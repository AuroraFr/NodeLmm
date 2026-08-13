import torch
from torch.utils.data import Dataset


# ══════════════════════════════════════════════════════════════════════════
# Unified dataset + collate
# ══════════════════════════════════════════════════════════════════════════
#
# Both data sources (simulation dataframes and preprocessed real data) are
# represented identically: every subject is
#     (id, x_aug (Ti, 1+2K), y (Ti,), obs_mask (Ti,), static (Ks,))
# with x_aug = [time(1), x(K), mask(K)]. A single collate (collate_xaug) pads
# variable-length subjects; for a fixed grid it pads nothing.

class XAugDataset(Dataset):
    """Unified dataset over subjects already in x_aug form.

    Build it with one of the factories:
      - from_long_df    : long-format dataframe (one row per visit)
      - from_processed  : list of dicts from process_data (real pipeline)
    """
    def __init__(self, subjects):
        self.subjects = subjects   # list of (id, x_aug, y, obs_mask, static)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return self.subjects[idx]

    @classmethod
    def from_long_df(cls, df, id_col, time_col, x_cols, y_col,
                     static_cols, dtype=torch.float32):
        """Build from a long dataframe. Covariates are treated as fully
        observed, so the mask block is all ones; padding masks are added by
        collate_xaug. x_aug per subject = [time(1), x(K), mask(K)=1]."""
        subjects = []
        for sid, g in df.groupby(id_col, sort=False):
            g = g.sort_values(time_col)
            t = torch.tensor(g[time_col].values, dtype=dtype)             # (Ti,)
            x = torch.tensor(g[list(x_cols)].values, dtype=dtype)         # (Ti,K)
            y = torch.tensor(g[y_col].values, dtype=dtype)                # (Ti,)
            s = torch.tensor(g[list(static_cols)].iloc[0].values, dtype=dtype)
            mask_block = torch.ones_like(x)                               # (Ti,K)
            x_aug = torch.cat([t.unsqueeze(-1), x, mask_block], dim=-1)   # (Ti,1+2K)
            obs = torch.ones(t.shape[0], dtype=dtype)
            subjects.append((sid, x_aug, y, obs, s))
        return cls(subjects)

    @classmethod
    def from_processed(cls, patient_data_list):
        """Build from process_data output (dicts already carrying x_aug)."""
        subjects = [
            (d["patient_id"], d["x_aug"], d["y"], d["target_mask"], d["s_i"])
            for d in patient_data_list
        ]
        return cls(subjects)


def collate_xaug(batch, pad_value=0.0):
    """Pad variable-length subjects to the batch max T.

    Padded rows repeat the last real time (so dt=0 in the ODE and the
    integrator does not advance through them), carry forward-filled covariates,
    and have their covariate-mask columns and obs_mask set to 0. A fixed-grid
    batch (all Ti equal) pads nothing.

    Returns (ids, x_aug (N,T,1+2K), y (N,T), obs_mask (N,T), static (N,Ks)).
    """
    ids, xa, ys, obs, ss = zip(*batch)
    N = len(xa)
    F = xa[0].shape[1]
    assert (F - 1) % 2 == 0, "x_aug must be [time, x(K), mask(K)]"
    K = (F - 1) // 2
    T = max(x.shape[0] for x in xa)
    dtype = xa[0].dtype

    x_aug = torch.full((N, T, F), float(pad_value), dtype=dtype)
    y = torch.full((N, T), float(pad_value), dtype=ys[0].dtype)
    m = torch.zeros((N, T), dtype=dtype)
    ss = torch.stack(ss, dim=0)

    for i, (x, yy, oo) in enumerate(zip(xa, ys, obs)):
        Ti = x.shape[0]
        x_aug[i, :Ti] = x
        y[i, :Ti] = yy
        m[i, :Ti] = oo
        if Ti < T:
            x_aug[i, Ti:] = x[Ti - 1]        # ffill time + covariates + mask
            x_aug[i, Ti:, 1 + K:] = 0.0      # padded → covariate mask 0
            y[i, Ti:] = yy[Ti - 1]
            # obs_mask m stays 0 for padded steps
    return list(ids), x_aug, y, m, ss