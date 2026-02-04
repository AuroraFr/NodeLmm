import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LongitudinalDataset(Dataset):
    def __init__(self, df, id_col, time_col, x_cols, y_col, static_cols=None, dtype=torch.float32):
        self.id_col = id_col
        self.time_col = time_col
        self.x_cols = list(x_cols)
        self.y_col = y_col
        self.static_cols = list(static_cols) if static_cols is not None else []
        self.dtype = dtype

        self.subjects = []
        for sid, g in df.groupby(id_col, sort=False):
            g = g.sort_values(time_col)

            t = torch.tensor(g[time_col].to_numpy(), dtype=dtype)                  # (Ti,)
            x = torch.tensor(g[self.x_cols].to_numpy(), dtype=dtype)               # (Ti, C)
            y = torch.tensor(g[y_col].to_numpy(), dtype=dtype)                     # (Ti,)

            if self.static_cols:
                s = torch.tensor(g[self.static_cols].iloc[0].to_numpy(), dtype=dtype)  # (S,)
            else:
                s = None

            self.subjects.append((t, x, y, s))

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return self.subjects[idx]


def collate_pad(batch, pad_value=0.0, time_pad_mode="extend"):
    """
    batch: list of (t, x, y, s)
      t: (Ti,), x: (Ti,C), y: (Ti,), s: (S,) or None
    returns:
      t_pad: (N,T)
      x_pad: (N,T,C)
      y_pad: (N,T)
      mask:  (N,T) {0,1}
      s:     (N,S) or None
      lengths: (N,)
    """
    ts, xs, ys, ss = zip(*batch)
    N = len(ts)
    C = xs[0].shape[1]
    lengths = torch.tensor([t.shape[0] for t in ts], dtype=torch.long)
    T = int(lengths.max().item())

    t_pad = torch.full((N, T), float(pad_value), dtype=ts[0].dtype)
    x_pad = torch.full((N, T, C), float(pad_value), dtype=xs[0].dtype)
    y_pad = torch.full((N, T), float(pad_value), dtype=ys[0].dtype)
    mask  = torch.zeros((N, T), dtype=ts[0].dtype)

    for i, (t, x, y) in enumerate(zip(ts, xs, ys)):
        Ti = t.shape[0]
        t_pad[i, :Ti] = t
        x_pad[i, :Ti] = x
        y_pad[i, :Ti] = y
        mask[i, :Ti]  = 1.0

        if time_pad_mode == "extend" and Ti < T:
            # keep time strictly increasing after last real time
            last = t[-1]
            dt = (t[-1] - t[-2]).abs() if Ti >= 2 else torch.tensor(1.0, dtype=t.dtype)
            dt = torch.clamp(dt, min=1e-3)
            extra = last + dt * torch.arange(1, T - Ti + 1, dtype=t.dtype)
            t_pad[i, Ti:] = extra
            # x_pad/y_pad stay padded; mask is 0 so they won't affect loss

    # static covariates
    if ss[0] is None:
        s_pad = None
    else:
        S = ss[0].shape[0]
        s_pad = torch.stack(ss, dim=0)  # (N,S)

    return t_pad, x_pad, y_pad, mask, s_pad, lengths