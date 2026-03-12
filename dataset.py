import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LongitudinalDataset(Dataset):
    def __init__(self, df, id_col, time_col, x_cols, y_col, static_cols=None, dtype=torch.float32):
        self.id_col = id_col
        self.time_col = time_col
        self.x_cols = list(x_cols)
        self.y_col = y_col
        self.static_cols = static_cols
        self.dtype = dtype

        self.subjects = []
        for sid, g in df.groupby(id_col, sort=False):
            g = g.sort_values(time_col)
            
            t = torch.tensor(g[time_col].values, dtype=dtype)                  # (Ti,)
            x = torch.tensor(g[self.x_cols].values, dtype=dtype)               # (Ti, C)
            y = torch.tensor(g[y_col].values, dtype=dtype)                     # (Ti,)
            s = torch.tensor(g[self.static_cols].iloc[0].values, dtype=dtype)  # (S,)

            self.subjects.append((sid, t, x, y, s))

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return self.subjects[idx]

def collate_pad(batch, pad_value=0.0):

    ids, ts, xs, ys, ss = zip(*batch)
    N = len(ts)
    C = xs[0].shape[1]
    lengths = torch.tensor([t.shape[0] for t in ts], dtype=torch.long)
    T = int(lengths.max().item())

    t_pad = torch.full((N, T), float(pad_value), dtype=ts[0].dtype)
    x_pad = torch.full((N, T, C), float(pad_value), dtype=xs[0].dtype)
    y_pad = torch.full((N, T), float(pad_value), dtype=ys[0].dtype)
    ss = torch.stack(ss, dim=0)
    mask  = torch.zeros((N, T), dtype=ts[0].dtype)

    for i, (t, x, y) in enumerate(zip(ts, xs, ys)):
        Ti = t.shape[0]

        # fill observed part
        t_pad[i, :Ti] = t
        x_pad[i, :Ti] = x
        y_pad[i, :Ti] = y
        mask[i, :Ti]  = 1.0

        if Ti < T:
            # ---- forward-fill x and y on padded positions ----
            x_last = x[Ti - 1]          # (C,)
            y_last = y[Ti - 1]          # scalar
            x_pad[i, Ti:] = x_last.unsqueeze(0).expand(T - Ti, C)
            y_pad[i, Ti:] = y_last
            t_pad[i, Ti:] = t[-1]
    c_pad = torch.cumsum(mask, dim=1)
    return ids, t_pad, x_pad, y_pad, c_pad, mask, ss