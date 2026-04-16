import torch
import torchcde
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pandas as pd

import torch
import torchcde

def masked_NLL(mu, y_pad, V, mask):
    """
    Batch-averaged masked Gaussian marginal NLL.
 
    For each subject i, extracts the observed sub-vector and computes:
 
        NLL_i = 0.5 * [log|V_i| + (y_i - mu_i)' V_i^{-1} (y_i - mu_i)
                        + n_i * log(2π)]
 
    Returns the average over the batch:  (1/N) * Σ_i NLL_i
 
    This convention ensures that:
      - The loss magnitude is independent of batch size
      - The scheduler sees a consistent scale across batches
      - Gradient magnitudes are stable regardless of N
 
    Args:
        mu:    (N, T)     population mean predictions
        V:     (N, T, T)  marginal covariance matrices
        y_pad: (N, T)     outcomes (0 at unobserved slots)
        mask:  (N, T)     binary mask (1 = observed, 0 = unobserved)
        jitter: float     diagonal jitter for numerical stability
 
    Returns:
        scalar: batch-averaged NLL
    """
    N = mu.shape[0]
    device, dtype = mu.device, mu.dtype
    total_nll = torch.tensor(0.0, device=device, dtype=dtype)
    n_valid = 0
 
    for i in range(N):
        idx = mask[i].bool()
        n_i = idx.sum()
        if n_i == 0:
            continue
 
        mu_i = mu[i, idx]                                     # (n_i,)
        y_i = y_pad[i, idx]                                    # (n_i,)
        V_i = V[i][idx][:, idx]                                # (n_i, n_i)
 
        r_i = y_i - mu_i                                       # (n_i,)
 
        L_i = torch.linalg.cholesky(V_i)
        Vinv_r = torch.cholesky_solve(
            r_i.unsqueeze(-1), L_i).squeeze(-1)                # (n_i,)
 
        log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L_i)))
 
        nll_i = 0.5 * (log_det + r_i @ Vinv_r
                        + n_i * math.log(2 * math.pi))
        total_nll = total_nll + nll_i
        n_valid += 1
 
    if n_valid == 0:
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
 
    return total_nll / n_valid