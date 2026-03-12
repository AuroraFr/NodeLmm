"""
Orthogonalization for semi-structured hybrid models.

Two approaches following Rügamer et al. (2023):

1. DURING TRAINING (soft penalty):
   Add λ_orth * ||W'h||² to the loss.
   Pushes h away from column space of W without hard constraints.
   Pro: simple, differentiable, one line in training loop.
   Con: only approximate orthogonality (depends on λ).

2. POST-HOC (PHO):
   After training, decompose h = h_W + h_⊥ where h_W is the projection
   onto col(W). Absorb h_W into β by re-estimating β from Y - h_⊥.
   Pro: exact orthogonality, doesn't affect training dynamics.
   Con: β changes after training; must re-run PDP with corrected model.

References:
  - Rügamer, Kolb & Klein (2023), "Semi-Structured Distributional Regression",
    The American Statistician.
  - Rügamer (2023), "A New PHO-rmula for Improved Performance of
    Semi-Structured Networks", ICML 2023.
"""

import torch
import numpy as np


# ====================================================================
# 1. DURING TRAINING: Soft orthogonality penalty
# ====================================================================

def reg_orthogonality(h, W, obs_mask):
    """
    Soft orthogonality penalty: ||W'h||² / n_obs

    Penalizes the correlation between h and each column of W,
    pooled across all observed (non-padded) positions in the batch.

    Args:
        h:        (N, T)       neural contribution
        W:        (N, T, n_W)  parametric design matrix
        obs_mask: (N, T)       binary mask (1=observed)

    Returns:
        scalar penalty (to be multiplied by λ_orth and added to loss)
    """
    # Mask h and W to observed positions only
    # h_masked: (N, T), W_masked: (N, T, n_W)
    h_masked = h * obs_mask                          # (N, T)
    W_masked = W * obs_mask.unsqueeze(-1)            # (N, T, n_W)

    # W'h pooled across batch: sum over N and T
    # For each column k of W: correlation_k = sum_{i,t} W_{i,t,k} * h_{i,t}
    Wh = (W_masked * h_masked.unsqueeze(-1)).sum(dim=(0, 1))  # (n_W,)

    n_obs = obs_mask.sum().clamp(min=1)

    # Penalty: ||W'h||² / n_obs²  (normalized so it's scale-invariant)
    penalty = (Wh ** 2).sum() / (n_obs ** 2)

    return penalty


def reg_orthogonality_per_subject(h, W, obs_mask):
    """
    Per-subject orthogonality penalty: average of ||W_i' h_i||² / n_i²

    More granular than the pooled version — penalizes within-subject
    correlation, which may be more appropriate for longitudinal data.

    Args:
        h:        (N, T)       neural contribution
        W:        (N, T, n_W)  parametric design matrix
        obs_mask: (N, T)       binary mask (1=observed)

    Returns:
        scalar penalty
    """
    h_masked = h * obs_mask                          # (N, T)
    W_masked = W * obs_mask.unsqueeze(-1)            # (N, T, n_W)

    # Per-subject W'h: (N, n_W) = sum over T of W_{i,t,k} * h_{i,t}
    Wh_i = (W_masked * h_masked.unsqueeze(-1)).sum(dim=1)  # (N, n_W)

    n_i = obs_mask.sum(dim=1).clamp(min=1)  # (N,)

    # ||W_i'h_i||² / n_i² per subject, then average
    penalty_per_subject = (Wh_i ** 2).sum(dim=1) / (n_i ** 2)  # (N,)

    return penalty_per_subject.mean()


# ====================================================================
# 2. POST-HOC ORTHOGONALIZATION (PHO)
# ====================================================================

@torch.no_grad()
def posthoc_orthogonalize(model, loader, device, beta_names=None):
    """
    Post-hoc orthogonalization following Rügamer (2023).

    Steps:
      1. Collect h and W at all observed time points
      2. Project h onto col(W): h_W = W (W'W)^{-1} W' h
      3. Compute h_⊥ = h - h_W
      4. Re-estimate β from Y - h_⊥ using OLS (or GLS with current V)
      5. Store corrected β in model

    The corrected model satisfies: W' h_⊥ ≈ 0 (exact at the data points).

    Args:
        model:      trained NeuralCDEModel
        loader:     DataLoader (full dataset, no shuffle)
        device:     torch device
        beta_names: optional list of names for printing

    Returns:
        beta_new:   corrected β vector
        diagnostics: dict with useful info
    """
    model.eval()

    all_W = []
    all_h = []
    all_Y = []
    all_mask = []

    for batch in loader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad  = t_pad.to(device)
        x_pad  = x_pad.to(device)
        y_pad  = y_pad.to(device)
        mask   = mask.to(device)
        c_mask = c_mask.to(device)
        s      = s.to(device)

        mu, V, AtA, Atb, h = model(t_pad, x_pad, c_mask, s, mask, y_pad=y_pad)

        # Build W explicitly
        W = model.decoder._build_W(t_pad, x_pad, s)  # (N, T, n_W)

        N, T = t_pad.shape
        for i in range(N):
            obs = mask[i].bool()
            all_W.append(W[i, obs].cpu())       # (n_i, n_W)
            all_h.append(h[i, obs].cpu())       # (n_i,)
            all_Y.append(y_pad[i, obs].cpu())   # (n_i,)

    # Stack all observed data
    W_all = torch.cat(all_W, dim=0).float()    # (n_total, n_W)
    h_all = torch.cat(all_h, dim=0).float()    # (n_total,)
    Y_all = torch.cat(all_Y, dim=0).float()    # (n_total,)

    n_total, n_W = W_all.shape

    # --- Step 1: Compute projection P_W = W (W'W)^{-1} W' ---
    WtW = W_all.t() @ W_all                                # (n_W, n_W)
    WtW_reg = WtW + 1e-6 * torch.eye(n_W)                  # regularize
    Wth = W_all.t() @ h_all                                 # (n_W,)

    # Projection coefficients: gamma = (W'W)^{-1} W'h
    gamma = torch.linalg.solve(WtW_reg, Wth)                # (n_W,)

    # h_W = W @ gamma (the part of h in col(W))
    h_W = W_all @ gamma                                      # (n_total,)
    h_orth = h_all - h_W                                     # (n_total,)

    # --- Step 2: Re-estimate β from Y - h_orth ---
    # β_new = (W'W)^{-1} W' (Y - h_orth)
    residual = Y_all - h_orth
    Wt_residual = W_all.t() @ residual                       # (n_W,)
    beta_new = torch.linalg.solve(WtW_reg, Wt_residual)     # (n_W,)

    # --- Step 3: Diagnostics ---
    beta_old = model.decoder._last_beta.cpu().float()

    # Correlation between h and W columns (before and after)
    corr_before = []
    corr_after = []
    for k in range(n_W):
        w_k = W_all[:, k]
        corr_before.append(torch.corrcoef(torch.stack([w_k, h_all]))[0, 1].item())
        corr_after.append(torch.corrcoef(torch.stack([w_k, h_orth]))[0, 1].item())

    # How much of h was in col(W)?
    h_W_norm = h_W.norm().item()
    h_orth_norm = h_orth.norm().item()
    h_total_norm = h_all.norm().item()
    frac_in_W = (h_W_norm ** 2) / (h_total_norm ** 2 + 1e-10)

    # Print results
    print("\n" + "=" * 60)
    print("POST-HOC ORTHOGONALIZATION (PHO)")
    print("=" * 60)

    print(f"\n  h decomposition:")
    print(f"    ||h||      = {h_total_norm:.4f}")
    print(f"    ||h_W||    = {h_W_norm:.4f}  ({frac_in_W*100:.1f}% of h² in col(W))")
    print(f"    ||h_orth|| = {h_orth_norm:.4f}")

    print(f"\n  Projection coefficients γ (h_W = W @ γ):")
    names = beta_names if beta_names else [f"w{k}" for k in range(n_W)]
    for name, g in zip(names, gamma):
        print(f"    {name:>20s}: γ = {g.item():+.6f}")

    print(f"\n  β comparison (old → new):")
    for name, old, new in zip(names, beta_old, beta_new):
        shift = new.item() - old.item()
        print(f"    {name:>20s}: {old.item():+.6f} → {new.item():+.6f}  (Δ = {shift:+.6f})")

    print(f"\n  Correlation of W columns with h (before → after PHO):")
    for name, cb, ca in zip(names, corr_before, corr_after):
        print(f"    {name:>20s}: {cb:+.4f} → {ca:+.4f}")

    # --- Step 4: Update model ---
    model.decoder._last_beta.copy_(beta_new.to(model.decoder._last_beta.device))
    print(f"\n  ✓ Updated model._last_beta with corrected β")

    diagnostics = {
        "beta_old": beta_old,
        "beta_new": beta_new,
        "gamma": gamma,
        "h_W_norm": h_W_norm,
        "h_orth_norm": h_orth_norm,
        "frac_in_W": frac_in_W,
        "corr_before": corr_before,
        "corr_after": corr_after,
    }

    return beta_new, diagnostics


@torch.no_grad()
def posthoc_orthogonalize_gls(model, loader, device, beta_names=None):
    """
    Post-hoc orthogonalization using GLS (accounts for V weighting).

    Same as posthoc_orthogonalize but uses V^{-1} weighting:
      γ = (W'V⁻¹W)⁻¹ W'V⁻¹h
      h_W = W @ γ
      h_orth = h - h_W
      β_new = (W'V⁻¹W)⁻¹ W'V⁻¹(Y - h_orth)

    This is more correct statistically but slower (requires V per subject).
    """
    model.eval()

    n_W = model.decoder.n_W
    WtVW = torch.zeros(n_W, n_W)
    WtVh = torch.zeros(n_W)
    WtVY = torch.zeros(n_W)
    WtVh_orth = torch.zeros(n_W)

    all_gamma_contributions = []

    # First pass: compute γ = (W'V⁻¹W)⁻¹ W'V⁻¹h
    for batch in loader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad  = t_pad.to(device)
        x_pad  = x_pad.to(device)
        y_pad  = y_pad.to(device)
        mask   = mask.to(device)
        c_mask = c_mask.to(device)
        s      = s.to(device)

        mu, V, _, _, h = model(t_pad, x_pad, c_mask, s, mask, y_pad=y_pad)
        W = model.decoder._build_W(t_pad, x_pad, s)

        N, T = t_pad.shape
        for i in range(N):
            obs = mask[i].bool()
            n_i = obs.sum()
            if n_i < 1:
                continue

            W_i = W[i, obs].cpu().float()        # (n_i, n_W)
            h_i = h[i, obs].cpu().float()        # (n_i,)
            y_i = y_pad[i, obs].cpu().float()    # (n_i,)
            V_i = V[i][obs][:, obs].cpu().float()  # (n_i, n_i)

            L_i = torch.linalg.cholesky(V_i + 1e-6 * torch.eye(n_i))
            A_i = torch.linalg.solve_triangular(L_i, W_i, upper=False)  # L⁻¹W
            bh_i = torch.linalg.solve_triangular(L_i, h_i.unsqueeze(-1), upper=False).squeeze(-1)  # L⁻¹h
            by_i = torch.linalg.solve_triangular(L_i, y_i.unsqueeze(-1), upper=False).squeeze(-1)  # L⁻¹y

            WtVW += A_i.t() @ A_i
            WtVh += A_i.t() @ bh_i
            WtVY += A_i.t() @ by_i

    WtVW_reg = WtVW + 1e-6 * torch.eye(n_W)
    gamma = torch.linalg.solve(WtVW_reg, WtVh)          # (n_W,)

    # β_new = (W'V⁻¹W)⁻¹ W'V⁻¹(Y - h + W@γ)
    # = (W'V⁻¹W)⁻¹ (W'V⁻¹Y - W'V⁻¹h + W'V⁻¹W @ γ)
    # = (W'V⁻¹W)⁻¹ W'V⁻¹Y - γ + γ
    # Wait, let me redo this properly.
    #
    # h_orth = h - W@γ
    # β_new = (W'V⁻¹W)⁻¹ W'V⁻¹(Y - h_orth) = (W'V⁻¹W)⁻¹ W'V⁻¹(Y - h + W@γ)
    # = (W'V⁻¹W)⁻¹ (W'V⁻¹Y - W'V⁻¹h) + γ
    # = (W'V⁻¹W)⁻¹ W'V⁻¹(Y-h) + γ

    beta_from_residual = torch.linalg.solve(WtVW_reg, WtVY - WtVh)
    beta_new = beta_from_residual + gamma

    beta_old = model.decoder._last_beta.cpu().float()

    # Print
    print("\n" + "=" * 60)
    print("POST-HOC ORTHOGONALIZATION — GLS (V⁻¹ weighted)")
    print("=" * 60)

    names = beta_names if beta_names else [f"w{k}" for k in range(n_W)]

    print(f"\n  Projection coefficients γ (GLS):")
    for name, g in zip(names, gamma):
        print(f"    {name:>20s}: γ = {g.item():+.6f}")

    print(f"\n  β comparison (old → new):")
    for name, old, new in zip(names, beta_old, beta_new):
        shift = new.item() - old.item()
        print(f"    {name:>20s}: {old.item():+.6f} → {new.item():+.6f}  (Δ = {shift:+.6f})")

    model.decoder._last_beta.copy_(beta_new.to(model.decoder._last_beta.device))
    print(f"\n  ✓ Updated model._last_beta with GLS-corrected β")

    return beta_new, {"gamma": gamma, "beta_old": beta_old, "beta_new": beta_new}