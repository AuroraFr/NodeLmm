"""
Full-parameter delta method for ∆PDP variance — Neural ODE-LMM.
CONTINUOUS-TIME version: evaluates PDP at exact target times by
augmenting each subject's time grid with the evaluation points.

Key differences from windowed version:
  - No time-windowed matching (_closest_obs_per_subject)
  - ODE is solved at exact evaluation times (augmented grid)
  - All subjects contribute at all eval times (no follow-up filtering)
  - Oracle evaluated at exact t_eval (no mean_times approximation)

Architecture and inference unchanged:
  - skip_gate / group_lasso regularisation (data-independent penalty)
  - M-estimator inference (Commenges et al., 2014)

Three variance estimators (selected via --sandwich):
  Default:  Cov(θ̂) = F⁻¹                 (F = Σ φ_i φ_iᵀ, penalised scores)
  Bayesian: Cov(θ̂) = J⁻¹                 (O'Sullivan 1988)
  Sandwich: Cov(θ̂) = J⁻¹ F J⁻¹           (robust, Commenges et al. 2014)
"""
from __future__ import annotations
import math, os, csv, time
import torch
import torch.nn.functional as F_torch
import numpy as np
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _param_list(model):
    """Ordered list of parameters that require grad."""
    return [p for p in model.parameters() if p.requires_grad]


def _param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _is_nn_param(name):
    """True for encoder/func/decoder network weights, False for β, D, σ², gates."""
    excluded = ('beta', 'log_D_diag', 'log_sigma2', 'D_off_diag',
                'skip_gate_logits', 'gate_logits')
    return not any(ex in name for ex in excluded)

def _cat_grads(grads):
    """Flatten and concatenate a tuple of gradient tensors."""
    return torch.cat([g.reshape(-1) for g in grads])


# ─────────────────────────────────────────────────────────
# 1. Per-subject NLL (differentiable)
# ─────────────────────────────────────────────────────────

def _per_subject_nll(mu, V, y_pad, mask, jitter=1e-4):
    """
    Masked Gaussian NLL for ONE subject (B=1).
    """
    mu = mu.squeeze(0)
    V = V.squeeze(0)
    y = y_pad.squeeze(0)
    m = mask.squeeze(0)

    idx = m.bool()
    n_i = idx.sum()
    if n_i == 0:
        return torch.tensor(0.0, device=mu.device, requires_grad=True)

    mu_obs = mu[idx]
    y_obs = y[idx]
    V_obs = V[idx][:, idx] + jitter * torch.eye(n_i, device=mu.device, dtype=mu.dtype)

    residual = y_obs - mu_obs
    L = torch.linalg.cholesky(V_obs)
    Vinv_r = torch.cholesky_solve(residual.unsqueeze(-1), L).squeeze(-1)
    log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L)))

    nll = 0.5 * (log_det + residual @ Vinv_r + n_i * math.log(2 * math.pi))
    return nll


# ─────────────────────────────────────────────────────────
# 2. Empirical Fisher  F = Σ_i  φ_i  φ_iᵀ
# ─────────────────────────────────────────────────────────

def _compute_penalty_gradient(model, lambda_reg, weight_decay):
    """
    Compute the data-independent penalty gradient:

        c = λ_reg · ∇_θ reg_term  +  λ_wd · (mask ⊙ θ)

    Weight decay applies ONLY to NN params (_is_nn_param), matching
    the training optimizer config.

    Returns:
        c: (P,) tensor on CPU, or None if no penalty
    """
    params = _param_list(model)
    P = sum(p.numel() for p in params)
    c = torch.zeros(P)

    has_penalty = False

    # --- reg_term gradient (skip_gate or group_lasso) ---
    if lambda_reg > 0:
        reg_mode = getattr(model, 'reg_mode', None)
        if reg_mode is not None:
            reg_dict = model.decoder._compute_reg(None)
            reg_term = reg_dict["reg_term"]
            if reg_term.requires_grad:
                grads = torch.autograd.grad(reg_term, params,
                                            allow_unused=True)
                grads = [g if g is not None else torch.zeros_like(p)
                         for g, p in zip(grads, params)]
                c += lambda_reg * _cat_grads(grads).cpu()
                has_penalty = True

    # --- weight decay gradient: λ_wd · θ_nn only ---
    if weight_decay > 0:
        offset = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            numel = p.numel()
            if _is_nn_param(name):
                c[offset:offset + numel] = weight_decay * p.detach().reshape(-1).cpu()
            offset += numel
        has_penalty = True

    return c if has_penalty else None


def compute_empirical_fisher(model, dataset, device, collate_fn,
                              lambda_reg=0.0, weight_decay=0.0,
                              verbose=True):
    """
    Compute F = Σ_i  φ_i  φ_iᵀ   where  φ_i = ∇_θ nll_i + c  (penalised score).

    c = λ_reg · ∇_θ reg_term + λ_wd · (mask ⊙ θ) is a data-independent constant
    computed once and added to each NLL score (Commenges et al., 2014, eq. 8-9).
    Weight decay applies only to NN weight matrices (not biases/variance params).

    At the penalised MLE: Σ φ_i = 0, so mean(φ_i) ≈ 0 (stationarity check).

    Returns:
        F: (P, P) tensor on CPU
        scores: (N, P) tensor on CPU  (penalised scores φ_i)
    """
    from torch.utils.data import DataLoader

    model.eval()
    params = _param_list(model)
    P = sum(p.numel() for p in params)
    N = len(dataset)

    c = _compute_penalty_gradient(model, lambda_reg, weight_decay)

    if verbose:
        if c is not None:
            parts = ["∇nll_i"]
            if lambda_reg > 0:
                reg_mode = getattr(model, 'reg_mode', 'unknown')
                parts.append(f"λ_reg·∇reg ({reg_mode})")
            if weight_decay > 0:
                # Count NN params
                n_wd = sum(p.numel() for n, p in model.named_parameters()
                           if p.requires_grad and _is_nn_param(n))
                parts.append(f"λ_wd·θ_nn [{n_wd}/{P} params]")
            print(f"    Score: φ_i = {' + '.join(parts)}")
            print(f"    Penalty gradient ||c|| = {c.norm().item():.4f}")
        else:
            print(f"    Score: φ_i = ∇nll_i  (no penalty)")

    fisher = torch.zeros(P, P)
    score_list = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn)

    t0 = time.time()
    for i, batch in enumerate(loader):
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        y_pad = y_pad.to(device)
        mask = mask.to(device)
        s = s.to(device)
        bmi_t = x_pad[:, :, 0:1]

        if mask.sum() == 0:
            continue

        mu, V, Z, D, sig2, _ = model(
            t_pad, x_pad, masks=None,
            static_covariates=s, bmi_t=bmi_t, obs_mask=mask
        )

        nll_i = _per_subject_nll(mu, V, y_pad, mask)

        grads = torch.autograd.grad(nll_i, params, retain_graph=False,
                                    allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, params)]
        phi_i = _cat_grads(grads).cpu()

        if c is not None:
            phi_i = phi_i + c

        fisher += torch.outer(phi_i, phi_i)
        score_list.append(phi_i)

        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"    Fisher: {i+1}/{N} subjects "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    scores = torch.stack(score_list)
    if verbose:
        print(f"    Fisher: done ({time.time()-t0:.1f}s), "
              f"cond = {torch.linalg.cond(fisher).item():.2e}")

    return fisher, scores


import torch.func as TF


def _compute_penalty_hessian(model, device, lambda_reg, verbose=True):
    """
    Compute ∇²(reg_term) for skip_gate or group_lasso via autograd.
    """
    reg_mode = getattr(model, 'reg_mode', None)
    if reg_mode is None or lambda_reg <= 0:
        return None

    params = _param_list(model)
    P = sum(p.numel() for p in params)

    reg_dict = model.decoder._compute_reg(None)
    reg_term = reg_dict["reg_term"]

    if not reg_term.requires_grad:
        if verbose:
            print(f"    Penalty Hessian: reg_term has no grad (penalty=0)")
        return None

    grad1 = torch.autograd.grad(reg_term, params, create_graph=True,
                                 allow_unused=True)
    grad1 = [g if g is not None else torch.zeros_like(p)
             for g, p in zip(grad1, params)]
    grad_flat = torch.cat([g.reshape(-1) for g in grad1])

    H_pen = torch.zeros(P, P)
    for j in range(P):
        if grad_flat[j].requires_grad:
            row = torch.autograd.grad(
                grad_flat[j], params,
                retain_graph=(j < P - 1),
                allow_unused=True,
            )
            row = [g if g is not None else torch.zeros_like(p)
                   for g, p in zip(row, params)]
            H_pen[j, :] = _cat_grads(row).cpu()

    H_pen = 0.5 * (H_pen + H_pen.T)

    if verbose:
        nnz = (H_pen.abs() > 1e-12).sum().item()
        print(f"    Penalty Hessian ({reg_mode}): "
              f"{nnz}/{P*P} nonzero entries, "
              f"||H_pen|| = {H_pen.norm().item():.4e}")

    return H_pen


def compute_hessian_explicit(model, dataset, device, collate_fn,
                             n_subsample=None, weight_decay=0.0,
                             lambda_reg=0.0,
                             verbose=True):
    """
    Compute J = Σ ∇²nll_i + N·λ_wd·diag(mask_nn) + N·λ_reg·∇²reg_term.

    Weight decay Hessian is diagonal, applied only to NN params.
    Reg penalty Hessian computed via autograd.
    """
    from torch.utils.data import DataLoader, Subset

    model.eval()
    params = _param_list(model)
    P = sum(p.numel() for p in params)
    N = len(dataset)

    if n_subsample is not None and n_subsample < N:
        indices = torch.randperm(N)[:n_subsample].tolist()
        subset = Subset(dataset, indices)
        scale = N / n_subsample
        M = n_subsample
        if verbose:
            print(f"    Hessian: subsampling {M}/{N} subjects (scale={scale:.2f})")
    else:
        subset = dataset
        scale = 1.0
        M = N

    loader = DataLoader(subset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn)

    param_names, param_shapes, param_numels = [], [], []
    flat_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            param_names.append(name)
            param_shapes.append(p.shape)
            param_numels.append(p.numel())
            flat_params.append(p.detach().reshape(-1))
    theta0 = torch.cat(flat_params).to(device)

    def _nll_from_flat(theta, batch_data):
        t_pad, x_pad, y_pad, mask, s = batch_data
        offset = 0
        param_dict = {}
        for name, shape, numel in zip(param_names, param_shapes, param_numels):
            param_dict[name] = theta[offset:offset+numel].view(shape)
            offset += numel

        bmi_t = x_pad[:, :, 0:1]
        mu, V, Z, D_mat, sig2, _ = TF.functional_call(
            model, param_dict,
            args=(t_pad, x_pad),
            kwargs=dict(masks=None, static_covariates=s,
                        bmi_t=bmi_t, obs_mask=mask)
        )
        return _per_subject_nll(mu, V, y_pad, mask)

    hessian = torch.zeros(P, P)
    t0 = time.time()
    n_done = 0

    for i, batch in enumerate(loader):
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad, x_pad = t_pad.to(device), x_pad.to(device)
        y_pad, mask, s = y_pad.to(device), mask.to(device), s.to(device)

        if mask.sum() == 0:
            continue

        batch_data = (t_pad, x_pad, y_pad, mask, s)

        H_i = torch.func.jacrev(torch.func.jacrev(
            lambda th: _nll_from_flat(th, batch_data)
        ))(theta0)

        hessian += H_i.cpu()
        n_done += 1

        if verbose and n_done % 10 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (M - n_done) / rate
            print(f"    Hessian: {n_done}/{M} subjects "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    hessian *= scale
    hessian = 0.5 * (hessian + hessian.T)

    # --- Add penalty Hessians to J ---
    N_full = len(dataset)

    # Weight decay: +N·λ_wd on NN-param diagonal only
    if weight_decay > 0:
        offset = 0
        n_wd = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            numel = p.numel()
            if _is_nn_param(name):
                for j in range(numel):
                    hessian[offset + j, offset + j] += N_full * weight_decay
                n_wd += numel
            offset += numel
        if verbose:
            print(f"    Added weight decay to J: {n_wd}/{P} NN params, "
                  f"+{N_full * weight_decay:.2e} on selected diag")

    # Reg penalty: +N·λ_reg·∇²reg_term
    if lambda_reg > 0:
        H_pen = _compute_penalty_hessian(model, device, lambda_reg, verbose)
        if H_pen is not None:
            hessian += N_full * lambda_reg * H_pen
            if verbose:
                print(f"    Added penalty Hessian to J: "
                      f"N·λ_reg = {N_full * lambda_reg:.2e}")

    if verbose:
        eigvals = torch.linalg.eigvalsh(hessian)
        print(f"    Hessian done ({time.time()-t0:.1f}s), "
              f"used {n_done} subjects")
        print(f"    J eigenvalue range: [{eigvals.min().item():.2e}, "
              f"{eigvals.max().item():.2e}]")
        n_neg = (eigvals < 0).sum().item()
        print(f"    J negative eigenvalues: {n_neg}/{P}")
        print(f"    J cond = {torch.linalg.cond(hessian).item():.2e}")
    return hessian


# ─────────────────────────────────────────────────────────
# 3. CONTINUOUS-TIME ∆PDP gradients
# ─────────────────────────────────────────────────────────


def compute_delta_pdp_gradients_continuous(
    model, dataset, device, collate_fn,
    bmi_lo: float,
    bmi_hi: float,
    t_eval: np.ndarray,
    verbose: bool = True,
    batch_size: int = 64,
):
    """
    Continuous-time ∆PDP gradients — BATCHED over subjects.

    ∆PDP(t) = (1/N) Σ_i [μ_i(t; BMI=hi) − μ_i(t; BMI=lo)]
    g(t)    = ∇_θ ∆PDP(t)

    All subjects share the same eval-only time grid [0] ∪ t_eval, so the
    forward pass is batched over subjects (batch_size=B).

    Backward passes: ceil(N/B) × L  instead of  N × L  (batch_size=1).
    For N=5859, B=64, L=7:  644 vs 41,013 — a ~64× speedup.
    """
    from torch.utils.data import DataLoader

    model.eval()
    params = _param_list(model)
    P = sum(p.numel() for p in params)
    N = len(dataset)
    L = len(t_eval)

    # --- Build shared eval-only grid: [0] ∪ t_eval, sorted unique ---
    t_grid_np = np.array(sorted(set([0.0] + list(map(float, t_eval)))),
                         dtype=np.float32)
    T_grid = len(t_grid_np)
    eval_indices = [int(np.argmin(np.abs(t_grid_np - float(te))))
                    for te in t_eval]

    # --- Scale substeps to keep h_eff ≈ training value ---
    orig_substeps = getattr(model.cfg, 'euler_steps_per_interval', None)
    model.cfg.euler_steps_per_interval = 4
    if verbose:
        print(f"    Eval-only grid: t={t_grid_np.tolist()}, "
              f"substeps=4 (was {orig_substeps}), batch_size={batch_size}")

    # --- Pre-extract all static covariates (fast, no grad) ---
    all_statics = []
    single_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                               collate_fn=collate_fn)
    x_dim = None
    with torch.no_grad():
        for batch in single_loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            all_statics.append(s.squeeze(0))
            if x_dim is None:
                x_dim = x_pad.shape[2]
    all_statics = torch.stack(all_statics)  # (N, S) on CPU

    if verbose:
        print(f"    Extracted {N} static covariate vectors, x_dim={x_dim}")

    # --- Shared grid tensors (pre-allocated) ---
    t_grid_1 = torch.tensor(t_grid_np, dtype=torch.float32,
                            device=device).unsqueeze(0)  # (1, T_grid)

    g_accum = torch.zeros(L, P)     # accumulated gradients (CPU)
    est_accum = np.zeros(L)          # accumulated estimates
    N_total = 0

    t0 = time.time()
    n_batches = (N + batch_size - 1) // batch_size

    for b_idx in range(n_batches):
        start = b_idx * batch_size
        end = min(start + batch_size, N)
        B = end - start

        # --- Build batch tensors on the shared grid ---
        s_batch = all_statics[start:end].to(device)           # (B, S)
        t_batch = t_grid_1.expand(B, -1)                       # (B, T_grid)
        obs_mask_batch = torch.ones(B, T_grid, device=device)  # (B, T_grid)

        x_hi = torch.zeros(B, T_grid, x_dim, device=device)
        x_lo = torch.zeros(B, T_grid, x_dim, device=device)
        x_hi[:, :, 0] = bmi_hi
        x_lo[:, :, 0] = bmi_lo

        # --- Two counterfactual forward passes (batched) ---
        mu_hi, _, _, _, _, _ = model(
            t_batch, x_hi, masks=None,
            static_covariates=s_batch, bmi_t=x_hi[:, :, 0:1],
            obs_mask=obs_mask_batch,
        )
        mu_lo, _, _, _, _, _ = model(
            t_batch, x_lo, masks=None,
            static_covariates=s_batch, bmi_t=x_lo[:, :, 0:1],
            obs_mask=obs_mask_batch,
        )

        # --- For each eval time: backward through batch sum ---
        for k, aug_idx in enumerate(eval_indices):
            # Sum of deltas across subjects in this batch (scalar, in graph)
            delta_batch_sum = (mu_hi[:, aug_idx] - mu_lo[:, aug_idx]).sum()

            retain = (k < L - 1)
            grads = torch.autograd.grad(
                delta_batch_sum, params,
                retain_graph=retain,
                allow_unused=True,
            )
            grads = [g if g is not None else torch.zeros_like(p)
                     for g, p in zip(grads, params)]
            g_accum[k] += _cat_grads(grads).cpu()
            est_accum[k] += delta_batch_sum.item()

        N_total += B

        if verbose and (b_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (b_idx + 1) / elapsed
            eta = (n_batches - b_idx - 1) / rate
            print(f"    PDP batched: {b_idx+1}/{n_batches} batches, "
                  f"{N_total}/{N} subjects "
                  f"({elapsed:.0f}s, ~{eta:.0f}s remaining)")

    # --- Restore substeps ---
    model.cfg.euler_steps_per_interval = orig_substeps

    # --- Normalise by N ---
    gradients = {}
    estimates = {}
    counts = {}

    for l in range(L):
        t = float(t_eval[l])
        counts[t] = N_total
        gradients[t] = g_accum[l] / N_total
        estimates[t] = est_accum[l] / N_total

    if verbose:
        elapsed = time.time() - t0
        print(f"\n    PDP done ({elapsed:.1f}s), "
              f"{N_total} subjects, {n_batches} batches × {L} eval times "
              f"= {n_batches * L} backward passes "
              f"(was {N_total * L} with batch_size=1)")
        for l in range(L):
            t = float(t_eval[l])
            print(f"    t={t:.1f}: ||g|| = {gradients[t].norm().item():.4f}, "
                  f"n = {counts[t]}, ∆PDP = {estimates[t]:.4f}")

    return gradients, estimates, counts


# ─────────────────────────────────────────────────────────
# 4. Regularise & invert helper
# ─────────────────────────────────────────────────────────

def _ledoit_wolf_shrink_fisher(fisher, scores, verbose=True):
    """
    Ledoit-Wolf shrinkage (Ledoit & Wolf, 2004) for the empirical Fisher.

    Shrinks F toward a scaled identity to correct the downward bias
    of small eigenvalues when N/P is finite:

        F_shrunk = (1 - α) F  +  α · (tr(F)/P) · I

    The optimal α minimises E[||F_shrunk/N - Σ_true||²_F].

    Args:
        fisher: (P, P) empirical Fisher  F = Σ φ_i φ_iᵀ
        scores: (N, P) per-subject penalised scores φ_i

    Returns:
        F_shrunk: (P, P) shrunk Fisher
        alpha:    optimal shrinkage intensity ∈ [0, 1]
    """
    N, P = scores.shape

    # Sample covariance  S = F / N
    S = fisher / N
    mu = torch.trace(S).item() / P          # target: μ I

    # δ² = ||S − μI||²_F  (distance from sample to target)
    delta_sq = (S - mu * torch.eye(P)).pow(2).sum().item()

    if delta_sq < 1e-30:
        if verbose:
            print(f"  Ledoit-Wolf: δ² ≈ 0, no shrinkage needed")
        return fisher.clone(), 0.0

    # β = (1/N²) Σ_i ||φ_i φ_iᵀ − S||²_F
    #   = (1/N²) [Σ_i ||φ_i||⁴  −  N ||S||²_F]
    norms_sq = (scores ** 2).sum(dim=1)           # (N,)  ||φ_i||²
    sum_norms_4 = (norms_sq ** 2).sum().item()     # Σ ||φ_i||⁴
    S_frob_sq = (S ** 2).sum().item()              # ||S||²_F

    beta = max((1.0 / N**2) * (sum_norms_4 - N * S_frob_sq), 0.0)

    # Optimal shrinkage intensity
    alpha = min(beta / delta_sq, 1.0)

    # Shrunk Fisher (scale of F, not S)
    trace_F = torch.trace(fisher).item()
    F_shrunk = (1.0 - alpha) * fisher + alpha * (trace_F / P) * torch.eye(P)

    if verbose:
        eig_orig = torch.linalg.eigvalsh(fisher)
        eig_shrunk = torch.linalg.eigvalsh(F_shrunk)
        print(f"  Ledoit-Wolf shrinkage:")
        print(f"    N/P = {N}/{P} = {N/P:.1f}")
        print(f"    α* = {alpha:.4f}  (0 = no shrinkage, 1 = full shrinkage to μI)")
        print(f"    target μ = tr(F)/P = {trace_F/P:.2e}")
        print(f"    F  eigenvalues: [{eig_orig.min().item():.2e}, "
              f"{eig_orig.max().item():.2e}], "
              f"cond = {eig_orig.max().item()/max(eig_orig.min().item(),1e-30):.2e}")
        print(f"    F* eigenvalues: [{eig_shrunk.min().item():.2e}, "
              f"{eig_shrunk.max().item():.2e}], "
              f"cond = {eig_shrunk.max().item()/max(eig_shrunk.min().item(),1e-30):.2e}")

    return F_shrunk, alpha


def _regularise_and_invert(M, label, LAMBDA=1e-4, verbose=True):
    """Marquardt-damp a PSD matrix and invert it."""
    diag_M = torch.diag(M)
    diag_M = torch.clamp(diag_M, min=1e-4 * diag_M.max())
    M_reg = M + LAMBDA * torch.diag(diag_M)
    cond = torch.linalg.cond(M_reg).item()
    if verbose:
        print(f"  {label}: diag range [{torch.diag(M).min().item():.2e}, "
              f"{torch.diag(M).max().item():.2e}], cond = {cond:.2e}")
    try:
        M_inv = torch.linalg.inv(M_reg)
    except torch.linalg.LinAlgError:
        if verbose:
            print(f"  WARNING: {label} not invertible, using pseudo-inverse")
        M_inv = torch.linalg.pinv(M_reg)
    return M_reg, M_inv


# ─────────────────────────────────────────────────────────
# 5. Main: full-parameter delta method variance (continuous)
# ─────────────────────────────────────────────────────────

def compute_full_delta_variance(
    model, dataset, device, collate_fn,
    bmi_lo=20.0, bmi_hi=35.0,
    t_eval=None,
    oracle_fn=None,
    sandwich=False,
    n_hessian_subsample=None,
    weight_decay=0.0,
    lambda_reg=0.0,
    batch_size_pdp=64,
    ledoit_wolf=False,
):
    """
    Full-parameter delta method for ∆PDP variance — continuous time.

    Args:
        oracle_fn: callable(t) → true ∆PDP(t), or None.
    """
    if t_eval is None:
        t_eval = np.array([2, 4, 8, 10], dtype=float)

    P = _param_count(model)
    delta_v = bmi_hi - bmi_lo
    var_method = "SANDWICH (J⁻¹FJ⁻¹)" if sandwich else "FISHER (F⁻¹)"

    print(f"\n{'='*60}")
    print(f"FULL-PARAMETER DELTA METHOD — {var_method}")
    print(f"  (Continuous-time PDP)")
    print(f"{'='*60}")
    print(f"  P = {P} parameters")
    print(f"  N = {len(dataset)} subjects")
    print(f"  BMI: lo={bmi_lo}, hi={bmi_hi}, Δv={delta_v}")
    print(f"  t_eval: {t_eval.tolist()}")
    if weight_decay > 0:
        print(f"  Weight decay: {weight_decay:.2e}")
    if lambda_reg > 0:
        reg_mode = getattr(model, 'reg_mode', 'unknown')
        print(f"  λ_reg: {lambda_reg:.2e} (mode: {reg_mode})")

    # --- Step 1: Empirical Fisher ---
    print(f"\nStep 1: Empirical Fisher F = Σ_i φ_i φ_iᵀ ...")
    fisher, scores = compute_empirical_fisher(
        model, dataset, device, collate_fn,
        lambda_reg=lambda_reg, weight_decay=weight_decay,
    )

    eigvals, eigvecs = torch.linalg.eigh(fisher)
    print(f"  Fisher rank (>1e-6): {(eigvals > 1e-6).sum().item()} / {P}")
    print(f"  Eigenvalue range: [{eigvals.min().item():.2e}, "
          f"{eigvals.max().item():.2e}]")

    mean_score = scores.mean(dim=0)
    mean_abs = scores.abs().mean(dim=0)
    ratio = mean_score.norm() / mean_abs.norm()
    mean_score_norm = mean_score.norm().item()
    mean_indiv_norm = scores.norm(dim=1).mean().item()
    print(f"  Stationarity: ||mean(φ)|| / ||mean(|φ|)|| = {ratio:.4f}")
    print(f"  ||mean(φ)|| = {mean_score_norm:.4e},  mean(||φ_i||) = {mean_indiv_norm:.4e}")

    LAMBDA = 1e-4

    if ledoit_wolf:
        print(f"\n  Applying Ledoit-Wolf shrinkage to F ...")
        fisher, lw_alpha = _ledoit_wolf_shrink_fisher(fisher, scores)

    fisher_reg, fisher_inv = _regularise_and_invert(fisher, "Fisher", LAMBDA)

    # --- Step 1b (optional): Hessian + Sandwich ---
    sandwich_cov = None
    bayes_cov = None
    if sandwich:
        print(f"\nStep 1b: Hessian J = ∇²(penalised objective) ...")
        J = compute_hessian_explicit(
            model, dataset, device, collate_fn,
            n_subsample=n_hessian_subsample,
            weight_decay=weight_decay, lambda_reg=lambda_reg,
        )
        J_reg, J_inv = _regularise_and_invert(J, "Hessian J", LAMBDA)
        sandwich_cov = J_inv @ fisher @ J_inv
        bayes_cov = J_inv

        diag_ratio = torch.diag(fisher) / torch.clamp(torch.diag(J_reg), min=1e-10)
        print(f"  diag(F)/diag(J): median={diag_ratio.median().item():.3f}, "
              f"mean={diag_ratio.mean().item():.3f}")

    # --- Step 2: Continuous-time ∆PDP gradients ---
    print(f"\nStep 2: Continuous-time ∆PDP gradients ...")
    gradients, estimates, counts = compute_delta_pdp_gradients_continuous(
        model, dataset, device, collate_fn,
        bmi_lo=bmi_lo, bmi_hi=bmi_hi,
        t_eval=t_eval,
        batch_size=batch_size_pdp,
    )

    # --- Step 3: Null space diagnostic & Var = gᵀ Cov g ---
    # Check how much of each ∆PDP gradient lies in the null space of the raw Fisher
    null_fracs = {}
    null_thresh = eigvals.max().item() * 1e-10  # relative threshold
    null_mask = eigvals < null_thresh
    n_null = null_mask.sum().item()
    print(f"\n  Null space diagnostic (relative thresh = {null_thresh:.2e}):")
    print(f"    {n_null}/{P} eigenvalues below threshold")
    for t in sorted(estimates.keys()):
        g = gradients[t]
        coeffs = eigvecs.T @ g
        frac_null = (coeffs[null_mask] ** 2).sum().item() / max((coeffs ** 2).sum().item(), 1e-30)
        null_fracs[t] = frac_null
        print(f"    t={t:.1f}: ||g_null||² / ||g||² = {frac_null:.4f}")

    print(f"\nStep 3: Var(∆PDP(t)) = g(t)ᵀ Cov g(t)")

    if sandwich:
        header = (f"  {'t':>6s}  {'∆PDP':>10s}  {'SE_F⁻¹':>10s}  "
                  f"{'SE_bayes':>10s}  {'SE_sand':>10s}  "
                  f"{'CI_lo':>10s}  {'CI_hi':>10s}")
    else:
        header = (f"  {'t':>6s}  {'∆PDP':>10s}  {'SE':>10s}  "
                  f"{'CI_lo':>10s}  {'CI_hi':>10s}")
    if oracle_fn is not None:
        header += f"  {'True':>10s}  {'Bias':>10s}"
    header += f"  {'n':>5s}"
    print(header)
    print(f"  {'-'*len(header)}")

    results = {}
    for t in sorted(estimates.keys()):
        g = gradients[t]
        est = estimates[t]
        n_t = counts[t]

        var_fisher = (g @ fisher_inv @ g).item()

        if sandwich:
            var_sand = (g @ sandwich_cov @ g).item()
            var_bayes = (g @ bayes_cov @ g).item()
            se_fisher = np.sqrt(max(var_fisher, 0.0))
            se_bayes = np.sqrt(max(var_bayes, 0.0))
            se_sand = np.sqrt(max(var_sand, 0.0))
            se_main = se_bayes
        else:
            var_sand = float('nan')
            var_bayes = float('nan')
            se_fisher = np.sqrt(max(var_fisher, 0.0))
            se_main = se_fisher
            se_bayes = float('nan')
            se_sand = float('nan')

        ci_lo = est - 1.96 * se_main
        ci_hi = est + 1.96 * se_main

        true_val = oracle_fn(t) if oracle_fn is not None else float('nan')
        bias = est - true_val if oracle_fn is not None else float('nan')

        results[t] = {
            'estimate': est, 'se': se_main,
            'var_fisher': var_fisher, 'var_sandwich': var_sand,
            'var_bayes': var_bayes,
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'true': true_val, 'bias': bias, 'n': n_t,
        }

        if sandwich:
            line = (f"  {t:6.1f}  {est:+10.4f}  {se_fisher:10.4f}  "
                    f"{se_bayes:10.4f}  {se_sand:10.4f}  "
                    f"{ci_lo:+10.4f}  {ci_hi:+10.4f}")
        else:
            line = (f"  {t:6.1f}  {est:+10.4f}  {se_main:10.4f}  "
                    f"{ci_lo:+10.4f}  {ci_hi:+10.4f}")
        if oracle_fn is not None:
            line += f"  {true_val:+10.4f}  {bias:+10.4f}"
        line += f"  {n_t:5d}"
        print(line)

    ret = {
        'results': results,
        'fisher': fisher.numpy(), 'fisher_inv': fisher_inv.numpy(),
        'gradients': {t: g.numpy() for t, g in gradients.items()},
        'estimates': estimates, 'counts': counts,
        't_eval': t_eval, 'P': P,
        'stationarity_ratio': ratio.item() if torch.is_tensor(ratio) else ratio,
        'mean_score_norm': mean_score_norm,
        'mean_indiv_norm': mean_indiv_norm,
        'null_fracs': null_fracs,
    }
    if sandwich:
        ret['hessian'] = J.numpy()
        ret['hessian_inv'] = J_inv.numpy()
        ret['sandwich_cov'] = sandwich_cov.numpy()
        ret['bayes_cov'] = bayes_cov.numpy()

    return ret


# ─────────────────────────────────────────────────────────
# 6. Multi-simulation aggregation
# ─────────────────────────────────────────────────────────

def aggregate_simulations(all_results, t_eval):
    """Aggregate across D simulations at eval times."""
    D = len(all_results)
    summary = {}

    # --- Aggregate stationarity and score diagnostics ---
    stationarity_ratios = [r.get('stationarity_ratio', np.nan) for r in all_results]
    mean_score_norms = [r.get('mean_score_norm', np.nan) for r in all_results]
    mean_indiv_norms = [r.get('mean_indiv_norm', np.nan) for r in all_results]
    summary['_diagnostics'] = {
        'mean_stationarity_ratio': np.nanmean(stationarity_ratios),
        'std_stationarity_ratio': np.nanstd(stationarity_ratios),
        'mean_score_norm': np.nanmean(mean_score_norms),
        'std_score_norm': np.nanstd(mean_score_norms),
        'mean_indiv_norm': np.nanmean(mean_indiv_norms),
        'std_indiv_norm': np.nanstd(mean_indiv_norms),
    }

    for t in t_eval:
        t_key = float(t)

        ests = np.array([r['results'][t_key]['estimate']
                         for r in all_results if t_key in r['results']])
        ses = np.array([r['results'][t_key]['se']
                        for r in all_results if t_key in r['results']])
        true_vals = np.array([r['results'][t_key]['true']
                              for r in all_results if t_key in r['results']])
        ci_los = np.array([r['results'][t_key]['ci_lo']
                           for r in all_results if t_key in r['results']])
        ci_his = np.array([r['results'][t_key]['ci_hi']
                           for r in all_results if t_key in r['results']])

        # Null space fractions
        null_fracs_t = np.array([r.get('null_fracs', {}).get(t_key, np.nan)
                                  for r in all_results])

        D_t = len(ests)
        if D_t == 0:
            continue

        mean_true = true_vals.mean()
        mean_hat = ests.mean()
        bias = mean_hat - mean_true
        var_mc = ests.var(ddof=1) if D_t > 1 else float('nan')
        mean_var_est = (ses ** 2).mean()
        mse = bias ** 2 + var_mc if D_t > 1 else float('nan')

        coverage = np.mean(
            (ci_los <= true_vals) & (true_vals <= ci_his)
        ) if D_t > 1 else float('nan')

        summary[t_key] = {
            'D': D_t, 'mean_hat': mean_hat, 'true': mean_true,
            'bias': bias, 'var_mc': var_mc,
            'mean_var_est': mean_var_est, 'mse': mse,
            'coverage95': coverage,
            'mean_null_frac': np.nanmean(null_fracs_t),
            'std_null_frac': np.nanstd(null_fracs_t),
        }

    return summary


# ─────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pyreadr
    from torch.utils.data import DataLoader
    from dataset import LongitudinalDataset, collate_pad
    from model_ODE_cumulative import NeuralODEModel, NeuralODEConfig

    parser = argparse.ArgumentParser(
        description="Full-parameter delta method — continuous-time PDP")
    parser.add_argument("--n_sims", type=int, default=100)
    parser.add_argument("--true_coeff", type=float, default=-0.05)
    parser.add_argument("--oracle_mode", type=str, default="cumulative",
                        choices=["cumulative", "instantaneous"])
    parser.add_argument("--output_csv", type=str,
                        default="results_simu/simulation_cumulative_noreg_summary_configB.csv")
    parser.add_argument("--data_dir", type=str, default="simu_datasets/S5_sims")
    parser.add_argument("--ckpt_dir", type=str,
                        default="checkpoints/model_selection_S2")
    parser.add_argument("--bmi_pairs", type=str, default=None)
    parser.add_argument("--sandwich", action="store_true")
    parser.add_argument("--n_hessian_subsample", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_reg", type=float, default=0.5)
    parser.add_argument("--reg_mode", type=str, default=None,
                        choices=[None, "skip_gate", "group_lasso"])
    parser.add_argument("--t_eval", type=str, default="0, 2, 4, 6, 8, 10, 12",
                        help="Comma-separated eval times")
    parser.add_argument("--hidden_channels", type=int, default=4)
    parser.add_argument("--batch_size_pdp", type=int, default=64,
                        help="Batch size for PDP gradient computation (default: 64)")
    parser.add_argument("--ode_solver", type=str, default="rk4")
    parser.add_argument("--euler_steps", type=int, default=4)
    parser.add_argument("--ledoit_wolf", action="store_true",
                        help="Apply Ledoit-Wolf shrinkage to Fisher before inversion")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Evaluation grid ---
    t_eval = np.array([float(x) for x in args.t_eval.split(',')], dtype=float)

    # --- BMI pairs ---
    if args.bmi_pairs:
        bmi_pairs = [tuple(map(float, p.split(':')))
                     for p in args.bmi_pairs.split(',')]
    else:
        grid = [20, 23, 26, 29, 32, 35]
        bmi_pairs = [(grid[i], grid[i+1]) for i in range(len(grid)-1)]
        bmi_pairs.append((23, 32))

    def make_oracle_fn(mode, coeff, bmi_lo, bmi_hi):
        delta_v = bmi_hi - bmi_lo
        if mode == "cumulative":
            return lambda t: coeff * delta_v * t
        else:
            return lambda t: coeff * delta_v

    time_col, y_col, id_col = "time", "ISA15_sim", "NUM_ID"
    x_cols = ["BMI_t"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    all_pair_results = {pair: [] for pair in bmi_pairs}

    ckpt_dir = os.path.dirname(args.output_csv) or '.'
    ckpt_file = os.path.join(ckpt_dir, "delta_cumulative_checkpoint_sel.pt")
    start_sim = 0

    if os.path.exists(ckpt_file):
        print(f"Found checkpoint: {ckpt_file}")
        ckpt_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        all_pair_results = ckpt_data['all_pair_results']
        start_sim = ckpt_data['completed_up_to'] + 1
        print(f"Resuming from simulation {start_sim}")

    for sim_idx in range(start_sim, args.n_sims):
        if args.n_sims > 1:
            data_path = f"{args.data_dir}/sim_{sim_idx+1:03d}.rds"
            ckpt_path = f"{args.ckpt_dir}/B_both_no_reg_sim00{sim_idx}.pt"
            print(f"\n{'#'*60}")
            print(f"# SIMULATION {sim_idx}")
            print(f"{'#'*60}")
        else:
            data_path = f"{args.data_dir}/sim_001.rds"
            ckpt_path = f"{args.ckpt_dir}/best_model_ode_0.pt"

        df = next(iter(pyreadr.read_r(data_path).values()))
        df["SEX"] = df["SEX"].astype("category")
        df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
        df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
        df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

        dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col,
                                      static_cols=static_cols)

        cfg = NeuralODEConfig(
            hidden_channels=args.hidden_channels,
            enc_mlp_hidden=16, func_mlp_hidden=16,
            dec_rho_hidden=16, dec_p=4, dec_q=3, depth=2, dropout=0.0,
            euler_steps_per_interval=4,
            ode_solver=args.ode_solver,
        )
        model = NeuralODEModel(
            x_dim=len(x_cols), static_dim=len(static_cols), cfg=cfg,
            n_tv=1, use_rho_net=True, use_neural_re=True,
            re_spline_cols=None, g_hidden=8, fullD=False,
            bmi_mean=0.0, bmi_std=1.0,
            use_bmi_skip=True, static_skip_dims=[0,1,2,3],
            reg_mode=args.reg_mode,
        ).to(device)

        checkpoint = torch.load(ckpt_path, map_location=device,
                                weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded: {ckpt_path}")
        print(model.decoder.bmi_mean.item(), model.decoder.bmi_std.item())

        for bmi_lo, bmi_hi in bmi_pairs:
            oracle_fn = make_oracle_fn(
                args.oracle_mode, args.true_coeff, bmi_lo, bmi_hi)

            result = compute_full_delta_variance(
                model, dataset, device, collate_pad,
                bmi_lo=bmi_lo, bmi_hi=bmi_hi,
                t_eval=t_eval,
                oracle_fn=oracle_fn,
                sandwich=args.sandwich,
                n_hessian_subsample=args.n_hessian_subsample,
                weight_decay=args.weight_decay,
                lambda_reg=args.lambda_reg,
                batch_size_pdp=args.batch_size_pdp,
                ledoit_wolf=args.ledoit_wolf,
            )
            all_pair_results[(bmi_lo, bmi_hi)].append(result)

        torch.save({
            'all_pair_results': all_pair_results,
            'completed_up_to': sim_idx,
        }, ckpt_file)
        print(f"  [Checkpoint] Saved after simulation {sim_idx}")

    # --- Aggregate & CSV ---
    os.makedirs(os.path.dirname(args.output_csv)
                if os.path.dirname(args.output_csv) else '.', exist_ok=True)

    header = ['time', 'BMI_lo', 'BMI_hi', 'D', 'mean_hat', 'mean_true',
              'bias', 'var_mc', 'mean_var_est', 'mse', 'coverage95',
              'mean_null_frac']
    csv_rows = []

    print(f"\n{'='*90}")
    print(f"AGGREGATED — Continuous-Time Delta Method (D={args.n_sims})")
    print(f"{'='*90}")
    print(f"{'t':>6s} {'lo':>4s} {'hi':>4s} {'D':>3s}  {'mean_hat':>10s} "
          f"{'mean_true':>10s} {'bias':>8s}  {'var_mc':>10s} {'var_est':>10s} "
          f"{'mse':>10s} {'cov95':>6s} {'null%':>6s}")
    print(f"{'-'*100}")

    for (bmi_lo, bmi_hi), results_list in all_pair_results.items():
        D = len(results_list)
        if D == 0:
            continue
        summary = aggregate_simulations(results_list, t_eval)

        # --- Print diagnostics ---
        diag = summary.get('_diagnostics', {})
        if diag:
            print(f"\n  Diagnostics (averaged over {D} replicates):")
            print(f"    ||mean(φ)||          = {diag['mean_score_norm']:.4e} "
                  f"± {diag['std_score_norm']:.4e}")
            print(f"    mean(||φ_i||)        = {diag['mean_indiv_norm']:.4e} "
                  f"± {diag['std_indiv_norm']:.4e}")
            print(f"    stationarity ratio   = {diag['mean_stationarity_ratio']:.4f} "
                  f"± {diag['std_stationarity_ratio']:.4f}")

        for t in t_eval:
            t_key = float(t)
            if t_key not in summary:
                continue
            s = summary[t_key]

            row = {
                'time': round(t, 2),
                'BMI_lo': int(bmi_lo), 'BMI_hi': int(bmi_hi),
                'D': s['D'],
                'mean_hat': s['mean_hat'], 'mean_true': s['true'],
                'bias': s['bias'], 'var_mc': s['var_mc'],
                'mean_var_est': s['mean_var_est'], 'mse': s['mse'],
                'coverage95': s['coverage95'],
                'mean_null_frac': s.get('mean_null_frac', float('nan')),
            }
            csv_rows.append(row)

            vm = f"{s['var_mc']:.6f}" if not np.isnan(s['var_mc']) else "NA"
            ms = f"{s['mse']:.6f}" if not np.isnan(s['mse']) else "NA"
            cv = f"{s['coverage95']:.2f}" if not np.isnan(s['coverage95']) else "NA"
            nf = f"{s.get('mean_null_frac', float('nan')):.4f}"
            print(f"{t:6.1f} {int(bmi_lo):4d} {int(bmi_hi):4d} "
                  f"{s['D']:3d}  {s['mean_hat']:+10.4f} "
                  f"{s['true']:+10.4f} {s['bias']:+8.4f}  "
                  f"{vm:>10s} {s['mean_var_est']:10.6f} "
                  f"{ms:>10s} {cv:>6s} {nf:>6s}")

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved to {args.output_csv}")