"""
Advanced variance estimation for ∆PDP — Neural ODE-LMM.

Two methods beyond the first-order delta method:

1. SECOND-ORDER DELTA METHOD
   Var(∆PDP) ≈ gᵀ Σ_θ g + ½ tr(H_∆ Σ_θ H_∆ Σ_θ)

   where H_∆ = ∇²_θ ∆PDP(θ) is the P×P Hessian of the estimand.
   The trace term is estimated stochastically via Hutchinson:
       tr(A B A B) ≈ (1/K) Σ_k  vₖᵀ A B A B vₖ
   using Hessian-vector products (no full H_∆ needed).

2. PARAMETRIC BOOTSTRAP
   Sample θ⁽ᵇ⁾ ~ N(θ̂, F⁻¹), compute ∆PDP(θ⁽ᵇ⁾) for each draw.
   Var = sample variance of {∆PDP(θ⁽ᵇ⁾)}.
   No Taylor expansion — captures all nonlinearity in θ ↦ ∆PDP.

Usage:
    python PDP_variance_advanced.py --method both --n_sims 1
    python PDP_variance_advanced.py --method bootstrap --n_bootstrap 500
    python PDP_variance_advanced.py --method second_order --n_hutchinson 50
"""
from __future__ import annotations
import math, os, csv, time, copy
import torch
import torch.nn.functional as F_torch
import numpy as np
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────
# Helpers (shared with PDP_Full_delta_method.py)
# ─────────────────────────────────────────────────────────

def _param_list(model):
    """Ordered list of parameters that require grad."""
    return [p for p in model.parameters() if p.requires_grad]


def _param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _cat_grads(grads):
    """Flatten and concatenate a tuple of gradient tensors."""
    return torch.cat([g.reshape(-1) for g in grads])


def _flat_params(model):
    """Return a flat vector of all trainable parameters."""
    return torch.cat([p.detach().reshape(-1) for p in _param_list(model)])


def _load_flat_params(model, flat_vec):
    """Load a flat parameter vector back into the model."""
    offset = 0
    for p in _param_list(model):
        n = p.numel()
        p.data.copy_(flat_vec[offset:offset + n].reshape(p.shape))
        offset += n


# ─────────────────────────────────────────────────────────
# Compute ∆PDP at current parameters (no grad needed)
# ─────────────────────────────────────────────────────────

def compute_delta_pdp_values(model, loader, device,
                              bmi_lo, bmi_hi, visit_times):
    """
    Compute ∆PDP point estimates at current model parameters.
    No gradient computation — fast.

    Returns:
        estimates: dict {vt: float}
        counts:    dict {vt: int}
    """
    model.eval()
    est_accum = {vt: 0.0 for vt in visit_times}
    n_accum = {vt: 0 for vt in visit_times}

    with torch.no_grad():
        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad = t_pad.to(device)
            x_pad = x_pad.to(device)
            mask = mask.to(device)
            s = s.to(device)

            B, T = t_pad.shape

            x_hi = x_pad.clone()
            x_hi[:, :, 0] = bmi_hi
            mu_hi, _, _, _, _, _ = model(
                t_pad, x_hi, masks=None,
                static_covariates=s, bmi_t=x_hi[:, :, 0:1], obs_mask=mask
            )

            x_lo = x_pad.clone()
            x_lo[:, :, 0] = bmi_lo
            mu_lo, _, _, _, _, _ = model(
                t_pad, x_lo, masks=None,
                static_covariates=s, bmi_t=x_lo[:, :, 0:1], obs_mask=mask
            )

            for vt in visit_times:
                for i in range(B):
                    obs_idx = torch.where(mask[i] > 0.5)[0]
                    if len(obs_idx) == 0:
                        continue
                    obs_times = t_pad[i, obs_idx]
                    closest = obs_idx[torch.argmin(torch.abs(obs_times - vt))]
                    est_accum[vt] += (mu_hi[i, closest] - mu_lo[i, closest]).item()
                    n_accum[vt] += 1

    estimates = {}
    for vt in visit_times:
        estimates[vt] = est_accum[vt] / n_accum[vt] if n_accum[vt] > 0 else 0.0
    return estimates, n_accum


# ─────────────────────────────────────────────────────────
# 3. ∆PDP gradient  g_ℓ = ∇_θ  ∆PDP_ℓ
# ─────────────────────────────────────────────────────────

def compute_delta_pdp_gradient(model, loader, device,
                                 bmi_lo, bmi_hi,
                                 visit_times,
                                 verbose=True):
    """
    Compute g_ℓ = ∇_θ ∆PDP_ℓ for each visit time.

    ∆PDP_ℓ = (1/n_ℓ) Σ_i [μ^{hi}_{i,c(i,ℓ)} − μ^{lo}_{i,c(i,ℓ)}]

    The gradient is accumulated across batches by linearity:
        g_ℓ = (1/n_ℓ) Σ_batch  ∇_θ  Σ_{i∈batch} δ_{i,ℓ}

    The computation graph spans both forward passes (hi and lo)
    because they share the same model parameters.

    Also returns the ∆PDP point estimates for each visit time.

    Returns:
        gradients: dict {vt: g_ℓ ∈ R^P}  (CPU)
        estimates: dict {vt: float}        ∆PDP estimates
        counts:    dict {vt: int}          subjects per visit time
    """
    model.eval()
    params = _param_list(model)
    P = sum(p.numel() for p in params)

    g_accum = {vt: torch.zeros(P) for vt in visit_times}
    est_accum = {vt: 0.0 for vt in visit_times}
    n_accum = {vt: 0 for vt in visit_times}

    for batch in loader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        mask = mask.to(device)
        s = s.to(device)

        B, T = t_pad.shape

        # --- Two counterfactual forward passes (with grad) ---
        x_hi = x_pad.clone()
        x_hi[:, :, 0] = bmi_hi
        mu_hi, _, _, _, _, _ = model(
            t_pad, x_hi, masks=None,
            static_covariates=s, bmi_t=x_hi[:, :, 0:1], obs_mask=mask
        )

        x_lo = x_pad.clone()
        x_lo[:, :, 0] = bmi_lo
        mu_lo, _, _, _, _, _ = model(
            t_pad, x_lo, masks=None,
            static_covariates=s, bmi_t=x_lo[:, :, 0:1], obs_mask=mask
        )

        # --- For each visit time, build differentiable sum ---
        for vt_idx, vt in enumerate(visit_times):
            delta_sum = torch.tensor(0.0, device=device)
            n_vt = 0

            for i in range(B):
                obs_idx = torch.where(mask[i] > 0.5)[0]
                if len(obs_idx) == 0:
                    continue
                obs_times = t_pad[i, obs_idx]
                closest = obs_idx[torch.argmin(torch.abs(obs_times - vt))]
                delta_sum = delta_sum + (mu_hi[i, closest] - mu_lo[i, closest])
                n_vt += 1

            if n_vt > 0:
                # Backward — keep graph for remaining visit times
                # allow_unused: g_net, L_unconstrained, log_residual_var
                # are not in the mu computation graph
                retain = (vt_idx < len(visit_times) - 1)
                grads = torch.autograd.grad(delta_sum, params,
                                            retain_graph=retain,
                                            allow_unused=True)
                # Replace None grads (unused params) with zeros
                grads = [g if g is not None else torch.zeros_like(p)
                         for g, p in zip(grads, params)]
                g_batch = _cat_grads(grads).cpu()

                g_accum[vt] += g_batch
                est_accum[vt] += delta_sum.item()
                n_accum[vt] += n_vt

    # Normalise
    gradients = {}
    estimates = {}
    for vt in visit_times:
        n = n_accum[vt]
        if n > 0:
            gradients[vt] = g_accum[vt] / n
            estimates[vt] = est_accum[vt] / n
        else:
            gradients[vt] = torch.zeros(P)
            estimates[vt] = 0.0

    if verbose:
        for vt in visit_times:
            print(f"    g_{int(vt)}: ||g|| = {gradients[vt].norm().item():.4f}, "
                  f"n = {n_accum[vt]}, ∆PDP = {estimates[vt]:.4f}")

    return gradients, estimates, n_accum


# ─────────────────────────────────────────────────────────
# Per-subject NLL (for Fisher computation)
# ─────────────────────────────────────────────────────────

def _per_subject_nll(mu, V, y_pad, mask, jitter=1e-4):
    """Masked Gaussian NLL for ONE subject (B=1)."""
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
# Empirical Fisher
# ─────────────────────────────────────────────────────────

def compute_empirical_fisher(model, dataset, device, collate_fn,
                              verbose=True):
    """
    Compute F = Σ_i s_i s_iᵀ and return score matrix S.
    Returns:
        fisher: (P, P) tensor on CPU
        scores: (N, P) tensor on CPU
    """
    from torch.utils.data import DataLoader

    model.eval()
    params = _param_list(model)
    P = sum(p.numel() for p in params)
    fisher = torch.zeros(P, P)
    score_list = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn)
    N = len(dataset)
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
            score_list.append(torch.zeros(P))
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
        s_i = _cat_grads(grads).cpu()

        fisher += torch.outer(s_i, s_i)
        score_list.append(s_i)

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


# ─────────────────────────────────────────────────────────
# Regularize and invert Fisher
# ─────────────────────────────────────────────────────────

def regularize_and_invert_fisher(fisher, lam=1e-4, verbose=True):
    """
    Marquardt damping + inversion.
    Returns:
        fisher_reg: (P, P)
        fisher_inv: (P, P)
    """
    diag_F = torch.diag(fisher)
    diag_F = torch.clamp(diag_F, min=1e-4 * diag_F.max())
    fisher_reg = fisher + lam * torch.diag(diag_F)

    if verbose:
        print(f"  Marquardt λ = {lam}")
        print(f"  Condition number: {torch.linalg.cond(fisher_reg).item():.2e}")

    try:
        fisher_inv = torch.linalg.inv(fisher_reg)
    except torch.linalg.LinAlgError:
        print("  WARNING: Fisher not invertible, using pseudo-inverse")
        fisher_inv = torch.linalg.pinv(fisher_reg)

    return fisher_reg, fisher_inv


# =============================================================
# METHOD 1: SECOND-ORDER DELTA METHOD
# =============================================================
#
# Var(∆PDP) ≈ gᵀ Σ g  +  ½ tr(H Σ H Σ)
#
# where Σ = F⁻¹, g = ∇_θ ∆PDP, H = ∇²_θ ∆PDP.
#
# The trace term is estimated via Hutchinson:
#   tr(H Σ H Σ) ≈ (1/K) Σ_k  vₖᵀ H Σ H Σ vₖ
#
# Each probe requires:
#   w₁ = Σ vₖ          (matrix-vector, cheap)
#   w₂ = H w₁          (Hessian-vector product via autograd)
#   w₃ = Σ w₂          (matrix-vector, cheap)
#   w₄ = H w₃          (Hessian-vector product via autograd)
#   estimate_k = vₖᵀ w₄
#
# The Hessian-vector product H·u is computed by differentiating
# the gradient-vector product: H·u = ∂/∂θ (gᵀ u) where
# g = ∇_θ ∆PDP is recomputed with create_graph=True.
# =============================================================

def _hvp_delta_pdp(model, loader, device, bmi_lo, bmi_hi,
                    vt, u_vec, visit_times):
    """
    Compute H_∆ · u  for a single visit time, where H_∆ = ∇²_θ ∆PDP_ℓ.

    Uses the identity: H·u = ∂/∂θ [ (∇_θ ∆PDP)ᵀ u ]

    Args:
        model: the model (parameters must have requires_grad=True)
        loader: data loader
        device: torch device
        bmi_lo, bmi_hi: BMI intervention values
        vt: the visit time for this HVP
        u_vec: (P,) direction vector (CPU)
        visit_times: all visit times (for batch processing)

    Returns:
        hv: (P,) tensor on CPU — the Hessian-vector product H·u
    """
    model.eval()
    params = _param_list(model)
    P = sum(p.numel() for p in params)

    # Assign u to parameter-shaped chunks for the dot product
    u_parts = []
    offset = 0
    for p in params:
        n = p.numel()
        u_parts.append(u_vec[offset:offset + n].reshape(p.shape).to(device))
        offset += n

    # Accumulate ∇_θ(gᵀu) = H·u across batches
    hv_accum = torch.zeros(P)
    n_total = 0

    for batch in loader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        mask = mask.to(device)
        s = s.to(device)
        B, T = t_pad.shape

        x_hi = x_pad.clone()
        x_hi[:, :, 0] = bmi_hi
        mu_hi, _, _, _, _, _ = model(
            t_pad, x_hi, masks=None,
            static_covariates=s, bmi_t=x_hi[:, :, 0:1], obs_mask=mask
        )

        x_lo = x_pad.clone()
        x_lo[:, :, 0] = bmi_lo
        mu_lo, _, _, _, _, _ = model(
            t_pad, x_lo, masks=None,
            static_covariates=s, bmi_t=x_lo[:, :, 0:1], obs_mask=mask
        )

        # Build differentiable ∆PDP sum for this visit time
        delta_sum = torch.tensor(0.0, device=device)
        n_vt = 0
        for i in range(B):
            obs_idx = torch.where(mask[i] > 0.5)[0]
            if len(obs_idx) == 0:
                continue
            obs_times = t_pad[i, obs_idx]
            closest = obs_idx[torch.argmin(torch.abs(obs_times - vt))]
            delta_sum = delta_sum + (mu_hi[i, closest] - mu_lo[i, closest])
            n_vt += 1

        if n_vt == 0:
            continue

        # First-order gradient with graph retained
        grads = torch.autograd.grad(delta_sum, params,
                                    create_graph=True,
                                    retain_graph=True,
                                    allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, params)]

        # Dot product gᵀu (scalar, differentiable)
        g_dot_u = sum((g * u_p).sum()
                      for g, u_p in zip(grads, u_parts))

        # Second-order: ∂/∂θ (gᵀu) = H·u
        hv_grads = torch.autograd.grad(g_dot_u, params,
                                        retain_graph=False,
                                        allow_unused=True)
        hv_grads = [g if g is not None else torch.zeros_like(p)
                    for g, p in zip(hv_grads, params)]
        hv_batch = _cat_grads(hv_grads).cpu()

        hv_accum += hv_batch
        n_total += n_vt

    if n_total > 0:
        hv_accum /= n_total

    return hv_accum


def compute_second_order_variance(model, loader, device, collate_fn, dataset,
                                   bmi_lo, bmi_hi, visit_times,
                                   fisher_inv,
                                   gradients, estimates,
                                   n_hutchinson=30,
                                   verbose=True):
    """
    Second-order delta method variance via Hutchinson trace estimator.

    Var(∆PDP_ℓ) ≈ gᵀ Σ g  +  ½ tr(H Σ H Σ)

    where Σ = F⁻¹ (fisher_inv), g = gradients, H via HVP.

    Args:
        fisher_inv: (P, P) tensor on CPU
        gradients:  dict {vt: g ∈ R^P} on CPU
        estimates:  dict {vt: float}
        n_hutchinson: number of Hutchinson probes

    Returns:
        results: dict {vt: {var_1st, var_2nd, var_total, trace_term, ...}}
    """
    P = fisher_inv.shape[0]
    Sigma = fisher_inv  # alias

    if verbose:
        print(f"\n{'='*60}")
        print(f"SECOND-ORDER DELTA METHOD")
        print(f"{'='*60}")
        print(f"  P = {P}, K = {n_hutchinson} Hutchinson probes")

    results = {}

    for vt in visit_times:
        g = gradients[vt]
        est = estimates[vt]

        # First-order term
        var_1st = (g @ Sigma @ g).item()

        # Hutchinson trace estimation:  tr(H Σ H Σ)
        # Each probe:  vᵀ H Σ H Σ v
        trace_estimates = []

        t0 = time.time()
        for k in range(n_hutchinson):
            # Rademacher probe vector
            v = torch.sign(torch.randn(P))

            # w1 = Σ v
            w1 = (Sigma @ v)

            # w2 = H w1  (Hessian-vector product)
            w2 = _hvp_delta_pdp(model, loader, device,
                                bmi_lo, bmi_hi, vt, w1, visit_times)

            # w3 = Σ w2
            w3 = (Sigma @ w2)

            # w4 = H w3  (Hessian-vector product)
            w4 = _hvp_delta_pdp(model, loader, device,
                                bmi_lo, bmi_hi, vt, w3, visit_times)

            # vᵀ w4 = vᵀ H Σ H Σ v
            trace_k = (v @ w4).item()
            trace_estimates.append(trace_k)

            if verbose and (k + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"    t={int(vt)}: probe {k+1}/{n_hutchinson} "
                      f"({elapsed:.1f}s), running trace = "
                      f"{np.mean(trace_estimates):.6f}")

        trace_term = np.mean(trace_estimates)
        trace_se = np.std(trace_estimates) / np.sqrt(n_hutchinson)

        # Second-order correction
        var_2nd = 0.5 * trace_term
        var_total = var_1st + var_2nd

        se_1st = np.sqrt(max(var_1st, 0.0))
        se_total = np.sqrt(max(var_total, 0.0))

        results[vt] = {
            'estimate': est,
            'var_1st': var_1st,
            'var_2nd': var_2nd,
            'var_total': var_total,
            'se_1st': se_1st,
            'se_total': se_total,
            'trace_term': trace_term,
            'trace_se': trace_se,
        }

        if verbose:
            print(f"  t={int(vt)}: var_1st={var_1st:.6f}, "
                  f"trace={trace_term:.6f} ± {trace_se:.6f}, "
                  f"var_2nd_correction={var_2nd:.6f}, "
                  f"var_total={var_total:.6f}")
            print(f"         SE_1st={se_1st:.4f}, SE_total={se_total:.4f}")

    return results


# =============================================================
# METHOD 2: PARAMETRIC BOOTSTRAP
# =============================================================
#
# Sample θ⁽ᵇ⁾ ~ N(θ̂, F⁻¹), compute ∆PDP(θ⁽ᵇ⁾), take variance.
# No Taylor expansion — captures full nonlinearity of θ ↦ ∆PDP.
# =============================================================

def compute_parametric_bootstrap_variance(
    model, loader, device,
    bmi_lo, bmi_hi, visit_times,
    fisher_inv,
    n_bootstrap=500,
    verbose=True,
):
    """
    Parametric bootstrap variance for ∆PDP.

    Samples θ⁽ᵇ⁾ ~ N(θ̂, F⁻¹), evaluates ∆PDP at each draw.

    Args:
        fisher_inv: (P, P) tensor on CPU
        n_bootstrap: number of bootstrap draws

    Returns:
        results: dict {vt: {var_boot, se_boot, estimates_boot, ...}}
    """
    P = fisher_inv.shape[0]

    if verbose:
        print(f"\n{'='*60}")
        print(f"PARAMETRIC BOOTSTRAP")
        print(f"{'='*60}")
        print(f"  P = {P}, B = {n_bootstrap} draws")

    # Save original parameters
    theta_hat = _flat_params(model)  # (P,)

    # Cholesky of F⁻¹ for sampling: θ⁽ᵇ⁾ = θ̂ + L z, z ~ N(0, I)
    try:
        # Ensure positive definiteness
        eigvals = torch.linalg.eigvalsh(fisher_inv)
        min_eig = eigvals.min().item()
        if min_eig < 0:
            # Add small ridge to make PD
            ridge = abs(min_eig) + 1e-6
            fisher_inv_pd = fisher_inv + ridge * torch.eye(P)
            if verbose:
                print(f"  F⁻¹ has negative eigenvalue ({min_eig:.2e}), "
                      f"added ridge {ridge:.2e}")
        else:
            fisher_inv_pd = fisher_inv

        L_cov = torch.linalg.cholesky(fisher_inv_pd)
    except torch.linalg.LinAlgError:
        print("  WARNING: Cholesky of F⁻¹ failed, using eigendecomposition")
        eigvals, eigvecs = torch.linalg.eigh(fisher_inv)
        eigvals = torch.clamp(eigvals, min=1e-10)
        L_cov = eigvecs @ torch.diag(torch.sqrt(eigvals))

    if verbose:
        print(f"  ||θ̂|| = {theta_hat.norm().item():.4f}")
        print(f"  ||L_cov|| = {L_cov.norm().item():.4f}")
        print(f"  max perturbation scale ~ {L_cov.norm(dim=0).max().item():.4f}")

    # Point estimate at θ̂
    est_hat, _ = compute_delta_pdp_values(model, loader, device,
                                           bmi_lo, bmi_hi, visit_times)

    # Bootstrap draws
    boot_estimates = {vt: [] for vt in visit_times}

    t0 = time.time()
    for b in range(n_bootstrap):
        # Sample θ⁽ᵇ⁾ = θ̂ + L z
        z_draw = torch.randn(P)
        theta_b = theta_hat + L_cov @ z_draw

        # Load perturbed parameters
        _load_flat_params(model, theta_b)

        # Compute ∆PDP
        est_b, _ = compute_delta_pdp_values(model, loader, device,
                                             bmi_lo, bmi_hi, visit_times)

        for vt in visit_times:
            boot_estimates[vt].append(est_b[vt])

        if verbose and (b + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (b + 1) / elapsed
            eta = (n_bootstrap - b - 1) / rate
            print(f"    Bootstrap: {b+1}/{n_bootstrap} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # Restore original parameters
    _load_flat_params(model, theta_hat)

    # Compute statistics
    results = {}
    for vt in visit_times:
        boot_arr = np.array(boot_estimates[vt])
        var_boot = np.var(boot_arr, ddof=1)
        se_boot = np.sqrt(var_boot)
        mean_boot = np.mean(boot_arr)
        median_boot = np.median(boot_arr)

        # Percentile CI
        ci_lo = np.percentile(boot_arr, 2.5)
        ci_hi = np.percentile(boot_arr, 97.5)

        # Skewness and kurtosis — diagnostics for nonlinearity
        skew = float(np.mean(((boot_arr - mean_boot) / se_boot) ** 3)) \
            if se_boot > 0 else 0.0
        kurt = float(np.mean(((boot_arr - mean_boot) / se_boot) ** 4) - 3.0) \
            if se_boot > 0 else 0.0

        results[vt] = {
            'estimate': est_hat[vt],
            'var_boot': var_boot,
            'se_boot': se_boot,
            'mean_boot': mean_boot,
            'median_boot': median_boot,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'skewness': skew,
            'excess_kurtosis': kurt,
            'boot_samples': boot_arr,
        }

        if verbose:
            print(f"  t={int(vt)}: est={est_hat[vt]:.4f}, "
                  f"var_boot={var_boot:.6f}, SE_boot={se_boot:.4f}")
            print(f"         CI_95=[{ci_lo:.4f}, {ci_hi:.4f}], "
                  f"skew={skew:.3f}, kurt={kurt:.3f}")

    if verbose:
        print(f"  Total time: {time.time() - t0:.1f}s")

    return results


# =============================================================
# MAIN: run both methods and compare
# =============================================================

def run_variance_comparison(
    model, dataset, loader, device, collate_fn,
    bmi_lo=20.0, bmi_hi=35.0,
    visit_times=np.array([0, 5, 10, 15]),
    true_beta_bmi=-0.30,
    true_beta_int=-0.05,
    n_hutchinson=30,
    n_bootstrap=500,
    method="both",
    verbose=True,
):
    """
    Run first-order, second-order, and/or parametric bootstrap variance.

    Args:
        method: "second_order", "bootstrap", or "both"

    Returns:
        dict with all results
    """
    P = _param_count(model)
    delta_v = bmi_hi - bmi_lo

    print(f"\n{'='*60}")
    print(f"ADVANCED VARIANCE ESTIMATION")
    print(f"{'='*60}")
    print(f"  P = {P}, N = {len(dataset)}")
    print(f"  BMI: lo={bmi_lo}, hi={bmi_hi}")
    print(f"  Method: {method}")

    # --- Step 1: Fisher ---
    print(f"\nStep 1: Empirical Fisher ...")
    fisher, scores = compute_empirical_fisher(model, dataset, device, collate_fn,
                                               verbose=verbose)
    fisher_reg, fisher_inv = regularize_and_invert_fisher(fisher, verbose=verbose)

    # --- Step 2: Gradient ---
    print(f"\nStep 2: ∆PDP gradients ...")
    gradients, estimates, counts = compute_delta_pdp_gradient(
        model, loader, device, bmi_lo, bmi_hi, visit_times, verbose=verbose
    )

    # True ∆PDP
    all_ages = []
    for batch in loader:
        _, _, _, _, _, _, s = batch
        all_ages.append(s[:, 1])
    mean_age = torch.cat(all_ages).mean().item()
    true_delta = delta_v * (true_beta_bmi + true_beta_int * mean_age)

    # --- First-order (always computed as baseline) ---
    var_1st = {}
    for vt in visit_times:
        g = gradients[vt]
        var_1st[vt] = (g @ fisher_inv @ g).item()

    # --- Method-specific ---
    results_2nd = None
    results_boot = None

    if method in ("second_order", "both"):
        results_2nd = compute_second_order_variance(
            model, loader, device, collate_fn, dataset,
            bmi_lo, bmi_hi, visit_times,
            fisher_inv, gradients, estimates,
            n_hutchinson=n_hutchinson, verbose=verbose,
        )

    if method in ("bootstrap", "both"):
        results_boot = compute_parametric_bootstrap_variance(
            model, loader, device,
            bmi_lo, bmi_hi, visit_times,
            fisher_inv, n_bootstrap=n_bootstrap,
            verbose=verbose,
        )

    # --- Summary comparison ---
    print(f"\n{'='*70}")
    print(f"COMPARISON — true ∆PDP = {true_delta:.4f}")
    print(f"{'='*70}")

    header_parts = [f"{'t':>4s}", f"{'∆PDP':>10s}", f"{'SE_1st':>10s}"]
    if results_2nd is not None:
        header_parts.append(f"{'SE_2nd':>10s}")
    if results_boot is not None:
        header_parts.append(f"{'SE_boot':>10s}")
    header_parts.extend([f"{'True':>10s}", f"{'Bias':>10s}"])
    print("  " + "  ".join(header_parts))
    print(f"  {'-' * (len(header_parts) * 12)}")

    for vt in visit_times:
        est = estimates[vt]
        se_1st = np.sqrt(max(var_1st[vt], 0.0))
        bias = est - true_delta

        parts = [f"{int(vt):4d}", f"{est:+10.4f}", f"{se_1st:10.4f}"]
        if results_2nd is not None:
            parts.append(f"{results_2nd[vt]['se_total']:10.4f}")
        if results_boot is not None:
            parts.append(f"{results_boot[vt]['se_boot']:10.4f}")
        parts.extend([f"{true_delta:+10.4f}", f"{bias:+10.4f}"])
        print("  " + "  ".join(parts))

    # Nonlinearity diagnostic
    if results_boot is not None and results_2nd is not None:
        print(f"\n  Nonlinearity diagnostics:")
        for vt in visit_times:
            ratio_2nd = results_2nd[vt]['var_total'] / max(var_1st[vt], 1e-12)
            ratio_boot = results_boot[vt]['var_boot'] / max(var_1st[vt], 1e-12)
            print(f"    t={int(vt)}: var_2nd/var_1st = {ratio_2nd:.3f}, "
                  f"var_boot/var_1st = {ratio_boot:.3f}, "
                  f"skew = {results_boot[vt]['skewness']:.3f}")

    return {
        'var_1st': var_1st,
        'results_2nd': results_2nd,
        'results_boot': results_boot,
        'fisher_inv': fisher_inv.numpy(),
        'gradients': {vt: g.numpy() for vt, g in gradients.items()},
        'estimates': estimates,
        'true_delta': true_delta,
        'mean_age': mean_age,
    }


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pyreadr
    from torch.utils.data import DataLoader
    from dataset import LongitudinalDataset, collate_pad
    from model_ODE import NeuralODEModel, NeuralODEConfig

    parser = argparse.ArgumentParser(
        description="Advanced variance estimation for ∆PDP")
    parser.add_argument("--n_sims", type=int, default=1)
    parser.add_argument("--data", type=str,
                        default="simu_datasets/S2a_sims/sim_001.rds")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/simulation_baseline_euler_4_ReLU/best_model_ode_0.pt")
    parser.add_argument("--method", type=str, default="both",
                        choices=["second_order", "bootstrap", "both"])
    parser.add_argument("--n_hutchinson", type=int, default=30)
    parser.add_argument("--n_bootstrap", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--true_beta_bmi", type=float, default=-0.30)
    parser.add_argument("--true_beta_int", type=float, default=-0.05)
    parser.add_argument("--output_csv", type=str,
                        default="results/variance_advanced.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visit_times = np.array([0, 5, 10, 15])

    time_col, y_col, id_col = "time", "ISA15_sim", "NUM_ID"
    x_cols = ["BMI_t", "rs1", "rs2"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    all_sim_results = []
    # --- Checkpoint: resume from last completed simulation ---
    ckpt_dir = os.path.dirname(args.output_csv) or '.'
    ckpt_file = os.path.join(ckpt_dir, "advanced_delta_method_checkpoint.pt")
    start_sim = 0

    if os.path.exists(ckpt_file):
        print(f"Found checkpoint: {ckpt_file}")
        ckpt_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        all_sim_results = ckpt_data['all_sim_results']
        start_sim = ckpt_data['completed_up_to'] + 1
        print(f"Resuming from simulation {start_sim} "
              f"({start_sim}/{args.n_sims} already done)")

    for sim_idx in range(start_sim, args.n_sims):
        if args.n_sims > 1:
            data_path = f"simu_datasets/S2a_sims/sim_{sim_idx+1:03d}.rds"
            ckpt_path = f"checkpoints/simulation_baseline_euler_4_ReLU/best_model_ode_{sim_idx}.pt"
            print(f"\n{'#'*60}")
            print(f"# SIMULATION {sim_idx}")
            print(f"{'#'*60}")
        else:
            data_path = "simu_datasets/S2a_sims_2/sim_000.rds"
            ckpt_path = "checkpoints/best_model_ode_full_skip_0.pt"

        # --- Data ---
        df = next(iter(pyreadr.read_r(data_path).values()))
        df["SEX"] = df["SEX"].astype("category")
        df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
        df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
        df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

        dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col,
                                    static_cols=static_cols)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_pad)

        # --- Model ---
        cfg = NeuralODEConfig(
            hidden_channels=8, enc_mlp_hidden=32, func_mlp_hidden=32,
            dec_rho_hidden=16, dec_p=4, dec_q=3, depth=2, dropout=0.0,
            euler_steps_per_interval=4,
        )
        model = NeuralODEModel(
            x_dim=len(x_cols), static_dim=len(static_cols), cfg=cfg,
            n_tv=1, use_rho_net=True, use_neural_re=True,
            re_spline_cols=[1, 2], g_hidden=16, fullD=True,
            bmi_mean=0.0, bmi_std=1.0, static_skip_dims=[1],
        ).to(device)

        checkpoint = torch.load(ckpt_path, map_location=device,
                                weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded: {ckpt_path}")

        # --- Run ---
        all_results = run_variance_comparison(
            model, dataset, loader, device, collate_pad,
            bmi_lo=20.0, bmi_hi=35.0,
            visit_times=visit_times,
            true_beta_bmi=args.true_beta_bmi,
            true_beta_int=args.true_beta_int,
            n_hutchinson=args.n_hutchinson,
            n_bootstrap=args.n_bootstrap,
            method=args.method,
        )
        all_sim_results.append(all_results)
        # --- Checkpoint: save after each simulation ---
        torch.save({
            'all_sim_results':all_sim_results,
            'completed_up_to': sim_idx,
        }, ckpt_file)
        print(f"  [Checkpoint] Saved after simulation {sim_idx} → {ckpt_file}")

    # --- Save per-simulation CSV ---
    os.makedirs(os.path.dirname(args.output_csv)
                if os.path.dirname(args.output_csv) else '.', exist_ok=True)

    has_2nd = all_sim_results[0]['results_2nd'] is not None
    has_boot = all_sim_results[0]['results_boot'] is not None

    header = ['sim', 'time', 'estimate', 'true', 'bias',
              'var_1st', 'se_1st']
    if has_2nd:
        header += ['var_2nd_total', 'se_2nd', 'trace_term', 'trace_se']
    if has_boot:
        header += ['var_boot', 'se_boot', 'ci_lo_boot', 'ci_hi_boot',
                    'skewness', 'excess_kurtosis']

    rows = []
    for sim_i, all_results in enumerate(all_sim_results):
        for vt in visit_times:
            row = {
                'sim': sim_i,
                'time': int(vt),
                'estimate': all_results['estimates'][vt],
                'true': all_results['true_delta'],
                'bias': all_results['estimates'][vt] - all_results['true_delta'],
                'var_1st': all_results['var_1st'][vt],
                'se_1st': np.sqrt(max(all_results['var_1st'][vt], 0.0)),
            }
            if has_2nd and all_results['results_2nd'] is not None:
                r2 = all_results['results_2nd'][vt]
                row['var_2nd_total'] = r2['var_total']
                row['se_2nd'] = r2['se_total']
                row['trace_term'] = r2['trace_term']
                row['trace_se'] = r2['trace_se']
            if has_boot and all_results['results_boot'] is not None:
                rb = all_results['results_boot'][vt]
                row['var_boot'] = rb['var_boot']
                row['se_boot'] = rb['se_boot']
                row['ci_lo_boot'] = rb['ci_lo']
                row['ci_hi_boot'] = rb['ci_hi']
                row['skewness'] = rb['skewness']
                row['excess_kurtosis'] = rb['excess_kurtosis']
            rows.append(row)

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-simulation CSV saved to {args.output_csv}")

    # ─────────────────────────────────────────────────────────
    # Aggregate across simulations
    # ─────────────────────────────────────────────────────────
    n_sims_done = len(all_sim_results)
    true_delta = all_sim_results[0]['true_delta']

    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS over {n_sims_done} simulations")
    print(f"  True ∆PDP = {true_delta:.4f}")
    print(f"{'='*70}")

    for vt in visit_times:
        ests = np.array([r['estimates'][vt] for r in all_sim_results])
        vars_1st = np.array([r['var_1st'][vt] for r in all_sim_results])

        emp_var = np.var(ests, ddof=1)
        mean_var_1st = np.mean(vars_1st)
        ses_1st = np.sqrt(np.maximum(vars_1st, 0.0))
        covers_1st = np.abs(ests - true_delta) <= 1.96 * ses_1st
        cov_1st = np.mean(covers_1st)

        print(f"\n  t = {int(vt)}:")
        print(f"    Mean ∆PDP estimate   = {np.mean(ests):+.4f}")
        print(f"    Mean bias            = {np.mean(ests) - true_delta:+.4f}")
        print(f"    Empirical Var(∆PDP)  = {emp_var:.6f}")
        print(f"    Mean Est. Var (1st)  = {mean_var_1st:.6f}")
        print(f"    Var ratio (est/emp)  = {mean_var_1st / max(emp_var, 1e-12):.3f}")
        print(f"    Coverage 95% (1st)   = {cov_1st:.3f}  ({int(covers_1st.sum())}/{n_sims_done})")

        if has_2nd:
            vars_2nd = np.array([r['results_2nd'][vt]['var_total']
                                 for r in all_sim_results])
            mean_var_2nd = np.mean(vars_2nd)
            ses_2nd = np.sqrt(np.maximum(vars_2nd, 0.0))
            covers_2nd = np.abs(ests - true_delta) <= 1.96 * ses_2nd
            cov_2nd = np.mean(covers_2nd)
            print(f"    Mean Est. Var (2nd)  = {mean_var_2nd:.6f}")
            print(f"    Var ratio (2nd/emp)  = {mean_var_2nd / max(emp_var, 1e-12):.3f}")
            print(f"    Coverage 95% (2nd)   = {cov_2nd:.3f}  ({int(covers_2nd.sum())}/{n_sims_done})")

        if has_boot:
            vars_boot = np.array([r['results_boot'][vt]['var_boot']
                                  for r in all_sim_results])
            mean_var_boot = np.mean(vars_boot)
            ses_boot = np.sqrt(np.maximum(vars_boot, 0.0))
            covers_boot = np.abs(ests - true_delta) <= 1.96 * ses_boot
            cov_boot = np.mean(covers_boot)
            covers_boot_pct = np.array([
                (r['results_boot'][vt]['ci_lo'] <= true_delta <= r['results_boot'][vt]['ci_hi'])
                for r in all_sim_results
            ])
            cov_boot_pct = np.mean(covers_boot_pct)
            print(f"    Mean Est. Var (boot) = {mean_var_boot:.6f}")
            print(f"    Var ratio (boot/emp) = {mean_var_boot / max(emp_var, 1e-12):.3f}")
            print(f"    Coverage 95% (boot-normal) = {cov_boot:.3f}  ({int(covers_boot.sum())}/{n_sims_done})")
            print(f"    Coverage 95% (boot-pctile) = {cov_boot_pct:.3f}  ({int(covers_boot_pct.sum())}/{n_sims_done})")

    # --- Save aggregate CSV ---
    agg_csv = args.output_csv.replace('.csv', '_aggregate.csv')
    agg_header = ['time', 'mean_estimate', 'mean_bias', 'empirical_var',
                  'mean_var_1st', 'var_ratio_1st', 'coverage_1st']
    if has_2nd:
        agg_header += ['mean_var_2nd', 'var_ratio_2nd', 'coverage_2nd']
    if has_boot:
        agg_header += ['mean_var_boot', 'var_ratio_boot',
                       'coverage_boot_normal', 'coverage_boot_percentile']

    agg_rows = []
    for vt in visit_times:
        ests = np.array([r['estimates'][vt] for r in all_sim_results])
        vars_1st = np.array([r['var_1st'][vt] for r in all_sim_results])
        emp_var = np.var(ests, ddof=1)
        ses_1st = np.sqrt(np.maximum(vars_1st, 0.0))

        agg_row = {
            'time': int(vt),
            'mean_estimate': np.mean(ests),
            'mean_bias': np.mean(ests) - true_delta,
            'empirical_var': emp_var,
            'mean_var_1st': np.mean(vars_1st),
            'var_ratio_1st': np.mean(vars_1st) / max(emp_var, 1e-12),
            'coverage_1st': np.mean(np.abs(ests - true_delta) <= 1.96 * ses_1st),
        }
        if has_2nd:
            vars_2nd = np.array([r['results_2nd'][vt]['var_total']
                                 for r in all_sim_results])
            ses_2nd = np.sqrt(np.maximum(vars_2nd, 0.0))
            agg_row['mean_var_2nd'] = np.mean(vars_2nd)
            agg_row['var_ratio_2nd'] = np.mean(vars_2nd) / max(emp_var, 1e-12)
            agg_row['coverage_2nd'] = np.mean(
                np.abs(ests - true_delta) <= 1.96 * ses_2nd)
        if has_boot:
            vars_boot = np.array([r['results_boot'][vt]['var_boot']
                                  for r in all_sim_results])
            ses_boot = np.sqrt(np.maximum(vars_boot, 0.0))
            agg_row['mean_var_boot'] = np.mean(vars_boot)
            agg_row['var_ratio_boot'] = np.mean(vars_boot) / max(emp_var, 1e-12)
            agg_row['coverage_boot_normal'] = np.mean(
                np.abs(ests - true_delta) <= 1.96 * ses_boot)
            agg_row['coverage_boot_percentile'] = np.mean([
                (r['results_boot'][vt]['ci_lo'] <= true_delta <= r['results_boot'][vt]['ci_hi'])
                for r in all_sim_results
            ])
        agg_rows.append(agg_row)

    with open(agg_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=agg_header)
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"\nAggregate CSV saved to {agg_csv}")