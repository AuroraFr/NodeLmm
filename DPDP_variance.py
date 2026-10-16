"""
Delta-method variance estimation for constrast of counterfactural predictions.

Three variance estimators (matching the simulation code):
  Default:  Cov(θ̂) = F⁻¹                 (F = Σ φᵢ φᵢᵀ, penalised scores)
  Bayesian: Cov(θ̂) = J⁻¹                 (O'Sullivan 1988)
  Sandwich: Cov(θ̂) = J⁻¹ F J⁻¹           (robust, Commenges et al. 2014)
"""
from __future__ import annotations
import math, time
import torch
import numpy as np
from typing import Dict

from PDP import (
    resample_xaug_to_grid,
    build_counterfactual_profile_xaug,
)


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _param_list(model):
    """Ordered list of parameters that require grad."""
    return [p for p in model.parameters() if p.requires_grad]


def _param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _cat_grads(grads):
    """Flatten and concatenate a tuple of gradient tensors."""
    return torch.cat([g.reshape(-1) for g in grads])


# ─────────────────────────────────────────────────────────
# 1. Per-subject NLL (differentiable)
# ─────────────────────────────────────────────────────────

def _per_subject_nll(mu, V, y_pad, mask, jitter=1e-4):
    """
    Masked Gaussian NLL for ONE subject (B=1).

    Uses model-returned V directly (no manual reconstruction).

    Args:
        mu:    (1, T)
        V:     (1, T, T)
        y_pad: (1, T)
        mask:  (1, T)

    Returns:
        scalar NLL (differentiable)
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
    V_obs = V[idx][:, idx] + jitter * torch.eye(
        n_i, device=mu.device, dtype=mu.dtype)

    residual = y_obs - mu_obs
    L = torch.linalg.cholesky(V_obs)
    Vinv_r = torch.cholesky_solve(residual.unsqueeze(-1), L).squeeze(-1)
    log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L)))

    nll = 0.5 * (log_det + residual @ Vinv_r + n_i * math.log(2 * math.pi))
    return nll


# ─────────────────────────────────────────────────────────
# 2. Penalty gradient (data-independent constant)
# ─────────────────────────────────────────────────────────

def _is_nn_param(name):

    if ('log_residual_var' in name or 'log_std' in name
            or 'beta_neural' in name):
        return False
    if name == 'decoder.rho_net.net.0.weight':   # group-lasso weight (wd=0)
        return False
    
    return True


def _compute_penalty_gradient(model, lambda_reg, weight_decay):
    """
    c = λ_reg · ∇_θ reg_term + λ_wd · θ

    Added to each NLL score to form φᵢ = ∇nllᵢ + c.
    Returns (P,) tensor on CPU, or None.
    """
    params = _param_list(model)
    P = sum(p.numel() for p in params)
    c = torch.zeros(P)
    has_penalty = False

    if lambda_reg > 0:
        reg_mode = getattr(model, 'reg_mode', None)
        if reg_mode is not None:
            reg_dict = model.decoder._compute_reg()
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
            numel = p.numel()
            is_nn = _is_nn_param(name)  # True for encoder/func/decoder weights
            if is_nn:
                c[offset:offset + numel] = weight_decay * p.detach().reshape(-1).cpu()
            # else: leave zeros (no weight decay on β, D, σ², gate logits)
            offset += numel
        has_penalty = True

    return c if has_penalty else None


# ─────────────────────────────────────────────────────────
# 3. Empirical Fisher  F = Σ_i  φ_i  φ_iᵀ
# ─────────────────────────────────────────────────────────

def compute_empirical_fisher(model, dataset, device, collate_fn,
                              lambda_reg=0.0, weight_decay=0.0,
                              max_subjects=None,
                              verbose=True):
    """
    Compute F = Σᵢ φᵢ φᵢᵀ where φᵢ = ∇_θ nllᵢ + c  (penalised score).

    Uses batch_size=1 so each forward/backward is one subject —
    no retain_graph issues with ODE solvers.

    Includes stationarity diagnostic:
      At the MPLE, Σᵢ φᵢ ≈ 0  (first-order optimality).
      Checks ||mean(φ)|| / mean(||φᵢ||).

    Returns:
        F:      (P, P) tensor on CPU
        scores: (N, P) tensor on CPU  (penalised scores φᵢ)
    """
    from torch.utils.data import DataLoader

    model.eval()
    params = _param_list(model)
    P = sum(p.numel() for p in params)
    N = len(dataset)

    if max_subjects is not None:
        N = min(N, max_subjects)

    c = _compute_penalty_gradient(model, lambda_reg, weight_decay)

    if verbose:
        print(f"    Parameters: P = {P}")
        print(f"    Subjects:   N = {N}  (N/P = {N/P:.2f})")
        if c is not None:
            parts = ["∇nll_i"]
            if lambda_reg > 0:
                reg_mode = getattr(model, 'reg_mode', 'unknown')
                parts.append(f"λ_reg·∇reg ({reg_mode}, λ={lambda_reg})")
            if weight_decay > 0:
                parts.append(f"λ_wd·θ (λ={weight_decay})")
            print(f"    Score: φ_i = {' + '.join(parts)}")
            print(f"    Penalty gradient ||c|| = {c.norm().item():.4f}")
        else:
            print(f"    Score: φ_i = ∇nll_i  (no penalty)")

    fisher = torch.zeros(P, P)
    score_list = []
    nll_grad_norms = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn)

    t0 = time.time()
    for i, batch in enumerate(loader):
        if max_subjects is not None and i >= max_subjects:
            break

        pids, x_aug, y_pad, target_mask, static = batch
        x_aug = x_aug.to(device)
        y_pad = y_pad.to(device)
        target_mask = target_mask.to(device)
        static = static.to(device)

        if target_mask.sum() == 0:
            continue

        mu, V, Z, D, sig2, _ = model(
            x_aug, static_covariates=static, obs_mask=target_mask
        )

        nll_i = _per_subject_nll(mu, V, y_pad, target_mask)

        grads = torch.autograd.grad(nll_i, params, retain_graph=False,
                                    allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, params)]
        nll_grad_i = _cat_grads(grads).cpu()

        nll_grad_norms.append(nll_grad_i.norm().item())

        if c is not None:
            phi_i = nll_grad_i + c
        else:
            phi_i = nll_grad_i

        fisher += torch.outer(phi_i, phi_i)
        score_list.append(phi_i)

        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"    Fisher: {i+1}/{N} subjects "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    scores = torch.stack(score_list) if score_list else torch.zeros(0, P)
    N_actual = len(score_list)

    # ─────────────────────────────────────────────────────
    # Stationarity diagnostic
    # ─────────────────────────────────────────────────────
    if verbose and N_actual > 0:
        elapsed = time.time() - t0
        print(f"    Fisher: done ({elapsed:.1f}s), N = {N_actual}")

        mean_score = scores.mean(dim=0)
        mean_score_norm = mean_score.norm().item()

        phi_norms = scores.norm(dim=1)
        mean_phi_norm = phi_norms.mean().item()
        mean_indiv_nll_norm = np.mean(nll_grad_norms)

        relative = (mean_score_norm / mean_phi_norm
                     if mean_phi_norm > 0 else float('inf'))

        print(f"\n    ── Stationarity diagnostic ──")
        print(f"    ||mean(φ)||         = {mean_score_norm:.4e}")
        print(f"    mean(||φᵢ||)        = {mean_phi_norm:.4e}")
        print(f"    relative            = {relative:.4e}  ")

        if c is None:
            # No penalty — check if mean NLL gradient is suspiciously large
            print(f"No penalty term,  mean(||∇nll_i||)    = {mean_indiv_nll_norm:.4e}")

        # ── Fisher conditioning ───────────────────────────
        diag_F = torch.diag(fisher)
        n_zero = (diag_F.abs() < 1e-10).sum().item()
        cond_raw = torch.linalg.cond(fisher).item()

        print(f"\n    ── Fisher conditioning ──")
        print(f"    diag(F) range: [{diag_F.min().item():.2e}, "
              f"{diag_F.max().item():.2e}]")
        print(f"    cond(F):       {cond_raw:.2e}")

        if n_zero > 0:
            print(f"    {n_zero}/{P} parameters have zero Fisher "
                  f"diagonal (dead/unused)")

        eigvals = torch.linalg.eigvalsh(fisher)
        eff_rank = (eigvals > eigvals.max() * 1e-8).sum().item()
        print(f"    eff_rank(F):   {eff_rank}/{P}")

    return fisher, scores


# ─────────────────────────────────────────────────────────
# 4. Fisher regularisation: three modes
# ─────────────────────────────────────────────────────────

def _ledoit_wolf_shrink_fisher(fisher, scores, verbose=True):
    """
    Ledoit-Wolf shrinkage (Ledoit & Wolf, 2004) for the empirical Fisher.

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
    S = fisher / N
    mu = torch.trace(S).item() / P

    delta_sq = (S - mu * torch.eye(P)).pow(2).sum().item()
    if delta_sq < 1e-30:
        if verbose:
            print(f"  Ledoit-Wolf: δ² ≈ 0, no shrinkage needed")
        return fisher.clone(), 0.0

    norms_sq = (scores ** 2).sum(dim=1)
    sum_norms_4 = (norms_sq ** 2).sum().item()
    S_frob_sq = (S ** 2).sum().item()
    beta = max((1.0 / N**2) * (sum_norms_4 - N * S_frob_sq), 0.0)

    alpha = min(beta / delta_sq, 1.0)

    trace_F = torch.trace(fisher).item()
    F_shrunk = (1.0 - alpha) * fisher + alpha * (trace_F / P) * torch.eye(P)

    if verbose:
        eig_orig = torch.linalg.eigvalsh(fisher)
        eig_shrunk = torch.linalg.eigvalsh(F_shrunk)
        print(f"  Ledoit-Wolf shrinkage:")
        print(f"    N/P = {N}/{P} = {N/P:.1f},  α* = {alpha:.4f}")
        print(f"    F  eigenvalues: [{eig_orig.min().item():.2e}, "
              f"{eig_orig.max().item():.2e}]")
        print(f"    F* eigenvalues: [{eig_shrunk.min().item():.2e}, "
              f"{eig_shrunk.max().item():.2e}]")

    return F_shrunk, alpha


def _active_subspace_decomp(fisher, threshold_ratio=1e-4, verbose=True):
    """
    Eigendecompose F and retain only the active subspace (λ_k > threshold).

    Variance is computed as:
        Var(h) = Σ_{k: λ_k > ε} (gᵀv_k)² / λ_k

    This avoids amplifying noise from near-zero eigenvalues without
    any shrinkage bias.

    Args:
        fisher:          (P, P) empirical Fisher
        threshold_ratio: keep eigenvalues > threshold_ratio × λ_max

    Returns:
        eigvecs_active: (P, K) — columns are active eigenvectors
        eigvals_active: (K,)   — active eigenvalues
    """
    eigvals, eigvecs = torch.linalg.eigh(fisher)
    threshold = threshold_ratio * eigvals.max().item()
    active = eigvals > threshold
    K = active.sum().item()
    P = fisher.shape[0]

    eigvecs_active = eigvecs[:, active]   # (P, K)
    eigvals_active = eigvals[active]      # (K,)

    if verbose:
        print(f"  Active subspace: {K}/{P} directions "
              f"(threshold = {threshold:.2e})")
        print(f"    Active eigenvalue range: [{eigvals_active.min().item():.2e}, "
              f"{eigvals_active.max().item():.2e}]")
        frac_trace = eigvals_active.sum().item() / max(eigvals.sum().item(), 1e-30)
        print(f"    Captures {frac_trace*100:.1f}% of tr(F)")

    return eigvecs_active, eigvals_active


def _variance_projected(g, eigvecs_active, eigvals_active):
    """
    Compute gᵀ F⁻¹ g projected onto the active subspace.

        Var = Σ_{k ∈ active} (gᵀ v_k)² / λ_k

    Args:
        g: (P,) gradient vector
        eigvecs_active: (P, K)
        eigvals_active: (K,)

    Returns:
        var: float
    """
    coeffs = eigvecs_active.T @ g        # (K,)
    var = (coeffs ** 2 / eigvals_active).sum().item()
    return var


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
# 5. ∇_θ PDP(profile, t) for ALL profiles in one data pass
# ─────────────────────────────────────────────────────────

def compute_all_profile_pdp_gradients(
    model, loader, device, profiles, eval_grid,
    target_col, n_tv, mask_type="binary",
    verbose=True,
):
    """
    Compute PDP means and ∇_θ PDP(tₗ) for ALL trajectory profiles
    in a SINGLE pass over the data.
 
    Args:
        model:     NeuralODEModel
        loader:    DataLoader
        device:    torch device
        profiles:  dict {name: (L,) array}
        eval_grid: (L,) numpy array
        target_col, n_tv, mask_type: intervention config
 
    Returns:
        pdp_means: dict {name: (L,) numpy array}
        pdp_grads: dict {name: (L, P) numpy array}
        N_total:   int
    """
    model.eval()
    params = _param_list(model)
    P = sum(p.numel() for p in params)
    L = len(eval_grid)
    profile_names = list(profiles.keys())
    n_profiles = len(profile_names)
 
    # Accumulators
    sum_mu = {pn: np.zeros(L) for pn in profile_names}
    sum_grad = {pn: np.zeros((L, P)) for pn in profile_names}
    N_total = 0
 
    if verbose:
        print(f"    Computing gradients for {n_profiles} profiles × "
              f"{L} time points in single data pass ...")
 
    for batch_idx, batch in enumerate(loader):
        pids, x_aug, y_pad, target_mask, static = batch
        x_aug = x_aug.to(device)
        target_mask = target_mask.to(device)
        static = static.to(device)
        N_batch = x_aug.shape[0]
 
        # Resample to grid per batch
        x_aug_grid, obs_mask_grid = resample_xaug_to_grid(
            x_aug, target_mask, eval_grid, n_tv,
        )
 
        # --- Forward pass for each profile (all in same graph) ---
        mu_dict = {}
        for pname, prof_values in profiles.items():
            x_aug_cf = build_counterfactual_profile_xaug(
                x_aug_grid, target_col=target_col,
                profile_values=prof_values, n_tv=n_tv,
                mask_type=mask_type,
            )
            mu, V, Z, D, sig2, reg = model(
                x_aug_cf, static_covariates=static, obs_mask=None,
            )
            mu_dict[pname] = mu  # (N_batch, L)
 
        N_total += N_batch
 
        # --- Backward: for each (profile, time_point) ---
        # Total backward passes = n_profiles × L
        # retain_graph for all except the very last
        for p_idx, pname in enumerate(profile_names):
            mu = mu_dict[pname]
            sum_mu[pname] += mu.sum(dim=0).detach().cpu().numpy()
 
            for ell in range(L):
                mu_sum_ell = mu[:, ell].sum()
                is_last = (p_idx == n_profiles - 1) and (ell == L - 1)
 
                grads = torch.autograd.grad(
                    mu_sum_ell, params,
                    retain_graph=(not is_last),
                    allow_unused=True,
                )
                grads = [g if g is not None else torch.zeros_like(p)
                         for g, p in zip(grads, params)]
                sum_grad[pname][ell] += _cat_grads(grads).cpu().numpy()
 
        if verbose and (batch_idx + 1) % 5 == 0:
            print(f"      batch {batch_idx+1}: {N_total} subjects processed")
 
    # Normalise
    PDPs = {pn: sum_mu[pn] / N_total for pn in profile_names}
    pdp_grads = {pn: sum_grad[pn] / N_total for pn in profile_names}
 
    if verbose:
        print(f"    Done. N = {N_total}")
        for pn in profile_names:
            norms = [np.linalg.norm(pdp_grads[pn][ell])
                     for ell in range(L)]
            print(f"      {pn:<20s}: mean ||g|| = {np.mean(norms):.4f}")
 
    return PDPs, pdp_grads, N_total

# ─────────────────────────────────────────────────────────
# 6. Full pipeline: trajectory-profile PDP with CI
# ─────────────────────────────────────────────────────────
 
# Default diagnostic contrasts for the real-data multi-profile path.
DEFAULT_DELTA_PAIRS = [
    ("late_decline", "stable_low"),
    ("stable_high", "late_spike"),
    ("stable_high", "gradual_rise"),
    ("gradual_decline", "stable_low"),
]

def _build_var_fn(variance_mode, *, fisher=None, scores=None, fisher_inv=None,
                  hessian=None, active_threshold=1e-4, LAMBDA=1e-4,
                  verbose=True):
    """Return g ↦ gᵀ Cov g for one of five estimators.
 
    Fisher-based (need fisher [+ scores]):
      "marquardt"        Cov = F⁻¹  (pre-damped F⁻¹ reused if given)
      "ledoit_wolf"      shrink F, then invert
      "active_subspace"  project onto active eigenvectors of F
    Hessian-based (need `hessian` = J):
      "bayesian"         Cov = J⁻¹                 (O'Sullivan 1988)
      "sandwich"         Cov = J⁻¹ F J⁻¹           (Commenges et al. 2014)
    """
    if variance_mode == "marquardt":
        if fisher_inv is not None:
            F_inv = (torch.from_numpy(fisher_inv).float()
                     if isinstance(fisher_inv, np.ndarray) else fisher_inv.float())
        else:
            _, F_inv = _regularise_and_invert(fisher, "Fisher", LAMBDA, verbose)
        return lambda g: (g @ F_inv @ g).item()
 
    if variance_mode == "ledoit_wolf":
        assert fisher is not None and scores is not None, \
            "ledoit_wolf requires fisher and scores"
        F_shrunk, lw_alpha = _ledoit_wolf_shrink_fisher(fisher, scores, verbose)
        _, F_inv = _regularise_and_invert(F_shrunk, "Fisher (LW)", LAMBDA, verbose)
        if verbose:
            print(f"  Variance mode: Ledoit-Wolf (α*={lw_alpha:.4f})")
        return lambda g: (g @ F_inv @ g).item()
 
    if variance_mode == "active_subspace":
        assert fisher is not None, "active_subspace requires fisher"
        ev, ea = _active_subspace_decomp(fisher, active_threshold, verbose)
        return lambda g: _variance_projected(g, ev, ea)
 
    if variance_mode == "bayesian":
        assert hessian is not None, "bayesian requires the Hessian J"
        _, J_inv = _regularise_and_invert(hessian, "Hessian J", LAMBDA, verbose)
        return lambda g: (g @ J_inv @ g).item()
 
    if variance_mode == "sandwich":
        assert hessian is not None and fisher is not None, \
            "sandwich requires the Hessian J and F"
        _, J_inv = _regularise_and_invert(hessian, "Hessian J", LAMBDA, verbose)
        cov = J_inv @ fisher @ J_inv
        return lambda g: (g @ cov @ g).item()
 
    raise ValueError(f"Unknown variance_mode: {variance_mode!r}")
 
def compute_contrast_pdp_with_ci(
    model, loader, device, profiles, eval_grid, pairs,
    target_col, n_tv, mask_type="binary",
    variance_mode="marquardt",
    fisher=None, scores=None, fisher_inv=None, hessian=None,
    dataset=None, collate_fn=None,
    lambda_reg=0.0, weight_decay=0.0, n_hessian_subsample=None,
    true_contrast_fns=None, alpha=0.05, active_threshold=1e-4, verbose=True,
):
    """Canonical ΔPDP-with-CI over profile pairs.
 
    The estimand is always the constrast of the counterfactural predictions
    between the profiles, ΔPDP_ab(t) = PDP_a(t) − PDP_b(t);
    Since ∇ is linear, g_ab(t) = ∇PDP_a − ∇PDP_b, so
    all profiles are differentiated once (single data pass) and contrasts are
    formed by subtraction.
 
    Variance: one of five estimators via `variance_mode`. The Hessian J is
    computed ONLY for "bayesian"/"sandwich" (and only if not supplied).
 
    Inputs:
      loader      : batched DataLoader — used for the profile gradients.
      dataset,
      collate_fn  : needed only to compute F (if `fisher` is None) or J
                    (if a Hessian mode is requested and `hessian` is None).
      pairs       : list of (profile_a, profile_b) name tuples.
      true_contrast_fns  : optional {(a, b): callable(t) -> true ΔPDP} for validation.
 
    Returns a dict with:
      results[name]              per-profile {'mean','se','ci_lo','ci_hi','grad'}
      results['_delta_a_vs_b']   per-pair {'mean','estimate','se','ci_lo',
                                  'ci_hi','grad','true','bias','null_frac'}
      results['_meta']           {'stationarity_ratio','mean_score_norm',
                                  'mean_indiv_norm','P','N','variance_mode'}
    Always returns the validation fields (bias is NaN without truth).
    """
    from scipy.stats import norm
    z_crit = norm.ppf(1 - alpha / 2)
    needs_hessian = variance_mode in ("bayesian", "sandwich")
    needs_raw_fisher = not (variance_mode == "marquardt"
                            and fisher_inv is not None)
 
    # ── Fisher (compute only if needed and not supplied) ────────────────
    if fisher is None and needs_raw_fisher:
        assert dataset is not None and collate_fn is not None, \
            "raw Fisher required → need dataset and collate_fn to compute it"
        fisher, scores = compute_empirical_fisher(
            model, dataset, device, collate_fn,
            lambda_reg=lambda_reg, weight_decay=weight_decay, verbose=verbose)
 
    have_fisher = fisher is not None
    if have_fisher:
        eigvals, eigvecs = torch.linalg.eigh(fisher)
        null_mask = eigvals < eigvals.max().item() * 1e-10
 
        def _null_frac(g):
            coeffs = eigvecs.T @ g
            return (coeffs[null_mask] ** 2).sum().item() / max(
                (coeffs ** 2).sum().item(), 1e-30)
    else:
        def _null_frac(g):
            return float('nan')
 
    # stationarity diagnostic (needs per-subject scores)
    if scores is not None and scores.numel() > 0:
        mean_score = scores.mean(dim=0)
        stationarity_ratio = (mean_score.norm()
                              / scores.abs().mean(dim=0).norm()).item()
        mean_score_norm = mean_score.norm().item()
        mean_indiv_norm = scores.norm(dim=1).mean().item()
    else:
        stationarity_ratio = mean_score_norm = mean_indiv_norm = float('nan')
 
    # ── Hessian (only for bayesian/sandwich) ────────────────────────────
    if needs_hessian and hessian is None:
        assert dataset is not None and collate_fn is not None, \
            f"{variance_mode} → need dataset and collate_fn to compute J"
        hessian = compute_hessian_explicit(
            model, dataset, device, collate_fn,
            n_subsample=n_hessian_subsample,
            weight_decay=weight_decay, lambda_reg=lambda_reg, verbose=verbose)
 
    var_fn = _build_var_fn(
        variance_mode, fisher=fisher, scores=scores, fisher_inv=fisher_inv,
        hessian=hessian, active_threshold=active_threshold, verbose=verbose)
 
    # ── Profile gradients (single pass) ─────────────────────────────────
    PDPs, pdp_grads, N_total = compute_all_profile_pdp_gradients(
        model, loader, device, profiles, eval_grid,
        target_col, n_tv, mask_type, verbose)
 
    L = len(eval_grid)
    results = {}
 
    # per-profile PDP with CI
    for pname in profiles:
        se = np.array([np.sqrt(max(var_fn(
            torch.from_numpy(pdp_grads[pname][ell]).float()), 0.0))
            for ell in range(L)])
        results[pname] = {
            'mean': PDPs[pname], 
            'se': se,
            'ci_lo': PDPs[pname] - z_crit * se,
            'ci_hi': PDPs[pname] + z_crit * se,
            'grad': pdp_grads[pname],
        }
 
    # per-pair ΔPDP with CI + validation fields
    true_contrast_fns = true_contrast_fns or {}
    for pa, pb in pairs:
        if pa not in results or pb not in results:
            continue
        delta_mean = results[pa]['mean'] - results[pb]['mean']
        delta_grad = results[pa]['grad'] - results[pb]['grad']
        se = np.zeros(L); nf = np.zeros(L)
        for ell in range(L):
            g = torch.from_numpy(delta_grad[ell]).float()
            se[ell] = np.sqrt(max(var_fn(g), 0.0))
            nf[ell] = _null_frac(g)
        ofn = true_contrast_fns.get((pa, pb))
        true = (np.array([ofn(float(eval_grid[ell])) for ell in range(L)])
                if ofn is not None else np.full(L, np.nan))
        results[f'_delta_{pa}_vs_{pb}'] = {
            'mean': delta_mean, 'estimate': delta_mean, 
            'se': se,
            'ci_lo': delta_mean - z_crit * se,
            'ci_hi': delta_mean + z_crit * se,
            'grad': delta_grad, 
            'true': true, 'bias': delta_mean - true,
            'null_frac': nf,
        }
        if verbose:
            n_sig = int(((results[f'_delta_{pa}_vs_{pb}']['ci_lo'] > 0)
                         | (results[f'_delta_{pa}_vs_{pb}']['ci_hi'] < 0)).sum())
            print(f"    ΔPDP {pa} − {pb}: {n_sig}/{L} times significant")
 
    results['_meta'] = {
        'stationarity_ratio': stationarity_ratio,
        'mean_score_norm': mean_score_norm,
        'mean_indiv_norm': mean_indiv_norm,
        'P': _param_count(model), 'N': N_total,
        'variance_mode': variance_mode,
    }
    return results
 
 
def compute_trajectory_profile_pdp_with_ci(
    model, loader, device, profiles, eval_grid,
    fisher_inv, target_col, n_tv,
    mask_type="binary", alpha=0.05,
    verbose=True,
    variance_mode="marquardt",
    fisher=None, scores=None,
    active_threshold=1e-4,
):
    """Backward-compat wrapper: real-data multi-profile path with the default
    diagnostic contrasts. Delegates to compute_profile_delta_pdp_with_ci."""
    results = compute_contrast_pdp_with_ci(
        model, loader, device, profiles, eval_grid,
        pairs=DEFAULT_DELTA_PAIRS,
        target_col=target_col, n_tv=n_tv, mask_type=mask_type,
        variance_mode=variance_mode,
        fisher=fisher, scores=scores, fisher_inv=fisher_inv,
        alpha=alpha, active_threshold=active_threshold, verbose=verbose)
    # legacy alias
    if '_delta_late_decline_vs_late_spike' in results:
        results['_delta_eb_ls'] = results['_delta_late_decline_vs_late_spike']
    return results
 
 
import torch.func as TF

def _compute_penalty_hessian(model, lambda_reg, verbose=True):
    """∇²(reg_term) for skip_gate / group_lasso via autograd (or None)."""
    reg_mode = getattr(model, 'reg_mode', None)
    if reg_mode is None or lambda_reg <= 0:
        return None
 
    params = _param_list(model)
    P = sum(p.numel() for p in params)
 
    reg_dict = model.decoder._compute_reg()
    reg_term = reg_dict["reg_term"]
    if not reg_term.requires_grad:
        if verbose:
            print("    Penalty Hessian: reg_term has no grad (penalty=0)")
        return None
 
    grad1 = torch.autograd.grad(reg_term, params, create_graph=True,
                                allow_unused=True)
    grad1 = [g if g is not None else torch.zeros_like(p)
             for g, p in zip(grad1, params)]
    grad_flat = torch.cat([g.reshape(-1) for g in grad1])
 
    H_pen = torch.zeros(P, P)
    for j in range(P):
        if grad_flat[j].requires_grad:
            row = torch.autograd.grad(grad_flat[j], params,
                                      retain_graph=(j < P - 1),
                                      allow_unused=True)
            row = [g if g is not None else torch.zeros_like(p)
                   for g, p in zip(row, params)]
            H_pen[j, :] = _cat_grads(row).cpu()
 
    H_pen = 0.5 * (H_pen + H_pen.T)
    if verbose:
        nnz = (H_pen.abs() > 1e-12).sum().item()
        print(f"    Penalty Hessian ({reg_mode}): {nnz}/{P*P} nonzero, "
              f"||H_pen|| = {H_pen.norm().item():.4e}")
    return H_pen
 
 
def compute_hessian_explicit(model, dataset, device, collate_fn,
                             n_subsample=None, weight_decay=0.0,
                             lambda_reg=0.0, verbose=True):
    """J = Σ ∇²nll_i + N·λ_wd·diag(mask_nn) + N·λ_reg·∇²reg_term.
 
    Unified x_aug signature: model(x_aug, static_covariates=, obs_mask=).
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
            print(f"    Hessian: subsampling {M}/{N} (scale={scale:.2f})")
    else:
        subset, scale, M = dataset, 1.0, N
 
    loader = DataLoader(subset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn)
 
    param_names, param_shapes, param_numels, flat_params = [], [], [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            param_names.append(name)
            param_shapes.append(p.shape)
            param_numels.append(p.numel())
            flat_params.append(p.detach().reshape(-1))
    theta0 = torch.cat(flat_params).to(device)
 
    def _nll_from_flat(theta, batch_data):
        x_aug, y_pad, mask, s = batch_data
        offset = 0
        param_dict = {}
        for name, shape, numel in zip(param_names, param_shapes, param_numels):
            param_dict[name] = theta[offset:offset + numel].view(shape)
            offset += numel
        mu, V, Z, D_mat, sig2, _ = TF.functional_call(
            model, param_dict, args=(x_aug,),
            kwargs=dict(static_covariates=s, obs_mask=mask))
        return _per_subject_nll(mu, V, y_pad, mask)
 
    hessian = torch.zeros(P, P)
    t0 = time.time()
    n_done = 0
    for i, batch in enumerate(loader):
        pids, x_aug, y_pad, mask, s = batch
        x_aug, y_pad = x_aug.to(device), y_pad.to(device)
        mask, s = mask.to(device), s.to(device)
        if mask.sum() == 0:
            continue
        batch_data = (x_aug, y_pad, mask, s)
        H_i = torch.func.jacrev(torch.func.jacrev(
            lambda th: _nll_from_flat(th, batch_data)))(theta0)
        hessian += H_i.cpu()
        n_done += 1
        if verbose and n_done % 10 == 0:
            elapsed = time.time() - t0
            eta = (M - n_done) / max(n_done / elapsed, 1e-9)
            print(f"    Hessian: {n_done}/{M} ({elapsed:.0f}s, ~{eta:.0f}s left)")
 
    hessian *= scale
    hessian = 0.5 * (hessian + hessian.T)
 
    N_full = len(dataset)
    if weight_decay > 0:
        offset = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            numel = p.numel()
            if _is_nn_param(name):
                for j in range(numel):
                    hessian[offset + j, offset + j] += N_full * weight_decay
            offset += numel
    if lambda_reg > 0:
        H_pen = _compute_penalty_hessian(model, lambda_reg, verbose)
        if H_pen is not None:
            hessian += N_full * lambda_reg * H_pen
 
    if verbose:
        eigvals = torch.linalg.eigvalsh(hessian)
        print(f"    Hessian done ({time.time()-t0:.1f}s), used {n_done} subjects")
        print(f"    J eig range: [{eigvals.min().item():.2e}, "
              f"{eigvals.max().item():.2e}], "
              f"{(eigvals < 0).sum().item()}/{P} negative, "
              f"cond={torch.linalg.cond(hessian).item():.2e}")
    return hessian
 
 
def aggregate_simulations(all_results, t_eval, pair_key):
    """Aggregate Var_est vs Var_MC + 95% coverage across D replicates.
 
    Consumes the output of compute_profile_delta_pdp_with_ci directly: each
    element of `all_results` is one replicate's result dict, and `pair_key` is
    the contrast to aggregate, e.g. '_delta_hi_vs_lo'. Per-pair fields
    (mean, se, ci_lo, ci_hi, true, null_frac) are arrays over the eval grid,
    aligned with `t_eval`; stationarity diagnostics live in results['_meta'].
    """
    def _meta(r, k):
        return r.get('_meta', {}).get(k, np.nan)
 
    summary = {}
    stat = [_meta(r, 'stationarity_ratio') for r in all_results]
    msn = [_meta(r, 'mean_score_norm') for r in all_results]
    min_ = [_meta(r, 'mean_indiv_norm') for r in all_results]
    summary['_diagnostics'] = {
        'mean_stationarity_ratio': np.nanmean(stat),
        'std_stationarity_ratio': np.nanstd(stat),
        'mean_score_norm': np.nanmean(msn),
        'mean_indiv_norm': np.nanmean(min_),
    }
 
    # keep only replicates that actually contain this contrast
    reps = [r for r in all_results if pair_key in r]
 
    for ell, t in enumerate(t_eval):
        tk = float(t)
        ests = np.array([r[pair_key]['mean'][ell] for r in reps])
        ses = np.array([r[pair_key]['se'][ell] for r in reps])
        trues = np.array([r[pair_key]['true'][ell] for r in reps])
        ci_los = np.array([r[pair_key]['ci_lo'][ell] for r in reps])
        ci_his = np.array([r[pair_key]['ci_hi'][ell] for r in reps])
        null_fracs_t = np.array(
            [r[pair_key].get('null_frac', np.full(len(t_eval), np.nan))[ell]
             for r in reps])
        D_t = len(ests)
        if D_t == 0:
            continue
        bias = ests.mean() - np.nanmean(trues)
        var_mc = ests.var(ddof=1) if D_t > 1 else float('nan')
        coverage = (np.mean((ci_los <= trues) & (trues <= ci_his))
                    if D_t > 1 else float('nan'))
        summary[tk] = {
            'D': D_t, 
            'mean_hat': ests.mean(), 'true': np.nanmean(trues),
            'bias': bias, 
            'var_mc': var_mc,
            'mean_var_est': (ses ** 2).mean(),
            'mse': (bias ** 2 + var_mc) if D_t > 1 else float('nan'),
            'coverage95': coverage,
            'mean_null_frac': np.nanmean(null_fracs_t),
        }
    return summary