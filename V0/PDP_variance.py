"""
Delta-method variance estimation for PDP / ΔPDP — Neural ODE-LMM (real 3C data).

Adapted from PDP_delta_cumulative_effect.py for the continuous-time PDP
on the real 3C cohort.

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

from PDP_continuous_time import (
    resample_xaug_to_grid,
    build_profile_xaug_continuous,
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
    """True for encoder/func/decoder network weights, False for β, D, σ², gates."""
    excluded = ('beta', 'log_D_diag', 'log_sigma2', 'D_off_diag',
                'skip_gate_logits', 'gate_logits')
    return not any(ex in name for ex in excluded)

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
      Checks ||mean(φ)|| / mean(||φᵢ||) — should be < 0.05.

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
        print(f"    relative            = {relative:.4e}  "
              f"(should be < 0.05)")

        if c is None:
            # No penalty — check if mean NLL gradient is suspiciously large
            print(f"    mean(||∇nll_i||)    = {mean_indiv_nll_norm:.4e}")

            if relative > 0.05:
                print(f"\n    ⚠ WARNING: ||mean(∇nll_i)|| is large relative "
                      f"to individual score norms.")
                print(f"      This suggests the model was trained with "
                      f"regularisation (weight_decay / lambda_reg)")
                print(f"      but the penalty gradient c_pen was NOT "
                      f"included in the Fisher scores.")
                print(f"      → Pass the training lambda_reg and "
                      f"weight_decay to compute_empirical_fisher().")
                print(f"      Expected: ||mean(∇nll_i)|| ≈ ||c_pen|| "
                      f"at the MPLE.")
        else:
            if relative > 0.05:
                print(f"\n    ⚠ WARNING: stationarity violation. "
                      f"||mean(φ)|| / mean(||φᵢ||) = {relative:.4e} > 0.05")
                print(f"      Possible causes:")
                print(f"        - Optimizer did not fully converge")
                print(f"        - lambda_reg or weight_decay values "
                      f"don't match training")
                print(f"        - Gradient computation issue "
                      f"(e.g. ODE solver tolerance)")
            else:
                print(f"    ✓ Stationarity OK")

        # ── Fisher conditioning ───────────────────────────
        diag_F = torch.diag(fisher)
        n_zero = (diag_F.abs() < 1e-10).sum().item()
        cond_raw = torch.linalg.cond(fisher).item()

        print(f"\n    ── Fisher conditioning ──")
        print(f"    diag(F) range: [{diag_F.min().item():.2e}, "
              f"{diag_F.max().item():.2e}]")
        print(f"    cond(F):       {cond_raw:.2e}")

        if n_zero > 0:
            print(f"    ⚠ {n_zero}/{P} parameters have zero Fisher "
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

    Follows the same pattern as compute_delta_pdp_gradients() in
    PDP_delta_cumulative_effect.py:
      - Multiple forward passes per batch (one per profile)
      - For each time point, backward with retain_graph
      - Accumulate gradients across batches

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

        # Resample to grid ONCE per batch
        x_aug_grid, obs_mask_grid = resample_xaug_to_grid(
            x_aug, target_mask, eval_grid, n_tv,
        )

        # --- Forward pass for each profile (all in same graph) ---
        mu_dict = {}
        for pname, prof_values in profiles.items():
            x_aug_cf = build_profile_xaug_continuous(
                x_aug_grid, target_col=target_col,
                profile_values=prof_values, n_tv=n_tv,
                mask_type=mask_type,
            )
            mu, V, Z, D, sig2, reg = model(
                x_aug_cf, static_covariates=static, obs_mask=obs_mask_grid,
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
    pdp_means = {pn: sum_mu[pn] / N_total for pn in profile_names}
    pdp_grads = {pn: sum_grad[pn] / N_total for pn in profile_names}

    if verbose:
        print(f"    Done. N = {N_total}")
        for pn in profile_names:
            norms = [np.linalg.norm(pdp_grads[pn][ell])
                     for ell in range(L)]
            print(f"      {pn:<20s}: mean ||g|| = {np.mean(norms):.4f}")

    return pdp_means, pdp_grads, N_total


# ─────────────────────────────────────────────────────────
# 6. Full pipeline: trajectory-profile PDP with CI
# ─────────────────────────────────────────────────────────

def compute_trajectory_profile_pdp_with_ci(
    model, loader, device, profiles, eval_grid,
    fisher_inv, target_col, n_tv,
    mask_type="binary", alpha=0.05,
    target_name="covariate", verbose=True,
    variance_mode="marquardt",
    fisher=None, scores=None,
    active_threshold=1e-4,
):
    """
    Compute trajectory-profile PDP with delta-method CI.

    Single pass over the data for all profiles (follows reference pattern).

    Args:
        fisher_inv:     (P, P) tensor — F⁻¹ (used when variance_mode="marquardt")
        fisher:         (P, P) tensor — F   (needed for "ledoit_wolf" and "active_subspace")
        scores:         (N, P) tensor — φ_i (needed for "ledoit_wolf")
        variance_mode:  "marquardt"        — use pre-computed F⁻¹ with Marquardt damping
                        "ledoit_wolf"      — shrink F, then invert
                        "active_subspace"  — project onto active eigenvectors of F
        active_threshold: eigenvalue threshold ratio for active_subspace mode
        profiles:       dict {name: (L,) array} from make_profiles_continuous

    Returns:
        dict {profile_name: {'mean', 'se', 'ci_lo', 'ci_hi', 'grad'}}
    """
    from scipy.stats import norm
    z_crit = norm.ppf(1 - alpha / 2)

    # --- Prepare variance computation based on mode ---
    if variance_mode == "ledoit_wolf":
        assert fisher is not None and scores is not None, \
            "ledoit_wolf mode requires fisher and scores"
        F_shrunk, lw_alpha = _ledoit_wolf_shrink_fisher(fisher, scores, verbose)
        _, F_inv = _regularise_and_invert(F_shrunk, "Fisher (LW)", LAMBDA=1e-4,
                                          verbose=verbose)
        var_fn = lambda g: (g @ F_inv @ g).item()
        if verbose:
            print(f"  Variance mode: Ledoit-Wolf (α*={lw_alpha:.4f})")

    elif variance_mode == "active_subspace":
        assert fisher is not None, "active_subspace mode requires fisher"
        eigvecs_active, eigvals_active = _active_subspace_decomp(
            fisher, threshold_ratio=active_threshold, verbose=verbose)
        var_fn = lambda g: _variance_projected(g, eigvecs_active, eigvals_active)
        if verbose:
            print(f"  Variance mode: active subspace "
                  f"(threshold={active_threshold:.1e})")

    else:  # "marquardt" — default
        if isinstance(fisher_inv, np.ndarray):
            F_inv = torch.from_numpy(fisher_inv).float()
        else:
            F_inv = fisher_inv.float()
        var_fn = lambda g: (g @ F_inv @ g).item()
        if verbose:
            print(f"  Variance mode: Marquardt (pre-computed F⁻¹)")

    # All profiles in one data pass
    pdp_means, pdp_grads, N_total = compute_all_profile_pdp_gradients(
        model, loader, device, profiles, eval_grid,
        target_col, n_tv, mask_type, verbose,
    )

    L = len(eval_grid)
    results = {}

    for pname in profiles:
        se = np.zeros(L)
        for ell in range(L):
            g = torch.from_numpy(pdp_grads[pname][ell]).float()
            var_ell = var_fn(g)
            se[ell] = np.sqrt(max(var_ell, 0.0))

        results[pname] = {
            'mean': pdp_means[pname],
            'se': se,
            'ci_lo': pdp_means[pname] - z_crit * se,
            'ci_hi': pdp_means[pname] + z_crit * se,
            'grad': pdp_grads[pname],
        }

    # ── Diagnostic: pairwise ΔPDP with CI ─────────────────
    pairs = [
        # ("late_decline", "late_spike"),
        # ("stable_high", "stable_low"),
        # ("gradual_decline", "gradual_rise"),
        ("late_decline", "stable_low"),
        # ("late_decline", "gradual_decline"),
        ("stable_high", "late_spike"),
        ("stable_high", "gradual_rise"),
        ("gradual_decline", "stable_low"),
    ]
    for pa, pb in pairs:
        if pa not in results or pb not in results:
            continue
        delta_mean = results[pa]['mean'] - results[pb]['mean']
        delta_grad = results[pa]['grad'] - results[pb]['grad']

        delta_se = np.zeros(L)
        for ell in range(L):
            g = torch.from_numpy(delta_grad[ell]).float()
            delta_se[ell] = np.sqrt(max(var_fn(g), 0.0))

        key = f'_delta_{pa}_vs_{pb}'
        results[key] = {
            'mean': delta_mean,
            'se': delta_se,
            'ci_lo': delta_mean - z_crit * delta_se,
            'ci_hi': delta_mean + z_crit * delta_se,
        }

        if verbose:
            label_a = pa.replace('_', ' ')
            label_b = pb.replace('_', ' ')
            n_sig = sum(1 for ell in range(L)
                        if (delta_mean[ell] - z_crit * delta_se[ell] > 0
                            or delta_mean[ell] + z_crit * delta_se[ell] < 0))
            print(f"\n    ΔPDP: {label_a} − {label_b}: "
                  f"{n_sig}/{L} times significant")
            print(f"    {'Time':>8s}  {'Diff':>8s}  {'SE':>8s}  "
                  f"{'95% CI':>20s}  {'Sig?':>8s}")
            for ell in range(L):
                t = eval_grid[ell]
                m = delta_mean[ell]
                s = delta_se[ell]
                lo, hi = m - z_crit * s, m + z_crit * s
                sig = "yes" if hi < 0 or lo > 0 else "no"
                print(f"    {t:8.1f}  {m:+8.3f}  {s:8.3f}  "
                      f"[{lo:+.3f}, {hi:+.3f}]  {sig:>8s}")

    # Backward compatibility: also store under the old key
    if '_delta_late_decline_vs_late_spike' in results:
        results['_delta_eb_ls'] = results['_delta_late_decline_vs_late_spike']

    return results