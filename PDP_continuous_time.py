"""
Continuous-time PDP analysis for the Neural ODE-LMM on the 3C cohort.

Evaluates the trajectory-profile PDP on a regular time grid (e.g. 0,1,...,14)
instead of the canonical visit times.  Every subject contributes at every
grid point — no censoring-based filtering, no max_dist alignment.

v2: warm/cold colour scheme + delta-method CI plotting.

Usage:
    from PDP_continuous_time import (
        make_profiles_continuous,
        resample_xaug_to_grid,
        compute_trajectory_profile_pdp_continuous,
        compute_trajectory_profile_pdp_with_ci,   # delta-method CI
        plot_trajectory_profile_pdp_continuous,
        plot_trajectory_profile_pdp_delta,
        plot_all_pairwise_delta_pdp,
    )
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ═════════════════════════════════════════════════════════════════════
#  Warm / cold colour palette
# ═════════════════════════════════════════════════════════════════════
#
#  Warm = profiles starting or staying at HIGH values (burden)
#  Cold = profiles starting or staying at LOW values (protective)
#
#  Visual logic:  at early times warm profiles have high covariate
#  values and cold profiles have low values.  The diagnostic pair
#  (early_burden vs late_spike) is warm vs cold.

PROFILE_COLOURS = {
    # ── Cold (low / protective) ──────────────────────────────────
    "stable_low":      "#1565C0",   # deep blue
    "late_spike":      "#0288D1",   # medium blue
    "gradual_rise":    "#00838F",   # teal
    # ── Warm (high / burden) ─────────────────────────────────────
    "stable_high":     "#C62828",   # deep red
    "late_decline":    "#E64A19",   # deep orange
    "gradual_decline": "#F57C00",   # amber
}

PROFILE_LABELS = {
    "stable_low":      "Stable low (Q25)",
    "stable_high":     "Stable high (Q75)",
    "late_spike":      "Late spike (Q25→Q75)",
    "late_decline":    "Late Decline (Q75→Q25)",
    "gradual_rise":    "Gradual rise",
    "gradual_decline": "Gradual decline",
}

# Canonical plot order: cold first, then warm
PROFILE_ORDER = [
    "stable_low", "late_spike", "gradual_rise",
    "stable_high", "late_decline", "gradual_decline",
]


# ═════════════════════════════════════════════════════════════════════
#  Regular evaluation grid
# ═════════════════════════════════════════════════════════════════════

def make_eval_grid(t_max=14.0, n_points=15):
    """Regular time grid from 0 to t_max (inclusive)."""
    return np.linspace(0, t_max, n_points)


# ═════════════════════════════════════════════════════════════════════
#  Profiles on continuous grid
# ═════════════════════════════════════════════════════════════════════

def make_profiles_continuous(eval_grid, v_lo, v_hi):
    """
    Six counterfactual trajectory profiles evaluated on a continuous grid.

    Args:
        eval_grid: (L,) numpy array of evaluation times
        v_lo:      low value (e.g. Q25)
        v_hi:      high value (e.g. Q75)

    Returns:
        dict {profile_name: (L,) numpy array}
    """
    L = len(eval_grid)
    t_norm = (eval_grid - eval_grid[0]) / (eval_grid[-1] - eval_grid[0])

    profiles = {
        "stable_low":      np.full(L, v_lo),
        "stable_high":     np.full(L, v_hi),
        "late_spike":      np.where(t_norm < 0.5, v_lo, v_hi),
        "late_decline":    np.where(t_norm < 0.5, v_hi, v_lo),
        "gradual_rise":    v_lo + (v_hi - v_lo) * t_norm,
        "gradual_decline": v_hi + (v_lo - v_hi) * t_norm,
    }
    return profiles


# ═════════════════════════════════════════════════════════════════════
#  Resample x_aug onto a regular grid
# ═════════════════════════════════════════════════════════════════════

def resample_xaug_to_grid(x_aug, obs_mask, eval_grid, n_tv):
    """
    Resample x_aug from its original (irregular) time slots onto a regular grid.

    For each subject and each covariate channel:
      - Identify observed slots (obs_mask > 0)
      - Linearly interpolate covariate values to eval_grid
      - Beyond last observation: LOCF
      - Before first observation: first-observation-carried-backward

    Args:
        x_aug:     (N, T_orig, 1+2K) original augmented input
        obs_mask:  (N, T_orig) binary mask of observed visits
        eval_grid: (L,) numpy array of new time points
        n_tv:      K, number of time-varying covariates

    Returns:
        x_aug_new:    (N, L, 1+2K) resampled x_aug
        obs_mask_new: (N, L) all-ones mask
    """
    K = n_tv
    N, T_orig, D = x_aug.shape
    L = len(eval_grid)
    device = x_aug.device
    dtype = x_aug.dtype

    x_aug_new = torch.zeros(N, L, D, device=device, dtype=dtype)

    grid_tensor = torch.tensor(eval_grid, device=device, dtype=dtype)
    x_aug_new[:, :, 0] = grid_tensor.unsqueeze(0).expand(N, -1)

    for i in range(N):
        t_orig = x_aug[i, :, 0].cpu().numpy()
        obs_i = obs_mask[i].cpu().numpy() > 0.5

        if not obs_i.any():
            continue

        t_obs = t_orig[obs_i]

        for k in range(K):
            cov_col = 1 + k
            mask_col = 1 + K + k

            vals_orig = x_aug[i, :, cov_col].cpu().numpy()
            vals_obs = vals_orig[obs_i]

            interp_vals = np.interp(eval_grid, t_obs, vals_obs)
            x_aug_new[i, :, cov_col] = torch.tensor(
                interp_vals, device=device, dtype=dtype)

            mask_orig = x_aug[i, :, mask_col].cpu().numpy()
            if mask_orig.max() > 1.5:  # cumulative mask
                x_aug_new[i, :, mask_col] = torch.arange(
                    1, L + 1, device=device, dtype=dtype)
            else:
                x_aug_new[i, :, mask_col] = 1.0

    obs_mask_new = torch.ones(N, L, device=device, dtype=dtype)
    return x_aug_new, obs_mask_new


# ═════════════════════════════════════════════════════════════════════
#  Build profile x_aug on regular grid
# ═════════════════════════════════════════════════════════════════════

def build_profile_xaug_continuous(x_aug_grid, target_col, profile_values,
                                   n_tv, mask_type="binary"):
    """Replace target covariate in resampled x_aug with profile values."""
    K = n_tv
    N, L, _ = x_aug_grid.shape
    x_aug_cf = x_aug_grid.clone()

    cov_col = 1 + target_col
    mask_col = 1 + K + target_col

    prof = torch.tensor(profile_values, device=x_aug_grid.device,
                        dtype=x_aug_grid.dtype)
    x_aug_cf[:, :, cov_col] = prof.unsqueeze(0).expand(N, -1)

    if mask_type == "binary":
        x_aug_cf[:, :, mask_col] = 1.0
    else:
        x_aug_cf[:, :, mask_col] = torch.arange(
            1, L + 1, device=x_aug_grid.device, dtype=x_aug_grid.dtype
        ).unsqueeze(0).expand(N, -1)

    return x_aug_cf


# ═════════════════════════════════════════════════════════════════════
#  Core: continuous-time trajectory-profile PDP
# ═════════════════════════════════════════════════════════════════════

def compute_trajectory_profile_pdp_continuous(
    model, loader, device, profiles, eval_grid,
    target_col=0, n_tv=5, mask_type="binary",
    target_name="covariate",fisher=None, scores=None
):
    """
    Compute trajectory-profile PDP on a continuous regular time grid.

    Returns:
        results:    dict {profile_name: (N, L) numpy array of per-subject pop means}
        eval_grid:  (L,) numpy array
        n_subjects: int
    """
    model.eval()
    L = len(eval_grid)

    results = {pname: [] for pname in profiles}
    n_total = 0

    print(f"  Computing continuous-time trajectory-profile PDP for {target_name}")
    print(f"    Grid: {L} points on [{eval_grid[0]:.1f}, {eval_grid[-1]:.1f}]")
    print(f"    Profiles: {list(profiles.keys())}")

    with torch.no_grad():
        for pname, prof_values in profiles.items():
            batch_mus = []

            for batch in loader:
                pids, x_aug, y_pad, target_mask, static = batch
                x_aug = x_aug.to(device)
                target_mask = target_mask.to(device)
                static = static.to(device)
                N_batch = x_aug.shape[0]

                x_aug_grid, obs_mask_grid = resample_xaug_to_grid(
                    x_aug, target_mask, eval_grid, n_tv,
                )

                x_aug_cf = build_profile_xaug_continuous(
                    x_aug_grid, target_col=target_col,
                    profile_values=prof_values, n_tv=n_tv,
                    mask_type=mask_type,
                )

                mu, V, Z, D, sig2, reg_dict = model(
                    x_aug_cf,
                    static_covariates=static,
                    obs_mask=obs_mask_grid,
                )
                batch_mus.append(mu.cpu().numpy())

                if pname == list(profiles.keys())[0]:
                    n_total += N_batch

            results[pname] = np.concatenate(batch_mus, axis=0)

    # ── Print summary ────────────────────────────────────────────
    print(f"\n    N subjects = {n_total}")
    print(f"\n    {'Profile':<20s}", end="")
    for t in eval_grid:
        print(f"  t={t:<5.1f}", end="")
    print()
    print(f"    {'-'*20 + '-'*8*L}")

    for pname in profiles:
        mu_all = results[pname]
        means = mu_all.mean(axis=0)
        print(f"    {pname:<20s}", end="")
        for l in range(L):
            print(f"  {means[l]:>7.2f}", end="")
        print()

    # ── Diagnostic: early_burden vs late_spike ────────────────────
    if "late_decline" in results and "late_spike" in results:
        eb = results["late_decline"].mean(axis=0)
        ls = results["late_spike"].mean(axis=0)
        diff = eb - ls
        n = results["late_decline"].shape[0]
        diff_subj = results["late_decline"] - results["late_spike"]
        se = diff_subj.std(axis=0) / np.sqrt(n)

        print(f"\n    Diagnostic: late_decline − late_spike (cross-subject SE)")
        print(f"    {'Time':>8s}  {'Diff':>8s}  {'SE':>8s}  "
              f"{'95% CI':>20s}  {'Interp':>20s}")
        for l in range(L):
            t = eval_grid[l]
            m = diff[l]
            s = se[l]
            ci_lo, ci_hi = m - 1.96 * s, m + 1.96 * s
            interp = ("cumulative ✓" if ci_hi < -0.01
                       else "instantaneous" if ci_lo < 0 < ci_hi
                       else "unexpected (+)")
            print(f"    {t:8.1f}  {m:+8.3f}  {s:8.3f}  "
                  f"[{ci_lo:+.3f}, {ci_hi:+.3f}]  {interp:>20s}")

    return results, eval_grid, n_total


# ═════════════════════════════════════════════════════════════════════
#  Core: continuous-time trajectory-profile PDP  WITH  delta-method CI
# ═════════════════════════════════════════════════════════════════════

def _flatten_grad(params):
    """Collect .grad from each parameter into a single 1-D numpy array."""
    pieces = []
    for p in params:
        if p.grad is not None:
            pieces.append(p.grad.detach().cpu().reshape(-1).numpy())
        else:
            pieces.append(np.zeros(p.numel()))
    return np.concatenate(pieces)


def compute_trajectory_profile_pdp_with_ci(
    model, loader, device, profiles, eval_grid,
    target_col=0, n_tv=5, mask_type="binary",
    target_name="covariate", fisher_inv=None,
):
    """
    Compute trajectory-profile PDP **with delta-method CI**.

    For each profile p and grid point ℓ, computes:
        PDP_p(t_ℓ) = (1/N) Σ_i  μ_i(t_ℓ ; x^cf_p)
        g_pℓ       = (1/N) Σ_i  ∂μ_i(t_ℓ)/∂θ          (via backprop)
        SE_pℓ      = sqrt( g_pℓ^T  F^{-1}  g_pℓ )

    Gradient computation uses batch_size=1 with independent forward/backward
    per subject to avoid retain_graph errors from the ODE solver.

    Args:
        model:      trained Neural ODE-LMM
        loader:     DataLoader (any batch size — subjects processed one-by-one)
        device:     torch device
        profiles:   dict {name: (L,) array of profile values}
        eval_grid:  (L,) numpy array
        target_col: covariate index
        n_tv:       number of time-varying covariates
        mask_type:  "binary" or "cumulative"
        target_name: label for printing
        fisher_inv: (P, P) numpy array or torch tensor, F^{-1}

    Returns:
        ci_results: dict {profile_name: {'mean': (L,), 'grad': (L,P),
                          'se': (L,), 'ci_lo': (L,), 'ci_hi': (L,)}}
        eval_grid:  (L,) numpy array
        n_subjects: int
    """
    model.eval()
    L = len(eval_grid)

    # Identify parameters for gradient collection
    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)

    print(f"  Computing trajectory-profile PDP with CI for {target_name}")
    print(f"    Grid: {L} points on [{eval_grid[0]:.1f}, {eval_grid[-1]:.1f}]")
    print(f"    Profiles: {list(profiles.keys())}")
    print(f"    Total parameters: {P}")

    # ── Step 1: compute means (no grad, fast) ──────────────────────
    with torch.no_grad():
        all_mus = {pname: [] for pname in profiles}
        n_total = 0
        for pname, prof_values in profiles.items():
            for batch in loader:
                pids, x_aug, y_pad, target_mask, static = batch
                x_aug = x_aug.to(device)
                target_mask = target_mask.to(device)
                static = static.to(device)

                x_aug_grid, obs_mask_grid = resample_xaug_to_grid(
                    x_aug, target_mask, eval_grid, n_tv)
                x_aug_cf = build_profile_xaug_continuous(
                    x_aug_grid, target_col=target_col,
                    profile_values=prof_values, n_tv=n_tv,
                    mask_type=mask_type)
                mu, *_ = model(
                    x_aug_cf, static_covariates=static,
                    obs_mask=obs_mask_grid)
                all_mus[pname].append(mu.cpu().numpy())

                if pname == list(profiles.keys())[0]:
                    n_total += x_aug.shape[0]

        for pname in profiles:
            all_mus[pname] = np.concatenate(all_mus[pname], axis=0)  # (N, L)

    print(f"    N subjects = {n_total}")

    # ── Step 2: compute gradients (with grad, subject-by-subject) ──
    #    For each profile p:  g_p[ℓ] = (1/N) Σ_i  ∂μ_i(t_ℓ)/∂θ
    #
    #    Uses retain_graph to get all L gradients from a single forward pass.
    #    Falls back to L separate forward passes if retain_graph fails.

    grad_all = {}  # {pname: (L, P)}

    for pname, prof_values in profiles.items():
        g_accum = np.zeros((L, P), dtype=np.float64)
        subj_idx = 0

        print(f"    Gradients for {pname}:", end=" ", flush=True)

        for batch in loader:
            pids, x_aug, y_pad, target_mask, static = batch
            x_aug = x_aug.to(device)
            target_mask = target_mask.to(device)
            static = static.to(device)
            N_batch = x_aug.shape[0]

            x_aug_grid, obs_mask_grid = resample_xaug_to_grid(
                x_aug, target_mask, eval_grid, n_tv)
            x_aug_cf = build_profile_xaug_continuous(
                x_aug_grid, target_col=target_col,
                profile_values=prof_values, n_tv=n_tv,
                mask_type=mask_type)

            for i in range(N_batch):
                # -- Try retain_graph approach first (1 forward, L backward) --
                try:
                    model.zero_grad()
                    mu_i, *_ = model(
                        x_aug_cf[i:i+1],
                        static_covariates=static[i:i+1],
                        obs_mask=obs_mask_grid[i:i+1],
                    )
                    for ell in range(L):
                        model.zero_grad()
                        mu_i[0, ell].backward(
                            retain_graph=(ell < L - 1))
                        g_accum[ell] += _flatten_grad(params) / n_total
                except RuntimeError:
                    # -- Fallback: L separate forward passes per subject --
                    for ell in range(L):
                        model.zero_grad()
                        mu_i, *_ = model(
                            x_aug_cf[i:i+1],
                            static_covariates=static[i:i+1],
                            obs_mask=obs_mask_grid[i:i+1],
                        )
                        mu_i[0, ell].backward()
                        g_accum[ell] += _flatten_grad(params) / n_total

                subj_idx += 1
                if subj_idx % 200 == 0:
                    print(f"{subj_idx}", end=" ", flush=True)

        grad_all[pname] = g_accum.astype(np.float64)
        print(f"done ({subj_idx})")

    # ── Step 3: assemble ci_results ────────────────────────────────
    if fisher_inv is not None:
        if isinstance(fisher_inv, np.ndarray):
            F_inv = torch.from_numpy(fisher_inv).float()
        else:
            F_inv = fisher_inv.float()
    else:
        F_inv = None

    ci_results = {}
    for pname in profiles:
        mu_mean = all_mus[pname].mean(axis=0)   # (L,)
        g = grad_all[pname]                      # (L, P)

        if F_inv is not None:
            se = np.zeros(L)
            for ell in range(L):
                gvec = torch.from_numpy(g[ell]).float()
                se[ell] = np.sqrt(max((gvec @ F_inv @ gvec).item(), 0.0))
        else:
            se = np.full(L, np.nan)

        ci_results[pname] = {
            'mean':  mu_mean,
            'grad':  g,
            'se':    se,
            'ci_lo': mu_mean - 1.96 * se,
            'ci_hi': mu_mean + 1.96 * se,
        }

    # ── Step 4: pre-compute pairwise ΔPDP SE ──────────────────────
    pnames = list(profiles.keys())
    for ia in range(len(pnames)):
        for ib in range(ia + 1, len(pnames)):
            pa, pb = pnames[ia], pnames[ib]
            delta_grad = grad_all[pa] - grad_all[pb]  # (L, P)
            delta_mean = ci_results[pa]['mean'] - ci_results[pb]['mean']

            if F_inv is not None:
                delta_se = np.zeros(L)
                for ell in range(L):
                    gvec = torch.from_numpy(delta_grad[ell]).float()
                    delta_se[ell] = np.sqrt(
                        max((gvec @ F_inv @ gvec).item(), 0.0))
            else:
                delta_se = np.full(L, np.nan)

            key = f'_delta_{pa}_vs_{pb}'
            ci_results[key] = {
                'mean': delta_mean,
                'se':   delta_se,
                'ci_lo': delta_mean - 1.96 * delta_se,
                'ci_hi': delta_mean + 1.96 * delta_se,
            }

    # ── Print SE comparison ───────────────────────────────────────
    if "late_decline" in profiles and "stable_low" in profiles:
        pa, pb = "late_decline", "stable_low"
        cross_se = (all_mus[pa] - all_mus[pb]).std(axis=0) / np.sqrt(n_total)
        dkey = f'_delta_{pa}_vs_{pb}'
        delta_se = ci_results[dkey]['se'] if dkey in ci_results else np.full(L, np.nan)
        print(f"\n  SE comparison: cross-subject vs delta-method")
        print(f"    {'Time':>8s}  {'cross-SE':>10s}  {'delta-SE':>10s}  {'ratio':>10s}")
        for ell in range(L):
            r = delta_se[ell] / cross_se[ell] if cross_se[ell] > 0 else np.nan
            print(f"    {eval_grid[ell]:8.1f}  {cross_se[ell]:10.4f}  "
                  f"{delta_se[ell]:10.4f}  {r:10.2f}")

    return ci_results, eval_grid, n_total


def build_counterfactual_xaug(x_aug, target_col, target_value, n_tv,
                               mask_type="binary", mode="constant",
                               slope=None):
    """
    Build counterfactual x_aug by intervening on a single covariate channel.

    Args:
        x_aug:        (N, T, 1+2K) original augmented input
        target_col:   int, index in [0, K-1] of the target covariate
        target_value: scalar, counterfactual value
        n_tv:         K, number of time-varying covariates
        mask_type:    "binary" or "cumulative"
        mode:         "constant", "linear", or "shifted"
        slope:        float, for linear mode

    Returns:
        x_aug_cf:     (N, T, 1+2K) counterfactual x_aug
    """
    K = n_tv
    N, T, _ = x_aug.shape
    x_aug_cf = x_aug.clone()

    # Column indices in x_aug
    cov_col = 1 + target_col              # x_interp column
    mask_col = 1 + K + target_col         # mask column

    # ── Intervene on covariate value ────────────────────────────────────
    if mode == "constant":
        x_aug_cf[:, :, cov_col] = target_value

    elif mode == "linear":
        if slope is None:
            raise ValueError("slope required for linear mode")
        t_pad = x_aug[:, :, 0]                                 # (N, T)
        x_aug_cf[:, :, cov_col] = target_value + slope * t_pad

    elif mode == "shifted":
        # Shift each subject's trajectory to have mean = target_value
        # (preserves individual dynamics, changes level)
        real_vals = x_aug[:, :, cov_col]                        # (N, T)
        real_mask = x_aug[:, :, mask_col]                       # (N, T)
        n_obs = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_subj = (real_vals * real_mask).sum(dim=1, keepdim=True) / n_obs
        x_aug_cf[:, :, cov_col] = real_vals - mean_subj + target_value

    else:
        raise ValueError(f"Unknown mode: '{mode}'")

    # ── Set mask to fully observed (the intervention is "known") ────────
    if mask_type == "binary":
        x_aug_cf[:, :, mask_col] = 1.0
    else:  # "cumulative"
        x_aug_cf[:, :, mask_col] = torch.arange(
            1, T + 1, device=x_aug.device, dtype=x_aug.dtype
        ).unsqueeze(0).expand(N, -1)

    return x_aug_cf


# ═════════════════════════════════════════════════════════════════════
#  Constant-intervention PDP (continuous time)
# ═════════════════════════════════════════════════════════════════════

def compute_pdp_continuous(
    model, loader, device, intervention_values, eval_grid,
    target_col=0, n_tv=5, mask_type="binary",
    mode="constant", slope=None,
    age_col=1, target_name="covariate",
):
    """Compute constant-value PDP on a regular time grid."""
    # from PDP_analysis_ODE_real import build_counterfactual_xaug

    model.eval()
    L = len(eval_grid)
    results = {v: [] for v in intervention_values}
    all_ages = []
    n_total = 0

    print(f"  Computing continuous-time PDP for {target_name} (mode='{mode}')")

    with torch.no_grad():
        for v in intervention_values:
            batch_mus = []

            for batch in loader:
                pids, x_aug, y_pad, target_mask, static = batch
                x_aug = x_aug.to(device)
                target_mask = target_mask.to(device)
                static = static.to(device)

                x_aug_grid, obs_mask_grid = resample_xaug_to_grid(
                    x_aug, target_mask, eval_grid, n_tv,
                )

                x_aug_cf = build_counterfactual_xaug(
                    x_aug_grid, target_col=target_col,
                    target_value=v, n_tv=n_tv,
                    mask_type=mask_type, mode=mode, slope=slope,
                )

                mu, V, Z, D, sig2, reg_dict = model(
                    x_aug_cf,
                    static_covariates=static,
                    obs_mask=obs_mask_grid,
                )
                batch_mus.append(mu.cpu().numpy())

                if v == intervention_values[0]:
                    all_ages.append(static[:, age_col].cpu().numpy())
                    n_total += x_aug.shape[0]

            results[v] = np.concatenate(batch_mus, axis=0)

    ages = np.concatenate(all_ages, axis=0)
    return results, ages, eval_grid, n_total


# ═════════════════════════════════════════════════════════════════════
#  Plotting — trajectory-profile PDP (cross-subject SE)
# ═════════════════════════════════════════════════════════════════════

def plot_trajectory_profile_pdp_continuous(
    results, eval_grid, save_path="traj_profile_pdp_continuous.png",
    target_name="covariate", visit_times=None,
):
    """
    Plot trajectory-profile PDP (warm/cold scheme, cross-subject SE).
    Single panel with all 6 profiles and full legend.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for pname in PROFILE_ORDER:
        if pname not in results:
            continue
        mu_all = results[pname]
        N = mu_all.shape[0]
        mean = mu_all.mean(axis=0)
        se = mu_all.std(axis=0) / np.sqrt(N)

        color = PROFILE_COLOURS.get(pname, "grey")
        label = PROFILE_LABELS.get(pname, pname)

        ax.plot(eval_grid, mean, '-', color=color, linewidth=1.8, label=label)
        # ax.fill_between(eval_grid, mean - 1.96 * se, mean + 1.96 * se,
        #                  color=color, alpha=0.12)

    if visit_times is not None:
        for vt in visit_times:
            ax.axvline(vt, color='grey', linestyle=':', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('E[IST]')
    ax.set_title(f'Trajectory-profile PDP of {target_name}')
    # ax.legend(loc='best', fontsize=9, framealpha=0.9)
    from profile_legend import add_profile_legend
    profiles_in_plot = [p for p in PROFILE_ORDER if p in results]
    add_profile_legend(ax, profiles_in_plot, PROFILE_COLOURS, PROFILE_LABELS,
                    loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {save_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════
#  Plotting — trajectory-profile PDP (delta-method CI)       [NEW]
# ═════════════════════════════════════════════════════════════════════

def plot_trajectory_profile_pdp_delta(
    ci_results, eval_grid, save_path="traj_profile_pdp_delta.png",
    target_name="covariate", visit_times=None,
    n_subjects=None,
):
    """
    Plot trajectory-profile PDP with delta-method confidence intervals.
    Single panel with all 6 profiles and full legend.

    Args:
        ci_results: dict from compute_trajectory_profile_pdp_with_ci()
                    {profile_name: {'mean', 'se', 'ci_lo', 'ci_hi'}}
        eval_grid:  (L,) numpy array
        save_path:  output path
        target_name: covariate name
        visit_times: canonical visit times for reference lines
        n_subjects: N (for subtitle)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for pname in PROFILE_ORDER:
        if pname not in ci_results:
            continue
        r = ci_results[pname]
        color = PROFILE_COLOURS.get(pname, "grey")
        label = PROFILE_LABELS.get(pname, pname)

        ax.plot(eval_grid, r['mean'], '-', color=color,
                linewidth=1.8, label=label)
        ax.fill_between(eval_grid, r['ci_lo'], r['ci_hi'],
                         color=color, alpha=0.15)

    if visit_times is not None:
        for vt in visit_times:
            ax.axvline(vt, color='grey', linestyle=':', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('E[IST]')
    subtitle = 'delta-method 95% CI'
    if n_subjects is not None:
        subtitle += f', N={n_subjects}'
    ax.set_title(f'Trajectory-profile PDP of {target_name}\n({subtitle})')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {save_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════
#  Plotting — constant-intervention PDP (warm→cold colormap)
# ═════════════════════════════════════════════════════════════════════

def plot_pdp_continuous(results, eval_grid, intervention_values,
                        save_path="pdp_continuous.png",
                        target_name="covariate", visit_times=None):
    """Plot constant-intervention PDP with warm/cold colormap."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(intervention_values)))

    for v_idx, v in enumerate(intervention_values):
        mu_all = results[v]
        mean = mu_all.mean(axis=0)
        se = mu_all.std(axis=0) / np.sqrt(mu_all.shape[0])

        ax.plot(eval_grid, mean, '-', color=colors[v_idx],
                label=f'{target_name}={v:.1f}', linewidth=2)
        ax.fill_between(eval_grid, mean - 1.96 * se, mean + 1.96 * se,
                         color=colors[v_idx], alpha=0.1)

    if visit_times is not None:
        for vt in visit_times:
            ax.axvline(vt, color='grey', linestyle=':', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Predicted IST')
    ax.set_title(f'Marginal PDP of {target_name} on IST (continuous time)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {save_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════
#  Plotting — ΔPDP
# ═════════════════════════════════════════════════════════════════════

def plot_delta_pdp_continuous(results, eval_grid, val_lo, val_hi,
                               save_path="delta_pdp_continuous.png",
                               target_name="covariate"):
    """Plot ΔPDP on continuous time grid."""
    delta = results[val_hi] - results[val_lo]
    N = delta.shape[0]
    mean = delta.mean(axis=0)
    se = delta.std(axis=0) / np.sqrt(N)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_grid, mean, '-', color='#C62828', linewidth=2)
    ax.fill_between(eval_grid, mean - 1.96 * se, mean + 1.96 * se,
                     color='#C62828', alpha=0.15)
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(f'ΔPDP ({target_name} {val_lo} → {val_hi})')
    ax.set_title(f'ΔPDP of {target_name} on IST (continuous time, N={N})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {save_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════
#  Plotting — ΔPDP between profiles with delta-method CI     [NEW]
# ═════════════════════════════════════════════════════════════════════

def plot_delta_profile_pdp_delta(
    ci_results, eval_grid,
    profile_a="late_decline", profile_b="late_spike",
    save_path="delta_profile_pdp.png",
    target_name="covariate", visit_times=None,
    n_subjects=None, fisher_inv=None,
):
    """
    Plot ΔPDP = PDP(profile_a) − PDP(profile_b) with delta-method CI.

    Red dots mark time points where the CI excludes zero (significant).
    """
    import torch

    a = ci_results[profile_a]
    b = ci_results[profile_b]
    delta_mean = a['mean'] - b['mean']
    delta_grad = a['grad'] - b['grad']   # (L, P)

    # Compute delta-method SE
    delta_key = f'_delta_{profile_a}_vs_{profile_b}'
    if delta_key in ci_results:
        delta_se = ci_results[delta_key]['se']
    elif '_delta_eb_ls' in ci_results and profile_a == "late_decline":
        delta_se = ci_results['_delta_eb_ls']['se']
    elif fisher_inv is not None:
        if isinstance(fisher_inv, np.ndarray):
            F_inv = torch.from_numpy(fisher_inv).float()
        else:
            F_inv = fisher_inv.float()

        delta_se = np.zeros(len(eval_grid))
        for ell in range(len(eval_grid)):
            g = torch.from_numpy(delta_grad[ell]).float()
            delta_se[ell] = np.sqrt(max((g @ F_inv @ g).item(), 0.0))
    elif '_delta_eb_ls' in ci_results:
        delta_se = ci_results['_delta_eb_ls']['se']
    else:
        delta_se = np.zeros_like(delta_mean)

    ci_lo = delta_mean - 1.96 * delta_se
    ci_hi = delta_mean + 1.96 * delta_se

    label_a = PROFILE_LABELS.get(profile_a, profile_a)
    label_b = PROFILE_LABELS.get(profile_b, profile_b)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(eval_grid, delta_mean, '-', color='#37474F', linewidth=2,
            label=f'{label_a} − {label_b}')
    ax.fill_between(eval_grid, ci_lo, ci_hi,
                     color='#546E7A', alpha=0.2, label='95% CI (delta method)')
    ax.axhline(0, color='#B71C1C', linestyle='--', linewidth=1, alpha=0.7,
               label='No difference')

    # # Mark significant time points
    # sig_times = []
    # for ell in range(len(eval_grid)):
    #     if ci_hi[ell] < 0 or ci_lo[ell] > 0:
    #         sig_times.append(eval_grid[ell])
    #         ax.plot(eval_grid[ell], delta_mean[ell], 'o',
    #                 color='#D32F2F', markersize=6, zorder=5)

    if visit_times is not None:
        for vt in visit_times:
            ax.axvline(vt, color='grey', linestyle=':', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('ΔPDP')

    subtitle_parts = [target_name, 'delta-method 95% CI']
    if n_subjects is not None:
        subtitle_parts.append(f'N={n_subjects}')
    # if sig_times:
    #     subtitle_parts.append(f'significant at {len(sig_times)}/{len(eval_grid)} times')
    # else:
    #     subtitle_parts.append('not significant at any time')

    ax.set_title(f'ΔPDP: {label_a} vs {label_b}\n({", ".join(subtitle_parts)})')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {save_path}")
    plt.close()


def plot_all_pairwise_delta_pdp(
    ci_results, eval_grid,
    save_path="delta_all_pairs.png",
    target_name="covariate", visit_times=None,
    n_subjects=None, fisher_inv=None,
):
    """
    Plot ΔPDP for three diagnostic pairs side by side:
      - early_burden vs late_spike  (path-dependence)
      - stable_high vs stable_low   (level effect)
      - gradual_decline vs gradual_rise (trend effect)
    """
    import torch

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
    pairs = [(a, b) for a, b in pairs
             if a in ci_results and b in ci_results]

    if not pairs:
        print("  No profile pairs available for pairwise ΔPDP plot")
        return

    if isinstance(fisher_inv, np.ndarray):
        F_inv = torch.from_numpy(fisher_inv).float()
    elif fisher_inv is not None:
        F_inv = fisher_inv.float()
    else:
        F_inv = None

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    if len(pairs) == 1:
        axes = [axes]

    pair_colors = ['#37474F', '#1B5E20', '#4A148C', "#C62828", "#E64A19", "#F57C00"]

    # First pass: compute all CIs to find global y range
    all_ci_lo = []
    all_ci_hi = []
    all_data = []

    for idx, (pa, pb) in enumerate(pairs):
        a, b = ci_results[pa], ci_results[pb]
        delta_mean = a['mean'] - b['mean']
        delta_grad = a['grad'] - b['grad']

        delta_key = f'_delta_{pa}_vs_{pb}'
        if delta_key in ci_results:
            delta_se = ci_results[delta_key]['se']
        elif F_inv is not None:
            delta_se = np.zeros(len(eval_grid))
            for ell in range(len(eval_grid)):
                g = torch.from_numpy(delta_grad[ell]).float()
                delta_se[ell] = np.sqrt(max((g @ F_inv @ g).item(), 0.0))
        else:
            delta_se = np.zeros_like(delta_mean)

        ci_lo = delta_mean - 1.96 * delta_se
        ci_hi = delta_mean + 1.96 * delta_se
        all_ci_lo.append(ci_lo.min())
        all_ci_hi.append(ci_hi.max())
        all_data.append((delta_mean, delta_se, ci_lo, ci_hi))

    # Global y limits with 5% padding
    y_min = min(all_ci_lo)
    y_max = max(all_ci_hi)
    y_pad = 0.05 * (y_max - y_min)
    shared_ylim = (y_min - y_pad, y_max + y_pad)

    # Second pass: plot
    for idx, (pa, pb) in enumerate(pairs):
        ax = axes[idx]
        delta_mean, delta_se, ci_lo, ci_hi = all_data[idx]
        color = pair_colors[idx % len(pair_colors)]

        ax.plot(eval_grid, delta_mean, '-', color=color, linewidth=2)
        ax.fill_between(eval_grid, ci_lo, ci_hi, color=color, alpha=0.15)
        ax.axhline(0, color='#B71C1C', linestyle='--', linewidth=1, alpha=0.7)

        # n_sig = 0
        # for ell in range(len(eval_grid)):
        #     if ci_hi[ell] < 0 or ci_lo[ell] > 0:
        #         ax.plot(eval_grid[ell], delta_mean[ell], 'o',
        #                 color='#D32F2F', markersize=5, zorder=5)
        #         n_sig += 1

        if visit_times is not None:
            for vt in visit_times:
                ax.axvline(vt, color='grey', linestyle=':', alpha=0.2, linewidth=0.5)

        ax.set_xlabel('Time (years)', fontsize=13)
        if idx % int(len(pairs)/4) == 0:
            ax.set_ylabel('ΔPDP', fontsize=13)
        ax.tick_params(labelsize=11)

        ax.set_ylim(shared_ylim)

        label_a = PROFILE_LABELS.get(pa, pa)
        label_b = PROFILE_LABELS.get(pb, pb)
        # sig_str = f'{n_sig}/{len(eval_grid)} sig.' if n_sig > 0 else 'n.s.'
        ax.set_title(f'{label_a}− {label_b}', fontsize=18)
        ax.grid(True, alpha=0.3)

    # fig.suptitle(f'Pairwise ΔPDP for {target_name} (delta-method 95% CI)',
    #              fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {save_path}")
    plt.close()