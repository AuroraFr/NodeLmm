"""
Continuous-time PDP analysis for the Neural ODE-LMM on the 3C cohort.

Evaluates the trajectory-profile PDP on a regular time grid (e.g. 0,1,...,14)
instead of the canonical visit times.  Every subject contributes at every
grid point — no censoring-based filtering, no max_dist alignment.

Drop-in addition to PDP_analysis_ODE_real.py.

Usage:
    from PDP_continuous_time import (
        make_profiles_continuous,
        resample_xaug_to_grid,
        compute_trajectory_profile_pdp_continuous,
        plot_trajectory_profile_pdp_continuous,
    )
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Regular evaluation grid
# ─────────────────────────────────────────────

def make_eval_grid(t_max=14.0, n_points=15):
    """Regular time grid from 0 to t_max (inclusive)."""
    return np.linspace(0, t_max, n_points)


# ─────────────────────────────────────────────
# Profiles on continuous grid
# ─────────────────────────────────────────────

def make_profiles_continuous(eval_grid, v_lo, v_hi):
    """
    Six counterfactual trajectory profiles evaluated on a continuous grid.

    Matches the shape logic of make_profiles() but works on arbitrary grids.

    Args:
        eval_grid: (L,) numpy array of evaluation times
        v_lo:      low value (e.g. Q25)
        v_hi:      high value (e.g. Q75)

    Returns:
        dict {profile_name: (L,) numpy array}
    """
    L = len(eval_grid)
    t_norm = (eval_grid - eval_grid[0]) / (eval_grid[-1] - eval_grid[0])  # [0, 1]

    profiles = {
        "stable_low":      np.full(L, v_lo),
        "stable_high":     np.full(L, v_hi),
        "late_spike":     np.where(t_norm < 0.5, v_lo, v_hi),
        "early_burden":   np.where(t_norm < 0.5, v_hi, v_lo),
        "gradual_rise":    v_lo + (v_hi - v_lo) * t_norm,
        "gradual_decline": v_hi + (v_lo - v_hi) * t_norm,
    }
    return profiles


# ─────────────────────────────────────────────
# Resample x_aug onto a regular grid
# ─────────────────────────────────────────────

def resample_xaug_to_grid(x_aug, obs_mask, eval_grid, n_tv):
    """
    Resample x_aug from its original (irregular) time slots onto a regular grid.

    For each subject and each covariate channel:
      - Identify observed slots (obs_mask > 0)
      - Linearly interpolate covariate values to eval_grid
      - Beyond last observation: LOCF (last-observation-carried-forward)
      - Before first observation: first-observation-carried-backward

    Args:
        x_aug:     (N, T_orig, 1+2K) original augmented input
        obs_mask:  (N, T_orig) binary mask of observed visits
        eval_grid: (L,) numpy array of new time points
        n_tv:      K, number of time-varying covariates

    Returns:
        x_aug_new: (N, L, 1+2K) resampled x_aug
        obs_mask_new: (N, L) all-ones mask (every grid point is "observed")
    """
    K = n_tv
    N, T_orig, D = x_aug.shape
    L = len(eval_grid)
    device = x_aug.device
    dtype = x_aug.dtype

    x_aug_new = torch.zeros(N, L, D, device=device, dtype=dtype)

    # Time column
    grid_tensor = torch.tensor(eval_grid, device=device, dtype=dtype)
    x_aug_new[:, :, 0] = grid_tensor.unsqueeze(0).expand(N, -1)

    # For each subject, interpolate covariates and masks
    for i in range(N):
        # Original time points for this subject
        t_orig = x_aug[i, :, 0].cpu().numpy()        # (T_orig,)
        obs_i = obs_mask[i].cpu().numpy() > 0.5       # (T_orig,) bool

        if not obs_i.any():
            # No observations at all — keep zeros (rare edge case)
            continue

        # Observed time points
        t_obs = t_orig[obs_i]

        # Interpolate each covariate channel
        for k in range(K):
            cov_col = 1 + k
            mask_col = 1 + K + k

            vals_orig = x_aug[i, :, cov_col].cpu().numpy()
            vals_obs = vals_orig[obs_i]

            # np.interp handles LOCF at boundaries by default
            # (clamps to first/last observed value)
            interp_vals = np.interp(eval_grid, t_obs, vals_obs)
            x_aug_new[i, :, cov_col] = torch.tensor(
                interp_vals, device=device, dtype=dtype)

            # Mask: set to fully observed on the grid
            # For cumulative mask, we set 1, 2, 3, ..., L
            mask_orig = x_aug[i, :, mask_col].cpu().numpy()
            if mask_orig.max() > 1.5:  # cumulative mask
                x_aug_new[i, :, mask_col] = torch.arange(
                    1, L + 1, device=device, dtype=dtype)
            else:  # binary mask
                x_aug_new[i, :, mask_col] = 1.0

    # All-ones obs_mask (every subject evaluated at every grid point)
    obs_mask_new = torch.ones(N, L, device=device, dtype=dtype)

    return x_aug_new, obs_mask_new


# ─────────────────────────────────────────────
# Build profile x_aug on regular grid
# ─────────────────────────────────────────────

def build_profile_xaug_continuous(x_aug_grid, target_col, profile_values,
                                   n_tv, mask_type="binary"):
    """
    Replace target covariate in resampled x_aug with profile values.

    Args:
        x_aug_grid:     (N, L, 1+2K) resampled x_aug on regular grid
        target_col:     int, index in [0, K-1]
        profile_values: (L,) array of profile values
        n_tv:           K
        mask_type:      "binary" or "cumulative"

    Returns:
        x_aug_cf: (N, L, 1+2K) counterfactual
    """
    K = n_tv
    N, L, _ = x_aug_grid.shape
    x_aug_cf = x_aug_grid.clone()

    cov_col = 1 + target_col
    mask_col = 1 + K + target_col

    # Set target covariate to profile
    prof = torch.tensor(profile_values, device=x_aug_grid.device,
                        dtype=x_aug_grid.dtype)
    x_aug_cf[:, :, cov_col] = prof.unsqueeze(0).expand(N, -1)

    # Set target mask to fully observed
    if mask_type == "binary":
        x_aug_cf[:, :, mask_col] = 1.0
    else:
        x_aug_cf[:, :, mask_col] = torch.arange(
            1, L + 1, device=x_aug_grid.device, dtype=x_aug_grid.dtype
        ).unsqueeze(0).expand(N, -1)

    return x_aug_cf


# ─────────────────────────────────────────────
# Core: continuous-time trajectory-profile PDP
# ─────────────────────────────────────────────

def compute_trajectory_profile_pdp_continuous(
    model, loader, device, profiles, eval_grid,
    target_col=0, n_tv=5, mask_type="binary",
    target_name="covariate",
):
    """
    Compute trajectory-profile PDP on a continuous regular time grid.

    Unlike compute_trajectory_profile_pdp(), this:
      - Evaluates on a user-specified regular grid (not visit times)
      - ALL subjects contribute at ALL grid points (no censoring filter)
      - Non-intervened covariates are linearly interpolated to the grid
      - Returns direct population averages (no _closest_obs alignment)

    Args:
        model:       NeuralODEModel
        loader:      DataLoader (RealDataset + collate_real)
        device:      torch device
        profiles:    dict from make_profiles_continuous(eval_grid, ...)
        eval_grid:   (L,) numpy array of evaluation times
        target_col:  column index in x_interp [0..K-1]
        n_tv:        K, number of time-varying covariates
        mask_type:   "binary" or "cumulative"
        target_name: name for printing

    Returns:
        results:   dict {profile_name: (N, L) numpy array of pop means}
        eval_grid: (L,) numpy array (echoed back for convenience)
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

                # Step 1: resample non-intervened covariates to regular grid
                x_aug_grid, obs_mask_grid = resample_xaug_to_grid(
                    x_aug, target_mask, eval_grid, n_tv,
                )

                # Step 2: replace target covariate with profile
                x_aug_cf = build_profile_xaug_continuous(
                    x_aug_grid, target_col=target_col,
                    profile_values=prof_values, n_tv=n_tv,
                    mask_type=mask_type,
                )

                # Step 3: forward pass on the regular grid
                mu, V, Z, D, sig2, reg_dict = model(
                    x_aug_cf,
                    static_covariates=static,
                    obs_mask=obs_mask_grid,
                )
                # mu: (N_batch, L) — population mean at each grid point
                batch_mus.append(mu.cpu().numpy())

                if pname == list(profiles.keys())[0]:
                    n_total += N_batch

            results[pname] = np.concatenate(batch_mus, axis=0)  # (N, L)

    # ── Print summary ────────────────────────────────────────────────
    print(f"\n    N subjects = {n_total}")
    print(f"\n    {'Profile':<20s}", end="")
    for t in eval_grid:
        print(f"  t={t:<5.1f}", end="")
    print()
    print(f"    {'-'*20 + '-'*8*L}")

    for pname in profiles:
        mu_all = results[pname]  # (N, L)
        means = mu_all.mean(axis=0)  # (L,)
        print(f"    {pname:<20s}", end="")
        for l in range(L):
            print(f"  {means[l]:>7.2f}", end="")
        print()

    # ── Diagnostic: early_burden vs late_spike ────────────────────────
    if "early_burden" in results and "late_spike" in results:
        eb = results["early_burden"].mean(axis=0)
        ls = results["late_spike"].mean(axis=0)
        diff = eb - ls
        n = results["early_burden"].shape[0]
        diff_subj = results["early_burden"] - results["late_spike"]  # (N, L)
        se = diff_subj.std(axis=0) / np.sqrt(n)

        print(f"\n    Diagnostic: early_burden − late_spike (continuous)")
        print(f"    {'Time':>8s}  {'Diff':>8s}  {'SE':>8s}  {'95% CI':>20s}  {'Interp':>20s}")
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


# ─────────────────────────────────────────────
# Constant-intervention PDP (continuous time)
# ─────────────────────────────────────────────

def compute_pdp_continuous(
    model, loader, device, intervention_values, eval_grid,
    target_col=0, n_tv=5, mask_type="binary",
    mode="constant", slope=None,
    age_col=1, target_name="covariate",
):
    """
    Compute constant-value PDP on a regular time grid.

    Same as compute_pdp() but on continuous time.
    """
    from PDP_analysis_ODE_real import build_counterfactual_xaug

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

                # Resample to regular grid
                x_aug_grid, obs_mask_grid = resample_xaug_to_grid(
                    x_aug, target_mask, eval_grid, n_tv,
                )

                # Apply constant intervention on the grid
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

            results[v] = np.concatenate(batch_mus, axis=0)  # (N, L)

    ages = np.concatenate(all_ages, axis=0)
    return results, ages, eval_grid, n_total


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_trajectory_profile_pdp_continuous(
    results, eval_grid, save_path="traj_profile_pdp_continuous.png",
    target_name="covariate", visit_times=None,
):
    """
    Plot trajectory-profile PDP on continuous time grid.

    Two panels:
      Left:  all 6 profiles with 95% CI
      Right: diagnostic pair (late_spike vs early_burden)
    """
    profile_colours = {
        "stable_low":      "#2166AC",
        "stable_high":     "#B2182B",
        "late_spike":      "#F4A582",
        "early_burden":    "#D6604D",
        "gradual_rise":    "#92C5DE",
        "gradual_decline": "#4393C3",
    }
    profile_labels = {
        "stable_low":      "Stable low",
        "stable_high":     "Stable high",
        "late_spike":      "Late spike",
        "early_burden":    "Late decline",
        "gradual_rise":    "Gradual rise",
        "gradual_decline": "Gradual decline",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left panel: all profiles ──────────────────────────────────────
    ax = axes[0]
    for pname, mu_all in results.items():
        N = mu_all.shape[0]
        mean = mu_all.mean(axis=0)
        se = mu_all.std(axis=0) / np.sqrt(N)
        lo = mean - 1.96 * se
        hi = mean + 1.96 * se

        color = profile_colours.get(pname, "grey")
        label = profile_labels.get(pname, pname)

        ax.plot(eval_grid, mean, '-', color=color, linewidth=1.8, label=label)
        ax.fill_between(eval_grid, lo, hi, color=color, alpha=0.12)

    if visit_times is not None:
        for vt in visit_times:
            ax.axvline(vt, color='grey', linestyle=':', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('E[ISA15]')
    ax.set_title(f'Trajectory-profile PDP of {target_name}\n(continuous time, N={mu_all.shape[0]})')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Right panel: diagnostic pair ──────────────────────────────────
    ax = axes[1]
    for pname in ['late_spike', 'early_burden', 'stable_low', 'stable_high']:
        if pname not in results:
            continue
        mu_all = results[pname]
        N = mu_all.shape[0]
        mean = mu_all.mean(axis=0)
        se = mu_all.std(axis=0) / np.sqrt(N)

        color = profile_colours[pname]
        label = profile_labels[pname]
        ax.plot(eval_grid, mean, '-', color=color, linewidth=1.8, label=label)
        ax.fill_between(eval_grid, mean - 1.96 * se, mean + 1.96 * se,
                         color=color, alpha=0.12)

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('E[ISA15]')
    ax.set_title('Diagnostic Pair: Late Spike vs Early Burden')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {save_path}")
    plt.close()


def plot_pdp_continuous(results, eval_grid, intervention_values,
                        save_path="pdp_continuous.png",
                        target_name="covariate", visit_times=None):
    """Plot constant-intervention PDP on continuous time grid."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(intervention_values)))

    for v_idx, v in enumerate(intervention_values):
        mu_all = results[v]  # (N, L)
        mean = mu_all.mean(axis=0)
        se = mu_all.std(axis=0) / np.sqrt(mu_all.shape[0])

        ax.plot(eval_grid, mean, '-', color=colors[v_idx],
                label=f'{target_name}={v}', linewidth=2)
        ax.fill_between(eval_grid, mean - 1.96 * se, mean + 1.96 * se,
                         color=colors[v_idx], alpha=0.1)

    if visit_times is not None:
        for vt in visit_times:
            ax.axvline(vt, color='grey', linestyle=':', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Predicted ISA15')
    ax.set_title(f'Marginal PDP of {target_name} on ISA15 (continuous time)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {save_path}")
    plt.close()


def plot_delta_pdp_continuous(results, eval_grid, val_lo, val_hi,
                               save_path="delta_pdp_continuous.png",
                               target_name="covariate"):
    """Plot ΔPDP on continuous time grid."""
    delta = results[val_hi] - results[val_lo]  # (N, L)
    N = delta.shape[0]
    mean = delta.mean(axis=0)
    se = delta.std(axis=0) / np.sqrt(N)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_grid, mean, '-', color='firebrick', linewidth=2)
    ax.fill_between(eval_grid, mean - 1.96 * se, mean + 1.96 * se,
                     color='firebrick', alpha=0.15)
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(f'ΔPDP ({target_name} {val_lo} → {val_hi})')
    ax.set_title(f'ΔPDP of {target_name} on ISA15 (continuous time, N={N})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {save_path}")
    plt.close()