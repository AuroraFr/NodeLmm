"""
PDP.py — PDP construction + plotting.

Provides:
  - trajectory profiles + evaluation grid
  - resample_xaug_to_grid : irregular visits -> continuous time
  - counterfactual / profile x_aug builders
  - PDP (no CI)
  - all plotting (profiles, constant-intervention, delta, pairwise-delta)

"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

PROFILE_ORDER = [
    "stable_low", "late_spike", "gradual_rise",
    "stable_high", "late_decline", "gradual_decline",
]

def make_eval_grid(t_max=14.0, n_points=15):
    """Regular time grid from 0 to t_max (inclusive)."""
    return np.linspace(0, t_max, n_points)

def make_profiles(eval_grid, v_lo, v_hi):
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

def resample_xaug_to_grid(x_aug, obs_mask, eval_grid, n_tv):
    """
    Resample x_aug from its original (irregular) time slots onto a continuous grid.

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

def build_couterfactual_profile_xaug(x_aug_grid, target_col, profile_values,
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

def compute_pdp_by_profile(
    model, loader, device, profiles, eval_grid,
    target_col=0, n_tv=5, mask_type="binary",
    target_name="covariate"
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

                x_aug_cf = build_couterfactual_profile_xaug(
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

def plot_pdp(results, eval_grid, intervention_values,
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
    # ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {save_path}")
    plt.close()

def plot_all_pairwise_delta_pdp(
    ci_results, eval_grid,
    save_path="delta_all_pairs.eps",
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
    pair_fill_colors = [
    '#B0BEC5', '#C8E6C9', '#E1BEE7',
    '#FFCDD2']

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
        fill  = pair_fill_colors[idx % len(pair_fill_colors)]

        ax.plot(eval_grid, delta_mean, '-', color=color, linewidth=2)
        ax.fill_between(eval_grid, ci_lo, ci_hi, color=fill, alpha=0.4)
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
        # ax.grid(True, alpha=0.3)

    # fig.suptitle(f'Pairwise ΔPDP for {target_name} (delta-method 95% CI)',
    #              fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, format='eps')
    print(f"  → {save_path}")
    plt.close()