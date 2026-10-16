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
    Reproduce what _integrate_manual feeds f_phi at time t:
      - values: linear interpolation between slots
      - masks:  mask of the LAST slot with time <= t. Missed visits are slots
                with a duplicated time (dt = 0); the integrator skips them and
                then uses their mask, so the last duplicate is the one in force.
    obs_mask is not used (kept for signature compatibility).
    """
    K = n_tv
    N, T, D = x_aug.shape
    xa = x_aug.detach().cpu().numpy()
    grid = np.asarray(eval_grid, dtype=xa.dtype)
    L = len(grid)

    out = np.zeros((N, L, D), dtype=xa.dtype)
    out[:, :, 0] = grid

    for i in range(N):
        t = xa[i, :, 0]                                   # non-decreasing, may repeat
        idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, T - 1)
        t_u, first = np.unique(t, return_index=True)      # values equal at duplicates
        for k in range(K):
            c, m = 1 + k, 1 + K + k
            out[i, :, c] = np.interp(grid, t_u, xa[i, first, c])
            out[i, :, m] = xa[i, idx, m]

    kw = dict(device=x_aug.device, dtype=x_aug.dtype)
    return torch.as_tensor(out, **kw), torch.ones(N, L, **kw)

def build_counterfactual_profile_xaug(x_aug_grid, target_col, profile_values, n_tv,
                                      mask_type="binary", target_mask="real",
                                      visit_count=None):
    """
    Replace the target covariate's values with v(eval_grid).

    target_mask:
      "real"     -> keep the subject's real held mask (eq. 3.13); same for both mask types
      "observed" -> fictive covariate treated as measured at every visit:
                    binary: 1 ; cumulative: number of visits <= t (needs visit_count)
    """
    K = n_tv
    N, L, _ = x_aug_grid.shape
    x_aug_cf = x_aug_grid.clone()
    cov_col = 1 + target_col
    mask_col = 1 + K + target_col

    prof = torch.as_tensor(profile_values, device=x_aug_grid.device, dtype=x_aug_grid.dtype)
    assert prof.shape == (L,), "profile_values must be v(eval_grid)"
    x_aug_cf[:, :, cov_col] = prof.unsqueeze(0).expand(N, -1)

    if target_mask == "real":
        pass
    elif target_mask == "observed":
        if mask_type == "binary":
            x_aug_cf[:, :, mask_col] = 1.0
        elif mask_type == "cumulative":
            if visit_count is None:
                raise ValueError("cumulative 'observed' mask needs visit_count from resampling")
            x_aug_cf[:, :, mask_col] = visit_count
        else:
            raise ValueError(f"unknown mask_type {mask_type!r}")
    else:
        raise ValueError(f"unknown target_mask {target_mask!r}")

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

                x_aug_cf = build_counterfactual_profile_xaug(
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
#  Plotting — trajectory-profile PDP (cross-subject SE)
# ═════════════════════════════════════════════════════════════════════
def plot_profile_pdp(
    results, eval_grid, save_path="traj_profile_pdp.png", visit_times=None,
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
    # ax.set_title(f'Trajectory-profile PDP of {target_name}')
    # ax.legend(loc='best', fontsize=9, framealpha=0.9)
    from utils import add_profile_legend
    profiles_in_plot = [p for p in PROFILE_ORDER if p in results]
    add_profile_legend(ax, profiles_in_plot, PROFILE_COLOURS, PROFILE_LABELS,
                    loc='best')
    # ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, format='eps')
    print(f"  → {save_path}")
    plt.close()

# def plot_all_pairwise_delta_pdp(
#     ci_results, eval_grid,
#     save_path="delta_all_pairs.eps",
#     target_name="covariate", visit_times=None,
#     n_subjects=None, fisher_inv=None,
# ):
#     """
#     Plot ΔPDP for three diagnostic pairs side by side:
#       - early_burden vs late_spike  (path-dependence)
#       - stable_high vs stable_low   (level effect)
#       - gradual_decline vs gradual_rise (trend effect)
#     """
#     import torch

#     pairs = [
#         # ("late_decline", "late_spike"),
#         # ("stable_high", "stable_low"),
#         # ("gradual_decline", "gradual_rise"),
#         ("late_decline", "stable_low"),
#         # ("late_decline", "gradual_decline"),
#         ("stable_high", "late_spike"),
#         ("stable_high", "gradual_rise"),
#         ("gradual_decline", "stable_low"),
#     ]
#     pairs = [(a, b) for a, b in pairs
#              if a in ci_results and b in ci_results]

#     if not pairs:
#         print("  No profile pairs available for pairwise ΔPDP plot")
#         return

#     if isinstance(fisher_inv, np.ndarray):
#         F_inv = torch.from_numpy(fisher_inv).float()
#     elif fisher_inv is not None:
#         F_inv = fisher_inv.float()
#     else:
#         F_inv = None

#     fig, axes = plt.subplots(2, 2, figsize=(18, 12))
#     axes = axes.flatten()
#     if len(pairs) == 1:
#         axes = [axes]

#     pair_colors = ['#37474F', '#1B5E20', '#4A148C', "#C62828", "#E64A19", "#F57C00"]
#     pair_fill_colors = [
#     '#B0BEC5', '#C8E6C9', '#E1BEE7',
#     '#FFCDD2']

#     # First pass: compute all CIs to find global y range
#     all_ci_lo = []
#     all_ci_hi = []
#     all_data = []

#     for idx, (pa, pb) in enumerate(pairs):
#         a, b = ci_results[pa], ci_results[pb]
#         delta_mean = a['mean'] - b['mean']
#         delta_grad = a['grad'] - b['grad']

#         delta_key = f'_delta_{pa}_vs_{pb}'
#         if delta_key in ci_results:
#             delta_se = ci_results[delta_key]['se']
#         elif F_inv is not None:
#             delta_se = np.zeros(len(eval_grid))
#             for ell in range(len(eval_grid)):
#                 g = torch.from_numpy(delta_grad[ell]).float()
#                 delta_se[ell] = np.sqrt(max((g @ F_inv @ g).item(), 0.0))
#         else:
#             delta_se = np.zeros_like(delta_mean)

#         ci_lo = delta_mean - 1.96 * delta_se
#         ci_hi = delta_mean + 1.96 * delta_se
#         all_ci_lo.append(ci_lo.min())
#         all_ci_hi.append(ci_hi.max())
#         all_data.append((delta_mean, delta_se, ci_lo, ci_hi))

#     # Global y limits with 5% padding
#     y_min = min(all_ci_lo)
#     y_max = max(all_ci_hi)
#     y_pad = 0.05 * (y_max - y_min)
#     shared_ylim = (y_min - y_pad, y_max + y_pad)

#     # Second pass: plot
#     for idx, (pa, pb) in enumerate(pairs):
#         ax = axes[idx]
#         delta_mean, delta_se, ci_lo, ci_hi = all_data[idx]
#         color = pair_colors[idx % len(pair_colors)]
#         fill  = pair_fill_colors[idx % len(pair_fill_colors)]

#         ax.plot(eval_grid, delta_mean, '-', color=color, linewidth=2)
#         ax.fill_between(eval_grid, ci_lo, ci_hi, color=fill, alpha=0.4)
#         ax.axhline(0, color='#B71C1C', linestyle='--', linewidth=1, alpha=0.7)

#         # n_sig = 0
#         # for ell in range(len(eval_grid)):
#         #     if ci_hi[ell] < 0 or ci_lo[ell] > 0:
#         #         ax.plot(eval_grid[ell], delta_mean[ell], 'o',
#         #                 color='#D32F2F', markersize=5, zorder=5)
#         #         n_sig += 1

#         if visit_times is not None:
#             for vt in visit_times:
#                 ax.axvline(vt, color='grey', linestyle=':', alpha=0.2, linewidth=0.5)

#         ax.set_xlabel('Time (years)', fontsize=13)
#         if idx % int(len(pairs)/4) == 0:
#             ax.set_ylabel(r'$\Delta\hat{Y}^{*}(t)$', fontsize=13)
#         ax.tick_params(labelsize=11)

#         ax.set_ylim(shared_ylim)

#         label_a = PROFILE_LABELS.get(pa, pa)
#         label_b = PROFILE_LABELS.get(pb, pb)
#         # sig_str = f'{n_sig}/{len(eval_grid)} sig.' if n_sig > 0 else 'n.s.'
#         ax.set_title(f'{label_a}− {label_b}', fontsize=18)
#         # ax.grid(True, alpha=0.3)

#     # fig.suptitle(f'Pairwise ΔPDP for {target_name} (delta-method 95% CI)',
#     #              fontsize=12, y=1.02)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, format='eps')
#     print(f"  → {save_path}")
#     plt.close()

def plot_all_pairwise_delta_pdp(
    ci_results, eval_grid,
    save_path="delta_all_pairs.eps",
    target_name="covariate", visit_times=None,
    n_subjects=None, fisher_inv=None,
    profiles=None,          # dict: profile_name -> array of covariate values over eval_grid
    profile_grid=None,      # time grid for profiles (defaults to eval_grid)
):
    import torch
    from matplotlib.gridspec import GridSpec

    pairs = [
        ("late_decline", "stable_low"),
        ("stable_high", "late_spike"),
        ("stable_high", "gradual_rise"),
        ("gradual_decline", "stable_low"),
    ]
    pairs = [(a, b) for a, b in pairs
             if a in ci_results and b in ci_results]

    if not pairs:
        print("  No profile pairs available for pairwise ΔPDP plot")
        return

    if profile_grid is None:
        profile_grid = eval_grid

    if isinstance(fisher_inv, np.ndarray):
        F_inv = torch.from_numpy(fisher_inv).float()
    elif fisher_inv is not None:
        F_inv = fisher_inv.float()
    else:
        F_inv = None

    n_pairs = len(pairs)
    nrows = 2
    ncols = (n_pairs + 1) // 2

    # GridSpec: for each pair, a narrow profile column + a wide contrast column
    fig = plt.figure(figsize=(11 * ncols, 6 * nrows))
    gs = GridSpec(nrows, ncols * 2, figure=fig,
                 width_ratios=[2, 5] * ncols, wspace=0.5, hspace=0.5)

    pair_colors = ['#37474F', '#1B5E20', '#4A148C', '#C62828']
    pair_fill_colors = ['#B0BEC5', '#C8E6C9', '#E1BEE7', '#FFCDD2']
    profile_colors = {'a': '#D32F2F', 'b': '#1565C0'}  # red vs blue for the two profiles

    # --- First pass: compute all CIs for shared y-limits ---
    all_data = []
    all_ci_lo, all_ci_hi = [], []

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

    y_min, y_max = min(all_ci_lo), max(all_ci_hi)
    y_pad = 0.05 * (y_max - y_min)
    shared_ylim = (y_min - y_pad, y_max + y_pad)

    # --- Second pass: plot ---
    for idx, (pa, pb) in enumerate(pairs):
        row = idx // ncols
        col = idx % ncols

        ax_prof = fig.add_subplot(gs[row, col * 2])       # narrow: profiles
        ax_delta = fig.add_subplot(gs[row, col * 2 + 1])   # wide: contrast

        delta_mean, delta_se, ci_lo, ci_hi = all_data[idx]
        color = pair_colors[idx % len(pair_colors)]
        fill = pair_fill_colors[idx % len(pair_fill_colors)]

        label_a = PROFILE_LABELS.get(pa, pa)
        label_b = PROFILE_LABELS.get(pb, pb)

        # ── Profile panel ──
        # if profiles is not None and pa in profiles and pb in profiles:
        #     ax_prof.plot(profile_grid, profiles[pa], '-',
        #                  color=profile_colors['a'], linewidth=1.8, label=label_a)
        #     ax_prof.plot(profile_grid, profiles[pb], '-',
        #                  color=profile_colors['b'], linewidth=1.8, label=label_b)
        #     ax_prof.legend(fontsize=8, loc='best', framealpha=0.7)
        # ax_prof.set_xlabel('Time (years)', fontsize=10)
        # ax_prof.set_ylabel(target_name, fontsize=10)
        # ax_prof.tick_params(labelsize=9)
        # ax_prof.set_title('Profiles', fontsize=11)

        # ── Contrast panel ──
        ax_delta.plot(eval_grid, delta_mean, '-', color=color, linewidth=2)
        ax_delta.fill_between(eval_grid, ci_lo, ci_hi, color=fill, alpha=0.4)
        ax_delta.axhline(0, color='#B71C1C', linestyle='--', linewidth=1, alpha=0.7)

        if profiles is not None and pa in profiles and pb in profiles:
            ax_prof.plot(profile_grid, profiles[pa], '-',
                        color=profile_colors['a'], linewidth=1.5, label=label_a)
            ax_prof.plot(profile_grid, profiles[pb], '-',
                        color=profile_colors['b'], linewidth=1.5, label=label_b)
            ax_prof.legend(fontsize=7, loc='best', framealpha=0.7)
        ax_prof.set_xlabel('Time', fontsize=8)
        ax_prof.set_ylabel('')
        ax_prof.tick_params(labelsize=7)
        ax_prof.set_title('', fontsize=9)
        ax_prof.spines['top'].set_visible(False)
        ax_prof.spines['right'].set_visible(False)
        ax_prof.grid(False)
        ax_prof.set_yticks([])

        if visit_times is not None:
            for vt in visit_times:
                ax_delta.axvline(vt, color='grey', linestyle=':', alpha=0.2, linewidth=0.5)

        ax_delta.set_xlabel('Time (years)', fontsize=13)
        ax_delta.set_ylabel(r'$\Delta\hat{Y}^{*}(t)$', fontsize=13)
        ax_delta.tick_params(labelsize=11)
        ax_delta.set_ylim(shared_ylim)
        ax_delta.set_title(f'{label_a} − {label_b}', fontsize=14)

    plt.savefig(save_path, dpi=300, format='eps')
    print(f"  → {save_path}")
    plt.close()

def _build_dense_grid(visit_times, dense_step=1.0):
    """Dense integration grid that includes the eval (report) times.
 
    For fixed-step ODE solvers the integration step size follows the spacing of
    the grid points, so the accumulated ∫₀ᵗ BMI dτ (cumulative scenario) is only
    approximated well on a fine, zero-anchored mesh. This returns a dense grid
    (superset of `visit_times`, spacing ≤ dense_step) plus the indices of the
    report times within it.
 
    NOTE: for the cumulative scenario `visit_times` should start at 0 so the
    integration begins at the accumulation origin.
 
    Returns:
        t_dense:      (M,) sorted numpy array (⊇ visit_times)
        eval_indices: list[int] — positions of visit_times in t_dense
    """
    visit_times = np.asarray(visit_times, dtype=float)
    t_min, t_max = float(visit_times[0]), float(visit_times[-1])
    t_dense = np.arange(t_min, t_max + 1e-9, dense_step)
    t_dense = np.union1d(t_dense, visit_times)
    t_dense = np.sort(t_dense)
    eval_indices = [int(np.searchsorted(t_dense, vt)) for vt in visit_times]
    return t_dense, eval_indices
