"""
Trajectory-profile PDP analysis for Neural ODE model — Scenario 5 diagnostic.

Scenario 5 DGP: h5(t) = -0.05 * integral_0^t BMI(tau) dtau
The cognitive effect depends on CUMULATIVE BMI burden, not current BMI.

This module constructs counterfactual BMI trajectory profiles that dissociate
current BMI from cumulative BMI, then checks which pathway the model uses:
  - If the model learned the integral → early_burden predicts worse cognition
  - If the model uses the instantaneous skip → late_spike predicts worse cognition

Profiles (all defined relative to each subject's own follow-up duration Ti):
  1. stable_low:   BMI(t) = bmi_lo  for all t
  2. stable_high:  BMI(t) = bmi_hi  for all t
  3. late_spike:   BMI = bmi_lo until 0.75*Ti, ramp to bmi_hi by Ti
  4. early_burden: BMI = bmi_hi until 0.50*Ti, ramp down to bmi_lo by 0.60*Ti

Key diagnostic pair: late_spike vs early_burden
  At the final visit:  late_spike has HIGH current BMI, LOW cumulative
                        early_burden has LOW current BMI, HIGH cumulative
  Under true S5 DGP:   early_burden → worse cognition

Usage:
    python PDP_profiles_ODE.py
    python PDP_profiles_ODE.py --checkpoint checkpoints/best_model_ode_full_skip_0.pt
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict


# ─────────────────────────────────────────────
# Profile construction
# ─────────────────────────────────────────────

def make_profiles(t_pad, mask, bmi_lo=22.0, bmi_hi=30.0):
    """
    Construct counterfactual BMI trajectory profiles for each subject.

    All profiles are defined relative to each subject's own follow-up
    duration Ti = max observed time, so subjects with different visit
    schedules get the same trajectory *shape*.

    Args:
        t_pad:  (N, T) padded observation times
        mask:   (N, T) binary observation mask
        bmi_lo: low BMI level (default 22 = healthy)
        bmi_hi: high BMI level (default 30 = obese class I)

    Returns:
        profiles: OrderedDict {name: (N, T) BMI tensor}
    """
    N, T = t_pad.shape

    # Per-subject max observed time
    obs = mask > 0.5
    # Use large negative for unobserved so they don't affect max
    t_masked = t_pad.clone()
    t_masked[~obs] = -1.0
    T_max = t_masked.max(dim=1, keepdim=True).values  # (N, 1)
    T_max = T_max.clamp(min=1.0)

    t_frac = t_pad / T_max  # (N, T) fractional time in [0, ~1]

    profiles = OrderedDict()

    # 1. Stable low: constant at bmi_lo
    profiles["stable_low"] = torch.full_like(t_pad, bmi_lo)

    # 2. Stable high: constant at bmi_hi
    profiles["stable_high"] = torch.full_like(t_pad, bmi_hi)

    # 3. Late spike: bmi_lo until 75% of follow-up, linear ramp to bmi_hi
    #    t_frac < 0.75 → bmi_lo
    #    0.75 ≤ t_frac ≤ 1.0 → linear interpolation
    alpha = ((t_frac - 0.75) / 0.25).clamp(0.0, 1.0)
    profiles["late_spike"] = bmi_lo + (bmi_hi - bmi_lo) * alpha

    # 4. Early burden: bmi_hi until 50%, ramp down to bmi_lo by 60%, stay low
    #    t_frac < 0.50 → bmi_hi
    #    0.50 ≤ t_frac ≤ 0.60 → linear ramp down
    #    t_frac > 0.60 → bmi_lo
    beta = ((t_frac - 0.50) / 0.10).clamp(0.0, 1.0)
    profiles["late_decline"] = bmi_hi - (bmi_hi - bmi_lo) * beta

    # 5. Gradual rise: linear from bmi_lo at t=0 to bmi_hi at t=Ti
    profiles["gradual_rise"] = bmi_lo + (bmi_hi - bmi_lo) * t_frac.clamp(0.0, 1.0)

    # 6. Gradual decline: linear from bmi_hi at t=0 to bmi_lo at t=Ti
    profiles["gradual_decline"] = bmi_hi - (bmi_hi - bmi_lo) * t_frac.clamp(0.0, 1.0)

    return profiles


def compute_profile_integrals(t_pad, mask, profiles):
    """
    Compute the approximate cumulative BMI integral for each profile and subject
    using the trapezoidal rule on observed time points.

    Returns:
        integrals: dict {name: (N,) tensor of integral_0^Ti BMI(tau) dtau}
    """
    integrals = {}
    N, T = t_pad.shape

    for name, bmi_profile in profiles.items():
        integral = torch.zeros(N)
        for i in range(N):
            obs_idx = torch.where(mask[i] > 0.5)[0]
            if len(obs_idx) < 2:
                continue
            t_obs = t_pad[i, obs_idx]
            bmi_obs = bmi_profile[i, obs_idx]
            # Trapezoidal rule
            dt = t_obs[1:] - t_obs[:-1]
            bmi_avg = 0.5 * (bmi_obs[1:] + bmi_obs[:-1])
            integral[i] = (dt * bmi_avg).sum()
        integrals[name] = integral

    return integrals


# ─────────────────────────────────────────────
# PDP computation with trajectory profiles
# ─────────────────────────────────────────────

def compute_pdp_profiles(model, loader, device, bmi_lo=22.0, bmi_hi=30.0,
                         bmi_col=0, profile_names=None):
    """
    Compute PDP for trajectory-profile BMI interventions.

    For each profile, replaces the entire BMI trajectory in x_pad (and bmi_t)
    with the counterfactual, runs the full ODE forward pass, and collects
    the population mean mu(t).

    Unlike scalar PDP (constant BMI), here the ODE dynamics ARE affected
    because bmi_t changes the z(t) trajectory when BMI is in the dynamics.

    Args:
        model:         NeuralODEModel
        loader:        DataLoader
        device:        torch device
        bmi_lo, bmi_hi: BMI levels for profile construction
        bmi_col:       column index of BMI in x_pad (default 0)
        profile_names: list of profile names to compute (None = all)

    Returns:
        results:    dict {profile_name: (N, max_T) population mean predictions}
        profiles:   dict {profile_name: (N, max_T) BMI trajectories used}
        integrals:  dict {profile_name: (N,) cumulative BMI integrals}
        ages:       (N,) tensor of AGEc values
        masks:      (N, max_T) observation mask
        times:      (N, max_T) real times
    """
    model.eval()

    # --- First pass: collect all data to build profiles ---
    all_t_pad, all_x_pad, all_mask, all_static = [], [], [], []
    all_y_pad, all_c_mask = [], []

    with torch.no_grad():
        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            all_t_pad.append(t_pad)
            all_x_pad.append(x_pad)
            all_y_pad.append(y_pad)
            all_c_mask.append(c_mask)
            all_mask.append(mask)
            all_static.append(s)

    # Pad to common max T across batches
    max_T = max(t.shape[1] for t in all_t_pad)
    t_all = _pad_and_cat_2d(all_t_pad, max_T)
    x_all = _pad_and_cat_3d(all_x_pad, max_T)
    y_all = _pad_and_cat_2d(all_y_pad, max_T)
    mask_all = _pad_and_cat_2d(all_mask, max_T)
    s_all = torch.cat(all_static, dim=0)

    N = t_all.shape[0]
    ages = s_all[:, 1]  # AGEc is col 1

    print(f"  Building profiles (N={N}, max_T={max_T}, "
          f"bmi_lo={bmi_lo}, bmi_hi={bmi_hi})")

    # --- Build profiles ---
    all_profiles = make_profiles(t_all, mask_all, bmi_lo=bmi_lo, bmi_hi=bmi_hi)

    if profile_names is not None:
        all_profiles = OrderedDict(
            (k, v) for k, v in all_profiles.items() if k in profile_names
        )

    # --- Compute integrals ---
    integrals = compute_profile_integrals(t_all, mask_all, all_profiles)

    # Print integral summary
    print(f"\n  Profile integral summary (mean ± std of integral_0^Ti BMI(tau) dtau):")
    for name, integ in integrals.items():
        obs_mask_any = mask_all.sum(dim=1) > 1
        vals = integ[obs_mask_any]
        print(f"    {name:20s}: {vals.mean():8.1f} ± {vals.std():6.1f}")

    # --- Forward pass for each profile ---
    results = {}
    BATCH = 256  # process in chunks to avoid OOM

    print(f"\n  Computing forward passes...")
    with torch.no_grad():
        for prof_name, bmi_profile in all_profiles.items():
            print(f"    Profile: {prof_name}")
            all_mu = []

            for start in range(0, N, BATCH):
                end = min(start + BATCH, N)

                t_b = t_all[start:end].to(device)
                x_b = x_all[start:end].to(device)
                m_b = mask_all[start:end].to(device)
                s_b = s_all[start:end].to(device)

                # Counterfactual: replace BMI with profile
                x_cf = x_b.clone()
                bmi_cf = bmi_profile[start:end].to(device)
                x_cf[:, :, bmi_col] = bmi_cf

                # bmi_t for ODE dynamics: (N, T, 1)
                bmi_t_cf = bmi_cf.unsqueeze(-1)

                mu, V, _, _, _, _= model(
                    t_b, x_cf, masks=None,
                    static_covariates=s_b,
                    bmi_t=bmi_t_cf,
                    obs_mask=m_b
                )
                all_mu.append(mu.cpu())

            results[prof_name] = torch.cat(all_mu, dim=0)

    return results, all_profiles, integrals, ages, mask_all, t_all


# ─────────────────────────────────────────────
# Profile diagnostic: cumulative vs instantaneous
# ─────────────────────────────────────────────

def compute_profile_diagnostic(results, integrals, masks, times, ages,
                               visit_times=None, true_coeff=-0.05):
    """
    The core S5 diagnostic: does the model use cumulative or instantaneous BMI?

    Compares predictions across profiles at each visit time and checks
    whether the ordering matches the cumulative integral (true S5)
    or the current BMI value (instantaneous shortcut).

    Args:
        results:    dict {profile_name: (N, T) predictions}
        integrals:  dict {profile_name: (N,) cumulative integrals}
        masks:      (N, T) observation mask
        times:      (N, T) real times
        ages:       (N,) AGEc values
        visit_times: array of times at which to evaluate
        true_coeff: coefficient in h5(t) = true_coeff * integral (default -0.05)

    Returns:
        diagnostic: dict with comparison results
    """
    masks_np = masks.numpy()
    times_np = times.numpy()
    ages_np = ages.numpy()

    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])

    profiles_to_compare = list(results.keys())
    N = masks_np.shape[0]

    print(f"\n{'='*70}")
    print(f"SCENARIO 5 PROFILE DIAGNOSTIC")
    print(f"  True DGP: h5(t) = {true_coeff} * integral_0^t BMI(tau) dtau")
    print(f"{'='*70}")

    # --- 1. Mean predictions per profile at each visit time ---
    print(f"\n  Mean predictions by profile:")
    print(f"  {'Time':>6s}", end="")
    for name in profiles_to_compare:
        print(f"  {name:>16s}", end="")
    print(f"  {'n':>6s}")
    print(f"  {'-'*(8 + 18*len(profiles_to_compare) + 8)}")

    profile_means = {name: {} for name in profiles_to_compare}

    for vt in visit_times:
        print(f"  {vt:6.1f}", end="")
        n_obs = 0
        for name in profiles_to_compare:
            mu_np = results[name].numpy()
            preds, _  = _get_closest_preds_windowed(mu_np, masks_np, times_np, vt)
            profile_means[name][vt] = preds
            if len(preds) > 0:
                print(f"  {preds.mean():16.3f}", end="")
                n_obs = len(preds)
            else:
                print(f"  {'N/A':>16s}", end="")
        print(f"  {n_obs:6d}")

    # --- 2. Key diagnostic: late_spike vs early_burden ---
    if "late_spike" in results and "late_decline" in results:
        print(f"\n  KEY DIAGNOSTIC: late_spike vs early_burden")
        print(f"  At final visits: late_spike has HIGH current BMI, LOW cumulative")
        print(f"                   early_burden has LOW current BMI, HIGH cumulative")
        print(f"  Under true S5:   early_burden should predict WORSE cognition")
        print(f"                   (lower ISA15 = worse)")
        print()
        print(f"  {'Time':>6s}  {'late_spike':>12s}  {'late_decline':>14s}  "
              f"{'Δ(EB−LS)':>10s}  {'Signal':>12s}")
        print(f"  {'-'*62}")

        diagnostic = {}
        for vt in visit_times:
            ls = profile_means["late_spike"].get(vt, np.array([]))
            eb = profile_means["late_decline"].get(vt, np.array([]))
            if len(ls) > 10 and len(eb) > 10:
                delta = eb.mean() - ls.mean()
                if delta < -0.05:
                    signal = "CUMULATIVE"
                elif delta > 0.05:
                    signal = "INSTANTANEOUS"
                else:
                    signal = "AMBIGUOUS"

                diagnostic[vt] = {
                    "late_spike": ls.mean(),
                    "late_decline": eb.mean(),
                    "delta": delta,
                    "signal": signal,
                }
                print(f"  {vt:6.1f}  {ls.mean():12.3f}  {eb.mean():14.3f}  "
                      f"{delta:+10.3f}  {signal:>12s}")

    # --- 3. Compare with oracle integral-based predictions ---
    print(f"\n  Oracle integral comparison:")
    print(f"  {'Profile':>20s}  {'mean integral':>14s}  {'true h5(T)':>10s}")
    print(f"  {'-'*50}")
    for name in profiles_to_compare:
        obs_mask_any = masks.sum(dim=1) > 1
        integ_vals = integrals[name][obs_mask_any].numpy()
        mean_integ = integ_vals.mean()
        true_effect = true_coeff * mean_integ
        print(f"  {name:>20s}  {mean_integ:14.1f}  {true_effect:+10.3f}")

    # --- 4. Pairwise ΔPDPs ---
    print(f"\n  Pairwise ΔPDP at last available visit time:")
    last_vt = visit_times[-1]
    pairs = [
        ("stable_high", "stable_low"),
        ("late_decline", "late_spike"),
        ("gradual_decline", "gradual_rise"),
    ]
    for p_hi, p_lo in pairs:
        if p_hi in profile_means and p_lo in profile_means:
            hi = profile_means[p_hi].get(last_vt, np.array([]))
            lo = profile_means[p_lo].get(last_vt, np.array([]))
            if len(hi) > 10 and len(lo) > 10:
                delta_pred = hi.mean() - lo.mean()
                # Oracle
                integ_hi = integrals[p_hi][masks.sum(dim=1) > 1].mean().item()
                integ_lo = integrals[p_lo][masks.sum(dim=1) > 1].mean().item()
                delta_oracle = true_coeff * (integ_hi - integ_lo)
                print(f"    Δ({p_hi} − {p_lo}):")
                print(f"      Estimated = {delta_pred:+.3f}")
                print(f"      Oracle    = {delta_oracle:+.3f}")

    return diagnostic


# ─────────────────────────────────────────────
# Skip ablation test
# ─────────────────────────────────────────────

def compute_skip_ablation(model, loader, device, bmi_lo=22.0, bmi_hi=30.0,
                          bmi_col=0):
    """
    Ablation test: compare full model vs zeroing the BMI skip at eval time.

    Runs two forward passes per profile:
      1. Full model (BMI in ODE dynamics + decoder skip)
      2. Ablated: BMI in ODE dynamics, but decoder sees BMI = bmi_mean (skip nullified)

    If the skip contributes little, both should give similar predictions.
    If the skip dominates, ablating it should dramatically change predictions.

    Returns:
        results_full:    dict {profile: (N, T)} full model predictions
        results_ablated: dict {profile: (N, T)} skip-ablated predictions
    """
    model.eval()

    # Collect all data
    all_t, all_x, all_mask, all_static = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            all_t.append(t_pad)
            all_x.append(x_pad)
            all_mask.append(mask)
            all_static.append(s)

    max_T = max(t.shape[1] for t in all_t)
    t_all = _pad_and_cat_2d(all_t, max_T)
    x_all = _pad_and_cat_3d(all_x, max_T)
    mask_all = _pad_and_cat_2d(all_mask, max_T)
    s_all = torch.cat(all_static, dim=0)
    N = t_all.shape[0]

    profiles = make_profiles(t_all, mask_all, bmi_lo=bmi_lo, bmi_hi=bmi_hi)
    # Focus on the diagnostic pair
    profiles = OrderedDict(
        (k, v) for k, v in profiles.items()
        if k in ["stable_low", "stable_high", "late_spike", "late_decline"]
    )

    bmi_mean_val = model.decoder.bmi_mean.item()

    results_full = {}
    results_ablated = {}
    BATCH = 256

    print(f"\n  Skip ablation test (bmi_mean={bmi_mean_val:.2f})...")

    with torch.no_grad():
        for prof_name, bmi_profile in profiles.items():
            all_mu_full = []
            all_mu_ablated = []

            for start in range(0, N, BATCH):
                end = min(start + BATCH, N)
                t_b = t_all[start:end].to(device)
                x_b = x_all[start:end].to(device)
                m_b = mask_all[start:end].to(device)
                s_b = s_all[start:end].to(device)
                bmi_cf = bmi_profile[start:end].to(device)

                # --- Full model ---
                x_cf = x_b.clone()
                x_cf[:, :, bmi_col] = bmi_cf
                bmi_t_cf = bmi_cf.unsqueeze(-1)
                mu_full, _, _, _, _, _ = model(
                    t_b, x_cf, masks=None,
                    static_covariates=s_b, bmi_t=bmi_t_cf,
                    obs_mask=m_b, y_pad=None,
                )
                all_mu_full.append(mu_full.cpu())

                # --- Ablated: BMI in dynamics, but decoder skip sees bmi_mean ---
                x_ablated = x_b.clone()
                x_ablated[:, :, bmi_col] = bmi_mean_val  # decoder will standardize to ~0
                # ODE still sees the profile BMI
                mu_abl, _, _, _, _, _ = model(
                    t_b, x_ablated, masks=None,
                    static_covariates=s_b, bmi_t=bmi_t_cf,
                    obs_mask=m_b, y_pad=None,
                )
                all_mu_ablated.append(mu_abl.cpu())

            results_full[prof_name] = torch.cat(all_mu_full, dim=0)
            results_ablated[prof_name] = torch.cat(all_mu_ablated, dim=0)

    return results_full, results_ablated, profiles, mask_all, t_all


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_profiles(profiles, t_pad, mask, save_path="figures/bmi_profiles.png",
                  n_subjects=5, seed=42):
    """
    Visualize the BMI trajectory profiles for a few example subjects.
    """
    N = t_pad.shape[0]
    rng = np.random.RandomState(seed)

    # Pick subjects with long follow-up
    obs_counts = mask.sum(dim=1)
    long_fu = torch.where(obs_counts >= 5)[0]
    if len(long_fu) > n_subjects:
        chosen = long_fu[rng.choice(len(long_fu), n_subjects, replace=False)]
    else:
        chosen = long_fu

    n_profiles = len(profiles)
    fig, axes = plt.subplots(1, n_profiles, figsize=(4 * n_profiles, 4), sharey=True)
    if n_profiles == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(chosen)))

    for ax_idx, (prof_name, bmi_prof) in enumerate(profiles.items()):
        ax = axes[ax_idx]
        for c_idx, subj_i in enumerate(chosen):
            obs_idx = torch.where(mask[subj_i] > 0.5)[0]
            t_obs = t_pad[subj_i, obs_idx].numpy()
            bmi_obs = bmi_prof[subj_i, obs_idx].numpy()
            ax.plot(t_obs, bmi_obs, 'o-', color=colors[c_idx],
                    linewidth=1.5, markersize=3, alpha=0.7)
        ax.set_title(prof_name.replace('_', ' ').title(), fontsize=10)
        ax.set_xlabel('Time (years)')
        if ax_idx == 0:
            ax.set_ylabel('BMI (kg/m²)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(18, 35)

    plt.suptitle('Counterfactual BMI Trajectory Profiles', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Profile visualization saved to {save_path}")
    plt.close()


def plot_pdp_profiles(results, masks, times, profiles,
                      save_path="figures/pdp_profiles.png",
                      visit_times=None):
    """
    Plot PDP over time for each trajectory profile.
    """
    masks_np = masks.numpy()
    times_np = times.numpy()

    if visit_times is None:
        visit_times = np.array([0, 2, 4, 7, 10, 12])

    # Color map for profiles
    profile_colors = {
        "stable_low":      "#2196F3",  # blue
        "stable_high":     "#F44336",  # red
        "late_spike":      "#F44336",  # orange
        "late_decline":    "#2196F3",  # purple #F44336 #9C27B0
        "gradual_rise":    "#4CAF50",  # green
        "gradual_decline": "#795548",  # brown
    }
    profile_styles = {
        "stable_low":      "-",
        "stable_high":     "-",
        "late_spike":      "--",
        "late_decline":    "--",
        "gradual_rise":    ":",
        "gradual_decline": ":",
    }

    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig, ax2 = plt.subplots(figsize=(7, 5))

    # # --- Left panel: PDP curves ---
    # ax = axes[0]
    # for prof_name in results:
    #     mu_np = results[prof_name].numpy()
    #     closest = _closest_obs_per_subject_arr(mu_np, masks_np, times_np, visit_times)

    #     mean_pred, vt_plot = [], []
    #     for vt in visit_times:
    #         preds = closest[vt]
    #         if len(preds) > 10:
    #             mean_pred.append(preds.mean())
    #             vt_plot.append(vt)

    #     if vt_plot:
    #         color = profile_colors.get(prof_name, "grey")
    #         style = profile_styles.get(prof_name, "-")
    #         ax.plot(vt_plot, mean_pred, f'o{style}', color=color,
    #                 label=prof_name.replace('_', ' '), linewidth=2, markersize=5)

    # ax.set_xlabel('Time (years)')
    # ax.set_ylabel('Mean predicted ISA15')
    # ax.set_title('PDP by Trajectory Profile')
    # ax.legend(fontsize=8, loc='lower left')
    # ax.grid(True, alpha=0.3)

    # --- Right panel: diagnostic pair zoom ---
    # ax2 = axes[0]
    # print(results)
    if "late_spike" in results and "late_decline" in results:
        # for prof_name in ["late_spike", "late_decline", "stable_low", "stable_high"]:
        for prof_name in ["late_spike", "late_decline"]:
            if prof_name not in results:
                continue
            mu_np = results[prof_name].numpy()
            closest = _closest_obs_per_subject_arr(mu_np, masks_np, times_np, visit_times)

            mean_pred, vt_plot = [], []
            for vt in visit_times:
                preds = closest[vt]
                if len(preds) > 10:
                    mean_pred.append(preds.mean())
                    vt_plot.append(vt)

            if vt_plot:
                color = profile_colors.get(prof_name, "grey")
                style = profile_styles.get(prof_name, "-")
                lw = 3 if prof_name in ["late_spike", "late_decline"] else 1.5
                ax2.plot(vt_plot, mean_pred, f'o{style}', color=color,
                         label=prof_name.replace('_', ' '), linewidth=lw, markersize=6)

        ax2.set_xlabel('Time (years)')
        ax2.set_ylabel('E[IST]')
        ax2.set_title('Diagnostic Pair: Late Spike vs Late decline')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  PDP profiles plot saved to {save_path}")
    plt.close()


def plot_skip_ablation(results_full, results_ablated, masks, times,
                       save_path="figures/skip_ablation.png",
                       visit_times=None):
    """
    Plot full model vs skip-ablated predictions for each profile.
    """
    masks_np = masks.numpy()
    times_np = times.numpy()

    if visit_times is None:
        visit_times = np.array([0, 2, 4, 7, 10, 12])

    prof_names = list(results_full.keys())
    n = len(prof_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for idx, prof_name in enumerate(prof_names):
        ax = axes[idx]
        for label, res, color, ls in [
            ("Full model", results_full, "#2196F3", "-"),
            ("Skip ablated", results_ablated, "#F44336", "--"),
        ]:
            mu_np = res[prof_name].numpy()
            closest = _closest_obs_per_subject_arr(mu_np, masks_np, times_np, visit_times)
            mean_pred, vt_plot = [], []
            for vt in visit_times:
                preds = closest[vt]
                if len(preds) > 10:
                    mean_pred.append(preds.mean())
                    vt_plot.append(vt)
            if vt_plot:
                ax.plot(vt_plot, mean_pred, f'o{ls}', color=color,
                        label=label, linewidth=2, markersize=5)

        ax.set_title(prof_name.replace('_', ' ').title(), fontsize=10)
        ax.set_xlabel('Time (years)')
        if idx == 0:
            ax.set_ylabel('E[IST]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Skip Ablation: Full Model vs BMI Skip Nullified', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Skip ablation plot saved to {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _get_closest_preds_windowed(mu_np, masks_np, times_np, vt, 
                                 max_dist=1.5):
    """
    Get predictions for subjects with an observation within 
    max_dist years of target time vt.
    
    Returns:
        preds:      array of predictions
        actual_times: array of actual observation times (for oracle correction)
    """
    N = mu_np.shape[0]
    preds, actual_times = [], []
    for i in range(N):
        obs_idx = np.where(masks_np[i] > 0.5)[0]
        if len(obs_idx) == 0:
            continue
        obs_times = times_np[i, obs_idx]
        dists = np.abs(obs_times - vt)
        best = np.argmin(dists)
        if dists[best] <= max_dist:
            preds.append(mu_np[i, obs_idx[best]])
            actual_times.append(obs_times[best])
    return np.array(preds), np.array(actual_times)

def _get_closest_preds(mu_np, masks_np, times_np, vt):
    """Get closest prediction to visit time vt for all subjects."""
    N = mu_np.shape[0]
    preds = []
    for i in range(N):
        obs_idx = np.where(masks_np[i] > 0.5)[0]
        if len(obs_idx) == 0:
            continue
        obs_times = times_np[i, obs_idx]
        closest = obs_idx[np.argmin(np.abs(obs_times - vt))]
        preds.append(mu_np[i, closest])
    return np.array(preds)


def _closest_obs_per_subject_arr(mu_np, masks_np, times_np, visit_times):
    """For each visit time, collect closest prediction per subject."""
    result = {}
    for vt in visit_times:
        result[vt], _ = _get_closest_preds_windowed(mu_np, masks_np, times_np, vt)
    return result


def _pad_and_cat_2d(tensors, max_T):
    """Pad list of (N_i, T_i) tensors to (sum N_i, max_T)."""
    padded = []
    for m in tensors:
        if m.shape[1] < max_T:
            pad = torch.zeros(m.shape[0], max_T - m.shape[1])
            m = torch.cat([m, pad], dim=1)
        padded.append(m)
    return torch.cat(padded, dim=0)


def _pad_and_cat_3d(tensors, max_T):
    """Pad list of (N_i, T_i, C) tensors to (sum N_i, max_T, C)."""
    C = tensors[0].shape[2]
    padded = []
    for m in tensors:
        if m.shape[1] < max_T:
            pad = torch.zeros(m.shape[0], max_T - m.shape[1], C)
            m = torch.cat([m, pad], dim=1)
        padded.append(m)
    return torch.cat(padded, dim=0)