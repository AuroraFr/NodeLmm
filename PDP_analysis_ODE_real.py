"""
PDP analysis for the Neural ODE-LMM on the real 3C dataset.

The model receives x_aug = [time(1), x_interp(K), mask(K)].
A covariate enters the model in two places:
  1. ODE dynamics:  dz/dt = f(z, x_interp(t), mask(t), t)
  2. Decoder skip:  skip(t) = gate ⊙ [x_interp_std(t), mask(t), static]

A full PDP intervention replaces the target covariate consistently in BOTH
x_interp and mask channels, then re-integrates the ODE from scratch.

Convention for counterfactual mask:
  - binary mask:      set to 1 at all slots (the intervention value is "known")
  - cumulative mask:  set to 1, 2, 3, ..., T

Includes:
  - build_counterfactual_xaug:       build intervened x_aug (constant/linear/shifted)
  - build_profile_xaug:             build intervened x_aug (trajectory profile)
  - compute_pdp:                     population-level PDP (mu only)
  - compute_pdp_with_blup:           subject-level ICE with BLUP random effects
  - compute_delta_pdp:               marginal ΔPDP at visit times
  - compute_delta_pdp_stratified:    ΔPDP by age tertiles
  - make_profiles:                   define 6 counterfactual trajectory shapes
  - compute_trajectory_profile_pdp:  profile PDP (path-dependence diagnostic)
  - plot_pdp, plot_pdp_marginal, plot_delta_pdp, plot_trajectory_profile_pdp
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Preprocess_3C import EXPECTED_TIMES


# ── Canonical visit grid ─────────────────────────────────────────────────────
VISIT_TIMES_3C = EXPECTED_TIMES


# ─────────────────────────────────────────────
# Counterfactual builder
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Helper: closest observation per subject
# ─────────────────────────────────────────────

def _closest_obs_per_subject(mu, masks_np, times_np, visit_times):
    """
    For each target visit time, find the closest observed time point
    per subject and return the prediction at that point.
    """
    N, T = mu.shape
    result = {}

    for vt in visit_times:
        preds = []
        for i in range(N):
            obs_idx = np.where(masks_np[i] > 0.5)[0]
            if len(obs_idx) == 0:
                continue
            obs_times = times_np[i, obs_idx]
            closest = obs_idx[np.argmin(np.abs(obs_times - vt))]
            preds.append(mu[i, closest])
        result[vt] = np.array(preds)

    return result

def _closest_obs_per_subject_2(mu, masks_np, times_np, visit_times,
                              max_dist=1.0):
    N, T = mu.shape
    result = {}
    for vt in visit_times:
        preds = []
        for i in range(N):
            obs_idx = np.where(masks_np[i] > 0.5)[0]
            if len(obs_idx) == 0:
                continue
            obs_times = times_np[i, obs_idx]
            best = np.argmin(np.abs(obs_times - vt))
            if np.abs(obs_times[best] - vt) <= max_dist:
                preds.append(mu[i, obs_idx[best]])
        result[vt] = np.array(preds)
    return result


# ─────────────────────────────────────────────
# Core PDP computation (population mean only)
# ─────────────────────────────────────────────

def compute_pdp(model, loader, device, intervention_values, target_col=0,
                n_tv=5, mask_type="binary", mode="constant", slope=None,
                age_col=1, target_name="covariate"):
    """
    Compute PDP for a target covariate.

    For each intervention value v:
      1. Build counterfactual x_aug with target set to v
      2. Re-run full forward pass (encoder + ODE + decoder)
      3. Collect population mean mu(t)

    Args:
        model:               NeuralODEModel
        loader:              DataLoader (RealDataset + collate_real)
        device:              torch device
        intervention_values: list of counterfactual values
        target_col:          column index in x_interp [0..K-1]
        n_tv:                K, number of time-varying covariates
        mask_type:           "binary" or "cumulative"
        mode:                "constant", "linear", or "shifted"
        slope:               float, for linear mode
        age_col:             AGEc column index in static covariates
        target_name:         name for printing

    Returns:
        results: dict {value: (N, T) numpy array of population means}
        ages:    (N,) numpy array of AGEc
        masks:   (N, T) numpy array of target_mask
        times:   (N, T) numpy array of times
    """
    model.eval()

    results = {v: [] for v in intervention_values}
    all_ages = []
    all_masks = []
    all_times = []

    print(f"  Computing PDP for {target_name} (col={target_col}, mode='{mode}')")

    # Estimate population slope if needed
    if mode == "linear" and slope is None:
        all_vals, all_t = [], []
        K = n_tv
        with torch.no_grad():
            for batch in loader:
                _, x_aug, _, target_mask, static = batch
                x_interp = x_aug[:, :, 1:1+K]
                t_pad = x_aug[:, :, 0]
                obs = target_mask > 0.5
                all_vals.append(x_interp[:, :, target_col][obs].numpy())
                all_t.append(t_pad[obs].numpy())
        val_flat = np.concatenate(all_vals)
        t_flat = np.concatenate(all_t)
        from numpy.polynomial.polynomial import polyfit
        c = polyfit(t_flat, val_flat, 1)
        slope = c[1]
        print(f"  Estimated population slope: {slope:.4f} per year")

    with torch.no_grad():
        for v in intervention_values:
            batch_mus = []

            for batch in loader:
                pids, x_aug, y_pad, target_mask, static = batch
                x_aug = x_aug.to(device)
                target_mask = target_mask.to(device)
                static = static.to(device)

                # Build counterfactual
                x_aug_cf = build_counterfactual_xaug(
                    x_aug, target_col=target_col,
                    target_value=v, n_tv=n_tv,
                    mask_type=mask_type, mode=mode, slope=slope,
                )

                # Forward under counterfactual
                mu, V, Z, D, sig2, reg_dict = model(
                    x_aug_cf,
                    static_covariates=static,
                    obs_mask=target_mask,
                )
                batch_mus.append(mu.cpu())

                # Collect metadata on first pass
                if v == intervention_values[0]:
                    all_ages.append(static[:, age_col].cpu())
                    all_masks.append(target_mask.cpu())
                    all_times.append(x_aug[:, :, 0].cpu())

            results[v] = torch.cat(batch_mus, dim=0).numpy()

    ages = torch.cat(all_ages, dim=0).numpy()
    masks = torch.cat(all_masks, dim=0).numpy()
    times = torch.cat(all_times, dim=0).numpy()

    return results, ages, masks, times


# ─────────────────────────────────────────────
# ICE with BLUP random effects
# ─────────────────────────────────────────────

def compute_pdp_with_blup(model, loader, device, intervention_values,
                           target_col=0, n_tv=5, mask_type="binary",
                           mode="constant", slope=None,
                           age_col=1, target_name="covariate"):
    """
    Compute subject-level ICE curves INCLUDING random effects via BLUP.

    Step 1: compute BLUP from OBSERVED data (real covariates)
    Step 2: for each counterfactual value, re-run forward and add Z @ b_hat
    """
    model.eval()

    results_pop = {v: [] for v in intervention_values}
    results_subj = {v: [] for v in intervention_values}
    all_blup = []
    all_ages = []
    all_masks = []
    all_times = []

    print(f"  Computing ICE with BLUP for {target_name} "
          f"(col={target_col}, mode='{mode}')")

    with torch.no_grad():
        # ── Step 1: BLUP from observed data ─────────────────────────────
        print("    Step 1: computing BLUP from observed data...")
        for batch in loader:
            pids, x_aug, y_pad, target_mask, static = batch
            x_aug = x_aug.to(device)
            y_pad = y_pad.to(device)
            target_mask = target_mask.to(device)
            static = static.to(device)

            mu_obs, V_obs, Z_obs, D_obs, sig2_obs, _ = model(
                x_aug,
                static_covariates=static,
                obs_mask=target_mask,
            )

            # Per-subject BLUP: b_hat_i = D Z_i^T V_i^{-1} (y_i - mu_i)
            N_batch, T = mu_obs.shape
            q = Z_obs.shape[2]
            b_hat_batch = torch.zeros(N_batch, q, device=device)

            for i in range(N_batch):
                idx = target_mask[i].bool()
                n_i = idx.sum()
                if n_i < 1:
                    continue

                r_i = y_pad[i, idx] - mu_obs[i, idx]
                V_i = V_obs[i][idx][:, idx]
                Z_i = Z_obs[i, idx]

                L_i = torch.linalg.cholesky(V_i)
                Vinv_r = torch.cholesky_solve(
                    r_i.unsqueeze(-1), L_i).squeeze(-1)

                b_hat_batch[i] = D_obs @ Z_i.t() @ Vinv_r

            all_blup.append(b_hat_batch.cpu())
            all_ages.append(static[:, age_col].cpu())
            all_masks.append(target_mask.cpu())
            all_times.append(x_aug[:, :, 0].cpu())

        blup = torch.cat(all_blup, dim=0)

        # ── Step 2: counterfactual predictions ──────────────────────────
        print("    Step 2: computing counterfactual predictions...")
        for v in intervention_values:
            batch_mus = []
            batch_subj = []
            blup_offset = 0

            for batch in loader:
                pids, x_aug, y_pad, target_mask, static = batch
                x_aug = x_aug.to(device)
                target_mask = target_mask.to(device)
                static = static.to(device)

                N_batch = x_aug.shape[0]

                x_aug_cf = build_counterfactual_xaug(
                    x_aug, target_col=target_col,
                    target_value=v, n_tv=n_tv,
                    mask_type=mask_type, mode=mode, slope=slope,
                )

                mu_cf, V_cf, Z_cf, D_cf, sig2_cf, _ = model(
                    x_aug_cf,
                    static_covariates=static,
                    obs_mask=target_mask,
                )

                # Subject-level: mu + Z @ b_hat
                b_batch = blup[blup_offset:blup_offset + N_batch].to(device)
                Zb = (Z_cf * b_batch.unsqueeze(1)).sum(dim=-1)
                y_subj = mu_cf + Zb

                batch_mus.append(mu_cf.cpu())
                batch_subj.append(y_subj.cpu())
                blup_offset += N_batch

            results_pop[v] = torch.cat(batch_mus, dim=0).numpy()
            results_subj[v] = torch.cat(batch_subj, dim=0).numpy()

    ages = torch.cat(all_ages, dim=0).numpy()
    masks = torch.cat(all_masks, dim=0).numpy()
    times = torch.cat(all_times, dim=0).numpy()

    return results_pop, results_subj, blup, ages, masks, times


# ─────────────────────────────────────────────
# ΔPDP computation
# ─────────────────────────────────────────────

def compute_delta_pdp(results, ages, masks, times, val_lo, val_hi,
                      visit_times=None, target_name="covariate"):
    """
    Compute marginal ΔPDP = PDP(val_hi) - PDP(val_lo) at each visit time.
    """
    mu_lo = results[val_lo]
    mu_hi = results[val_hi]
    delta = mu_hi - mu_lo
    delta_v = val_hi - val_lo

    if visit_times is None:
        visit_times = VISIT_TIMES_3C

    closest = _closest_obs_per_subject_2(delta, masks, times, visit_times)

    print(f"\nΔPDP for {target_name} ({val_lo} → {val_hi}, Δv = {delta_v}):")
    print(f"  {'Time':>6s}  {'ΔPDP':>10s}  {'SE':>10s}  {'n':>6s}")
    print(f"  {'-'*40}")

    estimated = {}
    for vt in visit_times:
        d = closest[vt]
        if len(d) > 10:
            est = d.mean()
            se = d.std() / np.sqrt(len(d))
            estimated[vt] = est
            print(f"  {vt:6.0f}  {est:+10.4f}  {se:10.4f}  {len(d):6d}")

    if estimated:
        avg_delta = np.mean(list(estimated.values()))
        print(f"\n  Average ΔPDP = {avg_delta:+.4f}")
        print(f"  Per-unit effect = {avg_delta / delta_v:+.4f}")

    return estimated


def compute_delta_pdp_stratified(results, ages, masks, times,
                                  val_lo, val_hi, visit_times=None,
                                  target_name="covariate"):
    """
    Compute ΔPDP stratified by age tertiles.
    """
    mu_lo = results[val_lo]
    mu_hi = results[val_hi]
    delta = mu_hi - mu_lo
    delta_v = val_hi - val_lo

    if visit_times is None:
        visit_times = VISIT_TIMES_3C

    q33, q67 = np.percentile(ages, [33, 67])
    age_groups = {
        f'Young (AGEc < {q33:.1f})': ages < q33,
        f'Middle ({q33:.1f} ≤ AGEc < {q67:.1f})':
            (ages >= q33) & (ages < q67),
        f'Old (AGEc ≥ {q67:.1f})': ages >= q67,
    }

    print(f"\n{'='*70}")
    print(f"Stratified ΔPDP for {target_name} ({val_lo} → {val_hi})")
    print(f"{'='*70}")

    summary = {}

    for group_name, group_mask in age_groups.items():
        n_group = group_mask.sum()
        mean_age = ages[group_mask].mean()

        delta_group = delta[group_mask]
        masks_group = masks[group_mask]
        times_group = times[group_mask]

        closest_delta = _closest_obs_per_subject_2(
            delta_group, masks_group, times_group, visit_times)

        print(f"\n  {group_name} (n={n_group}, mean AGEc={mean_age:.2f})")
        print(f"  {'Time':>6s}  {'ΔPDP':>8s}  {'SE':>8s}  {'n':>6s}")
        print(f"  {'-'*35}")

        all_deltas = []
        for vt in visit_times:
            d = closest_delta[vt]
            if len(d) > 10:
                mean_d = d.mean()
                se_d = d.std() / np.sqrt(len(d))
                all_deltas.append(mean_d)
                print(f"  {vt:6.0f}  {mean_d:+8.3f}  {se_d:8.3f}  {len(d):6d}")

        summary[group_name] = {
            "mean_age": mean_age,
            "n": int(n_group),
            "estimated": np.mean(all_deltas) if all_deltas else 0,
            "per_unit": np.mean(all_deltas) / delta_v if all_deltas else 0,
        }

    print(f"\n  {'='*70}")
    print(f"  {'Group':<35s} {'mean AGEc':>10s} {'ΔPDP':>10s} {'per-unit':>10s}")
    print(f"  {'-'*70}")
    for group_name, vals in summary.items():
        print(f"  {group_name:<35s} {vals['mean_age']:>+10.2f} "
              f"{vals['estimated']:>+10.3f} {vals['per_unit']:>+10.4f}")

    return summary


# ─────────────────────────────────────────────
# Trajectory-profile PDP
# ─────────────────────────────────────────────

def make_profiles(visit_times=None, v_lo=None, v_hi=None):
    """
    Define counterfactual trajectory profiles for a covariate.

    Matches the R function make_profiles() exactly:
      - takes visit_times (the actual time vector, not just a count)
      - uses ceiling(n / 2) for the switch point

    Args:
        visit_times: array-like of canonical visit times
                     (default: VISIT_TIMES_3C)
        v_lo:        low value (e.g. Q25)
        v_hi:        high value (e.g. Q75)

    Returns:
        dict of {profile_name: (T,) numpy array of covariate values}
    """
    import math
    if visit_times is None:
        visit_times = VISIT_TIMES_3C
    n = len(visit_times)
    half = math.ceil(n / 2)          # R: ceiling(n / 2)
    return {
        "stable_low":      np.full(n, v_lo),
        "stable_high":     np.full(n, v_hi),
        "late_spike":      np.array([v_lo]*half + [v_hi]*(n - half)),
        "late_decline":    np.array([v_hi]*half + [v_lo]*(n - half)),
        "gradual_rise":    np.linspace(v_lo, v_hi, n),
        "gradual_decline": np.linspace(v_hi, v_lo, n),
    }


def build_profile_xaug(x_aug, target_col, profile_values, n_tv,
                        mask_type="binary"):
    """
    Build counterfactual x_aug with a trajectory profile.

    Unlike build_counterfactual_xaug (which sets a constant value),
    this sets the target covariate to profile_values[t] at each slot t.

    Args:
        x_aug:          (N, T, 1+2K) original
        target_col:     int, index in [0, K-1]
        profile_values: (T,) array of covariate values per slot
        n_tv:           K
        mask_type:      "binary" or "cumulative"

    Returns:
        x_aug_cf: (N, T, 1+2K) counterfactual
    """
    K = n_tv
    N, T, _ = x_aug.shape
    x_aug_cf = x_aug.clone()

    cov_col = 1 + target_col
    mask_col = 1 + K + target_col

    # Set covariate to profile values at each slot
    prof = torch.tensor(profile_values, device=x_aug.device,
                        dtype=x_aug.dtype)
    x_aug_cf[:, :, cov_col] = prof.unsqueeze(0).expand(N, -1)

    # Set mask to fully observed
    if mask_type == "binary":
        x_aug_cf[:, :, mask_col] = 1.0
    else:
        x_aug_cf[:, :, mask_col] = torch.arange(
            1, T + 1, device=x_aug.device, dtype=x_aug.dtype
        ).unsqueeze(0).expand(N, -1)

    return x_aug_cf


def compute_trajectory_profile_pdp(model, loader, device, profiles,
                                    target_col=0, n_tv=5,
                                    mask_type="binary",
                                    target_name="covariate"):
    """
    Compute trajectory-profile PDP.

    For each profile, replace the target covariate path with the profile
    values, re-integrate the ODE, and compute the population mean at
    each visit.

    This is the key diagnostic for path-dependence:
      - If early_burden ≈ late_spike → model sees only current value
        (instantaneous effect, same as HLME)
      - If early_burden < late_spike → model learned cumulative burden
        (the ODE accumulated the high-BMI history)

    Args:
        model:       NeuralODEModel
        loader:      DataLoader
        device:      torch device
        profiles:    dict from make_profiles()
        target_col:  column index in x_interp [0..K-1]
        n_tv:        K
        mask_type:   "binary" or "cumulative"
        target_name: name for printing

    Returns:
        results: dict {profile_name: {time: mean_prediction}}
        masks:   (N, T) numpy array
        times:   (N, T) numpy array
    """
    model.eval()

    results = {pname: [] for pname in profiles}
    all_masks = []
    all_times = []

    print(f"  Computing trajectory-profile PDP for {target_name}")
    print(f"    Profiles: {list(profiles.keys())}")

    with torch.no_grad():
        for pname, prof_values in profiles.items():
            batch_mus = []

            for batch in loader:
                pids, x_aug, y_pad, target_mask, static = batch
                x_aug = x_aug.to(device)
                target_mask = target_mask.to(device)
                static = static.to(device)

                x_aug_cf = build_profile_xaug(
                    x_aug, target_col=target_col,
                    profile_values=prof_values, n_tv=n_tv,
                    mask_type=mask_type,
                )

                mu, V, Z, D, sig2, reg_dict = model(
                    x_aug_cf,
                    static_covariates=static,
                    obs_mask=target_mask,
                )
                batch_mus.append(mu.cpu())

                if pname == list(profiles.keys())[0]:
                    all_masks.append(target_mask.cpu())
                    all_times.append(x_aug[:, :, 0].cpu())

            results[pname] = torch.cat(batch_mus, dim=0).numpy()

    masks = torch.cat(all_masks, dim=0).numpy()
    times = torch.cat(all_times, dim=0).numpy()

    # Print summary
    visit_times = VISIT_TIMES_3C
    print(f"\n    {'Profile':<20s}", end="")
    for vt in visit_times:
        print(f"  T={vt:<5.0f}", end="")
    print()
    print(f"    {'-'*80}")

    for pname in profiles:
        mu = results[pname]
        closest = _closest_obs_per_subject_2(mu, masks, times, visit_times)
        print(f"    {pname:<20s}", end="")
        for vt in visit_times:
            d = closest[vt]
            if len(d) > 10:
                print(f"  {d.mean():>7.2f}", end="")
            else:
                print(f"  {'N/A':>7s}", end="")
        print()

    # Diagnostic: early_burden vs late_spike
    if "late_decline" in results and "late_spike" in results:
        eb = results["late_decline"]
        ls = results["late_spike"]
        diff = eb - ls
        closest_diff = _closest_obs_per_subject_2(diff, masks, times, visit_times)
        print(f"\n    Diagnostic: late_decline − late_spike")
        print(f"    {'Time':>8s}  {'Diff':>8s}  {'SE':>8s}  {'Interp':>20s}")
        for vt in visit_times:
            d = closest_diff[vt]
            if len(d) > 10:
                m = d.mean()
                se = d.std() / np.sqrt(len(d))
                interp = ("cumulative ✓" if m < -0.1
                          else "instantaneous" if abs(m) < 0.1
                          else "unexpected (+)")
                print(f"    {vt:8.0f}  {m:+8.3f}  {se:8.3f}  {interp:>20s}")

    return results, masks, times


def plot_trajectory_profile_pdp(results, masks, times,
                                 save_path="traj_profile_pdp.png",
                                 visit_times=None,
                                 target_name="covariate",
                                 ylim=None,
                                 title=None,
                                 figsize=(9, 6)):
    """
    Plot trajectory-profile PDP.

    Visually aligned with R's plot_trajectory_profile_pdp() so that
    the two panels look consistent when placed side-by-side in a Beamer slide.

    Args:
        ylim:    (ymin, ymax) tuple for shared Y-axis scale across models.
                 If None, matplotlib auto-scales.
        title:   Custom title (None → default; '' → no title).
        figsize: Figure size in inches (default 9×6 to match R ggsave).
    """
    if visit_times is None:
        visit_times = VISIT_TIMES_3C

    # ── Shared palette (identical to R) ──────────────────────────────────
    profile_colours = {
        "stable_low":      "#2166AC",   # blue
        "stable_high":     "#B2182B",   # dark red
        "late_spike":      "#FF7F00",   # orange
        "late_decline":    "#6A3D9A",   # purple
        "gradual_rise":    "#33A02C",   # green
        "gradual_decline": "#E31A1C",   # bright red
    }

    profile_linestyles = {
        "stable_low":      "-",
        "stable_high":     "-",
        "late_spike":      "--",
        "late_decline":    "--",
        "gradual_rise":    "-.",
        "gradual_decline": "-.",
    }

    profile_markers = {
        "stable_low":      "o",
        "stable_high":     "s",
        "late_spike":      "^",
        "late_decline":    "v",
        "gradual_rise":    "D",
        "gradual_decline": "d",
    }

    profile_labels = {
        "stable_low":      "Stable low",
        "stable_high":     "Stable high",
        "late_spike":      "Late spike",
        "late_decline":    "Late decline",
        "gradual_rise":    "Gradual rise",
        "gradual_decline": "Gradual decline",
    }

    # ── Figure setup (theme_minimal–like) ────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')
    fig.set_facecolor('white')

    # Subtle grey grid (matches ggplot theme_minimal)
    ax.grid(True, color='#D9D9D9', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Remove top/right spines (theme_minimal style)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#636363')
    ax.spines['bottom'].set_color('#636363')
    ax.tick_params(colors='#636363', labelsize=11)

    # ── Plot each profile ────────────────────────────────────────────────
    for pname, mu in results.items():
        closest = _closest_obs_per_subject_2(mu, masks, times, visit_times)

        t_plot, mean_plot, lo_plot, hi_plot = [], [], [], []
        for vt in visit_times:
            d = closest[vt]
            if len(d) > 10:
                m = d.mean()
                se = d.std() / np.sqrt(len(d))
                t_plot.append(vt)
                mean_plot.append(m)
                lo_plot.append(m - 1.96 * se)
                hi_plot.append(m + 1.96 * se)

        color = profile_colours.get(pname, "grey")
        ls = profile_linestyles.get(pname, "-")
        marker = profile_markers.get(pname, "o")
        label = profile_labels.get(pname, pname)

        ax.fill_between(t_plot, lo_plot, hi_plot, color=color,
                        alpha=0.10, zorder=1)
        ax.plot(t_plot, mean_plot, linestyle=ls, marker=marker, color=color,
                linewidth=1.3, markersize=5, label=label, zorder=2)

    # ── Labels ───────────────────────────────────────────────────────────
    ax.set_xlabel('Time (years)', fontsize=13, color='#252525')
    ax.set_ylabel('E[IST]', fontsize=13, color='#252525')

    if title is None:
        ax.set_title(f'Trajectory-profile PDP of {target_name}',
                     fontsize=13, fontweight='bold', color='#252525')
    elif title != '':
        ax.set_title(title, fontsize=13, fontweight='bold', color='#252525')

    if ylim is not None:
        ax.set_ylim(ylim)

    # Legend at bottom (matches R legend.position = "bottom")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              fontsize=10, ncol=3, frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  \u2192 {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_pdp(results, ages, masks, times, intervention_values,
             save_path="pdp_by_age.png", visit_times=None,
             target_name="covariate"):
    """Plot PDP over time, stratified by age tertiles."""
    q33, q67 = np.percentile(ages, [33, 67])

    age_groups = {
        f'Young (AGEc < {q33:.1f})': ages < q33,
        f'Middle ({q33:.1f} ≤ AGEc < {q67:.1f})':
            (ages >= q33) & (ages < q67),
        f'Old (AGEc ≥ {q67:.1f})': ages >= q67,
    }

    if visit_times is None:
        visit_times = VISIT_TIMES_3C

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(intervention_values)))

    for ax_idx, (group_name, group_mask) in enumerate(age_groups.items()):
        ax = axes[ax_idx]

        for v_idx, v in enumerate(intervention_values):
            mu_group = results[v][group_mask]
            masks_group = masks[group_mask]
            times_group = times[group_mask]

            closest = _closest_obs_per_subject_2(
                mu_group, masks_group, times_group, visit_times)
            mean_pred, visit_t_plot = [], []
            for vt in visit_times:
                if len(closest[vt]) > 10:
                    mean_pred.append(closest[vt].mean())
                    visit_t_plot.append(vt)

            if visit_t_plot:
                ax.plot(visit_t_plot, mean_pred, 'o-', color=colors[v_idx],
                        label=f'{target_name}={v}', linewidth=2, markersize=5)

        ax.set_title(group_name, fontsize=11)
        ax.set_xlabel('Time (years)')
        if ax_idx == 0:
            ax.set_ylabel('Predicted ISA15')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'PDP of {target_name} on ISA15, stratified by age (3C)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {save_path}")
    plt.close()


def plot_pdp_marginal(results, masks, times, intervention_values,
                      save_path="pdp_marginal.png", visit_times=None,
                      ice_results=None, ice_n=50, seed=1,
                      target_name="covariate"):
    """
    Plot marginal PDP + ICE curves.

    If ice_results is provided (from compute_pdp_with_blup), plots
    subject-level predictions WITH random effects.
    """
    if visit_times is None:
        visit_times = VISIT_TIMES_3C

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(intervention_values)))

    N = masks.shape[0]
    rng = np.random.RandomState(seed)
    ice_ids = rng.choice(N, min(ice_n, N), replace=False)

    ice_source = ice_results if ice_results is not None else results

    # ICE curves (grey)
    for v in intervention_values:
        mu = ice_source[v]
        for i in ice_ids:
            ice_times, ice_preds = [], []
            for vt in visit_times:
                obs_idx = np.where(masks[i] > 0.5)[0]
                if len(obs_idx) == 0:
                    continue
                obs_times = times[i, obs_idx]
                closest = obs_idx[np.argmin(np.abs(obs_times - vt))]
                ice_times.append(vt)
                ice_preds.append(mu[i, closest])
            if len(ice_times) > 1:
                ax.plot(ice_times, ice_preds, '-', color='grey',
                        alpha=0.08, linewidth=0.5, zorder=1)

    # PDP curves (colored)
    for v_idx, v in enumerate(intervention_values):
        closest = _closest_obs_per_subject_2(
            results[v], masks, times, visit_times)

        mean_pred, visit_t_plot = [], []
        for vt in visit_times:
            if len(closest[vt]) > 10:
                mean_pred.append(closest[vt].mean())
                visit_t_plot.append(vt)

        ax.plot(visit_t_plot, mean_pred, 'o-', color=colors[v_idx],
                label=f'{target_name}={v}', linewidth=2.5, markersize=6,
                zorder=2)

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Predicted ISA15')
    title = f'Marginal PDP of {target_name} on ISA15 (3C)'
    if ice_results is not None:
        title += ' + ICE with BLUP'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {save_path}")
    plt.close()


def plot_delta_pdp(results, masks, times, val_lo, val_hi,
                   save_path="delta_pdp.png", visit_times=None,
                   target_name="covariate"):
    """Plot ΔPDP over time with 95% pointwise confidence bands."""
    delta = results[val_hi] - results[val_lo]

    if visit_times is None:
        visit_times = VISIT_TIMES_3C

    closest = _closest_obs_per_subject_2(delta, masks, times, visit_times)

    t_plot, mean_plot, lo_plot, hi_plot = [], [], [], []
    for vt in visit_times:
        d = closest[vt]
        if len(d) > 10:
            m = d.mean()
            se = d.std() / np.sqrt(len(d))
            t_plot.append(vt)
            mean_plot.append(m)
            lo_plot.append(m - 1.96 * se)
            hi_plot.append(m + 1.96 * se)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_plot, mean_plot, 'o-', color='firebrick',
            linewidth=2, markersize=6)
    ax.fill_between(t_plot, lo_plot, hi_plot, color='firebrick', alpha=0.15)
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(f'ΔPDP ({target_name} {val_lo} → {val_hi})')
    ax.set_title(f'ΔPDP of {target_name} on ISA15 (3C)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {save_path}")
    plt.close()