"""
PDP analysis for the Neural ODE-LMM on the real 3C dataset.

BMI enters the model in two places:
  1. ODE dynamics:  dz/dt = f(z, t, [x_filled, cumask], static)
  2. Decoder skip:  mu = rho(z(t), BMI_std(t), AGEc) @ beta

A full PDP intervention (Profile A — total effect) must replace BMI
consistently in BOTH pathways and re-integrate the ODE from scratch.

For the counterfactual ode_inject:
  - BMI channel in x_filled → set to constant v
  - BMI channel in cumask   → set to fully observed (incrementing 1,2,3,...)
  - All other channels      → left at their observed values

Includes:
  - compute_pdp:               population-level PDP (mu only)
  - compute_pdp_with_blup:     subject-level ICE with BLUP random effects
  - compute_delta_pdp:         marginal ΔPDP at visit times
  - compute_delta_pdp_stratified: ΔPDP by age tertiles
  - plot_pdp, plot_pdp_marginal: visualization
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Canonical visit grid ─────────────────────────────────────────────────────
VISIT_TIMES_3C = np.array([0, 2, 4, 7, 10, 12])


# ─────────────────────────────────────────────
# Counterfactual builder
# ─────────────────────────────────────────────

def build_counterfactual(x_filled, cumask, t_pad, mask, target_value,
                         target_col=0, mode="constant", slope=None):
    """
    Build counterfactual x_filled, cumask, and ode_inject for a covariate intervention.

    Args:
        x_filled:     (N, T, K) forward-filled covariates
        cumask:       (N, T, K) cumulative observation mask
        t_pad:        (N, T)    padded times
        mask:         (N, T)    outcome observation mask
        target_value: scalar    counterfactual value for the target covariate
        target_col:   int       column index in x_filled of the target covariate
        mode:         str       "constant", "linear", or "shifted"
        slope:        float     slope for linear mode

    Returns:
        x_cf:          (N, T, K)  counterfactual x_filled
        ode_inject_cf: (N, T, 2K) counterfactual [x_filled, cumask] for ODE
    """
    N, T, K = x_filled.shape

    x_cf = x_filled.clone()

    if mode == "constant":
        x_cf[:, :, target_col] = target_value

    elif mode == "linear":
        if slope is None:
            raise ValueError("slope required for linear mode")
        x_cf[:, :, target_col] = target_value + slope * t_pad

    elif mode == "shifted":
        real_vals = x_filled[:, :, target_col]
        masked_vals = real_vals * mask
        n_obs = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_subj = masked_vals.sum(dim=1, keepdim=True) / n_obs
        x_cf[:, :, target_col] = real_vals - mean_subj + target_value

    else:
        raise ValueError(f"Unknown mode: '{mode}'")

    # Build counterfactual cumask: target channel is "fully observed"
    cumask_cf = cumask.clone()
    cumask_cf[:, :, target_col] = torch.arange(
        1, T + 1, device=cumask.device, dtype=cumask.dtype
    ).unsqueeze(0).expand(N, -1)

    ode_inject_cf = torch.cat([x_cf, cumask_cf], dim=-1)

    return x_cf, ode_inject_cf


# ─────────────────────────────────────────────
# Core PDP computation (population mean only)
# ─────────────────────────────────────────────

def compute_pdp(model, loader, device, intervention_values, target_col=0,
                mode="constant", slope=None, age_col=1, target_name="covariate"):
    """
    Compute PDP for a target covariate on the real 3C dataset.

    For each intervention value v:
      1. Replace target covariate in x_filled AND ode_inject with counterfactual
      2. Set target's cumask to fully observed
      3. Re-run full forward pass (encoder + ODE + decoder)
      4. Collect population mean mu(t)

    Args:
        model:               NeuralODEModel
        loader:              DataLoader (from RealDataset + collate_real)
        device:              torch device
        intervention_values: list of counterfactual values for the target covariate
        target_col:          column index in x_filled of the target covariate
        mode:                "constant", "linear", or "shifted"
        slope:               slope for linear mode
        age_col:             AGEc column index in static covariates
        target_name:         name of target covariate (for printing)

    Returns:
        results: dict {value: (N, T) tensor of population mean predictions}
        ages:    (N,) tensor of AGEc values
        masks:   (N, T) observation mask
        times:   (N, T) padded times
    """
    model.eval()

    results = {v: [] for v in intervention_values}
    all_ages = []
    all_masks = []
    all_times = []

    print(f"  Computing PDP for {target_name} (col={target_col}, mode='{mode}')")

    # Estimate slope if needed
    if mode == "linear" and slope is None:
        all_vals, all_t = [], []
        with torch.no_grad():
            for batch in loader:
                _, t_pad_b, x_filled_b, _, _, mask_b, _ = batch
                obs = mask_b > 0.5
                all_vals.append(x_filled_b[:, :, target_col][obs].numpy())
                all_t.append(t_pad_b[obs].numpy())
        val_flat = np.concatenate(all_vals)
        t_flat = np.concatenate(all_t)
        from numpy.polynomial.polynomial import polyfit
        c = polyfit(t_flat, val_flat, 1)
        slope = c[1]
        print(f"  Estimated population slope for {target_name}: {slope:.4f} per year")

    with torch.no_grad():
        for v in intervention_values:
            batch_mus = []

            for batch in loader:
                _, t_pad, x_filled, cumask, y_pad, mask, s = batch
                t_pad = t_pad.to(device)
                x_filled = x_filled.to(device)
                cumask = cumask.to(device)
                mask = mask.to(device)
                s = s.to(device)

                x_cf, ode_inject_cf = build_counterfactual(
                    x_filled, cumask, t_pad, mask,
                    target_value=v, target_col=target_col,
                    mode=mode, slope=slope,
                )

                mu, V, _, _, _, _ = model(
                    t_pad, x_cf, masks=cumask,
                    static_covariates=s,
                    ode_inject=ode_inject_cf,
                    obs_mask=mask,
                    y_pad=None,
                )
                batch_mus.append(mu.cpu())

                if v == intervention_values[0]:
                    all_ages.append(s[:, age_col].cpu())
                    all_masks.append(mask.cpu())
                    all_times.append(t_pad.cpu())

            results[v] = torch.cat(batch_mus, dim=0)

    ages = torch.cat(all_ages, dim=0)
    masks = torch.cat(all_masks, dim=0)
    times = torch.cat(all_times, dim=0)

    return results, ages, masks, times


# ─────────────────────────────────────────────
# ICE with BLUP random effects
# ─────────────────────────────────────────────

def compute_pdp_with_blup(model, loader, device, intervention_values, target_col=0,
                          mode="constant", slope=None, age_col=1, target_name="covariate"):
    """
    Compute subject-level ICE curves INCLUDING random effects via BLUP.

    Step 1: compute BLUP from OBSERVED data
    Step 2: for each counterfactual value, re-run forward pass and add Z @ b_hat
    """
    model.eval()

    results_pop = {v: [] for v in intervention_values}
    results_subj = {v: [] for v in intervention_values}
    all_blup = []
    all_ages = []
    all_masks = []
    all_times = []

    print(f"  Computing ICE with BLUP for {target_name} (col={target_col}, mode='{mode}')")

    # Estimate slope if needed
    if mode == "linear" and slope is None:
        all_vals, all_t = [], []
        with torch.no_grad():
            for batch in loader:
                _, t_pad_b, x_filled_b, _, _, mask_b, _ = batch
                obs = mask_b > 0.5
                all_vals.append(x_filled_b[:, :, target_col][obs].numpy())
                all_t.append(t_pad_b[obs].numpy())
        val_flat = np.concatenate(all_vals)
        t_flat = np.concatenate(all_t)
        from numpy.polynomial.polynomial import polyfit
        c = polyfit(t_flat, val_flat, 1)
        slope = c[1]

    with torch.no_grad():
        # --- Step 1: compute BLUP from observed data ---
        print("    Step 1: computing BLUP from observed data...")
        for batch in loader:
            _, t_pad, x_filled, cumask, y_pad, mask, s = batch
            t_pad = t_pad.to(device)
            x_filled = x_filled.to(device)
            cumask = cumask.to(device)
            y_pad = y_pad.to(device)
            mask = mask.to(device)
            s = s.to(device)

            # Forward on OBSERVED data (real BMI)
            ode_inject_obs = torch.cat([x_filled, cumask], dim=-1)
            mu_obs, V_obs, Z_obs, D_obs, sig2_obs, _ = model(
                t_pad, x_filled, masks=cumask,
                static_covariates=s, ode_inject=ode_inject_obs,
                obs_mask=mask, y_pad=None,
            )

            # BLUP: b_hat_i = D Z_i^T V_i^{-1} (y_i - mu_i)
            residual = (y_pad - mu_obs) * mask                      # (N, T)
            Z_masked = Z_obs * mask.unsqueeze(-1)                   # (N, T, q)

            # Masked V
            mask_outer = mask.unsqueeze(-1) * mask.unsqueeze(-2)    # (N, T, T)
            jitter = 1e-4
            V_masked = V_obs * mask_outer + jitter * torch.eye(
                t_pad.shape[1], device=device).unsqueeze(0)

            # Solve V^{-1} r via Cholesky
            L_V = torch.linalg.cholesky(V_masked)
            Vinv_r = torch.cholesky_solve(
                residual.unsqueeze(-1), L_V
            ).squeeze(-1)                                            # (N, T)

            # b_hat = D @ Z^T @ V^{-1} @ r
            b_hat = D_obs @ (Z_masked.transpose(1, 2) @ Vinv_r.unsqueeze(-1))
            b_hat = b_hat.squeeze(-1)                                # (N, q)
            all_blup.append(b_hat.cpu())

            all_ages.append(s[:, age_col].cpu())
            all_masks.append(mask.cpu())
            all_times.append(t_pad.cpu())

        blup = torch.cat(all_blup, dim=0)

        # --- Step 2: compute predictions under each intervention value ---
        print("    Step 2: computing counterfactual predictions...")
        for v in intervention_values:
            batch_mus = []
            batch_subj = []
            blup_offset = 0

            for batch in loader:
                _, t_pad, x_filled, cumask, y_pad, mask, s = batch
                t_pad = t_pad.to(device)
                x_filled = x_filled.to(device)
                cumask = cumask.to(device)
                mask = mask.to(device)
                s = s.to(device)

                N_batch = t_pad.shape[0]

                x_cf, ode_inject_cf = build_counterfactual(
                    x_filled, cumask, t_pad, mask,
                    target_value=v, target_col=target_col,
                    mode=mode, slope=slope,
                )

                # Forward under counterfactual
                mu_cf, V_cf, Z_cf, D_cf, sig2_cf, _ = model(
                    t_pad, x_cf, masks=cumask,
                    static_covariates=s, ode_inject=ode_inject_cf,
                    obs_mask=mask, y_pad=None,
                )

                # Subject-level: mu + Z @ b_hat
                b_batch = blup[blup_offset:blup_offset + N_batch].to(device)
                Zb = (Z_cf * b_batch.unsqueeze(1)).sum(dim=-1)       # (N, T)
                y_subj = mu_cf + Zb

                batch_mus.append(mu_cf.cpu())
                batch_subj.append(y_subj.cpu())
                blup_offset += N_batch

            results_pop[v] = torch.cat(batch_mus, dim=0)
            results_subj[v] = torch.cat(batch_subj, dim=0)

    ages = torch.cat(all_ages, dim=0)
    masks = torch.cat(all_masks, dim=0)
    times = torch.cat(all_times, dim=0)

    return results_pop, results_subj, blup, ages, masks, times


# ─────────────────────────────────────────────
# Helper: closest observation per subject
# ─────────────────────────────────────────────

def _closest_obs_per_subject(mu, masks_np, times_np, visit_times):
    """
    For each target visit time, find the closest observed time point per subject.
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


# ─────────────────────────────────────────────
# ΔPDP computation (no oracle — real data)
# ─────────────────────────────────────────────

def compute_delta_pdp(results, ages, masks, times, val_lo, val_hi,
                      visit_times=None, target_name="covariate"):
    """
    Compute marginal ΔPDP = PDP(val_hi) - PDP(val_lo) at each visit time.

    Returns:
        estimated: dict {visit_time: estimated_delta}
    """
    mu_lo = results[val_lo].numpy()
    mu_hi = results[val_hi].numpy()
    masks_np = masks.numpy()
    times_np = times.numpy()

    delta = mu_hi - mu_lo
    delta_v = val_hi - val_lo

    if visit_times is None:
        visit_times = VISIT_TIMES_3C

    closest = _closest_obs_per_subject(delta, masks_np, times_np, visit_times)

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

    # Per-unit effect (ΔPDP / Δv)
    if estimated:
        avg_delta = np.mean(list(estimated.values()))
        print(f"\n  Average ΔPDP = {avg_delta:+.4f}")
        print(f"  Per-unit BMI effect = {avg_delta / delta_v:+.4f}")

    return estimated


def compute_delta_pdp_stratified(results, ages, masks, times,
                                  val_lo, val_hi, visit_times=None,
                                  target_name="covariate"):
    """
    Compute ΔPDP stratified by age tertiles.
    """
    mu_lo = results[val_lo].numpy()
    mu_hi = results[val_hi].numpy()
    masks_np = masks.numpy()
    ages_np = ages.numpy()
    times_np = times.numpy()

    delta = mu_hi - mu_lo
    delta_v = val_hi - val_lo

    if visit_times is None:
        visit_times = VISIT_TIMES_3C

    q33, q67 = np.percentile(ages_np, [33, 67])
    age_groups = {
        f'Young (AGEc < {q33:.1f})': ages_np < q33,
        f'Middle ({q33:.1f} ≤ AGEc < {q67:.1f})': (ages_np >= q33) & (ages_np < q67),
        f'Old (AGEc ≥ {q67:.1f})': ages_np >= q67,
    }

    print(f"\n{'='*70}")
    print(f"Stratified ΔPDP for {target_name} ({val_lo} → {val_hi})")
    print(f"{'='*70}")

    summary = {}

    for group_name, group_mask in age_groups.items():
        n_group = group_mask.sum()
        mean_age = ages_np[group_mask].mean()

        delta_group = delta[group_mask]
        masks_group = masks_np[group_mask]
        times_group = times_np[group_mask]

        closest_delta = _closest_obs_per_subject(delta_group, masks_group,
                                                  times_group, visit_times)

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
# Plotting
# ─────────────────────────────────────────────

def plot_pdp(results, ages, masks, times, intervention_values,
             save_path="pdp_by_age.png", visit_times=None,
             target_name="covariate"):
    """Plot PDP over time, stratified by age tertiles."""
    age_np = ages.numpy()
    q33, q67 = np.percentile(age_np, [33, 67])

    age_groups = {
        f'Young (AGEc < {q33:.1f})': age_np < q33,
        f'Middle ({q33:.1f} ≤ AGEc < {q67:.1f})': (age_np >= q33) & (age_np < q67),
        f'Old (AGEc ≥ {q67:.1f})': age_np >= q67,
    }

    masks_np = masks.numpy()
    times_np = times.numpy()

    if visit_times is None:
        visit_times = VISIT_TIMES_3C

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(intervention_values)))

    for ax_idx, (group_name, group_mask) in enumerate(age_groups.items()):
        ax = axes[ax_idx]

        for v_idx, v in enumerate(intervention_values):
            mu = results[v].numpy()
            mu_group = mu[group_mask]
            masks_group = masks_np[group_mask]
            times_group = times_np[group_mask]

            closest = _closest_obs_per_subject(mu_group, masks_group,
                                                times_group, visit_times)
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

    plt.suptitle(f'PDP of {target_name} on ISA15, stratified by baseline age (3C cohort)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"PDP saved to {save_path}")
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
    masks_np = masks.numpy()
    times_np = times.numpy()

    if visit_times is None:
        visit_times = VISIT_TIMES_3C

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(intervention_values)))

    N = masks_np.shape[0]
    rng = np.random.RandomState(seed)
    ice_ids = rng.choice(N, min(ice_n, N), replace=False)

    # Use subject-level predictions if available, else population mean
    ice_source = ice_results if ice_results is not None else results

    # Plot ICE curves (grey)
    for v in intervention_values:
        mu = ice_source[v].numpy()
        for i in ice_ids:
            ice_times, ice_preds = [], []
            for vt in visit_times:
                obs_idx = np.where(masks_np[i] > 0.5)[0]
                if len(obs_idx) == 0:
                    continue
                obs_times = times_np[i, obs_idx]
                closest = obs_idx[np.argmin(np.abs(obs_times - vt))]
                ice_times.append(vt)
                ice_preds.append(mu[i, closest])
            if len(ice_times) > 1:
                ax.plot(ice_times, ice_preds, '-', color='grey',
                        alpha=0.08, linewidth=0.5, zorder=1)

    # Plot PDP curves (colored, on top)
    for v_idx, v in enumerate(intervention_values):
        mu = results[v].numpy()
        closest = _closest_obs_per_subject(mu, masks_np, times_np, visit_times)

        mean_pred, visit_t_plot = [], []
        for vt in visit_times:
            if len(closest[vt]) > 10:
                mean_pred.append(closest[vt].mean())
                visit_t_plot.append(vt)

        ax.plot(visit_t_plot, mean_pred, 'o-', color=colors[v_idx],
                label=f'{target_name}={v}', linewidth=2.5, markersize=6, zorder=2)

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Predicted ISA15')
    title = f'Marginal PDP of {target_name} on ISA15 (3C cohort)'
    if ice_results is not None:
        title += ' + ICE with BLUP (grey)'
    else:
        title += ' + ICE (grey)'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Marginal PDP saved to {save_path}")
    plt.close()


def plot_delta_pdp(results, masks, times, val_lo, val_hi,
                   save_path="delta_pdp.png", visit_times=None,
                   target_name="covariate"):
    """
    Plot ΔPDP over time with 95% pointwise confidence bands.
    """
    mu_lo = results[val_lo].numpy()
    mu_hi = results[val_hi].numpy()
    masks_np = masks.numpy()
    times_np = times.numpy()

    delta = mu_hi - mu_lo

    if visit_times is None:
        visit_times = VISIT_TIMES_3C

    closest = _closest_obs_per_subject(delta, masks_np, times_np, visit_times)

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
    ax.plot(t_plot, mean_plot, 'o-', color='firebrick', linewidth=2, markersize=6)
    ax.fill_between(t_plot, lo_plot, hi_plot, color='firebrick', alpha=0.15)
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(f'ΔPDP ({target_name} {val_lo} → {val_hi})')
    ax.set_title(f'ΔPDP of {target_name} on ISA15 (3C cohort)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"ΔPDP plot saved to {save_path}")
    plt.close()