"""
PDP analysis for the Neural ODE + BMI Skip model.

Key simplifications vs CDE version:
  - No torchcde dependency
  - No cumulative mask manipulation
  - No cubic/causal interpolation concerns
  - BMI enters ONLY through x_pad → decoder skip — guaranteed separation

Includes:
  - compute_pdp:               population-level PDP (mu only, no random effects)
  - compute_pdp_with_blup:     subject-level ICE including BLUP random effects
  - compute_delta_pdp:         marginal ΔPDP at visit times
  - compute_delta_pdp_stratified: ΔPDP by age tertiles
  - compute_true_delta_pdp:    oracle ΔPDP from DGP parameters
  - plot_pdp, plot_pdp_marginal: visualization
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Core PDP computation (population mean only)
# ─────────────────────────────────────────────

def compute_pdp(model, loader, device, bmi_values, n_tv=1,
                bmi_mode="constant", bmi_slope=None, bmi_col=0,
                interp=None):
    """
    Compute PDP for BMI interventions using the Neural ODE model.

    For each BMI value v, replace BMI(t) in x_pad with the counterfactual,
    run the full forward pass, and collect the population mean mu(t).

    The ODE dynamics are INDEPENDENT of BMI (it's not in the control path),
    so z(t) is the same for all v — only the decoder output changes.

    Args:
        model:      NeuralODEModel
        loader:     DataLoader
        device:     torch device
        bmi_values: list of BMI intervention values
        n_tv:       ignored (kept for API compatibility)
        bmi_mode:   "constant", "linear", or "shifted"
        bmi_slope:  slope for linear mode (auto-estimated if None)
        bmi_col:    column index of BMI in x_pad (default 0)
        interp:     ignored (kept for API compatibility)

    Returns:
        results: dict {bmi_value: (N, max_T) tensor of population mean predictions}
        ages:    (N,) tensor of AGEc values
        masks:   (N, max_T) observation mask
        times:   (N, max_T) real times
    """
    model.eval()

    results = {v: [] for v in bmi_values}
    all_ages = []
    all_masks = []
    all_times = []

    print(f"  Computing PDP (ODE model, bmi_mode='{bmi_mode}')")

    # Estimate BMI slope if needed
    if bmi_mode == "linear" and bmi_slope is None:
        all_bmi, all_t = [], []
        with torch.no_grad():
            for batch in loader:
                _, t_pad_b, x_pad_b, _, _, mask_b, _ = batch
                obs = mask_b > 0.5
                all_bmi.append(x_pad_b[:, :, bmi_col][obs].numpy())
                all_t.append(t_pad_b[obs].numpy())
        bmi_flat = np.concatenate(all_bmi)
        t_flat = np.concatenate(all_t)
        from numpy.polynomial.polynomial import polyfit
        c = polyfit(t_flat, bmi_flat, 1)
        bmi_slope = c[1]
        print(f"  Estimated population BMI slope: {bmi_slope:.4f} per year")

    with torch.no_grad():
        for bmi_v in bmi_values:
            batch_mus = []

            for batch in loader:
                _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
                t_pad = t_pad.to(device)
                x_pad = x_pad.to(device)
                mask  = mask.to(device)
                s     = s.to(device)

                N, T = t_pad.shape

                # --- Counterfactual BMI intervention ---
                x_cf = x_pad.clone()

                if bmi_mode == "constant":
                    x_cf[:, :, bmi_col] = bmi_v

                elif bmi_mode == "linear":
                    x_cf[:, :, bmi_col] = bmi_v + bmi_slope * t_pad

                elif bmi_mode == "shifted":
                    bmi_real = x_pad[:, :, bmi_col]
                    bmi_masked = bmi_real * mask
                    n_obs = mask.sum(dim=1, keepdim=True).clamp(min=1)
                    bmi_mean_subj = bmi_masked.sum(dim=1, keepdim=True) / n_obs
                    x_cf[:, :, bmi_col] = bmi_real - bmi_mean_subj + bmi_v

                else:
                    raise ValueError(f"Unknown bmi_mode: '{bmi_mode}'")

                # Forward pass — masks=None (ODE ignores c_mask)
                mu, V, _, _, _ = model(t_pad, x_cf, masks=None,
                                       static_covariates=s, bmi_t=x_cf[:, :, bmi_col:bmi_col+1], obs_mask=mask,
                                       y_pad=None)
                batch_mus.append(mu.cpu())

                # Collect metadata on first BMI value only
                if bmi_v == bmi_values[0]:
                    all_ages.append(s[:, 1].cpu())   # AGEc is col 1
                    all_masks.append(mask.cpu())
                    all_times.append(t_pad.cpu())

            # Pad to common max T
            max_T = max(m.shape[1] for m in batch_mus)
            padded = []
            for m in batch_mus:
                if m.shape[1] < max_T:
                    pad = torch.zeros(m.shape[0], max_T - m.shape[1])
                    m = torch.cat([m, pad], dim=1)
                padded.append(m)
            results[bmi_v] = torch.cat(padded, dim=0)

    ages = torch.cat(all_ages, dim=0)

    # Pad masks and times
    max_T = max(m.shape[1] for m in all_masks)
    padded_masks, padded_times = [], []
    for m, t in zip(all_masks, all_times):
        if m.shape[1] < max_T:
            m = torch.cat([m, torch.zeros(m.shape[0], max_T - m.shape[1])], dim=1)
            t = torch.cat([t, torch.zeros(t.shape[0], max_T - t.shape[1])], dim=1)
        padded_masks.append(m)
        padded_times.append(t)

    masks = torch.cat(padded_masks, dim=0)
    times = torch.cat(padded_times, dim=0)

    return results, ages, masks, times


# ─────────────────────────────────────────────
# ICE with BLUP random effects
# ─────────────────────────────────────────────

def compute_pdp_with_blup(model, loader, device, bmi_values, bmi_col=0,
                          bmi_mode="constant", bmi_slope=None):
    """
    Compute subject-level ICE curves INCLUDING random effects via BLUP.

    For each subject i and BMI intervention v:
      Y_subj(t; v) = mu(t; v) + Z(t) @ b_hat_i

    where b_hat_i is the BLUP computed from the OBSERVED data (real BMI),
    and mu(t; v) is the population mean under counterfactual BMI = v.

    The BLUP captures subject-specific deviations (fast/slow decliners)
    and is computed ONCE from observed data — it does NOT change with v.

    Returns:
        results_pop:  dict {v: (N, T) population mean}
        results_subj: dict {v: (N, T) subject-level predictions with BLUP}
        blup:         (N, q) estimated random effects
        ages, masks, times: metadata tensors
    """
    model.eval()

    results_pop = {v: [] for v in bmi_values}
    results_subj = {v: [] for v in bmi_values}
    all_blup = []
    all_ages = []
    all_masks = []
    all_times = []

    print(f"  Computing ICE with BLUP (ODE model, bmi_mode='{bmi_mode}')")

    # Estimate slope if needed
    if bmi_mode == "linear" and bmi_slope is None:
        all_bmi, all_t = [], []
        with torch.no_grad():
            for batch in loader:
                _, t_pad_b, x_pad_b, _, _, mask_b, _ = batch
                obs = mask_b > 0.5
                all_bmi.append(x_pad_b[:, :, bmi_col][obs].numpy())
                all_t.append(t_pad_b[obs].numpy())
        bmi_flat = np.concatenate(all_bmi)
        t_flat = np.concatenate(all_t)
        from numpy.polynomial.polynomial import polyfit
        c = polyfit(t_flat, bmi_flat, 1)
        bmi_slope = c[1]

    with torch.no_grad():
        # --- Step 1: compute BLUP from observed data ---
        print("    Step 1: computing BLUP from observed data...")
        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad = t_pad.to(device)
            x_pad = x_pad.to(device)
            y_pad = y_pad.to(device)
            mask  = mask.to(device)
            s     = s.to(device)

            # Forward on OBSERVED data (real BMI)
            mu_obs, V_obs, Z_obs, D_obs, sig2_obs = model(
                t_pad, x_pad, masks=None,
                static_covariates=s, obs_mask=mask, y_pad=None
            )

            # BLUP: b_hat_i = D Z_i^T V_i^{-1} (y_i - mu_i)
            # Apply masking
            residual = (y_pad - mu_obs) * mask                    # (N, T)
            Z_masked = Z_obs * mask.unsqueeze(-1)                 # (N, T, q)

            # Masked V
            mask_outer = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (N, T, T)
            jitter = 1e-4
            V_masked = V_obs * mask_outer + jitter * torch.eye(
                t_pad.shape[1], device=device).unsqueeze(0)

            # Solve V^{-1} r via Cholesky
            L_V = torch.linalg.cholesky(V_masked)                # (N, T, T)
            Vinv_r = torch.cholesky_solve(
                residual.unsqueeze(-1), L_V                       # (N, T, 1)
            ).squeeze(-1)                                          # (N, T)

            # b_hat = D @ Z^T @ V^{-1} @ r
            b_hat = D_obs @ (Z_masked.transpose(1, 2) @ Vinv_r.unsqueeze(-1))  # (N, q, 1)
            b_hat = b_hat.squeeze(-1)                              # (N, q)
            all_blup.append(b_hat.cpu())

            all_ages.append(s[:, 1].cpu())
            all_masks.append(mask.cpu())
            all_times.append(t_pad.cpu())

        blup = torch.cat(all_blup, dim=0)                         # (N_total, q)

        # --- Step 2: compute predictions under each BMI intervention ---
        print("    Step 2: computing counterfactual predictions...")
        for bmi_v in bmi_values:
            batch_mus = []
            batch_subj = []
            blup_offset = 0

            for batch in loader:
                _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
                t_pad = t_pad.to(device)
                x_pad = x_pad.to(device)
                mask  = mask.to(device)
                s     = s.to(device)

                N_batch = t_pad.shape[0]

                # Counterfactual BMI
                x_cf = x_pad.clone()
                if bmi_mode == "constant":
                    x_cf[:, :, bmi_col] = bmi_v
                elif bmi_mode == "linear":
                    x_cf[:, :, bmi_col] = bmi_v + bmi_slope * t_pad
                elif bmi_mode == "shifted":
                    bmi_real = x_pad[:, :, bmi_col]
                    bmi_masked = bmi_real * mask
                    n_obs = mask.sum(dim=1, keepdim=True).clamp(min=1)
                    bmi_mean_subj = bmi_masked.sum(dim=1, keepdim=True) / n_obs
                    x_cf[:, :, bmi_col] = bmi_real - bmi_mean_subj + bmi_v

                # Forward under counterfactual
                mu_cf, V_cf, Z_cf, D_cf, sig2_cf = model(
                    t_pad, x_cf, masks=None,
                    static_covariates=s, obs_mask=mask, y_pad=None
                )

                # Subject-level: mu + Z @ b_hat
                b_batch = blup[blup_offset:blup_offset + N_batch].to(device)  # (N, q)
                Zb = (Z_cf * b_batch.unsqueeze(1)).sum(dim=-1)     # (N, T)
                y_subj = mu_cf + Zb                                 # (N, T)

                batch_mus.append(mu_cf.cpu())
                batch_subj.append(y_subj.cpu())
                blup_offset += N_batch

            # Pad and concatenate
            max_T = max(m.shape[1] for m in batch_mus)
            results_pop[bmi_v] = _pad_and_cat(batch_mus, max_T)
            results_subj[bmi_v] = _pad_and_cat(batch_subj, max_T)

    ages = torch.cat(all_ages, dim=0)
    max_T = max(m.shape[1] for m in all_masks)
    masks = _pad_and_cat(all_masks, max_T)
    times = _pad_and_cat(all_times, max_T)

    return results_pop, results_subj, blup, ages, masks, times


def _pad_and_cat(tensors, max_T):
    """Pad list of (N_i, T_i) tensors to (sum N_i, max_T) and concatenate."""
    padded = []
    for m in tensors:
        if m.shape[1] < max_T:
            pad = torch.zeros(m.shape[0], max_T - m.shape[1])
            m = torch.cat([m, pad], dim=1)
        padded.append(m)
    return torch.cat(padded, dim=0)


# ─────────────────────────────────────────────
# Helper: closest observation per subject
# ─────────────────────────────────────────────

def _closest_obs_per_subject(mu, masks_np, times_np, visit_times):
    """
    For each target visit time, find the closest observed time point per subject.
    Ensures ALL subjects contribute at every visit time (no survivor selection).
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
# ΔPDP computation
# ─────────────────────────────────────────────

def compute_delta_pdp(results, ages, masks, times, bmi_lo=20, bmi_hi=35,
                      true_beta_bmi=-0.30, true_beta_int=-0.05,
                      visit_times=None):
    """
    Compute marginal ΔPDP = PDP(bmi_hi) - PDP(bmi_lo) at each visit time.
    Returns dict of {visit_time: estimated_delta} and {visit_time: true_delta}.
    """
    mu_lo = results[bmi_lo].numpy()
    mu_hi = results[bmi_hi].numpy()
    masks_np = masks.numpy()
    times_np = times.numpy()
    ages_np = ages.numpy()

    delta = mu_hi - mu_lo
    delta_v = bmi_hi - bmi_lo

    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])

    # True marginal ΔPDP uses mean AGEc
    mean_age = ages_np.mean()
    true_marginal = delta_v * (true_beta_bmi + true_beta_int * mean_age)

    closest = _closest_obs_per_subject(delta, masks_np, times_np, visit_times)

    print(f"\nΔPDP (BMI {bmi_lo} → {bmi_hi}):")
    print(f"  True marginal = Δv × (β_BMI + β_int × mean_AGEc)")
    print(f"               = {delta_v} × ({true_beta_bmi} + {true_beta_int} × {mean_age:.3f})")
    print(f"               = {true_marginal:.4f}")

    print(f"\n  {'Time':>6s}  {'Estimated':>10s}  {'True':>10s}  {'Bias':>10s}  {'n':>6s}")
    print(f"  {'-'*50}")

    estimated = {}
    true_ref = {}
    for vt in visit_times:
        d = closest[vt]
        if len(d) > 10:
            est = d.mean()
            estimated[vt] = est
            true_ref[vt] = true_marginal
            print(f"  {vt:6.0f}  {est:+10.4f}  {true_marginal:+10.4f}  "
                  f"{est - true_marginal:+10.4f}  {len(d):6d}")

    return estimated, true_ref


def compute_delta_pdp_stratified(results, ages, masks, times,
                                  bmi_lo=20, bmi_hi=35,
                                  true_beta_bmi=-0.30, true_beta_int=-0.05,
                                  visit_times=None):
    """
    Compute ΔPDP stratified by age tertiles.
    """
    mu_lo = results[bmi_lo].numpy()
    mu_hi = results[bmi_hi].numpy()
    masks_np = masks.numpy()
    ages_np = ages.numpy()
    times_np = times.numpy()

    delta = mu_hi - mu_lo
    delta_v = bmi_hi - bmi_lo

    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])

    q33, q67 = np.percentile(ages_np, [33, 67])
    age_groups = {
        f'Young (AGEc < {q33:.1f})': ages_np < q33,
        f'Middle ({q33:.1f} ≤ AGEc < {q67:.1f})': (ages_np >= q33) & (ages_np < q67),
        f'Old (AGEc ≥ {q67:.1f})': ages_np >= q67,
    }

    print(f"\n{'='*70}")
    print(f"Stratified ΔPDP (BMI {bmi_lo} → {bmi_hi})")
    print(f"  True β_BMI = {true_beta_bmi}, β_BMI×AGEc = {true_beta_int}")
    print(f"{'='*70}")

    summary = {}

    for group_name, group_mask in age_groups.items():
        n_group = group_mask.sum()
        mean_age = ages_np[group_mask].mean()

        true_slope = true_beta_bmi + true_beta_int * mean_age
        true_delta = true_slope * delta_v

        delta_group = delta[group_mask]
        masks_group = masks_np[group_mask]
        times_group = times_np[group_mask]

        closest_delta = _closest_obs_per_subject(delta_group, masks_group,
                                                  times_group, visit_times)

        print(f"\n  {group_name} (n={n_group}, mean AGEc={mean_age:.2f})")
        print(f"  True ΔPDP = ({true_slope:.3f}) × {delta_v} = {true_delta:.3f}")
        print(f"  {'Time':>6s}  {'ΔPDP':>8s}  {'n':>6s}")
        print(f"  {'-'*30}")

        all_deltas = []
        for vt in visit_times:
            d = closest_delta[vt]
            if len(d) > 10:
                mean_d = d.mean()
                all_deltas.append(mean_d)
                print(f"  {vt:6.0f}  {mean_d:+8.3f}  {len(d):6d}")

        summary[group_name] = {
            "estimated": np.mean(all_deltas) if all_deltas else 0,
            "true": true_delta,
        }

    print(f"\n  {'='*60}")
    print(f"  {'Group':<35s} {'mean AGEc':>10s} {'Estimated':>10s} {'True':>10s}")
    print(f"  {'-'*65}")
    for group_name, vals in summary.items():
        mean_age = ages_np[age_groups[group_name]].mean()
        print(f"  {group_name:<35s} {mean_age:>+10.3f} "
              f"{vals['estimated']:>+10.3f} {vals['true']:>+10.3f}")

    # Weighted average check
    total_n = sum(age_groups[g].sum() for g in age_groups)
    weighted_est = sum(
        summary[g]['estimated'] * age_groups[g].sum() / total_n
        for g in age_groups
    )
    weighted_true = sum(
        summary[g]['true'] * age_groups[g].sum() / total_n
        for g in age_groups
    )
    print(f"\n  Weighted avg:  est={weighted_est:+.3f}  true={weighted_true:+.3f}")

    return summary


def compute_true_delta_pdp(ages, masks, times, bmi_lo=20, bmi_hi=35,
                           true_beta_bmi=-0.30, true_beta_int=-0.05,
                           visit_times=None):
    """
    Compute the TRUE ΔPDP from the data-generating parameters.
    """
    ages_np = ages.numpy()
    delta_v = bmi_hi - bmi_lo

    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])

    marginal = delta_v * (true_beta_bmi + true_beta_int * ages_np.mean())

    q33, q67 = np.percentile(ages_np, [33, 67])
    age_groups = {
        f'Young (AGEc < {q33:.1f})': ages_np < q33,
        f'Middle ({q33:.1f} ≤ AGEc < {q67:.1f})': (ages_np >= q33) & (ages_np < q67),
        f'Old (AGEc ≥ {q67:.1f})': ages_np >= q67,
    }

    print(f"\n{'='*60}")
    print(f"TRUE ΔPDP (BMI {bmi_lo} → {bmi_hi})")
    print(f"  β_BMI = {true_beta_bmi}, β_int = {true_beta_int}, Δv = {delta_v}")
    print(f"{'='*60}")
    print(f"\n  Marginal: Δv × (β_BMI + β_int × mean_AGEc)")
    print(f"  = {delta_v} × ({true_beta_bmi} + {true_beta_int} × {ages_np.mean():.3f})")
    print(f"  = {marginal:.4f}")

    print(f"\n  {'Group':<35s} {'mean AGEc':>10s} {'ΔPDP':>10s} {'n':>6s}")
    print(f"  {'-'*65}")

    true_deltas = {}
    for group_name, group_mask in age_groups.items():
        mean_age = ages_np[group_mask].mean()
        delta = delta_v * (true_beta_bmi + true_beta_int * mean_age)
        true_deltas[group_name] = delta
        print(f"  {group_name:<35s} {mean_age:>+10.3f} {delta:>+10.4f} "
              f"{group_mask.sum():>6d}")

    return marginal, true_deltas, age_groups


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_pdp(results, ages, masks, times, bmi_values,
             save_path="pdp_bmi.png", visit_times=None):
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
        visit_times = np.array([0, 5, 10, 15])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(bmi_values)))

    for ax_idx, (group_name, group_mask) in enumerate(age_groups.items()):
        ax = axes[ax_idx]

        for bmi_idx, bmi_v in enumerate(bmi_values):
            mu = results[bmi_v].numpy()
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
                ax.plot(visit_t_plot, mean_pred, 'o-', color=colors[bmi_idx],
                        label=f'BMI={bmi_v}', linewidth=2, markersize=5)

        ax.set_title(group_name, fontsize=11)
        ax.set_xlabel('Time (years)')
        if ax_idx == 0:
            ax.set_ylabel('Predicted ISA15')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('PDP of BMI on ISA15, stratified by baseline age', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"PDP saved to {save_path}")
    plt.close()


def plot_pdp_marginal(results, masks, times, bmi_values,
                      save_path="pdp_marginal.png", visit_times=None,
                      ice_results=None, ice_n=50, seed=1):
    """
    Plot marginal PDP + ICE curves.

    If ice_results is provided (from compute_pdp_with_blup), plots
    subject-level predictions WITH random effects.
    Otherwise plots ICE from population-mean predictions (no RE).

    Args:
        results:     dict {v: (N, T)} population mean PDP
        ice_results: dict {v: (N, T)} subject-level predictions with BLUP (optional)
        ice_n:       number of ICE curves to draw
    """
    masks_np = masks.numpy()
    times_np = times.numpy()

    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(bmi_values)))

    N = masks_np.shape[0]
    rng = np.random.RandomState(seed)
    ice_ids = rng.choice(N, min(ice_n, N), replace=False)

    # Use subject-level predictions if available, else population mean
    ice_source = ice_results if ice_results is not None else results
    ice_label = "ICE (with RE)" if ice_results is not None else "ICE (pop. mean)"

    # Plot ICE curves (grey)
    for bmi_v in bmi_values:
        mu = ice_source[bmi_v].numpy()
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

    # Plot PDP curves (colored, on top) — always from population mean
    for bmi_idx, bmi_v in enumerate(bmi_values):
        mu = results[bmi_v].numpy()
        closest = _closest_obs_per_subject(mu, masks_np, times_np, visit_times)

        mean_pred, visit_t_plot = [], []
        for vt in visit_times:
            if len(closest[vt]) > 10:
                mean_pred.append(closest[vt].mean())
                visit_t_plot.append(vt)

        ax.plot(visit_t_plot, mean_pred, 'o-', color=colors[bmi_idx],
                label=f'BMI={bmi_v}', linewidth=2.5, markersize=6, zorder=2)

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Predicted ISA15')
    title = 'Marginal PDP of BMI on ISA15'
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