"""
Compute Partial Dependence Plots (PDP) for BMI, stratified by AGE.
If the CDE learned the interaction, the BMI slope should be steeper for older subjects.

Run after training — requires `model` and `loader` in scope.
Add this at the end of train_step1.py.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_pdp(model, loader, device, bmi_values, n_tv=2, interp="linear",
                bmi_mode="constant", bmi_slope=None, bmi_col=1):
    """
    Compute CAUSAL PDP for BMI interventions.
    
    At each time index ℓ, the CDE only integrates over the path up to ℓ.
    - interp="linear": inherently causal (dX at t only uses t-1 and t)
                        Run CDE once on full path — z(t) is automatically causal.
    - interp="cubic":  truncate path at each ℓ, recompute cubic coefficients.
                        More expensive (O(T²)) but causal with smoother interpolation.
    
    BMI intervention modes:
    - bmi_mode="constant": replace BMI(t) with constant bmi_v at all times (original)
    - bmi_mode="linear":   replace BMI(t) with bmi_v + bmi_slope * t
                           Preserves temporal dynamics (dBMI/dt = bmi_slope ≠ 0)
                           so the CDE sees a non-flat covariate path.
                           If bmi_slope is None, estimated from population mean BMI drift.
    - bmi_mode="shifted":  replace BMI_i(t) with BMI_i(t) - mean(BMI_i) + bmi_v
                           Preserves each subject's individual BMI temporal dynamics
                           while shifting the mean level to bmi_v.
    
    Returns:
        results: dict mapping bmi_value → (N, max_T) tensor of predictions
        ages, masks, times: (N,) or (N, max_T) tensors
    """
    import torchcde
    model.eval()
    
    results = {v: [] for v in bmi_values}
    all_ages = []
    all_masks = []
    all_times = []
    
    print(f"  Computing causal PDP with interp='{interp}', bmi_mode='{bmi_mode}'")
    
    # If linear mode and no slope given, estimate from data
    if bmi_mode == "linear" and bmi_slope is None:
        # Estimate population-average BMI slope from the data
        all_bmi, all_t = [], []
        with torch.no_grad():
            for batch in loader:
                _, t_pad_b, x_pad_b, _, _, mask_b, _ = batch
                obs = mask_b > 0.5
                all_bmi.append(x_pad_b[:, :, bmi_col][obs].numpy())
                all_t.append(t_pad_b[obs].numpy())
        bmi_flat = np.concatenate(all_bmi)
        t_flat = np.concatenate(all_t)
        # Simple linear regression: BMI = a + slope * t
        from numpy.polynomial.polynomial import polyfit
        c = polyfit(t_flat, bmi_flat, 1)
        bmi_slope = c[1]
        print(f"  Estimated population BMI slope: {bmi_slope:.4f} per year")
    
    with torch.no_grad():
        for bmi_v in bmi_values:
            batch_mus = []
            
            for batch in loader:
                _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
                t_pad  = t_pad.to(device)
                x_pad  = x_pad.to(device)
                mask   = mask.to(device)
                c_mask = c_mask.to(device)
                s      = s.to(device)
                
                N, T = t_pad.shape
                
                # Counterfactual BMI intervention
                x_cf = x_pad.clone()
                
                if bmi_mode == "constant":
                    # Original: flat BMI at bmi_v
                    x_cf[:, :, bmi_col] = bmi_v
                    
                elif bmi_mode == "linear":
                    # Linear trajectory: BMI(t) = bmi_v + slope * t
                    # Preserves dBMI/dt so CDE sees temporal dynamics
                    x_cf[:, :, bmi_col] = bmi_v + bmi_slope * t_pad
                    
                elif bmi_mode == "shifted":
                    # Shift each subject's real BMI trajectory to have mean = bmi_v
                    # Preserves individual temporal dynamics (shape, variability)
                    bmi_real = x_pad[:, :, bmi_col]              # (N, T)
                    # Compute per-subject mean BMI (at observed times only)
                    bmi_masked = bmi_real * mask                  # zero out padded
                    n_obs = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (N, 1)
                    bmi_mean = bmi_masked.sum(dim=1, keepdim=True) / n_obs  # (N, 1)
                    # Shift: preserve dynamics, change level
                    x_cf[:, :, bmi_col] = bmi_real - bmi_mean + bmi_v
                    
                else:
                    raise ValueError(f"bmi_mode must be 'constant', 'linear', or 'shifted', "
                                     f"got '{bmi_mode}'")
                
                if interp == "linear":
                    # Linear interpolation is causal: z(t_k) only depends on X_0..X_k
                    # Run CDE once on full path
                    mu, V, _, _ , _ = model(t_pad, x_cf, c_mask, s, mask,
                                        y_pad=None, interp="linear")
                    batch_mus.append(mu.cpu())
                    
                elif interp == "cubic":
                    # Cubic is non-causal → truncate path at each ℓ, recompute
                    mu_all = torch.zeros(N, T, device=device)
                    
                    x_cde = x_cf[:, :, :n_tv]  # (N, T, n_tv)
                    X_in_full = torch.cat([t_pad[..., None], x_cde, c_mask[..., None]], dim=-1)
                    
                    for ell in range(1, T + 1):
                        # Truncate path to [0, ℓ]
                        X_in_trunc = X_in_full[:, :ell, :]         # (N, ℓ, C)
                        grid_trunc = torch.arange(ell, device=device, dtype=t_pad.dtype)
                        
                        # Cubic spline on truncated path (causal: no future data)
                        if ell < 2:
                            # Need at least 2 points for interpolation
                            continue
                        
                        coeffs = torchcde.natural_cubic_coeffs(X_in_trunc)
                        X = torchcde.CubicSpline(coeffs)
                        
                        # Encoder
                        x0 = X.evaluate(grid_trunc[0])
                        encoder_in = torch.cat([x0[:, :-1], s], dim=-1)
                        z0 = model.encoder(encoder_in)
                        
                        # CDE integration up to ℓ
                        model.func.set_context(X, s)
                        zt = torchcde.cdeint(
                            X=X, z0=z0, func=model.func,
                            t=grid_trunc,
                            method=model.cfg.solver,
                            options={"step_size": 1.0},
                            atol=model.cfg.atol, rtol=model.cfg.rtol,
                            adjoint=False
                        )  # (N, ℓ, H)
                        
                        zt = model.z_norm(zt)
                        
                        # Only need prediction at the last time point (ℓ-1)
                        z_ell = zt[:, -1, :]  # (N, H)
                        
                        # Decode at time ℓ-1 (0-indexed)
                        rho = model.decoder.rho_net(z_ell.unsqueeze(1))
                        rho = model.decoder.rho_norm(rho)
                        h_ell = (rho * model.decoder.beta_neural).sum(dim=-1).squeeze(1)  # (N,)
                        
                        # Parametric part at time ℓ-1
                        W_full = model.decoder._build_W(t_pad, x_cf, s)  # (N, T, n_W)
                        W_ell = W_full[:, ell-1, :]                       # (N, n_W)
                        beta = model.decoder._last_beta
                        mu_param_ell = (W_ell * beta).sum(dim=-1)        # (N,)
                        
                        mu_all[:, ell-1] = mu_param_ell + h_ell
                    
                    batch_mus.append(mu_all.cpu())
                
                else:
                    raise ValueError(f"interp must be 'cubic' or 'linear', got '{interp}'")
                
                if bmi_v == bmi_values[0]:
                    all_ages.append(s[:, 1].cpu())
                    all_masks.append(mask.cpu())
                    all_times.append(t_pad.cpu())
            
            # Pad to common max T across batches
            max_T = max(m.shape[1] for m in batch_mus)
            padded = []
            for m in batch_mus:
                if m.shape[1] < max_T:
                    pad = torch.zeros(m.shape[0], max_T - m.shape[1])
                    m = torch.cat([m, pad], dim=1)
                padded.append(m)
            results[bmi_v] = torch.cat(padded, dim=0)  # (N, max_T)
    
    ages = torch.cat(all_ages, dim=0)         # (N,)
    
    # Pad masks and times to common max T
    max_T = max(m.shape[1] for m in all_masks)
    padded_masks = []
    padded_times = []
    for m, t in zip(all_masks, all_times):
        if m.shape[1] < max_T:
            m = torch.cat([m, torch.zeros(m.shape[0], max_T - m.shape[1])], dim=1)
            t = torch.cat([t, torch.zeros(t.shape[0], max_T - t.shape[1])], dim=1)
        padded_masks.append(m)
        padded_times.append(t)
    
    masks = torch.cat(padded_masks, dim=0)    # (N, max_T)
    times = torch.cat(padded_times, dim=0)    # (N, max_T)
    
    return results, ages, masks, times


def _closest_obs_per_subject(mu, masks_np, times_np, visit_times):
    """
    For each target visit time, find the closest observed time point per subject.
    This ensures ALL subjects contribute at every visit time (no survivor selection).
    
    Like R's: slice_min(order_by = abs(time - t0), n = 1)
    
    Args:
        mu:          (N, T) predictions
        masks_np:    (N, T) observation mask
        times_np:    (N, T) real times
        visit_times: array of target times
    
    Returns:
        dict: {vt: array of predictions, one per subject} for each visit time
    """
    N, T = mu.shape
    result = {}
    
    for vt in visit_times:
        preds = []
        for i in range(N):
            # Find observed time points for this subject
            obs_idx = np.where(masks_np[i] > 0.5)[0]
            if len(obs_idx) == 0:
                continue
            
            # Find closest observed time to target
            obs_times = times_np[i, obs_idx]
            closest = obs_idx[np.argmin(np.abs(obs_times - vt))]
            preds.append(mu[i, closest])
        
        result[vt] = np.array(preds)
    
    return result


def plot_pdp(results, ages, masks, times, bmi_values, save_path="pdp_bmi.png",
             visit_times=None):
    """
    Plot PDP over time for different BMI values, stratified by age tertiles.
    Uses closest observation per subject (R-style) to avoid survivor selection.
    """
    # Age tertiles
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
        visit_times = np.array([0, 5, 10, 5])
    
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
            
            mean_pred = []
            visit_t_plot = []
            for vt in visit_times:
                if len(closest[vt]) > 10:
                    mean_pred.append(closest[vt].mean())
                    visit_t_plot.append(vt)
            
            if len(visit_t_plot) > 0:
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


def plot_pdp_marginal(results, masks, times, bmi_values, save_path="pdp_bmi_marginal.png",
                      visit_times=None, ice_n=30, seed=1, ages=None):
    """
    Plot marginal PDP (averaged over all subjects) + optional ICE curves.
    Uses closest observation per subject (R-style) to avoid survivor selection.
    """
    masks_np = masks.numpy()
    times_np = times.numpy()
    
    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(bmi_values)))
    
    N = masks_np.shape[0]
    
    # Select ICE subjects
    rng = np.random.RandomState(seed)
    ice_ids = rng.choice(N, min(ice_n, N), replace=False)
    
    # Plot ICE curves (grey, behind PDP)
    for bmi_idx, bmi_v in enumerate(bmi_values):
        mu = results[bmi_v].numpy()
        
        for i in ice_ids:
            ice_times = []
            ice_preds = []
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
    for bmi_idx, bmi_v in enumerate(bmi_values):
        mu = results[bmi_v].numpy()
        closest = _closest_obs_per_subject(mu, masks_np, times_np, visit_times)
        
        mean_pred = []
        visit_t_plot = []
        for vt in visit_times:
            if len(closest[vt]) > 10:
                mean_pred.append(closest[vt].mean())
                visit_t_plot.append(vt)
        
        ax.plot(visit_t_plot, mean_pred, 'o-', color=colors[bmi_idx],
                label=f'BMI={bmi_v}', linewidth=2.5, markersize=6, zorder=2)
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Predicted ISA15')
    ax.set_title('Marginal PDP of BMI on ISA15 + ICE (grey)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Marginal PDP saved to {save_path}")
    plt.close()

def compute_true_delta_pdp(ages, masks, times, bmi_lo=20, bmi_hi=35,
                           true_beta_bmi=-0.175, true_beta_int=-0.015,
                           visit_times=None):
    """
    Compute the TRUE ΔPDP from the data-generating parameters, at every visit time.
    
    For Scenario 2 (linear BMI + BMI×AGEc interaction, instantaneous effect):
      Y_ij = ... + β_BMI * BMI_ij + β_int * BMI_ij * AGEc_i + ...
    
    The true ΔPDP = Δv × (β_BMI + β_int × mean_AGEc_group).
    This is TIME-INVARIANT in Scenario 2 (no time×BMI interaction),
    but we return it at each visit time for easy comparison with model estimates.
    
    Returns:
        marginal_per_t: dict {vt: scalar} — marginal ΔPDP at each visit time
        stratified_per_t: dict {group_name: {vt: scalar}} — per-group per-time
        age_groups: dict {group_name: bool_mask} — for reuse
    """
    ages_np = ages.numpy() if hasattr(ages, 'numpy') else np.asarray(ages)
    delta_v = bmi_hi - bmi_lo
    
    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])
    
    # Marginal (all subjects): time-invariant for Scenario 2
    mean_age_all = ages_np.mean()
    marginal_scalar = delta_v * (true_beta_bmi + true_beta_int * mean_age_all)
    marginal_per_t = {vt: marginal_scalar for vt in visit_times}
    
    # Stratified by age tertiles
    q33, q67 = np.percentile(ages_np, [33, 67])
    age_groups = {
        f'Young (AGEc < {q33:.1f})': ages_np < q33,
        f'Middle ({q33:.1f} ≤ AGEc < {q67:.1f})': (ages_np >= q33) & (ages_np < q67),
        f'Old (AGEc ≥ {q67:.1f})': ages_np >= q67,
    }
    
    print(f"\n{'='*60}")
    print(f"TRUE ΔPDP (BMI {bmi_lo} → {bmi_hi})")
    print(f"  β_BMI = {true_beta_bmi}, β_int = {true_beta_int}, Δv = {delta_v}")
    print(f"  mean AGEc (all) = {mean_age_all:.3f}")
    print(f"{'='*60}")
    print(f"\n  Marginal: Δv × (β_BMI + β_int × mean_AGEc)")
    print(f"  = {delta_v} × ({true_beta_bmi} + {true_beta_int} × {mean_age_all:.3f})")
    print(f"  = {marginal_scalar:.4f}  [constant across all visit times]")
    
    print(f"\n  {'Group':<35s} {'mean AGEc':>10s} {'ΔPDP':>10s} {'n':>6s}")
    print(f"  {'-'*65}")
    
    stratified_per_t = {}
    for group_name, group_mask in age_groups.items():
        mean_age = ages_np[group_mask].mean()
        n_group = group_mask.sum()
        delta = delta_v * (true_beta_bmi + true_beta_int * mean_age)
        stratified_per_t[group_name] = {vt: delta for vt in visit_times}
        print(f"  {group_name:<35s} {mean_age:>+10.3f} {delta:>+10.4f} {n_group:>6d}")
    
    print()
    return marginal_per_t, stratified_per_t, age_groups


def compute_delta_pdp_stratified(results, ages, masks, times, bmi_lo=20, bmi_hi=35,
                                  true_beta_bmi=-0.175, true_beta_int=-0.015,
                                  visit_times=None):
    """
    Compute ΔPDP stratified by age tertiles.
    Uses closest observation per subject (R-style) to avoid survivor selection.

    Args:
        true_beta_bmi: true main BMI coefficient
        true_beta_int: true BMI×AGEc interaction coefficient
    """
    mu_lo = results[bmi_lo].numpy()
    mu_hi = results[bmi_hi].numpy()
    masks_np = masks.numpy()
    ages_np = ages.numpy()
    times_np = times.numpy()
    
    delta = mu_hi - mu_lo  # (N, T)
    
    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])
    
    # Age tertiles
    q33, q67 = np.percentile(ages_np, [33, 67])
    age_groups = {
        f'Young (AGEc < {q33:.1f})': ages_np < q33,
        f'Middle ({q33:.1f} ≤ AGEc < {q67:.1f})': (ages_np >= q33) & (ages_np < q67),
        f'Old (AGEc ≥ {q67:.1f})': ages_np >= q67,
    }
    
    delta_v = bmi_hi - bmi_lo
    
    print(f"\n{'='*70}")
    print(f"Stratified ΔPDP (BMI {bmi_lo} → {bmi_hi})")
    print(f"  True β_BMI = {true_beta_bmi}, β_BMI×AGEc = {true_beta_int}")
    print(f"{'='*70}")
    
    summary = {}
    
    for group_name, group_mask in age_groups.items():
        n_group = group_mask.sum()
        mean_age = ages_np[group_mask].mean()
        
        # True ΔPDP for this group
        true_slope = true_beta_bmi + true_beta_int * mean_age
        true_delta = true_slope * delta_v
        
        delta_group = delta[group_mask]
        masks_group = masks_np[group_mask]
        times_group = times_np[group_mask]
        
        # Also get lo/hi for gap display
        mu_lo_group = mu_lo[group_mask]
        mu_hi_group = mu_hi[group_mask]
        
        closest_delta = _closest_obs_per_subject(delta_group, masks_group,
                                                  times_group, visit_times)
        closest_lo = _closest_obs_per_subject(mu_lo_group, masks_group,
                                               times_group, visit_times)
        closest_hi = _closest_obs_per_subject(mu_hi_group, masks_group,
                                               times_group, visit_times)
        
        print(f"\n  {group_name} (n={n_group}, mean AGEc={mean_age:.2f})")
        print(f"  True ΔPDP = ({true_slope:.3f}) × {delta_v} = {true_delta:.3f}")
        print(f"  {'Time':>6s}  {'ΔPDP':>8s}  {'n':>6s}  {'gap(lo-hi)':>12s}")
        print(f"  {'-'*40}")
        
        all_deltas = []
        for vt in visit_times:
            d = closest_delta[vt]
            if len(d) > 10:
                mean_d = d.mean()
                mean_lo = closest_lo[vt].mean()
                mean_hi = closest_hi[vt].mean()
                all_deltas.append(mean_d)
                print(f"  {vt:6.0f}  {mean_d:+8.3f}  {len(d):6d}  "
                      f"{mean_lo:.1f} → {mean_hi:.1f}")
        
        summary[group_name] = {
            "estimated": np.mean(all_deltas) if all_deltas else 0,
            "true": true_delta,
        }
    
    # Summary table
    print(f"\n  {'='*50}")
    print(f"  Summary: average ΔPDP across all time points")
    print(f"  {'Group':<30s} {'Estimated':>10s} {'True':>10s}")
    print(f"  {'-'*50}")
    for group_name, vals in summary.items():
        print(f"  {group_name:<30s} {vals['estimated']:+10.3f} {vals['true']:+10.3f}")
    print()


def compute_delta_pdp(results, ages, masks, times, bmi_lo=20, bmi_hi=35,
                      true_beta_bmi=-0.175, true_beta_int=-0.015,
                      visit_times=None):
    """
    Compute ΔPDP = PDP(bmi_hi) - PDP(bmi_lo) at each visit time.
    Uses closest observation per subject (R-style) to avoid survivor selection.

    Returns:
        estimated_per_t: dict {vt: mean_delta} — model-estimated ΔPDP at each visit
        true_per_t: dict {vt: true_delta} — oracle ΔPDP at each visit (for comparison)
    """
    mu_lo = results[bmi_lo].numpy()
    mu_hi = results[bmi_hi].numpy()
    masks_np = masks.numpy()
    ages_np = ages.numpy() if hasattr(ages, 'numpy') else np.asarray(ages)
    times_np = times.numpy()
    
    delta = mu_hi - mu_lo  # (N, T)
    delta_v = bmi_hi - bmi_lo
    
    # Correct true marginal: includes interaction term
    mean_age = ages_np.mean()
    true_marginal = delta_v * (true_beta_bmi + true_beta_int * mean_age)
    
    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])
    
    closest = _closest_obs_per_subject(delta, masks_np, times_np, visit_times)
    
    print(f"\nΔPDP (BMI {bmi_lo} → {bmi_hi}):")
    print(f"  True marginal = Δv × (β_BMI + β_int × mean_AGEc)")
    print(f"               = {delta_v} × ({true_beta_bmi} + {true_beta_int} × {mean_age:.3f})")
    print(f"               = {true_marginal:.4f}")
    print(f"\n  {'Time':>6s}  {'Estimated':>10s}  {'True':>10s}  {'Bias':>10s}  {'n':>6s}")
    print(f"  {'-'*50}")
    
    estimated_per_t = {}
    true_per_t = {}
    for vt in visit_times:
        d = closest[vt]
        if len(d) > 10:
            mean_d = d.mean()
            estimated_per_t[vt] = mean_d
            true_per_t[vt] = true_marginal
            bias = mean_d - true_marginal
            print(f"  {vt:6.0f}  {mean_d:+10.4f}  {true_marginal:+10.4f}  {bias:+10.4f}  {len(d):6d}")
    
    print()
    return estimated_per_t, true_per_t


if __name__ != "__main__":
    # When imported, provide convenience functions
    pass