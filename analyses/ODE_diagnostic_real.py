"""
ODE latent-state diagnostic: what did the ODE learn about a covariate?

Extract z(t) trajectories under the six counterfactual profiles
and compare them to diagnose:
  - Instantaneous (current-value): z(t) tracks current covariate group
  - Baseline + current: z(t) depends on cov(0) and cov(t) only
  - Path-dependent (cumulative): z(t) differs for profiles with
    same endpoints but different paths

Usage:
    from ODE_diagnostic_real import extract_zt_profiles, plot_zt_diagnostic

    zt_dict, pdp_dict, profiles = extract_zt_profiles(model, loader, device, ...)
    plot_zt_diagnostic(zt_dict, pdp_dict, profiles, visit_times)
"""
from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt

from PDP_continuous_time import (
    resample_xaug_to_grid,
    make_profiles_continuous,
    build_profile_xaug_continuous,
    PROFILE_COLOURS,
    PROFILE_LABELS,
    PROFILE_ORDER,
)


def _param_list(model):
    return [p for p in model.parameters() if p.requires_grad]


# ─────────────────────────────────────────────────────────────────────
#  Core extraction
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_zt_profiles(model, loader, device,
                        visit_times, q25, q75,
                        n_tv,
                        covariate_idx=0,
                        max_subjects=None,
                        mask_type="binary",
                        verbose=True):
    """
    Extract z(t) and mu(t) under six counterfactual covariate profiles.

    Uses the same profile construction, resampling, and counterfactual
    building as PDP_continuous_time to guarantee identical results.
    Extracts z(t) by calling encoder → _integrate → z_norm directly
    (no return_hidden needed).

    Args:
        model:         trained NeuralODEModel
        loader:        DataLoader (any batch size)
        device:        torch device
        visit_times:   (L,) array of anchor times
        q25, q75:      covariate quantiles for profile construction
        n_tv:          K, number of time-varying covariates
        covariate_idx: which covariate to intervene on (0-indexed)
        max_subjects:  cap number of subjects (None = all)
        mask_type:     "binary" or "cumulative" — must match training
        verbose:       print summary

    Returns:
        zt_dict:  {profile_key: (N, L, d) np.array}  latent states
        pdp_dict: {profile_key: (N, L) np.array}     population predictions
        profiles: {profile_key: (L,) np.array}       the covariate profiles
    """
    model.eval()
    profiles = make_profiles_continuous(visit_times, q25, q75)
    grid = np.array(visit_times, dtype=np.float64)
    L = len(grid)
    K = n_tv

    zt_accum = {name: [] for name in profiles}
    mu_accum = {name: [] for name in profiles}
    n_total = 0

    for batch in loader:
        pids, x_aug, y_pad, target_mask, static = batch
        x_aug = x_aug.to(device)
        target_mask = target_mask.to(device)
        static = static.to(device)
        B = x_aug.shape[0]

        if max_subjects is not None and n_total >= max_subjects:
            break

        # Resample onto eval grid (PDP version — handles cumulative masks)
        x_aug_grid, obs_mask_grid = resample_xaug_to_grid(
            x_aug, target_mask, grid, K
        )

        for name, prof_values in profiles.items():
            # Build counterfactual (same as PDP path)
            x_cf = build_profile_xaug_continuous(
                x_aug_grid, target_col=covariate_idx,
                profile_values=prof_values, n_tv=K,
                mask_type=mask_type,
            )

            # --- Extract z(t) manually ---
            t_pad = x_cf[:, :, 0]
            x_interp = x_cf[:, :, 1:1+K]
            mask = x_cf[:, :, 1+K:1+2*K]

            t0 = t_pad[:, 0:1]
            x_baseline = x_interp[:, 0]
            enc_in = torch.cat([t0, x_baseline, static], dim=-1)
            z0 = model.encoder(enc_in)

            zt = model._integrate(z0, t_pad, x_interp, mask)
            zt = model.z_norm(zt)

            # --- Decoder (standard path — guaranteed correct) ---
            mu, V, Z, D, sig2, reg_dict = model.decoder(
                zt, x_interp, mask, static, obs_mask=obs_mask_grid,
            )

            zt_accum[name].append(zt.cpu().numpy())
            mu_accum[name].append(mu.cpu().numpy())

        n_total += B

    # Concatenate across batches
    zt_dict = {name: np.concatenate(arrs, axis=0)
               for name, arrs in zt_accum.items()}
    pdp_dict = {name: np.concatenate(arrs, axis=0)
                for name, arrs in mu_accum.items()}

    if verbose:
        N = zt_dict[list(zt_dict.keys())[0]].shape[0]
        d = zt_dict[list(zt_dict.keys())[0]].shape[2]
        print(f"Extracted z(t): N={N}, L={L}, d={d}")
        print(f"Profiles: {list(profiles.keys())}")

    return zt_dict, pdp_dict, profiles


# ─────────────────────────────────────────────────────────────────────
#  Plotting — z(t) dimensions + decoded output
# ─────────────────────────────────────────────────────────────────────

def plot_zt_diagnostic(zt_dict, pdp_dict, profiles, visit_times,
                       dims_to_plot=None, save_path=None,
                       figsize_per_dim=(14, 3)):
    """
    Plot z(t) trajectories averaged across subjects for each profile.

    Creates:
      1. One row: covariate profiles (input)
      2. One row per latent dimension: mean z_k(t) for each profile
      3. One summary row: mu(t) = rho(z)^T beta for each profile
    """
    profile_names = list(zt_dict.keys())
    N, L, d = zt_dict[profile_names[0]].shape
    t = np.array(visit_times)

    if dims_to_plot is None:
        dims_to_plot = list(range(d))

    # Use PDP colour scheme
    colors = {k: PROFILE_COLOURS.get(k, 'gray') for k in profile_names}
    linestyles = {}
    for k in profile_names:
        if "stable" in k:
            linestyles[k] = "-"
        elif "late" in k:
            linestyles[k] = "--"
        else:
            linestyles[k] = "-."

    n_rows = len(dims_to_plot) + 2  # profiles + latent dims + mu
    fig, axes = plt.subplots(n_rows, 1,
                              figsize=(figsize_per_dim[0],
                                       figsize_per_dim[1] * n_rows),
                              sharex=True)

    # --- Row 0: covariate profiles (input) ---
    ax = axes[0]
    for name in profile_names:
        label = PROFILE_LABELS.get(name, name)
        ax.plot(t, profiles[name],
                color=colors.get(name, 'gray'),
                linestyle=linestyles.get(name, '-'),
                linewidth=2, label=label)
    ax.set_ylabel("Covariate profile")
    ax.set_title("Counterfactual profiles (input)")
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Rows 1..d: latent dimensions ---
    for row_idx, k in enumerate(dims_to_plot):
        ax = axes[row_idx + 1]
        for name in profile_names:
            z_mean = zt_dict[name][:, :, k].mean(axis=0)
            z_std = zt_dict[name][:, :, k].std(axis=0)
            ax.plot(t, z_mean,
                    color=colors.get(name, 'gray'),
                    linestyle=linestyles.get(name, '-'),
                    linewidth=2)
            ax.fill_between(t, z_mean - z_std, z_mean + z_std,
                            color=colors.get(name, 'gray'), alpha=0.08)
        ax.set_ylabel(f"z_{k}(t)")
        ax.set_title(f"Latent dimension {k}")
        ax.grid(True, alpha=0.3)

    # --- Last row: decoded population prediction mu(t) ---
    ax = axes[-1]
    for name in profile_names:
        mu_mean = pdp_dict[name].mean(axis=0)
        label = PROFILE_LABELS.get(name, name)
        ax.plot(t, mu_mean,
                color=colors.get(name, 'gray'),
                linestyle=linestyles.get(name, '-'),
                linewidth=2, label=label)
    ax.set_ylabel("E[IST]")
    ax.set_xlabel("Time (years)")
    ax.set_title("Population prediction (PDP)")
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────
#  Plotting — pairwise z(t) comparisons
# ─────────────────────────────────────────────────────────────────────

def plot_pairwise_diagnostic(zt_dict, visit_times, save_path=None):
    """
    Focused pairwise comparisons to diagnose the type of covariate effect.

    Panel 1: Late spike vs Stable high (after crossover)
        Same current value but different baseline.
        If z(t) identical → pure current value
        If z(t) differs → baseline or path effect

    Panel 2: Late spike vs Gradual rise (same endpoints)
        Same cov(0)≈Q25, same cov(T)≈Q75, different paths.
        If z(t) identical → baseline + current (no path memory)
        If z(t) differs → genuine path-dependence

    Panel 3: Late decline vs Gradual decline (same endpoints)
        Same diagnostic as Panel 2.
    """
    profile_names = list(zt_dict.keys())
    N, L, d = zt_dict[profile_names[0]].shape
    t = np.array(visit_times)

    pairs = [
        ("late_spike", "stable_high",
         "Same current value after crossover\n→ differ = baseline/history effect"),
        ("late_spike", "gradual_rise",
         "Same endpoints, different paths\n→ differ = genuine path-dependence"),
        ("late_decline", "gradual_decline",
         "Same endpoints, different paths\n→ differ = genuine path-dependence"),
    ]

    # Filter to available profiles
    pairs = [(a, b, t_) for a, b, t_ in pairs
             if a in zt_dict and b in zt_dict]

    if not pairs:
        print("No matching profile pairs found for pairwise diagnostic.")
        return

    # Pick top-4 most variable latent dims
    all_z = np.concatenate([zt_dict[n] for n in profile_names], axis=0)
    dim_var = all_z.std(axis=(0, 1))
    top_dims = np.argsort(dim_var)[-4:][::-1]

    fig, axes = plt.subplots(len(pairs), len(top_dims),
                              figsize=(4 * len(top_dims), 4 * len(pairs)),
                              sharex=True)
    if len(pairs) == 1:
        axes = axes[np.newaxis, :]

    for row, (name_a, name_b, title) in enumerate(pairs):
        for col, k in enumerate(top_dims):
            ax = axes[row, col]

            za_mean = zt_dict[name_a][:, :, k].mean(axis=0)
            zb_mean = zt_dict[name_b][:, :, k].mean(axis=0)
            za_std = zt_dict[name_a][:, :, k].std(axis=0)
            zb_std = zt_dict[name_b][:, :, k].std(axis=0)

            label_a = PROFILE_LABELS.get(name_a, name_a)
            label_b = PROFILE_LABELS.get(name_b, name_b)

            ax.plot(t, za_mean, color=PROFILE_COLOURS.get(name_a, 'blue'),
                    linewidth=2, label=label_a)
            ax.plot(t, zb_mean, color=PROFILE_COLOURS.get(name_b, 'red'),
                    linestyle='--', linewidth=2, label=label_b)
            ax.fill_between(t, za_mean - za_std, za_mean + za_std,
                            color=PROFILE_COLOURS.get(name_a, 'blue'), alpha=0.1)
            ax.fill_between(t, zb_mean - zb_std, zb_mean + zb_std,
                            color=PROFILE_COLOURS.get(name_b, 'red'), alpha=0.1)

            diff = np.abs(za_mean - zb_mean).mean()
            ax.set_title(f"z_{k}(t)  |Δ|={diff:.4f}", fontsize=10)

            if col == 0:
                ax.set_ylabel(title, fontsize=8)
            if row == len(pairs) - 1:
                ax.set_xlabel("Time (years)")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Pairwise z(t) diagnostic: what did the ODE learn?",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────
#  Quantitative summary
# ─────────────────────────────────────────────────────────────────────

def print_diagnostic_summary(zt_dict, pdp_dict, visit_times):
    """
    Print a quantitative summary comparing z(t) and decoded mu(t)
    across diagnostic pairs.
    """
    t = np.array(visit_times)
    L = len(t)
    profile_names = list(zt_dict.keys())
    d = zt_dict[profile_names[0]].shape[2]

    print("\n" + "=" * 70)
    print("ODE DIAGNOSTIC SUMMARY")
    print("=" * 70)

    # ── Decoded output table ────────────────────────────────────────
    print("\n  DECODED OUTPUT: E[ρ(z)ᵀβ] per profile")
    header = f"    {'Profile':<25s}"
    for vt in visit_times:
        header += f"  t={vt:<5.1f}"
    print(header)
    print(f"    {'-' * (25 + 8 * L)}")
    for name in profile_names:
        mu_mean = pdp_dict[name].mean(axis=0)
        row = f"    {PROFILE_LABELS.get(name, name):<25s}"
        for ell in range(L):
            row += f"  {mu_mean[ell]:>7.2f}"
        print(row)

    # ── z(t) pairwise tests ─────────────────────────────────────────
    tests = [
        ("CURRENT-VALUE TEST",
         "late_spike", "stable_high",
         "Same current value after crossover. If L2≈0 → pure current-value."),
        ("PATH-DEPENDENCE TEST (rising)",
         "late_spike", "gradual_rise",
         "Same endpoints, different paths. If L2≈0 → baseline+current only."),
        ("PATH-DEPENDENCE TEST (declining)",
         "late_decline", "gradual_decline",
         "Same endpoints, different paths. If L2≈0 → baseline+current only."),
        ("BASELINE TEST",
         "stable_low", "late_spike",
         "Same baseline (Q25). Early z(t) should be identical."),
    ]

    for test_name, name_a, name_b, interpretation in tests:
        if name_a not in zt_dict or name_b not in zt_dict:
            print(f"\n  {test_name}: SKIPPED (profile not found)")
            continue

        za = zt_dict[name_a].mean(axis=0)   # (L, d)
        zb = zt_dict[name_b].mean(axis=0)

        l2_per_time = np.sqrt(((za - zb) ** 2).sum(axis=1))
        l2_total = np.sqrt(((za - zb) ** 2).sum())
        z_scale = np.sqrt((za ** 2).sum() + (zb ** 2).sum()) / 2
        l2_relative = l2_total / (z_scale + 1e-10)

        # Also compare decoded mu
        mu_a = pdp_dict[name_a].mean(axis=0)
        mu_b = pdp_dict[name_b].mean(axis=0)
        mu_diff = mu_a - mu_b

        label_a = PROFILE_LABELS.get(name_a, name_a)
        label_b = PROFILE_LABELS.get(name_b, name_b)

        print(f"\n  {test_name}")
        print(f"    {label_a}  vs  {label_b}")
        print(f"    {interpretation}")
        print(f"    z(t) L2 total: {l2_total:.6f}  (relative: {l2_relative:.4f})")
        print(f"    {'t':>8s}  {'||Δz||':>10s}  {'Δmu':>10s}")
        for ell, vt in enumerate(visit_times):
            print(f"    {vt:8.1f}  {l2_per_time[ell]:10.6f}  {mu_diff[ell]:+10.4f}")

    # ── Verdict ─────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("INTERPRETATION GUIDE:")
    print("  • Current-value test L2 ≈ 0 → ODE ignores history, pure current-value")
    print("  • Current-value test L2 > 0, path test L2 ≈ 0 → baseline + current")
    print("  • Path test L2 > 0 → genuine path-dependence")
    print("  • Δmu is the definitive check: decoded prediction difference")
    print("=" * 70)