"""
ODE latent-state diagnostic: what did the ODE learn about BMI?

Extract z(t) trajectories under the six counterfactual BMI profiles
and compare them to diagnose:
  - Instantaneous (current-value): z(t) tracks current BMI group
  - Baseline + current: z(t) depends on BMI(0) and BMI(t) only
  - Path-dependent (cumulative): z(t) differs for profiles with
    same endpoints but different paths

Usage:
    from ode_diagnostic import extract_zt_profiles, plot_zt_diagnostic

    zt_dict, pdp_dict = extract_zt_profiles(model, loader, device, ...)
    plot_zt_diagnostic(zt_dict, pdp_dict, visit_times, save_path="zt_diag.png")
"""
from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt


def _param_list(model):
    return [p for p in model.parameters() if p.requires_grad]


def _build_profiles(visit_times, q25, q75):
    """
    Build the six counterfactual BMI profiles on the visit_times grid.

    Returns:
        profiles: dict {name: (L,) np.array of BMI values}
    """
    L = len(visit_times)
    t = np.array(visit_times, dtype=np.float64)
    t_min, t_max = t[0], t[-1]
    t_mid = (t_min + t_max) / 2.0

    # Linear interpolation helper
    def _lerp(t_arr, t0, v0, t1, v1):
        frac = np.clip((t_arr - t0) / (t1 - t0 + 1e-12), 0, 1)
        return v0 + frac * (v1 - v0)

    profiles = {
        "Stable low (Q25)": np.full(L, q25),
        "Stable high (Q75)": np.full(L, q75),
        "Late spike (Q25→Q75)": np.where(t <= t_mid, q25,
                                          _lerp(t, t_mid, q25, t_max, q75)),
        "Late Decline (Q75→Q25)": np.where(t <= t_mid, q75,
                                            _lerp(t, t_mid, q75, t_max, q25)),
        "Gradual rise": _lerp(t, t_min, q25, t_max, q75),
        "Gradual decline": _lerp(t, t_min, q75, t_max, q25),
    }
    return profiles


def _resample_batch_to_grid(t_pad, x_pad, mask, grid):
    """Resample batch covariates onto a fixed grid via linear interpolation."""
    B, T_orig, C = x_pad.shape
    M = len(grid)
    device = x_pad.device
    dtype = x_pad.dtype

    t_grid = torch.tensor(grid, device=device, dtype=dtype
                          ).unsqueeze(0).expand(B, -1).clone()
    x_grid = torch.zeros(B, M, C, device=device, dtype=dtype)
    mask_grid = torch.ones(B, M, device=device, dtype=dtype)

    t_np = t_pad.cpu().numpy()
    x_np = x_pad.cpu().numpy()
    m_np = mask.cpu().numpy()

    for i in range(B):
        obs_i = m_np[i] > 0.5
        if not obs_i.any():
            continue
        t_obs = t_np[i, obs_i]
        for c in range(C):
            vals_obs = x_np[i, obs_i, c]
            x_grid[i, :, c] = torch.tensor(
                np.interp(grid, t_obs, vals_obs),
                device=device, dtype=dtype)

    return t_grid, x_grid, mask_grid



@torch.no_grad()
def extract_zt_profiles(model, loader, device,
                        visit_times, q25, q75,
                        covariate_idx=0,
                        max_subjects=None,
                        verbose=True):
    """
    Extract z(t) and mu(t) under six counterfactual BMI profiles.

    Works with NeuralODEModel from model_ODE_skipgate.py:
      - Encoder:  z(0) = Enc(t0, static)
      - ODE:      dz/dt = f(z, t, BMI(t))   via model._integrate()
      - z_norm:   LayerNorm on z(t)
      - Decoder:  mu, V, ... = decoder(z_t, x_pad, static, obs_mask)

    For each subject, replaces covariate_idx with each profile's values
    on the visit_times grid, runs the ODE forward, and collects:
      - z(t) ∈ R^d  at each anchor time (after z_norm)
      - mu(t) population prediction from decoder

    Args:
        model:         trained NeuralODEModel
        loader:        DataLoader (any batch size)
        device:        torch device
        visit_times:   (L,) array of anchor times
        q25, q75:      covariate quantiles for profile construction
        covariate_idx: which channel in x_pad to intervene on (default 0 = BMI)
        max_subjects:  cap number of subjects (None = all)

    Returns:
        zt_dict:  {profile_name: (N, L, d) np.array}  latent states
        pdp_dict: {profile_name: (N, L) np.array}     population predictions
        profiles: {profile_name: (L,) np.array}        the BMI profiles themselves
    """
    model.eval()
    profiles = _build_profiles(visit_times, q25, q75)
    grid = np.array(visit_times, dtype=np.float64)
    L = len(grid)

    # Collect all z(t) and mu(t) per profile
    zt_accum = {name: [] for name in profiles}
    mu_accum = {name: [] for name in profiles}
    n_total = 0

    for batch in loader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        mask = mask.to(device)
        s = s.to(device)
        B = t_pad.shape[0]

        if max_subjects is not None and n_total >= max_subjects:
            break

        # Resample onto common grid
        t_grid, x_grid, mask_grid = _resample_batch_to_grid(
            t_pad, x_pad, mask, grid)

        for name, bmi_vals in profiles.items():
            # --- Counterfactual x_pad: replace BMI channel ---
            x_cf = x_grid.clone()
            x_cf[:, :, covariate_idx] = torch.tensor(
                bmi_vals, device=device, dtype=x_cf.dtype
            ).unsqueeze(0).expand(B, -1)

            # bmi_t for ODE: (N, T, 1) matching model.forward convention
            bmi_t_cf = x_cf[:, :, covariate_idx:covariate_idx + 1]

            # --- Encoder: z(0) = Enc(t0, static) ---
            t0 = t_grid[:, 0:1]                               # (B, 1)
            encoder_in = torch.cat([t0, s], dim=-1)            # (B, 1 + static_dim)
            z0 = model.encoder(encoder_in)                     # (B, H)

            # --- ODE integration: uses model._integrate ---
            zt = model._integrate(
                z0, t_grid, s,
                bmi_t=bmi_t_cf                                 # (B, L, 1)
            )                                                   # (B, L, H)

            # --- LayerNorm on z(t) ---
            zt_normed = model.z_norm(zt)                       # (B, L, H)

            # --- Decoder: get mu(t) via full decoder forward ---
            # decoder expects: z_t (B,T,H), x_pad (B,T,Cx), static (B,Cs)
            mu, V, Z_re, D, sig2, reg_dict = model.decoder(
                zt_normed, x_cf, s, obs_mask=mask_grid
            )                                                   # mu: (B, L)

            zt_accum[name].append(zt_normed.cpu().numpy())
            mu_accum[name].append(mu.cpu().numpy())

        n_total += B

    # Concatenate across batches
    zt_dict = {name: np.concatenate(arrs, axis=0) for name, arrs in zt_accum.items()}
    pdp_dict = {name: np.concatenate(arrs, axis=0) for name, arrs in mu_accum.items()}

    if verbose:
        N = zt_dict[list(zt_dict.keys())[0]].shape[0]
        d = zt_dict[list(zt_dict.keys())[0]].shape[2]
        print(f"Extracted z(t): N={N} subjects, L={L} times, d={d} dims")
        print(f"Profiles: {list(profiles.keys())}")

    return zt_dict, pdp_dict, profiles


def plot_zt_diagnostic(zt_dict, pdp_dict, profiles, visit_times,
                       dims_to_plot=None, save_path=None,
                       figsize_per_dim=(14, 3)):
    """
    Plot z(t) trajectories averaged across subjects for each profile.

    Creates:
      1. One row per latent dimension: mean z_k(t) for each profile
      2. One summary row: mu(t) = rho(z)^T beta for each profile
      3. Pairwise comparison panels for key diagnostic pairs

    Args:
        zt_dict:  {profile_name: (N, L, d)}
        pdp_dict: {profile_name: (N, L)}
        profiles: {profile_name: (L,) BMI values}
        visit_times: (L,)
        dims_to_plot: list of latent dimensions to show (None = all)
        save_path: if set, save figure
    """
    profile_names = list(zt_dict.keys())
    N, L, d = zt_dict[profile_names[0]].shape
    t = np.array(visit_times)

    if dims_to_plot is None:
        dims_to_plot = list(range(d))

    # Color scheme matching PDP plots
    colors = {
        "Stable low (Q25)": "#1f77b4",
        "Late spike (Q25→Q75)": "#17becf",
        "Gradual rise": "#d62728",
        "Stable high (Q75)": "#e377c2",
        "Late Decline (Q75→Q25)": "#ff7f0e",
        "Gradual decline": "#bcbd22",
    }
    linestyles = {
        "Stable low (Q25)": "-",
        "Late spike (Q25→Q75)": "--",
        "Gradual rise": "-.",
        "Stable high (Q75)": "-",
        "Late Decline (Q75→Q25)": "--",
        "Gradual decline": "-.",
    }

    n_rows = len(dims_to_plot) + 2  # latent dims + mu + BMI profiles
    fig, axes = plt.subplots(n_rows, 1,
                              figsize=(figsize_per_dim[0],
                                       figsize_per_dim[1] * n_rows),
                              sharex=True)

    # --- Row 0: BMI profiles themselves ---
    ax = axes[0]
    for name in profile_names:
        ax.plot(t, profiles[name],
                color=colors.get(name, 'gray'),
                linestyle=linestyles.get(name, '-'),
                linewidth=2, label=name)
    ax.set_ylabel("BMI profile")
    ax.set_title("Counterfactual BMI profiles (input)")
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Rows 1..d: latent dimensions ---
    for row_idx, k in enumerate(dims_to_plot):
        ax = axes[row_idx + 1]
        for name in profile_names:
            z_mean = zt_dict[name][:, :, k].mean(axis=0)  # (L,)
            z_std = zt_dict[name][:, :, k].std(axis=0)    # (L,)
            ax.plot(t, z_mean,
                    color=colors.get(name, 'gray'),
                    linestyle=linestyles.get(name, '-'),
                    linewidth=2, label=name)
            ax.fill_between(t, z_mean - z_std, z_mean + z_std,
                            color=colors.get(name, 'gray'), alpha=0.08)
        ax.set_ylabel(f"z_{k}(t)")
        ax.set_title(f"Latent dimension {k}")
        ax.grid(True, alpha=0.3)

    # --- Last row: population prediction mu(t) ---
    ax = axes[-1]
    for name in profile_names:
        mu_mean = pdp_dict[name].mean(axis=0)
        ax.plot(t, mu_mean,
                color=colors.get(name, 'gray'),
                linestyle=linestyles.get(name, '-'),
                linewidth=2, label=name)
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


def plot_pairwise_diagnostic(zt_dict, visit_times,
                              save_path=None):
    """
    Focused pairwise comparisons to diagnose the type of BMI effect.

    Panel 1: Late spike vs Stable high (after crossover)
        Same current BMI (Q75) but different baseline.
        If z(t) identical → pure current value
        If z(t) differs → baseline or path effect

    Panel 2: Late spike vs Gradual rise (same endpoints)
        Same BMI(0)≈Q25, same BMI(T)≈Q75, different paths.
        If z(t) identical → baseline + current (no path memory)
        If z(t) differs → genuine path-dependence

    Panel 3: Late decline vs Gradual decline (same endpoints)
        Same BMI(0)≈Q75, same BMI(T)≈Q25, different paths.
        Same diagnostic as Panel 2.
    """
    profile_names = list(zt_dict.keys())
    N, L, d = zt_dict[profile_names[0]].shape
    t = np.array(visit_times)

    pairs = [
        ("Late spike (Q25→Q75)", "Stable high (Q75)",
         "Same current BMI (Q75) after crossover\n→ differ = baseline/history effect"),
        ("Late spike (Q25→Q75)", "Gradual rise",
         "Same endpoints, different paths\n→ differ = genuine path-dependence"),
        ("Late Decline (Q75→Q25)", "Gradual decline",
         "Same endpoints, different paths\n→ differ = genuine path-dependence"),
    ]

    # Pick top-4 most variable latent dims
    all_z = np.concatenate([zt_dict[n] for n in profile_names], axis=0)
    dim_var = all_z.std(axis=(0, 1))  # (d,) variance across subjects & time
    top_dims = np.argsort(dim_var)[-4:][::-1]

    fig, axes = plt.subplots(len(pairs), len(top_dims),
                              figsize=(4 * len(top_dims), 4 * len(pairs)),
                              sharex=True)

    for row, (name_a, name_b, title) in enumerate(pairs):
        if name_a not in zt_dict or name_b not in zt_dict:
            continue
        for col, k in enumerate(top_dims):
            ax = axes[row, col] if len(pairs) > 1 else axes[col]

            za_mean = zt_dict[name_a][:, :, k].mean(axis=0)
            zb_mean = zt_dict[name_b][:, :, k].mean(axis=0)
            za_std = zt_dict[name_a][:, :, k].std(axis=0)
            zb_std = zt_dict[name_b][:, :, k].std(axis=0)

            ax.plot(t, za_mean, 'b-', linewidth=2, label=name_a)
            ax.plot(t, zb_mean, 'r--', linewidth=2, label=name_b)
            ax.fill_between(t, za_mean - za_std, za_mean + za_std,
                            color='blue', alpha=0.1)
            ax.fill_between(t, zb_mean - zb_std, zb_mean + zb_std,
                            color='red', alpha=0.1)

            # Annotate difference
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


def print_diagnostic_summary(zt_dict, visit_times):
    """
    Print a quantitative summary comparing z(t) across diagnostic pairs.

    Reports L2 distance between mean z(t) trajectories for:
      1. Late spike vs Stable high → current-value test
      2. Late spike vs Gradual rise → path-dependence test
    """
    t = np.array(visit_times)
    L = len(t)
    profile_names = list(zt_dict.keys())
    d = zt_dict[profile_names[0]].shape[2]

    print("\n" + "=" * 70)
    print("ODE DIAGNOSTIC SUMMARY")
    print("=" * 70)

    tests = [
        ("CURRENT-VALUE TEST",
         "Late spike (Q25→Q75)", "Stable high (Q75)",
         "Same current BMI after crossover. If L2≈0 → pure current-value."),
        ("PATH-DEPENDENCE TEST (rising)",
         "Late spike (Q25→Q75)", "Gradual rise",
         "Same endpoints, different paths. If L2≈0 → baseline+current only."),
        ("PATH-DEPENDENCE TEST (declining)",
         "Late Decline (Q75→Q25)", "Gradual decline",
         "Same endpoints, different paths. If L2≈0 → baseline+current only."),
        ("BASELINE TEST",
         "Stable low (Q25)", "Late spike (Q25→Q75)",
         "Same baseline (Q25). Early z(t) should be identical."),
    ]

    for test_name, name_a, name_b, interpretation in tests:
        if name_a not in zt_dict or name_b not in zt_dict:
            print(f"\n  {test_name}: SKIPPED (profile not found)")
            continue

        za = zt_dict[name_a].mean(axis=0)  # (L, d)
        zb = zt_dict[name_b].mean(axis=0)  # (L, d)

        # Per-time L2 distance
        l2_per_time = np.sqrt(((za - zb) ** 2).sum(axis=1))  # (L,)
        # Overall
        l2_total = np.sqrt(((za - zb) ** 2).sum())
        # Normalise by scale of z
        z_scale = np.sqrt((za ** 2).sum() + (zb ** 2).sum()) / 2
        l2_relative = l2_total / (z_scale + 1e-10)

        print(f"\n  {test_name}")
        print(f"    {name_a}  vs  {name_b}")
        print(f"    {interpretation}")
        print(f"    L2 total: {l2_total:.6f}  (relative: {l2_relative:.4f})")
        for ell, vt in enumerate(visit_times):
            print(f"      t={vt:5.1f}: ||Δz|| = {l2_per_time[ell]:.6f}")

    # --- Verdict ---
    print(f"\n{'─' * 70}")
    print("INTERPRETATION GUIDE:")
    print("  • Current-value test L2 ≈ 0 → ODE ignores history, pure current-value")
    print("  • Current-value test L2 > 0, path test L2 ≈ 0 → baseline + current")
    print("  • Path test L2 > 0 → genuine path-dependence (ODE encodes trajectory shape)")
    print("=" * 70)