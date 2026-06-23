"""
Diagnose what the ODE learned on the 3C real data.

Runs the z(t) diagnostic for BMI (and optionally glucose, HDL)
to determine whether the ODE learned cumulative, instantaneous,
or baseline effects for each covariate.

Usage:
    python run_real_diag.py --covariate BMI
    python run_real_diag.py --covariate GLUC
    python run_real_diag.py --covariate HDL
"""
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from Preprocess_3C import process_data
from train_ODE_real import RealDataset, collate_real
from model_ODE_real import NeuralODEModel, NeuralODEConfig
from ODE_diagnostic_real import (
    extract_zt_profiles,
    plot_zt_diagnostic,
    plot_pairwise_diagnostic,
    print_diagnostic_summary,
)
from PDP_continuous_time import PROFILE_COLOURS, PROFILE_LABELS, PROFILE_ORDER


# ─────────────────────────────────────────────────
# Profile shapes plot
# ─────────────────────────────────────────────────

def plot_profile_shapes(profiles, visit_times, covariate_name,
                        q25=None, q75=None, save_path=None):
    """
    Plot the six counterfactual intervention profiles.
    """
    t = np.array(visit_times)
    fig, ax = plt.subplots(figsize=(8, 4))

    order = [k for k in PROFILE_ORDER if k in profiles]
    if not order:
        order = list(profiles.keys())

    for name in order:
        vals = profiles[name]
        label = PROFILE_LABELS.get(name, name)
        color = PROFILE_COLOURS.get(name, 'gray')
        ls = '-' if 'stable' in name else ('--' if 'late' in name else '-.')
        lw = 2.5 if 'stable' in name else 2
        ax.plot(t, vals, linewidth=lw, label=label,
                color=color, linestyle=ls)

    if q25 is not None:
        ax.axhline(q25, color='gray', linestyle=':', alpha=0.5,
                    label=f'Q25 = {q25:.1f}')
    if q75 is not None:
        ax.axhline(q75, color='gray', linestyle=':', alpha=0.5,
                    label=f'Q75 = {q75:.1f}')

    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel(covariate_name, fontsize=12)
    ax.set_title(f"Counterfactual intervention profiles — {covariate_name}",
                 fontsize=13)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────
# Covariate-specific settings
# ─────────────────────────────────────────────────
COVARIATE_CONFIG = {
    "BMI": {"col_idx": 0, "q25": 23.1, "q75": 28.4},
    "SBP": {"col_idx": 1, "q25": 126.0, "q75": 155.0},
    "DBP": {"col_idx": 2, "q25": 70.0, "q75": 85.0},
    "GLUC": {"col_idx": 3, "q25": 4, "q75": 10},
    "HDL": {"col_idx": 4, "q25": 1.27, "q75": 1.75},
}


# ─────────────────────────────────────────────────
# Latent distance plot
# ─────────────────────────────────────────────────

def plot_latent_distance(zt_dict, visit_times, covariate_name,
                         profiles=None, q25=None, q75=None,
                         save_path=None):
    """
    Plot total L2 distance between diagnostic profile pairs over time,
    with profile shapes shown in the top panel.
    """
    t = np.array(visit_times)

    pairs = [
        ("late_spike", "stable_high",
         "Same current value after crossover"),
        ("late_spike", "gradual_rise",
         "Same endpoints, different paths (rising)"),
        ("late_decline", "gradual_decline",
         "Same endpoints, different paths (declining)"),
    ]
    pairs = [(a, b, lab) for a, b, lab in pairs
             if a in zt_dict and b in zt_dict]

    if not pairs:
        print("No matching pairs for latent distance plot.")
        return

    n_rows = 2 if profiles is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3.5 * n_rows),
                             gridspec_kw={'height_ratios': [1, 1.2] if n_rows == 2 else [1]})
    if n_rows == 1:
        axes = [axes]

    # --- Top panel: profile shapes ---
    if profiles is not None:
        ax = axes[0]
        order = [k for k in PROFILE_ORDER if k in profiles]
        if not order:
            order = list(profiles.keys())
        for name in order:
            vals = profiles[name]
            label = PROFILE_LABELS.get(name, name)
            color = PROFILE_COLOURS.get(name, 'gray')
            ls = '-' if 'stable' in name else ('--' if 'late' in name else '-.')
            ax.plot(t, vals, linewidth=2, label=label,
                    color=color, linestyle=ls)
        if q25 is not None:
            ax.axhline(q25, color='gray', linestyle=':', alpha=0.4)
        if q75 is not None:
            ax.axhline(q75, color='gray', linestyle=':', alpha=0.4)
        ax.set_ylabel(covariate_name, fontsize=11)
        ax.set_title(f"Counterfactual profiles", fontsize=12)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t[0], t[-1])

    # --- Bottom panel: L2 distances ---
    ax = axes[-1]
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    linestyles = ['-', '--', '-.']

    for idx, (name_a, name_b, label) in enumerate(pairs):
        za = zt_dict[name_a].mean(axis=0)
        zb = zt_dict[name_b].mean(axis=0)
        dist = np.sqrt(((za - zb) ** 2).sum(axis=1))
        ax.plot(t, dist, linewidth=2, label=label,
                color=colors[idx], linestyle=linestyles[idx])

    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel(
        r"$\|\bar{\Lambda}_a(t) - \bar{\Lambda}_b(t)\|$", fontsize=12)
    ax.set_title(
        f"Latent state divergence — {covariate_name}", fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────
# Latent PDP plot (per-dimension response to intervention)
# ─────────────────────────────────────────────────

def plot_latent_pdp(zt_dict, visit_times, covariate_name,
                    top_k=4, save_path=None):
    """
    Plot the top-k most responsive latent dimensions under the
    six counterfactual profiles.
    """
    profile_names = list(zt_dict.keys())
    N, L, d = zt_dict[profile_names[0]].shape
    t = np.array(visit_times)

    # Find top-k most responsive dimensions
    all_z = np.stack([zt_dict[n].mean(axis=0) for n in profile_names])
    dim_spread = all_z.std(axis=0).mean(axis=0)  # (d,)
    top_dims = np.argsort(dim_spread)[-top_k:][::-1]

    fig, axes = plt.subplots(1, top_k, figsize=(4 * top_k, 3.5),
                             sharey=False)
    if top_k == 1:
        axes = [axes]

    for col, k in enumerate(top_dims):
        ax = axes[col]
        for name in profile_names:
            z_mean = zt_dict[name][:, :, k].mean(axis=0)
            label = PROFILE_LABELS.get(name, name)
            color = PROFILE_COLOURS.get(name, 'gray')
            ls = '-' if 'stable' in name else ('--' if 'late' in name else '-.')
            ax.plot(t, z_mean, linewidth=2, label=label,
                    color=color, linestyle=ls)

        ax.set_xlabel("Time (years)")
        ax.set_title(f"$\\Lambda_{{{k}}}(t)$", fontsize=12)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7, loc='best')

    fig.suptitle(
        f"Latent PDP — top {top_k} dimensions responding to "
        f"{covariate_name}", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────
# Cumulative vs Baseline+Current diagnostic
# ─────────────────────────────────────────────────

def _make_temp_spike(t, q25, q75, peak_frac=0.5):
    """Q25 → Q75 → Q25: triangle spike peaking at peak_frac of follow-up."""
    L = len(t)
    peak_idx = int(L * peak_frac)
    vals = np.zeros(L)
    for i in range(L):
        if i <= peak_idx:
            vals[i] = q25 + (q75 - q25) * (i / peak_idx)
        else:
            vals[i] = q75 - (q75 - q25) * ((i - peak_idx) / (L - 1 - peak_idx))
    return vals


def _make_temp_dip(t, q25, q75, dip_frac=0.5):
    """Q75 → Q25 → Q75: triangle dip."""
    L = len(t)
    dip_idx = int(L * dip_frac)
    vals = np.zeros(L)
    for i in range(L):
        if i <= dip_idx:
            vals[i] = q75 - (q75 - q25) * (i / dip_idx)
        else:
            vals[i] = q25 + (q75 - q25) * ((i - dip_idx) / (L - 1 - dip_idx))
    return vals


def _make_curved_rise(t, q25, q75, curvature='linear'):
    """Q25 → Q75 with different curvatures."""
    t_norm = (t - t[0]) / (t[-1] - t[0])  # [0, 1]
    if curvature == 'linear':
        frac = t_norm
    elif curvature == 'concave':  # late rise
        frac = t_norm ** 2
    elif curvature == 'convex':   # early rise
        frac = 1 - (1 - t_norm) ** 2
    else:
        frac = t_norm
    return q25 + (q75 - q25) * frac


def plot_cumulative_diagnostic(model, loader, device, visit_times,
                                q25, q75, n_tv, covariate_idx=0,
                                covariate_name="BMI",
                                mask_type="binary",
                                max_subjects=None,
                                save_path=None):
    """
    Cumulative vs Baseline+Current diagnostic (3×2 figure).

    Column 1: Constant low vs Temp spike (same start=Q25, same end=Q25)
    Column 2: Constant high vs Temp dip (same start=Q75, same end=Q75)
    Column 3: Linear vs Concave vs Convex rise (all Q25→Q75)

    If Δ ≈ 0 at endpoint → model uses only (baseline, current)
    If Δ > 0 at endpoint → model learned cumulative/path-dependent effect
    """
    from ODE_diagnostic_real import extract_zt_profiles
    from PDP_continuous_time import make_profiles_continuous

    t = np.array(visit_times)
    L = len(t)

    # --- Define special profiles ---
    constant_low = np.full(L, q25)
    constant_high = np.full(L, q75)
    temp_spike = _make_temp_spike(t, q25, q75)
    temp_dip = _make_temp_dip(t, q25, q75)
    linear_rise = _make_curved_rise(t, q25, q75, 'linear')
    concave_rise = _make_curved_rise(t, q25, q75, 'concave')
    convex_rise = _make_curved_rise(t, q25, q75, 'convex')

    all_profiles = {
        'constant_low': constant_low,
        'temp_spike': temp_spike,
        'constant_high': constant_high,
        'temp_dip': temp_dip,
        'linear_rise': linear_rise,
        'concave_rise': concave_rise,
        'convex_rise': convex_rise,
    }

    # --- Extract PDP for each profile ---
    zt_dict, pdp_dict, _ = extract_zt_profiles(
        model, loader, device,
        visit_times=visit_times,
        q25=q25, q75=q75,
        n_tv=n_tv,
        covariate_idx=covariate_idx,
        max_subjects=max_subjects,
    )

    # Override with our custom profiles — re-extract
    from PDP_continuous_time import (
        resample_xaug_to_grid,
        build_profile_xaug_continuous,
    )

    model.eval()
    mu_results = {name: [] for name in all_profiles}
    n_total = 0

    with torch.no_grad():
        for batch in loader:
            pids, x_aug, y_pad, target_mask, static = batch
            x_aug = x_aug.to(device)
            target_mask = target_mask.to(device)
            static = static.to(device)
            B = x_aug.shape[0]

            if max_subjects is not None and n_total >= max_subjects:
                break

            grid = np.array(visit_times, dtype=np.float64)
            K = n_tv
            x_aug_grid, obs_mask_grid = resample_xaug_to_grid(
                x_aug, target_mask, grid, K)

            for name, prof_vals in all_profiles.items():
                x_cf = build_profile_xaug_continuous(
                    x_aug_grid, target_col=covariate_idx,
                    profile_values=prof_vals, n_tv=K,
                    mask_type=mask_type)

                t_pad = x_cf[:, :, 0]
                x_interp = x_cf[:, :, 1:1+K]
                mask = x_cf[:, :, 1+K:1+2*K]

                t0 = t_pad[:, 0:1]
                x_baseline = x_interp[:, 0]
                enc_in = torch.cat([t0, x_baseline, static], dim=-1)
                z0 = model.encoder(enc_in)
                zt = model._integrate(z0, t_pad, x_interp, mask)
                zt = model.z_norm(zt)
                mu, V, Z, D, sig2, reg = model.decoder(
                    zt, x_interp, mask, static, obs_mask=obs_mask_grid)
                mu_results[name].append(mu.cpu().numpy())

            n_total += B

    pdp_custom = {name: np.concatenate(arrs, axis=0).mean(axis=0)
                  for name, arrs in mu_results.items()}

    # --- Plot 3×2 figure ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Column 1: Constant low vs Temp spike
    ax = axes[0, 0]
    ax.plot(t, constant_low, 'b-', linewidth=2, label='Constant low')
    ax.plot(t, temp_spike, 'r--', linewidth=2,
            label=f'Temp spike (Q25→Q75→Q25)')
    ax.set_ylabel(covariate_name)
    ax.set_title(f"start={q25:.1f}, end={q25:.1f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    delta_1 = pdp_custom['temp_spike'][-1] - pdp_custom['constant_low'][-1]
    ax.plot(t, pdp_custom['constant_low'], 'b-o', linewidth=2,
            markersize=4, label='Constant low')
    ax.plot(t, pdp_custom['temp_spike'], 'r--o', linewidth=2,
            markersize=4, label='Temp spike')
    ax.set_ylabel("E[IST]")
    ax.set_xlabel("Time (years)")
    ax.set_title(f"E[IST]  |  Δ at t={t[-1]:.0f}: {delta_1:.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Column 2: Constant high vs Temp dip
    ax = axes[0, 1]
    ax.plot(t, constant_high, 'r-', linewidth=2, label='Constant high')
    ax.plot(t, temp_dip, 'b--', linewidth=2,
            label=f'Temp dip (Q75→Q25→Q75)')
    ax.set_title(f"start={q75:.1f}, end={q75:.1f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    delta_2 = pdp_custom['temp_dip'][-1] - pdp_custom['constant_high'][-1]
    ax.plot(t, pdp_custom['constant_high'], 'r-o', linewidth=2,
            markersize=4, label='Constant high')
    ax.plot(t, pdp_custom['temp_dip'], 'b--o', linewidth=2,
            markersize=4, label='Temp dip')
    ax.set_xlabel("Time (years)")
    ax.set_title(f"E[IST]  |  Δ at t={t[-1]:.0f}: {delta_2:.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Column 3: Linear vs Concave vs Convex rise
    ax = axes[0, 2]
    ax.plot(t, linear_rise, 'g-', linewidth=2, label='Linear')
    ax.plot(t, concave_rise, 'm--', linewidth=2, label='Concave (late rise)')
    ax.plot(t, convex_rise, 'c:', linewidth=2, label='Convex (early rise)')
    ax.set_title(f"start={q25:.1f}, end={q75:.1f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    delta_3 = (pdp_custom['convex_rise'][-1]
               - pdp_custom['concave_rise'][-1])
    ax.plot(t, pdp_custom['linear_rise'], 'g-o', linewidth=2,
            markersize=4, label='Linear')
    ax.plot(t, pdp_custom['concave_rise'], 'm--o', linewidth=2,
            markersize=4, label='Concave (late)')
    ax.plot(t, pdp_custom['convex_rise'], 'c:o', linewidth=2,
            markersize=4, label='Convex (early)')
    ax.set_xlabel("Time (years)")
    ax.set_title(
        f"E[IST]  |  Δ(convex−concave) at t={t[-1]:.0f}: {delta_3:.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Cumulative vs Baseline+Current Diagnostic — {covariate_name}\n"
        f"If Δ ≈ 0 at endpoint: model uses only (baseline, current), "
        f"not full trajectory",
        fontsize=12, y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default="3C_dataset/train_3C_data_1.csv")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_model_ode_real3C_practice_group_lasso_seed3000.pt")
    parser.add_argument("--covariate", type=str, default="BMI",
                        choices=list(COVARIATE_CONFIG.keys()))
    parser.add_argument("--q25", type=float, default=None)
    parser.add_argument("--q75", type=float, default=None)
    parser.add_argument("--max_subjects", type=int, default=4687)
    parser.add_argument("--save_dir", type=str, default="diagnostics")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Covariate settings ---
    cov_cfg = COVARIATE_CONFIG[args.covariate]
    col_idx = cov_cfg["col_idx"]
    q25 = args.q25 if args.q25 is not None else cov_cfg["q25"]
    q75 = args.q75 if args.q75 is not None else cov_cfg["q75"]

    print(f"Covariate: {args.covariate} (col_idx={col_idx})")
    print(f"  Q25={q25}, Q75={q75}")

    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    ckpt_cfg = checkpoint['config']
    print(f"Checkpoint: {args.checkpoint}")

    # ── Feature definitions ─────────────────────────────────────────────
    id_col = "NUM_ID"
    target_col = "ISA15"
    time_varying_features = ckpt_cfg.get('time_varying_features',
                                         ["BMI", "PAS", "PAD", "GLUC", "HDL"])
    static_features = ckpt_cfg.get('static_features',
                                    ["SEX_code", "AGEc", "DIPNIV_2", "DIPNIV_3"])
    K = len(time_varying_features)
    Ks = len(static_features)
    interp_method = ckpt_cfg.get('interp_method', 'linear')
    mask_type = ckpt_cfg.get('mask_type', 'binary')
    cov_means = checkpoint['cov_means']
    cov_stds = checkpoint['cov_stds']

    print(f"  Covariates: {time_varying_features}")
    print(f"  Statics:    {static_features}")

    # ── Load and preprocess ─────────────────────────────────────────────
    df = pd.read_csv(args.data)
    if "AGEc" not in df.columns:
        all_df = pd.read_csv("3C_dataset/data_3C.csv")
        baseline_age = all_df.groupby(id_col)["AGE0"].transform("first")
        baseline_age_mean = baseline_age.mean()
        df["AGEc"] = (df.groupby(id_col)["AGE0"].transform("first")
                       - baseline_age_mean)

    patient_data = process_data(
        df=df, id_col=id_col,
        time_varying_features=time_varying_features,
        static_features=static_features,
        target_col=target_col,
        interp_method=interp_method,
        mask_type=mask_type,
    )
    print(f"  Preprocessed {len(patient_data)} patients")

    dataset = RealDataset(patient_data)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_real)

    # ── Rebuild model ───────────────────────────────────────────────────
    cfg = NeuralODEConfig(
        hidden_channels=ckpt_cfg['hidden_channels'],
        enc_mlp_hidden=ckpt_cfg.get('enc_mlp_hidden', 16),
        func_mlp_hidden=ckpt_cfg.get('func_mlp_hidden', 16),
        dec_rho_hidden=ckpt_cfg.get('dec_rho_hidden', 16),
        dec_p=ckpt_cfg.get('dec_p', 4),
        dec_q=ckpt_cfg.get('dec_q', 3),
        depth=ckpt_cfg.get('depth', 2),
        dropout=0.0,
        euler_steps_per_interval=ckpt_cfg.get('euler_steps', 4),
        ode_solver=ckpt_cfg.get('ode_solver', 'rk4'),
        use_rho_norm=ckpt_cfg.get('use_rho_norm', True)
    )

    model = NeuralODEModel(
        n_tv=K, static_dim=Ks, cfg=cfg,
        use_rho_net=True, use_neural_re=True,
        g_hidden=8, fullD=False,
        cov_means=cov_means, cov_stds=cov_stds,
        use_dynamic_skip=True,
        static_skip_dims=list(range(Ks)),
        reg_mode=ckpt_cfg.get('reg_mode', None),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    # --- Visit times ---
    visit_times = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # ─────────────────────────────────────────────────
    # Step 1: Extract z(t) under counterfactual profiles
    # ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"EXTRACTING z(t) UNDER COUNTERFACTUAL {args.covariate} PROFILES")
    print("=" * 60)

    zt_dict, pdp_dict, profiles = extract_zt_profiles(
        model, loader, device,
        visit_times=visit_times,
        q25=q25, q75=q75,
        n_tv=K,
        covariate_idx=col_idx,
        max_subjects=args.max_subjects,
    )

    # ─────────────────────────────────────────────────
    # Step 2: Quantitative diagnostic
    # ─────────────────────────────────────────────────
    print_diagnostic_summary(zt_dict, pdp_dict, visit_times)

    # ─────────────────────────────────────────────────
    # Step 3: Correlation with cumulative integral
    # ─────────────────────────────────────────────────
    cov = args.covariate
    d = zt_dict[list(zt_dict.keys())[0]].shape[2]
    t = np.array(visit_times)

    print("\n" + "=" * 60)
    print(f"CORRELATION: z_k(t) vs integral({cov}) ACROSS PROFILES")
    print("=" * 60)
    print(f"  High |r| = ODE learned cumulative-like representation")
    print(f"  Low |r|  = ODE uses non-cumulative encoding\n")

    for ell, vt in enumerate(visit_times):
        if vt == 0:
            continue
        cum_vals, z_vals = [], []
        for name, cov_vals in profiles.items():
            cum = np.trapz(cov_vals[:ell + 1], t[:ell + 1])
            cum_vals.append(cum)
            z_vals.append(zt_dict[name][:, ell, :].mean(axis=0))

        cum_arr = np.array(cum_vals)
        z_arr = np.array(z_vals)
        corrs = []
        for k in range(d):
            if z_arr[:, k].std() < 1e-10:
                corrs.append(0.0)
            else:
                corrs.append(np.corrcoef(cum_arr, z_arr[:, k])[0, 1])

        best_k = np.argmax(np.abs(corrs))
        print(f"  t={vt:5.1f}: best dim z_{best_k} "
              f"(r={corrs[best_k]:+.4f}), "
              f"all |r|: {[f'{abs(c):.3f}' for c in corrs]}")

    # ─────────────────────────────────────────────────
    # Step 4: Correlation with current value
    # ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"CORRELATION: z_k(t) vs CURRENT {cov} ACROSS PROFILES")
    print("=" * 60)
    print(f"  High |r| = ODE tracks current value")
    print(f"  Low |r|  = ODE does not track current value\n")

    for ell, vt in enumerate(visit_times):
        current_vals, z_vals = [], []
        for name, cov_vals in profiles.items():
            current_vals.append(cov_vals[ell])
            z_vals.append(zt_dict[name][:, ell, :].mean(axis=0))

        cur_arr = np.array(current_vals)
        z_arr = np.array(z_vals)

        if cur_arr.std() < 1e-10:
            print(f"  t={vt:5.1f}: all profiles same value — skipped")
            continue

        corrs = []
        for k in range(d):
            if z_arr[:, k].std() < 1e-10:
                corrs.append(0.0)
            else:
                corrs.append(np.corrcoef(cur_arr, z_arr[:, k])[0, 1])

        best_k = np.argmax(np.abs(corrs))
        print(f"  t={vt:5.1f}: best dim z_{best_k} "
              f"(r={corrs[best_k]:+.4f}), "
              f"all |r|: {[f'{abs(c):.3f}' for c in corrs]}")

    # ─────────────────────────────────────────────────
    # Step 5: Plots
    # ─────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)

    # Profile shapes (the six intervention paths)
    save0 = os.path.join(args.save_dir, f"profiles_3C_{cov}.png")
    plot_profile_shapes(profiles, visit_times, cov,
                        q25=q25, q75=q75, save_path=save0)

    # Full z(t) diagnostic (all dims + PDP)
    save1 = os.path.join(args.save_dir, f"zt_all_3C_{cov}.png")
    plot_zt_diagnostic(zt_dict, pdp_dict, profiles, visit_times,
                       save_path=save1)

    # Pairwise z(t) comparisons
    save2 = os.path.join(args.save_dir, f"zt_pairs_3C_{cov}.png")
    plot_pairwise_diagnostic(zt_dict, visit_times, save_path=save2)

    # Latent PDP (top-4 responsive dimensions)
    save3 = os.path.join(args.save_dir, f"zt_latent_pdp_3C_{cov}.png")
    plot_latent_pdp(zt_dict, visit_times, cov,
                    top_k=4, save_path=save3)

    # Latent distance (L2 divergence between diagnostic pairs)
    save4 = os.path.join(args.save_dir, f"zt_latent_distance_3C_{cov}.png")
    plot_latent_distance(zt_dict, visit_times, cov,
                         profiles=profiles, q25=q25, q75=q75,
                         save_path=save4)

    # Cumulative vs Baseline+Current diagnostic (3×2 figure)
    save5 = os.path.join(args.save_dir,
                         f"cumulative_diagnostic_3C_{cov}.png")
    plot_cumulative_diagnostic(
        model, loader, device, visit_times,
        q25=q25, q75=q75, n_tv=K, covariate_idx=col_idx,
        covariate_name=cov, mask_type=mask_type,
        max_subjects=args.max_subjects,
        save_path=save5,
    )

    print(f"\nDone. Plots saved to {args.save_dir}/")
    print(f"\nInterpretation guide:")
    print(f"  Latent distance plot:")
    print(f"    • Pair 1 growing → ODE encodes history beyond current value")
    print(f"    • Pairs 2-3 growing → genuine path-dependence")
    print(f"    • Pairs 2-3 flat → ODE only tracks endpoints, no path memory")
    print(f"  Correlations:")
    print(f"    • integral corr >> current corr → cumulative effect")
    print(f"    • current corr >> integral corr → instantaneous effect")


if __name__ == "__main__":
    main()