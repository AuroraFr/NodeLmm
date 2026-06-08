"""
Standalone PDP evaluation script for Neural ODE + BMI Skip model.
Evaluates PDP at anchor times directly (not at observation times).

Usage:
    python PDP_ode.py
    python PDP_ode.py --checkpoint checkpoints/best_model_ode_skip.pt
    python PDP_ode.py --with_blup
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pyreadr
import argparse
from dataset import LongitudinalDataset, collate_pad

from PDP_analysis_ODE import (
    compute_pdp_with_blup,
    plot_pdp_marginal,
    compute_delta_pdp, compute_delta_pdp_stratified,
    compute_true_delta_pdp,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ═════════════════════════════════════════════════════════════════════════
# PDP at anchor times — runs ODE to each anchor time directly
# ═════════════════════════════════════════════════════════════════════════

def compute_pdp_at_anchors(model, dataset, device, bmi_values, anchor_times,
                            collate_fn, batch_size=128, n_tv=1):
    """
    Compute PDP at fixed anchor times by running the ODE to those times.

    For each subject and each BMI value v:
      - Construct synthetic input with t = anchor_times, BMI = v
      - Run the model forward
      - Extract population-mean prediction mu(t)

    Returns:
        results: {bmi_val: (N, L) array}  — predictions at anchor times
        ages:    (N,) array of AGEc values
    """
    from torch.utils.data import DataLoader

    # Collect all subjects
    all_statics = []
    all_first_x = []  # first observation for encoder

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn)

    for batch in loader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        all_statics.append(s.squeeze(0))
        # First observed time's covariates (for encoder context)
        m = mask.squeeze(0).bool()
        first_idx = m.nonzero(as_tuple=True)[0][0] if m.any() else 0
        all_first_x.append(x_pad.squeeze(0)[first_idx])

    N = len(all_statics)
    L = len(anchor_times)
    ages = np.array([s[1].item() for s in all_statics])  # AGEc is index 1

    # Build anchor time tensor
    t_anchor = torch.tensor(anchor_times, dtype=torch.float32)  # (L,)

    results = {}
    for v in bmi_values:
        preds = np.zeros((N, L))

        # Process in batches
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B = end - start

            # t_pad: (B, L)
            t_batch = t_anchor.unsqueeze(0).expand(B, -1).to(device)

            # x_pad: (B, L, x_dim) — BMI = v, rs1 = rs2 = 0
            x_dim = all_first_x[0].shape[0]
            x_batch = torch.zeros(B, L, x_dim, device=device)
            x_batch[:, :, 0] = v  # BMI = intervention value
            # rs1, rs2 = 0 (only affects RE basis, not population mean)

            # Static covariates
            s_batch = torch.stack(all_statics[start:end]).to(device)

            # mask: all observed
            mask_batch = torch.ones(B, L, device=device)

            # bmi_t for skip connection
            bmi_t = x_batch[:, :, 0:1]

            with torch.no_grad():
                mu, V, _, _, _, _ = model(
                    t_batch, x_batch, masks=None,
                    static_covariates=s_batch,
                    bmi_t=bmi_t, obs_mask=mask_batch
                )

            preds[start:end] = mu.squeeze(-1).cpu().numpy()

        results[v] = preds
        print(f"    BMI={v}: mean PDP = "
              f"[{', '.join(f'{preds.mean(axis=0)[ell]:.2f}' for ell in range(L))}]")

    return results, ages


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDP analysis for Neural ODE model")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/simulation_baseline_skip_noreg_norhonorm_diagoD/best_model_ode_0.pt")
    parser.add_argument("--data", type=str,
                        default="simu_datasets/S2a_sims/sim_001.rds")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--prefix", type=str, default="figures/baseline_pdp_ode")
    parser.add_argument("--true_beta_bmi", type=float, default=-0.30)
    parser.add_argument("--true_beta_int", type=float, default=-0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ────────────────────────────────────────────────────────────
    time_col = "time"
    y_col = "ISA15_sim"
    id_col = "NUM_ID"
    x_cols = ["BMI_t", "rs1", "rs2"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    df = next(iter(pyreadr.read_r(args.data).values()))
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col,
                                  static_cols=static_cols)

    n_tv = 1

    # ── Build model ─────────────────────────────────────────────────────
    from model_ODE_skipgate import NeuralODEModel, NeuralODEConfig

    cfg = NeuralODEConfig(
        hidden_channels=8,
        enc_mlp_hidden=32,
        func_mlp_hidden=32,
        dec_rho_hidden=16,
        dec_p=4,
        dec_q=3,
        depth=2,
        dropout=0.0,
        euler_steps_per_interval=4,
    )

    model = NeuralODEModel(
        x_dim=len(x_cols),
        static_dim=len(static_cols),
        cfg=cfg,
        n_tv=n_tv,
        use_rho_net=True,
        use_neural_re=True,
        re_spline_cols=None,
        g_hidden=16,
        fullD=False,
        bmi_mean=0.0,
        bmi_std=1.0,
        use_bmi_skip=True,
        static_skip_dims=[1],
        reg_mode=None
    ).to(device)

    # ── Load checkpoint ─────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"  epoch = {checkpoint.get('epoch', '?')}")
        loss_val = checkpoint.get('best_test_loss', None)
        if loss_val is not None:
            print(f"  loss  = {loss_val:.4f}")
    else:
        model.load_state_dict(checkpoint, strict=False)

    print(f"  bmi_mean = {model.decoder.bmi_mean.item():.4f}")
    print(f"  bmi_std  = {model.decoder.bmi_std.item():.4f}")

    # ── Print model parameters ──────────────────────────────────────────
    model.eval()
    print(f"\n{'='*60}")
    print(f"MODEL PARAMETERS (Neural ODE + BMI Skip)")
    print(f"{'='*60}")

    sig2 = torch.exp(model.decoder.log_residual_var).item()
    print(f"  sigma2 = {sig2:.6f}")

    bn = model.decoder.beta_neural.detach()
    print(f"  beta_neural = {bn.cpu().tolist()}")

    if model.decoder.L_unconstrained is not None:
        D = model.decoder._build_D(device=torch.device('cpu'), dtype=torch.float32)
        print(f"  D diag = {D.diag().tolist()}")

    if model.decoder.skip_gate_logit is not None:
        gates = torch.sigmoid(model.decoder.skip_gate_logit).detach().cpu()
        gate_names = ['BMI', 'AGEc'] if gates.numel() > 1 else ['AGEc']
        print(f"  Skip gates:")
        for g, name in enumerate(gate_names):
            print(f"    {name:>8s}: {gates[g]:.4f}")

    # ── True parameters ─────────────────────────────────────────────────
    TRUE_BETA_BMI = args.true_beta_bmi
    TRUE_BETA_INT = args.true_beta_int
    print(f"\nTrue beta: BMI = {TRUE_BETA_BMI}, BMI*AGEc = {TRUE_BETA_INT}")

    # ── PDP at anchor times ─────────────────────────────────────────────
    bmi_values = [20, 23, 26, 29, 32, 35]
    anchor_times = [0.0, 5.0, 10.0, 15.0]

    print(f"\n{'='*60}")
    print(f"PDP ANALYSIS (constant BMI, evaluated at anchor times)")
    print(f"  BMI values   = {bmi_values}")
    print(f"  Anchor times = {anchor_times}")
    print(f"{'='*60}")

    results, ages = compute_pdp_at_anchors(
        model, dataset, device, bmi_values, anchor_times,
        collate_fn=collate_pad, batch_size=args.batch_size, n_tv=n_tv,
    )

    anchor_arr = np.array(anchor_times)
    L = len(anchor_arr)

    # ── Marginal PDP: truth=lines, estimated=dots, ICE=grey ─────────────
    mean_age = ages.mean()
    mean_bmi = np.mean(bmi_values)
    slope = TRUE_BETA_BMI + TRUE_BETA_INT * mean_age
    delta_bmi = bmi_values[-1] - bmi_values[0]

    # Estimated PDP means at anchor times
    est_pdp = {v: results[v].mean(axis=0) for v in bmi_values}
    mean_pdp = np.mean([est_pdp[v] for v in bmi_values], axis=0)

    cmap = plt.cm.RdYlBu_r
    colors_pdp = [cmap(i / (len(bmi_values) - 1)) for i in range(len(bmi_values))]
    t_fine = np.linspace(0, max(anchor_times), 200)

    fig, ax = plt.subplots(figsize=(10, 7))

    # ICE curves (grey)
    n_ice = min(200, results[bmi_values[0]].shape[0])
    rng = np.random.RandomState(42)
    ice_idx = rng.choice(results[bmi_values[0]].shape[0], n_ice, replace=False)
    for v in bmi_values:
        for i in ice_idx:
            ax.plot(anchor_arr, results[v][i], color='grey', alpha=0.05,
                    linewidth=0.5)

    # Oracle: coloured lines
    for j, v in enumerate(bmi_values):
        true_at_anchor = mean_pdp + slope * (v - mean_bmi)
        true_fine = np.interp(t_fine, anchor_arr, true_at_anchor)
        ax.plot(t_fine, true_fine, '-', color=colors_pdp[j],
                linewidth=2.5, alpha=0.85)

    # Estimated: coloured dots
    for j, v in enumerate(bmi_values):
        ax.scatter(anchor_arr, est_pdp[v], color=colors_pdp[j],
                   s=90, zorder=4, edgecolors='black', linewidth=0.7,
                   label=f'BMI={v}')

    ax.set_xlabel('Time (years)', fontsize=13)
    ax.set_ylabel('Predicted IST', fontsize=13)
    ax.legend(fontsize=10, title='Estimated (dots) / Oracle (lines)',
              loc='lower left')
    ax.set_title('Marginal PDP of BMI on IST + ICE (grey)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{args.prefix}_marginal.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {args.prefix}_marginal.png")

    # ── Stratified PDP: truth=lines, estimated=dots ─────────────────────
    age_terciles = np.percentile(ages, [33, 67])
    age_groups = [
        ('Young', -np.inf, age_terciles[0]),
        ('Middle', age_terciles[0], age_terciles[1]),
        ('Old', age_terciles[1], np.inf),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for g, (label, a_lo, a_hi) in enumerate(age_groups):
        ax = axes[g]
        gmask = (ages >= a_lo) & (ages < a_hi)
        if gmask.sum() == 0:
            continue
        mean_age_g = ages[gmask].mean()
        slope_g = TRUE_BETA_BMI + TRUE_BETA_INT * mean_age_g

        est_g = {v: results[v][gmask].mean(axis=0) for v in bmi_values}
        mean_pdp_g = np.mean([est_g[v] for v in bmi_values], axis=0)

        for j, v in enumerate(bmi_values):
            true_at_anchor = mean_pdp_g + slope_g * (v - mean_bmi)
            true_fine = np.interp(t_fine, anchor_arr, true_at_anchor)
            ax.plot(t_fine, true_fine, '-', color=colors_pdp[j],
                    linewidth=2.5, alpha=0.85)

        for j, v in enumerate(bmi_values):
            ax.scatter(anchor_arr, est_g[v], color=colors_pdp[j],
                       s=90, zorder=4, edgecolors='black', linewidth=0.7,
                       label=f'BMI={v}')

        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_title(f'{label} (AGEc={mean_age_g:.1f}, n={gmask.sum()})',
                     fontsize=12)
        if g == 0:
            ax.set_ylabel('Predicted IST', fontsize=12)
        if g == 2:
            ax.legend(fontsize=8, loc='upper right',
                      title='Est. (dots) / Oracle (lines)')

    fig.suptitle('PDP of BMI by age tercile — Oracle vs Estimated', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{args.prefix}_by_age.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {args.prefix}_by_age.png")

    # ── ΔPDP: marginal ──────────────────────────────────────────────────
    bmi_lo, bmi_hi = bmi_values[0], bmi_values[-1]
    delta_pdp_est = results[bmi_hi].mean(axis=0) - results[bmi_lo].mean(axis=0)
    delta_pdp_true = (TRUE_BETA_BMI + TRUE_BETA_INT * mean_age) * (bmi_hi - bmi_lo)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_fine, np.full_like(t_fine, delta_pdp_true), '-',
            color='#2166AC', linewidth=2.5,
            label=f'Oracle ΔPDP = {delta_pdp_true:.2f}')
    ax.scatter(anchor_arr, delta_pdp_est, color='#B2182B', s=90, zorder=3,
               edgecolors='black', linewidth=0.8, label='Estimated ΔPDP')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (years)', fontsize=13)
    ax.set_ylabel(r'$\Delta$PDP (difference in E[IST])', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_title(f'Marginal ΔPDP: BMI = {bmi_hi} vs {bmi_lo}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{args.prefix}_dpdp_marginal.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {args.prefix}_dpdp_marginal.png")

    # ── ΔPDP: stratified by age ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    age_colors = ['#4393C3', '#808080', '#D6604D']
    for g, (label, a_lo, a_hi) in enumerate(age_groups):
        gmask = (ages >= a_lo) & (ages < a_hi)
        if gmask.sum() == 0:
            continue
        mean_age_g = ages[gmask].mean()
        true_val = (TRUE_BETA_BMI + TRUE_BETA_INT * mean_age_g) * (bmi_hi - bmi_lo)
        est_val = results[bmi_hi][gmask].mean(axis=0) - results[bmi_lo][gmask].mean(axis=0)

        ax.plot(t_fine, np.full_like(t_fine, true_val), '-',
                color=age_colors[g], linewidth=2.5, alpha=0.8,
                label=f'{label} oracle ({true_val:.2f})')
        ax.scatter(anchor_arr, est_val, color=age_colors[g],
                   s=90, zorder=3, edgecolors='black', linewidth=0.8,
                   label=f'{label} estimated')

    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (years)', fontsize=13)
    ax.set_ylabel(r'$\Delta$PDP (difference in E[IST])', fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.set_title(f'ΔPDP by age tercile: BMI = {bmi_hi} vs {bmi_lo}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{args.prefix}_dpdp_stratified.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {args.prefix}_dpdp_stratified.png")

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  True: beta_BMI={TRUE_BETA_BMI}, beta_int={TRUE_BETA_INT}")
    print(f"  True marginal ΔPDP = {delta_pdp_true:.4f}")
    print(f"\n  {'Time':>6s}  {'Estimated':>10s}  {'True':>10s}  {'Bias':>10s}")
    print(f"  {'-'*42}")
    for ell, t in enumerate(anchor_arr):
        est = delta_pdp_est[ell]
        print(f"  {t:6.0f}  {est:+10.4f}  {delta_pdp_true:+10.4f}  "
              f"{est - delta_pdp_true:+10.4f}")

    print(f"\nDone.")