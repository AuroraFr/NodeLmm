"""
PDP evaluation for Neural ODE-LMM on the real 3C cohort.

Usage:
    python PDP_ode_real.py
    python PDP_ode_real.py --checkpoint checkpoints/best_model_ode_real_3C.pt
    python PDP_ode_real.py --with_blup
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import os

from Preprocess_3C import process_data, EXPECTED_TIMES
from train_ODE_real import RealDataset, collate_real
from model_ODE_real import NeuralODEModel, NeuralODEConfig
from PDP_analysis_ODE_real import (
    compute_pdp, compute_pdp_with_blup,
    plot_pdp, plot_pdp_marginal, plot_delta_pdp,
    compute_delta_pdp, compute_delta_pdp_stratified,
    make_profiles, compute_trajectory_profile_pdp,
    plot_trajectory_profile_pdp,
    VISIT_TIMES_3C,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDP analysis — real 3C dataset")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_model_ode_real_3C_sepreg.pt")
    parser.add_argument("--data", type=str,
                        default="3C_dataset/train_3C_data_1.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--mode", type=str, default="constant",
                        choices=["constant", "linear", "shifted"],
                        help="Counterfactual mode for PDP interventions")
    parser.add_argument("--slope", type=float, default=None,
                        help="Slope for linear mode (auto-estimated if omitted)")
    parser.add_argument("--prefix", type=str, default="figures/pdp_real_sepreg")
    parser.add_argument("--with_blup", action="store_true",
                        help="Include BLUP random effects in ICE curves")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ─────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    ckpt_cfg = checkpoint['config']

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  epoch     = {checkpoint.get('epoch', '?')}")
    print(f"  test loss = {checkpoint.get('best_test_loss', '?'):.4f}")
    print(ckpt_cfg)

    # ── Feature definitions from checkpoint ─────────────────────────────
    id_col = "NUM_ID"
    target_col = "ISA15"
    time_varying_features = ckpt_cfg['time_varying_features']
    static_features = ckpt_cfg['static_features']
    K = len(time_varying_features)
    Ks = len(static_features)
    interp_method = ckpt_cfg.get('interp_method', 'linear')
    mask_type = ckpt_cfg.get('mask_type', 'binary')

    cov_means = checkpoint['cov_means']
    cov_stds = checkpoint['cov_stds']

    print(f"  Covariates: {time_varying_features}")
    print(f"  Statics:    {static_features}")
    print(f"  Interp:     {interp_method}, mask: {mask_type}")
    for k, feat in enumerate(time_varying_features):
        print(f"    {feat:>6s}: mean={cov_means[k]:.4f}, std={cov_stds[k]:.4f}")

    # ── Load and preprocess ─────────────────────────────────────────────
    print(f"\nLoading 3C dataset...")
    df = pd.read_csv(args.data)

    if "AGEc" not in df.columns:
        all_df = pd.read_csv("3C_dataset/data_3C.csv")
        baseline_age = all_df.groupby(id_col)["AGE0"].transform("first")
        baseline_age_mean = baseline_age.mean()
        df["AGEc"] = df.groupby(id_col)["AGE0"].transform("first") - baseline_age_mean

    patient_data = process_data(
        df=df,
        id_col=id_col,
        time_varying_features=time_varying_features,
        static_features=static_features,
        target_col=target_col,
        interp_method=interp_method,
        mask_type=mask_type,
    )
    print(f"  Preprocessed {len(patient_data)} patients")

    # ── Dataset and loader ──────────────────────────────────────────────
    dataset = RealDataset(patient_data)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_real)

    # ── Rebuild model from checkpoint ───────────────────────────────────
    cfg = NeuralODEConfig(
        hidden_channels=ckpt_cfg['hidden_channels'],
        enc_mlp_hidden=ckpt_cfg.get('enc_mlp_hidden', 16),
        func_mlp_hidden=ckpt_cfg.get('func_mlp_hidden', 32),
        dec_rho_hidden=ckpt_cfg.get('dec_rho_hidden', 16),
        dec_p=ckpt_cfg.get('dec_p', 4),
        dec_q=ckpt_cfg.get('dec_q', 3),
        depth=ckpt_cfg.get('depth', 2),
        dropout=0.0,
        euler_steps_per_interval=ckpt_cfg['euler_steps'],
        ode_solver=ckpt_cfg.get('ode_solver', 'euler'),
        use_rho_norm=ckpt_cfg.get('use_rho_norm', True),
    )

    model = NeuralODEModel(
        n_tv=K,
        static_dim=Ks,
        cfg=cfg,
        use_rho_net=True,
        use_neural_re=True,
        g_hidden=16,
        fullD=False,
        cov_means=cov_means,
        cov_stds=cov_stds,
        use_dynamic_skip=ckpt_cfg.get('use_dynamic_skip', True),
        static_skip_dims=ckpt_cfg.get('static_skip_dims', list(range(Ks))),
        reg_mode=ckpt_cfg.get('reg_mode', None),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    # ── Print model info ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"MODEL PARAMETERS")
    print(f"{'='*60}")

    sig2 = torch.exp(model.decoder.log_residual_var).item()
    D = model.decoder._build_D(device, torch.float32).detach().cpu()
    beta = model.decoder.beta_neural.detach().cpu()

    print(f"  σ² = {sig2:.6f}")
    print(f"  β  = {beta.tolist()}")
    print(f"  D diag = {D.diag().tolist()}")

    if model.decoder.skip_gate_logit is not None:
        gates = torch.sigmoid(model.decoder.skip_gate_logit).detach().cpu()
        names = time_varying_features + static_features
        print(f"  Skip gates:")
        for g, name in enumerate(names):
            print(f"    {name:>10s}: {gates[g]:.4f}")

    if hasattr(model.decoder, '_group_col_indices') and \
       model.decoder.reg_mode == "group_lasso":
        W = model.decoder.rho_net.net[0].weight.detach().cpu()
        offset = model.decoder.latent_dim
        names = time_varying_features + static_features
        print(f"  Group lasso norms:")
        for g, cols in enumerate(model.decoder._group_col_indices):
            W_cols = W[:, [offset + c for c in cols]]
            print(f"    {names[g]:>10s}: {W_cols.norm(p='fro').item():.4f}")

    # ── PDP Analysis ────────────────────────────────────────────────────
    visit_times = VISIT_TIMES_3C

    # ── Intervention ranges ─────────────────────────────────────────────
    # BMI: data-driven Q25 / Q75  (same as R)
    # Other covariates: hardcoded clinical ranges (same in R and Python)

    bmi_vals = df["BMI"].dropna().values
    bmi_q25 = float(np.percentile(bmi_vals, 25))
    bmi_q75 = float(np.percentile(bmi_vals, 75))
    bmi_q05 = float(np.percentile(bmi_vals, 5))
    bmi_q95 = float(np.percentile(bmi_vals, 95))

    print(f"  BMI quantiles: Q05={bmi_q05:.2f}  Q25={bmi_q25:.2f}  "
          f"Q75={bmi_q75:.2f}  Q95={bmi_q95:.2f}")

    # Intervention grids
    INTERVENTION_GRIDS = {
        "BMI":  sorted(set(
                    list(np.linspace(bmi_q05, bmi_q95, 6)) +
                    [bmi_q25, bmi_q75]
                )),
        "PAS":  [110, 120, 130, 140, 150, 160],
        "PAD":  [60, 65, 70, 75, 80, 85],
        "GLUC": [4, 5, 6, 7, 8, 10],
        "HDL":  [0.8, 1.0, 1.2, 1.5, 1.8, 2.2],
    }

    # ΔPDP and trajectory-profile ranges
    DELTA_RANGES = {
        "BMI":  (bmi_q25, bmi_q75),    # data-driven
        "PAS":  (120, 150),             # hardcoded clinical range
        "PAD":  (65, 80),               # hardcoded clinical range
        "GLUC": (4, 10),              # hardcoded clinical range
        "HDL":  (0.8, 2.2),            # hardcoded clinical range
    }

    print(f"\nIntervention ranges:")
    for feat in time_varying_features:
        v_lo, v_hi = DELTA_RANGES[feat]
        src = "Q25/Q75" if feat == "BMI" else "hardcoded"
        print(f"  {feat:>6s}: {v_lo:.2f} → {v_hi:.2f}  ({src})")

    os.makedirs(os.path.dirname(args.prefix) or ".", exist_ok=True)
    suffix = f"_{args.mode}" if args.mode != "constant" else ""

    all_summaries = {}

    for col_idx, feat_name in enumerate(time_varying_features):
        print(f"\n{'='*60}")
        print(f"PDP ANALYSIS: {feat_name} (col={col_idx}, mode={args.mode})")
        print(f"{'='*60}")

        values = INTERVENTION_GRIDS.get(feat_name)
        if values is None:
            print(f"  Skipping {feat_name}: no intervention grid defined")
            continue

        val_lo, val_hi = DELTA_RANGES[feat_name]
        print(f"  Intervention values = {values}")
        print(f"  ΔPDP range: {val_lo} → {val_hi}")

        if args.with_blup:
            results_pop, results_subj, blup, ages, masks, times = \
                compute_pdp_with_blup(
                    model, eval_loader, device, values,
                    target_col=col_idx, n_tv=K,
                    mask_type=mask_type,
                    mode=args.mode, slope=args.slope,
                    target_name=feat_name,
                )
            results = results_pop

            plot_pdp_marginal(
                results, masks, times, values,
                save_path=f"{args.prefix}_{feat_name}_marginal_blup{suffix}.png",
                visit_times=visit_times,
                ice_results=results_subj, ice_n=100,
                target_name=feat_name,
            )
        else:
            results, ages, masks, times = compute_pdp(
                model, eval_loader, device, values,
                target_col=col_idx, n_tv=K,
                mask_type=mask_type,
                mode=args.mode, slope=args.slope,
                target_name=feat_name,
            )

            plot_pdp_marginal(
                results, masks, times, values,
                save_path=f"{args.prefix}_{feat_name}_marginal{suffix}.png",
                visit_times=visit_times,
                target_name=feat_name,
            )

        # Age-stratified PDP
        plot_pdp(results, ages, masks, times, values,
                 save_path=f"{args.prefix}_{feat_name}_by_age{suffix}.png",
                 visit_times=visit_times, target_name=feat_name)

        # ΔPDP
        estimated = compute_delta_pdp(
            results, ages, masks, times,
            val_lo=val_lo, val_hi=val_hi,
            visit_times=visit_times, target_name=feat_name,
        )

        # ΔPDP stratified by age
        summary = compute_delta_pdp_stratified(
            results, ages, masks, times,
            val_lo=val_lo, val_hi=val_hi,
            visit_times=visit_times, target_name=feat_name,
        )

        # ΔPDP plot
        plot_delta_pdp(results, masks, times,
                       val_lo=val_lo, val_hi=val_hi,
                       save_path=f"{args.prefix}_{feat_name}_delta{suffix}.png",
                       visit_times=visit_times, target_name=feat_name)

        # Trajectory-profile PDP (path-dependence diagnostic)
        profiles = make_profiles(visit_times=VISIT_TIMES_3C,
                                 v_lo=val_lo, v_hi=val_hi)
        traj_results, traj_masks, traj_times = compute_trajectory_profile_pdp(
            model, eval_loader, device, profiles,
            target_col=col_idx, n_tv=K,
            mask_type=mask_type,
            target_name=feat_name,
        )

        plot_trajectory_profile_pdp(
            traj_results, traj_masks, traj_times,
            save_path=f"{args.prefix}_{feat_name}_traj_profile{suffix}.png",
            visit_times=visit_times, target_name=feat_name,
        )

        all_summaries[feat_name] = {
            "delta_pdp": estimated,
            "stratified": summary,
            "val_lo": val_lo, "val_hi": val_hi,
        }

    # ── Global summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY — ALL COVARIATES")
    print(f"{'='*60}")
    print(f"  Model: Neural ODE-LMM (3C cohort)")
    print(f"  Mode:  {args.mode}")
    print(f"  Reg:   {ckpt_cfg.get('reg_mode', 'None')}")

    print(f"\n  {'Covariate':<10s} {'Range':<15s} {'avg ΔPDP':>10s} {'per-unit':>10s}")
    print(f"  {'-'*50}")
    for feat_name, info in all_summaries.items():
        est = info["delta_pdp"]
        delta_v = info["val_hi"] - info["val_lo"]
        if est:
            avg = np.mean(list(est.values()))
            print(f"  {feat_name:<10s} {info['val_lo']}→{info['val_hi']:<10} "
                  f"{avg:>+10.4f} {avg/delta_v:>+10.4f}")

    print(f"\nDone.")