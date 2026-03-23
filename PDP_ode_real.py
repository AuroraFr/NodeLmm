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

from Preprocess_3C import process_data
from train_ODE_real import RealDataset, collate_real
from model_ODE_real import NeuralODEModel, NeuralODEConfig
from PDP_analysis_ODE_real import (
    compute_pdp, compute_pdp_with_blup,
    plot_pdp, plot_pdp_marginal, plot_delta_pdp,
    compute_delta_pdp, compute_delta_pdp_stratified,
    VISIT_TIMES_3C,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDP analysis — real 3C dataset")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_model_ode_real_3C.pt")
    parser.add_argument("--data", type=str,
                        default="3C_dataset/train_3C_data.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--mode", type=str, default="constant",
                        choices=["constant", "linear", "shifted"],
                        help="Counterfactual mode for PDP interventions")
    parser.add_argument("--slope", type=float, default=None,
                        help="Slope for linear mode (auto-estimated if omitted)")
    parser.add_argument("--prefix", type=str, default="figures/pdp_real")
    parser.add_argument("--with_blup", action="store_true",
                        help="Include BLUP random effects in ICE curves")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Feature definitions (must match training) ----
    id_col = "NUM_ID"
    target_col = "ISA15"
    time_varying_features = ["BMI", "PAS", "PAD", "GLUC", "HDL"]
    static_features = ["SEX_code", "AGEc", "DIPNIV_2", "DIPNIV_3"]
    K = len(time_varying_features)
    S = len(static_features)

    # ---- Load and preprocess ----
    print("Loading 3C dataset...")
    df = pd.read_csv(args.data)

    if "AGEc" not in df.columns:
        baseline_age = df.groupby(id_col)["AGE0"].transform("first")
        df["AGEc"] = baseline_age - baseline_age.mean()

    patient_data = process_data(
        df=df,
        id_col=id_col,
        time_varying_features=time_varying_features,
        static_features=static_features,
        target_col=target_col,
    )
    print(f"Preprocessed {len(patient_data)} patients")

    # ---- Build dataset and loader ----
    dataset = RealDataset(patient_data, n_tv=K)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_real)

    # ---- Load checkpoint ----
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    ckpt_config = checkpoint.get('config', {})
    ckpt_config = checkpoint.get('config', {})
    x_mean_list = checkpoint.get('x_mean', None)
    x_std_list = checkpoint.get('x_std', None)
    x_mean = torch.tensor(x_mean_list, dtype=torch.float32) if x_mean_list else None
    x_std = torch.tensor(x_std_list, dtype=torch.float32) if x_std_list else None

    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"  epoch     = {checkpoint.get('epoch', '?')}")
    print(f"  test loss = {checkpoint.get('best_test_loss', '?'):.4f}")
    if x_mean is not None:
        feat_names = checkpoint.get('time_varying_features', time_varying_features)
        for k, feat in enumerate(feat_names):
            print(f"  {feat:>6s}: mean={x_mean[k]:.4f}, std={x_std[k]:.4f}")
    print(f"  config    = {ckpt_config}")

    # ---- Build model (must match training config) ----
    n_ode_inject = 2 * K

    cfg = NeuralODEConfig(
        hidden_channels=ckpt_config.get('hidden_channels', 8),
        enc_mlp_hidden=32,
        func_mlp_hidden=32,
        dec_rho_hidden=16,
        dec_p=4,
        dec_q=3,
        depth=2,
        dropout=0.0,
        euler_steps_per_interval=ckpt_config.get('euler_steps', 4),
    )

    model = NeuralODEModel(
        x_dim=K,
        static_dim=S,
        cfg=cfg,
        n_tv=K,
        n_ode_inject=n_ode_inject,
        use_rho_net=ckpt_config.get('use_rho_net', True),
        use_neural_re=ckpt_config.get('use_neural_re', True),
        re_spline_cols=None,
        g_hidden=16,
        fullD=True,
        x_mean=x_mean,
        x_std=x_std,
        skip_covariate_cols=ckpt_config.get('skip_covariate_cols', list(range(K))),
        static_skip_dims=ckpt_config.get('static_skip_dims', list(range(S))),
        reg_mode=None,   # no regularization at inference
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # ---- Print model parameters ----
    print(f"\n{'='*60}")
    print(f"MODEL PARAMETERS")
    print(f"{'='*60}")

    sig2 = torch.exp(model.decoder.log_residual_var).item()
    print(f"  sigma2 = {sig2:.6f}")

    bn = model.decoder.beta_neural.detach()
    print(f"  beta_neural = {bn.cpu().tolist()}")

    if model.decoder.L_unconstrained is not None:
        D = model.decoder._build_D(device=torch.device('cpu'), dtype=torch.float32)
        print(f"  D matrix ({D.shape[0]}x{D.shape[1]}):")
        for i in range(D.shape[0]):
            print(f"    [{', '.join(f'{D[i,j]:+.4f}' for j in range(D.shape[1]))}]")

    # ---- PDP Analysis: loop over all covariates ----
    visit_times = VISIT_TIMES_3C

    # Define intervention grid for each covariate (approximate clinical ranges)
    # These will be refined once you see the actual data distributions
    INTERVENTION_GRIDS = {
        "BMI":  [20, 23, 26, 29, 32, 35],
        "PAS":  [110, 120, 130, 140, 150, 160],
        "PAD":  [60, 65, 70, 75, 80, 85],
        "GLUC": [4, 5, 6, 7, 8, 10],
        "HDL":  [0.8, 1.0, 1.2, 1.5, 1.8, 2.2],
    }

    # ΔPDP lo/hi for each covariate
    DELTA_RANGES = {
        "BMI":  (20, 35),
        "PAS":  (110, 160),
        "PAD":  (60, 85),
        "GLUC": (4, 10),
        "HDL":  (0.8, 2.2),
    }

    import os
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
            results_pop, results_subj, blup, ages, masks, times = compute_pdp_with_blup(
                model, eval_loader, device, values,
                target_col=col_idx, mode=args.mode,
                slope=args.slope, target_name=feat_name,
            )
            results = results_pop

            plot_pdp_marginal(results, masks, times, values,
                              save_path=f"{args.prefix}_{feat_name}_marginal_blup{suffix}.png",
                              visit_times=visit_times,
                              ice_results=results_subj, ice_n=100,
                              target_name=feat_name)
        else:
            results, ages, masks, times = compute_pdp(
                model, eval_loader, device, values,
                target_col=col_idx, mode=args.mode,
                slope=args.slope, target_name=feat_name,
            )

            plot_pdp_marginal(results, masks, times, values,
                              save_path=f"{args.prefix}_{feat_name}_marginal{suffix}.png",
                              visit_times=visit_times,
                              target_name=feat_name)

        # Stratified PDP by age
        plot_pdp(results, ages, masks, times, values,
                 save_path=f"{args.prefix}_{feat_name}_by_age{suffix}.png",
                 visit_times=visit_times, target_name=feat_name)

        # ΔPDP
        estimated = compute_delta_pdp(
            results, ages, masks, times,
            val_lo=val_lo, val_hi=val_hi,
            visit_times=visit_times, target_name=feat_name,
        )

        # Stratified ΔPDP
        summary = compute_delta_pdp_stratified(
            results, ages, masks, times,
            val_lo=val_lo, val_hi=val_hi,
            visit_times=visit_times, target_name=feat_name,
        )

        # ΔPDP plot
        plot_delta_pdp(results, masks, times,
                       val_lo=val_lo, val_hi=val_hi,
                       save_path=f"{args.prefix}_{feat_name}_delta_pdp{suffix}.png",
                       visit_times=visit_times, target_name=feat_name)

        all_summaries[feat_name] = {
            "delta_pdp": estimated,
            "stratified": summary,
            "val_lo": val_lo, "val_hi": val_hi,
        }

    # ---- Global summary ----
    print(f"\n{'='*60}")
    print(f"SUMMARY — ALL COVARIATES")
    print(f"{'='*60}")
    print(f"  Model: Neural ODE-LMM (real 3C)")
    print(f"  Mode: {args.mode}")

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