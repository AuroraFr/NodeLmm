"""
Runner for continuous-time PDP analysis — Neural ODE-LMM on 3C cohort.

Drop-in companion to PDP_ode_real.py, using a regular time grid
instead of canonical visit times.

Usage:
    python PDP_ode_real_continuous.py
    python PDP_ode_real_continuous.py --n_points 50   # finer grid
    python PDP_ode_real_continuous.py --n_points 8    # coarse (compare with visit-time)
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
from PDP_analysis_ODE_real import VISIT_TIMES_3C, make_profiles
from PDP_continuous_time import (
    make_eval_grid,
    make_profiles_continuous,
    compute_trajectory_profile_pdp_continuous,
    compute_pdp_continuous,
    plot_trajectory_profile_pdp_continuous,
    plot_pdp_continuous,
    plot_delta_pdp_continuous,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continuous-time PDP analysis — Neural ODE-LMM on 3C")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_model_ode_real_3C_sepreg.pt")
    parser.add_argument("--data", type=str,
                        default="3C_dataset/train_3C_data_1.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_points", type=int, default=15,
                        help="Number of grid points (default 15 = yearly 0..14)")
    parser.add_argument("--t_max", type=float, default=14.0,
                        help="Maximum time (years)")
    parser.add_argument("--prefix", type=str,
                        default="figures/pdp_real_sepreg_continuous")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ─────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    ckpt_cfg = checkpoint['config']

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  epoch     = {checkpoint.get('epoch', '?')}")
    print(f"  test loss = {checkpoint.get('best_test_loss', '?'):.4f}")

    # ── Feature definitions ─────────────────────────────────────────────
    id_col = "NUM_ID"
    target_col = "ISA15"
    time_varying_features = ckpt_cfg['time_varying_features']
    static_features = ckpt_cfg['static_features']
    K = len(time_varying_features)
    Ks = len(static_features)
    interp_method = ckpt_cfg.get('interp_method', 'ffill')
    mask_type = ckpt_cfg.get('mask_type', 'cumulative')
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
        df["AGEc"] = df.groupby(id_col)["AGE0"].transform("first") - baseline_age_mean

    patient_data = process_data(
        df=df, id_col=id_col,
        time_varying_features=time_varying_features,
        static_features=static_features,
        target_col=target_col,
        interp_method=interp_method,
        mask_type=mask_type,
    )
    print(f"  Preprocessed {len(patient_data)} patients")
    print(ckpt_cfg)

    dataset = RealDataset(patient_data)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_real)

    # ── Rebuild model ───────────────────────────────────────────────────
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
        use_rho_norm=ckpt_cfg.get('use_rho_norm', True)
    )

    model = NeuralODEModel(
        n_tv=K, static_dim=Ks, cfg=cfg,
        use_rho_net=True, use_neural_re=True,
        g_hidden=16, fullD=False,
        cov_means=cov_means, cov_stds=cov_stds,
        use_dynamic_skip=ckpt_cfg.get('use_dynamic_skip', True),
        static_skip_dims=ckpt_cfg.get('static_skip_dims', list(range(Ks))),
        reg_mode=ckpt_cfg.get('reg_mode', None),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    # ── Evaluation grid ─────────────────────────────────────────────────
    eval_grid = make_eval_grid(t_max=args.t_max, n_points=args.n_points)
    print(f"\n{'='*60}")
    print(f"CONTINUOUS-TIME PDP ANALYSIS")
    print(f"  Grid: {args.n_points} points on [0, {args.t_max}]")
    print(f"  Grid points: {eval_grid}")
    print(f"  (cf. canonical visit times: {VISIT_TIMES_3C})")
    print(f"{'='*60}")

    bmi_vals = df["BMI"].dropna().values
    bmi_q25 = float(np.percentile(bmi_vals, 25))
    bmi_q75 = float(np.percentile(bmi_vals, 75))
    bmi_q05 = float(np.percentile(bmi_vals, 5))
    bmi_q95 = float(np.percentile(bmi_vals, 95))

    print(f"  BMI quantiles: Q05={bmi_q05:.2f}  Q25={bmi_q25:.2f}  "
          f"Q75={bmi_q75:.2f}  Q95={bmi_q95:.2f}")


    INTERVENTION_GRIDS = {
        # "BMI":  [20, 23, 26, 29, 32, 35],
        "BMI":  sorted(set(
                    list(np.linspace(bmi_q05, bmi_q95, 6)) +
                    [bmi_q25, bmi_q75]
                )),
        "GLUC": [4, 5, 6, 7, 8, 10],
        "HDL":  [0.8, 1.0, 1.2, 1.5, 1.8, 2.2],
    }
    DELTA_RANGES = {
        # "BMI":  (20, 35),
        "BMI":  (bmi_q25, bmi_q75),    # data-driven
        "PAS":  (110, 160),
        "PAD":  (60, 85),
        "GLUC": (4, 10),
        "HDL":  (0.8, 2.2),
    }

    os.makedirs(os.path.dirname(args.prefix) or ".", exist_ok=True)

    # ── Run analyses ────────────────────────────────────────────────────
    for col_idx, feat_name in enumerate(time_varying_features):
        if feat_name not in DELTA_RANGES:
            continue

        val_lo, val_hi = DELTA_RANGES[feat_name]
        values = INTERVENTION_GRIDS.get(feat_name)

        print(f"\n{'='*60}")
        print(f"PDP ANALYSIS: {feat_name} (col={col_idx}, continuous time)")
        print(f"{'='*60}")

        # ── 1. Constant-intervention PDP ────────────────────────────────
        if values is not None:
            results_const, ages, _, n_subj = compute_pdp_continuous(
                model, eval_loader, device, values, eval_grid,
                target_col=col_idx, n_tv=K,
                mask_type=mask_type, target_name=feat_name,
            )

            plot_pdp_continuous(
                results_const, eval_grid, values,
                save_path=f"{args.prefix}_{feat_name}_marginal.png",
                target_name=feat_name,
                visit_times=VISIT_TIMES_3C,
            )

            # ΔPDP
            if val_lo in results_const and val_hi in results_const:
                plot_delta_pdp_continuous(
                    results_const, eval_grid, val_lo, val_hi,
                    save_path=f"{args.prefix}_{feat_name}_delta.png",
                    target_name=feat_name,
                )

        # ── 2. Trajectory-profile PDP ───────────────────────────────────
        profiles = make_profiles_continuous(eval_grid, v_lo=val_lo, v_hi=val_hi)

        traj_results, _, n_subj = compute_trajectory_profile_pdp_continuous(
            model, eval_loader, device, profiles, eval_grid,
            target_col=col_idx, n_tv=K,
            mask_type=mask_type, target_name=feat_name,
        )

        plot_trajectory_profile_pdp_continuous(
            traj_results, eval_grid,
            save_path=f"{args.prefix}_{feat_name}_traj_profile.png",
            target_name=feat_name,
            visit_times=VISIT_TIMES_3C,
        )

    print(f"\nDone. All figures saved with prefix: {args.prefix}")