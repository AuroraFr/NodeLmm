"""
Runner for continuous-time PDP analysis — Neural ODE-LMM on 3C cohort.

v2: adds delta-method variance estimation (Section 3.2 of the paper).

Usage:
    python PDP_ode_real_continuous.py
    python PDP_ode_real_continuous.py --delta_method               # full CI
    python PDP_ode_real_continuous.py --delta_method --fisher_max 2000  # faster
    python PDP_ode_real_continuous.py --n_points 50                # finer grid
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
    plot_trajectory_profile_pdp_delta,
    plot_pdp_continuous,
    plot_delta_pdp_continuous,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continuous-time PDP analysis — Neural ODE-LMM on 3C")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_model_ode_real_3C_regnone_H8_seed42.pt")
    parser.add_argument("--data", type=str,
                        default="3C_dataset/train_3C_data_1.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_points", type=int, default=15,
                        help="Number of grid points (default 15 = yearly 0..14)")
    parser.add_argument("--t_max", type=float, default=14.0,
                        help="Maximum time (years)")
    parser.add_argument("--prefix", type=str,
                        default="figures/pdp_real_noreg_H8_continuous")

    # Delta-method options
    parser.add_argument("--delta_method", action="store_true",
                        help="Compute delta-method CI (requires Fisher)")
    parser.add_argument("--fisher_max", type=int, default=None,
                        help="Max subjects for Fisher (None = all)")
    parser.add_argument("--fisher_cache", type=str, default=None,
                        help="Path to save/load cached Fisher inverse")
    parser.add_argument("--damping", type=float, default=1e-4,
                        help="Marquardt damping for Fisher inversion")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ─────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    ckpt_cfg = checkpoint['config']
    print(ckpt_cfg)

    print(f"Checkpoint: {args.checkpoint}")
    # print(f"  epoch     = {checkpoint.get('epoch', '?')}")
    # print(f"  test loss = {checkpoint.get('best_test_loss', '?'):.4f}")

    # ── Feature definitions ─────────────────────────────────────────────
    id_col = "NUM_ID"
    target_col = "ISA15"
    time_varying_features = ckpt_cfg.get('time_varying_features', ["BMI", "PAS", "PAD", "GLUC", "HDL"])
    static_features = ckpt_cfg.get('static_features', ["SEX_code", "AGEc", "DIPNIV_2", "DIPNIV_3"])
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
    print(ckpt_cfg)

    dataset = RealDataset(patient_data)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size,
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
        use_dynamic_skip=False,
        static_skip_dims=list(range(Ks)),
        # reg_mode=ckpt_cfg.get('reg_mode', None),
        reg_mode=None
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
    print(f"  Delta-method CI: {'ON' if args.delta_method else 'OFF'}")
    print(f"{'='*60}")

    bmi_vals = df["BMI"].dropna().values
    bmi_q25 = float(np.percentile(bmi_vals, 25))
    bmi_q75 = float(np.percentile(bmi_vals, 75))
    bmi_q05 = float(np.percentile(bmi_vals, 5))
    bmi_q95 = float(np.percentile(bmi_vals, 95))

    print(f"  BMI quantiles: Q05={bmi_q05:.2f}  Q25={bmi_q25:.2f}  "
          f"Q75={bmi_q75:.2f}  Q95={bmi_q95:.2f}")

    INTERVENTION_GRIDS = {
        "BMI":  sorted(set(
                    list(np.linspace(bmi_q05, bmi_q95, 6)) +
                    [bmi_q25, bmi_q75]
                )),
        "GLUC": [4, 5, 6, 7, 8, 10],
        "HDL":  [0.8, 1.0, 1.2, 1.5, 1.8, 2.2],
    }
    DELTA_RANGES = {
        "BMI":  (bmi_q25, bmi_q75),
        "GLUC": (4, 10),
        "HDL":  (0.8, 2.2),
    }

    os.makedirs(os.path.dirname(args.prefix) or ".", exist_ok=True)

    # ── Fisher computation (once, shared across covariates) ─────────────
    fisher_inv = None
    if args.delta_method:
        from PDP_variance import (
            compute_empirical_fisher, _regularise_and_invert,
        )
        from train_ODE_real import collate_real

        # Check for cached Fisher
        if args.fisher_cache and os.path.exists(args.fisher_cache):
            print(f"\n  Loading cached Fisher inverse from {args.fisher_cache}")
            fisher_inv = torch.from_numpy(np.load(args.fisher_cache)).float()
        else:
            print(f"\n{'='*60}")
            print(f"COMPUTING EMPIRICAL FISHER INFORMATION")
            print(f"{'='*60}")

            lambda_reg = ckpt_cfg.get('lambda_gate', 0.1)
            lambda_wd = ckpt_cfg.get('lambda_wd', 1e-5)

            fisher, scores = compute_empirical_fisher(
                model, dataset, device, collate_fn=collate_real,
                lambda_reg=0.1,
                weight_decay=lambda_wd,
                max_subjects=args.fisher_max,
                verbose=True,
            )

            # Regularise and invert
            _, fisher_inv = _regularise_and_invert(
                fisher, "F", LAMBDA=args.damping, verbose=True,
            )

            # Stationarity check: mean score should be ≈ 0
            if scores.shape[0] > 0:
                mean_score = scores.mean(dim=0)
                print(f"  Stationarity check: ||mean(φ)|| = "
                      f"{mean_score.norm().item():.4e}")

            # Cache for reuse
            if args.fisher_cache:
                np.save(args.fisher_cache, fisher_inv.numpy())
                print(f"  Fisher inverse cached to {args.fisher_cache}")

    # ── Run analyses ────────────────────────────────────────────────────
    for col_idx, feat_name in enumerate(time_varying_features):
        if feat_name not in DELTA_RANGES:
            continue

        val_lo, val_hi = DELTA_RANGES[feat_name]
        values = INTERVENTION_GRIDS.get(feat_name)

        print(f"\n{'='*60}")
        print(f"PDP ANALYSIS: {feat_name} (col={col_idx}, continuous time)")
        print(f"{'='*60}")

        # ── 1. Cross-subject SE (always) ────────────────────────────────
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

        # ── 2. Delta-method CI (if requested) ──────────────────────────
        if args.delta_method and fisher_inv is not None:
            from PDP_variance import compute_trajectory_profile_pdp_with_ci
            from PDP_continuous_time import (
                plot_delta_profile_pdp_delta,
                plot_all_pairwise_delta_pdp,
            )

            print(f"\n  Computing delta-method CI for {feat_name} ...")

            ci_results = compute_trajectory_profile_pdp_with_ci(
                model, eval_loader, device, profiles, eval_grid,
                fisher_inv=fisher_inv,
                target_col=col_idx, n_tv=K,
                mask_type=mask_type,
                target_name=feat_name,
                verbose=True,
            )

            from pdp_real_wald_test import wald_test_all_pairs

            wald_results = wald_test_all_pairs(
                ci_results, eval_grid, fisher_inv,
                late_cutoff=7.0,   # second half of follow-up
                verbose=True,
            )

            # Profile PDP with CI bands
            plot_trajectory_profile_pdp_delta(
                ci_results, eval_grid,
                save_path=f"{args.prefix}_{feat_name}_traj_profile_delta.png",
                target_name=feat_name,
                visit_times=VISIT_TIMES_3C,
                n_subjects=n_subj,
            )

            # ΔPDP: early_burden vs late_spike (path-dependence diagnostic)
            if "late_decline" in ci_results and "late_spike" in ci_results:
                plot_delta_profile_pdp_delta(
                    ci_results, eval_grid,
                    profile_a="late_decline", profile_b="late_spike",
                    save_path=f"{args.prefix}_{feat_name}_delta_eb_vs_ls.png",
                    target_name=feat_name,
                    visit_times=VISIT_TIMES_3C,
                    n_subjects=n_subj,
                    fisher_inv=fisher_inv,
                )

            # All three diagnostic pairs side by side
            plot_all_pairwise_delta_pdp(
                ci_results, eval_grid,
                save_path=f"{args.prefix}_{feat_name}_delta_all_pairs.png",
                target_name=feat_name,
                visit_times=VISIT_TIMES_3C,
                n_subjects=n_subj,
                fisher_inv=fisher_inv,
            )

            # Print comparison of SE methods
            print(f"\n    SE comparison: cross-subject vs delta-method")
            print(f"    {'Time':>8s}  {'cross-SE':>10s}  {'delta-SE':>10s}  "
                  f"{'ratio':>8s}")
            for ell in range(len(eval_grid)):
                t = eval_grid[ell]
                if "late_decline" in traj_results and "late_spike" in traj_results:
                    diff_subj = (traj_results["late_decline"]
                                 - traj_results["late_spike"])
                    cross_se = diff_subj.std(axis=0)[ell] / np.sqrt(n_subj)
                else:
                    cross_se = float('nan')

                if '_delta_eb_ls' in ci_results:
                    delta_se = ci_results['_delta_eb_ls']['se'][ell]
                else:
                    delta_se = float('nan')

                ratio = delta_se / cross_se if cross_se > 0 else float('nan')
                print(f"    {t:8.1f}  {cross_se:10.4f}  {delta_se:10.4f}  "
                      f"{ratio:8.2f}")

    print(f"\nDone. All figures saved with prefix: {args.prefix}")