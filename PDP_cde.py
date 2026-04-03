"""
Standalone PDP evaluation script for Neural CDE model.

Includes:
  A) Standard PDP (constant level interventions) — all scenarios
  B) Trajectory-class PDP (S5/S6 only) — volatile, ramp, step, shifted

Usage:
    python PDP_CDE.py --scenario S6
    python PDP_CDE.py --scenario S6 --no_trajectory
    python PDP_CDE.py --scenario S2 --inject_x --encoder_sees_covariates
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pyreadr
import argparse
import os
from dataset import LongitudinalDataset, collate_pad

from PDP_analysis_CDE import (
    compute_pdp,
    plot_pdp, plot_pdp_marginal,
    compute_delta_pdp, compute_delta_pdp_stratified,
    compute_true_delta_pdp,
    compute_trajectory_delta_pdp,
    plot_trajectory_delta_pdp,
    plot_all_trajectory_profiles,
    compute_pdp_causal
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ── Scenario presets ──
SCENARIO_DEFAULTS = {
    "S2": {
        "true_beta_bmi": -0.30,
        "true_beta_int": -0.05,
        "alpha": None,
        "data": "simu_datasets/S2a_sims_2/sim_001.rds",
        "checkpoint": "checkpoints/best_CDE_S2.pt",
        "x_cols": ["BMI_t", "rs1", "rs2"],
    },
    "S5": {
        "true_beta_bmi": 0.0,
        "true_beta_int": 0.0,
        "alpha": -0.5,
        "data": "simu_datasets/S5_sims/sim_001.rds",
        "checkpoint": "checkpoints/best_CDE_S5.pt",
        "x_cols": ["BMI_t"],
    },
    "S6": {
        "true_beta_bmi": 0.0,
        "true_beta_int": 0.0,
        "alpha": -0.15,
        "data": "simu_datasets/S6_sims/sim_001.rds",
        "checkpoint": "checkpoints/best_CDE_S6.pt",
        "x_cols": ["BMI_t"],
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDP analysis for Neural CDE model")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="S6",
                        choices=["S2", "S5", "S6"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bmi_mode", type=str, default="constant",
                        choices=["constant", "linear", "shifted"])
    parser.add_argument("--bmi_slope", type=float, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--true_beta_bmi", type=float, default=None)
    parser.add_argument("--true_beta_int", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--interp", type=str, default="cubic",
                        choices=["linear", "cubic"])
    # Model architecture
    parser.add_argument("--inject_x", action="store_true", default=False)
    parser.add_argument("--augment_order", type=int, default=2)
    parser.add_argument("--encoder_sees_covariates", action="store_true", default=False)
    # Trajectory PDP
    parser.add_argument("--no_trajectory", action="store_true", default=False)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--period", type=float, default=5.0)
    parser.add_argument("--slope_slow", type=float, default=0.05)
    parser.add_argument("--slope_fast", type=float, default=0.3)
    parser.add_argument("--bmi_before", type=float, default=25.0)
    parser.add_argument("--bmi_after", type=float, default=30.0)
    parser.add_argument("--t_step", type=float, default=2.0)
    args = parser.parse_args()

    # ── Apply scenario defaults ──
    defaults = SCENARIO_DEFAULTS[args.scenario]
    if args.data is None:
        args.data = defaults["data"]
    if args.checkpoint is None:
        args.checkpoint = defaults["checkpoint"]
    if args.true_beta_bmi is None:
        args.true_beta_bmi = defaults["true_beta_bmi"]
    if args.true_beta_int is None:
        args.true_beta_int = defaults["true_beta_int"]
    if args.alpha is None:
        args.alpha = defaults["alpha"]
    if args.prefix is None:
        args.prefix = f"figures/pdp_cde_{args.scenario}"

    x_cols = defaults["x_cols"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Fixed PDP parameters ──
    bmi_values = [20, 23, 26, 29, 32, 35]
    bmi_lo, bmi_hi = 20, 35
    visit_times = np.array([0, 5, 10, 15])

    # ── Data ──
    time_col = "time"
    y_col = "ISA15_sim"
    id_col = "NUM_ID"
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    df = next(iter(pyreadr.read_r(args.data).values()))
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col,
                                  static_cols=static_cols)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_pad)

    n_tv = 1

    # ── Build model ──
    from model_CDE import NeuralCDEModel, NeuralCDEConfig

    cfg = NeuralCDEConfig(
        hidden_channels=8,
        enc_mlp_hidden=32,
        func_mlp_hidden=32,
        dec_rho_hidden=16,
        dec_p=4,
        dec_q=3,
        depth=2,
        dropout=0.0,
    )

    if "rs1" in x_cols and "rs2" in x_cols:
        use_neural_re = False
        re_spline_cols = [x_cols.index("rs1"), x_cols.index("rs2")]
    else:
        use_neural_re = True
        re_spline_cols = None

    model = NeuralCDEModel(
        x_dim=len(x_cols),
        static_dim=len(static_cols),
        cfg=cfg,
        n_tv=n_tv,
        inject_x=args.inject_x,
        augment_order=args.augment_order,
        encoder_sees_covariates=args.encoder_sees_covariates,
        use_rho_net=True,
        use_neural_re=use_neural_re,
        g_hidden=16,
        re_spline_cols=re_spline_cols,
        fullD=True,
        tv_skip_cols=None,
        static_skip_dims=None,
        reg_mode=None,
    ).to(device)

    # ── Load checkpoint ──
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded: {args.checkpoint}")
        loss_val = checkpoint.get('best_test_loss', None)
        if loss_val is not None:
            print(f"  best test loss = {loss_val:.4f}")
    else:
        model.load_state_dict(checkpoint, strict=False)

    # ---- Build eval loader from test set ----
    from torch.utils.data import Subset
    if isinstance(checkpoint, dict) and 'test_idx' in checkpoint:
        test_idx = checkpoint['test_idx']
        eval_dataset = Subset(dataset, test_idx)
        print(f"  Evaluating on test set: {len(test_idx)} subjects")
    else:
        eval_dataset = dataset
        print(f"  No test_idx in checkpoint — evaluating on full dataset ({len(dataset)} subjects)")
    
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_pad)

    # ── Model info ──
    model.eval()
    print(f"\n{'='*60}")
    print(f"MODEL: Neural CDE — Scenario {args.scenario}")
    print(f"{'='*60}")
    model.describe()

    TRUE_BETA_BMI = args.true_beta_bmi
    TRUE_BETA_INT = args.true_beta_int
    print(f"\nTrue DGP: beta_BMI={TRUE_BETA_BMI}, beta_int={TRUE_BETA_INT}")
    if args.alpha is not None:
        print(f"  alpha={args.alpha}")

    # ══════════════════════════════════════════
    # A) STANDARD PDP
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"A) STANDARD PDP (mode={args.bmi_mode})")
    print(f"  BMI = {bmi_values}, ΔPDP: {bmi_lo} -> {bmi_hi}")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.prefix) or ".", exist_ok=True)
    suffix = f"_{args.bmi_mode}" if args.bmi_mode != "constant" else ""

    results, ages, masks, times = compute_pdp(
        model, eval_loader, device, bmi_values,
        n_tv=n_tv, bmi_col=0,
        bmi_mode=args.bmi_mode, bmi_slope=args.bmi_slope,
        interp=args.interp,
    )

    plot_pdp(results, ages, masks, times, bmi_values,
             save_path=f"{args.prefix}_by_age{suffix}.png",
             visit_times=visit_times)
    plot_pdp_marginal(results, masks, times, bmi_values,
                      save_path=f"{args.prefix}_marginal{suffix}.png",
                      visit_times=visit_times)

    compute_true_delta_pdp(
        ages, masks, times, bmi_lo=bmi_lo, bmi_hi=bmi_hi,
        true_beta_bmi=TRUE_BETA_BMI, true_beta_int=TRUE_BETA_INT,
        visit_times=visit_times,
    )
    estimated, true_ref = compute_delta_pdp(
        results, ages, masks, times,
        bmi_lo=bmi_lo, bmi_hi=bmi_hi,
        true_beta_bmi=TRUE_BETA_BMI, true_beta_int=TRUE_BETA_INT,
        visit_times=visit_times,
    )
    compute_delta_pdp_stratified(
        results, ages, masks, times,
        bmi_lo=bmi_lo, bmi_hi=bmi_hi,
        true_beta_bmi=TRUE_BETA_BMI, true_beta_int=TRUE_BETA_INT,
        visit_times=visit_times,
    )

    # ══════════════════════════════════════════
    # B) TRAJECTORY-CLASS PDP (S5/S6 only)
    # ══════════════════════════════════════════
    if args.scenario in ("S5", "S6") and not args.no_trajectory and args.alpha is not None:
        print(f"\n{'='*60}")
        print(f"B) TRAJECTORY-CLASS PDP (alpha={args.alpha})")
        print(f"{'='*60}")

        all_traj = {}

        # B1: Volatile vs Stable
        est, orc, _, _, _, _ = compute_trajectory_delta_pdp(
            model, eval_loader, device,
            profile="volatile_vs_stable", alpha=args.alpha,
            amplitude=args.amplitude, period=args.period,
            interp=args.interp, visit_times=visit_times,
        )
        all_traj["volatile_vs_stable"] = (est, orc)
        plot_trajectory_delta_pdp(est, orc, "volatile_vs_stable",
                                  f"{args.prefix}_traj_volatile{suffix}.png")

        # B2: Fast vs Slow Ramp
        est, orc, _, _, _, _ = compute_trajectory_delta_pdp(
            model, eval_loader, device,
            profile="fast_vs_slow_ramp", alpha=args.alpha,
            slope_fast=args.slope_fast, slope_slow=args.slope_slow,
            interp=args.interp, visit_times=visit_times,
        )
        all_traj["fast_vs_slow_ramp"] = (est, orc)
        plot_trajectory_delta_pdp(est, orc, "fast_vs_slow_ramp",
                                  f"{args.prefix}_traj_ramp{suffix}.png")

        # B3: Step vs Stable
        est, orc, _, _, _, _ = compute_trajectory_delta_pdp(
            model, eval_loader, device,
            profile="step_vs_stable", alpha=args.alpha,
            bmi_before=args.bmi_before, bmi_after=args.bmi_after,
            t_step=args.t_step,
            interp=args.interp, visit_times=visit_times,
        )
        all_traj["step_vs_stable"] = (est, orc)
        plot_trajectory_delta_pdp(est, orc, "step_vs_stable",
                                  f"{args.prefix}_traj_step{suffix}.png")

        # B4: Shifted (negative control)
        est, orc, _, _, _, _ = compute_trajectory_delta_pdp(
            model, eval_loader, device,
            profile="shifted", alpha=args.alpha,
            interp=args.interp, visit_times=visit_times,
        )
        all_traj["shifted"] = (est, orc)
        plot_trajectory_delta_pdp(est, orc, "shifted",
                                  f"{args.prefix}_traj_shifted{suffix}.png")

        # Summary
        plot_all_trajectory_profiles(all_traj,
                                     f"{args.prefix}_traj_summary{suffix}.png")

    # ══════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"SUMMARY — CDE Scenario {args.scenario}")
    print(f"{'='*60}")
    print(f"  inject_x={args.inject_x}, augment={args.augment_order}, "
          f"encoder_covs={args.encoder_sees_covariates}")

    if estimated:
        print(f"\n  Standard ΔPDP ({bmi_lo} -> {bmi_hi}):")
        print(f"  {'Time':>6s}  {'Estimated':>10s}  {'True':>10s}  {'Bias':>10s}")
        print(f"  {'-'*42}")
        for vt in sorted(estimated.keys()):
            print(f"  {vt:6.0f}  {estimated[vt]:+10.4f}  "
                  f"{true_ref[vt]:+10.4f}  {estimated[vt]-true_ref[vt]:+10.4f}")

    print(f"\nDone.")