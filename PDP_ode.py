"""
Standalone PDP evaluation script for Neural ODE + BMI Skip model.

Usage:
    python PDP_ode.py
    python PDP_ode.py --checkpoint checkpoints/best_model_ode_skip.pt
    python PDP_ode.py --bmi_mode shifted
    python PDP_ode.py --with_blup   # include random effects in ICE
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pyreadr
import argparse
from dataset import LongitudinalDataset, collate_pad

from PDP_analysis_ODE import (
    compute_pdp, compute_pdp_with_blup,
    plot_pdp, plot_pdp_marginal,
    compute_delta_pdp, compute_delta_pdp_stratified,
    compute_true_delta_pdp,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDP analysis for Neural ODE model")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_model_ode_full_skip_0.pt")
    parser.add_argument("--data", type=str,
                        default="simu_datasets/S2a_sims/sim_001.rds")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bmi_mode", type=str, default="constant",
                        choices=["constant", "linear", "shifted"])
    parser.add_argument("--bmi_slope", type=float, default=None)
    parser.add_argument("--prefix", type=str, default="figures/pdp_ode")
    parser.add_argument("--with_blup", action="store_true",
                        help="Include BLUP random effects in ICE curves")
    parser.add_argument("--true_beta_bmi", type=float, default=-0.30)
    parser.add_argument("--true_beta_int", type=float, default=-0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
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
    eval_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_pad)

    n_tv = 1

    # ---- Build model ----
    from model_ODE import NeuralODEModel, NeuralODEConfig

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
        re_spline_cols=[1, 2],
        g_hidden=16,
        fullD=True,
        bmi_mean=0.0,   # placeholder — overwritten by checkpoint
        bmi_std=1.0,
        static_skip_dims=[1],
    ).to(device)

    # ---- Load checkpoint ----
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"  epoch = {checkpoint.get('epoch', '?')}")
        loss_val = checkpoint.get('best_test_loss', None)
        if loss_val is not None:
            print(f"  loss  = {loss_val:.4f}")
        if 'config' in checkpoint:
            print(f"  config = {checkpoint['config']}")
    else:
        model.load_state_dict(checkpoint, strict=False)

    # Verify BMI stats loaded correctly
    print(f"  bmi_mean = {model.decoder.bmi_mean.item():.4f}")
    print(f"  bmi_std  = {model.decoder.bmi_std.item():.4f}")

    # ---- Print model parameters ----
    model.eval()
    print(f"\n{'='*60}")
    print(f"MODEL PARAMETERS (Neural ODE + BMI Skip)")
    print(f"{'='*60}")

    sig2 = torch.exp(model.decoder.log_residual_var).item()
    print(f"  sigma2 = {sig2:.6f}")

    bn = model.decoder.beta_neural.detach()
    print(f"  beta_neural = {bn.cpu().tolist()}")
    print(f"  beta_neural norm = {bn.norm().item():.4f}")

    if model.decoder.L_unconstrained is not None:
        D = model.decoder._build_D(device=torch.device('cpu'), dtype=torch.float32)
        print(f"  D matrix ({D.shape[0]}x{D.shape[1]}):")
        for i in range(D.shape[0]):
            print(f"    [{', '.join(f'{D[i,j]:+.4f}' for j in range(D.shape[1]))}]")

    # ---- True parameters ----
    TRUE_BETA_BMI = args.true_beta_bmi
    TRUE_BETA_INT = args.true_beta_int
    print(f"\nTrue β (Scenario 2):")
    print(f"  β_BMI = {TRUE_BETA_BMI}, β_BMI×AGEc = {TRUE_BETA_INT}")

    # ---- PDP Analysis ----
    bmi_values = [20, 23, 26, 29, 32, 35]
    bmi_col = 0   # BMI is always col 0 in x_pad
    visit_times = np.array([0, 5, 10, 15])

    print(f"\n{'='*60}")
    print(f"PDP ANALYSIS (mode={args.bmi_mode})")
    print(f"  BMI values = {bmi_values}")
    print(f"  visit times = {visit_times.tolist()}")
    print(f"  with_blup = {args.with_blup}")
    print(f"{'='*60}")

    suffix = f"_{args.bmi_mode}" if args.bmi_mode != "constant" else ""

    if args.with_blup:
        # --- PDP + ICE with BLUP ---
        results_pop, results_subj, blup, ages, masks, times = compute_pdp_with_blup(
            model, eval_loader, device, bmi_values,
            bmi_col=bmi_col, bmi_mode=args.bmi_mode, bmi_slope=args.bmi_slope,
        )
        results = results_pop  # for delta_pdp computation

        print(f"\n  BLUP stats: shape={blup.shape}")
        print(f"    norm: mean={blup.norm(dim=1).mean():.3f}, "
              f"max={blup.norm(dim=1).max():.3f}")

        # Plot with BLUP ICE
        plot_pdp_marginal(results, masks, times, bmi_values,
                          save_path=f"{args.prefix}_marginal_blup{suffix}.png",
                          visit_times=visit_times,
                          ice_results=results_subj, ice_n=100)

    else:
        # --- Standard PDP (no BLUP) ---
        results, ages, masks, times = compute_pdp(
            model, eval_loader, device, bmi_values,
            bmi_col=bmi_col, bmi_mode=args.bmi_mode, bmi_slope=args.bmi_slope,
        )

        # Plot without BLUP
        plot_pdp_marginal(results, masks, times, bmi_values,
                          save_path=f"{args.prefix}_marginal{suffix}.png",
                          visit_times=visit_times)

    # ---- Stratified PDP plot ----
    plot_pdp(results, ages, masks, times, bmi_values,
             save_path=f"{args.prefix}_by_age{suffix}.png",
             visit_times=visit_times)

    # ---- True ΔPDP ----
    compute_true_delta_pdp(
        ages, masks, times,
        bmi_lo=20, bmi_hi=35,
        true_beta_bmi=TRUE_BETA_BMI, true_beta_int=TRUE_BETA_INT,
        visit_times=visit_times,
    )

    # ---- Estimated ΔPDP (marginal) ----
    estimated, true_ref = compute_delta_pdp(
        results, ages, masks, times,
        bmi_lo=20, bmi_hi=35,
        true_beta_bmi=TRUE_BETA_BMI, true_beta_int=TRUE_BETA_INT,
        visit_times=visit_times,
    )

    # ---- Estimated ΔPDP (stratified) ----
    compute_delta_pdp_stratified(
        results, ages, masks, times,
        bmi_lo=20, bmi_hi=35,
        true_beta_bmi=TRUE_BETA_BMI, true_beta_int=TRUE_BETA_INT,
        visit_times=visit_times,
    )

    # ---- Summary ----
    delta_v = bmi_values[-1] - bmi_values[0]
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: Neural ODE + BMI Skip")
    print(f"  BMI mode: {args.bmi_mode}")
    print(f"  True β_BMI={TRUE_BETA_BMI}, β_int={TRUE_BETA_INT}")

    if estimated:
        print(f"\n  {'Time':>6s}  {'Estimated':>10s}  {'True':>10s}  {'Bias':>10s}")
        print(f"  {'-'*42}")
        for vt in sorted(estimated.keys()):
            est = estimated[vt]
            tru = true_ref[vt]
            print(f"  {vt:6.0f}  {est:+10.4f}  {tru:+10.4f}  {est-tru:+10.4f}")

    print(f"\nDone.")