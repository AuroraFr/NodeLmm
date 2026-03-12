"""
Standalone PDP evaluation script.
Loads a trained checkpoint and runs PDP analysis without retraining.

Usage:
    python PDP.py                                          # pure NN model (default)
    python PDP.py --hybrid --checkpoint checkpoints/best_model_hybrid.pt
    python PDP.py --bmi_mode shifted
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pyreadr
import argparse
from dataset import LongitudinalDataset, collate_pad

from pdp_analysis import (compute_pdp, plot_pdp, plot_pdp_marginal,
                          compute_delta_pdp, compute_delta_pdp_stratified,
                          compute_true_delta_pdp)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def print_model_params(model, is_hybrid, beta_names=None):
    """
    Print model parameters safely for both model types.
    For hybrid: prints parametric β (if available) + neural weight.
    For pure NN: prints neural weight norms and variance components.
    """
    print(f"\n{'='*60}")
    print(f"MODEL PARAMETERS ({'hybrid' if is_hybrid else 'pure NN'})")
    print(f"{'='*60}")

    # --- Residual variance (both models) ---
    if hasattr(model.decoder, 'log_residual_var'):
        sig2 = torch.exp(model.decoder.log_residual_var).item()
        print(f"  {'sigma2':>20s} = {sig2:.6f}")

    if is_hybrid:
        # Hybrid: parametric β stored as nn.Parameter or cached in _last_beta
        beta = None
        if hasattr(model.decoder, 'beta') and model.decoder.beta is not None:
            beta = model.decoder.beta.detach().cpu()
        elif hasattr(model.decoder, '_last_beta') and model.decoder._last_beta is not None:
            beta = model.decoder._last_beta.detach().cpu()

        if beta is not None and beta_names is not None:
            print(f"  Beta layout: {beta_names}")
            n_print = min(len(beta_names), len(beta))
            for name, val in zip(beta_names[:n_print], beta[:n_print]):
                print(f"  {name:>20s} = {val.item():+.6f}")
        elif beta is not None:
            print(f"  Beta (raw): {beta.tolist()}")
        else:
            print(f"  [β not available — may need a forward pass to populate]")

        # Neural component weight norm
        if hasattr(model.decoder, 'w_neural'):
            bn = model.decoder.w_neural.detach()
            print(f"  {'h(z) weight norm':>20s} = {bn.norm().item():.6f}")
        elif hasattr(model.decoder, 'beta_neural'):
            bn = model.decoder.beta_neural.detach()
            print(f"  {'β_neural norm':>20s} = {bn.norm().item():.6f}")

    else:
        # Pure NN model: ρ(z)·β_neural + g(z)·b structure
        if hasattr(model.decoder, 'beta_neural'):
            bn = model.decoder.beta_neural.detach()
            print(f"  {'β_neural norm':>20s} = {bn.norm().item():.6f}")
            print(f"  {'β_neural':>20s} = {bn.cpu().tolist()}")
        if hasattr(model.decoder, 'use_rho_net'):
            print(f"  {'rho_net':>20s} = {'enabled' if model.decoder.use_rho_net else 'disabled'}")
        if hasattr(model.decoder, 'use_neural_re'):
            print(f"  {'neural RE (g_net)':>20s} = {'enabled' if model.decoder.use_neural_re else 'disabled'}")

    # --- Random effects covariance D (both models, if present) ---
    if hasattr(model.decoder, 'L_unconstrained') and model.decoder.L_unconstrained is not None:
        try:
            D = model.decoder._build_D(device=torch.device('cpu'), dtype=torch.float32)
            print(f"  D matrix ({D.shape[0]}x{D.shape[1]}):")
            for i in range(D.shape[0]):
                print(f"    [{', '.join(f'{D[i,j]:+.4f}' for j in range(D.shape[1]))}]")
        except Exception as e:
            print(f"  D matrix: [error: {e}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDP analysis on trained model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_hybrid_neural_fe_spline_re.pt")
    parser.add_argument("--pure_nn", action="store_true",
                        help="Use pure NN model (model) instead of hybrid (model_hybrid_reg)")
    parser.add_argument("--data", type=str, default="simu_datasets/S2a_sims_2/sim_001.rds")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bmi_mode", type=str, default="constant",
                        choices=["constant", "linear", "shifted"],
                        help="BMI intervention mode for PDP")
    parser.add_argument("--bmi_slope", type=float, default=None,
                        help="BMI slope for linear mode (auto-estimated if None)")
    parser.add_argument("--interp", type=str, default="linear",
                        choices=["linear", "cubic"])
    parser.add_argument("--prefix", type=str, default=None,
                        help="Prefix for output figure paths (auto: figures/pdp_{hybrid|neural})")
    # True params — override for different scenarios
    parser.add_argument("--true_beta_bmi", type=float, default=-0.30,
                        help="True beta_BMI (Scenario 2 default: -0.30)")
    parser.add_argument("--true_beta_int", type=float, default=-0.05,
                        help="True beta_BMIxAGEc (Scenario 2 default: -0.05)")
    args = parser.parse_args()

    is_hybrid = not args.pure_nn

    # Auto-set prefix based on model type
    if args.prefix is None:
        args.prefix = f"figures/pdp_{'hybrid' if is_hybrid else 'neural'}"

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

    dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col, static_cols=static_cols)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)

    # ---- Spline knots (from R, for hybrid model) ----
    fe_knots    = np.array([1.769863, 6.693151])
    fe_boundary = np.array([0.0, 13.50685])
    re_knots    = np.array([3.567123])
    re_boundary = np.array([0.0, 13.50685])

    n_tv = 1
    interaction_pairs = [(0, 1)]  # BMI_t x AGEc

    # ---- Build model ----
    if is_hybrid:
        from model_hybrid_reg import NeuralCDEModel, NeuralCDEConfig
        cfg = NeuralCDEConfig(
            hidden_channels=4,
            enc_mlp_hidden=16,
            func_mlp_hidden=16,
            dec_rho_hidden=16,
            dec_p=4,
            dec_q=3,
            depth=1,
            dropout=0.0,
        )
        model = NeuralCDEModel(
            x_dim=len(x_cols),
            static_dim=len(static_cols),
            cfg=cfg,
            fe_spline_knots=fe_knots,
            fe_spline_boundary=fe_boundary,
            re_spline_knots=re_knots,
            re_spline_boundary=re_boundary,
            interaction_pairs=interaction_pairs,
            precomputed_splines=True,
            n_tv=n_tv,
            include_fe_splines=False,
            use_rho_net=False
        ).to(device)
    else:
        from model import NeuralCDEModel, NeuralCDEConfig
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
        model = NeuralCDEModel(
            x_dim=len(x_cols),
            static_dim=len(static_cols),
            cfg=cfg,
            n_tv=n_tv,
            use_rho_net=True,
            use_neural_re=True,
            g_hidden=16,
            bmi_mean=0.0,
            bmi_std = 1.0
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
        print(f"Loaded state dict: {args.checkpoint}")

    print("LOADED", model.decoder.bmi_std, model.decoder.bmi_mean)

    # ---- Beta names (only meaningful for hybrid) ----
    beta_names = None
    if is_hybrid:
        beta_names = ["intercept"]
        beta_names += x_cols[:n_tv] + static_cols
        if interaction_pairs:
            for tv_i, s_i in interaction_pairs:
                beta_names.append(f"{x_cols[tv_i]}x{static_cols[s_i]}")

    # ---- Print model parameters ----
    model.eval()
    print_model_params(model, is_hybrid=is_hybrid, beta_names=beta_names)

    # ---- True parameters ----
    TRUE_BETA_BMI = args.true_beta_bmi
    TRUE_BETA_INT = args.true_beta_int

    print(f"\nTrue beta_0 (Scenario 2):")
    print(f"  beta_BMI = {TRUE_BETA_BMI}, beta_BMIxAGEc = {TRUE_BETA_INT}")

    # ---- PDP Analysis ----
    bmi_values = [20, 23, 26, 29, 32, 35]
    bmi_col = x_cols.index("BMI_t")
    visit_times = np.array([0, 5, 10, 15])

    print(f"\n{'='*60}")
    print(f"PDP ANALYSIS (mode={args.bmi_mode}, interp={args.interp})")
    print(f"  model = {'hybrid' if is_hybrid else 'pure NN'}")
    print(f"  BMI values = {bmi_values}")
    print(f"  visit times = {visit_times.tolist()}")
    print(f"{'='*60}")

    results, ages, masks, times = compute_pdp(
        model, eval_loader, device, bmi_values,
        n_tv=n_tv, interp=args.interp,
        bmi_mode=args.bmi_mode, bmi_slope=args.bmi_slope,
        bmi_col=bmi_col,
    )

    suffix = f"_{args.bmi_mode}" if args.bmi_mode != "constant" else ""
    model_tag = "hybrid" if is_hybrid else "neural"

    # ---- 1. True delta_PDP ----
    true_marginal, true_stratified, age_groups = compute_true_delta_pdp(
        ages, masks, times,
        bmi_lo=20, bmi_hi=35,
        true_beta_bmi=TRUE_BETA_BMI, true_beta_int=TRUE_BETA_INT,
        visit_times=visit_times,
    )

    # ---- 2. Plots ----
    plot_pdp(results, ages, masks, times, bmi_values,
             save_path=f"{args.prefix}_by_age{suffix}_{model_tag}.png",
             visit_times=visit_times)
    plot_pdp_marginal(results, masks, times, bmi_values,
                      save_path=f"{args.prefix}_marginal{suffix}_{model_tag}.png",
                      visit_times=visit_times)

    # ---- 3. Estimated delta_PDP (marginal) ----
    estimated, true_ref = compute_delta_pdp(
        results, ages, masks, times,
        bmi_lo=20, bmi_hi=35,
        true_beta_bmi=TRUE_BETA_BMI, true_beta_int=TRUE_BETA_INT,
        visit_times=visit_times,
    )

    # ---- 4. Estimated delta_PDP (stratified by age) ----
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
    print(f"  Model type: {'hybrid (beta*BMI + h(z))' if is_hybrid else 'pure NN (rho(z)*beta_neural)'}")
    print(f"  BMI mode: {args.bmi_mode}")
    print(f"  True beta_BMI={TRUE_BETA_BMI}, beta_int={TRUE_BETA_INT}")

    if is_hybrid and beta_names is not None:
        # For hybrid: compare parametric-only vs full (parametric+neural) delta_PDP
        beta = None
        if hasattr(model.decoder, 'beta') and model.decoder.beta is not None:
            beta = model.decoder.beta.detach().cpu()
        elif hasattr(model.decoder, '_last_beta') and model.decoder._last_beta is not None:
            beta = model.decoder._last_beta.detach().cpu()

        if beta is not None and "BMI_t" in beta_names:
            bmi_idx = beta_names.index("BMI_t")
            beta_bmi = beta[bmi_idx].item()
            int_name = "BMI_txAGEc"
            if int_name in beta_names:
                int_idx = beta_names.index(int_name)
                beta_int_est = beta[int_idx].item()
                mean_age = ages.numpy().mean() if hasattr(ages, 'numpy') else float(ages.mean())
                param_only_delta = delta_v * (beta_bmi + beta_int_est * mean_age)
                print(f"  Estimated beta_BMI = {beta_bmi:.4f} (true = {TRUE_BETA_BMI})")
                print(f"  Estimated beta_int = {beta_int_est:.4f} (true = {TRUE_BETA_INT})")
                print(f"  Parametric-only delta_PDP = {param_only_delta:.4f}")
            else:
                print(f"  Estimated beta_BMI = {beta_bmi:.4f} (true = {TRUE_BETA_BMI})")
                print(f"  Parametric-only delta_PDP = {beta_bmi * delta_v:.4f}")

    # Print estimated vs true at each visit time
    if estimated:
        print(f"\n  {'Time':>6s}  {'Estimated':>10s}  {'True':>10s}  {'Bias':>10s}")
        print(f"  {'-'*42}")
        for vt in sorted(estimated.keys()):
            est = estimated[vt]
            tru = true_ref[vt]
            print(f"  {vt:6.0f}  {est:+10.4f}  {tru:+10.4f}  {est-tru:+10.4f}")

    print(f"\nDone.")