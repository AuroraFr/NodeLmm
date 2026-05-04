"""
Runner script for Scenario 5 trajectory-profile PDP diagnostic.

Usage:
    python run_pdp_profiles.py
    python run_pdp_profiles.py --checkpoint checkpoints/best_model_ode_full_skip_0.pt
    python run_pdp_profiles.py --bmi_lo 22 --bmi_hi 30
    python run_pdp_profiles.py --ablation   # run skip ablation test
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pyreadr
import argparse
import os

from dataset import LongitudinalDataset, collate_pad
from model_ODE_torchdiff import NeuralODEModel, NeuralODEConfig
from PDP_analysis_profiles_ODE import (
    compute_pdp_profiles,
    compute_profile_diagnostic,
    compute_skip_ablation,
    _get_closest_preds_windowed,
    plot_profiles,
    plot_pdp_profiles,
    plot_skip_ablation,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S5 profile PDP diagnostic")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/simulation_cumulative_effect_diagoD_noBMIInEncoder_norhonorm/best_model_ode_1.pt")
    parser.add_argument("--data", type=str,
                        default="simu_datasets/S5_sims/sim_002.rds")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bmi_lo", type=float, default=20.0)
    parser.add_argument("--bmi_hi", type=float, default=28.0)
    parser.add_argument("--true_coeff", type=float, default=-0.05,
                        help="True h5 coefficient: h5(t) = coeff * integral BMI(tau) dtau")
    parser.add_argument("--prefix", type=str, default="figures/s5_profiles_noskip")
    parser.add_argument("--ablation", action="store_true",
                        help="Run skip ablation test")
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    time_col = "time"
    y_col = "ISA15_sim"
    id_col = "NUM_ID"
    x_cols = ["BMI_t"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    print(f"Loading data from {args.data}...")
    df = next(iter(pyreadr.read_r(args.data).values()))
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col,
                                  static_cols=static_cols)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_pad)

    # ---- Model ----
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
        ode_solver='rk4'
    )

    model = NeuralODEModel(
        x_dim=len(x_cols),
        static_dim=len(static_cols),
        cfg=cfg,
        n_tv=1,
        use_rho_net=True,
        use_neural_re=True,
        re_spline_cols=None,
        g_hidden=16,
        fullD=False,
        bmi_mean=0.0,
        bmi_std=1.0,
        use_bmi_skip=False,
        static_skip_dims=None,
        reg_mode=None
    ).to(device)

    # ---- Load checkpoint ----
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"  epoch = {checkpoint.get('epoch', '?')}")
        print(f"  loss  = {checkpoint.get('best_test_loss', '?')}")
    else:
        model.load_state_dict(checkpoint, strict=False)

    print(f"  bmi_mean = {model.decoder.bmi_mean.item():.4f}")
    print(f"  bmi_std  = {model.decoder.bmi_std.item():.4f}")
    model.eval()

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

    # ---- Visit times (adapt to your simulation) ----
    visit_times = np.array([0, 4, 8, 10])

    # ================================================================
    # 1. Compute PDP for all profiles
    # ================================================================
    print(f"\n{'='*70}")
    print(f"TRAJECTORY-PROFILE PDP ANALYSIS (Scenario 5)")
    print(f"  bmi_lo={args.bmi_lo}, bmi_hi={args.bmi_hi}")
    print(f"  true h5 coeff = {args.true_coeff}")
    print(f"{'='*70}")

    results, profiles, integrals, ages, masks, times = compute_pdp_profiles(
        model, eval_loader, device,
        bmi_lo=args.bmi_lo, bmi_hi=args.bmi_hi,
    )

    # ================================================================
    # 2. Visualize the profiles themselves
    # ================================================================
    # plot_profiles(profiles, times, masks,
    #               save_path=f"{args.prefix}_shapes.png")

    # ================================================================
    # 3. Plot PDP results
    # ================================================================
    plot_pdp_profiles(results, masks, times, profiles,
                      save_path=f"{args.prefix}_pdp.png",
                      visit_times=visit_times)

    # ================================================================
    # 4. Run the core diagnostic
    # ================================================================
    diagnostic = compute_profile_diagnostic(
        results, integrals, masks, times, ages,
        visit_times=visit_times,
        true_coeff=args.true_coeff,
    )

    # ================================================================
    # 5. Skip ablation (optional)
    # ================================================================
    if args.ablation:
        print(f"\n{'='*70}")
        print(f"SKIP ABLATION TEST")
        print(f"{'='*70}")

        results_full, results_ablated, abl_profiles, abl_masks, abl_times = \
            compute_skip_ablation(
                model, eval_loader, device,
                bmi_lo=args.bmi_lo, bmi_hi=args.bmi_hi,
            )

        # Print comparison
        print(f"\n  Full vs Ablated predictions at last visit:")
        last_vt = visit_times[-1]
        masks_np = abl_masks.numpy()
        times_np = abl_times.numpy()

        print(f"  {'Profile':>20s}  {'Full':>10s}  {'Ablated':>10s}  {'Δ(F−A)':>10s}")
        print(f"  {'-'*55}")
        for prof_name in results_full:
            mu_full = results_full[prof_name].numpy()
            mu_abl = results_ablated[prof_name].numpy()

            preds_full = []
            preds_abl = []
            for i in range(mu_full.shape[0]):
                obs_idx = np.where(masks_np[i] > 0.5)[0]
                if len(obs_idx) == 0:
                    continue
                obs_times = times_np[i, obs_idx]
                closest = obs_idx[np.argmin(np.abs(obs_times - last_vt))]
                preds_full.append(mu_full[i, closest])
                preds_abl.append(mu_abl[i, closest])

            pf = np.mean(preds_full)
            pa = np.mean(preds_abl)
            print(f"  {prof_name:>20s}  {pf:10.3f}  {pa:10.3f}  {pf - pa:+10.3f}")

        plot_skip_ablation(results_full, results_ablated, abl_masks, abl_times,
                           save_path=f"{args.prefix}_ablation.png",
                           visit_times=visit_times)

    # ================================================================
    # Summary — all visit times
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY — ALL VISIT TIMES")
    print(f"{'='*70}")

    if diagnostic:
        # Header
        print(f"\n  {'t':>4s}  {'late_spike':>12s}  {'late_decline':>14s}  "
              f"{'Δ(EB−LS)':>10s}  {'Signal':>14s}")
        print(f"  {'-'*60}")

        for vt in sorted(diagnostic.keys()):
            d = diagnostic[vt]
            print(f"  {vt:4.0f}  {d['late_spike']:12.3f}  {d['late_decline']:14.3f}  "
                  f"{d['delta']:+10.3f}  {d['signal']:>14s}")

        # Oracle comparison table: ΔPDP (stable_high - stable_low) vs true
        print(f"\n  {'─'*70}")
        print(f"  ΔPDP (stable_high − stable_low) vs oracle:")
        print(f"  {'t':>4s}  {'Model ΔPDP':>12s}  {'True ΔPDP':>12s}  {'Bias':>10s}")
        print(f"  {'-'*45}")

        masks_np = masks.numpy()
        times_np = times.numpy()
        for vt in visit_times:
            preds_hi, times_hi = _get_closest_preds_windowed(
                results["stable_high"].numpy(), masks_np, times_np, vt)
            preds_lo, times_lo = _get_closest_preds_windowed(
                results["stable_low"].numpy(), masks_np, times_np, vt)

            if len(preds_hi) > 10:
                model_delta = preds_hi.mean() - preds_lo.mean()

                # Oracle at each subject's ACTUAL observation time
                oracle_hi = args.true_coeff * args.bmi_hi * times_hi
                oracle_lo = args.true_coeff * args.bmi_lo * times_lo
                true_delta = oracle_hi.mean() - oracle_lo.mean()

                bias = model_delta - true_delta
                print(f"  t={vt:5.1f}  n={len(preds_hi):4d}  "
                    f"Model={model_delta:+.3f}  Oracle={true_delta:+.3f}  "
                    f"Bias={bias:+.3f}")

        # Final interpretation
        latest = max(diagnostic.keys())
        d = diagnostic[latest]
        print()
        if d['signal'] == "CUMULATIVE":
            print(f"  ✓ Model learned cumulative BMI effect through z(t)")
            print(f"    (early burden → worse cognition, as in true S5 DGP)")
        elif d['signal'] == "INSTANTANEOUS":
            print(f"  ✗ Model is using the instantaneous BMI skip connection")
            print(f"    (late spike → worse cognition, opposite of true S5 DGP)")
            print(f"    Consider: removing skip, increasing ODE capacity, or longer training")
        else:
            print(f"  ? Ambiguous — model may partially use both pathways")
            print(f"    Run --ablation to check skip contribution")

    print(f"\nDone. Figures saved to {args.prefix}_*.png")