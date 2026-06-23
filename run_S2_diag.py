"""
Example: diagnose what the ODE learned in Scenario S2 (cumulative BMI burden).

DGP: h_s = -0.05 * ∫₀ᵗ BMI(τ) dτ

Expected diagnostic results:
  - Current-value test:   ||Δz|| > 0  (history matters)
  - Path-dependence test: ||Δz|| > 0  (different paths → different ∫BMI)
  - Baseline test:        ||Δz|| ≈ 0  early on (same BMI → same z)

Usage:
    python run_diagnostic_S2.py --sim_idx 0 --ckpt_path checkpoints/best_model_ode_0.pt
"""
import argparse
import torch
import numpy as np
import pyreadr
from torch.utils.data import DataLoader

from dataset import LongitudinalDataset, collate_pad
from model_ODE_cumulative import NeuralODEModel, NeuralODEConfig
from ODE_diagnostic_simu import (
    extract_zt_profiles,
    plot_zt_diagnostic,
    plot_pairwise_diagnostic,
    print_diagnostic_summary,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_idx", type=int, default=0)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--q25", type=float, default=23.1,
                        help="BMI 25th percentile")
    parser.add_argument("--q75", type=float, default=28.4,
                        help="BMI 75th percentile")
    parser.add_argument("--max_subjects", type=int, default=5859,
                        help="Cap subjects for speed (None=all)")
    parser.add_argument("--save_dir", type=str, default="diagnostics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Paths ---
    if args.data_path is None:
        args.data_path = f"simu_datasets/S2a_sims/sim_{args.sim_idx + 1:03d}.rds"
    if args.ckpt_path is None:
        args.ckpt_path = f"checkpoints/simulation_cumulative_effect_diagoD_skipgate_norhonorm/best_model_ode_{args.sim_idx}.pt"

    # --- Data ---
    df = next(iter(pyreadr.read_r(args.data_path).values()))
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    time_col, y_col, id_col = "time", "ISA15_sim", "NUM_ID"
    x_cols = ["BMI_t"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col,
                                  static_cols=static_cols)
    loader = DataLoader(dataset, batch_size=64, shuffle=False,
                        collate_fn=collate_pad)

    # --- Model ---
    # Adjust config to match your training setup
    cfg = NeuralODEConfig(
        hidden_channels=8, enc_mlp_hidden=16, func_mlp_hidden=16,
        dec_rho_hidden=16, dec_p=4, dec_q=3, depth=2, dropout=0.0,
        euler_steps_per_interval=4,
    )
    model = NeuralODEModel(
        x_dim=len(x_cols), static_dim=len(static_cols), cfg=cfg,
        n_tv=1, use_rho_net=True, use_neural_re=True,
        re_spline_cols=None, g_hidden=8, fullD=False,
        bmi_mean=0.0, bmi_std=1.0, static_skip_dims=[1], use_bmi_skip=True,
        reg_mode='skip_gate',
    ).to(device)

    checkpoint = torch.load(args.ckpt_path, map_location=device,
                            weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    print(f"Loaded: {args.ckpt_path}")

    # --- Visit times ---
    visit_times = np.array([0, 2, 4, 6, 8, 10, 12])

    # ─────────────────────────────────────────────────
    # Step 1: Extract z(t) under counterfactual profiles
    # ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXTRACTING z(t) UNDER COUNTERFACTUAL BMI PROFILES")
    print("=" * 60)

    zt_dict, pdp_dict, profiles = extract_zt_profiles(
        model, loader, device,
        visit_times=visit_times,
        q25=args.q25,
        q75=args.q75,
        covariate_idx=0,
        max_subjects=args.max_subjects,
    )

    # ─────────────────────────────────────────────────
    # Step 2: Quantitative diagnostic
    # ─────────────────────────────────────────────────
    print_diagnostic_summary(zt_dict, visit_times)

    # ─────────────────────────────────────────────────
    # Step 3: Compute oracle ∫BMI for each profile
    #         to verify z(t) tracks the integral
    # ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ORACLE CUMULATIVE BMI FOR EACH PROFILE")
    print("=" * 60)
    print(f"  DGP: h_s = -0.05 * ∫₀ᵗ BMI(τ) dτ")
    print(f"  If ODE learned cumulative effect, z(t) should correlate")
    print(f"  with ∫BMI across profiles.\n")

    t = np.array(visit_times)
    for name, bmi_vals in profiles.items():
        # Trapezoidal integral of BMI profile
        cum_bmi = np.zeros_like(t, dtype=float)
        for ell in range(1, len(t)):
            dt = t[ell] - t[ell - 1]
            cum_bmi[ell] = cum_bmi[ell - 1] + 0.5 * (bmi_vals[ell - 1] + bmi_vals[ell]) * dt
        oracle_effect = -0.05 * cum_bmi

        # Mean z norm at each time
        z_mean = zt_dict[name].mean(axis=0)  # (L, d)
        z_norm = np.sqrt((z_mean ** 2).sum(axis=1))  # (L,)

        print(f"  {name}:")
        for ell, vt in enumerate(visit_times):
            print(f"    t={vt:5.1f}: ∫BMI={cum_bmi[ell]:8.1f}, "
                  f"oracle_h={oracle_effect[ell]:+7.2f}, "
                  f"||z||={z_norm[ell]:.4f}")

    # ─────────────────────────────────────────────────
    # Step 4: Correlation between z(t) and ∫BMI
    # ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CORRELATION: z_k(t) vs ∫BMI ACROSS PROFILES")
    print("=" * 60)

    d = zt_dict[list(zt_dict.keys())[0]].shape[2]

    # At each time, collect (profile_mean_z_k, oracle_cum_bmi) pairs
    for ell, vt in enumerate(visit_times):
        if vt == 0:
            continue  # all ∫BMI = 0 at t=0

        cum_vals = []
        z_vals = []
        for name, bmi_vals in profiles.items():
            # Oracle cumulative
            cum = np.trapz(bmi_vals[:ell + 1], t[:ell + 1])
            cum_vals.append(cum)
            z_vals.append(zt_dict[name][:, ell, :].mean(axis=0))  # (d,)

        cum_arr = np.array(cum_vals)  # (6,)
        z_arr = np.array(z_vals)      # (6, d)

        # Correlation of each z_k with ∫BMI across the 6 profiles
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
    # Step 5: Plots
    # ─────────────────────────────────────────────────
    import os
    os.makedirs(args.save_dir, exist_ok=True)

    save1 = os.path.join(args.save_dir, f"zt_all_S2_sim{args.sim_idx}.png")
    plot_zt_diagnostic(zt_dict, pdp_dict, profiles, visit_times,
                       save_path=save1)

    save2 = os.path.join(args.save_dir, f"zt_pairs_S2_sim{args.sim_idx}.png")
    plot_pairwise_diagnostic(zt_dict, visit_times, save_path=save2)

    print(f"\nDone. Plots saved to {args.save_dir}/")


if __name__ == "__main__":
    main()