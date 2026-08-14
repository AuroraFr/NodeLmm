"""
Runner for continuous-time PDP analysis — Neural ODE-LMM on 3C cohort.

v3: adds --zero_skip for pure path-dependence testing (group lasso models).

Usage:
    python PDP_ode_real_continuous.py --delta_method
    python PDP_ode_real_continuous.py --delta_method --zero_skip   # ODE-only
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import os

from Preprocess_3C import process_data
from train_ODE_real import RealDataset, collate_real
from model_ODE_real import NeuralODEModel, NeuralODEConfig
from PDP_continuous_time import (
    make_eval_grid,
    make_profiles_continuous,
    compute_trajectory_profile_pdp_continuous,
    plot_trajectory_profile_pdp_continuous,
    plot_trajectory_profile_pdp_delta,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ════════════════════════════════════════════════════════════════════
#  Skip-zeroing context manager (works with any reg_mode)
# ════════════════════════════════════════════════════════════════════

class ZeroSkipContext:
    """
    Temporarily zero the skip input in the decoder.

    Works with group_lasso, skip_gate, or no regularisation.
    The rho_net still receives the correct input dimension
    (latent_dim + skip_dim), but all skip columns are zero —
    so the prediction depends only on z(t).
    """
    def __init__(self, model):
        self.decoder = model.decoder
        self._original = None

    def __enter__(self):
        original_fn = self.decoder._build_skip_input

        def _zero_skip(x_interp, mask, static, N, T):
            skip = original_fn(x_interp, mask, static, N, T)
            if skip is not None:
                return torch.zeros_like(skip)
            return skip

        self._original = original_fn
        self.decoder._build_skip_input = _zero_skip
        return self

    def __exit__(self, *args):
        self.decoder._build_skip_input = self._original


# ════════════════════════════════════════════════════════════════════
#  Wald test pairs
# ════════════════════════════════════════════════════════════════════

# Standard contrasts (as in the paper)
STANDARD_PAIRS = [
    ("late_decline", "late_spike"),
    ("stable_high", "stable_low"),
    ("gradual_decline", "gradual_rise"),
]

# Matched-endpoint pairs for pure path-dependence
MATCHED_ENDPOINT_PAIRS = [
    ("gradual_rise", "late_spike"),         # both Q25 → Q75
    ("gradual_decline", "late_decline"),     # both Q75 → Q25
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continuous-time PDP analysis — Neural ODE-LMM on 3C")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/cv_final_model.pt")
    parser.add_argument("--data", type=str,
                        default="3C_dataset/train_3C_data_1.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_points", type=int, default=13,
                        help="Number of grid points")
    parser.add_argument("--t_max", type=float, default=12.0,
                        help="Maximum time (years)")
    parser.add_argument("--prefix", type=str,
                        default="figures/pdp_real3C_cv")

    # Delta-method options
    parser.add_argument("--delta_method", action="store_true",
                        help="Compute delta-method CI (requires Fisher)")
    parser.add_argument("--fisher_max", type=int, default=None,
                        help="Max subjects for Fisher (None = all)")
    parser.add_argument("--fisher_cache", type=str, default=None,
                        help="Path to save/load cached Fisher inverse")
    parser.add_argument("--damping", type=float, default=1e-4,
                        help="Marquardt damping for Fisher inversion")

    # Skip ablation
    parser.add_argument("--zero_skip", action="store_true",
                        help="Zero all skip inputs — predictions from z(t) only. "
                             "Use with matched-endpoint pairs for pure "
                             "path-dependence testing.")

    # Gate ablation (skip_gate models only)
    parser.add_argument("--zero_dynamic_gates", action="store_true",
                        help="Hard-zero all dynamic covariate gates before PDP")
    parser.add_argument("--zero_gates_list", type=str, nargs="*", default=None,
                        help="Zero specific gates by name (e.g. HDL BMI)")

    # Wald test options
    parser.add_argument("--matched_endpoint_wald", action="store_true",
                        help="Include matched-endpoint Wald tests "
                             "(gradual_rise vs late_spike, etc.)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ─────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    ckpt_cfg = checkpoint['config']

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  epoch     = {checkpoint.get('epoch', '?')}")
    # print(f"  test loss = {checkpoint.get('best_test_loss', '?'):.4f}")
    print(f"  reg_mode  = {ckpt_cfg.get('reg_mode', None)}")
    print(f"  lambda_reg= {ckpt_cfg.get('lambda_reg', None)}")

    # ── Feature definitions ─────────────────────────────────────────────
    id_col = "NUM_ID"
    target_col = "ISA15"
    time_varying_features = ckpt_cfg.get('time_varying_features',
                                          ["BMI", "PAS", "PAD", "GLUC", "HDL"])
    static_features = ckpt_cfg.get('static_features',
                                    ["SEX_code", "AGEc", "DIPNIV_2", "DIPNIV_3"])
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
        use_rho_norm=ckpt_cfg.get('use_rho_norm', True),
    )

    model = NeuralODEModel(
        n_tv=K, static_dim=Ks, cfg=cfg,
        use_rho_net=True, use_neural_re=True,
        g_hidden=8, fullD=False,
        cov_means=cov_means, cov_stds=cov_stds,
        use_dynamic_skip=ckpt_cfg.get('use_dynamic_skip', True),
        static_skip_dims=list(range(Ks)),
        reg_mode=ckpt_cfg.get('reg_mode', None),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    # ── Gate ablation (skip_gate models) ────────────────────────────────
    gate_logit = getattr(model.decoder, 'skip_gate_logit', None)
    if gate_logit is not None:
        all_gate_names = time_varying_features + static_features
        original_gates = torch.sigmoid(gate_logit).detach().cpu()
        print(f"\n  Original gate values:")
        for g, name in enumerate(all_gate_names):
            print(f"    {name:>8s}: σ(α) = {original_gates[g]:.4f}")

        if args.zero_dynamic_gates:
            n_dynamic = len(time_varying_features)
            with torch.no_grad():
                gate_logit.data[:n_dynamic] = -50.0
            print(f"\n  *** ALL DYNAMIC GATES HARD-ZEROED ***")
            zeroed_gates = torch.sigmoid(gate_logit).detach().cpu()
            for g, name in enumerate(all_gate_names):
                print(f"    {name:>8s}: σ(α) = {zeroed_gates[g]:.6f}")

        elif args.zero_gates_list:
            with torch.no_grad():
                for name in args.zero_gates_list:
                    if name in all_gate_names:
                        idx = all_gate_names.index(name)
                        gate_logit.data[idx] = -50.0
                        print(f"\n  *** Gate {name} (idx={idx}) HARD-ZEROED ***")
                    else:
                        print(f"\n  WARNING: {name} not in {all_gate_names}")
            zeroed_gates = torch.sigmoid(gate_logit).detach().cpu()
            print(f"\n  Gate values after ablation:")
            for g, name in enumerate(all_gate_names):
                print(f"    {name:>8s}: σ(α) = {zeroed_gates[g]:.6f}")

    # ── Zero-skip mode announcement ─────────────────────────────────────
    if args.zero_skip:
        print(f"\n  *** ZERO-SKIP MODE: predictions from z(t) only ***")
        print(f"      All skip inputs zeroed at evaluation time.")
        print(f"      Matched-endpoint contrasts test pure ODE path-dependence.")

    # ── Evaluation grid ─────────────────────────────────────────────────
    eval_grid = make_eval_grid(t_max=args.t_max, n_points=args.n_points)
    print(f"\n{'='*60}")
    print(f"CONTINUOUS-TIME PDP ANALYSIS")
    print(f"  Grid: {args.n_points} points on [0, {args.t_max}]")
    print(f"  Delta-method CI: {'ON' if args.delta_method else 'OFF'}")
    print(f"  Zero-skip:       {'ON' if args.zero_skip else 'OFF'}")
    print(f"  Matched-endpoint Wald: "
          f"{'ON' if args.matched_endpoint_wald else 'OFF'}")
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
        "GLUC": np.array([4, 5, 6, 7, 8, 10]),
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

        if args.fisher_cache and os.path.exists(args.fisher_cache):
            print(f"\n  Loading cached Fisher inverse from {args.fisher_cache}")
            fisher_inv = torch.from_numpy(np.load(args.fisher_cache)).float()
        else:
            print(f"\n{'='*60}")
            print(f"COMPUTING EMPIRICAL FISHER INFORMATION")
            print(f"{'='*60}")

            lambda_reg = ckpt_cfg.get('lambda_reg', 0.1)
            lambda_wd = ckpt_cfg.get('lambda_wd', 1e-5)

            fisher, scores = compute_empirical_fisher(
                model, dataset, device, collate_fn=collate_real,
                lambda_reg=lambda_reg,
                weight_decay=lambda_wd,
                max_subjects=args.fisher_max,
                verbose=True,
            )

            _, fisher_inv = _regularise_and_invert(
                fisher, "F", LAMBDA=args.damping, verbose=True,
            )

            if scores.shape[0] > 0:
                mean_score = scores.mean(dim=0)
                print(f"  Stationarity check: ||mean(φ)|| = "
                      f"{mean_score.norm().item():.4e}")

            if args.fisher_cache:
                np.save(args.fisher_cache, fisher_inv.numpy())
                print(f"  Fisher inverse cached to {args.fisher_cache}")

    # ── Determine which Wald pairs to test ──────────────────────────────
    wald_pairs = list(STANDARD_PAIRS)
    if args.matched_endpoint_wald or args.zero_skip:
        wald_pairs += MATCHED_ENDPOINT_PAIRS
        print(f"\n  Wald test pairs: standard + matched-endpoint")
    else:
        print(f"\n  Wald test pairs: standard only")

    # ── Suffix for filenames ────────────────────────────────────────────
    suffix = "_zeroskip" if args.zero_skip else ""

    # ── Run analyses ────────────────────────────────────────────────────
    skip_ctx = ZeroSkipContext(model) if args.zero_skip else None

    for col_idx, feat_name in enumerate(time_varying_features):
        if feat_name not in DELTA_RANGES:
            continue

        val_lo, val_hi = DELTA_RANGES[feat_name]

        print(f"\n{'='*60}")
        mode_str = " [ODE-only, skip=0]" if args.zero_skip else ""
        print(f"PDP ANALYSIS: {feat_name} (col={col_idx}){mode_str}")
        print(f"{'='*60}")

        profiles = make_profiles_continuous(eval_grid, v_lo=val_lo, v_hi=val_hi)

        # ── 1. Cross-subject SE ─────────────────────────────────────────
        if args.zero_skip:
            with ZeroSkipContext(model):
                traj_results, _, n_subj = compute_trajectory_profile_pdp_continuous(
                    model, eval_loader, device, profiles, eval_grid,
                    target_col=col_idx, n_tv=K,
                    mask_type=mask_type, target_name=feat_name,
                )
        else:
            traj_results, _, n_subj = compute_trajectory_profile_pdp_continuous(
                model, eval_loader, device, profiles, eval_grid,
                target_col=col_idx, n_tv=K,
                mask_type=mask_type, target_name=feat_name,
            )

        plot_trajectory_profile_pdp_continuous(
            traj_results, eval_grid,
            save_path=f"{args.prefix}_{feat_name}_traj_profile{suffix}.png",
            target_name=f"{feat_name}{' (ODE only)' if args.zero_skip else ''}",
            visit_times=None,
        )

        # ── 2. Delta-method CI ──────────────────────────────────────────
        if args.delta_method and fisher_inv is not None:
            from PDP_variance import compute_trajectory_profile_pdp_with_ci
            from PDP_continuous_time import (
                plot_delta_profile_pdp_delta,
                plot_all_pairwise_delta_pdp,
            )

            print(f"\n  Computing delta-method CI for {feat_name}{mode_str} ...")

            if args.zero_skip:
                with ZeroSkipContext(model):
                    ci_results = compute_trajectory_profile_pdp_with_ci(
                        model, eval_loader, device, profiles, eval_grid,
                        fisher_inv=fisher_inv,
                        target_col=col_idx, n_tv=K,
                        mask_type=mask_type,
                        target_name=feat_name,
                        verbose=True,
                    )
            else:
                ci_results = compute_trajectory_profile_pdp_with_ci(
                    model, eval_loader, device, profiles, eval_grid,
                    fisher_inv=fisher_inv,
                    target_col=col_idx, n_tv=K,
                    mask_type=mask_type,
                    target_name=feat_name,
                    verbose=True,
                )

            # ── Wald tests ──────────────────────────────────────────────
            from pdp_real_wald_test import wald_test_all_pairs

            wald_results = wald_test_all_pairs(
                ci_results, eval_grid, fisher_inv,
                late_cutoff=7.0,
                pairs=wald_pairs,
                verbose=True,
            )

            # ── Profile PDP with CI bands ───────────────────────────────
            plot_trajectory_profile_pdp_delta(
                ci_results, eval_grid,
                save_path=f"{args.prefix}_{feat_name}_traj_profile_delta{suffix}.png",
                target_name=f"{feat_name}{' (ODE only)' if args.zero_skip else ''}",
                visit_times=None,
                n_subjects=n_subj,
            )

            # ── Pairwise ΔPDP plots ─────────────────────────────────────
            if "late_decline" in ci_results and "late_spike" in ci_results:
                plot_delta_profile_pdp_delta(
                    ci_results, eval_grid,
                    profile_a="late_decline", profile_b="late_spike",
                    save_path=f"{args.prefix}_{feat_name}_delta_eb_vs_ls{suffix}.png",
                    target_name=feat_name,
                    visit_times=None,
                    n_subjects=n_subj,
                    fisher_inv=fisher_inv,
                )

            # All pairwise
            plot_all_pairwise_delta_pdp(
                ci_results, eval_grid,
                save_path=f"{args.prefix}_{feat_name}_delta_all_pairs{suffix}.png",
                target_name=feat_name,
                visit_times=None,
                n_subjects=n_subj,
                fisher_inv=fisher_inv,
            )

            # ── Matched-endpoint ΔPDP plots ─────────────────────────────
            for pa, pb in MATCHED_ENDPOINT_PAIRS:
                if pa in ci_results and pb in ci_results:
                    plot_delta_profile_pdp_delta(
                        ci_results, eval_grid,
                        profile_a=pa, profile_b=pb,
                        save_path=f"{args.prefix}_{feat_name}_delta_{pa}_vs_{pb}{suffix}.png",
                        target_name=feat_name,
                        visit_times=None,
                        n_subjects=n_subj,
                        fisher_inv=fisher_inv,
                    )

            # ── SE comparison ───────────────────────────────────────────
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
    if args.zero_skip:
        print(f"  (zero-skip mode: all outputs have '{suffix}' suffix)")