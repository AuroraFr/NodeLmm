"""
Monte Carlo CI for ∆PDP — Neural ODE-LMM.

Validation counterpart to the delta method in DPDP_cumulative_effect.py.
Draws θ⁽ᵐ⁾ ~ N(θ̂, F⁻¹) and evaluates ∆PDP(t; θ⁽ᵐ⁾) for each draw,
then reports pointwise quantile-based CIs.

If the linearization g(θ)ᵀ F⁻¹ g(θ) is accurate, MC CIs ≈ delta CIs.

Reference:
    Delta method:   Var(∆PDP) = ∇_θ ∆PDP ᵀ  F⁻¹  ∇_θ ∆PDP
    MC method:      Draw θ⁽ᵐ⁾ ~ N(θ̂, F⁻¹), m=1,...,M
                    Compute ∆PDP(t; θ⁽ᵐ⁾) for each draw
                    CI = pointwise [2.5%, 97.5%] quantiles
"""
from __future__ import annotations
import time
import torch
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────
# Helpers (duplicated for standalone use)
# ─────────────────────────────────────────────────────────

def _param_list(model):
    return [p for p in model.parameters() if p.requires_grad]


def _param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────
# 1. Cholesky with PD projection fallback
# ─────────────────────────────────────────────────────────

def _safe_cholesky(fisher_inv, verbose=True):
    """
    Compute L such that L Lᵀ = F⁻¹, with eigenvalue clipping fallback.

    F⁻¹ is theoretically PD but may have small negative eigenvalues
    from float32 accumulation in the inversion step.  The fallback
    projects onto the PD cone by clamping eigenvalues to a relative
    floor (λ_max · 1e-10), which is the minimal perturbation that
    restores Cholesky factorizability.
    """
    P = fisher_inv.shape[0]
    F_cpu = fisher_inv.cpu().double()
    F_cpu = (F_cpu + F_cpu.T) / 2              # enforce exact symmetry

    try:
        L = torch.linalg.cholesky(F_cpu)
        if verbose:
            print(f"    Cholesky succeeded in float64")
    except torch.linalg.LinAlgError:
        eigvals, eigvecs = torch.linalg.eigh(F_cpu)
        n_neg = (eigvals < 0).sum().item()
        if verbose:
            print(f"    Fisher⁻¹ not PD: {n_neg}/{P} negative eigenvalues")
            print(f"    eigenvalue range: [{eigvals.min().item():.2e}, "
                  f"{eigvals.max().item():.2e}]")
        floor = eigvals.max().item() * 1e-10
        eigvals = eigvals.clamp(min=floor)
        F_pd = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        F_pd = (F_pd + F_pd.T) / 2
        L = torch.linalg.cholesky(F_pd)
        if verbose:
            print(f"    Projected onto PD cone (floor = {floor:.2e})")

    return L.float()


# ─────────────────────────────────────────────────────────
# 2. Evaluate ∆PDP(t; θ) for current model parameters
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate_delta_pdp(model, dataset, device, collate_fn,
                        bmi_lo, bmi_hi, t_eval):
    """
    Compute ∆PDP(t) = (1/N) Σ_i [μ_i(t; BMI=hi) − μ_i(t; BMI=lo)]
    at the current parameter values.

    Uses the same continuous-time eval grid approach as
    compute_delta_pdp_gradients_continuous() in DPDP_cumulative_effect.py,
    but without backward passes (no gradients needed).

    Returns:
        delta_pdp: (L,) numpy array of ∆PDP at each eval time
    """
    from torch.utils.data import DataLoader

    model.eval()
    L = len(t_eval)

    # --- Build shared eval-only grid ---
    t_grid_np = np.array(sorted(set([0.0] + list(map(float, t_eval)))),
                         dtype=np.float32)
    T_grid = len(t_grid_np)
    eval_indices = [int(np.argmin(np.abs(t_grid_np - float(te))))
                    for te in t_eval]

    # --- Scale substeps to match training ---
    orig_substeps = getattr(model.cfg, 'euler_steps_per_interval', None)
    model.cfg.euler_steps_per_interval = 4

    # --- Pre-extract statics ---
    single_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                               collate_fn=collate_fn)
    all_statics = []
    x_dim = None
    for batch in single_loader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        all_statics.append(s.squeeze(0))
        if x_dim is None:
            x_dim = x_pad.shape[2]
    all_statics = torch.stack(all_statics)
    N = len(all_statics)

    # --- Batched forward ---
    batch_size = 64
    t_grid_1 = torch.tensor(t_grid_np, dtype=torch.float32,
                            device=device).unsqueeze(0)
    delta_accum = np.zeros(L)
    N_total = 0

    n_batches = (N + batch_size - 1) // batch_size
    for b_idx in range(n_batches):
        start = b_idx * batch_size
        end = min(start + batch_size, N)
        B = end - start

        s_batch = all_statics[start:end].to(device)
        t_batch = t_grid_1.expand(B, -1)
        obs_mask_batch = torch.ones(B, T_grid, device=device)

        x_hi = torch.zeros(B, T_grid, x_dim, device=device)
        x_lo = torch.zeros(B, T_grid, x_dim, device=device)
        x_hi[:, :, 0] = bmi_hi
        x_lo[:, :, 0] = bmi_lo

        mu_hi, _, _, _, _, _ = model(
            t_batch, x_hi, masks=None,
            static_covariates=s_batch, bmi_t=x_hi[:, :, 0:1],
            obs_mask=obs_mask_batch,
        )
        mu_lo, _, _, _, _, _ = model(
            t_batch, x_lo, masks=None,
            static_covariates=s_batch, bmi_t=x_lo[:, :, 0:1],
            obs_mask=obs_mask_batch,
        )

        for k, aug_idx in enumerate(eval_indices):
            delta_accum[k] += (mu_hi[:, aug_idx] - mu_lo[:, aug_idx]).sum().item()

        N_total += B

    # Restore substeps
    model.cfg.euler_steps_per_interval = orig_substeps

    return delta_accum / N_total


# ─────────────────────────────────────────────────────────
# 3. MC ∆PDP CI
# ─────────────────────────────────────────────────────────

def mc_delta_pdp_ci(
    model, dataset, device, collate_fn,
    fisher_inv,
    bmi_lo=20.0, bmi_hi=35.0,
    t_eval=None,
    oracle_fn=None,
    n_mc=500, seed=42,
    verbose=True,
):
    """
    Monte Carlo CI for ∆PDP(t).

    Method:
      1. Save θ̂ = current parameters
      2. Cholesky factor: L Lᵀ = F⁻¹
      3. For m = 1, ..., M:
         a. Draw z ~ N(0, I),  set θ⁽ᵐ⁾ = θ̂ + L z
         b. Load θ⁽ᵐ⁾ into model
         c. Compute ∆PDP(t; θ⁽ᵐ⁾) via _evaluate_delta_pdp
      4. Restore θ̂
      5. Report pointwise quantiles and SE

    Args:
        model:       NeuralODEModel (will be temporarily modified)
        dataset:     LongitudinalDataset
        device:      torch device
        collate_fn:  collation function
        fisher_inv:  (P, P) inverse Fisher matrix (from delta method pipeline)
        bmi_lo, bmi_hi: BMI intervention pair
        t_eval:      (L,) eval times
        oracle_fn:   callable(t) → true ∆PDP(t), or None
        n_mc:        number of Monte Carlo draws
        seed:        random seed for reproducibility

    Returns:
        dict with keys: t_eval, estimates_hat, mc_mean, mc_se,
                         ci_lo, ci_hi, curves (M×L array of all draws)
    """
    if t_eval is None:
        t_eval = np.array([2, 4, 8, 10], dtype=float)
    L = len(t_eval)

    # --- Save θ̂ ---
    trainable_params = _param_list(model)
    theta_hat = torch.nn.utils.parameters_to_vector(
        trainable_params
    ).detach().cpu()
    P = len(theta_hat)

    print(f"\n{'='*60}")
    print(f"MONTE CARLO ∆PDP CI")
    print(f"{'='*60}")
    print(f"  P = {P} parameters")
    print(f"  N = {len(dataset)} subjects")
    print(f"  BMI: lo={bmi_lo}, hi={bmi_hi}")
    print(f"  t_eval: {t_eval.tolist()}")
    print(f"  M = {n_mc} draws, seed = {seed}")

    # --- Step 1: ∆PDP at θ̂ (point estimate) ---
    print(f"\nStep 1: ∆PDP at θ̂ ...")
    delta_hat = _evaluate_delta_pdp(
        model, dataset, device, collate_fn,
        bmi_lo, bmi_hi, t_eval,
    )
    print(f"  Point estimates:")
    for k, t in enumerate(t_eval):
        print(f"    t={t:.1f}: ∆PDP = {delta_hat[k]:+.4f}")

    # --- Step 2: Cholesky of F⁻¹ ---
    print(f"\nStep 2: Cholesky of F⁻¹ ...")
    fisher_inv_t = torch.as_tensor(fisher_inv).float() if not torch.is_tensor(fisher_inv) else fisher_inv.float()
    L_cov = _safe_cholesky(fisher_inv_t, verbose=verbose)

    # --- Step 3: MC draws ---
    print(f"\nStep 3: MC sampling ...")
    curves = np.zeros((n_mc, L))

    torch.manual_seed(seed)
    t0 = time.time()

    for m in range(n_mc):
        z = torch.randn(P)
        theta_m = theta_hat + L_cov @ z

        # Load perturbed parameters
        torch.nn.utils.vector_to_parameters(theta_m, trainable_params)

        # Evaluate ∆PDP at θ⁽ᵐ⁾
        curves[m] = _evaluate_delta_pdp(
            model, dataset, device, collate_fn,
            bmi_lo, bmi_hi, t_eval,
        )

        if verbose and (m + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (m + 1) / elapsed
            eta = (n_mc - m - 1) / rate
            print(f"    MC draw {m+1}/{n_mc} "
                  f"({elapsed:.0f}s, ~{eta:.0f}s remaining)")

    # --- Step 4: Restore θ̂ ---
    torch.nn.utils.vector_to_parameters(theta_hat, trainable_params)
    model.eval()

    # --- Step 5: Summarise ---
    mc_mean = np.mean(curves, axis=0)
    mc_se = np.std(curves, axis=0)
    ci_lo = np.percentile(curves, 2.5, axis=0)
    ci_hi = np.percentile(curves, 97.5, axis=0)

    elapsed = time.time() - t0
    print(f"\n  MC done ({elapsed:.1f}s)")
    print(f"\n  {'t':>6s}  {'∆PDP(θ̂)':>10s}  {'MC mean':>10s}  "
          f"{'MC SE':>10s}  {'CI_lo':>10s}  {'CI_hi':>10s}", end="")
    if oracle_fn is not None:
        print(f"  {'True':>10s}", end="")
    print()
    print(f"  {'-'*72}")

    for k, t in enumerate(t_eval):
        line = (f"  {t:6.1f}  {delta_hat[k]:+10.4f}  {mc_mean[k]:+10.4f}  "
                f"{mc_se[k]:10.4f}  {ci_lo[k]:+10.4f}  {ci_hi[k]:+10.4f}")
        if oracle_fn is not None:
            line += f"  {oracle_fn(t):+10.4f}"
        print(line)

    results = {
        't_eval': t_eval,
        'estimates_hat': delta_hat,       # ∆PDP at θ̂ (point estimate)
        'mc_mean': mc_mean,               # mean of ∆PDP across draws
        'mc_se': mc_se,                   # std of ∆PDP across draws
        'ci_lo': ci_lo,                   # 2.5% quantile
        'ci_hi': ci_hi,                   # 97.5% quantile
        'curves': curves,                 # (M, L) all MC draws
        'n_mc': n_mc,
    }

    return results


# ─────────────────────────────────────────────────────────
# 4. Comparison helper: Delta vs MC
# ─────────────────────────────────────────────────────────

def compare_delta_vs_mc(delta_results, mc_results, t_eval, oracle_fn=None):
    """
    Print side-by-side comparison of delta method and MC CIs.

    Args:
        delta_results: dict from compute_full_delta_variance()
        mc_results:    dict from mc_delta_pdp_ci()
        t_eval:        eval times
        oracle_fn:     callable(t) → true ∆PDP(t), or None
    """
    print(f"\n{'='*80}")
    print(f"COMPARISON: Delta Method vs Monte Carlo ({mc_results['n_mc']} draws)")
    print(f"{'='*80}")
    print(f"  {'t':>6s}  {'∆PDP':>10s}  {'SE_δ':>10s}  {'SE_MC':>10s}  "
          f"{'Ratio':>8s}  {'CI_δ':>22s}  {'CI_MC':>22s}", end="")
    if oracle_fn is not None:
        print(f"  {'True':>10s}", end="")
    print()
    print(f"  {'-'*100}")

    for k, t in enumerate(t_eval):
        t_key = float(t)
        d = delta_results['results'][t_key]
        se_delta = d['se']
        se_mc = mc_results['mc_se'][k]
        ratio = se_delta / se_mc if se_mc > 0 else float('nan')

        line = (f"  {t:6.1f}  {d['estimate']:+10.4f}  "
                f"{se_delta:10.4f}  {se_mc:10.4f}  {ratio:8.3f}  "
                f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]  "
                f"[{mc_results['ci_lo'][k]:+.3f}, {mc_results['ci_hi'][k]:+.3f}]")
        if oracle_fn is not None:
            line += f"  {oracle_fn(t):+10.4f}"
        print(line)

    # Jensen's inequality diagnostic
    mean_shift = np.mean(np.abs(mc_results['mc_mean'] - mc_results['estimates_hat']))
    print(f"\n  Mean |E_MC[∆PDP] - ∆PDP(θ̂)| = {mean_shift:.4f}")
    print(f"  (nonzero → Jensen's inequality, curvature in ∆PDP(θ))")


# ─────────────────────────────────────────────────────────
# 5. Integration into simulation loop
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    import pyreadr
    from torch.utils.data import DataLoader
    from dataset import LongitudinalDataset, collate_pad
    from model_ODE_cumulative import NeuralODEModel, NeuralODEConfig
    from DPDP_cumulative_effect import (
        compute_empirical_fisher,
        _ledoit_wolf_shrink_fisher,
        _regularise_and_invert,
        compute_delta_pdp_gradients_continuous,
    )

    parser = argparse.ArgumentParser(
        description="Delta vs MC comparison for ∆PDP — single simulation")
    parser.add_argument("--sim_idx", type=int, default=0,
                        help="Simulation index (0-based)")
    parser.add_argument("--true_coeff", type=float, default=-0.05)
    parser.add_argument("--oracle_mode", type=str, default="cumulative",
                        choices=["cumulative", "instantaneous"])
    parser.add_argument("--data_dir", type=str, default="simu_datasets/S5_sims")
    parser.add_argument("--ckpt_dir", type=str,
                        default="checkpoints/simulation_cumulative_effect_diagoD_grouplasso_norhonorm_2")
    parser.add_argument("--bmi_lo", type=float, default=23.0)
    parser.add_argument("--bmi_hi", type=float, default=28.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--reg_mode", type=str, default='group_lasso',
                        choices=[None, "skip_gate", "group_lasso"])
    parser.add_argument("--t_eval", type=str, default="0, 2, 4, 6, 8, 10, 12",
                        help="Comma-separated eval times")
    parser.add_argument("--hidden_channels", type=int, default=4)
    parser.add_argument("--batch_size_pdp", type=int, default=64)
    parser.add_argument("--ode_solver", type=str, default="rk4")
    parser.add_argument("--n_mc", type=int, default=500,
                        help="Number of Monte Carlo draws")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ledoit_wolf", action="store_true",
                        help="Apply Ledoit-Wolf shrinkage to Fisher")
    parser.add_argument("--output_dir", type=str, default="results_simu")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Evaluation grid ---
    t_eval = np.array([float(x) for x in args.t_eval.split(',')], dtype=float)

    # --- Oracle ---
    bmi_lo, bmi_hi = args.bmi_lo, args.bmi_hi
    delta_v = bmi_hi - bmi_lo
    if args.oracle_mode == "cumulative":
        oracle_fn = lambda t: args.true_coeff * delta_v * t
    else:
        oracle_fn = lambda t: args.true_coeff * delta_v

    # --- Load data (sim_idx=0 → sim_001.rds, ckpt best_model_ode_0.pt) ---
    sim_idx = args.sim_idx
    data_path = f"{args.data_dir}/sim_{sim_idx+1:03d}.rds"
    ckpt_path = f"{args.ckpt_dir}/best_model_ode_{sim_idx}.pt"

    print(f"\n{'#'*60}")
    print(f"# DELTA vs MC COMPARISON — Simulation {sim_idx}")
    print(f"# Data: {data_path}")
    print(f"# Checkpoint: {ckpt_path}")
    print(f"# BMI pair: ({bmi_lo}, {bmi_hi})")
    print(f"{'#'*60}")

    time_col, y_col, id_col = "time", "ISA15_sim", "NUM_ID"
    x_cols = ["BMI_t"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    df = next(iter(pyreadr.read_r(data_path).values()))
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col,
                                  static_cols=static_cols)
    print(f"  N = {len(dataset)} subjects")

    # --- Build model ---
    cfg = NeuralODEConfig(
        hidden_channels=args.hidden_channels,
        enc_mlp_hidden=16, func_mlp_hidden=16,
        dec_rho_hidden=16, dec_p=4, dec_q=3, depth=2, dropout=0.0,
        euler_steps_per_interval=4,
        ode_solver=args.ode_solver,
    )
    model = NeuralODEModel(
        x_dim=len(x_cols), static_dim=len(static_cols), cfg=cfg,
        n_tv=1, use_rho_net=True, use_neural_re=True,
        re_spline_cols=None, g_hidden=8, fullD=False,
        bmi_mean=0.0, bmi_std=1.0,
        use_bmi_skip=True, static_skip_dims=[0, 1, 2, 3],
        reg_mode=args.reg_mode,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    print(f"  Loaded: {ckpt_path}")
    print(f"  bmi_mean={model.decoder.bmi_mean.item():.4f}, "
          f"bmi_std={model.decoder.bmi_std.item():.4f}")

    P = _param_count(model)
    print(f"  P = {P} parameters")

    # ================================================================
    # PART A: Delta method (reuses code from DPDP_cumulative_effect.py)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"PART A: DELTA METHOD")
    print(f"{'='*60}")

    # Step A1: Empirical Fisher
    print(f"\n  Step A1: Empirical Fisher F = Σ φ_i φ_iᵀ ...")
    fisher, scores = compute_empirical_fisher(
        model, dataset, device, collate_pad,
        lambda_reg=args.lambda_reg, weight_decay=args.weight_decay,
    )

    # Stationarity check
    mean_score = scores.mean(dim=0)
    mean_indiv_norm = scores.norm(dim=1).mean().item()
    mean_abs = scores.abs().mean(dim=0)
    ratio = mean_score.norm() / mean_abs.norm()
    print(f"  Stationarity: ||mean(φ)|| / ||mean(|φ|)|| = {ratio:.4f}")
    print(f"  ||mean(φ)|| = {mean_score.norm().item():.4e}, "
          f"mean(||φ_i||) = {mean_indiv_norm:.4e}")

    # Shrinkage + inversion
    if args.ledoit_wolf:
        print(f"\n  Applying Ledoit-Wolf shrinkage ...")
        fisher, lw_alpha = _ledoit_wolf_shrink_fisher(fisher, scores)

    fisher_reg, fisher_inv = _regularise_and_invert(fisher, "Fisher", LAMBDA=1e-4)

    # Step A2: ∆PDP gradients
    print(f"\n  Step A2: ∆PDP gradients ...")
    gradients, estimates, counts = compute_delta_pdp_gradients_continuous(
        model, dataset, device, collate_pad,
        bmi_lo=bmi_lo, bmi_hi=bmi_hi,
        t_eval=t_eval,
        batch_size=args.batch_size_pdp,
    )

    # Step A3: Var = gᵀ F⁻¹ g
    print(f"\n  Step A3: Var(∆PDP) = gᵀ F⁻¹ g")
    delta_results = {'results': {}}
    print(f"  {'t':>6s}  {'∆PDP':>10s}  {'SE':>10s}  "
          f"{'CI_lo':>10s}  {'CI_hi':>10s}  {'True':>10s}  {'Bias':>10s}")
    print(f"  {'-'*76}")

    for t in sorted(estimates.keys()):
        g = gradients[t]
        est = estimates[t]
        var_est = (g @ fisher_inv @ g).item()
        se = np.sqrt(max(var_est, 0.0))
        ci_lo = est - 1.96 * se
        ci_hi = est + 1.96 * se
        true_val = oracle_fn(t)
        bias = est - true_val

        delta_results['results'][t] = {
            'estimate': est, 'se': se,
            'var_fisher': var_est,
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'true': true_val, 'bias': bias,
        }
        print(f"  {t:6.1f}  {est:+10.4f}  {se:10.4f}  "
              f"{ci_lo:+10.4f}  {ci_hi:+10.4f}  "
              f"{true_val:+10.4f}  {bias:+10.4f}")

    delta_results['fisher_inv'] = fisher_inv.numpy()

    # ================================================================
    # PART B: Monte Carlo
    # ================================================================
    mc_result = mc_delta_pdp_ci(
        model, dataset, device, collate_pad,
        fisher_inv=fisher_inv,
        bmi_lo=bmi_lo, bmi_hi=bmi_hi,
        t_eval=t_eval,
        oracle_fn=oracle_fn,
        n_mc=args.n_mc, seed=args.seed,
    )

    # ================================================================
    # PART C: Side-by-side comparison
    # ================================================================
    compare_delta_vs_mc(delta_results, mc_result, t_eval, oracle_fn)

    # ================================================================
    # Save results
    # ================================================================
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir,
                            f"delta_vs_mc_sim{sim_idx}.pt")
    torch.save({
        'sim_idx': sim_idx,
        'bmi_lo': bmi_lo, 'bmi_hi': bmi_hi,
        't_eval': t_eval,
        'delta_results': delta_results,
        'mc_results': {k: v for k, v in mc_result.items() if k != 'curves'},
        'mc_curves': mc_result['curves'],
        'oracle_mode': args.oracle_mode,
        'true_coeff': args.true_coeff,
    }, out_path)
    print(f"\n  Saved to {out_path}")