"""
Full-parameter delta method for ∆PDP variance — Neural ODE-LMM.

Generalises the LMM delta method (paper §4.4.1, eq. 36–39) to all model
parameters θ = {encoder, ODE, ρ_net, β, g_net, D, σ²}:

    Var(∆PDP_ℓ) = g_ℓᵀ  F⁻¹  g_ℓ

where
    g_ℓ  = ∇_θ  ∆PDP_ℓ          gradient of estimand w.r.t. ALL params
    F    = Σ_i  s_i  s_iᵀ       empirical Fisher (per-subject NLL scores)

Steps:
  1. Empirical Fisher  F ∈ R^{P×P}  via batch_size=1 forward+backward
  2. ∆PDP gradient  g_ℓ  via differentiable counterfactual forward passes
  3. Var = g_ℓᵀ F⁻¹ g_ℓ ;  SE, 95 % CI

With P ≈ 1 300 parameters, F⁻¹ is trivial to compute.
"""
from __future__ import annotations
import math, os, csv, time
import torch
import torch.nn.functional as F_torch
import numpy as np
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _param_list(model):
    """Ordered list of parameters that require grad."""
    return [p for p in model.parameters() if p.requires_grad]


def _param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _cat_grads(grads):
    """Flatten and concatenate a tuple of gradient tensors."""
    return torch.cat([g.reshape(-1) for g in grads])


# ─────────────────────────────────────────────────────────
# 1. Per-subject NLL (differentiable)
# ─────────────────────────────────────────────────────────

def _per_subject_nll(mu, V, y_pad, mask, jitter=1e-4):
    """
    Masked Gaussian NLL for ONE subject (B=1).

    Args:
        mu:    (1, T)
        V:     (1, T, T)
        y_pad: (1, T)
        mask:  (1, T)

    Returns:
        scalar NLL (differentiable)
    """
    mu = mu.squeeze(0)
    V = V.squeeze(0)
    y = y_pad.squeeze(0)
    m = mask.squeeze(0)

    idx = m.bool()
    n_i = idx.sum()
    if n_i == 0:
        return torch.tensor(0.0, device=mu.device, requires_grad=True)

    mu_obs = mu[idx]
    y_obs = y[idx]
    V_obs = V[idx][:, idx] + jitter * torch.eye(n_i, device=mu.device, dtype=mu.dtype)

    residual = y_obs - mu_obs
    L = torch.linalg.cholesky(V_obs)
    Vinv_r = torch.cholesky_solve(residual.unsqueeze(-1), L).squeeze(-1)
    log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L)))

    nll = 0.5 * (log_det + residual @ Vinv_r + n_i * math.log(2 * math.pi))
    return nll


# ─────────────────────────────────────────────────────────
# 2. Empirical Fisher  F = Σ_i  s_i  s_iᵀ
# ─────────────────────────────────────────────────────────

def compute_empirical_fisher(model, dataset, device, collate_fn,
                              verbose=True):
    """
    Compute F = Σ_i  s_i  s_iᵀ   where  s_i = ∇_θ ℓ_i(θ̂).

    Uses batch_size=1 so each forward/backward is one subject.
    With P ≈ 1300 and N ≈ 5000, takes ~30–60 s.

    Returns:
        F: (P, P) tensor on CPU
    """
    from torch.utils.data import DataLoader

    model.eval()   # no dropout; LayerNorm same in eval/train
    params = _param_list(model)
    P = sum(p.numel() for p in params)

    fisher = torch.zeros(P, P)

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn)
    N = len(dataset)

    t0 = time.time()
    for i, batch in enumerate(loader):
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        y_pad = y_pad.to(device)
        mask = mask.to(device)
        s = s.to(device)
        bmi_t = x_pad[:, :, 0:1]

        # Skip subjects with no observations
        if mask.sum() == 0:
            continue

        # Forward (with grad)
        mu, V, Z, D, sig2,_ = model(
            t_pad, x_pad, masks=None,
            static_covariates=s, bmi_t=bmi_t, obs_mask=mask
        )

        # Per-subject NLL
        nll_i = _per_subject_nll(mu, V, y_pad, mask)

        # Score vector
        grads = torch.autograd.grad(nll_i, params, retain_graph=False,
                                    allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, params)]
        s_i = _cat_grads(grads).cpu()    # (P,)

        # Accumulate outer product
        fisher += torch.outer(s_i, s_i)

        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"    Fisher: {i+1}/{N} subjects "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    if verbose:
        print(f"    Fisher: done ({time.time()-t0:.1f}s), "
              f"cond = {torch.linalg.cond(fisher).item():.2e}")

    return fisher


# ─────────────────────────────────────────────────────────
# 3. ∆PDP gradient  g_ℓ = ∇_θ  ∆PDP_ℓ
# ─────────────────────────────────────────────────────────

def compute_delta_pdp_gradients(model, loader, device,
                                 bmi_lo, bmi_hi,
                                 visit_times,
                                 verbose=True):
    """
    Compute g_ℓ = ∇_θ ∆PDP_ℓ for each visit time.

    ∆PDP_ℓ = (1/n_ℓ) Σ_i [μ^{hi}_{i,c(i,ℓ)} − μ^{lo}_{i,c(i,ℓ)}]

    The gradient is accumulated across batches by linearity:
        g_ℓ = (1/n_ℓ) Σ_batch  ∇_θ  Σ_{i∈batch} δ_{i,ℓ}

    The computation graph spans both forward passes (hi and lo)
    because they share the same model parameters.

    Also returns the ∆PDP point estimates for each visit time.

    Returns:
        gradients: dict {vt: g_ℓ ∈ R^P}  (CPU)
        estimates: dict {vt: float}        ∆PDP estimates
        counts:    dict {vt: int}          subjects per visit time
    """
    model.eval()
    params = _param_list(model)
    P = sum(p.numel() for p in params)

    g_accum = {vt: torch.zeros(P) for vt in visit_times}
    est_accum = {vt: 0.0 for vt in visit_times}
    n_accum = {vt: 0 for vt in visit_times}

    for batch in loader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        mask = mask.to(device)
        s = s.to(device)

        B, T = t_pad.shape

        # --- Two counterfactual forward passes (with grad) ---
        x_hi = x_pad.clone()
        x_hi[:, :, 0] = bmi_hi
        mu_hi, _, _, _, _, _ = model(
            t_pad, x_hi, masks=None,
            static_covariates=s, bmi_t=x_hi[:, :, 0:1], obs_mask=mask
        )

        x_lo = x_pad.clone()
        x_lo[:, :, 0] = bmi_lo
        mu_lo, _, _, _, _, _ = model(
            t_pad, x_lo, masks=None,
            static_covariates=s, bmi_t=x_lo[:, :, 0:1], obs_mask=mask
        )

        # --- For each visit time, build differentiable sum ---
        for vt_idx, vt in enumerate(visit_times):
            delta_sum = torch.tensor(0.0, device=device)
            n_vt = 0

            for i in range(B):
                obs_idx = torch.where(mask[i] > 0.5)[0]
                if len(obs_idx) == 0:
                    continue
                obs_times = t_pad[i, obs_idx]
                closest = obs_idx[torch.argmin(torch.abs(obs_times - vt))]
                delta_sum = delta_sum + (mu_hi[i, closest] - mu_lo[i, closest])
                n_vt += 1

            if n_vt > 0:
                # Backward — keep graph for remaining visit times
                # allow_unused: g_net, L_unconstrained, log_residual_var
                # are not in the mu computation graph
                retain = (vt_idx < len(visit_times) - 1)
                grads = torch.autograd.grad(delta_sum, params,
                                            retain_graph=retain,
                                            allow_unused=True)
                # Replace None grads (unused params) with zeros
                grads = [g if g is not None else torch.zeros_like(p)
                         for g, p in zip(grads, params)]
                g_batch = _cat_grads(grads).cpu()

                g_accum[vt] += g_batch
                est_accum[vt] += delta_sum.item()
                n_accum[vt] += n_vt

    # Normalise
    gradients = {}
    estimates = {}
    for vt in visit_times:
        n = n_accum[vt]
        if n > 0:
            gradients[vt] = g_accum[vt] / n
            estimates[vt] = est_accum[vt] / n
        else:
            gradients[vt] = torch.zeros(P)
            estimates[vt] = 0.0

    if verbose:
        for vt in visit_times:
            print(f"    g_{int(vt)}: ||g|| = {gradients[vt].norm().item():.4f}, "
                  f"n = {n_accum[vt]}, ∆PDP = {estimates[vt]:.4f}")

    return gradients, estimates, n_accum


# ─────────────────────────────────────────────────────────
# 4. Main: full-parameter delta method variance
# ─────────────────────────────────────────────────────────

def compute_full_delta_variance(
    model, dataset, loader, device, collate_fn,
    bmi_lo=20.0, bmi_hi=35.0,
    visit_times=np.array([0, 5, 10, 15]),
    true_beta_bmi=-0.175,
    true_beta_int=-0.015,
):
    """
    Full-parameter delta method for ∆PDP variance.

    Var(∆PDP_ℓ) = g_ℓᵀ  F⁻¹  g_ℓ

    Returns dict with results per visit time + Fisher + gradients.
    """
    P = _param_count(model)
    delta_v = bmi_hi - bmi_lo

    print(f"\n{'='*60}")
    print(f"FULL-PARAMETER DELTA METHOD")
    print(f"{'='*60}")
    print(f"  P = {P} parameters")
    print(f"  N = {len(dataset)} subjects")
    print(f"  BMI: lo={bmi_lo}, hi={bmi_hi}, Δv={delta_v}")

    # --- Step 1: Empirical Fisher ---
    print(f"\nStep 1: Empirical Fisher F = Σ_i s_i s_iᵀ ...")
    fisher = compute_empirical_fisher(model, dataset, device, collate_fn)

    # Marquardt damping: F_reg = F + λ · diag(F)  (Marquardt, 1963)
    # Scale-invariant: each parameter's ridge is proportional to its own curvature.
    # Floor on diag(F) for dead parameters (F_jj = 0).
    LAMBDA = 1e-4
    diag_F = torch.diag(fisher)
    diag_F = torch.clamp(diag_F, min=1e-4* diag_F.max())
    fisher_reg = fisher + LAMBDA * torch.diag(diag_F)

    cond = torch.linalg.cond(fisher_reg).item()
    print(f"  Marquardt λ = {LAMBDA}")
    print(f"  diag(F): min={torch.diag(fisher).min().item():.2e}, "
          f"max={torch.diag(fisher).max().item():.2e}")
    print(f"  Condition number: {cond:.2e}")

    try:
        fisher_inv = torch.linalg.inv(fisher_reg)
    except torch.linalg.LinAlgError:
        print("  WARNING: Fisher not invertible, using pseudo-inverse")
        fisher_inv = torch.linalg.pinv(fisher_reg)

    # --- Step 2: ∆PDP gradients ---
    print(f"\nStep 2: ∆PDP gradients g_ℓ = ∇_θ ∆PDP_ℓ ...")
    gradients, estimates, counts = compute_delta_pdp_gradients(
        model, loader, device, bmi_lo, bmi_hi, visit_times
    )

    # --- Step 3: Var = gᵀ F⁻¹ g ---
    # Compute mean AGEc for true ∆PDP
    all_ages = []
    for batch in loader:
        _, _, _, _, _, _, s = batch
        all_ages.append(s[:, 1])
    mean_age = torch.cat(all_ages).mean().item()
    true_delta = delta_v * (true_beta_bmi + true_beta_int * mean_age)

    print(f"\nStep 3: Var(∆PDP_ℓ) = g_ℓᵀ F⁻¹ g_ℓ")
    print(f"  mean AGEc = {mean_age:.4f}")
    print(f"  true ∆PDP = {true_delta:.4f}")

    print(f"\n  {'Time':>6s}  {'∆PDP':>10s}  {'SE':>10s}  "
          f"{'CI_lo':>10s}  {'CI_hi':>10s}  {'True':>10s}  {'Bias':>10s}")
    print(f"  {'-'*72}")

    results = {}
    for vt in visit_times:
        g = gradients[vt]
        est = estimates[vt]

        var = (g @ fisher_inv @ g).item()
        se = np.sqrt(max(var, 0.0))

        ci_lo = est - 1.96 * se
        ci_hi = est + 1.96 * se
        bias = est - true_delta

        results[vt] = {
            'estimate': est,
            'se': se,
            'var': var,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'true': true_delta,
            'bias': bias,
        }

        print(f"  {vt:6.0f}  {est:+10.4f}  {se:10.4f}  "
              f"{ci_lo:+10.4f}  {ci_hi:+10.4f}  {true_delta:+10.4f}  {bias:+10.4f}")

    return {
        'results': results,
        'fisher': fisher.numpy(),
        'fisher_inv': fisher_inv.numpy(),
        'gradients': {vt: g.numpy() for vt, g in gradients.items()},
        'estimates': estimates,
        'counts': counts,
        'P': P,
        'mean_age': mean_age,
    }


# ─────────────────────────────────────────────────────────
# 5. Multi-simulation aggregation
# ─────────────────────────────────────────────────────────

def aggregate_simulations(all_results, visit_times):
    """
    Aggregate across D simulations. Compute var_mc, var_est, coverage.
    Output format matches LMM CSV.
    """
    D = len(all_results)
    summary = {}

    for vt in visit_times:
        ests = np.array([r['results'][vt]['estimate'] for r in all_results])
        ses = np.array([r['results'][vt]['se'] for r in all_results])

        # true_vals = np.array([r['results'][vt]['true'] for r in all_results])
        # coverage = np.mean((ci_los <= true_vals) & (true_vals <= ci_his))
        true_val = all_results[0]['results'][vt]['true']
        ci_los = np.array([r['results'][vt]['ci_lo'] for r in all_results])
        ci_his = np.array([r['results'][vt]['ci_hi'] for r in all_results])

        mean_hat = ests.mean()
        bias = mean_hat - true_val
        var_mc = ests.var(ddof=1) if D > 1 else float('nan')
        mean_var_est = (ses ** 2).mean()
        mse = bias ** 2 + var_mc if D > 1 else float('nan')
        rmse = np.sqrt(mse) if D > 1 else float('nan')
        coverage = np.mean((ci_los <= true_val) & (true_val <= ci_his)) if D > 1 else float('nan')

        summary[vt] = {
            'mean_hat': mean_hat,
            'true': true_val,
            'bias': bias,
            'var_mc': var_mc,
            'mean_var_est': mean_var_est,
            'mse': mse,
            'rmse': rmse,
            'coverage95': coverage,
        }

    return summary


# ─────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pyreadr
    from torch.utils.data import DataLoader
    from dataset import LongitudinalDataset, collate_pad
    from model_ODE import NeuralODEModel, NeuralODEConfig

    parser = argparse.ArgumentParser(
        description="Full-parameter delta method for ∆PDP variance")
    parser.add_argument("--n_sims", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--true_beta_bmi", type=float, default=-0.30)
    parser.add_argument("--true_beta_int", type=float, default=-0.05)
    parser.add_argument("--output_csv", type=str,
                        default="results/delta_pdp_full_delta_ode.csv")
    parser.add_argument("--bmi_pairs", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visit_times = np.array([0, 5, 10, 15])

    # BMI pairs
    if args.bmi_pairs:
        bmi_pairs = [tuple(map(float, p.split(':')))
                     for p in args.bmi_pairs.split(',')]
    else:
        grid = [20, 23, 26, 29, 32, 35]
        bmi_pairs = [(grid[i], grid[i+1]) for i in range(len(grid)-1)]
        bmi_pairs.append((20, 35))

    time_col, y_col, id_col = "time", "ISA15_sim", "NUM_ID"
    x_cols = ["BMI_t", "rs1", "rs2"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    all_pair_results = {pair: [] for pair in bmi_pairs}

    for sim_idx in range(9, 9+args.n_sims):
        if args.n_sims > 1:
            data_path = f"simu_datasets/S2a_sims/sim_{sim_idx+1:03d}.rds"
            ckpt_path = f"checkpoints/best_model_ode_full_skip_{sim_idx}.pt"
            print(f"\n{'#'*60}")
            print(f"# SIMULATION {sim_idx}")
            print(f"{'#'*60}")
        else:
            data_path = "simu_datasets/S2a_sims_2/sim_000.rds"
            ckpt_path = "checkpoints/best_model_ode_full_skip_0.pt"

        # --- Data ---
        df = next(iter(pyreadr.read_r(data_path).values()))
        df["SEX"] = df["SEX"].astype("category")
        df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
        df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
        df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

        dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col,
                                      static_cols=static_cols)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_pad)

        # --- Model ---
        cfg = NeuralODEConfig(
            hidden_channels=8, enc_mlp_hidden=32, func_mlp_hidden=32,
            dec_rho_hidden=16, dec_p=4, dec_q=3, depth=2, dropout=0.0,
            euler_steps_per_interval=4,
        )
        model = NeuralODEModel(
            x_dim=len(x_cols), static_dim=len(static_cols), cfg=cfg,
            n_tv=1, use_rho_net=True, use_neural_re=True,
            re_spline_cols=[1, 2], g_hidden=16, fullD=True,
            bmi_mean=0.0, bmi_std=1.0, static_skip_dims=[1],
        ).to(device)

        checkpoint = torch.load(ckpt_path, map_location=device,
                                weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded: {ckpt_path}")

        # --- Compute for each BMI pair ---
        for bmi_lo, bmi_hi in bmi_pairs:
            result = compute_full_delta_variance(
                model, dataset, loader, device, collate_pad,
                bmi_lo=bmi_lo, bmi_hi=bmi_hi,
                visit_times=visit_times,
                true_beta_bmi=args.true_beta_bmi,
                true_beta_int=args.true_beta_int,
            )
            all_pair_results[(bmi_lo, bmi_hi)].append(result)

    # --- Aggregate & CSV ---
    os.makedirs(os.path.dirname(args.output_csv)
                if os.path.dirname(args.output_csv) else '.', exist_ok=True)

    header = ['time', 'BMI_lo', 'BMI_hi', 'D', 'mean_hat', 'mean_true',
              'bias', 'var_mc', 'mean_var_est', 'mse', 'rmse', 'coverage95']
    csv_rows = []

    print(f"\n{'='*90}")
    print(f"AGGREGATED — Full Delta Method (D={args.n_sims})")
    print(f"{'='*90}")
    print(f"{'t':>4s} {'lo':>4s} {'hi':>4s} {'D':>3s}  {'mean_hat':>10s} "
          f"{'mean_true':>10s} {'bias':>8s}  {'var_mc':>10s} {'var_est':>10s} "
          f"{'mse':>10s} {'cov95':>6s}")
    print(f"{'-'*90}")

    for (bmi_lo, bmi_hi), results_list in all_pair_results.items():
        D = len(results_list)
        summary = aggregate_simulations(results_list, visit_times) if D > 0 else {}

        for vt in visit_times:
            if D == 1:
                r = results_list[0]['results'][vt]
                row = {
                    'time': int(vt), 'BMI_lo': int(bmi_lo), 'BMI_hi': int(bmi_hi),
                    'D': D, 'mean_hat': r['estimate'], 'mean_true': r['true'],
                    'bias': r['bias'], 'var_mc': float('nan'),
                    'mean_var_est': r['var'], 'mse': float('nan'),
                    'rmse': float('nan'), 'coverage95': float('nan'),
                }
            else:
                s = summary[vt]
                row = {
                    'time': int(vt), 'BMI_lo': int(bmi_lo), 'BMI_hi': int(bmi_hi),
                    'D': D, 'mean_hat': s['mean_hat'], 'mean_true': s['true'],
                    'bias': s['bias'], 'var_mc': s['var_mc'],
                    'mean_var_est': s['mean_var_est'], 'mse': s['mse'],
                    'rmse': s['rmse'], 'coverage95': s['coverage95'],
                }
            csv_rows.append(row)

            vm = f"{row['var_mc']:.6f}" if not np.isnan(row['var_mc']) else "NA"
            ms = f"{row['mse']:.6f}" if not np.isnan(row['mse']) else "NA"
            cv = f"{row['coverage95']:.2f}" if not np.isnan(row['coverage95']) else "NA"
            print(f"{int(vt):4d} {int(bmi_lo):4d} {int(bmi_hi):4d} {D:3d}  "
                  f"{row['mean_hat']:+10.4f} {row['mean_true']:+10.4f} "
                  f"{row['bias']:+8.4f}  {vm:>10s} {row['mean_var_est']:10.6f} "
                  f"{ms:>10s} {cv:>6s}")

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved to {args.output_csv}")