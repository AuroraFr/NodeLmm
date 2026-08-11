"""
Prediction and evaluation for Neural ODE-LMM on the 3C cohort.

Computes:
  1. Marginal log-likelihood (train + test)
  2. Fit mode: BLUP predictions using all outcomes → fitted MSE
  3. Forecasting mode: ODE up to t*, outcomes before t* → prediction MSE
  4. Population-averaged predictions at each visit time
  5. Export all results to CSV

Usage:
    python prediction_ODE.py
    python prediction_ODE.py --checkpoint checkpoints/best_model_ode_real_3C.pt
"""

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

from Preprocess_3C import process_data, EXPECTED_TIMES
from train_ODE_real import RealDataset, collate_real, compute_covariate_stats
from model_ODE_real import NeuralODEModel, NeuralODEConfig
from utils import masked_NLL

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ====================================================================
# 1. LOG-LIKELIHOOD
# ====================================================================

@torch.no_grad()
def compute_log_likelihood(model, loader, device):
    """
    Compute total and per-subject marginal log-likelihood.
    Returns positive log-likelihood (LL = -NLL).
    """
    model.eval()
    total_nll = 0.0
    n_subjects = 0
    n_obs = 0

    for batch in loader:
        pids, x_aug, y_pad, target_mask, static = batch
        x_aug = x_aug.to(device)
        y_pad = y_pad.to(device)
        target_mask = target_mask.to(device)
        static = static.to(device)

        mu, V, Z, D, sig2, reg_dict = model(
            x_aug, static_covariates=static, obs_mask=target_mask)

        nll = masked_NLL(mu, y_pad, V, target_mask)

        N = x_aug.shape[0]
        total_nll += nll.item() * N
        n_subjects += N
        n_obs += target_mask.sum().item()

    total_ll = -total_nll
    return {
        "total_LL": total_ll,
        "avg_LL_per_subject": total_ll / n_subjects,
        "n_subjects": n_subjects,
        "n_obs": int(n_obs),
    }


# ====================================================================
# 2. BLUP
# ====================================================================

@torch.no_grad()
def compute_blup(mu, V, y_pad, target_mask, Z, D, sig2, jitter=1e-6):
    """
    Best Linear Unbiased Predictor for random effects.

    b_hat_i = D Z_i' V_i^{-1} (y_i - mu_i)

    Returns:
        b_hat:   (N, q)
        y_blup:  (N, T) subject-specific predictions (mu + Z @ b_hat)
    """
    N, T = mu.shape
    q = Z.shape[2]
    device, dtype = mu.device, mu.dtype

    b_hat = torch.zeros(N, q, device=device, dtype=dtype)
    y_blup = mu.clone()

    for i in range(N):
        obs = target_mask[i].bool()
        n_i = obs.sum()
        if n_i < 1:
            continue

        mu_i = mu[i, obs]
        y_i = y_pad[i, obs]
        V_i = V[i][obs][:, obs]
        Z_i = Z[i, obs]
        r_i = y_i - mu_i

        L_i = torch.linalg.cholesky(
            V_i + jitter * torch.eye(n_i, device=device, dtype=dtype))
        Vinv_r = torch.cholesky_solve(
            r_i.unsqueeze(-1), L_i).squeeze(-1)

        b_hat[i] = D @ Z_i.t() @ Vinv_r
        y_blup[i] = mu[i] + Z[i] @ b_hat[i]

    return b_hat, y_blup


# ====================================================================
# 3. FIT MODE
# ====================================================================

@torch.no_grad()
def predict_fit_mode(model, loader, device):
    """
    Fit mode: condition on ALL observed outcomes for BLUP.

    Returns DataFrame with columns:
        patient_id, time, y_true, mu_pop, y_blup
    """
    model.eval()
    rows = []

    for batch in loader:
        pids, x_aug, y_pad, target_mask, static = batch
        x_aug = x_aug.to(device)
        y_pad = y_pad.to(device)
        target_mask = target_mask.to(device)
        static = static.to(device)

        mu, V, Z, D, sig2, reg_dict = model(
            x_aug, static_covariates=static, obs_mask=target_mask)

        b_hat, y_blup = compute_blup(mu, V, y_pad, target_mask, Z, D, sig2)

        # Extract times from x_aug
        t_pad = x_aug[:, :, 0]                                # (N, T)
        N, T = t_pad.shape

        for i in range(N):
            for j in range(T):
                if target_mask[i, j] > 0.5:
                    rows.append({
                        "patient_id": pids[i],
                        "time": t_pad[i, j].item(),
                        "y_true": y_pad[i, j].item(),
                        "mu_pop": mu[i, j].item(),
                        "y_blup": y_blup[i, j].item(),
                    })

    return pd.DataFrame(rows)


# ====================================================================
# 4. FORECASTING MODE
# ====================================================================

@torch.no_grad()
def predict_forecast_mode(model, loader, device):
    """
    Forecasting mode:
      For each visit k >= 1:
        - Truncate x_aug to [0, ..., k]
        - Run ODE integration on the truncated grid
        - BLUP from outcomes [y_0, ..., y_{k-1}]
        - Predict y_k

    Returns DataFrame with columns:
        patient_id, time, y_true, y_pred, mu_pop, visit_index
    """
    model.eval()
    rows = []

    for batch in loader:
        pids, x_aug, y_pad, target_mask, static = batch
        x_aug = x_aug.to(device)
        y_pad = y_pad.to(device)
        target_mask = target_mask.to(device)
        static = static.to(device)

        t_pad = x_aug[:, :, 0]                                 # (N, T)
        N, T = t_pad.shape

        for i in range(N):
            obs_idx = torch.where(target_mask[i] > 0.5)[0]
            n_i = len(obs_idx)
            if n_i < 2:
                continue

            for k_pos in range(0, n_i):    # ← start from 0, not 1
                k = obs_idx[k_pos].item()

                # Truncate x_aug at visit k (inclusive)
                x_aug_trunc = x_aug[i:i+1, :k+1, :]
                mask_trunc = target_mask[i:i+1, :k+1]
                static_i = static[i:i+1]

                mu_k, V_k, Z_k, D_k, sig2_k, _ = model(
                    x_aug_trunc,
                    static_covariates=static_i,
                    obs_mask=mask_trunc,
                )

                mu_current = mu_k[0, k]
                Z_current = Z_k[0, k]

                if k_pos == 0:
                    # First visit: population prediction only
                    y_pred_k = mu_current
                else:
                    # BLUP from past outcomes
                    past_idx = obs_idx[:k_pos]
                    n_past = len(past_idx)

                    mu_past = mu_k[0, past_idx]
                    y_past = y_pad[i, past_idx]
                    Z_past = Z_k[0, past_idx]
                    V_past = V_k[0][past_idx][:, past_idx]
                    r_past = y_past - mu_past

                    L_past = torch.linalg.cholesky(
                        V_past + 1e-6 * torch.eye(n_past, device=device,
                                                dtype=V_k.dtype))
                    Vinv_r = torch.cholesky_solve(
                        r_past.unsqueeze(-1), L_past).squeeze(-1)

                    b_hat = D_k @ (Z_past.t() @ Vinv_r)
                    y_pred_k = mu_current + Z_current @ b_hat

                rows.append({
                    "patient_id": pids[i],
                    "time": t_pad[i, k].item(),
                    "y_true": y_pad[i, k].item(),
                    "mu_pop": mu_current.item(),
                    "y_blup": y_pred_k.item(),
                    "visit_index": k_pos,
                })

    return pd.DataFrame(rows)


# ====================================================================
# 5. UTILITIES
# ====================================================================

def compute_mse(df, y_col="y_true", pred_col="y_blup"):
    return ((df[y_col] - df[pred_col]) ** 2).mean()


def population_averaged_predictions(df_fit, visit_times=None, max_dist=1.5):
    if visit_times is None:
        visit_times = EXPECTED_TIMES.tolist()

    rows = []
    subjects = df_fit["patient_id"].unique()

    for vt in visit_times:
        mu_pops, y_blups, y_trues = [], [], []
        for sid in subjects:
            df_s = df_fit[df_fit["patient_id"] == sid]
            closest_idx = (df_s["time"] - vt).abs().idxmin()
            row = df_s.loc[closest_idx]
            if abs(row["time"] - vt) <= max_dist:      # ← only if close enough
                mu_pops.append(row["mu_pop"])
                y_blups.append(row["y_blup"])
                y_trues.append(row["y_true"])

        rows.append({
            "time": vt,
            "n": len(mu_pops),
            "mean_y_true": np.mean(y_trues) if y_trues else np.nan,
            "se_y_true": np.std(y_trues) / np.sqrt(len(y_trues)) if y_trues else np.nan,
            "mean_mu_pop": np.mean(mu_pops) if mu_pops else np.nan,
            "mean_y_blup": np.mean(y_blups) if y_blups else np.nan,
        })

    return pd.DataFrame(rows)


# ====================================================================
# 6. MONTE-CARLO CONFIDENCE INTERVALS  (via empirical Fisher)
# ====================================================================

# ====================================================================
# 6a. DIFFERENTIABLE BLUP (for delta method)
# ====================================================================

def _blup_differentiable(mu, V, y_pad, target_mask, Z, D, sig2, jitter=1e-6):
    """
    Same as compute_blup but WITHOUT @torch.no_grad(),
    so gradients flow through for the delta method.
    Only supports batch_size=1 (N=1).
    """
    N, T = mu.shape
    q = Z.shape[2]
    device, dtype = mu.device, mu.dtype

    b_hat = torch.zeros(N, q, device=device, dtype=dtype)
    y_blup = mu.clone()

    for i in range(N):
        obs = target_mask[i].bool()
        n_i = obs.sum()
        if n_i < 1:
            continue

        mu_i = mu[i, obs]
        y_i = y_pad[i, obs]
        V_i = V[i][obs][:, obs]
        Z_i = Z[i, obs]
        r_i = y_i - mu_i

        L_i = torch.linalg.cholesky(
            V_i + jitter * torch.eye(n_i, device=device, dtype=dtype))
        Vinv_r = torch.cholesky_solve(
            r_i.unsqueeze(-1), L_i).squeeze(-1)

        b_hat[i] = D @ Z_i.t() @ Vinv_r
        y_blup[i] = mu[i] + Z[i] @ b_hat[i]

    return b_hat, y_blup


# ====================================================================
# 6b. DELTA METHOD CI FOR POPULATION-AVERAGED PREDICTIONS
# ====================================================================

def delta_method_prediction_ci(model, dataset, collate_fn, device, fisher_inv,
                                visit_times=None, max_dist=1.5, verbose=True):
    """
    Delta method 95% CI for population-averaged BLUP predictions.

    Var(μ̄_BLUP(t)) = ∇_θ μ̄_BLUP(t)ᵀ · F⁻¹ · ∇_θ μ̄_BLUP(t)

    where ∇_θ μ̄_BLUP(t) = (1/nₜ) Σᵢ ∇_θ ŷᵢ_BLUP(t)

    One forward+backward per subject per visit time.
    Exact (no MC noise), much faster than MC with 500 draws.
    """
    if visit_times is None:
        visit_times = EXPECTED_TIMES.tolist()
    n_times = len(visit_times)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in trainable_params)

    # Accumulators per visit time
    grad_accum = torch.zeros(n_times, P)
    pred_accum = np.zeros(n_times)
    counts = np.zeros(n_times, dtype=int)

    model.eval()
    loader_1 = DataLoader(dataset, batch_size=1, shuffle=False,
                          collate_fn=collate_fn)

    for s_idx, batch in enumerate(loader_1):
        pids, x_aug, y_pad, target_mask, static = batch
        x_aug = x_aug.to(device)
        y_pad = y_pad.to(device)
        target_mask = target_mask.to(device)
        static = static.to(device)

        # ── Forward WITH gradients ──────────────────────────────────
        mu, V, Z, D, sig2, _ = model(
            x_aug, static_covariates=static, obs_mask=target_mask)

        # ── Differentiable BLUP ─────────────────────────────────────
        _, y_blup = _blup_differentiable(
            mu, V, y_pad, target_mask, Z, D, sig2)

        # ── Find which visit times this subject contributes to ──────
        t_pad = x_aug[0, :, 0].detach()
        obs_idx = torch.where(target_mask[0] > 0.5)[0]
        obs_times = t_pad[obs_idx]

        # Collect (visit_time_index, obs_index) pairs
        vt_hits = []
        for k, vt in enumerate(visit_times):
            if len(obs_times) == 0:
                continue
            dists = (obs_times - vt).abs()
            min_dist, min_pos = dists.min(0)
            if min_dist.item() <= max_dist:
                j = obs_idx[min_pos.item()].item()
                vt_hits.append((k, j))

        if len(vt_hits) == 0:
            continue

        # ── Backward for each visit time ────────────────────────────
        for hit_idx, (k, j) in enumerate(vt_hits):
            model.zero_grad()
            is_last = (hit_idx == len(vt_hits) - 1)
            y_blup[0, j].backward(retain_graph=not is_last)

            # Collect gradient
            grad_i = torch.cat([
                p.grad.detach().reshape(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=device)
                for p in trainable_params
            ]).cpu()

            grad_accum[k] += grad_i
            pred_accum[k] += y_blup[0, j].item()
            counts[k] += 1

        if verbose and (s_idx + 1) % 500 == 0:
            print(f"    Subject {s_idx+1}/{len(dataset)}")

    # ── Compute pointwise CI ────────────────────────────────────────
    fisher_inv_cpu = fisher_inv.cpu()

    results = {
        "time": np.array(visit_times),
        "mean_blup": np.zeros(n_times),
        "se_blup": np.zeros(n_times),
        "ci_lo_blup": np.zeros(n_times),
        "ci_hi_blup": np.zeros(n_times),
    }

    if verbose:
        print(f"\n  Delta method CI (averaged BLUP predictions):")
        print(f"  {'Time':>6s}  {'Mean':>8s}  {'nₜ':>5s}  "
              f"{'SE':>8s}  {'95% CI':>20s}")

    for k in range(n_times):
        if counts[k] == 0:
            continue

        n_k = counts[k]
        mean_pred = pred_accum[k] / n_k
        mean_grad = grad_accum[k] / n_k     # ∇_θ μ̄(t)

        # Var(μ̄(t)) = ∇ᵀ F⁻¹ ∇
        var_k = (mean_grad @ fisher_inv_cpu @ mean_grad).item()
        se_k = np.sqrt(max(var_k, 0.0))

        results["mean_blup"][k] = mean_pred
        results["se_blup"][k] = se_k
        results["ci_lo_blup"][k] = mean_pred - 1.96 * se_k
        results["ci_hi_blup"][k] = mean_pred + 1.96 * se_k

        if verbose:
            print(f"  {visit_times[k]:6.1f}  {mean_pred:8.3f}  {n_k:5d}  "
                  f"{se_k:8.4f}  "
                  f"[{results['ci_lo_blup'][k]:.3f}, "
                  f"{results['ci_hi_blup'][k]:.3f}]")

    return results


def compute_fisher_for_ci(model, dataset, collate_fn, device,
                           lambda_reg=0.0, weight_decay=0.0,
                           variance_mode="ledoit_wolf",
                           fisher_cache=None, verbose=True):
    """
    Compute (or load cached) Fisher inverse for CI computation.
    Shared between prediction CI and ΔPDP CI.
    """
    from PDP_variance import (
        compute_empirical_fisher,
        _ledoit_wolf_shrink_fisher,
        _regularise_and_invert,
    )

    if fisher_cache and os.path.exists(fisher_cache):
        if verbose:
            print(f"  Loading cached Fisher inverse from {fisher_cache}")
        return torch.from_numpy(np.load(fisher_cache)).float()

    if verbose:
        print(f"\n  Computing empirical Fisher (N={len(dataset)}) ...")
        print(f"  Variance mode: {variance_mode}")

    fisher, scores = compute_empirical_fisher(
        model, dataset, device, collate_fn=collate_fn,
        lambda_reg=lambda_reg, weight_decay=weight_decay,
        verbose=verbose,
    )

    if variance_mode == "ledoit_wolf":
        F_shrunk, lw_alpha = _ledoit_wolf_shrink_fisher(
            fisher, scores, verbose=verbose)
        _, fisher_inv = _regularise_and_invert(
            F_shrunk, "Fisher (LW)", LAMBDA=1e-6, verbose=verbose)
    elif variance_mode == "marquardt":
        _, fisher_inv = _regularise_and_invert(
            fisher, "Fisher (Marquardt)", LAMBDA=1e-4, verbose=verbose)
    else:
        raise ValueError(f"Unknown variance_mode: {variance_mode}")

    if fisher_cache:
        np.save(fisher_cache, fisher_inv.numpy())
        if verbose:
            print(f"  Fisher inverse cached to {fisher_cache}")

    return fisher_inv


# ====================================================================
# 6c. MONTE-CARLO CI (alternative to delta method)
# ====================================================================
 
@torch.no_grad()
def mc_population_ci(model, loader, device, fisher_inv,
                     visit_times=None, n_mc=500, seed=42,
                     max_dist=1.5, verbose=True):
    """
    Monte Carlo 95% CI for population-averaged BLUP predictions.
 
    Method:
      1. Draw θ⁽ᵐ⁾ ~ N(θ̂, F⁻¹),  m = 1, ..., M
      2. For each draw, load θ⁽ᵐ⁾, run predict_fit_mode → pop-avg curve
      3. Pointwise quantiles [2.5%, 97.5%]
 
    Should give same width as delta method if linearization is accurate.
    """
    if visit_times is None:
        visit_times = EXPECTED_TIMES.tolist()
    n_times = len(visit_times)
 
    theta_hat = torch.nn.utils.parameters_to_vector(
        [p for p in model.parameters() if p.requires_grad]
    ).detach().cpu()
    P = len(theta_hat)
 
    if verbose:
        print(f"\n  MC sampling: {n_mc} draws from N(θ̂, F⁻¹), P = {P}")
 
    # ------------------------------------------------------------------
    # Cholesky of F⁻¹ for efficient sampling
    # Work in float64 throughout to avoid numerical non-PD artifacts
    # ------------------------------------------------------------------
    F_cpu = fisher_inv.cpu().double()
    F_cpu = (F_cpu + F_cpu.T) / 2                # enforce exact symmetry
 
    try:
        L_cov = torch.linalg.cholesky(F_cpu)
    except torch.linalg.LinAlgError:
        # F⁻¹ has negative eigenvalues from float32 inversion artifacts
        # Project onto PD cone via eigenvalue clipping
        eigvals, eigvecs = torch.linalg.eigh(F_cpu)
        n_neg = (eigvals < 0).sum().item()
        if verbose:
            print(f"  Fisher⁻¹ not PD: {n_neg}/{P} negative eigenvalues")
            print(f"  eigenvalue range: [{eigvals.min().item():.2e}, "
                  f"{eigvals.max().item():.2e}]")
        # Relative floor: preserve structure, clamp only degenerate directions
        floor = eigvals.max().item() * 1e-10
        eigvals = eigvals.clamp(min=floor)
        F_pd = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        F_pd = (F_pd + F_pd.T) / 2
        L_cov = torch.linalg.cholesky(F_pd)
        if verbose:
            print(f"  Projected onto PD cone (floor = {floor:.2e})")
 
    L_cov = L_cov.float()                        # back to float32 for sampling
 
    curves_blup = np.zeros((n_mc, n_times))
    curves_pop = np.zeros((n_mc, n_times))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
 
    torch.manual_seed(seed)
    for m in range(n_mc):
        z = torch.randn(P)
        theta_m = theta_hat + L_cov @ z
 
        torch.nn.utils.vector_to_parameters(theta_m, trainable_params)
 
        df_fit_m = predict_fit_mode(model, loader, device)
        df_pop_m = population_averaged_predictions(
            df_fit_m, visit_times=visit_times, max_dist=max_dist)
 
        curves_blup[m] = df_pop_m["mean_y_blup"].values
        curves_pop[m] = df_pop_m["mean_mu_pop"].values
 
        if verbose and (m + 1) % 100 == 0:
            print(f"    MC draw {m+1}/{n_mc}")
 
    # Restore original parameters
    torch.nn.utils.vector_to_parameters(theta_hat, trainable_params)
    model.eval()
 
    results = {
        "time": np.array(visit_times),
        "mean_blup": np.mean(curves_blup, axis=0),
        "se_blup": np.std(curves_blup, axis=0),
        "ci_lo_blup": np.percentile(curves_blup, 2.5, axis=0),
        "ci_hi_blup": np.percentile(curves_blup, 97.5, axis=0),
        "mean_pop": np.mean(curves_pop, axis=0),
        "se_pop": np.std(curves_pop, axis=0),
        "ci_lo_pop": np.percentile(curves_pop, 2.5, axis=0),
        "ci_hi_pop": np.percentile(curves_pop, 97.5, axis=0),
    }
 
    if verbose:
        print(f"\n  MC CI results (averaged BLUP):")
        print(f"  {'Time':>6s}   {'Mean':>8s}   {'SE':>8s}   {'95% CI':>20s}")
        for i, t in enumerate(visit_times):
            print(f"  {t:6.1f}   {results['mean_blup'][i]:8.3f}   "
                  f"{results['se_blup'][i]:8.3f}   "
                  f"[{results['ci_lo_blup'][i]:.3f}, "
                  f"{results['ci_hi_blup'][i]:.3f}]")
        print(f"\n  MC CI results (population μ):")
        for i, t in enumerate(visit_times):
            print(f"  {t:6.1f}   [{results['ci_lo_pop'][i]:.3f}, "
                  f"{results['ci_hi_pop'][i]:.3f}]")
 
    return results


# ====================================================================
# 7. PLOTS
# ====================================================================

def plot_individual_predictions(df_fit, n_subjects=25, ncols=5,
                                hlme_df=None,
                                save_path="predictions_individual.png"):
    subjects = df_fit["patient_id"].unique()
    n_plot = min(n_subjects, len(subjects))
    rng = np.random.RandomState(42)
    selected = rng.choice(subjects, n_plot, replace=False)

    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False, sharey=True)
    axes = axes.ravel()

    for p_idx, sid in enumerate(selected):
        ax = axes[p_idx]
        df_s = df_fit[df_fit["patient_id"] == sid].sort_values("time")

        ax.scatter(df_s["time"], df_s["y_true"], c="blue", s=20, zorder=3,
                   label="Observed")
        ax.plot(df_s["time"], df_s["y_blup"], "g--", linewidth=1.5,
                label="Neural ODE-LMM")
        ax.plot(df_s["time"], df_s["mu_pop"], "r:", linewidth=1, alpha=0.6,
                label="ODE pop")

        if hlme_df is not None:
            hlme_s = hlme_df[hlme_df["patient_id"] == sid].sort_values("time")
            if len(hlme_s) > 0:
                ax.plot(hlme_s["time"], hlme_s["y_blup"], "k--",
                        linewidth=1.5, label="HLME")

        ax.set_title(f"ID {sid} (n={len(df_s)})", fontsize=9)
        if p_idx % ncols == 0:
            ax.set_ylabel("IST")
        if p_idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Time (years)")

    for k in range(n_plot, len(axes)):
        axes[k].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)
    # fig.suptitle("Individual BLUP predictions (fit mode) — Neural ODE-LMM",
    #              fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {save_path}")


def plot_population_averaged(df_pop, hlme_pop=None, mode="fit",
                             ci_dict=None, hlme_ci_dict=None,
                             save_path="predictions_population_mc.png"):
    """
    Population-averaged prediction plot with optional 95% CI as error bars.

    Args:
        df_pop:       DataFrame with observed means/SE and model predictions
        hlme_pop:     DataFrame with HLME predictions (optional)
        mode:         "fit" or "forecast"
        ci_dict:      dict from delta_method_prediction_ci() (optional)
        hlme_ci_dict: dict with keys 'time', 'ci_lo', 'ci_hi' — HLME CI (optional)
        save_path:    output path
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # ── Observed: mean ± 1.96 SE (95% CI) ───────────────────────────
    ax.errorbar(df_pop["time"], df_pop["mean_y_true"],
                yerr=1.96 * df_pop["se_y_true"],
                fmt="o-", color="blue", linewidth=2, markersize=6,
                capsize=4, label="Observed (mean, 95% CI)")

    # ── ODE BLUP with delta method CI error bars ────────────────────
    label = "Neural ODE-LMM (95% CI)" if mode == "fit" else "Neural ODE-LMM (forecast)"
    print(df_pop["mean_y_blup"].values)
    if ci_dict is not None:
        y_ode = df_pop["mean_y_blup"].values 
        yerr_lo = y_ode - ci_dict["ci_lo_blup"]
        yerr_hi = ci_dict["ci_hi_blup"] - y_ode
        print(yerr_hi, yerr_lo)
        print(df_pop["mean_y_blup"].values, ci_dict['mean_blup'])
        ax.errorbar(ci_dict["time"], y_ode,
                    yerr=[yerr_lo, yerr_hi],
                    fmt="o--", color="green", linewidth=2, markersize=6,
                    capsize=4, label=label)
    else:
        ax.plot(df_pop["time"], df_pop["mean_y_blup"], "o--", color="green",
                linewidth=2, markersize=6, label=label)

    # ── HLME with CI error bars ─────────────────────────────────────
    if hlme_pop is not None:
        y_hlme = hlme_pop["mean_pred"].values
        yerr_lo = y_hlme - hlme_pop["ci_lo_delta"]
        yerr_hi = hlme_pop["ci_hi_delta"] - y_hlme
        print(yerr_hi, yerr_lo)
        ax.errorbar(hlme_pop["time"], y_hlme,
                    yerr=[yerr_lo, yerr_hi],
                    fmt="o--", color="black", linewidth=2, markersize=6,
                    capsize=4, label="LMM (95% CI)")

    ax.set_xlabel("Time since inclusion (years)", fontsize=12)
    ax.set_ylabel("IST", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {save_path}")


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prediction and evaluation for Neural ODE-LMM (3C cohort)")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/cv_final_model.pt")
    parser.add_argument("--data", type=str,
                        default="3C_dataset/train_3C_data_1.csv")
    parser.add_argument("--hlme_csv", type=str, default='results_3C/hlme/',
                        help="CSV with HLME predictions for comparison")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="results_3C")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    # ── CI arguments ──
    parser.add_argument("--compute_ci", action="store_true",
                        help="Compute 95%% CI for population-averaged predictions")
    parser.add_argument("--ci_method", type=str, default="delta",
                        choices=["delta", "mc", "both"],
                        help="CI method: delta (exact, fast), mc (no linearization), "
                             "both (cross-check)")
    parser.add_argument("--n_mc", type=int, default=500,
                        help="Number of MC draws (only for mc/both)")
    parser.add_argument("--fisher_cache", type=str, default=None,
                        help="Path to cached F⁻¹ (.npy); computed if absent")
    parser.add_argument("--lambda_reg", type=float, default=0.1,
                        help="Group lasso λ (default: read from checkpoint)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="AdamW weight decay (default: read from checkpoint)")
    parser.add_argument("--variance_mode", type=str, default="ledoit_wolf",
                        choices=["ledoit_wolf", "marquardt"],
                        help="Variance estimation mode (default: ledoit_wolf)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ─────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    ckpt_cfg = checkpoint['config']
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    # print(f"  Best test loss: {checkpoint.get('best_test_loss', '?'):.4f}")

    # ── Feature definitions from checkpoint ─────────────────────────────
    id_col = "NUM_ID"
    target_col = "ISA15"
    time_varying_features = ckpt_cfg.get('time_varying_features', ["BMI", "PAS", "PAD", "GLUC", "HDL"])
    static_features = ckpt_cfg.get('static_features', ["SEX_code", "AGEc", "DIPNIV_2", "DIPNIV_3"])
    K = len(time_varying_features)
    Ks = len(static_features)
    interp_method = ckpt_cfg.get('interp_method', 'linear')
    mask_type = ckpt_cfg.get('mask_type', 'binary')

    # ── Load and preprocess data ────────────────────────────────────────
    df = pd.read_csv(args.data)
    test_df = pd.read_csv("3C_dataset/test_3C_data.csv")

    if "AGEc" not in df.columns:
        # all_df = pd.read_csv("3C_dataset/data_3C.csv")
        # baseline_age = all_df.groupby(id_col)["AGE0"].transform("first")
        # baseline_age_mean = baseline_age.mean()
        # df["AGEc"] = df.groupby(id_col)["AGE0"].transform("first") - baseline_age_mean
        # test_df["AGEc"] = test_df.groupby(id_col)["AGE0"].transform("first") - baseline_age_mean

        all_df = pd.read_csv("3C_dataset/data_3C.csv")
        baseline_age_mean = all_df.groupby(id_col)["AGE0"].first().mean()
        print('baseline_age_mean', baseline_age_mean)
        df["AGEc"] = df.groupby(id_col)["AGE0"].transform("first") - baseline_age_mean
        test_df["AGEc"] = test_df.groupby(id_col)["AGE0"].transform("first") - baseline_age_mean

    patient_data = process_data(
        df=df,
        id_col=id_col,
        time_varying_features=time_varying_features,
        static_features=static_features,
        target_col=target_col,
        interp_method=interp_method,
        mask_type=mask_type,
    )

    test_patient_data = process_data(
        df=test_df,
        id_col=id_col,
        time_varying_features=time_varying_features,
        static_features=static_features,
        target_col=target_col,
        interp_method=interp_method,
        mask_type=mask_type,
    )

    full_dataset = RealDataset(patient_data)
    test_dataset = RealDataset(test_patient_data)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_real)
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_real)

    # print(f"  Subjects: {N} total → {len(train_idx)} train, {len(test_idx)} test")

    # ── Rebuild model from checkpoint config ────────────────────────────
    cfg = NeuralODEConfig(
        hidden_channels=ckpt_cfg['hidden_channels'],
        enc_mlp_hidden=ckpt_cfg.get('enc_mlp_hidden', 16),
        func_mlp_hidden=ckpt_cfg.get('func_mlp_hidden', 32),
        dec_rho_hidden=ckpt_cfg.get('dec_rho_hidden', 16),
        dec_p=ckpt_cfg.get('dec_p', 4),
        dec_q=ckpt_cfg.get('dec_q', 3),
        depth=ckpt_cfg.get('depth', 2),
        euler_steps_per_interval=4,
        ode_solver=ckpt_cfg.get('ode_solver', 'rk4'),
        use_rho_norm=ckpt_cfg.get('use_rho_norm', True)
    )

    model = NeuralODEModel(
        n_tv=K,
        static_dim=Ks,
        cfg=cfg,
        use_rho_net=True,
        use_neural_re=True,
        g_hidden=8,
        fullD=False,
        cov_means=checkpoint['cov_means'],
        cov_stds=checkpoint['cov_stds'],
        static_skip_dims=ckpt_cfg.get('static_skip_dims', list(range(Ks))),
        use_dynamic_skip=True,
        reg_mode='group_lasso',
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    # Print model info
    sig2 = torch.exp(model.decoder.log_residual_var).item()
    D = model.decoder._build_D(device, torch.float32).detach().cpu()
    print(f"  σ² = {sig2:.4f}")
    print(f"  D diag = {D.diag().tolist()}")
    print(f"  β = {model.decoder.beta_neural.detach().cpu().tolist()}")

    # ── HLME predictions ────────────────────────────────────────────────
    hlme_train_fit = None
    hlme_train_pred = None
    hlme_val_fit = None
    hlme_val_pred = None
    if args.hlme_csv and os.path.exists(args.hlme_csv):
        hlme_train_fit = pd.read_csv(args.hlme_csv+"fit_preds_hlme_train.csv")
        hlme_train_pred = pd.read_csv(args.hlme_csv+"forecast_preds_hlme_train.csv")
        hlme_val_fit = pd.read_csv(args.hlme_csv+"fit_preds_hlme_val.csv")
        hlme_val_pred = pd.read_csv(args.hlme_csv+"forecast_preds_hlme_val.csv")
        print(f"  HLME train fit predictions: {args.hlme_csv} ({len(hlme_train_fit)} rows)")
        print(f"  HLME train forecast predictions: {args.hlme_csv} ({len(hlme_train_pred)} rows)")
        print(f"  HLME val fit predictions: {args.hlme_csv} ({len(hlme_val_fit)} rows)")
        print(f"  HLME val forecast predictions: {args.hlme_csv} ({len(hlme_val_pred)} rows)")

    # ================================================================
    # 1. LOG-LIKELIHOOD
    # ================================================================
    print(f"\n{'='*60}")
    print("1. LOG-LIKELIHOOD")
    print(f"{'='*60}")

    for name, loader in [("train", train_loader), ("test", test_loader)]:
        ll = compute_log_likelihood(model, loader, device)
        print(f"  {name:6s}: LL = {ll['total_LL']:.2f}, "
              f"avg/subj = {ll['avg_LL_per_subject']:.4f}, "
              f"n = {ll['n_subjects']}, n_obs = {ll['n_obs']}")

    # ================================================================
    # 2. FIT MODE
    # ================================================================
    print(f"\n{'='*60}")
    print("2. FIT MODE (trajectory reconstruction)")
    print(f"{'='*60}")

    for name, loader in [("train", train_loader), ("test", test_loader)]:
        df_fit = predict_fit_mode(model, loader, device)
        mse_pop = compute_mse(df_fit, pred_col="mu_pop")
        mse_blup = compute_mse(df_fit, pred_col="y_blup")
        print(f"  {name:6s}: MSE(pop) = {mse_pop:.4f}, "
              f"MSE(BLUP) = {mse_blup:.4f}, n_obs = {len(df_fit)}")

        csv_path = os.path.join(args.output_dir, f"fit_{name.lower()}.csv")
        df_fit.to_csv(csv_path, index=False)

    # Population-averaged
    df_fit_train = predict_fit_mode(model, train_loader, device)
    df_fit_test = predict_fit_mode(model, test_loader, device)
    df_pred_train = predict_forecast_mode(model, train_loader, device)
    df_pred_test = predict_forecast_mode(model, test_loader, device)
    df_pop_test = population_averaged_predictions(df_fit_test)
    df_pop_train = population_averaged_predictions(df_fit_train)
    print(f"\n  Population-averaged (fit mode):")
    print(df_pop_test.to_string(index=False))

    # ================================================================
    # 3. FORECASTING MODE
    # ================================================================
    print(f"\n{'='*60}")
    print("3. FORECASTING MODE")
    print(f"{'='*60}")

    for name, loader in [("train", train_loader), ("test", test_loader)]:
        df_fc = predict_forecast_mode(model, loader, device)
        if len(df_fc) > 0:
            mse_pop = compute_mse(df_fc, y_col="y_true", pred_col="mu_pop")
            mse_pred = compute_mse(df_fc, y_col="y_true", pred_col="y_blup")
            print(f"  {name:6s}: MSE(pop) = {mse_pop:.4f}, "
                  f"MSE(pred) = {mse_pred:.4f}, n = {len(df_fc)}")

            csv_path = os.path.join(args.output_dir,
                                    f"forecast_{name.lower()}.csv")
            df_fc.to_csv(csv_path, index=False)

    # ================================================================
    # 4. CONFIDENCE INTERVALS (optional)
    # ================================================================
    ci_ode = None
    if args.compute_ci:
        print(f"\n{'='*60}")
        print(f"4. CONFIDENCE INTERVALS (method={args.ci_method})")
        print(f"{'='*60}")

        # Read regularisation from checkpoint if not overridden
        lam_reg = (args.lambda_reg if args.lambda_reg is not None
                   else ckpt_cfg.get('lambda_reg', 0.0))
        lam_wd = (args.weight_decay if args.weight_decay is not None
                  else ckpt_cfg.get('lambda_wd', 0.0))
        print(f"  λ_reg = {lam_reg},  λ_wd = {lam_wd}")

        fisher_cache_path = (args.fisher_cache or
                             os.path.join(args.output_dir,
                                          f"fisher_inv_pred_{args.variance_mode}.npy"))

        # Step 1: Fisher inverse (from TRAINING data — Var(θ̂))
        fisher_inv = compute_fisher_for_ci(
            model, full_dataset, collate_real, device,
            lambda_reg=lam_reg, weight_decay=lam_wd,
            variance_mode=args.variance_mode,
            fisher_cache=fisher_cache_path,
            verbose=True,
        )

        # Step 2a: Delta method
        if args.ci_method in ("delta", "both"):
            print(f"\n  ── Delta method CI ──")
            print(f"  Computing gradients on test data (N={len(test_dataset)}) ...")
            delta_ci = delta_method_prediction_ci(
                model, test_dataset, collate_real, device,
                fisher_inv=fisher_inv,
                visit_times=EXPECTED_TIMES.tolist(),
                verbose=True,
            )
            ci_df = pd.DataFrame(delta_ci)
            ci_df.to_csv(os.path.join(args.output_dir, "delta_ci_ode.csv"),
                         index=False)
            ci_ode = delta_ci  # use delta for plotting by default

        # Step 2b: Monte Carlo
        if args.ci_method in ("mc", "both"):
            print(f"\n  ── Monte Carlo CI (M={args.n_mc}) ──")
            mc_ci = mc_population_ci(
                model, test_loader, device,
                fisher_inv=fisher_inv,
                visit_times=EXPECTED_TIMES.tolist(),
                n_mc=args.n_mc, seed=args.seed,
                verbose=True,
            )
            mc_df = pd.DataFrame(mc_ci)
            mc_df.to_csv(os.path.join(args.output_dir, "mc_ci_ode.csv"),
                         index=False)
            if args.ci_method == "mc":
                ci_ode = mc_ci  # use MC for plotting

        # Step 2c: Compare if both
        if args.ci_method == "both":
            print(f"\n  ── Comparison: Delta vs MC ──")
            print(f"  {'Time':>6s}  {'Delta SE':>10s}  {'MC SE':>10s}  "
                  f"{'Ratio':>8s}")
            for i, t in enumerate(EXPECTED_TIMES.tolist()):
                d_se = delta_ci["se_blup"][i]
                m_se = mc_ci["se_blup"][i]
                ratio = d_se / m_se if m_se > 0 else float('nan')
                print(f"  {t:6.1f}  {d_se:10.4f}  {m_se:10.4f}  "
                      f"{ratio:8.3f}")

    # ================================================================
    # 5. PLOTS
    # ================================================================
    print(f"\n{'='*60}")
    print("5. PLOTS")
    print(f"{'='*60}")

    plot_individual_predictions(
        df_fit_train, n_subjects=25, ncols=5,
        hlme_df=hlme_train_fit,
        save_path=os.path.join(args.output_dir, "fit_individual_train.png"))
    
    plot_individual_predictions(
        df_pred_test, n_subjects=25, ncols=5,
        hlme_df=hlme_val_pred,
        save_path=os.path.join(args.output_dir, "pred_individual_val.png"))

    hlme_pop = None
    if hlme_val_fit is not None:
        hlme_pop = population_averaged_predictions(
            hlme_val_fit)
    hlme_pop = pd.read_csv('results_3C/pop_ci_hlme_val.csv', delimiter=",")
    print(hlme_pop)
    print(ci_ode)
    plot_population_averaged(
        df_pop_test, hlme_pop=hlme_pop, mode="fit",
        ci_dict=ci_ode,
        save_path=os.path.join(args.output_dir, "population_average.png"))

    print("\nDone.")