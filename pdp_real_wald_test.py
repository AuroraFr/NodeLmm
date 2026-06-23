"""
Omnibus Wald tests for trajectory-profile ΔPDP contrasts.

Three tests per pairwise contrast:
  1. Time-averaged Z-test:       H₀: (1/L) Σₗ ΔPDP(tₗ) = 0
  2. Late-time averaged Z-test:  H₀: (1/L') Σ_{ℓ>ℓ₀} ΔPDP(tₗ) = 0
  3. Joint Wald χ² test:         H₀: ΔPDP(t₁) = ... = ΔPDP(t_L) = 0

All three use the delta-method covariance through F⁻¹, consistent
with the pointwise CIs in PDP_variance.py.

Usage:
    from wald_test_pdp import wald_test_all_pairs

    # ci_results from compute_trajectory_profile_pdp_with_ci()
    wald_results = wald_test_all_pairs(ci_results, eval_grid, fisher_inv)
"""
from __future__ import annotations
import numpy as np
import torch
from scipy import stats


# ─────────────────────────────────────────────────────────
#  Core: Wald tests for a single pairwise ΔPDP contrast
# ─────────────────────────────────────────────────────────

def wald_test_delta_pdp(
    delta_mean: np.ndarray,      # (L,)   point estimates
    delta_grad: np.ndarray,      # (L, P) gradient matrix D
    fisher_inv: torch.Tensor | np.ndarray,  # (P, P) F⁻¹
    eval_grid: np.ndarray,       # (L,)   time points
    late_cutoff: float = 7.0,    # only grid points with t >= this for late-time test
    chi2_reg: float = 1e-6,      # Tikhonov regularisation for Σ_Δ inversion
) -> dict:
    """
    Compute omnibus Wald tests for H₀: ΔPDP(t) ≡ 0.

    Returns dict with keys:
      'time_avg':  {stat, pvalue, delta_bar, se, description}
      'late_avg':  {stat, pvalue, delta_bar, se, t_start, description}
      'joint_chi2': {stat, df, pvalue, description}
      'joint_chi2_late': {stat, df, pvalue, t_start, description}
    """
    if isinstance(fisher_inv, np.ndarray):
        F_inv = torch.from_numpy(fisher_inv).float()
    else:
        F_inv = fisher_inv.float()

    L, P = delta_grad.shape
    D = torch.from_numpy(delta_grad).float()   # (L, P)
    delta = delta_mean                           # (L,)

    results = {}

    # ── 1. Time-averaged Z-test (all time points) ────────
    d_bar = D.mean(dim=0)                        # (P,)  average gradient
    delta_bar = delta.mean()                     # scalar
    var_bar = (d_bar @ F_inv @ d_bar).item()
    se_bar = np.sqrt(max(var_bar, 0.0))

    if se_bar > 0:
        z_stat = delta_bar / se_bar
        p_val = 2 * stats.norm.sf(abs(z_stat))
    else:
        z_stat, p_val = 0.0, 1.0

    results['time_avg'] = {
        'stat': z_stat,
        'pvalue': p_val,
        'delta_bar': delta_bar,
        'se': se_bar,
        'ci_lo': delta_bar - 1.96 * se_bar,
        'ci_hi': delta_bar + 1.96 * se_bar,
        'description': f'Z = {z_stat:.3f}, p = {p_val:.4f}  '
                        f'(Δ̄ = {delta_bar:.3f} ± {1.96*se_bar:.3f})',
    }

    # ── 2. Late-time averaged Z-test ─────────────────────
    late_mask = eval_grid >= late_cutoff
    if late_mask.sum() >= 2:
        d_bar_late = D[late_mask].mean(dim=0)
        delta_bar_late = delta[late_mask].mean()
        var_late = (d_bar_late @ F_inv @ d_bar_late).item()
        se_late = np.sqrt(max(var_late, 0.0))

        if se_late > 0:
            z_late = delta_bar_late / se_late
            p_late = 2 * stats.norm.sf(abs(z_late))
        else:
            z_late, p_late = 0.0, 1.0

        results['late_avg'] = {
            'stat': z_late,
            'pvalue': p_late,
            'delta_bar': delta_bar_late,
            'se': se_late,
            'ci_lo': delta_bar_late - 1.96 * se_late,
            'ci_hi': delta_bar_late + 1.96 * se_late,
            't_start': late_cutoff,
            'n_points': int(late_mask.sum()),
            'description': f'Z = {z_late:.3f}, p = {p_late:.4f}  '
                            f'(Δ̄_late = {delta_bar_late:.3f} ± '
                            f'{1.96*se_late:.3f}, t≥{late_cutoff})',
        }

    # ── 3. Joint Wald χ² test (all time points) ──────────
    results['joint_chi2'] = _chi2_wald(
        delta, D, F_inv, chi2_reg, label='all')

    # ── 4. Joint Wald χ² test (late time points) ─────────
    if late_mask.sum() >= 2:
        res_late = _chi2_wald(
            delta[late_mask], D[late_mask], F_inv, chi2_reg, label='late')
        res_late['t_start'] = late_cutoff
        res_late['n_points'] = int(late_mask.sum())
        results['joint_chi2_late'] = res_late

    return results


def _chi2_wald(delta, D, F_inv, reg, label=''):
    """
    W = Δ̂ᵀ Σ_Δ⁻¹ Δ̂  where Σ_Δ = D F⁻¹ Dᵀ.

    Regularise Σ_Δ with Tikhonov to handle near-singularity
    from highly correlated adjacent time points.
    """
    delta_t = torch.from_numpy(np.asarray(delta)).float()
    Sigma_delta = D @ F_inv @ D.T                 # (L, L) or (L', L')
    k = Sigma_delta.shape[0]

    # Tikhonov regularisation
    Sigma_reg = Sigma_delta + reg * torch.eye(k)

    # Check effective rank
    eigvals = torch.linalg.eigvalsh(Sigma_reg)
    eff_rank = (eigvals > eigvals.max() * 1e-8).sum().item()

    try:
        L_chol = torch.linalg.cholesky(Sigma_reg)
        v = torch.cholesky_solve(delta_t.unsqueeze(-1), L_chol).squeeze(-1)
        W = (delta_t @ v).item()
    except torch.linalg.LinAlgError:
        # Fall back to pseudo-inverse
        Sigma_pinv = torch.linalg.pinv(Sigma_reg)
        W = (delta_t @ Sigma_pinv @ delta_t).item()

    # Use effective rank as df (accounts for collinearity)
    df = min(eff_rank, k)
    p_val = stats.chi2.sf(W, df=df)

    return {
        'stat': W,
        'df': df,
        'df_nominal': k,
        'eff_rank': eff_rank,
        'pvalue': p_val,
        'description': f'χ²({df}) = {W:.2f}, p = {p_val:.4f}  '
                        f'(eff_rank={eff_rank}/{k})',
    }


# ─────────────────────────────────────────────────────────
#  Convenience: run all pairwise contrasts
# ─────────────────────────────────────────────────────────

DEFAULT_PAIRS = [
    ("gradual_rise", "late_spike"),
    ("stable_high", "stable_low"),
    ("gradual_decline", "late_decline"),
]

PAIR_LABELS = {
    ("late_decline", "late_spike"):
        "Late Decline − Late spike (path-dependence)",
    ("stable_high", "stable_low"):
        "Stable high − Stable low (level effect)",
    ("gradual_decline", "gradual_rise"):
        "Gradual decline − Gradual rise (trend effect)",
}


def wald_test_all_pairs(
    ci_results: dict,
    eval_grid: np.ndarray,
    fisher_inv: torch.Tensor | np.ndarray,
    pairs: list | None = None,
    late_cutoff: float = 7.0,
    chi2_reg: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    Run omnibus Wald tests for all pairwise ΔPDP contrasts.

    Args:
        ci_results:  output of compute_trajectory_profile_pdp_with_ci()
        eval_grid:   (L,) time grid
        fisher_inv:  (P, P) F⁻¹
        pairs:       list of (profile_a, profile_b) tuples
        late_cutoff: time threshold for late-time tests
        chi2_reg:    regularisation for Σ_Δ inversion

    Returns:
        dict {(pa, pb): wald_test_delta_pdp result}
    """
    if pairs is None:
        pairs = DEFAULT_PAIRS

    all_results = {}

    for pa, pb in pairs:
        if pa not in ci_results or pb not in ci_results:
            continue

        delta_mean = ci_results[pa]['mean'] - ci_results[pb]['mean']
        delta_grad = ci_results[pa]['grad'] - ci_results[pb]['grad']

        res = wald_test_delta_pdp(
            delta_mean, delta_grad, fisher_inv, eval_grid,
            late_cutoff=late_cutoff, chi2_reg=chi2_reg,
        )
        all_results[(pa, pb)] = res

        if verbose:
            label = PAIR_LABELS.get((pa, pb), f'{pa} − {pb}')
            print(f"\n  ═══ {label} ═══")
            print(f"    Time-averaged:     {res['time_avg']['description']}")
            if 'late_avg' in res:
                print(f"    Late-time avg:     {res['late_avg']['description']}")
            print(f"    Joint χ² (all):    {res['joint_chi2']['description']}")
            if 'joint_chi2_late' in res:
                print(f"    Joint χ² (late):   {res['joint_chi2_late']['description']}")

    # ── Summary table ────────────────────────────────────
    if verbose and all_results:
        print("\n  ┌─────────────────────────────────────────"
              "───────────────────────────────────┐")
        print(f"  │ {'Contrast':<35s} │ {'Z (all)':<14s} │ "
              f"{'Z (late)':<14s} │ {'χ² (all)':<14s} │")
        print("  ├─────────────────────────────────────────"
              "───────────────────────────────────┤")
        for (pa, pb), res in all_results.items():
            label = f'{pa} − {pb}'[:35]
            z_all = (f"{res['time_avg']['stat']:+.2f} "
                     f"(p={res['time_avg']['pvalue']:.3f})")
            z_late = (f"{res['late_avg']['stat']:+.2f} "
                      f"(p={res['late_avg']['pvalue']:.3f})"
                      if 'late_avg' in res else '—')
            chi2 = (f"{res['joint_chi2']['stat']:.1f} "
                    f"(p={res['joint_chi2']['pvalue']:.3f})")
            print(f"  │ {label:<35s} │ {z_all:<14s} │ "
                  f"{z_late:<14s} │ {chi2:<14s} │")
        print("  └─────────────────────────────────────────"
              "───────────────────────────────────┘")

    return all_results