"""
Last-Layer Variance for ∆PDP — Neural ODE-LMM.

The decoder is:  μ_{iℓ} = ρ(z_{iℓ}, BMI_std, AGEc) · β_neural

Since ∆PDP_ℓ = γ_ℓᵀ · β_neural  (linear in the last layer),
we compute:
    Var(β_neural) = (Σ_i X̂_iᵀ V_i⁻¹ X̂_i)⁻¹       [GLS Fisher]
    Var(∆PDP_ℓ)  = γ_ℓᵀ · Var(β_neural) · γ_ℓ       [delta method]

where:
    X̂_{iℓ}  = ρ_net(z_{iℓ}, BMI_std, AGEc)  ∈ R^p   (fixed-effect features)
    V_i      = Z_i D Z_iᵀ + σ² I                       (marginal covariance)
    γ_ℓ      = (1/n_ℓ) Σ_i m_{iℓ} [ρ^{hi}_{iℓ} − ρ^{lo}_{iℓ}]

This is the last-layer Laplace / sandwich approach described in
Dorigatti et al. (2023, AISTATS) applied to the ∆PDP estimand.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────
# 1.  Extract ρ features from the decoder (no β dot product)
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_rho_features(model, t_pad, x_pad, static, obs_mask, bmi_t=None):
    """
    Run the model forward and return the ρ features BEFORE the β dot product.

    Returns:
        rho:  (N, T, p)   — output of rho_net + LayerNorm
        Z_re: (N, T, q)   — random-effect design matrix
        D:    (q, q)       — RE covariance
        sig2: scalar       — residual variance
    """
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    t_pad = t_pad.to(device)
    x_pad = x_pad.to(device)
    static = static.to(device)
    obs_mask = obs_mask.to(device)
    if bmi_t is not None:
        bmi_t = bmi_t.to(device)

    N, T = t_pad.shape

    # --- Encoder + ODE ---
    t0 = t_pad[:, 0:1]
    bmi0 = x_pad[:, 0, 0:model.n_tv]
    encoder_in = torch.cat([t0, bmi0, static], dim=-1)
    z0 = model.encoder(encoder_in)
    zt = model._euler_integrate(z0, t_pad, static, bmi_t)
    zt = model.z_norm(zt)

    # --- Decoder internals (replicate forward without the β dot product) ---
    dec = model.decoder
    H = zt.shape[-1]

    # BMI standardization
    bmi_raw = x_pad[:, :, 0:1]
    bmi_std_val = (bmi_raw - dec.bmi_mean) / dec.bmi_std

    # Static skip
    if dec.static_skip_dims:
        static_exp = static[:, dec.static_skip_dims].unsqueeze(1).expand(-1, T, -1)
        skip_input = torch.cat([bmi_std_val, static_exp], dim=-1)
    else:
        skip_input = bmi_std_val

    rho_input = torch.cat([zt, skip_input], dim=-1)

    # ρ features
    rho = dec.rho_net(rho_input)       # (N, T, p)
    rho = dec.rho_norm(rho)            # (N, T, p)

    # Random-effect design matrix Z
    if dec.use_neural_re:
        Z_re = dec.g_net(zt)
        Z_re = dec.g_norm(Z_re)
    else:
        ones = torch.ones(N, T, 1, device=device, dtype=dtype)
        if dec.re_spline_cols is not None:
            rs_cols = x_pad[:, :, dec.re_spline_cols]
            Z_re = torch.cat([ones, rs_cols], dim=-1)
        else:
            Z_re = ones

    if obs_mask is not None:
        Z_re = Z_re * obs_mask.unsqueeze(-1)

    D = dec._build_D(device, dtype)
    sig2 = torch.exp(dec.log_residual_var)

    return rho, Z_re, D, sig2


@torch.no_grad()
def extract_rho_counterfactual(model, t_pad, x_pad, static, obs_mask,
                                bmi_value: float):
    """
    Run model with BMI replaced by a constant counterfactual value,
    return ρ features.

    Replicates exactly what PDP_ode.py / compute_pdp does:
      x_cf = x_pad.clone(); x_cf[:,:,0] = bmi_value
      model(t_pad, x_cf, ..., bmi_t=x_cf[:,:,0:1], ...)

    BMI flows through ALL three stages:
      1. Encoder  (baseline BMI0 = bmi_value)
      2. ODE      (bmi_t = bmi_value at all times)
      3. Decoder   (skip connection uses bmi_value)
    """
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    t_pad = t_pad.to(device)
    x_pad = x_pad.to(device)
    static = static.to(device)
    obs_mask = obs_mask.to(device)

    N, T = t_pad.shape

    # --- Build counterfactual x_cf (same as PDP_ode.py) ---
    x_cf = x_pad.clone()
    x_cf[:, :, 0] = bmi_value                          # BMI col = 0
    bmi_t_cf = x_cf[:, :, 0:1]                         # (N, T, 1)

    # --- Encoder with counterfactual baseline BMI ---
    t0 = t_pad[:, 0:1]
    bmi0_cf = x_cf[:, 0, 0:model.n_tv]                 # counterfactual BMI0
    encoder_in = torch.cat([t0, bmi0_cf, static], dim=-1)
    z0 = model.encoder(encoder_in)

    # --- ODE with counterfactual bmi_t ---
    zt = model._euler_integrate(z0, t_pad, static, bmi_t_cf)
    zt = model.z_norm(zt)

    # --- Decoder ρ features with counterfactual BMI ---
    dec = model.decoder

    bmi_std_cf = (bmi_t_cf - dec.bmi_mean) / dec.bmi_std

    if dec.static_skip_dims:
        static_exp = static[:, dec.static_skip_dims].unsqueeze(1).expand(-1, T, -1)
        skip_input = torch.cat([bmi_std_cf, static_exp], dim=-1)
    else:
        skip_input = bmi_std_cf

    rho_input = torch.cat([zt, skip_input], dim=-1)

    rho = dec.rho_net(rho_input)
    rho = dec.rho_norm(rho)

    return rho   # (N, T, p)


# ─────────────────────────────────────────────────────────
# 2.  Compute GLS Fisher information for β_neural
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def compute_fisher_beta(rho_all: torch.Tensor,
                        Z_re_all: torch.Tensor,
                        D: torch.Tensor,
                        sig2: torch.Tensor,
                        obs_mask_all: torch.Tensor,
                        jitter: float = 1e-4) -> torch.Tensor:
    """
    Compute the GLS Fisher information matrix for β_neural:

        I(β) = Σ_i  X̂_iᵀ V_i⁻¹ X̂_i

    where X̂_i = ρ_i (the rho features) and V_i = Z_i D Z_iᵀ + σ² I.

    We apply the observation mask to restrict to observed time points.

    Args:
        rho_all:     (N, T, p)  — ρ features for all subjects
        Z_re_all:    (N, T, q)  — RE design matrices
        D:           (q, q)     — RE covariance
        sig2:        scalar     — residual variance
        obs_mask_all:(N, T)     — observation mask

    Returns:
        I_beta: (p, p) — Fisher information matrix
    """
    N, T, p = rho_all.shape
    device = rho_all.device
    dtype = rho_all.dtype

    I_beta = torch.zeros(p, p, device=device, dtype=dtype)

    for i in range(N):
        mask_i = obs_mask_all[i]                      # (T,)
        idx = mask_i.bool()                           # observed indices

        if idx.sum() == 0:
            continue

        # Extract observed rows
        rho_i = rho_all[i, idx]                       # (n_i, p)
        Z_i = Z_re_all[i, idx]                        # (n_i, q)
        n_i = rho_i.shape[0]

        # V_i = Z_i D Z_iᵀ + σ² I
        V_i = Z_i @ D @ Z_i.t() + (sig2 + jitter) * torch.eye(n_i, device=device, dtype=dtype)

        # V_i⁻¹ via Cholesky
        try:
            L_i = torch.linalg.cholesky(V_i)
            V_inv_rho = torch.cholesky_solve(rho_i, L_i)   # (n_i, p)
        except torch.linalg.LinAlgError:
            # Fallback to direct solve
            V_inv_rho = torch.linalg.solve(V_i, rho_i)

        # Accumulate: X̂_iᵀ V_i⁻¹ X̂_i
        I_beta += rho_i.t() @ V_inv_rho                # (p, p)

    return I_beta


# ─────────────────────────────────────────────────────────
# 3.  Compute γ vector and ∆PDP variance at each visit time
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def compute_gamma_vectors(rho_hi: torch.Tensor,
                          rho_lo: torch.Tensor,
                          obs_mask: torch.Tensor,
                          times: torch.Tensor,
                          visit_times: np.ndarray) -> Dict[float, torch.Tensor]:
    """
    Compute γ_ℓ = (1/n_ℓ) Σ_i m_{iℓ} (ρ^{hi}_{iℓ} − ρ^{lo}_{iℓ})

    for each visit time in visit_times.

    Uses the same closest-observation matching as _closest_obs_per_subject
    in PDP_analysis_ODE.py: no distance threshold, ALL subjects contribute.

    Args:
        rho_hi:  (N, T, p)  — ρ features under BMI = v_hi
        rho_lo:  (N, T, p)  — ρ features under BMI = v_lo
        obs_mask:(N, T)     — observation mask
        times:   (N, T)     — observation times
        visit_times: array of times at which to evaluate

    Returns:
        gammas: dict {t: γ_t ∈ R^p}
    """
    N, T, p = rho_hi.shape
    device = rho_hi.device
    dtype = rho_hi.dtype
    delta_rho = rho_hi - rho_lo                        # (N, T, p)

    gammas = {}
    for vt in visit_times:
        gamma_sum = torch.zeros(p, device=device, dtype=dtype)
        n_obs = 0

        for i in range(N):
            obs_idx = torch.where(obs_mask[i] > 0.5)[0]
            if len(obs_idx) == 0:
                continue
            obs_times = times[i, obs_idx]
            closest = obs_idx[torch.argmin(torch.abs(obs_times - vt))]
            gamma_sum += delta_rho[i, closest]
            n_obs += 1

        if n_obs > 0:
            gammas[vt] = gamma_sum / n_obs
        else:
            gammas[vt] = torch.zeros(p, device=device, dtype=dtype)

    return gammas


# ─────────────────────────────────────────────────────────
# 4.  Main: compute ∆PDP variance (last-layer)
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def _accumulate_fisher_per_batch(rho_batch, Z_re_batch, D, sig2, mask_batch,
                                  jitter=1e-4,
                                  y_batch=None, beta=None):
    """
    Accumulate Fisher contribution for one batch (variable T per batch is OK).

    If y_batch and beta are provided, also accumulates the sandwich meat:
        M = Σ_i s_i s_iᵀ   where  s_i = X̂_iᵀ V_i⁻¹ (Y_i − X̂_i β)

    Returns:
        I_acc:  (p, p) — Fisher contribution
        M_acc:  (p, p) — Meat contribution (or None if y_batch/beta not given)
    """
    N, T, p = rho_batch.shape
    device = rho_batch.device
    dtype = rho_batch.dtype
    I_acc = torch.zeros(p, p, device=device, dtype=dtype)

    compute_meat = (y_batch is not None) and (beta is not None)
    M_acc = torch.zeros(p, p, device=device, dtype=dtype) if compute_meat else None

    for i in range(N):
        idx = mask_batch[i].bool()
        if idx.sum() == 0:
            continue
        rho_i = rho_batch[i, idx]                # (n_i, p)
        Z_i = Z_re_batch[i, idx]                 # (n_i, q)
        n_i = rho_i.shape[0]

        V_i = Z_i @ D @ Z_i.t() + (sig2 + jitter) * torch.eye(
            n_i, device=device, dtype=dtype)
        try:
            L_i = torch.linalg.cholesky(V_i)
            V_inv_rho = torch.cholesky_solve(rho_i, L_i)
        except torch.linalg.LinAlgError:
            V_inv_rho = torch.linalg.solve(V_i, rho_i)

        # Fisher: X̂_iᵀ V_i⁻¹ X̂_i
        I_acc += rho_i.t() @ V_inv_rho

        # Sandwich meat: s_i s_iᵀ  where s_i = X̂_iᵀ V_i⁻¹ r_i
        if compute_meat:
            y_i = y_batch[i, idx]                # (n_i,)
            r_i = y_i - rho_i @ beta             # (n_i,)  residual
            # V_i⁻¹ r_i
            try:
                V_inv_r = torch.cholesky_solve(r_i.unsqueeze(-1), L_i).squeeze(-1)
            except Exception:
                V_inv_r = torch.linalg.solve(V_i, r_i)
            s_i = rho_i.t() @ V_inv_r            # (p,)  per-subject score
            M_acc += s_i.unsqueeze(1) * s_i.unsqueeze(0)  # outer product

    return I_acc, M_acc


@torch.no_grad()
def _accumulate_gamma_per_batch(delta_rho, mask_batch, times_batch,
                                 visit_times, p):
    """
    Accumulate γ numerator and count for one batch.

    Mirrors _closest_obs_per_subject from PDP_analysis_ODE.py exactly:
    for each subject, find the observed index closest to each visit time
    (no distance threshold — ALL subjects contribute at every visit time).
    """
    N, T, _ = delta_rho.shape
    device = delta_rho.device
    dtype = delta_rho.dtype

    gamma_sums = {vt: torch.zeros(p, device=device, dtype=dtype)
                  for vt in visit_times}
    gamma_counts = {vt: 0 for vt in visit_times}

    for i in range(N):
        # obs_idx = np.where(masks_np[i] > 0.5)[0]
        obs_idx = torch.where(mask_batch[i] > 0.5)[0]
        if len(obs_idx) == 0:
            continue

        obs_times = times_batch[i, obs_idx]             # (n_obs,)

        for vt in visit_times:
            # closest = obs_idx[np.argmin(np.abs(obs_times - vt))]
            closest = obs_idx[torch.argmin(torch.abs(obs_times - vt))]
            gamma_sums[vt] += delta_rho[i, closest]
            gamma_counts[vt] += 1

    return gamma_sums, gamma_counts


@torch.no_grad()
def compute_delta_pdp_variance(
    model,
    dataloader,
    device: torch.device,
    bmi_lo: float = 20.0,
    bmi_hi: float = 35.0,
    visit_times: np.ndarray = np.array([0, 5, 10, 15]),
    true_beta_bmi: float = -0.175,
    true_beta_int: float = -0.015,
) -> Dict[str, object]:
    """
    Compute the last-layer variance of ∆PDP for the Neural ODE model.

    Processes data per-batch (never concatenates across batches) to handle
    variable padding lengths T across batches.

    Steps:
        1. Per-batch forward pass → accumulate Fisher I(β) and γ vectors
        2. Var(β) = I(β)⁻¹
        3. Var(∆PDP_ℓ) = γ_ℓᵀ · Var(β) · γ_ℓ

    Returns:
        Dictionary with var_beta, fisher, results, beta_neural, gammas.
    """
    model.eval()
    model = model.to(device)

    p = model.decoder.p
    q = model.decoder.q
    dec = model.decoder

    # --- Pass 1: accumulate Fisher I(β) and sandwich meat M(β) ---
    I_beta = torch.zeros(p, p, device='cpu')
    M_beta = torch.zeros(p, p, device='cpu')     # sandwich meat
    all_ages = []     # for computing mean(AGEc) later
    N_total = 0
    D_cpu = None
    sig2_cpu = None

    beta_neural = model.decoder.beta_neural.detach().cpu()

    print(f"\n{'='*60}")
    print(f"LAST-LAYER VARIANCE FOR ∆PDP (Fisher + Sandwich)")
    print(f"{'='*60}")
    print(f"Pass 1: Fisher I(β) and sandwich meat M(β)...")

    for batch in dataloader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        bmi_t = x_pad[:, :, 0:1]

        rho, Z_re, D, sig2 = extract_rho_features(
            model, t_pad, x_pad, s, mask, bmi_t=bmi_t
        )

        # Accumulate Fisher and meat on CPU
        rho_cpu = rho.cpu()
        Z_re_cpu = Z_re.cpu()
        D_cpu = D.cpu()
        sig2_cpu = sig2.cpu()
        mask_cpu = mask.cpu()
        y_cpu = y_pad.cpu()

        I_batch, M_batch = _accumulate_fisher_per_batch(
            rho_cpu, Z_re_cpu, D_cpu, sig2_cpu, mask_cpu,
            y_batch=y_cpu, beta=beta_neural)
        I_beta += I_batch
        M_beta += M_batch

        # Collect AGEc (column 1 of static)
        all_ages.append(s[:, 1].cpu())
        N_total += t_pad.shape[0]

    all_ages = torch.cat(all_ages)
    mean_age = all_ages.mean().item()

    print(f"  N subjects = {N_total}")
    print(f"  p (feature dim) = {p}, q (RE dim) = {q}")
    print(f"  σ² = {sig2_cpu.item():.6f}")
    print(f"  I(β) condition number: {torch.linalg.cond(I_beta).item():.2e}")

    # Var_fisher(β) = I(β)⁻¹
    try:
        I_beta_inv = torch.linalg.inv(I_beta)
    except torch.linalg.LinAlgError:
        print("  WARNING: Fisher not invertible, using pseudo-inverse")
        I_beta_inv = torch.linalg.pinv(I_beta)

    var_beta_fisher = I_beta_inv

    # Var_sandwich(β) = I(β)⁻¹ · M · I(β)⁻¹
    var_beta_sandwich = I_beta_inv @ M_beta @ I_beta_inv

    print(f"  β_neural = {beta_neural.tolist()}")
    print(f"  SE_fisher(β)   = {torch.sqrt(torch.diag(var_beta_fisher)).tolist()}")
    print(f"  SE_sandwich(β) = {torch.sqrt(torch.diag(var_beta_sandwich)).tolist()}")

    # --- Pass 2: counterfactual ρ → accumulate γ vectors ---
    print(f"\nPass 2: counterfactual ρ and γ vectors...")
    print(f"  BMI: lo={bmi_lo}, hi={bmi_hi}, Δv={bmi_hi - bmi_lo}")

    gamma_sums = {vt: torch.zeros(p) for vt in visit_times}
    gamma_counts = {vt: 0 for vt in visit_times}

    for batch in dataloader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch

        rho_hi = extract_rho_counterfactual(
            model, t_pad, x_pad, s, mask, bmi_value=bmi_hi
        ).cpu()
        rho_lo = extract_rho_counterfactual(
            model, t_pad, x_pad, s, mask, bmi_value=bmi_lo
        ).cpu()

        delta_rho = rho_hi - rho_lo                     # (B, T, p)

        g_sums, g_counts = _accumulate_gamma_per_batch(
            delta_rho, mask.cpu(), t_pad.cpu(), visit_times, p)

        for vt in visit_times:
            gamma_sums[vt] += g_sums[vt]
            gamma_counts[vt] += g_counts[vt]

    # Normalize γ
    gammas = {}
    for vt in visit_times:
        n_vt = gamma_counts[vt]
        if n_vt > 0:
            gammas[vt] = gamma_sums[vt] / n_vt
        else:
            gammas[vt] = torch.zeros(p)
        print(f"  t={vt}: n_obs={n_vt}, ||γ||={gammas[vt].norm().item():.4f}")

    # --- Step 3: Var(∆PDP) = γᵀ Var(β) γ  [both Fisher and Sandwich] ---
    delta_v = bmi_hi - bmi_lo
    results = {}

    print(f"\n  {'Time':>6s}  {'∆PDP_est':>10s}  {'SE_fish':>10s}  {'SE_sand':>10s}  "
          f"{'CI_lo':>10s}  {'CI_hi':>10s}  {'True':>10s}  {'Bias':>10s}")
    print(f"  {'-'*84}")

    for vt in visit_times:
        gamma = gammas[vt]                              # (p,)

        # ∆PDP estimate = γᵀ β
        delta_pdp_est = (gamma * beta_neural).sum().item()

        # Fisher variance = γᵀ I(β)⁻¹ γ
        var_fisher = (gamma @ var_beta_fisher @ gamma).item()
        se_fisher = np.sqrt(max(var_fisher, 0.0))

        # Sandwich variance = γᵀ I(β)⁻¹ M I(β)⁻¹ γ
        var_sandwich = (gamma @ var_beta_sandwich @ gamma).item()
        se_sandwich = np.sqrt(max(var_sandwich, 0.0))

        # Use SANDWICH as the primary SE for CI
        se_delta = se_sandwich
        ci_lo = delta_pdp_est - 1.96 * se_delta
        ci_hi = delta_pdp_est + 1.96 * se_delta

        # True ∆PDP (Scenario 2: β_BMI + β_int × mean(AGEc))
        true_delta = delta_v * (true_beta_bmi + true_beta_int * mean_age)

        bias = delta_pdp_est - true_delta

        results[vt] = {
            'estimate': delta_pdp_est,
            'se': se_delta,               # sandwich SE (primary)
            'se_fisher': se_fisher,
            'se_sandwich': se_sandwich,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'true': true_delta,
            'bias': bias,
            'gamma': gamma.numpy(),
        }

        print(f"  {vt:6.0f}  {delta_pdp_est:+10.4f}  {se_fisher:10.4f}  {se_sandwich:10.4f}  "
              f"{ci_lo:+10.4f}  {ci_hi:+10.4f}  {true_delta:+10.4f}  {bias:+10.4f}")

    print(f"\n  Mean AGEc = {mean_age:.4f}")
    print(f"  True formula: ∆PDP = Δv × (β_BMI + β_int × mean_AGEc)")
    print(f"               = {delta_v} × ({true_beta_bmi} + {true_beta_int} × {mean_age:.4f})")
    print(f"               = {true_delta:.4f}")
    print(f"\n  Note: CIs use sandwich SE (accounts for upstream misspecification).")
    print(f"  Fisher SE is reported for comparison.")

    return {
        'var_beta_fisher': var_beta_fisher.numpy(),
        'var_beta_sandwich': var_beta_sandwich.numpy(),
        'fisher': I_beta.numpy(),
        'meat': M_beta.numpy(),
        'results': results,
        'beta_neural': beta_neural.numpy(),
        'gammas': {vt: g.numpy() for vt, g in gammas.items()},
    }


# ─────────────────────────────────────────────────────────
# 5.  Multi-simulation aggregation
# ─────────────────────────────────────────────────────────

def aggregate_simulations(
    all_results: List[Dict],
    visit_times: np.ndarray,
) -> Dict:
    """
    Aggregate ∆PDP variance results across D simulated datasets.

    Computes:
        - Mean estimated ∆PDP
        - Mean SE (from last-layer variance)
        - Monte Carlo variance of ∆PDP across simulations
        - Coverage of 95% CIs
        - MSE = Bias² + MC Variance

    Args:
        all_results: list of dicts from compute_delta_pdp_variance
        visit_times: array of visit times

    Returns:
        Summary dict
    """
    D = len(all_results)

    summary = {}
    for vt in visit_times:
        estimates = [r['results'][vt]['estimate'] for r in all_results]
        ses = [r['results'][vt]['se'] for r in all_results]
        trues = [r['results'][vt]['true'] for r in all_results]
        ci_los = [r['results'][vt]['ci_lo'] for r in all_results]
        ci_his = [r['results'][vt]['ci_hi'] for r in all_results]

        est_arr = np.array(estimates)
        se_arr = np.array(ses)
        true_val = np.mean(trues)  # should be same across sims

        mean_est = est_arr.mean()
        bias = mean_est - true_val
        var_mc = est_arr.var(ddof=1)
        mean_se = se_arr.mean()
        var_est = mean_se ** 2
        mse = bias ** 2 + var_mc

        # Coverage: how often does the 95% CI contain the true value?
        coverage = np.mean([
            ci_lo <= true_val <= ci_hi
            for ci_lo, ci_hi in zip(ci_los, ci_his)
        ])

        summary[vt] = {
            'mean_est': mean_est,
            'true': true_val,
            'bias': bias,
            'var_mc': var_mc,
            'var_est': var_est,
            'mse': mse,
            'mean_se': mean_se,
            'coverage_95': coverage,
        }

    print(f"\n{'='*80}")
    print(f"AGGREGATED RESULTS OVER {D} SIMULATIONS (Last-Layer Variance)")
    print(f"{'='*80}")
    print(f"  {'t':>4s}  {'∆PDP':>8s}  {'∆PDP₀':>8s}  {'Bias':>8s}  "
          f"{'Var_MC':>9s}  {'Var_Est':>9s}  {'MSE':>9s}  {'Cov95':>6s}")
    print(f"  {'-'*70}")

    for vt in visit_times:
        s = summary[vt]
        print(f"  {vt:4.0f}  {s['mean_est']:+8.4f}  {s['true']:+8.4f}  "
              f"{s['bias']:+8.4f}  {s['var_mc']:9.5f}  {s['var_est']:9.5f}  "
              f"{s['mse']:9.5f}  {s['coverage_95']:6.1%}")

    return summary


# ─────────────────────────────────────────────────────────
# 6.  Standalone entrypoint
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import csv
    import pyreadr
    from torch.utils.data import DataLoader
    from dataset import LongitudinalDataset, collate_pad
    from model_ODE import NeuralODEModel, NeuralODEConfig

    parser = argparse.ArgumentParser(description="Last-layer ∆PDP variance")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_model_ode_full_skip_0.pt")
    parser.add_argument("--data", type=str,
                        default="simu_datasets/S2a_sims/sim_001.rds")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--true_beta_bmi", type=float, default=-0.30)
    parser.add_argument("--true_beta_int", type=float, default=-0.05)
    parser.add_argument("--n_sims", type=int, default=1,
                        help="Number of simulations to aggregate (1=single)")
    parser.add_argument("--output_csv", type=str,
                        default="results/delta_pdp_variance_ode.csv")
    # BMI pairs: same as LMM analysis
    parser.add_argument("--bmi_pairs", type=str, default=None,
                        help="Comma-sep pairs: 20:23,20:26,... (default: standard grid)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visit_times = np.array([0, 5, 10, 15])

    # BMI pair grid (matching LMM analysis)
    if args.bmi_pairs:
        bmi_pairs = [tuple(map(float, p.split(':'))) for p in args.bmi_pairs.split(',')]
    else:
        # Standard grid: consecutive pairs from [20, 23, 26, 29, 32, 35]
        bmi_grid = [20, 23, 26, 29, 32, 35]
        bmi_pairs = [(bmi_grid[i], bmi_grid[i+1]) for i in range(len(bmi_grid)-1)]
        # Also add 20→35 for the full range
        bmi_pairs.append((20, 35))

    print(f"BMI pairs: {bmi_pairs}")
    print(f"N simulations: {args.n_sims}")

    # Column definitions
    time_col, y_col, id_col = "time", "ISA15_sim", "NUM_ID"
    x_cols = ["BMI_t", "rs1", "rs2"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    # Storage: {(bmi_lo, bmi_hi): [result_sim0, result_sim1, ...]}
    all_pair_results = {pair: [] for pair in bmi_pairs}

    for sim_idx in range(9, args.n_sims):
        if args.n_sims > 1:
            data_path = f"simu_datasets/S2a_sims_2/sim_{sim_idx+1:03d}.rds"
            ckpt_path = f"checkpoints/best_model_ode_full_skip_{sim_idx}.pt"
            print(f"\n{'#'*60}")
            print(f"# SIMULATION {sim_idx}")
            print(f"{'#'*60}")
        else:
            data_path = args.data
            ckpt_path = args.checkpoint

        # --- Load data ---
        df = next(iter(pyreadr.read_r(data_path).values()))
        df["SEX"] = df["SEX"].astype("category")
        df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
        df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
        df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

        dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col,
                                      static_cols=static_cols)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_pad)

        # --- Build model ---
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

        # --- Load checkpoint ---
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        print(f"Loaded: {ckpt_path}")

        # --- Compute variance for each BMI pair ---
        for bmi_lo, bmi_hi in bmi_pairs:
            result = compute_delta_pdp_variance(
                model, loader, device,
                bmi_lo=bmi_lo, bmi_hi=bmi_hi,
                visit_times=visit_times,
                true_beta_bmi=args.true_beta_bmi,
                true_beta_int=args.true_beta_int,
            )
            all_pair_results[(bmi_lo, bmi_hi)].append(result)

    # --- Aggregate across simulations and write CSV ---
    import os
    os.makedirs(os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else '.', exist_ok=True)

    csv_rows = []
    header = ['time', 'BMI_lo', 'BMI_hi', 'D', 'mean_hat', 'mean_true',
              'bias', 'var_mc', 'var_fisher', 'var_sandwich', 'mse', 'rmse',
              'coverage95_fisher', 'coverage95_sandwich']

    print(f"\n{'='*110}")
    print(f"AGGREGATED RESULTS — Last-Layer Variance (D={args.n_sims} simulations)")
    print(f"{'='*110}")
    print(f"{'time':>4s} {'lo':>4s} {'hi':>4s} {'D':>3s}  {'mean_hat':>10s} {'mean_true':>10s} "
          f"{'bias':>8s}  {'var_mc':>10s} {'var_fish':>10s} {'var_sand':>10s} "
          f"{'cov_fish':>8s} {'cov_sand':>8s}")
    print(f"{'-'*105}")

    for (bmi_lo, bmi_hi), results_list in all_pair_results.items():
        D = len(results_list)

        for vt in visit_times:
            estimates = [r['results'][vt]['estimate'] for r in results_list]
            ses_fish = [r['results'][vt]['se_fisher'] for r in results_list]
            ses_sand = [r['results'][vt]['se_sandwich'] for r in results_list]
            trues = [r['results'][vt]['true'] for r in results_list]
            ci_los = [r['results'][vt]['ci_lo'] for r in results_list]
            ci_his = [r['results'][vt]['ci_hi'] for r in results_list]

            est_arr = np.array(estimates)
            se_fish_arr = np.array(ses_fish)
            se_sand_arr = np.array(ses_sand)
            true_val = np.mean(trues)

            mean_hat = est_arr.mean()
            bias = mean_hat - true_val
            var_mc = est_arr.var(ddof=1) if D > 1 else float('nan')
            var_fisher = (se_fish_arr ** 2).mean()
            var_sandwich = (se_sand_arr ** 2).mean()
            mse = (bias ** 2 + var_mc) if D > 1 else float('nan')
            rmse = np.sqrt(mse) if D > 1 else float('nan')

            # Coverage (Fisher-based CIs)
            if D > 1:
                ci_fish_los = [e - 1.96 * s for e, s in zip(estimates, ses_fish)]
                ci_fish_his = [e + 1.96 * s for e, s in zip(estimates, ses_fish)]
                coverage_fisher = np.mean([
                    lo <= true_val <= hi
                    for lo, hi in zip(ci_fish_los, ci_fish_his)
                ])
                # Coverage (Sandwich-based CIs) — these are the ci_lo/ci_hi stored
                coverage_sandwich = np.mean([
                    lo <= true_val <= hi
                    for lo, hi in zip(ci_los, ci_his)
                ])
            else:
                coverage_fisher = float('nan')
                coverage_sandwich = float('nan')

            row = {
                'time': int(vt),
                'BMI_lo': int(bmi_lo),
                'BMI_hi': int(bmi_hi),
                'D': D,
                'mean_hat': mean_hat,
                'mean_true': true_val,
                'bias': bias,
                'var_mc': var_mc,
                'var_fisher': var_fisher,
                'var_sandwich': var_sandwich,
                'mse': mse,
                'rmse': rmse,
                'coverage95_fisher': coverage_fisher,
                'coverage95_sandwich': coverage_sandwich,
            }
            csv_rows.append(row)

            # Print
            vm_str = f"{var_mc:.6f}" if not np.isnan(var_mc) else "NA"
            cov_f_str = f"{coverage_fisher:.2f}" if not np.isnan(coverage_fisher) else "NA"
            cov_s_str = f"{coverage_sandwich:.2f}" if not np.isnan(coverage_sandwich) else "NA"
            print(f"{int(vt):4d} {int(bmi_lo):4d} {int(bmi_hi):4d} {D:3d}  "
                  f"{mean_hat:+10.4f} {true_val:+10.4f} {bias:+8.4f}  "
                  f"{vm_str:>10s} {var_fisher:10.6f} {var_sandwich:10.6f} "
                  f"{cov_f_str:>8s} {cov_s_str:>8s}")

    # Write CSV
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nCSV saved to {args.output_csv}")
    print(f"Columns: {','.join(header)}")