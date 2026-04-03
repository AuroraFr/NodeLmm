"""
PDP analysis for Neural CDE-LMM (generalized).

Two classes of interventions:

A) Standard PDP (constant level interventions):
   BMI(t) = v for all t → dBMI = 0 → oracle ΔPDP = 0 for S6

B) Trajectory-class PDP (path interventions):
   Compare counterfactual BMI trajectories that differ in dynamics.
   Tests whether the CDE captures rate-of-change effects.

Intervention profiles and their S6 oracles:

  DGP: h₆(t) = α ∫₀ᵗ |dBMI/dτ| dτ,  β_BMI = 0

  ┌──────────────────────────┬──────────────────────────────────────────┐
  │ Profile                  │ Oracle Δ = α × [TV(path₁) - TV(path₀)] │
  ├──────────────────────────┼──────────────────────────────────────────┤
  │ Constant v₁ vs v₀       │ 0 (both TV = 0)                         │
  │ Shifted v₁ vs v₀        │ 0 (same shape → same TV)                │
  │ Stable vs Volatile       │ α × 4A × t/T  (sinusoidal TV)          │
  │ Slow ramp vs fast ramp  │ α × (|s₁| - |s₀|) × t                  │
  │ Step vs stable           │ 0 if t<t*, α×|Δv| if t≥t*             │
  └──────────────────────────┴──────────────────────────────────────────┘
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ═════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════

def _closest_obs_per_subject(mu, masks_np, times_np, visit_times):
    N, T = mu.shape
    result = {}
    for vt in visit_times:
        preds = []
        for i in range(N):
            obs_idx = np.where(masks_np[i] > 0.5)[0]
            if len(obs_idx) == 0:
                continue
            obs_times = times_np[i, obs_idx]
            closest = obs_idx[np.argmin(np.abs(obs_times - vt))]
            preds.append(mu[i, closest])
        result[vt] = np.array(preds)
    return result


def _pad_and_cat(tensors, max_T):
    padded = []
    for m in tensors:
        if m.shape[1] < max_T:
            pad = torch.zeros(m.shape[0], max_T - m.shape[1])
            m = torch.cat([m, pad], dim=1)
        padded.append(m)
    return torch.cat(padded, dim=0)


# ═════════════════════════════════════════════
# Counterfactual path constructors
# ═════════════════════════════════════════════

def make_constant_path(x_pad, bmi_col=0, value=25.0):
    """BMI(t) = v for all t. TV = 0."""
    x_cf = x_pad.clone()
    x_cf[:, :, bmi_col] = value
    return x_cf


def make_shifted_path(x_pad, mask, bmi_col=0, value=25.0):
    """BMI_i(t) = BMI_i(t) - mean(BMI_i) + v. Same shape, shifted level. TV unchanged."""
    x_cf = x_pad.clone()
    bmi_real = x_pad[:, :, bmi_col]
    bmi_masked = bmi_real * mask
    n_obs = mask.sum(dim=1, keepdim=True).clamp(min=1)
    bmi_mean = bmi_masked.sum(dim=1, keepdim=True) / n_obs
    x_cf[:, :, bmi_col] = bmi_real - bmi_mean + value
    return x_cf


def make_stable_path(x_pad, bmi_col=0):
    """BMI(t) = BMI₀ for all t (freeze at baseline). TV = 0."""
    x_cf = x_pad.clone()
    x_cf[:, :, bmi_col] = x_pad[:, 0:1, bmi_col]
    return x_cf


def make_volatile_path(x_pad, t_pad, bmi_col=0, amplitude=1, period=5.0):
    """
    BMI(t) = BMI₀ + A·sin(2πt/T).
    Same time-averaged mean as baseline. Adds sinusoidal volatility.

    Continuous TV(t) = 4A × t/T  (4A per full period).
    """
    x_cf = x_pad.clone()
    bmi0 = x_pad[:, 0:1, bmi_col]
    x_cf[:, :, bmi_col] = bmi0 + amplitude * torch.sin(2 * np.pi * t_pad / period)
    return x_cf


def make_ramp_path(x_pad, t_pad, bmi_col=0, bmi_start=25.0, slope=0.5):
    """
    BMI(t) = bmi_start + slope × t.

    TV(t) = |slope| × t.
    """
    x_cf = x_pad.clone()
    x_cf[:, :, bmi_col] = bmi_start + slope * t_pad
    return x_cf


def make_step_path(x_pad, t_pad, bmi_col=0, bmi_before=25.0, bmi_after=30.0, t_step=5.0):
    """
    BMI jumps from bmi_before to bmi_after at t_step.

    TV = |bmi_after - bmi_before| (single jump).
    h₆(t) = 0 for t < t_step, α×|Δv| for t ≥ t_step.
    """
    x_cf = x_pad.clone()
    before = (t_pad < t_step).float()
    x_cf[:, :, bmi_col] = bmi_before * before + bmi_after * (1 - before)
    return x_cf


# ═════════════════════════════════════════════
# Oracle ΔPDP functions (S6 specific)
# ═════════════════════════════════════════════

def oracle_constant(visit_times):
    """Constant intervention: ΔPDP = 0 at all times."""
    return {vt: 0.0 for vt in visit_times}


def oracle_shifted(visit_times):
    """Shifted intervention: same TV → ΔPDP = 0 at all times."""
    return {vt: 0.0 for vt in visit_times}


def oracle_volatile_vs_stable_discrete(loader, device, bmi_col=0,
                                        alpha=-0.15, amplitude=3.0, period=5.0,
                                        visit_times=None):
    """
    Oracle using discrete TV — matches what the CDE actually integrates.
    """
    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])

    all_delta_tv = {vt: [] for vt in visit_times}

    for batch in loader:
        _, t_pad, x_pad, _, _, mask, _ = batch
        N, T = t_pad.shape

        for i in range(N):
            obs = torch.where(mask[i] > 0.5)[0]
            if len(obs) < 2:
                continue
            t_obs = t_pad[i, obs].numpy()

            # Volatile path at observed times
            bmi0 = x_pad[i, obs[0], bmi_col].item()
            bmi_vol = bmi0 + amplitude * np.sin(2 * np.pi * t_obs / period)

            # Stable path
            bmi_stable = np.full_like(t_obs, bmi0)

            # Discrete TV
            tv_vol = np.cumsum(np.abs(np.diff(bmi_vol, prepend=bmi_vol[0])))
            tv_sta = np.cumsum(np.abs(np.diff(bmi_stable, prepend=bmi_stable[0])))

            for vt in visit_times:
                closest = np.argmin(np.abs(t_obs - vt))
                delta_tv = tv_vol[closest] - tv_sta[closest]
                all_delta_tv[vt].append(alpha * delta_tv)

    oracle = {vt: np.mean(all_delta_tv[vt]) for vt in visit_times}

    print(f"\n  Discrete oracle (volatile vs stable):")
    print(f"  {'Time':>6s}  {'Continuous':>12s}  {'Discrete':>12s}")
    print(f"  {'-'*35}")
    for vt in visit_times:
        continuous = alpha * 4 * amplitude * vt / period
        print(f"  {vt:6.0f}  {continuous:+12.4f}  {oracle[vt]:+12.4f}")

    return oracle


def oracle_ramp(visit_times, alpha=-0.15, slope_fast=1.0, slope_slow=0.2):
    """
    Fast ramp vs slow ramp:
      TV_fast(t) = |slope_fast| × t
      TV_slow(t) = |slope_slow| × t
      Δ(t) = α × (|slope_fast| - |slope_slow|) × t
    """
    return {vt: alpha * (abs(slope_fast) - abs(slope_slow)) * vt for vt in visit_times}


def oracle_step_vs_stable(visit_times, alpha=-0.15, bmi_before=25.0, bmi_after=30.0,
                           t_step=5.0):
    """
    Step vs stable:
      TV_stable = 0
      TV_step = |Δv| at t_step, then 0
      Δ(t) = 0 for t < t_step, α×|Δv| for t ≥ t_step
    """
    delta_bmi = abs(bmi_after - bmi_before)
    return {vt: (alpha * delta_bmi if vt >= t_step else 0.0) for vt in visit_times}


# ═════════════════════════════════════════════
# A) Standard PDP (constant level interventions)
# ═════════════════════════════════════════════

def compute_pdp(model, loader, device, bmi_values, n_tv=1,
                bmi_col=0, bmi_mode="constant", bmi_slope=None,
                interp="cubic"):
    model.eval()
    results = {v: [] for v in bmi_values}
    all_ages, all_masks, all_times = [], [], []

    print(f"  Computing causal PDP (CDE, interp='{interp}', bmi_mode='{bmi_mode}')")

    # Estimate slope if needed
    if bmi_mode == "linear" and bmi_slope is None:
        all_bmi, all_t = [], []
        with torch.no_grad():
            for batch in loader:
                _, t_pad_b, x_pad_b, _, _, mask_b, _ = batch
                obs = mask_b > 0.5
                all_bmi.append(x_pad_b[:, :, bmi_col][obs].numpy())
                all_t.append(t_pad_b[obs].numpy())
        bmi_flat = np.concatenate(all_bmi)
        t_flat = np.concatenate(all_t)
        from numpy.polynomial.polynomial import polyfit
        c = polyfit(t_flat, bmi_flat, 1)
        bmi_slope = c[1]
        print(f"  Estimated population BMI slope: {bmi_slope:.4f} per year")

    with torch.no_grad():
        for bmi_v in bmi_values:
            batch_mus = []
            for batch in loader:
                _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
                t_pad = t_pad.to(device)
                x_pad = x_pad.to(device)
                mask = mask.to(device)
                c_mask = c_mask.to(device)
                s = s.to(device)
                N, T = t_pad.shape

                # Counterfactual
                if bmi_mode == "constant":
                    x_cf = make_constant_path(x_pad, bmi_col, bmi_v)
                elif bmi_mode == "linear":
                    x_cf = make_ramp_path(x_pad, t_pad, bmi_col, bmi_v, bmi_slope)
                elif bmi_mode == "shifted":
                    x_cf = make_shifted_path(x_pad, mask, bmi_col, bmi_v)
                else:
                    raise ValueError(f"Unknown bmi_mode: '{bmi_mode}'")

                # # Causal: truncate at each ℓ, call model() normally
                # mu_all = torch.zeros(N, T, device=t_pad.device)

                # for ell in range(2, T + 1):
                #     # Truncate all inputs to [0, ℓ)
                #     t_trunc     = t_pad[:, :ell]
                #     x_trunc     = x_cf[:, :ell, :]
                #     mask_trunc  = mask[:, :ell]
                #     if c_mask.dim() == 2:
                #         cmask_trunc = c_mask[:, :ell]
                #     else:
                #         cmask_trunc = c_mask[:, :ell, :]

                #     # Model handles augmentation, encoder, CDE, decoder
                #     mu_ell, _, _, _, _, _ = model(
                #         t_trunc, x_trunc, cmask_trunc, s,
                #         obs_mask=mask_trunc, y_pad=None,
                #         interp=interp
                #     )
                #     # Extract prediction at last time point
                #     mu_all[:, ell - 1] = mu_ell[:, -1]

                # batch_mus.append(mu_all.cpu())
                mu, V, Z, D, sig2, _ = model(
                    t_pad, x_cf, c_mask, s, mask,
                    y_pad=None, interp=interp
                )
                batch_mus.append(mu.cpu())

                if bmi_v == bmi_values[0]:
                    all_ages.append(s[:, 1].cpu())
                    all_masks.append(mask.cpu())
                    all_times.append(t_pad.cpu())

            max_T = max(m.shape[1] for m in batch_mus)
            results[bmi_v] = _pad_and_cat(batch_mus, max_T)

    ages = torch.cat(all_ages, dim=0)
    max_T = max(m.shape[1] for m in all_masks)
    masks = _pad_and_cat(all_masks, max_T)
    times = _pad_and_cat(all_times, max_T)
    return results, ages, masks, times


def compute_pdp_causal(model, loader, device, bmi_values, n_tv=1,
                       bmi_col=0, bmi_mode="constant", bmi_slope=None,
                       interp="cubic", visit_times=None):
    """
    PDP in prediction mode: at each visit time, model sees path up to t only.
    """
    model.eval()
    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])

    results = {v: [] for v in bmi_values}
    all_ages, all_masks, all_times = [], [], []

    with torch.no_grad():
        for bmi_v in bmi_values:
            batch_preds = []

            for batch in loader:
                _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
                t_pad = t_pad.to(device)
                x_pad = x_pad.to(device)
                mask = mask.to(device)
                c_mask = c_mask.to(device)
                s = s.to(device)
                N, T = t_pad.shape

                # Counterfactual
                if bmi_mode == "constant":
                    x_cf = make_constant_path(x_pad, bmi_col, bmi_v)
                elif bmi_mode == "shifted":
                    x_cf = make_shifted_path(x_pad, mask, bmi_col, bmi_v)
                else:
                    x_cf = make_constant_path(x_pad, bmi_col, bmi_v)

                # For each subject, predict at each observed time
                # using path truncated up to that time
                mu_all = torch.zeros(N, T, device=device)

                for i in range(N):
                    obs_idx = torch.where(mask[i] > 0.5)[0]
                    if len(obs_idx) < 2:
                        continue

                    for k_pos in range(1, len(obs_idx)):
                        k = obs_idx[k_pos].item()

                        # Truncate path up to visit k (inclusive)
                        t_trunc = t_pad[i:i+1, :k+1]
                        x_trunc = x_cf[i:i+1, :k+1, :]
                        mask_trunc = mask[i:i+1, :k+1]
                        if c_mask.dim() == 2:
                            cm_trunc = c_mask[i:i+1, :k+1]
                        else:
                            cm_trunc = c_mask[i:i+1, :k+1, :]

                        mu_k, _, _, _, _, _ = model(
                            t_trunc, x_trunc, cm_trunc, s[i:i+1],
                            obs_mask=mask_trunc, y_pad=None,
                            interp=interp
                        )
                        mu_all[i, k] = mu_k[0, -1]

                    # First visit: use 2-point forward if possible
                    k0 = obs_idx[0].item()
                    if len(obs_idx) >= 2:
                        k1 = obs_idx[1].item()
                        t_tr = t_pad[i:i+1, :k1+1]
                        x_tr = x_cf[i:i+1, :k1+1, :]
                        m_tr = mask[i:i+1, :k1+1]
                        cm_tr = c_mask[i:i+1, :k1+1] if c_mask.dim() == 2 \
                            else c_mask[i:i+1, :k1+1, :]
                        mu_k0, _, _, _, _, _ = model(
                            t_tr, x_tr, cm_tr, s[i:i+1],
                            obs_mask=m_tr, y_pad=None, interp=interp
                        )
                        mu_all[i, k0] = mu_k0[0, 0]

                batch_preds.append(mu_all.cpu())

                if bmi_v == bmi_values[0]:
                    all_ages.append(s[:, 1].cpu())
                    all_masks.append(mask.cpu())
                    all_times.append(t_pad.cpu())

            max_T = max(m.shape[1] for m in batch_preds)
            results[bmi_v] = _pad_and_cat(batch_preds, max_T)

    ages = torch.cat(all_ages, dim=0)
    max_T = max(m.shape[1] for m in all_masks)
    masks = _pad_and_cat(all_masks, max_T)
    times = _pad_and_cat(all_times, max_T)
    return results, ages, masks, times

# ═════════════════════════════════════════════
# B) Trajectory-class PDP
# ═════════════════════════════════════════════

def _run_trajectory(model, loader, device, path_fn, interp="cubic"):
    model.eval()
    all_mus, all_masks, all_times = [], [], []

    with torch.no_grad():
        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad = t_pad.to(device)
            x_pad = x_pad.to(device)
            mask = mask.to(device)
            c_mask = c_mask.to(device)
            s = s.to(device)

            x_cf = path_fn(x_pad, t_pad, mask)

            mu, _, _, _, _, _ = model(
                t_pad, x_cf, c_mask, s,
                obs_mask=mask, y_pad=None,
                interp=interp
            )

            all_mus.append(mu.cpu())
            all_masks.append(mask.cpu())
            all_times.append(t_pad.cpu())

    max_T = max(m.shape[1] for m in all_mus)
    mus = _pad_and_cat(all_mus, max_T)
    masks = _pad_and_cat(all_masks, max_T)
    times = _pad_and_cat(all_times, max_T)
    return mus, masks, times

def _run_trajectory_causal(model, loader, device, path_fn, interp="cubic"):
    model.eval()
    all_mus, all_masks, all_times = [], [], []

    with torch.no_grad():
        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad = t_pad.to(device)
            x_pad = x_pad.to(device)
            mask = mask.to(device)
            c_mask = c_mask.to(device)
            s = s.to(device)
            N, T = t_pad.shape

            x_cf = path_fn(x_pad, t_pad, mask)

            mu_all = torch.zeros(N, T, device=device)

            for i in range(N):
                obs_idx = torch.where(mask[i] > 0.5)[0]
                if len(obs_idx) < 2:
                    continue

                # First visit: use path up to second visit
                k0 = obs_idx[0].item()
                k1 = obs_idx[1].item()
                t_tr = t_pad[i:i+1, :k1+1]
                x_tr = x_cf[i:i+1, :k1+1, :]
                m_tr = mask[i:i+1, :k1+1]
                cm_tr = c_mask[i:i+1, :k1+1] if c_mask.dim() == 2 \
                    else c_mask[i:i+1, :k1+1, :]
                mu_k, _, _, _, _, _ = model(
                    t_tr, x_tr, cm_tr, s[i:i+1],
                    obs_mask=m_tr, y_pad=None, interp=interp
                )
                mu_all[i, k0] = mu_k[0, 0]

                # Remaining visits: truncate path up to visit k
                for k_pos in range(1, len(obs_idx)):
                    k = obs_idx[k_pos].item()
                    t_tr = t_pad[i:i+1, :k+1]
                    x_tr = x_cf[i:i+1, :k+1, :]
                    m_tr = mask[i:i+1, :k+1]
                    cm_tr = c_mask[i:i+1, :k+1] if c_mask.dim() == 2 \
                        else c_mask[i:i+1, :k+1, :]
                    mu_k, _, _, _, _, _ = model(
                        t_tr, x_tr, cm_tr, s[i:i+1],
                        obs_mask=m_tr, y_pad=None, interp=interp
                    )
                    mu_all[i, k] = mu_k[0, -1]

            all_mus.append(mu_all.cpu())
            all_masks.append(mask.cpu())
            all_times.append(t_pad.cpu())

    max_T = max(m.shape[1] for m in all_mus)
    mus = _pad_and_cat(all_mus, max_T)
    masks = _pad_and_cat(all_masks, max_T)
    times = _pad_and_cat(all_times, max_T)
    return mus, masks, times


def compute_trajectory_delta_pdp(model, loader, device,
                                  profile="volatile_vs_stable",
                                  alpha=-0.15,
                                  amplitude=1.0, period=5.0,
                                  slope_fast=0.3, slope_slow=0.05,
                                  bmi_before=25.0, bmi_after=30.0, t_step=2.0,
                                  bmi_col=0, interp="linear",
                                  visit_times=None):
    """
    Compute trajectory-class ΔPDP: compare two counterfactual path types.

    Profiles:
      "volatile_vs_stable": sinusoidal vs frozen BMI
      "fast_vs_slow_ramp":  steep vs gentle linear ramp
      "step_vs_stable":     step function vs frozen BMI
      "shifted":            shifted level (negative control, oracle=0)

    Returns:
        estimated: dict {vt: mean_delta}
        oracle:    dict {vt: true_delta}
    """
    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])

    print(f"\n{'='*60}")
    print(f"Trajectory ΔPDP: {profile}")
    print(f"{'='*60}")

    # Define path pairs and oracle
    if profile == "volatile_vs_stable":
        path_fn_1 = lambda x, t, m: make_volatile_path(x, t, bmi_col, amplitude, period)
        path_fn_0 = lambda x, t, m: make_stable_path(x, bmi_col)
        oracle = oracle_volatile_vs_stable_discrete(
            loader, device, bmi_col=bmi_col,
            alpha=alpha, amplitude=amplitude, period=period,
            visit_times=visit_times
        )
        label_1, label_0 = f"Volatile (A={amplitude}, T={period})", "Stable (frozen)"

    elif profile == "fast_vs_slow_ramp":
        path_fn_1 = lambda x, t, m: make_ramp_path(x, t, bmi_col, 25.0, slope_fast)
        path_fn_0 = lambda x, t, m: make_ramp_path(x, t, bmi_col, 25.0, slope_slow)
        oracle = oracle_ramp(visit_times, alpha, slope_fast, slope_slow)
        label_1, label_0 = f"Fast ramp (slope={slope_fast})", f"Slow ramp (slope={slope_slow})"

    elif profile == "step_vs_stable":
        path_fn_1 = lambda x, t, m: make_step_path(x, t, bmi_col, bmi_before, bmi_after, t_step)
        path_fn_0 = lambda x, t, m: make_stable_path(x, bmi_col)
        oracle = oracle_step_vs_stable(visit_times, alpha, bmi_before, bmi_after, t_step)
        label_1 = f"Step ({bmi_before}->{bmi_after} at t={t_step})"
        label_0 = "Stable (frozen)"

    elif profile == "shifted":
        path_fn_1 = lambda x, t, m: make_shifted_path(x, m, bmi_col, 35.0)
        path_fn_0 = lambda x, t, m: make_shifted_path(x, m, bmi_col, 20.0)
        oracle = oracle_shifted(visit_times)
        label_1, label_0 = "Shifted to 35", "Shifted to 20"

    else:
        raise ValueError(f"Unknown profile: {profile}")

    print(f"  Path 1: {label_1}")
    print(f"  Path 0: {label_0}")

    # Run both paths
    mu_1, masks, times = _run_trajectory(model, loader, device, path_fn_1, interp)
    mu_0, _, _ = _run_trajectory(model, loader, device, path_fn_0, interp)

    # Compute delta
    delta = (mu_1 - mu_0).numpy()
    masks_np = masks.numpy()
    times_np = times.numpy()

    closest = _closest_obs_per_subject(delta, masks_np, times_np, visit_times)

    print(f"\n  {'Time':>6s}  {'Estimated':>10s}  {'Oracle':>10s}  "
          f"{'Bias':>10s}  {'n':>6s}")
    print(f"  {'-'*50}")

    estimated = {}
    for vt in visit_times:
        d = closest[vt]
        if len(d) > 10:
            est = d.mean()
            orc = oracle[vt]
            estimated[vt] = est
            print(f"  {vt:6.0f}  {est:+10.4f}  {orc:+10.4f}  "
                  f"{est - orc:+10.4f}  {len(d):6d}")

    return estimated, oracle, mu_1, mu_0, masks, times


# ═════════════════════════════════════════════
# Standard ΔPDP (for constant interventions)
# ═════════════════════════════════════════════

def compute_true_delta_pdp(ages, masks, times, bmi_lo=20, bmi_hi=35,
                           true_beta_bmi=0.0, true_beta_int=0.0,
                           visit_times=None):
    ages_np = ages.numpy() if hasattr(ages, 'numpy') else np.asarray(ages)
    delta_v = bmi_hi - bmi_lo
    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])

    mean_age = ages_np.mean()
    marginal = delta_v * (true_beta_bmi + true_beta_int * mean_age)

    q33, q67 = np.percentile(ages_np, [33, 67])
    age_groups = {
        f'Young (AGEc < {q33:.1f})': ages_np < q33,
        f'Middle ({q33:.1f} <= AGEc < {q67:.1f})':
            (ages_np >= q33) & (ages_np < q67),
        f'Old (AGEc >= {q67:.1f})': ages_np >= q67,
    }

    print(f"\n{'='*60}")
    print(f"TRUE ΔPDP (BMI {bmi_lo} -> {bmi_hi})")
    print(f"  β_BMI={true_beta_bmi}, β_int={true_beta_int}, Δv={delta_v}")
    print(f"  Marginal = {marginal:.4f}")
    print(f"{'='*60}")

    print(f"\n  {'Group':<35s} {'mean AGEc':>10s} {'ΔPDP':>10s} {'n':>6s}")
    print(f"  {'-'*65}")
    stratified = {}
    for gname, gmask in age_groups.items():
        mean_a = ages_np[gmask].mean()
        d = delta_v * (true_beta_bmi + true_beta_int * mean_a)
        stratified[gname] = {vt: d for vt in visit_times}
        print(f"  {gname:<35s} {mean_a:>+10.3f} {d:>+10.4f} {gmask.sum():>6d}")

    return {vt: marginal for vt in visit_times}, stratified, age_groups


def compute_delta_pdp(results, ages, masks, times, bmi_lo=20, bmi_hi=35,
                      true_beta_bmi=0.0, true_beta_int=0.0, visit_times=None):
    mu_lo = results[bmi_lo].numpy()
    mu_hi = results[bmi_hi].numpy()
    masks_np = masks.numpy()
    ages_np = ages.numpy() if hasattr(ages, 'numpy') else np.asarray(ages)
    times_np = times.numpy()
    delta = mu_hi - mu_lo
    delta_v = bmi_hi - bmi_lo
    mean_age = ages_np.mean()
    true_marginal = delta_v * (true_beta_bmi + true_beta_int * mean_age)

    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])

    closest = _closest_obs_per_subject(delta, masks_np, times_np, visit_times)

    print(f"\nΔPDP (BMI {bmi_lo} -> {bmi_hi}):")
    print(f"  True marginal = {true_marginal:.4f}")
    print(f"\n  {'Time':>6s}  {'Estimated':>10s}  {'True':>10s}  "
          f"{'Bias':>10s}  {'n':>6s}")
    print(f"  {'-'*50}")

    estimated, true_ref = {}, {}
    for vt in visit_times:
        d = closest[vt]
        if len(d) > 10:
            est = d.mean()
            estimated[vt] = est
            true_ref[vt] = true_marginal
            print(f"  {vt:6.0f}  {est:+10.4f}  {true_marginal:+10.4f}  "
                  f"{est - true_marginal:+10.4f}  {len(d):6d}")
    return estimated, true_ref


def compute_delta_pdp_stratified(results, ages, masks, times,
                                  bmi_lo=20, bmi_hi=35,
                                  true_beta_bmi=0.0, true_beta_int=0.0,
                                  visit_times=None):
    mu_lo = results[bmi_lo].numpy()
    mu_hi = results[bmi_hi].numpy()
    masks_np = masks.numpy()
    ages_np = ages.numpy() if hasattr(ages, 'numpy') else np.asarray(ages)
    times_np = times.numpy()
    delta = mu_hi - mu_lo
    delta_v = bmi_hi - bmi_lo

    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])

    q33, q67 = np.percentile(ages_np, [33, 67])
    age_groups = {
        f'Young (AGEc < {q33:.1f})': ages_np < q33,
        f'Middle ({q33:.1f} <= AGEc < {q67:.1f})':
            (ages_np >= q33) & (ages_np < q67),
        f'Old (AGEc >= {q67:.1f})': ages_np >= q67,
    }

    print(f"\n{'='*70}")
    print(f"Stratified ΔPDP (BMI {bmi_lo} -> {bmi_hi})")
    print(f"{'='*70}")

    summary = {}
    for gname, gmask in age_groups.items():
        mean_age = ages_np[gmask].mean()
        true_delta = delta_v * (true_beta_bmi + true_beta_int * mean_age)
        dg = delta[gmask]
        mg = masks_np[gmask]
        tg = times_np[gmask]
        closest = _closest_obs_per_subject(dg, mg, tg, visit_times)

        print(f"\n  {gname} (n={gmask.sum()}, mean AGEc={mean_age:.2f})")
        print(f"  Oracle = {true_delta:.3f}")
        print(f"  {'Time':>6s}  {'ΔPDP':>8s}  {'n':>6s}")
        print(f"  {'-'*30}")
        all_d = []
        for vt in visit_times:
            d = closest[vt]
            if len(d) > 10:
                all_d.append(d.mean())
                print(f"  {vt:6.0f}  {d.mean():+8.3f}  {len(d):6d}")
        summary[gname] = {
            "estimated": np.mean(all_d) if all_d else 0, "true": true_delta}

    print(f"\n  {'Group':<30s} {'Estimated':>10s} {'True':>10s} {'Bias':>10s}")
    print(f"  {'-'*60}")
    for g, v in summary.items():
        print(f"  {g:<30s} {v['estimated']:+10.3f} {v['true']:+10.3f} "
              f"{v['estimated']-v['true']:+10.3f}")
    return summary


# ═════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════

def plot_pdp(results, ages, masks, times, bmi_values,
             save_path="pdp_bmi.png", visit_times=None):
    age_np = ages.numpy()
    q33, q67 = np.percentile(age_np, [33, 67])
    age_groups = {
        f'Young (AGEc < {q33:.1f})': age_np < q33,
        f'Middle ({q33:.1f} <= AGEc < {q67:.1f})':
            (age_np >= q33) & (age_np < q67),
        f'Old (AGEc >= {q67:.1f})': age_np >= q67,
    }
    masks_np, times_np = masks.numpy(), times.numpy()
    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(bmi_values)))
    for ax_idx, (gname, gmask) in enumerate(age_groups.items()):
        ax = axes[ax_idx]
        for bi, bv in enumerate(bmi_values):
            mu = results[bv].numpy()[gmask]
            c = _closest_obs_per_subject(mu, masks_np[gmask], times_np[gmask], visit_times)
            mp, vtp = [], []
            for vt in visit_times:
                if len(c[vt]) > 10:
                    mp.append(c[vt].mean()); vtp.append(vt)
            if vtp:
                ax.plot(vtp, mp, 'o-', color=colors[bi], label=f'BMI={bv}',
                        linewidth=2, markersize=5)
        ax.set_title(gname, fontsize=11)
        ax.set_xlabel('Time (years)')
        if ax_idx == 0: ax.set_ylabel('Predicted ISA15')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle('PDP of BMI on ISA15 (CDE), stratified by age', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"PDP saved to {save_path}"); plt.close()


def plot_pdp_marginal(results, masks, times, bmi_values,
                      save_path="pdp_marginal.png", visit_times=None,
                      ice_n=30, seed=1):
    masks_np, times_np = masks.numpy(), times.numpy()
    if visit_times is None:
        visit_times = np.array([0, 5, 10, 15])
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(bmi_values)))
    N = masks_np.shape[0]
    rng = np.random.RandomState(seed)
    ice_ids = rng.choice(N, min(ice_n, N), replace=False)

    for bv in bmi_values:
        mu = results[bv].numpy()
        for i in ice_ids:
            it, ip = [], []
            for vt in visit_times:
                oi = np.where(masks_np[i] > 0.5)[0]
                if len(oi) == 0: continue
                cl = oi[np.argmin(np.abs(times_np[i, oi] - vt))]
                it.append(vt); ip.append(mu[i, cl])
            if len(it) > 1:
                ax.plot(it, ip, '-', color='grey', alpha=0.08, linewidth=0.5, zorder=1)

    for bi, bv in enumerate(bmi_values):
        mu = results[bv].numpy()
        c = _closest_obs_per_subject(mu, masks_np, times_np, visit_times)
        mp, vtp = [], []
        for vt in visit_times:
            if len(c[vt]) > 10:
                mp.append(c[vt].mean()); vtp.append(vt)
        ax.plot(vtp, mp, 'o-', color=colors[bi], label=f'BMI={bv}',
                linewidth=2.5, markersize=6, zorder=2)

    ax.set_xlabel('Time (years)'); ax.set_ylabel('Predicted ISA15')
    ax.set_title('Marginal PDP of BMI on ISA15 (CDE) + ICE (grey)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Marginal PDP saved to {save_path}"); plt.close()


def plot_trajectory_delta_pdp(estimated, oracle, profile="volatile_vs_stable",
                               save_path="trajectory_delta_pdp.png"):
    """Plot trajectory ΔPDP: estimated vs oracle over time."""
    vts = sorted(estimated.keys())
    est_vals = [estimated[vt] for vt in vts]
    orc_vals = [oracle[vt] for vt in vts]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: estimated vs oracle
    ax = axes[0]
    ax.plot(vts, est_vals, 'o-', color='steelblue', linewidth=2, markersize=8,
            label='CDE estimated')
    ax.plot(vts, orc_vals, 's--', color='red', linewidth=2, markersize=8,
            label='Oracle')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Trajectory ΔPDP')
    ax.set_title(f'Trajectory ΔPDP: {profile}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: bias
    ax = axes[1]
    bias = [est_vals[i] - orc_vals[i] for i in range(len(vts))]
    ax.bar(vts, bias, width=1.5, color='coral', alpha=0.8)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Bias (estimated - oracle)')
    ax.set_title(f'Bias: {profile}')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Trajectory ΔPDP plot saved to {save_path}")
    plt.close()


def plot_all_trajectory_profiles(all_results, save_path="trajectory_profiles_summary.png"):
    """
    Summary plot: estimated vs oracle for all trajectory profiles.

    all_results: dict {profile_name: (estimated, oracle)} from compute_trajectory_delta_pdp
    """
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    colors = {'volatile_vs_stable': 'steelblue', 'fast_vs_slow_ramp': 'green',
              'step_vs_stable': 'orange', 'shifted': 'purple'}

    for ax, (profile, (est, orc)) in zip(axes, all_results.items()):
        vts = sorted(est.keys())
        e = [est[vt] for vt in vts]
        o = [orc[vt] for vt in vts]
        c = colors.get(profile, 'steelblue')

        ax.plot(vts, e, 'o-', color=c, linewidth=2, markersize=8, label='CDE')
        ax.plot(vts, o, 's--', color='red', linewidth=2, markersize=8, label='Oracle')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('ΔPDP')
        ax.set_title(profile.replace('_', ' ').title())
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('S6 — CDE Trajectory ΔPDP across profiles', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"All profiles plot saved to {save_path}")
    plt.close()