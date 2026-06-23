"""
ODE Dynamics Diagnostics: Impulse Response & Latent Sensitivity
================================================================

Two diagnostics to understand what the ODE learned for each covariate:

1. IMPULSE RESPONSE
   Add a +1 SD pulse to covariate k at one time window, track how the
   predicted outcome (and latent state) differ from the unperturbed
   reference over the full follow-up.  Fast decay ⇒ instantaneous;
   persistent shift ⇒ cumulative / path-dependent.

2. LATENT SENSITIVITY (Memory Kernel)
   Compute ∂μ(t*) / ∂x_k(τ) for τ ≤ t* via autograd through the ODE.
   This gives the "memory kernel": how sensitive the current prediction
   is to past covariate values.  A sharply decaying kernel ⇒ short
   memory (instantaneous); a flat/growing kernel ⇒ long memory
   (cumulative).

Usage:
  python ode_diagnostics.py \
      --ckpt checkpoints/best_model_ode_real3C_practice_skipgate_H8_seed42.pt \
      --data_dir 3C_dataset \
      --n_subjects 200 \
      --output_dir diagnostics_output

Requires: model_ODE_real.py, Preprocess_3C.py, utils.py in the path.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Imports from your codebase ──
from Preprocess_3C import process_data, EXPECTED_TIMES
from model_ODE_real import NeuralODEModel, NeuralODEConfig
from train_ODE_real import RealDataset, collate_real, compute_covariate_stats


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def load_model_and_data(ckpt_path, data_dir, device="cpu"):
    """Load checkpoint, reconstruct model, and prepare data."""

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["config"]

    # ── Reconstruct config ──
    cfg = NeuralODEConfig(
        hidden_channels=cfg_dict["hidden_channels"],
        enc_mlp_hidden=cfg_dict.get("enc_mlp_hidden", 16),
        func_mlp_hidden=cfg_dict["func_mlp_hidden"],
        dec_rho_hidden=cfg_dict["dec_rho_hidden"],
        dec_p=cfg_dict["dec_p"],
        dec_q=cfg_dict["dec_q"],
        depth=cfg_dict["depth"],
        euler_steps_per_interval=cfg_dict["euler_steps"],
        ode_solver=cfg_dict["ode_solver"],
        use_rho_norm=cfg_dict.get("use_rho_norm", True),
    )

    K = cfg_dict["n_tv"]
    Ks = cfg_dict["static_dim"]
    time_varying_features = cfg_dict["time_varying_features"]
    static_features = cfg_dict["static_features"]
    interp_method = cfg_dict.get("interp_method", "linear")
    mask_type = cfg_dict.get("mask_type", "binary")
    reg_mode = cfg_dict.get("reg_mode", "skip_gate")
    static_skip_dims = cfg_dict.get("static_skip_dims", list(range(Ks)))
    use_dynamic_skip = cfg_dict.get("use_dynamic_skip", True)
    cov_means = cfg_dict.get("cov_means", ckpt.get("cov_means"))
    cov_stds = cfg_dict.get("cov_stds", ckpt.get("cov_stds"))

    # ── Load data ──
    id_col = "NUM_ID"
    target_col = "ISA15"
    df = pd.read_csv(os.path.join(data_dir, "train_3C_data_1.csv"))

    if "AGEc" not in df.columns:
        all_df = pd.read_csv(os.path.join(data_dir, "data_3C.csv"))
        baseline_age_mean = all_df.groupby(id_col)["AGE0"].first().mean()
        df["AGEc"] = (
            df.groupby(id_col)["AGE0"].transform("first") - baseline_age_mean
        )

    patient_data = process_data(
        df=df, id_col=id_col,
        time_varying_features=time_varying_features,
        static_features=static_features,
        target_col=target_col,
        interp_method=interp_method,
        mask_type=mask_type,
    )
    dataset = RealDataset(patient_data)

    # ── Build model ──
    model = NeuralODEModel(
        n_tv=K, static_dim=Ks, cfg=cfg,
        use_rho_net=True,
        use_neural_re=True,
        g_hidden=8,
        fullD=False,
        cov_means=cov_means,
        cov_stds=cov_stds,
        static_skip_dims=static_skip_dims,
        use_dynamic_skip=use_dynamic_skip,
        reg_mode=reg_mode,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    info = {
        "K": K, "Ks": Ks,
        "time_varying_features": time_varying_features,
        "static_features": static_features,
        "cov_means": np.array(cov_means),
        "cov_stds": np.array(cov_stds),
    }
    return model, dataset, info


def get_batch(dataset, indices, device="cpu"):
    """Extract a batch from dataset at given indices."""
    batch = collate_real([dataset[i] for i in indices])
    pids, x_aug, y_pad, target_mask, static = batch
    return (
        x_aug.to(device),
        y_pad.to(device),
        target_mask.to(device),
        static.to(device),
    )


def get_population_mu(model, x_aug, static, target_mask):
    """Run forward pass, return population mean mu (N, T) and latent zt (N, T, H)."""
    with torch.no_grad():
        mu, V, Z, D, sig2, reg_dict, zt = model(
            x_aug, static_covariates=static,
            obs_mask=target_mask, return_hidden=True,
        )
    return mu, zt


# ═══════════════════════════════════════════════════════════════════════
#  1. IMPULSE RESPONSE
# ═══════════════════════════════════════════════════════════════════════

def impulse_response(model, dataset, info, indices,
                     perturb_covariates=None,
                     pulse_time_idx=1,
                     pulse_duration=1,
                     pulse_magnitude_sd=1.0,
                     device="cpu"):
    """
    Compute the impulse response for each covariate.

    For each covariate k:
      - Reference: original x_aug
      - Perturbed: add pulse_magnitude_sd * std_k to covariate k at
        time indices [pulse_time_idx, ..., pulse_time_idx + pulse_duration - 1],
        then revert to original values at subsequent times.

    Args:
        model:  trained NeuralODEModel
        dataset: RealDataset
        info:   dict with K, cov_stds, time_varying_features
        indices: subject indices to use
        perturb_covariates: list of covariate indices to perturb
                            (default: all K)
        pulse_time_idx: grid index where pulse starts (0-indexed)
        pulse_duration: number of grid points the pulse lasts
        pulse_magnitude_sd: pulse size in SD units
        device: torch device

    Returns:
        results: dict keyed by covariate name, each containing:
            delta_mu:  (T,) mean Δμ across subjects
            delta_zt:  (T, H) mean Δz across subjects
            se_mu:     (T,) standard error of Δμ
    """
    K = info["K"]
    cov_stds = info["cov_stds"]
    feat_names = info["time_varying_features"]

    if perturb_covariates is None:
        perturb_covariates = list(range(K))

    x_aug, y_pad, target_mask, static = get_batch(dataset, indices, device)
    N, T, _ = x_aug.shape

    # Reference predictions
    mu_ref, zt_ref = get_population_mu(model, x_aug, static, target_mask)

    results = {}
    for k_idx in perturb_covariates:
        feat_name = feat_names[k_idx]
        sd_k = cov_stds[k_idx]

        # Create perturbed x_aug: add pulse to covariate k
        x_aug_pert = x_aug.clone()

        # Perturb x_interp (columns 1..K in x_aug)
        t_start = pulse_time_idx
        t_end = min(pulse_time_idx + pulse_duration, T)
        x_aug_pert[:, t_start:t_end, 1 + k_idx] += pulse_magnitude_sd * sd_k

        # Predictions under perturbation
        mu_pert, zt_pert = get_population_mu(
            model, x_aug_pert, static, target_mask
        )

        # Differences
        delta_mu = (mu_pert - mu_ref).cpu().numpy()          # (N, T)
        delta_zt = (zt_pert - zt_ref).cpu().numpy()          # (N, T, H)

        results[feat_name] = {
            "delta_mu_mean": delta_mu.mean(axis=0),           # (T,)
            "delta_mu_se": delta_mu.std(axis=0) / np.sqrt(N), # (T,)
            "delta_zt_mean": np.linalg.norm(
                delta_zt.mean(axis=0), axis=-1),               # (T,) L2 norm
            "delta_mu_all": delta_mu,                          # (N, T) full
            "pulse_window": (t_start, t_end),
        }

    return results


def plot_impulse_response(results, info, output_dir, pulse_time_idx=1,
                          pulse_duration=1):
    """
    Plot impulse response: Δμ(t) and ||Δz(t)|| for each covariate.
    Two panels: (a) Δμ(t), (b) ||Δz(t)||.
    """
    feat_names = info["time_varying_features"]
    times = EXPECTED_TIMES

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel (a): Δμ(t) ──
    ax = axes[0]
    for feat_name in feat_names:
        if feat_name not in results:
            continue
        r = results[feat_name]
        ax.plot(times, r["delta_mu_mean"], "o-", label=feat_name, linewidth=2)
        ax.fill_between(
            times,
            r["delta_mu_mean"] - 1.96 * r["delta_mu_se"],
            r["delta_mu_mean"] + 1.96 * r["delta_mu_se"],
            alpha=0.15,
        )

    # Shade pulse window
    t_start = pulse_time_idx
    t_end = min(pulse_time_idx + pulse_duration, len(times))
    ax.axvspan(times[t_start], times[t_end - 1], alpha=0.08, color="grey",
               label="pulse window")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Δ E[IST]  (perturbed − reference)")
    ax.set_title("(a) Impulse response: predicted outcome")
    ax.legend(fontsize=9)

    # ── Panel (b): ||Δz(t)|| ──
    ax = axes[1]
    for feat_name in feat_names:
        if feat_name not in results:
            continue
        r = results[feat_name]
        ax.plot(times, r["delta_zt_mean"], "o-", label=feat_name, linewidth=2)

    ax.axvspan(times[t_start], times[t_end - 1], alpha=0.08, color="grey",
               label="pulse window")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("||Δz(t)||₂  (mean over subjects)")
    ax.set_title("(b) Impulse response: latent state displacement")
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Impulse Response (+1 SD pulse at t={times[pulse_time_idx]:.0f} yr)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir,
                        f"impulse_response_t{pulse_time_idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_impulse_decay(results, info, output_dir, pulse_time_idx=1):
    """
    Normalised decay plot: |Δμ(t)| / |Δμ(t_pulse_end)|, showing how
    quickly the effect decays after the pulse ends.
    """
    feat_names = info["time_varying_features"]
    times = EXPECTED_TIMES

    fig, ax = plt.subplots(figsize=(8, 5))

    for feat_name in feat_names:
        if feat_name not in results:
            continue
        r = results[feat_name]
        _, t_end = r["pulse_window"]
        t_end = min(t_end, len(times) - 1)

        # Normalise by the value at pulse end
        peak = np.abs(r["delta_mu_mean"][t_end])
        if peak < 1e-8:
            continue

        decay = np.abs(r["delta_mu_mean"]) / peak
        # Only plot from pulse end onwards
        ax.plot(times[t_end:], decay[t_end:], "o-",
                label=feat_name, linewidth=2)

    ax.axhline(1.0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(0.0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("|Δμ(t)| / |Δμ(t_end_pulse)|")
    ax.set_title("Normalised impulse decay after pulse ends")
    ax.legend(fontsize=10)
    ax.set_ylim(-0.1, 2.5)

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f"impulse_decay_t{pulse_time_idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  2. LATENT SENSITIVITY / MEMORY KERNEL
# ═══════════════════════════════════════════════════════════════════════

def compute_sensitivity_kernel(model, dataset, info, indices,
                               target_covariates=None,
                               device="cpu"):
    """
    Compute the memory kernel: ∂μ(t*) / ∂x_k(τ) for all (τ, t*) pairs.

    For each covariate k and each pair (τ, t*) with τ ≤ t*, compute the
    sensitivity of the population mean at time t* to the covariate value
    at time τ, averaged over subjects.

    This uses autograd through the ODE solver.  We process subjects
    individually (batch_size=1) to avoid graph memory issues.

    Args:
        model:  trained NeuralODEModel
        dataset: RealDataset
        info:   dict with K, time_varying_features
        indices: subject indices to use
        target_covariates: list of covariate indices (default: all)
        device: torch device

    Returns:
        kernels: dict keyed by covariate name, each containing:
            kernel:  (T, T) array where kernel[τ_idx, t_idx] =
                     mean ∂μ(t_idx) / ∂x_k(τ_idx)
            kernel_abs: same but with absolute values before averaging
    """
    K = info["K"]
    feat_names = info["time_varying_features"]
    if target_covariates is None:
        target_covariates = list(range(K))

    T = len(EXPECTED_TIMES)

    # Accumulate kernels across subjects
    kernel_accum = {k: np.zeros((T, T)) for k in target_covariates}
    kernel_abs_accum = {k: np.zeros((T, T)) for k in target_covariates}
    n_valid = 0

    for si, subj_idx in enumerate(indices):
        if (si + 1) % 10 == 0:
            print(f"    subject {si+1}/{len(indices)}")

        x_aug_i, y_i, mask_i, static_i = get_batch(
            dataset, [subj_idx], device
        )
        # x_aug_i: (1, T, 1+2K)

        for k_idx in target_covariates:
            # We need gradients w.r.t. x_interp values at time τ.
            # Strategy: make x_interp_k a leaf, reconstruct x_aug via
            # autograd-safe column replacement (no in-place ops).
            x_aug_c = x_aug_i.detach().clone()

            # Leaf tensor for covariate k: (1, T)
            x_interp_k = x_aug_c[:, :, 1 + k_idx].clone().requires_grad_(True)

            # Autograd-safe reconstruction: build x_aug column by column
            cols = []
            for c in range(x_aug_c.shape[2]):
                if c == 1 + k_idx:
                    cols.append(x_interp_k.unsqueeze(-1))  # (1, T, 1)
                else:
                    cols.append(x_aug_c[:, :, c:c+1])       # (1, T, 1)
            x_aug_grad = torch.cat(cols, dim=-1)  # (1, T, 1+2K)

            # Forward pass WITH gradients
            mu, V, Z, D, sig2, reg_dict, zt = model(
                x_aug_grad,
                static_covariates=static_i,
                obs_mask=mask_i,
                return_hidden=True,
            )
            # mu: (1, T)

            # Compute ∂μ(t*) / ∂x_k(τ) for each t*
            kernel_k = np.zeros((T, T))
            for t_star in range(T):
                # Backward pass for μ at t_star
                model.zero_grad()
                if x_interp_k.grad is not None:
                    x_interp_k.grad.zero_()

                mu_t = mu[0, t_star]
                mu_t.backward(retain_graph=(t_star < T - 1))

                if x_interp_k.grad is not None:
                    grad = x_interp_k.grad[0].detach().cpu().numpy()  # (T,)
                    kernel_k[:, t_star] = grad

            kernel_accum[k_idx] += kernel_k
            kernel_abs_accum[k_idx] += np.abs(kernel_k)

        n_valid += 1

    # Average
    kernels = {}
    for k_idx in target_covariates:
        feat_name = feat_names[k_idx]
        kernels[feat_name] = {
            "kernel": kernel_accum[k_idx] / n_valid,
            "kernel_abs": kernel_abs_accum[k_idx] / n_valid,
        }

    return kernels


def plot_sensitivity_kernels(kernels, info, output_dir):
    """
    Plot the memory kernel heatmap for each covariate.

    kernel[τ, t*]: sensitivity of prediction at t* to covariate at τ.
    Only the lower triangle (τ ≤ t*) is causal.
    """
    times = EXPECTED_TIMES
    T = len(times)
    feat_names = list(kernels.keys())
    n_cov = len(feat_names)

    fig, axes = plt.subplots(1, n_cov, figsize=(5 * n_cov, 4.5))
    if n_cov == 1:
        axes = [axes]

    for ax, feat_name in zip(axes, feat_names):
        K = kernels[feat_name]["kernel_abs"]

        # Mask upper triangle (τ > t* is non-causal)
        mask = np.triu(np.ones_like(K, dtype=bool), k=1)
        K_masked = np.ma.array(K, mask=mask)

        im = ax.imshow(K_masked, origin="lower", aspect="auto",
                       cmap="YlOrRd",
                       extent=[times[0], times[-1], times[0], times[-1]])
        ax.set_xlabel("t* (prediction time)")
        ax.set_ylabel("τ (covariate time)")
        ax.set_title(f"|∂μ(t*)/∂{feat_name}(τ)|")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add diagonal reference
        ax.plot([times[0], times[-1]], [times[0], times[-1]],
                "w--", linewidth=1, alpha=0.5)

    fig.suptitle("Memory Kernel: sensitivity of prediction to past covariates",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "sensitivity_kernel_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_sensitivity_decay(kernels, info, output_dir):
    """
    For each covariate, plot the sensitivity at the final time t*=T
    as a function of the source time τ.  This is the last column of
    the kernel matrix — the "memory profile" of each covariate.
    """
    times = EXPECTED_TIMES
    T = len(times)
    feat_names = list(kernels.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Signed kernel at final time
    ax = axes[0]
    for feat_name in feat_names:
        K = kernels[feat_name]["kernel"]
        profile = K[:, -1]  # ∂μ(t_final)/∂x_k(τ) for each τ
        ax.plot(times, profile, "o-", label=feat_name, linewidth=2)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("τ (covariate time, years)")
    ax.set_ylabel("∂μ(t=12) / ∂x_k(τ)")
    ax.set_title("(a) Sensitivity of final prediction to each past time")
    ax.legend(fontsize=9)

    # (b) Normalised absolute decay (each covariate normalised by its peak)
    ax = axes[1]
    for feat_name in feat_names:
        K_abs = kernels[feat_name]["kernel_abs"]
        profile = K_abs[:, -1]
        peak = profile.max()
        if peak < 1e-10:
            continue
        ax.plot(times, profile / peak, "o-", label=feat_name, linewidth=2)
    ax.set_xlabel("τ (covariate time, years)")
    ax.set_ylabel("|∂μ(t=12)/∂x_k(τ)| / max")
    ax.set_title("(b) Normalised sensitivity profile (final prediction)")
    ax.legend(fontsize=9)

    fig.suptitle("Memory Profile at t*=12 years", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "sensitivity_decay_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_multi_pulse(multi_results, info, output_dir):
    """
    For each covariate, plot Δμ at the final time as a function of
    the pulse injection time.  This is the "influence function":
    how much does a +1 SD perturbation at time τ affect the final
    prediction?

    A flat curve ⇒ all past times equally important (cumulative).
    A rising curve (recent τ dominates) ⇒ instantaneous/tracking.
    """
    times = EXPECTED_TIMES
    T = len(times)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Raw Δμ(t=12) vs pulse time
    ax = axes[0]
    for feat_name, pulse_dict in multi_results.items():
        pulse_times = sorted(pulse_dict.keys())
        delta_final = [pulse_dict[t]["delta_mu_mean"][-1] for t in pulse_times]
        ax.plot([times[t] for t in pulse_times], delta_final,
                "o-", label=feat_name, linewidth=2)

    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("τ (pulse injection time, years)")
    ax.set_ylabel("Δμ(t=12)  from +1 SD pulse at τ")
    ax.set_title("(a) Influence of transient perturbation on final outcome")
    ax.legend(fontsize=10)

    # (b) Normalised (absolute)
    ax = axes[1]
    for feat_name, pulse_dict in multi_results.items():
        pulse_times = sorted(pulse_dict.keys())
        delta_final = np.array(
            [pulse_dict[t]["delta_mu_mean"][-1] for t in pulse_times]
        )
        abs_delta = np.abs(delta_final)
        peak = abs_delta.max()
        if peak < 1e-8:
            continue
        ax.plot([times[t] for t in pulse_times], abs_delta / peak,
                "o-", label=feat_name, linewidth=2)

    ax.set_xlabel("τ (pulse injection time, years)")
    ax.set_ylabel("|Δμ(t=12)| / max  (normalised)")
    ax.set_title("(b) Normalised influence on final prediction")
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.15)

    fig.suptitle("Multi-pulse influence analysis",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "multi_pulse_influence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  3. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════

def compute_summary_table(impulse_results, kernels, info, pulse_time_idx=1):
    """
    Produce a summary table comparing covariates on key metrics.
    """
    times = EXPECTED_TIMES
    rows = []

    for feat_name in info["time_varying_features"]:
        row = {"Covariate": feat_name}

        # ── Impulse response metrics ──
        if feat_name in impulse_results:
            r = impulse_results[feat_name]
            _, t_end = r["pulse_window"]
            t_end = min(t_end, len(times) - 1)

            peak = np.abs(r["delta_mu_mean"][t_end])
            final = np.abs(r["delta_mu_mean"][-1])
            row["Peak |Δμ|"] = f"{peak:.4f}"
            row["Final |Δμ|"] = f"{final:.4f}"
            row["Retention (final/peak)"] = (
                f"{final/peak:.2f}" if peak > 1e-8 else "—"
            )

        # ── Sensitivity metrics ──
        if feat_name in kernels:
            K_abs = kernels[feat_name]["kernel_abs"]
            # Ratio: sensitivity to earliest time vs latest time
            # at the final prediction
            profile = K_abs[:, -1]
            if profile[-1] > 1e-10:
                ratio = profile[0] / profile[-1]
                row["Sens. ratio (τ=0)/(τ=12)"] = f"{ratio:.2f}"
            else:
                row["Sens. ratio (τ=0)/(τ=12)"] = "—"

            # Average sensitivity across all past times
            row["Mean |∂μ/∂x|"] = f"{profile.mean():.4f}"

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ODE dynamics diagnostics: impulse response & sensitivity"
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, default="3C_dataset",
                        help="Directory containing 3C CSV files")
    parser.add_argument("--n_subjects", type=int, default=200,
                        help="Number of subjects for diagnostics")
    parser.add_argument("--n_subjects_sensitivity", type=int, default=50,
                        help="Number of subjects for sensitivity "
                             "(slower, uses autograd)")
    parser.add_argument("--output_dir", type=str,
                        default="diagnostics_output",
                        help="Output directory for plots")
    parser.add_argument("--pulse_time_idx", type=int, default=1,
                        help="Grid index for pulse start (0=t0, 1=t2, ...)")
    parser.add_argument("--pulse_duration", type=int, default=1,
                        help="Number of grid points the pulse lasts")
    parser.add_argument("--pulse_sd", type=float, default=1.0,
                        help="Pulse magnitude in SD units")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_sensitivity", action="store_true",
                        help="Skip the sensitivity kernel (option 2)")
    args = parser.parse_args()

    device = "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("ODE DYNAMICS DIAGNOSTICS")
    print("=" * 60)

    # ── Load ──
    print("\nLoading model and data...")
    model, dataset, info = load_model_and_data(
        args.ckpt, args.data_dir, device
    )
    N = len(dataset)
    print(f"  Dataset: {N} subjects")
    print(f"  Covariates: {info['time_varying_features']}")
    print(f"  Grid: {EXPECTED_TIMES}")

    # Print gate values if available
    if hasattr(model.decoder, 'skip_gate_logit') and \
       model.decoder.skip_gate_logit is not None:
        gates = torch.sigmoid(model.decoder.skip_gate_logit).detach().cpu()
        names = info["time_varying_features"] + info["static_features"]
        print(f"\n  Gate values:")
        for g, name in enumerate(names):
            print(f"    {name:>8s}: {gates[g]:.4f}")

    # ── Select subjects ──
    indices_impulse = np.random.choice(
        N, size=min(args.n_subjects, N), replace=False
    )
    indices_sens = np.random.choice(
        N, size=min(args.n_subjects_sensitivity, N), replace=False
    )

    # ══════════════════════════════════════════════════════════════════
    #  1. Impulse Response
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"1. IMPULSE RESPONSE")
    print(f"   Pulse: +{args.pulse_sd} SD at grid index {args.pulse_time_idx} "
          f"(t = {EXPECTED_TIMES[args.pulse_time_idx]:.0f} yr), "
          f"duration = {args.pulse_duration} grid point(s)")
    print(f"   Subjects: {len(indices_impulse)}")
    print(f"{'─'*60}")

    impulse_results = impulse_response(
        model, dataset, info, indices_impulse,
        perturb_covariates=None,  # all covariates
        pulse_time_idx=args.pulse_time_idx,
        pulse_duration=args.pulse_duration,
        pulse_magnitude_sd=args.pulse_sd,
        device=device,
    )

    # Print summary
    print(f"\n  {'Covariate':>8s}  {'Peak |Δμ|':>10s}  {'Final |Δμ|':>10s}  "
          f"{'Retention':>10s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")
    for feat_name in info["time_varying_features"]:
        if feat_name not in impulse_results:
            continue
        r = impulse_results[feat_name]
        _, t_end = r["pulse_window"]
        t_end = min(t_end, len(EXPECTED_TIMES) - 1)
        peak = np.abs(r["delta_mu_mean"][t_end])
        final = np.abs(r["delta_mu_mean"][-1])
        ret = final / peak if peak > 1e-8 else 0
        print(f"  {feat_name:>8s}  {peak:10.4f}  {final:10.4f}  {ret:10.2f}")

    plot_impulse_response(
        impulse_results, info, args.output_dir,
        pulse_time_idx=args.pulse_time_idx,
        pulse_duration=args.pulse_duration,
    )
    plot_impulse_decay(
        impulse_results, info, args.output_dir,
        pulse_time_idx=args.pulse_time_idx,
    )

    # ── Multi-pulse: pulse at each grid point separately ──
    print(f"\n  Multi-pulse analysis (pulse at each grid time):")
    focus_covs = [i for i, n in enumerate(info["time_varying_features"])
                  if n in ("BMI", "HDL", "GLUC")]
    multi_results = {}
    for t_idx in range(len(EXPECTED_TIMES) - 1):
        mr = impulse_response(
            model, dataset, info, indices_impulse,
            perturb_covariates=focus_covs,
            pulse_time_idx=t_idx,
            pulse_duration=1,
            pulse_magnitude_sd=args.pulse_sd,
            device=device,
        )
        for feat_name, data in mr.items():
            if feat_name not in multi_results:
                multi_results[feat_name] = {}
            multi_results[feat_name][t_idx] = data

    # Plot multi-pulse: for each covariate, Δμ(t=12) as function of pulse time
    plot_multi_pulse(multi_results, info, args.output_dir)

    # ══════════════════════════════════════════════════════════════════
    #  2. Sensitivity Kernel (Memory Kernel)
    # ══════════════════════════════════════════════════════════════════
    kernels = {}
    if not args.skip_sensitivity:
        print(f"\n{'─'*60}")
        print(f"2. SENSITIVITY KERNEL (Memory Kernel)")
        print(f"   Subjects: {len(indices_sens)} "
              f"(autograd through ODE, one at a time)")
        print(f"{'─'*60}")

        # Only compute for BMI and HDL to keep it focused
        target_cov = [
            i for i, name in enumerate(info["time_varying_features"])
            if name in ("BMI", "HDL", "GLUC")
        ]
        print(f"  Target covariates: "
              f"{[info['time_varying_features'][i] for i in target_cov]}")

        kernels = compute_sensitivity_kernel(
            model, dataset, info, indices_sens,
            target_covariates=target_cov,
            device=device,
        )

        # Print summary
        print(f"\n  Sensitivity at t*=12 yr:")
        print(f"  {'Covariate':>8s}  "
              + "  ".join(f"τ={t:.0f}" for t in EXPECTED_TIMES))
        print(f"  {'─'*8}  " + "  ".join("─" * 6 for _ in EXPECTED_TIMES))
        for feat_name, data in kernels.items():
            profile = data["kernel_abs"][:, -1]
            vals = "  ".join(f"{v:6.4f}" for v in profile)
            print(f"  {feat_name:>8s}  {vals}")

        plot_sensitivity_kernels(kernels, info, args.output_dir)
        plot_sensitivity_decay(kernels, info, args.output_dir)

    # ══════════════════════════════════════════════════════════════════
    #  3. Summary Table
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"3. SUMMARY TABLE")
    print(f"{'─'*60}")

    summary = compute_summary_table(
        impulse_results, kernels, info,
        pulse_time_idx=args.pulse_time_idx,
    )
    print(summary.to_string(index=False))

    # Save
    summary.to_csv(
        os.path.join(args.output_dir, "diagnostics_summary.csv"),
        index=False,
    )
    print(f"\n  Saved: {os.path.join(args.output_dir, 'diagnostics_summary.csv')}")

    print(f"\n{'='*60}")
    print(f"INTERPRETATION GUIDE")
    print(f"{'='*60}")
    print("""
  IMPULSE RESPONSE:
    Retention ≈ 1.0 → ODE preserves the perturbation (cumulative/path-dependent)
    Retention ≈ 0.0 → ODE forgets the perturbation (instantaneous/tracking)

  SENSITIVITY KERNEL:
    Flat profile (all τ contribute equally) → cumulative effect
    Peaked at τ ≈ t* (recent times dominate) → instantaneous effect

  EXPECTED PATTERNS (if trajectory-profile PDP is correct):
    BMI:  high retention, flat sensitivity → path-dependent
    HDL:  low retention, peaked sensitivity → instantaneous (or data-sparse)
    GLUC: intermediate
    """)


if __name__ == "__main__":
    main()