"""
Prediction and evaluation for Neural ODE-LMM.

Computes:
  1. Marginal log-likelihood (train + test)
  2. Fit mode: BLUP predictions using all outcomes → fitted MSE
  3. Forecasting mode: ODE up to t*, outcomes before t* → prediction MSE
  4. Population-averaged predictions at each visit time
  5. Export all results to CSV for comparison with HLME

Usage:
    python prediction_ode.py
    python prediction_ode.py --checkpoint checkpoints/best_model_ode_skip.pt
    python prediction_ode.py --hlme_csv hlme_predictions.csv
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyreadr
import argparse
import os

from dataset import LongitudinalDataset, collate_pad
from model_ODE import NeuralODEModel, NeuralODEConfig
from utils import masked_NLL

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ====================================================================
# 1. LIKELIHOOD
# ====================================================================

@torch.no_grad()
def compute_log_likelihood(model, loader, device):
    """
    Compute total and per-subject marginal log-likelihood.
    LL = -NLL (we return the positive log-likelihood).
    """
    model.eval()
    total_nll = 0.0
    n_subjects = 0
    n_obs = 0

    for batch in loader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        y_pad = y_pad.to(device)
        mask  = mask.to(device)
        s     = s.to(device)

        mu, V, _, _, _ = model(t_pad, x_pad, masks=None,
                               static_covariates=s, bmi_t=x_pad[:,:,0:1], obs_mask=mask, y_pad=None)
        nll = masked_NLL(mu, y_pad, V, mask)

        N = t_pad.shape[0]
        total_nll += nll.item() * N
        n_subjects += N
        n_obs += mask.sum().item()

    total_ll = -total_nll
    avg_ll = total_ll / n_subjects

    return {
        "total_LL": total_ll,
        "avg_LL_per_subject": avg_ll,
        "n_subjects": n_subjects,
        "n_obs": int(n_obs),
    }


# ====================================================================
# 2. BLUP COMPUTATION
# ====================================================================

@torch.no_grad()
def compute_blup(mu, V, y_pad, mask, Z, D, sig2, jitter=1e-6):
    """
    Compute Best Linear Unbiased Predictor for random effects.

    b_hat_i = D Z_i' V_i^{-1} (y_i - mu_i)

    Args:
        mu:    (N, T) population mean predictions
        V:     (N, T, T) marginal covariance
        y_pad: (N, T) observed outcomes
        mask:  (N, T) observation mask
        Z:     (N, T, q) random effects design
        D:     (q, q) random effects covariance
        sig2:  scalar residual variance

    Returns:
        b_hat:   (N, q) estimated random effects
        y_blup:  (N, T) subject-specific predictions (mu + Z @ b_hat)
    """
    N, T = mu.shape
    q = Z.shape[2]
    device, dtype = mu.device, mu.dtype

    b_hat = torch.zeros(N, q, device=device, dtype=dtype)
    y_blup = mu.clone()

    for i in range(N):
        obs = mask[i].bool()
        n_i = obs.sum()
        if n_i < 1:
            continue

        mu_i = mu[i, obs]                          # (n_i,)
        y_i  = y_pad[i, obs]                       # (n_i,)
        V_i  = V[i][obs][:, obs]                   # (n_i, n_i)
        Z_i  = Z[i, obs]                           # (n_i, q)

        r_i = y_i - mu_i                           # (n_i,)

        L_i = torch.linalg.cholesky(
            V_i + 1e-6 * torch.eye(n_i, device=device, dtype=dtype)
        )
        Vinv_r = torch.cholesky_solve(
            r_i.unsqueeze(-1), L_i
        ).squeeze(-1)                               # (n_i,)

        b_hat[i] = D @ Z_i.t() @ Vinv_r            # (q,)
        y_blup[i] = mu[i] + (Z[i] @ b_hat[i])      # (T,)

    return b_hat, y_blup


# ====================================================================
# 3. FIT MODE — full trajectory reconstruction
# ====================================================================

@torch.no_grad()
def predict_fit_mode(model, loader, device):
    """
    Fit mode: condition on ALL observed outcomes for BLUP.
    ODE uses full time grid, decoder sees full BMI path.

    Returns DataFrame with columns:
        subject_idx, time, y_true, mu_pop, y_blup, mask
    """
    model.eval()
    rows = []

    for batch in loader:
        ids, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        y_pad = y_pad.to(device)
        mask  = mask.to(device)
        s     = s.to(device)

        # Forward pass
        mu, V, Z, D, sig2 = model(t_pad, x_pad, masks=None,
                                   static_covariates=s, bmi_t=x_pad[:,:,0:1],obs_mask=mask,
                                   y_pad=None)

        # BLUP
        b_hat, y_blup = compute_blup(mu, V, y_pad, mask, Z, D, sig2)

        N, T = t_pad.shape
        for i in range(N):
            for j in range(T):
                if mask[i, j] > 0.5:
                    rows.append({
                        "subject_idx": ids[i],
                        "time": t_pad[i, j].item(),
                        "y_true": y_pad[i, j].item(),
                        "mu_pop": mu[i, j].item(),
                        "y_blup": y_blup[i, j].item(),
                    })

    return pd.DataFrame(rows)


# ====================================================================
# 4. FORECASTING MODE — predict current visit from past outcomes
# ====================================================================

@torch.no_grad()
def predict_forecast_mode(model, loader, device):
    """
    Forecasting mode for Neural ODE:
      - For each visit k >= 1, truncate time grid at t_k
      - Run ODE integration on [0, t_k]
      - BLUP from outcomes [y_0, ..., y_{k-1}]
      - Predict y_k

    Unlike the CDE version, we don't need to worry about causal
    interpolation — the ODE only depends on the current state,
    time, and BMI(t), all of which are available.

    Returns DataFrame with columns:
        subject_idx, time, y_true, y_pred, mu_pop, visit_index
    """
    model.eval()
    rows = []
    subject_idx = 0

    for batch in loader:
        ids, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        y_pad = y_pad.to(device)
        mask  = mask.to(device)
        s     = s.to(device)

        N = t_pad.shape[0]

        for i in range(N):
            obs_idx = torch.where(mask[i] > 0.5)[0]
            n_i = len(obs_idx)
            if n_i < 2:
                subject_idx += 1
                continue

            for k_pos in range(1, n_i):
                k = obs_idx[k_pos].item()  # padded index of current visit

                # --- Truncate at visit k (inclusive) ---
                t_trunc    = t_pad[i:i+1, :k+1]           # (1, k+1)
                x_trunc    = x_pad[i:i+1, :k+1, :]        # (1, k+1, Cx)
                mask_trunc = mask[i:i+1, :k+1]             # (1, k+1)

                # --- Forward on truncated sequence ---
                mu_k, V_k, Z_k, D_k, sig2_k = model(
                    t_trunc, x_trunc, masks=None,
                    static_covariates=s[i:i+1],
                    bmi_t = x_trunc[:,:,0:1],
                    obs_mask=mask_trunc, y_pad=None
                )

                # --- Past visits: observed indices strictly before k ---
                past_idx = obs_idx[:k_pos]
                # These are valid in the truncated grid (all < k+1)
                n_past = len(past_idx)

                # --- BLUP from past outcomes ---
                mu_past = mu_k[0, past_idx]                    # (n_past,)
                y_past  = y_pad[i, past_idx]                   # (n_past,)
                Z_past  = Z_k[0, past_idx]                     # (n_past, q)
                V_past  = V_k[0][past_idx][:, past_idx]        # (n_past, n_past)

                r_past = y_past - mu_past                      # (n_past,)

                L_past = torch.linalg.cholesky(
                    V_past + 1e-6 * torch.eye(n_past, device=device, dtype=V_k.dtype)
                )
                Vinv_r = torch.cholesky_solve(
                    r_past.unsqueeze(-1), L_past
                ).squeeze(-1)                                  # (n_past,)

                b_hat = D_k @ (Z_past.t() @ Vinv_r)          # (q,)

                # --- Predict at current visit k ---
                mu_current = mu_k[0, k]                        # scalar
                Z_current  = Z_k[0, k]                         # (q,)
                y_pred_k   = mu_current + Z_current @ b_hat    # scalar

                rows.append({
                    "subject_idx": ids[i],
                    "time":        t_pad[i, k].item(),
                    "y_true":      y_pad[i, k].item(),
                    "mu_pop":      mu_current.item(),
                    "y_pred":      y_pred_k.item(),
                    "visit_index": k_pos,
                })

            subject_idx += 1

    return pd.DataFrame(rows)


# ====================================================================
# 5. MSE COMPUTATION
# ====================================================================

def compute_mse(df, y_col="y_true", pred_col="y_blup"):
    """Compute MSE from a predictions DataFrame."""
    residuals = df[y_col] - df[pred_col]
    return (residuals ** 2).mean()


# ====================================================================
# 6. POPULATION-AVERAGED PREDICTIONS AT VISIT TIMES
# ====================================================================

def population_averaged_predictions(df_fit, visit_times=None):
    """
    Compute population-averaged predictions at each visit time.
    Uses closest observation per subject (R-style).
    """
    if visit_times is None:
        visit_times = [0, 2, 4, 7, 10, 12]

    rows = []
    subjects = df_fit["subject_idx"].unique()

    for vt in visit_times:
        mu_pops = []
        y_blups = []
        y_trues = []

        for sid in subjects:
            df_s = df_fit[df_fit["subject_idx"] == sid]
            closest_idx = (df_s["time"] - vt).abs().idxmin()
            row = df_s.loc[closest_idx]
            mu_pops.append(row["mu_pop"])
            y_blups.append(row["y_blup"])
            y_trues.append(row["y_true"])

        rows.append({
            "time": vt,
            "n": len(mu_pops),
            "mean_y_true": np.mean(y_trues),
            "se_y_true": np.std(y_trues) / np.sqrt(len(y_trues)),
            "mean_mu_pop": np.mean(mu_pops),
            "mean_y_blup": np.mean(y_blups),
        })

    return pd.DataFrame(rows)


# ====================================================================
# 7. PLOTTING
# ====================================================================

def plot_individual_predictions(df_fit, n_subjects=25, ncols=5,
                                hlme_df=None, save_path="predictions_individual.png"):
    """
    Plot individual BLUP predictions for a random sample of subjects.
    Optionally overlay HLME predictions from CSV.
    """
    subjects = df_fit["subject_idx"].unique()
    n_plot = min(n_subjects, len(subjects))
    rng = np.random.RandomState(42)
    selected = rng.choice(subjects, n_plot, replace=False)

    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False, sharey=True)
    axes = axes.ravel()

    for p_idx, sid in enumerate(selected):
        ax = axes[p_idx]
        df_s = df_fit[df_fit["subject_idx"] == sid].sort_values("time")

        ax.scatter(df_s["time"], df_s["y_true"], c="blue", s=20, zorder=3,
                   label="Observed")
        ax.plot(df_s["time"], df_s["y_blup"], "g--", linewidth=1.5,
                label="ODE BLUP")
        ax.plot(df_s["time"], df_s["mu_pop"], "r:", linewidth=1, alpha=0.6,
                label="ODE pop")

        if hlme_df is not None:
            hlme_s = hlme_df[hlme_df["subject_idx"] == sid].sort_values("time")
            if len(hlme_s) > 0:
                ax.plot(hlme_s["time"], hlme_s["y_pred"], "k--", linewidth=1.5,
                        label="HLME")
                ax.plot(hlme_s["time"], hlme_s["mu_pop"], "b:", linewidth=1, alpha=0.6,
                label="HLME pop")

        ax.set_title(f"Subject {sid} (n={len(df_s)})", fontsize=9)
        if p_idx % ncols == 0:
            ax.set_ylabel("ISA15")
        if p_idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Time (years)")

    for k in range(n_plot, len(axes)):
        axes[k].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle("Individual BLUP predictions (fit mode) — Neural ODE",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Individual predictions saved to {save_path}")


def plot_population_averaged(df_pop, hlme_pop=None, mode="fit",
                             save_path="predictions_population.png"):
    """Plot population-averaged predictions over time."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(df_pop["time"], df_pop["mean_y_true"], yerr=df_pop["se_y_true"],
                fmt="o-", color="blue", linewidth=2, markersize=6,
                capsize=3, label="Observed (mean ± SE)")

    pred_col = "mean_y_blup" if mode == "fit" else "mean_y_pred"
    label = "ODE BLUP" if mode == "fit" else "ODE forecast"
    if pred_col in df_pop.columns:
        ax.plot(df_pop["time"], df_pop[pred_col], "o--", color="green",
                linewidth=2, markersize=6, label=label)

    ax.plot(df_pop["time"], df_pop["mean_mu_pop"], "o:", color="red",
            linewidth=1.5, markersize=5, alpha=0.7, label="ODE pop mean")

    if hlme_pop is not None:
        print(hlme_pop.columns)
        ax.plot(hlme_pop["time"], hlme_pop["mean_mu_pop"], "o--", color="black",
                linewidth=2, markersize=6, label="HLME")

    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("ISA15", fontsize=12)
    ax.set_title(f"Population-averaged predictions ({mode} mode) — Neural ODE",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Population-averaged plot saved to {save_path}")


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prediction and evaluation for Neural ODE-LMM"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_model_ode_full_skip.pt")
    parser.add_argument("--data", type=str,
                        default="simu_datasets/S2a_sims_2/sim_001.rds")
    parser.add_argument("--hlme_csv", type=str, default="results/HLME_S2a_sim_2_preds/predictions/fit_preds_001.csv",
                        help="CSV with HLME predictions (columns: subject_idx, time, y_pred)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    time_col = "time"
    y_col = "ISA15_sim"
    id_col = "NUM_ID"
    x_cols = ["BMI_t", "rs1", "rs2"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    df = next(iter(pyreadr.read_r(args.data).values()))
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    full_dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col,
                                       static_cols=static_cols)

    # ---- Train/test split (subject level) ----
    N = len(full_dataset)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(N)
    n_test = int(N * args.test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset  = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_pad)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_pad)
    full_loader  = DataLoader(full_dataset, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_pad)

    print(f"Subjects: {N} total -> {len(train_idx)} train, {len(test_idx)} test")

    # ---- Model ----
    n_tv = 1

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
    )

    model = NeuralODEModel(
        x_dim=len(x_cols),
        static_dim=len(static_cols),
        cfg=cfg,
        n_tv=n_tv,
        use_rho_net=True,
        use_neural_re=True,
        re_spline_cols=[1, 2],
        g_hidden=16,
        fullD=True,
        bmi_mean=0.0,    # placeholder — overwritten by checkpoint
        bmi_std=1.0,
        static_skip_dims=[1],
    ).to(device)

    # ---- Load checkpoint ----
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded: {args.checkpoint}")
        print(f"  epoch = {checkpoint.get('epoch', '?')}")
        loss_val = checkpoint.get('best_test_loss', None)
        if loss_val is not None:
            print(f"  best test loss = {loss_val:.4f}")
    else:
        model.load_state_dict(checkpoint, strict=False)

    # Verify BMI stats
    print(f"  bmi_mean = {model.decoder.bmi_mean.item():.4f}")
    print(f"  bmi_std  = {model.decoder.bmi_std.item():.4f}")

    # Print model info
    if model.decoder.L_unconstrained is not None:
        D = model.decoder._build_D(device=torch.device('cpu'), dtype=torch.float32)
        print(f"  D matrix:\n{D}")
    print(f"  sigma2 = {torch.exp(model.decoder.log_residual_var).item():.4f}")

    # ---- HLME predictions ----
    hlme_df = None
    if args.hlme_csv and os.path.exists(args.hlme_csv):
        hlme_df = pd.read_csv(args.hlme_csv)
        print(f"Loaded HLME predictions: {args.hlme_csv} ({len(hlme_df)} rows)")

    print(hlme_df.columns)
    # ================================================================
    # 1. LOG-LIKELIHOOD
    # ================================================================
    print(f"\n{'='*60}")
    print("1. LOG-LIKELIHOOD")
    print(f"{'='*60}")

    for name, loader in [("Full", full_loader)]:
        ll = compute_log_likelihood(model, loader, device)
        print(f"  {name:6s}: LL = {ll['total_LL']:.2f}, "
              f"avg LL/subject = {ll['avg_LL_per_subject']:.4f}, "
              f"n_subjects = {ll['n_subjects']}, n_obs = {ll['n_obs']}")

    # ================================================================
    # 2. FIT MODE PREDICTIONS
    # ================================================================
    print(f"\n{'='*60}")
    print("2. FIT MODE (trajectory reconstruction)")
    print(f"{'='*60}")

    for name, loader in [("Full", full_loader)]:
        df_fit = predict_fit_mode(model, loader, device)
        mse_pop = compute_mse(df_fit, pred_col="mu_pop")
        mse_blup = compute_mse(df_fit, pred_col="y_blup")
        print(f"  {name:6s}: MSE(pop) = {mse_pop:.4f}, MSE(BLUP) = {mse_blup:.4f}, "
              f"n_obs = {len(df_fit)}")

        csv_path = os.path.join(args.output_dir, f"fit_ode_{name.lower()}.csv")
        df_fit.to_csv(csv_path, index=False)
        print(f"    -> {csv_path}")

    # Full dataset fit for plots
    df_fit_full = predict_fit_mode(model, full_loader, device)

    # Population-averaged
    df_pop_fit = population_averaged_predictions(df_fit_full)
    print(f"\n  Population-averaged (fit mode):")
    print(df_pop_fit.to_string(index=False))

    # ================================================================
    # 3. FORECASTING MODE
    # ================================================================
    print(f"\n{'='*60}")
    print("3. FORECASTING MODE (ODE up to t*, outcomes before t*)")
    print(f"{'='*60}")

    for name, loader in [("Full", full_loader)]:
        df_fc = predict_forecast_mode(model, loader, device)
        mse_pop = compute_mse(df_fc, y_col="y_true", pred_col="mu_pop")
        mse_pred = compute_mse(df_fc, y_col="y_true", pred_col="y_pred")
        print(f"  {name:6s}: MSE(pop) = {mse_pop:.4f}, MSE(pred) = {mse_pred:.4f}, "
              f"n_predictions = {len(df_fc)}")

        csv_path = os.path.join(args.output_dir, f"forecast_ode_{name.lower()}.csv")
        df_fc.to_csv(csv_path, index=False)
        print(f"    -> {csv_path}")

    # ================================================================
    # 4. PLOTS
    # ================================================================
    print(f"\n{'='*60}")
    print("4. PLOTS")
    print(f"{'='*60}")

    # Individual predictions (fit mode)
    plot_individual_predictions(
        df_fit_full, n_subjects=25, ncols=5,
        hlme_df=hlme_df,
        save_path=os.path.join(args.output_dir, "fit_predictions_ode.png")
    )

    # Population-averaged (fit mode)
    hlme_pop = None
    if hlme_df is not None:
        hlme_pop = population_averaged_predictions(
            hlme_df.rename(columns={"y_pred": "y_blup"}),
        )

    plot_population_averaged(
        df_pop_fit, hlme_pop=hlme_pop, mode="fit",
        save_path=os.path.join(args.output_dir, "predictions_population_ode_fit.png")
    )

    # # ================================================================
    # # 5. SUMMARY TABLE
    # # ================================================================
    # print(f"\n{'='*60}")
    # print("5. SUMMARY")
    # print(f"{'='*60}")

    # summary_rows = []
    # for name, loader in [("Train", train_loader), ("Test", test_loader)]:
    #     ll = compute_log_likelihood(model, loader, device)
    #     df_fit = predict_fit_mode(model, loader, device)
    #     df_fc = predict_forecast_mode(model, loader, device)

    #     summary_rows.append({
    #         "Set": name,
    #         "LL": ll["total_LL"],
    #         "Fitted_MSE": compute_mse(df_fit, pred_col="y_blup"),
    #         "Pred_MSE": compute_mse(df_fc, y_col="y_true", pred_col="y_pred"),
    #         "n_subjects": ll["n_subjects"],
    #         "n_obs": ll["n_obs"],
    #     })

    # df_summary = pd.DataFrame(summary_rows)
    # print(df_summary.to_string(index=False))

    # csv_path = os.path.join(args.output_dir, "summary_metrics_ode.csv")
    # df_summary.to_csv(csv_path, index=False)
    # print(f"\n  -> {csv_path}")

    print("\nDone.")