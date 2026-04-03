"""
Prediction and evaluation for Neural CDE-LMM.

Computes:
  1. Marginal log-likelihood (train + test)
  2. Fit mode: BLUP predictions using all outcomes -> fitted MSE
  3. Forecasting mode: CDE up to t*, outcomes before t* -> prediction MSE
  4. Population-averaged predictions at each visit time
  5. Export all results to CSV for comparison with HLME / ODE

Key difference from ODE:
  - CDE forward requires cumulative masks (c_mask)
  - With interp="linear", CDE is automatically causal: z(t_k) depends only on X_0..X_k
  - Forecasting truncates path at each visit for correctness
  - Returns 6 values: (mu, V, Z, D, sig2, reg_dict)

Usage:
    python prediction_CDE.py
    python prediction_CDE.py --checkpoint checkpoints/best_CDE_S6.pt --scenario S6
    # S6 strict (matches your current training config)
    python prediction_CDE.py --scenario S6 --augment_order 2 --encoder_sees_covariates

    # S6 without augmentation
    python prediction_CDE.py --scenario S6 --augment_order 1

    # S2 with injection
    python prediction_CDE.py --scenario S2 --inject_x --augment_order 1 --encoder_sees_covariates
"""

import torch
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
from model_CDE import NeuralCDEModel, NeuralCDEConfig
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
        c_mask = c_mask.to(device)
        s     = s.to(device)

        mu, V, _, _, _, _ = model(
            t_pad, x_pad, c_mask, s,
            obs_mask=mask, y_pad=None, interp="linear"
        )
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

        mu_i = mu[i, obs]
        y_i  = y_pad[i, obs]
        V_i  = V[i][obs][:, obs]
        Z_i  = Z[i, obs]

        r_i = y_i - mu_i

        L_i = torch.linalg.cholesky(
            V_i + 1e-6 * torch.eye(n_i, device=device, dtype=dtype)
        )
        Vinv_r = torch.cholesky_solve(
            r_i.unsqueeze(-1), L_i
        ).squeeze(-1)

        b_hat[i] = D @ Z_i.t() @ Vinv_r
        y_blup[i] = mu[i] + (Z[i] @ b_hat[i])

    return b_hat, y_blup


# ====================================================================
# 3. FIT MODE — full trajectory reconstruction
# ====================================================================

@torch.no_grad()
def predict_fit_mode(model, loader, device):
    """
    Fit mode: condition on ALL observed outcomes for BLUP.
    CDE runs on full path with linear interpolation (causal).

    Returns DataFrame with columns:
        subject_idx, time, y_true, mu_pop, y_blup
    """
    model.eval()
    rows = []

    for batch in loader:
        ids, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        y_pad = y_pad.to(device)
        mask  = mask.to(device)
        c_mask = c_mask.to(device)
        s     = s.to(device)

        mu, V, Z, D, sig2, _ = model(
            t_pad, x_pad, c_mask, s,
            obs_mask=mask, y_pad=None, interp="linear"
        )

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
    Forecasting mode for Neural CDE:
      For each visit k >= 1:
        - Truncate path at t_k
        - Run CDE on [0, t_k] with linear interpolation (causal)
        - BLUP from outcomes [y_0, ..., y_{k-1}]
        - Predict y_k

    Returns DataFrame with columns:
        subject_idx, time, y_true, y_pred, mu_pop, visit_index
    """
    model.eval()
    rows = []

    for batch in loader:
        ids, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        y_pad = y_pad.to(device)
        mask  = mask.to(device)
        c_mask = c_mask.to(device)
        s     = s.to(device)

        N = t_pad.shape[0]

        for i in range(N):
            obs_idx = torch.where(mask[i] > 0.5)[0]
            n_i = len(obs_idx)
            if n_i < 2:
                continue

            for k_pos in range(1, n_i):
                k = obs_idx[k_pos].item()

                # Truncate path at visit k (inclusive)
                t_trunc    = t_pad[i:i+1, :k+1]
                x_trunc    = x_pad[i:i+1, :k+1, :]
                mask_trunc = mask[i:i+1, :k+1]
                cmask_trunc = c_mask[i:i+1, :k+1] if c_mask.dim() == 2 else c_mask[i:i+1, :k+1, :]

                # Forward on truncated path (linear = causal)
                mu_k, V_k, Z_k, D_k, sig2_k, _ = model(
                    t_trunc, x_trunc, cmask_trunc,
                    static_covariates=s[i:i+1],
                    obs_mask=mask_trunc, y_pad=None,
                    interp="linear"
                )

                # Past visits: observed indices strictly before k
                past_idx = obs_idx[:k_pos]
                n_past = len(past_idx)

                # BLUP from past outcomes
                mu_past = mu_k[0, past_idx]
                y_past  = y_pad[i, past_idx]
                Z_past  = Z_k[0, past_idx]
                V_past  = V_k[0][past_idx][:, past_idx]

                r_past = y_past - mu_past

                L_past = torch.linalg.cholesky(
                    V_past + 1e-6 * torch.eye(n_past, device=device, dtype=V_k.dtype)
                )
                Vinv_r = torch.cholesky_solve(
                    r_past.unsqueeze(-1), L_past
                ).squeeze(-1)

                b_hat = D_k @ (Z_past.t() @ Vinv_r)

                # Predict at current visit k
                mu_current = mu_k[0, k]
                Z_current  = Z_k[0, k]
                y_pred_k   = mu_current + Z_current @ b_hat

                rows.append({
                    "subject_idx": ids[i],
                    "time":        t_pad[i, k].item(),
                    "y_true":      y_pad[i, k].item(),
                    "mu_pop":      mu_current.item(),
                    "y_pred":      y_pred_k.item(),
                    "visit_index": k_pos,
                })

    return pd.DataFrame(rows)


# ====================================================================
# 5. MSE COMPUTATION
# ====================================================================

def compute_mse(df, y_col="y_true", pred_col="y_blup"):
    residuals = df[y_col] - df[pred_col]
    return (residuals ** 2).mean()


# ====================================================================
# 6. POPULATION-AVERAGED PREDICTIONS
# ====================================================================

def population_averaged_predictions(df_fit, visit_times=None):
    if visit_times is None:
        visit_times = [0, 2, 4, 7, 10, 12]

    rows = []
    subjects = df_fit["subject_idx"].unique()

    for vt in visit_times:
        mu_pops, y_blups, y_trues = [], [], []
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
                                hlme_df=None, ode_df=None,
                                save_path="predictions_individual_cde.png"):
    """
    Plot individual BLUP predictions for a random sample of subjects.
    Optionally overlay HLME and/or ODE predictions.
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
                label="CDE BLUP")
        ax.plot(df_s["time"], df_s["mu_pop"], "r:", linewidth=1, alpha=0.6,
                label="CDE pop")

        if hlme_df is not None:
            hlme_s = hlme_df[hlme_df["subject_idx"] == sid].sort_values("time")
            if len(hlme_s) > 0:
                ax.plot(hlme_s["time"], hlme_s["y_pred"], "k--", linewidth=1.5,
                        label="HLME")

        if ode_df is not None:
            ode_s = ode_df[ode_df["subject_idx"] == sid].sort_values("time")
            if len(ode_s) > 0:
                ax.plot(ode_s["time"], ode_s["y_blup"], "m--", linewidth=1.5,
                        label="ODE BLUP")

        ax.set_title(f"Subject {sid} (n={len(df_s)})", fontsize=9)
        if p_idx % ncols == 0:
            ax.set_ylabel("ISA15")
        if p_idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Time (years)")

    for k in range(n_plot, len(axes)):
        axes[k].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=9)
    fig.suptitle("Individual BLUP predictions (fit mode) — Neural CDE",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Individual predictions saved to {save_path}")


def plot_population_averaged(df_pop, hlme_pop=None, ode_pop=None, mode="fit",
                             save_path="predictions_population_cde.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(df_pop["time"], df_pop["mean_y_true"], yerr=df_pop["se_y_true"],
                fmt="o-", color="blue", linewidth=2, markersize=6,
                capsize=3, label="Observed (mean +/- SE)")

    pred_col = "mean_y_blup" if mode == "fit" else "mean_y_pred"
    label = "CDE BLUP" if mode == "fit" else "CDE forecast"
    if pred_col in df_pop.columns:
        ax.plot(df_pop["time"], df_pop[pred_col], "o--", color="green",
                linewidth=2, markersize=6, label=label)

    ax.plot(df_pop["time"], df_pop["mean_mu_pop"], "o:", color="red",
            linewidth=1.5, markersize=5, alpha=0.7, label="CDE pop mean")

    if hlme_pop is not None:
        ax.plot(hlme_pop["time"], hlme_pop["mean_mu_pop"], "o--", color="black",
                linewidth=2, markersize=6, label="HLME")

    if ode_pop is not None:
        pred_ode = "mean_y_blup" if mode == "fit" else "mean_y_pred"
        if pred_ode in ode_pop.columns:
            ax.plot(ode_pop["time"], ode_pop[pred_ode], "o--", color="magenta",
                    linewidth=2, markersize=6, label="ODE BLUP")

    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("ISA15", fontsize=12)
    ax.set_title(f"Population-averaged predictions ({mode} mode) — Neural CDE",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Population-averaged plot saved to {save_path}")


# ====================================================================
# SCENARIO PRESETS
# ====================================================================

SCENARIO_DEFAULTS = {
    "S2": {
        "data": "simu_datasets/S2a_sims_2/sim_001.rds",
        "checkpoint": "checkpoints/best_CDE_S2.pt",
        "x_cols": ["BMI_t", "rs1", "rs2"],
    },
    "S5": {
        "data": "simu_datasets/S5_sims/sim_001.rds",
        "checkpoint": "checkpoints/best_CDE_S5.pt",
        "x_cols": ["BMI_t"],
    },
    "S6": {
        "data": "simu_datasets/S6_sims/sim_001.rds",
        "checkpoint": "checkpoints/best_CDE_S6.pt",
        "x_cols": ["BMI_t"],
    },
}


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prediction and evaluation for Neural CDE-LMM"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="S6",
                        choices=["S2", "S5", "S6"])
    parser.add_argument("--hlme_csv", type=str, default=None)
    parser.add_argument("--ode_csv", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    # Model architecture flags
    parser.add_argument("--inject_x", action="store_true", default=False)
    parser.add_argument("--augment_order", type=int, default=2)
    parser.add_argument("--encoder_sees_covariates", action="store_true", default=False)
    args = parser.parse_args()

    # Apply scenario defaults
    defaults = SCENARIO_DEFAULTS[args.scenario]
    if args.data is None:
        args.data = defaults["data"]
    if args.checkpoint is None:
        args.checkpoint = defaults["checkpoint"]
    x_cols = defaults["x_cols"]

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ──
    time_col = "time"
    y_col = "ISA15_sim"
    id_col = "NUM_ID"
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    df = next(iter(pyreadr.read_r(args.data).values()))
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    full_dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col,
                                       static_cols=static_cols)

    # ── Train/test split ──
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

    # ── Model ──
    n_tv = 1

    cfg = NeuralCDEConfig(
        hidden_channels=8,
        enc_mlp_hidden=32,
        func_mlp_hidden=32,
        dec_rho_hidden=16,
        dec_p=4,
        dec_q=3,
        depth=2,
        dropout=0.0,
    )

    # RE configuration
    if "rs1" in x_cols and "rs2" in x_cols:
        use_neural_re = False
        re_spline_cols = [x_cols.index("rs1"), x_cols.index("rs2")]
    else:
        use_neural_re = True
        re_spline_cols = None

    model = NeuralCDEModel(
        x_dim=len(x_cols),
        static_dim=len(static_cols),
        cfg=cfg,
        n_tv=n_tv,
        inject_x=args.inject_x,
        augment_order=args.augment_order,
        encoder_sees_covariates=args.encoder_sees_covariates,
        use_rho_net=True,
        use_neural_re=use_neural_re,
        g_hidden=16,
        re_spline_cols=re_spline_cols,
        fullD=True,
        tv_skip_cols=None,
        static_skip_dims=None,
        reg_mode=None,
    ).to(device)

    # ── Load checkpoint ──
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded: {args.checkpoint}")
        loss_val = checkpoint.get('best_test_loss', None)
        if loss_val is not None:
            print(f"  best test loss = {loss_val:.4f}")
        if 'config' in checkpoint:
            print(f"  config = {checkpoint['config']}")
    else:
        model.load_state_dict(checkpoint, strict=False)

    # Print model info
    model.eval()
    model.describe()
    print(f"  sigma2 = {torch.exp(model.decoder.log_residual_var).item():.4f}")

    if model.decoder.L_unconstrained is not None:
        D = model.decoder._build_D(device=torch.device('cpu'), dtype=torch.float32)
        print(f"  D matrix:\n{D}")

    # ── HLME / ODE predictions ──
    hlme_df = None
    if args.hlme_csv and os.path.exists(args.hlme_csv):
        hlme_df = pd.read_csv(args.hlme_csv)
        print(f"Loaded HLME predictions: {len(hlme_df)} rows")

    ode_df = None
    if args.ode_csv and os.path.exists(args.ode_csv):
        ode_df = pd.read_csv(args.ode_csv)
        print(f"Loaded ODE predictions: {len(ode_df)} rows")

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
    # 2. FIT MODE
    # ================================================================
    print(f"\n{'='*60}")
    print("2. FIT MODE (trajectory reconstruction)")
    print(f"{'='*60}")

    df_fit_full = predict_fit_mode(model, full_loader, device)
    mse_pop = compute_mse(df_fit_full, pred_col="mu_pop")
    mse_blup = compute_mse(df_fit_full, pred_col="y_blup")
    print(f"  Full: MSE(pop) = {mse_pop:.4f}, MSE(BLUP) = {mse_blup:.4f}, "
          f"n_obs = {len(df_fit_full)}")

    csv_path = os.path.join(args.output_dir, "fit_cde_full.csv")
    df_fit_full.to_csv(csv_path, index=False)
    print(f"    -> {csv_path}")

    df_pop_fit = population_averaged_predictions(df_fit_full)
    print(f"\n  Population-averaged (fit mode):")
    print(df_pop_fit.to_string(index=False))

    # ================================================================
    # 3. FORECASTING MODE
    # ================================================================
    print(f"\n{'='*60}")
    print("3. FORECASTING MODE (CDE up to t*, outcomes before t*)")
    print(f"{'='*60}")

    df_fc = predict_forecast_mode(model, full_loader, device)
    mse_fc_pop = compute_mse(df_fc, y_col="y_true", pred_col="mu_pop")
    mse_fc_pred = compute_mse(df_fc, y_col="y_true", pred_col="y_pred")
    print(f"  Full: MSE(pop) = {mse_fc_pop:.4f}, MSE(pred) = {mse_fc_pred:.4f}, "
          f"n_predictions = {len(df_fc)}")

    csv_path = os.path.join(args.output_dir, "forecast_cde_full.csv")
    df_fc.to_csv(csv_path, index=False)
    print(f"    -> {csv_path}")

    # ================================================================
    # 4. PLOTS
    # ================================================================
    print(f"\n{'='*60}")
    print("4. PLOTS")
    print(f"{'='*60}")

    plot_individual_predictions(
        df_fit_full, n_subjects=25, ncols=5,
        hlme_df=hlme_df, ode_df=ode_df,
        save_path=os.path.join(args.output_dir, "fit_predictions_cde.png")
    )

    hlme_pop = None
    if hlme_df is not None:
        hlme_pop = population_averaged_predictions(
            hlme_df.rename(columns={"y_pred": "y_blup"})
        )

    ode_pop = None
    if ode_df is not None:
        ode_pop = population_averaged_predictions(ode_df)

    plot_population_averaged(
        df_pop_fit, hlme_pop=hlme_pop, ode_pop=ode_pop, mode="fit",
        save_path=os.path.join(args.output_dir, "predictions_population_cde_fit.png")
    )

    # ================================================================
    # 5. SUMMARY
    # ================================================================
    print(f"\n{'='*60}")
    print(f"SUMMARY — CDE Scenario {args.scenario}")
    print(f"{'='*60}")
    print(f"  Fit MSE (pop):  {mse_pop:.4f}")
    print(f"  Fit MSE (BLUP): {mse_blup:.4f}")
    print(f"  Pred MSE (pop): {mse_fc_pop:.4f}")
    print(f"  Pred MSE (BLUP):{mse_fc_pred:.4f}")

    print("\nDone.")