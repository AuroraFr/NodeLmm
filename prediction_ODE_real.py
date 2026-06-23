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
# 6. PLOTS
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
                label="ODE BLUP")
        ax.plot(df_s["time"], df_s["mu_pop"], "r:", linewidth=1, alpha=0.6,
                label="ODE pop")

        if hlme_df is not None:
            hlme_s = hlme_df[hlme_df["patient_id"] == sid].sort_values("time")
            if len(hlme_s) > 0:
                ax.plot(hlme_s["time"], hlme_s["y_blup"], "k--",
                        linewidth=1.5, label="HLME")

        ax.set_title(f"ID {sid} (n={len(df_s)})", fontsize=9)
        if p_idx % ncols == 0:
            ax.set_ylabel("ISA15")
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
                             save_path="predictions_population.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(df_pop["time"], df_pop["mean_y_true"],
                yerr=df_pop["se_y_true"],
                fmt="o-", color="blue", linewidth=2, markersize=6,
                capsize=3, label="Observed (mean ± SE)")

    pred_col = "mean_y_blup" if mode == "fit" else "mean_y_pred"
    label = "ODE BLUP" if mode == "fit" else "ODE forecast"
    if pred_col in df_pop.columns:
        ax.plot(df_pop["time"], df_pop[pred_col], "o--", color="green",
                linewidth=2, markersize=6, label=label)

    # ax.plot(df_pop["time"], df_pop["mean_mu_pop"], "o:", color="red",
    #         linewidth=1.5, markersize=5, alpha=0.7, label="ODE pop mean")

    if hlme_pop is not None and "mean_y_blup" in hlme_pop.columns:
        ax.plot(hlme_pop["time"], hlme_pop["mean_y_blup"], "o--",
                color="black", linewidth=2, markersize=6, label="HLME BLUP")

    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("ISA15", fontsize=12)
    # ax.set_title(f"Population-averaged predictions ({mode}) — Neural ODE-LMM",
    #              fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
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

    if model.decoder.skip_gate_logit is not None:
        gates = torch.sigmoid(model.decoder.skip_gate_logit).detach().cpu()
        names = time_varying_features + static_features
        print(f"  Skip gates:")
        for g, name in enumerate(names):
            print(f"    {name:>8s}: {gates[g]:.4f}")

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
    df_fit_test = predict_fit_mode(model, train_loader, device)
    df_pred_train = predict_forecast_mode(model, test_loader, device)
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
    # 4. PLOTS
    # ================================================================
    print(f"\n{'='*60}")
    print("4. PLOTS")
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

    plot_population_averaged(
        df_pop_test, hlme_pop=hlme_pop, mode="fit",
        save_path=os.path.join(args.output_dir, "population_average.png"))

    print("\nDone.")