"""
Model selection evaluation script.
Computes forecasting log-likelihood and MSE metrics on train and test sets.

Metrics:
  - Marginal LL (fit mode): condition on all outcomes
  - Fitted MSE: population prediction vs observed (mu = W@beta + h)
  - Prediction MSE (forecasting): BLUP prediction vs observed
    For each subject at time t*, condition on outcomes before t*,
    predict Y(t*) using BLUP.

Usage:
  python evaluate.py --checkpoint best_model_hybrid_reg.pt
"""
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pyreadr
import argparse
from dataset import LongitudinalDataset, collate_pad
from model import NeuralCDEModel, NeuralCDEConfig
from utils import masked_NLL

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def compute_fitted_mse(model, loader, device):
    """
    Fitted MSE: population mean prediction vs observed.
    MSE = (1/N_obs) Σ (Y_ij - mu_ij)² where mu = W@beta + h
    """
    model.eval()
    total_se = 0.0
    total_obs = 0

    with torch.no_grad():
        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad  = t_pad.to(device)
            x_pad  = x_pad.to(device)
            y_pad  = y_pad.to(device)
            mask   = mask.to(device)
            c_mask = c_mask.to(device)
            s      = s.to(device)

            mu, V, _, _, h = model(t_pad, x_pad, c_mask, s, mask, y_pad=None)

            residuals = (y_pad - mu) ** 2 * mask
            total_se += residuals.sum().item()
            total_obs += mask.sum().item()

    return total_se / max(total_obs, 1)


def compute_conditional_mse(model, loader, device):
    """
    Conditional (BLUP) MSE: subject-specific prediction vs observed.
    For each subject, compute BLUP b_i from ALL observed outcomes,
    then MSE = (1/N_obs) Σ (Y_ij - mu_ij - Z_ij @ b_i)²
    This is the "fit mode" conditional prediction.
    """
    model.eval()
    total_se = 0.0
    total_obs = 0

    with torch.no_grad():
        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad  = t_pad.to(device)
            x_pad  = x_pad.to(device)
            y_pad  = y_pad.to(device)
            mask   = mask.to(device)
            c_mask = c_mask.to(device)
            s      = s.to(device)

            mu, V, _, _, h = model(t_pad, x_pad, c_mask, s, mask, y_pad=None)

            N, T = t_pad.shape

            # Build Z and D for BLUP
            Z = model.decoder._build_Z(t_pad, x_pad)   # (N, T, q)
            Z = Z * mask.unsqueeze(-1)
            D = model.decoder._build_D(device, t_pad.dtype)
            sig2 = torch.exp(model.decoder.log_residual_var)

            for i in range(N):
                obs_idx = mask[i].bool()
                n_i = obs_idx.sum()
                if n_i < 1:
                    continue

                Z_i = Z[i, obs_idx]                      # (n_i, q)
                V_i = V[i][obs_idx][:, obs_idx]          # (n_i, n_i)
                r_i = y_pad[i, obs_idx] - mu[i, obs_idx] # (n_i,)

                # BLUP: b_i = D Z_i' V_i^{-1} r_i
                V_inv_r = torch.linalg.solve(V_i, r_i)   # (n_i,)
                b_i = D @ Z_i.t() @ V_inv_r              # (q,)

                # Conditional prediction
                mu_cond = mu[i, obs_idx] + Z_i @ b_i     # (n_i,)
                se = ((y_pad[i, obs_idx] - mu_cond) ** 2).sum().item()
                total_se += se
                total_obs += n_i.item()

    return total_se / max(total_obs, 1)


def compute_forecasting_mse(model, loader, device):
    """
    Forecasting MSE: at each visit t* (except the first), condition on
    outcomes BEFORE t*, compute BLUP, predict Y(t*).

    For CDE: covariates up to t* are used (forecasting mode 2),
    but outcomes only up to t*-1 for BLUP.

    MSE = (1/N_pred) Σ_{i,j>1} (Y_ij - Ŷ_ij)²
    """
    model.eval()
    total_se = 0.0
    total_pred = 0

    with torch.no_grad():
        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad  = t_pad.to(device)
            x_pad  = x_pad.to(device)
            y_pad  = y_pad.to(device)
            mask   = mask.to(device)
            c_mask = c_mask.to(device)
            s      = s.to(device)

            # Full forward pass — mu uses all covariates (CDE integrates full path)
            mu, V, _, _, h = model(t_pad, x_pad, c_mask, s, mask, y_pad=None)

            N, T = t_pad.shape

            # Build Z and D
            Z = model.decoder._build_Z(t_pad, x_pad)
            Z = Z * mask.unsqueeze(-1)
            D = model.decoder._build_D(device, t_pad.dtype)

            for i in range(N):
                obs_idx = mask[i].bool()
                obs_positions = torch.where(obs_idx)[0]
                n_i = len(obs_positions)
                if n_i < 2:
                    continue

                # For each observed visit j > 0, predict using BLUP from visits 0..j-1
                for j_idx in range(1, n_i):
                    j = obs_positions[j_idx]

                    # Past observations: visits 0..j-1
                    past_pos = obs_positions[:j_idx]

                    Z_past = Z[i, past_pos]                    # (j_idx, q)
                    V_past = V[i][past_pos][:, past_pos]       # (j_idx, j_idx)
                    r_past = y_pad[i, past_pos] - mu[i, past_pos]  # (j_idx,)

                    # BLUP from past only
                    try:
                        V_inv_r = torch.linalg.solve(V_past, r_past)
                        b_i = D @ Z_past.t() @ V_inv_r        # (q,)
                    except:
                        continue

                    # Predict at current visit j
                    Z_j = Z[i, j]                              # (q,)
                    y_pred = mu[i, j] + Z_j @ b_i              # scalar
                    y_true = y_pad[i, j]

                    total_se += (y_true - y_pred).item() ** 2
                    total_pred += 1

    return total_se / max(total_pred, 1)


def compute_marginal_ll(model, loader, device, n_W):
    """Marginal log-likelihood (per subject average)."""
    model.eval()
    total_nll = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad  = t_pad.to(device)
            x_pad  = x_pad.to(device)
            y_pad  = y_pad.to(device)
            mask   = mask.to(device)
            c_mask = c_mask.to(device)
            s      = s.to(device)

            mu, V, _, _, h = model(t_pad, x_pad, c_mask, s, mask, y_pad=None)
            loss = masked_NLL(mu, y_pad, V, mask)

            total_nll += loss.item()
            count += 1

    return total_nll / max(count, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_model_hybrid_reg.pt")
    parser.add_argument("--data", type=str, default="simu_datasets/S2a_sim/sim_001.rds")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    time_col = "time"
    y_col = "ISA15_sim"
    id_col = "NUM_ID"
    x_cols = ["GLUC_t", "BMI_t", "ns1", "ns2", "ns3", "rs1", "rs2"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    df = next(iter(pyreadr.read_r(args.data).values()))
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    full_dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col, static_cols=static_cols)

    # ---- Load checkpoint ----
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    train_idx = checkpoint.get('train_idx', None)
    test_idx = checkpoint.get('test_idx', None)

    if train_idx is None or test_idx is None:
        print("WARNING: No train/test split in checkpoint. Using full dataset.")
        train_idx = np.arange(len(full_dataset))
        test_idx = np.arange(len(full_dataset))

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset  = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)

    print(f"Train: {len(train_idx)} subjects, Test: {len(test_idx)} subjects")

    # ---- Spline knots ----
    fe_knots    = np.array([1.769863, 6.693151])
    fe_boundary = np.array([0.0, 13.50685])
    re_knots    = np.array([3.567123])
    re_boundary = np.array([0.0, 13.50685])

    n_tv = 2

    # ---- Rebuild model from config ----
    cfg = NeuralCDEConfig(
        hidden_channels=config.get('hidden_channels', 8),
        enc_mlp_hidden=32,
        func_mlp_hidden=32,
        dec_rho_hidden=16,
        dec_p=4,
        dec_q=3,
        depth=2,
    )

    model = NeuralCDEModel(
        x_dim=len(x_cols),
        static_dim=len(static_cols),
        cfg=cfg,
        fe_spline_knots=fe_knots,
        fe_spline_boundary=fe_boundary,
        re_spline_knots=re_knots,
        re_spline_boundary=re_boundary,
        interaction_pairs=config.get('interaction_pairs', None),
        precomputed_splines=True,
        n_tv=n_tv,
        include_fe_splines=config.get('include_fe_splines', True),
        use_rho_net=config.get('use_rho_net', True),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded from {args.checkpoint}")

    # ---- Print stored beta ----
    beta = model.decoder._last_beta.detach().cpu()
    beta_names = ["intercept"]
    if model.decoder.include_fe_splines:
        beta_names += [f"ns{i+1}" for i in range(model.decoder.fe_spline_df)]
    beta_names += x_cols[:n_tv] + static_cols
    print(f"\nβ coefficients:")
    for name, val in zip(beta_names, beta):
        print(f"  {name:>20s} = {val.item():+.6f}")

    n_W = model.decoder.n_W

    # ====================================================================
    # EVALUATION METRICS
    # ====================================================================
    print("\n" + "="*70)
    print(f"{'Metric':<30s} {'Train':>12s} {'Test':>12s}")
    print("="*70)

    # 1. Marginal log-likelihood
    train_ll = compute_marginal_ll(model, train_loader, device, n_W)
    test_ll  = compute_marginal_ll(model, test_loader, device, n_W)
    print(f"{'Marginal NLL (per batch)':<30s} {train_ll:>12.4f} {test_ll:>12.4f}")

    # 2. Fitted MSE (population mean)
    train_fit_mse = compute_fitted_mse(model, train_loader, device)
    test_fit_mse  = compute_fitted_mse(model, test_loader, device)
    print(f"{'Fitted MSE (mu only)':<30s} {train_fit_mse:>12.4f} {test_fit_mse:>12.4f}")

    # 3. Conditional MSE (BLUP, fit mode — all outcomes)
    train_cond_mse = compute_conditional_mse(model, train_loader, device)
    test_cond_mse  = compute_conditional_mse(model, test_loader, device)
    print(f"{'Conditional MSE (BLUP fit)':<30s} {train_cond_mse:>12.4f} {test_cond_mse:>12.4f}")

    # 4. Forecasting MSE (BLUP, past outcomes only)
    train_fc_mse = compute_forecasting_mse(model, train_loader, device)
    test_fc_mse  = compute_forecasting_mse(model, test_loader, device)
    print(f"{'Forecasting MSE (BLUP pred)':<30s} {train_fc_mse:>12.4f} {test_fc_mse:>12.4f}")

    print("="*70)

    print(f"\nFor comparison with HLME Table 3:")
    print(f"  HLME Model 4 (train): LL=-79474, Fitted MSE=8.78, Pred MSE=21.01")
    print(f"  HLME Model 4 (test):  LL=-20006, Fitted MSE=9.10, Pred MSE=21.82")