"""
Grid-search driver for the Neural ODE-LMM on the real 3C cohort.

Protocol (three CSVs, three roles, no additional splits):
  - train_3C_data_1.csv : training data
  - val_3C_data_1.csv   : early stopping AND model selection across configs
  - test_3C_data.csv    : final unbiased report on the winner (touched once)

Per-config flow:
  train on full train CSV -> val CSV for early stopping -> record val metrics.
Selection:
  winner = argmin over configs of val marginal NLL.
Final:
  retrain winner on train with val for early stopping -> evaluate on test.

Covariate standardisation stats are computed once, from the training CSV only.
Run:
    python cv_neural_ode_lmm.py
"""
from __future__ import annotations

import json
import time
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Preprocess_3C import process_data
from model_ODE_real import NeuralODEModel, NeuralODEConfig
from utils import masked_NLL
from train_ODE_real import RealDataset, collate_real, compute_covariate_stats


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

@dataclass
class CVConfig:
    # Architecture
    hidden_channels: int = 4
    enc_mlp_hidden: int = 16
    func_mlp_hidden: int = 16
    dec_rho_hidden: int = 16
    dec_p: int = 4
    dec_q: int = 3
    depth: int = 2
    dropout: float = 0.0
    euler_steps_per_interval: int = 4
    ode_solver: str = "rk4"

    # Decoder variants
    use_rho_net: bool = True
    use_rho_norm: bool = False  
    use_neural_re: bool = True
    g_hidden: int = 8
    fullD: bool = False

    # Skip architecture
    use_dynamic_skip: bool = True
    static_skip_all: bool = True
    static_skip_dims: int = 4,
    # Regularisation
    reg_mode: Optional[str] = None
    lambda_reg: float = 0.0

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 1000
    patience: int = 300

    def key(self) -> str:
        return (f"h{self.hidden_channels}_p{self.dec_p}_q{self.dec_q}"
                f"_reg-{self.reg_mode}_lam{self.lambda_reg:g}"
                f"_skip-{int(self.use_dynamic_skip)}"
                f"_rhonorm-{int(self.use_rho_norm)}"
                f"_nre-{int(self.use_neural_re)}")


# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────

def prepare_datasets(
    train_csv: str = "3C_dataset/train_3C_data_1.csv",
    val_csv: str = "3C_dataset/val_3C_data_1.csv",
    test_csv: str = "3C_dataset/test_3C_data.csv",
    all_csv: str = "3C_dataset/data_3C.csv",
    id_col: str = "NUM_ID",
    target_col: str = "ISA15",
    time_varying_features=("BMI", "PAS", "PAD", "GLUC", "HDL"),
    static_features=("SEX_code", "AGEc", "DIPNIV_2", "DIPNIV_3"),
    interp_method: str = "linear",
    mask_type: str = "binary",
):
    """Load the three CSVs and return (train_ds, val_ds, test_ds, info)."""
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    if "AGEc" not in df_train.columns:
        all_df = pd.read_csv(all_csv)
        baseline_age = all_df.groupby(id_col)["AGE0"].transform("first")
        mu_age = baseline_age.mean()
        for df_ in (df_train, df_val, df_test):
            df_["AGEc"] = (
                df_.groupby(id_col)["AGE0"].transform("first") - mu_age)

    kwargs = dict(
        id_col=id_col,
        time_varying_features=list(time_varying_features),
        static_features=list(static_features),
        target_col=target_col,
        interp_method=interp_method,
        mask_type=mask_type,
    )
    return (
        RealDataset(process_data(df=df_train, **kwargs)),
        RealDataset(process_data(df=df_val, **kwargs)),
        RealDataset(process_data(df=df_test, **kwargs)),
        {
            "n_tv": len(time_varying_features),
            "static_dim": len(static_features),
            "time_varying_features": list(time_varying_features),
            "static_features": list(static_features),
        },
    )


# ─────────────────────────────────────────────────────────────────
# Model construction from CVConfig
# ─────────────────────────────────────────────────────────────────

def build_model(cfg: CVConfig, n_tv: int, static_dim: int,
                cov_means, cov_stds, device):
    ode_cfg = NeuralODEConfig(
        hidden_channels=cfg.hidden_channels,
        enc_mlp_hidden=cfg.enc_mlp_hidden,
        func_mlp_hidden=cfg.func_mlp_hidden,
        dec_rho_hidden=cfg.dec_rho_hidden,
        dec_p=cfg.dec_p,
        dec_q=cfg.dec_q,
        depth=cfg.depth,
        dropout=cfg.dropout,
        euler_steps_per_interval=cfg.euler_steps_per_interval,
        ode_solver=cfg.ode_solver,
    )
    static_skip_dims = list(range(static_dim)) if cfg.static_skip_all else []
    model = NeuralODEModel(
        n_tv=n_tv,
        static_dim=static_dim,
        cfg=ode_cfg,
        use_rho_net=cfg.use_rho_net,
        use_neural_re=cfg.use_neural_re,
        g_hidden=cfg.g_hidden,
        fullD=cfg.fullD,
        cov_means=cov_means.tolist() if torch.is_tensor(cov_means) else cov_means,
        cov_stds=cov_stds.tolist() if torch.is_tensor(cov_stds) else cov_stds,
        static_skip_dims=static_skip_dims,
        use_dynamic_skip=cfg.use_dynamic_skip,
        reg_mode=cfg.reg_mode,
    ).to(device)

    # rho_norm ablation: replace with Identity when turned off
    if cfg.use_rho_net and not cfg.use_rho_norm:
        model.decoder.rho_norm = nn.Identity()
    return model


# ─────────────────────────────────────────────────────────────────
# Training: train on train_loader, early-stop on val_loader
# ─────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, cfg: CVConfig, device,
                verbose: bool = True):
    """
    Train on train_loader, early-stop on val_loader.

    Mirrors train_ODE_real.py exactly:
      - Adam + weight_decay, ReduceLROnPlateau on val NLL
      - gradient clipping at 1.0
      - best-val-NLL checkpointing, early stop after `patience` epochs
        without improvement
    """

    nn_weights, gate_params, var_params, fe_params = [], [], [], []
    for n, p in model.named_parameters():
        print(n)
        if 'skip_gate_logit' in n:
            gate_params.append(p)
        elif 'log_residual_var' in n or 'log_std' in n:
            var_params.append(p)
        elif 'beta_neural' in n:
            fe_params.append(p)
        else:
            nn_weights.append(p)

    print(gate_params, fe_params, var_params)
    opt = torch.optim.AdamW([
        {'params': nn_weights, 'weight_decay': cfg.weight_decay},
        {'params': gate_params, 'weight_decay': 0.0},
        {'params': var_params, 'weight_decay': 0.0},
        {'params': fe_params,  'weight_decay': 0.0},  # or a small value if desired
    ])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=50,
        verbose=False)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, cfg.max_epochs + 1):
        # --- train ---
        model.train()
        n_batches = 0
        train_loss_sum = 0.0
        for _, x_aug, y, target_mask, static in train_loader:
            x_aug = x_aug.to(device); y = y.to(device)
            target_mask = target_mask.to(device); static = static.to(device)

            mu, V, _, _, _, reg_dict = model(
                x_aug, static_covariates=static, obs_mask=target_mask)
            loss = masked_NLL(mu, y, V, target_mask)
            if reg_dict and "reg_term" in reg_dict:
                loss = loss + cfg.lambda_reg * reg_dict["reg_term"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_sum += loss.item()
            n_batches += 1

        # --- val (for early stopping; raw NLL, no reg term) ---
        model.eval()
        val_nll_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for _, x_aug, y, target_mask, static in val_loader:
                x_aug = x_aug.to(device); y = y.to(device)
                target_mask = target_mask.to(device); static = static.to(device)
                mu, V, _, _, _, _ = model(
                    x_aug, static_covariates=static, obs_mask=target_mask)
                val_nll_sum += masked_NLL(mu, y, V, target_mask).item()
                val_batches += 1
        avg_val = val_nll_sum / max(val_batches, 1)
        scheduler.step(avg_val)

        if verbose and (epoch % 25 == 0 or epoch == 1):
            avg_tr = train_loss_sum / max(n_batches, 1)
            print(f"    ep {epoch:4d}  train={avg_tr:.4f}  "
                  f"val={avg_val:.4f}  best={best_val:.4f}")

        if avg_val < best_val - 1e-4:
            best_val = avg_val
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# ─────────────────────────────────────────────────────────────────
# BLUP + metrics
# ─────────────────────────────────────────────────────────────────

def _blup_point_predictions(mu, V, Z, D, sig2, y, target_mask,
                            forecast_cutoff=None):
    """Y_hat = mu + Z @ b_hat per subject. Forecast mode drops visits >= cutoff."""
    N, T = mu.shape
    Y_hat = mu.clone()
    for i in range(N):
        obs = target_mask[i].bool()
        if forecast_cutoff is not None:
            c = int(forecast_cutoff[i].item())
            keep = torch.zeros_like(obs)
            keep[:c] = obs[:c]
            obs = keep
        n_obs = int(obs.sum().item())
        if n_obs == 0:
            continue
        Z_obs = Z[i, obs]
        r = (y[i, obs] - mu[i, obs]).unsqueeze(-1)
        V_oo = Z_obs @ D @ Z_obs.t() + sig2 * torch.eye(
            n_obs, device=mu.device, dtype=mu.dtype)
        b_hat = D @ Z_obs.t() @ torch.linalg.solve(V_oo, r)
        Y_hat[i] = mu[i] + Z[i] @ b_hat.squeeze(-1)
    return Y_hat


def evaluate_loader(model, loader, device):
    """
    Evaluate on any held-out loader (val or test).

    Returns:
      - nll_per_subject:  marginal NLL (b_i integrated out) — model selection criterion
      - mse_fit_per_obs:  MSE with BLUP using all visits
      - mse_fc_per_obs:   MSE with BLUP using all visits except the last (forecast)
    """
    model.eval()
    tot = {
        "nll_sum": 0.0, "n_subj": 0, "n_obs_total": 0,
        "sqerr_fit_sum": 0.0,
        "sqerr_fc_sum": 0.0, "n_fc": 0,
    }
    with torch.no_grad():
        for _, x_aug, y, target_mask, static in loader:
            x_aug = x_aug.to(device); y = y.to(device)
            target_mask = target_mask.to(device); static = static.to(device)
            mu, V, Z, D, sig2, _ = model(
                x_aug, static_covariates=static, obs_mask=target_mask)

            N, T = y.shape
            tot["nll_sum"] += masked_NLL(mu, y, V, target_mask).item() * N
            tot["n_subj"] += N
            tot["n_obs_total"] += int(target_mask.sum().item())

            # Fit MSE: BLUP from all observed visits
            Y_hat_fit = _blup_point_predictions(
                mu, V, Z, D, sig2, y, target_mask, forecast_cutoff=None)
            sq = ((Y_hat_fit - y) ** 2) * target_mask
            tot["sqerr_fit_sum"] += sq.sum().item()

            # Forecast MSE: BLUP from all visits except the last
            cutoff = torch.zeros(N, dtype=torch.long, device=device)
            last_obs = torch.full((N,), -1, dtype=torch.long, device=device)
            for i in range(N):
                obs_idx = target_mask[i].bool().nonzero(as_tuple=True)[0]
                if len(obs_idx) >= 2:
                    last_obs[i] = obs_idx[-1].item()
                    cutoff[i] = obs_idx[-1].item()
            Y_hat_fc = _blup_point_predictions(
                mu, V, Z, D, sig2, y, target_mask, forecast_cutoff=cutoff)
            for i in range(N):
                j = int(last_obs[i].item())
                if j < 0:
                    continue
                tot["sqerr_fc_sum"] += (Y_hat_fc[i, j] - y[i, j]).item() ** 2
                tot["n_fc"] += 1

    n_subj = max(tot["n_subj"], 1)
    n_obs = max(tot["n_obs_total"], 1)
    n_fc = max(tot["n_fc"], 1)
    return {
        "nll_per_subject":   tot["nll_sum"] / n_subj,
        "mse_fit_per_obs":   tot["sqerr_fit_sum"] / n_obs,
        "mse_fc_per_obs":    tot["sqerr_fc_sum"] / n_fc,
        "n_subjects":        tot["n_subj"],
        "n_obs_forecast":    tot["n_fc"],
    }


# ─────────────────────────────────────────────────────────────────
# One config: train + evaluate on val
# ─────────────────────────────────────────────────────────────────

def run_config(
    cfg: CVConfig,
    train_loader, val_loader,
    cov_means, cov_stds,
    info: dict, device: str,
    verbose: bool = True,
    init_seed: int = 42
):
    """
    Train the config on the train CSV with early stopping on the val CSV,
    and return the full set of val metrics plus the trained model.
    """

    # Reset RNG so every config starts from the same initialisation state.
    # Layers with identical shapes across configs will get identical weights;
    # layers that differ (because the architecture differs) will still get
    # deterministic draws from the same generator sequence.
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)


    model = build_model(cfg, info["n_tv"], info["static_dim"],
                        cov_means, cov_stds, device)
    model, best_val_nll = train_model(
        model, train_loader, val_loader, cfg, device, verbose=verbose)
    val_metrics = evaluate_loader(model, val_loader, device)
    val_metrics["best_early_stopping_nll"] = best_val_nll
    return model, val_metrics


# ─────────────────────────────────────────────────────────────────
# Grid search: one training per config
# ─────────────────────────────────────────────────────────────────

def grid_search(
    configs: list[CVConfig],
    train_dataset, val_dataset, info: dict,
    out_path: Optional[str] = None,
    device: str = "cuda",
    save_best_model_path: Optional[str] = "cv_best_model.pt",
):
    """
    For each config: train on train CSV, early-stop + select on val CSV.
    Rank by val marginal NLL.

    The best config's trained model is saved to disk so we don't need to
    retrain it for the test evaluation.
    """
    cov_means, cov_stds = compute_covariate_stats(
        train_dataset, list(range(len(train_dataset))), info["n_tv"])

    train_loader = DataLoader(
        train_dataset, batch_size=configs[0].batch_size,
        shuffle=True, collate_fn=collate_real)
    val_loader = DataLoader(
        val_dataset, batch_size=configs[0].batch_size,
        shuffle=False, collate_fn=collate_real)

    all_results = []
    best_val_nll = float("inf")

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {cfg.key()}")
        t0 = time.time()
        try:
            model, val_metrics = run_config(
                cfg, train_loader, val_loader,
                cov_means, cov_stds, info, device, verbose=True)
            v = val_metrics
            print(f"  VAL  nll={v['nll_per_subject']:.4f}  "
                  f"fit_mse={v['mse_fit_per_obs']:.4f}  "
                  f"fc_mse={v['mse_fc_per_obs']:.4f}")

            res = {
                "config_key": cfg.key(), "config": asdict(cfg),
                "val_metrics": val_metrics,
                "elapsed_s": time.time() - t0, "status": "ok",
            }

            # Save the best model to disk so we don't retrain for test
            if (save_best_model_path is not None
                    and val_metrics["nll_per_subject"] < best_val_nll):
                best_val_nll = val_metrics["nll_per_subject"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "cov_means": cov_means.tolist(),
                    "cov_stds": cov_stds.tolist(),
                    "val_metrics": val_metrics,
                }, save_best_model_path)
                print(f"  [new best -> saved to {save_best_model_path}]")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            res = {"config_key": cfg.key(), "config": asdict(cfg),
                   "status": "error", "error": repr(e),
                   "elapsed_s": time.time() - t0}
            print(f"  ERROR: {e}")

        all_results.append(res)
        if out_path is not None:
            Path(out_path).write_text(json.dumps(all_results, indent=2))

    # Rank
    ok = [r for r in all_results if r.get("status") == "ok"]
    ok.sort(key=lambda r: r["val_metrics"]["nll_per_subject"])

    print("\n" + "=" * 80)
    print("Top 10 configs by VAL NLL")
    print("=" * 80)
    for r in ok[:10]:
        v = r["val_metrics"]
        print(f"  {r['config_key']:70s}")
        print(f"     nll={v['nll_per_subject']:.4f}  "
              f"fit_mse={v['mse_fit_per_obs']:.4f}  "
              f"fc_mse={v['mse_fc_per_obs']:.4f}")
    return all_results


# ─────────────────────────────────────────────────────────────────
# Winner -> test (touched once)
# ─────────────────────────────────────────────────────────────────

def evaluate_winner_on_test(
    train_dataset, val_dataset, test_dataset, info: dict,
    winner_ckpt_path: str = "cv_best_model.pt",
    device: str = "cuda",
    retrain: bool = False,
):
    """
    Evaluate the winner on the test CSV.

    By default uses the model saved during grid_search (`retrain=False`).
    Set `retrain=True` to retrain from scratch on train + early-stop on val.
    """
    ckpt = torch.load(winner_ckpt_path, map_location=device, weights_only=False)
    cfg = CVConfig(**ckpt["config"])
    cov_means = torch.tensor(ckpt["cov_means"])
    cov_stds = torch.tensor(ckpt["cov_stds"])

    print(f"\n{'='*80}")
    print(f"FINAL: evaluating winner on test CSV")
    print(f"Winner: {cfg.key()}")
    print(f"{'='*80}")

    if retrain:
        print("Retraining winner from scratch...")
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            shuffle=True, collate_fn=collate_real)
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.batch_size,
            shuffle=False, collate_fn=collate_real)
        model, val_metrics = run_config(
            cfg, train_loader, val_loader,
            cov_means, cov_stds, info, device, verbose=True)
    else:
        print("Reusing the model saved during grid search.")
        model = build_model(cfg, info["n_tv"], info["static_dim"],
                            cov_means, cov_stds, device)
        model.load_state_dict(ckpt["model_state_dict"])
        val_metrics = ckpt["val_metrics"]

    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False, collate_fn=collate_real)
    test_metrics = evaluate_loader(model, test_loader, device)

    print("\nVAL (from selection):")
    for k, v in val_metrics.items():
        print(f"  {k}: {v}")
    print("\nTEST (goes in the paper):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")
    return model, val_metrics, test_metrics, cfg, cov_means, cov_stds


# ─────────────────────────────────────────────────────────────────
# Empirical-Fisher LCVa (optional diagnostic)
# ─────────────────────────────────────────────────────────────────

def compute_lcva_empirical_fisher(model, loader, device, ridge: float = 1e-3):
    """Post-hoc LCVa on a single trained model. Run once on the winner."""
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    scores = []
    total_ll = 0.0
    n_subj = 0

    for _, x_aug, y, target_mask, static in loader:
        B = x_aug.shape[0]
        for i in range(B):
            model.zero_grad()
            xi = x_aug[i:i+1].to(device)
            si = static[i:i+1].to(device)
            yi = y[i:i+1].to(device)
            mi = target_mask[i:i+1].to(device)
            mu, V, _, _, _, _ = model(xi, static_covariates=si, obs_mask=mi)
            nll_i = masked_NLL(mu, yi, V, mi)
            nll_i.backward()
            g = torch.cat([p.grad.detach().flatten()
                           for p in params if p.grad is not None]).cpu()
            scores.append(g)
            total_ll += (-nll_i).item()
            n_subj += 1

    S = torch.stack(scores, dim=0)
    F = S.t() @ S
    F_reg = F + ridge * torch.diag(torch.diag(F).clamp(min=1e-8))
    try:
        F_inv = torch.linalg.inv(F_reg)
        p_eff = float(torch.trace(F_inv @ F).item())
    except Exception:
        p_eff = float("nan")

    n = max(n_subj, 1)
    return {
        "L_per_subject": total_ll / n,
        "p_eff": p_eff,
        "LCVa_per_subject": -total_ll / n + p_eff / n,
        "n_subjects": n_subj,
    }


# ─────────────────────────────────────────────────────────────────
# Grid builders
# ─────────────────────────────────────────────────────────────────

def build_tier1_grid() -> list[CVConfig]:
    """Tier 1: reg_mode × lambda × use_dynamic_skip."""
    grid = []
    for reg_mode in [None, "skip_gate"]:
        lams = [0.0] if reg_mode is None else [0.1, 1.0]
        for lam in lams:
            for use_skip in [True, False]:
                grid.append(CVConfig(
                    reg_mode=reg_mode,
                    lambda_reg=lam,
                    use_dynamic_skip=use_skip,
                    use_rho_norm=True,
                    static_skip_dims = 4
                ))
    return grid


def build_tier2_grid(best: CVConfig) -> list[CVConfig]:
    """Tier 2: around the winner, sweep rho_norm, latent dim, p."""
    grid = []
    for rho_norm in [False, True]:
        for h in [4, 8, 16]:
            for p in [2, 4]:
                cfg = CVConfig(**asdict(best))
                cfg.use_rho_norm = rho_norm
                cfg.hidden_channels = h
                cfg.dec_p = p
                grid.append(cfg)
    return grid


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    # ---- 1. Load data ----
    train_ds, val_ds, test_ds, info = prepare_datasets()
    print(f"Train: {len(train_ds)} subjects")
    print(f"Val:   {len(val_ds)} subjects  (early stopping + selection)")
    print(f"Test:  {len(test_ds)} subjects  (touched once, at the end)")

    # ---- 2. Tier-1 grid ----
    tier1 = build_tier1_grid()
    print(f"\nTier 1: {len(tier1)} configurations")
    tier1_results = grid_search(
        tier1, train_ds, val_ds, info,
        out_path="cv_results_tier1.json",
        save_best_model_path="cv_best_tier1.pt",
        device=str(device))

    # ---- 3. Tier-2 around the Tier-1 winner ----
    ok1 = [r for r in tier1_results if r.get("status") == "ok"]
    ok1.sort(key=lambda r: r["val_metrics"]["nll_per_subject"])
    best_tier1 = CVConfig(**ok1[0]["config"])
    print(f"\nBest Tier 1: {best_tier1.key()}")

    tier2 = build_tier2_grid(best_tier1)
    print(f"\nTier 2: {len(tier2)} configurations")
    tier2_results = grid_search(
        tier2, train_ds, val_ds, info,
        out_path="cv_results_tier2.json",
        save_best_model_path="cv_best_tier2.pt",
        device=str(device))

    # ---- 4. Pick final winner, evaluate on test ----
    ok2 = [r for r in tier2_results if r.get("status") == "ok"]
    ok2.sort(key=lambda r: r["val_metrics"]["nll_per_subject"])

    # Use whichever tier produced the best val NLL
    best_tier2_val = ok2[0]["val_metrics"]["nll_per_subject"]
    best_tier1_val = ok1[0]["val_metrics"]["nll_per_subject"]
    winner_path = ("cv_best_tier2.pt" if best_tier2_val < best_tier1_val
                   else "cv_best_tier1.pt")

    (model_final, val_metrics, test_metrics,
     final_cfg, cov_means, cov_stds) = evaluate_winner_on_test(
        train_ds, val_ds, test_ds, info,
        winner_ckpt_path=winner_path, device=str(device), retrain=False)

    # ---- 5. Optional: LCVa diagnostic on the winner ----
    full_train_loader = DataLoader(
        train_ds, batch_size=final_cfg.batch_size,
        shuffle=False, collate_fn=collate_real)
    lcva = compute_lcva_empirical_fisher(
        model_final, full_train_loader, device=str(device))
    print(f"\nLCVa: {lcva}")

    # ---- 6. Save final artifacts ----
    torch.save({
        "model_state_dict": model_final.state_dict(),
        "config": asdict(final_cfg),
        "cov_means": cov_means.tolist(),
        "cov_stds": cov_stds.tolist(),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }, "cv_final_model.pt")
    print("\nSaved cv_final_model.pt")