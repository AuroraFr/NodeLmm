"""
Model selection experiment for S1 (instantaneous BMI × age interaction).

Compares configs over 10–20 replicates to validate that NLL-based
model selection recovers the correct pathway. Saves checkpoints for
post-hoc ΔPDP bias analysis.

Usage:
    python train_ODE_model_selection.py --scenario S1 --n_sims 10 --data_dir simu_datasets/S1_sims
"""
from __future__ import annotations

import os
import json
import time
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import pyreadr
from dataset import LongitudinalDataset, collate_pad
from model_ODE_baseline import NeuralODEModel, NeuralODEConfig
from utils import masked_NLL


@dataclass
class ExpConfig:
    name: str
    use_bmi_skip: bool
    use_bmi_in_ode: bool
    use_learned_z0: bool
    reg_mode: str | None
    lambda_reg: float
    hidden_channels: int = 4
    dec_p: int = 4
    dec_q: int = 3
    skip_statics: bool = True


def get_s1_configs() -> list[ExpConfig]:
    return [
        # ExpConfig(name="A_skip_only", use_bmi_skip=True, use_bmi_in_ode=False,
        #           use_learned_z0=True, reg_mode=None, lambda_reg=0.0),
        # ExpConfig(name="B_both_no_reg", use_bmi_skip=True, use_bmi_in_ode=True,
        #           use_learned_z0=False, reg_mode=None, lambda_reg=0.0),
        # ExpConfig(name="C_both_group_lasso", use_bmi_skip=True, use_bmi_in_ode=True,
        #           use_learned_z0=False, reg_mode="group_lasso", lambda_reg=0.1),
        ExpConfig(name="D_noskip", use_bmi_skip=False, use_bmi_in_ode=True,
                  use_learned_z0=False, reg_mode=None, lambda_reg=0, skip_statics=False),
    ]


def build_model(exp, bmi_mean, bmi_std, device):
    cfg = NeuralODEConfig(
        hidden_channels=exp.hidden_channels, enc_mlp_hidden=16,
        func_mlp_hidden=16, dec_rho_hidden=16,
        dec_p=exp.dec_p, dec_q=exp.dec_q,
        depth=2, dropout=0.0, euler_steps_per_interval=4,
    )
    return NeuralODEModel(
        x_dim=1, static_dim=4, cfg=cfg, n_tv=1,
        use_rho_net=True, use_neural_re=True,
        re_spline_cols=None, g_hidden=8, fullD=False,
        bmi_mean=bmi_mean, bmi_std=bmi_std,
        use_bmi_skip=exp.use_bmi_skip,
        static_skip_dims= [0, 1, 2, 3] if exp.skip_statics else None,
        reg_mode=exp.reg_mode,
        use_learned_z0=exp.use_learned_z0,
        use_bmi_in_ode=exp.use_bmi_in_ode,
    ).to(device)


def train_one(model, train_loader, test_loader, device,
              lambda_reg=0.0, lr=1e-3, wd=1e-5,
              max_epochs=1000, patience=300, print_every=50,
              save_path=None, save_meta=None):
    """Train with early stopping. Returns best test NLL.
    If save_path is given, saves the best checkpoint."""

    nn_weights, gate_params, var_params, fe_params = [], [], [], []
    for n, p in model.named_parameters():
        if 'skip_gate_logit' in n:
            gate_params.append(p)
        elif 'log_residual_var' in n or 'log_std' in n:
            var_params.append(p)
        elif 'beta_neural' in n:
            fe_params.append(p)
        else:
            nn_weights.append(p)

    opt = torch.optim.AdamW([
        {'params': nn_weights, 'weight_decay': wd},
        {'params': gate_params, 'weight_decay': 0.0},
        {'params': var_params, 'weight_decay': 0.0},
        {'params': fe_params, 'weight_decay': 0.0},
    ], lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=50)

    best_test = float('inf')
    best_state = None
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_sum, n_batch = 0.0, 0
        for batch in train_loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad, x_pad = t_pad.to(device), x_pad.to(device)
            y_pad, mask, s = y_pad.to(device), mask.to(device), s.to(device)

            mu, V, _, _, _, reg_dict = model(
                t_pad, x_pad, masks=None,
                static_covariates=s, bmi_t=x_pad[:, :, 0:1], obs_mask=mask)

            loss = masked_NLL(mu, y_pad, V, mask)
            if reg_dict and "reg_term" in reg_dict and lambda_reg > 0:
                loss = loss + lambda_reg * reg_dict["reg_term"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_sum += loss.item()
            n_batch += 1

        model.eval()
        test_sum, t_batch = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
                t_pad, x_pad = t_pad.to(device), x_pad.to(device)
                y_pad, mask, s = y_pad.to(device), mask.to(device), s.to(device)
                mu, V, _, _, _, _ = model(
                    t_pad, x_pad, masks=None,
                    static_covariates=s, bmi_t=x_pad[:, :, 0:1], obs_mask=mask)
                test_sum += masked_NLL(mu, y_pad, V, mask).item()
                t_batch += 1

        avg_test = test_sum / max(t_batch, 1)
        scheduler.step(avg_test)

        if avg_test < best_test - 1e-4:
            best_test = avg_test
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if epoch % print_every == 0 or epoch == 1:
            avg_tr = train_sum / max(n_batch, 1)
            print(f"      ep {epoch:4d}  train={avg_tr:.4f}  "
                  f"test={avg_test:.4f}  best={best_test:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path:
            ckpt = {'model_state_dict': best_state, 'best_test_nll': best_test}
            if save_meta:
                ckpt.update(save_meta)
            torch.save(ckpt, save_path)
            print(f"    Checkpoint saved: {save_path}")

    return best_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="S1", choices=["S1", "S2"])
    parser.add_argument("--n_sims", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default="simu_datasets/S1_sims")
    parser.add_argument("--output", type=str, default="results_simu/model_selection_S1_2.json")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/model_selection_S1")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    BATCH_SIZE = 128
    TEST_RATIO = 0.2

    configs = get_s1_configs()
    config_names = [c.name for c in configs]
    x_cols = ["BMI_t"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    all_results = {c.name: [] for c in configs}
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for sim_idx in range(args.n_sims):
        print(f"\n{'#'*60}")
        print(f"# SIMULATION {sim_idx}")
        print(f"{'#'*60}")

        path = f"{args.data_dir}/sim_{sim_idx + 1:03d}.rds"
        df = next(iter(pyreadr.read_r(path).values()))
        df["SEX"] = df["SEX"].astype("category")
        df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
        df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
        df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

        dataset = LongitudinalDataset(df, "NUM_ID", "time", x_cols, "ISA15_sim",
                                      static_cols=static_cols)

        N = len(dataset)
        rng = np.random.RandomState(SEED)
        indices = rng.permutation(N)
        n_test = int(N * TEST_RATIO)
        train_idx = indices[n_test:]
        test_idx = indices[:n_test]
        train_ds = Subset(dataset, train_idx)
        test_ds = Subset(dataset, test_idx)

        train_bmis = torch.cat([dataset[idx][2][:, 0] for idx in train_idx])
        bmi_mean = train_bmis.mean().item()
        bmi_std = train_bmis.std().item()

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_pad)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                 shuffle=False, collate_fn=collate_pad)

        for exp in configs:
            print(f"\n  Config: {exp.name}")
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            np.random.seed(SEED)

            model = build_model(exp, bmi_mean, bmi_std, device)
            print(f"    params: {sum(p.numel() for p in model.parameters())}")

            ckpt_path = os.path.join(args.ckpt_dir,
                                     f"{exp.name}_sim{sim_idx:03d}.pt")
            save_meta = {
                'config': asdict(exp),
                'sim_idx': sim_idx,
                'bmi_mean': bmi_mean,
                'bmi_std': bmi_std,
                'train_idx': train_idx.tolist(),
                'test_idx': test_idx.tolist(),
            }

            t0 = time.time()
            best_nll = train_one(
                model, train_loader, test_loader, device,
                lambda_reg=exp.lambda_reg, print_every=100,
                save_path=ckpt_path, save_meta=save_meta,
            )
            elapsed = time.time() - t0
            print(f"    best test NLL = {best_nll:.4f}  ({elapsed:.0f}s)")

            all_results[exp.name].append(best_nll)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ─── Aggregate ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"MODEL SELECTION — {args.scenario} ({args.n_sims} replicates)")
    print(f"{'='*70}")
    print(f"{'Config':<30s} {'Mean NLL':>10s} {'SE':>8s} {'#Best':>6s}")
    print(f"{'-'*54}")

    n_sims = args.n_sims
    wins = {name: 0 for name in config_names}
    for si in range(n_sims):
        nlls = {name: all_results[name][si] for name in config_names}
        wins[min(nlls, key=nlls.get)] += 1

    for name in config_names:
        vals = np.array(all_results[name])
        print(f"{name:<30s} {vals.mean():10.4f} "
              f"{vals.std()/np.sqrt(len(vals)):8.4f} "
              f"{wins[name]:6d}/{n_sims}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'scenario': args.scenario, 'n_sims': n_sims,
            'configs': {n: asdict(c) for n, c in zip(config_names, configs)},
            'test_nlls': all_results, 'wins': wins,
            'ckpt_dir': args.ckpt_dir,
        }, f, indent=2)
    print(f"\nSaved to {args.output}")
    print(f"Checkpoints in {args.ckpt_dir}/")