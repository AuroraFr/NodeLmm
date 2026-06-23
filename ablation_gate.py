"""
Gate ablation: hard-zero the skip gates and evaluate held-out LL.

Loads a trained checkpoint, sets all gate logits to -∞ (so σ(α) → 0),
then evaluates the held-out NLL.  Compares with the original gate values.

Usage:
  python ablate_gates.py \
      --ckpt checkpoints/best_model_ode_real3C_practice_skipgate_H8_seed42.pt \
      --data_dir 3C_dataset
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd

from Preprocess_3C import process_data
from model_ODE_real import NeuralODEModel, NeuralODEConfig
from train_ODE_real import RealDataset, collate_real, compute_covariate_stats
from torch.utils.data import DataLoader
from utils import masked_NLL


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["config"]

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

    model = NeuralODEModel(
        n_tv=cfg_dict["n_tv"],
        static_dim=cfg_dict["static_dim"],
        cfg=cfg,
        use_rho_net=True,
        use_neural_re=True,
        g_hidden=8,
        fullD=False,
        cov_means=cfg_dict.get("cov_means", ckpt.get("cov_means")),
        cov_stds=cfg_dict.get("cov_stds", ckpt.get("cov_stds")),
        static_skip_dims=cfg_dict.get("static_skip_dims"),
        use_dynamic_skip=cfg_dict.get("use_dynamic_skip", True),
        reg_mode=cfg_dict.get("reg_mode", "skip_gate"),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, cfg_dict


def evaluate_nll(model, loader, device):
    model.eval()
    total_nll = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            _, x_aug, y_pad, target_mask, static = batch
            x_aug = x_aug.to(device)
            y_pad = y_pad.to(device)
            target_mask = target_mask.to(device)
            static = static.to(device)

            mu, V, Z, D, sig2, reg_dict = model(
                x_aug, static_covariates=static, obs_mask=target_mask,
            )
            loss = masked_NLL(mu, y_pad, V, target_mask)
            total_nll += loss.item()
            count += 1
    return total_nll / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="3C_dataset")
    args = parser.parse_args()

    device = "cpu"

    # ── Load model ──
    model, cfg_dict = load_model(args.ckpt, device)

    # ── Load test data ──
    tv_feats = cfg_dict["time_varying_features"]
    static_feats = cfg_dict["static_features"]
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_3C_data.csv"))

    if "AGEc" not in test_df.columns:
        all_df = pd.read_csv(os.path.join(args.data_dir, "data_3C.csv"))
        mean_age = all_df.groupby("NUM_ID")["AGE0"].first().mean()
        test_df["AGEc"] = (
            test_df.groupby("NUM_ID")["AGE0"].transform("first") - mean_age
        )

    test_data = process_data(
        df=test_df, id_col="NUM_ID",
        time_varying_features=tv_feats,
        static_features=static_feats,
        target_col="ISA15",
        interp_method=cfg_dict.get("interp_method", "linear"),
        mask_type=cfg_dict.get("mask_type", "binary"),
    )
    test_loader = DataLoader(
        RealDataset(test_data), batch_size=128,
        shuffle=False, collate_fn=collate_real,
    )

    # ── Print original gates ──
    gate_logits = model.decoder.skip_gate_logit
    names = tv_feats + static_feats
    original_gates = torch.sigmoid(gate_logits).detach().cpu()

    print("=" * 55)
    print("GATE ABLATION")
    print("=" * 55)
    print(f"\nOriginal gate values:")
    for g, name in enumerate(names):
        print(f"  {name:>8s}: σ(α) = {original_gates[g]:.4f}")

    # ── Evaluate with original gates ──
    nll_original = evaluate_nll(model, test_loader, device)
    print(f"\nHeld-out NLL (original gates): {nll_original:.4f}")

    # ── Hard-zero ALL gates ──
    with torch.no_grad():
        saved_logits = gate_logits.data.clone()
        gate_logits.data.fill_(-50.0)  # σ(-50) ≈ 0

    nll_all_zero = evaluate_nll(model, test_loader, device)
    print(f"Held-out NLL (all gates → 0):  {nll_all_zero:.4f}")
    print(f"  ΔNLL = {nll_all_zero - nll_original:+.4f}")

    # ── Restore and zero only DYNAMIC gates ──
    with torch.no_grad():
        gate_logits.data.copy_(saved_logits)
        n_dynamic = len(tv_feats)
        gate_logits.data[:n_dynamic].fill_(-50.0)

    nll_dyn_zero = evaluate_nll(model, test_loader, device)
    print(f"Held-out NLL (dynamic → 0):    {nll_dyn_zero:.4f}")
    print(f"  ΔNLL = {nll_dyn_zero - nll_original:+.4f}")

    # ── Restore and zero only STATIC gates ──
    with torch.no_grad():
        gate_logits.data.copy_(saved_logits)
        gate_logits.data[n_dynamic:].fill_(-50.0)

    nll_static_zero = evaluate_nll(model, test_loader, device)
    print(f"Held-out NLL (static → 0):     {nll_static_zero:.4f}")
    print(f"  ΔNLL = {nll_static_zero - nll_original:+.4f}")

    # ── Per-covariate ablation: zero one gate at a time ──
    print(f"\nPer-covariate ablation (zero one gate at a time):")
    print(f"  {'Covariate':>8s}  {'NLL':>10s}  {'ΔNLL':>10s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}")

    for g, name in enumerate(names):
        with torch.no_grad():
            gate_logits.data.copy_(saved_logits)
            gate_logits.data[g] = -50.0

        nll_g = evaluate_nll(model, test_loader, device)
        delta = nll_g - nll_original
        print(f"  {name:>8s}  {nll_g:10.4f}  {delta:+10.4f}")

    # ── Restore original ──
    with torch.no_grad():
        gate_logits.data.copy_(saved_logits)

    # ── Interpretation ──
    print(f"\n{'='*55}")
    print("INTERPRETATION")
    print(f"{'='*55}")
    print(f"""
  ΔNLL ≈ 0:   residual skip signal is negligible
               → benefit was purely in optimization
  ΔNLL > 0:   residual skip signal matters
               → model relies on the small leak through gates
    """)


if __name__ == "__main__":
    main()