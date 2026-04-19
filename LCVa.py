"""
Run LCVa on one or several saved model checkpoints and compare rankings
against the validation-NLL ranking.

Usage:
    # Option A: LCVa on just the winner (fast, reports one number)
    python run_lcva.py --checkpoints cv_final_model.pt

    # Option B: LCVa on a list of top-K checkpoints saved by grid_search
    python run_lcva.py --checkpoints ckpt_rank1.pt ckpt_rank2.pt ckpt_rank3.pt
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from CV_NODE_LMM import (
    CVConfig, prepare_datasets, build_model, compute_lcva_empirical_fisher,
)
from train_ODE_real import collate_real


def lcva_for_checkpoint(ckpt_path: str, train_ds, info, device):
    """Load a checkpoint, rebuild the model, compute LCVa."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = CVConfig(**ckpt["config"])
    cov_means = torch.tensor(ckpt["cov_means"])
    cov_stds = torch.tensor(ckpt["cov_stds"])

    model = build_model(cfg, info["n_tv"], info["static_dim"],
                        cov_means, cov_stds, device)
    model.load_state_dict(ckpt["model_state_dict"])

    P_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=False, collate_fn=collate_real)
    print(f"\n{cfg.key()}")
    print(f"  Nominal parameters P = {P_total}")
    print(f"  Computing LCVa (this may take 5-15 min)...")

    lcva = compute_lcva_empirical_fisher(model, loader, device=device)
    lcva["P_total"] = P_total
    lcva["config_key"] = cfg.key()
    # Recover val NLL if stored in the checkpoint
    if "val_metrics" in ckpt:
        lcva["val_fc_nll"] = ckpt["val_metrics"].get("nll_forecast_per_obs")
        lcva["val_fit_nll"] = ckpt["val_metrics"].get("nll_fit_per_subject")

    print(f"  L per subject     = {lcva['L_per_subject']:.4f}")
    print(f"  p_eff             = {lcva['p_eff']:.2f}")
    print(f"  p_eff / P         = {lcva['p_eff']/P_total:.4f}")
    print(f"  LCVa per subject  = {lcva['LCVa_per_subject']:.4f}")
    if "val_fc_nll" in lcva:
        print(f"  (val fc NLL was  = {lcva['val_fc_nll']:.4f})")
    return lcva


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="One or more .pt files saved by grid_search.")
    ap.add_argument("--out", default="lcva_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, _, _, info = prepare_datasets()

    results = []
    for ckpt in args.checkpoints:
        res = lcva_for_checkpoint(ckpt, train_ds, info, str(device))
        results.append(res)

    # Rank
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("Comparison")
        print("=" * 80)
        print(f"{'config':60s}  {'p_eff':>8s}  {'LCVa':>8s}  {'val_fc_nll':>10s}")
        by_lcva = sorted(results, key=lambda r: r["LCVa_per_subject"])
        for r in by_lcva:
            v = r.get("val_fc_nll", float("nan"))
            print(f"{r['config_key']:60s}  {r['p_eff']:>8.2f}  "
                  f"{r['LCVa_per_subject']:>8.4f}  {v:>10.4f}")

        # Rank agreement
        by_val = sorted(results, key=lambda r: r.get("val_fc_nll", float("inf")))
        if by_lcva[0]["config_key"] == by_val[0]["config_key"]:
            print("\n  LCVa and val NLL agree on the winner.")
        else:
            print(f"\n  LCVa winner: {by_lcva[0]['config_key']}")
            print(f"  Val winner:  {by_val[0]['config_key']}")
            print("  Rankings disagree -- decide how to frame this in the paper.")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")