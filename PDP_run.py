"""
pdp_run.py — config-driven runner for PDP / ΔPDP-with-CI.

Standalone: it rebuilds the model from the checkpoint's embedded `train_config`
(saved by train_ODE.py) and does its own data loading. Two settings, one core:

  setting="real":
    - rebuild model from a single checkpoint
    - load the TRAINING data (Fisher and the ΔPDP gradient must be averaged over
      the same sample the model was fit on, for the delta method to be valid)
    - compute the empirical Fisher (λ_reg / weight_decay taken from train_config)
    - ΔPDP-with-CI over named profile pairs, chosen `variance_mode`
    - write pairwise-ΔPDP plots + results

  setting="simulation":
    - loop over replicate checkpoints (glob)
    - each: rebuild model, load that replicate's training split, run a flat
      lo/hi ΔPDP contrast with a `true_contrast` oracle
    - aggregate across replicates → Var_est vs Var_MC + 95% coverage table

Run:
    python pdp_run.py --config configs/pdp_real.json
    python pdp_run.py --config configs/pdp_sim.json
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Optional, List
import os
import json
import glob
import argparse

import numpy as np
import torch

from model import NeuralODEModel, NeuralODEConfig
from Dataset import XAugDataset, collate_xaug
import PDP
import DPDP_variance as DV


# ══════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class PDPConfig:
    name: str = "pdp"
    setting: str = "real"                 # "real" | "simulation"

    # ── checkpoints ─────────────────────────────────────────────────────
    checkpoint: str = ""                  # real: one path; simulation: glob

    # ── target intervention ─────────────────────────────────────────────
    target_feature: str = "BMI"           # must be in train_config's TV features
    v_lo: float = 24.0
    v_hi: float = 30.0

    # ── evaluation grids ────────────────────────────────────────────────
    ci_grid: List[float] = field(default_factory=lambda: [2.0, 4.0, 8.0, 10.0])
    plot_t_max: float = 14.0              # fine grid for smooth PDP-mean plots
    plot_n_points: int = 15

    # ── contrasts / profiles ────────────────────────────────────────────
    profile_set: str = "named"            # "named" (real) | "lohi" (simulation)
    pairs: List[List[str]] = field(
        default_factory=lambda: [["stable_high", "stable_low"]])
    mask_type: str = "binary"

    # ── variance ────────────────────────────────────────────────────────
    variance_mode: str = "marquardt"      # marquardt|ledoit_wolf|active_subspace|bayesian|sandwich
    n_hessian_subsample: Optional[int] = None
    fisher_max_subjects: Optional[int] = None
    active_threshold: float = 1e-4

    # ── simulation oracle ───────────────────────────────────────────────
    oracle: dict = field(default_factory=lambda: {"kind": "none"})

    # ── output ──────────────────────────────────────────────────────────
    output_dir: str = "pdp_out"
    make_plots: bool = True
    verbose: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "PDPConfig":
        valid = {f.name for f in fields(cls)}
        unknown = set(d) - valid
        if unknown:
            raise ValueError(f"Unknown PDP config keys: {sorted(unknown)}")
        return cls(**d)


def load_pdp_config(path: str) -> PDPConfig:
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            import yaml
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)
    return PDPConfig.from_dict(raw)


# ══════════════════════════════════════════════════════════════════════════
# Rebuild model from a train_ODE checkpoint
# ══════════════════════════════════════════════════════════════════════════

def rebuild_model_from_checkpoint(ckpt_path, device):
    """Reconstruct the model from the embedded train_config + cov stats and
    load its weights. Returns (model, train_config_dict)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tc = ckpt["train_config"]
    cov_means = ckpt["cov_means"]
    cov_stds = ckpt["cov_stds"]

    K = len(tc["time_varying_features"])
    Ks = len(tc["static_features"])
    skip_dims = (tc["static_skip_dims"] if tc.get("static_skip_dims") is not None
                 else list(range(Ks)))

    model = NeuralODEModel(
        n_tv=K, static_dim=Ks,
        cfg=NeuralODEConfig(**tc["cfg"]),
        use_rho_net=tc["use_rho_net"],
        use_neural_re=tc["use_neural_re"],
        g_hidden=tc["g_hidden"],
        fullD=tc["fullD"],
        cov_means=cov_means, cov_stds=cov_stds,
        static_skip_dims=skip_dims,
        use_dynamic_skip=tc["use_dynamic_skip"],
        reg_mode=tc["reg_mode"],
        include_mask=tc["include_mask"],
        use_dynamic_in_ode=tc["use_dynamic_in_ode"],
        encode_baseline_cov=tc["encode_baseline_cov"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, tc


# ══════════════════════════════════════════════════════════════════════════
# Data loading (own, dispatched on train_config's data_kind)
# ══════════════════════════════════════════════════════════════════════════

def load_training_dataset(tc: dict, sim_index: Optional[int] = None):
    """Load the TRAINING data the model was fit on, as an XAugDataset.

    csv_real : the pre-split train_csv.
    rds_sim  : replicate `sim_index`, split reproduced from split_seed/test_ratio,
               returning the training subjects.
    """
    data_kind = tc["data_kind"]

    if data_kind == "csv_real":
        import pandas as pd
        from Preprocess_3C import process_data
        df = pd.read_csv(tc["train_csv"])
        if "AGEc" not in df.columns:
            all_df = pd.read_csv(tc["all_csv"])
            base_mean = all_df.groupby(tc["id_col"])["AGE0"].first().mean()
            df["AGEc"] = df.groupby(tc["id_col"])["AGE0"].transform("first") - base_mean
        patient_data = process_data(
            df=df, id_col=tc["id_col"],
            time_varying_features=tc["time_varying_features"],
            static_features=tc["static_features"],
            target_col=tc["target_col"],
            interp_method=tc["interp_method"], mask_type=tc["mask_type"])
        return XAugDataset.from_processed(patient_data)

    if data_kind == "rds_sim":
        import pyreadr
        assert sim_index is not None, "rds_sim needs a sim_index"
        path = tc["data_dir"] + f"/sim_{sim_index + 1:03d}.rds"
        df = next(iter(pyreadr.read_r(path).values()))
        df["SEX"] = df["SEX"].astype("category")
        df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
        df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
        df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")
        full = XAugDataset.from_long_df(
            df, tc["id_col"], tc["time_col"], tc["time_varying_features"],
            tc["target_col"], tc["static_features"])
        # reproduce the exact training split
        N = len(full)
        rng = np.random.RandomState(tc["split_seed"])
        idx = rng.permutation(N)
        n_test = int(N * tc["test_ratio"])
        train_idx = idx[n_test:]
        return torch.utils.data.Subset(full, train_idx.tolist())

    raise ValueError(f"Unknown data_kind {data_kind!r}")


# ══════════════════════════════════════════════════════════════════════════
# Oracle (true ΔPDP) for simulation validation
# ══════════════════════════════════════════════════════════════════════════
 
def _mean_agec(dataset, agec_static_col):
    print(dataset)
    """Mean of AGEc over the evaluation sample (the term `a` in the oracle).
    AGEc is a static covariate; static[:, agec_static_col] is per subject."""
    vals = []
    for i in range(len(dataset)):
        static = dataset[i][4]                      # (Ks,)  s_i
        vals.append(float(static[agec_static_col]))
    return float(np.mean(vals))
 
 
def build_oracle(spec: dict, v_lo, v_hi, dataset):
    """Return a callable(t) -> true ΔPDP for a constant-BMI contrast
    (v_hi vs v_lo), marginalised over the evaluation sample's AGEc.
 
    Δv = v_hi - v_lo,  a = mean(AGEc) over `dataset`.
 
    kinds (DGP coefficients supplied in the config):
      {"kind": "linear",     "beta_bmi": .., "beta_int": .., "agec_col": j}
          contemporaneous LMM with a BMI×AGEc interaction:
            ΔPDP(t) = Δv · (β_BMI + a·β_int)          [constant in t]
      {"kind": "cumulative", "beta_bmi": .., "beta_int": .., "alpha": ..,
                             "agec_col": j}
          adds α ∫₀ᵗ BMI dτ; under constant v this is α·v·t, so
            ΔPDP(t) = Δv · (β_BMI + a·β_int + α·t)     [linear in t]
    """
    kind = spec["kind"]
    dv = v_hi - v_lo
    beta_bmi = float(spec["beta_bmi"])
    beta_int = float(spec["beta_int"])
    a = _mean_agec(dataset, int(spec["agec_col"]))
    base = beta_bmi + a * beta_int
 
    if kind == "linear":
        val = dv * base
        return lambda t: val
    if kind == "cumulative":
        alpha = float(spec["alpha"])
        return lambda t: dv * (base + alpha * float(t))
    raise ValueError(f"Unknown oracle kind: {kind!r} (linear | cumulative)")


# ══════════════════════════════════════════════════════════════════════════
# Shared: Fisher + contrast on a training dataset
# ══════════════════════════════════════════════════════════════════════════

def _target_col(tc: dict, feature: str) -> int:
    tvf = tc["time_varying_features"]
    if feature not in tvf:
        raise ValueError(f"target_feature {feature!r} not in {tvf}")
    return tvf.index(feature)


def _fisher(model, dataset, device, tc, cfg: PDPConfig):
    """Empirical Fisher with λ_reg / weight_decay taken from train_config
    (they MUST match training for the stationarity condition to hold)."""
    fisher, scores = DV.compute_empirical_fisher(
        model, dataset, device, collate_xaug,
        lambda_reg=tc["lambda_reg"], weight_decay=tc["wd"],
        max_subjects=cfg.fisher_max_subjects, verbose=cfg.verbose)
    _, fisher_inv = DV._regularise_and_invert(fisher, "Fisher", verbose=cfg.verbose)
    return fisher, scores, fisher_inv


# ══════════════════════════════════════════════════════════════════════════
# Real-data run
# ══════════════════════════════════════════════════════════════════════════

def run_real(cfg: PDPConfig, device):
    from torch.utils.data import DataLoader
    os.makedirs(cfg.output_dir, exist_ok=True)

    model, tc = rebuild_model_from_checkpoint(cfg.checkpoint, device)
    n_tv = len(tc["time_varying_features"])
    target_col = _target_col(tc, cfg.target_feature)

    dataset = load_training_dataset(tc)                    # eval on TRAIN
    loader = DataLoader(dataset, batch_size=64, shuffle=False,
                        collate_fn=collate_xaug)
    ci_grid = np.asarray(cfg.ci_grid, dtype=float)

    # PDP on a fine grid (for smooth plots)
    plot_grid = PDP.make_eval_grid(cfg.plot_t_max, cfg.plot_n_points)
    plot_profiles = PDP.make_profiles(plot_grid, cfg.v_lo, cfg.v_hi)
    PDP.compute_pdp_by_profile(
        model, loader, device, plot_profiles, plot_grid,
        target_col=target_col, n_tv=n_tv, mask_type=cfg.mask_type,
        target_name=cfg.target_feature)

    # Fisher on the training sample (λ from train_config)
    fisher, scores, fisher_inv = _fisher(model, dataset, device, tc, cfg)

    # ΔPDP-with-CI on the (coarser) CI grid — pass fisher AND fisher_inv so the
    # stationarity / null-space diagnostics are available on real data too.
    ci_profiles = PDP.make_profiles(ci_grid, cfg.v_lo, cfg.v_hi)
    pairs = [tuple(p) for p in cfg.pairs]
    ci_results = DV.compute_contrast_pdp_with_ci(
        model, loader, device, ci_profiles, ci_grid, pairs,
        target_col=target_col, n_tv=n_tv, mask_type=cfg.mask_type,
        variance_mode=cfg.variance_mode,
        fisher=fisher, scores=scores, fisher_inv=fisher_inv,
        dataset=dataset, collate_fn=collate_xaug,
        lambda_reg=tc["lambda_reg"], weight_decay=tc["wd"],
        n_hessian_subsample=cfg.n_hessian_subsample,
        active_threshold=cfg.active_threshold, verbose=cfg.verbose)

    if cfg.make_plots:
        PDP.plot_all_pairwise_delta_pdp(
            ci_results, ci_grid,
            save_path=os.path.join(cfg.output_dir, f"{cfg.name}_delta_pairs.eps"),
            target_name=cfg.target_feature, fisher_inv=fisher_inv)

    np.savez(os.path.join(cfg.output_dir, f"{cfg.name}_results.npz"),
             ci_grid=ci_grid,
             **{k: v for k, v in _flatten_ci(ci_results).items()})
    print(f"  real ΔPDP done: {cfg.output_dir}")
    return ci_results


def _flatten_ci(ci_results):
    """Flatten the nested result dict into npz-friendly arrays."""
    out = {}
    for key, d in ci_results.items():
        if key == "_meta" or not isinstance(d, dict):
            continue
        for fld in ("mean", "se", "ci_lo", "ci_hi"):
            if fld in d:
                out[f"{key}.{fld}"] = np.asarray(d[fld])
    return out


# ══════════════════════════════════════════════════════════════════════════
# Simulation run
# ══════════════════════════════════════════════════════════════════════════

def _sim_index_from_ckpt(path):
    """Best-effort replicate index from a checkpoint filename."""
    import re
    m = re.findall(r"(\d+)", os.path.basename(path))
    return int(m[-1]) if m else None


def run_simulation(cfg: PDPConfig, device):
    from torch.utils.data import DataLoader
    os.makedirs(cfg.output_dir, exist_ok=True)

    ckpts = sorted(glob.glob(cfg.checkpoint))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints match {cfg.checkpoint!r}")
    ci_grid = np.asarray(cfg.ci_grid, dtype=float)
    L = len(ci_grid)
    pair = ("hi", "lo")
    pair_key = "_delta_hi_vs_lo"

    all_results = []
    for ckpt_path in ckpts:
        sim_index = _sim_index_from_ckpt(ckpt_path)
        if cfg.verbose:
            print(f"\n── replicate {sim_index} ({os.path.basename(ckpt_path)}) ──")
        model, tc = rebuild_model_from_checkpoint(ckpt_path, device)
        n_tv = len(tc["time_varying_features"])
        target_col = _target_col(tc, cfg.target_feature)

        dataset = load_training_dataset(tc, sim_index=sim_index)
        loader = DataLoader(dataset, batch_size=64, shuffle=False,
                            collate_fn=collate_xaug)
        oracle = build_oracle(cfg.oracle, cfg.v_lo, cfg.v_hi, dataset)
        profiles = {"lo": [cfg.v_lo] * L, "hi": [cfg.v_hi] * L}
        true_fns = {pair: oracle} if oracle is not None else None

        res = DV.compute_contrast_pdp_with_ci(
            model, loader, device, profiles, ci_grid, [pair],
            target_col=target_col, n_tv=n_tv, mask_type=cfg.mask_type,
            variance_mode=cfg.variance_mode,
            dataset=dataset, collate_fn=collate_xaug,
            lambda_reg=tc["lambda_reg"], weight_decay=tc["wd"],
            n_hessian_subsample=cfg.n_hessian_subsample,
            true_contrast_fns=true_fns,
            active_threshold=cfg.active_threshold, verbose=cfg.verbose)
        all_results.append(res)
        break

    summary = DV.aggregate_simulations(all_results, ci_grid, pair_key=pair_key)
    _print_sim_summary(summary, ci_grid)

    np.savez(os.path.join(cfg.output_dir, f"{cfg.name}_coverage.npz"),
             ci_grid=ci_grid,
             **{f"t{t}": np.array([summary[float(t)].get(k, np.nan)
                                   for k in ("mean_hat", "true", "bias",
                                             "var_mc", "mean_var_est",
                                             "coverage95")])
                for t in ci_grid if float(t) in summary})
    print(f"\n  simulation validation done ({len(ckpts)} replicates) "
          f" {cfg.output_dir}")
    return summary


def _print_sim_summary(summary, ci_grid):
    diag = summary.get("_diagnostics", {})
    print(f"\n{'='*72}")
    print("SIMULATION VALIDATION — Var_est vs Var_MC + 95% coverage")
    print(f"  mean stationarity ratio = "
          f"{diag.get('mean_stationarity_ratio', float('nan')):.4f}")
    print(f"{'='*72}")
    print(f"  {'t':>6s}  {'ΔPDP_hat':>9s}  {'true':>8s}  {'bias':>8s}  "
          f"{'Var_MC':>9s}  {'Var_est':>9s}  {'cover95':>8s}  {'D':>4s}")
    for t in ci_grid:
        tk = float(t)
        if tk not in summary:
            continue
        s = summary[tk]
        print(f"  {tk:6.1f}  {s['mean_hat']:+9.4f}  {s['true']:+8.4f}  "
              f"{s['bias']:+8.4f}  {s['var_mc']:9.4f}  {s['mean_var_est']:9.4f}  "
              f"{s['coverage95']:8.3f}  {s['D']:4d}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PDP / ΔPDP-with-CI runner")
    parser.add_argument("--config", required=True, help="JSON/YAML PDP config")
    parser.add_argument("--checkpoint", default=None,
                        help="override checkpoint path/glob")
    args = parser.parse_args()

    cfg = load_pdp_config(args.config)
    if args.checkpoint:
        cfg.checkpoint = args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(f"PDP RUNNER — {cfg.name}  (setting={cfg.setting}, "
          f"variance_mode={cfg.variance_mode}, device={device})")
    print("=" * 72)

    if cfg.setting == "real":
        run_real(cfg, device)
    elif cfg.setting == "simulation":
        run_simulation(cfg, device)
    else:
        raise ValueError(f"Unknown setting {cfg.setting!r} (real | simulation)")


if __name__ == "__main__":
    main()