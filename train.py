"""
Run:
    python train.py --config configs/my_dataset.json
    python train.py --config configs/config_basic.json          # YAML needs pyyaml

Data sources (config field `data_kind`):
    "rds_sim"  : .rds replicate files → XAugDataset.from_long_df,
                 subject-level split done internally with `split_seed`. The
                 long dataframe is converted to per-subject x_aug (mask block
                 all-ones, since sims are fully observed); collate_xaug pads
                 variable-length subjects. The trainer loops over
                 sim_start..sim_end.
    "csv_real" : pre-split CSVs → process_data → XAugDataset.from_processed;
                 x_aug already carries [time, x(K), mask(K)]. collate_xaug
                 pads nothing (fixed grid).

To add a new data source, write a prepare_*() that returns a Bundle whose
loaders emit the unified 5-tuple (use collate_xaug); the core is unchanged.

Example configs for the three original setups are provided as
configs/config_basic.json or configs/config_cumulative.json / config_real.json — copy one
and edit it for your dataset. 
Regenerate them with:  python make_configs.py
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields, asdict
from typing import Optional, List
import os
import json
import argparse
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model import NeuralODEModel, NeuralODEConfig
from Dataset import XAugDataset, collate_xaug
from utils import masked_NLL

warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    name: str = "run"
    data_kind: str = "csv_real"          # "rds_sim" | "csv_real"

    # ── features ──────────────────────────────────────────────────────────
    time_varying_features: List[str] = field(default_factory=list)
    static_features: List[str] = field(default_factory=list)
    target_col: str = "ISA15"
    id_col: str = "NUM_ID"
    time_col: str = "time"

    # ── rds_sim data ──────────────────────────────────────────────────────
    data_dir: str = ""
    sim_start: int = 0                   # inclusive, 0-based file index
    sim_end: int = 0                     # exclusive
    split_seed: int = 42
    test_ratio: float = 0.2

    # ── csv_real data ─────────────────────────────────────────────────────
    train_csv: str = ""
    val_csv: str = ""
    all_csv: str = ""
    interp_method: str = "linear"        # "ffill" | "linear" | "cubic"
    mask_type: str = "binary"            # "binary" | "cumulative"
    seeds: List[int] = field(default_factory=lambda: [42])

    # ── model flags ───────────────────────────────────────────────────────
    include_mask: bool = False
    use_dynamic_in_ode: bool = True
    encode_baseline_cov: bool = False
    use_neural_re: bool = True
    g_hidden: int = 8
    fullD: bool = False
    use_rho_net: bool = True
    use_dynamic_skip: bool = True
    static_skip_dims: Optional[List[int]] = None   # None ⇒ all statics
    reg_mode: Optional[str] = None
    lambda_reg: float = 0.0

    # ── model architecture ────────────────────────────────────────────────
    cfg: NeuralODEConfig = field(default_factory=NeuralODEConfig)

    # ── training ──────────────────────────────────────────────────────────
    lr: float = 1e-3
    wd: float = 1e-5
    epochs: int = 1000
    batch_size: int = 128
    print_every: int = 25
    patience_init: int = 150
    patience_reset: int = 150
    sched_patience: int = 50

    # ── checkpoint ────────────────────────────────────────────────────────
    ckpt_dir: str = "checkpoints"
    ckpt_template: str = "best_model_ode_{tag}.pt"

    # ── (de)serialisation ─────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = dict(d)
        cfg_raw = d.pop("cfg", {}) or {}
        cfg_valid = {f.name for f in fields(NeuralODEConfig)}
        cfg_unknown = set(cfg_raw) - cfg_valid
        if cfg_unknown:
            raise ValueError(f"Unknown cfg keys: {sorted(cfg_unknown)}")
        cfg = NeuralODEConfig(**cfg_raw)

        valid = {f.name for f in fields(cls)} - {"cfg"}
        unknown = set(d) - valid
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(cfg=cfg, **d)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str) -> TrainConfig:
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as e:
                raise ImportError(
                    "YAML config requested but pyyaml is not installed. "
                    "Use a .json config or `pip install pyyaml`.") from e
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)
    return TrainConfig.from_dict(raw)


# ══════════════════════════════════════════════════════════════════════════
# Covariate statistics (training set only)
# ══════════════════════════════════════════════════════════════════════════

def compute_cov_stats(dataset, train_idx, K):
    """Per-channel mean/std from observed covariate values in the training
    set. Reads x_aug = [time, x(K), mask(K)] and uses the covariate mask block,
    so it works for both sources: sims have an all-ones mask (all values used),
    real data uses only truly observed values."""
    vals = {k: [] for k in range(K)}
    for idx in train_idx:
        x_aug = dataset[idx][1]               # (Ti, 1+2K)
        x_interp = x_aug[:, 1:1 + K]
        cov_mask = x_aug[:, 1 + K:1 + 2 * K]
        for k in range(K):
            obs_k = cov_mask[:, k] > 0.5
            if obs_k.any():
                vals[k].append(x_interp[obs_k, k])
    means, stds = torch.zeros(K), torch.zeros(K)
    for k in range(K):
        v = torch.cat(vals[k])
        means[k], stds[k] = v.mean(), v.std().clamp(min=1e-6)
    return means, stds


# ══════════════════════════════════════════════════════════════════════════
# Data preparation to get common bundle
# ══════════════════════════════════════════════════════════════════════════

class Bundle:
    """Everything run_training needs, regardless of data source."""
    def __init__(self, train_loader, test_loader, cov_means, cov_stds,
                 K, Ks):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cov_means = cov_means
        self.cov_stds = cov_stds
        self.K = K
        self.Ks = Ks


def unpack_batch(batch, device):
    """collate_xaug yields the unified 5-tuple
    (ids, x_aug, y, obs_mask, static) with x_aug = [time, x(K), mask(K)].
    Reorder to the training signature and move to device."""
    _ids, x_aug, y_pad, obs_mask, static = batch
    return (x_aug.to(device), static.to(device),
            y_pad.to(device), obs_mask.to(device))


def prepare_sim(cfg: TrainConfig, sim_index: int) -> Bundle:
    import pyreadr   # .rds reader — only needed for this data source

    K = len(cfg.time_varying_features)
    Ks = len(cfg.static_features)

    path = cfg.data_dir + f"/sim_{sim_index + 1:03d}.rds"
    df = next(iter(pyreadr.read_r(path).values()))
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    full_dataset = XAugDataset.from_long_df(
        df, cfg.id_col, cfg.time_col, cfg.time_varying_features,
        cfg.target_col, cfg.static_features)

    N = len(full_dataset)
    rng = np.random.RandomState(cfg.split_seed)
    indices = rng.permutation(N)
    n_test = int(N * cfg.test_ratio)
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    cov_means, cov_stds = compute_cov_stats(full_dataset, train_idx, K)

    train_loader = DataLoader(Subset(full_dataset, train_idx),
                              batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_xaug)
    test_loader = DataLoader(Subset(full_dataset, test_idx),
                             batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate_xaug)
    print(f"  Subjects: {N} → {len(train_idx)} train / {len(test_idx)} test")
    return Bundle(train_loader, test_loader, cov_means, cov_stds,
                  K, Ks)


def prepare_real(cfg: TrainConfig) -> Bundle:
    import pandas as pd
    from Preprocess_3C import process_data

    K = len(cfg.time_varying_features)
    Ks = len(cfg.static_features)

    df = pd.read_csv(cfg.train_csv)
    test_df = pd.read_csv(cfg.val_csv)

    if "AGEc" not in df.columns:
        all_df = pd.read_csv(cfg.all_csv)
        base_mean = all_df.groupby(cfg.id_col)["AGE0"].first().mean()
        print("  baseline_age_mean", base_mean)
        df["AGEc"] = df.groupby(cfg.id_col)["AGE0"].transform("first") - base_mean
        test_df["AGEc"] = test_df.groupby(cfg.id_col)["AGE0"].transform("first") - base_mean

    kw = dict(id_col=cfg.id_col,
              time_varying_features=cfg.time_varying_features,
              static_features=cfg.static_features,
              target_col=cfg.target_col,
              interp_method=cfg.interp_method,
              mask_type=cfg.mask_type)
    patient_data = process_data(df=df, **kw)
    test_patient_data = process_data(df=test_df, **kw)
    print(f"  Preprocessed {len(patient_data)} train / "
          f"{len(test_patient_data)} test patients")

    train_dataset = XAugDataset.from_processed(patient_data)
    test_dataset = XAugDataset.from_processed(test_patient_data)

    cov_means, cov_stds = compute_cov_stats(
        train_dataset, list(range(len(train_dataset))), K)
    for k, feat in enumerate(cfg.time_varying_features):
        print(f"    {feat:>6s}: mean={cov_means[k]:.4f}, std={cov_stds[k]:.4f}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, collate_fn=collate_xaug)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size,
                             shuffle=False, collate_fn=collate_xaug)
    return Bundle(train_loader, test_loader, cov_means, cov_stds,
                  K, Ks)


# Data-source registry. Add new loaders here; the training core is unchanged.
_SIM_KINDS = {"rds_sim"}
_REAL_KINDS = {"csv_real"}


# ══════════════════════════════════════════════════════════════════════════
# Model / optimiser
# ══════════════════════════════════════════════════════════════════════════

def build_model(cfg: TrainConfig, bundle: Bundle, device):
    skip_dims = (cfg.static_skip_dims if cfg.static_skip_dims is not None
                 else list(range(bundle.Ks)))
    model = NeuralODEModel(
        n_tv=bundle.K,
        static_dim=bundle.Ks,
        cfg=cfg.cfg,
        use_rho_net=cfg.use_rho_net,
        use_neural_re=cfg.use_neural_re,
        g_hidden=cfg.g_hidden,
        fullD=cfg.fullD,
        cov_means=bundle.cov_means.tolist(),
        cov_stds=bundle.cov_stds.tolist(),
        static_skip_dims=skip_dims,
        use_dynamic_skip=cfg.use_dynamic_skip,
        reg_mode=cfg.reg_mode,
        include_mask=cfg.include_mask,
        use_dynamic_in_ode=cfg.use_dynamic_in_ode,
        encode_baseline_cov=cfg.encode_baseline_cov,
    ).to(device)
    return model


def build_optimizer(model, cfg: TrainConfig):
    """AdamW with the group-lasso first-layer weight excluded from weight decay
    (it already carries the group-lasso penalty; double shrinkage otherwise).
    Variance and fixed-effect parameters are also decay-free."""
    gl_weight = model.decoder.rho_net.net[0].weight
    gl_ids = {id(gl_weight)}

    nn_weights, var_params, fe_params = [], [], []
    for n, p in model.named_parameters():
        if id(p) in gl_ids:
            continue
        elif "log_residual_var" in n or "log_std" in n:
            var_params.append(p)
        elif "beta_neural" in n:
            fe_params.append(p)
        else:
            nn_weights.append(p)

    return torch.optim.AdamW([
        {"params": nn_weights, "weight_decay": cfg.wd},
        {"params": [gl_weight], "weight_decay": 0.0},
        {"params": var_params, "weight_decay": 0.0},
        {"params": fe_params,  "weight_decay": 0.0},
    ], lr=cfg.lr)


# ══════════════════════════════════════════════════════════════════════════
# Training core (source-agnostic)
# ══════════════════════════════════════════════════════════════════════════

def run_training(cfg: TrainConfig, bundle: Bundle, tag, device):
    model = build_model(cfg, bundle, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params}")
    print(f"  flags: include_mask={cfg.include_mask}, "
          f"use_dynamic_in_ode={cfg.use_dynamic_in_ode}, "
          f"encode_baseline_cov={cfg.encode_baseline_cov}")
    print(f"  reg_mode={cfg.reg_mode}, lambda={cfg.lambda_reg}, "
          f"solver={cfg.cfg.ode_solver}")

    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg.sched_patience)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    reg = cfg.reg_mode if cfg.reg_mode is not None else "noreg"
    ckpt_name = cfg.ckpt_template.format(tag=tag, lam=cfg.lambda_reg, reg=reg,
                                         name=cfg.name)
    ckpt_path = os.path.join(cfg.ckpt_dir, ckpt_name)

    feat_names = cfg.time_varying_features + cfg.static_features

    best_test = float("inf")
    best_state = None
    patience = cfg.patience_init

    for epoch in range(1, cfg.epochs + 1):
        if patience == 0:
            break

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total_nll, count = 0.0, 0
        reg_dict = {}
        for batch in bundle.train_loader:
            x_aug, static, y, obs_mask = unpack_batch(batch, device)
            mu, V, _, _, _, reg_dict = model(
                x_aug, static_covariates=static, obs_mask=obs_mask)
            loss = masked_NLL(mu, y, V, obs_mask)
            if reg_dict and "reg_term" in reg_dict:
                loss = loss + cfg.lambda_reg * reg_dict["reg_term"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_nll += loss.item()
            count += 1
        avg_train = total_nll / max(count, 1)

        # ── Test ──────────────────────────────────────────────────────────
        model.eval()
        test_nll, test_count = 0.0, 0
        with torch.no_grad():
            for batch in bundle.test_loader:
                x_aug, static, y, obs_mask = unpack_batch(batch, device)
                mu, V, _, _, _, _ = model(
                    x_aug, static_covariates=static, obs_mask=obs_mask)
                test_nll += masked_NLL(mu, y, V, obs_mask).item()
                test_count += 1
        avg_test = test_nll / max(test_count, 1)
        scheduler.step(avg_test)

        # ── Checkpoint on improvement ─────────────────────────────────────
        if avg_test < best_test:
            best_test = avg_test
            patience = cfg.patience_reset
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({
                "model_state_dict": best_state,
                "best_test_loss": best_test,
                "epoch": epoch,
                "cov_means": bundle.cov_means.tolist(),
                "cov_stds": bundle.cov_stds.tolist(),
                "name": cfg.name,
                "train_config": cfg.to_dict(),
            }, ckpt_path)
        else:
            patience -= 1

        # ── Log ───────────────────────────────────────────────────────────
        if epoch % cfg.print_every == 0 or epoch == 1:
            _log_epoch(model, cfg, epoch, avg_train, avg_test, best_test,
                       reg_dict, feat_names, device)

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    print(f"  ✓ best test NLL = {best_test:.4f}  →  {ckpt_path}")
    return model, best_test


def _log_epoch(model, cfg, epoch, avg_train, avg_test, best_test,
               reg_dict, feat_names, device):
    beta = model.decoder.beta_neural.detach().cpu()
    sig2 = torch.exp(model.decoder.log_residual_var).item()
    D = model.decoder._build_D(device, torch.float32).detach().cpu()
    print(f"\n  Epoch {epoch:5d} | train NLL = {avg_train:.4f} | "
          f"test NLL = {avg_test:.4f} | best = {best_test:.4f}")
    print(f"      sigma2 = {sig2:.4f}")
    print(f"      D diag = {[round(v, 4) for v in D.diag().tolist()]}")
    print(f"      beta   = {[f'{v:.4f}' for v in beta.tolist()]}")
    if cfg.reg_mode == "skip_gate" and "gate_values" in reg_dict:
        gv = reg_dict["gate_values"]
        print("      gates:")
        for g, name in enumerate(feat_names[:len(gv)]):
            print(f"        {name:>8s}: {gv[g]:.4f}")
    elif cfg.reg_mode == "group_lasso" and "group_norms" in reg_dict:
        gn = reg_dict["group_norms"]
        print("      group norms:")
        for g, name in enumerate(feat_names[:len(gn)]):
            print(f"        {name:>8s}: {gn[g]:.4f}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="Generic config-driven Neural ODE-LMM trainer")
    parser.add_argument("--config", required=True,
                        help="path to a JSON (or YAML) config file")
    parser.add_argument("--data", default=None,
                        help="override data_dir (sim) or train_csv (real)")
    parser.add_argument("--ckpt", default=None,
                        help="override checkpoint directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data:
        if cfg.data_kind in _SIM_KINDS:
            cfg.data_dir = args.data
        else:
            cfg.train_csv = args.data
    if args.ckpt:
        cfg.ckpt_dir = args.ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"NEURAL ODE-LMM — config: {cfg.name}  (device: {device})")
    print("=" * 60)

    if cfg.data_kind in _SIM_KINDS:
        for sim_index in range(cfg.sim_start, cfg.sim_end):
            print(f"\n── replicate {sim_index} ({cfg.data_dir}) ─────────────")
            set_seed(cfg.split_seed)
            bundle = prepare_sim(cfg, sim_index)
            run_training(cfg, bundle, tag=str(sim_index), device=device)
    elif cfg.data_kind in _REAL_KINDS:
        for seed in cfg.seeds:
            print(f"\n── seed {seed} ──────────────────────────────────")
            set_seed(seed)
            bundle = prepare_real(cfg)
            run_training(cfg, bundle, tag=str(seed), device=device)
    else:
        raise ValueError(
            f"Unknown data_kind {cfg.data_kind!r}; "
            f"expected one of {sorted(_SIM_KINDS | _REAL_KINDS)}")


if __name__ == "__main__":
    main()