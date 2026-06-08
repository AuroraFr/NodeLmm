"""
Neural ODE-LMM training on the real 3C cohort dataset.

Architecture (refactored multi-covariate version):
  - Encoder:  z(0) = Enc(t0, x_baseline[K], static[Ks])
  - Dynamics: dz/dt = f(z, x_interp(t), mask(t), t)   ← no static in ODE
  - Skip:     gate ⊙ [x_interp_std(t), mask(t), static_skip]
  - Decoder:  mu(t) = rho(z(t), skip(t))^T beta
  - RE:       Z = g(z(t))  (neural random effects)

Input: x_aug from Preprocess_3C.py with layout [time(1), x_interp(K), mask(K)].
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import numpy as np
import pandas as pd
from Preprocess_3C import process_data, EXPECTED_TIMES
from model_ODE_real import NeuralODEModel, NeuralODEConfig
from utils import masked_NLL

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ── Dataset wrapper ──────────────────────────────────────────────────────────

class RealDataset(Dataset):
    """
    Wraps the list of dicts from process_data into a PyTorch Dataset.

    Returns x_aug directly — the model unpacks it internally.
    """
    def __init__(self, patient_data_list):
        self.data = patient_data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return (
            d["patient_id"],
            d["x_aug"],            # (T, 1+2K)
            d["y"],                # (T,)
            d["target_mask"],      # (T,)
            d["s_i"],              # (Ks,)
        )


def collate_real(batch):
    """
    Collate for RealDataset. All patients share the canonical grid (T=6),
    so no padding needed — just stack.
    """
    pids, x_augs, ys, masks, statics = zip(*batch)

    return (
        list(pids),
        torch.stack(x_augs),      # (N, T, 1+2K)
        torch.stack(ys),           # (N, T)
        torch.stack(masks),        # (N, T)
        torch.stack(statics),      # (N, Ks)
    )


# ── Covariate statistics (training set only) ─────────────────────────────────

def compute_covariate_stats(dataset, train_idx, n_tv):
    """
    Compute per-channel mean and std from training-set observed values only.

    Uses the covariate-level mask (columns K+1..2K in x_aug) to identify
    truly observed values, NOT the outcome observation mask.

    Args:
        dataset:   RealDataset instance
        train_idx: array of training subject indices
        n_tv:      number of time-varying covariates (K)

    Returns:
        cov_means: (K,) tensor
        cov_stds:  (K,) tensor
    """
    K = n_tv
    vals = {k: [] for k in range(K)}

    for idx in train_idx:
        _, x_aug, _, _, _ = dataset[idx]
        # x_aug layout: [time(1), x_interp(K), mask(K)]
        x_interp = x_aug[:, 1:1+K]           # (T, K)
        cov_mask = x_aug[:, 1+K:1+2*K]       # (T, K)

        for k in range(K):
            obs_k = cov_mask[:, k] > 0.5      # truly observed at this slot
            if obs_k.any():
                vals[k].append(x_interp[obs_k, k])

    cov_means = torch.zeros(K)
    cov_stds = torch.zeros(K)
    for k in range(K):
        v = torch.cat(vals[k])
        cov_means[k] = v.mean()
        cov_stds[k] = v.std().clamp(min=1e-6)

    return cov_means, cov_stds


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEEDS = [10, 200, 300, 400, 500, 600]
    # ── Config ──────────────────────────────────────────────────────────
    for SEED  in SEEDS:
        LR = 1e-3
        WD = 1e-5
        EPOCHS = 1000
        BATCH_SIZE = 128
        PRINT_EVERY = 25
        # SEED = 42
        LAMBDA_REG = 0.1
        INTERP_METHOD = "linear"      # "ffill", "linear", or "cubic"
        MASK_TYPE = "binary"           # "binary" or "cumulative"
        REG_MODE = None        # None, "skip_gate", or "group_lasso"
        use_dynamic_skip = False

        print("=" * 60)
        print("NEURAL ODE-LMM — REAL 3C DATASET")
        print("=" * 60)

        # ── Feature definitions ─────────────────────────────────────────────
        id_col = "NUM_ID"
        target_col = "ISA15"

        time_varying_features = ["BMI", "PAS", "PAD", "GLUC", "HDL"]
        K = len(time_varying_features)

        static_features = ["SEX_code", "AGEc", "DIPNIV_2", "DIPNIV_3"]
        Ks = len(static_features)

        # ── Load and preprocess data ────────────────────────────────────────
        print(f"\nLoading 3C dataset...")
        df = pd.read_csv("3C_dataset/train_3C_data_1.csv")
        test_df = pd.read_csv("3C_dataset/val_3C_data_1.csv")

        if "AGEc" not in df.columns:
            all_df = pd.read_csv("3C_dataset/data_3C.csv")
            baseline_age = all_df.groupby(id_col)["AGE0"].transform("first")
            baseline_age_mean = baseline_age.mean()
            df["AGEc"] = df.groupby(id_col)["AGE0"].transform("first") - baseline_age_mean
            test_df["AGEc"] = test_df.groupby(id_col)["AGE0"].transform("first") - baseline_age_mean

        patient_data = process_data(
            df=df,
            id_col=id_col,
            time_varying_features=time_varying_features,
            static_features=static_features,
            target_col=target_col,
            interp_method=INTERP_METHOD,
            mask_type=MASK_TYPE,
        )
        print(f"Preprocessed {len(patient_data)} patients")
        print(f"x_aug shape: {patient_data[0]['x_aug'].shape}  "
            f"= [time(1), x_interp({K}), mask({K})]")
        print(f"Interpolation: {INTERP_METHOD}, mask: {MASK_TYPE}")

        test_patient_data = process_data(df=test_df,
            id_col=id_col,
            time_varying_features=time_varying_features,
            static_features=static_features,
            target_col=target_col,
            interp_method=INTERP_METHOD,
            mask_type=MASK_TYPE,)

        # ── Dataset ─────────────────────────────────────────────────────────
        train_dataset = RealDataset(patient_data)
        test_dataset = RealDataset(test_patient_data)

        # # ── Train/test split (subject level) ────────────────────────────────
        # N = len(full_dataset)
        # rng = np.random.RandomState(SEED)
        # indices = rng.permutation(N)
        # n_test = int(N * TEST_RATIO)
        # test_idx = indices[:n_test]
        # train_idx = indices[n_test:]

        # train_dataset = Subset(full_dataset, train_idx)
        # test_dataset = Subset(full_dataset, test_idx)

        # ── Covariate statistics from training set ──────────────────────────
        cov_means, cov_stds = compute_covariate_stats(train_dataset, list(range(len(train_dataset))), K)

        print(f"\nCovariate stats (train set, truly observed values only):")
        for k, feat in enumerate(time_varying_features):
            print(f"  {feat:>6s}: mean={cov_means[k]:.4f}, std={cov_stds[k]:.4f}")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=collate_real)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, collate_fn=collate_real)
        print(f"Subjects size: train : {len(train_dataset)}, test: {len(test_dataset)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        np.random.seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # ── Model ───────────────────────────────────────────────────────────
        cfg = NeuralODEConfig(
            hidden_channels=8,
            enc_mlp_hidden=16,
            func_mlp_hidden=16,
            dec_rho_hidden=16,
            dec_p=4,
            dec_q=3,
            depth=2,
            dropout=0.0,
            euler_steps_per_interval=4,
            ode_solver='rk4',
            use_rho_norm=True
        )

        model = NeuralODEModel(
            n_tv=K,
            static_dim=Ks,
            cfg=cfg,
            use_rho_net=True,
            use_neural_re=True,
            g_hidden=8,
            fullD=False,
            cov_means=cov_means.tolist(),
            cov_stds=cov_stds.tolist(),
            static_skip_dims=list(range(Ks)),     # all statics in skip
            # static_skip_dims=None,
            use_dynamic_skip=use_dynamic_skip,                 # all K covariates in skip
            reg_mode=REG_MODE,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params}")
        print(f"Architecture:")
        print(f"  Encoder:  z(0) = Enc(t0, x_baseline[{K}], static[{Ks}])")
        print(f"  ODE:      dz/dt = f(z, x_interp[{K}], mask[{K}], t)")
        print(f"  Skip:     gate ⊙ [x_std[{K}], mask[{K}], static[{Ks}]]")
        print(f"  Decoder:  mu = rho(z(t), skip) @ beta[{cfg.dec_p}]")
        print(f"  RE:       Z = g(z(t))  (neural, q={cfg.dec_q})")
        print(f"  ODE solver: {cfg.ode_solver}, sub-steps: {cfg.euler_steps_per_interval}")
        print(f"  Reg mode: {REG_MODE}, lambda: {LAMBDA_REG}")

        # ── Optimiser ───────────────────────────────────────────────────────
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
        optimizer = torch.optim.AdamW([
            {'params': nn_weights, 'weight_decay': WD},
            {'params': gate_params, 'weight_decay': 0.0},
            {'params': var_params, 'weight_decay': 0.0},
            {'params': fe_params,  'weight_decay': 0.0},  # or a small value if desired
        ])
        # optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )

        # ── Checkpoint ──────────────────────────────────────────────────────
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = "checkpoints/best_model_ode_real_3C_regnone_H8_seed"+str(SEED)+".pt"

        # ── Training loop ───────────────────────────────────────────────────
        best_test_loss = float("inf")
        best_state = None

        for epoch in range(1, EPOCHS + 1):

            # ── Train ───────────────────────────────────────────────────────
            model.train()
            total_nll = 0.0
            count = 0

            for batch in train_loader:
                pids, x_aug, y_pad, target_mask, static = batch
                x_aug = x_aug.to(device)
                y_pad = y_pad.to(device)
                target_mask = target_mask.to(device)
                static = static.to(device)

                mu, V, Z, D, sig2, reg_dict = model(
                    x_aug,
                    static_covariates=static,
                    obs_mask=target_mask,
                )

                loss = masked_NLL(mu, y_pad, V, target_mask)

                if reg_dict and "reg_term" in reg_dict:
                    loss = loss + LAMBDA_REG * reg_dict["reg_term"]

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_nll += loss.item()
                count += 1

            avg_train = total_nll / max(count, 1)

            # ── Test ────────────────────────────────────────────────────────
            model.eval()
            test_nll = 0.0
            test_count = 0

            with torch.no_grad():
                for batch in test_loader:
                    pids, x_aug, y_pad, target_mask, static = batch
                    x_aug = x_aug.to(device)
                    y_pad = y_pad.to(device)
                    target_mask = target_mask.to(device)
                    static = static.to(device)

                    mu, V, Z, D, sig2, reg_dict = model(
                        x_aug,
                        static_covariates=static,
                        obs_mask=target_mask,
                    )
                    loss = masked_NLL(mu, y_pad, V, target_mask)
                    test_nll += loss.item()
                    test_count += 1

            avg_test = test_nll / max(test_count, 1)
            scheduler.step(avg_test)

            if avg_test < best_test_loss:
                best_test_loss = avg_test
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                torch.save({
                    'model_state_dict': best_state,
                    'best_test_loss': best_test_loss,
                    'epoch': epoch,
                    'cov_means': cov_means.tolist(),
                    'cov_stds': cov_stds.tolist(),
                    'config': {
                        'hidden_channels': cfg.hidden_channels,
                        'func_mlp_hidden': cfg.func_mlp_hidden,
                        'dec_rho_hidden': cfg.dec_rho_hidden,
                        'dec_p': cfg.dec_p,
                        'dec_q': cfg.dec_q,
                        'depth': cfg.depth,
                        'euler_steps': cfg.euler_steps_per_interval,
                        'ode_solver': cfg.ode_solver,
                        'n_tv': K,
                        'static_dim': Ks,
                        'time_varying_features': time_varying_features,
                        'static_features': static_features,
                        'interp_method': INTERP_METHOD,
                        'mask_type': MASK_TYPE,
                        'reg_mode': REG_MODE,
                        'lambda_reg': LAMBDA_REG,
                        'static_skip_dims': list(range(Ks)),
                        'use_dynamic_skip': use_dynamic_skip,
                        'use_rho_norm': cfg.use_rho_norm
                    },
                }, ckpt_path)

            if epoch % PRINT_EVERY == 0 or epoch == 1:
                beta = model.decoder.beta_neural.detach().cpu()
                sig2_val = torch.exp(model.decoder.log_residual_var).item()
                D_val = model.decoder._build_D(device, torch.float32).detach().cpu()

                print(f"\nEpoch {epoch:5d} | train NLL = {avg_train:.4f} | "
                    f"test NLL = {avg_test:.4f} | best = {best_test_loss:.4f}")
                print(f"    σ² = {sig2_val:.4f}")
                print(f"    D = {D_val.diag().tolist()}"
                    if not model.decoder.fullD
                    else f"    D diag = {D_val.diag().tolist()}")
                print(f"    β = {[f'{v:.4f}' for v in beta.tolist()]}")

                # Print regularisation info
                if REG_MODE == "skip_gate" and "gate_values" in reg_dict:
                    gv = reg_dict["gate_values"]
                    print(gv)
                    names = time_varying_features + static_features
                    # names =  static_features
                    print(f"    gates:")
                    for g, name in enumerate(names):
                        print(f"      {name:>8s}: {gv[g]:.4f}")
                elif REG_MODE == "group_lasso" and "group_norms" in reg_dict:
                    gn = reg_dict["group_norms"]
                    names = time_varying_features + static_features
                    # names =  static_features
                    print(f"    group norms:")
                    for g, name in enumerate(names):
                        print(f"      {name:>8s}: {gn[g]:.4f}")

        # ── Restore best ────────────────────────────────────────────────────
        if best_state is not None:
            model.load_state_dict(best_state, strict=False)

        # ── Final summary ───────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("FINAL RESULTS (Neural ODE-LMM — Real 3C)")
        print("=" * 60)
        print(f"  Best test NLL = {best_test_loss:.4f}")
        print(f"  β = {model.decoder.beta_neural.detach().cpu().tolist()}")
        print(f"  σ² = {torch.exp(model.decoder.log_residual_var).item():.4f}")
        D_final = model.decoder._build_D(device, torch.float32).detach().cpu()
        print(f"  D (RE covariance):")
        for r in range(D_final.shape[0]):
            print(f"    [{', '.join(f'{D_final[r,c]:.4f}' for c in range(D_final.shape[1]))}]")

        if REG_MODE == "skip_gate":
            gates = torch.sigmoid(model.decoder.skip_gate_logit).detach().cpu()
            names = time_varying_features + static_features
            print(f"  Final gate values:")
            for g, name in enumerate(names):
                print(f"    {name:>8s}: {gates[g]:.4f}")