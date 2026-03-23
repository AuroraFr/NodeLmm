"""
Neural ODE-LMM training on the real 3C cohort dataset.

Architecture:
  - Encoder:  z(0) = Enc(t0, x0_all, static)
  - Dynamics: dz/dt = f(z, t, x_filled(t), cumask(t), static)
  - Decoder:  mu(t) = rho(z(t), BMI_std(t), AGEc) @ beta_neural
  - RE:       Z = g(z(t))  (neural random effects)

All K time-varying covariates + K cumulative masks are injected into the ODE.
BMI skip connection in the decoder is optional.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import numpy as np
import pandas as pd
from Preprocess_3C import process_data
from model_ODE_real import NeuralODEModel, NeuralODEConfig
from utils import masked_NLL

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ── Dataset wrapper ──────────────────────────────────────────────────────────

class RealDataset(Dataset):
    """
    Wraps the list of dicts from process_data into a proper PyTorch Dataset.

    Each sample returns:
        patient_id, t, x_filled, cumask, y, target_mask, static
    """
    def __init__(self, patient_data_list, n_tv):
        """
        Args:
            patient_data_list: output of process_data()
            n_tv: number of time-varying features (K).
                  x_aug layout: [time(1), x_filled(K), cumask(K)]
        """
        self.data = patient_data_list
        self.n_tv = n_tv

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        x_aug = d["x_aug"]           # (T, 1 + K + K)
        t = d["t"]                   # (T,)
        y = d["y"]                   # (T,)
        target_mask = d["target_mask"]  # (T,)
        s_i = d["s_i"]              # (S,)
        pid = d["patient_id"]

        K = self.n_tv
        # Split x_aug into components
        # col 0: time (redundant with t, but kept for consistency)
        x_filled = x_aug[:, 1:1+K]         # (T, K) forward-filled covariates
        cumask = x_aug[:, 1+K:1+2*K]       # (T, K) cumulative observation mask

        return pid, t, x_filled, cumask, y, target_mask, s_i


def collate_real(batch):
    """
    Collate for RealDataset. All patients share the same canonical grid (T=6),
    so no padding is needed — just stack.
    """
    pids, ts, x_filleds, cumasks, ys, masks, statics = zip(*batch)

    return (
        torch.stack(pids) if isinstance(pids[0], torch.Tensor) else list(pids),
        torch.stack(ts),           # (N, T)
        torch.stack(x_filleds),    # (N, T, K)
        torch.stack(cumasks),      # (N, T, K)
        torch.stack(ys),           # (N, T)
        torch.stack(masks),        # (N, T)
        torch.stack(statics),      # (N, S)
    )


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ---- Config ----
    LR = 1e-3
    WD = 1e-5
    EPOCHS = 1000
    BATCH_SIZE = 128
    PRINT_EVERY = 25
    TEST_RATIO = 0.2
    SEED = 42
    LAMBDA_ORTHO = 0.1

    print("=" * 60)
    print("NEURAL ODE-LMM — REAL 3C DATASET")
    print("=" * 60)

    # ---- Feature definitions ----
    id_col = "NUM_ID"
    target_col = "ISA15"

    # Time-varying covariates injected into ODE dynamics
    time_varying_features = ["BMI", "PAS", "PAD", "GLUC", "HDL"]
    K = len(time_varying_features)

    # Static covariates
    static_features = ["SEX_code", "AGEc", "DIPNIV_2", "DIPNIV_3"]
    S = len(static_features)

    # BMI column index within x_filled (for skip connection)
    # BMI column index (for reference in analysis, not used in model)
    BMI_COL = 0

    # ---- Load and preprocess data ----
    print("\nLoading 3C dataset...")
    df = pd.read_csv("3C_dataset/train_3C_data.csv")  # adapt path to your data

    # Compute centered age at baseline
    if "AGEc" not in df.columns:
        baseline_age = df.groupby(id_col)["AGE0"].transform("first")
        df["AGEc"] = baseline_age - baseline_age.mean()

    patient_data = process_data(
        df=df,
        id_col=id_col,
        time_varying_features=time_varying_features,
        static_features=static_features,
        target_col=target_col,
    )
    print(f"Preprocessed {len(patient_data)} patients")
    print(f"x_aug shape per patient: {patient_data[0]['x_aug'].shape}")
    print(f"  = [time(1), x_filled({K}), cumask({K})]")

    # ---- Build dataset ----
    full_dataset = RealDataset(patient_data, n_tv=K)

    # ---- Train/test split (subject level) ----
    N = len(full_dataset)
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(N)
    n_test = int(N * TEST_RATIO)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # ---- Covariate stats from training set only ----
    # Collect all observed values per channel for standardization
    train_vals = {k: [] for k in range(K)}
    for idx in train_idx:
        sample = full_dataset[idx]
        x_filled = sample[2]           # (T, K)
        mask = sample[5]               # (T,)
        obs = mask > 0.5
        for k in range(K):
            train_vals[k].append(x_filled[obs, k])

    # Per-channel mean and std (from training set only)
    x_mean = torch.zeros(K)
    x_std = torch.zeros(K)
    for k in range(K):
        vals = torch.cat(train_vals[k])
        x_mean[k] = vals.mean()
        x_std[k] = vals.std().clamp(min=1e-6)

    print(f"\nCovariate stats (train set, observed values only):")
    for k, feat in enumerate(time_varying_features):
        print(f"  {feat:>6s}: mean={x_mean[k]:.4f}, std={x_std[k]:.4f}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_real)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_real)
    print(f"Subjects: {N} total → {len(train_idx)} train, {len(test_idx)} test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model ----
    # n_ode_inject = K covariates + K cumulative masks
    n_ode_inject = 2 * K

    cfg = NeuralODEConfig(
        hidden_channels=8,
        enc_mlp_hidden=32,
        func_mlp_hidden=32,
        dec_rho_hidden=16,
        dec_p=4,
        dec_q=3,
        depth=2,
        dropout=0.0,
        euler_steps_per_interval=4,
    )

    model = NeuralODEModel(
        x_dim=K,                    # number of time-varying features
        static_dim=S,
        cfg=cfg,
        n_tv=K,                     # all K baseline covariates go to encoder
        n_ode_inject=n_ode_inject,  # K covariates + K masks into ODE dynamics
        use_rho_net=True,
        use_neural_re=True,         # neural RE (no spline columns in real data)
        re_spline_cols=None,
        g_hidden=16,
        fullD=True,
        x_mean=x_mean,
        x_std=x_std,
        skip_covariate_cols=list(range(K)),   # all time-varying covariates skip to decoder
        static_skip_dims=list(range(S)),      # all static covariates skip to decoder
        reg_mode="l1_skip",
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    print(f"Architecture:")
    print(f"  Encoder:  z(0) = Enc(t0, x0_all[{K}], static[{S}])")
    print(f"  ODE:      dz/dt = f(z, t, x_filled[{K}], cumask[{K}], static)")
    print(f"  Decoder:  mu = rho(z(t), x_skip_std[{K}], static_skip[{S}]) @ beta")
    print(f"  RE:       Z = g(z(t))  (neural)")
    print(f"  Euler sub-steps: {cfg.euler_steps_per_interval}")
    print(f"  Skip covariates: {time_varying_features}")
    print(f"  Skip statics:    {static_features}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )

    # ---- Checkpoint ----
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/best_model_ode_real_3C.pt"

    # ---- Training loop ----
    best_test_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):

        # ---- Train ----
        model.train()
        total_nll = 0.0
        count = 0

        for batch in train_loader:
            _, t_pad, x_filled, cumask, y_pad, mask, s = batch
            t_pad = t_pad.to(device)
            x_filled = x_filled.to(device)      # (N, T, K)
            cumask = cumask.to(device)           # (N, T, K)
            y_pad = y_pad.to(device)
            mask = mask.to(device)
            s = s.to(device)

            # Concatenate covariates + cumulative masks for ODE injection
            ode_inject = torch.cat([x_filled, cumask], dim=-1)  # (N, T, 2K)

            mu, V, _, _, _, reg_dict = model(
                t_pad, x_filled, masks=cumask,
                static_covariates=s,
                ode_inject=ode_inject,
                obs_mask=mask,
                y_pad=None,
            )
            loss = masked_NLL(mu, y_pad, V, mask)

            if reg_dict and "reg_term" in reg_dict:
                loss = loss + LAMBDA_ORTHO * reg_dict["reg_term"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_nll += loss.item()
            count += 1

        avg_train = total_nll / max(count, 1)

        # ---- Test ----
        model.eval()
        test_nll = 0.0
        test_count = 0

        with torch.no_grad():
            for batch in test_loader:
                _, t_pad, x_filled, cumask, y_pad, mask, s = batch
                t_pad = t_pad.to(device)
                x_filled = x_filled.to(device)
                cumask = cumask.to(device)
                y_pad = y_pad.to(device)
                mask = mask.to(device)
                s = s.to(device)

                ode_inject = torch.cat([x_filled, cumask], dim=-1)

                mu, V, _, _, _, reg_dict = model(
                    t_pad, x_filled, masks=cumask,
                    static_covariates=s,
                    ode_inject=ode_inject,
                    obs_mask=mask,
                    y_pad=None,
                )
                loss = masked_NLL(mu, y_pad, V, mask)
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
                'train_idx': train_idx,
                'test_idx': test_idx,
                'x_mean': x_mean.tolist(),
                'x_std': x_std.tolist(),
                'time_varying_features': time_varying_features,
                'config': {
                    'hidden_channels': cfg.hidden_channels,
                    'euler_steps': cfg.euler_steps_per_interval,
                    'n_ode_inject': n_ode_inject,
                    'time_varying_features': time_varying_features,
                    'static_features': static_features,
                    'use_rho_net': True,
                    'use_neural_re': True,
                    'skip_covariate_cols': list(range(K)),
                    'static_skip_dims': list(range(S)),
                    'approach': 'ode_real_3C',
                },
            }, ckpt_path)

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            beta_neural = model.decoder.beta_neural.detach().cpu()
            sig2 = torch.exp(model.decoder.log_residual_var).item()
            D = model.decoder._build_D(device,
                    model.decoder.log_residual_var.dtype).detach().cpu()

            print(f"\nEpoch {epoch:5d} | train NLL = {avg_train:.4f} | "
                  f"test NLL = {avg_test:.4f} | best test = {best_test_loss:.4f}")
            print(f"    sigma2 = {sig2:.4f}")
            print(f"    D:")
            for r in range(D.shape[0]):
                print(f"      [{', '.join(f'{D[r,c]:.4f}' for c in range(D.shape[1]))}]")
            print(f"    beta_neural = [{', '.join(f'{v:.4f}' for v in beta_neural)}]")

            if reg_dict and "skip_contrib_mean" in reg_dict:
                print(f"    skip_contrib_mean = {reg_dict['skip_contrib_mean']:.4f}")

    # ---- Restore best ----
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # ---- Final summary ----
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Neural ODE — Real 3C)")
    print("=" * 60)
    print(f"  best test NLL = {best_test_loss:.4f}")
    print(f"  beta_neural = {model.decoder.beta_neural.detach().cpu().tolist()}")
    print(f"  sigma2 = {torch.exp(model.decoder.log_residual_var).item():.4f}")
    D_final = model.decoder._build_D(device,
                model.decoder.log_residual_var.dtype).detach().cpu()
    print(f"  D (RE covariance):")
    for r in range(D_final.shape[0]):
        print(f"    [{', '.join(f'{D_final[r,c]:.4f}' for c in range(D_final.shape[1]))}]")