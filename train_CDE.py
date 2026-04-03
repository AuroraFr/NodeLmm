"""
Neural CDE-LMM training for Scenario 6: Rate-of-change sensitivity.

DGP: h_6(t) = alpha * integral_0^t |dBMI/dtau| dtau
     with BMI volatility correlated with |BMI_0 - 25| (OU process)
     and beta_BMI = 0 (no direct level effect)

Configuration:
  - x_cols = ["BMI_t"] — CDE sees BMI through dX
  - tv_skip_cols = None — no skip (pure latent)
  - use_neural_re = True — learned RE basis g(z(t))
  - Oracle DeltaPDP = 0 under constant interventions
"""
import torch
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
import pyreadr
from dataset import LongitudinalDataset, collate_pad
from model_CDE import NeuralCDEModel, NeuralCDEConfig, probe_latent_space
from utils import masked_NLL

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":

    # ── Config ──
    LR = 1e-3
    WD = 1e-5
    EPOCHS = 500
    BATCH_SIZE = 128
    PRINT_EVERY = 25
    TEST_RATIO = 0.2
    SEED = 42

    # S6 DGP parameters (for logging)
    ALPHA_6 = -0.15
    SIGMA_BASE = 0.8
    GAMMA = 0.3
    THETA = 0.5
    n_tv = 1

    print("=" * 60)
    print("SCENARIO 6: RATE-OF-CHANGE SENSITIVITY (CDE)")
    print(f"  h_6(t) = {ALPHA_6} * integral |dBMI/dt| dt")
    print(f"  OU volatility: sigma_base={SIGMA_BASE}, gamma={GAMMA}, theta={THETA}")
    print(f"  Oracle DeltaPDP = 0 under constant interventions")
    print("=" * 60)

    # ── Data ──
    time_col = "time"
    y_col = "ISA15_sim"
    id_col = "NUM_ID"
    x_cols = ["BMI_t"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    path = "simu_datasets/S6_sims/sim_001.rds"
    df = next(iter(pyreadr.read_r(path).values()))
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    full_dataset = LongitudinalDataset(
        df, id_col, time_col, x_cols, y_col, static_cols=static_cols
    )
    print(f"Loaded: {len(full_dataset)} subjects")

    # ── Train/test split ──
    N = len(full_dataset)
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(N)
    n_test = int(N * TEST_RATIO)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # BMI stats from training set
    train_bmis = []
    for idx in train_idx:
        sample = full_dataset[idx]
        train_bmis.append(sample[2][:, 0])   # (T,) — BMI_t
    train_bmis = torch.cat(train_bmis)
    bmi_mean = train_bmis.mean().item()
    bmi_std = train_bmis.std().item()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_pad)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_pad)

    print(f"Split: {len(train_idx)} train, {len(test_idx)} test")
    print(f"BMI (train): mean={bmi_mean:.2f}, std={bmi_std:.2f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_tv = 1

    # ── Model ──
    cfg = NeuralCDEConfig(
        hidden_channels=8,
        enc_mlp_hidden=32,
        func_mlp_hidden=32,
        dec_rho_hidden=16,
        dec_p=4,
        dec_q=3,
        depth=2,
        dropout=0.0,
    )

    model = NeuralCDEModel(
        x_dim=len(x_cols),
        static_dim=len(static_cols),
        cfg=cfg,
        n_tv=n_tv,
        use_rho_net=True,
        use_neural_re=True,
        g_hidden=16,
        re_spline_cols=None,
        # S6: no skip — CDE must learn volatility effect from dX alone
        tv_skip_cols=None,
        static_skip_dims=None,
        reg_mode=None,
        inject_x=False, augment_order=2,
        encoder_sees_covariates=False
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params} parameters")
    print(f"  CDE control: [time, BMI_t, cum_mask] (3 channels)")
    print(f"  Decoder FE: rho(z(t)) @ beta  (no skip)")
    print(f"  Decoder RE: g(z(t)) -> R^{cfg.dec_q}")
    model.decoder.describe_skip()

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )

    # ── Training ──
    os.makedirs("checkpoints", exist_ok=True)
    best_test_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_nll = 0.0
        count = 0

        for batch in train_loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch

            t_pad = t_pad.to(device)
            x_pad = x_pad.to(device)
            y_pad = y_pad.to(device)
            mask = mask.to(device)
            c_mask = c_mask.to(device)
            s = s.to(device)

            mu, V, Z, D, sig2, reg_dict = model(
                t_pad, x_pad, c_mask, s, mask, y_pad=None
            )
            loss = masked_NLL(mu, y_pad, V, mask)

            # Add regularization if configured
            if "reg_term" in reg_dict:
                reg_lambda = 0.1
                loss = loss + reg_lambda * reg_dict["reg_term"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_nll += loss.item()
            count += 1

        avg_train = total_nll / max(count, 1)

        # ── Test ──
        model.eval()
        test_nll = 0.0
        test_count = 0
        with torch.no_grad():
            for batch in test_loader:
                _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
                t_pad = t_pad.to(device)
                x_pad = x_pad.to(device)
                y_pad = y_pad.to(device)
                mask = mask.to(device)
                c_mask = c_mask.to(device)
                s = s.to(device)

                mu, V, Z, D, sig2, reg_dict = model(
                    t_pad, x_pad, c_mask, s, mask, y_pad=None
                )
                loss = masked_NLL(mu, y_pad, V, mask)
                test_nll += loss.item()
                test_count += 1

        avg_test = test_nll / max(test_count, 1)
        scheduler.step(avg_test)

        if avg_test < best_test_loss:
            best_test_loss = avg_test
            best_state = {
                k: v.clone() for k, v in model.state_dict().items()
                if 'X_spline' not in k
            }
            torch.save({
                'model_state_dict': best_state,
                'best_test_loss': best_test_loss,
                'train_idx': train_idx,
                'test_idx': test_idx,
                'bmi_mean': bmi_mean,
                'bmi_std': bmi_std,
                'config': {
                    'hidden_channels': cfg.hidden_channels,
                    'use_rho_net': True,
                    'use_neural_re': True,
                    'tv_skip_cols': None,
                    'static_skip_dims': None,
                    'reg_mode': None,
                    'approach': 'CDE_S6_rate_of_change',
                    'scenario': {
                        'alpha': ALPHA_6,
                        'sigma_base': SIGMA_BASE,
                        'gamma': GAMMA,
                        'theta': THETA,
                        'oracle_delta_pdp': 0.0,
                    },
                },
            }, 'checkpoints/best_CDE_S6.pt')

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            beta_neural = model.decoder.beta_neural.detach().cpu()
            sig2_val = torch.exp(model.decoder.log_residual_var).item()
            D_val = model.decoder._build_D(device, model.decoder.log_residual_var.dtype)
            D_val = D_val.detach().cpu()

            print(f"\nEpoch {epoch:5d} | train NLL = {avg_train:.4f} | "
                  f"test NLL = {avg_test:.4f} | best = {best_test_loss:.4f}")
            print(f"    sigma2 = {sig2_val:.4f}")
            print(f"    D =")
            for i in range(D_val.shape[0]):
                print(f"      [{', '.join(f'{D_val[i,j]:.4f}' for j in range(D_val.shape[1]))}]")
            print(f"    beta = [{', '.join(f'{v:.4f}' for v in beta_neural)}]")

    # ── Restore best ──
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # ── Latent space probe ──
    print("\n" + "=" * 60)
    print("LATENT SPACE PROBE")
    print("=" * 60)
    eval_loader = DataLoader(full_dataset, batch_size=256, shuffle=False,
                             collate_fn=collate_pad)
    probe_latent_space(model, eval_loader, device, x_cols, static_cols)

    # ── Final report ──
    print("\n" + "=" * 60)
    print("FINAL RESULTS (CDE — Scenario 6)")
    print("=" * 60)
    print(f"  Best test NLL: {best_test_loss:.4f}")
    print(f"  beta = {model.decoder.beta_neural.detach().cpu().tolist()}")
    print(f"  sigma2 = {torch.exp(model.decoder.log_residual_var).item():.4f}")
    D_final = model.decoder._build_D(device, model.decoder.log_residual_var.dtype)
    D_final = D_final.detach().cpu()
    print(f"  D =")
    for i in range(D_final.shape[0]):
        print(f"    [{', '.join(f'{D_final[i,j]:.4f}' for j in range(D_final.shape[1]))}]")
    model.decoder.describe_skip()

    # ── Debug: shifted must be identical ──
print("\n" + "=" * 60)
print("DEBUG: Shifted path identity check")
print("=" * 60)

from PDP_analysis_CDE import make_shifted_path

model.eval()
with torch.no_grad():
    batch = next(iter(eval_loader))
    _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
    t_pad = t_pad.to(device)
    x_pad = x_pad.to(device)
    mask = mask.to(device)
    c_mask = c_mask.to(device)
    s = s.to(device)

    x_20 = make_shifted_path(x_pad, mask, bmi_col=0, value=20.0)
    x_35 = make_shifted_path(x_pad, mask, bmi_col=0, value=35.0)

    # Check increments are identical
    d20 = x_20[:, 1:, 0] - x_20[:, :-1, 0]
    d35 = x_35[:, 1:, 0] - x_35[:, :-1, 0]
    print(f"  Max diff in BMI increments: {(d20 - d35).abs().max():.2e}")

    # Check raw BMI values differ
    print(f"  Max diff in BMI values:     {(x_20[:,:,0] - x_35[:,:,0]).abs().max():.2e}")

    # Run both through model
    mu_20, _, _, _, _, _ = model(t_pad, x_20, c_mask, s, mask, y_pad=None, interp="linear")
    mu_35, _, _, _, _, _ = model(t_pad, x_35, c_mask, s, mask, y_pad=None, interp="linear")

    diff_mu = (mu_20 - mu_35).abs()
    print(f"  Max diff in mu:             {diff_mu.max():.2e}")
    print(f"  Mean diff in mu:            {diff_mu[mask > 0.5].mean():.2e}")

    if diff_mu.max() > 1e-4:
        print("  ⚠ BUG: shifted paths should produce IDENTICAL outputs!")
        print("  Checking intermediate values...")

        # Check augmented paths
        from model_CDE import augment_path
        x_tv_20 = x_20[:, :, :1]
        x_tv_35 = x_35[:, :, :1]
        cm = c_mask.unsqueeze(-1) if c_mask.dim() == 2 else c_mask[:, :, :1]
        x_aug_20, _ = augment_path(x_tv_20, cm)
        x_aug_35, _ = augment_path(x_tv_35, cm)
        print(f"  Max diff in cum_abs:        {(x_aug_20[:,:,1] - x_aug_35[:,:,1]).abs().max():.2e}")
        print(f"  Max diff in cum_sq:         {(x_aug_20[:,:,2] - x_aug_35[:,:,2]).abs().max():.2e}")

        # Check X_in
        X_in_20 = torch.cat([t_pad[...,None], x_aug_20, cm.expand(-1,-1,3)], dim=-1)
        X_in_35 = torch.cat([t_pad[...,None], x_aug_35, cm.expand(-1,-1,3)], dim=-1)
        print(f"  Max diff in X_in:           {(X_in_20 - X_in_35).abs().max():.2e}")
        print(f"  Max diff in X_in col 0 (t): {(X_in_20[:,:,0] - X_in_35[:,:,0]).abs().max():.2e}")
        print(f"  Max diff in X_in col 1(BMI):{(X_in_20[:,:,1] - X_in_35[:,:,1]).abs().max():.2e}")
        print(f"  Max diff in X_in col 2(ca): {(X_in_20[:,:,2] - X_in_35[:,:,2]).abs().max():.2e}")
        print(f"  Max diff in X_in col 3(cs): {(X_in_20[:,:,3] - X_in_35[:,:,3]).abs().max():.2e}")
    else:
        print("  ✓ Outputs are identical (as expected)")