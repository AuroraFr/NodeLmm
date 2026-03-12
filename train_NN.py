"""
Hybrid Neural CDE-LMM: Neural FE + Fixed Spline RE.
Fixed effects are entirely neural: mu = rho(z(t)) @ beta_neural
Random effects use classical spline basis: Z = [1, rs1(t), rs2(t)], b_i ~ N(0, D)

Uses same data encoding as semi-parametric model:
  - x_pad: [BMI_t, rs1, rs2]  (rs1/rs2 are precomputed R spline columns for RE)
  - static: [SEX_code, AGEc, DIPNIV2, DIPNIV3]
  - CDE sees: [time, BMI_t, cumulative_mask]  (only first n_tv=1 columns)
  - CDEFunc input: [z(t), x(t), static] — same as semi-parametric
  - Decoder RE: Z = [1, rs1, rs2] from x_pad (not learned from z(t))
"""
import torch
from torch.utils.data import DataLoader, Subset
import os
import math
import numpy as np
import pyreadr
import pandas as pd
from dataset import LongitudinalDataset, collate_pad
from model import NeuralCDEModel, NeuralCDEConfig
from utils import masked_NLL

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":

    # ---- Config ----
    LR = 1e-3
    WD = 1e-5
    EPOCHS = 1000
    BATCH_SIZE = 128
    PRINT_EVERY = 25
    TEST_RATIO = 0.2
    SEED = 42

    print("="*60)
    print("HYBRID: NEURAL FE + FIXED SPLINE RE")
    print("="*60)

    # ---- Data ----
    time_col = "time"
    y_col = "ISA15_sim"
    id_col = "NUM_ID"
    # Pure neural FE + fixed spline RE
    # CDE sees only BMI_t (n_tv=1); rs1, rs2 ride along in x_pad for the decoder RE
    x_cols = ["BMI_t", "rs1", "rs2"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    path = "simu_datasets/S2a_sims_2/sim_001.rds"
    df = next(iter(pyreadr.read_r(path).values()))
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    full_dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col, static_cols=static_cols)
    print(full_dataset[0]) 

    # ---- Train/test split (subject level) ----
    N = len(full_dataset)
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(N)
    n_test = int(N * TEST_RATIO)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset  = Subset(full_dataset, test_idx)
    # Extract BMI stats from training subjects only
    # Depends on how full_dataset stores covariates — e.g.:
    train_bmis = []
    for idx in train_idx:
        sample = full_dataset[idx]
        covariates = sample[2]       # (T, 3) — BMI is column 0
        bmi_vals = covariates[:, 0]  # (T,)
        train_bmis.append(bmi_vals)

    train_bmis = torch.cat(train_bmis)
    bmi_mean = train_bmis.mean()
    bmi_std = train_bmis.std()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad)

    print(f"Subjects: {N} total → {len(train_idx)} train, {len(test_idx)} test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_tv = 1   # only BMI_t is a real time-varying covariate for the CDE

    # ---- Model: neural FE + classical spline RE ----
    cfg = NeuralCDEConfig(
        hidden_channels=8,
        enc_mlp_hidden=32,
        func_mlp_hidden=32,
        dec_rho_hidden=16,
        dec_p=4,
        dec_q=3,                    # q=3: intercept + rs1 + rs2
        depth=2,
        dropout=0.0,
    )

    model = NeuralCDEModel(
        x_dim=len(x_cols),           # 3: BMI_t, rs1, rs2
        static_dim=len(static_cols),
        cfg=cfg,
        n_tv=n_tv,
        use_rho_net=True,           # FE: rho(z(t)) @ beta_neural (fully neural)
        use_neural_re=False,        # RE: Z = [1, rs1(t), rs2(t)] from data (classical)
        re_spline_cols=[1, 2],
        g_hidden=16,
        bmi_mean=bmi_mean, bmi_std=bmi_std
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Hybrid formulation:")
    print(f"  FE: mu = rho(z(t)) @ beta_neural  (fully neural)")
    print(f"  RE: Z = [1, rs1(t), rs2(t)] (fixed spline basis from data)")
    print(f"       b_i ~ N(0, D)")
    print(f"  No parametric W@beta")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )

    # ---- Train ----
    best_test_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        # ---- Training ----
        model.train()
        total_nll = 0.0
        count = 0

        for batch in train_loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad  = t_pad.to(device)
            x_pad  = x_pad.to(device)
            y_pad  = y_pad.to(device)
            mask   = mask.to(device)
            c_mask = c_mask.to(device)
            s      = s.to(device)

            # y_pad=None → uses frozen zero beta, no AtA/Atb
            mu, V, _,_,_ = model(t_pad, x_pad, c_mask, s, mask, y_pad=None)
            loss = masked_NLL(mu, y_pad, V, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_nll += loss.item()
            count += 1

        avg_train = total_nll / max(count, 1)

        # ---- Test evaluation ----
        model.eval()
        test_nll = 0.0
        test_count = 0
        with torch.no_grad():
            for batch in test_loader:
                _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
                t_pad  = t_pad.to(device)
                x_pad  = x_pad.to(device)
                y_pad  = y_pad.to(device)
                mask   = mask.to(device)
                c_mask = c_mask.to(device)
                s      = s.to(device)

                mu, V,_,_,_ = model(t_pad, x_pad, c_mask, s, mask, y_pad=None)
                loss = masked_NLL(mu, y_pad, V, mask)
                test_nll += loss.item()
                test_count += 1

        avg_test = test_nll / max(test_count, 1)
        scheduler.step(avg_test)

        if avg_test < best_test_loss:
            best_test_loss = avg_test
            best_state = {k: v.clone() for k, v in model.state_dict().items()
                         if 'X_spline' not in k}
            clean_state = {k: v for k, v in model.state_dict().items() if 'X_spline' not in k}
            torch.save({
                'model_state_dict': best_state,
                'best_test_loss': best_test_loss,
                'train_idx': train_idx,
                'test_idx': test_idx,
                'config': {
                    'hidden_channels': cfg.hidden_channels,
                    'include_fe_splines': False,
                    'use_rho_net': True,
                    'use_neural_re': False,
                    'interaction_pairs': None,
                    'approach': 'hybrid_neural_fe_spline_re',
                },
            }, 'checkpoints/best_model_hybrid_neural_fe_spline_re.pt')

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            beta_neural = model.decoder.beta_neural.detach().cpu()
            sig2 = torch.exp(model.decoder.log_residual_var).item()
            D = model.decoder._build_D(device, model.decoder.log_residual_var.dtype).detach().cpu()

            print(f"\nEpoch {epoch:5d} | train NLL = {avg_train:.4f} | "
                  f"test NLL = {avg_test:.4f} | best test = {best_test_loss:.4f}")
            print(f"    sigma2 = {sig2:.4f}")
            for i in range(D.shape[0]):
                print(f"    [{', '.join(f'{D[i,j]:.4f}' for j in range(D.shape[1]))}]")
            print(f"    beta_neural norm = {beta_neural.norm().item():.4f}")
            print(f"    beta_neural = [{', '.join(f'{v:.4f}' for v in beta_neural)}]")

    # ---- Restore best ----
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # ---- Save ----
    clean_state = {k: v for k, v in model.state_dict().items() if 'X_spline' not in k}
    torch.save({
        'model_state_dict': clean_state,
        'best_test_loss': best_test_loss,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'config': {
            'hidden_channels': cfg.hidden_channels,
            'include_fe_splines': False,
            'use_rho_net': True,
            'use_neural_re': False,
            'interaction_pairs': None,
            'approach': 'hybrid_neural_fe_spline_re',
        },
    }, 'checkpoints/best_model_hybrid_neural_fe_spline_re.pt')

    # ---- Final + PDP ----
    print("\n" + "="*60)
    print("FINAL RESULTS (Hybrid: Neural FE + Spline RE)")
    print("="*60)
    print(f"  beta_neural = {model.decoder.beta_neural.detach().cpu().tolist()}")
    print(f"  sigma2 = {torch.exp(model.decoder.log_residual_var).item():.4f}")
    D_final = model.decoder._build_D(device, model.decoder.log_residual_var.dtype).detach().cpu()
    print(f"  D (RE covariance):")
    for i in range(D_final.shape[0]):
        print(f"    [{', '.join(f'{D_final[i,j]:.4f}' for j in range(D_final.shape[1]))}]")
    print(f"  NOTE: FE is fully neural — PDP is the only way to assess covariate effects")
    print(f"  RE: classical [1, rs1(t), rs2(t)] spline basis")

    print("\n--- PDP Analysis ---")
    from pdp_analysis import (compute_pdp, plot_pdp, plot_pdp_marginal,
                              compute_delta_pdp, compute_delta_pdp_stratified)

    eval_loader = DataLoader(full_dataset, batch_size=256, shuffle=False, collate_fn=collate_pad)
    bmi_values = [20, 23, 26, 29, 32, 35]

    results, ages, masks, times = compute_pdp(
        model, eval_loader, device, bmi_values, n_tv=n_tv, interp="linear"
    )
    plot_pdp(results, ages, masks, times, bmi_values, save_path="pdp_hybrid_by_age.png")
    plot_pdp_marginal(results, masks, times, bmi_values, save_path="pdp_hybrid_marginal.png")
    delta = compute_delta_pdp(results, masks, bmi_lo=20, bmi_hi=35)
    compute_delta_pdp_stratified(results, ages, masks, times, bmi_lo=20, bmi_hi=35)