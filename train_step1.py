"""
Step 1: Full Neural CDE-LMM on Scenario 2 (BMI×AGE interaction).
Uses precomputed R splines. Interaction in W → neural part should learn ~nothing.

x_pad layout: ["GLUC_t", "BMI_t", "ns1", "ns2", "ns3", "rs1", "rs2"]
              |--- CDE sees ---|  |------- decoder only ---------|
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pyreadr
import pandas as pd
from dataset import LongitudinalDataset, collate_pad
from model_step1 import NeuralCDEModel, NeuralCDEConfig
from utils import masked_NLL

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":

    # ---- Config ----
    LR = 1e-3
    WD = 1e-5
    EPOCHS = 5000
    BATCH_SIZE = 128
    PRINT_EVERY = 25

    # ---- Data ----
    time_col = "time"
    y_col = "ISA15_sim"
    id_col = "NUM_ID"

    # x_pad: first 2 = real covariates (CDE input), rest = R spline columns (decoder only)
    x_cols = ["GLUC_t", "BMI_t", "ns1", "ns2", "ns3", "rs1", "rs2"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    path = "simu_datasets/S2a_sim/sim_001.rds"
    df = next(iter(pyreadr.read_r(path).values()))

    # Encode SEX
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")

    # Encode DIPNIV (reference = level 1)
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    print(f"DIPNIV2 count: {df['DIPNIV2'].sum():.0f}, DIPNIV3 count: {df['DIPNIV3'].sum():.0f}")

    dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col, static_cols=static_cols)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Spline knots from R ----
    fe_knots    = np.array([1.769863, 6.693151])
    fe_boundary = np.array([0.0, 13.50685])
    re_knots    = np.array([3.567123])
    re_boundary = np.array([0.0, 13.50685])

    # ---- Interaction: BMI_t × AGEc ----
    # In x_cols[:n_tv]: BMI_t is index 1
    # In static_cols: AGEc is index 1
    interaction_pairs = [(1, 1)]

    n_tv = 2  # only GLUC_t, BMI_t are real covariates

    # ---- Model ----
    cfg = NeuralCDEConfig(
        hidden_channels=4,
        enc_mlp_hidden=16,
        func_mlp_hidden=16,
        dec_rho_hidden=8,
        dec_p=4,
        dec_q=3,          # intercept + 2 RE splines
        depth=2,
        dropout=0.0,
    )

    model = NeuralCDEModel(
        x_dim=len(x_cols),         # total columns in x_pad (7)
        static_dim=len(static_cols),  # 4
        cfg=cfg,
        fe_spline_knots=fe_knots,
        fe_spline_boundary=fe_boundary,
        re_spline_knots=re_knots,
        re_spline_boundary=re_boundary,
        interaction_pairs=interaction_pairs,
        precomputed_splines=True,
        n_tv=n_tv,                 # CDE only sees first 2 columns
    ).to(device)

    # ---- Print model info ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Decoder n_tv={model.decoder.n_tv}, n_W={model.decoder.n_W}, "
          f"precomputed={model.decoder.precomputed_splines}")
    print(f"CDE input_channels={model.input_channels} (should be {1 + n_tv})")

    # Beta layout
    beta_names = ["intercept"] + [f"ns{i+1}" for i in range(model.decoder.fe_spline_df)]
    beta_names += x_cols[:n_tv] + static_cols
    for tv_i, s_i in interaction_pairs:
        beta_names.append(f"{x_cols[tv_i]}x{static_cols[s_i]}")
    print(f"Beta layout ({len(beta_names)}): {beta_names}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )

    # ---- Train ----
    model.train()
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        count = 0

        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad  = t_pad.to(device)
            x_pad  = x_pad.to(device)
            y_pad  = y_pad.to(device)
            mask   = mask.to(device)
            c_mask = c_mask.to(device)
            s      = s.to(device)

            mu, V = model(t_pad, x_pad, c_mask, s, mask)
            loss = masked_NLL(mu, y_pad, V, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += loss.item()
            count += 1

        avg = total / max(count, 1)
        scheduler.step(avg)

        if avg < best_loss:
            best_loss = avg

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            beta = model.decoder.beta.detach().cpu()
            beta_neural = model.decoder.beta_neural.detach().cpu()

            print(f"\nEpoch {epoch:5d} | loss = {avg:.4f} | best = {best_loss:.4f}")
            print("  --- Parametric beta ---")
            for name, val in zip(beta_names, beta):
                print(f"    {name:>20s} = {val.item():+.4f}")

            sig2 = torch.exp(model.decoder.log_residual_var).item()
            print(f"    {'sigma2':>20s} = {sig2:.4f}")

            print(f"  --- Neural contribution ---")
            print(f"    beta_neural norm = {beta_neural.norm().item():.4f}")
            print(f"    beta_neural = [{', '.join(f'{v:.4f}' for v in beta_neural)}]")

    # ---- Final ----
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    beta = model.decoder.beta.detach().cpu()
    for name, val in zip(beta_names, beta):
        print(f"  {name:>20s} = {val.item():+.6f}")

    print(f"\nTrue β₀ (Scenario 2):")
    print(f"  intercept=30.71, ns1=-2.82, ns2=0.02, ns3=-3.14")
    print(f"  GLUC=-0.95, BMI=-0.175, SEX=1.85, AGE=-0.51")
    print(f"  DIPNIV2=2.67, DIPNIV3=3.32, BMIxAGE=-0.015")

    print(f"\nNeural beta_neural norm = {model.decoder.beta_neural.detach().norm().item():.6f}")
    print(f"(Should be ~0 since true model is linear with interaction)")