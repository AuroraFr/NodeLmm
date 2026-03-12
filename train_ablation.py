"""
Pure LMM ablation: uses Decoder directly with precomputed R splines.
No CDE, no neural head — just beta, D, sigma2.
Should recover HLME coefficients exactly.

x_cols layout: ["GLUC_t", "BMI_t", "ns1", "ns2", "ns3", "rs1", "rs2"]
               |--- n_tv=2 ---|  |-- fe_spline=3 --|  |-- re=2 --|
"""
import torch
from torch.utils.data import DataLoader
import os
import math
import numpy as np
import pyreadr
import pandas as pd
from dataset import LongitudinalDataset, collate_pad
from model_ablation import Decoder
from utils import masked_NLL

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":

    # ---- Config ----
    LR = 1e-2
    WD = 0.0
    EPOCHS = 5000
    BATCH_SIZE = 128
    PRINT_EVERY = 50

    # ---- Data ----
    time_col = "time"
    y_col = "ISA15_sim"
    id_col = "NUM_ID"

    # Time-varying: first 2 are real covariates, rest are R spline columns
    x_cols = ["GLUC_t", "BMI_t", "ns1", "ns2", "ns3", "rs1", "rs2"]

    # Static covariates (one-hot DIPNIV)
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    path = "simu_datasets/S2a_sim/sim_001.rds"
    df = next(iter(pyreadr.read_r(path).values()))

    # Encode SEX (binary → 0/1 is fine)
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")

    # Encode DIPNIV (1,2,3 → two dummies, reference=1)
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    # Verify
    print(f"DIPNIV2 count: {df['DIPNIV2'].sum():.0f}, DIPNIV3 count: {df['DIPNIV3'].sum():.0f}")

    dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col, static_cols=static_cols)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Spline knots (from R — only needed for df computation) ----
    fe_knots    = np.array([1.769863, 6.693151])
    fe_boundary = np.array([0.0, 13.50685])
    re_knots    = np.array([3.567123])
    re_boundary = np.array([0.0, 13.50685])

    # ---- Interaction: BMI_t × AGEc ----
    # In x_cols[:n_tv]: BMI_t is index 1
    # In static_cols: AGEc is index 1
    interaction_pairs = [(1, 1)]
    interaction_pairs = None

    n_tv = 2          # GLUC_t, BMI_t
    q = 3             # intercept + 2 RE splines

    # ---- Build Decoder only (no CDE) ----
    latent_dim = 4  # dummy, won't be used

    decoder = Decoder(
        latent_dim=latent_dim,
        p=4,               # neural basis dim (zeroed out, doesn't matter)
        q=q,
        fullD=True,
        n_static=len(static_cols),
        n_tv=n_tv,
        fe_spline_knots=fe_knots,
        fe_spline_boundary=fe_boundary,
        re_spline_knots=re_knots,
        re_spline_boundary=re_boundary,
        interaction_pairs=interaction_pairs,
        precomputed_splines=True,
    ).to(device)

    # Zero out neural contribution → mu_neural = 0 everywhere
    with torch.no_grad():
        decoder.beta_neural.zero_()

    # Only train: beta, D (L_unconstrained), sigma2
    for name, param in decoder.named_parameters():
        if name in ["beta", "log_residual_var", "L_unconstrained"]:
            param.requires_grad = True
            print(f"  TRAINABLE: {name:40s}  shape={list(param.shape)}")
        else:
            param.requires_grad = False

    trainable_params = [p for p in decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WD)

    print(f"\nTotal params: {sum(p.numel() for p in decoder.parameters())}")
    print(f"Trainable:    {sum(p.numel() for p in trainable_params)}")

    # ---- Beta layout ----
    beta_names = ["intercept"] + [f"ns{i+1}" for i in range(decoder.fe_spline_df)]
    beta_names += x_cols[:n_tv] + static_cols
    if interaction_pairs is not None:
        for tv_i, s_i in interaction_pairs:
            beta_names.append(f"{x_cols[tv_i]}x{static_cols[s_i]}")
    print(f"Beta layout: {beta_names}")
    print(f"n_W = {decoder.n_W}")

    # ---- Train ----
    decoder.train()
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
            s      = s.to(device)

            # Dummy z_t — zeroed out via beta_neural, so content doesn't matter
            N, T = t_pad.shape
            z_t = torch.zeros(N, T, latent_dim, device=device)

            mu, V = decoder(z_t, t_pad, x_pad, s, obs_mask=mask)
            loss = masked_NLL(mu, y_pad, V, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total += loss.item()
            count += 1

        avg = total / max(count, 1)
        if avg < best_loss:
            best_loss = avg
            torch.save(
                {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                "checkpoints/simuS2a_model_ablation_nointeraction.pt",
            )

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            beta = decoder.beta.detach().cpu()
            print(f"\nEpoch {epoch:5d} | loss = {avg:.4f} | best = {best_loss:.4f}")
            for name, val in zip(beta_names, beta):
                print(f"  {name:>20s} = {val.item():+.4f}")
            sig2 = torch.exp(decoder.log_residual_var).item()
            print(f"  {'sigma2':>20s} = {sig2:.4f}")
            if decoder.L_unconstrained is not None:
                L = torch.tril(decoder.L_unconstrained.detach().cpu())
                diag = torch.diagonal(L)
                diag_pos = torch.nn.functional.softplus(diag) + decoder.D_diag_min
                L = L - torch.diag(diag) + torch.diag(diag_pos)
                D = L @ L.t()
                print(f"  {'D diag':>20s} = [{', '.join(f'{D[i,i]:.4f}' for i in range(q))}]")

    # ---- Final ----
    print("\n" + "="*60)
    print("FINAL ESTIMATED BETA:")
    print("="*60)
    beta = decoder.beta.detach().cpu()
    for name, val in zip(beta_names, beta):
        print(f"  {name:>20s} = {val.item():+.6f}")

    print(f"\nHLME reference (Scenario 2):")
    print(f"  intercept ≈ 30.71,  ns1 ≈ -2.82,  ns2 ≈ 0.02,   ns3 ≈ -3.14")
    print(f"  GLUC ≈ -0.95,  BMI ≈ -0.175,  SEX ≈ 1.85,  AGE ≈ -0.51")
    print(f"  DIPNIV2 ≈ 2.67,  DIPNIV3 ≈ 3.32,  BMIxAGE ≈ -0.015")
    print(f"  sigma2 ≈ 13.1 (3.62^2)")