"""
Neural ODE-LMM: Neural ODE +/- BMI Skip Connection.

Architecture:
  - Encoder:  z(0) = Enc(t0, BMI0, static)
  - Dynamics: dz/dt = f(z, t, static, x(t))          ← Neural ODE
  - Decoder:  mu(t) = rho(z(t), BMI_std(t)) @ beta_neural  ← BMI skip connection
  - RE:       Z = [1, rs1(t), rs2(t)] or g(z(t))          (classical spline basis/neural network)

BMI is completely removed from the ODE dynamics.
This guarantees PDP separation at all times.

Data encoding (same as CDE version for compatibility):
  - x_pad: [BMI_t, rs1, rs2]
  - static: [SEX_code, AGEc, DIPNIV2, DIPNIV3]
  - c_mask from dataloader is accepted but IGNORED
"""
import torch
from torch.utils.data import DataLoader, Subset
import os
import math
import numpy as np
import pyreadr
import pandas as pd
from dataset import LongitudinalDataset, collate_pad
from model_ODE_torchdiff import NeuralODEModel, NeuralODEConfig
from utils import masked_NLL
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Neural ODE-LMM training")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/simulation_cumulative_effect_diagoD_noBMIInEncoder")
    parser.add_argument("--data", type=str,
                        default="simu_datasets/S5_sims")
    args = parser.parse_args()

    # ---- Config ----
    LR = 1e-3
    WD = 1e-5
    EPOCHS = 1000
    BATCH_SIZE = 128
    PRINT_EVERY = 25
    TEST_RATIO = 0.2
    SEED = 42

    print("=" * 60)
    print("NEURAL ODE +/- BMI SKIP CONNECTION")
    print("=" * 60)

    # ---- Data ----
    time_col = "time"
    y_col = "ISA15_sim"
    id_col = "NUM_ID"
    x_cols = ["BMI_t", "rs1", "rs2"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    for i in range(100):
        path = args.data + f"/sim_{i+1:03d}.rds"
        df = next(iter(pyreadr.read_r(path).values()))
        df["SEX"] = df["SEX"].astype("category")
        df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
        df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
        df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

        full_dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col,
                                        static_cols=static_cols)
        print(f"Sample 0: {full_dataset[0]}")

        # ---- Train/test split (subject level) ----
        N = len(full_dataset)
        rng = np.random.RandomState(SEED)
        indices = rng.permutation(N)
        n_test = int(N * TEST_RATIO)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        train_dataset = Subset(full_dataset, train_idx)
        test_dataset = Subset(full_dataset, test_idx)

        # ---- BMI stats from training set only ----
        train_bmis = []
        for idx in train_idx:
            sample = full_dataset[idx]
            covariates = sample[2]        # (T_i, 3)
            bmi_vals = covariates[:, 0]   # (T_i,)
            train_bmis.append(bmi_vals)

        train_bmis = torch.cat(train_bmis)
        bmi_mean = train_bmis.mean().item()
        bmi_std = train_bmis.std().item()
        print(f"BMI stats (train): mean={bmi_mean:.4f}, std={bmi_std:.4f}")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=collate_pad)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, collate_fn=collate_pad)
        print(f"Subjects: {N} total → {len(train_idx)} train, {len(test_idx)} test")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_tv = 1   # only BMI_t

        # ---- Model ----
        cfg = NeuralODEConfig(
            hidden_channels=8,
            enc_mlp_hidden=32,
            func_mlp_hidden=32,
            dec_rho_hidden=16,
            dec_p=4,
            dec_q=3,                      # q=3: intercept + rs1 + rs2
            depth=2,
            dropout=0.0,
            euler_steps_per_interval=4,
            ode_solver='rk4'
        )

        # static_skip_dims=[1] passes AGEc directly to decoder
        # This helps the NN learn BMI × AGEc interactions easily
        model = NeuralODEModel(
            x_dim=len(x_cols),
            static_dim=len(static_cols),
            cfg=cfg,
            n_tv=n_tv,
            use_rho_net=True,
            use_neural_re=True,
            re_spline_cols=None,
            g_hidden=16,
            fullD=False,
            bmi_mean=bmi_mean,
            bmi_std=bmi_std,
            use_bmi_skip=False,
            static_skip_dims=None,         # AGEc skip to decoder
            reg_mode=None
        ).to(device)

        LAMBDA_REG = 0.5

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params}")
        print(f"Architecture:")
        print(f"  Encoder:  z(0) = Enc(t0, BMI0, static)")
        print(f"  ODE:      dz/dt = f(z, t, x(t))")
        print(f"  Decoder:  mu = rho(z(t), BMI_std, AGEc) @ beta_neural")
        print(f"  RE:       g(z(t))")
        print(f"  Euler sub-steps: {cfg.euler_steps_per_interval}")

        # ---- Optimizer ----
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )

        # ---- Checkpoint dir ----
        os.makedirs(args.checkpoint, exist_ok=True)
        ckpt_path = args.checkpoint+"/best_model_ode_"+str(i)+".pt"

        # ---- Train ----
        best_test_loss = float("inf")
        best_state = None
        patience = 300

        for epoch in range(1, EPOCHS + 1):

            if patience == 0:
                break

            # ---- Training ----
            model.train()
            total_nll = 0.0
            count = 0

            for batch in train_loader:
                _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
                t_pad = t_pad.to(device)
                x_pad = x_pad.to(device)
                y_pad = y_pad.to(device)
                mask = mask.to(device)
                s = s.to(device)
                # c_mask is NOT passed to the model (ODE ignores it)

                mu, V, _, _, _ , reg_dict = model(t_pad, x_pad, masks=None,
                                    static_covariates=s, bmi_t=x_pad[:, :, 0:1], obs_mask=mask)
                loss = masked_NLL(mu, y_pad, V, mask)

                if reg_dict and "reg_term" in reg_dict:
                    loss = loss + LAMBDA_REG * reg_dict["reg_term"]

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
                    t_pad = t_pad.to(device)
                    x_pad = x_pad.to(device)
                    y_pad = y_pad.to(device)
                    mask = mask.to(device)
                    s = s.to(device)

                    mu, V, _, _, _, reg_term = model(t_pad, x_pad, masks=None,
                                        static_covariates=s, bmi_t=x_pad[:, :, 0:1], obs_mask=mask)
                    test_loss = masked_NLL(mu, y_pad, V, mask)
                    test_nll += test_loss.item()
                    test_count += 1

            avg_test = test_nll / max(test_count, 1)
            scheduler.step(avg_test)

            if avg_test < best_test_loss:
                best_test_loss = avg_test
                patience = 300
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                torch.save({
                    'model_state_dict': best_state,
                    'best_test_loss': best_test_loss,
                    'epoch': epoch,
                    'train_idx': train_idx,
                    'test_idx': test_idx,
                    'bmi_mean': bmi_mean,
                    'bmi_std': bmi_std,
                    'config': {
                        'hidden_channels': cfg.hidden_channels,
                        'euler_steps': cfg.euler_steps_per_interval,
                        'use_rho_net': True,
                        'use_neural_re': True,
                        'static_skip_dims': [1],
                        'approach': 'ode_bmi_skip',
                    },
                }, ckpt_path)
            else:
                patience = patience - 1

            if epoch % PRINT_EVERY == 0 or epoch == 1:
                beta_neural = model.decoder.beta_neural.detach().cpu()
                sig2 = torch.exp(model.decoder.log_residual_var).item()
                D = model.decoder._build_D(device,
                        model.decoder.log_residual_var.dtype).detach().cpu()

                print(f"\nEpoch {epoch:5d} | train NLL = {avg_train:.4f} | "
                    f"test NLL = {avg_test:.4f} | best test = {best_test_loss:.4f}")
                print(f"    sigma2 = {sig2:.4f}")
                print(f"    D:")
                for i in range(D.shape[0]):
                    print(f"      [{', '.join(f'{D[i,j]:.4f}' for j in range(D.shape[1]))}]")
                print(f"    beta_neural = [{', '.join(f'{v:.4f}' for v in beta_neural)}]")

        # # ---- Restore best ----
        # if best_state is not None:
        #     model.load_state_dict(best_state, strict=False)
 
        # # ============================================================
        # # Fine-tuning: converge to MLE (no regularization)
        # # ============================================================
        # FT_LR = 1e-5
        # FT_EPOCHS = 200
        # FT_PRINT_EVERY = 25
 
        # print(f"\n{'='*60}")
        # print(f"FINE-TUNING: LR={FT_LR}, WD=0, L1=0, {FT_EPOCHS} epochs")
        # print(f"{'='*60}")
 
        # optimizer_ft = torch.optim.Adam(model.parameters(), lr=FT_LR,
        #                                 weight_decay=0.0)
 
        # # Use FULL dataset (train+test) for MLE — no early stopping
        # full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE,
        #                          shuffle=True, collate_fn=collate_pad)
 
        # for epoch in range(1, FT_EPOCHS + 1):
        #     model.train()
        #     total_nll = 0.0
        #     count = 0
 
        #     for batch in full_loader:
        #         _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        #         t_pad = t_pad.to(device)
        #         x_pad = x_pad.to(device)
        #         y_pad = y_pad.to(device)
        #         mask = mask.to(device)
        #         s = s.to(device)
 
        #         mu, V, _, _, _, reg_dict = model(
        #             t_pad, x_pad, masks=None,
        #             static_covariates=s, bmi_t=x_pad[:, :, 0:1],
        #             obs_mask=mask
        #         )
        #         # Pure NLL — NO regularization
        #         loss = masked_NLL(mu, y_pad, V, mask)
 
        #         optimizer_ft.zero_grad(set_to_none=True)
        #         loss.backward()
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #         optimizer_ft.step()
 
        #         total_nll += loss.item()
        #         count += 1
 
        #     avg_nll = total_nll / max(count, 1)
 
        #     if epoch % FT_PRINT_EVERY == 0 or epoch == 1:
        #         print(f"  FT epoch {epoch:4d} | NLL = {avg_nll:.4f}")
 
        # # Save fine-tuned (MLE) checkpoint
        # ft_ckpt_path = ckpt_path.replace(".pt", "_mle.pt")
        # torch.save({
        #     'model_state_dict': {k: v.clone()
        #                          for k, v in model.state_dict().items()},
        #     'ft_epochs': FT_EPOCHS,
        #     'ft_lr': FT_LR,
        #     'bmi_mean': bmi_mean,
        #     'bmi_std': bmi_std,
        #     'config': {
        #         'hidden_channels': cfg.hidden_channels,
        #         'euler_steps': cfg.euler_steps_per_interval,
        #         'use_rho_net': True,
        #         'use_neural_re': True,
        #         'static_skip_dims': [1],
        #         'approach': 'ode_bmi_skip_mle',
        #     },
        # }, ft_ckpt_path)
        # print(f"  MLE checkpoint saved → {ft_ckpt_path}")