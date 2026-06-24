"""
Neural ODE-LMM: Neural ODE +/- BMI Skip Connection.

Architecture:
  - Encoder:  z(0) = Enc(t0, BMI0, static)
  - Dynamics: dz/dt = f(z, t, static)          ← Neural ODE
  - Decoder:  mu(t) = rho(z(t), BMI_std(t)) @ beta_neural  ← BMI skip connection
  - RE:       Z = [1, rs1(t), rs2(t)] or g(z(t))          (classical spline basis/neural network)

This guarantees PDP separation at all times.

Data encoding (same as CDE version for compatibility):
  - x_pad: [BMI_t, rs1, rs2]
  - static: [SEX_code, AGEc, DIPNIV2, DIPNIV3]
  - c_mask from dataloader is accepted but IGNORED
"""
import torch
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
import pyreadr
from dataset import LongitudinalDataset, collate_pad
from model_ODE_baseline import NeuralODEModel, NeuralODEConfig
from utils import masked_NLL
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Neural ODE-LMM training")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/simulation_baseline_noreg_seed42_rhonorm_diagoD")
    parser.add_argument("--data", type=str,
                        default="simu_datasets/S2a_sims")
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
    x_cols = ["BMI_t"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    for i in range(20, 21):
        path = args.data + f"/sim_{i+1:03d}.rds"
        df = next(iter(pyreadr.read_r(path).values()))
        df["SEX"] = df["SEX"].astype("category")
        df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
        df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
        df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

        # print(f"DIPNIV unique: {df['DIPNIV'].unique()}")
        # print(f"DIPNIV3 sum: {df['DIPNIV3'].sum()}")
        # print(f"DIPNIV3 mean: {df['DIPNIV3'].mean()}")
        # break

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
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        np.random.seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        n_tv = 1   # only BMI_t

        # ---- Model ----
        cfg = NeuralODEConfig(
            hidden_channels=8,
            enc_mlp_hidden=16,
            func_mlp_hidden=16,
            dec_rho_hidden=16,
            dec_p=4,
            dec_q=3,                      # q=3: intercept + rs1 + rs2
            depth=2,
            dropout=0.0,
            euler_steps_per_interval=4,
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
            g_hidden=8,
            fullD=False,
            bmi_mean=bmi_mean,
            bmi_std=bmi_std,
            use_bmi_skip=True,
            static_skip_dims=[0,1,2,3],        
            reg_mode=None,
            use_learned_z0=False
        ).to(device)

        LAMBDA_REG = 0.0
        REG_MODE = None

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params}")
        print(f"Architecture:")
        print(f"  Encoder:  z(0) = Enc(t0, BMI0, static)")
        print(f"  ODE:      dz/dt = f(z, t, x(t), static)")
        print(f"  Decoder:  mu = rho(z(t), BMI_std, AGEc) @ beta_neural")
        print(f"  RE:       g(z(t))")
        print(f"  Euler sub-steps: {cfg.euler_steps_per_interval}")

        # ── Optimiser ───────────────────────────────────────────────────────
        nn_weights, var_params, fe_params = [], [], []
        gl_param_set = {id(model.rho_net.net[0].weight)}

        for n, p in model.named_parameters():
            if id(p) in gl_param_set:
                continue
            elif 'log_residual_var' in n or 'log_std' in n:
                var_params.append(p)
            elif 'beta_neural' in n:
                fe_params.append(p)
            else:
                nn_weights.append(p)

        gl_params = [model.rho_net.net[0].weight]

        optimizer = torch.optim.AdamW([
            {'params': nn_weights, 'weight_decay': WD},
            {'params': gl_params,  'weight_decay': 0.0},
            {'params': var_params, 'weight_decay': 0.0},
            {'params': fe_params,  'weight_decay': 0.0},
        ])

        # ---- Optimizer ----
        # optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
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

                if REG_MODE == "skip_gate" and "gate_values" in reg_dict:
                    gv = reg_dict["gate_values"]
                    print(gv)
                    names = ["AGEc"]
                    print(f"    gates:")
                    for g, name in enumerate(names):
                        print(f"      {name:>8s}: {gv[g]:.4f}")
