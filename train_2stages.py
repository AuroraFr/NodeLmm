"""
Two-stage training on Scenario 2 with regularization options.
  Stage 1: Pure LMM ablation → β* (already done)
  Stage 2: Freeze β*, train CDE + D + σ² with regularization on h(z(t))

β is frozen by construction — never updated, never contaminated.

Regularization options (can be combined):
  1. LAMBDA_SMOOTH: temporal smoothness penalty on h(z(t))
     → penalizes |h(t+1) - h(t)|², discourages spurious accumulation
  2. LAMBDA_H_MAG: L2 penalty on h(z(t)) magnitude
     → pushes h toward zero, neural part only activates when it really helps
  3. LAMBDA_BETA_NEURAL: L2 penalty on beta_neural weights
     → limits overall neural contribution capacity
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pyreadr
from dataset import LongitudinalDataset, collate_pad
from model_hybrid import NeuralCDEModel, NeuralCDEConfig
from utils import masked_NLL

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def reg_temporal_smoothness(h, obs_mask):
    """
    Penalize |h(t+1) - h(t)|² at observed positions.
    Discourages the CDE from building up spurious cumulative effects.
    """
    # Differences between consecutive time points
    dh = h[:, 1:] - h[:, :-1]                          # (N, T-1)
    # Mask: both t and t+1 must be observed
    mask_pairs = obs_mask[:, 1:] * obs_mask[:, :-1]     # (N, T-1)
    n_pairs = mask_pairs.sum().clamp(min=1)
    return (dh ** 2 * mask_pairs).sum() / n_pairs


def reg_h_magnitude(h, obs_mask):
    """
    Penalize |h(t)|² at observed positions.
    Pushes neural contribution toward zero — only activates when needed.
    """
    n_obs = obs_mask.sum().clamp(min=1)
    return (h ** 2 * obs_mask).sum() / n_obs


def reg_beta_neural(model):
    """
    L2 penalty on beta_neural weights.
    Limits the overall capacity of the neural fixed effect.
    """
    return model.decoder.beta_neural.pow(2).sum()


if __name__ == "__main__":

    # ================================================================
    # REGULARIZATION CONFIG — toggle and tune here
    # ================================================================
    LAMBDA_SMOOTH      = 0.1     # temporal smoothness on h(z(t)), 0 = off
    LAMBDA_H_MAG       = 0.01    # L2 on h(z(t)) magnitude, 0 = off
    LAMBDA_BETA_NEURAL = 0.0     # L2 on beta_neural, 0 = off

    print("="*60)
    print("REGULARIZATION CONFIG")
    print("="*60)
    print(f"  LAMBDA_SMOOTH      = {LAMBDA_SMOOTH}")
    print(f"  LAMBDA_H_MAG       = {LAMBDA_H_MAG}")
    print(f"  LAMBDA_BETA_NEURAL = {LAMBDA_BETA_NEURAL}")
    print()

    # ---- Training config ----
    LR = 1e-3
    WD = 1e-5
    EPOCHS = 500
    BATCH_SIZE = 128
    PRINT_EVERY = 25

    # ---- Data ----
    time_col = "time"
    y_col = "ISA15_sim"
    id_col = "NUM_ID"
    x_cols = ["GLUC_t", "BMI_t", "ns1", "ns2", "ns3", "rs1", "rs2"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    path = "simu_datasets/S2a_sim/sim_001.rds"
    df = next(iter(pyreadr.read_r(path).values()))
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col, static_cols=static_cols)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Spline knots from R ----
    fe_knots    = np.array([1.769863, 6.693151])
    fe_boundary = np.array([0.0, 13.50685])
    re_knots    = np.array([3.567123])
    re_boundary = np.array([0.0, 13.50685])

    n_tv = 2
    interaction_pairs = None  # No interaction in W — CDE must discover it

    # ---- Model (larger architecture) ----
    cfg = NeuralCDEConfig(
        hidden_channels=8,       # was 4
        enc_mlp_hidden=32,       # was 16
        func_mlp_hidden=32,      # was 16
        dec_rho_hidden=16,       # was 8
        dec_p=4,
        dec_q=3,
        depth=2,
        dropout=0.0,
    )

    model = NeuralCDEModel(
        x_dim=len(x_cols),
        static_dim=len(static_cols),
        cfg=cfg,
        fe_spline_knots=fe_knots,
        fe_spline_boundary=fe_boundary,
        re_spline_knots=re_knots,
        re_spline_boundary=re_boundary,
        interaction_pairs=interaction_pairs,
        precomputed_splines=True,
        n_tv=n_tv,
    ).to(device)

    # ---- Print model info ----
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: hidden_channels=8, enc=32, func=32, rho=16, dec_p=4")
    print(f"Total parameters: {total_params}")

    # ============================================================
    # STAGE 1: Freeze β from pure LMM ablation (no interaction)
    # ============================================================
    ablation_beta = torch.tensor([
        +29.2650,   # intercept
        -2.8255,    # ns1
        -0.0429,    # ns2
        -3.1169,    # ns3
        -0.1062,    # GLUC_t
        -0.175,    # BMI_t
        +1.8933,    # SEX_code
        -0.9311,    # AGEc
        +2.6670,    # DIPNIV2
        +3.4408,    # DIPNIV3
    ], dtype=torch.float32)

    with torch.no_grad():
        model.decoder._last_beta.copy_(ablation_beta.to(device))

    beta_names = ["intercept"] + [f"ns{i+1}" for i in range(model.decoder.fe_spline_df)]
    beta_names += x_cols[:n_tv] + static_cols

    print("\nFrozen β (from Stage 1 ablation):")
    for name, val in zip(beta_names, ablation_beta):
        print(f"  {name:>20s} = {val.item():+.4f}")

    # ============================================================
    # STAGE 2: Train CDE + neural head + D + σ²
    # ============================================================
    trainable_params = list(model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )

    print(f"\nTrainable parameters: {sum(p.numel() for p in trainable_params)}")
    print("(β is frozen — not in optimizer)\n")

    # ---- Train ----
    model.train()
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        total_nll = 0.0
        total_smooth = 0.0
        total_hmag = 0.0
        total_bn = 0.0
        count = 0

        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad  = t_pad.to(device)
            x_pad  = x_pad.to(device)
            y_pad  = y_pad.to(device)
            mask   = mask.to(device)
            c_mask = c_mask.to(device)
            s      = s.to(device)

            # Forward: y_pad=None → uses frozen _last_beta, no AtA/Atb
            mu, V, _, _, h = model(t_pad, x_pad, c_mask, s, mask, y_pad=None)

            # NLL
            loss_nll = masked_NLL(mu, y_pad, V, mask)

            # Regularization on h
            loss_reg = torch.tensor(0.0, device=device)

            if LAMBDA_SMOOTH > 0:
                ls = reg_temporal_smoothness(h, mask)
                loss_reg = loss_reg + LAMBDA_SMOOTH * ls
                total_smooth += ls.item()

            if LAMBDA_H_MAG > 0:
                lh = reg_h_magnitude(h, mask)
                loss_reg = loss_reg + LAMBDA_H_MAG * lh
                total_hmag += lh.item()

            if LAMBDA_BETA_NEURAL > 0:
                lb = reg_beta_neural(model)
                loss_reg = loss_reg + LAMBDA_BETA_NEURAL * lb
                total_bn += lb.item()

            loss = loss_nll + loss_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_nll += loss_nll.item()
            count += 1

        avg_nll = total_nll / max(count, 1)
        scheduler.step(avg_nll)

        if avg_nll < best_loss:
            best_loss = avg_nll
            # Exclude torchcde spline buffers (size varies by batch)
            best_state = {k: v.clone() for k, v in model.state_dict().items()
                         if 'X_spline' not in k}
            # ---- Save checkpoint ----
            # Exclude torchcde spline buffers (size varies by batch)
            clean_state = {k: v for k, v in model.state_dict().items() if 'X_spline' not in k}
            torch.save({
                'epoch': EPOCHS,
                'model_state_dict': clean_state,
                'loss': best_loss,
                'ablation_beta': ablation_beta,
                'config': {
                    'LAMBDA_SMOOTH': LAMBDA_SMOOTH,
                    'LAMBDA_H_MAG': LAMBDA_H_MAG,
                    'LAMBDA_BETA_NEURAL': LAMBDA_BETA_NEURAL,
                    'hidden_channels': cfg.hidden_channels,
                },
            }, 'best_model_twostage_reg.pt')

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            beta = model.decoder._last_beta.detach().cpu()
            beta_neural = model.decoder.beta_neural.detach().cpu()

            print(f"\nEpoch {epoch:5d} | NLL = {avg_nll:.4f} | best = {best_loss:.4f}")

            # Regularization losses
            reg_str = []
            if LAMBDA_SMOOTH > 0:
                reg_str.append(f"smooth={total_smooth/max(count,1):.4f}")
            if LAMBDA_H_MAG > 0:
                reg_str.append(f"h_mag={total_hmag/max(count,1):.4f}")
            if LAMBDA_BETA_NEURAL > 0:
                reg_str.append(f"bn={total_bn/max(count,1):.4f}")
            if reg_str:
                print(f"          | reg: {', '.join(reg_str)}")

            print("  --- Frozen β (should not change) ---")
            for name, val in zip(beta_names, beta):
                print(f"    {name:>20s} = {val.item():+.4f}")

            sig2 = torch.exp(model.decoder.log_residual_var).item()
            print(f"    {'sigma2':>20s} = {sig2:.4f}")

            print(f"  --- Neural contribution ---")
            print(f"    beta_neural norm = {beta_neural.norm().item():.4f}")
            print(f"    beta_neural = [{', '.join(f'{v:.4f}' for v in beta_neural)}]")

            # D matrix
            D = model.decoder._build_D(device=torch.device('cpu'), dtype=torch.float32)
            print(f"  --- D matrix ---")
            for i in range(D.shape[0]):
                print(f"    [{', '.join(f'{D[i,j]:.4f}' for j in range(D.shape[1]))}]")

    # ---- Restore best ----
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # ---- Final ----
    print("\n" + "="*60)
    print("FINAL RESULTS (Two-stage + regularization)")
    print("="*60)
    beta = model.decoder._last_beta.detach().cpu()
    for name, val in zip(beta_names, beta):
        print(f"  {name:>20s} = {val.item():+.6f}")
    bn = model.decoder.beta_neural.detach()
    print(f"  beta_neural norm = {bn.norm().item():.4f}")
    print(f"  beta_neural = [{', '.join(f'{v:.4f}' for v in bn.cpu())}]")

    # ---- PDP Analysis ----
    print("\n--- PDP Analysis ---")
    from pdp_analysis import compute_pdp, plot_pdp, plot_pdp_marginal, compute_delta_pdp

    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_pad)
    bmi_values = [20, 23, 26, 29, 32, 35]

    for interp in ["linear"]:
        print(f"\n  === interp = {interp} ===")
        results, ages, masks, times = compute_pdp(
            model, eval_loader, device, bmi_values, n_tv=n_tv, interp=interp
        )

        suffix = f"twostage_reg_{interp}"
        plot_pdp(results, ages, masks, times, bmi_values,
                 save_path=f"pdp_{suffix}_by_age.png")
        plot_pdp_marginal(results, masks, times, bmi_values,
                         save_path=f"pdp_{suffix}_marginal.png")
        delta = compute_delta_pdp(results, masks, bmi_lo=20, bmi_hi=35)

    bmi_idx = beta_names.index("BMI_t")
    beta_bmi = beta[bmi_idx].item()
    print(f"\n--- Summary ---")
    print(f"  β_BMI = {beta_bmi:.4f} (frozen from ablation)")
    print(f"  Parametric ΔPDP = {beta_bmi * 15:.4f}")
    print(f"  True ΔPDP ≈ -2.625")
    print(f"  Regularization: smooth={LAMBDA_SMOOTH}, h_mag={LAMBDA_H_MAG}, bn={LAMBDA_BETA_NEURAL}")