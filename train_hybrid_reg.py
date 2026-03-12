"""
Hybrid approach on Scenario 2:
  - W = [1, GLUC, BMI, SEX, AGE, DIPNIV2, DIPNIV3] — no splines, no interaction
  - β computed analytically (GLS) per epoch, accumulated across batches
  - CDE learns: time trend + BMI×AGE interaction + residual structure
  - Regularization on h(z(t)) to control neural leakage into β

x_pad layout: ["GLUC_t", "BMI_t", "ns1", "ns2", "ns3", "rs1", "rs2"]
              |--- CDE sees ---|  |------- RE splines (decoder) ------|
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pyreadr
from dataset import LongitudinalDataset, collate_pad
from model_hybrid_reg import NeuralCDEModel, NeuralCDEConfig
from utils import masked_NLL
import random
from Orthogonalization import reg_orthogonality

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def seed_worker(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

def reg_temporal_smoothness(h, obs_mask):
    """Penalize |h(t+1) - h(t)|² at observed positions."""
    dh = h[:, 1:] - h[:, :-1]
    mask_pairs = obs_mask[:, 1:] * obs_mask[:, :-1]
    n_pairs = mask_pairs.sum().clamp(min=1)
    return (dh ** 2 * mask_pairs).sum() / n_pairs


def reg_h_magnitude(h, obs_mask):
    """Penalize |h(t)|² at observed positions."""
    n_obs = obs_mask.sum().clamp(min=1)
    return (h ** 2 * obs_mask).sum() / n_obs

def reg_beta_neural(model):
    """L2 penalty on neural readout weights."""
    if model.decoder.use_rho_net:
        return model.decoder.beta_neural.pow(2).sum()
    else:
        return model.decoder.w_neural.pow(2).sum()

if __name__ == "__main__":

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ================================================================
    # REGULARIZATION CONFIG
    # ================================================================
    LAMBDA_SMOOTH      = 0.0     # temporal smoothness on h(z(t))
    LAMBDA_H_MAG       = 0.0    # L2 on h(z(t)) magnitude
    LAMBDA_BETA_NEURAL = 0.0    # L2 on beta_neural

    print("="*60)
    print("HYBRID: Analytical β + CDE (no FE splines in W)")
    print("="*60)
    print(f"  LAMBDA_SMOOTH      = {LAMBDA_SMOOTH}")
    print(f"  LAMBDA_H_MAG       = {LAMBDA_H_MAG}")
    print(f"  LAMBDA_BETA_NEURAL = {LAMBDA_BETA_NEURAL}")
    print()

    # ---- Training config ----
    LR = 1e-3
    WD = 1e-5
    EPOCHS = 1000
    BATCH_SIZE = 128
    PRINT_EVERY = 25
    WARMUP_EPOCHS = 0

    # ---- Data ----
    time_col = "time"
    y_col = "ISA15_sim"
    id_col = "NUM_ID"

    # x_pad: GLUC, BMI for CDE; ns1-ns3, rs1-rs2 for RE splines only
    x_cols = ["BMI_t", "ns1", "ns2", "ns3", "rs1", "rs2"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    path = "simu_datasets/S2a_sims_2/sim_001.rds"
    df = next(iter(pyreadr.read_r(path).values()))
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col, static_cols=static_cols)
    # loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=collate_pad,
                    worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(SEED))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Spline knots (only for RE, FE splines not in W) ----
    fe_knots    = np.array([1.769863, 6.693151])     # needed for precomputed_splines slicing
    fe_boundary = np.array([0.0, 13.50685])
    re_knots    = np.array([3.567123])
    re_boundary = np.array([0.0, 13.50685])

    n_tv = 1
    interaction_pairs = [(0, 1)]  # no interaction in W

    # ---- Model ----
    cfg = NeuralCDEConfig(
        hidden_channels=4,
        enc_mlp_hidden=16,
        func_mlp_hidden=16,
        dec_rho_hidden=16,
        dec_p=4,
        dec_q=3,
        depth=1,
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
        include_fe_splines=False,    # CDE learns time trend instead
        use_rho_net=False
    ).to(device)

    # ---- Print model info ----
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Decoder n_tv={model.decoder.n_tv}, n_W={model.decoder.n_W}")

    # Beta layout (no FE splines)
    beta_names = ["intercept"]
    beta_names += x_cols[:n_tv] + static_cols
    if interaction_pairs:
        for tv_i, s_i in interaction_pairs:
            beta_names.append(f"{x_cols[tv_i]}x{static_cols[s_i]}")
    print(f"Beta layout ({len(beta_names)}): {beta_names}")
    print(f"W = [1, GLUC, BMI, SEX, AGE, DIPNIV2, DIPNIV3] — no splines")
    print(f"CDE must learn: time trend + interaction")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )

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

        # Accumulate AtA/Atb across all batches
        n_W = model.decoder.n_W
        AtA_epoch = torch.zeros(n_W, n_W, device=device)
        Atb_epoch = torch.zeros(n_W, device=device)

        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad  = t_pad.to(device)
            x_pad  = x_pad.to(device)
            y_pad  = y_pad.to(device)
            mask   = mask.to(device)
            c_mask = c_mask.to(device)
            s      = s.to(device)

            # Forward with y_pad to get AtA/Atb, but beta uses previous epoch's value
            mu, V, AtA_batch, Atb_batch, h = model(
                t_pad, x_pad, c_mask, s, mask, y_pad=y_pad
            )

            # NLL
            loss_nll = masked_NLL(mu, y_pad, V, mask)

            
            LAMBDA_ORTH = 0.1

            # Inside your batch loop, after the forward pass:
            W = model.decoder._build_W(t_pad, x_pad, s)
            loss_orth = reg_orthogonality(h, W, mask)
            loss = loss_nll + LAMBDA_ORTH * loss_orth

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_nll += loss_nll.item()
            count += 1

            # Accumulate (already detached in _compute_analytical_beta)
            if AtA_batch is not None:
                AtA_epoch += AtA_batch
                Atb_epoch += Atb_batch

        # Solve beta from full-epoch accumulation
        if epoch > WARMUP_EPOCHS:
            model.decoder.solve_beta(AtA_epoch, Atb_epoch)

        avg_nll = total_nll / max(count, 1)
        scheduler.step(avg_nll)

        if avg_nll < best_loss:
            best_loss = avg_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()
                         if 'X_spline' not in k}
            # ---- Save checkpoint ----
            clean_state = {k: v for k, v in model.state_dict().items() if 'X_spline' not in k}
            torch.save({
                'epoch': EPOCHS,
                'model_state_dict': clean_state,
                'loss': best_loss,
                'config': {
                    'LAMBDA_SMOOTH': LAMBDA_SMOOTH,
                    'LAMBDA_H_MAG': LAMBDA_H_MAG,
                    'LAMBDA_BETA_NEURAL': LAMBDA_BETA_NEURAL,
                    'hidden_channels': cfg.hidden_channels,
                    'approach': 'hybrid_no_fe_splines',
                },
            }, 'checkpoints/best_model_hybrid.pt')

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            beta = model.decoder._last_beta.detach().cpu()
            if model.decoder.use_rho_net:
                beta_neural = model.decoder.beta_neural.detach().cpu()
            else:
                beta_neural = model.decoder.w_neural.detach().cpu()

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

            print("  --- Analytical β ---")
            for name, val in zip(beta_names, beta):
                print(f"    {name:>20s} = {val.item():+.4f}")

            sig2 = torch.exp(model.decoder.log_residual_var).item()
            print(f"    {'sigma2':>20s} = {sig2:.4f}")
            
            # Diagonal D
            print(model.decoder.log_std, model.decoder.fullD)
            if model.decoder.log_std is not None:
                std = torch.exp(model.decoder.log_std).detach().cpu()
                D_diag = std ** 2
                print(f"  D diag = [{', '.join(f'{v:.4f}' for v in D_diag)}]")

            # Full D
            if model.decoder.L_unconstrained is not None:
                D = model.decoder._build_D(device=torch.device('cpu'), dtype=torch.float32)
                print(f"  D matrix:")
                for i in range(D.shape[0]):
                    print(f"    [{', '.join(f'{D[i,j]:.4f}' for j in range(D.shape[1]))}]")

            print(f"  --- Neural contribution ---")
            print(f"    beta_neural norm = {beta_neural.norm().item():.4f}")
            print(f"    beta_neural = [{', '.join(f'{v:.4f}' for v in beta_neural)}]")

    # ---- Restore best ----
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # ---- Final ----
    print("\n" + "="*60)
    print("FINAL RESULTS (Hybrid + regularization)")
    print("="*60)
    beta = model.decoder._last_beta.detach().cpu()
    for name, val in zip(beta_names, beta):
        print(f"  {name:>20s} = {val.item():+.6f}")

    if model.decoder.use_rho_net:
        bn = model.decoder.beta_neural.detach()
    else:
        bn = model.decoder.w_neural.detach()
    print(f"  neural weight norm = {bn.norm().item():.4f}")

    print(f"\nTrue β₀ (Scenario 2):")
    print(f"  BMI = -0.175, GLUC = -0.95, SEX = 1.85, AGE = -0.51")
    print(f"  DIPNIV2 = 2.67, DIPNIV3 = 3.32")

    # ---- PDP Analysis ----
    print("\n--- PDP Analysis ---")
    from pdp_analysis import (compute_pdp, plot_pdp, plot_pdp_marginal,
                              compute_delta_pdp, compute_delta_pdp_stratified)

    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_pad)
    bmi_values = [20, 23, 26, 29, 32, 35]
    bmi_col = x_cols.index("BMI_t")  # = 0 now

    results, ages, masks, times = compute_pdp(
        model, eval_loader, device, bmi_values, n_tv=n_tv, interp="linear",bmi_col=bmi_col
    )

    plot_pdp(results, ages, masks, times, bmi_values,
             save_path="figures/pdp_hybrid_reg_by_age.png")
    plot_pdp_marginal(results, masks, times, bmi_values,
                     save_path="figures/pdp_hybrid_reg_marginal.png")
    delta = compute_delta_pdp(results, masks, bmi_lo=20, bmi_hi=35)
    compute_delta_pdp_stratified(results, ages, masks, times, bmi_lo=20, bmi_hi=35)

    bmi_idx = beta_names.index("BMI_t")
    beta_bmi = beta[bmi_idx].item()
    print(f"\n--- Summary ---")
    print(f"  β_BMI = {beta_bmi:.4f} (analytical, should be ≈ -0.30)")
    print(f"  Parametric ΔPDP = {beta_bmi * 15:.4f}")
    print(f"  True ΔPDP ≈ ", -0.30*15)
    print(f"  Regularization: smooth={LAMBDA_SMOOTH}, h_mag={LAMBDA_H_MAG}")