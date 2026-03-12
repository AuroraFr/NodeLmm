"""
Hybrid approach: analytical β + CDE + regularization.
With train/test split — save model when TEST loss improves.

W = [1, (optional: splines), GLUC, BMI, SEX, AGE, DIPNIV2, DIPNIV3]
CDE learns residual structure (time trend, interactions, nonlinearities).
"""
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pyreadr
from dataset import LongitudinalDataset, collate_pad
from model_hybrid import NeuralCDEModel, NeuralCDEConfig
from utils import masked_NLL

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


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


def eval_epoch(model, loader, device, n_W):
    """Evaluate NLL on a dataset (no gradient, no regularization)."""
    model.eval()
    total_nll = 0.0
    count = 0

    # Accumulate AtA/Atb for analytical beta on eval set
    AtA_eval = torch.zeros(n_W, n_W, device=device)
    Atb_eval = torch.zeros(n_W, device=device)

    with torch.no_grad():
        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad  = t_pad.to(device)
            x_pad  = x_pad.to(device)
            y_pad  = y_pad.to(device)
            mask   = mask.to(device)
            c_mask = c_mask.to(device)
            s      = s.to(device)

            mu, V, AtA_batch, Atb_batch, h = model(
                t_pad, x_pad, c_mask, s, mask, y_pad=y_pad
            )
            loss_nll = masked_NLL(mu, y_pad, V, mask)

            total_nll += loss_nll.item()
            count += 1

            if AtA_batch is not None:
                AtA_eval += AtA_batch
                Atb_eval += Atb_batch

    model.train()
    return total_nll / max(count, 1), AtA_eval, Atb_eval


if __name__ == "__main__":

    # ================================================================
    # CONFIG
    # ================================================================
    LAMBDA_SMOOTH      = 0.1
    LAMBDA_H_MAG       = 0.01
    LAMBDA_BETA_NEURAL = 0.0

    LR = 1e-3
    WD = 1e-5
    EPOCHS = 500
    BATCH_SIZE = 128
    PRINT_EVERY = 25
    TEST_RATIO = 0.2
    SEED = 42
    patience = 300

    print("="*60)
    print("HYBRID: Analytical β + CDE + regularization")
    print("="*60)
    print(f"  LAMBDA_SMOOTH      = {LAMBDA_SMOOTH}")
    print(f"  LAMBDA_H_MAG       = {LAMBDA_H_MAG}")
    print(f"  LAMBDA_BETA_NEURAL = {LAMBDA_BETA_NEURAL}")
    print(f"  TEST_RATIO         = {TEST_RATIO}")
    print()

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

    full_dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col, static_cols=static_cols)

    # ---- Train/test split (subject level) ----
    N = len(full_dataset)
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(N)
    n_test = int(N * TEST_RATIO)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset  = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad)

    print(f"Subjects: {N} total → {len(train_idx)} train, {len(test_idx)} test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Spline knots ----
    fe_knots    = np.array([1.769863, 6.693151])
    fe_boundary = np.array([0.0, 13.50685])
    re_knots    = np.array([3.567123])
    re_boundary = np.array([0.0, 13.50685])

    n_tv = 2
    interaction_pairs = None

    # ---- Model ----
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
        fe_spline_knots=fe_knots,
        fe_spline_boundary=fe_boundary,
        re_spline_knots=re_knots,
        re_spline_boundary=re_boundary,
        interaction_pairs=interaction_pairs,
        precomputed_splines=True,
        n_tv=n_tv,
        include_fe_splines=False,
        use_rho_net=False,
    ).to(device)

    # ---- Print model info ----
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Decoder n_W={model.decoder.n_W}, use_rho_net={model.decoder.use_rho_net}")

    beta_names = ["intercept"]
    beta_names += x_cols[:n_tv] + static_cols
    if interaction_pairs:
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
    best_test_loss = float("inf")
    best_state = None
    n_W = model.decoder.n_W

    for epoch in range(1, EPOCHS + 1):

        if patience == 0:
            print('patiences reached')
            break
        # ---- Training pass ----
        model.train()
        total_nll = 0.0
        total_smooth = 0.0
        total_hmag = 0.0
        total_bn = 0.0
        count = 0

        AtA_epoch = torch.zeros(n_W, n_W, device=device)
        Atb_epoch = torch.zeros(n_W, device=device)

        for batch in train_loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad  = t_pad.to(device)
            x_pad  = x_pad.to(device)
            y_pad  = y_pad.to(device)
            mask   = mask.to(device)
            c_mask = c_mask.to(device)
            s      = s.to(device)

            mu, V, AtA_batch, Atb_batch, h = model(
                t_pad, x_pad, c_mask, s, mask, y_pad=y_pad
            )

            loss_nll = masked_NLL(mu, y_pad, V, mask)

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

            if AtA_batch is not None:
                AtA_epoch += AtA_batch
                Atb_epoch += Atb_batch

        # Solve beta from full training set
        model.decoder.solve_beta(AtA_epoch, Atb_epoch)

        avg_train_nll = total_nll / max(count, 1)

        # ---- Test evaluation ----
        avg_test_nll, _, _ = eval_epoch(model, test_loader, device, n_W)

        scheduler.step(avg_test_nll)

        # Save if test loss improves
        if avg_test_nll < best_test_loss:
            best_test_loss = avg_test_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()
                         if 'X_spline' not in k}
            patience = 300
        else:
            patience = patience - 1

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            beta = model.decoder._last_beta.detach().cpu()
            if model.decoder.use_rho_net:
                neural_w = model.decoder.beta_neural.detach().cpu()
            else:
                neural_w = model.decoder.w_neural.detach().cpu()

            print(f"\nEpoch {epoch:5d} | train NLL = {avg_train_nll:.4f} | "
                  f"test NLL = {avg_test_nll:.4f} | best test = {best_test_loss:.4f}")

            reg_str = []
            if LAMBDA_SMOOTH > 0:
                reg_str.append(f"smooth={total_smooth/max(count,1):.4f}")
            if LAMBDA_H_MAG > 0:
                reg_str.append(f"h_mag={total_hmag/max(count,1):.4f}")
            if reg_str:
                print(f"          | reg: {', '.join(reg_str)}")

            print("  --- Analytical β ---")
            for name, val in zip(beta_names, beta):
                print(f"    {name:>20s} = {val.item():+.4f}")

            sig2 = torch.exp(model.decoder.log_residual_var).item()
            print(f"    {'sigma2':>20s} = {sig2:.4f}")

            print(f"  --- Neural contribution ---")
            print(f"    neural weight norm = {neural_w.norm().item():.4f}")

    # ---- Restore best (by test loss) ----
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        print(f"\nRestored best model (test NLL = {best_test_loss:.4f})")

    # ---- Save checkpoint ----
    clean_state = {k: v for k, v in model.state_dict().items() if 'X_spline' not in k}
    torch.save({
        'model_state_dict': clean_state,
        'best_test_loss': best_test_loss,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'config': {
            'LAMBDA_SMOOTH': LAMBDA_SMOOTH,
            'LAMBDA_H_MAG': LAMBDA_H_MAG,
            'LAMBDA_BETA_NEURAL': LAMBDA_BETA_NEURAL,
            'hidden_channels': cfg.hidden_channels,
            'include_fe_splines': False,
            'use_rho_net': False,
            'interaction_pairs': interaction_pairs,
        },
    }, 'best_model_hybrid_reg.pt')

    # ---- Final ----
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    beta = model.decoder._last_beta.detach().cpu()
    for name, val in zip(beta_names, beta):
        print(f"  {name:>20s} = {val.item():+.6f}")

    # ---- PDP ----
    print("\n--- PDP Analysis ---")
    from pdp_analysis import (compute_pdp, plot_pdp, plot_pdp_marginal,
                              compute_delta_pdp, compute_delta_pdp_stratified)

    eval_loader = DataLoader(full_dataset, batch_size=256, shuffle=False, collate_fn=collate_pad)
    bmi_values = [20, 23, 26, 29, 32, 35]

    results, ages, masks, times = compute_pdp(
        model, eval_loader, device, bmi_values, n_tv=n_tv, interp="linear"
    )
    plot_pdp(results, ages, masks, times, bmi_values, save_path="pdp_hybrid_reg_by_age.png")
    plot_pdp_marginal(results, masks, times, bmi_values, save_path="pdp_hybrid_reg_marginal.png")
    delta = compute_delta_pdp(results, masks, bmi_lo=20, bmi_hi=35)
    compute_delta_pdp_stratified(results, ages, masks, times, bmi_lo=20, bmi_hi=35)

    bmi_idx = beta_names.index("BMI_t")
    beta_bmi = beta[bmi_idx].item()
    print(f"\n--- Summary ---")
    print(f"  β_BMI = {beta_bmi:.4f}")
    print(f"  Parametric ΔPDP = {beta_bmi * 15:.4f}")
    print(f"  True ΔPDP ≈ -2.625")