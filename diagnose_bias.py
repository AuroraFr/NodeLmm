"""
Diagnose bias: compute final training NLL, stationarity ratio, and ∆PDP bias
across all simulations. Outputs CSV + scatter plot for correlation analysis.

Usage:
    python diagnose_bias.py --n_sims 100
    python diagnose_bias.py --n_sims 100 --output_csv results/bias_diagnosis.csv
"""
import math, os, time
import torch
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader
import numpy as np
import pyreadr
import argparse
import csv

from dataset import LongitudinalDataset, collate_pad
from model_ODE import NeuralODEModel, NeuralODEConfig
from PDP_analysis_ODE import compute_pdp, compute_delta_pdp

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────
# NLL computation (batched, no grad)
# ─────────────────────────────────────────────────────────

def compute_total_nll(model, loader, device, jitter=1e-4):
    """
    Compute total NLL across all subjects (batched for speed).
    Returns total NLL, number of subjects, mean NLL per subject.
    """
    model.eval()
    total_nll = 0.0
    n_subjects = 0

    with torch.no_grad():
        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad = t_pad.to(device)
            x_pad = x_pad.to(device)
            y_pad = y_pad.to(device)
            mask = mask.to(device)
            s = s.to(device)
            bmi_t = x_pad[:, :, 0:1]

            mu, V, Z, D, sig2, _ = model(
                t_pad, x_pad, masks=None,
                static_covariates=s, bmi_t=bmi_t, obs_mask=mask
            )

            B = mu.shape[0]
            for i in range(B):
                m = mask[i].bool()
                n_i = m.sum()
                if n_i == 0:
                    continue

                mu_obs = mu[i, m]
                y_obs = y_pad[i, m]
                V_obs = V[i][m][:, m] + jitter * torch.eye(
                    n_i, device=device, dtype=mu.dtype)

                residual = y_obs - mu_obs
                L = torch.linalg.cholesky(V_obs)
                Vinv_r = torch.cholesky_solve(
                    residual.unsqueeze(-1), L).squeeze(-1)
                log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L)))

                nll_i = 0.5 * (log_det + residual @ Vinv_r
                               + n_i * math.log(2 * math.pi))
                total_nll += nll_i.item()
                n_subjects += 1

    mean_nll = total_nll / n_subjects if n_subjects > 0 else float('nan')
    return total_nll, n_subjects, mean_nll


# ─────────────────────────────────────────────────────────
# Stationarity ratio (lightweight — no full Fisher)
# ─────────────────────────────────────────────────────────

def compute_stationarity_ratio(model, dataset, device, collate_fn,
                                max_subjects=1000):
    """
    Compute ||mean(s)|| / ||mean(|s|)|| on a subset of subjects.
    """
    from torch.utils.data import DataLoader, Subset

    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)

    n_use = min(max_subjects, len(dataset))
    subset = Subset(dataset, list(range(n_use)))
    loader = DataLoader(subset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn)

    score_sum = torch.zeros(P)
    score_abs_sum = torch.zeros(P)
    n = 0

    for batch in loader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        y_pad = y_pad.to(device)
        mask = mask.to(device)
        s = s.to(device)
        bmi_t = x_pad[:, :, 0:1]

        if mask.sum() == 0:
            continue

        mu, V, Z, D, sig2, _ = model(
            t_pad, x_pad, masks=None,
            static_covariates=s, bmi_t=bmi_t, obs_mask=mask
        )

        # Per-subject NLL
        m = mask.squeeze(0).bool()
        n_i = m.sum()
        mu_obs = mu.squeeze(0)[m]
        y_obs = y_pad.squeeze(0)[m]
        V_obs = V.squeeze(0)[m][:, m] + 1e-4 * torch.eye(
            n_i, device=device, dtype=mu.dtype)
        residual = y_obs - mu_obs
        L = torch.linalg.cholesky(V_obs)
        Vinv_r = torch.cholesky_solve(residual.unsqueeze(-1), L).squeeze(-1)
        log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
        nll = 0.5 * (log_det + residual @ Vinv_r + n_i * math.log(2 * math.pi))

        grads = torch.autograd.grad(nll, params, retain_graph=False,
                                    allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, params)]
        s_i = torch.cat([g.reshape(-1) for g in grads]).cpu()

        score_sum += s_i
        score_abs_sum += s_i.abs()
        n += 1

    mean_score = score_sum / n
    mean_abs = score_abs_sum / n
    ratio = mean_score.norm() / mean_abs.norm()

    return ratio.item(), mean_score.norm().item()


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose bias across simulations")
    parser.add_argument("--n_sims", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--true_beta_bmi", type=float, default=-0.30)
    parser.add_argument("--true_beta_int", type=float, default=-0.05)
    parser.add_argument("--stationarity_subjects", type=int, default=1000,
                        help="Max subjects for stationarity ratio (speed)")
    parser.add_argument("--output_csv", type=str,
                        default="results/bias_diagnosis.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    time_col, y_col, id_col = "time", "ISA15_sim", "NUM_ID"
    x_cols = ["BMI_t", "rs1", "rs2"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    bmi_values = [20, 23, 26, 29, 32, 35]
    visit_times = np.array([0, 5, 10, 15])

    header = [
        'sim', 'mean_nll', 'stationarity_ratio', 'mean_score_norm',
        'bias_t0', 'bias_t5', 'bias_t10', 'bias_t15',
        'delta_pdp_t0', 'delta_pdp_t5', 'delta_pdp_t10', 'delta_pdp_t15',
        'true_t0', 'true_t5', 'true_t10', 'true_t15',
        'ckpt_epoch', 'ckpt_loss',
    ]
    rows = []

    for sim_idx in range(args.n_sims):
        data_path = f"simu_datasets/S2a_sims/sim_{sim_idx+1:03d}.rds"
        ckpt_path = f"checkpoints/simulation_baseline/best_model_ode_{sim_idx}.pt"

        print(f"\n{'='*60}")
        print(f"SIMULATION {sim_idx}")
        print(f"{'='*60}")

        # --- Data ---
        df = next(iter(pyreadr.read_r(data_path).values()))
        df["SEX"] = df["SEX"].astype("category")
        df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
        df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
        df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

        dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col,
                                      static_cols=static_cols)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_pad)

        # --- Model ---
        cfg = NeuralODEConfig(
            hidden_channels=8, enc_mlp_hidden=32, func_mlp_hidden=32,
            dec_rho_hidden=16, dec_p=4, dec_q=3, depth=2, dropout=0.0,
            euler_steps_per_interval=4,
        )
        model = NeuralODEModel(
            x_dim=len(x_cols), static_dim=len(static_cols), cfg=cfg,
            n_tv=1, use_rho_net=True, use_neural_re=True,
            re_spline_cols=[1, 2], g_hidden=16, fullD=True,
            bmi_mean=0.0, bmi_std=1.0, static_skip_dims=[1],
        ).to(device)

        checkpoint = torch.load(ckpt_path, map_location=device,
                                weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            ckpt_epoch = checkpoint.get('epoch', -1)
            ckpt_loss = checkpoint.get('best_test_loss', float('nan'))
        else:
            model.load_state_dict(checkpoint, strict=False)
            ckpt_epoch = -1
            ckpt_loss = float('nan')
        model.eval()
        print(f"  Loaded: {ckpt_path} (epoch={ckpt_epoch}, loss={ckpt_loss:.4f})")

        # --- 1. Training NLL ---
        total_nll, n_subj, mean_nll = compute_total_nll(model, loader, device)
        print(f"  NLL: total={total_nll:.2f}, mean={mean_nll:.4f}, n={n_subj}")

        # --- 2. Stationarity ratio ---
        ratio, mean_s_norm = compute_stationarity_ratio(
            model, dataset, device, collate_fn=collate_pad,
            max_subjects=args.stationarity_subjects)
        print(f"  Stationarity: ratio={ratio:.4f}, ||mean(s)||={mean_s_norm:.4f}")

        # --- 3. ∆PDP bias ---
        results, ages, masks, times = compute_pdp(
            model, loader, device, bmi_values,
            bmi_col=0, bmi_mode="constant",
        )
        estimated, true_ref = compute_delta_pdp(
            results, ages, masks, times,
            bmi_lo=20, bmi_hi=35,
            true_beta_bmi=args.true_beta_bmi,
            true_beta_int=args.true_beta_int,
            visit_times=visit_times,
        )

        biases = {}
        for vt in visit_times:
            est = estimated.get(vt, float('nan'))
            tru = true_ref.get(vt, float('nan'))
            biases[vt] = est - tru
        print(f"  Bias: {', '.join(f't{int(vt)}={biases[vt]:+.4f}' for vt in visit_times)}")

        # --- Collect row ---
        row = {
            'sim': sim_idx,
            'mean_nll': mean_nll,
            'stationarity_ratio': ratio,
            'mean_score_norm': mean_s_norm,
            'bias_t0': biases[0], 'bias_t5': biases[5],
            'bias_t10': biases[10], 'bias_t15': biases[15],
            'delta_pdp_t0': estimated.get(0, float('nan')),
            'delta_pdp_t5': estimated.get(5, float('nan')),
            'delta_pdp_t10': estimated.get(10, float('nan')),
            'delta_pdp_t15': estimated.get(15, float('nan')),
            'true_t0': true_ref.get(0, float('nan')),
            'true_t5': true_ref.get(5, float('nan')),
            'true_t10': true_ref.get(10, float('nan')),
            'true_t15': true_ref.get(15, float('nan')),
            'ckpt_epoch': ckpt_epoch,
            'ckpt_loss': ckpt_loss,
        }
        rows.append(row)

    # ─────────────────────────────────────────────────────────
    # Save CSV
    # ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_csv)
                if os.path.dirname(args.output_csv) else '.', exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {args.output_csv}")

    # ─────────────────────────────────────────────────────────
    # Summary statistics
    # ─────────────────────────────────────────────────────────
    biases_t0 = np.array([r['bias_t0'] for r in rows])
    nlls = np.array([r['mean_nll'] for r in rows])
    ratios = np.array([r['stationarity_ratio'] for r in rows])
    abs_biases = np.abs(biases_t0)

    print(f"\n{'='*60}")
    print(f"SUMMARY (D={len(rows)})")
    print(f"{'='*60}")
    print(f"  |bias_t0|: mean={abs_biases.mean():.4f}, "
          f"median={np.median(abs_biases):.4f}, "
          f"max={abs_biases.max():.4f}")
    print(f"  mean_nll:  mean={nlls.mean():.4f}, std={nlls.std():.4f}")
    print(f"  ratio:     mean={ratios.mean():.4f}, std={ratios.std():.4f}")

    # Correlation
    corr_nll = np.corrcoef(abs_biases, nlls)[0, 1]
    corr_ratio = np.corrcoef(abs_biases, ratios)[0, 1]
    print(f"\n  Corr(|bias_t0|, mean_nll)    = {corr_nll:+.4f}")
    print(f"  Corr(|bias_t0|, ratio)       = {corr_ratio:+.4f}")

    # Flag high-bias simulations
    threshold = 1.0
    high_bias = [r for r in rows if abs(r['bias_t0']) > threshold]
    print(f"\n  Simulations with |bias_t0| > {threshold}: "
          f"{len(high_bias)} / {len(rows)}")
    if high_bias:
        print(f"  {'sim':>5s}  {'bias_t0':>10s}  {'mean_nll':>10s}  "
              f"{'ratio':>8s}  {'epoch':>6s}  {'ckpt_loss':>10s}")
        print(f"  {'-'*56}")
        for r in sorted(high_bias, key=lambda x: abs(x['bias_t0']),
                         reverse=True):
            print(f"  {r['sim']:5d}  {r['bias_t0']:+10.4f}  "
                  f"{r['mean_nll']:10.4f}  {r['stationarity_ratio']:8.4f}  "
                  f"{r['ckpt_epoch']:6d}  {r['ckpt_loss']:10.4f}")