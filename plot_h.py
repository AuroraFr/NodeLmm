"""
Plot the neural contribution h(z(t)) over time.
- Grey lines: individual subject h trajectories
- Red line: mean h(t) across subjects at each observed time
- Blue dashed: smoothed mean trend

Usage:
    python plot_h_over_time.py --checkpoint best_model.pt --data_path <path_to_data>

Adapt the imports and data loading to match your project structure.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from model_hybrid_reg import NeuralCDEModel, NeuralCDEConfig
from dataset import LongitudinalDataset, collate_pad
from torch.utils.data import DataLoader
import pyreadr

# ============================================================
# TRUE PARAMETERS (Scenario 2)
# ============================================================

TRUE_BETA = {
    "intercept": 30.712,
    "ns1":       -2.817,
    "ns2":        0.020,
    "ns3":       -3.135,
    "BMI":       -0.175,
    "GLUC":      -0.954,
    "AGEc":      -0.510,
    "SEX":        1.845,
    "DIPNIV2":    2.665,
    "DIPNIV3":    3.324,
    "BMI_x_AGEc": -0.015,
}

TRUE_BETA = {
    "intercept": 30.71,
    "ns1":       -2.817,
    "ns2":        0.020,
    "ns3":       -3.135,
    "BMI":       -0.30,
    "AGEc":      -0.510,
    "SEX":        1.845,
    "DIPNIV2":    2.665,
    "DIPNIV3":    3.324,
    "BMI_x_AGEc": -0.05,
}


def extract_h_and_truth(model, dataloader, device="cpu",
                        true_beta=None,
                        bmi_col=0, ns_cols=None, age_static_col=1,
                        interaction_in_W=False):
    """
    For each subject at each observed time:
      - h_learned: from the model's neural component
      - h_true_time: beta1*ns1 + beta2*ns2 + beta3*ns3  (time trend only)
      - h_true_target: what h SHOULD learn given current W
        - if interaction_in_W: same as h_true_time
        - if not: h_true_time + beta_int * BMI(t) * AGEc

    Args:
        true_beta: dict with keys "ns1","ns2","ns3","BMI_x_AGEc"
        bmi_col:   index of BMI in x_pad
        ns_cols:   list of indices for ns1,ns2,ns3 in x_pad (e.g. [1,2,3])
        age_static_col: index of AGEc in static (default 1)
        interaction_in_W: if True, interaction is parametric -> h only learns splines
    """
    if true_beta is None:
        true_beta = {
            "ns1": -2.817, "ns2": 0.020, "ns3": -3.135,
            "BMI_x_AGEc": -0.05,
        }

    if ns_cols is None:
        ns_cols = [1, 2, 3]

    model.eval()

    all_times = []
    all_h_learned = []
    all_h_true_time = []
    all_h_true_target = []
    all_ages = []

    with torch.no_grad():
        for batch in dataloader:
            ids, t_pad, x_pad, y_pad, c_mask, mask, static = batch
            t_pad  = t_pad.to(device)
            x_pad  = x_pad.to(device)
            y_pad  = y_pad.to(device)
            mask   = mask.to(device)
            c_mask = c_mask.to(device)
            static = static.to(device)

            mu, V, AtA, Atb, h = model(t_pad, x_pad, c_mask, static, mask, y_pad=y_pad)

            h_np = h.cpu().numpy()
            mask_np = mask.cpu().numpy()
            t_np = t_pad.cpu().numpy()
            x_np = x_pad.cpu().numpy()
            s_np = static.cpu().numpy()

            B = h_np.shape[0]
            for i in range(B):
                obs = mask_np[i] > 0.5
                ti = t_np[i, obs]
                hi = h_np[i, obs]

                ns1 = x_np[i, obs, ns_cols[0]]
                ns2 = x_np[i, obs, ns_cols[1]]
                ns3 = x_np[i, obs, ns_cols[2]]

                bmi = x_np[i, obs, bmi_col]
                agec = s_np[i, age_static_col]

                h_time = (true_beta["ns1"] * ns1 +
                          true_beta["ns2"] * ns2 +
                          true_beta["ns3"] * ns3)

                if interaction_in_W:
                    h_target = h_time.copy()
                else:
                    h_interaction = true_beta["BMI_x_AGEc"] * bmi * agec
                    h_target = h_time + h_interaction

                all_times.append(ti)
                all_h_learned.append(hi)
                all_h_true_time.append(h_time)
                all_h_true_target.append(h_target)
                all_ages.append(agec)

    return {
        "all_times": all_times,
        "all_h_learned": all_h_learned,
        "all_h_true_time": all_h_true_time,
        "all_h_true_target": all_h_true_target,
        "all_ages": all_ages,
        "interaction_in_W": interaction_in_W,
    }


def plot_h_comparison(all_times, all_h_learned, all_h_true_time, all_h_true_target,
                      all_ages, interaction_in_W=False,
                      save_path="h_comparison.png"):
    """
    Three-panel plot:
      Left:   mean learned h vs mean true target over time
      Middle: scatter of individual h_learned vs h_true_target
      Right:  h by age group to check interaction capture
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    t_flat = np.concatenate(all_times)
    h_flat = np.concatenate(all_h_learned)
    ht_flat = np.concatenate(all_h_true_time)
    htarget_flat = np.concatenate(all_h_true_target)
    ages_flat = np.concatenate([np.full_like(all_times[i], all_ages[i])
                                 for i in range(len(all_ages))])

    # ---- Panel 1: Mean over time ----
    ax = axes[0]
    n_bins = 40
    t_min, t_max = t_flat.min(), t_flat.max()
    edges = np.linspace(t_min, t_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mean_learned = np.full(n_bins, np.nan)
    mean_true_time = np.full(n_bins, np.nan)
    mean_true_target = np.full(n_bins, np.nan)

    for b in range(n_bins):
        m = (t_flat >= edges[b]) & (t_flat < edges[b + 1])
        if m.sum() > 5:
            mean_learned[b] = h_flat[m].mean()
            mean_true_time[b] = ht_flat[m].mean()
            mean_true_target[b] = htarget_flat[m].mean()

    valid = ~np.isnan(mean_learned)
    ax.plot(centers[valid], mean_learned[valid], "r-", linewidth=2.5, label="Learned h(z(t))")

    if interaction_in_W:
        # h should match splines only — show one reference line
        ax.plot(centers[valid], mean_true_time[valid], "b--", linewidth=2,
                label="True target (splines only)")
    else:
        # h should match splines + interaction — show both references
        ax.plot(centers[valid], mean_true_time[valid], "b--", linewidth=2,
                label="True (splines only)")
        ax.plot(centers[valid], mean_true_target[valid], "g--", linewidth=2,
                label="True (splines + interaction)")

    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("h", fontsize=12)
    ax.set_title("Population mean: learned vs true", fontsize=13)
    ax.legend(fontsize=9)

    # ---- Panel 2: Scatter h_learned vs h_true_target ----
    ax2 = axes[1]
    n_plot = min(len(h_flat), 10000)
    idx = np.random.choice(len(h_flat), n_plot, replace=False)
    ax2.scatter(htarget_flat[idx], h_flat[idx], alpha=0.05, s=3, color="steelblue")

    lo = min(htarget_flat[idx].min(), h_flat[idx].min())
    hi_val = max(htarget_flat[idx].max(), h_flat[idx].max())
    ax2.plot([lo, hi_val], [lo, hi_val], "r--", linewidth=1.5, label="y = x")

    from numpy.polynomial.polynomial import polyfit
    c = polyfit(htarget_flat[idx], h_flat[idx], 1)
    x_fit = np.linspace(lo, hi_val, 100)
    ax2.plot(x_fit, c[0] + c[1] * x_fit, "g-", linewidth=1.5,
             label=f"fit: {c[1]:.2f}x + {c[0]:.2f}")

    corr = np.corrcoef(htarget_flat[idx], h_flat[idx])[0, 1]

    target_label = "True target (splines only)" if interaction_in_W else "True target (splines + interaction)"
    ax2.set_xlabel(target_label, fontsize=12)
    ax2.set_ylabel("Learned h(z(t))", fontsize=12)
    ax2.set_title(f"Learned vs true target (r = {corr:.3f})", fontsize=13)
    ax2.legend(fontsize=9)

    # ---- Panel 3: Mean h by age group over time ----
    ax3 = axes[2]

    age_q = np.percentile(ages_flat, [33, 67])
    age_groups = [
        ("Young (AGEc < {:.1f})".format(age_q[0]), ages_flat < age_q[0]),
        ("Middle", (ages_flat >= age_q[0]) & (ages_flat < age_q[1])),
        ("Old (AGEc > {:.1f})".format(age_q[1]), ages_flat >= age_q[1]),
    ]
    colors_learned = ["#2196F3", "#FF9800", "#F44336"]
    colors_true = ["#90CAF9", "#FFE0B2", "#FFCDD2"]

    for (label, mask_age), c_learn, c_true in zip(age_groups, colors_learned, colors_true):
        t_sub = t_flat[mask_age]
        h_sub = h_flat[mask_age]
        htarget_sub = htarget_flat[mask_age]

        bin_h = np.full(n_bins, np.nan)
        bin_ht = np.full(n_bins, np.nan)
        for b in range(n_bins):
            m = (t_sub >= edges[b]) & (t_sub < edges[b + 1])
            if m.sum() > 5:
                bin_h[b] = h_sub[m].mean()
                bin_ht[b] = htarget_sub[m].mean()

        v = ~np.isnan(bin_h)
        ax3.plot(centers[v], bin_h[v], "-", color=c_learn, linewidth=2,
                 label=f"Learned: {label}")
        ax3.plot(centers[v], bin_ht[v], "--", color=c_true, linewidth=2,
                 label=f"True: {label}")

    ax3.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax3.set_xlabel("Time (years)", fontsize=12)
    ax3.set_ylabel("h", fontsize=12)
    title_suffix = "(h = splines only)" if interaction_in_W else "(h = splines + interaction)"
    ax3.set_title(f"h by age group {title_suffix}", fontsize=13)
    ax3.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {save_path}")

    # ---- Print summary ----
    print(f"\n--- Comparison summary ---")
    print(f"  Interaction in W: {interaction_in_W}")
    print(f"  Correlation (learned, target):      {np.corrcoef(h_flat, htarget_flat)[0,1]:.4f}")
    print(f"  Correlation (learned, spline only):  {np.corrcoef(h_flat, ht_flat)[0,1]:.4f}")
    print(f"  Learned h:    mean={h_flat.mean():.3f}, std={h_flat.std():.3f}, "
          f"range=[{h_flat.min():.3f}, {h_flat.max():.3f}]")
    print(f"  True target:  mean={htarget_flat.mean():.3f}, std={htarget_flat.std():.3f}, "
          f"range=[{htarget_flat.min():.3f}, {htarget_flat.max():.3f}]")
    print(f"  True splines: mean={ht_flat.mean():.3f}, std={ht_flat.std():.3f}, "
          f"range=[{ht_flat.min():.3f}, {ht_flat.max():.3f}]")
    print(f"  Scale ratio (learned_std / target_std): {h_flat.std() / (htarget_flat.std() + 1e-10):.3f}")

if __name__ == "__main__":

    # ---- Data ----
    time_col = "time"
    y_col = "ISA15_sim"
    id_col = "NUM_ID"

    # x_pad: GLUC, BMI for CDE; ns1-ns3, rs1-rs2 for RE splines only
    x_cols = ["BMI_t", "ns1", "ns2", "ns3", "rs1", "rs2"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    path = "simu_datasets/S2a_sim/sim_001.rds"
    df = next(iter(pyreadr.read_r(path).values()))
    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
    df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
    df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

    # ---- Spline knots (only for RE, FE splines not in W) ----
    fe_knots    = np.array([1.769863, 6.693151])     # needed for precomputed_splines slicing
    fe_boundary = np.array([0.0, 13.50685])
    re_knots    = np.array([3.567123])
    re_boundary = np.array([0.0, 13.50685])
    
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
    checkpoint = torch.load("checkpoints/best_model_hybrid.pt", map_location="cpu")
    
    model = NeuralCDEModel(
        x_dim=len(x_cols),
        static_dim=len(static_cols),
        cfg=cfg,
        fe_spline_knots=fe_knots,
        fe_spline_boundary=fe_boundary,
        re_spline_knots=re_knots,
        re_spline_boundary=re_boundary,
        interaction_pairs=[(0, 1)],
        precomputed_splines=True,
        n_tv=1,
        include_fe_splines=False,    # CDE learns time trend instead
        use_rho_net=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col, static_cols=static_cols)
    loader = DataLoader(dataset, batch_size=64, shuffle=False,collate_fn=collate_pad)
    
    TRUE_BETA = {"ns1": -2.817, "ns2": 0.020, "ns3": -3.135, "BMI_x_AGEc": -0.05}

    # Case 1: interaction IN W (your current good result)
    data = extract_h_and_truth(model, loader,
                            true_beta=TRUE_BETA,
                            bmi_col=0, ns_cols=[1, 2, 3],
                            interaction_in_W=True)
    plot_h_comparison(**data, save_path="h_comparison_int_in_W.png")