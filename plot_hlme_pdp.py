"""
Plot HLME trajectory-profile PDP and ΔPDP from R-exported CSVs.
Matches Neural ODE-LMM color scheme, profile legend, and layout.

Usage:
    python plot_hlme_pdp.py
    python plot_hlme_pdp.py --dir hlme_pdp_exports --feat BMI GLUC HDL
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

from profile_legend import add_profile_legend

# ── Same style as Neural ODE-LMM ────────────────────────────────────

PROFILE_ORDER = [
    "stable_low", "late_spike", "gradual_rise",
    "stable_high", "late_decline", "gradual_decline",
]

PROFILE_COLOURS = {
    "stable_low":      "#1565C0",
    "late_spike":      "#0288D1",
    "gradual_rise":    "#00838F",
    "stable_high":     "#C62828",
    "late_decline":    "#E64A19",
    "gradual_decline": "#F57C00",
}

PROFILE_LABELS = {
    "stable_low":      "Stable low (Q25)",
    "late_spike":      "Late spike (Q25→Q75)",
    "gradual_rise":    "Gradual rise",
    "stable_high":     "Stable high (Q75)",
    "late_decline":    "Late decline (Q75→Q25)",
    "gradual_decline": "Gradual decline",
}

# PAIR_FILL = {
#     "stable_low":      "#37474F",
#     "late_spike":      "#1B5E20",
#     "gradual_rise":    "#4A148C",
#     "stable_high":     "#C62828",
#     "late_decline":    "#E64A19",
#     "gradual_decline": "#F57C00",
# }

 # One distinct color per pair (line + fill)
PAIR_LINE_COLORS = [
    '#37474F', '#1B5E20', '#4A148C',
    '#C62828', '#E64A19', '#F57C00',
    '#0288D1', '#00838F',
]
PAIR_FILL_COLORS = [
    '#B0BEC5', '#C8E6C9', '#E1BEE7',
    '#FFCDD2', '#FFCCBC', '#FFE0B2',
    '#37474F', '#F57C00',
]


VISIT_TIMES_3C = [0, 2, 4, 7, 10, 12]

DEFAULT_PAIRS = [
        ("late_decline", "late_spike"),
        ("stable_high", "stable_low"),
        ("gradual_decline", "gradual_rise"),
        ("late_decline", "stable_low"),
        ("late_decline", "gradual_decline"),
        ("stable_high", "late_spike"),
        ("stable_high", "gradual_rise"),
        ("gradual_decline", "stable_low"),
]


def plot_hlme_traj_pdp(csv_path, save_path, feat_name="covariate"):
    """Plot trajectory-profile PDP with schematic legend."""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    profiles_in_plot = []
    for pname in PROFILE_ORDER:
        sub = df[df["profile"] == pname]
        if sub.empty:
            continue
        color = PROFILE_COLOURS.get(pname, "grey")
        ax.plot(sub["time"], sub["mean"], '-', color=color, linewidth=1.8)
        profiles_in_plot.append(pname)

    for vt in VISIT_TIMES_3C:
        ax.axvline(vt, color='grey', linestyle=':', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Time (years)', fontsize=13)
    ax.set_ylabel('E[IST]', fontsize=13)
    ax.set_title(f'Trajectory-profile PDP of {feat_name} — HLME', fontsize=14)
    ax.grid(True, alpha=0.3)

    add_profile_legend(ax, profiles_in_plot, PROFILE_COLOURS, PROFILE_LABELS,
                       loc='best')

    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


def plot_hlme_delta_pdp(csv_path, save_path, feat_name="covariate",
                         wald_csv=None):
    """Plot pairwise ΔPDP with delta-method CI, 2×4 layout, profile colors."""
    df = pd.read_csv(csv_path)

    # Match pairs to DEFAULT_PAIRS order
    available_pairs = df["pair"].unique()
    pairs_ordered = []
    for pa, pb in DEFAULT_PAIRS:
        pair_str = f"{pa} - {pb}"
        if pair_str in available_pairs:
            pairs_ordered.append((pa, pb, pair_str))

    n_pairs = len(pairs_ordered)
    if n_pairs == 0:
        print(f"  No pairs found in {csv_path}")
        return

    # Layout: 2×4 if 8 pairs, else 1×n
    if n_pairs <= 4:
        nrows, ncols = 1, n_pairs
    else:
        nrows, ncols = 2, 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 12))
    axes = np.atleast_1d(axes).flatten()

    # First pass: find global y range
    all_ymin, all_ymax = [], []
    pair_data = []
    for pa, pb, pair_str in pairs_ordered:
        sub = df[df["pair"] == pair_str].sort_values("time")
        all_ymin.append(sub["ci_lo"].min())
        all_ymax.append(sub["ci_hi"].max())
        pair_data.append(sub)

    y_min = min(all_ymin)
    y_max = max(all_ymax)
    y_pad = 0.05 * (y_max - y_min)
    shared_ylim = (y_min - y_pad, y_max + y_pad)

    # Second pass: plot
    for idx, (pa, pb, pair_str) in enumerate(pairs_ordered):
        ax = axes[idx]
        sub = pair_data[idx]

        # color = PROFILE_COLOURS.get(pa, '#37474F')
        # fill  = PAIR_FILL.get(pa, '#E0E0E0')

        color = PAIR_LINE_COLORS[idx % len(PAIR_LINE_COLORS)]
        fill  = PAIR_FILL_COLORS[idx % len(PAIR_FILL_COLORS)]

        ax.fill_between(sub["time"], sub["ci_lo"], sub["ci_hi"],
                        color=fill, alpha=0.4)
        ax.plot(sub["time"], sub["delta"], '-', color=color, linewidth=2)
        ax.axhline(0, color='#B71C1C', linestyle='--', linewidth=1, alpha=0.7)

        # Significant points
        sig = sub[sub["sig"] == True]
        if not sig.empty:
            ax.scatter(sig["time"], sig["delta"], color='#D32F2F',
                       s=40, zorder=5)

        for vt in VISIT_TIMES_3C:
            ax.axvline(vt, color='grey', linestyle=':', alpha=0.2, linewidth=0.5)

        ax.set_ylim(shared_ylim)
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_xlabel('Time (years)')
        if idx % ncols == 0:
            ax.set_ylabel('ΔPDP')

        label_a = PROFILE_LABELS.get(pa, pa)
        label_b = PROFILE_LABELS.get(pb, pb)
        n_sig = int(sub["sig"].sum())
        L = len(sub)
        sig_str = f'{n_sig}/{L} sig.' if n_sig > 0 else 'n.s.'
        ax.set_title(f'{label_a}\n− {label_b}\n({sig_str})', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused panels
    for j in range(len(pairs_ordered), len(axes)):
        axes[j].set_visible(False)

    # fig.suptitle(f'Pairwise ΔPDP for {feat_name} (delta-method 95% CI) — HLME',
    #              fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")

    # Print Wald test summary
    if wald_csv and os.path.exists(wald_csv):
        wald_df = pd.read_csv(wald_csv)
        print(f"\n  HLME Wald tests for {feat_name}:")
        print(f"  {'Contrast':<45s} │ {'Z (all)':<14s} │ {'Z (late)':<14s}")
        print(f"  {'─'*45}─┼─{'─'*14}─┼─{'─'*14}")
        for _, row in wald_df.iterrows():
            z_all = f"{row['z_all']:+.2f} (p={row['p_all']:.3f})"
            z_late = (f"{row['z_late']:+.2f} (p={row['p_late']:.3f})"
                      if pd.notna(row['z_late']) else "—")
            print(f"  {row['pair']:<45s} │ {z_all:<14s} │ {z_late:<14s}")


def plot_hlme_all(data_dir="hlme_pdp_exports", out_dir="figures",
                   features=None):
    """Plot all HLME PDP figures."""
    if features is None:
        features = ["BMI", "GLUC", "HDL"]

    os.makedirs(out_dir, exist_ok=True)

    for feat in features:
        traj_csv  = os.path.join(data_dir, f"hlme_traj_pdp_{feat}.csv")
        delta_csv = os.path.join(data_dir, f"hlme_delta_pdp_{feat}.csv")
        wald_csv  = os.path.join(data_dir, f"hlme_wald_test_{feat}.csv")

        if os.path.exists(traj_csv):
            plot_hlme_traj_pdp(
                traj_csv,
                save_path=os.path.join(out_dir, f"hlme_traj_pdp_{feat}.png"),
                feat_name=feat,
            )

        if os.path.exists(delta_csv):
            plot_hlme_delta_pdp(
                delta_csv,
                save_path=os.path.join(out_dir, f"hlme_delta_pdp_{feat}.png"),
                feat_name=feat,
                wald_csv=wald_csv if os.path.exists(wald_csv) else None,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot HLME PDP from R exports")
    parser.add_argument("--dir", type=str, default="hlme_pdp_exports")
    parser.add_argument("--out", type=str, default="figures")
    parser.add_argument("--feat", nargs="+", default=["BMI", "GLUC", "HDL"])
    args = parser.parse_args()

    plot_hlme_all(data_dir=args.dir, out_dir=args.out, features=args.feat)