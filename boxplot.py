"""
Boxplot of Delta-PDP estimates vs. true value, per anchor and per method.

Expected input: a long-format CSV with one row per (replication, method, anchor).
Required columns:
    - replication: int, replication index r = 1, ..., R
    - method:      str, e.g. "Oracle LMM", "Neural ODE-LMM"
    - anchor:      float, anchor time t (years)
    - dpdp_hat:    float, the estimated Delta-PDP for that (r, method, anchor)
    - dpdp_true:   float, the oracle Delta-PDP at that anchor (constant within anchor)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# Colours: method -> (fill, edge)
METHOD_STYLES = {
    "Oracle LMM":     ("#8ecae6", "#023047"),
    "Neural ODE-LMM": ("#ffb703", "#fb8500"),
    # Add more methods here if needed; any unknown method gets grey.
}
DEFAULT_STYLE = ("#d9d9d9", "#525252")


def plot_dpdp_boxplot(
    df: pd.DataFrame,
    method_order: list[str] | None = None,
    anchor_order: list[float] | None = None,
    title: str | None = None,
    ylabel: str = r"$\widehat{\Delta \mathrm{PDP}}$",
    xlabel: str = "Anchor time (years)",
    true_line_label: str = r"True $\Delta \mathrm{PDP}_0$",
    figsize: tuple[float, float] = (8.0, 5.0),
    box_width: float = 0.7,
    group_gap: float = 1.0,   # extra space between anchor groups
) -> plt.Figure:
    """
    Build the boxplot. Returns the matplotlib Figure.
    """
    if method_order is None:
        method_order = sorted(df["method"].unique())
    if anchor_order is None:
        anchor_order = sorted(df["anchor"].unique())

    n_methods = len(method_order)
    n_anchors = len(anchor_order)

    # x-positions: for each anchor we place n_methods boxes side-by-side,
    # then leave group_gap of empty space before the next anchor.
    group_width = n_methods + group_gap
    # Centre of each anchor group
    group_centres = np.arange(n_anchors) * group_width + (n_methods - 1) / 2

    fig, ax = plt.subplots(figsize=figsize)

    # Draw one boxplot per (anchor, method); keep handles for legend
    for m_idx, method in enumerate(method_order):
        positions = np.arange(n_anchors) * group_width + m_idx
        data_per_anchor = []
        for a in anchor_order:
            sub = df[(df["method"] == method) & (df["anchor"] == a)]
            data_per_anchor.append(sub["dpdp_hat"].values)

        fill, edge = METHOD_STYLES.get(method, DEFAULT_STYLE)
        bp = ax.boxplot(
            data_per_anchor,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            boxprops=dict(facecolor=fill, edgecolor=edge, linewidth=1.2),
            medianprops=dict(color=edge, linewidth=1.5),
            whiskerprops=dict(color=edge, linewidth=1.0),
            capprops=dict(color=edge, linewidth=1.0),
            flierprops=dict(marker="o", markerfacecolor=fill,
                            markeredgecolor=edge, markersize=4, alpha=0.7),
        )

    # True value: one red dashed horizontal segment per anchor group
    true_by_anchor = (
        df.groupby("anchor")["dpdp_true"]
        .first()
        .reindex(anchor_order)
    )
    for a_idx, a in enumerate(anchor_order):
        left  = a_idx * group_width - 0.5
        right = a_idx * group_width + (n_methods - 1) + 0.5
        ax.hlines(
            true_by_anchor.loc[a], left, right,
            colors="red", linestyles="dashed", linewidth=1.5,
            zorder=3,
        )

    # Axis cosmetics
    ax.set_xticks(group_centres)
    ax.set_xticklabels([f"$t = {a:g}$" for a in anchor_order])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.3)
    if title is not None:
        ax.set_title(title)

    # Legend: one patch per method plus the red dashed line
    legend_elems = []
    for method in method_order:
        fill, edge = METHOD_STYLES.get(method, DEFAULT_STYLE)
        legend_elems.append(Patch(facecolor=fill, edgecolor=edge, label=method))
    legend_elems.append(
        plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.5,
                   label=true_line_label)
    )
    ax.legend(handles=legend_elems, loc="best", frameon=True)

    fig.tight_layout()
    return fig


# ---------- Demo data so the script runs stand-alone ----------
def _make_demo_dataframe(seed: int = 0) -> pd.DataFrame:
    """Generate a demo CSV-like DataFrame matching the S1 table structure."""
    rng = np.random.default_rng(seed)
    anchors = [0, 5, 10, 15]
    R = 100
    true_val = -1.725

    rows = []
    # Oracle LMM
    for a in anchors:
        for r in range(R):
            rows.append({
                "replication": r,
                "method": "Oracle LMM",
                "anchor": a,
                "dpdp_hat": rng.normal(true_val - 0.025, np.sqrt(0.074)),
                "dpdp_true": true_val,
            })
    # Neural ODE-LMM (from your table)
    neural_params = {
        0:  (-0.106, 0.016),
        5:  (-0.108, 0.017),
        10: (-0.094, 0.021),
        15: (-0.098, 0.022),
    }
    for a in anchors:
        bias, var = neural_params[a]
        for r in range(R):
            rows.append({
                "replication": r,
                "method": "Neural ODE-LMM",
                "anchor": a,
                "dpdp_hat": rng.normal(true_val + bias, np.sqrt(var)),
                "dpdp_true": true_val,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=None,
                        help="Path to long-format CSV. If omitted, a demo is generated.")
    parser.add_argument("--out", type=Path, default=Path("s1_boxplot.png"),
                        help="Output PNG path.")
    parser.add_argument("--title", type=str,
                        default=r"Scenario S1: distribution of $\widehat{\Delta \mathrm{PDP}}$ across $R = 100$ replications")
    args = parser.parse_args()

    if args.csv is None:
        print("No --csv given; using demo data.")
        df = _make_demo_dataframe()
    else:
        df = pd.read_csv(args.csv)

    fig = plot_dpdp_boxplot(
        df,
        method_order=["Oracle LMM", "Neural ODE-LMM"],
        title=args.title,
    )
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()