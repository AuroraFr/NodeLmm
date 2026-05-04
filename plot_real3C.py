"""
Plot longitudinal trajectories of 20 randomly selected subjects from the 3C cohort.
Panels: BMI, GLUC (glucose), PAS, PAD, HDL, ISA15.

Usage:
    python plot_3C_subjects.py --data path/to/3C_data.csv --n_subjects 20 --seed 42

Adjust COLUMN NAMES below to match your dataset.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Column name mapping — adjust to match your CSV ───────────────────────────
COL_ID    = "NUM_ID"          # subject identifier
COL_TIME  = "time"        # visit time (years from baseline, or age, etc.)
COVARIATES = {
    "BMI":  "BMI",
    "PAS":  "PAS",
    "PAD":  "PAD",
    "GLUC": "GLUC",
    "HDL":  "HDL",
    "IST": "ISA15",
}

# ── Plotting configuration ────────────────────────────────────────────────────
PANEL_ORDER = ["BMI", "GLUC", "PAS", "PAD", "HDL", "IST"]
YLABELS = {
    "BMI":  "BMI (kg/m²)",
    "GLUC": "Glucose (g/L)",
    "PAS":  "Systolic BP (mmHg)",
    "PAD":  "Diastolic BP (mmHg)",
    "HDL":  "HDL (g/L)",
    "IST": "IST",
}
COLORS = plt.cm.tab20(np.linspace(0, 1, 20))


def load_data(path: str) -> pd.DataFrame:
    """Load dataset — handles csv / tsv / parquet."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    sep = "\t" if path.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def plot_subjects(df: pd.DataFrame, ids: np.ndarray, seed: int):
    """Create a 3×2 panel figure with individual trajectories."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    for ax, var in zip(axes, PANEL_ORDER):
        col = COVARIATES[var]
        for i, sid in enumerate(ids):
            sub = df[df[COL_ID] == sid].sort_values(COL_TIME)
            t = sub[COL_TIME].values
            y = sub[col].values
            # Drop NaN for this variable (sparse observation pattern)
            mask = ~np.isnan(y)
            if mask.sum() == 0:
                continue
            ax.plot(t[mask], y[mask], "-o", color=COLORS[i % 20],
                    alpha=0.7, markersize=3, linewidth=1)
        ax.set_ylabel(YLABELS.get(var, var), fontsize=11)
        ax.set_title(var, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    for ax in axes[-2:]:
        ax.set_xlabel("Time", fontsize=11)

    fig.suptitle(f"3C Cohort — {len(ids)} randomly selected subjects (seed={seed})",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot 3C subject trajectories")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to 3C dataset (CSV/TSV/Parquet)")
    parser.add_argument("--n_subjects", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="3C_subject_trajectories.png")
    args = parser.parse_args()

    df = load_data(args.data)
    if "time" not in df.columns:
        df["SUIVI"] = pd.to_datetime(df["SUIVI"])
        df["time"] = df.groupby(COL_ID)["SUIVI"].transform(
            lambda s: (s - s.min()).dt.total_seconds() / (365.25 * 24 * 3600)
        )
    print(f"Loaded {len(df)} rows, {df[COL_ID].nunique()} unique subjects")
    print(f"Columns: {list(df.columns)}")

    # Select random subjects
    rng = np.random.default_rng(args.seed)
    all_ids = df[COL_ID].unique()
    selected = rng.choice(all_ids, size=min(args.n_subjects, len(all_ids)),
                          replace=False)
    print(f"Selected {len(selected)} subjects: {selected[:5]}...")

    fig = plot_subjects(df, selected, args.seed)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()