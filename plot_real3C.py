"""
Plot longitudinal trajectories of 20 randomly selected subjects from the 3C cohort.
Top two rows (5 panels): time-varying exposures (blue).
Bottom row (1 panel): cognitive outcome IST (red).
Usage:
    python plot_3C_subjects.py --data path/to/3C_data.csv --n_subjects 20 --seed 42
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Column name mapping ──────────────────────────────────────────────────────
COL_ID   = "NUM_ID"
COL_TIME = "time"
COVARIATES = {
    "BMI":  "BMI",
    "PAS":  "PAS",
    "PAD":  "PAD",
    "GLUC": "GLUC",
    "HDL":  "HDL",
    "IST":  "ISA15",
}

# ── Panel layout: exposures first, then outcome ──────────────────────────────
EXPOSURE_PANELS = ["BMI", "GLUC", "PAS", "PAD", "HDL"]
OUTCOME_PANELS  = ["IST"]

YLABELS = {
    "BMI":  "BMI (kg/m²)",
    "GLUC": "Glucose (mmol/L)",
    "PAS":  "Systolic BP (mmHg)",
    "PAD":  "Diastolic BP (mmHg)",
    "HDL":  "HDL (mmol/L)",
    "IST":  "IST",
}

COLOR_EXPOSURE = "#1f77b4"   # blue
COLOR_OUTCOME  = "#d62728"   # red


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    sep = "\t" if path.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def plot_subjects(df: pd.DataFrame, ids: np.ndarray):
    """
    3-row figure:
      Row 1: BMI, GLUC          (exposures, blue)
      Row 2: PAS, PAD, HDL      (exposures, blue)  — 3 columns
      Row 3: IST                (outcome, red)      — centered single panel
    """
    fig = plt.figure(figsize=(18, 15))

    # Row 1: 2 panels
    ax_bmi  = fig.add_subplot(3, 3, 1)
    ax_gluc = fig.add_subplot(3, 3, 2)
    # Row 2: 3 panels
    ax_pas  = fig.add_subplot(3, 3, 4)
    ax_pad  = fig.add_subplot(3, 3, 5)
    ax_hdl  = fig.add_subplot(3, 3, 6)
    # Row 3: centered outcome panel
    ax_ist  = fig.add_subplot(3, 3, 8)  # middle of bottom row

    panel_map = {
        "BMI": ax_bmi, "GLUC": ax_gluc,
        "PAS": ax_pas, "PAD": ax_pad, "HDL": ax_hdl,
        "IST": ax_ist,
    }

    all_panels = EXPOSURE_PANELS + OUTCOME_PANELS
    for var in all_panels:
        ax = panel_map[var]
        col = COVARIATES[var]
        color = COLOR_OUTCOME if var in OUTCOME_PANELS else COLOR_EXPOSURE

        for sid in ids:
            sub = df[df[COL_ID] == sid].sort_values(COL_TIME)
            t = sub[COL_TIME].values
            y = sub[col].values
            mask = ~np.isnan(y)
            if mask.sum() == 0:
                continue
            ax.plot(t[mask], y[mask], "-o", color=color,
                    alpha=0.35, markersize=4, linewidth=1.2)

        ax.set_ylabel(YLABELS[var], fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3)

    # X-axis label only on bottom-row panels
    for var in ["PAS", "PAD", "HDL", "IST"]:
        panel_map[var].set_xlabel("Time since inclusion (years)", fontsize=14)

    # Panel labels (a)–(f)
    labels = list("abcdef")
    for lab, var in zip(labels, ["BMI", "GLUC", "PAS", "PAD", "HDL", "IST"]):
        ax = panel_map[var]
        ax.set_title(f"({lab})", fontsize=15, fontweight="bold", loc="left")

    plt.tight_layout()
    # # Add caption-style text below the figure
    # caption = (
    #     "Fig. 3: Observed individual trajectories for a random subset of participants "
    #     "in the 3C cohort. "
    #     "Panels (a)–(e) (blue): time-varying exposures; "
    #     "panel (f) (red): cognitive outcome. "
    #     "BMI = body mass index (kg/m²); "
    #     "PAS = systolic blood pressure (mmHg); "
    #     "PAD = diastolic blood pressure (mmHg); "
    #     "HDL = high-density lipoprotein cholesterol (mmol/L); "
    #     "IST = Isaacs Set Test score. "
    #     "Time is measured in years since cohort inclusion."
    # )
    # fig.text(0.5, -0.02, caption, ha="center", fontsize=9,
    #          wrap=True,
    #          bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none"))

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot 3C subject trajectories")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--n_subjects", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="3C_subject_trajectories.eps")
    args = parser.parse_args()

    df = load_data(args.data)
    if "time" not in df.columns:
        df["SUIVI"] = pd.to_datetime(df["SUIVI"])
        df["time"] = df.groupby(COL_ID)["SUIVI"].transform(
            lambda s: (s - s.min()).dt.total_seconds() / (365.25 * 24 * 3600)
        )

    print(f"Loaded {len(df)} rows, {df[COL_ID].nunique()} unique subjects")

    rng = np.random.default_rng(args.seed)
    all_ids = df[COL_ID].unique()
    selected = rng.choice(all_ids, size=min(args.n_subjects, len(all_ids)),
                          replace=False)
    print(f"Selected {len(selected)} subjects")

    fig = plot_subjects(df, selected)
    fig.savefig(args.output, dpi=300, format='eps')
    print(f"Saved to {args.output}")
    # plt.show()


if __name__ == "__main__":
    main()