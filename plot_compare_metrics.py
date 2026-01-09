import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

experiments = [
    "Neural CDE\n(BMI+PAD+HDL+GLUC)",
    # "Neural CDE\n(Full Longitudinal features)",
    "BMI",
    "GLUC",
    "HDL",
    "PAD",
]

# Pad every metric to length = len(experiments) using np.nan for missing values
N = len(experiments)
def pad(vals, N=N):
    vals = list(vals)
    return vals + [np.nan] * (N - len(vals))

data = {
    # ("Train", "-2LL"):         pad([-63012*-2, -63058*-2, -63688*-2, -63090*-2, -63093*-2, -63429*-2]),
    ("Train", "-2LL"):         pad([-63012*-2, -63688*-2, -63090*-2, -63093*-2, -63429*-2]),
    # ("Train", "Fitted MSE"): pad([8.60, 8.86, None, None, None, None]),
    # ("Train", "Pred MSE"):   pad([22.00, 22.78, 23.09, 22.07, 22.08, 22.54]),  # padded to 6
    ("Train", "Pred MSE"):   pad([22.00, 23.09, 22.07, 22.08, 22.54]),  # padded to 6
    # ("Test", "-2LL"):          pad([-19937*-2, -19980*-2, None, None, None, None]),
    # ("Test", "Fitted MSE"):  pad([9.03, 9.26, None, None, None, None]),
    # ("Test", "Pred MSE"):    pad([22.62, 23.56, None, None, None, None]),
}

experiments = [
    "Neural CDE\n(BMI+PAD+HDL+GLUC)",
    "BMI",
    "GLUC",
    "HDL",
    "PAD",
]

N = len(experiments)

def pad(vals, N=N):
    vals = list(vals)
    return vals + [np.nan] * (N - len(vals))

data = {
    ("Train", "-2LL"):     pad([-63012*-2, -63688*-2, -63090*-2, -63093*-2, -63429*-2]),
    ("Train", "MSE"): pad([22.00, 23.09, 22.07, 22.08, 22.54]),
}

df = pd.DataFrame(data, index=experiments)
df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Split", "Metric"])
print(df)

def plot_perm_degradation_from_df(
    df,
    baseline_experiment="Neural CDE\n(BMI+PAD+HDL+GLUC)",
    split="Train",
    metrics=("-2LL", "MSE"),
    outpath="figures/perm_degradation.pdf",
    drop_baseline=True,
):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    plt.rcParams["font.size"] = 16

    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(5.0 * len(metrics), 3.0),
        constrained_layout=True
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):

        if (split, metric) not in df.columns:
            ax.set_axis_off()
            continue

        col = pd.to_numeric(df[(split, metric)], errors="coerce")
        base = col.loc[baseline_experiment]

        if pd.isna(base):
            ax.set_axis_off()
            continue

        # Δ = permuted - baseline  (positive = worse, since lower is better for both -2LL and MSE)
        delta = (col - base).dropna()

        if drop_baseline and baseline_experiment in delta.index:
            delta = delta.drop(index=baseline_experiment)

        if delta.empty:
            ax.set_axis_off()
            continue

        # sort: biggest degradation on top (like importance)
        delta = delta.sort_values(ascending=False)

        y = np.arange(len(delta))
        ax.axvline(0.0, linewidth=1)
        ax.hlines(y, 0.0, delta.values, linewidth=2)
        ax.scatter(delta.values, y, zorder=3)

        ax.set_yticks(y)
        ax.set_yticklabels(delta.index)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)

        ax.set_title(f"Δ {metric}", fontsize=18)
        ax.set_xlabel("Degradation relative to baseline", fontsize=16)

        # allow small negative deltas (sometimes permutation helps by noise)
        xmin = min(0.0, float(delta.min()))
        xmax = max(0.0, float(delta.max()))
        pad = 0.05 * (xmax - xmin + 1e-12)
        ax.set_xlim(xmin - pad, xmax + pad)

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

# call it
plot_perm_degradation_from_df(df)
