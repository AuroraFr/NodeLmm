import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

experiments = [
    "Neural CDE\n(BMI+PAD+HDL+GLUC)",
    "Neural CDE\n(Full Longitudinal features)",
    "Neural CDE\n(VIMP ON BMI)",
    "Neural CDE\n(VIMP ON GLUC)",
    "Neural CDE\n(VIMP ON HDL)",
    "Neural CDE\n(VIMP ON PAD)",
]

# Pad every metric to length = len(experiments) using np.nan for missing values
N = len(experiments)
def pad(vals, N=N):
    vals = list(vals)
    return vals + [np.nan] * (N - len(vals))

data = {
    ("Train", "LL"):         pad([-63012, -63058, -63688, -63090, -63093, -63429]),
    ("Train", "Fitted MSE"): pad([8.60, 8.86, None, None, None, None]),
    ("Train", "Pred MSE"):   pad([22.00, 22.78, 23.09, 22.07, 22.08, 22.54]),  # padded to 6
    ("Test", "LL"):          pad([-19937, -19980, None, None, None, None]),
    ("Test", "Fitted MSE"):  pad([9.03, 9.26, None, None, None, None]),
    ("Test", "Pred MSE"):    pad([22.62, 23.56, None, None, None, None]),
}

df = pd.DataFrame(data, index=experiments)
df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Split", "Metric"])

def plot_compare_models(df, outpath="figures/compare_models.pdf"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    splits  = ["Train", "Test"]
    metrics = ["LL", "Fitted MSE", "Pred MSE"]

    fig, axes = plt.subplots(len(splits), len(metrics),
                             figsize=(6.2 * len(metrics), 4.8 * len(splits)),
                             constrained_layout=True)

    # axes indexing convenience
    if len(splits) == 1 and len(metrics) == 1:
        axes = np.array([[axes]])
    elif len(splits) == 1:
        axes = np.array([axes])
    elif len(metrics) == 1:
        axes = np.array([[ax] for ax in axes])

    for r, split in enumerate(splits):
        for c, metric in enumerate(metrics):
            ax = axes[r, c]

            if (split, metric) not in df.columns:
                ax.set_axis_off()
                continue

            s = pd.to_numeric(df[(split, metric)], errors="coerce").dropna()

            if s.empty:
                ax.set_axis_off()
                continue

            # sorting: LL higher is better; MSE lower is better
            ascending = (metric != "LL")
            s = s.sort_values(ascending=ascending)

            y = np.arange(len(s.index))
            ax.hlines(y, xmin=s.min(), xmax=s, linewidth=2)  # lollipop stems
            ax.scatter(s.values, y, zorder=3)

            ax.set_yticks(y)
            ax.set_yticklabels(s.index)
            ax.invert_yaxis()  # best on top after sorting
            ax.set_title(f"{split} — {metric}")
            ax.grid(True, axis="x", alpha=0.3)

            if metric == "LL":
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)

plot_compare_models(df)
