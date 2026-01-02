import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_df = pd.read_csv("3C_dataset/train_3C_data_1.csv", na_values=["NA", ""])

train_df["SUIVI"] = pd.to_datetime(train_df["SUIVI"])
train_df["time"] = (train_df["SUIVI"] - train_df["SUIVI"].min()).dt.total_seconds() / (60 * 60 * 24 * 365)
fig, ax = plt.subplots(figsize=(8, 5))

# for subject, df_sub in train_df.groupby("NUM_ID"):
#     ax.plot(df_sub["time"], df_sub["ISA15"], alpha=0.4)

# ax.set_xlabel("Time (years)")
# ax.set_ylabel("ISA15")

# plt.savefig("figures/3C_data.pdf")
# plt.close()


def compute_global_bin_percentiles(df, time_col, value_col, percentiles=[5, 50, 95]):
    # Build bins
    bins = [0, 1.5, 3.5, 5.5, 8.5, 11, 14]
    
    # Assign each row to a bin
    df = df.copy()
    df["bin"] = pd.cut(df[time_col], bins=bins, labels=False, include_lowest=True)
    print(df[df.bin == 0.0])

    rows = []
    for b in range(len(bins) - 1):
        sub = df[df["bin"] == b]

        row = {
            "segment_start": bins[b],
            "segment_end": bins[b+1],
        }

        if len(sub) == 0:
            for p in percentiles:
                row[f"p{p}"] = np.nan
        else:
            bin_vals = sub[value_col].values
            for p in percentiles:
                row[f"p{p}"] = np.percentile(bin_vals, p)

        rows.append(row)

    return pd.DataFrame(rows)


key_times = [0, 2, 4, 7, 10, 12]

summary_df = compute_global_bin_percentiles(
    train_df,
    time_col="time",
    value_col="ISA15"
)

print(summary_df)

summary_df["bin_center"] = (summary_df["segment_start"] + summary_df["segment_end"]) / 2

# Extract x (time) and y (percentiles)
x = summary_df["bin_center"].values
y5  = summary_df["p5"].values
y50 = summary_df["p50"].values
y95 = summary_df["p95"].values

# Plot
plt.figure(figsize=(10, 6))

plt.plot(x, y50, label="Median (50%)", linewidth=3)
plt.plot(x, y5,  label="5%", linestyle="--")
plt.plot(x, y95, label="95%", linestyle="--")

# Optional shading between percentiles
plt.fill_between(x, y5, y95, alpha=0.2, label="5–95% band")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Percentile Trajectories Over Time")
plt.legend()
plt.savefig("figures/percentile_3C_data.pdf")


