import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def make_midpoint_bins(key_times, extend=0.5):
    key_times = np.array(sorted(key_times))
    midpoints = (key_times[:-1] + key_times[1:]) / 2
    bins = np.concatenate([[key_times[0]], midpoints, [key_times[-1] + extend]])
    return bins

def compute_global_bin_percentiles(pred_list, time_col, value_col, key_times, percentiles=[5, 50, 95]):
    # ---- Convert list of prediction dicts → DataFrame ----
    rows = []
    for d in pred_list:
        times = d["time"]
        values = np.asarray(d[value_col])
        pid = d["NUM_ID"]
        
        for t, v in zip(times, values):
            rows.append({time_col: t, value_col: v, "id": pid})

    df = pd.DataFrame(rows)
    print(df)

    # ---- Now reuse your existing logic ----
    bins = [0, 2.5, 4.5, 8.5, 11, 14]
    df = df.copy()
    df["bin"] = pd.cut(df[time_col], bins=bins, labels=False, include_lowest=True)

    out_rows = []
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
            vals = sub[value_col].values
            for p in percentiles:
                row[f"p{p}"] = np.percentile(vals, p)

        out_rows.append(row)

    return pd.DataFrame(out_rows)

train_predictions = []
hlme_predictions = pd.read_csv('results/ISA15_Model_3_train_predicted.csv', sep=',')
for name, group in hlme_predictions.groupby("NUM_ID"):
    df = group.reset_index(drop=True)
    train_predictions.append(df)

key_times = [0, 2, 4, 7, 10, 12]

summary_df = compute_global_bin_percentiles(
    train_predictions,
    time_col="time",
    value_col="Y_predicted",
    key_times=key_times
)
print("summary df", summary_df)
summary_df["bin_center"] = (summary_df["segment_start"] + summary_df["segment_end"]) / 2

# Extract x (time) and y (percentiles)
x = summary_df["bin_center"].values
y5  = summary_df["p5"].values
y50 = summary_df["p50"].values
y95 = summary_df["p95"].values

HLME_pop_pred = pd.read_csv('results/Model_3_pop_pred.csv',sep=',')
fig, ax = plt.subplots(figsize=(8, 5))
for subject, df_sub in HLME_pop_pred.groupby("NUM_ID"):
    ax.plot(df_sub["time"], df_sub["pred_m"], alpha=0.4)

ax.set_xlabel("Time (years)")
ax.set_ylabel("ISA15")

plt.savefig("figures/hlme_pop_prediction.pdf")
plt.close()

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
plt.savefig("figures/hlme_percentile_pred_3C_data.pdf")