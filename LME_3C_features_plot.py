import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

def get_medical_grid(X_tensor, feature_name, num_points=5):
    """
    Generates a grid based on data distribution + medical logic.
    X_tensor: (Batch, Time, Channels)
    """
    # 1. Extract all valid values (flatten and remove NaNs)
    vals = X_tensor[feature_name].values
    vals = vals[~np.isnan(vals)]
    
    # 2. Calculate Data Limits (5th to 95th percentile)
    # We ignore the top/bottom 5% to avoid physics-breaking outliers
    low_lim = np.percentile(vals, 5)
    high_lim = np.percentile(vals, 95)
    
    print(f"Feature {feature_name}: Data Range [{low_lim:.1f}, {high_lim:.1f}]")
    
    # 3. Generate Grid
    # We use linear spacing between the 5th and 95th percentile.
    grid = np.linspace(low_lim, high_lim, num_points)
    
    return grid

def create_fictive_profiles(grid, n_time=6):
    """
    Create fictive profiles for a given feature based on a grid of values.

    Returns:
        profiles: Torch tensor (5, n_time)
            5 scenarios × n_time timepoints:
            - constant low
            - constant mid
            - constant high
            - increasing (low -> high)
            - decreasing (high -> low)
    """
    assert len(grid) >= 2, "Grid must have at least 2 values."

    plt.rcParams["font.size"] = 16

    low = grid[0]
    high = grid[-1]
    mid = grid[len(grid)//2]

    profiles = []

    # 1) Constant low
    profiles.append(torch.full((n_time,), low))

    # 2) Constant mid
    profiles.append(torch.full((n_time,), mid))

    # 3) Constant high
    profiles.append(torch.full((n_time,), high))

    # 4) Increasing profile (low -> high)
    inc_profile = torch.linspace(low, high, steps=n_time)

    # 5) Decreasing profile (high -> low)
    dec_profile = torch.linspace(high, low, steps=n_time)

    profiles.append(inc_profile)
    profiles.append(dec_profile)

    return torch.stack(profiles, dim=0)

train_df = pd.read_csv("3C_dataset/train_3C_data_1.csv", na_values=["NA", ""])

train_df["SUIVI"] = pd.to_datetime(train_df["SUIVI"])
train_df["time"] = (
    (train_df["SUIVI"] - train_df.groupby("NUM_ID")["SUIVI"].transform("min"))
      .dt.total_seconds() / (60 * 60 * 24 * 365)
)
plt.rcParams["font.size"] = 16
fig, ax = plt.subplots(figsize=(8, 6))
print(len(train_df['NUM_ID'].unique()))
grids = {}
name = 'BMI'
grids[name] = get_medical_grid(train_df, name, num_points=6)
print(f"{name} Grid: {grids[name]} for {len(train_df['NUM_ID'].unique())}")
profils = create_fictive_profiles(grids[name])
print(profils)

sample_ids = train_df['NUM_ID'].sample(n=500, random_state=42).tolist()
for subject, df_sub in train_df.groupby("NUM_ID"):
    if subject in sample_ids:
        plot_df = df_sub.dropna(subset=["BMI", "time"])
        ax.plot(plot_df["time"], plot_df["BMI"], alpha=0.4, color='grey')

colors = plt.cm.viridis(np.linspace(0, 1, 5))
for idx, profil in enumerate(profils):
    ax.plot(np.array([0, 2, 4, 7, 10, 12]), profil, color=colors[idx])
ax.set_xlabel("Time since the first visit (years)")
ax.set_ylabel("BMI")

plt.savefig("figures/BMI_data_fictives_profils.pdf")
plt.close()



