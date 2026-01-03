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

def create_fictive_profiles(grid, n_time=6, jump_idx=3):
    """
    Create 6 fictive profiles for a given feature based on a grid of values.
    
    grid: array-like of values returned by get_medical_grid(...)
          e.g. shape (5,) or (num_points,)
    n_time: number of timepoints (default: 6)
    jump_idx: time index where the jump begins (default: 3)
    
    Returns:
        profiles: Torch tensor (6, n_time)
            6 scenarios × n_time timepoints
    """
    assert len(grid) >= 2, "Grid must have at least 2 values."

    low = grid[0]
    high = grid[-1]
    
    # Some extra intermediate values (optional)
    mid_low = grid[1] if len(grid) > 2 else (low + high) / 3
    mid     = grid[len(grid)//2]
    mid_high = grid[-2] if len(grid) > 2 else (2*high + low) / 3
    
    profiles = []

    # 1) Constant low
    profiles.append(torch.full((n_time,), low))

    # 2) Constant mid-low
    profiles.append(torch.full((n_time,), mid_low))

    # 3) Constant mid
    profiles.append(torch.full((n_time,), mid))

    # 4) Constant mid-high
    profiles.append(torch.full((n_time,), mid_high))

    # 5) Constant high
    profiles.append(torch.full((n_time,), high))

    # 6) Jump profile (low → high)
    jump_profile = torch.full((n_time,), low)
    jump_profile[jump_idx:] = high

    jump_profile2 = torch.full((n_time,), high)
    jump_profile2[jump_idx:] = low

    profiles.append(jump_profile)
    profiles.append(jump_profile2)
    print(profiles)

    return torch.stack(profiles, dim=0)

train_df = pd.read_csv("3C_dataset/train_3C_data_1.csv", na_values=["NA", ""])

train_df["SUIVI"] = pd.to_datetime(train_df["SUIVI"])
train_df["time"] = (
    (train_df["SUIVI"] - train_df.groupby("NUM_ID")["SUIVI"].transform("min"))
      .dt.total_seconds() / (60 * 60 * 24 * 365)
)

fig, ax = plt.subplots(figsize=(8, 5))
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
        ax.plot(plot_df["time"], plot_df["BMI"], alpha=0.4)

for profil in profils:
    ax.plot(np.array([0, 2, 4, 7, 10, 12]), profil, 'bo', alpha=0.4)
ax.set_xlabel("Time (years)")
ax.set_ylabel("BMI")

plt.savefig("figures/BMI_data.pdf")
plt.close()



