import torch
import torchcde
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from LME_3C_evaluation import *
import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pandas as pd
from LME_3C_model import *

def get_representative_indices_latent(model, dataloader, num_subjects=20, device='cuda'):
    """
    1. Extracts latent vectors (z) for all test subjects.
    2. Clusters them into k groups (k = num_subjects).
    3. Picks the specific subject closest to the center of each cluster.
    """
    model.eval()
    model.to(device)
    
    all_latents = []
    
    print("Extracting latent representations...")
    with torch.no_grad():
        for batch in dataloader:
            # 1. Unpack Batch
            t, y, s_i, mask, id, t_i, x_aug = [d.to(device) for d in batch.values()]
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_aug)
            X = torchcde.CubicSpline(coeffs)
            rnn_input_x = X.evaluate(t[0])
            rnn_input = torch.cat([rnn_input_x, s_i], dim=-1)
            z0 = model.encoder(rnn_input)
            z0_augmented = torch.cat([z0, s_i], dim=-1)            
            all_latents.append(z0.cpu().numpy())
            
    # Concatenate all batches: (Total_Test_Samples, Latent_Dim)
    X_latent = np.concatenate(all_latents, axis=0)
    
    print(f"Clustering {X_latent.shape[0]} subjects into {num_subjects} archetypes...")
    
    # 2. K-Means Clustering
    kmeans = KMeans(n_clusters=num_subjects, random_state=42, n_init=10)
    kmeans.fit(X_latent)
    cluster_centers = kmeans.cluster_centers_
    
    # 3. Find the closest real subject to each cluster center
    representative_indices = []
    
    # Calculate distance from every point to every center
    # dists shape: (Total_Samples, Num_Clusters)
    dists = cdist(X_latent, cluster_centers, metric='euclidean')
    
    for k in range(num_subjects):
        # Find index of the sample closest to center k
        closest_idx = np.argmin(dists[:, k])
        representative_indices.append(closest_idx)
        
    print(f"Selected Indices: {representative_indices}")
    return representative_indices

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

def get_modified_batch(X_batch, feature_idx, target_val):
    """
    Modify the trajectory of 'feature_idx' so its values become 'target_val'.
    and fills NaNs with the target value.
    DOES NOT touch other channels (like masks).
    """
    X_mod = X_batch.clone()
    X_mod[:, :, feature_idx] = target_val
    # X_mod[:, :, feature_idx] = torch.nan_to_num(
    #     X_mod[:, :, feature_idx], 
    #     nan=float(target_val)
    # )
    
    return X_mod

def compute_cde_ice_representatives(
    model, 
    ice_loader,       # The dataloader containing ONLY the K representatives
    feature_idx, 
    test_values,      # e.g., [Low_Value, High_Value] or a grid
    device='cuda'
):
    model.eval()
    model.to(device)
    print(test_values)
    # Storage: Dictionary to hold results per subject
    # Structure: { subject_id: { val_1: curve, val_2: curve } }
    ice_results = {}
    
    print(f"Computing ICE for {len(ice_loader)} representatives...")
    
    with torch.no_grad():
        # Iterate through each representative subject
        for i, batch in enumerate(ice_loader):
            
            # Unpack (Batch size is 1 here)
            t, y, s_i, mask, id, t_i, x_aug = [d.to(device) for d in batch.values()]
            
            # Get the unique ID or index of this subject for labeling
            subj_id = f"Subject_Cluster_{i}" 
            ice_results[subj_id] = {}
            
            # Loop through the grid values (Counterfactuals)
            for val in test_values:
                time_length = 10
                
                # 1. SHIFT PERTURBATION
                x_aug_shifted = get_modified_batch(x_aug, feature_idx, val)
                
                # 2. RE-INTERPOLATION
                coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_aug_shifted)
                X_cde = torchcde.CubicSpline(coeffs)
                
                # 3. PREDICT (Fixed Effects)
                # We use Fixed Effects to see how the model structure treats this profile
                pred_mean = model(s_i, X_cde, time_length, return_components=True)[0]
                
                # 4. CLEAN OUTPUT
                if pred_mean.dim() == 3:
                    pred_mean = pred_mean.squeeze(-1)
                
                ice_results[subj_id][val] = pred_mean.detach().cpu().numpy()

    return ice_results

def compute_pdp(
    model, 
    dataloader, 
    features,
    feature_idx, 
    grid_values, 
    device='cuda'
):
    """
    Computes PDP for Neural CDE using Fixed Effects.
    """
    model.eval()
    model.to(device)
    
    # Storage for the final averaged curves
    pdp_curves = []
    
    # Setup plotting
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(grid_values))) # Generate colors
    
    pred_means_original = []
    labels = [
    "Constant Low",
    "Constant Mid-Low",
    "Constant Mid",
    "Constant Mid-High",
    "Constant High",
    "Jump Low→High",
    "Jump High→Low"
]
    
    for i, val in enumerate(tqdm(grid_values, desc="Grid Loop")):
        pred_means = []
        time_length = 10
        
        with torch.no_grad():
            for batch in dataloader:
                t, y, s_i, mask, id, t_i, x_aug = [d.to(device) for d in batch.values()]
                B, T, D = x_aug.size()

                x_aug_modified = get_modified_batch(x_aug, feature_idx, val)

                preds_over_time = []  # will hold arrays of shape (B,)

                for j in range(1, time_length):  # j = 1..time_length-1
                    x_hist = x_aug_modified[:, :j+1, :]  # (B, j+1, D)

                    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_hist)
                    X = torchcde.CubicSpline(coeffs)

                    pred_j = model(s_i, X, j+1, return_components=True)[0]  # expected (B, j+1, 1) or (B, j+1)

                    if pred_j.dim() == 3 and pred_j.size(-1) == 1:
                        pred_j = pred_j.squeeze(-1)  # -> (B, j+1)

                    pred_j = pred_j.detach().cpu().numpy()  # (B, j+1)

                    if j == 1:
                        preds_over_time.append(pred_j[:, 0])   # t0  -> (B,)
                    preds_over_time.append(pred_j[:, -1])      # tj  -> (B,)

                pred_mean_batch = np.stack(preds_over_time, axis=0).T  # (B, time_length)
                pred_means.append(pred_mean_batch)
        
        
        pred_means = np.concatenate(pred_means, axis=0).reshape(-1, time_length)
        pred_means_low = np.percentile(pred_means, 5, axis=0)
        pred_means_high = np.percentile(pred_means, 95, axis=0)
        
        avg_trajectory = pred_means.mean(axis=0)
        # yerr = np.vstack([avg_trajectory - pred_means_low, pred_means_high - avg_trajectory])
        
        pdp_curves.append(avg_trajectory)
        # plt.errorbar(np.linspace(0, 12, 10), avg_trajectory, yerr=yerr, fmt='D',         # Diamond marker
        #      color=colors[i], ecolor=colors[i],
        #      elinewidth=2, capsize=6)
        plt.plot(np.linspace(0, 12, 10), avg_trajectory, label=f'Val={labels[i]}', color=colors[i], alpha=0.8)
    
    pred_means_original = np.array(pred_means_original)
    avg_trajectory_original = pred_means_original.mean(axis=0)
    # plt.plot(np.linspace(0, 12, 10), avg_trajectory_original, label=f'real_data', color='grey', linestyle='--')
    # Finalize Plot
    plt.xlabel('Time Steps')
    plt.ylabel('Predicted Outcome (Fixed Effect)')
    plt.title(f'PDP Trajectories: Feature '+features[feature_idx])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("figures/PDP_"+features[feature_idx]+".pdf")

    return np.array(pdp_curves)

def convert_pred_list_to_df(pred_list, value_col, time_col):
     # ---- Convert list of prediction dicts → DataFrame ----
    rows = []
    for d in pred_list:
        times = d["time"]
        values = np.asarray(d[value_col])
        if "id" in d.keys():
            pid = d["id"]
        else:
            pid = d['NUM_ID']
        
        for t, v in zip(times, values):
            rows.append({time_col: t, value_col: v, "id": pid})

    df = pd.DataFrame(rows)
    return df

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

def plot_representative_ice(ice_results, feature_name):
    """
    Plots one subplot per representative subject.
    Each subplot contains the curves for the different grid values.
    """
    subjects = list(ice_results.keys())
    num_sub = len(subjects)
    labels = [
    "Constant Low",
    "Constant Mid-Low",
    "Constant Mid",
    "Constant Mid-High",
    "Constant High",
    "Jump Low→High",
    "Jump High→Low"
]
    
    # Dynamic grid layout
    cols = 2
    rows = math.ceil(num_sub / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Color map for the different grid values
    test_vals = list(ice_results[subjects[0]].keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_vals)))
    
    for idx, subj in enumerate(subjects):
        ax = axes[idx]
        
        curves = ice_results[subj]
        
        for i, (val, curve) in enumerate(curves.items()):
            time_steps = np.linspace(0, 12, len(curve))
            ax.plot(time_steps, curve, label=f'{labels[i]}', 
                    color=colors[i], linewidth=2)
            
        ax.set_title(f"Representative: {subj}")
        ax.grid(True, alpha=0.3)
        
        # Only put legend on the first plot to avoid clutter
        if idx == 0:
            ax.legend(fontsize='small')

    # Labels
    fig.text(0.5, 0.04, 'Time Steps', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Predicted Outcome (Fixed Effect)', va='center', rotation='vertical', fontsize=12)
    plt.suptitle(f"ICE Analysis: Heterogeneity of {feature_name} Effect", fontsize=16)
    plt.savefig("figures/ICE_"+feature_name+".pdf")


def permute_bmi_keep_length_truncate_or_keep(
    df: pd.DataFrame,
    id_col: str = "NUM_ID",
    time_col: str = "time",
    perm_col: str = "BMI",
    seed: int = 0,
):
    """
    - Keeps the dataframe same length and same row order.
    - For each recipient subject r, pick a donor subject d (random permutation of subject IDs).
    - Map donor BMI to recipient visits by nearest donor time with UNIQUE donor-time usage.
    - If donor has MORE BMI points than recipient visits: extra donor points are ignored (truncate).
    - If donor has FEWER BMI points than recipient visits: remaining recipient visits keep original BMI.
    - You handle masks yourself.

    Output columns added:
      BMI_perm, BMI_donor_id, BMI_donor_time
    """
    df["time"] = (df["SUIVI"] - df["SUIVI"].min()).dt.total_seconds() / (60 * 60 * 24 * 365)
    d = df.copy()
    # default: keep original BMI everywhere; overwrite where we can match donor points
    d["BMI_perm"] = pd.to_numeric(d[perm_col], errors="coerce").to_numpy(copy=True)
    d["BMI_donor_id"] = np.nan
    d["BMI_donor_time"] = pd.NaT

    rng = np.random.default_rng(seed)

    # keep subject order as in file
    ids = d[id_col].dropna().astype(str).drop_duplicates().to_list()
    perm_ids = rng.permutation(ids)
    mapping = dict(zip(ids, perm_ids))  # recipient -> donor

    # group without sorting full df
    by_id = {str(k): v for k, v in d.groupby(id_col, sort=False)}

    used_time = {}

    for rid in ids:
        did = mapping[rid]
        used_time[rid] = []

        rec = by_id.get(rid)
        don = by_id.get(did)
        if rec is None or don is None:
            continue

        # sort within subject only for matching; indices stay original
        rec_sorted = rec.sort_values(time_col)
        don_sorted = don.sort_values(time_col)

        # donor BMI points
        don_pts = don_sorted[[time_col, perm_col]]
        if don_pts.shape[0] == 0:
            continue

        td = don_pts[time_col].to_numpy()   # (K,)
        yd = pd.to_numeric(don_pts[perm_col], errors="coerce").to_numpy(dtype=float)
        K = td.shape[0]

        used = np.zeros(K, dtype=bool)

        # recipient visit times
        rec_vis = rec_sorted[[time_col]]
        tr = rec_vis[time_col].to_numpy(dtype="datetime64[ns]")   # (R,)
        rec_idx_sorted = rec_vis.index.to_numpy()                 # original row indices
        R = tr.shape[0]

        # We can assign at most min(R, K) recipient visits (truncate donor if K>R; keep rest if K<R)
        max_assign = min(R, K)

        # Greedy unique nearest matching for the first max_assign recipient visits in time order.
        # Remaining recipient visits keep original BMI_perm (already set).
        assigned = 0
        for k in range(R):
            if assigned >= max_assign or used.all():
                break

            tq = tr[k]
            dist = np.abs(td.astype("int64") - tq.astype("int64"))
            dist[used] = np.iinfo(np.int64).max
            j = int(dist.argmin())
            if dist[j] == np.iinfo(np.int64).max:
                break

            row_idx = rec_idx_sorted[k]
            d.at[row_idx, perm_col+"_perm"] = yd[j]
            d.at[row_idx, perm_col+"_donor_id"] = did
            d.at[row_idx, perm_col+"_donor_time"] = td[j]

            used[j] = True
            used_time[rid].append(str(pd.Timestamp(td[j]).date()))
            assigned += 1

        # If donor has longer sequence: remaining donor points are ignored automatically (truncate).
        # If donor has shorter sequence: remaining recipient visits keep original BMI_perm.

    d["BMI_donor_id"] = d["BMI_donor_id"].astype("object")
    return d, used_time, mapping

import numpy as np
import pandas as pd
from scipy.stats import t

def compute_global_bin_means_with_ci(
    df, time_col, value_col,
    bins=None, ci=0.95, include_count=True
):
    if bins is None:
        bins = np.array([0., 1., 3., 5.5, 8.5, 11., 14.])

    k = 6  # number of bins
    qs = np.linspace(0, 1, k+1)
    bins_q = np.unique(np.quantile(df[time_col].dropna(), qs))

    df["qbin"] = pd.cut(df[time_col], bins=bins_q, include_lowest=True, labels=False)

    df = df.copy()
    df["bin"] = pd.cut(df[time_col], bins=bins, labels=False, include_lowest=True)

    out_rows = []
    for b in range(len(bins) - 1):
        sub = df[df["bin"] == b][value_col].dropna()
        n = len(sub)

        mean = sub.mean() if n else np.nan
        std  = sub.std(ddof=1) if n > 1 else np.nan
        se   = (std / np.sqrt(n)) if n > 1 else np.nan

        # t critical for the bin (better than 1.96 for small n)
        if n > 1:
            alpha = 1 - ci
            tcrit = t.ppf(1 - alpha/2, df=n-1)
            ci_low  = mean - 1.96 * se
            ci_high = mean + 1.96 * se
        else:
            ci_low = np.nan
            ci_high = np.nan

        row = {
            "segment_start": bins[b],
            "segment_end": bins[b + 1],
            "mean": mean,
            "std": std,
            "se": se,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
        if include_count:
            row["count"] = n

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def compute_binned_means_with_ci(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    bins=None,
    ci: float = 0.95,
    method: str = "t",                 # "t" (i.i.d.) or "subject_bootstrap"
    subject_col: str | None = None,    # required if method="subject_bootstrap"
    subject_weighted: bool = True,     # only used for subject_bootstrap
    n_boot: int = 2000,
    random_state: int = 0,
    include_count: bool = True,
    right: bool = False,               # matches your original (right=False) behavior
    include_lowest: bool = True,
) -> pd.DataFrame:
    """
    Compute binned means and CIs for REAL observed data.

    - method="t": classic t-interval per bin (assumes i.i.d. within bin)
    - method="subject_bootstrap": subject-level bootstrap (recommended for longitudinal data)

    subject_weighted=True (recommended):
        1) average within (subject, bin)
        2) then average across subjects
      This avoids subjects with many measurements dominating a bin.

    Returns a DataFrame with bin edges, midpoints, mean, CI, and counts.
    """

    if bins is None:
        bins = [0, 1.5, 3.5, 5.5, 8.5, 11, 14]
    bins = np.asarray(bins, dtype=float)
    if np.any(np.diff(bins) <= 0):
        raise ValueError("`bins` must be strictly increasing bin edges.")

    df0 = df[[time_col, value_col] + ([subject_col] if subject_col else [])].copy()
    df0 = df0.dropna(subset=[time_col, value_col])

    # Create bin index 0..K-1 and keep interval endpoints
    cut = pd.cut(df0[time_col], bins=bins, labels=False, right=right, include_lowest=include_lowest)
    df0["_bin"] = cut.astype("float")  # float to allow NaN
    df0 = df0.dropna(subset=["_bin"])
    df0["_bin"] = df0["_bin"].astype(int)

    K = len(bins) - 1
    bin_left = bins[:-1]
    bin_right = bins[1:]
    bin_mid = 0.5 * (bin_left + bin_right)

    out = pd.DataFrame({
        "segment_start": bin_left,
        "segment_end": bin_right,
        "segment_mid": bin_mid,
        "bin": np.arange(K, dtype=int),
    })

    if method.lower() == "t":
        # Classic per-bin stats (i.i.d. assumption)
        g = df0.groupby("_bin", sort=False)[value_col]
        mean = g.mean()
        std = g.std(ddof=1)
        n = g.size()

        out["mean"] = out["bin"].map(mean).to_numpy()
        out["std"]  = out["bin"].map(std).to_numpy()
        out["count"] = out["bin"].map(n).fillna(0).astype(int).to_numpy()

        # SE and t-interval
        se = out["std"] / np.sqrt(out["count"].replace(0, np.nan))
        out["se"] = se

        alpha = 1 - ci
        # t critical depends on df = n-1, so compute rowwise safely
        tcrit = np.full(K, np.nan, dtype=float)
        valid = out["count"].to_numpy() > 1
        tcrit[valid] = t.ppf(1 - alpha / 2, df=out.loc[valid, "count"].to_numpy() - 1)

        out["ci_low"]  = out["mean"] - tcrit * out["se"]
        out["ci_high"] = out["mean"] + tcrit * out["se"]

        if not include_count:
            out = out.drop(columns=["count"])

        # order columns similar to your original
        cols = ["segment_start", "segment_end", "segment_mid", "mean", "std", "se", "ci_low", "ci_high"]
        if include_count:
            cols.append("count")
        return out[cols]

    elif method.lower() == "subject_bootstrap":
        if subject_col is None:
            raise ValueError("subject_col must be provided when method='subject_bootstrap'.")

        df1 = df[[subject_col, time_col, value_col]].copy()
        df1 = df1.dropna(subset=[time_col, value_col])
        df1["_bin"] = pd.cut(df1[time_col], bins=bins, labels=False, right=right, include_lowest=include_lowest)
        df1 = df1.dropna(subset=["_bin"])
        df1["_bin"] = df1["_bin"].astype(int)

        subjects = df1[subject_col].unique()
        if len(subjects) == 0:
            raise ValueError("No subjects found after binning/NA removal.")

        # Point estimate
        if subject_weighted:
            sbm = (
                df1.groupby([subject_col, "_bin"], sort=False)[value_col]
                   .mean()
                   .reset_index()
            )
            point = sbm.groupby("_bin", sort=False)[value_col].mean()
            n_subjects_bin = sbm.groupby("_bin", sort=False)[subject_col].nunique()
        else:
            point = df1.groupby("_bin", sort=False)[value_col].mean()
            n_subjects_bin = df1.groupby("_bin", sort=False)[subject_col].nunique()

        n_obs_bin = df1.groupby("_bin", sort=False)[value_col].size()

        out["mean"] = out["bin"].map(point).to_numpy()
        out["n_obs"] = out["bin"].map(n_obs_bin).fillna(0).astype(int).to_numpy()
        out["n_subjects"] = out["bin"].map(n_subjects_bin).fillna(0).astype(int).to_numpy()

        # Pre-split by subject for fast bootstrap
        groups = {sid: g for sid, g in df1.groupby(subject_col, sort=False)}
        rng = np.random.default_rng(random_state)

        boot = np.full((n_boot, K), np.nan, dtype=float)

        for b in range(n_boot):
            sampled = rng.choice(subjects, size=len(subjects), replace=True)
            boot_df = pd.concat([groups[sid] for sid in sampled], ignore_index=True)

            if subject_weighted:
                boot_sbm = (
                    boot_df.groupby([subject_col, "_bin"], sort=False)[value_col]
                           .mean()
                           .reset_index()
                )
                m = boot_sbm.groupby("_bin", sort=False)[value_col].mean()
            else:
                m = boot_df.groupby("_bin", sort=False)[value_col].mean()

            # align to all bins 0..K-1
            boot[b, :] = pd.Series(m).reindex(np.arange(K)).to_numpy()

        alpha = 1 - ci
        out["ci_low"] = np.nanpercentile(boot, 100 * (alpha / 2), axis=0)
        out["ci_high"] = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)

        # Optional: bootstrap sd/se for the mean estimate (informative, not required)
        out["boot_sd_mean"] = np.nanstd(boot, axis=0, ddof=1)

        cols = [
            "segment_start", "segment_end", "segment_mid",
            "mean", "ci_low", "ci_high",
            "n_obs", "n_subjects", "boot_sd_mean"
        ]
        if not include_count:
            cols = [c for c in cols if c not in ("n_obs", "n_subjects")]
        return out[cols]

    else:
        raise ValueError("method must be 't' or 'subject_bootstrap'.")



# ------------------- Example usage -------------------
# df must have columns like: subject_id, time_bin, y
# df = pd.read_csv("your_data.csv")
# ci_df = subject_bootstrap_mean_ci(
#     df, subject_col="subject_id", time_col="time", y_col="y",
#     n_boot=2000, ci=0.95, random_state=42
# )
# print(ci_df)

# Optional plotting (matplotlib, no custom colors)
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(ci_df["time"], ci_df["mean"])
# plt.fill_between(ci_df["time"], ci_df["ci_low"], ci_df["ci_high"], alpha=0.2)
# plt.xlabel("time")
# plt.ylabel("mean(y)")
# plt.show()



def save_CDE_predictions(model, dataset, df):
    predicitons = []
    for _, patient_id in enumerate(df['NUM_ID'].unique().tolist()):
        # ax = axes[idx]
        # Get individual patient data
        sample_patient_data = filter_patient_with_id(patient_id, dataset)
        
        # Compute predicted trajectory
        t_points, seq_preds, actual_y, pop_preds = calculate_sequential_blup_forecasting(model, sample_patient_data, device)
        pred_dict = {'time':t_points, 'ISA15':seq_preds, "id": patient_id, 'pop_pred':pop_preds}
        predicitons.append(pred_dict)

    np.save("results/CDE_3C_predictions.npy", predicitons)