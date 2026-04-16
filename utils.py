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

import torch
import torchcde

def _batched_cholesky_with_jitter(A, jitter0=1e-8, max_tries=10):
    # symmetrize
    A = 0.5 * (A + A.transpose(-1, -2))
    N, K, _ = A.shape
    I = torch.eye(K, device=A.device, dtype=A.dtype).unsqueeze(0)

    jitter = jitter0
    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(A + jitter * I)
        if (info == 0).all():
            return L
        jitter *= 10.0
    raise torch._C._LinAlgError("Cholesky failed even after jitter escalation.")

def masked_NLL(mu, y_pad, V, mask):
    """
    Batch-averaged masked Gaussian marginal NLL.
 
    For each subject i, extracts the observed sub-vector and computes:
 
        NLL_i = 0.5 * [log|V_i| + (y_i - mu_i)' V_i^{-1} (y_i - mu_i)
                        + n_i * log(2π)]
 
    Returns the average over the batch:  (1/N) * Σ_i NLL_i
 
    This convention ensures that:
      - The loss magnitude is independent of batch size
      - The scheduler sees a consistent scale across batches
      - Gradient magnitudes are stable regardless of N
 
    Args:
        mu:    (N, T)     population mean predictions
        V:     (N, T, T)  marginal covariance matrices
        y_pad: (N, T)     outcomes (0 at unobserved slots)
        mask:  (N, T)     binary mask (1 = observed, 0 = unobserved)
        jitter: float     diagonal jitter for numerical stability
 
    Returns:
        scalar: batch-averaged NLL
    """
    N = mu.shape[0]
    device, dtype = mu.device, mu.dtype
    total_nll = torch.tensor(0.0, device=device, dtype=dtype)
    n_valid = 0
 
    for i in range(N):
        idx = mask[i].bool()
        n_i = idx.sum()
        if n_i == 0:
            continue
 
        mu_i = mu[i, idx]                                     # (n_i,)
        y_i = y_pad[i, idx]                                    # (n_i,)
        V_i = V[i][idx][:, idx]                                # (n_i, n_i)
 
        r_i = y_i - mu_i                                       # (n_i,)
 
        L_i = torch.linalg.cholesky(V_i)
        Vinv_r = torch.cholesky_solve(
            r_i.unsqueeze(-1), L_i).squeeze(-1)                # (n_i,)
 
        log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L_i)))
 
        nll_i = 0.5 * (log_det + r_i @ Vinv_r
                        + n_i * math.log(2 * math.pi))
        total_nll = total_nll + nll_i
        n_valid += 1
 
    if n_valid == 0:
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
 
    return total_nll / n_valid

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

def get_modified_batch(X_batch, feature_idx, target_val):
    """
    Modify the trajectory of 'feature_idx' so its values become 'target_val'.
    and fills NaNs with the target value.
    DOES NOT touch other channels (like masks).
    """
    X_mod = X_batch.clone()
    X_mod[:, :, feature_idx] = target_val
    
    return X_mod

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


def _resample_to_common_real_time_grid(x_aug, t_i, mask, t_query, time_ch=None):
    """
    Resample each subject's (irregular) covariate path onto a shared real-time grid t_query.
    Uses a *subject-specific* spline parameterized by that subject's real times tb.

    Args
    ----
    x_aug:   (B,T,D) padded covariates (may include time as a channel)
    t_i:     (B,T)   padded real times
    mask:    (B,T)   bool or {0,1}, True where valid
    t_query: (Q,)    shared real-time grid (e.g., linspace(0,12,10))
    time_ch: int or None, index of the time channel in x_aug (if present)

    Returns
    -------
    x_q:          (B,Q,D) covariates evaluated at t_query
    support_mask: (B,Q)   True where t_query is within [tmin, tmax] for that subject
    """
    B, T, D = x_aug.shape
    Q = t_query.numel()
    device = x_aug.device
    dtype = x_aug.dtype
    print(x_aug, t_i)

    x_q = torch.empty((B, Q, D), device=device, dtype=dtype)
    support_mask = torch.empty((B, Q), device=device, dtype=torch.bool)

    maskb = mask.bool() if mask.dtype != torch.bool else mask

    for b in range(B):
        mb = maskb[b]
        tb = t_i[b, mb]
        if tb.dim() == 2 and tb.size(-1) == 1:
            tb = tb.squeeze(-1)          # (Tb,)
        tb = tb.reshape(-1)              # force 1D
        xb = x_aug[b, mb, :]   # (Tb,D)
        # Subject-specific spline in REAL time
        coeffs_b = torchcde.hermite_cubic_coefficients_with_backward_differences(xb, t=tb)
        Xb = torchcde.CubicSpline(coeffs_b)

        # Evaluate covariates at shared real-time grid
        xqb = Xb.evaluate(t_query)  # (Q,D)

        x_q[b] = xqb

        tmin, tmax = tb[0], tb[-1]
        support_mask[b] = (t_query >= tmin) & (t_query <= tmax)

    return x_q, support_mask


@torch.no_grad()
def _predict_on_shared_grid_like_training(model, s_i, X, Q=None):
    """
    The model was trained with index-time splines, with real time supplied as a channel.
    So here we build the spline WITHOUT passing t=... (index-time), but x_q contains time_ch=t_query.
    """
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)  # index-time
    X = torchcde.CubicSpline(coeffs)
    pred = model(s_i, X, 6, return_components=True)[0]  # (B,Q,1) or (B,Q)
    if pred.dim() == 3 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    return pred  # (B,Q)


def compute_pdp(
    model,
    dataloader,
    features,
    feature_idx,
    grid_values,
    *,
    time_ch=None,                 # index of time channel in x_aug, e.g. TIME_CH
    t_query=None,                 # torch tensor (Q,), e.g. torch.linspace(0,12,10)
    device="cuda",
    use_delta=False,               # True => plot mean(delta-ICE); False => plot mean(pred_mod)
    savepath=None
):
    """
    PDP for Neural CDE on a shared *real-time* grid + delta-ICE.

    - Resamples each subject's irregular path (t_i varies) onto a common grid t_query.
    - Predicts baseline and modified predictions on that grid.
    - delta-ICE = pred_mod - pred_orig
    - PDP curve = mean over subjects of delta-ICE (recommended) or pred_mod (if use_delta=False)
    - CI: pointwise normal CI of the mean using only subjects supported at that time (no extrapolation).

    Returns
    -------
    pdp_curves: (len(grid_values), Q) numpy array
    """

    model.eval()
    model.to(device)

    # if t_query is None:
    #     t_query = torch.linspace(0.0, 12.0, 10, device=device)
    # else:
    #     t_query = t_query.to(device)
    # Q = t_query.numel()
    # t_query_np = t_query.detach().cpu().numpy()

    labels = [
        "Constant Low",
        "Constant Mid",
        "Constant High",
        "Inscreasing",
        "Decreasing",
    ]

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 16
    colors = plt.cm.viridis(np.linspace(0, 1, len(grid_values)))

    pdp_curves = []

    for gi, val in enumerate(tqdm(grid_values, desc="Grid Loop")):

        all_effects = []     # list of (B,Q) tensors/arrays
        all_support = []     # list of (B,Q) bool arrays
        all_ti = []

        with torch.no_grad():
            for batch in dataloader:
                t, y, s_i, mask, ids, t_i, x_aug = [d.to(device) for d in batch.values()]

                # # 1) Resample ORIGINAL covariates to shared real-time grid
                # x_q, support = _resample_to_common_real_time_grid(
                #     x_aug=x_aug, t_i=t_i, mask=mask, t_query=t_query, time_ch=time_ch
                # )

                # 2) Baseline prediction
                pred_orig = _predict_on_shared_grid_like_training(model, s_i, x_aug)  # (B,Q)

                # 3) Modify covariates on the shared grid (your existing function)
                x_aug_mod = get_modified_batch(x_aug, feature_idx, val)

                # 4) Modified prediction
                pred_mod = _predict_on_shared_grid_like_training(model, s_i, x_aug_mod)  # (B,Q)

                # 5) delta-ICE (recommended) or absolute
                eff = (pred_mod - pred_orig) if use_delta else pred_mod

                all_effects.append(eff.detach().cpu().numpy())          # (B,Q)
                all_support.append(mask.detach().cpu().numpy()) 
                all_ti.append(t_i.detach().cpu().numpy())     # (B,Q)

        eff_np = np.concatenate(all_effects, axis=0)  
        ti_np = np.concatenate(all_ti, axis=0).squeeze()     # (N,Q)
        support = np.concatenate(all_support, axis=0) # (N,Q) bool
        N = eff_np.shape[0]

        centers = np.array([0, 2, 4, 7, 10, 12], dtype=float)
        edges = np.concatenate(([-np.inf], (centers[:-1] + centers[1:]) / 2, [np.inf]))

        N, Q = eff.shape
        K = len(edges) - 1

        mean_curve = np.full(K, np.nan, dtype=float)
        se_curve   = np.full(K, np.nan, dtype=float)
        nsubj_curve = np.zeros(K, dtype=int)

        N, Q = eff_np.shape

        # support: (N,Q) -> boolean
        support = (support > 0.5)

        mean_curve = np.full(K, np.nan, dtype=float)
        se_curve   = np.full(K, np.nan, dtype=float)

        for k in range(K):
            lo, hi = edges[k], edges[k+1]

            idx = support & (ti_np >= lo) & (ti_np < hi)   # (N,Q) boolean

            vals = eff_np[idx]  # 1D array of all points in the bin
            n_k = vals.size

            if n_k > 0:
                mean_curve[k] = vals.mean()
                se_curve[k] = vals.std(ddof=1) / np.sqrt(n_k) if n_k > 1 else 0.0

        ci_low  = mean_curve - 1.96 * se_curve
        ci_high = mean_curve + 1.96 * se_curve
        yerr = np.vstack([mean_curve - ci_low, ci_high - mean_curve])

        pdp_curves.append(mean_curve)
        print(yerr.shape)

        plt.errorbar(
            centers[1:,], mean_curve[1:,], yerr=yerr[:,1:], fmt="D",
            color=colors[gi], ecolor=colors[gi], elinewidth=2, capsize=6
        )
        plt.plot(centers[1:,], mean_curve[1:,], color=colors[gi], alpha=0.85, label=labels[gi])

    plt.xlabel("Time (years)")
    plt.ylabel("Mean delta-ICE" if use_delta else "Mean ISA15 prediction")
    plt.title(f"PDP ({'delta-ICE' if use_delta else 'pred'}): {features[feature_idx]}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if savepath is None:
        savepath = (
            f"figures/PDP_{features[feature_idx]}_deltaICE.pdf"
            if use_delta else
            f"figures/PDP_{features[feature_idx]}.pdf"
        )
    plt.savefig(savepath)

    return np.stack(pdp_curves, axis=0)

def select_ids_by_bmi(
    df: pd.DataFrame,
    bmi_min: float = None,
    bmi_max: float = None,
    *,
    how: str = "baseline",
    baseline_time: float = 0.0,
    min_visits: int = 1,
) -> pd.Index:
    d = df.dropna(subset=["NUM_ID", "time", "BMI"]).copy()
    d = d.sort_values(["NUM_ID", "time"])

    def in_range(x):
        ok = np.ones(len(x), dtype=bool)
        if bmi_min is not None:
            ok &= (x >= bmi_min)
        if bmi_max is not None:
            ok &= (x <= bmi_max)
        return ok

    g = d.groupby("NUM_ID", sort=False)

    counts = g.size()
    keep_ids = counts[counts >= min_visits].index
    d = d[d["NUM_ID"].isin(keep_ids)]
    g = d.groupby("NUM_ID", sort=False)

    if how == "baseline":
        idx = g.apply(lambda x: (x["time"] - baseline_time).abs().idxmin())
        base = d.loc[idx.values, ["NUM_ID", "BMI"]].set_index("NUM_ID")
        return base.index[in_range(base["BMI"].values)]

    if how == "mean":
        m = g["BMI"].mean()
        return m.index[in_range(m.values)]

    if how == "last":
        last = g.tail(1).set_index("NUM_ID")
        return last.index[in_range(last["BMI"].values)]

    if how == "ever":
        ever = g["BMI"].apply(lambda s: in_range(s.values).any())
        return ever[ever].index

    if how == "always":
        always = g["BMI"].apply(lambda s: in_range(s.values).all())
        return always[always].index

    raise ValueError(f"Unknown how={how}")


def binned_mean_ci(
    df: pd.DataFrame,
    *,
    time_col="time",
    y_col="ISA15",
    id_col="NUM_ID",
    bins=None,
    bin_labels=None,
    min_n: int = 10,
):
    """
    Compute mean + 95% CI per time bin.
    CI is normal approx: mean ± 1.96 * (sd/sqrt(n)) within bin.
    (For repeated measures this is descriptive, not a strict inference.)
    """
    d = df.dropna(subset=[time_col, y_col]).copy()

    if bins is None:
        # Example bins aligned with your visit schedule
        bins = [-0.5, 1, 3, 5.5, 8.5, 11, 13]  # -> around 0,2,4,7,10,12
    if bin_labels is None:
        # use bin midpoints as labels
        mids = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
        bin_labels = [f"{m:.1f}" for m in mids]

    d["time_bin"] = pd.cut(d[time_col], bins=bins, labels=bin_labels, include_lowest=True)

    out = (
        d.groupby("time_bin", observed=True)[y_col]
        .agg(n="count", mean="mean", std="std")
        .reset_index()
    )

    out["se"] = out["std"] / np.sqrt(out["n"])
    out["ci_low"] = out["mean"] - 1.96 * out["se"]
    out["ci_high"] = out["mean"] + 1.96 * out["se"]

    # drop bins with too few points
    out = out[out["n"] >= min_n].copy()

    # numeric x for plotting (use midpoints)
    x_mids = np.array([float(s) for s in out["time_bin"].astype(str)])
    out["x"] = x_mids
    return out.sort_values("x")


def plot_binned_groups(
    df: pd.DataFrame,
    groups: dict,
    *,
    bins=None,
    min_n=10,
    title="ISA15 mean by time bins",
):
    """
    groups: dict name -> ids (iterable)
    """
    plt.figure(figsize=(10, 6))

    for name, ids in groups.items():
        dsub = df[df["NUM_ID"].isin(ids)]
        stats = binned_mean_ci(dsub, bins=bins, min_n=min_n)

        x = stats["x"].to_numpy()
        y = stats["mean"].to_numpy()
        yerr = np.vstack([y - stats["ci_low"].to_numpy(), stats["ci_high"].to_numpy() - y])

        plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=5, linewidth=2, label=f"{name} (npts={stats['n'].sum()})")

    plt.xlabel("Time (years, bin midpoints)")
    plt.ylabel("ISA15 (mean ± 95% CI)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('figures/'+title+'.pdf')

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

def compute_global_bin_means_with_ci(
    df, time_col, value_col,
    bins=None, ci=0.95, include_count=True
):
    if bins is None:
        bins = np.array([0., 1., 3., 5.5, 8.5, 11., 14.])

    k = 6  # number of bins
    qs = np.linspace(0, 1, k+1)
    bins_q = np.unique(np.quantile(df[time_col].dropna(), qs))

    # df["qbin"] = pd.cut(df[time_col], bins=bins_q, include_lowest=True, labels=False)

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

from scipy.stats import t

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


def save_predictions(model, dataset, df, mode='fit'):
    predicitons = []
    for _, patient_id in enumerate(df['NUM_ID'].unique().tolist()):

        sample_patient_data = filter_patient_with_id(patient_id, dataset)
        
        # Compute predicted trajectory
        if mode == "fit":
            t_points, seq_preds, actual_y, pop_preds = fitted_trajectory(model, sample_patient_data, device)
        elif mode == "pred":
            t_points, seq_preds, actual_y, pop_preds = calculate_sequential_blup_forecasting(model, sample_patient_data, device)
        
        pred_dict = {'time':t_points, 'ISA15':seq_preds, "id": patient_id, 'pop_pred':pop_preds}
        predicitons.append(pred_dict)

    np.save("results/CDE_3C_"+mode+".npy", predicitons)


def plot_mean_predictions(train_df, model="CDE", mode="pred",
                          prediction_file="results/CDE_3C_train_predictions.npy", hlme_prediction_file="results/ISA15_Model_4_train_pred.csv"):
    CDE_predictions = np.load(prediction_file, allow_pickle=True)
    CDE_predictions_list = CDE_predictions.tolist()
    predictions_df = pd.DataFrame(CDE_predictions_list)
    # predictions_df = predictions_df.drop(columns=["pop_pred"])
    cols_to_explode = ["time", "ISA15"]

    df_long = (
        predictions_df
        .explode(cols_to_explode, ignore_index=True)
        .assign(
            time=lambda d: pd.to_numeric(d["time"]),
            ISA15=lambda d: pd.to_numeric(d["ISA15"])
        )
    )

    hlme_predictions = pd.read_csv(hlme_prediction_file, sep=',')
    value_col = "ISA15"
    time_col  = "time"

    # df_y0 = (df_long.sort_values([ "id", time_col ])
    #                  .groupby("id", as_index=False)
    #                  .first()[["id", value_col]]
    #         )

    # id_to_y0 = dict(zip(df_y0["id"], df_y0[value_col]))
    # new_rows = pd.DataFrame({
    #     "NUM_ID": list(id_to_y0.keys()),
    #     "time": 0,
    #     "Y_predicted": list(id_to_y0.values()),
    #     "Y_observed": list(id_to_y0.values())
    # })

    # hlme_predictions = (
    #     pd.concat([new_rows, hlme_predictions], ignore_index=True)
    #       .sort_values(by=["NUM_ID", "time"], ascending=[True, True])
    #       .reset_index(drop=True)
    # )
    import textwrap
    train_df["time"] = (
        (train_df["SUIVI"] - train_df.groupby("NUM_ID")["SUIVI"].transform("min"))
          .dt.total_seconds() / (60 * 60 * 24 * 365)
    )
    value_col = "ISA15"
    binned_mean_observations = compute_global_bin_means_with_ci(train_df, time_col, value_col)
    binned_mean_predictions = compute_global_bin_means_with_ci(df_long, time_col, value_col)
    value_col = "Yfitted"
    binned_mean_predictions_hlme = compute_global_bin_means_with_ci(hlme_predictions, time_col, value_col)
    x = (binned_mean_observations["segment_start"] + binned_mean_observations["segment_end"]) / 2
    y = binned_mean_observations["mean"]
    hat_y = binned_mean_predictions["mean"]
    hat_y_hlme = binned_mean_predictions_hlme['mean']

    yerr_lower = binned_mean_observations["mean"] - binned_mean_observations["ci_low"]
    yerr_upper = binned_mean_observations["ci_high"] - binned_mean_observations["mean"]
    yerr = np.vstack([yerr_lower[1:,], yerr_upper[1:,]])

    yerr_lower = binned_mean_predictions["mean"] - binned_mean_predictions["ci_low"]
    yerr_upper = binned_mean_predictions["ci_high"] - binned_mean_predictions["mean"]
    hat_yerr = np.vstack([yerr_lower, yerr_upper])

    plt.figure(figsize=(8,4))
    plt.errorbar(x[1:,], y[1:,], yerr=yerr, fmt='-',elinewidth=2, capthick=2,capsize=5, label="observations", alpha=0.4)
    # plt.errorbar(x, hat_y, yerr=yerr, fmt='D',elinewidth=2, capthick=2,capsize=5, label="CDE predictions")
    plt.scatter(x[1:,], hat_y[1:,], marker="D", label="ODE conditional predictions", color="black")
    plt.scatter(x[1:,], hat_y_hlme[1:,], marker='o',label="HLME conditional predictions", color="orange")
    plt.xlabel("Follow-up time (years since first visit)")
    plt.ylabel("ISA15")
    plt.ylim(30, 36)
    plt.legend(loc="best")
    title = "Mean of the observations (with 95% confidence interval) and of the conditional predictions from CDE model and of the conditional predictions from HLME model by time intervals defined according to visit times"
    # plt.title("\n".join(textwrap.wrap(title, width=50)))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/"+model+"_mean_trajectory_"+mode+".pdf",format='pdf', bbox_inches='tight')