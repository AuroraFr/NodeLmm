"""
Preprocessing for the 3C cohort real dataset.

Produces per-patient dictionaries with:
  - x_aug:  (T, d_aug) augmented covariate tensor
  - t:      (T,)       padded observation times
  - y:      (T,)       target values (0 at unobserved slots)
  - s_i:    (S,)       static covariates
  - target_mask: (T,)  1 where outcome is observed, 0 otherwise
  - patient_id:  scalar

The augmented layout:
  Column 0:            time
  Columns 1..K:        interpolated/filled dynamic covariates
  Columns K+1..2K:     observation mask (binary or cumulative, see mask_type)

Interpolation methods:
  - "ffill":   forward-fill then backward-fill (piecewise constant)
  - "linear":  per-channel linear interpolation between observed values,
               with ffill/bfill for extrapolation beyond observed range
  - "cubic":   per-channel cubic spline interpolation (falls back to linear
               for channels with < 4 observations)

Key design choices:
  1. Observation indicators are tracked explicitly (not inferred from value == 0).
  2. Masks are computed from true observation indicators, BEFORE any filling.
  3. Interpolation is performed per-channel to handle heterogeneous observation
     patterns (e.g., BMI at every visit, glucose at 2-3 visits).
  4. For the ODE encoder, linear interpolation gives continuous x̄_i(t) with
     piecewise-constant derivative — compatible with future CDE extension.
"""

import numpy as np
import pandas as pd
import torch
import warnings
from typing import List, Dict, Optional, Literal
from scipy.interpolate import CubicSpline


# ── Canonical visit grid ──────────────────────────────────────────────────────

EXPECTED_TIMES = np.array([0.0, 2.0, 4.0, 7.0, 10.0, 12.0])


# ── Time-slot assignment ─────────────────────────────────────────────────────

def assign_time_slots(times: np.ndarray) -> np.ndarray:
    """
    Map each observed time to the nearest canonical slot (greedy, no collisions).

    Args:
        times: (n_visits,) actual follow-up times for one patient.
    Returns:
        slot_indices: (n_visits,) index into EXPECTED_TIMES for each visit.
    """
    n_slots = len(EXPECTED_TIMES)

    # Compute distance matrix: (n_visits, n_slots)
    dists = np.abs(times[:, None] - EXPECTED_TIMES[None, :])

    # Greedy assignment: closest first, no slot reuse
    slot_indices = np.full(len(times), -1, dtype=int)
    used_slots = set()

    # Process visits in order of best fit (smallest distance first)
    visit_slot_pairs = []
    for i in range(len(times)):
        for j in range(n_slots):
            visit_slot_pairs.append((dists[i, j], i, j))
    visit_slot_pairs.sort(key=lambda x: x[0])

    assigned_visits = set()
    for _, visit_idx, slot_idx in visit_slot_pairs:
        if visit_idx in assigned_visits or slot_idx in used_slots:
            continue
        slot_indices[visit_idx] = slot_idx
        assigned_visits.add(visit_idx)
        used_slots.add(slot_idx)

    assert np.all(slot_indices >= 0), "Some visits could not be assigned to slots"
    return slot_indices


# ── Interpolation methods ────────────────────────────────────────────────────

def fill_ffill(arr: np.ndarray) -> np.ndarray:
    """
    Forward-fill then backward-fill a 2D array column-wise.

    Forward-fill propagates the last observed value forward in time.
    Backward-fill handles leading NaNs (e.g., glucose first measured at T4)
    by carrying the first observed value backward.

    Args:
        arr: (T, K) array with NaN for missing entries.
    Returns:
        filled: (T, K) with no NaNs remaining.
    """
    df = pd.DataFrame(arr)
    df = df.ffill().bfill()
    return df.values.astype(np.float32)


def fill_linear(arr: np.ndarray, grid_times: np.ndarray) -> np.ndarray:
    """
    Per-channel linear interpolation between observed values.

    For each feature column k:
      - Identify slots where the value is truly observed (not NaN).
      - Linearly interpolate at all grid times between the first and last
        observed slot.
      - Extrapolate outside the observed range using the nearest observed
        value (i.e., constant extrapolation = ffill/bfill at boundaries).

    This gives a continuous x̄_i(t) whose derivative dx/dt is piecewise
    constant — well-suited as input to an ODE vector field and compatible
    with future CDE extensions where dz = f(z, x^s) dX.

    Args:
        arr:        (T, K) array with NaN for missing entries.
        grid_times: (T,) canonical time values for each slot.
    Returns:
        filled: (T, K) with no NaNs remaining.
    """
    T, K = arr.shape
    filled = np.copy(arr)

    for k in range(K):
        col = arr[:, k]
        obs_mask = ~np.isnan(col)
        n_obs = obs_mask.sum()

        if n_obs == 0:
            # No observations at all — fill with 0 (should not happen in practice)
            warnings.warn(f"Feature column {k} has no observations; filling with 0.")
            filled[:, k] = 0.0
        elif n_obs == 1:
            # Single observation — constant everywhere
            filled[:, k] = col[obs_mask][0]
        else:
            # Linear interpolation between observed values
            obs_times = grid_times[obs_mask]
            obs_vals = col[obs_mask]
            # np.interp extrapolates with boundary values by default
            filled[:, k] = np.interp(grid_times, obs_times, obs_vals)

    return filled.astype(np.float32)


def fill_cubic(arr: np.ndarray, grid_times: np.ndarray) -> np.ndarray:
    """
    Per-channel cubic spline interpolation between observed values.

    For each feature column k:
      - If >= 4 observations: fit a natural cubic spline (bc_type='natural'),
        clamped to the observed range to prevent extrapolation oscillation.
      - If 2-3 observations: fall back to linear interpolation.
      - If 1 observation: constant everywhere.

    Boundary handling: outside the range [t_first_obs, t_last_obs], the
    spline is NOT extrapolated (boundary values are held constant) to avoid
    the wild oscillations that cubic splines produce with sparse data.

    WARNING: For channels with only 2-3 observations across 14 years
    (e.g., glucose, HDL), this automatically falls back to linear.
    The cubic option is most useful for densely observed channels (BMI, BP).

    Args:
        arr:        (T, K) array with NaN for missing entries.
        grid_times: (T,) canonical time values for each slot.
    Returns:
        filled: (T, K) with no NaNs remaining.
    """
    T, K = arr.shape
    filled = np.copy(arr)

    for k in range(K):
        col = arr[:, k]
        obs_mask = ~np.isnan(col)
        n_obs = obs_mask.sum()

        if n_obs == 0:
            warnings.warn(f"Feature column {k} has no observations; filling with 0.")
            filled[:, k] = 0.0
        elif n_obs == 1:
            filled[:, k] = col[obs_mask][0]
        elif n_obs < 4:
            # Too few points for cubic — fall back to linear
            obs_times = grid_times[obs_mask]
            obs_vals = col[obs_mask]
            filled[:, k] = np.interp(grid_times, obs_times, obs_vals)
        else:
            # Cubic spline with natural boundary conditions
            obs_times = grid_times[obs_mask]
            obs_vals = col[obs_mask]
            cs = CubicSpline(obs_times, obs_vals, bc_type='natural')

            # Evaluate, but clamp to observed time range to avoid
            # extrapolation artifacts
            t_min, t_max = obs_times[0], obs_times[-1]

            for t_idx in range(T):
                t_val = grid_times[t_idx]
                if t_val < t_min:
                    filled[t_idx, k] = obs_vals[0]       # constant left extrap
                elif t_val > t_max:
                    filled[t_idx, k] = obs_vals[-1]       # constant right extrap
                else:
                    filled[t_idx, k] = cs(t_val)

    return filled.astype(np.float32)


# Dispatcher
INTERP_METHODS = {
    "ffill": lambda arr, grid_times: fill_ffill(arr),
    "linear": fill_linear,
    "cubic": fill_cubic,
}


# ── Main preprocessing function ──────────────────────────────────────────────

def process_data(
    df: pd.DataFrame,
    id_col: str,
    time_varying_features: List[str],
    static_features: List[str],
    target_col: str,
    metabolic_baseline_features: Optional[List[str]] = None,
    interp_method: Literal["ffill", "linear", "cubic"] = "ffill",
    mask_type: Literal["cumulative", "binary"] = "cumulative",
) -> List[Dict]:
    """
    Preprocess 3C cohort data into padded tensors for the Neural ODE/CDE model.

    Args:
        df: Raw dataframe with one row per (patient, visit).
        id_col: Column name for patient ID.
        time_varying_features: List of column names for dynamic covariates
                               (e.g., ["BMI", "PAS", "PAD", "GLUC", "HDL"]).
        static_features: List of column names for static covariates
                         (e.g., ["AGEc", "SEX_code", "DIPNIV_2", "DIPNIV_3"]).
        target_col: Column name for the outcome (e.g., "ISA15").
        metabolic_baseline_features: Optional list of baseline metabolic features
                                     to include (e.g., ["HDL0", "GLUC0", "BMI0", "CHOL0"]).
        interp_method: Interpolation method for time-varying covariates.
            - "ffill":  forward-fill / backward-fill (piecewise constant)
            - "linear": per-channel linear interpolation
            - "cubic":  per-channel cubic spline (falls back to linear if < 4 obs)
        mask_type: Type of observation mask appended to x_aug.
            - "cumulative": cumulative_mask[t, k] = number of times feature k
              has been observed up to and including slot t. Tells the ODE vector
              field about observation density over history.
            - "binary": obs_indicator[t, k] = 1 iff feature k was truly measured
              at slot t. Lets the vector field distinguish measured vs. imputed
              values at each time step.

    Returns:
        all_patient_data: List of per-patient dicts.
            Each dict contains:
              x_aug:  (T, 1 + 2K)   augmented covariates
              t:      (T,)          observation times
              y:      (T,)          target values
              s_i:    (S,)          static covariates
              target_mask: (T,)     observation indicator for target
              patient_id:  scalar
              interp_method: str    method used (for provenance)
              mask_type: str        mask type used (for provenance)
    """
    if interp_method not in INTERP_METHODS:
        raise ValueError(
            f"Unknown interp_method '{interp_method}'. "
            f"Choose from: {list(INTERP_METHODS.keys())}"
        )

    interpolate_fn = INTERP_METHODS[interp_method]
    n_slots = len(EXPECTED_TIMES)
    K = len(time_varying_features)

    # ── Encode categoricals ─────────────────────────────────────────────────
    df = df.copy()

    # SEX → SEX_code (binary: 0/1)
    if "SEX" in df.columns and "SEX_code" not in df.columns:
        if df["SEX"].dtype.name == "category":
            df["SEX_code"] = df["SEX"].cat.codes.astype(float)
        else:
            vals = sorted(df["SEX"].dropna().unique())
            df["SEX_code"] = df["SEX"].map({v: float(i) for i, v in enumerate(vals)})

    # DIPNIV → DIPNIV_2, DIPNIV_3 (dummy encoding, reference = 1)
    if "DIPNIV" in df.columns:
        dipniv_clean = df["DIPNIV"].astype(float).astype("Int64").astype(str)
        levels = sorted([l for l in dipniv_clean.unique() if l not in ("<NA>", "nan")])
        for level in levels:
            col_name = f"DIPNIV_{level}"
            if level != "1" and col_name not in df.columns:
                df[col_name] = (dipniv_clean == level).astype(float)

    # ── Compute follow-up time ────────────────────────────────────────────
    if "time" not in df.columns:
        df["SUIVI"] = pd.to_datetime(df["SUIVI"])
        df["time"] = df.groupby(id_col)["SUIVI"].transform(
            lambda s: (s - s.min()).dt.total_seconds() / (365.25 * 24 * 3600)
        )

    # ── Add baseline metabolic features ───────────────────────────────────
    if metabolic_baseline_features is not None:
        feature_to_source = {
            "HDL0": "HDL", "GLUC0": "GLUC", "BMI0": "BMI", "CHOL0": "CHOL"
        }
        for bf in metabolic_baseline_features:
            src = feature_to_source.get(bf, bf.rstrip("0"))
            baseline_vals = df.groupby(id_col).apply(
                lambda g: g.sort_values("time").iloc[0][src]
            )
            df[bf] = df[id_col].map(baseline_vals)

    # ── Verify required columns exist ────────────────────────────────────
    missing_static = [c for c in static_features if c not in df.columns]
    if missing_static:
        available = [c for c in df.columns if any(
            k in c.upper() for k in ["SEX", "DIP", "AGE"]
        )]
        raise KeyError(
            f"Static features not found: {missing_static}. "
            f"Available candidates: {available}"
        )

    missing_tv = [c for c in time_varying_features if c not in df.columns]
    if missing_tv:
        raise KeyError(f"Time-varying features not found: {missing_tv}")

    # ── Build per-patient data ─────────────────────────────────────────────
    all_patient_data = []

    for pid, group in df.groupby(id_col):
        patient_df = group.sort_values("time")
        times = patient_df["time"].values.astype(np.float32)
        n_visits = len(times)

        # ── Assign visits to canonical time slots ─────────────────────────
        slot_indices = assign_time_slots(times)

        # ── Pad target onto the canonical grid ────────────────────────────
        y = np.zeros(n_slots, dtype=np.float32)
        target_mask = np.zeros(n_slots, dtype=np.float32)
        padded_time = np.full(n_slots, np.nan, dtype=np.float32)

        target_vals = patient_df[target_col].values.astype(np.float32)
        for v in range(n_visits):
            s = slot_indices[v]
            y[s] = target_vals[v]
            target_mask[s] = 1.0
            padded_time[s] = times[v]

        # Forward-fill then backward-fill time (handles leading NaN slots)
        padded_time_filled = pd.Series(padded_time).ffill().bfill().values.astype(np.float32)

        # ── Pad dynamic features onto the canonical grid ──────────────────
        # Initialize as NaN — this is the TRUE missing indicator
        x_raw = np.full((n_slots, K), np.nan, dtype=np.float32)

        feature_vals = patient_df[time_varying_features].values.astype(np.float32)
        for v in range(n_visits):
            s = slot_indices[v]
            x_raw[s, :] = feature_vals[v, :]
            # Note: some features may be NaN even at observed visits
            # (e.g., glucose not measured at T2). This is correctly preserved.

        # ── Observation mask (computed BEFORE any filling/interpolation) ────
        # obs_indicator[t, k] = 1 if feature k was truly observed at slot t
        obs_indicator = (~np.isnan(x_raw)).astype(np.float32)

        if mask_type == "binary":
            mask = obs_indicator
        else:  # "cumulative"
            mask = np.cumsum(obs_indicator, axis=0).astype(np.float32)

        # ── Interpolate dynamic features ──────────────────────────────────
        x_interp = interpolate_fn(x_raw, padded_time_filled)

        # ── Augmented input ───────────────────────────────────────────────
        # Layout: [time | x_interp | mask]
        time_col = padded_time_filled.reshape(-1, 1)     # (T, 1)
        x_aug = np.concatenate([time_col, x_interp, mask], axis=1)
        # Shape: (T, 1 + K + K) = (T, 1 + 2K)

        # ── Static features ──────────────────────────────────────────────
        s_i = patient_df[static_features].iloc[0].values.astype(np.float32)

        # ── Build patient dict ────────────────────────────────────────────
        patient_dict = {
            "x_aug": torch.tensor(x_aug, dtype=torch.float32),
            "t": torch.tensor(padded_time_filled, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "s_i": torch.tensor(s_i, dtype=torch.float32),
            "target_mask": torch.tensor(target_mask, dtype=torch.float32),
            "patient_id": pid,
            "interp_method": interp_method,
            "mask_type": mask_type,
        }

        if metabolic_baseline_features is not None:
            mb = patient_df[metabolic_baseline_features].iloc[0].values.astype(np.float32)
            patient_dict["metabolic_baseline"] = torch.tensor(mb, dtype=torch.float32)

        all_patient_data.append(patient_dict)

    return all_patient_data


# ── Utility: describe x_aug layout ───────────────────────────────────────────

def describe_layout(K: int, mask_type: str = "cumulative") -> str:
    """Return a human-readable description of the x_aug column layout."""
    mask_label = "cumulative observation mask" if mask_type == "cumulative" \
        else "binary observation mask"
    lines = [
        f"  Column 0:              time",
        f"  Columns 1..{K}:          interpolated dynamic covariates",
        f"  Columns {K+1}..{2*K}:        {mask_label}",
        f"  Total dimension:       {1 + 2 * K}",
    ]
    return "\n".join(lines)


# ── Usage example ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("3C Preprocessing Pipeline")
    print("=" * 50)
    print(f"Canonical time grid: {EXPECTED_TIMES}")
    print()

    K_example = 5  # BMI, PAS, PAD, GLUC, HDL
    print("Available interpolation methods:")
    print("  ffill  — forward-fill / backward-fill (piecewise constant)")
    print("  linear — per-channel linear interpolation (continuous, dX/dt piecewise constant)")
    print("  cubic  — per-channel cubic spline (smooth, falls back to linear if < 4 obs)")
    print()

    for mt in ["cumulative", "binary"]:
        print(f"x_aug layout (mask_type='{mt}'):")
        print(describe_layout(K_example, mask_type=mt))
        print()

    # ── Quick test with synthetic data ────────────────────────────────────
    print("Running interpolation sanity check...")
    print("-" * 40)

    # Simulate a patient with glucose observed at T0 and T7 only
    grid = EXPECTED_TIMES.copy()
    x_test = np.full((6, 2), np.nan, dtype=np.float32)
    # Feature 0 (BMI): observed at T0, T2, T4, T7, T10
    x_test[0, 0] = 25.0   # T=0
    x_test[1, 0] = 25.5   # T=2
    x_test[2, 0] = 26.0   # T=4
    x_test[3, 0] = 26.5   # T=7
    x_test[4, 0] = 27.0   # T=10
    # Feature 1 (GLUC): observed at T0 and T7 only
    x_test[0, 1] = 5.0    # T=0
    x_test[3, 1] = 6.5    # T=7

    print(f"  Raw data (NaN = missing):")
    for s in range(6):
        print(f"    T={grid[s]:5.1f}  BMI={x_test[s,0]:>6}  GLUC={x_test[s,1]:>6}")
    print()

    for method in ["ffill", "linear", "cubic"]:
        fn = INTERP_METHODS[method]
        result = fn(x_test.copy(), grid)
        print(f"  {method}:")
        for s in range(6):
            print(f"    T={grid[s]:5.1f}  BMI={result[s,0]:6.2f}  GLUC={result[s,1]:6.2f}")
        print()

    print('Binary observation mask:')
    obs = (~np.isnan(x_test)).astype(int)
    for s in range(6):
        print(f'    T={grid[s]:5.1f}  BMI={obs[s,0]}  GLUC={obs[s,1]}')
    print()

    print('Cumulative observation mask:')
    cum = np.cumsum(obs, axis=0)
    for s in range(6):
        print(f'    T={grid[s]:5.1f}  BMI={cum[s,0]}  GLUC={cum[s,1]}')