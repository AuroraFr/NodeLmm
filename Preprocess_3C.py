"""
Preprocessing for the 3C cohort real dataset.

Produces per-patient dictionaries with:
  - x_aug:  (T, 1 + K + K)  = [time, forward-filled covariates, cumulative mask]
  - t:      (T,)             padded observation times (forward-filled)
  - y:      (T,)             target values (0 at unobserved slots)
  - s_i:    (S,)             static covariates
  - target_mask: (T,)        1 where outcome is observed, 0 otherwise
  - patient_id:  scalar

Key design choices:
  1. Observation indicators are tracked explicitly (not inferred from value == 0).
  2. Cumulative mask is computed from true observation indicators, BEFORE any filling.
  3. Forward-fill is applied column-wise; leading NaNs are backward-filled.
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional


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


# ── Forward-fill + backward-fill ─────────────────────────────────────────────

def fill_missing(arr: np.ndarray) -> np.ndarray:
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


# ── Main preprocessing function ──────────────────────────────────────────────

def process_data(
    df: pd.DataFrame,
    id_col: str,
    time_varying_features: List[str],
    static_features: List[str],
    target_col: str,
    metabolic_baseline_features: Optional[List[str]] = None,
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

    Returns:
        all_patient_data: List of per-patient dicts.
    """
    n_slots = len(EXPECTED_TIMES)
    K = len(time_varying_features)

    # ── Encode categoricals ─────────────────────────────────────────────────
    df = df.copy()

    # SEX → SEX_code (binary: 0/1)
    if "SEX" in df.columns and "SEX_code" not in df.columns:
        if df["SEX"].dtype.name == "category":
            df["SEX_code"] = df["SEX"].cat.codes.astype(float)
        else:
            # Works for int, float, or string — maps unique values to 0/1
            vals = sorted(df["SEX"].dropna().unique())
            df["SEX_code"] = df["SEX"].map({v: float(i) for i, v in enumerate(vals)})

    # DIPNIV → DIPNIV_2, DIPNIV_3 (dummy encoding, reference = 1)
    if "DIPNIV" in df.columns:
        # Normalize level labels to clean integers (handles float like 2.0 → "2")
        dipniv_clean = df["DIPNIV"].astype(float).astype("Int64").astype(str)
        # Int64 (nullable) keeps NaN as <NA>, str gives "<NA>"
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

        # ── Cumulative observation mask (computed BEFORE any filling) ─────
        # obs_indicator[t, k] = 1 if feature k was truly observed at slot t
        obs_indicator = (~np.isnan(x_raw)).astype(np.float32)
        # cumulative_mask[t, k] = number of times feature k observed up to slot t
        cumulative_mask = np.cumsum(obs_indicator, axis=0).astype(np.float32)

        # ── Forward-fill + backward-fill dynamic features ─────────────────
        x_filled = fill_missing(x_raw)

        # ── Augmented input: [time, x_filled, cumulative_mask] ────────────
        time_col = padded_time_filled.reshape(-1, 1)     # (T, 1)
        x_aug = np.concatenate([time_col, x_filled, cumulative_mask], axis=1)
        # Shape: (T, 1 + K + K)

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
        }

        if metabolic_baseline_features is not None:
            mb = patient_df[metabolic_baseline_features].iloc[0].values.astype(np.float32)
            patient_dict["metabolic_baseline"] = torch.tensor(mb, dtype=torch.float32)

        all_patient_data.append(patient_dict)

    return all_patient_data


# ── Usage example ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example usage (adapt paths and column names to your data)
    print("3C Preprocessing Pipeline")
    print("=" * 50)
    print(f"Canonical time grid: {EXPECTED_TIMES}")
    print()
    print("Expected x_aug layout for K dynamic features:")
    print("  Column 0:        time")
    print("  Columns 1..K:    forward-filled dynamic covariates")
    print("  Columns K+1..2K: cumulative observation mask")
    print()
    print("The cumulative mask tells the ODE vector field how many times")
    print("each covariate channel has been truly observed up to that time.")
    print("This is especially important for channels like glucose/HDL/LDL")
    print("that are only measured at 2-3 visits across 14 years.")