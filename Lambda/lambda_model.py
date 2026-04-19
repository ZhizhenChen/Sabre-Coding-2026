"""
Lambda model for Poisson Process and GLM-based TTL estimation.

PART 1 — SHARED DATA PREPARATION
Three new helper functions for bucketing and encoding:
- bucket_lead_time(horizon): Maps lead_time_days -> 7 buckets
- bucket_duration(duration): Maps duration_nights -> 7 buckets
- encode_hotel(df, min_count=4): Sparse hotel treatment (< min_count -> "rare_hotel")

PART 2 — POISSON PROCESS
build_lambda_table_poisson(): Groups by (hotel_encoded, duration_bucket, lead_time_bucket, rate_source),
computes lambda = total_changes / total_exposure_hours, builds lookup dictionary.

PART 3 — POISSON GLM
build_lambda_table_glm(): Fits categorical GLM with reference level selection based on global_rate,
extracts lambda correctly (fittedvalues / exposure_hours), builds lookup dictionary.

PART 4 — FALLBACK WATERFALL
serve_glm_ttl(): 4-level fallback when looking up lambda at serving time.

PART 5 — SANITY CHECK
print_sanity_check(): Validates that lambda ranges are sensible and TTL values are correct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
from data_processing.pipeline import (
    lambda_bucket_lead_time as _pipeline_bucket_lead_time,
    lambda_bucket_duration as _pipeline_bucket_duration,
    lambda_encode_hotel as _pipeline_encode_hotel,
    prepare_lambda_enriched_dataframe as _pipeline_prepare_lambda_enriched_dataframe,
    aggregate_lambda_bucket_groups as _pipeline_aggregate_lambda_bucket_groups,
    split_lambda_prepared_by_time as _pipeline_split_lambda_prepared_by_time,
)

TTL_NUMERATOR_TARGET_80 = 0.2231


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# PART 1 — SHARED DATA PREPARATION: Bucketing and Encoding Functions
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def bucket_lead_time(horizon: int) -> str:
    """
    Map lead_time_days (days between query and stay start) to one of 7 buckets.
    Remove rows where lead_time < 0 before calling this.
    """
    return _pipeline_bucket_lead_time(horizon)


def bucket_duration(duration: int) -> str:
    """
    Map duration_nights (length of stay in nights) to one of 7 buckets.
    Remove rows where duration <= 0 before calling this.
    """
    return _pipeline_bucket_duration(duration)


def encode_hotel(df: pd.DataFrame, min_count: int = 4) -> pd.Series:
    """
    Implement sparse-hotel treatment:
    - Count occurrences of each hotel_code
    - hotel_codes with count < min_count → "rare_hotel"
    - All others → original hotel_code
    Returns a new Series called hotel_encoded.
    """
    return _pipeline_encode_hotel(df, min_count=min_count)


def _prepare_enriched_dataframe(df_enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Shared Part 1 data prep used by Poisson and GLM.
    - Remove invalid lead_time and duration rows
    - Add lead_time_bucket, duration_bucket, hotel_encoded
    """
    return _pipeline_prepare_lambda_enriched_dataframe(df_enriched, min_hotel_count=4)


def _aggregate_bucket_groups(df_prepared: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to one row per (hotel_encoded, duration_bucket, lead_time_bucket, rate_source),
    with total_changes, total_exposure_hours, n_intervals, lambda_poisson, implied_ttl_poisson_hours.
    """
    return _pipeline_aggregate_lambda_bucket_groups(
        df_prepared=df_prepared,
        ttl_numerator_target_80=TTL_NUMERATOR_TARGET_80,
    )


def _time_split_prepared_df(
    df_prepared: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Timestamp]]:
    """
    Time-based split on rq_timestamp.
    Returns (train_df, test_df, split_ts).
    """
    return _pipeline_split_lambda_prepared_by_time(df_prepared=df_prepared, train_ratio=train_ratio)


def _compute_global_lambda(group_agg: pd.DataFrame, default_lambda: float = 0.1) -> float:
    if group_agg.empty:
        return float(default_lambda)
    total_changes = float(pd.to_numeric(group_agg["total_changes"], errors="coerce").fillna(0).sum())
    valid_exposure = pd.to_numeric(group_agg["total_exposure_hours"], errors="coerce")
    valid_exposure = valid_exposure.where(valid_exposure > 0, np.nan)
    total_exposure = float(valid_exposure.sum(skipna=True))
    if total_exposure <= 0:
        return float(default_lambda)
    return total_changes / total_exposure


def _is_valid_lambda(value: object) -> bool:
    if value is None:
        return False
    try:
        lam = float(value)
    except Exception:
        return False
    return np.isfinite(lam) and lam > 0


def build_lambda_lookup_dict(
    lambda_table: pd.DataFrame,
    lambda_col: str,
    min_intervals: int = 5,
) -> Dict[Tuple[str, str, str, str], float]:
    """
    Build lookup dictionary keyed by:
    (hotel_encoded, duration_bucket, lead_time_bucket, rate_source)
    """
    if lambda_table.empty or lambda_col not in lambda_table.columns:
        return {}
    needed = ["hotel_encoded", "duration_bucket", "lead_time_bucket", "rate_source", "n_intervals", lambda_col]
    if any(c not in lambda_table.columns for c in needed):
        return {}

    use = lambda_table[lambda_table["n_intervals"] >= int(min_intervals)].copy()
    if use.empty:
        return {}
    use = use[use[lambda_col].apply(_is_valid_lambda)].copy()
    if use.empty:
        return {}

    lookup: Dict[Tuple[str, str, str, str], float] = {}
    for row in use.itertuples(index=False):
        key = (str(row.hotel_encoded), str(row.duration_bucket), str(row.lead_time_bucket), str(row.rate_source))
        lookup[key] = float(getattr(row, lambda_col))
    return lookup


def _frequent_hotels_from_df(df_prepared: pd.DataFrame, min_count: int = 4) -> set[str]:
    if df_prepared.empty:
        return set()
    counts = df_prepared["hotel_code"].value_counts()
    return set(counts[counts >= int(min_count)].index.astype(str).tolist())


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# EXISTING KM SUPPORT (DO NOT TOUCH)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class KMSummary:
    n_intervals: int
    n_events: int
    total_exposure_hours: float
    median_hours: float
    rmst_hours: float
    lambda_km_median: float
    lambda_km_rmst: float
    lambda_empirical: float


def _km_survival(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)

    mask = np.isfinite(durations) & (durations > 0)
    durations = durations[mask]
    events = events[mask]

    if len(durations) == 0:
        return np.array([]), np.array([])

    order = np.argsort(durations)
    d_sorted = durations[order]
    e_sorted = events[order]

    times, first_idx, counts = np.unique(d_sorted, return_index=True, return_counts=True)

    n_at_risk = np.cumsum(counts[::-1])[::-1].astype(float)
    d_t = np.add.reduceat((e_sorted == 1).astype(float), first_idx)

    hazard = np.where(n_at_risk > 0.0, 1.0 - (d_t / n_at_risk), 1.0)
    survival = np.cumprod(hazard)
    return times, survival


def _median_from_survival(times: np.ndarray, survival: np.ndarray) -> float:
    if len(times) == 0:
        return math.inf
    hit = np.where(survival <= 0.5)[0]
    if len(hit) == 0:
        return math.inf
    return float(times[hit[0]])


def _rmst_from_survival(times: np.ndarray, survival: np.ndarray) -> float:
    if len(times) == 0:
        return math.nan
    prev_t = 0.0
    prev_s = 1.0
    area = 0.0
    for t, s in zip(times, survival):
        area += prev_s * (t - prev_t)
        prev_t = float(t)
        prev_s = float(s)
    return area


def estimate_lambda_km(group_df: pd.DataFrame, assume_sorted: bool = False) -> KMSummary:
    g = group_df.copy() if assume_sorted else group_df.sort_values("rq_timestamp").copy()
    # delta_hours = time between consecutive requests in hours
    g["delta_hours"] = g["rq_timestamp"].diff().dt.total_seconds() / 3600.0
    g["event_change"] = g["price_change"].astype(int)

    # First row has no preceding interval.
    km_df = g.dropna(subset=["delta_hours"])
    km_df = km_df[km_df["delta_hours"] > 0]

    durations = km_df["delta_hours"].to_numpy(dtype=float)
    events = km_df["event_change"].to_numpy(dtype=int)

    times, survival = _km_survival(durations, events)
    median_hours = _median_from_survival(times, survival)
    rmst_hours = _rmst_from_survival(times, survival)

    total_exposure_hours = float(durations.sum()) if len(durations) else 0.0
    n_events = int(events.sum())
    n_intervals = int(len(durations))

    lambda_empirical = (n_events / total_exposure_hours) if total_exposure_hours > 0 else 0.0
    lambda_km_median = (math.log(2.0) / median_hours) if math.isfinite(median_hours) and median_hours > 0 else math.nan
    lambda_km_rmst = (1.0 / rmst_hours) if np.isfinite(rmst_hours) and rmst_hours > 0 else math.nan

    return KMSummary(
        n_intervals=n_intervals,
        n_events=n_events,
        total_exposure_hours=total_exposure_hours,
        median_hours=median_hours,
        rmst_hours=rmst_hours,
        lambda_km_median=lambda_km_median,
        lambda_km_rmst=lambda_km_rmst,
        lambda_empirical=lambda_empirical,
    )


def _group_level_stats(df_enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-group interval/event/exposure statistics used by all methods.
    Now applies bucketing and encoding first.
    """
    df = _prepare_enriched_dataframe(df_enriched)
    if df.empty:
        return pd.DataFrame()
    
    rows = []
    group_cols = ["hotel_encoded", "duration_bucket", "lead_time_bucket", "rate_source"]
    sorted_df = df.sort_values(group_cols + ["rq_timestamp"])

    for keys, g in sorted_df.groupby(group_cols, dropna=False, sort=False):
        summary = estimate_lambda_km(g, assume_sorted=True)
        rows.append(
            {
                "hotel_encoded": str(keys[0]),
                "duration_bucket": str(keys[1]),
                "lead_time_bucket": str(keys[2]),
                "rate_source": str(keys[3]),
                "n_intervals": summary.n_intervals,
                "n_events": summary.n_events,
                "total_exposure_hours": summary.total_exposure_hours,
                "lambda_empirical": summary.lambda_empirical,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def build_lambda_table_km(df_enriched: pd.DataFrame, min_intervals: int = 5) -> pd.DataFrame:
    """Kaplan-Meier based lambda estimation (legacy behavior — do NOT modify)."""
    # NOTE: This function is not being updated. Kept as-is for backward compatibility.
    # The specification says "Do not modify anything outside the two files listed above"
    # and "Do not touch: build_lambda_table_km() and _build_km_ttl_lookup()"
    # So this uses the old grouping. Only Poisson and GLM are being updated.
    
    out = _group_level_stats(df_enriched)
    if out.empty:
        return out

    out = out[out["n_intervals"] >= min_intervals].copy()
    if out.empty:
        return out

    out["lambda_final"] = out["lambda_empirical"]
    out["lambda_method"] = "km"
    return out.sort_values("n_intervals", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# PART 2 — POISSON PROCESS: New Implementation with Bucketing
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def build_lambda_table_poisson(df_enriched: pd.DataFrame, min_intervals: int = 5) -> pd.DataFrame:
    """
    Poisson Process rate estimation.
    lambda = total_changes / total_exposure_hours per (hotel_encoded, duration_bucket, lead_time_bucket, rate_source) group.
    
    Key changes vs old code:
    1. Buckets FIRST, then groups (not post-hoc averaging of individual lambdas)
    2. Groups by (hotel_encoded, duration_bucket, lead_time_bucket, rate_source)
    3. Computes global_lambda as sum(total_changes) / sum(total_exposure_hours)
    4. Builds lookup dictionary by 4-tuple
    """
    
    df = _prepare_enriched_dataframe(df_enriched)
    group_agg = _aggregate_bucket_groups(df)
    if group_agg.empty:
        return pd.DataFrame()

    # Correct global lambda must be sum(total_changes) / sum(total_exposure_hours) across ALL groups.
    global_lambda = _compute_global_lambda(group_agg, default_lambda=0.1)

    out = group_agg.copy()
    out["lambda_final"] = out["lambda_poisson"]
    out.loc[~out["lambda_final"].apply(_is_valid_lambda), "lambda_final"] = global_lambda
    out["lambda_method"] = "poisson"
    out["_global_lambda"] = global_lambda
    out["_in_lookup"] = out["n_intervals"] >= int(min_intervals)

    lookup_dict = build_lambda_lookup_dict(out, lambda_col="lambda_poisson", min_intervals=min_intervals)
    out.attrs["lambda_lookup_dict_poisson"] = lookup_dict
    out.attrs["global_lambda"] = global_lambda
    out.attrs["frequent_hotels"] = _frequent_hotels_from_df(df, min_count=4)

    return out.sort_values("n_intervals", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# PART 3 — POISSON GLM: New Implementation with Reference Level Selection and Bug Fixes
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def build_lambda_table_glm(
    df_enriched: pd.DataFrame,
    min_intervals: int = 5,
    use_negative_binomial: bool = True,
    enable_train_test_split: bool = False,
    train_ratio: float = 0.8,
) -> pd.DataFrame:
    """
    GLM-based lambda estimation with proper reference level selection and lambda extraction.
    
    Formula:
        total_changes ~ C(hotel_encoded, Treatment(ref=hotel))
                      + C(duration_bucket, Treatment(ref=dur))
                      + C(lead_time_bucket, Treatment(ref=lt))
                      + C(rate_source, Treatment(ref=rs))
        offset = log(total_exposure_hours)
    
    Key fixes:
    1. Use categorical encoding for ALL features (not continuous lead_time)
    2. Select reference levels based on global_rate (population-average change rate)
    3. Extract lambda correctly: fittedvalues / exposure_hours (NOT exp(fittedvalues))
    4. Implement 4-level fallback waterfall for serving
    """
    
    df = _prepare_enriched_dataframe(df_enriched)
    if df.empty:
        return pd.DataFrame()

    if enable_train_test_split:
        df_train, df_test, split_ts = _time_split_prepared_df(df, train_ratio=train_ratio)
    else:
        df_train, df_test, split_ts = df.copy(), pd.DataFrame(columns=df.columns), None

    train_group_agg = _aggregate_bucket_groups(df_train)
    if train_group_agg.empty:
        return pd.DataFrame()

    test_group_agg = _aggregate_bucket_groups(df_test) if not df_test.empty else pd.DataFrame()

    # Keep lookup threshold strict, but allow adaptive fit threshold if data is sparse.
    effective_min_intervals = max(int(min_intervals), 5)
    fit_min_intervals = effective_min_intervals
    glm_train = pd.DataFrame()
    candidate_thresholds = sorted(set([effective_min_intervals, 4, 3, 2, 1]), reverse=True)
    for candidate in candidate_thresholds:
        cand_train = train_group_agg[
            (train_group_agg["total_exposure_hours"] > 0) & (train_group_agg["n_intervals"] >= int(candidate))
        ].copy()
        if cand_train.empty:
            continue
        # Need enough signal to fit a count model.
        if len(cand_train) >= 10 and float(cand_train["total_changes"].sum()) > 0:
            glm_train = cand_train
            fit_min_intervals = int(candidate)
            break
    if glm_train.empty:
        # Last fallback: keep any non-empty candidate, even if weak signal.
        for candidate in candidate_thresholds:
            cand_train = train_group_agg[
                (train_group_agg["total_exposure_hours"] > 0) & (train_group_agg["n_intervals"] >= int(candidate))
            ].copy()
            if not cand_train.empty:
                glm_train = cand_train
                fit_min_intervals = int(candidate)
                break

    global_lambda = _compute_global_lambda(train_group_agg, default_lambda=0.1)

    if glm_train.empty:
        def _fallback_frame(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
            out = frame.copy()
            out["dataset_split"] = split_name
            out["lambda_glm"] = np.nan
            out["implied_ttl_glm_hours"] = np.nan
            out["lambda_final"] = out["lambda_poisson"]
            out.loc[~out["lambda_final"].apply(_is_valid_lambda), "lambda_final"] = global_lambda
            out["lambda_method"] = "glm_fallback_empty_train"
            out["_global_lambda"] = global_lambda
            out["_ref_hotel"] = "rare_hotel"
            out["_ref_lt"] = "same_day"
            out["_ref_dur"] = "1_night"
            out["_ref_rs"] = "unknown"
            out["_in_lookup"] = out["n_intervals"] >= effective_min_intervals
            return out

        out_train = _fallback_frame(train_group_agg, "train")
        out_test = _fallback_frame(test_group_agg, "test") if not test_group_agg.empty else pd.DataFrame()
        out = out_train if out_test.empty else pd.concat([out_train, out_test], ignore_index=True)
        out.attrs["lambda_lookup_dict_glm"] = {}
        out.attrs["global_lambda"] = global_lambda
        out.attrs["ref_hotel"] = "rare_hotel"
        out.attrs["ref_lt"] = "same_day"
        out.attrs["ref_dur"] = "1_night"
        out.attrs["ref_rs"] = "unknown"
        out.attrs["frequent_hotels"] = _frequent_hotels_from_df(df_train, min_count=4)
        out.attrs["train_test_split_enabled"] = bool(enable_train_test_split)
        out.attrs["train_ratio"] = float(train_ratio)
        out.attrs["split_timestamp_utc"] = str(split_ts) if split_ts is not None else None
        out.attrs["n_rows_train"] = int(len(df_train))
        out.attrs["n_rows_test"] = int(len(df_test))
        out.attrs["glm_test_metrics"] = {}
        return out

    # Reference levels must be selected from row-level df_enriched prep.
    global_rate = float(pd.to_numeric(df_train["price_change"], errors="coerce").fillna(0).mean())

    def _pick_reference_from_row_level(series_rate: pd.Series, fallback: str) -> str:
        if series_rate.empty:
            return fallback
        return str((series_rate - global_rate).abs().idxmin())

    hotel_rates = (
        df_train[df_train["hotel_encoded"] != "rare_hotel"]
        .groupby("hotel_encoded", dropna=False)["price_change"]
        .mean()
    )
    lt_rates = df_train.groupby("lead_time_bucket", dropna=False)["price_change"].mean()
    dur_rates = df_train.groupby("duration_bucket", dropna=False)["price_change"].mean()
    rs_rates = df_train.groupby("rate_source", dropna=False)["price_change"].mean()

    ref_hotel = _pick_reference_from_row_level(hotel_rates, "rare_hotel")
    ref_lt = _pick_reference_from_row_level(lt_rates, "same_day")
    ref_dur = _pick_reference_from_row_level(dur_rates, "1_night")
    fallback_rs = (
        str(df_train["rate_source"].dropna().astype(str).iloc[0])
        if not df_train["rate_source"].dropna().empty
        else "unknown"
    )
    ref_rs = _pick_reference_from_row_level(rs_rates, fallback_rs)

    # Ensure reference level exists in GLM training categories.
    def _ensure_in_training(reference_value: str, col: str) -> str:
        train_levels = set(glm_train[col].astype(str).unique().tolist())
        if reference_value in train_levels:
            return reference_value
        if not train_levels:
            return reference_value
        return sorted(train_levels)[0]

    ref_hotel = _ensure_in_training(ref_hotel, "hotel_encoded")
    ref_lt = _ensure_in_training(ref_lt, "lead_time_bucket")
    ref_dur = _ensure_in_training(ref_dur, "duration_bucket")
    ref_rs = _ensure_in_training(ref_rs, "rate_source")

    def _rel_diff(rate_value: float, base_value: float) -> float:
        if not np.isfinite(rate_value) or not np.isfinite(base_value):
            return np.nan
        if abs(base_value) < 1e-12:
            return math.inf if abs(rate_value) > 0 else 0.0
        return abs(rate_value - base_value) / abs(base_value)

    ref_rates = {
        "hotel": float(hotel_rates.get(ref_hotel, np.nan)),
        "lead_time": float(lt_rates.get(ref_lt, np.nan)),
        "duration": float(dur_rates.get(ref_dur, np.nan)),
        "rate_source": float(rs_rates.get(ref_rs, np.nan)),
    }

    print("\n" + "="*100)
    print("POISSON GLM — REFERENCE LEVEL SELECTION")
    print("="*100)
    print(f"Global change rate: {global_rate:.6f}")
    print(f"  ref_hotel:       {ref_hotel} (rate={ref_rates['hotel']:.6f})")
    print(f"  ref_lt:          {ref_lt} (rate={ref_rates['lead_time']:.6f})")
    print(f"  ref_dur:         {ref_dur} (rate={ref_rates['duration']:.6f})")
    print(f"  ref_rs:          {ref_rs} (rate={ref_rates['rate_source']:.6f})")
    for name, rate in ref_rates.items():
        rel = _rel_diff(rate, global_rate)
        if np.isfinite(rel) and rel > 0.5:
            print(
                f"  WARNING: ref_{name} differs from global_rate by "
                f"{(rel * 100.0):.1f}% (closest available level, not near-average)."
            )
    print("="*100 + "\n")

    # Try to fit GLM
    try:
        import importlib
        smf = importlib.import_module("statsmodels.formula.api")
        sm = importlib.import_module("statsmodels.api")

        glm_train["total_changes"] = glm_train["total_changes"].astype(int)
        glm_train["log_exposure"] = np.log(glm_train["total_exposure_hours"])

        # Build dynamic formula with reference levels
        formula = (
            f"total_changes ~ "
            f"C(hotel_encoded, Treatment(reference='{ref_hotel}')) + "
            f"C(lead_time_bucket, Treatment(reference='{ref_lt}')) + "
            f"C(duration_bucket, Treatment(reference='{ref_dur}')) + "
            f"C(rate_source, Treatment(reference='{ref_rs}'))"
        )

        family = sm.families.NegativeBinomial() if use_negative_binomial else sm.families.Poisson()

        model = None
        fitted_family_name = None
        families_to_try = []
        if use_negative_binomial:
            families_to_try = [
                ("negative_binomial", sm.families.NegativeBinomial()),
                ("poisson", sm.families.Poisson()),
            ]
        else:
            families_to_try = [
                ("poisson", sm.families.Poisson()),
                ("negative_binomial", sm.families.NegativeBinomial()),
            ]

        fit_errors: list[str] = []
        for fam_name, fam in families_to_try:
            try:
                model = smf.glm(
                    formula=formula,
                    data=glm_train,
                    family=fam,
                    offset=glm_train["log_exposure"],
                ).fit()
                fitted_family_name = fam_name
                break
            except Exception as fam_exc:
                fit_errors.append(f"{fam_name}: {fam_exc}")
                continue

        if model is None:
            raise RuntimeError(" ; ".join(fit_errors) if fit_errors else "GLM fit failed with unknown error.")

        print(
            "GLM fit complete: "
            f"family={fitted_family_name}, "
            f"fit_min_intervals={fit_min_intervals}, "
            f"nobs={int(model.nobs)}, "
            f"df_model={int(model.df_model)}, "
            f"df_resid={int(model.df_resid)}, "
            f"llf={float(model.llf):.4f}"
        )
        
        seen_levels_by_col = {
            col: set(glm_train[col].astype(str).unique().tolist())
            for col in ["hotel_encoded", "lead_time_bucket", "duration_bucket", "rate_source"]
        }

        def _predict_lambda_for_groups(group_frame: pd.DataFrame) -> pd.DataFrame:
            if group_frame.empty:
                return group_frame.copy()
            frame = group_frame.copy()
            for col, ref_value in [
                ("hotel_encoded", ref_hotel),
                ("lead_time_bucket", ref_lt),
                ("duration_bucket", ref_dur),
                ("rate_source", ref_rs),
            ]:
                frame[col] = frame[col].astype(str)
                frame[col] = frame[col].where(frame[col].isin(seen_levels_by_col[col]), ref_value)
            frame["log_exposure"] = np.log(frame["total_exposure_hours"].clip(lower=1e-12))
            predicted_events = model.predict(frame, offset=frame["log_exposure"])
            exposure_for_div = frame["total_exposure_hours"].where(frame["total_exposure_hours"] > 0, np.nan)
            frame["lambda_glm"] = pd.to_numeric(predicted_events / exposure_for_div, errors="coerce")
            frame["lambda_glm"] = frame["lambda_glm"].replace([np.inf, -np.inf], np.nan)
            frame["implied_ttl_glm_hours"] = TTL_NUMERATOR_TARGET_80 / frame["lambda_glm"]
            frame["lambda_final"] = frame["lambda_glm"]
            frame.loc[~frame["lambda_final"].apply(_is_valid_lambda), "lambda_final"] = global_lambda
            frame["lambda_method"] = "glm_nb" if use_negative_binomial else "glm_poisson"
            frame["_global_lambda"] = global_lambda
            frame["_ref_hotel"] = ref_hotel
            frame["_ref_lt"] = ref_lt
            frame["_ref_dur"] = ref_dur
            frame["_ref_rs"] = ref_rs
            frame["_in_lookup"] = frame["n_intervals"] >= effective_min_intervals
            return frame

        train_pred = _predict_lambda_for_groups(train_group_agg)
        train_pred["dataset_split"] = "train"
        test_pred = _predict_lambda_for_groups(test_group_agg) if not test_group_agg.empty else pd.DataFrame()
        if not test_pred.empty:
            test_pred["dataset_split"] = "test"

        lookup_dict = build_lambda_lookup_dict(train_pred, lambda_col="lambda_glm", min_intervals=effective_min_intervals)

        glm_test_metrics: Dict[str, float] = {}
        if not test_pred.empty:
            valid = test_pred[
                (test_pred["n_intervals"] >= effective_min_intervals)
                & test_pred["lambda_poisson"].apply(_is_valid_lambda)
                & test_pred["lambda_glm"].apply(_is_valid_lambda)
            ].copy()
            if not valid.empty:
                err = pd.to_numeric(valid["lambda_glm"], errors="coerce") - pd.to_numeric(valid["lambda_poisson"], errors="coerce")
                mae = float(np.abs(err).mean())
                rmse = float(np.sqrt(np.mean(np.square(err))))
                denom = pd.to_numeric(valid["lambda_poisson"], errors="coerce").clip(lower=1e-12)
                mape = float((np.abs(err) / denom).mean())
                glm_test_metrics = {
                    "n_eval_groups": int(len(valid)),
                    "mae_lambda": mae,
                    "rmse_lambda": rmse,
                    "mape_lambda": mape,
                }

        out = train_pred if test_pred.empty else pd.concat([train_pred, test_pred], ignore_index=True)
        out.attrs["lambda_lookup_dict_glm"] = lookup_dict
        out.attrs["global_lambda"] = global_lambda
        out.attrs["ref_hotel"] = ref_hotel
        out.attrs["ref_lt"] = ref_lt
        out.attrs["ref_dur"] = ref_dur
        out.attrs["ref_rs"] = ref_rs
        out.attrs["frequent_hotels"] = _frequent_hotels_from_df(df_train, min_count=4)
        out.attrs["train_test_split_enabled"] = bool(enable_train_test_split)
        out.attrs["train_ratio"] = float(train_ratio)
        out.attrs["split_timestamp_utc"] = str(split_ts) if split_ts is not None else None
        out.attrs["n_rows_train"] = int(len(df_train))
        out.attrs["n_rows_test"] = int(len(df_test))
        out.attrs["fit_min_intervals"] = int(fit_min_intervals)
        out.attrs["lookup_min_intervals"] = int(effective_min_intervals)
        out.attrs["glm_fitted_family"] = str(fitted_family_name) if fitted_family_name is not None else None
        out.attrs["glm_test_metrics"] = glm_test_metrics
        return out.sort_values(["dataset_split", "n_intervals"], ascending=[True, False]).reset_index(drop=True)

    except Exception as exc:
        print(f"GLM fitting failed: {exc}")
        print("Falling back to empirical lambda per group with global fill.")

        def _fallback_exception_frame(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
            out = frame.copy()
            out["dataset_split"] = split_name
            out["lambda_glm"] = np.nan
            out["implied_ttl_glm_hours"] = np.nan
            out["lambda_final"] = out["lambda_poisson"]
            out.loc[~out["lambda_final"].apply(_is_valid_lambda), "lambda_final"] = global_lambda
            out["lambda_method"] = "glm_fallback_exception"
            out["_global_lambda"] = global_lambda
            out["_ref_hotel"] = ref_hotel
            out["_ref_lt"] = ref_lt
            out["_ref_dur"] = ref_dur
            out["_ref_rs"] = ref_rs
            out["_in_lookup"] = out["n_intervals"] >= effective_min_intervals
            return out

        out_train = _fallback_exception_frame(train_group_agg, "train")
        out_test = _fallback_exception_frame(test_group_agg, "test") if not test_group_agg.empty else pd.DataFrame()
        out = out_train if out_test.empty else pd.concat([out_train, out_test], ignore_index=True)
        out.attrs["lambda_lookup_dict_glm"] = {}
        out.attrs["global_lambda"] = global_lambda
        out.attrs["ref_hotel"] = ref_hotel
        out.attrs["ref_lt"] = ref_lt
        out.attrs["ref_dur"] = ref_dur
        out.attrs["ref_rs"] = ref_rs
        out.attrs["frequent_hotels"] = _frequent_hotels_from_df(df_train, min_count=4)
        out.attrs["train_test_split_enabled"] = bool(enable_train_test_split)
        out.attrs["train_ratio"] = float(train_ratio)
        out.attrs["split_timestamp_utc"] = str(split_ts) if split_ts is not None else None
        out.attrs["n_rows_train"] = int(len(df_train))
        out.attrs["n_rows_test"] = int(len(df_test))
        out.attrs["glm_test_metrics"] = {}
        return out





# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# PART 4 — FALLBACK WATERFALL: Serving Functions
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def serve_poisson_ttl(
    hotel_code: str,
    lead_time_days: int,
    duration_nights: int,
    rate_source: str,
    lambda_lookup_dict: Dict[Tuple[str, str, str, str], float],
    global_lambda: float,
    frequent_hotels: set,
    target_freshness: float = 0.8,
    min_ttl_seconds: int = 60,
    max_ttl_seconds: int = 24 * 3600,
) -> int:
    """
    Poisson serving logic:
    1. Bucket lead_time and duration
    2. Encode hotel with rare fallback
    3. Lookup 4-part key
    4. If missing/NaN/<=0, use global_lambda
    5. TTL seconds = 0.2231 / lambda * 3600 (for TARGET_FRESHNESS=0.80)
    """
    lt_bucket = bucket_lead_time(lead_time_days)
    dur_bucket = bucket_duration(duration_nights)
    hotel_encoded = hotel_code if str(hotel_code) in frequent_hotels else "rare_hotel"

    key = (str(hotel_encoded), str(dur_bucket), str(lt_bucket), str(rate_source))
    lam = lambda_lookup_dict.get(key)
    if not _is_valid_lambda(lam):
        lam = global_lambda
    if not _is_valid_lambda(lam):
        lam = 0.1

    return ttl_from_lambda(float(lam), target_freshness, min_ttl_seconds, max_ttl_seconds)

def serve_glm_ttl(
    hotel_code: str,
    lead_time_days: int,
    duration_nights: int,
    rate_source: str,
    lambda_lookup_dict: Dict[Tuple[str, str, str, str], float],
    global_lambda: float,
    frequent_hotels: set,
    ref_dur: str,
    ref_rs: str,
    target_freshness: float = 0.8,
    min_ttl_seconds: int = 60,
    max_ttl_seconds: int = 24 * 3600,
) -> Tuple[int, str]:
    """
    4-level fallback waterfall for GLM lambda lookup at serving time.
    
    Returns: (ttl_seconds, waterfall_level_used)
    """
    
    # Apply bucketing/encoding at serving time
    lt_bucket = bucket_lead_time(lead_time_days)
    dur_bucket = bucket_duration(duration_nights)
    hotel_enc = hotel_code if hotel_code in frequent_hotels else "rare_hotel"
    
    # LEVEL 1: Full personalised prediction
    key = (hotel_enc, dur_bucket, lt_bucket, rate_source)
    lam = lambda_lookup_dict.get(key)
    if _is_valid_lambda(lam):
        ttl = ttl_from_lambda(float(lam), target_freshness, min_ttl_seconds, max_ttl_seconds)
        return ttl, "level_1_full"
    
    # LEVEL 2: Rare hotel fallback (lose hotel, keep duration/lt/rs)
    key = ("rare_hotel", dur_bucket, lt_bucket, rate_source)
    lam = lambda_lookup_dict.get(key)
    if _is_valid_lambda(lam):
        ttl = ttl_from_lambda(float(lam), target_freshness, min_ttl_seconds, max_ttl_seconds)
        return ttl, "level_2_rare_hotel"
    
    # LEVEL 3: Lead time signal only (use ref_dur, ref_rs)
    key = ("rare_hotel", ref_dur, lt_bucket, ref_rs)
    lam = lambda_lookup_dict.get(key)
    if _is_valid_lambda(lam):
        ttl = ttl_from_lambda(float(lam), target_freshness, min_ttl_seconds, max_ttl_seconds)
        return ttl, "level_3_lead_time_only"
    
    # LEVEL 4: Global rate (guaranteed fallback)
    ttl = ttl_from_lambda(global_lambda, target_freshness, min_ttl_seconds, max_ttl_seconds)
    return ttl, "level_4_global"


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS: Unchanged from Original
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def build_lambda_table_fallback(df_enriched: pd.DataFrame, min_intervals: int = 1) -> pd.DataFrame:
    """Robust fallback: empirical per-group lambda with global fill."""
    out = _group_level_stats(df_enriched)
    if out.empty:
        return out

    out = out[out["n_intervals"] >= min_intervals].copy()
    if out.empty:
        return out

    total_exposure = float(out["total_exposure_hours"].sum())
    total_events = float(out["n_events"].sum())
    global_lambda = (total_events / total_exposure) if total_exposure > 0 else 0.1

    out["lambda_fallback"] = (out["n_events"] / out["total_exposure_hours"]).replace([np.inf, -np.inf], np.nan).fillna(global_lambda)
    out["lambda_final"] = out["lambda_fallback"].clip(lower=1e-9)
    out["lambda_method"] = "fallback"
    return out.sort_values("n_intervals", ascending=False).reset_index(drop=True)


def weighted_global_lambda(
    frame: pd.DataFrame,
    lambda_col: str = "lambda_final",
    weight_col: str = "total_exposure_hours",
    default_lambda: float = 0.1,
) -> float:
    """Compute global lambda using sum/sum (correct)."""
    if frame.empty or lambda_col not in frame.columns or weight_col not in frame.columns:
        return float(default_lambda)

    total_lambda_weighted = (frame[lambda_col] * frame[weight_col]).sum()
    total_weight = frame[weight_col].sum()
    
    if total_weight <= 0:
        return float(default_lambda)
    
    value = total_lambda_weighted / total_weight
    if not np.isfinite(value) or value <= 0:
        return float(default_lambda)
    
    return float(value)


def bucket_weighted_lambda(
    frame: pd.DataFrame,
    bucket_col: str = "lead_time_bucket",
    lambda_col: str = "lambda_final",
    weight_col: str = "total_exposure_hours",
) -> dict[str, float]:
    """Compute weighted lambda per bucket."""
    if frame.empty or bucket_col not in frame.columns or lambda_col not in frame.columns:
        return {}

    result = {}
    for bucket in frame[bucket_col].unique():
        bucket_data = frame[frame[bucket_col] == bucket]
        total_lambda_weighted = (bucket_data[lambda_col] * bucket_data[weight_col]).sum()
        total_weight = bucket_data[weight_col].sum()
        
        if total_weight > 0:
            result[str(bucket)] = total_lambda_weighted / total_weight
    
    return result


def ttl_from_lambda(
    lambda_val: float,
    target_freshness: float = 0.8,
    min_ttl_seconds: int = 60,
    max_ttl_seconds: int = 24 * 3600,
) -> int:
    """Convert lambda to TTL seconds with bounds and numeric guards."""
    safe_lambda = max(float(lambda_val), 1e-12)
    target = float(target_freshness)
    if abs(target - 0.8) < 1e-12:
        ttl_hours = TTL_NUMERATOR_TARGET_80 / safe_lambda
    else:
        safe_target = min(max(target, 1e-6), 1.0 - 1e-6)
        ttl_hours = -math.log(safe_target) / safe_lambda
    return int(max(min_ttl_seconds, min(max_ttl_seconds, ttl_hours * 3600.0)))


def build_lambda_table(
    df_enriched: pd.DataFrame,
    min_intervals: int = 5,
    method: str = "km",
    **kwargs,
) -> pd.DataFrame:
    """Unified lambda table builder."""
    method_norm = str(method).strip().lower()
    if method_norm == "km":
        if kwargs:
            raise ValueError("Extra kwargs are not supported for method='km'.")
        return build_lambda_table_km(df_enriched=df_enriched, min_intervals=min_intervals)
    if method_norm == "poisson":
        if kwargs:
            raise ValueError("Extra kwargs are not supported for method='poisson'.")
        return build_lambda_table_poisson(df_enriched=df_enriched, min_intervals=min_intervals)
    if method_norm == "glm":
        return build_lambda_table_glm(df_enriched=df_enriched, min_intervals=min_intervals, **kwargs)
    if method_norm == "fallback":
        if kwargs:
            raise ValueError("Extra kwargs are not supported for method='fallback'.")
        return build_lambda_table_fallback(df_enriched=df_enriched, min_intervals=max(1, min_intervals))
    raise ValueError(f"Unsupported method='{method}'. Use one of: km, poisson, glm, fallback.")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# PART 5 — SANITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def print_sanity_check(
    lambda_poisson_table: pd.DataFrame,
    lambda_glm_table: pd.DataFrame,
    target_freshness: float = 0.8,
) -> None:
    """
    Validate that lambda and TTL ranges are sensible.
    Checks for the double-exponentiation bug and other issues.
    """
    print("\n" + "=" * 100)
    print("SANITY CHECK — Lambda and TTL Ranges")
    print("=" * 100)

    global_lambda = np.nan
    if not lambda_poisson_table.empty and "_global_lambda" in lambda_poisson_table.columns:
        global_lambda = float(lambda_poisson_table["_global_lambda"].iloc[0])
    elif not lambda_glm_table.empty and "_global_lambda" in lambda_glm_table.columns:
        global_lambda = float(lambda_glm_table["_global_lambda"].iloc[0])
    elif not lambda_poisson_table.empty and {"total_changes", "total_exposure_hours"}.issubset(lambda_poisson_table.columns):
        global_lambda = _compute_global_lambda(lambda_poisson_table, default_lambda=np.nan)

    if np.isfinite(global_lambda) and global_lambda > 0:
        global_ttl_hours = TTL_NUMERATOR_TARGET_80 / global_lambda
        print(f"global_lambda: {global_lambda:.10f}")
        print(f"global_implied_ttl_hours: {global_ttl_hours:.6f}")
    else:
        global_ttl_hours = np.nan
        print("global_lambda: NaN")
        print("global_implied_ttl_hours: NaN")

    lam_glm = pd.Series(dtype=float)
    ttl_glm = pd.Series(dtype=float)
    if not lambda_glm_table.empty and "lambda_glm" in lambda_glm_table.columns:
        lam_glm = pd.to_numeric(lambda_glm_table["lambda_glm"], errors="coerce")
        lam_glm = lam_glm[(lam_glm > 0) & np.isfinite(lam_glm)]
        ttl_glm = TTL_NUMERATOR_TARGET_80 / lam_glm
        if len(lam_glm) > 0:
            print(f"lambda_glm_min: {lam_glm.min():.10f}")
            print(f"lambda_glm_max: {lam_glm.max():.10f}")
            print(f"implied_ttl_glm_min_hours: {ttl_glm.min():.6f}")
            print(f"implied_ttl_glm_max_hours: {ttl_glm.max():.6f}")
        else:
            print("lambda_glm_min: NaN")
            print("lambda_glm_max: NaN")
            print("implied_ttl_glm_min_hours: NaN")
            print("implied_ttl_glm_max_hours: NaN")

    lam_poisson = pd.Series(dtype=float)
    ttl_poisson = pd.Series(dtype=float)
    if not lambda_poisson_table.empty and "lambda_poisson" in lambda_poisson_table.columns:
        lam_poisson = pd.to_numeric(lambda_poisson_table["lambda_poisson"], errors="coerce")
        lam_poisson = lam_poisson[(lam_poisson > 0) & np.isfinite(lam_poisson)]
        ttl_poisson = TTL_NUMERATOR_TARGET_80 / lam_poisson
        if len(lam_poisson) > 0:
            print(f"lambda_poisson_min: {lam_poisson.min():.10f}")
            print(f"lambda_poisson_max: {lam_poisson.max():.10f}")
            print(f"implied_ttl_poisson_min_hours: {ttl_poisson.min():.6f}")
            print(f"implied_ttl_poisson_max_hours: {ttl_poisson.max():.6f}")
        else:
            print("lambda_poisson_min: NaN")
            print("lambda_poisson_max: NaN")
            print("implied_ttl_poisson_min_hours: NaN")
            print("implied_ttl_poisson_max_hours: NaN")

    if len(lam_glm) > 0 and (lam_glm.min() < 1e-6 or lam_glm.max() > 100):
        print("WARNING: lambda_glm range indicates potential double-exponentiation bug.")
        print("Check extraction uses fittedvalues / total_exposure_hours (without np.exp).")

    if len(ttl_glm) > 0:
        ttl_mode = ttl_glm.round(6).mode()
        if len(ttl_mode) > 0:
            mode_val = float(ttl_mode.iloc[0])
            mode_share = float((ttl_glm.round(6) == round(mode_val, 6)).mean())
            ttl_min = float(ttl_glm.min())
            if mode_share >= 0.8 and mode_val <= ttl_min * 1.05:
                print("WARNING: most implied_ttl_glm values are identical and near the minimum.")
                print("This pattern is consistent with the double-exponentiation bug.")

    if len(ttl_glm) > 0 and np.isfinite(global_ttl_hours) and global_ttl_hours > 0:
        median_glm_ttl = float(ttl_glm.median())
        if median_glm_ttl > 0:
            ratio = max(global_ttl_hours / median_glm_ttl, median_glm_ttl / global_ttl_hours)
            if ratio > 10:
                print("WARNING: global implied TTL differs from median implied_ttl_glm by >10x.")
                print("Recheck global_lambda calculation is sum(total_changes)/sum(total_exposure_hours).")

    if (
        not lambda_poisson_table.empty
        and "lambda_poisson" in lambda_poisson_table.columns
        and "n_intervals" in lambda_poisson_table.columns
    ):
        poiss = lambda_poisson_table.copy()
        poiss["lambda_poisson"] = pd.to_numeric(poiss["lambda_poisson"], errors="coerce")
        poiss = poiss[np.isfinite(poiss["lambda_poisson"])]
        if not poiss.empty:
            max_idx = poiss["lambda_poisson"].idxmax()
            max_row = poiss.loc[max_idx]
            if float(max_row["lambda_poisson"]) > 100 and int(max_row["n_intervals"]) < 5:
                print("WARNING: lambda_poisson max is from group with n_intervals < 5.")
                print("Apply min_intervals filter before building lookup dictionaries.")

    print("\n" + "=" * 100 + "\n")
