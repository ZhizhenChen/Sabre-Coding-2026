from __future__ import annotations

import argparse
import importlib
import math
import sys
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
LAMBDA_DIR = ROOT_DIR / "Lambda"
LRU_DIR = ROOT_DIR / "LRU"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(LAMBDA_DIR) not in sys.path:
    sys.path.insert(0, str(LAMBDA_DIR))
if str(LRU_DIR) not in sys.path:
    sys.path.insert(0, str(LRU_DIR))

from Cache_System_Workflow.sabre_cache_workflow_v2 import TruthPriceProvider, PreparedWorkflowInput, RequestContext, SabreCacheWorkflow
from demand_forecasting.model_input import DemandScoreGenerator
from LRU.simulate_lru_baseline import VanillaLRUCache
from data_processing.pipeline import DataPipelineProcessor

lambda_model = importlib.import_module("lambda_model")
build_lambda_table = lambda_model.build_lambda_table
build_lambda_lookup_dict = lambda_model.build_lambda_lookup_dict
_km_survival = lambda_model._km_survival
_median_from_survival = lambda_model._median_from_survival
_ttl_from_lambda_model = lambda_model.ttl_from_lambda
_bucket_lead_time_model = lambda_model.bucket_lead_time
_bucket_duration_model = lambda_model.bucket_duration

LEAD_TIME_LABELS = ["same_day", "1_to_3d", "4_to_7d", "8_to_14d", "15_to_30d", "31_to_90d", "91plus"]
LEAD_TIME_BREAKS = [0, 3, 7, 14, 30, 90]
TTL_TARGET_FRESHNESS = 0.9
MIN_TTL_SECONDS = 1
MAX_TTL_SECONDS = 24 * 3600
DEFAULT_LAMBDA = 0.1


@dataclass
class EvalSummary:
    ttl_method: str
    total_requests: int
    unique_keys: int
    hit_count: int
    miss_count: int
    provider_calls: int
    provider_refresh_calls: int
    provider_prefetch_calls: int
    provider_miss_calls: int
    hit_rate: float
    stale_response_count: int
    stale_rate_served_pct: float
    prewarm_precision: float
    useful_prewarm: int
    total_prewarm: int
    eviction_count: int
    eviction_accuracy: float
    wrong_eviction_count: int
    avg_query_count_after_eviction: float
    api_call_reduction_pct_vs_lru: float
    admission_scores: List[float]


@dataclass
class LRUSummary:
    total_requests: int
    unique_keys: int
    hit_count: int
    miss_count: int
    provider_calls: int
    hit_rate: float
    stale_response_count: int
    stale_rate_served_pct: float
    eviction_count: int
    eviction_accuracy: float
    wrong_eviction_count: int
    avg_query_count_after_eviction: float



def _build_truth_price_lookup(source_df: pd.DataFrame) -> Dict[str, Dict[str, List[Any]]]:
    """
    Build a time-ordered price lookup for each cache_key from unexploded request data.
    
    For each cache_key, stores all observed price offers in chronological order.
    This allows matching served prices against ground truth at the exact request time.
    
    Args:
        source_df: unexploded request data (convertedrate_infos not yet expanded)
    
    Returns:
        Dict mapping cache_key -> List of {timestamp, offers} dicts, sorted by timestamp
    """
    lookup: Dict[str, Dict[str, List[Any]]] = {}
    
    df = source_df.dropna(subset=["cache_key", "rq_timestamp", "convertedrate_infos"]).copy()
    df["rq_timestamp"] = pd.to_datetime(df["rq_timestamp"], errors="coerce", utc=True)
    df = df.sort_values(["cache_key", "rq_timestamp"])
    
    for cache_key, group in df.groupby("cache_key", dropna=False):
        timestamps: List[Any] = []
        offers_by_timestamp: List[List[Dict[str, Any]]] = []
        for row in group.itertuples(index=False):
            timestamp = row.rq_timestamp
            offers = []
            
            # Extract all offers from convertedrate_infos
            rate_infos = row.convertedrate_infos
            group = row.rate_group
            if isinstance(rate_infos, np.ndarray):
                rate_infos = rate_infos.tolist()
            elif isinstance(rate_infos, tuple):
                rate_infos = list(rate_infos)

            if rate_infos is not None and isinstance(rate_infos, list):
                rate_group = group
                for offer in rate_infos:
                    if isinstance(offer, dict):
                        price = offer.get("amount_after_tax")
                        source = offer.get("rate_source", "unknown")
                        if price is not None:
                            offers.append({
                                "source": str(source),
                                "rate_group": (str(rate_group) if rate_group is not None else None),
                                "price": float(price),
                            })
            
            if offers:
                timestamps.append(timestamp)
                offers_by_timestamp.append(offers)

        if timestamps:
            lookup[str(cache_key)] = {
                "timestamps": timestamps,
                "offers": offers_by_timestamp,
            }
    
    return lookup


def _extract_price(payload: Any) -> float | None:
    try:
        offers = payload.get("offers", [])
        if not offers:
            return None
        return float(offers[0].get("price"))
    except Exception:
        return None


def _extract_rate_group(payload: Any) -> str | None:
    try:
        group = payload.get("rate_group")
        if group is not None and str(group).strip() != "":
            return str(group)
        offers = payload.get("offers", [])
        if not offers:
            return None
        first = offers[0] if isinstance(offers[0], dict) else {}
        group = first.get("rate_group")
        if group is None or str(group).strip() == "":
            return None
        return str(group)
    except Exception:
        return None


def _get_truth_rate_group_for_request(
    cache_key: str,
    request_timestamp: Any,
    truth_price_by_key: Dict[str, Dict[str, List[Any]]],
) -> str | None:
    if cache_key not in truth_price_by_key:
        return None

    time_series = truth_price_by_key[cache_key]
    request_ts = pd.Timestamp(request_timestamp).to_pydatetime() if not isinstance(request_timestamp, (pd.Timestamp, datetime)) else request_timestamp
    if hasattr(request_ts, "replace"):
        request_ts = request_ts.replace(tzinfo=timezone.utc) if request_ts.tzinfo is None else request_ts

    timestamps = time_series.get("timestamps", [])
    offers_list = time_series.get("offers", [])
    closest_idx = bisect_right(timestamps, request_ts) - 1
    if closest_idx >= 0 and closest_idx < len(offers_list):
        offers = offers_list[closest_idx]
        if offers:
            first = offers[0] if isinstance(offers[0], dict) else {}
            group = first.get("rate_group")
            if group is not None and str(group).strip() != "":
                return str(group)
    return None


def _get_truth_price_for_request(
    cache_key: str,
    request_timestamp: Any,
    truth_price_by_key: Dict[str, Dict[str, List[Any]]],
) -> float | None:
    """
    Get ground truth price for a request by matching its cache_key and timestamp.
    
    Finds the most recent observation of the cache_key with timestamp <= request_timestamp.
    
    Args:
        cache_key: the request's cache key
        request_timestamp: the request's timestamp
        truth_price_by_key: time-ordered price lookup
    
    Returns:
        First price from the matched observation's offers, or None if not found
    """
    if cache_key not in truth_price_by_key:
        return None
    
    time_series = truth_price_by_key[cache_key]
    request_ts = pd.Timestamp(request_timestamp).to_pydatetime() if not isinstance(request_timestamp, (pd.Timestamp, datetime)) else request_timestamp
    if hasattr(request_ts, 'replace'):
        request_ts = request_ts.replace(tzinfo=timezone.utc) if request_ts.tzinfo is None else request_ts
    
    timestamps = time_series.get("timestamps", [])
    offers_list = time_series.get("offers", [])
    closest_idx = bisect_right(timestamps, request_ts) - 1

    if closest_idx >= 0 and closest_idx < len(offers_list):
        offers = offers_list[closest_idx]
        if offers:
            return float(offers[0].get("price"))
    return None


def _assign_lead_time_bucket(lead_time_days: int) -> str:
    return str(_bucket_lead_time_model(int(lead_time_days)))


def _bucketize_lead_time(series: pd.Series) -> pd.Series:
    return series.fillna(0).astype(int).apply(_assign_lead_time_bucket)


def _ttl_from_lambda(lambda_val: float) -> int:
    return _ttl_from_lambda_model(
        lambda_val=lambda_val,
        target_freshness=TTL_TARGET_FRESHNESS,
        min_ttl_seconds=MIN_TTL_SECONDS,
        max_ttl_seconds=MAX_TTL_SECONDS,
    )


def _lambda_from_ttl_seconds(ttl_seconds: int) -> float:
    safe_ttl = max(float(ttl_seconds), 1.0)
    safe_target = min(max(float(TTL_TARGET_FRESHNESS), 1e-6), 1.0 - 1e-6)
    return float(-math.log(safe_target) / safe_ttl)


def _extract_rate_source_for_admission(row: Any) -> str:
    # source_df rows contain convertedrate_infos (list/ndarray of dict offers).
    rate_infos = getattr(row, "convertedrate_infos", None)
    if isinstance(rate_infos, np.ndarray):
        rate_infos = rate_infos.tolist()
    elif isinstance(rate_infos, tuple):
        rate_infos = list(rate_infos)
    if isinstance(rate_infos, list):
        for offer in rate_infos:
            if isinstance(offer, dict):
                rs = offer.get("rate_source")
                if rs is not None and str(rs).strip() != "":
                    return str(rs)
    # Fallback for exploded frames.
    rs_col = getattr(row, "rate_source", None)
    if rs_col is not None and str(rs_col).strip() != "":
        return str(rs_col)
    return "unknown"


def _safe_global_lambda_from_table(table: pd.DataFrame) -> float:
    global_lambda = table.attrs.get("global_lambda")
    if global_lambda is not None and lambda_model._is_valid_lambda(global_lambda):
        return float(global_lambda)
    valid = table[table["lambda_final"].apply(lambda_model._is_valid_lambda)].copy()
    if valid.empty:
        return DEFAULT_LAMBDA
    return _weighted_global_lambda(valid)


def _weighted_global_lambda(valid_df: pd.DataFrame) -> float:
    total_weight = pd.to_numeric(valid_df["total_exposure_hours"], errors="coerce").fillna(0.0).sum()
    if total_weight <= 0:
        return DEFAULT_LAMBDA
    return float((valid_df["lambda_final"] * valid_df["total_exposure_hours"]).sum() / total_weight)


def _default_bucket_lookups() -> Tuple[Dict[str, int], Dict[str, float]]:
    ttl_fallback = {bucket: _ttl_from_lambda(DEFAULT_LAMBDA) for bucket in LEAD_TIME_LABELS}
    lam_fallback = {bucket: DEFAULT_LAMBDA for bucket in LEAD_TIME_LABELS}
    return ttl_fallback, lam_fallback


def _filter_lookup_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "dataset_split" in result.columns:
        train_frame = result[result["dataset_split"] == "train"].copy()
        if not train_frame.empty:
            result = train_frame
    if "_in_lookup" in result.columns:
        in_lookup = result[result["_in_lookup"] == True].copy()  # noqa: E712
        if not in_lookup.empty:
            result = in_lookup
    return result


def _build_admission_lambda_state(source_df: pd.DataFrame, ttl_method: str) -> Dict[str, Any]:
    method = ttl_method.strip().lower()
    if method == "pp":
        table = build_lambda_table(source_df, min_intervals=1, method="poisson")
        lambda_col = "lambda_poisson" if "lambda_poisson" in table.columns else "lambda_final"
    elif method == "glm":
        table = build_lambda_table(source_df, min_intervals=1, method="glm")
        table = _filter_lookup_frame(table)
        lambda_col = "lambda_glm" if "lambda_glm" in table.columns else "lambda_final"
    elif method == "km":
        table = build_lambda_table(source_df, min_intervals=1, method="km")
        lambda_col = "lambda_final"
    else:
        # rule_based admission lambda still comes from lambda_model fallback estimation.
        table = build_lambda_table(source_df, min_intervals=1, method="fallback")
        lambda_col = "lambda_final"

    if table.empty:
        return {
            "method": method,
            "lambda_lookup": {},
            "global_lambda": DEFAULT_LAMBDA,
            "frequent_hotels": set(),
            "ref_dur": "1_night",
            "ref_rs": "unknown",
        }

    table = _filter_lookup_frame(table)

    lookup = build_lambda_lookup_dict(table, lambda_col=lambda_col, min_intervals=1)
    global_lambda = _safe_global_lambda_from_table(table)
    frequent_hotels = table.attrs.get("frequent_hotels")
    if frequent_hotels is None:
        counts = source_df["hotel_code"].astype(str).value_counts()
        frequent_hotels = set(counts[counts >= 4].index.tolist())

    return {
        "method": method,
        "lambda_lookup": lookup,
        "global_lambda": float(global_lambda),
        "frequent_hotels": set(frequent_hotels),
        "ref_dur": str(table.attrs.get("ref_dur", "1_night")),
        "ref_rs": str(table.attrs.get("ref_rs", "unknown")),
    }


def _admission_lambda_for_request(
    request: RequestContext,
    rate_source: str,
    admission_state: Dict[str, Any],
    ttl_lookup_by_bucket: Dict[str, int] | None = None,
) -> float:
    method = str(admission_state.get("method", "fallback"))
    lambda_lookup = admission_state.get("lambda_lookup", {})
    frequent_hotels = admission_state.get("frequent_hotels", set())
    global_lambda = float(admission_state.get("global_lambda", 0.1))
    ref_dur = str(admission_state.get("ref_dur", "1_night"))
    ref_rs = str(admission_state.get("ref_rs", "unknown"))

    lead_time_days = int(request.lead_time_days) if request.lead_time_days is not None else max(0, request.duration)
    lead_bucket = _assign_lead_time_bucket(lead_time_days)
    dur_bucket = _bucket_duration_model(int(request.duration))
    hotel_enc = str(request.hotel_code) if str(request.hotel_code) in frequent_hotels else "rare_hotel"
    rs = str(rate_source) if rate_source is not None else "unknown"

    if method == "glm":
        # Same waterfall shape as lambda_model.serve_glm_ttl, but return lambda.
        key = (hotel_enc, dur_bucket, lead_bucket, rs)
        lam = lambda_lookup.get(key)
        if lambda_model._is_valid_lambda(lam):
            return float(lam)
        key = ("rare_hotel", dur_bucket, lead_bucket, rs)
        lam = lambda_lookup.get(key)
        if lambda_model._is_valid_lambda(lam):
            return float(lam)
        key = ("rare_hotel", ref_dur, lead_bucket, ref_rs)
        lam = lambda_lookup.get(key)
        if lambda_model._is_valid_lambda(lam):
            return float(lam)
    else:
        key = (hotel_enc, dur_bucket, lead_bucket, rs)
        lam = lambda_lookup.get(key)
        if lambda_model._is_valid_lambda(lam):
            return float(lam)

    if lambda_model._is_valid_lambda(global_lambda):
        return float(global_lambda)

    if ttl_lookup_by_bucket is not None:
        ttl_seconds = ttl_lookup_by_bucket.get(lead_bucket)
        if ttl_seconds is not None:
            return _lambda_from_ttl_seconds(ttl_seconds)
    return DEFAULT_LAMBDA


def _build_rule_based_ttl_lookup() -> Dict[str, int]:
    # A simple deterministic baseline policy.
    return {
        "same_day": 1 * 3600,
        "1_to_3d": 2 * 3600,
        "4_to_7d": 4 * 3600,
        "8_to_14d": 6 * 3600,
        "15_to_30d": 8 * 3600,
        "31_to_90d": 10 * 3600,
        "91plus": 12 * 3600,
    }


def _build_km_ttl_lookup(source_df: pd.DataFrame) -> Dict[str, int]:
    if source_df.empty:
        return {}

    frame = source_df.copy()
    frame["lead_time_bucket"] = _bucketize_lead_time(frame["lead_time"])
    frame["rq_timestamp"] = pd.to_datetime(frame["rq_timestamp"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["rq_timestamp", "price_change", "lead_time_bucket", "cache_key"])

    ttl_lookup: Dict[str, int] = {}
    for bucket, bucket_df in frame.groupby("lead_time_bucket", dropna=False):
        grouped = bucket_df.sort_values(["cache_key", "rq_timestamp"]).copy()
        grouped["delta_hours"] = grouped.groupby("cache_key")["rq_timestamp"].diff().dt.total_seconds() / 3600.0
        grouped["event_change"] = grouped["price_change"].astype(int)
        km_df = grouped.dropna(subset=["delta_hours"])
        km_df = km_df[km_df["delta_hours"] > 0]
        if km_df.empty:
            continue

        durations = km_df["delta_hours"].to_numpy(dtype=float)
        events = km_df["event_change"].to_numpy(dtype=int)
        times, survival = _km_survival(durations, events)
        if len(times) == 0:
            continue

        hit = np.where(survival <= TTL_TARGET_FRESHNESS)[0]
        if len(hit) > 0:
            ttl_hours = float(times[hit[0]])
        else:
            med = _median_from_survival(times, survival)
            ttl_hours = float(med) if np.isfinite(med) else 1.0

        ttl_lookup[str(bucket)] = int(max(MIN_TTL_SECONDS, min(MAX_TTL_SECONDS, ttl_hours * 3600.0)))

    if len(ttl_lookup) < len(LEAD_TIME_LABELS):
        fallback_table = build_lambda_table(source_df, min_intervals=1, method="fallback")
        if not fallback_table.empty:
            fallback_table = fallback_table.copy()
            valid = fallback_table[
                fallback_table["lambda_final"].apply(lambda_model._is_valid_lambda)
            ].copy()
            global_lambda = _weighted_global_lambda(valid) if not valid.empty else DEFAULT_LAMBDA
            fallback_ttl = _ttl_from_lambda(global_lambda)
            for bucket in LEAD_TIME_LABELS:
                ttl_lookup.setdefault(bucket, fallback_ttl)

    return ttl_lookup


def _build_lambda_based_bucket_lookups(source_df: pd.DataFrame, method: str) -> Tuple[Dict[str, int], Dict[str, float]]:
    table = build_lambda_table(source_df, min_intervals=1, method=method)
    if table.empty:
        table = build_lambda_table(source_df, min_intervals=1, method="fallback")
    if table.empty:
        return _default_bucket_lookups()

    frame = _filter_lookup_frame(table)

    valid = frame[frame["lambda_final"].apply(lambda_model._is_valid_lambda)].copy()
    if valid.empty:
        return _default_bucket_lookups()

    weighted_lambda: Dict[str, float] = {}
    for bucket, bucket_df in valid.groupby("lead_time_bucket", dropna=False):
        weight = pd.to_numeric(bucket_df["total_exposure_hours"], errors="coerce").fillna(0.0).sum()
        if weight > 0:
            weighted_lambda[str(bucket)] = float((bucket_df["lambda_final"] * bucket_df["total_exposure_hours"]).sum() / weight)

    global_lambda_attr = frame.attrs.get("global_lambda")
    if global_lambda_attr is not None and lambda_model._is_valid_lambda(global_lambda_attr):
        global_lambda = float(global_lambda_attr)
    else:
        global_lambda = _weighted_global_lambda(valid)

    lookup: Dict[str, int] = {}
    lambda_lookup: Dict[str, float] = {}
    for bucket in LEAD_TIME_LABELS:
        lam = float(weighted_lambda.get(bucket, global_lambda))
        lambda_lookup[bucket] = lam
        lookup[bucket] = _ttl_from_lambda(lam)

    return lookup, lambda_lookup


def _build_ttl_lookup(source_df: pd.DataFrame, ttl_method: str) -> Dict[str, int]:
    method = ttl_method.strip().lower()
    if method == "rule_based":
        return _build_rule_based_ttl_lookup()
    if method == "km":
        return _build_km_ttl_lookup(source_df)
    if method == "pp":
        ttl_lookup, _ = _build_lambda_based_bucket_lookups(source_df, method="poisson")
        return ttl_lookup
    if method == "glm":
        ttl_lookup, _ = _build_lambda_based_bucket_lookups(source_df, method="glm")
        return ttl_lookup
    raise ValueError(f"Unsupported ttl_method='{ttl_method}'. Use one of: glm, pp, rule_based, km")


def _request_context_from_row(row: Any) -> RequestContext:
    return RequestContext(
        rq_timestamp=row.rq_timestamp.to_pydatetime() if hasattr(row.rq_timestamp, "to_pydatetime") else row.rq_timestamp,
        chain_code=str(row.chain_code),
        stay_start_date=str(pd.Timestamp(row.rq_stay_start_date).strftime("%Y-%m-%d")),
        stay_end_date=str(pd.Timestamp(row.rq_stay_end_date).strftime("%Y-%m-%d")),
        duration=int(row.duration),
        city_code=str(row.location_city_code),
        lead_time_days=int(row.lead_time) if hasattr(row, "lead_time") else None,
        cache_key_value=str(row.cache_key),
        hotel_code=str(row.hotel_code) if hasattr(row, "hotel_code") else None,
    )

def _prepare_requests(
    requests_df: pd.DataFrame,
    p_reuse_df: pd.DataFrame,
    ttl_lookup_by_bucket: Dict[str, int],
    admission_state: Dict[str, Any],
) -> List[PreparedWorkflowInput]:
    p_lookup = (
        p_reuse_df.dropna(subset=["cache_key", "p_reuse"])
        .groupby("cache_key", as_index=False)["p_reuse"]
        .mean()
    )
    merged = requests_df.merge(p_lookup, on=["cache_key"], how="left")
    merged["p_reuse"] = merged["p_reuse"].fillna(0.5)

    prepared: List[PreparedWorkflowInput] = []
    for row in merged.itertuples(index=False):
        request = _request_context_from_row(row)
        rate_source = _extract_rate_source_for_admission(row)
        # Admission lambda_i: prefer lambda-model bucket lookup; fallback to TTL-inverted lambda.
        lambda_i = _admission_lambda_for_request(
            request,
            rate_source=rate_source,
            admission_state=admission_state,
            ttl_lookup_by_bucket=ttl_lookup_by_bucket,
        )
        prepared.append(
            PreparedWorkflowInput(
                request=request,
                p_reuse=float(row.p_reuse),
                lambda_i=lambda_i,
            )
        )
    return prepared


def _format_distribution_lines(values: List[float], name: str) -> List[str]:
    if not values:
        return [f"{name}: no valid values"]
    s = pd.Series(values, dtype=float)
    desc = s.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    lines = [f"{name}:"]
    for k in ["count", "mean", "std", "min", "1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%", "max"]:
        if k in desc.index:
            lines.append(f"  {k}: {float(desc[k]):.6f}")
    return lines


def _compute_eviction_metrics(
    eviction_events: List[Tuple[str, int]],
    requests_by_key: Dict[str, List[int]],
) -> Tuple[int, float, int, float]:
    eviction_count = len(eviction_events)
    later_query_counts: List[int] = []
    queried_after_eviction = 0
    wrong_eviction_count = 0
    for evicted_key, evict_idx in eviction_events:
        future_queries = len(requests_by_key[evicted_key]) - bisect_right(requests_by_key[evicted_key], evict_idx)
        later_query_counts.append(future_queries)
        if future_queries > 0:
            queried_after_eviction += 1
        if future_queries > 5:
            wrong_eviction_count += 1
    eviction_accuracy = 1 - queried_after_eviction / eviction_count if eviction_count else 0.0
    avg_query_count_after_eviction = mean(later_query_counts) if later_query_counts else 0.0
    return eviction_count, eviction_accuracy, wrong_eviction_count, avg_query_count_after_eviction


def _run_ttl_method(
    ttl_method: str,
    requests_df: pd.DataFrame,
    prepared_requests: List[PreparedWorkflowInput],
    ttl_lookup_by_bucket: Dict[str, int],
    truth_price_by_key: Dict[str, Dict[str, List[Any]]],
    lru_provider_calls: int,
    controlled_capacity: int = 100,
    uncontrolled_capacity: int = 900,
    score_percentile: float = 0.7,
    prefetch_ratio: float = 0.2,
) -> EvalSummary:
    provider = TruthPriceProvider(truth_price_by_key)
    workflow = SabreCacheWorkflow(
        controlled_capacity=controlled_capacity,
        uncontrolled_capacity=uncontrolled_capacity,
        score_percentile=score_percentile,
        prefetch_ratio=prefetch_ratio,
        lead_time_breaks=LEAD_TIME_BREAKS,
        lead_time_labels=LEAD_TIME_LABELS,
        km_ttl_lookup_by_bucket=ttl_lookup_by_bucket,
    )

    candidate_tuples = [
        (
            item.request,
            item.p_reuse,
            item.lambda_i,
        )
        for item in prepared_requests
    ]
    prefetched_keys = workflow.prefetch_controlled(candidate_tuples, provider)

    hit_count = 0
    miss_count = 0
    stale_response_count = 0
    admission_scores: List[float] = []
    eviction_events: List[Tuple[str, int]] = []
    requests_by_key: Dict[str, List[int]] = defaultdict(list)
    hit_keys: set[str] = set()

    for idx, item in enumerate(prepared_requests, start=1):
        request = item.request
        key = request.cache_key()
        requests_by_key[key].append(idx)
        result = workflow.get(request, p_reuse=item.p_reuse, lambda_i=item.lambda_i, provider=provider)
        admission_scores.append(float(result.admission_score))
        evicted_controlled_key = getattr(result, "evicted_controlled_key", None)
        evicted_uncontrolled_key = getattr(result, "evicted_uncontrolled_key", None)
        # Only count keys that leave the whole cache system.
        # Controlled -> uncontrolled demotion is not a final eviction.
        if evicted_uncontrolled_key is not None:
            eviction_events.append((evicted_uncontrolled_key, idx))
        if evicted_controlled_key is None and evicted_uncontrolled_key is None and result.evicted_key is not None:
            # Backward-compatible fallback path.
            if result.tier == "uncontrolled":
                eviction_events.append((result.evicted_key, idx))

        if "hit" in result.source:
            hit_count += 1
            hit_keys.add(key)
            served_price = _extract_price(result.payload)
            served_rate_group = _extract_rate_group(result.payload)
            truth_price = _get_truth_price_for_request(key, request.rq_timestamp, truth_price_by_key)
            truth_rate_group = _get_truth_rate_group_for_request(key, request.rq_timestamp, truth_price_by_key)
            if (
                served_price is not None
                and truth_price is not None
                and served_rate_group is not None
                and truth_rate_group is not None
                and str(served_rate_group) == str(truth_rate_group)
                and abs(served_price - truth_price) / max(served_price, 1e-6) > 0.01
            ):
                stale_response_count += 1
        else:
            miss_count += 1

    total_requests = len(prepared_requests)
    unique_keys = len({item.request.cache_key() for item in prepared_requests})
    hit_rate = hit_count / total_requests if total_requests else 0.0

    useful_prewarm = len(set(prefetched_keys) & hit_keys)
    total_prewarm = len(prefetched_keys)
    prewarm_precision = useful_prewarm / total_prewarm if total_prewarm else 0.0

    evictions, eviction_accuracy, wrong_eviction_count, avg_query_count_after_eviction = _compute_eviction_metrics(
        eviction_events,
        requests_by_key,
    )
    stale_rate_served_pct = stale_response_count / hit_count if hit_count else 0.0

    api_call_reduction = 1.0 - (provider.calls / lru_provider_calls) if lru_provider_calls else 0.0
    provider_call_counts = workflow.provider_call_counts()

    return EvalSummary(
        ttl_method=ttl_method,
        total_requests=total_requests,
        unique_keys=unique_keys,
        hit_count=hit_count,
        miss_count=miss_count,
        provider_calls=provider.calls,
        provider_refresh_calls=int(provider_call_counts.get("refresh", 0)),
        provider_prefetch_calls=int(provider_call_counts.get("prefetch", 0)),
        provider_miss_calls=int(provider_call_counts.get("miss", 0)),
        hit_rate=hit_rate,
        stale_response_count=stale_response_count,
        stale_rate_served_pct=stale_rate_served_pct,
        prewarm_precision=prewarm_precision,
        useful_prewarm=useful_prewarm,
        total_prewarm=total_prewarm,
        eviction_count=evictions,
        eviction_accuracy=eviction_accuracy,
        wrong_eviction_count=wrong_eviction_count,
        avg_query_count_after_eviction=avg_query_count_after_eviction,
        api_call_reduction_pct_vs_lru=api_call_reduction * 100.0,
        admission_scores=admission_scores,
    )


def _run_lru_baseline(
    requests_df: pd.DataFrame,
    truth_price_by_key: Dict[str, Dict[str, List[Any]]],
    lru_capacity: int = 1000,
) -> LRUSummary:
    provider = TruthPriceProvider(truth_price_by_key)
    lru = VanillaLRUCache(capacity=lru_capacity)

    eviction_events: List[Tuple[str, int]] = []
    requests_by_key: Dict[str, List[int]] = defaultdict(list)
    stale_response_count = 0

    for idx, row in enumerate(requests_df.itertuples(index=False), start=1):
        request = _request_context_from_row(row)
        key = request.cache_key()
        requests_by_key[key].append(idx)
        result = lru.get(request, provider)
        if result.evicted_key is not None:
            eviction_events.append((result.evicted_key, idx))
        if "hit" in result.source:
            served_price = _extract_price(result.payload)
            served_rate_group = _extract_rate_group(result.payload)
            truth_price = _get_truth_price_for_request(key, request.rq_timestamp, truth_price_by_key)
            truth_rate_group = _get_truth_rate_group_for_request(key, request.rq_timestamp, truth_price_by_key)
            if (
                served_price is not None
                and truth_price is not None
                and served_rate_group is not None
                and truth_rate_group is not None
                and str(served_rate_group) == str(truth_rate_group)
                and abs(served_price - truth_price) / max(served_price, 1e-6) > 0.01
            ):
                stale_response_count += 1

    total_requests = len(requests_df)
    unique_keys = int(requests_df["cache_key"].nunique())
    hit_rate = lru.hits / total_requests if total_requests else 0.0
    eviction_count = len(eviction_events)

    later_query_counts: List[int] = []
    queried_after_eviction = 0
    wrong_eviction_count = 0
    for evicted_key, evict_idx in eviction_events:
        future_queries = len(requests_by_key[evicted_key]) - bisect_right(requests_by_key[evicted_key], evict_idx)
        later_query_counts.append(future_queries)
        if future_queries > 0:
            queried_after_eviction += 1
        if future_queries > 5:
            wrong_eviction_count += 1

    return LRUSummary(
        total_requests=total_requests,
        unique_keys=unique_keys,
        hit_count=lru.hits,
        miss_count=lru.misses,
        provider_calls=provider.calls,
        hit_rate=hit_rate,
        stale_response_count=stale_response_count,
        stale_rate_served_pct=stale_response_count / lru.hits if lru.hits else 0.0,
        eviction_count=eviction_count,
        eviction_accuracy=1- queried_after_eviction / eviction_count if eviction_count else 0.0,
        wrong_eviction_count=wrong_eviction_count,
        avg_query_count_after_eviction=mean(later_query_counts) if later_query_counts else 0.0,
    )


def run_ttl_method_eval(
    start_date: str | None = None,
    end_date: str | None = None,
    max_requests: int = 3000,
    output_path: str = "workflow_ttl_methods_eval_2026-02-07_3000.txt",
    controlled_capacity: int = 100,
    uncontrolled_capacity: int = 900,
    lru_capacity: int = 1000,
    score_percentile: float = 0.7,
    prefetch_ratio: float = 0.2,
) -> None:
    processor = DataPipelineProcessor(data_root=str(ROOT_DIR / "data" / "cleaned_partitioned"))

    source_df, prepared_df = processor.process(start_date=start_date, end_date=end_date, max_requests=max_requests)
    print(f"Loaded and processed data: source_df={len(source_df)} rows, prepared_df={len(prepared_df)} rows")

    # Build a dedicated frame for lambda estimation only (needs price_change).
    pricing_source_df = processor._process_rates_and_prices(source_df.copy())
    pricing_source_df = processor._compute_market_and_price_change(pricing_source_df)

    model_generator = DemandScoreGenerator()
    p_reuse_df = model_generator.generate_demand_scores(
        processed_df=prepared_df,
        source_df=source_df,
        num_samples=None,
        output_parquet=None,
    )
    print(f"Generated demand scores: p_reuse_df={len(p_reuse_df)} rows")

    # Build time-ordered truth price lookup from unexploded source data
    truth_price_by_key = _build_truth_price_lookup(source_df)
    print(f"Built truth price lookup for {len(truth_price_by_key)} unique cache keys")

    lru_summary = _run_lru_baseline(source_df, truth_price_by_key=truth_price_by_key, lru_capacity=lru_capacity)
    print(
        "Completed LRU baseline evaluation: "
        f"hit_rate={lru_summary.hit_rate:.4f}, "
        f"provider_calls={lru_summary.provider_calls}, "
        f"stale_rate={lru_summary.stale_rate_served_pct:.4f}, "
        f"evictions={lru_summary.eviction_count}"
    )

    ttl_methods = ["pp"]
    evals: List[EvalSummary] = []
    ttl_lookups: Dict[str, Dict[str, int]] = {}
    admission_states: Dict[str, Dict[str, Any]] = {}
    method_lambda_dist: Dict[str, List[float]] = {}
    method_ttl_dist: Dict[str, List[float]] = {}
    lines: List[str] = []
    for method in ttl_methods:
        # if method == "glm":
        #     ttl_lookup, _ = _build_lambda_based_bucket_lookups(pricing_source_df, method="glm")
        # elif method == "pp":
        #     ttl_lookup, _ = _build_lambda_based_bucket_lookups(pricing_source_df, method="poisson")
        # else:
        ttl_lookup = _build_ttl_lookup(pricing_source_df, ttl_method=method)
        ttl_lookups[method] = ttl_lookup
        admission_states[method] = _build_admission_lambda_state(pricing_source_df, ttl_method=method)
        # Prepare requests with lambda_i from this TTL method
        prepared_requests = _prepare_requests(
            source_df,
            p_reuse_df,
            ttl_lookup_by_bucket=ttl_lookup,
            admission_state=admission_states[method],
        )
        method_lambda_dist[method] = [float(x.lambda_i) for x in prepared_requests if np.isfinite(float(x.lambda_i))]
        ttl_vals: List[float] = []
        for x in prepared_requests:
            lead_time_days = int(x.request.lead_time_days) if x.request.lead_time_days is not None else max(0, x.request.duration)
            bucket = _assign_lead_time_bucket(lead_time_days)
            ttl = ttl_lookup.get(bucket)
            if ttl is not None:
                ttl_vals.append(float(ttl))
        method_ttl_dist[method] = ttl_vals
        print(f"Prepared requests for {method.upper()}: {len(prepared_requests)}")
        evals.append(
            _run_ttl_method(
                ttl_method=method,
                requests_df=source_df,
                prepared_requests=prepared_requests,
                ttl_lookup_by_bucket=ttl_lookup,
                truth_price_by_key=truth_price_by_key,
                lru_provider_calls=lru_summary.provider_calls,
                controlled_capacity=controlled_capacity,
                uncontrolled_capacity=uncontrolled_capacity,
                score_percentile=score_percentile,
                prefetch_ratio=prefetch_ratio,
            )
        )
        latest = evals[-1]
        print(
            f"Completed TTL method evaluation for {method.upper()}: "
            f"hit_rate={latest.hit_rate:.4f}, "
            f"provider_calls={latest.provider_calls}, "
            f"stale_rate={latest.stale_rate_served_pct:.4f}, "
            f"evictions={latest.eviction_count}"
        )

    lines.append("TTL METHOD EVAL (admission score uses TTL-implied lambda)")

    lines.append(f"start_date={start_date}")
    lines.append(f"end_date={end_date}")

    lines.append(f"sample_requests={len(source_df)}")
    lines.append(f"unique_request_keys={source_df['cache_key'].nunique()}")
    lines.append(f"source_rows_after_explode={len(pricing_source_df)}")
    lines.append(f"prepared_feature_rows={len(prepared_df)}")
    lines.append(f"p_reuse_rows={len(p_reuse_df)}")
    lines.append(f"controlled_capacity={controlled_capacity}")
    lines.append(f"uncontrolled_capacity={uncontrolled_capacity}")
    lines.append(f"lru_capacity={lru_capacity}")

    lines.append("")

    lines.append("=== LRU Baseline ===")
    lines.append(f"requests_total: {lru_summary.total_requests}")
    lines.append(f"hit_rate: {lru_summary.hit_rate:.4f}")
    lines.append(f"hit_count: {lru_summary.hit_count}")
    lines.append(f"miss_count: {lru_summary.miss_count}")
    lines.append(f"provider_total_calls: {lru_summary.provider_calls}")
    lines.append(f"stale_rate_served_pct: {lru_summary.stale_rate_served_pct:.4f}")
    lines.append(f"eviction_count: {lru_summary.eviction_count}")
    lines.append(f"eviction_accuracy: {lru_summary.eviction_accuracy:.4f}")
    lines.append(f"wrong_eviction_count: {lru_summary.wrong_eviction_count}")
    lines.append(f"avg_query_count_after_eviction: {lru_summary.avg_query_count_after_eviction:.4f}")
    lines.append("")

    for summary in evals:
        lines.append(f"=== TTL Method: {summary.ttl_method} ===")
        lines.append(f"requests_total: {summary.total_requests}")
        lines.append(f"hit_rate: {summary.hit_rate:.4f}")
        lines.append(f"hit_rate_improvement_pct_vs_lru: {(summary.hit_rate - lru_summary.hit_rate) / summary.hit_rate * 100.0 if summary.hit_rate else 0.0:.2f}")
        lines.append(f"hit_count: {summary.hit_count}")
        lines.append(f"miss_count: {summary.miss_count}")
        lines.append(f"provider_total_calls: {summary.provider_calls}")
        lines.append(f"provider_refresh_calls: {summary.provider_refresh_calls}")
        lines.append(f"provider_prefetch_calls: {summary.provider_prefetch_calls}")
        lines.append(f"provider_miss_calls: {summary.provider_miss_calls}")
        lines.append(f"api_call_reduction_pct_vs_lru: {summary.api_call_reduction_pct_vs_lru:.2f}")
        lines.append(f"stale_rate_served_pct: {summary.stale_rate_served_pct:.4f}")
        lines.append(f"prewarm_precision: {summary.prewarm_precision:.4f}")
        lines.append(f"eviction_count: {summary.eviction_count}")
        lines.append(f"eviction_accuracy: {summary.eviction_accuracy:.4f}")
        lines.append(f"wrong_eviction_count: {summary.wrong_eviction_count}")
        lines.append(f"avg_query_count_after_eviction: {summary.avg_query_count_after_eviction:.4f}")
        lines.append("ttl_lookup_by_bucket:")
        lookup = ttl_lookups.get(summary.ttl_method, {})
        for bucket in LEAD_TIME_LABELS:
            ttl = lookup.get(bucket)
            lines.append(f"  {bucket}: {ttl if ttl is not None else 'NA'}")
        lines.append("distribution:")
        lines.extend([f"  {x}" for x in _format_distribution_lines(method_lambda_dist.get(summary.ttl_method, []), "lambda_i")])
        lines.extend([f"  {x}" for x in _format_distribution_lines(summary.admission_scores, "admission_score")])
        lines.extend([f"  {x}" for x in _format_distribution_lines(method_ttl_dist.get(summary.ttl_method, []), "ttl_seconds")])
        lines.append("")

    output_file = ROOT_DIR / output_path
    output_file.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved ttl-method eval to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTL Method Evaluation with configurable cache parameters")
    parser.add_argument("--start-date", default=None, help="Optional start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Optional end date (YYYY-MM-DD)")
    parser.add_argument("--max-requests", type=int, default=10000000, help="Max requests to sample (default: 10000000)")
    parser.add_argument("--output-path", default="workflow_ttl_methods_eval_2026-02-07_all.txt", help="Output file path")
    parser.add_argument("--controlled-capacity", type=int, default=100, help="Controlled cache capacity (default: 100)")
    parser.add_argument("--uncontrolled-capacity", type=int, default=900, help="Uncontrolled cache capacity (default: 900)")
    parser.add_argument("--lru-capacity", type=int, default=1000, help="LRU baseline capacity (default: 1000)")
    parser.add_argument("--score-percentile", type=float, default=0.7, help="Score percentile (default: 0.7)")
    parser.add_argument("--prefetch-ratio", type=float, default=0.2, help="Prefetch ratio (default: 0.2)")
   
    args = parser.parse_args()
    
    run_ttl_method_eval(
        start_date=args.start_date,
        end_date=args.end_date,
        max_requests=args.max_requests,
        output_path=args.output_path,
        controlled_capacity=args.controlled_capacity,
        uncontrolled_capacity=args.uncontrolled_capacity,
        lru_capacity=args.lru_capacity,
        score_percentile=args.score_percentile,
        prefetch_ratio=args.prefetch_ratio,
    )
