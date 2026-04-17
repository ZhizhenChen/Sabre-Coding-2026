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
from midas.midas_score import build_midas_scores

lambda_model = importlib.import_module("lambda_model")
build_lambda_table = lambda_model.build_lambda_table
_km_survival = lambda_model._km_survival
_median_from_survival = lambda_model._median_from_survival
_weighted_global_lambda = lambda_model.weighted_global_lambda
_bucket_weighted_lambda = lambda_model.bucket_weighted_lambda
_ttl_from_lambda_model = lambda_model.ttl_from_lambda

LEAD_TIME_BREAKS = [1, 3, 7, 30]
LEAD_TIME_LABELS = ["same_day", "short", "mid", "long", "very_long"]
TTL_TARGET_FRESHNESS = 0.8
MIN_TTL_SECONDS = 60
MAX_TTL_SECONDS = 24 * 3600


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


@dataclass
class LRUSummary:
    total_requests: int
    unique_keys: int
    hit_count: int
    miss_count: int
    provider_calls: int
    hit_rate: float
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
            if isinstance(rate_infos, np.ndarray):
                rate_infos = rate_infos.tolist()
            elif isinstance(rate_infos, tuple):
                rate_infos = list(rate_infos)

            if rate_infos is not None and isinstance(rate_infos, list):
                for offer in rate_infos:
                    if isinstance(offer, dict):
                        price = offer.get("amount_after_tax")
                        source = offer.get("rate_source", "unknown")
                        if price is not None:
                            offers.append({
                                "source": str(source),
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
        Minimum price from the matched observation's offers, or None if not found
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
            return float(offers[0].get("price", 0.0))
    return None


def _assign_lead_time_bucket(lead_time_days: int) -> str:
    for boundary, label in zip(LEAD_TIME_BREAKS, LEAD_TIME_LABELS[:-1]):
        if lead_time_days <= boundary:
            return label
    return LEAD_TIME_LABELS[-1]


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


def _admission_lambda_for_request(
    request: RequestContext,
    ttl_lookup_by_bucket: Dict[str, int],
) -> float:
    lead_time_days = int(request.lead_time_days) if request.lead_time_days is not None else max(0, request.duration)
    bucket = _assign_lead_time_bucket(lead_time_days)
    ttl_seconds = ttl_lookup_by_bucket.get(bucket)
    if ttl_seconds is None:
        return 1.0
    return _lambda_from_ttl_seconds(ttl_seconds)


def _build_rule_based_ttl_lookup() -> Dict[str, int]:
    # A simple deterministic baseline policy.
    return {
        "same_day": 1 * 3600,
        "short": 2 * 3600,
        "mid": 4 * 3600,
        "long": 8 * 3600,
        "very_long": 12 * 3600,
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
            fallback_table["lead_time_bucket"] = _bucketize_lead_time(fallback_table["lead_time"])
            global_lambda = _weighted_global_lambda(
                fallback_table,
                lambda_col="lambda_final",
                weight_col="exposure_hours",
                default_lambda=0.1,
            )
            fallback_ttl = _ttl_from_lambda(global_lambda)
            for bucket in LEAD_TIME_LABELS:
                ttl_lookup.setdefault(bucket, fallback_ttl)

    return ttl_lookup


def _build_lambda_based_ttl_lookup(source_df: pd.DataFrame, method: str) -> Dict[str, int]:
    table = build_lambda_table(source_df, min_intervals=1, method=method)
    if table.empty:
        table = build_lambda_table(source_df, min_intervals=1, method="fallback")
    if table.empty:
        return {}

    frame = table.copy()
    frame["lead_time_bucket"] = _bucketize_lead_time(frame["lead_time"])

    weighted_lambda = _bucket_weighted_lambda(
        frame,
        bucket_col="lead_time_bucket",
        lambda_col="lambda_final",
        weight_col="exposure_hours",
    )

    global_lambda = _weighted_global_lambda(
        frame,
        lambda_col="lambda_final",
        weight_col="exposure_hours",
        default_lambda=0.1,
    )

    lookup: Dict[str, int] = {}
    for bucket in LEAD_TIME_LABELS:
        lam = float(weighted_lambda.get(bucket, global_lambda))
        lookup[bucket] = _ttl_from_lambda(lam)

    return lookup


def _build_ttl_lookup(source_df: pd.DataFrame, ttl_method: str) -> Dict[str, int]:
    method = ttl_method.strip().lower()
    if method == "rule_based":
        return _build_rule_based_ttl_lookup()
    if method == "km":
        return _build_km_ttl_lookup(source_df)
    if method == "pp":
        return _build_lambda_based_ttl_lookup(source_df, method="poisson")
    if method == "glm":
        return _build_lambda_based_ttl_lookup(source_df, method="glm")
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
    score_df: pd.DataFrame,
    ttl_lookup_by_bucket: Dict[str, int],
    score_col: str = "p_reuse",
) -> List[PreparedWorkflowInput]:
    if score_col not in score_df.columns:
        raise ValueError(f"score_df must include score column '{score_col}'.")

    p_lookup = (
        score_df.dropna(subset=["cache_key", score_col])
        .groupby("cache_key", as_index=False)[score_col]
        .mean()
    )
    p_lookup = p_lookup.rename(columns={score_col: "admission_score"})
    merged = requests_df.merge(p_lookup, on=["cache_key"], how="left")
    merged["admission_score"] = merged["admission_score"].fillna(0.5)

    prepared: List[PreparedWorkflowInput] = []
    for row in merged.itertuples(index=False):
        request = _request_context_from_row(row)
        # Calculate lambda_i from TTL lookup based on lead time
        lambda_i = _admission_lambda_for_request(request, ttl_lookup_by_bucket)
        prepared.append(
            PreparedWorkflowInput(
                request=request,
                p_reuse=float(row.admission_score),
                lambda_i=lambda_i,
            )
        )
    return prepared


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
            _admission_lambda_for_request(item.request, ttl_lookup_by_bucket),
        )
        for item in prepared_requests
    ]
    prefetched_keys = workflow.prefetch_controlled(candidate_tuples, provider)

    hit_count = 0
    miss_count = 0
    stale_response_count = 0
    eviction_events: List[Tuple[str, int]] = []
    requests_by_key: Dict[str, List[int]] = defaultdict(list)
    hit_keys: set[str] = set()

    for idx, item in enumerate(prepared_requests, start=1):
        request = item.request
        key = request.cache_key()
        requests_by_key[key].append(idx)
        admission_lambda = _admission_lambda_for_request(request, ttl_lookup_by_bucket)
        result = workflow.get(request, p_reuse=item.p_reuse, lambda_i=admission_lambda, provider=provider)
        if result.evicted_key is not None:
            eviction_events.append((result.evicted_key, idx))

        if "hit" in result.source:
            hit_count += 1
            hit_keys.add(key)
            served_price = _extract_price(result.payload)
            truth_price = _get_truth_price_for_request(key, request.rq_timestamp, truth_price_by_key)
            if served_price is not None and truth_price is not None and abs(served_price - truth_price) / max(served_price, 1e-6) > 0.01:
                stale_response_count += 1
        else:
            miss_count += 1

    total_requests = len(prepared_requests)
    unique_keys = len({item.request.cache_key() for item in prepared_requests})
    hit_rate = hit_count / total_requests if total_requests else 0.0

    useful_prewarm = len(set(prefetched_keys) & hit_keys)
    total_prewarm = len(prefetched_keys)
    prewarm_precision = useful_prewarm / total_prewarm if total_prewarm else 0.0

    evictions = len(eviction_events)
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

    eviction_accuracy = 1- queried_after_eviction / evictions if evictions else 0.0
    avg_query_count_after_eviction = mean(later_query_counts) if later_query_counts else 0.0
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

    for idx, row in enumerate(requests_df.itertuples(index=False), start=1):
        request = _request_context_from_row(row)
        key = request.cache_key()
        requests_by_key[key].append(idx)
        result = lru.get(request, provider)
        if result.evicted_key is not None:
            eviction_events.append((result.evicted_key, idx))

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
        eviction_count=eviction_count,
        eviction_accuracy=1- queried_after_eviction / eviction_count if eviction_count else 0.0,
        wrong_eviction_count=wrong_eviction_count,
        avg_query_count_after_eviction=mean(later_query_counts) if later_query_counts else 0.0,
    )


def _build_admission_scores(
    source_df: pd.DataFrame,
    p_reuse_df: pd.DataFrame,
    admission_source: str,
    *,
    w_demand: float,
    w_intent: float,
    eta: float,
    tau: float,
    alpha: float,
) -> tuple[pd.DataFrame, str]:
    source = admission_source.strip().lower()
    if source == "p_reuse":
        return p_reuse_df.copy(), "p_reuse"
    if source == "midas":
        midas_df = build_midas_scores(
            requests_df=source_df,
            p_reuse_df=p_reuse_df,
            w_demand=w_demand,
            w_intent=w_intent,
            eta=eta,
            tau=tau,
            alpha=alpha,
        )
        return midas_df, "midas_score"
    raise ValueError("admission_source must be one of: p_reuse, midas")


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
    admission_source: str = "midas",
    w_demand: float = 0.85,
    w_intent: float = 0.15,
    midas_eta: float = 0.35,
    midas_tau: float = 0.08,
    midas_alpha: float = 1.0,
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

    admission_scores_df, admission_score_col = _build_admission_scores(
        source_df=source_df,
        p_reuse_df=p_reuse_df,
        admission_source=admission_source,
        w_demand=w_demand,
        w_intent=w_intent,
        eta=midas_eta,
        tau=midas_tau,
        alpha=midas_alpha,
    )
    print(
        f"Built admission scores: source={admission_source}, "
        f"rows={len(admission_scores_df)}, score_col={admission_score_col}"
    )

    # Build time-ordered truth price lookup from unexploded source data
    truth_price_by_key = _build_truth_price_lookup(source_df)
    print(f"Built truth price lookup for {len(truth_price_by_key)} unique cache keys")

    lru_summary = _run_lru_baseline(source_df, truth_price_by_key=truth_price_by_key, lru_capacity=lru_capacity)
    print(f"Completed LRU baseline evaluation: {lru_summary}")


    ttl_methods = ["glm", "pp", "rule_based", "km"]
    evals: List[EvalSummary] = []
    ttl_lookups: Dict[str, Dict[str, int]] = {}
    lines: List[str] = []
    for method in ttl_methods:
        ttl_lookup = _build_ttl_lookup(pricing_source_df, ttl_method=method)
        ttl_lookups[method] = ttl_lookup
        # Prepare requests with lambda_i from this TTL method
        prepared_requests = _prepare_requests(
            source_df,
            admission_scores_df,
            ttl_lookup,
            score_col=admission_score_col,
        )
        print(f"Prepared requests for {method.upper()}: {len(prepared_requests)}")
        # lines.append(method.upper() + " TTL Lookup:")
        # for i in prepared_requests:
        #     lines.append(f"Prepared request: cache_key={i.request.cache_key()}, rq_timestamp = {i.request.rq_timestamp}, p_reuse={i.p_reuse:.4f}, lambda_i={i.lambda_i:.6f}")
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
        print(f"Completed TTL method evaluation for {method.upper()}: {evals[-1]}")
    
    
    # lines: List[str] = []
    lines.append("TTL METHOD EVAL (admission score + TTL-implied lambda)")

    lines.append(f"start_date={start_date}")
    lines.append(f"end_date={end_date}")

    lines.append(f"sample_requests={len(source_df)}")
    lines.append(f"unique_request_keys={source_df['cache_key'].nunique()}")
    lines.append(f"source_rows_after_explode={len(pricing_source_df)}")
    lines.append(f"prepared_feature_rows={len(prepared_df)}")
    lines.append(f"p_reuse_rows={len(p_reuse_df)}")
    lines.append(f"admission_source={admission_source}")
    lines.append(f"admission_score_col={admission_score_col}")
    lines.append(f"w_demand={w_demand}")
    lines.append(f"w_intent={w_intent}")
    lines.append(f"midas_eta={midas_eta}")
    lines.append(f"midas_tau={midas_tau}")
    lines.append(f"midas_alpha={midas_alpha}")
    lines.append("")

    lines.append("=== LRU Baseline ===")
    lines.append(f"requests_total: {lru_summary.total_requests}")
    lines.append(f"hit_rate: {lru_summary.hit_rate:.4f}")
    lines.append(f"hit_count: {lru_summary.hit_count}")
    lines.append(f"miss_count: {lru_summary.miss_count}")
    lines.append(f"provider_total_calls: {lru_summary.provider_calls}")
    lines.append("")

    for summary in evals:
        lines.append(f"=== TTL Method: {summary.ttl_method} ===")
        lines.append(f"requests_total: {summary.total_requests}")
        lines.append(f"hit_rate: {summary.hit_rate:.4f}")
        if lru_summary.hit_rate > 0:
            hit_rate_improvement = ((summary.hit_rate - lru_summary.hit_rate) / lru_summary.hit_rate) * 100.0
        else:
            hit_rate_improvement = 0.0
        lines.append(f"hit_rate_improvement_pct_vs_lru: {hit_rate_improvement:.2f}")
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
        lines.append(f"avg_query_count_after_eviction: {summary.avg_query_count_after_eviction:.4f}")
        lines.append("ttl_lookup_by_bucket:")
        lookup = ttl_lookups.get(summary.ttl_method, {})
        for bucket in LEAD_TIME_LABELS:
            ttl = lookup.get(bucket)
            lines.append(f"  {bucket}: {ttl if ttl is not None else 'NA'}")
        lines.append("")

    output_file = ROOT_DIR / output_path
    output_file.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved ttl-method eval to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTL Method Evaluation with configurable cache parameters")
    parser.add_argument("--start-date", default=None, help="Optional start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Optional end date (YYYY-MM-DD)")
    parser.add_argument("--max-requests", type=int, default=100000000, help="Max requests to sample (default: 100000000)")
    parser.add_argument("--output-path", default="workflow_ttl_methods_eval_2026-02-07_all.txt", help="Output file path")
    parser.add_argument("--controlled-capacity", type=int, default=20000, help="Controlled cache capacity (default: 20000)")
    parser.add_argument("--uncontrolled-capacity", type=int, default=180000, help="Uncontrolled cache capacity (default: 180000)")
    parser.add_argument("--lru-capacity", type=int, default=200000, help="LRU baseline capacity (default: 200000)")
    parser.add_argument("--score-percentile", type=float, default=0.7, help="Score percentile (default: 0.7)")
    parser.add_argument("--prefetch-ratio", type=float, default=0.2, help="Prefetch ratio (default: 0.2)")
    parser.add_argument("--admission-source", choices=["p_reuse", "midas"], default="midas", help="Admission score source")
    parser.add_argument("--w-demand", type=float, default=0.85, help="Demand weight for MIDAS blend")
    parser.add_argument("--w-intent", type=float, default=0.15, help="Intent weight for MIDAS blend")
    parser.add_argument("--midas-eta", type=float, default=0.35, help="Markov smoothing blend exponent")
    parser.add_argument("--midas-tau", type=float, default=0.08, help="Bound for MIDAS correction delta")
    parser.add_argument("--midas-alpha", type=float, default=1.0, help="Dirichlet smoothing for transition matrix")

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
        admission_source=args.admission_source,
        w_demand=args.w_demand,
        w_intent=args.w_intent,
        midas_eta=args.midas_eta,
        midas_tau=args.midas_tau,
        midas_alpha=args.midas_alpha,
    )
