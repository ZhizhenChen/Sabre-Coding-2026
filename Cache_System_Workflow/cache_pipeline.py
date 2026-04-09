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


def _resolve_case_dir(preferred: str, fallback: str) -> Path:
    p = ROOT_DIR / preferred
    if p.exists():
        return p
    return ROOT_DIR / fallback


LAMBDA_DIR = _resolve_case_dir("Lambda", "lambda")
LRU_DIR = ROOT_DIR / "LRU"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(LAMBDA_DIR) not in sys.path:
    sys.path.insert(0, str(LAMBDA_DIR))
if str(LRU_DIR) not in sys.path:
    sys.path.insert(0, str(LRU_DIR))

from Cache_System_Workflow.sabre_cache_workflow_v2 import PreparedWorkflowInput, RequestContext, SabreCacheWorkflow
from demand_forecasting.model_input import DemandScoreGenerator
from LRU.simulate_lru_baseline import VanillaLRUCache
from data_processing.pipeline import DataPipelineProcessor

lambda_model = importlib.import_module("lambda_model")
build_lambda_table = lambda_model.build_lambda_table
_km_survival = lambda_model._km_survival
_median_from_survival = lambda_model._median_from_survival

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
    hit_rate: float
    stale_response_count: int
    stale_rate_served_pct: float
    prewarm_precision: float
    useful_prewarm: int
    total_prewarm: int
    eviction_count: int
    eviction_accuracy: float
    wrong_eviction_count: int
    avg_query_after_eviction: float
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
    avg_query_after_eviction: float


class TruthPriceProvider:
    def __init__(self, truth_price_by_key: Dict[str, List[Dict[str, Any]]]) -> None:
        self.calls = 0
        self.truth_price_by_key = truth_price_by_key

    def __call__(self, request: RequestContext) -> Dict[str, Any]:
        self.calls += 1
        key = request.cache_key()
        offers = self.truth_price_by_key.get(key, [])
        return {
            "request_key": key,
            "offers": offers,
        }


def _build_truth_price_lookup(source_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    if "price" not in source_df.columns:
        raise ValueError("source_df must include price_per_day before building truth prices")
    if "rate_source" not in source_df.columns:
        raise ValueError("source_df must include rate_source before building truth prices")

    lookup: Dict[str, List[Dict[str, Any]]] = {}
    for cache_key, group in source_df.dropna(subset=["cache_key", "price", "rate_source"]).groupby("cache_key", dropna=False):
        # For each cache_key, collect all unique source-price pairs (take minimum price per source)
        offers = []
        for source, source_group in group.groupby("rate_source", dropna=False):
            min_price = float(source_group["price"].min())
            offers.append({
                "source": str(source),
                "price": min_price,
            })
        lookup[str(cache_key)] = offers
    return lookup


def _extract_price(payload: Any) -> float | None:
    try:
        offers = payload.get("offers", [])
        if not offers:
            return None
        return float(offers[0].get("price"))
    except Exception:
        return None


def _workflow_cache_keys(workflow: SabreCacheWorkflow) -> set[str]:
    with workflow._lock:
        return set(workflow._controlled.keys()) | set(workflow._uncontrolled.keys())


def _assign_lead_time_bucket(lead_time_days: int) -> str:
    for boundary, label in zip(LEAD_TIME_BREAKS, LEAD_TIME_LABELS[:-1]):
        if lead_time_days <= boundary:
            return label
    return LEAD_TIME_LABELS[-1]


def _bucketize_lead_time(series: pd.Series) -> pd.Series:
    return series.fillna(0).astype(int).apply(_assign_lead_time_bucket)


def _ttl_from_lambda(lambda_val: float) -> int:
    safe_lambda = max(float(lambda_val), 1e-6)
    ttl = -math.log(TTL_TARGET_FRESHNESS) / safe_lambda
    return int(max(MIN_TTL_SECONDS, min(MAX_TTL_SECONDS, ttl * 3600.0)))


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

    return ttl_lookup


def _build_lambda_based_ttl_lookup(source_df: pd.DataFrame, method: str) -> Dict[str, int]:
    table = build_lambda_table(source_df, min_intervals=1, method=method)
    if table.empty:
        return {}

    frame = table.copy()
    frame["lead_time_bucket"] = _bucketize_lead_time(frame["lead_time"])

    weighted_lambda = (
        frame.groupby("lead_time_bucket", dropna=False)
        .apply(
            lambda g: float(np.average(g["lambda_final"], weights=g["exposure_hours"]))
            if float(g["exposure_hours"].sum()) > 0
            else float(g["lambda_final"].mean())
        )
        .to_dict()
    )

    total_exposure = float(frame["exposure_hours"].sum())
    global_lambda = (
        float(np.average(frame["lambda_final"], weights=frame["exposure_hours"]))
        if total_exposure > 0
        else float(frame["lambda_final"].mean())
    )
    if not np.isfinite(global_lambda) or global_lambda <= 0:
        global_lambda = 0.1

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


def _prepare_requests(requests_df: pd.DataFrame, p_reuse_df: pd.DataFrame) -> List[PreparedWorkflowInput]:
    p_lookup = (
        p_reuse_df.dropna(subset=["cache_key", "p_reuse"])
        .groupby("cache_key", as_index=False)["p_reuse"]
        .mean()
    )
    merged = requests_df.merge(p_lookup, on=["cache_key"], how="left")
    merged["rq_timestamp"] = pd.to_datetime(merged["rq_timestamp"], errors="coerce", utc=True)
    merged = merged.dropna(subset=["rq_timestamp"]).sort_values("rq_timestamp", kind="stable").reset_index(drop=True)
    merged["p_reuse"] = merged["p_reuse"].fillna(0.5)

    prepared: List[PreparedWorkflowInput] = []
    for row in merged.itertuples(index=False):
        prepared.append(
            PreparedWorkflowInput(
                request=RequestContext(
                    rq_timestamp=row.rq_timestamp.to_pydatetime() if hasattr(row.rq_timestamp, "to_pydatetime") else row.rq_timestamp,
                    chain_code=str(row.chain_code),
                    stay_start_date=str(pd.Timestamp(row.rq_stay_start_date).strftime("%Y-%m-%d")),
                    stay_end_date=str(pd.Timestamp(row.rq_stay_end_date).strftime("%Y-%m-%d")),
                    duration=int(row.duration),
                    city_code=str(row.location_city_code),
                    lead_time_days=int(row.lead_time),
                    cache_key_value=str(row.cache_key),
                ),
                p_reuse=float(row.p_reuse),
                lambda_i=1.0,  # admission score uses p_reuse only
            )
        )
    return prepared


def _run_ttl_method(
    ttl_method: str,
    requests_df: pd.DataFrame,
    prepared_requests: List[PreparedWorkflowInput],
    ttl_lookup_by_bucket: Dict[str, int],
    truth_price_by_key: Dict[str, List[Dict[str, Any]]],
    lru_provider_calls: int,
    controlled_capacity: int = 100,
    uncontrolled_capacity: int = 900,
    score_percentile: float = 0.7,
    prefetch_ratio: float = 0.2,
    enable_background_refresh: bool = False,
) -> EvalSummary:
    provider = TruthPriceProvider(truth_price_by_key)
    workflow = SabreCacheWorkflow(
        controlled_capacity=controlled_capacity,
        uncontrolled_capacity=uncontrolled_capacity,
        score_percentile=score_percentile,
        prefetch_ratio=prefetch_ratio,
        enable_background_refresh=enable_background_refresh,
        lead_time_breaks=LEAD_TIME_BREAKS,
        lead_time_labels=LEAD_TIME_LABELS,
        km_ttl_lookup_by_bucket=ttl_lookup_by_bucket,
    )

    # Causal prewarming only: use candidates observed up to current replay time.
    # This avoids leaking future requests into prewarm decisions.
    prefetched_keys: set[str] = set()
    seen_candidates: Dict[str, Tuple[RequestContext, float, float]] = {}
    prefetch_every = max(25, int(max(1, controlled_capacity) * max(0.01, prefetch_ratio)))

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
        before_keys = _workflow_cache_keys(workflow)
        result = workflow.get(request, p_reuse=item.p_reuse, lambda_i=item.lambda_i, provider=provider)
        after_keys = _workflow_cache_keys(workflow)

        evicted_keys = before_keys - after_keys
        for evicted_key in evicted_keys:
            eviction_events.append((evicted_key, idx))

        if "hit" in result.source:
            hit_count += 1
            hit_keys.add(key)
            served_price = _extract_price(result.payload)
            truth_offers = truth_price_by_key.get(key, [])
            truth_price = float(truth_offers[0].get("price", 0.0)) if truth_offers else 0.0
            if served_price is not None and abs(served_price - truth_price) / max(served_price, 1e-6) > 0.01:
                stale_response_count += 1
        else:
            miss_count += 1

        prev = seen_candidates.get(key)
        if prev is None or float(item.p_reuse) > float(prev[1]):
            seen_candidates[key] = (item.request, float(item.p_reuse), float(item.lambda_i))

        if idx % prefetch_every == 0 and seen_candidates:
            admitted = workflow.prefetch_controlled(list(seen_candidates.values()), provider)
            prefetched_keys.update(admitted)

    total_requests = len(prepared_requests)
    unique_keys = len({item.request.cache_key() for item in prepared_requests})
    hit_rate = hit_count / total_requests if total_requests else 0.0

    useful_prewarm = len(prefetched_keys & hit_keys)
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

    eviction_accuracy = queried_after_eviction / evictions if evictions else 0.0
    avg_query_after_eviction = mean(later_query_counts) if later_query_counts else 0.0
    stale_rate_served_pct = stale_response_count / hit_count if hit_count else 0.0

    api_call_reduction = 1.0 - (provider.calls / lru_provider_calls) if lru_provider_calls else 0.0

    workflow.close()
    return EvalSummary(
        ttl_method=ttl_method,
        total_requests=total_requests,
        unique_keys=unique_keys,
        hit_count=hit_count,
        miss_count=miss_count,
        provider_calls=provider.calls,
        hit_rate=hit_rate,
        stale_response_count=stale_response_count,
        stale_rate_served_pct=stale_rate_served_pct,
        prewarm_precision=prewarm_precision,
        useful_prewarm=useful_prewarm,
        total_prewarm=total_prewarm,
        eviction_count=evictions,
        eviction_accuracy=eviction_accuracy,
        wrong_eviction_count=wrong_eviction_count,
        avg_query_after_eviction=avg_query_after_eviction,
        api_call_reduction_pct_vs_lru=api_call_reduction * 100.0,
    )


def _run_lru_baseline(requests_df: pd.DataFrame) -> LRUSummary:
    truth_price_by_key = _build_truth_price_lookup(requests_df)
    provider = TruthPriceProvider(truth_price_by_key)
    lru = VanillaLRUCache(capacity=1000)

    hit_count = 0
    miss_count = 0
    eviction_events: List[Tuple[str, int]] = []
    requests_by_key: Dict[str, List[int]] = defaultdict(list)

    for idx, row in enumerate(requests_df.itertuples(index=False), start=1):
        request = RequestContext(
            rq_timestamp=row.rq_timestamp.to_pydatetime() if hasattr(row.rq_timestamp, "to_pydatetime") else row.rq_timestamp,
            chain_code=str(row.chain_code),
            stay_start_date=str(pd.Timestamp(row.rq_stay_start_date).strftime("%Y-%m-%d")),
            stay_end_date=str(pd.Timestamp(row.rq_stay_end_date).strftime("%Y-%m-%d")),
            duration=int(row.duration),
            city_code=str(row.location_city_code),
            cache_key_value=str(row.cache_key),
        )
        key = request.cache_key()
        requests_by_key[key].append(idx)
        before_keys = set(lru._store.keys())
        result = lru.get(request, provider)
        after_keys = set(lru._store.keys())

        if result.source == "lru_hit":
            hit_count += 1
        else:
            miss_count += 1

        evicted_keys = before_keys - after_keys
        for evicted_key in evicted_keys:
            eviction_events.append((evicted_key, idx))

    total_requests = len(requests_df)
    unique_keys = int(requests_df["cache_key"].nunique())
    hit_rate = hit_count / total_requests if total_requests else 0.0
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
        hit_count=hit_count,
        miss_count=miss_count,
        provider_calls=provider.calls,
        hit_rate=hit_rate,
        eviction_count=eviction_count,
        eviction_accuracy=queried_after_eviction / eviction_count if eviction_count else 0.0,
        wrong_eviction_count=wrong_eviction_count,
        avg_query_after_eviction=mean(later_query_counts) if later_query_counts else 0.0,
    )


def run_ttl_method_eval(
    partition_date: str = "2026-02-07",
    max_requests: int = 3000,
    output_path: str = "workflow_ttl_methods_eval_2026-02-07_3000.txt",
    controlled_capacity: int = 100,
    uncontrolled_capacity: int = 900,
    score_percentile: float = 0.7,
    prefetch_ratio: float = 0.2,
    enable_background_refresh: bool = False,
) -> None:
    processor = DataPipelineProcessor(data_root=str(ROOT_DIR / "data" / "cleaned_partitioned"))
    source_df, prepared_df = processor.process(partition_date=partition_date, max_rows=max_requests)

    if source_df.empty:
        raise ValueError(f"No requests found for {partition_date} after sampling {max_requests} rows.")

    model_generator = DemandScoreGenerator(model_path=str(ROOT_DIR / "demand_forecasting" / "xgb_model.json"))
    p_reuse_df = model_generator.generate_demand_scores(
        processed_df=prepared_df,
        source_df=source_df,
        num_samples=None,
        output_parquet=None,
    )

    truth_price_by_key = _build_truth_price_lookup(source_df)
    prepared_requests = _prepare_requests(source_df, p_reuse_df)
    lru_summary = _run_lru_baseline(source_df)

    ttl_methods = ["glm", "pp", "rule_based", "km"]
    evals: List[EvalSummary] = []
    ttl_lookups: Dict[str, Dict[str, int]] = {}

    for method in ttl_methods:
        ttl_lookup = _build_ttl_lookup(source_df, ttl_method=method)
        ttl_lookups[method] = ttl_lookup
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
                enable_background_refresh=enable_background_refresh,
            )
        )

    lines: List[str] = []
    lines.append("TTL METHOD EVAL (admission score uses p_reuse only)")
    lines.append(f"partition_date={partition_date}")
    lines.append(f"sample_requests={len(source_df)}")
    lines.append(f"unique_request_keys={source_df['cache_key'].nunique()}")
    lines.append(f"prepared_feature_rows={len(prepared_df)}")
    lines.append(f"p_reuse_rows={len(p_reuse_df)}")
    lines.append("")

    lines.append("=== LRU Baseline ===")
    lines.append(f"requests_total: {lru_summary.total_requests}")
    lines.append(f"hit_rate: {lru_summary.hit_rate:.4f}")
    lines.append(f"provider_total_calls: {lru_summary.provider_calls}")
    lines.append("")

    for summary in evals:
        lines.append(f"=== TTL Method: {summary.ttl_method} ===")
        lines.append(f"requests_total: {summary.total_requests}")
        lines.append(f"hit_rate: {summary.hit_rate:.4f}")
        lines.append(f"provider_total_calls: {summary.provider_calls}")
        lines.append(f"api_call_reduction_pct_vs_lru: {summary.api_call_reduction_pct_vs_lru:.2f}")
        lines.append(f"stale_rate_served_pct: {summary.stale_rate_served_pct:.4f}")
        lines.append(f"prewarm_precision: {summary.prewarm_precision:.4f}")
        lines.append(f"eviction_count: {summary.eviction_count}")
        lines.append(f"eviction_accuracy: {summary.eviction_accuracy:.4f}")
        lines.append(f"avg_query_after_eviction: {summary.avg_query_after_eviction:.4f}")
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
    parser.add_argument("--partition-date", default="2026-02-07", help="Partition date (default: 2026-02-07)")
    parser.add_argument("--max-requests", type=int, default=10000000, help="Max requests to sample (default: 10000000)")
    parser.add_argument("--output-path", default="workflow_ttl_methods_eval_2026-02-07_all.txt", help="Output file path")
    parser.add_argument("--controlled-capacity", type=int, default=100, help="Controlled cache capacity (default: 100)")
    parser.add_argument("--uncontrolled-capacity", type=int, default=900, help="Uncontrolled cache capacity (default: 900)")
    parser.add_argument("--score-percentile", type=float, default=0.7, help="Score percentile (default: 0.7)")
    parser.add_argument("--prefetch-ratio", type=float, default=0.2, help="Prefetch ratio (default: 0.2)")
    parser.add_argument("--enable-background-refresh", action="store_true", help="Enable background refresh")
    
    args = parser.parse_args()
    
    run_ttl_method_eval(
        partition_date=args.partition_date,
        max_requests=args.max_requests,
        output_path=args.output_path,
        controlled_capacity=args.controlled_capacity,
        uncontrolled_capacity=args.uncontrolled_capacity,
        score_percentile=args.score_percentile,
        prefetch_ratio=args.prefetch_ratio,
        enable_background_refresh=args.enable_background_refresh,
    )
