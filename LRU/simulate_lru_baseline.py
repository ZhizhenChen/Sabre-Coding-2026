from __future__ import annotations

from collections import OrderedDict
from contextlib import redirect_stdout
from dataclasses import dataclass
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd

from Cache_System_Workflow.sabre_cache_workflow_v2 import RequestContext
from data_processing.pipeline import DataPipelineProcessor


class DemoProvider:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: RequestContext) -> Dict[str, Any]:
        self.calls += 1
        base_price = 120 + (self.calls * 7)
        return {
            "request_key": request.cache_key(),
            "provider_call": self.calls,
            "offers": [
                {"hotel_id": "H100", "price": float(base_price)},
                {"hotel_id": "H210", "price": float(base_price + 25)},
            ],
        }


class _Tee:
    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@dataclass
class LRUResult:
    key: str
    source: str
    payload: Dict[str, Any]
    cached: bool


class VanillaLRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, request: RequestContext, provider: DemoProvider) -> LRUResult:
        key = request.cache_key()
        if key in self._store:
            self.hits += 1
            payload = self._store[key]
            self._store.move_to_end(key)
            return LRUResult(
                key=key,
                source="lru_hit",
                payload=payload,
                cached=True,
            )

        self.misses += 1
        payload = provider(request)

        if self.capacity > 0 and len(self._store) >= self.capacity:
            self._store.popitem(last=False)
            self.evictions += 1

        if self.capacity > 0:
            self._store[key] = payload
            self._store.move_to_end(key)

        return LRUResult(
            key=key,
            source="lru_miss_admit",
            payload=payload,
            cached=True,
        )

    def snapshot(self) -> List[str]:
        lines: List[str] = []
        for idx, key in enumerate(self._store.keys(), start=1):
            lines.append(f"{idx}. key={key}")
        return lines

    def size(self) -> int:
        return len(self._store)


def _load_requests_from_partition(
    partition_date: str = "2026-02-07",
    partition_root: str = "data/cleaned_partitioned",
    max_requests: int | None = None,
) -> List[RequestContext]:
    processor = DataPipelineProcessor(data_root=partition_root)
    source_df, _ = processor.process(partition_date=partition_date, max_rows=max_requests)
    if source_df.empty:
        return []

    request_df = source_df.copy()
    # Keep request-level rows when explode-based features exist.
    if "rq_correlation_id" in request_df.columns:
        request_df = request_df.drop_duplicates(subset=["rq_correlation_id"])

    request_df = request_df.dropna(
        subset=["rq_timestamp", "rq_stay_start_date", "rq_stay_end_date", "chain_code", "location_city_code", "cache_key"]
    )
    request_df = request_df.sort_values("rq_timestamp", kind="stable").reset_index(drop=True)

    requests: List[RequestContext] = []
    for row in request_df.itertuples(index=False):
        stay_start_date = pd.Timestamp(row.rq_stay_start_date).strftime("%Y-%m-%d")
        stay_end_date = pd.Timestamp(row.rq_stay_end_date).strftime("%Y-%m-%d")
        duration = max(0, (pd.Timestamp(row.rq_stay_end_date) - pd.Timestamp(row.rq_stay_start_date)).days)
        requests.append(
            RequestContext(
                rq_timestamp=row.rq_timestamp.to_pydatetime(),
                chain_code=str(row.chain_code),
                stay_start_date=stay_start_date,
                stay_end_date=stay_end_date,
                duration=duration,
                city_code=str(row.location_city_code),
                cache_key_value=str(row.cache_key),
            )
        )

    return requests


def run_simulation(
    output_path: str = "simulation_output_lru_2026-02-07.txt",
    partition_date: str = "2026-02-07",
    max_requests: int | None = None,
) -> None:
    provider = DemoProvider()
    lru = VanillaLRUCache(capacity=1000)

    requests = _load_requests_from_partition(partition_date=partition_date, max_requests=max_requests)

    unique_keys = len({request.cache_key() for request in requests})
    total_requests = len(requests)

    with open(output_path, "w", encoding="utf-8") as output_file:
        with redirect_stdout(_Tee(sys.stdout, output_file)):
            print(f"=== LRU Simulation for cleaned_partitioned/date={partition_date} ===")
            print(f"Loaded requests: {total_requests}")
            print(f"Unique cache keys: {unique_keys}")
            print(f"Cache capacity: {lru.capacity}")

            request_details: List[Dict[str, Any]] = []
            for idx, request in enumerate(requests, start=1):
                result = lru.get(request, provider)
                if idx <= 20:
                    request_details.append(
                        {
                            "request_num": idx,
                            "timestamp": request.rq_timestamp.isoformat(),
                            "key": result.key,
                            "source": result.source,
                            "cache_len": lru.size(),
                        }
                    )
                if idx % 10000 == 0:
                    print(f"Processed {idx}/{total_requests} requests...")

            hit_rate = (lru.hits / total_requests) if total_requests else 0.0
            miss_rate = (lru.misses / total_requests) if total_requests else 0.0
            api_calls_saved = lru.hits
            reduction_pct = (api_calls_saved / total_requests * 100.0) if total_requests else 0.0
            unique_ratio = (unique_keys / total_requests) if total_requests else 0.0
            repeat_request_rate = 1.0 - unique_ratio

            print("\n=== Final Stats ===")
            final_stats: Dict[str, Any] = {
                "cache_size": lru.size(),
                "cache_capacity": lru.capacity,
                "requests_total": total_requests,
                "unique_keys": unique_keys,
                "hit_count": lru.hits,
                "miss_count": lru.misses,
                "eviction_count": lru.evictions,
                "hit_rate": round(hit_rate, 4),
                "miss_rate": round(miss_rate, 4),
                "api_calls_saved": api_calls_saved,
                "api_call_reduction_pct": round(reduction_pct, 2),
                "repeat_request_rate": round(repeat_request_rate, 4),
            }
            for k, v in final_stats.items():
                print(f"{k}: {v}")

            print("\n=== Sample Request Details ===")
            for detail in request_details:
                print(
                    f"Request {detail['request_num']} @ {detail['timestamp']}"
                    f" | key={detail['key']}"
                    f" | source={detail['source']}"
                    f" | cache_len={detail['cache_len']}"
                )


            print(f"\nprovider_total_calls: {provider.calls}")

    print(f"Saved simulation output to {output_path}")


if __name__ == "__main__":
    run_simulation()


#  "from simulate_lru_baseline import run_simulation; run_simulation(output_path='simulation_output_lru_2026-02-07_3000.txt', partition_date='2026-02-07', max_requests=3000)"