from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import heapq
import math
import threading
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol, Tuple


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    p = _clamp_01(p)
    idx = int(math.ceil((len(sorted_values) - 1) * p))
    return sorted_values[idx]


def _parse_yyyy_mm_dd(raw: str) -> Optional[datetime]:
    try:
        return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _assign_bucket_label(value: int, breaks: List[int], labels: List[str]) -> str:
    for index, boundary in enumerate(breaks):
        if value <= boundary:
            return labels[index]
    return labels[-1] if labels else str(value)


@dataclass(frozen=True)
class LeadTimeTTLBucket:
    start_day: int
    end_day: int
    ttl_seconds: int


@dataclass(frozen=True)
class RequestContext:
    rq_timestamp: datetime
    chain_code: str
    stay_start_date: str
    stay_end_date: str
    duration: int
    city_code: str
    lead_time_days: Optional[int] = None
    cache_key_value: Optional[str] = None
    hotel_code: Optional[str] = None

    def cache_key(self) -> str:
        """
        Cache key format:
        <hotel_code::city_code::stay_start_date::stay_end_date>
        """
        if self.cache_key_value:
            return self.cache_key_value
        if self.hotel_code is None:
            raise ValueError("hotel_code is required when cache_key_value is not provided")
        return (
            f"{self.hotel_code}::"
            f"{self.city_code}::"
            f"{self.stay_start_date}::"
            f"{self.stay_end_date}"
        )

@dataclass
class TruthPriceProvider:
    def __init__(self, truth_price_by_key: Dict[str, Dict[str, List[Any]]]) -> None:
        self.calls = 0
        self.truth_price_by_key = truth_price_by_key

    def __call__(self, request: RequestContext) -> Dict[str, Any]:
        self.calls += 1
        key = request.cache_key()

        series = self.truth_price_by_key.get(key)
        offers: List[Dict[str, Any]] = []
        if series is not None:
            timestamps = series.get("timestamps", [])
            offers_by_timestamp = series.get("offers", [])
            request_ts = request.rq_timestamp
            if request_ts.tzinfo is None:
                request_ts = request_ts.replace(tzinfo=timezone.utc)

            closest_idx = bisect_right(timestamps, request_ts) - 1
            if 0 <= closest_idx < len(offers_by_timestamp):
                offers = offers_by_timestamp[closest_idx]

        return {
            "request_key": key,
            "offers": offers,
            "rate_group": (
                offers[0].get("rate_group")
                if offers and isinstance(offers[0], dict) and offers[0].get("rate_group") is not None
                else None
            ),
        }

@dataclass
class CacheEntry:
    request: RequestContext
    payload: Any
    expires_at: datetime
    score: float
    p_reuse: float
    lambda_i: float
    recent_freq: int
    last_access_time: datetime
    age_seconds: float


@dataclass
class WorkflowResult:
    key: str
    source: str
    payload: Any
    cached: bool
    tier: str
    admission_score: float
    theta: float
    evicted_key: str | None = None


@dataclass(frozen=True)
class PreparedWorkflowInput:
    request: RequestContext
    p_reuse: float
    lambda_i: float


class SabreCacheWorkflow:
    """
    Two-tier cache workflow for Sabre hotel requests.

    Inputs per request:
    - request context
    - p_reuse: probability produced by demand forecasting model
    - lambda_i: model output in [0, 1] representing relative update intensity

    Workflow behavior:
    - score = p_reuse / max(lambda_i, score_lambda_floor)
    - theta = top 80 percentile of recent scores
    - controlled cache: score-threshold admission + min-heap eviction by score
    - uncontrolled cache: LRU eviction
    - controlled/uncontrolled hits serve cached payload; refresh happens on expiry

    TTL behavior (ipynb-aligned):
    - freshness target: TTL = -ln(target_freshness) / lambda
    - optional lead_time bucket lookup overrides formula when configured
    - bucket lookup can be supplied either as label->TTL map or as explicit ranges
    - fallback to formula if no bucket match exists

    Behavior:
    - Cache hit (not expired): return cached payload
    - Cache miss/expired: fetch from provider, optionally cache using admission rule
    """

    def __init__(
        self,
        controlled_capacity: int = 500,
        uncontrolled_capacity: int = 5000,
        score_percentile: float = 0.8,
        score_history_size: int = 10000,
        score_lambda_floor: float = 0.01,
        ttl_lambda_floor: float = 0.05,
        min_ttl_seconds: int = 60,
        max_ttl_seconds: int = 24 * 3600,
        api_cost: float = 1.0,
        stale_penalty_l: float = 3.0,
        value_v: float = 1800.0,
        uncontrolled_base_t: float = 900.0,
        target_freshness_controlled: float = 0.80,
        target_freshness_uncontrolled: float = 0.80,
        lead_time_breaks: Optional[List[int]] = None,
        lead_time_labels: Optional[List[str]] = None,
        km_ttl_lookup_by_bucket: Optional[Dict[str, int]] = None,
        controlled_ttl_lookup_by_lead_time: Optional[List[Tuple[int, int, int]]] = None,
        uncontrolled_ttl_lookup_by_lead_time: Optional[List[Tuple[int, int, int]]] = None,
        staleness_threshold: float = 0.01,
        max_cache_size_mb: float = 60.0,
        avg_entry_size_bytes: int = 1000,
        min_cache_util_fraction: float = 0.20,
        prefetch_ratio: float = 0.01,
        now_fn: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.controlled_capacity = controlled_capacity
        self.uncontrolled_capacity = uncontrolled_capacity
        self.score_percentile = score_percentile
        self.score_lambda_floor = score_lambda_floor
        self.ttl_lambda_floor = ttl_lambda_floor
        self.min_ttl_seconds = min_ttl_seconds
        self.max_ttl_seconds = max_ttl_seconds
        self.api_cost = api_cost
        self.stale_penalty_l = stale_penalty_l
        self.value_v = value_v
        self.uncontrolled_base_t = uncontrolled_base_t
        self.target_freshness_controlled = _clamp_01(target_freshness_controlled)
        self.target_freshness_uncontrolled = _clamp_01(target_freshness_uncontrolled)
        self.lead_time_breaks = lead_time_breaks or []
        self.lead_time_labels = lead_time_labels or []
        self.km_ttl_lookup_by_bucket = dict(km_ttl_lookup_by_bucket or {})
        self.controlled_ttl_lookup_by_lead_time = controlled_ttl_lookup_by_lead_time or []
        self.uncontrolled_ttl_lookup_by_lead_time = uncontrolled_ttl_lookup_by_lead_time or []
        self.staleness_threshold = _clamp_01(staleness_threshold)
        self.max_cache_size_mb = float(max_cache_size_mb)
        self.avg_entry_size_bytes = int(avg_entry_size_bytes)
        self.min_cache_util_fraction = _clamp_01(min_cache_util_fraction)
        self.min_cache_util_mb = self.max_cache_size_mb * self.min_cache_util_fraction
        self.prefetch_ratio = prefetch_ratio
        self._now_fn = now_fn
        self._refresh_provider: Optional[TruthPriceProvider] = None
        self._lock = threading.RLock()
        self._controlled: OrderedDict[str, CacheEntry] = OrderedDict()
        self._controlled_min_heap: List[Tuple[float, str]] = []
        self._uncontrolled: OrderedDict[str, CacheEntry] = OrderedDict()
        self._score_history: Deque[float] = deque(maxlen=score_history_size)
        self._theta: float = 0.0
        self._provider_call_counts: Dict[str, int] = {"refresh": 0, "prefetch": 0, "miss": 0}
        self._last_stay_date_purge_date: Optional[str] = None


    def _is_stay_date_expired(self, entry: CacheEntry, now: datetime) -> bool:
        """Check if the stay start date has passed."""
        stay_start = _parse_yyyy_mm_dd(entry.request.stay_start_date)
        if stay_start is not None and now.date() > stay_start.date():
            return True
        return False

    def _is_ttl_expired(self, entry: CacheEntry, now: datetime) -> bool:
        """Check if the TTL has expired."""
        return entry.expires_at <= now

    def _purge_stay_date_expired_entries(self, now: datetime) -> None:
        """Purge entries with expired stay_start_date (once per day)."""
        today = now.date().isoformat()
        
        # Only run once per day
        if self._last_stay_date_purge_date == today:
            return
        
        with self._lock:
            # Purge controlled entries
            controlled_keys = [key for key, entry in self._controlled.items() 
                             if self._is_stay_date_expired(entry, now)]
            for key in controlled_keys:
                del self._controlled[key]
            
            # Purge uncontrolled entries
            uncontrolled_keys = [key for key, entry in self._uncontrolled.items() 
                               if self._is_stay_date_expired(entry, now)]
            for key in uncontrolled_keys:
                del self._uncontrolled[key]
        
        self._last_stay_date_purge_date = today

    def _compute_score(self, p_reuse: float, lambda_i: float) -> float:
        safe_lambda = max(float(lambda_i), self.score_lambda_floor)
        return p_reuse / safe_lambda

    def _lead_time_days(self, request: RequestContext, now: datetime) -> int:
        if request.lead_time_days is not None:
            return max(0, int(request.lead_time_days))
        stay_start = _parse_yyyy_mm_dd(request.stay_start_date)
        if stay_start is None:
            return max(0, request.duration)
        return max(0, (stay_start.date() - now.date()).days)

    def _lookup_ttl_by_lead_time(
        self,
        lead_time_days: int,
        buckets: List[Tuple[int, int, int]],
    ) -> Optional[int]:
        for start_day, end_day, ttl_seconds in buckets:
            if start_day <= lead_time_days <= end_day:
                return ttl_seconds
        return None

    def _lookup_km_ttl(self, lead_time_days: int) -> Optional[int]:
        if not self.km_ttl_lookup_by_bucket:
            return None
        if self.lead_time_breaks and self.lead_time_labels:
            bucket_label = _assign_bucket_label(lead_time_days, self.lead_time_breaks, self.lead_time_labels)
            return self.km_ttl_lookup_by_bucket.get(bucket_label)
        return self.km_ttl_lookup_by_bucket.get(str(lead_time_days))

    def _compute_ttl_from_freshness(self, lambda_i: float, target_freshness: float) -> int:
        safe_lambda = max(float(lambda_i), self.ttl_lambda_floor)
        safe_target = min(max(float(target_freshness), 1e-6), 1.0 - 1e-6)
        ttl_seconds = -math.log(safe_target) / safe_lambda
        return int(max(self.min_ttl_seconds, min(self.max_ttl_seconds, ttl_seconds)))

    def is_stale(self, cached_price: float, true_price: float) -> bool:
        if cached_price <= 0:
            return False
        return abs(cached_price - true_price) / cached_price > self.staleness_threshold

    def estimate_cache_size_mb(self, key_count: int) -> float:
        return float(key_count * self.avg_entry_size_bytes / 1e6)

    def cache_size_is_feasible(self, key_count: int) -> bool:
        estimated_cache_mb = self.estimate_cache_size_mb(key_count)
        return self.min_cache_util_mb <= estimated_cache_mb <= self.max_cache_size_mb

    @staticmethod
    def choose_fixed_ttl_candidate(
        candidates: List[Dict[str, Any]],
        max_cache_size_mb: float,
        min_cache_util_mb: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Notebook-aligned fixed-TTL choice:
        1. Prefer candidates that satisfy cache-size feasibility
        2. Among feasible candidates, pick the lowest stale_pct
        3. If none are feasible, prefer candidates under the memory cap
        4. If still none, fall back to the smallest TTL
        """
        if not candidates:
            return None

        feasible = [row for row in candidates if row.get("feasible", False)]
        if feasible:
            return min(feasible, key=lambda row: row.get("stale_pct", float("inf")))

        under_cap = [row for row in candidates if row.get("estimated_cache_mb", float("inf")) <= max_cache_size_mb]
        if under_cap:
            return min(under_cap, key=lambda row: row.get("stale_pct", float("inf")))

        return min(candidates, key=lambda row: row.get("ttl_value", float("inf")))

    def _compute_controlled_ttl(
        self,
        request: RequestContext,
        lambda_i: float,
        now: datetime,
    ) -> int:
        lead_time_days = self._lead_time_days(request, now)
        km_ttl = self._lookup_km_ttl(lead_time_days)
        if km_ttl is not None:
            return int(max(self.min_ttl_seconds, min(self.max_ttl_seconds, km_ttl)))
        lookup_ttl = self._lookup_ttl_by_lead_time(lead_time_days, self.controlled_ttl_lookup_by_lead_time)
        if lookup_ttl is not None:
            return int(max(self.min_ttl_seconds, min(self.max_ttl_seconds, lookup_ttl)))
        return self._compute_ttl_from_freshness(lambda_i, self.target_freshness_controlled)


    def _compute_uncontrolled_ttl(
        self,
        request: RequestContext,
        lambda_i: float,
        now: datetime,
    ) -> int:
        lead_time_days = self._lead_time_days(request, now)
        km_ttl = self._lookup_km_ttl(lead_time_days)
        if km_ttl is not None:
            return int(max(self.min_ttl_seconds, min(self.max_ttl_seconds, km_ttl)))
        lookup_ttl = self._lookup_ttl_by_lead_time(lead_time_days, self.uncontrolled_ttl_lookup_by_lead_time)
        if lookup_ttl is not None:
            return int(max(self.min_ttl_seconds, min(self.max_ttl_seconds, lookup_ttl)))
        return self._compute_ttl_from_freshness(lambda_i, self.target_freshness_uncontrolled)

    def _update_theta(self, score: float) -> float:
        self._score_history.append(score)
        self._theta = _percentile(list(self._score_history), self.score_percentile)
        return self._theta

    def _touch_entry(self, entry: CacheEntry, now: datetime) -> None:
        age = (now - entry.last_access_time).total_seconds()
        entry.age_seconds += max(0.0, age)
        entry.last_access_time = now
        entry.recent_freq += 1

    def _record_provider_call(self, kind: str) -> None:
        with self._lock:
            self._provider_call_counts[kind] = self._provider_call_counts.get(kind, 0) + 1

    def provider_call_counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._provider_call_counts)

    def _refresh_controlled_entries(self, entry: CacheEntry, key: str, now: datetime) -> None:
        provider = self._refresh_provider
        if provider is None:
            return

        self._record_provider_call("refresh")
        payload = provider(entry.request)
        entry.payload = payload
        entry.expires_at = now + timedelta(
            seconds=self._compute_controlled_ttl(entry.request, entry.lambda_i, now)
        )

    def _refresh_uncontrolled_entries(self, entry: CacheEntry, key: str, now: datetime) -> None:
        provider = self._refresh_provider
        if provider is None:
            return

        self._record_provider_call("refresh")
        payload = provider(entry.request)
        entry.payload = payload
        entry.expires_at = now + timedelta(
            seconds=self._compute_uncontrolled_ttl(entry.request, entry.lambda_i, now)
        )

    def _get_from_controlled(self, key: str, now: datetime) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._controlled.get(key)
            if entry is None:
                return None
            # If TTL has expired, try to refresh on-hit
            if self._is_ttl_expired(entry, now):
                self._refresh_controlled_entries(entry, key, now)

            self._touch_entry(entry, now)
            self._controlled.move_to_end(key)
            return entry

    def _get_from_uncontrolled(self, key: str, now: datetime) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._uncontrolled.get(key)
            if entry is None:
                return None
            # If TTL has expired, refresh on-hit using uncontrolled TTL.
            if self._is_ttl_expired(entry, now):
                self._refresh_uncontrolled_entries(entry, key, now)
            self._touch_entry(entry, now)
            self._uncontrolled.move_to_end(key)
            return entry

    def _admit_to_controlled(self, key: str, entry: CacheEntry) -> Tuple[bool, str | None, CacheEntry | None]:
        with self._lock:
            if self.controlled_capacity <= 0:
                return False, None, None

            # Safety path: if the key already exists, update in-place.
            if key in self._controlled:
                self._controlled[key] = entry
                self._controlled.move_to_end(key)
                heapq.heappush(self._controlled_min_heap, (entry.score, key))
                return True, None, None

            if len(self._controlled) < self.controlled_capacity:
                self._controlled[key] = entry
                self._controlled.move_to_end(key)
                heapq.heappush(self._controlled_min_heap, (entry.score, key))
                return True, None, None

            while self._controlled_min_heap:
                min_score, min_key = self._controlled_min_heap[0]
                current = self._controlled.get(min_key)
                if current is None or current.score != min_score:
                    heapq.heappop(self._controlled_min_heap)
                    continue
                break

            if not self._controlled_min_heap:
                self._controlled[key] = entry
                self._controlled.move_to_end(key)
                heapq.heappush(self._controlled_min_heap, (entry.score, key))
                return True, None, None

            min_score, min_key = self._controlled_min_heap[0]
            if entry.score <= min_score:
                return False, None, None

            heapq.heappop(self._controlled_min_heap)
            evicted_entry = self._controlled[min_key]
            del self._controlled[min_key]
            self._controlled[key] = entry
            self._controlled.move_to_end(key)
            heapq.heappush(self._controlled_min_heap, (entry.score, key))
            return True, min_key, evicted_entry

    def _admit_to_uncontrolled(self, key: str, entry: CacheEntry) -> Tuple[bool, str | None]:
        with self._lock:
            if self.uncontrolled_capacity <= 0:
                return False, None

            if key in self._uncontrolled:
                self._uncontrolled[key] = entry
                self._uncontrolled.move_to_end(key)
                return True, None

            evicted_key: str | None = None

            if len(self._uncontrolled) >= self.uncontrolled_capacity:
                evicted_key, _ = self._uncontrolled.popitem(last=False)

            self._uncontrolled[key] = entry
            self._uncontrolled.move_to_end(key)
            return True, evicted_key

    def get(
        self,
        request: RequestContext,
        p_reuse: float,
        lambda_i: float,
        provider: TruthPriceProvider,
    ) -> WorkflowResult:
        self._refresh_provider = provider
        request_now = request.rq_timestamp
        
        # Purge stay-date expired entries if date has changed
        self._purge_stay_date_expired_entries(request_now)
        
        key = request.cache_key()
        score = self._compute_score(p_reuse, lambda_i)
        theta = self._update_theta(score)

        controlled_hit = self._get_from_controlled(key, request_now)
        if controlled_hit is not None:
            return WorkflowResult(
                key=key,
                source="controlled_hit",
                payload=controlled_hit.payload,
                cached=True,
                tier="controlled",
                admission_score=controlled_hit.score,
                theta=theta,
            )

        uncontrolled_hit = self._get_from_uncontrolled(key, request_now)
        if uncontrolled_hit is not None:
            return WorkflowResult(
                key=key,
                source="uncontrolled_hit",
                payload=uncontrolled_hit.payload,
                cached=True,
                tier="uncontrolled",
                admission_score=uncontrolled_hit.score,
                theta=theta,
            )

        self._record_provider_call("miss")
        payload = provider(request)
        controlled_ttl = self._compute_controlled_ttl(request, lambda_i, request_now)
        uncontrolled_ttl = self._compute_uncontrolled_ttl(request, lambda_i, request_now)

        controlled_entry = CacheEntry(
            request=request,
            payload=payload,
            expires_at=request_now + timedelta(seconds=controlled_ttl),
            score=score,
            p_reuse=p_reuse,
            lambda_i=lambda_i,
            recent_freq=1,
            last_access_time=request_now,
            age_seconds=0.0,
        )
        uncontrolled_entry = CacheEntry(
            request=request,
            payload=payload,
            expires_at=request_now + timedelta(seconds=uncontrolled_ttl),
            score=score,
            p_reuse=p_reuse,
            lambda_i=lambda_i,
            recent_freq=1,
            last_access_time=request_now,
            age_seconds=0.0,
        )

        if score >= theta:
            admitted_controlled, evicted_controlled_key, evicted_controlled_entry = self._admit_to_controlled(key, controlled_entry)
            if admitted_controlled:
                evicted_uncontrolled_key: str | None = None
                # Demote controlled-evicted key into uncontrolled tier.
                if evicted_controlled_key is not None and evicted_controlled_entry is not None:
                    _, evicted_uncontrolled_key = self._admit_to_uncontrolled(
                        evicted_controlled_key,
                        evicted_controlled_entry,
                    )
                return WorkflowResult(
                    key=key,
                    source="miss_admit_controlled",
                    payload=payload,
                    cached=True,
                    tier="controlled",
                    admission_score=score,
                    theta=theta,
                    evicted_key=evicted_controlled_key,
                )

        _, evicted_uncontrolled_key = self._admit_to_uncontrolled(key, uncontrolled_entry)

        return WorkflowResult(
            key=key,
            source="miss_admit_uncontrolled",
            payload=payload,
            cached=True,
            tier="uncontrolled",
            admission_score=score,
            theta=theta,
            evicted_key=evicted_uncontrolled_key,
        )

    def prefetch_controlled(
        self,
        candidates: List[Tuple[RequestContext, float, float]],
        provider: TruthPriceProvider,
    ) -> List[str]:
        """
        Prefetch top-k p_reuse requests into controlled cache when keys are not cached.
        k defaults to 1% of controlled capacity (min 1).
        """
        if not candidates:
            return []

        k = max(1, int(self.controlled_capacity * self.prefetch_ratio))
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

        # Check top candidates first and skip keys that already exist in either cache.
        selected: List[Tuple[RequestContext, float, float]] = []
        for request, p_reuse, lambda_i in sorted_candidates:
            if len(selected) >= k:
                break
            key = request.cache_key()
            with self._lock:
                if key in self._controlled or key in self._uncontrolled:
                    continue
            selected.append((request, p_reuse, lambda_i))

        admitted_keys: List[str] = []
        for request, p_reuse, lambda_i in selected:
            key = request.cache_key()
            self._record_provider_call("prefetch")
            payload = provider(request)
            ttl = self._compute_controlled_ttl(request, lambda_i, request.rq_timestamp)
            entry = CacheEntry(
                request=request,
                payload=payload,
                expires_at=request.rq_timestamp + timedelta(seconds=ttl),
                score=self._compute_score(p_reuse, lambda_i),
                p_reuse=p_reuse,
                lambda_i=lambda_i,
                recent_freq=1,
                last_access_time=request.rq_timestamp,
                age_seconds=0.0,
            )
            admitted, _, _ = self._admit_to_controlled(key, entry)
            if admitted:
                admitted_keys.append(key)

        return admitted_keys

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._controlled.pop(key, None)
            self._uncontrolled.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._controlled.clear()
            self._controlled_min_heap.clear()
            self._uncontrolled.clear()
            self._score_history.clear()
            self._theta = 0.0

    def _count_valid(self, store: Dict[str, CacheEntry] | OrderedDict[str, CacheEntry]) -> int:
        now = self._now_fn()
        return sum(1 for entry in store.values() if entry.expires_at > now)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "controlled_size": len(self._controlled),
                "controlled_valid": self._count_valid(self._controlled),
                "controlled_capacity": self.controlled_capacity,
                "uncontrolled_size": len(self._uncontrolled),
                "uncontrolled_valid": self._count_valid(self._uncontrolled),
                "uncontrolled_capacity": self.uncontrolled_capacity,
                "theta": self._theta,
                "score_history_size": len(self._score_history),
            }
