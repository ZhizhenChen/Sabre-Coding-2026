from __future__ import annotations

from collections import OrderedDict
from contextlib import redirect_stdout
from dataclasses import dataclass
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd

from Cache_System_Workflow.sabre_cache_workflow_v2 import RequestContext, TruthPriceProvider
from data_processing.pipeline import DataPipelineProcessor


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
    evicted_key: str | None = None


class VanillaLRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, request: RequestContext, provider: TruthPriceProvider) -> LRUResult:
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
                evicted_key=None,
            )

        self.misses += 1
        payload = provider(request)
        evicted_key: str | None = None

        if self.capacity > 0 and len(self._store) >= self.capacity:
            evicted_key, _ = self._store.popitem(last=False)
            self.evictions += 1

        if self.capacity > 0:
            self._store[key] = payload
            self._store.move_to_end(key)

        return LRUResult(
            key=key,
            source="lru_miss_admit",
            payload=payload,
            cached=True,
            evicted_key=evicted_key,
        )

    def snapshot(self) -> List[str]:
        lines: List[str] = []
        for idx, key in enumerate(self._store.keys(), start=1):
            lines.append(f"{idx}. key={key}")
        return lines

    def size(self) -> int:
        return len(self._store)
