"""
runtime_utils.py
Small runtime helpers for caching and rate limiting.
"""
from __future__ import annotations

from collections import OrderedDict, deque
import json
from threading import Lock
import time
from typing import Any


def stable_json(data: Any) -> str:
    """Return a stable JSON string for cache keys."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


class TTLCache:
    """Simple TTL + LRU cache (in-memory)."""

    def __init__(self, *, max_size: int = 256, ttl_s: int = 300) -> None:
        self._max_size = max(1, int(max_size))
        self._ttl_s = max(1, int(ttl_s))
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        now = time.time()
        with self._lock:
            self._data[key] = (now + self._ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"size": len(self._data), "max_size": self._max_size}


class SlidingWindowRateLimiter:
    """Basic sliding-window rate limiter (per key)."""

    def __init__(self, *, max_requests: int, window_s: int) -> None:
        self._max_requests = max(1, int(max_requests))
        self._window_s = max(1, int(window_s))
        self._buckets: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            bucket = self._buckets.setdefault(key, deque())
            while bucket and bucket[0] <= now - self._window_s:
                bucket.popleft()
            if len(bucket) >= self._max_requests:
                return False, len(bucket)
            bucket.append(now)
            return True, len(bucket)

    @property
    def window_s(self) -> int:
        return self._window_s

    @property
    def max_requests(self) -> int:
        return self._max_requests
