"""
Lightweight observability helpers for the Credit Risk RAG API.

Uses only the Python standard library so traces work in local/offline setups.
The module keeps a bounded in-memory trace buffer and prints one JSON line per
completed trace for easy log collection later.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from statistics import median
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import uuid4


RECENT_TRACE_LIMIT = int(os.getenv("RECENT_TRACE_LIMIT", "250"))

_TRACES: deque[dict[str, Any]] = deque(maxlen=RECENT_TRACE_LIMIT)
_LOCK = Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_request_id() -> str:
    return f"req_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def start_trace(
    route: str,
    *,
    model: str | None = None,
    question: str = "",
    facts_present: bool = False,
) -> dict[str, Any]:
    """Create a mutable trace object owned by the current request."""
    return {
        "_started_at": perf_counter(),
        "request_id": new_request_id(),
        "route": route,
        "model": model,
        "question_chars": len(question or ""),
        "facts_present": bool(facts_present),
        "created_at": utc_now_iso(),
        "status": "running",
        "error": None,
        "latency_ms": None,
        "retrieval": {},
        "generation": {},
        "events": [],
    }


def add_event(trace: dict[str, Any] | None, event: str, payload: dict[str, Any] | None = None) -> None:
    if trace is None:
        return
    trace.setdefault("events", []).append({
        "at": utc_now_iso(),
        "event": event,
        "payload": payload or {},
    })


def set_trace_section(trace: dict[str, Any] | None, section: str, values: dict[str, Any]) -> None:
    if trace is None:
        return
    current = trace.setdefault(section, {})
    if isinstance(current, dict):
        current.update(values)
    else:
        trace[section] = dict(values)


def finish_trace(
    trace: dict[str, Any] | None,
    *,
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any] | None:
    """Finalize, store, and log a trace. Safe to call at most once per trace."""
    if trace is None:
        return None
    if trace.get("status") != "running":
        return trace

    started_at = float(trace.pop("_started_at", perf_counter()))
    trace["latency_ms"] = round((perf_counter() - started_at) * 1000, 2)
    trace["status"] = status
    trace["error"] = error
    trace["finished_at"] = utc_now_iso()

    frozen = deepcopy(trace)
    with _LOCK:
        _TRACES.appendleft(frozen)

    try:
        print(json.dumps({"type": "trace", **frozen}, ensure_ascii=True))
    except Exception:
        pass
    return frozen


def recent_traces(limit: int = 50) -> list[dict[str, Any]]:
    with _LOCK:
        return deepcopy(list(_TRACES)[:limit])


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return round(ordered[lo] * (1 - frac) + ordered[hi] * frac, 2)


def metrics_summary() -> dict[str, Any]:
    traces = recent_traces(RECENT_TRACE_LIMIT)
    completed = [t for t in traces if t.get("status") != "running"]
    status_counts = Counter(str(t.get("status", "unknown")) for t in completed)

    latencies = [float(t["latency_ms"]) for t in completed if isinstance(t.get("latency_ms"), (int, float))]
    retrieval_latencies = [
        float(t["retrieval"]["latency_ms"])
        for t in completed
        if isinstance(t.get("retrieval"), dict)
        and isinstance(t["retrieval"].get("latency_ms"), (int, float))
    ]
    generation_latencies = [
        float(t["generation"]["latency_ms"])
        for t in completed
        if isinstance(t.get("generation"), dict)
        and isinstance(t["generation"].get("latency_ms"), (int, float))
    ]
    queue_waits = [
        float(t["generation"]["queue_wait_ms"])
        for t in completed
        if isinstance(t.get("generation"), dict)
        and isinstance(t["generation"].get("queue_wait_ms"), (int, float))
    ]

    tokens_per_sec = [
        float(t["generation"]["tokens_per_sec"])
        for t in completed
        if isinstance(t.get("generation"), dict)
        and isinstance(t["generation"].get("tokens_per_sec"), (int, float))
    ]

    total = len(completed)
    errors = status_counts.get("error", 0)
    empty_retrievals = sum(
        1
        for t in completed
        if isinstance(t.get("retrieval"), dict)
        and t["retrieval"].get("chunk_count") == 0
    )

    return {
        "window_size": total,
        "status_counts": dict(status_counts),
        "error_rate": round(errors / total, 4) if total else 0.0,
        "empty_retrieval_rate": round(empty_retrievals / total, 4) if total else 0.0,
        "latency_ms": {
            "p50": round(median(latencies), 2) if latencies else 0.0,
            "p95": _percentile(latencies, 0.95),
        },
        "retrieval_latency_ms": {
            "p50": round(median(retrieval_latencies), 2) if retrieval_latencies else 0.0,
            "p95": _percentile(retrieval_latencies, 0.95),
        },
        "generation_latency_ms": {
            "p50": round(median(generation_latencies), 2) if generation_latencies else 0.0,
            "p95": _percentile(generation_latencies, 0.95),
        },
        "queue_wait_ms": {
            "p50": round(median(queue_waits), 2) if queue_waits else 0.0,
            "p95": _percentile(queue_waits, 0.95),
        },
        "tokens_per_sec": {
            "median": round(median(tokens_per_sec), 2) if tokens_per_sec else 0.0,
        },
    }
