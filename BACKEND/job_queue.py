"""
Lightweight job queue facade.

The in-process backend is the default so local development works without Redis
or a worker service. The module deliberately presents a queue-shaped interface
that can be replaced by RQ, Celery, or Temporal at the service boundary.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import os
from threading import Lock
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from event_bus import publish_event


JOB_QUEUE_BACKEND = os.getenv("JOB_QUEUE_BACKEND", "in-process")
JOB_WORKERS = max(1, int(os.getenv("JOB_WORKERS", "2")))
JOB_HISTORY_SIZE = max(10, int(os.getenv("JOB_HISTORY_SIZE", "250")))

_EXECUTOR = ThreadPoolExecutor(max_workers=JOB_WORKERS)
_JOBS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_LOCK = Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _snapshot(job: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(job)


def _store(job: dict[str, Any]) -> None:
    with _LOCK:
        _JOBS[job["job_id"]] = job
        _JOBS.move_to_end(job["job_id"])
        while len(_JOBS) > JOB_HISTORY_SIZE:
            _JOBS.popitem(last=False)


def _update(job_id: str, **updates: Any) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        job.update(updates)
        _JOBS.move_to_end(job_id)
        return _snapshot(job)


def submit_job(
    job_type: str,
    fn: Callable[[], Any],
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit a background job and return its metadata snapshot."""
    job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    job = {
        "job_id": job_id,
        "type": job_type,
        "status": "queued",
        "backend": JOB_QUEUE_BACKEND,
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "latency_ms": None,
        "payload": payload or {},
        "result": None,
        "error": None,
    }
    _store(job)
    publish_event("job.queued", {"job_id": job_id, "type": job_type, "backend": JOB_QUEUE_BACKEND})

    def _run() -> None:
        started = perf_counter()
        _update(job_id, status="running", started_at=_now())
        publish_event("job.started", {"job_id": job_id, "type": job_type})
        try:
            result = fn()
            latency_ms = round((perf_counter() - started) * 1000, 2)
            _update(
                job_id,
                status="succeeded",
                finished_at=_now(),
                latency_ms=latency_ms,
                result=result,
                error=None,
            )
            publish_event("job.succeeded", {"job_id": job_id, "type": job_type, "latency_ms": latency_ms})
        except Exception as exc:
            latency_ms = round((perf_counter() - started) * 1000, 2)
            _update(
                job_id,
                status="failed",
                finished_at=_now(),
                latency_ms=latency_ms,
                error=str(exc),
            )
            publish_event(
                "job.failed",
                {"job_id": job_id, "type": job_type, "latency_ms": latency_ms, "error": str(exc)},
                severity="ERROR",
            )

    _EXECUTOR.submit(_run)
    return get_job(job_id) or job


def get_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return _snapshot(job) if job else None


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with _LOCK:
        jobs = list(_JOBS.values())[-max(1, int(limit)):]
        return [_snapshot(job) for job in reversed(jobs)]


def queue_stats() -> dict[str, Any]:
    with _LOCK:
        status_counts: dict[str, int] = {}
        for job in _JOBS.values():
            status = str(job.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
        history_size = len(_JOBS)
    return {
        "backend": JOB_QUEUE_BACKEND,
        "workers": JOB_WORKERS,
        "history_size": history_size,
        "max_history_size": JOB_HISTORY_SIZE,
        "status_counts": status_counts,
    }
