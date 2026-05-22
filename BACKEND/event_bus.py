"""
Event publishing helpers for logs, metrics, and trace events.

The default sink is structured stdout so local development stays dependency-free.
If OTEL_EXPORTER_OTLP_ENDPOINT is set, events are also sent as best-effort OTLP
HTTP JSON log records to an OpenTelemetry Collector.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from time import time_ns
from typing import Any
from urllib import request


SERVICE_NAME = os.getenv("SERVICE_NAME", "credit-risk-rag")
EVENT_LOG_ENABLED = os.getenv("EVENT_LOG_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").rstrip("/")
OTLP_TIMEOUT_S = float(os.getenv("OTEL_EXPORTER_TIMEOUT_S", os.getenv("OTEL_EXPORT_TIMEOUT_S", "1.5")))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _otlp_log_payload(name: str, payload: dict[str, Any], severity: str) -> dict[str, Any]:
    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": SERVICE_NAME}},
                    ],
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "credit-risk-rag.event_bus"},
                        "logRecords": [
                            {
                                "timeUnixNano": str(time_ns()),
                                "severityText": severity,
                                "body": {"stringValue": json.dumps({"event": name, **payload}, default=str)},
                                "attributes": [
                                    {"key": "event.name", "value": {"stringValue": name}},
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    }


def _send_otlp(name: str, payload: dict[str, Any], severity: str) -> None:
    if not OTLP_ENDPOINT:
        return
    url = f"{OTLP_ENDPOINT}/v1/logs"
    body = json.dumps(_otlp_log_payload(name, payload, severity), default=str).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        request.urlopen(req, timeout=OTLP_TIMEOUT_S).close()
    except Exception:
        # Telemetry must never break the product path.
        return


def publish_event(
    name: str,
    payload: dict[str, Any] | None = None,
    *,
    severity: str = "INFO",
    service: str | None = None,
) -> dict[str, Any]:
    """Publish an event to stdout and optional OTLP."""
    event = {
        "type": "event",
        "event": name,
        "service": service or SERVICE_NAME,
        "severity": severity,
        "created_at": _utc_now(),
        "payload": payload or {},
    }
    if EVENT_LOG_ENABLED:
        try:
            print(json.dumps(event, ensure_ascii=True, default=str))
        except Exception:
            pass
    _send_otlp(name, event, severity)
    return event
