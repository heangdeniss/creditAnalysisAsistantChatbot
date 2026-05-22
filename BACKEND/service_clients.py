"""HTTP clients for optional split-service deployment."""
from __future__ import annotations

import json
import os
from typing import Any
from urllib import request


TABULAR_SCORING_SERVICE_URL = os.getenv("TABULAR_SCORING_SERVICE_URL", "").rstrip("/")
SERVICE_CLIENT_TIMEOUT_S = float(os.getenv("SERVICE_CLIENT_TIMEOUT_S", "30"))


def _post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{base_url}{path}"
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=SERVICE_CLIENT_TIMEOUT_S) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def remote_tabular_enabled() -> bool:
    return bool(TABULAR_SCORING_SERVICE_URL)


def remote_predict(payload: dict[str, Any]) -> dict[str, Any]:
    response = _post_json(TABULAR_SCORING_SERVICE_URL, "/predict", payload)
    return response.get("scores", response)


def remote_batch_score(applicants: list[dict[str, Any]]) -> dict[str, Any]:
    return _post_json(TABULAR_SCORING_SERVICE_URL, "/batch-score", {"applicants": applicants})
