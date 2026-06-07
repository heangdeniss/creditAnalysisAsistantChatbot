"""
Lightweight contextual-bandit learner for credit offer policy decisions.

The supervised PD models still estimate risk. This module learns which action
works best for a borrower/offer context after the system receives outcome
feedback. It uses a per-action LinUCB model so it can learn online without
extra dependencies beyond NumPy.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np


ACTIONS = (
    "approve",
    "reject",
    "manual_review",
    "reduce_loan_amount",
    "conditional_rate",
    "request_documents",
)

FEATURE_DIM = 14
POLICY_VERSION = "linucb-credit-offer-v1"

_STATE_DIR = Path(os.getenv("POLICY_STATE_DIR", Path(__file__).resolve().parent / "policy_state"))
_STATE_PATH = _STATE_DIR / "linucb_policy.json"
_FEEDBACK_PATH = _STATE_DIR / "policy_feedback.jsonl"
_ALPHA = float(os.getenv("POLICY_LINUCB_ALPHA", "1.5"))
_RIDGE = float(os.getenv("POLICY_LINUCB_RIDGE", "1.0"))
_LEARNED_WEIGHT = float(os.getenv("POLICY_LEARNED_WEIGHT", "1.0"))
_LOCK = Lock()

_state: dict[str, Any] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _empty_action_state() -> dict[str, Any]:
    return {
        "A": (np.eye(FEATURE_DIM) * _RIDGE).tolist(),
        "b": np.zeros(FEATURE_DIM).tolist(),
        "updates": 0,
    }


def _empty_state() -> dict[str, Any]:
    return {
        "version": POLICY_VERSION,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "feature_dim": FEATURE_DIM,
        "alpha": _ALPHA,
        "ridge": _RIDGE,
        "learned_weight": _LEARNED_WEIGHT,
        "actions": {action: _empty_action_state() for action in ACTIONS},
        "feedback_count": 0,
    }


def _load_state() -> dict[str, Any]:
    global _state
    if _state is not None:
        return _state
    if _STATE_PATH.exists():
        try:
            with _STATE_PATH.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if loaded.get("version") == POLICY_VERSION and int(loaded.get("feature_dim", 0)) == FEATURE_DIM:
                for action in ACTIONS:
                    loaded.setdefault("actions", {}).setdefault(action, _empty_action_state())
                _state = loaded
                return _state
        except Exception:
            pass
    _state = _empty_state()
    return _state


def _save_state(state: dict[str, Any]) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = _STATE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)
    tmp_path.replace(_STATE_PATH)


def _append_feedback(row: dict[str, Any]) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    with _FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
        if np.isfinite(num):
            return num
    except (TypeError, ValueError):
        pass
    return default


def context_vector(applicant: dict[str, Any], candidate: dict[str, Any]) -> np.ndarray:
    """Return a bounded feature vector for one borrower/action candidate."""
    income = max(_safe_float(applicant.get("person_income")), 1.0)
    loan_amount = _safe_float(candidate.get("loan_amount"), _safe_float(applicant.get("loan_amnt")))
    lti = loan_amount / income
    prior_default = 1.0 if str(applicant.get("cb_person_default_on_file", "N")).upper() == "Y" else 0.0
    home_rent = 1.0 if str(applicant.get("person_home_ownership", "")).upper() == "RENT" else 0.0
    home_own = 1.0 if str(applicant.get("person_home_ownership", "")).upper() == "OWN" else 0.0

    return np.array([
        1.0,
        _safe_float(applicant.get("person_age")) / 100.0,
        min(income / 150000.0, 3.0),
        _safe_float(applicant.get("person_emp_length")) / 60.0,
        loan_amount / 50000.0,
        _safe_float(candidate.get("interest_rate"), _safe_float(applicant.get("loan_int_rate"))) / 40.0,
        _safe_float(applicant.get("cb_person_cred_hist_length")) / 60.0,
        min(lti, 3.0),
        _safe_float(candidate.get("pd_pct")) / 100.0,
        _safe_float(candidate.get("expected_profit")) / 5000.0,
        _safe_float(candidate.get("operational_cost")) / 500.0,
        _safe_float(candidate.get("customer_penalty")) / 500.0,
        prior_default,
        max(home_rent, home_own * 0.5),
    ], dtype=np.float64)


def _predict_action(action_state: dict[str, Any], x: np.ndarray) -> tuple[float, float, float]:
    A = np.asarray(action_state["A"], dtype=np.float64)
    b = np.asarray(action_state["b"], dtype=np.float64)
    inv_A = np.linalg.pinv(A)
    theta = inv_A @ b
    expected = float(theta @ x)
    uncertainty = float(_ALPHA * np.sqrt(max(0.0, x @ inv_A @ x)))
    return expected, uncertainty, expected + uncertainty


def score_candidates(applicant: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add learned LinUCB scores to action candidates and return a new list."""
    with _LOCK:
        state = _load_state()
        actions = state["actions"]
        scored = []
        for candidate in candidates:
            row = dict(candidate)
            action = str(row.get("action", ""))
            action_state = actions.get(action)
            if action_state is None:
                row["learned_reward"] = 0.0
                row["exploration_bonus"] = 0.0
                row["policy_score"] = float(row.get("reward", 0.0))
                scored.append(row)
                continue
            x = context_vector(applicant, row)
            learned_reward, exploration_bonus, ucb_score = _predict_action(action_state, x)
            row["learned_reward"] = round(learned_reward, 4)
            row["exploration_bonus"] = round(exploration_bonus, 4)
            row["policy_score"] = round(
                _safe_float(row.get("reward")) + (_LEARNED_WEIGHT * ucb_score),
                4,
            )
            row["policy_updates"] = int(action_state.get("updates", 0))
            scored.append(row)
        return scored


def reward_from_outcome(payload: dict[str, Any]) -> float:
    """Compute a feedback reward when the caller does not provide one directly."""
    if payload.get("reward") is not None:
        return _safe_float(payload.get("reward"))

    reward = _safe_float(payload.get("realized_profit"))
    if payload.get("customer_accepted") is True:
        reward += 75.0
    if payload.get("customer_accepted") is False:
        reward -= 25.0
    if payload.get("defaulted") is True:
        reward -= _safe_float(payload.get("loss_amount"), 500.0)
    if payload.get("officer_override") is True:
        reward -= 50.0
    if payload.get("fairness_flag") is True:
        reward -= 125.0
    return round(reward, 4)


def record_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    """Update one action model from outcome feedback and persist the event."""
    action = str(payload.get("action", "")).strip()
    if action not in ACTIONS:
        raise ValueError(f"Unknown policy action: {action}")

    applicant = payload.get("applicant")
    if not isinstance(applicant, dict):
        raise ValueError("Feedback payload must include an applicant object.")

    candidate = dict(payload.get("candidate") or {})
    candidate.setdefault("action", action)
    candidate.setdefault("loan_amount", applicant.get("loan_amnt"))
    candidate.setdefault("interest_rate", applicant.get("loan_int_rate"))
    candidate.setdefault("pd_pct", payload.get("pd_pct", 0.0))
    candidate.setdefault("expected_profit", payload.get("expected_profit", 0.0))

    reward = reward_from_outcome(payload)
    x = context_vector(applicant, candidate)

    with _LOCK:
        state = _load_state()
        action_state = state["actions"][action]
        A = np.asarray(action_state["A"], dtype=np.float64)
        b = np.asarray(action_state["b"], dtype=np.float64)
        A += np.outer(x, x)
        b += reward * x
        action_state["A"] = A.tolist()
        action_state["b"] = b.tolist()
        action_state["updates"] = int(action_state.get("updates", 0)) + 1
        state["feedback_count"] = int(state.get("feedback_count", 0)) + 1
        state["updated_at"] = _utc_now()
        _save_state(state)

        event = {
            "feedback_id": str(payload.get("feedback_id") or uuid.uuid4()),
            "created_at": _utc_now(),
            "action": action,
            "reward": reward,
            "model": payload.get("model"),
            "applicant": applicant,
            "candidate": candidate,
            "outcome": {
                key: payload.get(key)
                for key in (
                    "realized_profit",
                    "defaulted",
                    "loss_amount",
                    "customer_accepted",
                    "officer_override",
                    "fairness_flag",
                )
                if key in payload
            },
        }
        _append_feedback(event)

        learned_reward, exploration_bonus, ucb_score = _predict_action(action_state, x)
        return {
            "status": "ok",
            "feedback_id": event["feedback_id"],
            "action": action,
            "reward": reward,
            "updates": action_state["updates"],
            "learned_reward": round(learned_reward, 4),
            "exploration_bonus": round(exploration_bonus, 4),
            "ucb_score": round(ucb_score, 4),
        }


def policy_summary() -> dict[str, Any]:
    with _LOCK:
        state = _load_state()
        return {
            "version": state["version"],
            "updated_at": state["updated_at"],
            "feedback_count": int(state.get("feedback_count", 0)),
            "alpha": _ALPHA,
            "ridge": _RIDGE,
            "learned_weight": _LEARNED_WEIGHT,
            "state_path": str(_STATE_PATH),
            "feedback_path": str(_FEEDBACK_PATH),
            "actions": {
                action: {"updates": int(row.get("updates", 0))}
                for action, row in state["actions"].items()
            },
        }
