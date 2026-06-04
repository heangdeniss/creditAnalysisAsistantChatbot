"""
ml_predict.py  —  ML inference + SHAP explanations for the credit-risk API.

Models
  - logistic_regression : custom LR stored as {beta, mean, std, threshold}
  - catboost            : sklearn-API CatBoost stored as {model, scaler, threshold}
  - neural_network      : NumPy MLP stored as {params, scaler, threshold, features}
  - random_forest       : sklearn RandomForestClassifier stored as {model, threshold}
                          OR a bare RandomForestClassifier object

Public functions
  predict(raw)              → predictions from all models
  explain_shap(raw, model)  → SHAP attribution for a chosen model
"""
from __future__ import annotations

import math
import os
import pickle
from typing import Any

import joblib
import numpy as np

from feature_store import standardize_applicant

# Paths
_ML_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "ML_Model"))
_CALIBRATION_DIR = os.getenv("CALIBRATION_DIR", _ML_DIR)
_WARN_MISSING_CALIBRATION = os.getenv("CALIBRATION_WARN_MISSING", "0").lower() in {"1", "true", "yes"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


EXPECTED_LOSS_LGD = _env_float("EXPECTED_LOSS_LGD", 0.45)
SUGGESTION_TARGET_PD_PCT = _env_float("SUGGESTION_TARGET_PD_PCT", 5.0)
SUGGESTION_MIN_LOAN = _env_float("SUGGESTION_MIN_LOAN", 500.0)
SUGGESTION_LOAN_STEP = _env_float("SUGGESTION_LOAN_STEP", 50.0)
SUGGESTION_MAX_ITER = _env_int("SUGGESTION_MAX_ITER", 24)
SUGGESTION_MAX_LOAN = _env_float("SUGGESTION_MAX_LOAN", 50000.0)
SUGGESTION_MAX_LOAN_MULTIPLIER = _env_float("SUGGESTION_MAX_LOAN_MULTIPLIER", 2.0)

_MODEL_FILES: dict[str, str] = {
    "logistic_regression": os.path.join(_ML_DIR, "Logistic regression model.joblib"),
    "catboost":            os.path.join(_ML_DIR, "catboost_model_complete.pkl"),
    # NumPy MLP saved as {params, scaler, threshold, features}
    "neural_network":      os.path.join(_ML_DIR, "nn_model.pkl"),
    # sklearn RandomForestClassifier (bare or wrapped in a dict with a threshold key)
    "random_forest":       os.path.join(_ML_DIR, "random_forest_model.pkl"),
}

_CALIBRATION_CANDIDATES: dict[str, list[str]] = {
    "logistic_regression": [
        "logistic_regression_calibration.joblib",
        "logistic_regression_calibrator.joblib",
        "calibration_logistic_regression.joblib",
        "Logistic regression calibration.joblib",
    ],
    "catboost": [
        "catboost_calibration.joblib",
        "catboost_calibrator.joblib",
        "calibration_catboost.joblib",
    ],
    "neural_network": [
        "neural_network_calibration.joblib",
        "neural_network_calibrator.joblib",
        "calibration_neural_network.joblib",
    ],
    "random_forest": [
        "random_forest_calibration.joblib",
        "random_forest_calibrator.joblib",
        "calibration_random_forest.joblib",
    ],
}

# Human-readable feature labels for SHAP output
FEATURE_LABELS: dict[str, str] = {
    "person_age":                  "Age",
    "person_income":               "Annual Income",
    "person_emp_length":           "Employment Length",
    "loan_amnt":                   "Loan Amount",
    "loan_int_rate":               "Interest Rate",
    "cb_person_cred_hist_length":  "Credit History Length",
    "person_home_ownership_OTHER": "Home: Other",
    "person_home_ownership_OWN":   "Home: Own",
    "person_home_ownership_RENT":  "Home: Rent",
    "loan_intent_EDUCATION":       "Purpose: Education",
    "loan_intent_HOMEIMPROVEMENT": "Purpose: Home Impr.",
    "loan_intent_MEDICAL":         "Purpose: Medical",
    "loan_intent_PERSONAL":        "Purpose: Personal",
    "loan_intent_VENTURE":         "Purpose: Venture",
    "cb_person_default_on_file_Y": "Prior Default",
}

# Maps each one-hot dummy column → its parent categorical group key.
# Numerical features and cb_person_default_on_file_Y are not listed here
_DUMMY_GROUP: dict[str, str] = {
    "person_home_ownership_OTHER": "home_ownership",
    "person_home_ownership_OWN":   "home_ownership",
    "person_home_ownership_RENT":  "home_ownership",
    "loan_intent_EDUCATION":       "loan_intent",
    "loan_intent_HOMEIMPROVEMENT": "loan_intent",
    "loan_intent_MEDICAL":         "loan_intent",
    "loan_intent_PERSONAL":        "loan_intent",
    "loan_intent_VENTURE":         "loan_intent",
}

# Label shown when all dummies in a group are 0 (the dropped/baseline category).
_GROUP_BASELINE_LABEL: dict[str, str] = {
    "home_ownership": "Home: Mortgage",
    "loan_intent":    "Purpose: Debt Consol.",
}

# Business-logic helpers
def _risk_grade(pd_pct: float) -> str:
    """Return letter grade A-E from probability-of-default expressed as a %."""
    if pd_pct <= 2:   return "A"
    if pd_pct <= 5:   return "B"
    if pd_pct <= 7:   return "C"
    if pd_pct <= 10:  return "D"
    return "E"


def _decision(pd_pct: float) -> str:
    """Return credit decision from probability-of-default expressed as a %."""
    if pd_pct <= 5:   return "APPROVE"
    if pd_pct <= 10:  return "REVIEW"
    return "REJECT"


def _expected_loss_metrics(
    pd_pct: float,
    loan_amount: float,
    *,
    confidence_band: list[float] | None = None,
    lgd: float | None = None,
) -> dict[str, float]:
    lgd_value = EXPECTED_LOSS_LGD if lgd is None else float(lgd)
    lgd_value = float(np.clip(lgd_value, 0.0, 1.0))
    exposure = max(float(loan_amount), 0.0)

    pd_value = float(np.clip(pd_pct / 100.0, 0.0, 1.0))
    expected_loss = round(pd_value * exposure * lgd_value, 2)
    expected_loss_pct = round(pd_value * lgd_value * 100.0, 2)

    stressed_pd_pct = pd_pct
    if confidence_band and len(confidence_band) >= 2:
        stressed_pd_pct = max(stressed_pd_pct, float(confidence_band[1]))
    stressed_pd_value = float(np.clip(stressed_pd_pct / 100.0, 0.0, 1.0))
    stressed_loss = round(stressed_pd_value * exposure * lgd_value, 2)
    stressed_loss_pct = round(stressed_pd_value * lgd_value * 100.0, 2)

    return {
        "amount": expected_loss,
        "rate_pct": expected_loss_pct,
        "ead": round(exposure, 2),
        "lgd": round(lgd_value, 4),
        "base_pd_pct": round(pd_pct, 2),
        "stressed_amount": stressed_loss,
        "stressed_rate_pct": stressed_loss_pct,
        "stressed_pd_pct": round(stressed_pd_pct, 2),
    }

# Feature columns  (exact order from pd.get_dummies(drop_first=True))
FEATURE_COLS: list[str] = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "cb_person_cred_hist_length",
    # person_home_ownership dummies  (baseline = MORTGAGE)
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "person_home_ownership_RENT",
    # loan_intent dummies  (baseline = DEBTCONSOLIDATION)
    "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE",
    # cb_person_default_on_file dummy  (baseline = N)
    "cb_person_default_on_file_Y",
]  # 15 features

# Model cache
_cache: dict[str, Any] = {}
_calibration_cache: dict[str, Any | None] = {}
_calibration_warned: set[str] = set()
_cb_explainer = None  # shap.TreeExplainer for CatBoost (built once on first use)


def _load(name: str) -> Any | None:
    """Lazy-load and cache a model by name; returns None if unavailable."""
    if name in _cache:
        return _cache[name]

    path = _MODEL_FILES[name]
    if not os.path.exists(path):
        print(f"[ml] WARNING  {name} model not found at {path}")
        return None

    try:
        pkg = joblib.load(path)
        _cache[name] = pkg
        print(f"[ml] OK  loaded {name}")
        return pkg
    except Exception as exc:
        # Fallback for files saved with raw pickle (e.g. CatBoost)
        try:
            with open(path, "rb") as f:
                pkg = pickle.load(f)
            _cache[name] = pkg
            print(f"[ml] OK  loaded {name} (pickle fallback)")
            return pkg
        except Exception as exc2:
            print(f"[ml] ERROR  failed to load {name}: {exc2}")
            return None


def _load_calibrator(name: str) -> Any | None:
    """Load an optional saved probability calibrator for a model."""
    if name in _calibration_cache:
        return _calibration_cache[name]

    for filename in _CALIBRATION_CANDIDATES.get(name, []):
        path = os.path.join(_CALIBRATION_DIR, filename)
        if not os.path.exists(path):
            continue
        try:
            calibrator = joblib.load(path)
        except Exception:
            try:
                with open(path, "rb") as f:
                    calibrator = pickle.load(f)
            except Exception as exc:
                print(f"[ml] WARNING  failed to load calibration model for {name}: {exc}")
                calibrator = None
        _calibration_cache[name] = calibrator
        if calibrator is not None:
            print(f"[ml] OK  loaded calibration model for {name}")
        return calibrator

    _calibration_cache[name] = None
    if _WARN_MISSING_CALIBRATION and name not in _calibration_warned:
        print(f"[ml] WARNING  calibration model not found for {name}; using identity calibration")
        _calibration_warned.add(name)
    return None


def load_all_models() -> None:
    """Pre-warm all models at startup (called from FastAPI lifespan)."""
    for name in _MODEL_FILES:
        _load(name)
        _load_calibrator(name)


def _get_cb_explainer():
    """Return (and cache) a shap.TreeExplainer for the CatBoost model."""
    global _cb_explainer
    if _cb_explainer is not None:
        return _cb_explainer
    pkg = _load("catboost")
    if pkg is None:
        return None
    try:
        import shap
        _cb_explainer = shap.TreeExplainer(pkg["model"])
        print("[ml] OK  SHAP TreeExplainer ready")
        return _cb_explainer
    except Exception as exc:
        print(f"[ml] WARNING  SHAP unavailable: {exc}")
        return None

# Feature encoding  (raw dict  →  15-d float64 vector in FEATURE_COLS order)
def _encode(raw: dict) -> np.ndarray:
    raw = standardize_applicant(raw)
    home   = str(raw.get("person_home_ownership", "")).upper()
    intent = str(raw.get("loan_intent", "")).upper()
    cb_def = str(raw.get("cb_person_default_on_file", "N")).upper()

    row: dict[str, float] = {
        "person_age":                  float(raw["person_age"]),
        "person_income":               float(raw["person_income"]),
        "person_emp_length":           float(raw["person_emp_length"]),
        "loan_amnt":                   float(raw["loan_amnt"]),
        "loan_int_rate":               float(raw["loan_int_rate"]),
        "cb_person_cred_hist_length":  float(raw["cb_person_cred_hist_length"]),
        "person_home_ownership_OTHER": 1.0 if home   == "OTHER"           else 0.0,
        "person_home_ownership_OWN":   1.0 if home   == "OWN"             else 0.0,
        "person_home_ownership_RENT":  1.0 if home   == "RENT"            else 0.0,
        "loan_intent_EDUCATION":       1.0 if intent == "EDUCATION"       else 0.0,
        "loan_intent_HOMEIMPROVEMENT": 1.0 if intent == "HOMEIMPROVEMENT" else 0.0,
        "loan_intent_MEDICAL":         1.0 if intent == "MEDICAL"         else 0.0,
        "loan_intent_PERSONAL":        1.0 if intent == "PERSONAL"        else 0.0,
        "loan_intent_VENTURE":         1.0 if intent == "VENTURE"         else 0.0,
        "cb_person_default_on_file_Y": 1.0 if cb_def == "Y"              else 0.0,
    }
    return np.array([row[c] for c in FEATURE_COLS], dtype=np.float64)


# Scales X15 and prepends a bias column → (1, 16) array used by both models.
def _preprocess(X15: np.ndarray, scaler: Any) -> np.ndarray:
    return np.c_[np.ones((1, 1)), scaler.transform(X15.reshape(1, -1))]


def _sigmoid(z: float | np.ndarray) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))))


def _clip_probability(prob: float) -> float:
    return float(np.clip(prob, 0.0, 1.0))


def _logit(prob: float) -> float:
    p = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _coerce_probability(value: Any) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return _clip_probability(float(arr))
    if arr.ndim >= 2 and arr.shape[-1] > 1:
        return _clip_probability(float(arr.reshape(-1, arr.shape[-1])[0, 1]))
    return _clip_probability(float(arr.ravel()[0]))


def _calibrate_probability(model_name: str, raw_prob: float) -> tuple[float, dict]:
    calibrator = _load_calibrator(model_name)
    if calibrator is None:
        return _clip_probability(raw_prob), {"method": "identity", "available": False}

    try:
        if isinstance(calibrator, dict):
            method = str(calibrator.get("method", "")).lower()
            if method == "platt" or {"a", "b"} <= set(calibrator) or {"coef", "intercept"} <= set(calibrator):
                a = float(np.asarray(calibrator.get("a", calibrator.get("coef", 1.0))).ravel()[0])
                b = float(np.asarray(calibrator.get("b", calibrator.get("intercept", 0.0))).ravel()[0])
                return _sigmoid(a * _logit(raw_prob) + b), {"method": "platt", "available": True}

            model = calibrator.get("model") or calibrator.get("calibrator")
            if model is not None:
                calibrator = model

        X = np.array([[float(raw_prob)]])
        if hasattr(calibrator, "predict_proba"):
            return _coerce_probability(calibrator.predict_proba(X)), {"method": "predict_proba", "available": True}
        if hasattr(calibrator, "predict"):
            return _coerce_probability(calibrator.predict(X)), {"method": "predict", "available": True}
        if callable(calibrator):
            return _coerce_probability(calibrator(float(raw_prob))), {"method": "callable", "available": True}
    except Exception as exc:
        print(f"[ml] WARNING  calibration failed for {model_name}: {exc}; using identity calibration")

    return _clip_probability(raw_prob), {"method": "identity", "available": False}


def _score_pd_pct_for_amount(model_name: str, base_raw: dict, loan_amount: float) -> float:
    payload = dict(base_raw)
    payload["loan_amnt"] = float(loan_amount)
    X15 = _encode(payload)
    raw_prob, _ = _RUNNERS[model_name](X15)
    calibrated_prob, _ = _calibrate_probability(model_name, raw_prob)
    return float(calibrated_prob) * 100.0


def _suggestion_cap_amount(requested_amount: float) -> float:
    cap = max(float(requested_amount), 0.0)
    cap = max(cap, float(requested_amount) * float(SUGGESTION_MAX_LOAN_MULTIPLIER))
    max_cap = max(float(SUGGESTION_MAX_LOAN), float(requested_amount))
    return min(cap, max_cap)


def _search_max_loan_for_target(
    base_raw: dict,
    model_name: str,
    *,
    target_pd_pct: float,
    low: float,
    high: float,
) -> float:
    step = max(float(SUGGESTION_LOAN_STEP), 1.0)
    lo = float(low)
    hi = float(high)
    for _ in range(max(1, int(SUGGESTION_MAX_ITER))):
        if (hi - lo) <= step:
            break
        mid = (lo + hi) / 2.0
        pd_mid = _score_pd_pct_for_amount(model_name, base_raw, mid)
        if pd_mid <= target_pd_pct:
            lo = mid
        else:
            hi = mid
    return max(float(SUGGESTION_MIN_LOAN), math.floor(lo / step) * step)


def _suggest_loan_amount(
    base_raw: dict,
    model_name: str,
    *,
    target_pd_pct: float,
    current_pd_pct: float,
) -> dict:
    try:
        requested_amount = float(base_raw.get("loan_amnt", 0.0))
    except (TypeError, ValueError):
        requested_amount = 0.0

    if requested_amount <= 0:
        return {
            "status": "unavailable",
            "reason": "missing_loan_amount",
            "target_pd_pct": round(float(target_pd_pct), 2),
        }

    target = float(target_pd_pct)
    min_amount = max(float(SUGGESTION_MIN_LOAN), 0.0)
    requested_amount = max(requested_amount, min_amount)
    cap_amount = max(_suggestion_cap_amount(requested_amount), min_amount)

    if current_pd_pct <= target:
        pd_at_cap = _score_pd_pct_for_amount(model_name, base_raw, cap_amount)
        if pd_at_cap <= target:
            return {
                "status": "cap_reached",
                "target_pd_pct": round(target, 2),
                "requested_loan_amount": round(requested_amount, 2),
                "suggested_loan_amount": round(cap_amount, 2),
                "estimated_pd_pct": round(pd_at_cap, 2),
                "cap_loan_amount": round(cap_amount, 2),
            }

        suggested = _search_max_loan_for_target(
            base_raw,
            model_name,
            target_pd_pct=target,
            low=requested_amount,
            high=cap_amount,
        )
        pd_suggested = _score_pd_pct_for_amount(model_name, base_raw, suggested)
        status = "approved_as_is" if abs(suggested - requested_amount) < float(SUGGESTION_LOAN_STEP) else "increase_amount"
        return {
            "status": status,
            "target_pd_pct": round(target, 2),
            "requested_loan_amount": round(requested_amount, 2),
            "suggested_loan_amount": round(suggested, 2),
            "estimated_pd_pct": round(pd_suggested, 2),
            "cap_loan_amount": round(cap_amount, 2),
        }

    pd_at_min = _score_pd_pct_for_amount(model_name, base_raw, min_amount)
    if pd_at_min > target:
        return {
            "status": "unreachable",
            "target_pd_pct": round(target, 2),
            "requested_loan_amount": round(requested_amount, 2),
            "suggested_loan_amount": None,
            "estimated_pd_pct": round(pd_at_min, 2),
            "min_loan_amount": round(min_amount, 2),
        }

    suggested = _search_max_loan_for_target(
        base_raw,
        model_name,
        target_pd_pct=target,
        low=min_amount,
        high=requested_amount,
    )
    pd_suggested = _score_pd_pct_for_amount(model_name, base_raw, suggested)
    status = "approved_as_is" if abs(suggested - requested_amount) < float(SUGGESTION_LOAN_STEP) else "reduce_amount"
    return {
        "status": status,
        "target_pd_pct": round(target, 2),
        "requested_loan_amount": round(requested_amount, 2),
        "suggested_loan_amount": round(suggested, 2),
        "estimated_pd_pct": round(pd_suggested, 2),
    }


def _confidence_band(calibrated_prob: float, calibration: dict) -> list[float]:
    uncertainty = 1.0 - abs(float(calibrated_prob) - 0.5) * 2.0
    half_width = 0.04 + 0.08 * max(0.0, uncertainty)
    if not calibration.get("available"):
        half_width += 0.03
    lower = max(0.0, calibrated_prob - half_width)
    upper = min(1.0, calibrated_prob + half_width)
    return [round(lower * 100, 2), round(upper * 100, 2)]


def _predict_lr(X15: np.ndarray) -> tuple[float, int]:
    pkg = _load("logistic_regression")
    if pkg is None:
        raise RuntimeError("logistic_regression model not available")

    mean  = np.asarray(pkg["mean"])           # (15,)
    std   = np.asarray(pkg["std"])            # (15,)
    beta  = np.asarray(pkg["beta"]).flatten() # (16,)  [bias, w1..w15]
    thresh = float(pkg.get("threshold", 0.31))

    x_scaled = (X15 - mean) / std
    prob = _sigmoid(float(np.concatenate([[1.0], x_scaled]) @ beta))
    return prob, int(prob >= thresh)


def _predict_cb(X15: np.ndarray) -> tuple[float, int]:
    pkg = _load("catboost")
    if pkg is None:
        raise RuntimeError("catboost model not available")

    thresh = float(pkg.get("threshold", 0.5))
    X_in   = _preprocess(X15, pkg["scaler"])
    prob   = float(pkg["model"].predict_proba(X_in)[0, 1])
    return prob, int(prob >= thresh)

# Neural Network helpers  (pure NumPy — no framework dependency)

def _nn_relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def _nn_forward_logit(x: np.ndarray, params: dict) -> tuple[float, list]:
    """
    Forward pass through the MLP.
    Returns (logit, relu_masks) where relu_masks[i] is the activation
    mask for hidden layer i+1 (used during backprop for attribution).
    """
    L = len(params) // 2   # number of layers
    a = x.copy()
    masks: list[np.ndarray] = []

    for l in range(1, L + 1):
        W = params[f"W{l}"]
        b = params[f"b{l}"].flatten()
        z = W @ a + b
        if l < L:
            # Hidden layer — ReLU; store mask for backprop
            mask = (z > 0).astype(float)
            masks.append(mask)
            a = z * mask
        else:
            # Output layer — return raw logit (before sigmoid)
            logit = float(z[0])

    return logit, masks


def _nn_logit_gradient(x: np.ndarray, params: dict) -> np.ndarray:
    """
    Backpropagate to get ∂(logit) / ∂x  — used by Integrated Gradients.
    The gradient of the pre-sigmoid logit is exact; no sigmoid Jacobian needed.
    """
    L = len(params) // 2
    _, masks = _nn_forward_logit(x, params)

    # Seed gradient at the single output neuron
    grad = np.array([1.0])
    for l in range(L, 0, -1):
        W = params[f"W{l}"]
        grad = W.T @ grad
        if l > 1:
            # Apply ReLU mask of the previous hidden layer
            grad = grad * masks[l - 2]

    return grad  # shape (15,)


def _predict_nn(X15: np.ndarray) -> tuple[float, int]:
    """
    Run the NumPy MLP on a single 15-feature vector.
    Applies the same StandardScaler used during training.
    """
    pkg = _load("neural_network")
    if pkg is None:
        raise RuntimeError("neural_network model not available")

    thresh   = float(pkg.get("threshold", 0.5))
    x_scaled = pkg["scaler"].transform(X15.reshape(1, -1))[0]  # (15,)
    logit, _ = _nn_forward_logit(x_scaled, pkg["nn_params"])
    prob     = _sigmoid(logit)
    return prob, int(prob >= thresh)


def _predict_rf(X15: np.ndarray) -> tuple[float, int]:
    """
    Run a sklearn RandomForestClassifier on a single 15-feature vector.

    The saved file may be either:
      • a bare RandomForestClassifier  (most common), or
      • a dict with keys {"model": <RF>, "threshold": <float>}.

    Random Forests are scale-invariant so no scaler is applied unless the
    artifact explicitly includes one. Some saved artifacts include the same
    leading bias column used by CatBoost, so input shape is adapted to the
    model's fitted feature count.
    """
    pkg = _load("random_forest")
    if pkg is None:
        raise RuntimeError("random_forest model not available")

    # Unwrap dict wrapper if present
    if isinstance(pkg, dict):
        model  = pkg["model"]
        thresh = float(pkg.get("threshold", 0.5))
    else:
        model  = pkg          # bare RandomForestClassifier
        thresh = 0.5

    X_in = _rf_input(model, X15, pkg)
    prob = float(model.predict_proba(X_in)[0, 1])
    return prob, int(prob >= thresh)


def _rf_input(model: Any, X15: np.ndarray, pkg: Any | None = None) -> np.ndarray:
    """Return the Random Forest input matrix expected by the saved artifact."""
    n_features = int(getattr(model, "n_features_in_", len(X15)))
    if n_features == len(X15):
        return X15.reshape(1, -1)

    if n_features == len(X15) + 1:
        scaler = pkg.get("scaler") if isinstance(pkg, dict) else None
        if scaler is not None:
            return _preprocess(X15, scaler)
        return np.c_[np.ones((1, 1)), X15.reshape(1, -1)]

    raise ValueError(
        f"Random Forest expects {n_features} features, but encoded input has {len(X15)} "
        f"or {len(X15) + 1} with bias."
    )


_RUNNERS = {
    "logistic_regression": _predict_lr,
    "catboost": _predict_cb,
    "neural_network": _predict_nn,
    "random_forest": _predict_rf,
}


def _top_features_from_entries(entries: list[dict], *, method: str, unit: str, limit: int = 5) -> list[dict]:
    non_zero = [row for row in entries if abs(float(row.get("shap_value", 0.0))) > 1e-12]
    top = sorted(non_zero or entries, key=lambda row: abs(float(row.get("shap_value", 0.0))), reverse=True)[:limit]
    features = []
    for row in top:
        value = float(row.get("shap_value", 0.0))
        features.append({
            "feature": row.get("feature"),
            "display_name": row.get("display_name"),
            "direction": "up" if value > 0 else "down",
            "magnitude": round(abs(value), 6),
            "contribution": round(value, 6),
            "unit": unit,
            "method": method,
        })
    return features


def _linear_top_features(X15: np.ndarray) -> list[dict]:
    pkg = _load("logistic_regression")
    if pkg is None:
        raise RuntimeError("logistic_regression model not available")

    mean = np.asarray(pkg["mean"])
    std = np.asarray(pkg["std"])
    beta = np.asarray(pkg["beta"]).flatten()
    x_scaled = (X15 - mean) / std
    contributions = beta[1:] * x_scaled
    return _top_features_from_entries(
        _build_entries(X15, contributions),
        method="linear_coefficients",
        unit="log_odds",
    )


def _reference_vector(model_name: str, X15: np.ndarray) -> np.ndarray:
    pkg = _load(model_name)
    ref = X15.copy()
    if model_name == "logistic_regression" and pkg is not None:
        ref = np.asarray(pkg["mean"], dtype=np.float64).copy()
    elif isinstance(pkg, dict) and hasattr(pkg.get("scaler"), "mean_"):
        ref = np.asarray(pkg["scaler"].mean_, dtype=np.float64).copy()

    for idx, col in enumerate(FEATURE_COLS):
        if col in _DUMMY_GROUP or col == "cb_person_default_on_file_Y":
            ref[idx] = 0.0
    return ref


def _ablation_top_features(model_name: str, X15: np.ndarray, base_prob: float) -> list[dict]:
    runner = _RUNNERS[model_name]
    ref = _reference_vector(model_name, X15)
    contributions = np.zeros_like(X15, dtype=np.float64)

    for idx in range(len(X15)):
        if float(X15[idx]) == float(ref[idx]):
            continue
        ablated = X15.copy()
        ablated[idx] = ref[idx]
        ablated_prob, _ = runner(ablated)
        contributions[idx] = (float(base_prob) - float(ablated_prob)) * 100.0

    return _top_features_from_entries(
        _build_entries(X15, contributions),
        method="feature_ablation",
        unit="percentage_points",
    )


def _top_features(model_name: str, X15: np.ndarray, base_prob: float) -> list[dict]:
    if model_name == "logistic_regression":
        return _linear_top_features(X15)
    return _ablation_top_features(model_name, X15, base_prob)


# Public API
def predict(raw: dict, *, include_suggestions: bool = True) -> dict[str, dict | None]:
    """
    Run all available models on raw borrower features.
    Returns {model_name: {probability, prediction, label, grade, decision}}
    or {model_name: None} when a model file is absent.
    """
    try:
        loan_amount = float(raw.get("loan_amnt", 0.0))
    except (TypeError, ValueError):
        loan_amount = 0.0

    X15 = _encode(raw)

    results: dict[str, dict | None] = {}
    for name, runner in _RUNNERS.items():
        try:
            raw_prob, pred = runner(X15)
            calibrated_prob, calibration = _calibrate_probability(name, raw_prob)
            pd_pct = round(calibrated_prob * 100, 2)
            raw_pd_pct = round(raw_prob * 100, 2)
            confidence_band = _confidence_band(calibrated_prob, calibration)
            expected_loss = _expected_loss_metrics(
                pd_pct,
                loan_amount,
                confidence_band=confidence_band,
            )
            result_row = {
                "probability":              pd_pct,
                "raw_probability":          raw_pd_pct,
                "calibrated_probability":   pd_pct,
                "calibration":              calibration,
                "confidence_band":          confidence_band,
                "prediction":               pred,
                "label":                    "Default" if pred == 1 else "No Default",
                "grade":                    _risk_grade(pd_pct),
                "decision":                 _decision(pd_pct),
                "top_features":             _top_features(name, X15, raw_prob),
                "expected_loss":            expected_loss,
            }
            if include_suggestions:
                result_row["approval_suggestion"] = _suggest_loan_amount(
                    raw,
                    name,
                    target_pd_pct=SUGGESTION_TARGET_PD_PCT,
                    current_pd_pct=pd_pct,
                )
            results[name] = result_row
        except RuntimeError:
            results[name] = None
        except Exception as exc:
            results[name] = {"error": str(exc)}

    return results

# SHAP helpers

def _group_dummy_shap(entries: list[dict]) -> list[dict]:
    """
    Collapse one-hot dummy SHAP values into a single row per categorical feature.

    Example: person_home_ownership_OWN, _RENT, _OTHER all become one "Home: Own"
    row whose shap_value is the sum of all three dummy contributions.
    This prevents confusion where inactive dummies (e.g. Rent=0) appear to
    have a non-zero SHAP value because they were standardised against the
    training mean rather than zero.
    """
    acc: dict[str, dict] = {}  # group_key -> {shap_sum, active_label}
    result = []

    for e in entries:
        gkey = _DUMMY_GROUP.get(e["feature"])
        if gkey is None:
            # Not a grouped dummy — keep as-is
            result.append(e)
            continue
        if gkey not in acc:
            acc[gkey] = {"shap_sum": 0.0, "active_label": _GROUP_BASELINE_LABEL[gkey]}
        acc[gkey]["shap_sum"] += e["shap_value"]
        # Track which dummy is active (raw value = 1) to name the row correctly
        if e["raw_value"] == 1.0:
            acc[gkey]["active_label"] = e["display_name"]

    for gkey, g in acc.items():
        result.append({
            "feature":      gkey,
            "display_name": g["active_label"],
            "raw_value":    None,
            "shap_value":   round(g["shap_sum"], 6),
        })

    result.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return result


def _build_entries(X15: np.ndarray, shap_vals: np.ndarray) -> list[dict]:
    """Zip raw feature values with their SHAP contributions, then group dummies."""
    entries = [
        {
            "feature":      col,
            "display_name": FEATURE_LABELS.get(col, col),
            "raw_value":    round(float(X15[i]), 4),
            "shap_value":   round(float(shap_vals[i]), 6),
        }
        for i, col in enumerate(FEATURE_COLS)
    ]
    return _group_dummy_shap(entries)


def _explain_shap_catboost(raw: dict) -> dict:
    """
    CatBoost SHAP via shap.TreeExplainer.
    SHAP values are in log-odds space; base_value is the average log-odds
    output of the model across the training distribution.
    """
    pkg = _load("catboost")
    if pkg is None:
        raise RuntimeError("catboost model not available")

    X15  = _encode(raw)
    X_in = _preprocess(X15, pkg["scaler"])  # (1, 16)

    explainer = _get_cb_explainer()
    if explainer is None:
        raise RuntimeError("Could not create CatBoost SHAP explainer")

    sv = explainer.shap_values(X_in)

    # TreeExplainer may return a list (one array per class) or a 2-/3-D array.
    if isinstance(sv, list):
        sv = sv[1]              # class-1 = default
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    row_sv = sv[0]              # (16,)  — index 0 is the bias column, skip it

    ev = explainer.expected_value
    base_value = float(np.asarray(ev).flat[1]) if isinstance(ev, (list, np.ndarray)) else float(ev)

    return {
        "model":       "catboost",
        "base_value":  round(base_value, 6),
        "shap_values": _build_entries(X15, row_sv[1:]),  # drop bias index
    }


def _explain_shap_lr(raw: dict) -> dict:
    """
    Logistic-Regression SHAP via the closed-form linear formula.

    For f(x) = σ(β₀ + Σ βᵢ·x̃ᵢ),  φᵢ = βᵢ·x̃ᵢ  (log-odds contribution).
    base_value = β₀  (log-odds at the all-zero standardised reference point).
    Consistent with CatBoost: base + Σφᵢ = logit(predicted probability).
    """
    pkg = _load("logistic_regression")
    if pkg is None:
        raise RuntimeError("logistic_regression model not available")

    mean = np.asarray(pkg["mean"])           # (15,)
    std  = np.asarray(pkg["std"])            # (15,)
    beta = np.asarray(pkg["beta"]).flatten() # (16,)  [β₀, β₁..β₁₅]

    X15       = _encode(raw)
    x_scaled  = (X15 - mean) / std           # (15,)  standardised features
    shap_vals = beta[1:] * x_scaled          # (15,)  log-odds contributions
    base_value = float(beta[0])              # log-odds bias (same space as CatBoost)

    return {
        "model":       "logistic_regression",
        "base_value":  round(base_value, 6),
        "shap_values": _build_entries(X15, shap_vals),
    }


def _explain_shap_nn(raw: dict) -> dict:
    """
    Neural-Network feature attribution via Integrated Gradients.

    φᵢ = (xᵢ − baseᵢ) × (1/N) × Σₖ ∂logit(x_base + k/N·Δx) / ∂xᵢ

    Baseline: all-zeros in scaled space (the scaler's reference point).
    Completeness: Σφᵢ ≈ logit(p) − logit(p_base)  (exact as N → ∞).
    Values are in log-odds space, identical to the CatBoost/LR SHAP charts.
    """
    pkg = _load("neural_network")
    if pkg is None:
        raise RuntimeError("neural_network model not available")

    params = pkg["nn_params"]
    scaler = pkg["scaler"]

    X15      = _encode(raw)
    x_scaled = scaler.transform(X15.reshape(1, -1))[0]   # (15,) scaled input
    baseline = np.zeros_like(x_scaled)                    # all-zeros baseline
    delta    = x_scaled - baseline                        # interpolation step

    # Accumulate gradients at N equally-spaced points along the straight-line path
    N_STEPS      = 30
    accumulated  = np.zeros_like(x_scaled)
    for k in range(N_STEPS):
        alpha       = k / max(N_STEPS - 1, 1)
        x_interp    = baseline + alpha * delta
        accumulated += _nn_logit_gradient(x_interp, params)

    # Attribution = step × mean gradient
    shap_vals = delta * (accumulated / N_STEPS)   # (15,)  log-odds contributions

    # Base value = logit at the all-zeros scaled input
    base_logit, _ = _nn_forward_logit(baseline, params)

    return {
        "model":       "neural_network",
        "base_value":  round(float(base_logit), 6),
        "shap_values": _build_entries(X15, shap_vals),
    }


def _explain_shap_rf(raw: dict) -> dict:
    """
    Random-Forest feature attribution via feature ablation.

    For each feature i, the contribution is estimated as:
        φᵢ = (P(default | x) − P(default | x with feature i set to reference)) × 100 pp

    The reference vector uses the training-data mean for numeric features and
    0 for all one-hot dummies (equivalent to the MORTGAGE / DEBTCONSOLIDATION baseline).
    Values are in percentage-point space, consistent with the NN ablation output.
    """
    pkg = _load("random_forest")
    if pkg is None:
        raise RuntimeError("random_forest model not available")

    model = pkg["model"] if isinstance(pkg, dict) else pkg

    X15      = _encode(raw)
    base_prob, _ = _predict_rf(X15)

    ref = X15.copy()
    for idx, col in enumerate(FEATURE_COLS):
        if col in _DUMMY_GROUP or col == "cb_person_default_on_file_Y":
            ref[idx] = 0.0

    contributions = np.zeros_like(X15, dtype=np.float64)
    for idx in range(len(X15)):
        if float(X15[idx]) == float(ref[idx]):
            continue
        ablated          = X15.copy()
        ablated[idx]     = ref[idx]
        ablated_prob     = float(model.predict_proba(_rf_input(model, ablated, pkg))[0, 1])
        contributions[idx] = (base_prob - ablated_prob) * 100.0

    return {
        "model":       "random_forest",
        "base_value":  round(base_prob * 100.0, 4),   # baseline PD in pp
        "shap_values": _build_entries(X15, contributions),
    }


def explain_shap(raw: dict, model: str = "catboost") -> dict:
    """
    Return SHAP feature attributions for the given model.

    Args:
        raw:   borrower feature dict (same schema as predict())
        model: "catboost", "logistic_regression", "neural_network", or "random_forest"

    Returns:
        {model, base_value, shap_values: [{feature, display_name, raw_value, shap_value}]}
    """
    try:
        import shap as _shap  # noqa: F401  (just to verify installation)
    except ImportError:
        raise RuntimeError("shap package not installed — run: pip install shap")

    if model == "catboost":
        return _explain_shap_catboost(raw)
    if model == "logistic_regression":
        return _explain_shap_lr(raw)
    if model == "neural_network":
        return _explain_shap_nn(raw)
    if model == "random_forest":
        return _explain_shap_rf(raw)
    raise ValueError(
        f"Unknown model {model!r}. "
        "Choose 'catboost', 'logistic_regression', 'neural_network', or 'random_forest'."
    )
