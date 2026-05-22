"""
Central feature preprocessing contract for credit-risk tabular models.

This module is intentionally small and dependency-light so the same contract can
be used by the API gateway, a standalone scoring service, batch jobs, and future
training pipelines.
"""
from __future__ import annotations

from typing import Any

import numpy as np


FEATURE_CONTRACT_VERSION = "credit-risk-v1"

NUMERIC_FIELDS: dict[str, tuple[float | None, float | None]] = {
    "person_age": (18, 100),
    "person_income": (0, None),
    "person_emp_length": (0, 60),
    "loan_amnt": (500, None),
    "loan_int_rate": (1, 40),
    "cb_person_cred_hist_length": (0, 60),
}

CATEGORICAL_FIELDS: dict[str, tuple[str, ...]] = {
    "person_home_ownership": ("MORTGAGE", "OWN", "RENT", "OTHER"),
    "loan_intent": (
        "DEBTCONSOLIDATION",
        "EDUCATION",
        "HOMEIMPROVEMENT",
        "MEDICAL",
        "PERSONAL",
        "VENTURE",
    ),
    "cb_person_default_on_file": ("Y", "N"),
}

FEATURE_COLS: list[str] = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "cb_person_cred_hist_length",
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "person_home_ownership_RENT",
    "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE",
    "cb_person_default_on_file_Y",
]

FEATURE_LABELS: dict[str, str] = {
    "person_age": "Age",
    "person_income": "Annual Income",
    "person_emp_length": "Employment Length",
    "loan_amnt": "Loan Amount",
    "loan_int_rate": "Interest Rate",
    "cb_person_cred_hist_length": "Credit History Length",
    "person_home_ownership_OTHER": "Home: Other",
    "person_home_ownership_OWN": "Home: Own",
    "person_home_ownership_RENT": "Home: Rent",
    "loan_intent_EDUCATION": "Purpose: Education",
    "loan_intent_HOMEIMPROVEMENT": "Purpose: Home Impr.",
    "loan_intent_MEDICAL": "Purpose: Medical",
    "loan_intent_PERSONAL": "Purpose: Personal",
    "loan_intent_VENTURE": "Purpose: Venture",
    "cb_person_default_on_file_Y": "Prior Default",
}


def _coerce_float(raw: dict[str, Any], field: str, bounds: tuple[float | None, float | None]) -> float:
    if field not in raw:
        raise ValueError(f"Missing required feature: {field}")
    try:
        value = float(raw[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature {field} must be numeric.") from exc

    lower, upper = bounds
    if lower is not None and value < lower:
        raise ValueError(f"Feature {field} must be >= {lower}.")
    if upper is not None and value > upper:
        raise ValueError(f"Feature {field} must be <= {upper}.")
    return value


def _coerce_category(raw: dict[str, Any], field: str, allowed: tuple[str, ...]) -> str:
    if field not in raw:
        raise ValueError(f"Missing required feature: {field}")
    value = str(raw[field]).strip().upper()
    if value not in allowed:
        raise ValueError(f"Feature {field} must be one of: {', '.join(allowed)}.")
    return value


def standardize_applicant(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize borrower features before model inference."""
    applicant: dict[str, Any] = {}
    for field, bounds in NUMERIC_FIELDS.items():
        applicant[field] = _coerce_float(raw, field, bounds)
    for field, allowed in CATEGORICAL_FIELDS.items():
        applicant[field] = _coerce_category(raw, field, allowed)
    return applicant


def encode_applicant(raw: dict[str, Any]) -> np.ndarray:
    """Encode raw borrower features into the 15-column model feature vector."""
    applicant = standardize_applicant(raw)
    home = applicant["person_home_ownership"]
    intent = applicant["loan_intent"]
    cb_def = applicant["cb_person_default_on_file"]

    row: dict[str, float] = {
        "person_age": applicant["person_age"],
        "person_income": applicant["person_income"],
        "person_emp_length": applicant["person_emp_length"],
        "loan_amnt": applicant["loan_amnt"],
        "loan_int_rate": applicant["loan_int_rate"],
        "cb_person_cred_hist_length": applicant["cb_person_cred_hist_length"],
        "person_home_ownership_OTHER": 1.0 if home == "OTHER" else 0.0,
        "person_home_ownership_OWN": 1.0 if home == "OWN" else 0.0,
        "person_home_ownership_RENT": 1.0 if home == "RENT" else 0.0,
        "loan_intent_EDUCATION": 1.0 if intent == "EDUCATION" else 0.0,
        "loan_intent_HOMEIMPROVEMENT": 1.0 if intent == "HOMEIMPROVEMENT" else 0.0,
        "loan_intent_MEDICAL": 1.0 if intent == "MEDICAL" else 0.0,
        "loan_intent_PERSONAL": 1.0 if intent == "PERSONAL" else 0.0,
        "loan_intent_VENTURE": 1.0 if intent == "VENTURE" else 0.0,
        "cb_person_default_on_file_Y": 1.0 if cb_def == "Y" else 0.0,
    }
    return np.array([row[col] for col in FEATURE_COLS], dtype=np.float64)


def derived_features(raw: dict[str, Any]) -> dict[str, float | None]:
    """Return shared derived features for dashboards, jobs, and services."""
    applicant = standardize_applicant(raw)
    income = float(applicant["person_income"])
    loan_amount = float(applicant["loan_amnt"])
    loan_to_income_pct = None
    if income > 0:
        loan_to_income_pct = round((loan_amount / income) * 100, 2)
    return {"loan_to_income_pct": loan_to_income_pct}


def feature_contract() -> dict[str, Any]:
    """Describe the preprocessing contract used by training and inference."""
    return {
        "version": FEATURE_CONTRACT_VERSION,
        "numeric_fields": {
            field: {"min": bounds[0], "max": bounds[1]}
            for field, bounds in NUMERIC_FIELDS.items()
        },
        "categorical_fields": {
            field: list(values)
            for field, values in CATEGORICAL_FIELDS.items()
        },
        "feature_columns": FEATURE_COLS,
        "feature_labels": FEATURE_LABELS,
        "derived_features": ["loan_to_income_pct"],
    }
