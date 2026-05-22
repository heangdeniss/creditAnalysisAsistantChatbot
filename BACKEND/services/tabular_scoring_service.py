"""Standalone tabular credit-risk scoring service.

Run:
  uvicorn services.tabular_scoring_service:app --host 0.0.0.0 --port 8013
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from feature_store import derived_features, encode_applicant, feature_contract, standardize_applicant
from ml_predict import load_all_models, predict


SKIP_STARTUP_LOAD = os.getenv("SKIP_STARTUP_LOAD", "").lower() in {"1", "true", "yes"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not SKIP_STARTUP_LOAD:
        load_all_models()
    yield


app = FastAPI(title="Credit Risk Tabular Scoring Service", version="1.0.0", lifespan=lifespan)


class ApplicantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_age: float = Field(..., ge=18, le=100)
    person_income: float = Field(..., ge=0)
    person_home_ownership: str = Field(..., pattern="^(MORTGAGE|OWN|RENT|OTHER)$")
    person_emp_length: float = Field(..., ge=0, le=60)
    loan_intent: str = Field(..., pattern="^(DEBTCONSOLIDATION|EDUCATION|HOMEIMPROVEMENT|MEDICAL|PERSONAL|VENTURE)$")
    loan_amnt: float = Field(..., ge=500)
    loan_int_rate: float = Field(..., ge=1, le=40)
    cb_person_default_on_file: str = Field(..., pattern="^(Y|N)$")
    cb_person_cred_hist_length: float = Field(..., ge=0, le=60)


class BatchScoreRequest(BaseModel):
    applicants: list[ApplicantRequest] = Field(..., min_length=1, max_length=1000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "feature_contract": feature_contract()["version"]}


@app.get("/features/contract")
def get_feature_contract() -> dict:
    return feature_contract()


@app.post("/features/transform")
def transform(body: ApplicantRequest) -> dict:
    payload = body.model_dump()
    return {
        "standardized": standardize_applicant(payload),
        "derived_metrics": derived_features(payload),
        "feature_columns": feature_contract()["feature_columns"],
        "feature_vector": [float(value) for value in encode_applicant(payload)],
    }


@app.post("/predict")
def score(body: ApplicantRequest) -> dict:
    payload = body.model_dump()
    return {
        "input": standardize_applicant(payload),
        "derived_metrics": derived_features(payload),
        "scores": predict(payload),
    }


@app.post("/batch-score")
def batch_score(body: BatchScoreRequest) -> dict:
    rows = []
    for idx, applicant in enumerate(body.applicants, start=1):
        payload = applicant.model_dump()
        rows.append({
            "row": idx,
            "input": standardize_applicant(payload),
            "derived_metrics": derived_features(payload),
            "scores": predict(payload),
        })
    return {"count": len(rows), "rows": rows}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.tabular_scoring_service:app", host="0.0.0.0", port=8013, reload=False)
