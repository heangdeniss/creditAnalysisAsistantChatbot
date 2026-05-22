"""
app.py
FastAPI backend — routes: GET /health, POST /query, POST /stream,
                           POST /predict, POST /scenario, POST /ingest.
Generation settings (temperature, top_p, max_new_tokens) are fixed in
model_loader.py and are NOT accepted or forwarded from the frontend.

Run with:  python app.py
Docs at:   http://localhost:8000/docs
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from copy import deepcopy
from statistics import median
from time import perf_counter

os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"]                 = "0"
os.environ["USE_TORCH"]              = "1"
warnings.filterwarnings("ignore")
for _n in ("transformers", "sentence_transformers", "tensorflow", "tf_keras"):
    logging.getLogger(_n).setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(__file__))

from typing import Annotated
from contextlib import asynccontextmanager
from threading import Event
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rag_pipeline import (
    get_retriever,
    invalidate_retrieval_indexes,
    query_refusal_reason,
    rag_query,
    rag_stream,
    retrieval_settings,
    retriever_is_loaded,
)
from model_loader import DEFAULT_MODEL, load_model, model_is_loaded, model_is_busy, current_model_name, switch_model
from model_server_client import external_generation_enabled
from ml_predict import load_all_models, predict as ml_predict, explain_shap
from chroma_loader import chunk_texts, embed_query, load_chroma_db
from feature_store import derived_features, encode_applicant, feature_contract, standardize_applicant
from job_queue import get_job, list_jobs, queue_stats, submit_job
from service_clients import remote_batch_score, remote_predict, remote_tabular_enabled
from speech_to_text import transcribe_wav_bytes
from observability import (
    add_event,
    finish_trace,
    inc_counter,
    metrics_summary,
    new_request_id,
    recent_traces,
    start_trace,
)
from runtime_utils import SemanticTTLCache, SlidingWindowRateLimiter, TTLCache, stable_json
from eval.rag_eval import run_eval as run_rag_eval


# Config
def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


INGEST_API_KEY = os.getenv("INGEST_API_KEY", "")   # set to require auth on /ingest
SKIP_STARTUP_LOAD = os.getenv("SKIP_STARTUP_LOAD", "").lower() in {"1", "true", "yes"}

RAG_CACHE_TTL_S = int(os.getenv("RAG_CACHE_TTL_S", "300"))
RAG_CACHE_SIZE = int(os.getenv("RAG_CACHE_SIZE", "256"))
RAG_SEMANTIC_CACHE_ENABLED = _env_bool("RAG_SEMANTIC_CACHE_ENABLED", True)
RAG_SEMANTIC_CACHE_TTL_S = int(os.getenv("RAG_SEMANTIC_CACHE_TTL_S", str(RAG_CACHE_TTL_S)))
RAG_SEMANTIC_CACHE_SIZE = int(os.getenv("RAG_SEMANTIC_CACHE_SIZE", "128"))
RAG_SEMANTIC_CACHE_THRESHOLD = float(os.getenv("RAG_SEMANTIC_CACHE_THRESHOLD", "0.93"))
PREDICT_CACHE_TTL_S = int(os.getenv("PREDICT_CACHE_TTL_S", "300"))
PREDICT_CACHE_SIZE = int(os.getenv("PREDICT_CACHE_SIZE", "256"))

RAG_TIMEOUT_S = float(os.getenv("RAG_TIMEOUT_S", "60"))
PREDICT_TIMEOUT_S = float(os.getenv("PREDICT_TIMEOUT_S", "20"))
RAG_RETRIES = int(os.getenv("RAG_RETRIES", "1"))
PREDICT_RETRIES = int(os.getenv("PREDICT_RETRIES", "1"))
STREAM_TIMEOUT_S = float(os.getenv("STREAM_TIMEOUT_S", "180"))

RATE_LIMIT_WINDOW_S = int(os.getenv("RATE_LIMIT_WINDOW_S", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "90"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    if SKIP_STARTUP_LOAD:
        print("[startup] SKIP_STARTUP_LOAD=1 — skipping heavy model loads")
        yield
        return
    print("[startup] loading embeddings + retriever …")
    get_retriever()
    if external_generation_enabled():
        print("[startup] using external LLM model server")
    else:
        print(f"[startup] loading {DEFAULT_MODEL} …")
        load_model(DEFAULT_MODEL)
    print("[startup] loading ML models …")
    load_all_models()
    print("[startup] ✅ ready")
    yield


app = FastAPI(title="Credit Risk RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:3000",  "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("APP_WORKER_POOL", "4")))
RAG_CACHE = TTLCache(max_size=RAG_CACHE_SIZE, ttl_s=RAG_CACHE_TTL_S)
RAG_SEMANTIC_CACHE = SemanticTTLCache(
    max_size=RAG_SEMANTIC_CACHE_SIZE,
    ttl_s=RAG_SEMANTIC_CACHE_TTL_S,
)
PREDICT_CACHE = TTLCache(max_size=PREDICT_CACHE_SIZE, ttl_s=PREDICT_CACHE_TTL_S)
RATE_LIMITER = SlidingWindowRateLimiter(
    max_requests=RATE_LIMIT_MAX,
    window_s=RATE_LIMIT_WINDOW_S,
)
_RATE_LIMIT_EXEMPT = {"/health", "/metrics/summary", "/traces/recent"}


def _cache_key(prefix: str, payload: dict) -> str:
    return f"{prefix}:{stable_json(payload)}"


def _semantic_cache_namespace(payload: dict) -> str:
    return stable_json({
        "facts": payload.get("facts") or "",
        "history": payload.get("history") or [],
        "model": payload.get("model") or "",
    })


def _semantic_cache_allowed(payload: dict) -> bool:
    if not RAG_SEMANTIC_CACHE_ENABLED or RAG_SEMANTIC_CACHE_TTL_S <= 0:
        return False
    if query_refusal_reason(str(payload.get("question") or "")):
        return False
    return not payload.get("facts") and not payload.get("history")


def _clone_cached_rag_result(result: dict, question: str) -> dict:
    cloned = deepcopy(result)
    if isinstance(cloned, dict):
        cloned["question"] = question
    return cloned


def _run_with_timeout(fn, *, timeout_s: float, trace: dict | None, label: str):
    future = _EXECUTOR.submit(fn)
    try:
        return future.result(timeout=max(0.0, float(timeout_s)))
    except FuturesTimeoutError as exc:
        inc_counter(f"timeout_{label}")
        add_event(trace, f"{label}_timeout", {"timeout_s": timeout_s})
        raise TimeoutError(f"{label} timed out after {timeout_s}s") from exc


def _run_with_retries(fn, *, timeout_s: float, retries: int, trace: dict | None, label: str):
    last_exc: Exception | None = None
    for attempt in range(max(0, int(retries)) + 1):
        try:
            return _run_with_timeout(fn, timeout_s=timeout_s, trace=trace, label=label)
        except TimeoutError as exc:
            last_exc = exc
            if attempt < retries:
                inc_counter(f"retry_{label}")
                continue
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                inc_counter(f"retry_{label}")
                continue
            break
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label} failed without exception")


def _rag_cached(*, payload: dict, trace: dict | None, call_fn):
    cache_key = _cache_key("rag", payload)
    if RAG_CACHE_TTL_S > 0:
        cached = RAG_CACHE.get(cache_key)
        if cached is not None:
            inc_counter("rag_cache_hit")
            add_event(trace, "rag_cache_hit", {"cache_key": cache_key})
            return cached
    inc_counter("rag_cache_miss")

    semantic_embedding: list[float] | None = None
    semantic_namespace = ""
    if _semantic_cache_allowed(payload):
        semantic_namespace = _semantic_cache_namespace(payload)
        try:
            semantic_embedding = embed_query(str(payload.get("question") or ""))
            cached, similarity = RAG_SEMANTIC_CACHE.get_similar(
                namespace=semantic_namespace,
                embedding=semantic_embedding,
                min_similarity=RAG_SEMANTIC_CACHE_THRESHOLD,
            )
            if cached is not None:
                inc_counter("rag_semantic_cache_hit")
                add_event(trace, "rag_semantic_cache_hit", {"similarity": similarity})
                return _clone_cached_rag_result(cached, str(payload.get("question") or ""))
        except Exception as exc:
            add_event(trace, "rag_semantic_cache_unavailable", {"error": str(exc)})
    else:
        inc_counter("rag_semantic_cache_skip")

    result = _run_with_retries(
        call_fn,
        timeout_s=RAG_TIMEOUT_S,
        retries=RAG_RETRIES,
        trace=trace,
        label="rag",
    )
    if RAG_CACHE_TTL_S > 0:
        RAG_CACHE.set(cache_key, result)
    if semantic_embedding is not None and semantic_namespace:
        RAG_SEMANTIC_CACHE.set(
            key=cache_key,
            namespace=semantic_namespace,
            embedding=semantic_embedding,
            value=result,
        )
    return result


def _predict_cached(*, payload: dict, trace: dict | None, label: str):
    cache_key = _cache_key("predict", payload)
    if PREDICT_CACHE_TTL_S > 0:
        cached = PREDICT_CACHE.get(cache_key)
        if cached is not None:
            inc_counter("predict_cache_hit")
            add_event(trace, "predict_cache_hit", {"cache_key": cache_key, "label": label})
            return cached
    inc_counter("predict_cache_miss")
    predict_fn = remote_predict if remote_tabular_enabled() else ml_predict
    result = _run_with_retries(
        lambda: predict_fn(payload),
        timeout_s=PREDICT_TIMEOUT_S,
        retries=PREDICT_RETRIES,
        trace=trace,
        label=label,
    )
    if PREDICT_CACHE_TTL_S > 0:
        PREDICT_CACHE.set(cache_key, result)
    return result


@app.middleware("http")
async def rate_limit_and_log(request: Request, call_next):
    req_id = new_request_id()
    started = perf_counter()
    client_ip = request.client.host if request.client else "unknown"
    status_code = 500
    error: str | None = None
    response = None

    if request.url.path not in _RATE_LIMIT_EXEMPT:
        allowed, _ = RATE_LIMITER.allow(client_ip)
        if not allowed:
            inc_counter("rate_limit_block")
            status_code = 429
            response = JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again soon."},
            )
            response.headers["X-Request-Id"] = req_id
            return response

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-Id"] = req_id
        return response
    except Exception as exc:
        error = str(exc)
        status_code = 500
        raise
    finally:
        latency_ms = round((perf_counter() - started) * 1000, 2)
        inc_counter("http_requests_total")
        if status_code >= 400:
            inc_counter("http_requests_error")
        log = {
            "type": "request",
            "request_id": req_id,
            "method": request.method,
            "path": request.url.path,
            "status": status_code,
            "latency_ms": latency_ms,
            "client_ip": client_ip,
        }
        if error:
            log["error"] = error
        try:
            print(json.dumps(log, ensure_ascii=True))
        except Exception:
            pass


# Schemas
class HistoryMessage(BaseModel):
    role:    str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4096)


class QueryRequest(BaseModel):
    question: str                  = Field(..., min_length=1, max_length=2000)
    facts:    str                  = Field(default="", max_length=4096)
    model:    str                  = Field(default="llama-1b", pattern="^(llama-1b|llama-3b)$")
    history:  list[HistoryMessage] = Field(default_factory=list)  # prior turns for memory


class ModelSwitchRequest(BaseModel):
    model: str = Field(..., pattern="^(llama-1b|llama-3b)$")


class RetrievedChunk(BaseModel):
    citation_id: str
    document_id: str | None = None
    text:        str
    snippet:     str
    source:      str
    page:        str | int | None = None
    title:       str | None = None
    chunk_id:    str
    score:       float | None = None


class QueryResponse(BaseModel):
    question: str
    answer:   str
    context:  list[RetrievedChunk]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_age:                 float = Field(..., ge=18,  le=100)
    person_income:              float = Field(..., ge=0)
    person_home_ownership:      str   = Field(..., pattern="^(MORTGAGE|OWN|RENT|OTHER)$")
    person_emp_length:          float = Field(..., ge=0,   le=60)
    loan_intent:                str   = Field(..., pattern="^(DEBTCONSOLIDATION|EDUCATION|HOMEIMPROVEMENT|MEDICAL|PERSONAL|VENTURE)$")
    loan_amnt:                  float = Field(..., ge=500)
    loan_int_rate:              float = Field(..., ge=1,   le=40)
    cb_person_default_on_file:  str   = Field(..., pattern="^(Y|N)$")
    cb_person_cred_hist_length: float = Field(..., ge=0,   le=60)


class ScenarioOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str | None = Field(default=None, min_length=1, max_length=80)
    name:        str        = Field(..., min_length=1, max_length=120)

    person_age:                 float | None = Field(default=None, ge=18,  le=100)
    person_income:              float | None = Field(default=None, ge=0)
    person_home_ownership:      str | None   = Field(default=None, pattern="^(MORTGAGE|OWN|RENT|OTHER)$")
    person_emp_length:          float | None = Field(default=None, ge=0,   le=60)
    loan_intent:                str | None   = Field(default=None, pattern="^(DEBTCONSOLIDATION|EDUCATION|HOMEIMPROVEMENT|MEDICAL|PERSONAL|VENTURE)$")
    loan_amnt:                  float | None = Field(default=None, ge=500)
    loan_int_rate:              float | None = Field(default=None, ge=1,   le=40)
    cb_person_default_on_file:  str | None   = Field(default=None, pattern="^(Y|N)$")
    cb_person_cred_hist_length: float | None = Field(default=None, ge=0,   le=60)

    @model_validator(mode="after")
    def require_at_least_one_change(self) -> "ScenarioOverride":
        changed = self.model_dump(
            exclude={"scenario_id", "name"},
            exclude_none=True,
        )
        if not changed:
            raise ValueError("Scenario must override at least one applicant field.")
        return self


class ScenarioSimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_applicant: PredictRequest
    scenarios: list[ScenarioOverride] = Field(..., min_length=1, max_length=20)
    include_top_drivers: bool = Field(default=False)
    drivers_model: str = Field(
        default="catboost",
        pattern="^(catboost|logistic_regression|neural_network)$",
    )


class IngestRequest(BaseModel):
    texts:  list[Annotated[str, Field(min_length=1, max_length=8192)]] = Field(
        ..., min_length=1, max_length=50,
        description="Text passages to add (max 50 items, each \u2264 8\u202f192 chars).",
    )
    source: str = Field(default="api", description="Metadata label stored alongside each passage.")


class RagEvalRequest(BaseModel):
    model:          str        = Field(default="llama-1b", pattern="^(llama-1b|llama-3b)$")
    retrieval_only: bool       = Field(default=True)
    cases_path:     str | None = Field(default=None, description="Optional JSONL cases path on the backend host.")
    top_k:          int | None  = Field(default=None, ge=1, le=25)
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    ablation:       bool       = Field(default=False, description="Compare default retrieval settings against a broader retrieval set.")


class BatchScoreRequest(BaseModel):
    applicants: list[PredictRequest] = Field(..., min_length=1, max_length=1000)


def _derived_applicant_metrics(applicant: dict) -> dict:
    """Return local derived metrics that are useful for scenario comparison."""
    return derived_features(applicant)


def _scenario_changes(baseline_input: dict, overrides: dict) -> list[dict]:
    """Return a compact list of changed fields for a scenario."""
    changes = []
    for field, new_value in overrides.items():
        old_value = baseline_input.get(field)
        if old_value != new_value:
            changes.append({
                "field": field,
                "from": old_value,
                "to": new_value,
            })
    return changes


def _prediction_delta(baseline: dict | None, scenario: dict | None) -> dict | None:
    """Compare two model prediction rows returned by ml_predict.predict()."""
    if not baseline or not scenario:
        return None
    if baseline.get("error") or scenario.get("error"):
        return {
            "status": "error",
            "baseline_error": baseline.get("error"),
            "scenario_error": scenario.get("error"),
        }
    if "probability" not in baseline or "probability" not in scenario:
        return None

    baseline_probability = float(baseline["probability"])
    scenario_probability = float(scenario["probability"])
    probability_delta = round(scenario_probability - baseline_probability, 2)
    relative_delta_pct = None
    if baseline_probability != 0:
        relative_delta_pct = round((probability_delta / baseline_probability) * 100, 2)

    direction = "increased" if probability_delta > 0 else "decreased" if probability_delta < 0 else "held steady"
    decision_changed = baseline.get("decision") != scenario.get("decision")
    grade_changed = baseline.get("grade") != scenario.get("grade")
    label_changed = baseline.get("label") != scenario.get("label")
    decision_text = (
        f"Decision changed to {scenario.get('decision')} (was {baseline.get('decision')})."
        if decision_changed
        else "Decision unchanged."
    )
    impact_summary = (
        f"PD {direction} by {abs(probability_delta):.2f} pts; {decision_text}"
        if probability_delta != 0
        else f"PD unchanged at {baseline_probability:.2f}%; {decision_text}"
    )

    return {
        "status": "ok",
        "probability_delta": probability_delta,
        "relative_probability_delta_pct": relative_delta_pct,
        "decision_changed": decision_changed,
        "grade_changed": grade_changed,
        "label_changed": label_changed,
        "impact_summary": impact_summary,
    }


def _score_deltas(baseline_scores: dict, scenario_scores: dict) -> dict:
    """Return per-model deltas for all models present in either score dict."""
    model_names = sorted(set(baseline_scores) | set(scenario_scores))
    return {
        model_name: _prediction_delta(
            baseline_scores.get(model_name),
            scenario_scores.get(model_name),
        )
        for model_name in model_names
    }


def _top_driver_summary(applicant: dict, model_name: str, limit: int = 5) -> dict:
    """Best-effort local SHAP summary; failures are reported without blocking scoring."""
    try:
        explanation = explain_shap(applicant, model_name)
        drivers = []
        for row in explanation.get("shap_values", [])[:limit]:
            shap_value = float(row.get("shap_value", 0.0))
            drivers.append({
                "feature": row.get("feature"),
                "display_name": row.get("display_name"),
                "raw_value": row.get("raw_value"),
                "shap_value": round(shap_value, 6),
                "direction": "risk_up" if shap_value > 0 else "risk_down",
            })
        return {
            "model": model_name,
            "drivers": drivers,
            "error": None,
        }
    except Exception as exc:
        return {
            "model": model_name,
            "drivers": [],
            "error": str(exc),
        }


def _ingest_documents(body: IngestRequest) -> dict:
    db = load_chroma_db()
    chunks, metadatas = chunk_texts(body.texts, source=body.source)
    if not chunks:
        raise ValueError("No non-empty text chunks to ingest.")
    ids = db.add_texts(texts=chunks, metadatas=metadatas)
    invalidate_retrieval_indexes()
    return {
        "added": len(ids),
        "ids": ids,
        "chunk_size": retrieval_settings()["chunk_size"],
        "chunk_overlap": retrieval_settings()["chunk_overlap"],
    }


def _eval_kwargs(body: RagEvalRequest) -> dict:
    kwargs = {
        "model_name": body.model,
        "retrieval_only": body.retrieval_only,
        "ablation": body.ablation,
    }
    if body.top_k is not None:
        kwargs["top_k"] = body.top_k
    if body.score_threshold is not None:
        kwargs["score_threshold"] = body.score_threshold
    if body.cases_path:
        kwargs["cases_path"] = body.cases_path
    return kwargs


def _batch_score(body: BatchScoreRequest) -> dict:
    if remote_tabular_enabled():
        return remote_batch_score([applicant.model_dump() for applicant in body.applicants])

    rows = []
    for idx, applicant in enumerate(body.applicants, start=1):
        payload = applicant.model_dump()
        rows.append({
            "row": idx,
            "input": standardize_applicant(payload),
            "derived_metrics": derived_features(payload),
            "scores": ml_predict(payload),
        })
    return {"count": len(rows), "rows": rows}


# Routes
@app.get("/health")
def health() -> dict:
    if not model_is_loaded():
        raise HTTPException(status_code=503, detail="LLM not loaded yet.")
    if not retriever_is_loaded():
        raise HTTPException(status_code=503, detail="Retriever not loaded yet.")
    return {"status": "ok", "busy": model_is_busy(), "model": current_model_name()}


@app.get("/metrics/summary")
def metrics() -> dict:
    """Return lightweight in-process observability metrics for recent requests."""
    summary = metrics_summary()
    summary["cache"] = {
        "rag": RAG_CACHE.stats(),
        "rag_semantic": {
            **RAG_SEMANTIC_CACHE.stats(),
            "enabled": RAG_SEMANTIC_CACHE_ENABLED,
            "threshold": RAG_SEMANTIC_CACHE_THRESHOLD,
        },
        "predict": PREDICT_CACHE.stats(),
    }
    summary["rate_limit"] = {
        "window_s": RATE_LIMITER.window_s,
        "max_requests": RATE_LIMITER.max_requests,
    }
    summary["queue"] = queue_stats()
    summary["services"] = {
        "api_gateway": "in-process",
        "retrieval": os.getenv("RETRIEVAL_SERVICE_URL", "local"),
        "inference": os.getenv("LLM_MODEL_SERVER_URL", "local"),
        "tabular_scoring": os.getenv("TABULAR_SCORING_SERVICE_URL", "local"),
    }
    return summary


@app.get("/traces/recent")
def traces(limit: int = Query(default=25, ge=1, le=250)) -> dict:
    """Return recent completed request traces."""
    return {"traces": recent_traces(limit)}


@app.get("/features/contract")
def get_feature_contract() -> dict:
    """Return the shared tabular feature preprocessing contract."""
    return feature_contract()


@app.post("/features/transform")
def transform_features(body: PredictRequest) -> dict:
    """Return standardized applicant fields, derived metrics, and model vector."""
    payload = body.model_dump()
    vector = encode_applicant(payload)
    return {
        "contract_version": feature_contract()["version"],
        "standardized": standardize_applicant(payload),
        "derived_metrics": derived_features(payload),
        "feature_columns": feature_contract()["feature_columns"],
        "feature_vector": [float(value) for value in vector],
    }


@app.get("/jobs")
def jobs(limit: int = Query(default=50, ge=1, le=250)) -> dict:
    """List recent background jobs."""
    return {"jobs": list_jobs(limit)}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    """Return a background job by id."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.post("/jobs/ingest", status_code=202)
def enqueue_ingest(body: IngestRequest, request: Request) -> dict:
    """Queue document ingestion work."""
    if INGEST_API_KEY and request.headers.get("X-API-Key", "") != INGEST_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key header.")
    return submit_job(
        "ingest",
        lambda: _ingest_documents(body),
        payload={"source": body.source, "text_count": len(body.texts)},
    )


@app.post("/jobs/eval/rag", status_code=202)
def enqueue_rag_eval(body: RagEvalRequest) -> dict:
    """Queue a RAG evaluation run."""
    kwargs = _eval_kwargs(body)
    return submit_job("rag_eval", lambda: run_rag_eval(**kwargs), payload=kwargs)


@app.post("/jobs/batch-score", status_code=202)
def enqueue_batch_score(body: BatchScoreRequest) -> dict:
    """Queue batch tabular scoring."""
    return submit_job(
        "batch_score",
        lambda: _batch_score(body),
        payload={"applicants": len(body.applicants)},
    )


@app.post("/eval/rag")
def eval_rag(body: RagEvalRequest) -> dict:
    """Run the local RAG eval harness. Defaults to retrieval-only for speed."""
    try:
        return run_rag_eval(**_eval_kwargs(body))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> dict:
    history = [m.model_dump() for m in body.history] or None
    trace = start_trace(
        "/query",
        model=body.model,
        question=body.question,
        facts_present=bool(body.facts),
    )
    try:
        payload = {
            "question": body.question.strip(),
            "facts": body.facts,
            "history": history or [],
            "model": body.model,
        }
        result = _rag_cached(
            payload=payload,
            trace=trace,
            call_fn=lambda: rag_query(
                question=payload["question"],
                facts=payload["facts"],
                history=history,
                model_name=payload["model"],
                trace=trace,
            ),
        )
        finish_trace(trace, status="ok")
        return result
    except TimeoutError as exc:
        finish_trace(trace, status="error", error=str(exc))
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        finish_trace(trace, status="error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/stream")
async def stream(body: QueryRequest, request: Request) -> StreamingResponse:
    history    = [m.model_dump() for m in body.history] or None
    stop_event = Event()
    trace = start_trace(
        "/stream",
        model=body.model,
        question=body.question,
        facts_present=bool(body.facts),
    )

    sync_gen = rag_stream(
        question=body.question.strip(),
        facts=body.facts,
        history=history,
        model_name=body.model,
        stop_event=stop_event,
        trace=trace,
    )

    started = perf_counter()

    async def event_stream():
        for chunk in sync_gen:
            if STREAM_TIMEOUT_S > 0 and (perf_counter() - started) > STREAM_TIMEOUT_S:
                stop_event.set()
                add_event(trace, "stream_timeout", {"timeout_s": STREAM_TIMEOUT_S})
                payload = {"type": "error", "message": "Stream timed out. Please try again."}
                yield f"data: {json.dumps(payload)}\n\n"
                break
            if await request.is_disconnected():
                stop_event.set()   # tell generate_stream to stop yielding
                break
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/model")
def set_model(body: ModelSwitchRequest) -> dict:
    """Preload and activate a model immediately."""
    try:
        active = switch_model(body.model)
        return {"status": "ok", "model": active}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(..., description="PCM WAV audio from browser microphone."),
    language: str = Form(default="en"),
) -> dict:
    """Transcribe uploaded microphone audio using local whisper-medium."""
    if file.content_type and not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload audio.")

    try:
        payload = await file.read()
        text = transcribe_wav_bytes(payload, language=language)
        print(f"[transcribe] language={language} text={text}")
        return {"text": text}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc


@app.post("/predict")
def predict(body: PredictRequest) -> dict:
    """Run all available ML models and return credit-risk predictions."""
    trace = start_trace("/predict")
    try:
        payload = body.model_dump()
        result = _predict_cached(payload=payload, trace=trace, label="predict")
        finish_trace(trace, status="ok")
        return result
    except TimeoutError as exc:
        finish_trace(trace, status="error", error=str(exc))
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        finish_trace(trace, status="error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/scenario")
def simulate_scenarios(body: ScenarioSimulationRequest) -> dict:
    """Score a baseline applicant plus locally evaluated what-if scenarios."""
    trace = start_trace("/scenario")
    try:
        baseline_input = body.base_applicant.model_dump()
        baseline_scores = _predict_cached(payload=baseline_input, trace=trace, label="scenario_baseline")
        baseline = {
            "input": baseline_input,
            "derived_metrics": _derived_applicant_metrics(baseline_input),
            "scores": baseline_scores,
        }
        if body.include_top_drivers:
            baseline["top_drivers"] = _top_driver_summary(baseline_input, body.drivers_model)

        scenarios = []
        for scenario in body.scenarios:
            overrides = scenario.model_dump(
                exclude={"scenario_id", "name"},
                exclude_none=True,
            )
            candidate_input = {**baseline_input, **overrides}
            validated_input = PredictRequest.model_validate(candidate_input).model_dump()
            scenario_scores = _predict_cached(payload=validated_input, trace=trace, label="scenario")
            changes = _scenario_changes(baseline_input, overrides)

            item = {
                "scenario_id": scenario.scenario_id,
                "name": scenario.name,
                "overrides": overrides,
                "changes": changes,
                "input": validated_input,
                "derived_metrics": _derived_applicant_metrics(validated_input),
                "scores": scenario_scores,
                "deltas": _score_deltas(baseline_scores, scenario_scores),
            }
            if body.include_top_drivers:
                item["top_drivers"] = _top_driver_summary(validated_input, body.drivers_model)
            scenarios.append(item)

        result = {
            "baseline": baseline,
            "scenarios": scenarios,
            "field_notes": {
                "supported_inputs": list(PredictRequest.model_fields.keys()),
                "derived_only": ["loan_to_income_pct"],
                "unsupported_without_model_retraining": ["ltv", "dti", "debt"],
            },
        }
        finish_trace(trace, status="ok")
        return result
    except TimeoutError as exc:
        finish_trace(trace, status="error", error=str(exc))
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        finish_trace(trace, status="error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/explain")
def explain(
    body: PredictRequest,
    model: str = Query(
        default="catboost",
        pattern="^(catboost|logistic_regression|neural_network)$",
        description="Model to explain: 'catboost', 'logistic_regression', or 'neural_network'",
    ),
) -> dict:
    """Return SHAP feature attributions for a single borrower."""
    try:
        return explain_shap(body.model_dump(), model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/dashboard/stats")
def dashboard_stats(
    limit: int = Query(default=1500, ge=100, le=10000, description="Maximum number of docs to sample."),
) -> dict:
    """Return corpus-level statistics for the frontend dashboard."""
    try:
        db = load_chroma_db()
        collection = getattr(db, "_collection", None)
        if collection is None:
            raise RuntimeError("Could not access Chroma collection.")

        total_documents = int(collection.count())
        sample_size = min(total_documents, limit)

        length_buckets = {"0-100": 0, "101-250": 0, "251-500": 0, "501+": 0}
        source_distribution: dict[str, int] = {}

        if sample_size == 0:
            return {
                "documents": {
                    "total_documents": 0,
                    "sample_size": 0,
                    "sampled": False,
                    "average_words": 0.0,
                    "median_words": 0.0,
                    "average_characters": 0.0,
                    "source_distribution": source_distribution,
                    "length_buckets": length_buckets,
                },
                "retrieval": retrieval_settings(),
            }

        payload = collection.get(limit=sample_size, include=["documents", "metadatas"])
        docs = payload.get("documents") or []
        metas = payload.get("metadatas") or []

        word_counts: list[int] = []
        char_counts: list[int] = []
        sources = Counter()

        for idx, raw_doc in enumerate(docs):
            if isinstance(raw_doc, list):
                text = " ".join(str(part) for part in raw_doc if part is not None)
            else:
                text = "" if raw_doc is None else str(raw_doc)

            words = len(text.split())
            chars = len(text)
            word_counts.append(words)
            char_counts.append(chars)

            if words <= 100:
                length_buckets["0-100"] += 1
            elif words <= 250:
                length_buckets["101-250"] += 1
            elif words <= 500:
                length_buckets["251-500"] += 1
            else:
                length_buckets["501+"] += 1

            meta = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
            source = str(meta.get("source", "unknown")).strip() or "unknown"
            sources[source] += 1

        source_distribution = dict(sorted(sources.items(), key=lambda item: item[1], reverse=True))
        average_words = round(sum(word_counts) / len(word_counts), 2) if word_counts else 0.0
        median_words = round(float(median(word_counts)), 2) if word_counts else 0.0
        average_characters = round(sum(char_counts) / len(char_counts), 2) if char_counts else 0.0

        return {
            "documents": {
                "total_documents": total_documents,
                "sample_size": sample_size,
                "sampled": sample_size < total_documents,
                "average_words": average_words,
                "median_words": median_words,
                "average_characters": average_characters,
                "source_distribution": source_distribution,
                "length_buckets": length_buckets,
            },
            "retrieval": {
                **retrieval_settings(),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest", status_code=201)
def ingest(body: IngestRequest, request: Request) -> dict:
    """Add text documents to the ChromaDB knowledge base.

    Requires the ``X-API-Key`` header when ``INGEST_API_KEY`` env var is set.
    """
    if INGEST_API_KEY:
        if request.headers.get("X-API-Key", "") != INGEST_API_KEY:
            raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key header.")
    try:
        return _ingest_documents(body)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000,
                reload=False, workers=1, timeout_keep_alive=600)
