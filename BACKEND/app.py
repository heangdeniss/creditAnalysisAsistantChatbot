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

import logging
import os
import sys
import warnings
from collections import Counter
from statistics import median

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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rag_pipeline import (
    get_retriever,
    rag_query,
    rag_stream,
    retrieval_settings,
    retriever_is_loaded,
)
from model_loader import DEFAULT_MODEL, load_model, model_is_loaded, model_is_busy, current_model_name, switch_model
from ml_predict import load_all_models, predict as ml_predict, explain_shap
from chroma_loader import chunk_texts, load_chroma_db
from speech_to_text import transcribe_wav_bytes
from observability import finish_trace, metrics_summary, recent_traces, start_trace
from eval.rag_eval import run_eval as run_rag_eval


@asynccontextmanager
async def lifespan(_: FastAPI):
    print("[startup] loading embeddings + retriever …")
    get_retriever()
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


# Config
INGEST_API_KEY = os.getenv("INGEST_API_KEY", "")   # set to require auth on /ingest

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


def _derived_applicant_metrics(applicant: dict) -> dict:
    """Return local derived metrics that are useful for scenario comparison."""
    income = float(applicant.get("person_income") or 0)
    loan_amount = float(applicant.get("loan_amnt") or 0)
    loan_to_income_pct = None
    if income > 0:
        loan_to_income_pct = round((loan_amount / income) * 100, 2)
    return {
        "loan_to_income_pct": loan_to_income_pct,
    }


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

    return {
        "status": "ok",
        "probability_delta": probability_delta,
        "relative_probability_delta_pct": relative_delta_pct,
        "decision_changed": baseline.get("decision") != scenario.get("decision"),
        "grade_changed": baseline.get("grade") != scenario.get("grade"),
        "label_changed": baseline.get("label") != scenario.get("label"),
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
    return metrics_summary()


@app.get("/traces/recent")
def traces(limit: int = Query(default=25, ge=1, le=250)) -> dict:
    """Return recent completed request traces."""
    return {"traces": recent_traces(limit)}


@app.post("/eval/rag")
def eval_rag(body: RagEvalRequest) -> dict:
    """Run the local RAG eval harness. Defaults to retrieval-only for speed."""
    try:
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
        return run_rag_eval(**kwargs)
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
        result = rag_query(
            question=body.question.strip(),
            facts=body.facts,
            history=history,
            model_name=body.model,
            trace=trace,
        )
        finish_trace(trace, status="ok")
        return result
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

    async def event_stream():
        for chunk in sync_gen:
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
    try:
        return ml_predict(body.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/scenario")
def simulate_scenarios(body: ScenarioSimulationRequest) -> dict:
    """Score a baseline applicant plus locally evaluated what-if scenarios."""
    try:
        baseline_input = body.base_applicant.model_dump()
        baseline_scores = ml_predict(baseline_input)
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
            scenario_scores = ml_predict(validated_input)

            item = {
                "scenario_id": scenario.scenario_id,
                "name": scenario.name,
                "overrides": overrides,
                "input": validated_input,
                "derived_metrics": _derived_applicant_metrics(validated_input),
                "scores": scenario_scores,
                "deltas": _score_deltas(baseline_scores, scenario_scores),
            }
            if body.include_top_drivers:
                item["top_drivers"] = _top_driver_summary(validated_input, body.drivers_model)
            scenarios.append(item)

        return {
            "baseline": baseline,
            "scenarios": scenarios,
            "field_notes": {
                "supported_inputs": list(PredictRequest.model_fields.keys()),
                "derived_only": ["loan_to_income_pct"],
                "unsupported_without_model_retraining": ["ltv", "dti", "debt"],
            },
        }
    except Exception as exc:
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
        db = load_chroma_db()
        chunks, metadatas = chunk_texts(body.texts, source=body.source)
        if not chunks:
            raise HTTPException(status_code=400, detail="No non-empty text chunks to ingest.")
        ids = db.add_texts(texts=chunks, metadatas=metadatas)
        return {
            "added": len(ids),
            "ids": ids,
            "chunk_size": retrieval_settings()["chunk_size"],
            "chunk_overlap": retrieval_settings()["chunk_overlap"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000,
                reload=False, workers=1, timeout_keep_alive=600)
