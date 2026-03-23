"""
app.py
FastAPI backend — routes: GET /health, POST /query, POST /stream,
                           POST /predict, POST /ingest.
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
from pydantic import BaseModel, Field
from rag_pipeline import rag_query, rag_stream, get_retriever, retriever_is_loaded
from model_loader import DEFAULT_MODEL, load_model, model_is_loaded, model_is_busy, current_model_name, switch_model
from ml_predict import load_all_models, predict as ml_predict, explain_shap
from chroma_loader import load_chroma_db
from speech_to_text import transcribe_wav_bytes


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


class QueryResponse(BaseModel):
    question: str
    answer:   str
    context:  list[str]


class PredictRequest(BaseModel):
    person_age:                 float = Field(..., ge=18,  le=100)
    person_income:              float = Field(..., ge=0)
    person_home_ownership:      str   = Field(..., pattern="^(MORTGAGE|OWN|RENT|OTHER)$")
    person_emp_length:          float = Field(..., ge=0,   le=60)
    loan_intent:                str   = Field(..., pattern="^(DEBTCONSOLIDATION|EDUCATION|HOMEIMPROVEMENT|MEDICAL|PERSONAL|VENTURE)$")
    loan_amnt:                  float = Field(..., ge=500)
    loan_int_rate:              float = Field(..., ge=1,   le=40)
    cb_person_default_on_file:  str   = Field(..., pattern="^(Y|N)$")
    cb_person_cred_hist_length: float = Field(..., ge=0,   le=60)


class IngestRequest(BaseModel):
    texts:  list[Annotated[str, Field(min_length=1, max_length=8192)]] = Field(
        ..., min_length=1, max_length=50,
        description="Text passages to add (max 50 items, each \u2264 8\u202f192 chars).",
    )
    source: str = Field(default="api", description="Metadata label stored alongside each passage.")


# Routes
@app.get("/health")
def health() -> dict:
    if not model_is_loaded():
        raise HTTPException(status_code=503, detail="LLM not loaded yet.")
    if not retriever_is_loaded():
        raise HTTPException(status_code=503, detail="Retriever not loaded yet.")
    return {"status": "ok", "busy": model_is_busy(), "model": current_model_name()}


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> dict:
    history = [m.model_dump() for m in body.history] or None
    try:
        return rag_query(
            question=body.question.strip(),
            facts=body.facts,
            history=history,
            model_name=body.model,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/stream")
async def stream(body: QueryRequest, request: Request) -> StreamingResponse:
    history    = [m.model_dump() for m in body.history] or None
    stop_event = Event()

    sync_gen = rag_stream(
        question=body.question.strip(),
        facts=body.facts,
        history=history,
        model_name=body.model,
        stop_event=stop_event,
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
        metadatas = [{"source": body.source}] * len(body.texts)
        ids = db.add_texts(texts=body.texts, metadatas=metadatas)
        return {"added": len(ids), "ids": ids}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000,
                reload=False, workers=1, timeout_keep_alive=600)
