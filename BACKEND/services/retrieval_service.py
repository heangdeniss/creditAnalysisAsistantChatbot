"""Standalone retrieval service.

Run:
  uvicorn services.retrieval_service:app --host 0.0.0.0 --port 8012
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_pipeline import _retrieve_citations, get_retriever, retrieval_settings, retriever_is_loaded


SKIP_STARTUP_LOAD = os.getenv("SKIP_STARTUP_LOAD", "").lower() in {"1", "true", "yes"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not SKIP_STARTUP_LOAD:
        get_retriever()
    yield


app = FastAPI(title="Credit Risk Retrieval Service", version="1.0.0", lifespan=lifespan)


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=25)
    score_threshold: float | None = Field(default=None, ge=0, le=1)


@app.get("/health")
def health() -> dict:
    if not retriever_is_loaded() and not SKIP_STARTUP_LOAD:
        raise HTTPException(status_code=503, detail="Retriever not loaded yet.")
    return {"status": "ok", "retrieval": retrieval_settings()}


@app.post("/retrieve")
def retrieve(body: RetrieveRequest) -> dict:
    chunks = _retrieve_citations(
        body.question,
        top_k=body.top_k,
        score_threshold=body.score_threshold,
        history=body.history,
    )
    return {"chunks": chunks, "retrieval": retrieval_settings()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.retrieval_service:app", host="0.0.0.0", port=8012, reload=False)
