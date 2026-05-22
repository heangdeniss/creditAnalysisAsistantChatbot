"""Standalone LLM inference service.

Run:
  uvicorn services.inference_service:app --host 0.0.0.0 --port 8011
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_loader import DEFAULT_MODEL, current_model_name, generate, load_model, model_is_loaded
from model_server_client import external_generation_enabled


SKIP_STARTUP_LOAD = os.getenv("SKIP_STARTUP_LOAD", "").lower() in {"1", "true", "yes"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not SKIP_STARTUP_LOAD and not external_generation_enabled():
        load_model(DEFAULT_MODEL)
    yield


app = FastAPI(title="Credit Risk LLM Inference Service", version="1.0.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    system_prompt: str = Field(..., min_length=1, max_length=8192)
    user_message: str = Field(..., min_length=1, max_length=8192)
    history: list[dict] = Field(default_factory=list)
    model: str | None = Field(default=None)


@app.get("/health")
def health() -> dict:
    if not model_is_loaded():
        raise HTTPException(status_code=503, detail="LLM not loaded yet.")
    return {
        "status": "ok",
        "model": current_model_name(),
        "external_model_server": external_generation_enabled(),
    }


@app.post("/generate")
def generate_text(body: GenerateRequest) -> dict:
    answer = generate(
        system_prompt=body.system_prompt,
        user_message=body.user_message,
        history=body.history,
        model_name=body.model,
    )
    return {"answer": answer, "model": current_model_name()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.inference_service:app", host="0.0.0.0", port=8011, reload=False)
