"""
Optional external LLM model-server client.

Set LLM_MODEL_SERVER_URL to route generation to a vLLM OpenAI-compatible server
or a Hugging Face TGI server. If unset, model_loader uses the local Transformers
runtime exactly as before.
"""
from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any, Generator
from urllib import request

from observability import set_trace_section


LLM_MODEL_SERVER_URL = os.getenv("LLM_MODEL_SERVER_URL", "").rstrip("/")
LLM_MODEL_SERVER_TYPE = os.getenv("LLM_MODEL_SERVER_TYPE", "openai").strip().lower()
LLM_MODEL_SERVER_MODEL = os.getenv("LLM_MODEL_SERVER_MODEL", "llama-3.2")
LLM_MODEL_SERVER_API_KEY = os.getenv("LLM_MODEL_SERVER_API_KEY", "")
LLM_MODEL_SERVER_TIMEOUT_S = float(os.getenv("LLM_MODEL_SERVER_TIMEOUT_S", "120"))


def external_generation_enabled() -> bool:
    return bool(LLM_MODEL_SERVER_URL)


def external_model_name(fallback: str | None = None) -> str:
    return LLM_MODEL_SERVER_MODEL or fallback or "external-llm"


def _messages(system_prompt: str, user_message: str, history: list[dict] | None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    for item in history or []:
        role = str(item.get("role", "user"))
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": str(item.get("content", ""))})
    messages.append({"role": "user", "content": user_message})
    return messages


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if LLM_MODEL_SERVER_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_MODEL_SERVER_API_KEY}"
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    with request.urlopen(req, timeout=LLM_MODEL_SERVER_TIMEOUT_S) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _openai_url() -> str:
    if LLM_MODEL_SERVER_URL.endswith("/chat/completions"):
        return LLM_MODEL_SERVER_URL
    if LLM_MODEL_SERVER_URL.endswith("/v1"):
        return f"{LLM_MODEL_SERVER_URL}/chat/completions"
    return f"{LLM_MODEL_SERVER_URL}/v1/chat/completions"


def _tgi_url() -> str:
    if LLM_MODEL_SERVER_URL.endswith("/generate"):
        return LLM_MODEL_SERVER_URL
    return f"{LLM_MODEL_SERVER_URL}/generate"


def _tgi_prompt(system_prompt: str, user_message: str, history: list[dict] | None) -> str:
    chunks = [f"System: {system_prompt}"]
    for item in history or []:
        role = str(item.get("role", "user")).title()
        chunks.append(f"{role}: {item.get('content', '')}")
    chunks.append(f"User: {user_message}")
    chunks.append("Assistant:")
    return "\n\n".join(chunks)


def generate_via_model_server(
    *,
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
    model_name: str | None = None,
    trace: dict | None = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a full answer through vLLM/OpenAI-compatible or TGI HTTP APIs."""
    started = perf_counter()
    selected_model = external_model_name(model_name)

    if LLM_MODEL_SERVER_TYPE == "tgi":
        payload = {
            "inputs": _tgi_prompt(system_prompt, user_message, history),
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "return_full_text": False,
            },
        }
        data = _post_json(_tgi_url(), payload)
        answer = str(data.get("generated_text") or data.get("details", {}).get("generated_text") or "")
    else:
        payload = {
            "model": selected_model,
            "messages": _messages(system_prompt, user_message, history),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        data = _post_json(_openai_url(), payload)
        choices = data.get("choices") or []
        answer = ""
        if choices:
            message = choices[0].get("message") or {}
            answer = str(message.get("content") or choices[0].get("text") or "")

    latency_s = perf_counter() - started
    set_trace_section(trace, "generation", {
        "provider": LLM_MODEL_SERVER_TYPE,
        "model_server": LLM_MODEL_SERVER_URL,
        "model": selected_model,
        "latency_ms": round(latency_s * 1000, 2),
    })
    return answer.strip()


def stream_via_model_server(**kwargs: Any) -> Generator[str, None, None]:
    """Streaming facade for external servers.

    The first implementation yields a complete server response as one chunk; this
    keeps the gateway contract stable while model serving is externalized.
    """
    yield generate_via_model_server(**kwargs)
