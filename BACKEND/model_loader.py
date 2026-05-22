"""
model_loader.py
Loads local Llama 3.2 models and exposes
  - generate()        → returns full answer string
  - generate_stream() → yields tokens one-by-one via TextIteratorStreamer
"""
from __future__ import annotations

import logging
import os
import warnings
from threading import Thread, Event, Semaphore
from time import perf_counter
from typing import Generator
import gc

# Disable TF before any transformers import (saves ~5 s of startup)
os.environ["USE_TF"]                 = "0"
os.environ["USE_TORCH"]              = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from model_server_client import (
    external_generation_enabled,
    external_model_name,
    generate_via_model_server,
    stream_via_model_server,
)
from observability import add_event, set_trace_section

# Config
_BASE  = os.path.dirname(__file__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATHS: dict[str, str] = {
    "llama-1b": os.path.join(_BASE, "..", "model", "llama-1b"),
    "llama-3b": os.path.join(_BASE, "..", "model", "llama-3b"),
}
DEFAULT_MODEL  = "llama-1b"
CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", "2048"))
GPU_LAYERS     = max(0, int(os.getenv("GPU_LAYERS", "20")))

_GEN_KWARGS: dict = dict(
    do_sample=True,
    pad_token_id=None,   # filled in after tokenizer is loaded
    eos_token_id=None,
)

# Singletons
_tokenizer: AutoTokenizer | None        = None
_model: AutoModelForCausalLM | None     = None
_active_model_name: str | None          = None
_gen_lock = Semaphore(1)                # serialise concurrent generation requests


def normalize_model_name(model_name: str | None) -> str:
    """Normalise aliases to a supported model key."""
    if not model_name:
        return DEFAULT_MODEL

    key = model_name.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "1b": "llama-1b",
        "3b": "llama-3b",
        "llama1b": "llama-1b",
        "llama3b": "llama-3b",
    }
    key = aliases.get(key, key)

    if key not in MODEL_PATHS:
        raise ValueError(f"Unsupported model '{model_name}'. Use one of: {', '.join(MODEL_PATHS)}")
    return key


def model_is_loaded() -> bool:
    """Return True once the tokenizer and model have been loaded."""
    if external_generation_enabled():
        return True
    return _tokenizer is not None and _model is not None


def model_is_busy() -> bool:
    """Non-blocking check: True if the model is currently generating."""
    if external_generation_enabled():
        return False
    acquired = _gen_lock.acquire(blocking=False)
    if acquired:
        _gen_lock.release()
    return not acquired


def current_model_name() -> str:
    """Return the active model key, defaulting to llama-1b before first load."""
    if external_generation_enabled():
        return external_model_name(DEFAULT_MODEL)
    return _active_model_name or DEFAULT_MODEL


def _build_device_map(model_path: str):
    """Build a safe device map for Transformers loading.

    Note: a handcrafted CUDA/CPU split can trigger cross-device attention
    failures on larger checkpoints (e.g. llama-3b). For mixed placement,
    prefer Accelerate's built-in 'auto' dispatcher.
    """
    if DEVICE != "cuda":
        return "auto"

    if GPU_LAYERS == 0:
        return {"": "cpu"}

    cfg = AutoConfig.from_pretrained(model_path)
    num_layers = int(getattr(cfg, "num_hidden_layers", 0))
    if num_layers <= 0:
        return "auto"

    n_gpu = min(GPU_LAYERS, num_layers)
    if n_gpu >= num_layers:
        return "auto"

    # Mixed placement requested: rely on Accelerate auto-partitioning.
    return "auto"


def load_model(model_name: str | None = None) -> tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    """Load a requested model; unload and replace cached model if it changed."""
    global _tokenizer, _model, _active_model_name

    selected = normalize_model_name(model_name)
    if _tokenizer and _model and _active_model_name == selected:
        return _tokenizer, _model, selected

    if _model is not None:
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_path = MODEL_PATHS[selected]
    device_map = _build_device_map(model_path)

    force_cpu = isinstance(device_map, dict) and device_map.get("") == "cpu"
    torch_dtype = torch.float16 if DEVICE == "cuda" and not force_cpu else torch.float32

    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    _model.eval()
    _active_model_name = selected

    # Patch shared gen kwargs now that tokenizer is available
    _GEN_KWARGS["pad_token_id"] = _tokenizer.eos_token_id
    _GEN_KWARGS["eos_token_id"] = _tokenizer.eos_token_id
    return _tokenizer, _model, selected


def switch_model(model_name: str | None = None) -> str:
    """Synchronously switch the active model, waiting for current generation if needed."""
    if external_generation_enabled():
        return external_model_name(model_name)
    with _gen_lock:
        _, _, selected = load_model(model_name)
    return selected


def _build_input_ids(
    tokenizer: AutoTokenizer,
    system: str,
    user: str,
    history: list[dict] | None = None,
) -> torch.Tensor:
    """Apply Llama 3.2 Instruct chat template with optional prior turns."""
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=CONTEXT_LENGTH,
    )


# Generation constants
MAX_NEW_TOKENS:      int   = 256  #512 is overkill for 1B params
TEMPERATURE:         float = 0.7  # More Creative 0.3 is too stiff
TOP_P:               float = 0.9  # 0.1 is too restriction 
REPETITION_PENALTY: float = 1.1   # >1 penalises already-generated tokens
NO_REPEAT_NGRAM:     int   = 3    # blocks any 3-gram from repeating
QUEUED_SENTINEL:     str   = "\x00QUEUED\x00"  # never appears in real LLM output


def generate(
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
    model_name: str | None = None,
    trace: dict | None = None,
) -> str:
    """Return a complete generated answer (blocking)."""
    if external_generation_enabled():
        return generate_via_model_server(
            system_prompt=system_prompt,
            user_message=user_message,
            history=history,
            model_name=model_name,
            trace=trace,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

    wait_started = perf_counter()
    with _gen_lock:
        queue_wait_ms = round((perf_counter() - wait_started) * 1000, 2)
        tokenizer, model, _ = load_model(model_name)
        input_ids = _build_input_ids(tokenizer, system_prompt, user_message, history)
        input_ids = input_ids.to(next(model.parameters()).device)
        gen_started = perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
                **_GEN_KWARGS,
            )
        gen_latency_s = perf_counter() - gen_started

    tokens_out = int(output_ids.shape[-1] - input_ids.shape[-1])
    set_trace_section(trace, "generation", {
        "queue_wait_ms": queue_wait_ms,
        "latency_ms": round(gen_latency_s * 1000, 2),
        "input_tokens": int(input_ids.shape[-1]),
        "tokens_out": tokens_out,
        "tokens_per_sec": round(tokens_out / gen_latency_s, 2) if gen_latency_s > 0 else 0.0,
    })

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_stream(
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
    model_name: str | None = None,
    stop_event: Event | None = None,
    trace: dict | None = None,
) -> Generator[str, None, None]:
    """Yield decoded text tokens one-by-one as the model generates them.

    If stop_event is set (e.g. because the client disconnected) the generator
    stops yielding immediately and lets the daemon thread finish naturally.
    """
    if external_generation_enabled():
        yield from stream_via_model_server(
            system_prompt=system_prompt,
            user_message=user_message,
            history=history,
            model_name=model_name,
            trace=trace,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return

    lock_attempted = Event()
    is_queued      = Event()
    streamer_ready = Event()
    stream_state: dict[str, object] = {}

    def _run():
        wait_started = perf_counter()
        acquired = _gen_lock.acquire(blocking=False)
        lock_attempted.set()         # signals that we know our queue status
        if not acquired:
            is_queued.set()          # let the generator yield the queued sentinel
            _gen_lock.acquire()      # block until the previous generation finishes
        queue_wait_ms = round((perf_counter() - wait_started) * 1000, 2)
        try:
            if stop_event and stop_event.is_set():
                streamer_ready.set()
                return

            tokenizer, model, _ = load_model(model_name)
            input_ids = _build_input_ids(tokenizer, system_prompt, user_message, history)
            input_ids = input_ids.to(next(model.parameters()).device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            stream_state["streamer"] = streamer
            streamer_ready.set()

            _kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
                streamer=streamer,
                **_GEN_KWARGS,
            )

            if not (stop_event and stop_event.is_set()):
                gen_started = perf_counter()
                with torch.inference_mode():
                    output_ids = model.generate(**_kwargs)
                gen_latency_s = perf_counter() - gen_started
                tokens_out = int(output_ids.shape[-1] - input_ids.shape[-1])
                set_trace_section(trace, "generation", {
                    "queue_wait_ms": queue_wait_ms,
                    "latency_ms": round(gen_latency_s * 1000, 2),
                    "input_tokens": int(input_ids.shape[-1]),
                    "tokens_out": tokens_out,
                    "tokens_per_sec": round(tokens_out / gen_latency_s, 2) if gen_latency_s > 0 else 0.0,
                })
            else:
                add_event(trace, "generation_cancelled_before_start")
                streamer.on_finalized_text("", stream_end=True)
        except Exception as exc:
            stream_state["error"] = exc
            add_event(trace, "generation_error", {"error": str(exc)})
            streamer_ready.set()
            streamer = stream_state.get("streamer")
            if isinstance(streamer, TextIteratorStreamer):
                streamer.on_finalized_text("", stream_end=True)
        finally:
            _gen_lock.release()

    thread = Thread(target=_run, daemon=True)
    thread.start()
    lock_attempted.wait()            # returns in microseconds once thread attempts lock
    if is_queued.is_set():
        yield QUEUED_SENTINEL

    streamer_ready.wait()
    err = stream_state.get("error")
    if isinstance(err, Exception):
        raise err

    streamer = stream_state.get("streamer")
    if not isinstance(streamer, TextIteratorStreamer):
        return

    for token in streamer:
        if stop_event and stop_event.is_set():
            break
        yield token
    if not (stop_event and stop_event.is_set()):
        thread.join(timeout=2.0)
    # We only wait for normal completion so trace metrics are available.
    # After early disconnects the daemon thread may still be unwinding
    # model.generate while the client is already gone.
