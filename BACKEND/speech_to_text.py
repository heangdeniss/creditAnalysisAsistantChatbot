from __future__ import annotations

import io
import os
import wave
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 60

_MODEL_LOCK = Lock()
_PROCESSOR: WhisperProcessor | None = None
_MODEL: WhisperForConditionalGeneration | None = None
_DEVICE: str | None = None


def _find_model_dir() -> Path:
    base = Path(__file__).resolve().parent
    env_model_dir = os.getenv("WHISPER_MODEL_DIR", "").strip()
    candidates = [
        Path(env_model_dir) if env_model_dir else None,
        base.parent / "model" / "whisper-medium",
        Path("E:/models/whisper-medium"),
        Path("E:/Credit Risk Data Science/model/whisper-medium"),
        Path("E:/Credit Risk Data Science/RAG Llama 3.2 1B/model/whisper-medium"),
    ]
    candidates = [p for p in candidates if p is not None]

    required_common = ["config.json", "tokenizer.json", "preprocessor_config.json"]
    required_any_weight = ["model.safetensors", "pytorch_model.bin"]

    for path in candidates:
        if not path.exists():
            continue
        has_common = all((path / name).exists() for name in required_common)
        has_weight = any((path / name).exists() for name in required_any_weight)
        if has_common and has_weight:
            return path

    raise FileNotFoundError(
        "Whisper model files not found. "
        "Ensure whisper-medium includes model.safetensors or pytorch_model.bin, "
        "or set WHISPER_MODEL_DIR to a valid model folder."
    )


def _resolve_device() -> str:
    pref = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("WHISPER_DEVICE=cuda but CUDA is unavailable.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_whisper() -> tuple[WhisperProcessor, WhisperForConditionalGeneration, str]:
    global _PROCESSOR, _MODEL, _DEVICE
    if _PROCESSOR is not None and _MODEL is not None and _DEVICE is not None:
        return _PROCESSOR, _MODEL, _DEVICE

    with _MODEL_LOCK:
        if _PROCESSOR is not None and _MODEL is not None and _DEVICE is not None:
            return _PROCESSOR, _MODEL, _DEVICE

        model_dir = _find_model_dir()
        device = _resolve_device()
        processor = WhisperProcessor.from_pretrained(str(model_dir), local_files_only=True)

        model = WhisperForConditionalGeneration.from_pretrained(
            str(model_dir),
            local_files_only=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()

        _PROCESSOR = processor
        _MODEL = model
        _DEVICE = device
        return _PROCESSOR, _MODEL, _DEVICE


def _decode_pcm_wav(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            sample_rate = int(wav_file.getframerate())
            channels = int(wav_file.getnchannels())
            sample_width = int(wav_file.getsampwidth())
            frames = wav_file.readframes(wav_file.getnframes())
    except wave.Error as exc:
        raise ValueError("Unsupported audio format. Please send a valid WAV file.") from exc

    if sample_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio.astype(np.float32), sample_rate


def _resample_linear(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr or audio.size == 0:
        return audio.astype(np.float32)

    duration = audio.shape[0] / float(original_sr)
    target_len = max(1, int(duration * target_sr))
    x_old = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def transcribe_wav_bytes(audio_bytes: bytes, language: str = "en") -> str:
    if not audio_bytes:
        raise ValueError("No audio payload provided.")

    audio, sample_rate = _decode_pcm_wav(audio_bytes)
    if audio.size == 0:
        raise ValueError("Audio payload is empty.")

    audio = _resample_linear(audio, sample_rate, TARGET_SAMPLE_RATE)

    max_samples = TARGET_SAMPLE_RATE * MAX_AUDIO_SECONDS
    if audio.shape[0] > max_samples:
        raise ValueError(f"Audio is too long. Maximum supported duration is {MAX_AUDIO_SECONDS} seconds.")

    processor, model, device = _load_whisper()
    model_dtype = next(model.parameters()).dtype

    inputs = processor(audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device=device, dtype=model_dtype)

    lang = language.strip().lower() if language else "en"
    if not lang:
        lang = "en"
    if any(ch not in "abcdefghijklmnopqrstuvwxyz-" for ch in lang):
        raise ValueError("Invalid language code. Use values like 'en'.")

    with torch.inference_mode():
        predicted_ids = model.generate(
            input_features=input_features,
            max_new_tokens=128,
            language=lang,
            task="transcribe",
        )

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    return text
