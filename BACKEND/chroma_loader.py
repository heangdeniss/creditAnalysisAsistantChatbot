"""
chroma_loader.py
Loads the persisted ChromaDB vector store with E5 multilingual embeddings.
Singletons ensure the embedding model and DB are initialised only once.
"""
from __future__ import annotations

import logging
import os
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Config
def _env_int(*names: str, default: int, minimum: int = 0) -> int:
    for name in names:
        raw = os.getenv(name)
        if raw not in (None, ""):
            try:
                return max(minimum, int(raw))
            except ValueError:
                return default
    return default


_BASE        = os.path.dirname(__file__)
CHROMA_DIR   = os.path.join(_BASE, "..", "chroma_db")
COLLECTION   = "credit_risk_corpus"
EMBED_MODEL  = "intfloat/multilingual-e5-base"
CHUNK_SIZE   = _env_int("RAG_CHUNK_SIZE", "CHUNK_SIZE", default=800, minimum=100)
CHUNK_OVERLAP = _env_int("RAG_CHUNK_OVERLAP", "CHUNK_OVERLAP", default=120)
if CHUNK_OVERLAP >= CHUNK_SIZE:
    CHUNK_OVERLAP = max(0, CHUNK_SIZE // 5)

# Singletons
_embedding: HuggingFaceEmbeddings | None = None
_db: Chroma | None = None


class E5Embeddings(HuggingFaceEmbeddings):
    """Adds E5 passage/query prefixes required by multilingual-e5-base."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")


def load_chroma_db() -> Chroma:
    """Return the ChromaDB vector store, initialising it on first call."""
    global _embedding, _db
    if _db is not None:
        return _db

    _embedding = get_embedding_function()
    _db = Chroma(
        collection_name=COLLECTION,
        embedding_function=_embedding,
        persist_directory=CHROMA_DIR,
    )
    return _db


def get_embedding_function() -> E5Embeddings:
    """Return the shared E5 embedding model without constructing Chroma."""
    global _embedding
    if _embedding is None:
        _embedding = E5Embeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding


def embed_query(text: str) -> list[float]:
    """Embed a query using the same model/prefixing used by the vector store."""
    return get_embedding_function().embed_query(text)


def chunk_texts(texts: list[str], source: str = "api") -> tuple[list[str], list[dict]]:
    """Split incoming texts using configurable local chunk settings."""
    chunks: list[str] = []
    metadatas: list[dict] = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)

    for doc_idx, raw_text in enumerate(texts, start=1):
        text = str(raw_text or "").strip()
        if not text:
            continue

        document_id = f"{source}:{doc_idx}"
        start = 0
        chunk_idx = 1
        while start < len(text):
            end = min(len(text), start + CHUNK_SIZE)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                metadatas.append({
                    "source": source,
                    "document_id": document_id,
                    "chunk_id": f"{document_id}:chunk_{chunk_idx}",
                    "chunk_start": start,
                    "chunk_end": end,
                })
                chunk_idx += 1
            if end >= len(text):
                break
            start += step

    return chunks, metadatas
