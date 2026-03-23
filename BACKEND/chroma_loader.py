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
_BASE        = os.path.dirname(__file__)
CHROMA_DIR   = os.path.join(_BASE, "..", "chroma_db")
COLLECTION   = "credit_risk_corpus"
EMBED_MODEL  = "intfloat/multilingual-e5-base"

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

    _embedding = E5Embeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    _db = Chroma(
        collection_name=COLLECTION,
        embedding_function=_embedding,
        persist_directory=CHROMA_DIR,
    )
    return _db
