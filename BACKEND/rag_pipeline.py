"""
rag_pipeline.py
Orchestrates retrieval from ChromaDB and generation via Llama 3.2 1B.
Generation settings are fixed in model_loader.py — not passed from the API.
"""
from __future__ import annotations

import json
import os
import sys
from threading import Event
from typing import Generator

sys.path.insert(0, os.path.dirname(__file__))

from chroma_loader import load_chroma_db
from model_loader import generate, generate_stream, QUEUED_SENTINEL

# Config
TOP_K           = 4
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.45"))
NO_INFO         = "I don't have enough information in my knowledge base to answer that."

# Used for normal chat questions (no borrower data)
RAG_SYSTEM_PROMPT = (
    "You are a professional credit risk assistant. "
    "Answer the question using only the context provided. Be concise and stop when done — do not add extra sections, conclusions, or additional information. "
    "Format rules:\n"
    "- Plain English only. No jargon.\n"     
    "- Use '- ' (dash space) for bullet points. Never use * or • as bullets.\n"
    "- Every bullet must contain a complete sentence. No empty bullets.\n"
    "- Do not use asterisks (*) anywhere.\n"
    "- Do not add tables.\n"
    "- Do not add sections titled 'Additional Information', 'In Conclusion', or 'Note'.\n"
    "- Stop writing as soon as the question is fully answered.\n"
    f"- If the answer is not in the context, say exactly: \"{NO_INFO}\""
)

RAG_USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"

# Used when borrower facts are provided (predict explanation) — no RAG, direct answer
EXPLAIN_SYSTEM_PROMPT = (
    "You are a professional credit risk advisor writing a clear loan decision explanation for a regular person. "
    "Follow this structure EXACTLY — no more, no less:\n"
    "1. One sentence summarising the decision and the single strongest reason.\n"
    "2. A section headed '### Key Factors' containing exactly 3-5 bullet points. "
    "Each bullet MUST start with '- ' and be a single complete sentence that states a fact and explains its impact. "
    "Do NOT add sub-bullets, nested bullets, topic labels, or bold sub-headings inside this section. "
    "Do NOT leave any bullet empty.\n"
    "3. A section headed '### What This Means For You' with exactly 2-3 sentences of practical advice.\n"
    "Strict rules — violating any rule makes the response wrong:\n"
    "- Use the exact numbers from the data. Never invent or change figures.\n"
    "- Plain English only. No jargon, no ML internals, no algorithm explanations.\n"
    "- No asterisks, no HTML/XML tags, no HTML comments.\n"
    "- No extra headers, titles, or sections beyond the three listed above.\n"
    "- No empty lines between bullets.\n"
    "- Stop writing immediately after the last sentence of section 3."
    "Do not use HTML tags, XML tags, or HTML comments (e.g. no <!--, -->, <div>, etc.). "
    "Do not explain how the model algorithm works — focus only on the borrower's financial profile and numbers. "
    "The model name is for labelling only; treat every model the same way."
)

EXPLAIN_USER_TEMPLATE = """{facts}

{question}"""

# Retriever singleton
_retriever = None


def retriever_is_loaded() -> bool:
    """Return True once the ChromaDB retriever has been initialised."""
    return _retriever is not None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = load_chroma_db().as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": TOP_K, "score_threshold": SCORE_THRESHOLD},
        )
    return _retriever


# Helpers
def _retrieve(question: str) -> list[str]:
    return [doc.page_content for doc in get_retriever().invoke(question)]


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# Public API
def rag_query(
    question: str,
    facts: str = "",
    history: list[dict] | None = None,
    model_name: str | None = None,
) -> dict[str, object]:
    """Blocking RAG: retrieve → generate → return dict."""
    if facts:
        answer = generate(
            system_prompt=EXPLAIN_SYSTEM_PROMPT,
            user_message=EXPLAIN_USER_TEMPLATE.format(facts=facts, question=question),
            model_name=model_name,
        )
        return {"question": question, "context": [], "answer": answer}

    chunks = _retrieve(question)
    if not chunks:
        return {"question": question, "context": [], "answer": NO_INFO}
    answer = generate(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_message=RAG_USER_TEMPLATE.format(
            context="\n\n---\n\n".join(chunks), question=question
        ),
        history=history,
        model_name=model_name,
    )
    return {"question": question, "context": chunks, "answer": answer}


def rag_stream(
    question: str,
    facts: str = "",
    history: list[dict] | None = None,
    model_name: str | None = None,
    stop_event: Event | None = None,
) -> Generator[str, None, None]:
    """
    Streaming RAG as Server-Sent Events.
      {"type": "context", "chunks": [...]}   sent first
      {"type": "token",   "text":  str}      one per token
      {"type": "done"}                       end of stream
    """
    if facts:
        yield _sse({"type": "context", "chunks": []})
        for token in generate_stream(
            system_prompt=EXPLAIN_SYSTEM_PROMPT,
            user_message=EXPLAIN_USER_TEMPLATE.format(facts=facts, question=question),
            model_name=model_name,
            stop_event=stop_event,
        ):
            if token == QUEUED_SENTINEL:
                yield _sse({"type": "queued"})
                continue
            yield _sse({"type": "token", "text": token})
        yield _sse({"type": "done"})
        return

    chunks = _retrieve(question)
    yield _sse({"type": "context", "chunks": chunks})

    if not chunks:
        yield _sse({"type": "token", "text": NO_INFO})
        yield _sse({"type": "done"})
        return

    for token in generate_stream(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_message=RAG_USER_TEMPLATE.format(
            context="\n\n---\n\n".join(chunks), question=question
        ),
        history=history,
        model_name=model_name,
        stop_event=stop_event,
    ):
        if token == QUEUED_SENTINEL:
            yield _sse({"type": "queued"})
            continue
        yield _sse({"type": "token", "text": token})

    yield _sse({"type": "done"})
