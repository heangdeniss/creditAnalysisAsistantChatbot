"""
rag_pipeline.py
Orchestrates retrieval from ChromaDB and generation via Llama 3.2 1B.
Generation settings are fixed in model_loader.py — not passed from the API.
"""
from __future__ import annotations

import json
import os
import re
import sys
from threading import Event
from time import perf_counter
from typing import Generator

sys.path.insert(0, os.path.dirname(__file__))

from chroma_loader import load_chroma_db
from model_loader import generate, generate_stream, QUEUED_SENTINEL
from observability import add_event, finish_trace, set_trace_section

# Config
TOP_K           = 4
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.45"))
NO_INFO         = "I don't have enough information in my knowledge base to answer that."
_CITATION_RE    = re.compile(r"\[S\d+\]")
_SOURCE_PATH_RE = re.compile(
    r"(?:[\w .-]+[\\/])+[\w .()-]+\.(?:pdf|txt|csv|docx?|xlsx?|json|jsonl|md)",
    re.IGNORECASE,
)

# Used for normal chat questions (no borrower data)
RAG_SYSTEM_PROMPT = (
    "You are a professional credit risk assistant. "
    "Answer the question using only the context provided. Be concise and stop when done — do not add extra sections, conclusions, or additional information. "
    "When you use a retrieved source, cite it with its bracketed source id such as [S1] or [S2]. "
    "Only cite source ids that appear in the provided context. "
    "Do not mention source file names, file paths, folders, PDFs, text files, or training files in the answer. "
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
def _metadata_value(metadata: dict, *keys: str):
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _citation_from_doc(doc, index: int, score: float | None = None) -> dict:
    metadata = doc.metadata if isinstance(getattr(doc, "metadata", None), dict) else {}
    text = str(getattr(doc, "page_content", "") or "")
    source = str(_metadata_value(metadata, "source", "file_name", "filename", "path") or "knowledge-base")
    page = _metadata_value(metadata, "page", "page_number")
    title = _metadata_value(metadata, "title", "document_title")
    chunk_id = _metadata_value(metadata, "chunk_id", "id")
    if not chunk_id:
        page_part = page if page is not None else "na"
        chunk_id = f"{source}:{page_part}:{index}"

    return {
        "citation_id": f"S{index}",
        "text": text,
        "snippet": text[:600],
        "source": source,
        "page": page,
        "title": title,
        "chunk_id": str(chunk_id),
        "score": round(float(score), 4) if isinstance(score, (int, float)) else None,
    }


def _retrieve_citations(question: str, trace: dict | None = None) -> list[dict]:
    """Retrieve chunks with metadata and relevance scores when available."""
    started = perf_counter()
    db = load_chroma_db()
    pairs = []
    try:
        pairs = db.similarity_search_with_relevance_scores(
            question,
            k=TOP_K,
            score_threshold=SCORE_THRESHOLD,
        )
    except TypeError:
        pairs = db.similarity_search_with_relevance_scores(question, k=TOP_K)
        pairs = [(doc, score) for doc, score in pairs if score is None or score >= SCORE_THRESHOLD]
    except Exception as exc:
        add_event(trace, "retrieval_score_fallback", {"error": str(exc)})
        docs = get_retriever().invoke(question)
        pairs = [(doc, None) for doc in docs]

    chunks = [
        _citation_from_doc(doc, idx, score)
        for idx, (doc, score) in enumerate(pairs, start=1)
    ]

    set_trace_section(trace, "retrieval", {
        "latency_ms": round((perf_counter() - started) * 1000, 2),
        "top_k": TOP_K,
        "score_threshold": SCORE_THRESHOLD,
        "chunk_count": len(chunks),
        "scores": [c["score"] for c in chunks if c.get("score") is not None],
        "sources": [c["source"] for c in chunks],
    })
    return chunks


def _retrieve(question: str) -> list[str]:
    return [chunk["text"] for chunk in _retrieve_citations(question)]


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for chunk in chunks:
        header = f"[{chunk['citation_id']}] Source: Credit risk knowledge base"
        blocks.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(blocks)


def _citation_suffix(answer: str, chunks: list[dict]) -> str:
    if not chunks:
        return ""
    if NO_INFO.lower() in answer.lower():
        return ""
    if _CITATION_RE.search(answer):
        return ""
    return f" [{chunks[0]['citation_id']}]"


def _postprocess_cited_answer(answer: str, chunks: list[dict]) -> str:
    cleaned = re.sub(r"(?m)^\s*\*\s+", "- ", answer).strip()
    cleaned = _SOURCE_PATH_RE.sub("the knowledge base", cleaned)
    return f"{cleaned}{_citation_suffix(cleaned, chunks)}"


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# Public API
def rag_query(
    question: str,
    facts: str = "",
    history: list[dict] | None = None,
    model_name: str | None = None,
    trace: dict | None = None,
) -> dict[str, object]:
    """Blocking RAG: retrieve → generate → return dict."""
    if facts:
        answer = generate(
            system_prompt=EXPLAIN_SYSTEM_PROMPT,
            user_message=EXPLAIN_USER_TEMPLATE.format(facts=facts, question=question),
            model_name=model_name,
            trace=trace,
        )
        return {"question": question, "context": [], "answer": answer}

    chunks = _retrieve_citations(question, trace=trace)
    if not chunks:
        return {"question": question, "context": [], "answer": NO_INFO}
    answer = generate(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_message=RAG_USER_TEMPLATE.format(context=_format_context(chunks), question=question),
        history=history,
        model_name=model_name,
        trace=trace,
    )
    answer = _postprocess_cited_answer(answer, chunks)
    return {"question": question, "context": chunks, "answer": answer}


def rag_stream(
    question: str,
    facts: str = "",
    history: list[dict] | None = None,
    model_name: str | None = None,
    stop_event: Event | None = None,
    trace: dict | None = None,
) -> Generator[str, None, None]:
    """
    Streaming RAG as Server-Sent Events.
      {"type": "context", "chunks": [...]}   sent first
      {"type": "token",   "text":  str}      one per token
      {"type": "done"}                       end of stream
    """
    status = "ok"
    error: str | None = None
    try:
        if facts:
            yield _sse({"type": "context", "chunks": []})
            for token in generate_stream(
                system_prompt=EXPLAIN_SYSTEM_PROMPT,
                user_message=EXPLAIN_USER_TEMPLATE.format(facts=facts, question=question),
                model_name=model_name,
                stop_event=stop_event,
                trace=trace,
            ):
                if token == QUEUED_SENTINEL:
                    yield _sse({"type": "queued"})
                    continue
                yield _sse({"type": "token", "text": token})
            yield _sse({"type": "done"})
            return

        chunks = _retrieve_citations(question, trace=trace)
        yield _sse({"type": "context", "chunks": chunks})

        if not chunks:
            yield _sse({"type": "token", "text": NO_INFO})
            yield _sse({"type": "done"})
            return

        emitted_text: list[str] = []
        for token in generate_stream(
            system_prompt=RAG_SYSTEM_PROMPT,
            user_message=RAG_USER_TEMPLATE.format(context=_format_context(chunks), question=question),
            history=history,
            model_name=model_name,
            stop_event=stop_event,
            trace=trace,
        ):
            if token == QUEUED_SENTINEL:
                yield _sse({"type": "queued"})
                continue
            emitted_text.append(token)
            yield _sse({"type": "token", "text": token})

        suffix = _citation_suffix("".join(emitted_text), chunks)
        if suffix:
            yield _sse({"type": "token", "text": suffix})

        yield _sse({"type": "done"})
    except Exception as exc:
        status = "error"
        error = str(exc)
        yield _sse({"type": "error", "message": error})
    finally:
        if stop_event and stop_event.is_set() and status == "ok":
            status = "aborted"
        finish_trace(trace, status=status, error=error)
