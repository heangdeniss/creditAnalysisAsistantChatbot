"""
RAG evaluation runner for the Credit Risk RAG API.

Examples:
  python BACKEND/eval/rag_eval.py --retrieval-only
  python BACKEND/eval/rag_eval.py --model llama-1b
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


DEFAULT_CASES = Path(__file__).with_name("eval_cases.jsonl")
REPORT_DIR = Path(__file__).with_name("reports")


def configure_runtime(*, force_cpu: bool) -> None:
    """Apply safe runtime defaults before importing heavy ML modules."""
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["FORCE_CPU"] = "1"


def _load_rag_dependencies():
    from rag_pipeline import NO_INFO, rag_query, retrieval_settings, _retrieve_citations

    return NO_INFO, rag_query, retrieval_settings, _retrieve_citations


def _load_trace_helpers():
    from observability import finish_trace, start_trace

    return start_trace, finish_trace


def load_eval_cases(path: str | Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    cases = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                case = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not case.get("id") or not case.get("question"):
                raise ValueError(f"Eval case on line {line_no} must include id and question.")
            cases.append(case)
    return cases


def _contains_any(text: str, needles: list[str]) -> bool:
    haystack = text.lower()
    return any(str(needle).lower() in haystack for needle in needles)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "into", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "with", "you", "your",
}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    haystack = answer.lower()
    hits = sum(1 for kw in expected_keywords if str(kw).lower() in haystack)
    return round(hits / len(expected_keywords), 4)


def _source_hit(chunks: list[dict], expected_sources: list[str]) -> bool:
    if not expected_sources:
        return bool(chunks)
    sources = " ".join(
        " ".join(str(c.get(key, "")) for key in ("source", "document_id", "chunk_id")).lower()
        for c in chunks
    )
    return any(str(src).lower() in sources for src in expected_sources)


def _abstained(answer: str, chunks: list[dict], no_info: str) -> bool:
    if not chunks:
        return True
    return no_info.lower() in answer.lower()


def _overlap_score(candidate: str, reference: str) -> float:
    candidate_tokens = _tokens(candidate)
    if not candidate_tokens:
        return 0.0
    reference_tokens = _tokens(reference)
    return round(len(candidate_tokens & reference_tokens) / len(candidate_tokens), 4)


def _relevance_score(
    case: dict[str, Any],
    answer: str,
    chunks: list[dict],
    no_info: str,
    keyword_recall: float,
) -> float:
    if bool(case.get("should_abstain", False)):
        return 1.0 if _abstained(answer, chunks, no_info) else 0.0

    expected_answer = str(case.get("expected_answer") or "")
    if expected_answer:
        return _overlap_score(expected_answer, answer)
    return keyword_recall


def _grounding_score(
    case: dict[str, Any],
    answer: str,
    chunks: list[dict],
    no_info: str,
    retrieval_hit: bool,
) -> float:
    if bool(case.get("should_abstain", False)):
        return 1.0 if _abstained(answer, chunks, no_info) else 0.0
    if not chunks or no_info.lower() in answer.lower():
        return 0.0

    evidence = " ".join(str(c.get("text") or c.get("snippet") or "") for c in chunks)
    evidence_overlap = _overlap_score(answer, evidence)
    if case.get("expected_sources"):
        return round((evidence_overlap + (1.0 if retrieval_hit else 0.0)) / 2, 4)
    return evidence_overlap


def score_case(case: dict[str, Any], answer: str, chunks: list[dict], no_info: str) -> dict[str, Any]:
    expected_keywords = list(case.get("expected_keywords") or [])
    expected_sources = list(case.get("expected_sources") or [])
    forbidden_phrases = list(case.get("forbidden_phrases") or [])
    should_abstain = bool(case.get("should_abstain", False))

    keyword_recall = _keyword_recall(answer, expected_keywords)
    retrieval_hit = True if should_abstain else _source_hit(chunks, expected_sources)
    forbidden_hit = _contains_any(answer, forbidden_phrases)
    abstention_ok = _abstained(answer, chunks, no_info) if should_abstain else not _abstained(answer, chunks, no_info)
    relevance_score = _relevance_score(case, answer, chunks, no_info, keyword_recall)
    grounding_score = _grounding_score(case, answer, chunks, no_info, retrieval_hit)

    passed = (
        relevance_score >= float(case.get("min_relevance", case.get("min_keyword_recall", 0.67)))
        and grounding_score >= float(case.get("min_grounding", 0.5))
        and not forbidden_hit
        and abstention_ok
    )

    return {
        "keyword_recall": keyword_recall,
        "relevance_score": relevance_score,
        "grounding_score": grounding_score,
        "retrieval_hit": retrieval_hit,
        "forbidden_hit": forbidden_hit,
        "abstention_ok": abstention_ok,
        "passed": passed,
    }


def run_eval_case(
    case: dict[str, Any],
    *,
    model_name: str,
    retrieval_only: bool = False,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> dict[str, Any]:
    no_info, rag_query, _, retrieve_citations = _load_rag_dependencies()
    start_trace, finish_trace = _load_trace_helpers()
    question = str(case["question"])
    trace = start_trace("/eval/rag", model=model_name, question=question)
    started = perf_counter()

    chunks: list[dict] = []
    if retrieval_only:
        chunks = retrieve_citations(
            question,
            trace=trace,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        answer = " ".join(str(c.get("text", "")) for c in chunks) if chunks else no_info
    else:
        result = rag_query(
            question=question,
            model_name=model_name,
            trace=trace,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        chunks = list(result.get("context") or chunks)
        answer = str(result.get("answer", ""))

    latency_ms = round((perf_counter() - started) * 1000, 2)
    scores = score_case(case, answer, chunks, no_info)
    finish_trace(trace, status="ok")

    return {
        "id": case["id"],
        "question": question,
        "answer": answer,
        "retrieved": [
            {
                "citation_id": c.get("citation_id"),
                "document_id": c.get("document_id"),
                "source": c.get("source"),
                "page": c.get("page"),
                "score": c.get("score"),
                "chunk_id": c.get("chunk_id"),
            }
            for c in chunks
        ],
        "latency_ms": latency_ms,
        **scores,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "cases": 0,
            "pass_rate": 0.0,
            "fail_rate": 0.0,
            "avg_relevance": 0.0,
            "avg_grounding": 0.0,
            "retrieval_hit_rate": 0.0,
            "avg_keyword_recall": 0.0,
            "abstention_accuracy": 0.0,
            "avg_latency_ms": 0.0,
        }

    return {
        "cases": total,
        "pass_rate": round(sum(1 for r in results if r["passed"]) / total, 4),
        "fail_rate": round(sum(1 for r in results if not r["passed"]) / total, 4),
        "avg_relevance": round(sum(float(r["relevance_score"]) for r in results) / total, 4),
        "avg_grounding": round(sum(float(r["grounding_score"]) for r in results) / total, 4),
        "retrieval_hit_rate": round(sum(1 for r in results if r["retrieval_hit"]) / total, 4),
        "avg_keyword_recall": round(sum(float(r["keyword_recall"]) for r in results) / total, 4),
        "abstention_accuracy": round(sum(1 for r in results if r["abstention_ok"]) / total, 4),
        "avg_latency_ms": round(sum(float(r["latency_ms"]) for r in results) / total, 2),
    }


def write_eval_report(report: dict[str, Any], output_path: str | Path | None = None) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = REPORT_DIR / f"rag_eval_{stamp}.json"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def run_eval(
    *,
    cases_path: str | Path = DEFAULT_CASES,
    model_name: str = "llama-1b",
    retrieval_only: bool = False,
    top_k: int | None = None,
    score_threshold: float | None = None,
    ablation: bool = False,
) -> dict[str, Any]:
    cases = load_eval_cases(cases_path)
    results = [
        run_eval_case(
            case,
            model_name=model_name,
            retrieval_only=retrieval_only,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        for case in cases
    ]
    _, _, retrieval_settings, _ = _load_rag_dependencies()
    settings = retrieval_settings()
    active_retrieval = {
        **settings,
        "top_k": int(top_k or settings["top_k"]),
        "score_threshold": settings["score_threshold"] if score_threshold is None else float(score_threshold),
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model": model_name,
        "retrieval_only": retrieval_only,
        "retrieval": active_retrieval,
        "summary": summarize(results),
        "cases": results,
    }
    if ablation:
        report["ablation"] = run_ablation(
            cases,
            model_name=model_name,
            retrieval_only=retrieval_only,
            base_retrieval=active_retrieval,
        )
    return report


def run_ablation(
    cases: list[dict[str, Any]],
    *,
    model_name: str,
    retrieval_only: bool,
    base_retrieval: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare the active retrieval config with one broader local setting."""
    base_top_k = int(base_retrieval["top_k"])
    base_threshold = float(base_retrieval["score_threshold"])
    configs = [
        {"name": "active", "top_k": base_top_k, "score_threshold": base_threshold},
        {
            "name": "broader",
            "top_k": max(base_top_k + 2, 6),
            "score_threshold": max(0.0, round(base_threshold - 0.1, 4)),
        },
    ]

    entries = []
    for cfg in configs:
        results = [
            run_eval_case(
                case,
                model_name=model_name,
                retrieval_only=retrieval_only,
                top_k=cfg["top_k"],
                score_threshold=cfg["score_threshold"],
            )
            for case in cases
        ]
        entries.append({
            "name": cfg["name"],
            "retrieval": cfg,
            "summary": summarize(results),
        })
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Run credit-risk RAG eval cases.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="Path to JSONL eval cases.")
    parser.add_argument("--model", default="llama-1b", choices=["llama-1b", "llama-3b"])
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM generation for a fast retrieval check.")
    parser.add_argument("--top-k", type=int, default=None, help="Override RAG_TOP_K for this eval run.")
    parser.add_argument("--score-threshold", type=float, default=None, help="Override RAG_SCORE_THRESHOLD for this eval run.")
    parser.add_argument("--ablation", action="store_true", help="Compare active retrieval settings with a broader retrieval set.")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Allow CUDA if available (may crash on incompatible drivers).",
    )
    parser.add_argument("--output", default="", help="Optional JSON report path.")
    args = parser.parse_args()

    configure_runtime(force_cpu=not args.gpu)

    report = run_eval(
        cases_path=args.cases,
        model_name=args.model,
        retrieval_only=args.retrieval_only,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        ablation=args.ablation,
    )
    output = write_eval_report(report, args.output or None)
    print(json.dumps(report["summary"], indent=2))
    print(f"Report written to: {output}")
    return 0 if report["summary"]["pass_rate"] >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
