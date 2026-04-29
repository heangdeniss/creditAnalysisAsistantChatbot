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
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from rag_pipeline import NO_INFO, rag_query, _retrieve_citations  # noqa: E402
from observability import finish_trace, start_trace  # noqa: E402


DEFAULT_CASES = Path(__file__).with_name("eval_cases.jsonl")
REPORT_DIR = Path(__file__).with_name("reports")


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


def _keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    haystack = answer.lower()
    hits = sum(1 for kw in expected_keywords if str(kw).lower() in haystack)
    return round(hits / len(expected_keywords), 4)


def _source_hit(chunks: list[dict], expected_sources: list[str]) -> bool:
    if not expected_sources:
        return bool(chunks)
    sources = " ".join(str(c.get("source", "")).lower() for c in chunks)
    return any(str(src).lower() in sources for src in expected_sources)


def _abstained(answer: str, chunks: list[dict]) -> bool:
    if not chunks:
        return True
    return NO_INFO.lower() in answer.lower()


def score_case(case: dict[str, Any], answer: str, chunks: list[dict]) -> dict[str, Any]:
    expected_keywords = list(case.get("expected_keywords") or [])
    expected_sources = list(case.get("expected_sources") or [])
    forbidden_phrases = list(case.get("forbidden_phrases") or [])
    should_abstain = bool(case.get("should_abstain", False))

    keyword_recall = _keyword_recall(answer, expected_keywords)
    retrieval_hit = True if should_abstain else _source_hit(chunks, expected_sources)
    forbidden_hit = _contains_any(answer, forbidden_phrases)
    abstention_ok = _abstained(answer, chunks) if should_abstain else not _abstained(answer, chunks)

    passed = (
        keyword_recall >= float(case.get("min_keyword_recall", 0.67))
        and retrieval_hit
        and not forbidden_hit
        and abstention_ok
    )

    return {
        "keyword_recall": keyword_recall,
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
) -> dict[str, Any]:
    question = str(case["question"])
    trace = start_trace("/eval/rag", model=model_name, question=question)
    started = perf_counter()

    chunks: list[dict] = []
    if retrieval_only:
        chunks = _retrieve_citations(question, trace=trace)
        answer = " ".join(str(c.get("text", "")) for c in chunks) if chunks else NO_INFO
    else:
        result = rag_query(question=question, model_name=model_name, trace=trace)
        chunks = list(result.get("context") or chunks)
        answer = str(result.get("answer", ""))

    latency_ms = round((perf_counter() - started) * 1000, 2)
    scores = score_case(case, answer, chunks)
    finish_trace(trace, status="ok")

    return {
        "id": case["id"],
        "question": question,
        "answer": answer,
        "retrieved": [
            {
                "citation_id": c.get("citation_id"),
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
            "retrieval_hit_rate": 0.0,
            "avg_keyword_recall": 0.0,
            "abstention_accuracy": 0.0,
            "avg_latency_ms": 0.0,
        }

    return {
        "cases": total,
        "pass_rate": round(sum(1 for r in results if r["passed"]) / total, 4),
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
) -> dict[str, Any]:
    cases = load_eval_cases(cases_path)
    results = [
        run_eval_case(case, model_name=model_name, retrieval_only=retrieval_only)
        for case in cases
    ]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model": model_name,
        "retrieval_only": retrieval_only,
        "summary": summarize(results),
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run credit-risk RAG eval cases.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="Path to JSONL eval cases.")
    parser.add_argument("--model", default="llama-1b", choices=["llama-1b", "llama-3b"])
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM generation for a fast retrieval check.")
    parser.add_argument("--output", default="", help="Optional JSON report path.")
    args = parser.parse_args()

    report = run_eval(cases_path=args.cases, model_name=args.model, retrieval_only=args.retrieval_only)
    output = write_eval_report(report, args.output or None)
    print(json.dumps(report["summary"], indent=2))
    print(f"Report written to: {output}")
    return 0 if report["summary"]["pass_rate"] >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
