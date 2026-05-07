import os
import pytest

from eval.rag_eval import load_eval_cases, run_eval, score_case, summarize


def test_rag_eval_scoring_smoke():
    cases = load_eval_cases()
    assert len(cases) > 0
    case = cases[0]
    scores = score_case(case, "Sample answer", [], "no info")
    summary = summarize([
        {
            "passed": scores["passed"],
            "relevance_score": scores["relevance_score"],
            "grounding_score": scores["grounding_score"],
            "retrieval_hit": scores["retrieval_hit"],
            "keyword_recall": scores["keyword_recall"],
            "abstention_ok": scores["abstention_ok"],
            "latency_ms": 10.0,
        }
    ])
    assert "avg_relevance" in summary
    assert "avg_grounding" in summary


@pytest.mark.skipif(os.getenv("RUN_RAG_EVAL") != "1", reason="RUN_RAG_EVAL not enabled")
def test_rag_eval_runner_smoke():
    report = run_eval(retrieval_only=True, limit=2)
    assert "summary" in report
    assert report["summary"]["cases"] >= 1
