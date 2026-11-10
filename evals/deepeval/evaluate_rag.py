"""Run DeepEval against the existing RAG pipeline without changing its API."""
import json
from datetime import datetime

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from evals.common.evaluation_config import DATASET_PATH, REPORT_DIR
from evals.common.rag_adapter import run_rag
from evals.deepeval.metrics import build_metrics


def load_cases():
    rows = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return [row for row in rows if row.get("enabled", True) and row.get("input") and row.get("expected_output")]


def main():
    rows = load_cases()
    if not rows:
        raise SystemExit(f"No enabled, verified test cases found in {DATASET_PATH}. See evals/README.md.")

    test_cases = []
    for row in rows:
        result = run_rag(row["input"], row.get("conversation_history", []))
        test_cases.append(LLMTestCase(
            input=row["input"],
            actual_output=result["answer"],
            expected_output=row["expected_output"],
            retrieval_context=result["contexts"],
            context=row.get("reference_context", []),
        ))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    # DeepEval prints a detailed table and returns evaluation results.
    results = evaluate(test_cases=test_cases, metrics=build_metrics())
    out = REPORT_DIR / f"deepeval-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    serializable = {"test_cases": len(test_cases), "results": str(results)}
    out.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"DeepEval report saved: {out}")


if __name__ == "__main__":
    main()
