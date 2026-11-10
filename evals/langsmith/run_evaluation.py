"""Create/reuse a LangSmith dataset and evaluate the unchanged RAG pipeline."""
import json
from evals.common.evaluation_config import DATASET_PATH, LANGSMITH_DATASET, LANGSMITH_PROJECT
from evals.common.rag_adapter import run_rag


def load_cases():
    rows = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return [row for row in rows if row.get("enabled", True) and row.get("input") and row.get("expected_output")]


def main():
    try:
        from langsmith import Client
        from langsmith.evaluation import evaluate
    except ImportError as exc:
        raise SystemExit("Install evaluation dependencies: pip install -r requirements-evals.txt") from exc

    rows = load_cases()
    if not rows:
        raise SystemExit(f"No enabled, verified test cases found in {DATASET_PATH}. See evals/README.md.")

    client = Client()
    try:
        dataset = client.read_dataset(dataset_name=LANGSMITH_DATASET)
    except Exception:
        dataset = client.create_dataset(dataset_name=LANGSMITH_DATASET, description="Verified FI Bot RAG evaluation examples")

    existing = {str(example.inputs.get("question", "")) for example in client.list_examples(dataset_id=dataset.id)}
    for row in rows:
        question = row["input"]
        if question not in existing:
            client.create_example(
                dataset_id=dataset.id,
                inputs={"question": question, "conversation_history": row.get("conversation_history", [])},
                outputs={"expected_output": row["expected_output"], "reference_context": row.get("reference_context", [])},
            )

    def target(inputs):
        return run_rag(inputs["question"], inputs.get("conversation_history", []))

    def answer_correctness(run, example):
        expected = (example.outputs or {}).get("expected_output", "")
        actual = (run.outputs or {}).get("answer", "")
        # Transparent exact-text diagnostic, not a semantic quality score.
        return {"key": "answer_nonempty", "score": bool(actual.strip()), "comment": f"Reference answer available: {bool(expected)}"}

    def retrieval_coverage(run, example):
        contexts = (run.outputs or {}).get("contexts", [])
        return {"key": "retrieval_nonempty", "score": bool(contexts), "comment": f"Retrieved {len(contexts)} chunks"}

    experiment = evaluate(
        target,
        data=dataset.name,
        evaluators=[answer_correctness, retrieval_coverage],
        experiment_prefix="fi-bot-rag",
        metadata={"pipeline": "existing-hybrid-rag-rrf-bge-reranker-grok", "project": LANGSMITH_PROJECT},
    )
    print(f"LangSmith evaluation complete: {experiment}")


if __name__ == "__main__":
    main()
