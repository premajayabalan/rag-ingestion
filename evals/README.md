# FI Bot RAG Evaluation (isolated add-on)

This folder adds DeepEval and LangSmith runners around the existing RAG functions. No files under `src/` are changed by this integration. The FastAPI routes, request/response schema, retrieval, chunking, classification, embedding, database schema, and Grok client are untouched.

## What is evaluated

The adapter calls the current sequence: follow-up query resolution -> hybrid vector/keyword retrieval with RRF -> BGE reranking -> Grok answer generation. It captures the rewritten query, answer, retrieved contexts, and source scores. Evaluation scripts call this adapter directly; they do not start or instrument FastAPI.

## 1. Install

Activate the same Python virtual environment used by the backend, then install its existing dependencies as usual and run:

```bash
pip install -r requirements-evals.txt
```

Do not replace the existing backend dependency file. `requirements-evals.txt` is deliberately separate.

## 2. Configure credentials

Copy `.env.evals.example` values into your local `.env` (do not commit secrets). For LangSmith set `LANGSMITH_API_KEY`. DeepEval's built-in metrics need an LLM judge; the default configuration generally uses OpenAI, so configure `OPENAI_API_KEY` or explicitly configure a supported custom judge model as described in the DeepEval documentation. The FI Bot answer-generation model remains Grok; the judge model is separate from the model being evaluated.

LangSmith tracing of the production API is not enabled by this integration. LangSmith is used for standalone evaluation datasets/experiments only, so backend instrumentation is not required.

## 3. Prepare a trusted dataset

Edit `evals/datasets/evaluation_dataset.json`. Each enabled row must have:

- `id`: stable case identifier
- `input`: user question
- `expected_output`: human-verified reference answer
- `reference_context`: one or more exact passages from the knowledge base that support the reference answer
- `conversation_history`: optional list of `{ "role": "user|assistant", "content": "..." }`
- `enabled`: set to `true` only after review

The included row is intentionally disabled and is only a schema template. Do not evaluate using fabricated answers or unverified references. Include answerable, multi-hop, paraphrased, ambiguous, and unanswerable questions. For unanswerable cases, write a reference answer that explicitly says the supplied knowledge base does not contain the answer.

## 4. Run DeepEval

From the repository root:

```bash
python -m evals.deepeval.evaluate_rag
```

Metrics configured: Answer Relevancy, Faithfulness, Contextual Precision, Contextual Recall, and Hallucination. These use a separate judge LLM and can incur API cost. DeepEval may print its detailed metric table; a run summary is saved under `evals/reports/`.

## 5. Run LangSmith

```bash
python -m evals.langsmith.run_evaluation
```

The runner creates or reuses the named LangSmith dataset, syncs enabled test questions/reference outputs, then runs the existing RAG pipeline and records standalone experiment results. The included evaluators are operational checks (non-empty answer and non-empty retrieval), not semantic quality metrics. Use DeepEval for the configured semantic metrics, or add reviewed LangSmith LLM-as-judge evaluators as a later enhancement.

## Safety / operational notes

- The evaluation runner performs retrieval and Grok generation, so it requires the same database, model files, environment variables, and network access as the backend.
- It only reads from PostgreSQL through the existing retrieval function. It does not ingest, update, or delete database rows.
- Evaluation runs make external LLM calls and may incur Grok and judge-provider charges.
- Importing `src.retrival.rag_chat` loads the existing embedding and reranker models, which can require substantial RAM/VRAM and startup time.
- The adapter currently uses the existing global retrieval defaults (candidate pool 20 and final top 5). It does not change backend settings.
- DeepEval and LangSmith package APIs evolve. Install in a separate virtual environment first if you want to validate dependency compatibility before adding them to your backend environment.
- This is an offline evaluation integration, not production observability or live-request tracing.
