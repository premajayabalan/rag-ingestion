"""Read-only adapter around the existing FI Bot RAG functions.

No backend modules are edited. Importing this adapter loads the same embedding
and reranker models as the current retrieval module, so run it on a machine
with the backend dependencies/models configured.
"""
from types import SimpleNamespace


def run_rag(question: str, conversation_history=None):
    # Lazy import prevents model/database initialization when only managing datasets.
    from src.retrival.rag_chat import (
        resolve_search_query, retrieve_chunks, generate_answer,
    )

    history = [
        item if hasattr(item, "role") and hasattr(item, "content")
        else SimpleNamespace(role=item.get("role", "user"), content=item.get("content", ""))
        for item in (conversation_history or [])
    ]
    search_query = resolve_search_query(question, history)
    chunks = retrieve_chunks(question, search_query=search_query)
    answer = generate_answer(question, chunks, history)
    contexts = [str(chunk[1]) for chunk in chunks]
    sources = [
        {
            "chunk_id": str(chunk[0]),
            "content": str(chunk[1]),
            "section": str(chunk[2]),
            "rrf_score": float(chunk[4]),
            "reranker_score": float(chunk[5]),
        }
        for chunk in chunks
    ]
    return {
        "question": question,
        "search_query": search_query,
        "answer": answer,
        "contexts": contexts,
        "sources": sources,
    }
