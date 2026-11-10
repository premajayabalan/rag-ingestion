from langgraph.graph import StateGraph, START, END

from src.ingestion.state import IngestionState

from src.ingestion.nodes import (
    extract_document_node,
    chunk_document_node,
    classify_document_node,
    generate_embeddings_node,
    store_chunks_node,
    handle_ingestion_error,
)


def check_extraction_status(state: IngestionState):
    return "error" if state.get("status") == "failed" else "chunk"


def check_chunking_status(state: IngestionState):
    return "error" if state.get("status") == "failed" else "classify"


def check_classification_status(state: IngestionState):
    return "error" if state.get("status") == "failed" else "embed"


def check_embedding_status(state: IngestionState):
    return "error" if state.get("status") == "failed" else "store"


def check_storage_status(state: IngestionState):
    return "error" if state.get("status") == "failed" else "complete"


def build_ingestion_graph():

    workflow = StateGraph(IngestionState)

    # Register nodes
    workflow.add_node("extract", extract_document_node)
    workflow.add_node("chunk", chunk_document_node)
    workflow.add_node("classify", classify_document_node)
    workflow.add_node("embed", generate_embeddings_node)
    workflow.add_node("store", store_chunks_node)
    workflow.add_node("error", handle_ingestion_error)

    # Start workflow
    workflow.add_edge(START, "extract")

    # Extraction -> Chunking or Error
    workflow.add_conditional_edges(
        "extract",
        check_extraction_status,
        {
            "chunk": "chunk",
            "error": "error",
        },
    )

    # Chunking -> Classification or Error
    workflow.add_conditional_edges(
        "chunk",
        check_chunking_status,
        {
            "classify": "classify",
            "error": "error",
        },
    )

    # Classification -> Embeddings or Error
    workflow.add_conditional_edges(
        "classify",
        check_classification_status,
        {
            "embed": "embed",
            "error": "error",
        },
    )

    # Embeddings -> Database storage or Error
    workflow.add_conditional_edges(
        "embed",
        check_embedding_status,
        {
            "store": "store",
            "error": "error",
        },
    )

    # Storage -> Complete or Error
    workflow.add_conditional_edges(
        "store",
        check_storage_status,
        {
            "complete": END,
            "error": "error",
        },
    )

    # Error handler
    workflow.add_edge("error", END)

    return workflow.compile()


ingestion_graph = build_ingestion_graph()
