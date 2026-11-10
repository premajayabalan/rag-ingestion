
from typing import TypedDict, Optional, Any


class IngestionState(TypedDict):
    # Document identification
    document_id: str
    filename: str
    file_path: Optional[str]
    r2_object_key: Optional[str]

    # Extraction
    extracted_content: Optional[str]

    # Chunking
    chunks: Optional[list[dict[str, Any]]]

    # Classification
    classification_summary: Optional[dict[str, Any]]

    # Embeddings and database
    embeddings: Optional[list[list[float]]]
    stored_chunk_count: int

    # Workflow status
    status: str
    error: Optional[str]
