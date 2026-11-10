
from dataclasses import asdict, is_dataclass
from typing import Any

import json

from pgvector import Vector

from src.embedding.store_chunks import (
    get_gemini_client,
    generate_embeddings,
)

from src.database.connection import get_connection


from src.extract.pdf_extractor import (
    extract_document as extract_document_file,
)

from src.chunking.structure_aware_chunker import (
    StructureAwareChunker,
    classify_chunks,
)


def extract_document_node(state: dict[str, Any]) -> dict:
    """Extract document content using Docling."""

    try:
        file_path = state.get("file_path")

        if not file_path:
            raise ValueError("Document file path is missing.")

        print(f"Extracting document: {file_path}")

        extracted_content = extract_document_file(file_path)

        if not extracted_content or not extracted_content.strip():
            raise ValueError("Document extraction returned empty content.")

        print("Document extraction completed.")

        return {
            "extracted_content": extracted_content,
            "status": "extracted",
            "error": None,
        }

    except Exception as error:
        print(f"Extraction failed: {error}")

        return {
            "status": "failed",
            "error": str(error),
        }


def chunk_document_node(state: dict[str, Any]) -> dict:
    """Create structure-aware chunks from extracted content."""

    try:
        extracted_content = state.get("extracted_content")

        if not extracted_content:
            raise ValueError("Extracted content is missing.")

        print("Starting document chunking...")

        chunker = StructureAwareChunker(
            chunk_size=600,
            chunk_overlap=100,
        )

        chunks = chunker.create_chunks(
            extracted_content,
            document_id=state.get("document_id"),
            source_file=state.get("filename"),
        )

        if not chunks:
            raise ValueError("Chunking produced no chunks.")

        print(f"Chunking completed. Total chunks: {len(chunks)}")

        return {
            "chunks": chunks,
            "status": "chunked",
            "error": None,
        }

    except Exception as error:
        print(f"Chunking failed: {error}")

        return {
            "status": "failed",
            "error": str(error),
        }


def classify_document_node(state: dict[str, Any]) -> dict:
    """Classify chunks using the existing classification pipeline."""

    try:
        chunks = state.get("chunks")

        if not chunks:
            raise ValueError("No chunks available for classification.")

        print(f"Classifying {len(chunks)} chunks...")

        classification_time = classify_chunks(chunks)

        serialized_chunks = [
            asdict(chunk) if is_dataclass(chunk) else chunk
            for chunk in chunks
        ]

        classification_counts = {
            "PUBLIC": 0,
            "INTERNAL": 0,
            "CONFIDENTIAL": 0,
        }

        for chunk in serialized_chunks:
            metadata = chunk.get("metadata", {})
            classification = metadata.get("classification")

            if classification in classification_counts:
                classification_counts[classification] += 1

        print("Classification completed.")

        return {
            "chunks": serialized_chunks,
            "classification_summary": {
                "total_chunks": len(serialized_chunks),
                "classification_counts": classification_counts,
                "classification_time_seconds": classification_time,
            },
            "status": "classified",
            "error": None,
        }

    except Exception as error:
        print(f"Classification failed: {error}")

        return {
            "status": "failed",
            "error": str(error),
        }

def generate_embeddings_node(state: dict[str, Any]) -> dict:
    """Generate Gemini embeddings for classified chunks."""

    try:
        chunks = state.get("chunks")

        if not chunks:
            raise ValueError("No classified chunks available.")

        print(f"Generating embeddings for {len(chunks)} chunks...")

        client = get_gemini_client()

        embeddings = generate_embeddings(client, chunks)

        if len(embeddings) != len(chunks):
            raise ValueError("Embedding count does not match chunk count.")

        print(f"Generated {len(embeddings)} embeddings.")

        return {
            "embeddings": embeddings,
            "status": "embedded",
            "error": None,
        }

    except Exception as error:
        print(f"Embedding generation failed: {error}")

        return {
            "status": "failed",
            "error": str(error),
        }

def store_chunks_node(state: dict[str, Any]) -> dict:
    """Store classified chunks and Gemini embeddings in PostgreSQL."""

    try:
        chunks = state.get("chunks")
        embeddings = state.get("embeddings")

        if not chunks or not embeddings:
            raise ValueError("Chunks or embeddings are missing.")

        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts do not match.")

        print(f"Storing {len(chunks)} chunks in PostgreSQL...")

        with get_connection() as conn:

            with conn.cursor() as cur:

                for chunk, embedding in zip(chunks, embeddings):

                    if len(embedding) != 1536:
                        raise ValueError(
                            "Embedding must contain exactly 1536 dimensions."
                        )

                    vector = str(Vector(embedding))

                    cur.execute(
                        """
                        INSERT INTO document_chunks (
                            document_id,
                            chunk_id,
                            chunk_index,
                            content,
                            section_path,
                            content_type,
                            token_count,
                            metadata,
                            embedding
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s,
                            %s, %s::vector
                        )
                        """,
                        (
                            chunk.get("document_id"),
                            chunk.get("chunk_id"),
                            chunk.get("chunk_index"),
                            chunk["content"],
                            chunk.get("section_path"),
                            chunk.get("content_type"),
                            chunk.get("token_count"),
                            json.dumps(chunk.get("metadata", {})),
                            vector,
                        ),
                    )

        print(f"Successfully stored {len(chunks)} chunks.")

        return {
            "stored_chunk_count": len(chunks),
            "status": "completed",
            "error": None,
        }

    except Exception as error:
        print(f"Database storage failed: {error}")

        return {
            "status": "failed",
            "error": str(error),
        }


def handle_ingestion_error(state: dict[str, Any]) -> dict:
    """Record ingestion failure details."""

    error = state.get("error") or "Unknown ingestion error."

    print(f"Ingestion workflow failed: {error}")

    return {
        "status": "failed",
        "error": error,
    }
