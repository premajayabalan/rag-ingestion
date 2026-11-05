import json
from pathlib import Path

from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from src.database.connection import get_connection


CHUNKS_FILE = Path("output/chunks/chunks.json")
MODEL_NAME = "microsoft/harrier-oss-v1-270m"


def load_chunks():
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def store_chunks():
    chunks = load_chunks()

    if not chunks:
        print("No chunks found.")
        return

    print(f"Loaded {len(chunks)} chunks.")

    model = SentenceTransformer(MODEL_NAME)
    texts = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    with get_connection() as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            for chunk, embedding in zip(chunks, embeddings):
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
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        embedding,
                    ),
                )

    print(f"Successfully stored {len(chunks)} chunks in pgvector.")


if __name__ == "__main__":
    store_chunks()
