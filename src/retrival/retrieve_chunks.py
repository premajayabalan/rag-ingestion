
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from src.database.connection import get_connection


MODEL_NAME = "microsoft/harrier-oss-v1-270m"
TOP_K = 5

# Load the model once, rather than every time a question is asked
model = SentenceTransformer(MODEL_NAME)


def retrieve_chunks(question: str, top_k: int = TOP_K):
    # 1. Convert the user's question into an embedding
    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    # 2. Search pgvector for the closest chunks
    with get_connection() as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    content,
                    section_path,
                    metadata,
                    embedding <=> %s AS distance
                FROM document_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (query_embedding, query_embedding, top_k),
            )

            results = cur.fetchall()

    return results


def main():
    question = input("Ask a question about your documents: ").strip()

    if not question:
        print("Please enter a question.")
        return

    results = retrieve_chunks(question)

    if not results:
        print("No relevant chunks found.")
        return

    print(f"\nRetrieved {len(results)} relevant chunks:\n")

    for rank, result in enumerate(results, start=1):
        chunk_id, content, section_path, metadata, distance = result
        similarity = 1 - distance

        print(f"--- Result {rank} ---")
        print(f"Chunk ID: {chunk_id}")
        print(f"Section: {section_path}")
        print(f"Cosine similarity: {similarity:.4f}")
        print(f"Content:\n{content}\n")


if __name__ == "__main__":
    main()
