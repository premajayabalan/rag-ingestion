
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pgvector import Vector

from pgvector.psycopg import register_vector

from src.database.connection import get_connection


load_dotenv()

MODEL_NAME = "gemini-embedding-2"
EMBEDDING_DIMENSION = 1536
TOP_K = 5


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from environment variables.")

    return genai.Client(api_key=api_key)


# Initialize the client once instead of creating it for every question.
client = get_gemini_client()


def generate_query_embedding(question: str):
    """
    Convert the user's question into a Gemini Embedding 2 vector.
    """

    query_text = f"task: search result | query: {question}"

    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=query_text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        ),
    )

    if not response.embeddings:
        raise RuntimeError("Gemini returned an empty embedding response.")

    embedding = response.embeddings[0].values

    if not embedding:
        raise RuntimeError("Gemini returned an empty query embedding.")

    if len(embedding) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Expected {EMBEDDING_DIMENSION} dimensions, "
            f"received {len(embedding)}."
        )

    return embedding


def retrieve_chunks(question: str, top_k: int = TOP_K):

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # 1. Generate the query embedding using Gemini.
    query_embedding = generate_query_embedding(generate_query_embedding(question))

    # 2. Search PostgreSQL using cosine distance.
    with get_connection() as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute("SET LOCAL hnsw.ef_search = 100")

            cur.execute(
                
                """
                SELECT
                    id,
                    content,
                    section_path,
                    metadata,
                    embedding <=> %s::vector AS distance
                FROM document_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
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

