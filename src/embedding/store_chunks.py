
import json
import os
import math
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from pgvector.psycopg import register_vector

from src.database.connection import get_connection


load_dotenv()

CHUNKS_FILE = Path("output/chunks/chunks.json")

MODEL_NAME = "gemini-embedding-2"
EMBEDDING_DIMENSION = 1536

BATCH_SIZE = 50


def load_chunks():
    """Load classified chunks from the JSON file."""

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_gemini_client():
    """Create Google Gemini API client."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing from environment variables."
        )

    return genai.Client(api_key=api_key)


def prepare_document(chunk):
    """
    Format a document chunk for Gemini Embedding 2.

    Section path is used as the document title.
    """

    title = chunk.get("section_path") or "none"
    content = chunk["content"]

    return f"title: {title} | text: {content}"


def generate_embeddings(client, chunks):
    """Generate embeddings in batches."""

    all_embeddings = []

    total_chunks = len(chunks)

    for start in range(0, total_chunks, BATCH_SIZE):

        batch_chunks = chunks[start:start + BATCH_SIZE]

        formatted_contents = [
            types.Content(
                parts=[
                    types.Part.from_text(
                        text=prepare_document(chunk)
                    )
                ]
            )
            for chunk in batch_chunks
        ]

        print(
            f"Generating embeddings: "
            f"{start + 1}-{start + len(batch_chunks)} "
            f"of {total_chunks}"
        )

        response = client.models.embed_content(
            model=MODEL_NAME,
            contents=formatted_contents,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSION
            ),
        )

        if not response.embeddings:
            raise RuntimeError(
                "Gemini returned an empty embedding response."
            )

        if len(response.embeddings) != len(batch_chunks):
            raise RuntimeError(
                "Embedding count does not match chunk count."
            )

        for item in response.embeddings:

            values = item.values

            if not values:
                raise RuntimeError(
                    "Gemini returned an empty embedding vector."
                )

            if len(values) != EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"Expected {EMBEDDING_DIMENSION} dimensions, "
                    f"received {len(values)}."
                )

            # Gemini Embedding 2 automatically normalizes
            # embeddings when using 1536 dimensions.
            all_embeddings.append(values)

    return all_embeddings


def store_chunks():

    chunks = load_chunks()

    if not chunks:
        print("No chunks found.")
        return

    print(f"Loaded {len(chunks)} chunks.")

    client = get_gemini_client()

    # Generate embeddings before opening the database transaction.
    embeddings = generate_embeddings(client, chunks)

    print(f"Generated {len(embeddings)} embeddings.")

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

    print(
        f"Successfully stored {len(chunks)} chunks "
        f"with Gemini embeddings in pgvector."
    )


if __name__ == "__main__":
    store_chunks()
