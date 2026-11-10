
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field

from src.retrival.rag_chat import (retrieve_chunks, generate_answer,resolve_search_query)


app = FastAPI(title="Fluid Intelligence RAG API")


# Allow requests from our Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    conversation_history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    follow_up_question: Optional[str] = None


@app.get("/")
def home():
    return {"message": "Fluid Intelligence RAG API is running"}



@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        # 1. Resolve follow-up questions using conversation history
        search_query = resolve_search_query(
            question,
            request.conversation_history
        )

        # 2. Retrieve chunks using hybrid search
        retrieved_chunks = retrieve_chunks(
            question,
            search_query=search_query
        )

        # 3. Generate answer using original question and history
        result = generate_answer(
                    question,
                    retrieved_chunks,
                    request.conversation_history
                )

        # 4. Prepare source information
        sources = []

        for rank, chunk in enumerate(retrieved_chunks, start=1):

            chunk_id, content, section_path, metadata, rrf_score, reranker_score = chunk

            sources.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "section": section_path,
                "rrf_score": round(float(rrf_score), 6),
                "reranker_score": round(float(reranker_score), 4),
                "content": content
            })

        
        return {
            "answer": result["answer"],
            "follow_up_question": result["follow_up_question"]
        }

    except Exception as error:
        print(f"Chat API error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Something went wrong while generating the answer."
        )
