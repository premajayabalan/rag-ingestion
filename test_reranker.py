
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)

question = "What happens during the AI Readiness Assessment?"

chunks = [
    "The AI Readiness Assessment evaluates an organisation's AI literacy and AI maturity across eight dimensions.",

    "Fluid Intelligence helps organisations build AI-powered solutions and transform their business.",

    "The company provides training programmes to help employees improve their AI skills.",

    "The assessment identifies an organisation's readiness zone, bottlenecks and provides a roadmap of next steps."
]

# Create question and chunk pairs
pairs = [
    [question, chunk]
    for chunk in chunks
]

# Calculate relevance scores
scores = reranker.predict(pairs)

# Display results in descending relevance order
ranked_results = sorted(
    zip(scores, chunks),
    key=lambda x: x[0],
    reverse=True
)

for score, chunk in ranked_results:
    print(f"\nScore: {score:.4f}")
    print(f"Chunk: {chunk}")
