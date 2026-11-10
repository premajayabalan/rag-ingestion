
from src.llm.grok_client import grok_generate

answer = grok_generate(
    system_prompt="You are a helpful assistant.",
    user_prompt="Explain RAG in two simple sentences."
)

print("\nGrok answer:")
print(answer)
