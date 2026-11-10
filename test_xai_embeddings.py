
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("XAI_API_KEY")
BASE_URL = "https://api.x.ai/v1"

if not API_KEY:
    raise ValueError("XAI_API_KEY is missing from .env")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


# STEP 1: Check available embedding models
print("\nChecking available xAI embedding models...")

models_response = requests.get(
    f"{BASE_URL}/embedding-models",
    headers=headers,
    timeout=30
)

print("Model list status:", models_response.status_code)
models_response.raise_for_status()

models_data = models_response.json()
available_models = models_data.get("models", [])

if available_models:
    print("\nAvailable embedding models:")

    for model in available_models:
        print("Model ID:", model.get("id"))
        print("Input modalities:", model.get("input_modalities"))
        print("-" * 40)

    model_name = (
        os.getenv("XAI_EMBEDDING_MODEL")
        or available_models[0]["id"]
    )

else:
    print("\nNo embedding models listed for this API key.")

    # Probe the documented example model ID.
    # You can override this using XAI_EMBEDDING_MODEL in .env.
    model_name = os.getenv("XAI_EMBEDDING_MODEL", "v1")


# STEP 2: Actually generate an embedding
payload = {
    "model": "grok-embedding-small",
    "input": ["Testing Grok embedding generation"],
    "encoding_format": "float"
}

response = requests.post(
    "https://api.x.ai/v1/embeddings",
    headers=headers,
    json=payload,
    timeout=60
)

print("Status:", response.status_code)
print(response.text)

if response.ok:
    data = response.json()
    embedding = data["data"][0]["embedding"]
    print("Dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])
