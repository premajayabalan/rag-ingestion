
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("XAI_API_KEY")
BASE_URL = "https://api.x.ai/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# 1. Check available embedding models
print("\n--- EMBEDDING MODELS ---")

response = requests.get(
    f"{BASE_URL}/embedding-models",
    headers=headers,
    timeout=30,
)

print("Embedding endpoint status:", response.status_code)

if response.ok:
    models = response.json().get("models", [])

    if not models:
        print("No embedding models returned.")

    for model in models:
        print("Model:", model.get("id"))
        print("Aliases:", model.get("aliases"))
        print("Input modalities:", model.get("input_modalities"))

    # 2. Test embedding with the first available model
    if models:
        model_id = models[0]["id"]

        print("\n--- TEST EMBEDDING ---")
        print("Testing model:", model_id)

        embedding_response = requests.post(
            f"{BASE_URL}/embeddings",
            headers=headers,
            json={
                "model": model_id,
                "input": "Fluid Intelligence helps organisations become AI-native."
            },
            timeout=60,
        )

        print("Embedding status:", embedding_response.status_code)

        if embedding_response.ok:
            data = embedding_response.json()
            vector = data["data"][0]["embedding"]

            print("Embedding successful!")
            print("Vector dimensions:", len(vector))
            print("First five values:", vector[:5])
        else:
            print("Embedding error:", embedding_response.text)

else:
    print("Embedding model discovery error:", response.text)

# 3. Check available general models
print("\n--- AVAILABLE MODELS ---")

response = requests.get(
    f"{BASE_URL}/models",
    headers=headers,
    timeout=30,
)

print("Models endpoint status:", response.status_code)

if response.ok:
    for model in response.json().get("data", []):
        print(model.get("id"))
