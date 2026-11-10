
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("XAI_API_KEY")
model = os.getenv("XAI_MODEL", "grok-4.7")

if not api_key:
    raise ValueError("XAI_API_KEY not found in .env")

response = requests.post(
    "https://api.x.ai/v1/responses",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json={
        "model": model,
        "input": "Reply with: Grok API connection successful!"
    },
    timeout=60,
)

print("Status:", response.status_code)
response.raise_for_status()

data = response.json()

for item in data.get("output", []):
    if item.get("type") == "message":
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                print("Grok response:", content["text"])
