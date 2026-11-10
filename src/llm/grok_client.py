
import os
import requests

from dotenv import load_dotenv

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_MODEL = os.getenv(
    "XAI_MODEL",
    "grok-4.20-0309-non-reasoning"
)

XAI_URL = "https://api.x.ai/v1/responses"


def grok_generate(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    timeout: int = 120,
) -> str:
    """
    Generate text using the xAI Grok Responses API.
    """

    if not XAI_API_KEY:
        raise ValueError("XAI_API_KEY is missing")

    selected_model = model or XAI_MODEL

    payload = {
        "model": selected_model,
        "input": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    }

    response = requests.post(
        XAI_URL,
        headers={
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )

    response.raise_for_status()

    data = response.json()

    # Extract generated text
    output_text = data.get("output_text")

    if output_text:
        return output_text.strip()

    # Fallback extraction
    texts = []

    for item in data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    texts.append(content.get("text", ""))

    result = "\n".join(texts).strip()

    if not result:
        raise RuntimeError(
            f"Grok returned no text. Response: {data}"
        )

    return result
