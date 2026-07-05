"""
Cognee 1.x reads all LLM/embedding config from environment variables at import time.
This function just validates the key is present and prints confirmation.

Required env vars (set in .env):
  LLM_PROVIDER=openai          ← Groq is OpenAI-compatible
  LLM_ENDPOINT=https://api.groq.com/openai/v1
  LLM_API_KEY=gsk_...
  LLM_MODEL=llama-3.3-70b-versatile
  EMBEDDING_PROVIDER=fastembed
  EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
"""
import os


async def setup_cognee() -> None:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set GROQ_API_KEY (or LLM_API_KEY) in your .env file."
        )

    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    embedding = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    print(f"[cognee] ready — LLM={model} via Groq | embeddings=fastembed({embedding})")
