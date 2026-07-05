"""
Groq chat client.  Cognee provides the retrieved memory context;
this module handles the actual LLM call that turns context + question → answer.
"""
import os
from groq import AsyncGroq

_client: AsyncGroq | None = None


def _get() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _client


async def chat(
    user_message: str,
    context: list[str],
    system_prompt: str,
    history: list[dict] | None = None,
    model: str | None = None,
) -> str:
    """
    Build a chat completion with:
      - system_prompt  (role + persona)
      - retrieved memory injected into the system message
      - optional prior turn history
      - user_message
    """
    ctx_block = "\n\n---\n\n".join(context) if context else "(no prior memory for this query)"

    system = (
        f"{system_prompt}\n\n"
        f"## Retrieved Memory (from Cognee knowledge graph)\n"
        f"{ctx_block}"
    )

    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    resp = await _get().chat.completions.create(
        model=model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""
