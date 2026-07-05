"""
Thin wrapper around Cognee 1.x's four lifecycle operations:
  remember() → cognee.remember()
  recall()   → cognee.recall()
  improve()  → append a feedback note via remember()
  forget()   → cognee.forget(dataset=...)
"""
import cognee


async def remember(text: str, dataset: str) -> None:
    """Ingest text into Cognee — builds knowledge graph + vector index."""
    try:
        await cognee.remember(text, dataset_name=dataset)
    except Exception as e:
        print(f"[memory] remember failed for '{dataset}': {e}")
        raise


async def recall(query: str, dataset: str, limit: int = 6) -> list[str]:
    """Retrieve the most relevant memory chunks for a query."""
    try:
        raw = await cognee.recall(query_text=query, datasets=[dataset])
        return _extract(raw, limit)
    except Exception as e:
        print(f"[memory] recall failed for '{dataset}': {e}")
        return []


def _extract(raw: list, limit: int) -> list[str]:
    chunks: list[str] = []
    for r in (raw or [])[:limit]:
        if isinstance(r, str):
            chunks.append(r)
        elif isinstance(r, dict):
            chunks.append(r.get("text") or r.get("content") or str(r))
        elif hasattr(r, "text"):
            chunks.append(str(r.text))
        elif hasattr(r, "payload"):
            chunks.append(str(r.payload))
        else:
            chunks.append(str(r))
    return [c for c in chunks if c.strip()]


async def improve(text: str, dataset: str) -> None:
    """Store a feedback note — Cognee folds it into the knowledge graph."""
    await remember(f"[FEEDBACK/IMPROVEMENT NOTE]\n{text}", dataset)


async def forget(dataset: str) -> None:
    """Delete all memory associated with a dataset (right to erasure)."""
    try:
        await cognee.forget(dataset=dataset)
    except TypeError:
        # Some builds use dataset_name= instead of dataset=
        try:
            await cognee.forget(dataset_name=dataset)
        except Exception as e:
            print(f"[memory] forget failed for '{dataset}': {e}")
            raise
    except Exception as e:
        print(f"[memory] forget failed for '{dataset}': {e}")
        raise
