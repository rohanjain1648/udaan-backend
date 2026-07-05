# Udaan × Cognee Backend

FastAPI memory layer for the [Udaan Verified Athlete Passport](../README.md).
Four AI features, all powered by **Cognee** (knowledge graph) + **Groq** (Llama 3.3 70B).

## Quick start

```powershell
copy .env.example .env      # fill in GROQ_API_KEY
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

## Features

| Route prefix | Feature | Cognee ops used |
|---|---|---|
| `/coach` | Athlete career coach with long-term memory | remember · recall · improve · forget |
| `/scout` | Scout copilot — field observation log | remember · recall · forget |
| `/include` | Para-sport classification advisor | remember · recall |
| `/train` | Training memory loop with drill ratings | remember · recall · improve · forget |

## How Cognee + Groq fit together

```
User question
     │
     ▼  lib/memory.py
cognee.search(INSIGHTS) ──→ graph traversal + vector similarity
     │
     ▼  lib/groq_agent.py
Groq system prompt = persona + retrieved context chunks
     │
     ▼
Natural-language reply
```

Cognee holds the long-term memory. Groq generates the reply given the retrieved context.
Neither has persistent state between sessions — all continuity comes from Cognee's knowledge graph.

## The four Cognee operations

### `remember(text, dataset)` — `cognee.add() + cognee.cognify()`
Ingests text into the named dataset. `cognify()` uses Groq to extract entities and relationships and builds them into a knowledge graph. If `cognify()` fails (rate limit, error), the raw text is still vector-indexed so `recall()` still works.

### `recall(query, dataset)` — `cognee.search(SearchType.INSIGHTS)`
Searches the knowledge graph via graph traversal and semantic similarity. Returns the top-N most relevant context chunks. Falls back to `SearchType.CHUNKS` (plain vector search) if INSIGHTS fails.

### `improve(text, dataset)` — wrapper around `remember()`
Stores a structured feedback note tagged `[FEEDBACK/IMPROVEMENT NOTE]`. The note is ingested into the graph, where it connects to the relevant advice/drill node. Future `recall()` calls traverse to this note and the LLM adapts its recommendations accordingly.

### `forget(dataset)` — `cognee.prune.prune_data(datasets=[name])`
Deletes the entire named dataset from Cognee — all vectors, graph nodes, edges. In Udaan this implements the athlete's **right to erasure** under India's DPDP Act. A guardian taps "Delete" and nothing about that child remains in the system.

## Dataset naming

| Dataset | Scope |
|---|---|
| `coach_{uid}` | Per-athlete coaching memory |
| `scout_sessions` | Shared scout field log |
| `include_knowledge` | Global para-sport knowledge base (seeded on startup) |
| `train_{uid}` | Per-athlete training history |

## Environment

```
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
LLM_API_KEY=gsk_...        # Cognee reads this
LLM_PROVIDER=openai        # Groq is OpenAI-compatible
LLM_ENDPOINT=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

Only one API key is needed: `GROQ_API_KEY`. Embeddings run locally via FastEmbed (first run downloads ~130 MB).
