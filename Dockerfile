# ── Hugging Face Space (Docker SDK) — Udaan × Cognee backend ────────────────
# HF free CPU tier gives 16 GB RAM — enough for Cognee's graph-build pipeline
# (Render's free 512 MB was not).
FROM python:3.11-slim

# Build tools for any Cognee sub-dependencies that compile from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cognee's local graph/vector stores + any model caches must write to a
# guaranteed-writable path. On HF Spaces /tmp is always writable.
ENV COGNEE_DATA_DIR=/tmp/cognee_data \
    HF_HOME=/tmp/hf \
    FASTEMBED_CACHE_PATH=/tmp/fastembed \
    XDG_CACHE_HOME=/tmp/cache

# HF Spaces proxies the public URL to this port.
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
