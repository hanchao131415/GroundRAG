# ---- Stage 1: Frontend Build ----
FROM node:20-alpine AS frontend-builder

WORKDIR /app/web
COPY web/package.json web/package-lock.json* ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- Stage 2: Runtime ----
FROM python:3.12-slim

# System deps for faiss / PyMuPDF / docx
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -r -s /bin/bash app

WORKDIR /app

# ---- Backend dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Pre-download embedding model (bge-small-zh, ~100MB) ----
ENV HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; \
    HuggingFaceEmbeddings(model_name='BAAI/bge-small-zh-v1.5', \
    model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':True})" \
    && chown -R app:app /app/.cache

# ---- Copy backend code ----
COPY --chown=app:app app/ ./app/
COPY --chown=app:app rag_modules/ ./rag_modules/
COPY --chown=app:app config.py main.py ./

# ---- Copy frontend build ----
COPY --from=frontend-builder --chown=app:app /app/web/dist/ ./web/dist/

# ---- Copy data ----
COPY --chown=app:app data/ ./data/
COPY --chown=app:app scripts/ ./scripts/
COPY --chown=app:app tests/ ./tests/
COPY --chown=app:app evaluation/ ./evaluation/
COPY --chown=app:app locustfile.py pyproject.toml ./
COPY --chown=app:app .env.example ./

# ---- Generate sample docs ----
RUN python scripts/make_sample_docs.py

# ---- Runtime config ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

USER app
EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
