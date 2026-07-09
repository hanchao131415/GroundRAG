# GroundRAG

> **Enterprise RAG Knowledge Base** — 5-stage retrieval pipeline + RBAC isolation + Streaming SSE + Full-pipeline Trace + React SPA

[English](README.md) | [中文](README.zh-CN.md)

GroundRAG is a portfolio showcase of a production-style RAG (Retrieval-Augmented Generation) system. It implements the complete lifecycle: multi-format document ingestion → hybrid search → cross-encoder reranking → role-based access control → streaming chat with grounded citations → step-level trace observability — all served through a bilingual React SPA and deployable with a single `docker compose up`.

## Architecture

```
Browser (React SPA)
   │  POST /api/v1/chat (SSE: sources → token → trace → done)
   │  POST /api/v1/search   GET /api/v1/stats
   │  POST /auth/login      GET /auth/demo-users
   ▼
FastAPI ── EnterpriseRAGSystem.ask_stream()
   │
   ├── _decide_intent()      ── simple-query shortcut (saves ~20s LLM call)
   ├── _rewrite_query()      ── LLM query expansion
   ├── _retrieve()           ── RBAC → permission_aware_search()
   │
   └── Retrieval Pipeline (5 stages)
        ┌─────────────────────────────────────────┐
        │ ① Vector Recall    FAISS + bge-small-zh │
        │ ② BM25 Recall      jieba tokenizer       │
        │ ③ RRF Fusion       k=60                  │
        │ ④ MMR Dedup         diversity-preserving  │
        │ ⑤ BGE Reranker      cross-encoder score   │
        └─────────────────────────────────────────┘
   │
   ├── CacheService          ── semantic cache + safe-to-hit validation
   ├── Tracer                ── step-level (ms + token + $USD)
   ├── LLM Factory           ── DeepSeek / Zhipu / Anthropic
   └── FallbackLLM           ── primary → backup → error chain
```

## Features

- **5-Stage Retrieval** — Vector (FAISS cosine≥0.3) → BM25 (jieba) → RRF fusion (k=60) → MMR dedup → **BGE Reranker** (cross-encoder). Each stage logged with `【检索问题】` preamble for full traceability.
- **RBAC Permission Isolation** — Per-department FAISS sub-indexes ensure *true pre-filter* (not post-filter). Admin (`*`) sees everything. Switch users in the UI to watch HR/Finance/IT isolation live.
- **Streaming Chat + Citations** — Typed SSE events (`sources` → `token` → `trace` → `done`). Every answer cites source documents with file name, page number, department badge, and relevance score. `ask()` and `ask_stream()` share the same decision helpers (DRY).
- **Full-pipeline Trace** — Intent routing → query rewrite → retrieval → generation. Each step records: ms (μs precision for sub-ms steps), token usage (prompt/completion/total), estimated cost in USD. Collapsible `TracePanel` in the UI.
- **Bilingual UI** — 30 i18n keys covering all UI chrome. Chinese/English toggle in header. Language persisted in localStorage.
- **One-command Deploy** — Multi-stage Dockerfile (node frontend → python runtime). `bge-small-zh` + `bge-reranker-base` pre-downloaded at build time → zero cold start.

## Quick Start

### Prerequisites

- Python >= 3.12, < 3.14
- Node.js >= 18
- LLM API key (DeepSeek recommended; Zhipu, Anthropic also supported)

### Local

```bash
git clone https://github.com/your-username/GroundRAG.git
cd GroundRAG
cp .env.example .env          # edit: fill LLM_API_KEY
make setup                    # pip install + npm install
make docs                     # generate sample docs (5 depts x 5 formats)
make run                      # build frontend + start :8000
```

### Docker

```bash
cp .env.example .env          # edit: fill LLM_API_KEY
docker compose up --build     # http://localhost:8000
```

Models baked into image (~2 GB). First request is instant -- no download.

### Dev Mode

```bash
# Terminal 1
uvicorn app.api:app --reload --port 8000

# Terminal 2
cd web && npm run dev         # http://localhost:5173 (HMR)
```

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/auth/login` | -- | Login (`user_id`) -> JWT |
| `GET` | `/auth/demo-users` | -- | List demo users |
| `POST` | `/api/v1/chat` | JWT | Chat: `stream:true` -> typed SSE, `stream:false` -> JSON |
| `POST` | `/api/v1/search` | JWT | Pure retrieval (no LLM) |
| `GET` | `/api/v1/stats` | JWT | KB metrics (docs, chunks, dept distribution, cache) |
| `GET` | `/health` | -- | Health check |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12, FastAPI, LangChain 0.3 |
| **Embedding** | `bge-small-zh-v1.5` (512-dim, CPU) |
| **Vector Store** | FAISS (IP index, cosine similarity) |
| **BM25** | `rank_bm25` + jieba tokenizer |
| **Reranker** | `bge-reranker-base` (transformers-native, Windows-safe) |
| **LLM** | DeepSeek v4 / Zhipu GLM / Anthropic (protocol-adaptive factory) |
| **Frontend** | React 18, TypeScript, Vite 5, Tailwind CSS 3 |
| **Testing** | pytest (43 tests), Vitest + RTL (5 tests) |
| **Trace** | Custom Tracer -> JSONL; Langfuse (optional) |
| **Deploy** | Docker multi-stage, docker-compose |

## Demo Users

| User ID | Name | Departments | Role | Visibility |
|---------|------|-------------|------|------------|
| `zhangsan` | 张三 | `["HR"]` | 员工 | HR docs only |
| `lisi` | 李四 | `["财务"]` | 员工 | Finance docs only |
| `wangwu` | 王五 | `["IT"]` | 员工 | IT docs only |
| `admin` | 管理员 | `["*"]` | 管理员 | All documents |

## Project Structure

```
GroundRAG/
├── app/                    # FastAPI
│   ├── api.py              # Routes + SSE + static serving + SPA fallback
│   └── auth.py             # JWT
├── rag_modules/            # Core pipeline
│   ├── retrieval_optimization.py  # 5-stage retrieval + RBAC sub-index
│   ├── reranker.py                # BGE cross-encoder (transformers-native)
│   ├── generation_integration.py  # LLM generation + streaming with usage
│   ├── cache_service.py           # Semantic cache (thread-safe)
│   ├── index_construction.py      # Incremental FAISS indexing
│   ├── data_preparation.py        # Multi-format parser (5 formats)
│   ├── tracer.py                  # Step-level trace -> JSONL
│   ├── llm_factory.py             # Multi-provider LLM factory
│   ├── llm_fallback.py            # Primary -> backup chain
│   ├── observability.py           # Langfuse integration
│   ├── user_service.py            # User management
│   └── logging_config.py          # Structured logging
├── web/                    # React SPA
│   └── src/
│       ├── views/          # ChatView, SearchView, StatsView
│       ├── components/     # UserSwitcher, ChatWindow, MessageBubble,
│       │                   #   SourceCard, TracePanel, SearchResults
│       ├── hooks/          # useAuth, useSSEChat
│       ├── api/            # client.ts, sse.ts (pure SSE parser)
│       ├── i18n/           # 30-key zh/en dictionary
│       └── types.ts
├── data/
│   ├── docs/               # 11 sample docs (5 depts, 3k-5k chars each)
│   └── users.json
├── tests/                  # 43 pytest + 5 vitest
├── evaluation/             # Retrieval sweep + A/B comparison
├── scripts/                # make_sample_docs.py
├── Dockerfile              # Multi-stage (node + python, non-root)
├── docker-compose.yml
├── Makefile
└── README.md
```

## Known Limitations

- **Password-less login** -- Demo users authenticate by `user_id` only. A real system would add password hashing. Intentional simplification for portfolio demonstration.
- **No online deployment** -- Designed for local review. No CDN, no cloud hosting. CORS not needed (FastAPI serves frontend from same origin).
- **Token/cost are estimates** -- Streaming `include_usage` varies by provider and fallback chain. Treat as approximate.
- **FAISS pickle** -- Local index uses `allow_dangerous_deserialization=True`. Never load externally-uploaded index files.
- **Single-process** -- No task queue, no horizontal scaling. LLM calls go to a thread pool to avoid blocking the event loop.

## License

MIT -- see [LICENSE](LICENSE)

---

Built as an AI Engineer portfolio piece. Questions? Open an issue or reach out.
