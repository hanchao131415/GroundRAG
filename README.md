# GroundRAG

> **Enterprise RAG Knowledge Base** — Hybrid Retrieval + RBAC + Multi-level Cache + Full-pipeline Observability + React SPA

[English](README.md) | [中文](README.zh-CN.md)

GroundRAG is a production-grade Retrieval-Augmented Generation (RAG) system built as a portfolio showcase. It demonstrates the complete lifecycle of an enterprise knowledge base: multi-format document ingestion, hybrid search (vector + BM25), role-based access control with per-user department filtering, streaming chat with source citations, and full-pipeline trace observability — all wrapped in a React frontend with bilingual UI.

## Architecture

```
Browser (React SPA)
   │
   ├── /api/v1/chat (SSE streaming)
   ├── /api/v1/search (pure retrieval)
   ├── /api/v1/stats (KB metrics)
   ├── /auth/login | /auth/demo-users
   │
   ▼
FastAPI (app/api.py)
   │
   ├── EnterpriseRAGSystem.ask_stream()
   │   ├── _decide_intent()    ── query router / simple-query shortcut
   │   ├── _rewrite_query()    ── LLM query rewrite
   │   ├── _retrieve()         ── RBAC-aware hybrid search
   │   └── generation_module   ── streaming LLM with usage tracking
   │
   ├── RetrievalOptimizationModule
   │   ├── Vector Search (FAISS + bge-small-zh embeddings)
   │   ├── BM25 (jieba tokenizer)
   │   ├── RRF Fusion (k=60)
   │   ├── MMR Deduplication
   │   └── RBAC Sub-Index (per-department FAISS indexes)
   │
   ├── CacheService            ── semantic cache with safe-to-hit validation
   ├── Tracer                  ── step-level trace with ms + token + cost
   ├── LLM Factory             ── multi-provider: DeepSeek / Zhipu / Anthropic
   └── FallbackLLM             ── primary → backup → error chain
```

## Features

- **Hybrid Search** — Dense (FAISS vector) + Sparse (BM25 keyword) with RRF fusion and MMR deduplication. Multi-format document ingestion (PDF, DOCX, XLSX, Markdown, TXT).
- **RBAC Permission Isolation** — Each user sees only authorized departments. Per-department FAISS sub-indexes ensure *true pre-filter* (not post-filter). Admin (`*`) sees everything. Switch users in the UI to see isolation in action.
- **Streaming Chat with Citations** — Typed SSE events (sources → token → trace → done). Every answer cites its source documents with department badge, page number, and relevance score.
- **Full-pipeline Observability** — Every query traced: intent routing, query rewrite, retrieval, generation. Each step records milliseconds, token usage, and estimated cost (USD). Collapsible TracePanel in the UI.
- **Multi-level Cache** — Semantic cache with safe-to-hit validation (entity change detection, intent comparison, number sensitivity) before serving cached answers.
- **Bilingual UI** — Chinese/English toggle. All UI chrome translated; content (department names, document titles, LLM answers) left in original language.
- **One-command Deploy** — `docker compose up --build` brings everything up. Embedding model pre-downloaded at build time — no cold start on first request.

## Quick Start

### Prerequisites

- Python >= 3.12, < 3.14
- Node.js >= 18
- LLM API key (DeepSeek, Zhipu, or Anthropic)

### Local Development

```bash
# 1. Clone
git clone https://github.com/your-username/GroundRAG.git
cd GroundRAG

# 2. Configure
cp .env.example .env
# Edit .env and fill your LLM_API_KEY

# 3. Setup
make setup

# 4. Generate sample docs
make docs

# 5. Run (one command)
make run
# → Open http://localhost:8000
```

### Docker (recommended for reviewers)

```bash
cp .env.example .env
# Edit .env and fill your LLM_API_KEY
docker compose up --build
# → Open http://localhost:8000
# Embedding model baked into image — no cold start
```

### Dev Mode (two processes)

```bash
# Terminal 1: Backend
uvicorn app.api:app --reload --port 8000

# Terminal 2: Frontend (with HMR proxy)
cd web && npm run dev
# → Open http://localhost:5173
```

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/auth/login` | No | Login with `user_id` → JWT |
| `GET` | `/auth/demo-users` | No | List demo users for UI dropdown |
| `POST` | `/api/v1/chat` | JWT | **Streaming** (`stream: true` → SSE) or **non-streaming** chat with sources + trace |
| `POST` | `/api/v1/search` | JWT | Pure retrieval (no LLM) |
| `GET` | `/api/v1/stats` | JWT | Knowledge base metrics |
| `GET` | `/health` | No | Health check |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12, FastAPI, LangChain |
| **Vector Store** | FAISS (CPU) + `bge-small-zh-v1.5` embeddings |
| **Sparse Retrieval** | BM25 (rank_bm25 + jieba) |
| **LLM** | DeepSeek / Zhipu / Anthropic (protocol-adaptive factory) |
| **Frontend** | React 18, TypeScript, Vite 5, Tailwind CSS 3 |
| **Testing** | pytest (backend), Vitest + React Testing Library (frontend) |
| **Observability** | Custom Tracer → JSONL traces + Langfuse (optional) |
| **Deployment** | Docker (multi-stage), docker-compose |

## Demo Users

| User ID | Name | Departments | Role | What They See |
|---------|------|-------------|------|---------------|
| `zhangsan` | 张三 | `["HR"]` | 员工 | HR docs only |
| `lisi` | 李四 | `["财务"]` | 员工 | Finance docs only |
| `wangwu` | 王五 | `["IT"]` | 员工 | IT docs only |
| `admin` | 管理员 | `["*"]` | 管理员 | All documents |

## Project Structure

```
GroundRAG/
├── app/                    # FastAPI application
│   ├── api.py              # Routes: chat, search, stats, auth, static serving
│   └── auth.py             # JWT authentication
├── rag_modules/            # Core RAG pipeline
│   ├── retrieval_optimization.py  # Hybrid search + RBAC sub-index
│   ├── generation_integration.py  # LLM generation + streaming
│   ├── cache_service.py           # Semantic cache with safe-to-hit
│   ├── index_construction.py      # Document loading + chunking + FAISS
│   ├── data_preparation.py        # Multi-format parser
│   ├── tracer.py                  # Step-level trace
│   ├── llm_factory.py             # Multi-provider LLM
│   ├── llm_fallback.py            # Fallback chain
│   ├── reranker.py                # BGE Reranker (off by default)
│   ├── observability.py           # Langfuse integration
│   ├── user_service.py            # User management
│   └── logging_config.py          # Structured logging
├── web/                    # React SPA
│   └── src/
│       ├── views/          # ChatView, SearchView, StatsView
│       ├── components/     # UserSwitcher, ChatWindow, MessageBubble,
│       │                   #   SourceCard, TracePanel, SearchResults
│       ├── hooks/          # useAuth, useSSEChat
│       ├── api/            # client.ts (fetch + JWT), sse.ts (SSE parser)
│       ├── i18n/           # Lightweight zh/en bilingual
│       └── types.ts        # Shared TypeScript types
├── data/
│   ├── docs/               # Sample documents (5 depts × 5 formats)
│   └── users.json          # Demo user database
├── tests/                  # pytest + vitest suites
├── evaluation/             # Retrieval sweep + A/B comparison tools
├── scripts/                # make_sample_docs.py
├── Dockerfile              # Multi-stage (node + python)
├── docker-compose.yml
├── Makefile
└── README.md
```

## Known Limitations

- **Password-less login** — Demo users authenticate by `user_id` only. A real system would add password verification (intentional for portfolio demonstration).
- **Reranker disabled by default** — RRF + MMR fusion is used for production retrieval. BGE Reranker is available in `evaluation/` scripts but may crash on Windows (0xC0000005). Evaluation numbers ≠ production retrieval behavior.
- **Token/cost estimation** — Usage and cost are estimates. Streaming `include_usage` varies by provider and fallback chain. Treat as approximate.
- **No online deployment** — Designed for local review. No CDN, no cloud hosting. CORS not configured (same-origin serving by FastAPI).
- **FAISS pickle** — Index serialization uses `allow_dangerous_deserialization=True` for local trusted files. Never load externally-uploaded index files.

## License

MIT — see [LICENSE](LICENSE)

---

Built as an AI Engineer portfolio piece. For questions, open an issue or reach out.
