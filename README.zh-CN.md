# GroundRAG

> **企业级 RAG 知识库系统** — 混合检索 + RBAC 权限隔离 + 多级缓存 + 全链路可观测 + React 单页应用

[English](README.md) | [中文](README.zh-CN.md)

GroundRAG 是一个面向求职作品集展示的生产级 RAG（检索增强生成）系统。完整呈现企业知识库的全生命周期：多格式文档解析入库、混合检索（向量+BM25）、按部门隔离的 RBAC 权限控制、带来源引用的流式问答、全链路 Trace 可观测——全部集成在一个 React 前端中，支持中英双语切换。

## 系统架构

```
浏览器 (React SPA)
   │
   ├── /api/v1/chat (SSE 流式)
   ├── /api/v1/search (纯检索)
   ├── /api/v1/stats (知识库统计)
   ├── /auth/login | /auth/demo-users
   │
   ▼
FastAPI (app/api.py)
   │
   ├── EnterpriseRAGSystem.ask_stream()
   │   ├── _decide_intent()    ── 意图路由 / 明确问题短路
   │   ├── _rewrite_query()    ── LLM 查询改写
   │   ├── _retrieve()         ── 带权限的混合检索
   │   └── generation_module   ── 流式 LLM 生成 + token 统计
   │
   ├── RetrievalOptimizationModule
   │   ├── 向量检索 (FAISS + bge-small-zh 嵌入模型)
   │   ├── BM25 关键词检索 (jieba 分词)
   │   ├── RRF 融合 (k=60)
   │   ├── MMR 去重
   │   └── RBAC 子索引 (按部门分片 FAISS 索引)
   │
   ├── CacheService            ── 语义缓存 + safe-to-hit 校验
   ├── Tracer                  ── 步骤级 trace (ms + token + 成本)
   ├── LLM Factory             ── 多厂商适配: DeepSeek / 智谱 / Anthropic
   └── FallbackLLM             ── 主→备→报错 降级链
```

## 核心特性

- **混合检索** — 稠密向量（FAISS）+ 稀疏关键词（BM25），RRF 融合 + MMR 去重。支持 PDF、DOCX、XLSX、Markdown、TXT 五种格式文档解析入库。
- **RBAC 权限隔离** — 每个用户只能看到授权部门的文档。按部门构建 FAISS 子索引，实现真正的"先过滤再检索"（非事后过滤）。管理员（`*`）可见全部文档。在前端切换用户即可观察隔离效果。
- **流式问答 + 来源引用** — 类型化 SSE 事件流（sources → token → trace → done）。每条回答附带引用来源卡片：文件名、页码、所属部门、相关度评分。
- **全链路可观测** — 每次查询完整记录：意图路由、查询改写、检索、生成四个步骤。每个步骤记录耗时（ms）、Token 消耗和预估成本（美元）。前端 TracePanel 可折叠查看。
- **多级缓存** — 语义缓存 + safe-to-hit 校验（实体变化检测、意图对比、数字敏感度），确保缓存回答不包含过时信息。
- **中英双语 UI** — 顶栏一键切换中/英文。UI 框架文案全部翻译；数据内容（部门名、文档名、LLM 回答）保留原文。
- **一键部署** — `docker compose up --build` 启动全部服务。Embedding 模型在镜像构建时预下载，首请求零冷启动。

## 快速开始

### 环境要求

- Python >= 3.12, < 3.14
- Node.js >= 18
- LLM API Key（DeepSeek / 智谱 / Anthropic 均可）

### 本地开发

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/GroundRAG.git
cd GroundRAG

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY

# 3. 安装依赖
make setup

# 4. 生成样例文档
make docs

# 5. 一键启动
make run
# → 浏览器打开 http://localhost:8000
```

### Docker（推荐给 Reviewer）

```bash
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY
docker compose up --build
# → 浏览器打开 http://localhost:8000
# 模型已烘焙进镜像，无需等待下载
```

### 开发模式（前后端分离）

```bash
# 终端 1: 后端
uvicorn app.api:app --reload --port 8000

# 终端 2: 前端（带 HMR 热更新代理）
cd web && npm run dev
# → 浏览器打开 http://localhost:5173
```

## API 接口

| 方法 | 路径 | 鉴权 | 说明 |
|--------|------|------|-------------|
| `POST` | `/auth/login` | 无 | 用户登录（传 `user_id`）→ JWT |
| `GET` | `/auth/demo-users` | 无 | 获取演示用户列表（供前端下拉框） |
| `POST` | `/api/v1/chat` | JWT | **流式**（`stream: true`→SSE）或**非流式**问答，返回 sources + trace |
| `POST` | `/api/v1/search` | JWT | 纯检索模式（不调用 LLM） |
| `GET` | `/api/v1/stats` | JWT | 知识库统计数据 |
| `GET` | `/health` | 无 | 健康检查 |

## 技术栈

| 层级 | 技术选型 |
|-------|-----------|
| **后端** | Python 3.12, FastAPI, LangChain |
| **向量存储** | FAISS (CPU) + `bge-small-zh-v1.5` 嵌入模型 |
| **稀疏检索** | BM25 (rank_bm25 + jieba 分词) |
| **大模型** | DeepSeek / 智谱 / Anthropic（协议自适应工厂模式） |
| **前端** | React 18, TypeScript, Vite 5, Tailwind CSS 3 |
| **测试** | pytest（后端）, Vitest + React Testing Library（前端） |
| **可观测** | 自研 Tracer → JSONL 本地 trace + Langfuse（可选） |
| **部署** | Docker 多阶段构建, docker-compose |

## 演示用户

| 用户 ID | 姓名 | 所属部门 | 角色 | 可见范围 |
|---------|------|-------------|------|---------------|
| `zhangsan` | 张三 | `["HR"]` | 员工 | 仅 HR 文档 |
| `lisi` | 李四 | `["财务"]` | 员工 | 仅财务文档 |
| `wangwu` | 王五 | `["IT"]` | 员工 | 仅 IT 文档 |
| `admin` | 管理员 | `["*"]` | 管理员 | 全部文档 |

## 项目结构

```
GroundRAG/
├── app/                    # FastAPI 应用
│   ├── api.py              # 路由: chat, search, stats, auth, 静态文件托管
│   └── auth.py             # JWT 鉴权
├── rag_modules/            # RAG 核心管线
│   ├── retrieval_optimization.py  # 混合检索 + RBAC 子索引
│   ├── generation_integration.py  # LLM 生成 + 流式输出
│   ├── cache_service.py           # 语义缓存 + safe-to-hit
│   ├── index_construction.py      # 文档加载 + 分块 + FAISS 建库
│   ├── data_preparation.py        # 多格式文档解析
│   ├── tracer.py                  # 步骤级 trace
│   ├── llm_factory.py             # 多厂商 LLM 工厂
│   ├── llm_fallback.py            # 降级链
│   ├── reranker.py                # BGE Reranker（默认关闭）
│   ├── observability.py           # Langfuse 集成
│   ├── user_service.py            # 用户管理
│   └── logging_config.py          # 结构化日志
├── web/                    # React 单页应用
│   └── src/
│       ├── views/          # ChatView, SearchView, StatsView
│       ├── components/     # UserSwitcher, ChatWindow, MessageBubble,
│       │                   #   SourceCard, TracePanel, SearchResults
│       ├── hooks/          # useAuth, useSSEChat
│       ├── api/            # client.ts (fetch + JWT), sse.ts (SSE 解析)
│       ├── i18n/           # 轻量中英双语
│       └── types.ts        # TypeScript 类型定义
├── data/
│   ├── docs/               # 样例文档（5 部门 × 5 格式）
│   └── users.json          # 演示用户库
├── tests/                  # pytest + vitest 测试套件
├── evaluation/             # 检索调参 + A/B 对比工具
├── scripts/                # make_sample_docs.py
├── Dockerfile              # 多阶段构建（node 前端 + python 后端）
├── docker-compose.yml
├── Makefile
└── README.md
```

## 已知限制

- **无密码登录** — 演示用户仅凭 `user_id` 登录，真实系统需补充密码校验。此为作品集展示的刻意简化。
- **Reranker 默认关闭** — 生产检索路径使用 RRF + MMR 融合。BGE Reranker 在 `evaluation/` 目录下可用，但在 Windows 上可能崩溃（0xC0000005）。评测数据 ≠ 生产检索行为。
- **Token / 成本为估算值** — 用量和成本为近似值。流式 `include_usage` 在不同厂商和降级链下表现不一。
- **无公网部署** — 设计用于本地 Review，无 CDN、无云托管。未配置 CORS（FastAPI 同源托管）。
- **FAISS pickle 安全** — 索引导入使用了 `allow_dangerous_deserialization=True`，仅适用于本地受信文件。切勿加载外部上传的索引文件。

## 开源协议

MIT — 详见 [LICENSE](LICENSE)

---

AI 工程师求职作品。有问题请提 Issue 或直接联系。
