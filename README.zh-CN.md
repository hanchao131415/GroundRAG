# GroundRAG

> **企业级 RAG 知识库系统** — 5 阶段检索管线 + RBAC 权限隔离 + 流式 SSE + 全链路 Trace + React 单页应用

[English](README.md) | [中文](README.zh-CN.md)

GroundRAG 是一个面向求职展示的生产级 RAG（检索增强生成）系统。完整呈现企业知识库的全生命周期：多格式文档解析 → 混合检索 → cross-encoder 精排 → 角色权限控制 → 流式问答 + 来源引用 → 全链路可观测，全部集成在支持中英双语切换的 React 前端中，一条 `docker compose up` 即可启动。

## 系统架构

```
浏览器 (React SPA)
   │  POST /api/v1/chat (SSE: sources → token → trace → done)
   │  POST /api/v1/search   GET /api/v1/stats
   │  POST /auth/login      GET /auth/demo-users
   ▼
FastAPI ── EnterpriseRAGSystem.ask_stream()
   │
   ├── _decide_intent()      ── 明确问题短路（省 ~20s LLM 调用）
   ├── _rewrite_query()      ── LLM 查询扩展
   ├── _retrieve()           ── RBAC → permission_aware_search()
   │
   └── 检索管线 (5 阶段)
        ┌─────────────────────────────────────────┐
        │ ① 向量召回    FAISS + bge-small-zh      │
        │ ② BM25 召回   jieba 分词                  │
        │ ③ RRF 融合    k=60                        │
        │ ④ MMR 去重     保多样性                    │
        │ ⑤ BGE Reranker cross-encoder 精排         │
        └─────────────────────────────────────────┘
   │
   ├── CacheService          ── 语义缓存 + safe-to-hit 校验
   ├── Tracer                ── 步骤级 trace (ms + token + $USD)
   ├── LLM Factory           ── DeepSeek / 智谱 / Anthropic
   └── FallbackLLM           ── 主 → 备 → 报错 降级链
```

## 核心特性

- **5 阶段检索管线** — 向量宽松召回 (cosine≥0.3) → BM25 关键词 → RRF 融合 → MMR 去重 → **BGE Reranker 精排**。每阶段日志带 `【检索问题】` 前缀，全链路可追踪。
- **RBAC 权限隔离** — 按部门构建 FAISS 子索引，实现真正的"先过滤再检索"。管理员（`*`）可见全部，在前端切换用户即可观察 HR/财务/IT 的隔离效果。
- **流式问答 + 来源引用** — 类型化 SSE 事件流（`sources` → `token` → `trace` → `done`）。每条回答附带引用卡片：文件名、页码、部门标签、相关度评分。`ask()` 与 `ask_stream()` 共用决策逻辑，杜绝代码重复。
- **全链路 Trace** — 意图路由 → 查询改写 → 检索 → 生成。每步记录：毫秒耗时（亚毫秒级用 μs 显示）、Token 用量（prompt/completion/total）、预估美元成本。前端 TracePanel 可折叠查看。
- **中英双语 UI** — 30 个 i18n key 覆盖全部 UI 文案。顶栏一键切换中/EN，语言偏好存 localStorage。
- **知识库管理台** — 支持按部门上传、查看和删除 PDF/DOCX/Markdown/TXT/XLSX 文档，轮询索引状态，并允许管理员在后台触发全量重建。
- **生产化运行时** — 轻量存活探针、结构化就绪探针、后台模型初始化、有界 SSE 背压、客户端断开取消和原子文件写入。
- **一键部署** — 多阶段 Dockerfile（node 前端构建 → python 运行时）。`bge-small-zh` + `bge-reranker-base` 两个模型均在构建期预下载 → 首请求零冷启动。

## 快速开始

### 五分钟面试演示

```bash
make setup
make demo                     # 可重复执行：准备 11 份多格式文档
make run                      # 打开 http://localhost:8000
```

通过用户切换同时验证检索质量和权限隔离：

| 用户 | 提问 | 预期现象 |
|------|------|----------|
| `zhangsan` | `工作满3年年假几天？` | 引用 HR 制度，财务和 IT 文档不可见 |
| `lisi` | `一线城市住宿费报销上限是多少？` | 检索到财务部门的差旅表格 |
| `wangwu` | `公司密码多久更换一次？` | 检索到 IT 安全制度 |
| `admin` | `检索全部部门的制度文档并比较来源。` | 结果可以覆盖所有部门 |

管理员进入**知识库**上传文档后，可以观察后台重建状态。已有问答和检索继续使用当前索引；如果重建失败，系统显示降级状态但不会丢弃可用索引。

### 环境要求

- Python >= 3.12, < 3.14
- Node.js >= 18
- LLM API Key（推荐 DeepSeek，也支持智谱、Anthropic）

### 本地运行

```bash
git clone https://github.com/your-username/GroundRAG.git
cd GroundRAG
cp .env.example .env          # 编辑：填入 LLM_API_KEY
make setup                    # pip install + npm install
make docs                     # 生成样例文档（5 部门 x 5 格式）
make run                      # 构建前端 + 启动 :8000
```

### Docker

```bash
cp .env.example .env          # 编辑：填入 LLM_API_KEY
docker compose up --build     # → http://localhost:8000
```

模型烘焙进镜像（约 2 GB），首请求即时响应，无需等待下载。

### 开发模式

```bash
# 终端 1
uvicorn app.api:app --reload --port 8000

# 终端 2
cd web && npm run dev         # → http://localhost:5173 (HMR 热更新)
```

## API 接口

| 方法 | 路径 | 鉴权 | 说明 |
|--------|------|------|-------------|
| `POST` | `/auth/login` | -- | 登录（`user_id`）→ JWT |
| `GET` | `/auth/demo-users` | -- | 获取演示用户列表 |
| `POST` | `/api/v1/chat` | JWT | 问答：`stream:true` → 类型化 SSE，`stream:false` → JSON |
| `POST` | `/api/v1/search` | JWT | 纯检索（不调 LLM） |
| `GET` | `/api/v1/stats` | JWT | 知识库统计（文档数、分块数、部门分布、缓存） |
| `GET/POST` | `/api/v1/documents` | JWT | 查看或上传有部门权限的文档 |
| `DELETE` | `/api/v1/documents/{id}` | JWT | 删除有权限的文档并安排重建索引 |
| `POST` | `/api/v1/documents/reindex` | 管理员 | 后台触发全量索引重建 |
| `GET` | `/api/v1/index-status` | JWT | 查看索引状态和最近错误 |
| `GET` | `/health` | -- | 轻量进程存活检查（不会加载模型） |
| `GET` | `/ready` | -- | 结构化就绪检查；加载或索引时返回 `503` |

## 技术栈

| 层级 | 技术选型 |
|-------|-----------|
| **后端** | Python 3.12, FastAPI, LangChain 0.3 |
| **嵌入模型** | `bge-small-zh-v1.5`（512 维，CPU） |
| **向量存储** | FAISS（IP 索引，cosine 相似度） |
| **BM25** | `rank_bm25` + jieba 分词 |
| **精排模型** | `bge-reranker-base`（transformers 原生加载，Windows 兼容） |
| **大模型** | DeepSeek v4 / 智谱 GLM / Anthropic（协议自适应工厂） |
| **前端** | React 19, TypeScript 6, Vite 8, Tailwind CSS 3 |
| **测试** | pytest（66 个测试）, Vitest + RTL（10 个测试） |
| **可观测** | 自研 Tracer → JSONL；Langfuse（可选） |
| **部署** | Docker 多阶段构建, docker-compose |

## 演示用户

| 用户 ID | 姓名 | 所属部门 | 角色 | 可见范围 |
|---------|------|-------------|------|------------|
| `zhangsan` | 张三 | `["HR"]` | 员工 | 仅 HR 文档 |
| `lisi` | 李四 | `["财务"]` | 员工 | 仅财务文档 |
| `wangwu` | 王五 | `["IT"]` | 员工 | 仅 IT 文档 |
| `admin` | 管理员 | `["*"]` | 管理员 | 全部文档 |

## 项目结构

```
GroundRAG/
├── app/                    # FastAPI 应用
│   ├── api.py              # 路由 + SSE + 静态文件托管 + SPA fallback
│   └── auth.py             # JWT 鉴权
├── rag_modules/            # RAG 核心管线
│   ├── retrieval_optimization.py  # 5 阶段检索 + RBAC 子索引
│   ├── reranker.py                # BGE cross-encoder（transformers 原生）
│   ├── generation_integration.py  # LLM 生成 + 流式 + token 统计
│   ├── cache_service.py           # 语义缓存（线程安全）
│   ├── index_construction.py      # 增量 FAISS 索引构建
│   ├── data_preparation.py        # 多格式文档解析（5 种格式）
│   ├── tracer.py                  # 步骤级 trace → JSONL
│   ├── llm_factory.py             # 多厂商 LLM 工厂
│   ├── llm_fallback.py            # 主→备 降级链
│   ├── observability.py           # Langfuse 集成
│   ├── user_service.py            # 用户管理
│   └── logging_config.py          # 结构化日志
├── web/                    # React 单页应用
│   └── src/
│       ├── views/          # 问答、检索、统计、知识库页面
│       ├── components/     # UserSwitcher, ChatWindow, MessageBubble,
│       │                   #   SourceCard, TracePanel, SearchResults
│       ├── hooks/          # useAuth, useSSEChat
│       ├── api/            # client.ts, sse.ts（纯函数 SSE 解析）
│       ├── i18n/           # 30 key 中英词典
│       └── types.ts
├── data/
│   ├── docs/               # 11 份样例文档（5 部门，每篇 3k-5k 字）
│   └── users.json
├── tests/                  # 66 个 pytest + 10 个 vitest
├── evaluation/             # 检索调参 + A/B 对比工具
├── scripts/                # make_sample_docs.py
├── Dockerfile              # 多阶段构建（node + python，非 root）
├── docker-compose.yml
├── Makefile
└── README.md
```

## 已知限制

- **无密码登录** — 演示用户仅凭 `user_id` 登录。真实系统需补充密码哈希校验。此为作品展示的刻意简化。
- **无公网部署** — 设计用于本地 Review。无 CDN、云托管。CORS 无需配置（FastAPI 同源托管前端）。
- **Token/成本为估算值** — 流式 `include_usage` 在不同厂商和降级链下表现不一，视为近似值。
- **FAISS pickle 安全** — 本地索引使用 `allow_dangerous_deserialization=True`。切勿加载外部上传的索引文件。
- **单进程运行** — 无任务队列、无水平扩展。LLM 调用放入线程池以避免阻塞事件循环。

## 开源协议

MIT — 详见 [LICENSE](LICENSE)

---

AI 工程师求职作品。有问题请提 Issue 或直接联系。
