# GroundRAG — 前端 + 仓库改造设计

- **日期**: 2026-07-08
- **状态**: Draft（待用户 review）
- **作者**: brainstorming 会话产出
- **上游项目**: `project1-rag`（企业级 RAG 知识库问答系统）

---

## 1. 背景与目标

把现有的 `project1-rag` 后端改造成一个**可发布到 GitHub 的展示项目 GroundRAG**，主要交付：

1. **重命名**：`project1-rag` / `enterprise-rag` → **GroundRAG**。
2. **新增前端**（主要需求）：React + Vite 单页应用，把后端能力可视化展示出来。
3. **仓库卫生**：新建干净独立仓库，剔除个人笔记 / 真实公司文档 / 真实密钥。

**目标受众**：求职作品集（HR / 面试官打开 GitHub 就能 get 项目价值）。重点 = 专业 README + 一键本地运行 + 漂亮且能演示核心能力的前端。**不需要公网部署。**

## 2. 决策汇总

| 维度 | 决策 |
|---|---|
| 项目名 | **GroundRAG**（grounding 意象，突出防幻觉 + 引用溯源） |
| 目标 | 求职作品集 |
| 前端栈 | **React + Vite + TypeScript + Tailwind CSS** |
| 后端栈 | 沿用现有 FastAPI（最小改动） |
| 交付形态 | **新建独立 GitHub 仓库**（全新 `git init`，不带旧 history，避免旧 history 里的 `.env` 泄密） |
| 前端-后端耦合 | **方案 A**：monorepo，FastAPI 托管构建产物 `web/dist/`，reviewer 一条命令运行 |
| 用户/角色存储 | **保留 `data/users.json`**（不引入 SQLite） |
| 用户切换 UX | 前端下拉框（`UserSwitcher`）静默重新登录，非 CLI |
| SSE 协议 | 升级为**类型化 JSON 事件**（sources/token/trace/error） |
| README | **完整双语**：`README.md`(英文主) + `README.zh-CN.md`(中文)，顶部互链 |
| 前端 UI | **中英双语**：轻量 i18n（`useLang` + `t()` + zh/en 字典），头部中/EN 切换，默认中文 |
| 样例文档 | **合成样例**（虚构公司），替换真实公司文档 |

## 3. 展示功能（前端视图）

单页应用，顶部导航三个视图：

1. **💬 Chat（默认视图）**
   - `UserSwitcher`（头部下拉）：张三/HR、李四/财务、admin、未登录。
   - 流式 `ChatWindow`：用户/助手气泡，token 逐字渲染。
   - 每条助手回答下：`SourceCard` 列表（文件 / 页 / 部门 / 分数 / 预览）。
   - 每条助手回答下：可折叠 `TracePanel`（意图/改写/检索/生成 各步 ms + token + 估算成本）。
2. **🔍 Search**：纯检索模式（不调 LLM）→ 输入 → chunk 列表（分数 / 来源 / 预览）。
3. **📊 Stats**：文档数 / 部门分布 / 缓存命中数。

**RBAC 演示机制**：同一会话内切换用户 → 调 `/auth/login` 拿新 JWT → 后端按 JWT 的 user_id 现查部门 → 带权限检索。可实时看到"HR 搜不到财务文档"的隔离效果。

## 4. 不做（YAGNI）

- 公网部署 / 在线 Demo 托管。
- 登录密码（沿用后端已知的 A3 弱点，demo 简化；README 注明真实系统需补密码校验）。
- 生产路径启用 reranker（保持 `_init_reranker()→None` 现状；Windows 0xC0000005 崩溃风险，README 注明 eval 脚本可用）。
- 用户管理 UI（增删改用户）→ 因此不引入 SQLite。
- i18n 框架、账号持久化。

> ✅ **Docker 部署（Dockerfile + docker-compose）纳入范围**，见 §14。原列在 YAGNI，用户后续要求加入。

---

## 5. 仓库结构（GroundRAG 根）

把 `project1-rag/` 内容**上提到新仓库根目录**（不再有外层 `project1-rag/`），新增 `web/`：

```
GroundRAG/
├── README.md                 # 英文（主）
├── README.zh-CN.md           # 中文完整版
├── LICENSE                   # MIT
├── Makefile                  # make setup / dev / build / run
├── .env.example              # 安全模板（占位 key）
├── .gitignore                # 沿用并加固（.env / web/dist / .venv / *.pkl / *.faiss）
├── pyproject.toml            # name="groundrag"
├── requirements.txt
├── app/                      # FastAPI（沿用）
│   ├── __init__.py
│   ├── api.py                # 改：SSE 类型化事件 + sources/trace + 静态托管 + demo-users
│   └── auth.py               # 沿用
├── rag_modules/              # 沿用全部
├── config.py                 # 沿用
├── main.py                   # 沿用
├── tests/                    # 沿用 + 新增响应形状测试
├── evaluation/               # 沿用
├── data/
│   ├── docs/                 # ⚠️ 合成样例文档（虚构公司，保留 HR/财务/IT/行政/研发 结构）
│   ├── users.json            # 样例用户（zhangsan/lisi/admin）
│   └── (cache/ traces/ vector_index/ 运行期生成，gitignore)
├── web/                      # 🆕 Vite + React + TS
│   ├── package.json
│   ├── vite.config.ts        # dev 代理 /api /auth → http://localhost:8000
│   ├── tsconfig.json
│   ├── tailwind.config.js / postcss.config.js
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx           # 顶部导航 + 视图路由（state 切换，不引 router 依赖）
│       ├── api/
│       │   ├── client.ts     # fetch 封装 + JWT 注入
│       │   └── sse.ts        # SSE 类型化事件解析（fetch + ReadableStream）
│       ├── views/
│       │   ├── ChatView.tsx
│       │   ├── SearchView.tsx
│       │   └── StatsView.tsx
│       ├── components/
│       │   ├── UserSwitcher.tsx
│       │   ├── ChatWindow.tsx
│       │   ├── MessageBubble.tsx
│       │   ├── SourceCard.tsx
│       │   ├── TracePanel.tsx
│       │   └── SearchResults.tsx
│       ├── hooks/
│       │   ├── useAuth.ts    # JWT 状态 + login/demo-users
│       │   └── useSSEChat.ts # 流式问答状态机
│       ├── types.ts          # SSE 事件 / Source / Trace / User 类型
│       └── styles/index.css  # Tailwind 入口
└── docs/                     # 架构文档（沿用精选 + 本 spec）
    └── superpowers/specs/2026-07-08-groundrag-frontend-design.md
```

## 6. 后端改动（聚焦、最小）

### 6.1 SSE 升级为类型化 JSON 事件

`POST /api/v1/chat`（`stream=True`）改为按序发送：

```
data: {"type":"sources","items":[{"source","page","department","score","preview"}, ...]}

data: {"type":"token","text":"答案片段"}

data: {"type":"token","text":"..."}

data: {"type":"trace","trace":{"steps":[{"name","ms","tokens?"}],"total_ms","tokens":{"prompt","completion","total"},"cost_usd"}}

data: [DONE]
```

错误：`data: {"type":"error","message":"..."}\n\n` 后接 `data: [DONE]`。

### 6.2 `/chat` 非流式响应

`ChatResponse` 填真实数据（不再空）：

```python
class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]   # [{source, page, department, score, preview}]
    trace: dict           # {steps:[{name,ms,tokens?}], total_ms, tokens, cost_usd}
    trace_id: str         # 真实 trace_id（来自 Tracer）
```

### 6.3 流式路径接回 Tracer + token 统计

当前 `app/api.py` 的 `stream=True` 分支**绕过了 `rag.ask()`**（自写 query_rewrite + permission_aware_search + generate_answer_stream），导致无 trace、无 token。本次修复：

- 流式分支复用 `Tracer` 收集各步耗时 + token。
- 改用已有的 `generate_answer_stream_with_usage`（替换 `generate_answer_stream`）拿真实 token（usage 仅在最后一个 chunk）。
- 检索阶段记录检索到的 chunks → 发 `sources` 事件；生成阶段发 `token` 事件；结束发 `trace` 事件。
- **顺带修复** learn-codebase 阶段发现的"流式 vs ask() 路径分叉"问题。

实现建议：在 `EnterpriseRAGSystem` 上新增一个方法（如 `ask_stream_with_events`），统一封装"带权限的检索 + 流式生成 + trace 收集 + sources 提取"，供 API 流式分支调用；`ask()` 保持不变（CLI 仍用）。避免在 API 层重复编排逻辑。

### 6.4 新增 `GET /auth/demo-users`

供前端下拉框渲染（单一数据源，不在前端抄 users.json）：

```python
[{"user_id":"zhangsan","name":"张三","departments":["HR"],"role":"员工"}, ...]
```

复用 `user_service.users`（公开列表，均无密码 demo 账号，无敏感信息）。

### 6.5 FastAPI 托管前端（方案 A）

- 构建：`cd web && npm run build` → `web/dist/`。
- `app/api.py`：**在所有 API 路由注册之后**挂载：
  - `/assets`（Vite 产物）→ `StaticFiles(directory="web/dist/assets")`。
  - catch-all `GET /{path:path}` → 返回 `web/dist/index.html`（SPA 路由回退）。
- API 路由（`/health`、`/auth/*`、`/api/*`）先注册，优先匹配。
- dev 模式前端走 Vite 代理（`vite.config.ts` 代理 `/api`、`/auth` → `:8000`），**无需 CORS**（同源）。

### 6.6 `.env.example`

全字段模板，占位 key，注释说明每项用途。**绝不复制真实 key**。

---

## 7. 前端设计

### 7.1 技术栈

- React 18 + Vite 5 + TypeScript
- Tailwind CSS（配 frontend-design 技能做不落俗套的视觉）
- 不引 React Router（三视图用 state 切换，保持依赖精简）
- HTTP：原生 `fetch` 封装；SSE：`fetch` + `ReadableStream` 手解（`/chat` 是 POST，`EventSource` 仅支持 GET 不适用）

### 7.2 SSE 解析（`src/api/sse.ts`）

`fetch('/api/v1/chat', {method:'POST', body, headers:{Authorization}})` → 读 `response.body.getReader()` → 按 `data: ...\n\n` 分帧 → `JSON.parse` → 按 `type` 分发到回调（`onsources` / `ontoken` / `ontrace` / `onerror` / `ondone`）。

### 7.3 认证（`src/hooks/useAuth.ts`）

- `login(user_id)` → `POST /auth/login` → 存 JWT（内存 + localStorage）。
- `getDemoUsers()` → `GET /auth/demo-users`。
- 所有受保护请求注入 `Authorization: Bearer <jwt>`。
- 401 → 清 token + 提示重新选用户。

### 7.4 视图与组件（见 §5 目录树）

- `UserSwitcher`：下拉选用户 → 触发 `login` → 切换后清空当前对话（避免跨用户串内容）。
- `ChatWindow` + `MessageBubble`：流式渲染 token。
- `SourceCard`：渲染 `sources` 事件里的每条引用。
- `TracePanel`：折叠面板，渲染 `trace` 事件的 steps/tokens/cost。
- `SearchView` / `StatsView`：分别调 `/api/v1/search`、`/api/v1/stats`。

### 7.5 视觉方向

由 frontend-design 技能落地：克制的配色 + 清晰层级 + 流式打字效果 + 引用卡片 + trace 面板的 monospace 数据展示。避免默认 Tailwind 模板感。

---

### 7.6 中英双语（i18n）

- **轻量自研 i18n**，不引 react-i18next：`web/src/i18n/index.ts` 提供 `Lang`（`'zh'|'en'`）、`useLang()` 返回 `{ lang, setLang, t }`、完整 zh/en 字典（单文件两份对象）。
- **头部 LanguageToggle**（中 / EN 两个按钮），lang 存 localStorage，默认 `'zh'`。切换即时生效（context 驱动重渲染）。
- 所有用户可见 UI 文案走 `t('key')`；带参数用 `t('source_doc_label', {n: i})` 形式。
- **不翻译**数据内容：部门名、文档名、LLM 回答、用户姓名——只翻译 UI 框架文案（标签/按钮/占位符/空态/标题）。
- 字典是所有文案的单一来源；新增文案先加键再引用。

---

## 8. 数据卫生

- `data/docs/`：用**合成样例文档**替换 `mye/` 等真实公司文档。虚构公司（如"示例科技有限公司"），保留 HR / 财务 / IT / 行政 / 研发 部门结构，确保 RBAC 演示（HR↔财务↔admin 隔离）效果不变。年假天数 / 报销标准等内容用虚构但自洽的数值。样例**覆盖 md/txt/xlsx/docx/pdf 五种格式，每个部门含多种格式**（如 HR 含 md+txt+pdf，财务含 xlsx+docx），由 `scripts/make_sample_docs.py` 生成，演示后端多格式解析（PDF/DOCX/XLSX/MD/TXT）能力。
- `data/users.json`：沿用样例用户（zhangsan/lisi/admin）。
- `.env`：**绝不进新仓库**。新仓库用全新 `git init`，旧 history（含已 stage 的真实 key `.env`）不带过来，免去 scrub history。
- `AI-Engineer-Playbook/`、`demo/`、`data/project.jpg` 等个人/无关内容不进新仓库。

---

## 9. 错误处理

- **前端**：网络错误 / 401（token 过期 → 提示重选用户）/ 429（限流 → 友好提示）/ 500 / SSE `error` 事件 → 在对话区显示错误气泡，不崩页面。
- **后端**：流式异常时发 `error` 事件 + `[DONE]`；`Tracer.finish` 在 `finally` 中保证收尾（trace.jsonl 不断片）。沿用现有异常隔离风格。

## 10. 测试

- **后端**：现有 `tests/` 全过；新增 `tests/test_api_response.py`（或扩展 `test_core.py`）验证：
  - `/chat` 非流式返回真实 `sources` + `trace`（fake rag 注入）。
  - SSE 事件序列正确（sources → tokens → trace → [DONE]）。
  - `/auth/demo-users` 返回结构正确。
- **前端**：Vitest + React Testing Library，测 `sse.ts` 解析器 + `SourceCard`/`TracePanel` 渲染（轻量，覆盖关键纯函数）。
- **E2E（可选，后续）**：Playwright 跑一遍切换用户 → 提问 → 看引用/trace。

## 11. README / 作品集物料

- **双语**：`README.md`（英文主）+ `README.zh-CN.md`（中文），顶部互链 `[English](README.md) | [中文](README.zh-CN.md)`。
- 内容：项目一句话定位、特性亮点、架构图（数据流）、前置依赖（Python 3.12 / Node 18+ / LLM API key）、`make run` 一键运行、截图与 GIF 占位（前端建好后补）、已知限制（无密码登录、reranker 默认关）、技术亮点（面试向）。
- `LICENSE`：MIT（沿用 pyproject 声明）。
- `Makefile`：`setup`（装依赖）/ `dev`（起后端 + 前端 dev）/ `build`（前端构建）/ `run`（构建 + 起 uvicorn 单服务）。

## 12. 迁移 / 脚手架步骤（project1-rag → GroundRAG）

**落盘位置（已定）**：在当前工作区同级新建 `F:\code\python\GroundRAG\`（与 `agent-platform` 平级的独立目录），开发完成后再 push 到新 GitHub 仓库。**全新 `git init` / 全新 history**（不带旧 history，避免旧 `.env` 泄密）。

1. 新建 `F:\code\python\GroundRAG\` 根目录。
2. 拷贝 `project1-rag/` 下：`app/`、`rag_modules/`、`config.py`、`main.py`、`tests/`、`evaluation/`、`pyproject.toml`、`requirements.txt`、`docs/`、`locustfile.py` 到根。
3. 拷贝 `data/users.json`；`data/docs/` 用合成样例替换（不拷真实公司文档）。
4. **不拷** `.env`、`.venv/`、`data/cache/`、`data/traces/`、`data/vector_index/`、`__pycache__/`。
5. `pyproject.toml`：`name = "groundrag"`，描述更新。
6. 新建 `web/`（`npm create vite@latest web -- --template react-ts`）+ Tailwind 初始化。
7. 新建 `Makefile`、`.env.example`、`LICENSE`、`README.md`、`README.zh-CN.md`、加固 `.gitignore`。
8. 实施后端改动（§6）+ 前端（§7）。
9. `git init` + 首次提交（全新历史）。

## 13. 风险与注意

- **embedding 模型首次下载**（bge-small-zh ~100MB）：README 注明首次运行需联网下载；离线场景需预先放置模型。
- **reviewer 运行门槛**：需 Python 3.12 + Node 18+ + 自备 LLM key。README 明确前置条件，提供截图让不跑的人也能看效果。
- **流式 SSE 重构**：改动 `app/api.py` 流式分支 + 新增 system 方法，需保证 CLI `main.py` 的 `ask()` 不受影响（测试覆盖）。
- **样例文档自洽性**：合成文档的数值（年假天数等）要在多 chunk 间一致，否则演示出现矛盾答案。

## 14. 部署（Docker）

提供 `Dockerfile`（多阶段）+ `docker-compose.yml` + `.dockerignore` + `docs/deployment.md`，让 reviewer 一条 `docker compose up --build` 起服务，无需本地装 Python/Node。

- **多阶段 Dockerfile**：
  - Stage 1（前端构建）：`node:20-alpine`，`npm ci` + `npm run build` → `/web/dist`。
  - Stage 2（运行时）：`python:3.12-slim`，装系统依赖（git 等），`pip install -r requirements.txt`，**构建期预下载 bge embedding 模型并烘焙进镜像**（`RUN python -c "...HuggingFaceEmbeddings(...)"`），拷后端代码 + 从 stage1 拷 `web/dist`。
  - 环境变量：`HF_HOME` / `SENTENCE_TRANSFORMERS_HOME` 指向 `/app/.cache/...`（烘焙模型所在）；`OMP_NUM_THREADS=1` `MKL_NUM_THREADS=1`；运行时 `HF_HUB_OFFLINE=1`（用镜像内模型，不联网）。
  - **非 root 运行**：`useradd -m -r app` + `chown -R app /app` + `USER app`。
- **`.env` 不打进镜像**：通过 `docker-compose` 的 `env_file: .env` 运行时注入（绝不在 Dockerfile 里 COPY .env）。
- **docker-compose 卷**：
  - 命名卷 `rag-index` / `rag-cache` / `rag-traces` → `/app/data/{vector_index,cache,traces}`（运行期产物持久化）。
  - **不再挂模型卷**——模型已烘焙进镜像（挂卷反而会遮蔽）。`data/docs/` 与 `data/users.json` **来自镜像**（不挂载，避免空卷遮蔽样例文档）。
- **首请求不卡**：模型在镜像构建期已下载，容器首启即就绪（无需运行时下载 ~100MB）。镜像体积约 1.5–2GB（torch/transformers/faiss），README 注明。
- **部署文档** `docs/deployment.md`：前置（Docker）、`.env` 配置、`docker compose up --build`、构建期下模型说明、常见问题（构建期 HF 慢→`HF_ENDPOINT` 镜像）。
- **README** 双语各加一节 "🐳 Docker" 快速开始（与 `make run` 并列的替代方案）。
- **`.dockerignore`**：排除 `.git`、`.venv`、`.env`、`__pycache__`、`web/node_modules`、`web/dist`（镜像内构建）、`data/{cache,traces,vector_index}`、`.pytest_cache`、IDE 目录。

---

## 15. 验收标准

- [ ] 新仓库 `GroundRAG/` 干净：无真实 key、无真实公司文档、无个人笔记。
- [ ] `make run` 一条命令起服务，浏览器打开看到前端。
- [ ] Chat 视图：流式输出 + 引用卡片 + trace 面板正常。
- [ ] 切换用户（HR↔财务↔admin）能复现 RBAC 隔离。
- [ ] Search / Stats 视图正常取数。
- [ ] 后端 pytest 全过 + 新增响应形状测试通过。
- [ ] 前端 Vitest 关键测试通过。
- [ ] 双语 README 完整、含架构图与一键运行说明。
- [ ] `docker compose up --build` 一条命令起容器服务，`:8000` 可用；模型烘焙进镜像、首请求不卡；索引/缓存/traces 卷持久化（重建容器不重建索引）。
- [ ] `ask()` 与 `ask_stream()` 共用决策 helper（无重复逻辑）。
- [ ] 流式 `/chat` 不阻塞事件循环（并发请求不串行）。
- [ ] `CacheService` 加锁线程安全。
- [ ] `config.validate()` 对默认 `jwt_secret` 发 warning。
- [ ] Docker 容器非 root 运行；Dockerfile 构建期预下载模型。
- [ ] 配置缺失启动即 fail-fast（非首请求 500）。

---

## 16. 审计驱动的硬化项（生产就绪）

经"严格 RAG 架构师"复审补入，计划已落对应任务：

### 16.1 管线去重（DRY，P0）
`ask()`（CLI）与 `ask_stream()`（API 事件源）共用决策逻辑：抽 `_decide_intent / _rewrite_query / _retrieve` 三个 helper，两条管线都调它们，杜绝"改一处忘另一处"。

### 16.2 流式不阻塞事件循环（P1）
`/chat` 流式分支不在 `async def` 内直接迭代同步 `ask_stream`；改为**后台线程**跑 `ask_stream`，经 `asyncio.Queue` + `run_coroutine_threadsafe` 桥到 async 生成器，彻底不卡事件循环（并发请求不串行）。

### 16.3 CacheService 线程安全（P1）
`put/get/clear` 加 `threading.Lock`，防 `asyncio.to_thread` 并发下 `_cache`/`_hash_idx` 竞态、JSONL 损坏。

### 16.4 安全硬化（P1）
- `config.validate()` 检查 `jwt_secret`：等于默认值时 warning（不阻断本地）。
- Dockerfile 以**非 root 用户**运行（`useradd -m -r app` + `USER app`）。

### 16.5 启动 fail-fast + 镜像自包含（P1）
- FastAPI `lifespan` 启动时初始化 `_rag`（建索引）；配置缺失立即崩并打清晰日志，而非首请求 500。
- Dockerfile **构建期预下载** bge embedding 模型（镜像自包含，首请求秒回；代价是镜像更大）。

### 16.6 文档注明（P2，进 README"已知限制 / 安全边界"）
- token/成本为**估算**（依赖 streaming+include_usage，跨 provider/降级链行为不一）。
- 同源托管**不需 CORS**；前端若独立部署需自行加 CORS。
- 国内首启 HF 模型下载慢→设 `HF_ENDPOINT=https://hf-mirror.com`。
- FAISS pickle 反序列化（`allow_dangerous_deserialization`）仅用于本地受控文件，**永不加载外部上传索引**。
- 生产路径 **reranker 关闭**（RRF+MMR）；eval/ab_compare 用 reranker → **评测数字 ≠ 生产检索行为**。
