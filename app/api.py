"""
企业知识库 RAG API 服务（FastAPI）

本文件是整个系统的"门面"：浏览器/客户端只和它打交道，它再调用背后的
检索、生成、权限等模块。可理解为"前台接待"，负责：收请求 → 鉴权 →
限流 → 调度后端 → 返回结果。

接口（共 6 个端点）：
  GET  /health              — 健康检查（无需登录，给监控探针用）
  POST /auth/login          — 登录，签发 JWT（限流最严：5/min，防 user_id 枚举）
  GET  /auth/demo-users     — 列出 demo 用户（无需登录，供前端下拉框渲染）
  POST /api/v1/chat         — 问答接口（需 JWT，支持流式 SSE）
  POST /api/v1/search       — 纯检索接口（不调 LLM，省成本，用于调试）
  GET  /api/v1/stats        — 知识库统计（需 JWT）

注：FastAPI 还会自动暴露 /docs（Swagger UI）和 /redoc，无需手写。

用法：
  uvicorn app.api:app --reload --port 8000
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# 把项目根目录加入 sys.path，这样 `import config`、`from main import ...` 才能找到模块。
# __file__ 是当前文件 api.py，parent.parent 就是 project1-rag/ 根目录。
sys.path.insert(0, str(Path(__file__).parent.parent))

# 先配好日志（尽早调用，后续模块的日志才会按统一格式输出）
from rag_modules.logging_config import setup_logging
setup_logging()

# ---- 引入 Web 框架与各类依赖 ----
# FastAPI：现代异步 Web 框架，支持自动文档、类型校验、依赖注入
# HTTPException：抛出后自动转成 HTTP 错误响应（如 401/500）
# Request：原始请求对象（slowapi 限流需要它来取客户端 IP）
# Depends：FastAPI 依赖注入，用于把"鉴权"做成可复用依赖
from fastapi import FastAPI, HTTPException, Query, Request, Depends
# StreamingResponse：把一个"逐块产出数据"的生成器转成 HTTP 流式响应（SSE 用）
from fastapi.responses import StreamingResponse
# BaseModel/Field：Pydantic 数据模型，请求体进来会被自动校验（少传/多传/类型错都会报错）
from pydantic import BaseModel, Field
# slowapi：FastAPI 的限流库（基于内存计数，按 IP 分桶）
from slowapi import Limiter, _rate_limit_exceeded_handler
# get_remote_address：限流的 key 函数，按客户端 IP 区分（返回 IP 字符串）
from slowapi.util import get_remote_address
# RateLimitExceeded：超限时的异常类型，下面注册一个处理器把它转成 429 响应
from slowapi.errors import RateLimitExceeded

# 项目内部模块
from config import DEFAULT_CONFIG                # 全局配置（含 jwt_secret、jwt_expire_hours）
from main import EnterpriseRAGSystem             # RAG 主系统（检索+生成的大管家）
from app.auth import create_access_token, get_current_user  # JWT 签发 / JWT 校验依赖

# 本文件的日志器（日志会带 api. 前缀，方便在日志里定位是 Web 层还是后端层）
logger = logging.getLogger(__name__)

# ---- FastAPI 应用实例 ----
# 一个 app 就是一个"Web 服务"。路由函数（@app.get/...）注册到它上面。
# title/description/version 会自动渲染到 /docs 的 Swagger 页面，相当于在线 API 文档。
app = FastAPI(
    title="企业知识库 RAG 系统",
    description="混合检索 + Reranker + RBAC 权限隔离 + 多级缓存 + 全链路可观测",
    version="1.0.0",
)

# ---- 限流器配置 ----
# key_func=get_remote_address：按"客户端 IP"区分限流桶（同一 IP 共用一个配额）
# default_limits：没单独配限流的路由，默认每分钟最多 30 次（兜底防爬/防刷）
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
# 把限流器挂到 app.state，slowapi 内部会从这里取
app.state.limiter = limiter
# 注册异常处理器：触发限流时自动返回 429 Too Many Requests（而不是 500）
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---- 请求/响应模型（Pydantic）----
# 这些类定义"接口收发的 JSON 长什么样"。FastAPI 拿到请求体会自动校验字段，
# 缺字段/类型不对直接返回 422，无需手写 if 校验。返回时也会按模型过滤多余字段。

class LoginRequest(BaseModel):
    """登录请求体：只要一个 user_id（演示项目无密码）"""
    user_id: str = Field(..., description="用户ID")   # ... 表示必填（不能缺）

class TokenResponse(BaseModel):
    """登录成功返回：JWT token + 基本信息"""
    access_token: str                               # 签发的 JWT，后续请求放 Authorization 头
    token_type: str = "bearer"                      # OAuth2 标准写法，固定 bearer
    user_id: str                                    # 回显，方便客户端确认
    expires_in: int                                 # 有效期（秒），前端据此定时刷新

class ChatRequest(BaseModel):
    """问答请求体"""
    question: str = Field(..., description="用户问题")          # 必填，用户的问题
    stream: bool = Field(default=False, description="是否流式输出")  # True → 走 SSE 逐字返回

class ChatResponse(BaseModel):
    """非流式问答的返回（流式走 SSE，不走这个模型）"""
    answer: str                       # LLM 生成的回答
    sources: list[str] = []           # 引用来源（当前留空，可扩展为文档名+片段）
    trace_id: str = ""                # 全链路追踪 ID（可对接 OpenTelemetry）

class SearchRequest(BaseModel):
    """纯检索请求体"""
    question: str                     # 检索 query
    top_k: int = Field(default=3)     # 返回前 K 条最相关片段，默认 3

class SearchResponse(BaseModel):
    """纯检索结果：每个片段含来源、分数、预览文本"""
    results: list[dict]

# ---- RAG 系统单例（全局只初始化一次）----
# 加载向量索引、embedding 模型、LLM 客户端都很慢（秒级~十秒级），
# 所以做单例：进程内只建一次，之后所有请求复用同一个 _rag。
_rag: EnterpriseRAGSystem = None


def get_rag() -> EnterpriseRAGSystem:
    """
    懒加载 RAG 系统（首次请求时才初始化）。

    为什么懒加载而非在 startup 里初始化：
      - startup 不 await 完毕会拖慢服务启动；放这里让首个请求触发，逻辑简单可控。
      - 注意：单例模式只适合"无请求级状态"的对象。当前 _rag 不应存任何
        与具体用户/请求相关的字段（见下方 _resolve_departments 的 A2 说明）。
    """
    global _rag                       # 声明改的是模块级变量，不是局部变量
    if _rag is None:                  # 首次调用时才进
        logger.info("🚀 首次请求，初始化 RAG 系统...")
        _rag = EnterpriseRAGSystem()  # 创建系统实例（读配置、连 LLM/向量库）
        _rag.initialize()             # 初始化各子模块（embedding、检索、生成、缓存…）
        _rag.build_knowledge_base()   # 构建/加载向量索引（最耗时的一步）
        logger.info("✅ RAG 系统就绪")
    return _rag


def _resolve_departments(rag: EnterpriseRAGSystem, user_id: str):
    """
    从 JWT 解出的 user_id 现查"该用户可见哪些部门"（请求级，每次现查）。

    A2 修复（并发竞态）：权限信息每次请求现查现用、随调用栈传递，绝不挂在进程级
    单例 _rag 上——否则并发请求会互相覆盖 current_user，导致 A 用户用 B 的部门做
    权限过滤，从而跨权限泄露文档。简单说：单例不能存"谁登录了"这种逐请求的状态。
    """
    if not user_id:                                 # 没拿到 user_id 直接放行（无权限过滤）
        return None
    try:
        # 调 UserService 查 user_id → 部门列表（如 ["研发部","产品部"]）
        return rag.user_service.get_departments(user_id)
    except Exception:
        # 查询失败也别让整个请求挂掉，降级为"无权限过滤"
        return None


# ---- 生命周期事件 ----
# startup 钩子：服务启动时执行一次。这里只记日志，真正的初始化在 get_rag() 懒加载。
# （旧版 FastAPI 用 @app.on_event；新版推荐 lifespan，但此处保持兼容。）
@app.on_event("startup")
async def startup():
    logger.info("RAG API 服务启动")


# ---- 端点 1/5：健康检查 ----
@app.get("/health")
def health():
    """
    健康检查（无需认证）。

    用途：docker healthcheck / k8s liveness probe / 负载均衡探活。
    返回 200 即代表服务活着。注意会顺带触发 RAG 初始化（首请求可能慢）。
    """
    rag = get_rag()                                # 确保 RAG 已就绪（顺带懒加载）
    return {
        "status": "ok",                            # 固定 ok，探针只看 HTTP 状态码即可
        "rag_ready": rag is not None,              # 后端是否就绪
        "cache_size": rag.cache.stats()["total"] if rag.cache else 0,  # 缓存条目数（间接反映热度）
    }


# ---- 端点 2/5：登录（签发 JWT）----
@app.post("/auth/login", response_model=TokenResponse)
# 限流最严：每分钟 5 次。理由（A3 缓解）：本项目登录无密码（用 user_id 是否存在来
# "认证"），攻击者可暴力枚举 user_id 探测哪些账号存在。限流把枚举速度压到极低，
# 等于变相提高爆破成本。注：default_limits 30/min 会被这里更严的 5/min 覆盖。
@limiter.limit("5/minute")  # A3 缓解：登录无密码（已知短板），限流防 user_id 枚举爆破
def login(req: LoginRequest, request: Request):
    """
    登录获取 JWT（系统唯一的认证入口）。

    真实系统这里校验密码（查 DB / 接 SSO / OAuth）。
    本项目用 UserService 校验 user_id 是否存在（演示用，无密码）。

    为什么 request 形参必须有：slowapi 的 @limiter.limit 依赖它来取客户端 IP，
    形参名必须是 request；漏写会报错。这是 slowapi 的固定写法。
    """
    rag = get_rag()                                # 取 RAG 单例（要用它的 user_service）
    user = rag.user_service.authenticate(req.user_id)  # 校验 user_id 是否存在（无密码）
    if not user:
        # 不存在 → 401。注意：真实系统对"用户不存在/密码错"应返回相同模糊提示，
        # 避免泄露"哪个账号存在"；此处演示直接回显 user_id，配合限流缓解风险。
        raise HTTPException(status_code=401, detail=f"用户不存在: {req.user_id}")
    # 用 app.auth.create_access_token 签发 JWT：把 user_id 写进 token 的 sub 字段，
    # 并用配置里的密钥+过期时间签名。客户端拿到后，后续每个请求带在 Authorization 头。
    token = create_access_token(
        req.user_id,
        secret=DEFAULT_CONFIG.jwt_secret,          # 密钥来自配置（生产用环境变量）
        expires_hours=DEFAULT_CONFIG.jwt_expire_hours,
    )
    return TokenResponse(
        access_token=token,
        user_id=req.user_id,
        expires_in=DEFAULT_CONFIG.jwt_expire_hours * 3600,  # 小时 → 秒
    )


# ---- 端点 3/6：列出 demo 用户（供前端下拉框渲染）----
@app.get("/auth/demo-users")
def demo_users():
    """列出 demo 用户，供前端下拉框渲染（公开：均为无密码 demo 账号）。"""
    rag = get_rag()
    users = []
    for uid, info in rag.user_service.users.items():
        users.append({
            "user_id": uid,
            "name": info.get("name", uid),
            "departments": info.get("departments", []),
            "role": info.get("role", ""),
        })
    return {"users": users}


# ---- 端点 4/6：问答（核心接口）----
@app.post("/api/v1/chat")
# 限流：每 IP 每分钟 20 次。比全局 30/min 严——因为 chat 要调 LLM，
# 既贵又慢，更需防滥用。（叠加在默认 30/min 之上，以更严的为准。）
@limiter.limit("20/minute")  # 每个IP每分钟20次
# user_id = Depends(get_current_user)：FastAPI 依赖注入。
# 每次请求进来，FastAPI 自动调 get_current_user：从 Authorization 头取 JWT →
# 校验签名/过期 → 返回可信 user_id。校验失败它内部会抛 401，请求根本进不到这里。
# 所以 user_id 是服务端从 token 解出的，不是客户端传的 → 无法伪造。
async def chat(req: ChatRequest, request: Request, user_id: str = Depends(get_current_user)):
    """
    问答接口（异步，支持流式 SSE，需 JWT 认证）。

    整条链路：JWT 取 user_id → 现查该用户可见部门 → 带部门做权限过滤检索 →
    喂给 LLM 生成回答（流式或一次性）。权限信息全程随调用栈传递，不写单例。
    """
    rag = get_rag()
    # user_id 来自 JWT 解码，不是客户端请求体 → 无法伪造
    # A2 修复：权限现查现用，不写单例 current_user（并发安全）
    # 现查这个用户能看哪些部门（请求级，每次重新查，避免并发串号）
    deps = _resolve_departments(rag, user_id)

    if req.stream:
        # ===== 流式分支：用 SSE（Server-Sent Events）逐块返回 =====
        # SSE 协议：响应体是一行行 "data: <内容>\n\n"，浏览器用 EventSource 接收，
        # 每来一块就渲染一块（像 ChatGPT 打字机效果）。
        # deps 通过闭包传入 generate()，不读任何单例 → 并发安全。
        async def generate():
            try:
                from rag_modules.generation_integration import GenerationIntegrationModule
                # ① 查询改写：把口语化问题改成更适合检索的形式（如同义词扩展）
                rewritten = rag.generation_module.query_rewrite(req.question)
                # ② 带权限的检索：有部门信息就用 permission_aware_search（只召回
                #    该用户可见的文档），否则退回普通 hybrid_search（无过滤）
                if deps:
                    chunks = rag.retrieval_module.permission_aware_search(rewritten, deps, top_k=rag.config.top_k)
                else:
                    chunks = rag.retrieval_module.hybrid_search(rewritten, top_k=rag.config.top_k)
                # ③ 没检索到任何东西 → 提前结束流（避免把空内容丢给 LLM 瞎编）
                if not chunks:
                    yield "data: 知识库中未找到相关内容\n\n"
                    yield "data: [DONE]\n\n"        # [DONE] 是约定的流结束标记
                    return
                # ④ 流式生成：generate_answer_stream 是个生成器，每产出一小段就 yield 出去
                for piece in rag.generation_module.generate_answer_stream(req.question, chunks):
                    yield f"data: {piece}\n\n"     # SSE 格式：每条消息以两个换行结尾
                yield "data: [DONE]\n\n"            # 正常结束
            except Exception as e:
                # 生成中途出错也要通过流告知前端，不能让连接悬着
                yield f"data: [ERROR] {str(e)}\n\n"
        # StreamingResponse：把上面的 generate() 生成器包装成流式 HTTP 响应
        # media_type="text/event-stream" 是 SSE 的标准 Content-Type
        return StreamingResponse(generate(), media_type="text/event-stream")

    # ===== 非流式分支：一次性返回完整答案 =====
    try:
        # rag.ask 是同步阻塞函数（内部调 LLM 等待完整返回）。
        # 用 asyncio.to_thread 把它丢到线程池跑，避免阻塞 FastAPI 的事件循环，
        # 这样其他请求（包括流式的）不会被卡住。
        answer = await asyncio.to_thread(
            rag.ask, req.question, stream=False,
            user_departments=deps, user_id=user_id)   # 权限/身份作为参数传入，不挂单例
        return ChatResponse(answer=answer or "", sources=[], trace_id="")
    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))  # 500 = 服务端内部错误


# ---- 端点 4/5：纯检索（不调 LLM）----
@app.post("/api/v1/search", response_model=SearchResponse)
def search(req: SearchRequest, user_id: str = Depends(get_current_user)):
    """
    纯检索接口：不调 LLM，只返回命中的文档片段。

    用途：①调试检索质量（看召回对不对）②省成本（LLM 调用要钱，这里不调）。
    同样走 JWT 鉴权 + 权限过滤（A2 修复：不写单例）。
    """
    rag = get_rag()
    deps = _resolve_departments(rag, user_id)  # A2 修复：不写单例
    # 有部门信息 → 带权限过滤检索；没有 → 普通 hybrid_search
    chunks = rag.retrieval_module.permission_aware_search(
        req.question, deps, top_k=req.top_k) if deps else rag.retrieval_module.hybrid_search(req.question, top_k=req.top_k)

    # 把检索到的 chunk 对象转成前端好处理的字典
    results = []
    for c in chunks:
        results.append({
            "source": c.metadata.get("source", ""),          # 来源文件名
            "chunk_index": c.metadata.get("chunk_index"),    # 文档内分片序号
            # 分数：优先用 rerank 分（更准），没有就用向量相似度，再没有就 0
            "score": c.metadata.get("rerank_score") or c.metadata.get("vector_sim", 0),
            "preview": c.page_content[:200],                 # 取前 200 字作预览
        })
    return SearchResponse(results=results)


# ---- 端点 5/5：知识库统计 ----
@app.get("/api/v1/stats")
def stats(user_id: str = Depends(get_current_user)):
    """
    知识库统计（需 JWT 认证）。

    给运维/管理看：知识库有多少文档、缓存命中情况等。同样要登录，
    避免匿名探测系统规模。
    """
    rag = get_rag()
    return {
        # 文档/分片数量统计（data_module 负责）
        "knowledge_base": rag.data_module.get_statistics() if rag.data_module else {},
        # 缓存统计（命中率等，反映热度与成本节约）
        "cache": rag.cache.stats() if rag.cache else {},
    }
