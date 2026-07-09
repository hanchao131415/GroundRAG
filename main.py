"""
企业级 RAG 知识库问答系统 - 主入口

【本文件在系统中的定位：编排者 / Director】
main.py 不实现具体功能，它的职责是"把各个模块组装起来并控制执行顺序"。
这是"编排模式"（Orchestrator Pattern）——像一个指挥家，自己不演奏乐器，
但决定哪个乐手在什么时候演奏什么声部。

数据流全景（对照前面注释过的模块）：
    用户提问
      │
      ▼
    ┌────────────────────────────────────────────────────────┐
    │  ask() 主流程（本文件）                                  │
    │                                                          │
    │  ① 缓存查询 ────── 命中 → 秒回（跳过所有 LLM 调用）        │
    │  ② 意图路由 ────── 明确问题短路跳过（省 20s）              │
    │  ③ 查询改写 ────── 明确问题短路跳过（省 6s）               │
    │  ④ 权限检索 ────── permission_aware_search（RBAC 过滤）   │
    │  ⑤ 流式生成 ────── 防幻觉 prompt + 引用溯源 + token 统计  │
    │  ⑥ 存缓存 ──────── 下次相同/相似问题秒回                  │
    └────────────────────────────────────────────────────────┘
      │
      ▼
    返回答案（全程被 Tracer 追踪，记录每步耗时/token/成本）

跑通验证：
  python main.py          # 交互问答（启动后先选用户身份）
  python main.py rbac     # 权限隔离演示（自动跑3个场景）
"""

import sys          # 读取命令行参数（sys.argv）+ 异常退出（sys.exit）
import logging
from pathlib import Path
from typing import List, Optional

# 从 config.py 导入默认配置实例（dataclass 实例，全环境变量可配）
from config import DEFAULT_CONFIG
# 导入四大核心模块（数据/索引/检索/生成）—— 这就是 RAG 的骨架
from rag_modules import (
    DataPreparationModule,        # ① 数据接入与切分
    IndexConstructionModule,      # ② 向量化与索引
    RetrievalOptimizationModule,  # ④ 混合检索 + RBAC
    GenerationIntegrationModule,  # ⑤ LLM 生成
)
from rag_modules.user_service import UserService       # 用户→部门映射（RBAC 数据源）
from rag_modules.logging_config import setup_logging   # 统一日志配置

# 启动时配置日志（幂等：重复调用不会叠加 handler）
# 放在模块顶层：确保 import main 时日志就配好了，后续 logger 都能正常输出
setup_logging()
logger = logging.getLogger(__name__)


class EnterpriseRAGSystem:
    """企业知识库 RAG 系统（带 RBAC 权限）

    【系统设计：门面对象（Facade）】
    把 data/index/retrieval/generation 四大模块 + cache/user_service 装在一个对象里，
    对外只暴露 initialize() / build_knowledge_base() / ask() 三个方法。
    调用方不用关心内部有 6 个模块，只跟 EnterpriseRAGSystem 打交道。
    """

    def __init__(self, config=DEFAULT_CONFIG):
        """构造函数只做轻量初始化：存配置、建空壳、做配置校验。

        【为什么不在 __init__ 里就加载模型？】
        分离"构造"和"初始化"是工程惯例：
        - __init__ 应该快、无副作用、可失败回退（这里只 validate 配置）
        - initialize() 才做重活（加载 embedding 模型、连 LLM、初始化缓存）
        这样测试时可以只构造不初始化，单元测试更快。
        """
        self.config = config
        # 四大模块的引用，先置 None，等 initialize() 才真正创建
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None
        self.cache = None  # 多级缓存（initialize 时初始化）
        # UserService 轻量（只读 users.json），可以直接在构造时创建
        self.user_service = UserService()
        # CLI 模式下记录当前登录用户；API 模式不使用这个字段（避免 A2 并发竞态）
        self.current_user: Optional[str] = None

        # 【Fail-Fast 早失败原则】启动时立刻校验配置，缺 key/路径不存在直接退出
        # 比跑了一半才崩好得多（用户能立刻看到"哪里配错了"而不是"为什么中途挂了"）
        missing = config.validate()
        if missing:
            logger.error(f"配置缺失: {missing}")
            sys.exit(1)

    def initialize(self):
        """初始化所有模块（重操作：加载 embedding 模型 ~7s、建 LLM 连接）。

        【模块创建顺序的讲究】
        1. data_module / index_module / generation_module 互相独立，顺序无所谓
        2. cache 复用 index_module 的 embeddings（语义缓存要算 query 向量），
           所以必须在 index_module 之后创建——这是"依赖注入"，省一份模型内存
        """
        logger.info("🚀 初始化企业 RAG 系统...")
        # ① 数据模块：配置 chunk_size/overlap
        self.data_module = DataPreparationModule(
            self.config.data_path, self.config.chunk_size, self.config.chunk_overlap)
        # ② 索引模块：加载 embedding 模型（这一步最慢，约 7 秒）
        self.index_module = IndexConstructionModule(
            self.config.embedding_model, self.config.index_save_path)
        # ⑤ 生成模块：建 LLM 连接（含降级链自动装配）
        self.generation_module = GenerationIntegrationModule(
            llm_provider=self.config.llm_provider,
            llm_base_url=self.config.llm_base_url,
            llm_api_key=self.config.llm_api_key,
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        # 【依赖注入】多级缓存复用 index_module 的 embedding 模型做语义匹配
        # 为什么复用？bge-small-zh 约 100MB 内存，加载两份浪费；同一份模型查询/缓存共用
        # try/except：缓存是"锦上添花"，挂了不影响主流程（降级为无缓存）
        try:
            from rag_modules.cache_service import CacheService
            self.cache = CacheService(embeddings=self.index_module.embeddings)
            logger.info(f"✅ 多级缓存启用（{self.cache.stats()['total']} 条）")
        except Exception as e:
            logger.warning(f"缓存不可用: {e}")
            self.cache = None
        # Langfuse 云端可观测（同样是降级设计，key 没配就跳过）
        from rag_modules.observability import get_langfuse_handler
        lf_handler = get_langfuse_handler()
        if lf_handler:
            self.generation_module.set_langfuse_handler(lf_handler)
        logger.info("✅ 模块初始化完成")

    def _init_reranker(self):
        """加载 bge-reranker（transformers 原生加载，兼容 Windows CPU）。

        reranker.py v2 用 transformers 原生 AutoModelForSequenceClassification 加载，
        绕过了 FlagEmbedding C 扩展在 Windows 上的 0xC0000005 崩溃。
        首次下载模型约 1.1GB，之后缓存复用。加载失败时降级为 None（不阻塞主流程）。
        """
        try:
            from rag_modules.reranker import Reranker
            reranker = Reranker()
            logger.info("✅ bge-reranker 加载成功（cross-encoder 精排已启用）")
            return reranker
        except Exception as e:
            logger.warning(f"⚠️ bge-reranker 加载失败，降级为 RRF+MMR 无精排: {e}")
            return None

    def build_knowledge_base(self):
        """构建知识库：增量索引，文档变化时自动清缓存

        【整体流程】（对应数据流 ①②③）
        load_documents → chunk_documents → build_incremental → save_index
        → build_department_indexes → (有变化时)cache.clear → 建 retrieval_module
        """
        logger.info("📚 构建知识库（增量模式）...")
        # ① 加载所有文档（PDF/Excel/Word/MD），返回 Document 列表
        self.data_module.load_documents()
        # ② 切分（表格感知 + 递归切分），返回 chunk 列表
        chunks = self.data_module.chunk_documents()
        # ③ 增量构建向量索引：content_hash 对比，只对变化的 chunk 重新 embedding
        # 返回 (vectorstore, has_changes)：has_changes=True 表示有文档增删改
        vectorstore, has_changes = self.index_module.build_incremental(chunks)
        self.index_module.save_index()
        # 【RBAC 关键】按部门建子索引（真·先过滤用；零重复嵌入，从全库向量切片）
        dept_indexes = self.index_module.build_department_indexes(vectorstore)
        # 【坑29 缓存一致性】文档变了 → 缓存必须清空，否则返回旧答案
        # 场景：HR 年假制度改了（15天→20天），但缓存还存着"15天"的旧答案 → 误答
        if has_changes and self.cache:
            self.cache.clear()
        # ④ 检索模块：注入 vectorstore + chunks + config + 部门子索引
        self.retrieval_module = RetrievalOptimizationModule(
            vectorstore, chunks, reranker=self._init_reranker(),
            config=self.config, dept_indexes=dept_indexes)
        # 打印统计：文档数/chunk数/部门分布/平均chunk大小（运维监控用）
        stats = self.data_module.get_statistics()
        logger.info(f"📊 知识库统计: {stats}")

    # ===== 权限：身份识别 =====
    def login(self) -> Optional[str]:
        """模拟登录：选用户身份（真实系统用 JWT/Session 替代）

        【这是 CLI 演示用的简化登录】
        真实生产环境不会用 input() 选用户——应该走 app/auth.py 的 JWT 流程。
        这里保留是为了让用户能快速切换身份测试 RBAC（HR 看不到财务文档等场景）。
        """
        print("\n" + "=" * 60)
        print("🔐 选择登录用户（决定可见的文档范围）:")
        print(self.user_service.list_users())
        print("=" * 60)
        uid = input("输入用户ID（如 zhangsan/lisi/admin）: ").strip()
        user = self.user_service.authenticate(uid)
        if not user:
            print(f"❌ 用户 {uid} 不存在")
            return None
        self.current_user = uid  # CLI 单线程，写实例字段安全（API 并发路径不写这里）
        print(f"✅ 已登录: {user['name']}，可见部门: {user['departments']}")
        return uid

    def _current_departments(self) -> Optional[List[str]]:
        """获取当前用户的可见部门列表。

        返回 None 表示"无权限控制"（未登录，全库可见）——这是 CLI 的默认行为。
        返回 ["HR"] 表示只能看 HR 部门文档。
        返回 ["*"] 是 admin 通配符，表示全库可见。
        """
        if self.current_user:
            return self.user_service.get_departments(self.current_user)
        return None

    # ===== 问答主流程（系统的核心方法）=====
    def ask(self, question: str, stream: bool = True,
            user_departments: List[str] = None, user_id: str = None):
        """问答：缓存查询→短路路由→意图→改写→检索→生成（全程 trace）

        【A2 并发竞态修复的核心设计（重要！）】
        权限完全由 user_departments 参数决定（API 显式传参），不读实例的 self.current_user。
        为什么？因为 API 是多线程并发的：
          - 请求 A（HR 用户）和请求 B（财务用户）同时到达
          - 如果都读 self.current_user，可能 A 把它设成 HR，B 紧接着设成财务
          - A 执行到检索时读到的 self.current_user 已经是财务 → 跨权限泄露！
        所以 API 路径必须显式传 user_departments，绝不依赖实例状态。
        user_id 参数仅用于 trace 标识和打印，不参与权限判断。
        CLI 路径（单线程）不传参时，fallback 到 self.current_user，无竞态风险。
        """
        # 延迟导入：tracer 只在 ask 时才需要，避免启动时加载
        from rag_modules.tracer import Tracer, elapsed_ms, time_it

        # CLI 未显式传 departments 时，从当前登录用户推导（API 总是显式传）
        if user_departments is None:
            user_departments = self._current_departments()

        # 初始化链路追踪（记录每步耗时/token/成本，落盘 trace.jsonl）
        tracer = Tracer()
        effective_user = user_id or self.current_user   # trace 显示用的用户标识
        tr = tracer.start(question, effective_user)     # start 返回一个 trace 句柄
        print(f"\n❓ 问题: {question}")
        if user_departments:
            print(f"👤 当前用户: {effective_user}，可见部门: {user_departments}")

        try:
            # ===== ① 缓存查询（高频问题秒回，跳过所有 LLM 调用）=====
            # 【性能】缓存命中时延迟从 5 秒降到 0.1 毫秒（差 50000 倍）
            # 缓存带权限签名隔离：HR 和财务的缓存分开存，防跨权限泄露（坑26）
            if self.cache:
                t0 = time_it()
                hit = self.cache.get(question, departments=user_departments)
                if hit:
                    print(f"⚡ 缓存命中({hit['cache_type']})，直接返回")
                    print(f"✍️ 回答: {hit['answer']}")
                    tracer.step(tr, "缓存命中", question, hit["cache_type"], elapsed_ms(t0))
                    tracer.finish(tr, hit["answer"], provider=self.config.llm_provider)
                    tracer.print_summary(tr)
                    return hit["answer"]

            # ===== ② 意图路由（共享 helper：明确问题短路，省 ~20s）=====
            intent, ms = self._decide_intent(question)
            tracer.step(tr, "意图路由", question, intent, ms)
            print(f"🎯 意图: {intent}")
            # 闲聊直接结束，不走检索（比如"你好""谢谢"）
            if intent == "chitchat":
                print("💬 (闲聊，不检索)")
                tracer.finish(tr, "(闲聊)", provider=self.config.llm_provider)
                tracer.print_summary(tr)
                return

            # ===== ③ 查询改写（共享 helper：明确问题短路，省 ~6s）=====
            rewritten, ms = self._rewrite_query(question)
            tracer.step(tr, "查询改写", question, rewritten, ms)
            print(f"📝 查询改写: {question} → {rewritten}")

            # ===== ④ 权限感知检索（共享 helper：RBAC 核心）=====
            chunks, ms = self._retrieve(rewritten, user_departments)
            sources = [c.metadata.get("source", "?") for c in chunks]
            tracer.step(tr, "检索", rewritten, sources, ms)
            # 【拒答兜底】检索为空时，明确告知"未找到"，而不是让 LLM 硬编（防幻觉）
            if not chunks:
                print("❌ 未检索到（或当前用户无权访问相关文档）")
                tracer.finish(tr, "未找到", status="no_result", provider=self.config.llm_provider)
                tracer.print_summary(tr)
                return "知识库中未找到相关内容"
            print(f"🔍 检索到 {len(chunks)} 块，来源: {sources}")

            # ===== ⑤ 生成（支持流式/非流式）=====
            print("✍️ 回答:")
            t0 = time_it()
            if stream:
                # 【坑18/37 流式 token 统计】流式输出体验好（边生成边显示），
                # 但难点是拿 token 用量——只有最后一个 chunk 带 usage_metadata
                # generate_answer_stream_with_usage 返回 (文本片段, usage) 元组，
                # usage 在最后一个 chunk 才有值，前面都是 None
                collected = ""
                stream_usage = None
                for piece, usage in self.generation_module.generate_answer_stream_with_usage(
                        question, chunks):
                    collected += piece
                    print(piece, end="", flush=True)   # flush=True 立即输出，不等缓冲区
                    if usage:
                        stream_usage = usage  # 最后一个 chunk 带的真实 token 数
                print()
                answer = collected
                if stream_usage:
                    tok = stream_usage
                else:
                    # 【兜底】部分 provider 流式不返回 usage，按字符数粗估（中文≈2字符/token）
                    est = max(1, len(answer) // 2)
                    tok = {"prompt": est, "completion": est, "total": est * 2}
                tracer.step(tr, "生成", question, answer, elapsed_ms(t0), tok)
            else:
                # 非流式：一次性 invoke，直接拿完整响应
                resp = self.generation_module.llm.invoke(
                    self.generation_module.ANSWER_PROMPT.format_messages(
                        context=self.generation_module._build_context(chunks), question=question),
                    config=self.generation_module._cb())
                answer = resp.content if hasattr(resp, "content") else str(resp)
                print(answer)
                tok = _extract_tokens(resp)   # 从响应对象多路径提取 token 用量
                tracer.step(tr, "生成", question, answer, elapsed_ms(t0), tok)

            # ===== ⑥ 存缓存（下次相同/相似问题秒回）=====
            # 带权限签名存入：HR 的查询存 HR 缓存区，财务的查询存财务缓存区
            if self.cache and answer:
                self.cache.put(question, answer, departments=user_departments)

            tracer.finish(tr, answer, provider=self.config.llm_provider)
            tracer.print_summary(tr)
            return answer

        except Exception as e:
            # 【全程 trace 兜底】即使异常也要 finish trace，保证 trace.jsonl 完整
            # 否则出错的那次请求在 trace 里会"断片"（只有 start 没有 finish）
            logger.error(f"处理出错: {e}")
            tracer.finish(tr, "", status="error", error=str(e), provider=self.config.llm_provider)
            tracer.print_summary(tr)

    # ===== 共享决策 helper（ask 与 ask_stream 共用，单一事实源）=====
    def _decide_intent(self, question: str):
        """返回 (intent, elapsed_ms)。明确问题短路，省 ~20s LLM 调用。"""
        from rag_modules.tracer import elapsed_ms, time_it
        from rag_modules.cache_service import is_simple_query
        t0 = time_it()
        intent = "retrieval" if is_simple_query(question) else self.generation_module.query_router(question)
        return intent, elapsed_ms(t0)

    def _rewrite_query(self, question: str):
        """返回 (rewritten, elapsed_ms)。明确问题短路，省 ~6s。"""
        from rag_modules.tracer import elapsed_ms, time_it
        from rag_modules.cache_service import is_simple_query
        t0 = time_it()
        rewritten = question if is_simple_query(question) else self.generation_module.query_rewrite(question)
        return rewritten, elapsed_ms(t0)

    def _retrieve(self, rewritten: str, user_departments):
        """返回 (chunks, elapsed_ms)。有部门→permission_aware_search，否则 hybrid_search。"""
        from rag_modules.tracer import elapsed_ms, time_it
        t0 = time_it()
        if user_departments is not None:
            chunks = self.retrieval_module.permission_aware_search(rewritten, user_departments, top_k=self.config.top_k)
        else:
            chunks = self.retrieval_module.hybrid_search(rewritten, top_k=self.config.top_k)
        return chunks, elapsed_ms(t0)

    def ask_stream(self, question: str, user_departments=None, user_id: str = None):
        """流式问答，产出类型化事件 dict 序列（供 API 转 SSE）。

        事件序列：sources → token(若干) → trace → done（异常时插 error）。
        这条路径补齐了原 streaming 分支缺失的 trace/token 统计，
        并与非流式 ask() 对齐。
        """
        from rag_modules.tracer import Tracer, elapsed_ms, time_it

        if user_departments is None:
            user_departments = self._current_departments()

        tracer = Tracer()
        effective_user = user_id or self.current_user
        tr = tracer.start(question, effective_user)

        try:
            # 缓存命中短路
            if self.cache:
                t0 = time_it()
                hit = self.cache.get(question, departments=user_departments)
                if hit:
                    tracer.step(tr, "缓存命中", question, hit["cache_type"], elapsed_ms(t0))
                    yield {"type": "sources", "items": []}
                    yield {"type": "token", "text": hit["answer"]}
                    tracer.finish(tr, hit["answer"], provider=self.config.llm_provider)
                    yield {"type": "trace", "trace": self._trace_public(tr)}
                    yield {"type": "done"}
                    return

            intent, ms = self._decide_intent(question)
            tracer.step(tr, "意图路由", question, intent, ms)
            if intent == "chitchat":
                yield {"type": "sources", "items": []}
                yield {"type": "token", "text": "(闲聊，不检索)"}
                tracer.finish(tr, "(闲聊)", provider=self.config.llm_provider)
                yield {"type": "trace", "trace": self._trace_public(tr)}
                yield {"type": "done"}
                return

            rewritten, ms = self._rewrite_query(question)
            tracer.step(tr, "查询改写", question, rewritten, ms)

            chunks, ms = self._retrieve(rewritten, user_departments)
            sources = [self._source_item(c) for c in chunks]
            tracer.step(tr, "检索", rewritten, sources, ms)
            yield {"type": "sources", "items": sources}

            if not chunks:
                msg = "知识库中未找到相关内容"
                yield {"type": "token", "text": msg}
                tracer.finish(tr, msg, status="no_result", provider=self.config.llm_provider)
                yield {"type": "trace", "trace": self._trace_public(tr)}
                yield {"type": "done"}
                return

            # 流式生成 + usage
            t0 = time_it()
            usage = None
            collected = ""
            for piece, u in self.generation_module.generate_answer_stream_with_usage(question, chunks):
                collected += piece
                yield {"type": "token", "text": piece}
                if u:
                    usage = u
            if not usage:
                est = max(1, len(collected) // 2)
                usage = {"prompt": est, "completion": est, "total": est * 2}
            tracer.step(tr, "生成", question, collected, elapsed_ms(t0), usage)

            if self.cache and collected:
                self.cache.put(question, collected, departments=user_departments)

            tracer.finish(tr, collected, provider=self.config.llm_provider)
            yield {"type": "trace", "trace": self._trace_public(tr)}
            yield {"type": "done"}

        except Exception as e:
            logger.error(f"ask_stream 出错: {e}")
            tracer.finish(tr, "", status="error", error=str(e), provider=self.config.llm_provider)
            yield {"type": "error", "message": str(e)}
            yield {"type": "trace", "trace": self._trace_public(tr)}
            yield {"type": "done"}

    @staticmethod
    def _source_item(c) -> dict:
        md = c.metadata
        return {
            "source": md.get("source", ""),
            "page": md.get("page"),
            "department": md.get("department", ""),
            "score": md.get("rerank_score") or md.get("vector_sim", 0),
            "preview": c.page_content[:200],
        }

    @staticmethod
    def _trace_public(tr) -> dict:
        return {
            "trace_id": tr["trace_id"],
            "steps": [{"name": s["name"], "ms": s["ms"], "tokens": s.get("tokens")} for s in tr["steps"]],
            "total_ms": tr["total_ms"],
            "tokens": tr["tokens"],
            "cost_usd": tr["cost_usd"],
        }

    def run_interactive(self):
        """交互式问答循环（CLI 模式）

        【登录失败重试设计】
        原实现登录失败直接退出，用户得重新跑 python main.py（要等 7 秒重建索引）。
        改成 while 循环重试，登录错了能立刻重输，不用重启程序。
        """
        while not self.login():
            print("请重新输入有效用户ID\n")
        print("\n" + "=" * 60)
        print("🏢 企业知识库 RAG 系统（输入 '退出' 结束，'切换' 换身份）")
        print("=" * 60)
        while True:
            try:
                q = input("\n您的问题: ").strip()
                # 多种退出关键词（中英文都支持），空输入也退出
                if q.lower() in ("退出", "quit", "exit", ""):
                    break
                # 切换身份：测试 RBAC 时频繁用到（HR ↔ 财务 ↔ admin）
                if q in ("切换", "switch"):
                    self.login()
                    continue
                self.ask(q)
            except KeyboardInterrupt:    # Ctrl+C 优雅退出
                break
            except Exception as e:
                # 单次问答出错不退出整个程序（下一次问答还能继续）
                logger.error(f"处理出错: {e}")
        print("\n再见！")

    def demo_rbac(self):
        """权限隔离自动演示（4 个场景，无需手动登录）

        【这是验证 RBAC 是否生效的集成测试】
        通过 4 个对照场景证明权限隔离真的起作用了：
          - HR 搜财务问题 → 应搜不到（场景1 负面验证）
          - HR 搜 HR 问题 → 应能答（场景2 正面验证）
          - 财务搜财务 → 应能答（场景3 正面验证）
          - admin 搜任意 → 全可见（场景4 超级权限验证）
        """
        print("\n" + "=" * 60)
        print("🔐 权限隔离(RBAC)自动演示")
        print("=" * 60)

        # 场景1：HR 员工搜财务问题（应搜不到——因为财务文档不在 HR 部门子索引里）
        print("\n【场景1】张三(HR) 搜索 '住宿费报销上限' — 应只看HR文档:")
        self.current_user = "zhangsan"
        self.ask("住宿费报销上限多少", stream=False, user_departments=["HR"])

        # 场景2：HR 员工搜 HR 问题（应能答——HR 文档在权限范围内）
        print("\n" + "-" * 60)
        print("\n【场景2】张三(HR) 搜索 '年假几天' — 应能答:")
        self.ask("年假几天", stream=False, user_departments=["HR"])

        # 场景3：财务员工搜财务问题（应能答——同一问题换用户，结果不同）
        print("\n" + "-" * 60)
        print("\n【场景3】李四(财务) 搜索 '住宿费报销上限' — 应能答:")
        self.current_user = "lisi"
        self.ask("住宿费报销上限多少", stream=False, user_departments=["财务"])

        # 场景4：管理员全权（["*"] 通配符 → permission_aware_search 走全库 hybrid_search）
        print("\n" + "-" * 60)
        print("\n【场景4】管理员 搜索任意 — 全可见:")
        self.current_user = "admin"
        self.ask("年假几天", stream=False, user_departments=["*"])


def _extract_tokens(resp) -> dict:
    """从 LLM response 提取 token 用量（兼容各 provider / LangChain 版本）

    【坑37：为什么这么复杂？】
    不同 LLM provider 和不同 LangChain 版本，token 用量的存放位置完全不同：
      - LangChain v0.2.8+：resp.usage_metadata（dict）
      - OpenAI 原始：resp.response_metadata.token_usage
      - Anthropic：resp.response_metadata.usage
      - 某些版本：usage_metadata 是 pydantic model 而非 dict
      - 最坏情况：什么都没有，只能按字符估算
    所以要按优先级逐一尝试（① ② ③ ④ ⑤），命中任一就返回。
    """
    try:
        # ① LangChain 标准：AIMessage.usage_metadata（v0.2.8+ 推荐路径）
        um = getattr(resp, "usage_metadata", None)
        if um and isinstance(um, dict):
            return {
                "prompt": um.get("input_tokens", 0),
                "completion": um.get("output_tokens", 0),
                "total": um.get("total_tokens", 0),
            }
        # ② OpenAI 原始格式：response_metadata.token_usage（老版本/直接调 OpenAI）
        rm = getattr(resp, "response_metadata", None) or {}
        tu = rm.get("token_usage", {})
        if tu:
            return {
                "prompt": tu.get("prompt_tokens", 0),
                "completion": tu.get("completion_tokens", 0),
                "total": tu.get("total_tokens", 0),
            }
        # ③ Anthropic 格式：response_metadata.usage（Claude 系列）
        usage = rm.get("usage", {})
        if usage:
            return {
                "prompt": usage.get("input_tokens", 0),
                "completion": usage.get("output_tokens", 0),
                "total": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            }
        # ④ usage_metadata 是 pydantic model 而非 dict（部分 LangChain 版本）
        if um is not None:
            return {
                "prompt": getattr(um, "input_tokens", 0) or 0,
                "completion": getattr(um, "output_tokens", 0) or 0,
                "total": getattr(um, "total_tokens", 0) or 0,
            }
        # ⑤ 最后兜底：按回答字符数粗估（中文约 2 字符 ≈ 1 token）
        content = getattr(resp, "content", "") or ""
        est = max(1, len(content) // 2)
        return {"prompt": est, "completion": est, "total": est * 2}
    except Exception:
        # 提取 token 失败不应影响主流程（token 统计是"锦上添花"）
        return {"prompt": 0, "completion": 0, "total": 0}


# ===== 程序入口 =====
if __name__ == "__main__":
    # sys.argv[1] 是命令行第一个参数（[0] 是脚本名本身）
    # 无参数默认 "chat"（交互模式），传 "rbac" 跑权限演示
    mode = sys.argv[1] if len(sys.argv) > 1 else "chat"
    # 三步启动：构造 → 初始化（加载模型） → 构建知识库（建索引）
    sys_obj = EnterpriseRAGSystem()
    sys_obj.initialize()
    sys_obj.build_knowledge_base()
    # 根据模式分发：rbac 演示 或 交互问答
    if mode == "rbac":
        sys_obj.demo_rbac()
    else:
        sys_obj.run_interactive()
