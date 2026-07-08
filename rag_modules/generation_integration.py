"""
生成集成模块（企业级改造）

改造点（对照《真实RAG系统全貌》④生成子系统）：
- 模型：Moonshot 写死 → OpenAI 兼容协议（DeepSeek/OpenAI/本地可配）
- Prompt：烹饪助手 → 企业知识助手 + 多重防幻觉约束
- 生成：一次 invoke → 约束→生成→溯源→拒答 闭环

本模块在整条 RAG 链路中处于"最后一棒"：检索系统已经把候选文档喂进来，
本模块负责 ①调 LLM 生成答案 ②用 prompt 约束防幻觉 ③给每条事实打引用编号
④必要时改写查询 / 路由意图。可以理解成"质检员 + 播音员"合体。
"""

import logging
from typing import List

# —— 以下三类是 LangChain 的核心积木，理解它们是读懂 LCEL 链（| 管道）的前提 ——
# ChatPromptTemplate：聊天型模板，支持 {变量} 占位 + system/human 等角色
# PromptTemplate   ：纯文本模板（不分角色），改写/路由这类"单输入→纯文本输出"场景用它更轻
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# Document：LangChain 里"一段文本 + metadata"的标准容器，检索结果的统一格式
from langchain_core.documents import Document
# RunnablePassthrough：LCEL 链里的"透传"组件，把输入原样传给下游变量
from langchain_core.runnables import RunnablePassthrough
# StrOutputParser：把 AIMessage 的 content 抽成纯字符串，丢弃其余字段（坑：会丢 usage_metadata）
from langchain_core.output_parsers import StrOutputParser

# 模块级 logger：所有日志走统一 logging 体系，方便接入 Langfuse/云日志
logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """
    生成集成模块 - 负责 LLM 集成和企业知识问答生成。

    一个实例持有一个（或一组）已连好的 LLM 对象，对外暴露：
        - 生成类：generate_answer / _stream / _stream_with_usage（同步+异步）
        - 检索辅助类：query_rewrite（改写）/ query_router（路由）
        - 上下文拼装：_build_context
    构造时即调用 setup_llm()，构造完即可用，无需额外初始化。
    """

    def __init__(self, llm_provider: str, llm_base_url: str, llm_api_key: str,
                 model_name: str = "", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成模块（通过 LLM 工厂，多协议自适应）

        Args:
            llm_provider: 供应商键（zai/zhipu/deepseek/qwen/openai/local）或协议名
            llm_base_url: base_url（留空则用预设）
            llm_api_key: API key
            model_name: 模型名（留空则用预设）
            temperature: 生成温度（企业问答建议 0.1~0.3）
            max_tokens: 最大 token
        """
        # 把连接参数都存成实例属性，供 setup_llm() 和日志使用
        self.provider = llm_provider
        self.base_url = llm_base_url
        self.api_key = llm_api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None  # 占位，下面 setup_llm() 会真正赋值
        self.setup_llm()  # 构造即就绪：连上模型，避免"用的时候才发现连不上"

    def setup_llm(self):
        """
        通过工厂创建 LLM（接入任意供应商，零修改本模块）。

        【降级链自动装配】这是本方法的核心逻辑：
        - 检测配置项 LLM_FALLBACK_PROVIDERS（逗号分隔的备用供应商列表）。
        - 有配置  → 走 create_llm_with_fallback，主 LLM 调用失败/超时时自动按顺序切到备用，
                    保证生产可用性（主供应商宕机也能回答）。
        - 无配置  → 走 create_llm 单 LLM，零额外开销（不走包装层，省一次 fallback 判断）。
        整套判断只在初始化时执行一次，运行期 hot path 上没有任何分支判断。
        """
        # 延迟导入：避免模块加载时就强依赖 config（让单测/复用更灵活）
        from config import DEFAULT_CONFIG
        fb_providers = DEFAULT_CONFIG.llm_fallback_providers  # 逗号分隔字符串，空则视为未配置
        if fb_providers:
            # —— 分支A：有降级链配置 → 走 FallbackLLM ——
            from rag_modules.llm_fallback import create_llm_with_fallback
            # 主 LLM 参数包：用本实例 __init__ 收到的连接信息
            primary = {
                "provider": self.provider,
                "api_key": self.api_key,
                "base_url": self.base_url,
                "model": self.model_name,
            }
            # 把四组"逗号分隔字符串"拆成列表，保持各供应商按位置对齐
            providers = [p.strip() for p in fb_providers.split(",") if p.strip()]
            keys = [k.strip() for k in DEFAULT_CONFIG.llm_fallback_api_keys.split(",")]
            models = [m.strip() for m in DEFAULT_CONFIG.llm_fallback_models.split(",")]
            base_urls = [u.strip() for u in DEFAULT_CONFIG.llm_fallback_base_urls.split(",")]
            fallbacks = []
            for i, prov in enumerate(providers):
                # 防御性下标：备用项的 key/url/model 没配齐时，缺省值兜底（key 复用主 key、url/model 留空）
                fallbacks.append({
                    "provider": prov,
                    "api_key": keys[i] if i < len(keys) else self.api_key,
                    "base_url": base_urls[i] if i < len(base_urls) else "",
                    "model": models[i] if i < len(models) else "",
                })
            self.llm = create_llm_with_fallback(
                primary, fallbacks, self.temperature, self.max_tokens)
            logger.info(f"LLM 初始化完成（降级链: {self.provider} → {' → '.join(providers)}）")
        else:
            # —— 分支B：无降级 → 单 LLM，直接用工厂创建（最常用路径）——
            from rag_modules.llm_factory import create_llm
            self.llm = create_llm(
                provider=self.provider,
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            logger.info("LLM 初始化完成（单 LLM，无降级链）")

        # 打印当前加载的模型详情（便于确认实际连的是哪个模型）
        # 不同工厂返回的对象模型名属性名不一，用 getattr 兜底取 model_name
        _model = getattr(self.llm, "model_name", "") or self.model_name
        _name = type(self.llm).__name__
        logger.info(f"🤖 当前模型: {_model} | 供应商: {self.provider} | "
                    f"类型: {_name} | temp={self.temperature} max_tok={self.max_tokens} "
                    f"base={self.base_url}")
        # Langfuse 可观测回调，默认 None；由 observability.py 在启动后注入，不影响生成逻辑
        self.langfuse_handler = None  # Langfuse callback，由外部设置（observability.py）

    def set_langfuse_handler(self, handler):
        """设置 Langfuse callback handler（用于可观测上报）：启动后由 observability.py 注入。"""
        self.langfuse_handler = handler

    def _cb(self):
        """
        返回 callbacks config（供 chain.invoke/ainvoke 使用）。
        有 handler → {"callbacks":[...]}；无 → {}。
        这样调用方永远传一个合法 config，不用每次判空。
        """
        return {"callbacks": [self.langfuse_handler]} if self.langfuse_handler else {}

    # ===== ④ 生成子系统：防幻觉约束 Prompt =====
    #
    # 这是整个模块的"灵魂"，直接对应故障模式7（幻觉/胡说八道）的防御。
    # 四条核心约束逐条解释"为什么这么写"：
    #
    # 约束1「只能依据【上下文】」：
    #   LLM 自带海量参数化知识，会忍不住"帮忙补全"。企业知识（如报销标准、年假天数）
    #   一旦用错代价极大，所以必须把模型当"只会照本宣科的复读机"，切断它的先验。
    #
    # 约束2「无依据就说未找到」：
    #   与约束1配套的"安全阀"。允许的输出只有两种——要么引用上下文，要么明确拒答，
    #   不给"半编半真"的灰色地带。这一条把"幻觉"问题降级为"回答不出来"，可控得多。
    #
    # 约束3「每事实标 [文档X]」：
    #   强制溯源。用户/审计能逐条核对答案来自哪份文档，出问题可定位。
    #   X 对应下方 _build_context 里【文档1】【文档2】... 的编号，闭环。
    #
    # 约束4「不答无关内容」：
    #   防止模型"借题发挥"——比如问年假它顺嘴讲旅游攻略。把回答收敛到问题本身。
    #
    ANSWER_PROMPT = ChatPromptTemplate.from_template("""你是一位严谨的企业知识助手。请严格依据下方【上下文】回答用户问题。

【核心约束】
1. 只能依据【上下文】作答，不得使用上下文之外的知识。
2. 若【上下文】中没有相关信息，必须回答"知识库中未找到相关内容"，严禁编造。
3. 回答中每个事实陈述后，用 [文档X] 标注来源（X 为上下文中的文档编号）。
4. 不要回答与问题无关的内容。

【上下文】
{context}

【用户问题】
{question}

【回答】""")

    # —— 为什么 _REWRITE_PROMPT / _ROUTER_PROMPT 设成类级常量？——
    # PromptTemplate.from_template 内部要做：解析 {变量} 占位、构建 partial/校验等动作，
    # 这些都是一次性的构建成本。如果放到方法里每次调用重建，等于每次请求都重做一遍。
    # 放到类级（类体里直接赋值），解释器加载类时构建一次，全实例共享，hot path 零开销。
    # 这是"把不变的东西提到构造期"的通用优化手法。
    #
    # 查询改写 Prompt（类级别，避免每次调用重建 PromptTemplate）
    _REWRITE_PROMPT = PromptTemplate(
        template="""你是一个企业知识库检索查询改写助手。请分析用户问题，若过于模糊或口语化，改写为更利于检索的明确表述；若已明确，原样返回。只输出最终查询。
改写原则：保持原意，补全可能的企业术语（如"请假"→"年假 事假 病假 请假流程 审批"）。
示例："报销怎么搞"→"费用报销流程 报销审批 报销标准"
原始查询: {query}
改写结果:""",
        input_variables=["query"],
    )

    # 意图路由 Prompt（类级别，避免每次调用重建）
    # 设计要点：明确告诉模型"只返回 retrieval/chitchat/clarify 之一"，逼它输出可枚举值，
    # 下游 query_router 再做白名单兜底（见下方），双保险。
    _ROUTER_PROMPT = ChatPromptTemplate.from_template("""判断用户问题类型，只返回 retrieval/chitchat/clarify 之一。
retrieval=需查知识库(如"年假几天""报销流程")
chitchat=闲聊(如"你好""谢谢")
clarify=太模糊需澄清(如"帮我看看""怎么办")
用户问题: {query}
类型:""")

    # 同理：分隔线也是常量，提前算好避免每次 _build_context 重建（高频调用）
    _SEP_LINE = "=" * 50       # 预计算分隔线，避免每次调用重建
    _SEP = "\n" + _SEP_LINE + "\n"  # 文档间分隔符

    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成企业知识问答回答（约束 + 溯源 + 拒答）。

        这是经典的 LCEL（LangChain Expression Language）链式写法，用 | 串起 4 个组件：
            组装输入 → 套 prompt 模板 → 调 LLM → 解析输出。

        Args:
            query: 用户查询
            context_docs: 检索到的上下文文档

        Returns:
            带引用溯源的回答
        """
        # 先把文档列表拼成带【文档X】编号的纯文本上下文（含截断保护）
        context = self._build_context(context_docs)

        # —— LCEL 链构造（注意：这里只是"搭管道"，还没真正调用 LLM）——
        chain = (
            # 第1步：把链入口重组成 prompt 需要的两个变量。
            #   "question" 透传用户原始 query（RunnablePassthrough 当"=输入"用）；
            #   "context"  是闭包，取上面已算好的 context（不管输入是什么都返回它）。
            {"question": RunnablePassthrough(), "context": lambda _: context}
            # 第2步：用 ANSWER_PROMPT 把 {question}/{context} 渲染成最终提示词
            | self.ANSWER_PROMPT
            # 第3步：调 LLM 生成（self.llm 已是 Runnable）
            | self.llm
            # 第4步：把 AIMessage.content 抽成纯字符串返回
            | StrOutputParser()
        )
        # invoke 才真正执行整条链；config=self._cb() 注入可观测回调
        return chain.invoke(query, config=self._cb())

    async def generate_answer_async(self, query: str, context_docs: List[Document]) -> str:
        """
        异步生成（生产环境用 ainvoke，不阻塞事件循环）。

        与同步版完全同构，只把 invoke→ainvoke：FastAPI/SSE 这类异步框架里必须用，
        否则一次 LLM 调用（动辄数秒）会卡住整个事件循环，导致并发塌掉。
        """
        context = self._build_context(context_docs)
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | self.ANSWER_PROMPT
            | self.llm
            | StrOutputParser()
        )
        return await chain.ainvoke(query, config=self._cb())

    async def query_rewrite_async(self, query: str) -> str:
        """异步查询改写。链更短：模板 → LLM → 取字符串，无需重组输入（_REWRITE_PROMPT 只要 {query}）。"""
        chain = self._REWRITE_PROMPT | self.llm | StrOutputParser()
        response = await chain.ainvoke(query, config=self._cb())
        return response.strip()

    async def query_router_async(self, query: str) -> str:
        """异步意图路由。注意 _ROUTER_PROMPT 变量名是 {query}，所以前面要重组 {"query": ...}。"""
        chain = ({"query": RunnablePassthrough()} | self._ROUTER_PROMPT | self.llm | StrOutputParser())
        result = await chain.ainvoke(query, config=self._cb())
        return result.strip().lower()

    def generate_answer_stream(self, query: str, context_docs: List[Document]):
        """
        流式生成（SSE 用，只返回文本）。

        生成器函数（yield）：用 chain.stream 逐块吐 token，前端可边生成边显示，
        用户体感更快。此版本只产出字符串，丢弃了 usage 信息（见下一方法的对比）。
        """
        context = self._build_context(context_docs)
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | self.ANSWER_PROMPT
            | self.llm
            | StrOutputParser()
        )
        # stream() 返回可迭代对象，逐 chunk yield 出去
        for chunk in chain.stream(query, config=self._cb()):
            yield chunk

    def generate_answer_stream_with_usage(self, query: str, context_docs: List[Document]):
        """
        流式生成 + 精确 token 统计（坑18/37）。

        【为什么绕过 StrOutputParser】
        StrOutputParser 只取 AIMessage.content 转成字符串，会把 usage_metadata 等
        附带字段全丢弃。而我们恰恰需要 token 用量来计费/监控，所以这里链尾不放 parser，
        让原始 AIMessageChunk 直接流出来，再手工从其上提取 content + usage_metadata。

        【stream_options include_usage 的作用】
        单靠去掉 parser 还不够——OpenAI 协议默认在流式响应里不发用量统计。
        需要在 LLM 初始化时带 stream_options={"include_usage": True}（见工厂层），
        服务端才会在"最后一个 chunk"里附 usage_metadata，含真实的 input/output/total tokens。
        没有 include_usage，这里 um 永远是 None。

        Yields:
            (text_chunk: str, usage: dict | None)
            usage 仅在最后一个 chunk 非 None，格式: {"prompt": N, "completion": N, "total": N}
        """
        context = self._build_context(context_docs)
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | self.ANSWER_PROMPT
            | self.llm
            # 不用 StrOutputParser，保留 AIMessageChunk 的 metadata（这是本方法的关键改动）
        )
        for chunk in chain.stream(query, config=self._cb()):
            # 文本：正常 chunk 有 .content；个别实现可能只返回裸字符串，用 str() 兜底
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            # 用量：只在最后一个 chunk 上才有（其余 chunk 为 None）
            um = getattr(chunk, "usage_metadata", None)
            usage = None
            if um:
                # usage_metadata 可能是 dict（OpenAI 兼容）也可能是对象（pydantic），
                # 用 isinstance 判形后分别取值，两种风格都能兼容（防供应商差异）
                usage = {
                    "prompt": um.get("input_tokens", 0) if isinstance(um, dict) else getattr(um, "input_tokens", 0),
                    "completion": um.get("output_tokens", 0) if isinstance(um, dict) else getattr(um, "output_tokens", 0),
                    "total": um.get("total_tokens", 0) if isinstance(um, dict) else getattr(um, "total_tokens", 0),
                }
            yield content, usage

    def query_rewrite(self, query: str) -> str:
        """
        查询改写（企业领域）：把口语化/模糊问题改写为更适合检索的查询。
        对应《真实RAG全貌》③检索子系统。

        注意：本方法只是"基础改写"。真正的"短路"机制（明确问题直接跳过改写省 6~20s，
        见坑20/24）由调用方（如 orchestrator）在更上层实现——先判断 query 是否已足够明确，
        明确则根本不调本方法。本方法假设"确实需要改写"。
        """
        chain = self._REWRITE_PROMPT | self.llm | StrOutputParser()
        response = chain.invoke(query, config=self._cb()).strip()
        # 改写前后不同才打日志，便于观察改写效果（相同则说明本就明确，省日志噪声）
        if response != query:
            logger.info(f"查询改写: '{query}' → '{response}'")
        return response

    def query_router(self, query: str) -> str:
        """
        意图路由：判断问题类型，分流处理。
        对应《真实RAG全貌》③意图路由（省成本提准确）。
        返回: 'retrieval'(需检索) | 'chitchat'(闲聊) | 'clarify'(需澄清)

        同样，真正的"短路"（明确 retrieval 问题直接跳过路由）在上层实现以省掉这次 LLM 调用。
        最后一行的白名单兜底很重要：模型偶尔会输出多余文字（如带标点、解释），
        只要有任何一个不在三选一里，就默认走 retrieval——宁可在知识库白搜一次，也别误判成闲聊。
        """
        chain = (
            {"query": RunnablePassthrough()}
            | self._ROUTER_PROMPT
            | self.llm
            | StrOutputParser()
        )
        result = chain.invoke(query, config=self._cb()).strip().lower()
        # 白名单兜底：非三选一则视为 retrieval（最安全默认值）
        return result if result in ("retrieval", "chitchat", "clarify") else "retrieval"

    def _build_context(self, docs: List[Document], max_length: int = 3000) -> str:
        """
        构建上下文（带引用编号，用于溯源）。

        本方法决定"喂给 LLM 的上下文长什么样"，直接影响答案质量与成本。

        Args:
            docs: 检索到的文档
            max_length: 上下文最大长度（控制成本/Lost in the Middle）

        Returns:
            格式化的上下文，形如【文档1】【文档2】...

        【max_length=3000 截断防 Lost in the Middle（故障模式5）】
        研究表明：当上下文很长时，LLM 对"中间位置"的信息明显忽略（首尾记得牢，中间被遗忘）。
        3000 字符是一个务实上限——既给模型足够信息，又避免把无关/低相关文档硬塞进去
        反而冲淡关键内容、拉高 token 成本。超过即截断，宁可少给也不给坏。

        【为什么处处用 "".join 而非 +=】
        Python 字符串不可变，s += x 每次都会"新建一个更长的串 + 拷贝旧内容"，
        循环里反复 += 是 O(n²) 行为（n 越大越慢）。改用"先收集到 list，最后一次 join"
        是 O(n)，大上下文下差距明显。下面 meta_parts、context_parts 两处都用这手法。
        """
        if not docs:
            # 空检索的显式信号：返回提示文本，让 prompt 据此触发约束2（拒答）
            return "（知识库中无相关内容）"

        context_parts = []  # 收集每段文档文本，最后统一 join
        current_length = 0   # 累计已用长度，用于截断判断
        sep = self._SEP   # 本地别名，避免属性查找：循环里反复 self._SEP 会多一次属性解析
        # enumerate(docs, 1) 让编号从 1 开始（用户看到【文档1】而非【文档0】，更自然）
        for i, doc in enumerate(docs, 1):
            md = doc.metadata
            # 元信息：来源 + 页码 + 部门（溯源用），用列表 join 代替 += 减少中间字符串
            # 这里每段都先攒到 meta_parts 再 "".join，正是上面 docstring 讲的优化手法
            meta_parts = [f"【文档{i}】"]  # 编号必须与 ANSWER_PROMPT 的 [文档X] 对应
            src = md.get("source")
            if src:
                meta_parts.append(f" 来源:{src}")
            pg = md.get("page")
            if pg:
                meta_parts.append(f" 页:{pg}")
            dept = md.get("department")
            if dept:
                meta_parts.append(f" 部门:{dept}")
            meta = "".join(meta_parts)

            doc_text = f"{meta}\n{doc.page_content}\n"
            # 截断判断：加上这段会超 max_length 就停（保证不超长）
            if current_length + len(doc_text) > max_length:
                logger.warning(f"上下文超长({max_length})，截断于文档{i}")
                break
            context_parts.append(doc_text)
            current_length += len(doc_text)

        # 最后一次性 join：开头也加一条分隔线，与文档间分隔一致
        return sep + sep.join(context_parts)
