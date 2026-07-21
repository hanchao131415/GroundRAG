"""
LLM 降级链（FallbackLLM）

解决的真实问题：
  单 provider 挂了（429 限流 / 余额不足 / 网络中断），整个 RAG 系统就不可用。
  生产系统必须有降级：主模型挂 → 自动切备用模型，用户无感知。

设计：组合模式（不是继承）
  FallbackLLM 持有一个 LLM 列表 [primary, fallback1, fallback2]，
  invoke/stream 时依次尝试，记录哪个成功了，下次优先用（熔断学习）。

为什么不用 LangChain 的 RunnableLambda 包：
  RunnableLambda 拿不到 LLM response 的 usage_metadata（token 统计会丢）。
  所以手动实现 invoke/stream，直接转发到真实 LLM。

  【展开讲清楚】这是本模块最关键的设计决策，对应"故障模式10 / 坑31"：
    1. LangChain 提供了 RunnableLambda(fn)，可以把任意函数包装成 Runnable，
       调用方就能像用普通 LLM 一样 .invoke() / .stream()，看起来很省事。
    2. 但 RunnableLambda 只透传 fn 的"返回值本身"，它不会保留 LLM response
       对象上的 usage_metadata（即 input_tokens / output_tokens / total_tokens）。
       对 RAG 系统来说，token 统计是成本核算、配额控制、链路监控的命脉，
       一旦丢了，整套可观测性就瞎了。
    3. 我们的解法：不继承、不包装，用"组合模式"——FallbackLLM 自己持有
       真实 LLM 列表，invoke/stream 时把 messages/config 原样转发给真实 LLM，
       再把真实 LLM 的 response 原样返回。这样 usage_metadata 一个字节都不丢。
    4. 代价：FallbackLLM 不是真正的 LangChain Runnable（没有 .bind()/.with_fallbacks()
       等 Runnable 协议方法），但这套 RAG 只用到 invoke/stream，鸭子类型足够。

面试可讲的点：
  "降级链不是简单 try-except，而是带健康状态的 failover：
   成功的 provider 会被记住，失败的 provider 会冷却 N 秒再重试（熔断）"
"""

import logging
import time
from typing import Generator, List, Optional, Tuple

# 模块级 logger：日志会带上模块路径（rag_modules.llm_fallback），方便在 ELK 里过滤
logger = logging.getLogger(__name__)


class FallbackLLM:
    """
    LLM 降级链：primary 挂了自动切 fallback

    用法：
        llm = FallbackLLM([primary_llm, backup_llm])
        # 用法跟普通 LLM 一样（duck-type 兼容 LangChain Runnable）
        resp = llm.invoke(messages, config={...})
        for chunk in llm.stream(query):
            ...

    核心状态（三件套，记住它们就懂了这个类）：
        _llms             : 真实 LLM 列表，按下标引用
        _cooldown         : 每个 LLM 的"下次可用时间戳"，0=可用，>now=冷却中
        _last_success_idx : 上次成功的下标，下次优先试它（避免每次都先撞墙）
    """

    # __slots__ 锁死实例属性，省内存 + 防手滑写错属性名（拼写错误会直接报错）
    # 四个属性都是内部状态，不对外暴露（带下划线前缀 = 约定的"私有"）
    __slots__ = ("_llms", "_cooldown", "_cooldown_seconds", "_last_success_idx")

    def __init__(self, llms: List, cooldown_seconds: int = 60):
        """
        Args:
            llms: LLM 列表，第一个是 primary，其余是 fallback（顺序即优先级）
            cooldown_seconds: 失败后的冷却时长，默认 60s。
                              太短→故障 provider 还没恢复就被反复打；
                              太长→provider 恢复了也迟迟不用，浪费备用。
        """
        if not llms:
            # 空列表直接拒绝：降级链至少要有一个能用的，否则没意义
            raise ValueError("FallbackLLM 需要至少一个 LLM")
        self._llms = llms

        # 每个 LLM 的"下次可用时间"（0 = 可用；>now = 冷却中）
        # 注意 key 用 id(llm) 而不是下标：万一有人在外面把同一个 LLM 实例放进两次，
        # id() 能让它们共享冷却状态（更符合"同一个对象=同一个健康状态"的语义）
        self._cooldown: dict = {id(llm): 0.0 for llm in llms}

        self._cooldown_seconds = cooldown_seconds

        # 上次成功的 LLM（下次优先用它，避免每次都试 primary 再 fail）
        # 初始值 0 = 默认从 primary 开始试（没人成功过时，按列表顺序就是最优）
        self._last_success_idx = 0

        logger.info(f"🔀 FallbackLLM 初始化: {len(llms)} 个 LLM（冷却 {cooldown_seconds}s）")

    @property
    def active_llm(self):
        """当前活跃的 LLM（上次成功的那个，或第一个未冷却的）

        用途：外部想知道"现在实际上在用哪个模型"时调用，比如日志/监控打点。
        注意它不触发任何降级逻辑，只是个只读视图。
        """
        return self._llms[self._last_success_idx]

    def _ordered_llms(self) -> Generator[Tuple[int, object], None, None]:
        """按优先级 yield LLM：上次成功的排第一，冷却中的排最后（零分配 generator）。

        这是降级链的"调度大脑"——它不返回列表，而是一个个 yield，
        调用方拿到一个试一个，试成功就 break，省得提前把全列表建好。

        三段优先级（对应下面 1/2/3）：
          ① 上次成功的、且未冷却    → 最可能成功，先试
          ② 其余的、且未冷却        → 备选健康节点
          ③ 冷却中的                → 万不得已（所有健康的都挂了才回来试它）

        为什么用 generator 而不是返回 list：
          绝大多数情况第一个就成功了，后面的根本不用算；
          generator 是惰性的，"用到哪算到哪"，省 CPU 也省内存。
        """
        now = time.time()       # 一次取当前时间，本函数内统一用它判定（避免时间漂移）
        llms = self._llms       # 局部别名，循环里少一次属性查找（微优化）
        idx = self._last_success_idx  # 上次成功的下标，作为首选

        # 1) 上次成功的 LLM（若未冷却）
        # _cooldown[id] <= now 表示"到期了/可用"（=0 时永远 <= now）
        if self._cooldown[id(llms[idx])] <= now:
            yield idx, llms[idx]

        # 2) 其余未冷却的
        # 跳过 idx（已经在上面 yield 过了），把其它健康节点按原顺序交出去
        for i, llm in enumerate(llms):
            if i != idx and self._cooldown[id(llm)] <= now:
                yield i, llm

        # 3) 冷却中的（万不得已才试）
        # 走到这里说明 ①② 都试完且都失败了——只能硬着头皮去碰运气看冷却的是否已恢复
        # （冷却只是"建议等一会儿"，不是硬禁止；真没别的选时还是得试）
        for i, llm in enumerate(llms):
            if self._cooldown[id(llm)] > now:
                yield i, llm

    def invoke(self, messages, config=None, **kwargs):
        """
        调用 LLM（带降级）

        依次尝试 LLM 列表，任一成功即返回；
        全部失败则抛最后一个异常（不让错误信息丢失）。

        关键点：返回的是真实 LLM 的 response 对象，所以 usage_metadata 完整保留
        （这正是我们不用 RunnableLambda 的原因）。
        """
        config = config or {}    # config 可能为 None，统一成空 dict，方便后面传参
        last_error = None        # 记住最后一次异常，全失败时抛出去（保留原始堆栈链）

        # _ordered_llms() 给出"试的顺序"，这里是降级的核心循环
        for idx, llm in self._ordered_llms():
            model_name = self._llm_name(llm)  # 取模型名，纯粹为了日志好读
            try:
                # 真正的调用：messages/config/kwargs 原样转发给真实 LLM
                resp = llm.invoke(messages, config=config, **kwargs)

                # 成功：记住这个 LLM，下次优先用
                # 只有"换了一个新 LLM 才成功"时才打 info，避免每次成功都刷屏
                if idx != self._last_success_idx:
                    logger.info(f"✅ 切换到 LLM[{idx}] {model_name}（之前降级过，现已恢复）")
                    self._last_success_idx = idx  # 更新"成功记忆"，下次 _ordered_llms 把它排第一
                return resp                       # 立即返回，不再尝试后面的（成功即止）
            except Exception as e:
                # 任何异常都视为这个 LLM 不可用：429/超时/鉴权失败/网络断……一律降级
                last_error = e
                # 触发熔断：这个 LLM 冷却 N 秒
                # 设置成"当前时间 + 冷却时长"，即它的"下次可用时间戳"被推到未来
                self._cooldown[id(llm)] = time.time() + self._cooldown_seconds
                logger.warning(f"⚠️ LLM[{idx}] {model_name} 调用失败: {e}，冷却 {self._cooldown_seconds}s，尝试下一个")
                # 没有 break——继续 for 循环试下一个 LLM

        # 走到这里说明循环跑完了都没成功 → 所有 LLM 都不可用
        # 用 `from last_error` 把原始异常链上，调用方 traceback 能看到真正的根因
        raise RuntimeError(
            f"所有 LLM 均不可用（共 {len(self._llms)} 个）。最后错误: {last_error}"
        ) from last_error

    def stream(self, query, config=None, **kwargs):
        """
        流式调用（带降级）

        流式的降级更微妙：流开始后挂掉，只能整段重来（不能中途换 provider）。
        所以这里用 generator 包裹：如果流中途异常，重新从下一个 LLM 开始。

        【为什么这是全文件最微妙的地方】
          普通请求是"一次性"的：失败就失败，换一个重试，没副作用。
          但流式是"边产生边消费"的——调用方可能已经把前几个 chunk yield 出去、
          渲染到前端了。这时候 provider 挂了，你不可能：
            ❌ 把已经吐出去的 chunk 收回来（已经显示给用户了）
            ❌ 从中间断点续传（HTTP/SSE 协议不支持，provider 不知道你到哪了）
          唯一能做的：整段从头重来。但"重来"又会导致重复输出——
          所以下面的 collected 列表其实是为"未来可能的去重/统计"留的钩子，
          当前实现选择"接受重发"（生产里通常配合前端去重或干脆只重试未开始流的情况）。
        """
        config = config or {}
        last_error = None

        for idx, llm in self._ordered_llms():
            model_name = self._llm_name(llm)
            try:
                # 先尝试拿 stream iterator（stream() 是惰性的，错误在迭代时才暴露）
                # 这一行通常不会抛——它只是"建立流"，真正的 IO 在下面的 for chunk 里
                stream_iter = llm.stream(query, config=config, **kwargs)
                collected = []          # 收集本次流的所有 chunk（用于失败后诊断/未来去重）
                for chunk in stream_iter:
                    collected.append(chunk)  # 先存一份，再吐给调用方
                    yield chunk              # 把 chunk 流式交给调用方（前端立刻能看到）
                # 成功完成整个流：记住这个 LLM
                # 走到这里说明整个流正常结束（没抛异常），才确认"这个 provider 健康"
                if idx != self._last_success_idx:
                    logger.info(f"✅ 切换到 LLM[{idx}] {model_name}")
                    self._last_success_idx = idx
                return  # 流正常结束
                # 注意：generator 里 return 等价于 StopIteration，表示"我结束了，别再要 chunk"
            except Exception as e:
                # 流中途挂了：把当前 provider 设冷却，然后 continue 去试下一个
                # 代价：已经 yield 出去的 chunk 不会被收回（见上面方法 docstring 的说明）
                last_error = e
                self._cooldown[id(llm)] = time.time() + self._cooldown_seconds
                logger.warning(f"⚠️ LLM[{idx}] {model_name} 流式失败: {e}，冷却 {self._cooldown_seconds}s，尝试下一个")
                continue  # 注意是 continue 不是 break——继续外层 for 试下一个 provider

        # 所有 provider 的流都失败了
        raise RuntimeError(
            f"所有 LLM 流式调用均失败（共 {len(self._llms)} 个）。最后错误: {last_error}"
        ) from last_error

    @staticmethod
    def _llm_name(llm) -> str:
        """提取 LLM 的模型名（用于日志）

        不同 provider 的 LLM 对象属性名不统一（DeepSeek 叫 model_name，
        OpenAI 叫 model，自定义的可能啥都没有），所以做三级兜底：
          model_name → model → 类名。总能拿到个能看的东西。
        """
        # getattr(obj, name, default)：属性不存在就返回 default，不会抛 AttributeError
        # `or` 短路：前一个为空字符串/None 时，顺延用后一个
        return getattr(llm, "model_name", "") or getattr(llm, "model", "") or str(type(llm).__name__)


def create_llm_with_fallback(
    primary_config: dict,
    fallback_configs: Optional[List[dict]] = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
):
    """
    构建带降级的 LLM（封装工厂调用）

    这是给上层（如 RAG 链组装处）用的"一键构造"入口：
    传主配置 + 备用配置列表，返回一个能直接 .invoke() 的对象。
    上层不用关心返回的是 FallbackLLM 还是单个 LLM——鸭子类型，用法一样。

    Args:
        primary_config: 主 LLM 配置 {provider, api_key, base_url, model}
        fallback_configs: 备用 LLM 配置列表（可空 = 不降级，退回单 LLM）
        temperature/max_tokens: 生成参数（主备共用，保证回答风格一致）

    Returns:
        FallbackLLM（有备用时）或 单个 BaseChatModel（无备用时，零开销）

    【关键设计：无备用时返回单个 LLM，而不是包一层只有一个元素的 FallbackLLM】
      这样在没有配置降级链的部署里，调用路径上没有任何额外的降级判断开销，
      做到"有备用才付降级成本"。这是"按需复杂度"的体现。

    环境变量配置降级链（.env）：
        LLM_PROVIDER=deepseek           # 主
        LLM_FALLBACK_PROVIDERS=qwen,zhipu  # 备用（逗号分隔）
        LLM_FALLBACK_API_KEYS=key1,key2
    """
    # 延迟导入：避免模块加载时就触发 llm_factory 的依赖（可能是重的 langchain/openai）
    # 放在函数内部也是为了打破潜在的循环导入
    from rag_modules.llm_factory import create_llm

    llms = []  # 收集所有成功创建的 LLM，最后交给 FallbackLLM

    # 主 LLM
    # 注意：主 LLM 创建失败会直接抛异常（不像备用那样 try/except 跳过）——
    # 因为连主都没有，降级链就失去基准了，不如早失败早暴露配置问题
    primary = create_llm(
        provider=primary_config["provider"],
        api_key=primary_config["api_key"],
        base_url=primary_config.get("base_url", ""),   # .get() 带默认值，缺了也不报错
        model=primary_config.get("model", ""),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    llms.append(primary)

    # 备用 LLM（逐个创建，单个失败不影响其余）
    # 这里是"容错创建"：第 i 个备用创建失败，只 warning 跳过，不影响主和其它备用。
    # 场景：配了 3 个备用，其中 1 个 key 写错了，另外 2 个还能正常工作。
    if fallback_configs:
        for i, cfg in enumerate(fallback_configs):
            try:
                fb = create_llm(
                    provider=cfg["provider"],
                    api_key=cfg["api_key"],
                    base_url=cfg.get("base_url", ""),
                    model=cfg.get("model", ""),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                llms.append(fb)
            except Exception as e:
                # 单个备用挂了不影响整体——这正是降级链"鲁棒性"的第一道关：构建期也要能降级
                logger.warning(f"备用 LLM[{i}] {cfg.get('provider', '?')} 创建失败，跳过: {e}")

    # 如果一个备用都没成功创建（或根本没配），就退回单 LLM
    # 关键：返回的是裸 LLM，不是 FallbackLLM——调用方零开销，没有降级循环的开销
    if len(llms) == 1:
        logger.info("无降级链配置（LLM_FALLBACK_* 未设），使用单 LLM")
        return llms[0]

    # 有至少一个备用，才包装成降级链
    return FallbackLLM(llms)
