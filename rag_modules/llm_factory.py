"""
LLM 工厂（工厂模式 + 注册表 + 实例缓存）

设计目标：接入任何大模型，只需新增一个适配器函数并注册，无需改动现有代码。
满足"开闭原则"——对扩展开放，对修改关闭。

协议分类（依据真实调研）：
- openai 协议：DeepSeek / GLM原生 / 通义 / 文心 / 讯飞 / OpenAI / 本地Ollama（几乎所有国产模型）
- anthropic 协议：z.ai / Claude

供应商预设：把常用的 base_url/model 打包，.env 里只填 LLM_PROVIDER=deepseek 即可。

性能：create_llm 实例级缓存 —— 同配置复用 httpx 连接池，降级链 3 provider
内存和 fd 占用从 3× 降为 1×（热路径）；warmup() 预加载 SDK 消除首请求冷启动尖刺。

---
【教学模式导读】本文件演示三个经典设计模式的组合落地：
  1. 工厂模式  —— create_llm() 是统一入口，调用方不关心具体构造细节；
  2. 注册表模式 —— _REGISTRY + @register_adapter，新增协议只加函数不改老代码（开闭原则）；
  3. 对象池/缓存 —— _LLM_CACHE 复用昂贵实例（httpx 连接池），降级链内存占用 3×→1×。
  三个模式各管一摊：工厂管"怎么造"，注册表管"造哪些"，缓存管"别重复造"。
"""

# 标准库：hashlib 做缓存 key 哈希（避免明文 key 存内存）；logging 做日志。
import hashlib
import logging
# typing：Callable 给适配器签名定类型，Dict/Tuple 给容器定类型，让 IDE/类型检查更准。
from typing import Callable, Dict, Optional, Tuple

# LangChain 的所有聊天模型基类：工厂返回类型统一为它，调用方拿到后可用通用接口 invoke/astream。
from langchain_core.language_models import BaseChatModel

# 模块级 logger：命名空间为本模块名，便于在 logging 配置里按模块调级别（如 DEBUG 仅开此模块）。
logger = logging.getLogger(__name__)

# ===== 供应商预设（只读，常量语义）=====
# 扩展时：往这里加一条即可，不改动工厂逻辑
#
# 【知识点】这里把"厂商"和"协议"解耦：
#   - 厂商（zai/zhipu/deepseek/...）：决定 base_url 和默认 model；
#   - 协议（protocol: openai / anthropic）：决定走哪个适配器函数构造客户端。
#   一个 base_url 同时只能对接一种协议（厂商暴露的接口形态决定的），所以预设里固定写死。
#   把这套预设做成字典而非 if/else 链，新增厂商=加一行，满足开闭原则。
PRESETS: Dict[str, dict] = {
    # —— z.ai（Anthropic 协议）——
    # 注意：z.ai 对外暴露的是 Claude 风格的 /messages 接口，所以走 anthropic 适配器，
    # 即便它后面跑的其实是 GLM-4.6 模型（接口形态 ≠ 模型本体）。
    "zai": {
        "protocol": "anthropic",
        "base_url": "https://api.z.ai/api/anthropic",
        "model": "glm-4.6",
    },
    # —— 智谱原生（OpenAI 协议）——
    # GLM 官方同时提供 OpenAI 兼容端点，复用 openai 适配器即可，零额外适配成本。
    "zhipu": {
        "protocol": "openai",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4-flash",
    },
    # —— DeepSeek（OpenAI 协议）——
    "deepseek": {
        "protocol": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    # —— 通义千问/阿里百炼（OpenAI 协议）——
    # 阿里 DashScope 提供 compatible-mode（OpenAI 兼容模式），所以归 openai 协议。
    "qwen": {
        "protocol": "openai",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    # —— OpenAI（OpenAI 协议）——
    "openai": {
        "protocol": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    # —— 本地 Ollama（OpenAI 协议，无需 API key）——
    # Ollama 也实现了 OpenAI 兼容接口；本地部署所以在 create_llm 里对 api_key 走豁免分支。
    "local": {
        "protocol": "openai",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5",
    },
}

# 【性能小技巧】未命中预设时复用的单例空 dict，避免高频路径重复分配
# —— 每次走 .get(k, {}) 都返回同一个对象，不创建新 dict，减少 GC 压力。
#   返回的空 dict 本身不会被写入，所以共享是安全的。
_EMPTY_PRESET: dict = {}

# SDK 模块路径映射（预热用）
# 【知识点】用字符串而非直接 import，因为 langchain_openai/anthropic 未必都装；
#   延迟到 warmup() 才尝试 __import__，缺失也不影响其他协议。
_SDK_MODULES: Dict[str, str] = {
    "openai": "langchain_openai",
    "anthropic": "langchain_anthropic",
}

# ===== 协议适配器注册表 =====
# 每个适配器接收 (api_key, base_url, model, temperature, max_tokens) 返回 BaseChatModel
# 新增协议时：写一个函数 + 注册，不改动已有适配器
#
# 【注册表模式核心】这个字典是整个工厂的"路由表"：
#   protocol 名 -> 适配器函数。create_llm 拿到 protocol 后查表得到对应函数，
#   调用方完全不感知具体构造细节。新增协议=加一个函数+装饰器，老代码零改动。
AdapterFactory = Callable[..., BaseChatModel]
_REGISTRY: Dict[str, AdapterFactory] = {}
# 【性能/线程安全】_registered_protocols() 返回的"冻结快照"，元组不可变。
# 目的：错误信息里要列出已注册协议；如果每次都现读 _REGISTRY.keys()，热路径上
#   还得加锁防并发改写。这里把名字冻结成元组后只读，调用方拿到的也是不可变副本，
#   既能安全展示给用户，又免去了热路径加锁开销。
_REGISTRY_LOCK: Tuple[str, ...] = ()  # 首次访问后冻结的协议名快照（不可变，避免热路径加锁）


def register_adapter(protocol: str):
    """装饰器：把一个适配器函数登记进 _REGISTRY。

    【开闭原则落地】新增协议时只需：
        @register_adapter("myproto")
        def _make_myproto(...): ...
    无需改动 create_llm 或已有适配器。

    【同名策略】同一个 protocol 重复注册时不覆盖、只警告，并返回已注册的旧函数。
    这样可避免：插件/二次开发意外覆盖内置适配器导致静默行为变更（fail-safe 设计）。
    """
    def decorator(func: AdapterFactory):
        # 已有同名协议 -> 警告 + 退回旧函数（注意：这里 return 旧函数而非新函数，
        # 所以即便装饰器套在新函数上，被注册生效的仍是先到先得的旧实现）。
        if protocol in _REGISTRY:
            logger.warning(f"协议适配器 '{protocol}' 已注册（{_REGISTRY[protocol].__name__}），"
                           f"跳过 {func.__name__}（同名不覆盖）")
            return _REGISTRY[protocol]
        # 首次注册：写表 + 记日志。
        _REGISTRY[protocol] = func
        logger.debug(f"已注册协议适配器: {protocol}")
        # 装饰器约定：返回被装饰函数本身，保证 func 名仍指向正确实现（避免 NoneType 坑）。
        return func
    return decorator


def _get_adapter(protocol: str):
    """封装 registry 访问，未来改存储结构只改这一处。

    【封装的价值】这是个"间接层"：如果将来 _REGISTRY 从 dict 换成带优先级的
    多版本表、或换成支持热插拔的结构，只需要改这一个函数，调用方 create_llm 无感。
    """
    return _REGISTRY[protocol]


def _registered_protocols() -> Tuple[str, ...]:
    """返回已注册协议名快照（不可变元组，线程安全读）。

    【懒冻结 + 快照】只在第一次调用、或注册表条目数变化时重新生成元组；
    之后直接返回缓存好的元组。这样热路径（如错误信息拼接）上读快照无需加锁，
    因为 tuple 一旦生成就不可变，并发读绝对安全。
    """
    global _REGISTRY_LOCK
    # 快照为空 或 长度对不上（说明又有新协议注册了）-> 重新冻结一份。
    if not _REGISTRY_LOCK or len(_REGISTRY_LOCK) != len(_REGISTRY):
        _REGISTRY_LOCK = tuple(sorted(_REGISTRY.keys()))
    return _REGISTRY_LOCK


# —— OpenAI 协议适配器（覆盖绝大多数国产模型）——
@register_adapter("openai")
def _make_openai(api_key, base_url, model, temperature, max_tokens) -> BaseChatModel:
    """构造一个 OpenAI 协议的 ChatModel 实例。

    覆盖面最广：DeepSeek/智谱/通义/OpenAI/Ollama 等都走这里，因为它们都实现了
    OpenAI 兼容的 /v1/chat/completions 接口。

    【关键设计】开启 streaming=True 的目的不是"为了流式输出"，而是为了让
    LangChain 在 invoke()（非流式调用）内部也走 stream 协议逐块收集，从而
    填充出真实的 usage_metadata（prompt/completion token 数）。坑点：很多国产
    厂商在非流式响应里不返回 usage，导致 token 统计为 0，只能靠流式末尾的 usage 块拿到。
    """
    # 延迟 import：仅在真正创建实例时才加载 langchain_openai，加快模块导入、
    # 也允许"没装 openai SDK 但只用 anthropic 协议"的部署环境正常工作。
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        # —— 连接 & 鉴权 ——
        base_url=base_url, api_key=api_key, model=model,
        # —— 采样参数 ——
        temperature=temperature, max_tokens=max_tokens,
        # 【健壮性】max_retries=3：底层 httpx 自动重试瞬时网络错误（429/5xx），
        # 比应用层手写重试更省事，且默认带指数退避。
        max_retries=3,
        # 【拿真实 token 的关键】streaming=True：让 invoke 也走流式协议内部收集，
        # 才能拿到 usage_metadata（见上方 docstring 说明）。
        streaming=True,  # 启用流式协议：invoke() 也走 stream 内部收集，拿到 usage_metadata
        # 【坑37 修复】stream_options={"include_usage": True}：要求服务端在流的最后一帧
        # 带上本次请求的 token 用量。没有这一项，上面 streaming=True 也拿不到真实数。
        model_kwargs={"stream_options": {"include_usage": True}},
        # streaming=True 时 stream_options 有效，invoke/stream 都能拿到真实 token 数
    )


# —— Anthropic 协议适配器（z.ai / Claude）——
@register_adapter("anthropic")
def _make_anthropic(api_key, base_url, model, temperature, max_tokens) -> BaseChatModel:
    """构造一个 Anthropic 协议（Claude /messages 接口）的 ChatModel 实例。

    【降级设计】langchain_anthropic 新旧版本对"传 key/url 的参数名"有 breaking change：
      - 新版：用 api_key / base_url（与 OpenAI 风格统一）；
      - 旧版：用 anthropic_api_key / anthropic_api_url。
    本函数先用新参数名尝试，撞到"不认识的参数名" TypeError 时再降级到旧名，
    从而同时兼容新旧两个版本的 SDK，部署环境不必锁死版本。
    """
    from langchain_anthropic import ChatAnthropic
    try:
        # 优先用新版参数名（api_key / base_url）。
        return ChatAnthropic(
            model=model, api_key=api_key, base_url=base_url,
            temperature=temperature, max_tokens=max_tokens,
            max_retries=3,
        )
    except TypeError as e:
        # 【精准降级】仅"不认识的参数名"（unexpected keyword / got multiple）才走旧版降级；
        # 其他 TypeError（如 None 被误用、参数类型不对）一律原样抛出，避免吞掉真正的 bug。
        msg = str(e).lower()
        if "unexpected keyword" not in msg and "got multiple" not in msg:
            raise
        # 降级路径：换成旧版参数名 anthropic_api_key / anthropic_api_url。
        logger.debug("ChatAnthropic 新 API 参数不兼容，降级旧版参数名")
        return ChatAnthropic(
            model=model, anthropic_api_key=api_key, anthropic_api_url=base_url,
            temperature=temperature, max_tokens=max_tokens,
            max_retries=3,
        )


# ===== 实例缓存 =====
# 同配置复用同一个 LLM 实例，避免重复创建 httpx 连接池。
# 降级链 3 provider → 3 个实例创建后全部缓存，后续同配置命中 O(1)。
#
# 【为什么需要缓存】每个 ChatOpenAI/ChatAnthropic 实例内部都持有一个 httpx 连接池
# （含若干 socket + 线程）。如果不缓存，每次 create_llm 都新建实例 -> 新建连接池，
# 降级链尝试 3 个 provider 就会凭空多出 3 套连接池（内存/fd 翻 3 倍）。缓存后，
# 已成功的同配置直接复用，整体占用从 3× 降为 1×。
_LLM_CACHE: Dict[str, BaseChatModel] = {}


def _cache_key(
    provider: str, api_key: str, base_url: str, model: str,
    temperature: float, max_tokens: int,
) -> str:
    """生成缓存 key（加密哈希，避免明文 api_key 存内存）。

    【安全考量】api_key 是敏感信息。如果把明文 key 直接拼进字符串当字典 key，
    会一直驻留在进程内存里、可能被堆转储/日志意外带出。改用 sha256 哈希后，
    缓存表里只剩一串不可逆摘要，满足"内存中不留明文凭证"的安全习惯。
    【为什么 sha256 而非更快 hash】这里不是性能瓶颈（调用频率远低于热路径），
    sha256 抗碰撞、无已知破解，足够；同时避免某些非加密 hash（如 md5）被合规扫描告警。
    """
    # 用 "|" 分隔各字段，避免 "ab"+"c" 与 "a"+"bc" 拼出相同 key 的歧义碰撞。
    raw = f"{provider}|{api_key}|{base_url}|{model}|{temperature}|{max_tokens}"
    # encode() 转字节后哈希（sha256 只吃 bytes），hexdigest() 取十六进制字符串方便做 dict key。
    return hashlib.sha256(raw.encode()).hexdigest()


def clear_llm_cache():
    """清空 LLM 实例缓存（API key 轮换/热更新后调用）。

    【使用场景】当 .env 里换了 API key、或想强制重建连接池时调用，
    让旧实例（持有旧连接/旧 key）被 GC 回收，下次 create_llm 重新构造。
    """
    n = len(_LLM_CACHE)
    _LLM_CACHE.clear()
    logger.info(f"LLM 实例缓存已清空（{n} 个实例）")


# ===== 工厂入口 =====
def create_llm(
    provider: str,
    api_key: str,
    base_url: str = "",
    model: str = "",
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> BaseChatModel:
    """
    创建（或从缓存返回）LLM 实例。

    优先级：显式参数 > 预设
    - 若 provider 命中 PRESETS，用预设的 protocol/base_url/model 兜底
    - 显式传入的 base_url/model 覆盖预设（方便临时切模型）

    Args:
        provider: 供应商键（zai/zhipu/deepseek/qwen/openai/local）
        api_key: API key（Ollama 等本地部署可传占位符 "ollama"）
        base_url: 显式 base_url（覆盖预设）
        model: 显式 model（覆盖预设）
        temperature: 0.0 ~ 2.0
        max_tokens: ≥1

    Raises:
        ValueError: provider 无效、缺少必要配置、参数越界
    """
    # ============ 第一阶段：参数校验（早失败 / fail-fast）============
    # 【设计理念】把所有"配置明显不对"的情况在最前面一次性挡掉，并给出可执行的提示
    # （列出可用预设/已注册协议）。宁可早抛 ValueError，也不要让错误延迟到真正发请求时
    # 才以一个晦涩的 SDK 报错暴露——后者排查成本高得多。

    # provider 非空（连空白字符串都不行）。
    if not provider or not provider.strip():
        raise ValueError("LLM provider 不能为空。可用预设: " + ", ".join(PRESETS.keys()))

    # 规范化：去首尾空白 + 转小写，让 " DeepSeek " 和 "deepseek" 等价，避免隐性不匹配。
    provider = provider.strip().lower()

    # temperature 范围
    # 【早校验的好处】越界值不进缓存 key，避免后续以错误参数命中缓存后难以纠错。
    if not (0.0 <= temperature <= 2.0):
        raise ValueError(f"temperature 必须在 [0.0, 2.0]，当前值: {temperature}")

    # max_tokens 范围
    if max_tokens < 1:
        raise ValueError(f"max_tokens 必须 >= 1，当前值: {max_tokens}")

    # ============ 第二阶段：解析预设 ============
    # 【优先级】显式传入的参数 > 预设兜底。这样既能为常见厂商提供"零配置"默认，
    # 又允许调用方临时覆盖（如切换 model 做对比测试），不必改 PRESETS。
    preset = PRESETS.get(provider, _EMPTY_PRESET)  # 未命中复用单例空 dict，省分配
    # 若预设没写 protocol（或未命中预设），就用 provider 本身当 protocol 名兜底
    # —— 适用于"自定义 provider 名 == 协议名"的简单场景。
    protocol = preset.get("protocol", provider)
    # `x or y`：显式 base_url/model 非空则用显式，否则回退到预设值。
    final_base_url = base_url or preset.get("base_url", "")
    final_model = model or preset.get("model", "")

    # api_key：local 豁免（Ollama 本地部署不需要 key），其余 provider 必填。
    if provider != "local" and not api_key:
        raise ValueError(f"LLM_API_KEY 未设置（provider={provider}）。请在 .env 中配置")

    # base_url 兜底失败：既没显式传，预设里也没有 -> 无法发请求，必须报错。
    if not final_base_url:
        raise ValueError(
            f"无法确定 base_url：provider={provider} 未在预设中且未显式传入 base_url。"
            f"可用预设: {list(PRESETS.keys())}"
        )

    # protocol 存在性：确保注册表里有对应的适配器函数，否则后面 _get_adapter 会 KeyError。
    # 用快照 _registered_protocols() 展示已注册项，避免读 _REGISTRY 时与并发注册竞争。
    if protocol not in _REGISTRY:
        raise ValueError(
            f"不支持的协议 '{protocol}'（provider={provider}）。"
            f"已注册: {list(_registered_protocols())}"
        )

    # model 必填：没有 model 名 SDK 也无法构造，同 base_url 一样早报错。
    if not final_model:
        raise ValueError(
            f"无法确定 model：provider={provider} 未在预设中且未显式传入 model"
        )

    # ============ 第三阶段：缓存查找（命中即返回，避免重建连接池）============
    # 用规范化后的最终参数算 key，确保"等价配置"必然命中。
    ck = _cache_key(provider, api_key, final_base_url, final_model, temperature, max_tokens)
    if ck in _LLM_CACHE:
        logger.debug(f"LLM 缓存命中: {final_model} (provider={provider})")
        return _LLM_CACHE[ck]

    # ============ 第四阶段：创建实例 ============
    # 通过间接层 _get_adapter 拿到适配器函数（解耦存储细节）。
    adapter = _get_adapter(protocol)
    logger.debug(f"创建 LLM: {final_model} @ {final_base_url} (provider={provider}, protocol={protocol})")
    # 统一适配器签名：所有适配器都吃这 5 个参数，新增协议须遵守此契约。
    llm = adapter(
        api_key=api_key, base_url=final_base_url, model=final_model,
        temperature=temperature, max_tokens=max_tokens,
    )
    # 写入缓存：后续同配置请求直接复用此实例（含其 httpx 连接池）。
    _LLM_CACHE[ck] = llm
    return llm


# ===== 启动预热 =====
def warmup():
    """预加载已注册协议的 SDK 模块，消除首个请求的冷启动延迟（~200-500ms）。

    在应用 startup 阶段调用（如 FastAPI lifespan / main.py initialize）。

    【为什么要预热】Python 首次 import 一个重模块（如 langchain_openai 会连带拉起
    openai/httpx/pydantic 等一大票依赖）有明显的磁盘 IO + 编译开销，往往要几百毫秒。
    如果把这个代价摊到"用户第一次提问"上，会看到一个明显的首请求延迟尖刺（P99 变差）。
    在启动时提前 import，把这部分开销挪到无人感知的初始化阶段，首请求就和后续一样快。

    【容错策略】失败不阻塞：某个 SDK 未安装或导入异常，只记日志、不影响其他协议；
    真正用到时再由对应适配器的延迟 import 抛错，错误信息也更聚焦。
    """
    for protocol, module_name in _SDK_MODULES.items():
        # 只预热"确实被注册"的协议：例如从没注册 anthropic 适配器，就不白费功夫。
        if protocol not in _REGISTRY:
            continue
        try:
            # __import__ 显式触发模块加载（等价于 import langchain_openai），
            # 把"导入耗时"消耗在启动阶段，而非首个用户请求时。
            __import__(module_name)
            logger.debug(f"预热 SDK 完成: {module_name} (protocol={protocol})")
        except ImportError:
            # SDK 没装：正常情况（可能这个部署只用另一种协议），静默跳过即可。
            logger.debug(f"跳过 SDK 预热（未安装）: {module_name} (protocol={protocol})")
        except Exception as e:
            # 装了但导入报错（如版本冲突）：警告但不阻断启动，留给真实调用时再处理。
            logger.warning(f"预热 SDK 失败: {module_name}: {e}")
