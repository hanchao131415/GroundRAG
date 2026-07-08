"""
Langfuse 可观测性封装

作用：把 LLM/检索的调用自动上报到 Langfuse 云端，可视化看 trace。
接入方式：LangChain CallbackHandler（自动追踪 LangChain 每一步）。

降级设计：Langfuse key 没配/网络不通时返回 None，不影响主流程。
对应《真实RAG全貌》⑦可观测子系统。
"""

import os  # 标准库：读环境变量（基础设施层零依赖业务 config）
import logging  # 标准库：日志输出

# 模块级 logger：每个模块用自己的 __name__ 命名，日志里能看出是哪个模块打的
# 这样在统一日志配置里（logging_config.py）就能按名字分级/降噪
logger = logging.getLogger(__name__)


def get_langfuse_handler():
    """
    获取 Langfuse CallbackHandler（如果配置了 key）。

    本函数是"可观测性"与"主流程"之间的解耦点：
    - 配了 key → 返回可用 handler，主流程自动上报 trace 到云端；
    - 没配 key / 加载失败 → 返回 None，主流程照常跑（降级为只用本地 trace）。
    这种"返回 None 即可关闭"的设计，让可观测性变成"可插拔"特性，不污染业务代码。

    设计说明：直接读 os.getenv 而非从 config 模块导入——
    observability 是基础设施层，应零依赖业务 config；
    且 langfuse key 可能由平台注入（K8s secret / CI env），不经过 .env。
    这条原则叫"分层零依赖"：底层不依赖上层，避免循环 import 和部署耦合。

    Returns:
        CallbackHandler 实例（配了 key 且加载成功时） 或 None（降级时）
    """
    # ① 读 Langfuse 三件套：公钥、私钥、云端地址
    # 注意：这里用 os.getenv 直连环境变量，而不是 from config import XXX
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")   # 公钥（公开标识，可放前端）
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")   # 私钥（敏感，只放服务端/secret）
    # host 默认指向 Langfuse 官方 SaaS；自建私有化部署时改这里即可
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # ② 降级判断：key 缺任意一个 → 直接返回 None，主流程无需感知
    # 这就是"降级设计"的核心：失败不抛异常，而是返回 None，调用方用 None 当"无观测"
    if not public_key or not secret_key:
        logger.info("Langfuse 未配置 key，跳过（只用本地 trace）")
        return None  # ← 返回 None 而不是抛错，保证主流程不被可观测性拖累

    try:
        # langfuse 4.x: CallbackHandler 从环境变量读 key，不接受 key 参数
        # 需先设置环境变量（注意 4.x 用 LANGFUSE_HOST 不是 BASE_URL）
        # —— 即 SDK 自己去读 os.environ，所以我们把值"回写"进环境变量
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key   # 回写公钥到环境变量
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key   # 回写私钥到环境变量
        os.environ["LANGFUSE_HOST"] = host               # 回写云端地址（4.x 改名了）
        # 延迟导入（lazy import）：只有真正要用 langfuse 时才 import
        # 好处：① 没装 langfuse 的环境不会因 import 报错 ② 启动更快（少加载）
        from langfuse.langchain import CallbackHandler
        # 实例化 handler：此时它会从上面回写的环境变量里读 key
        handler = CallbackHandler()
        logger.info(f"✅ Langfuse 可观测已启用 @ {host}")
        return handler  # 返回可用的 handler，调用方把它塞进 LangChain 的 callbacks
    except Exception as e:
        # 兜底降级：任何异常（网络不通、版本不符、key 失效……）都不影响主流程
        # 只打一条 warning，依然返回 None —— 这就是"可观测性永远是可选的"
        logger.warning(f"Langfuse 加载失败，降级为本地 trace: {e}")
        return None
