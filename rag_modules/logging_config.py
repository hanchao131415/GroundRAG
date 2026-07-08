"""
统一日志配置

修复 review 第5条：日志配置散落各处（main.py 自己 basicConfig，API 层没有）。
统一到这里，所有模块用 logging.getLogger(__name__) 即可。
"""

import logging  # 标准库：日志框架核心
import sys  # 标准库：用 sys.stdout 作为日志输出流


def setup_logging(level: int = logging.INFO):
    """
    初始化全局日志（main.py 和 api.py 启动时各调一次，幂等）。

    幂等（idempotent）设计：多次调用效果等同一次。
    这里通过"root logger 已有 handler 就直接 return"实现——
    避免在 API + main 都调用时重复添加 handler 导致日志输出两遍。

    Args:
        level: 全局日志级别，默认 INFO（DEBUG/INFO/WARNING/ERROR/CRITICAL）
    """
    # root logger：logging.getLogger() 不传名就是"根 logger"
    # 所有模块 logging.getLogger(__name__) 的日志最终都会冒泡到 root
    root = logging.getLogger()
    # 幂等关键点：如果 root 已经挂了 handler，说明之前初始化过，直接返回
    # （main.py 调一次、api.py 再调一次，不会重复加 handler → 不会日志翻倍）
    if root.handlers:
        return  # 已初始化，幂等

    # 日志格式：时间 [级别] 模块名: 消息
    # 例：2026-07-02 10:00:00 [INFO] user_service: 用户登录: zhangsan
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  # 时间格式：精确到秒即可
    )
    # StreamHandler：把日志输出到流（这里选 stdout，方便容器/k8s 采集）
    # 也可换 FileHandler 写文件，但云原生一般都用 stdout + 日志采集器
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)  # 给 handler 绑定上面的格式

    # 把级别和 handler 挂到 root 上，全局生效
    root.setLevel(level)     # 设全局级别：低于此级别的日志直接丢弃
    root.addHandler(handler) # 注册 handler，root 才真正开始输出日志

    # 第三方库降噪：把依赖库的日志级别调高到 WARNING
    # 不然它们的 INFO/DEBUG 会刷屏，淹没业务日志。只留 WARNING 及以上。
    logging.getLogger("httpx").setLevel(logging.WARNING)           # HTTP 客户端：请求重试等 INFO 太吵
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING) # HF 下载模型时的进度日志
    logging.getLogger("faiss").setLevel(logging.WARNING)           # 向量检索库的内部日志
    logging.getLogger("jieba").setLevel(logging.WARNING)           # 中文分词库：每次加载词典都打日志
    logging.getLogger("openai").setLevel(logging.WARNING)          # OpenAI SDK：每次请求的 HTTP 日志
