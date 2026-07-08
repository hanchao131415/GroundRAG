"""
RAG 系统配置文件（企业级，环境变量可配）

真实 RAG 系统：配置不应写死，全部走环境变量 + .env。
这样开发用本地/便宜 API，正式切生产，无需改代码。

【模块定位】
本文件是整个 RAG 系统的"配置中枢"。其他模块（文档处理、检索、生成、
鉴权）不直接读环境变量，而是统一从这里拿一个 RAGConfig 实例。这样做的好处：
  1. 配置集中——改一处即可，避免散落在各文件的硬编码；
  2. 可测试——测试时传一份假配置，不依赖真实 .env；
  3. 可审计——所有可调参数一目了然，方便 code review 和运维交接。

【为什么用 dataclass 而不是 dict / pydantic】
  - 比 dict 强：有类型标注（IDE 自动补全、静态检查），字段名写错会直接报错，
    而字典写错键名要到运行时才暴雷。
  - 比 pydantic 轻：pydantic 会自动做类型校验/转换，但引入额外依赖、启动稍慢。
    这里是"企业内部、配置项明确"的场景，类型转换我们手动写 int()/float()
    一行搞定，足够清晰，所以选了标准库 dataclass（零依赖）。
  - dataclass 自带 __init__、__repr__，写起来像普通类，读起来也直观。
"""

import os  # 标准库：提供 os.getenv() 读环境变量
from dataclasses import dataclass, field  # dataclass：装饰器，自动生成 __init__/__repr__；field：用于带默认值的可变字段（本文件暂未用到，保留导入以备扩展）
from pathlib import Path  # 跨平台路径处理（Windows/Linux 路径分隔符不同，Path 帮你抹平差异）
from typing import Dict, Any  # 类型标注用：Dict 字典、Any 任意类型
import logging

from dotenv import load_dotenv  # 第三方库 python-dotenv：把项目根目录的 .env 文件内容加载进环境变量

logger = logging.getLogger(__name__)

# 加载 .env
# —— load_dotenv() 会读取同目录（或上级）的 .env 文件，把其中 KEY=VALUE 注入到
#    os.environ 中。注意它不会覆盖已经存在的真实环境变量（即系统环境变量优先）。
#    放在模块顶层：确保 import 本文件时，环境变量就已就绪，下面的 _env() 才能读到。
load_dotenv()


def _env(key: str, default: str = "") -> str:
    """读环境变量，空字符串为默认。

    这是一个内部辅助函数（前缀 _ 表示"模块内部用，别从外部调"）。
    作用：统一封装 os.getenv，让 dataclass 字段的默认值写起来更简洁。

    参数:
        key: 环境变量名，如 "LLM_API_KEY"
        default: 取不到时用的默认值（默认空串）

    返回:
        该环境变量的字符串值（os.getenv 永远返回 str 或 None，这里兜底成 str）

    注意：返回值恒为字符串。数字/浮点类型需要在调用处再用 int()/float() 转换，
          这就是下面各字段写 int(_env(...)) 的原因。
    """
    return os.getenv(key, default)


@dataclass  # 装饰器：让 Python 自动为这个类生成 __init__(按字段顺序)、__repr__ 等方法，省去手写样板
class RAGConfig:
    """RAG 系统配置类（全部环境变量可配）。

    【整体结构】用 5 大类配置覆盖 RAG 全链路：
      ① 数据接入   —— 文档从哪来、向量索引存哪
      ② 文档处理   —— 切块（chunk）的大小与重叠
      ③ 检索       —— 嵌入模型、向量/BM25 召回、RRF 融合、阈值过滤
      ④ 生成       —— LLM 主模型 + 降级备用链
      ⑤ 权限       —— JWT 鉴权密钥与过期时间

    【字段默认值的写法】每个字段都写成 `_env("ENV_NAME", 默认值)`，
    即"先读环境变量，读不到就用默认值"。这样：
      - 本地开发：不配环境变量也能跑（用默认值）；
      - 生产部署：只改 .env 或注入环境变量，代码一行不动。
    """

    # ===== ① 数据接入 =====
    # data_path：原始文档（PDF/Word/txt 等）的存放目录。
    #   默认值用 Path(__file__).parent 动态算出"本文件所在目录下的 data/docs"，
    #   这样无论项目被拷到哪个盘，相对路径都成立，不写死绝对路径。
    data_path: str = _env("RAG_DATA_PATH", str(Path(__file__).parent / "data" / "docs"))
    # index_save_path：构建好的向量索引（embedding + BM25 等）的持久化保存目录。
    #   索引构建很慢，存盘后下次启动可直接加载，不必每次重算。
    index_save_path: str = _env("RAG_INDEX_PATH", str(Path(__file__).parent / "data" / "vector_index"))

    # ===== ② 文档处理 =====
    # chunk_size：单个文档块（chunk）的最大字符数。500 是中文场景常用值——
    #   太小块数太多检索慢，太大语义稀释、召回不准。500 约等于一段话。
    chunk_size: int = int(_env("RAG_CHUNK_SIZE", "500"))
    # chunk_overlap：相邻块之间的重叠字符数。重叠是为了避免"正好把一句话/一个
    #   关键信息切到两半"，保证检索命中的块上下文完整。25 约为 chunk_size 的 5%。
    chunk_overlap: int = int(_env("RAG_CHUNK_OVERLAP", "25"))

    # ===== ③ 检索 =====
    # embedding_model：把文本转向量用的模型。BAAI/bge-small-zh-v1.5 是中文小模型，
    #   本地可跑、速度快、效果够用。换大模型可提升精度但更慢/更贵。
    embedding_model: str = _env("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    # top_k：最终送给 LLM 的检索结果条数。3 条 = 只喂最相关的 3 段，省 token、
    #   也减少噪声干扰 LLM。调大 recall 高但 context 更长更贵。
    top_k: int = int(_env("RAG_TOP_K", "3"))
    vector_search_k: int = int(_env("RAG_VECTOR_K", "5"))    # 向量召回宽度（融合前候选数）：向量库先粗召回 5 条作为候选
    bm25_search_k: int = int(_env("RAG_BM25_K", "5"))        # BM25 召回宽度：关键词检索（稀疏）同样先召回 5 条候选
    rrf_k: int = int(_env("RAG_RRF_K", "60"))                # RRF 平滑参数：Reciprocal Rank Fusion 把两路召回结果按排名融合，k=60 是论文经验值，越大融合越平滑
    vector_score_threshold: float = float(_env("RAG_VECTOR_SCORE_THRESHOLD", "0.3"))  # 向量召回 cosine 阈值：相似度低于 0.3 的直接丢弃，避免塞入不相关内容
    rerank_threshold: float = float(_env("RAG_RERANK_THRESHOLD", "0.3"))              # rerank 拒答阈值：精排后分数低于 0.3 说明库里没有相关内容，触发"拒答"而非硬编答案

    # ===== ④ 生成（LLM，OpenAI 兼容协议，可接 GLM/DeepSeek/OpenAI/本地）=====
    # 这一组是"主 LLM"配置。只要兼容 OpenAI 接口（/chat/completions），
    # 换 base_url + api_key + model 就能切换厂商，业务代码无需改动。
    llm_provider: str = _env("LLM_PROVIDER", "zhipu")   # zhipu | deepseek | openai | local：标识当前用哪家，方便日志/降级链判断
    llm_base_url: str = _env("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")  # API 入口地址（智谱 GLM 的 OpenAI 兼容端点）
    llm_api_key: str = _env("LLM_API_KEY", "")  # 调用密钥。务必通过 .env 注入，不要写进代码/提交到 git
    llm_model: str = _env("LLM_MODEL", "glm-4-flash")  # 具体模型名。flash 系列便宜快，适合 RAG 这种"已给资料做总结"的任务
    temperature: float = float(_env("LLM_TEMPERATURE", "0.1"))  # 采样温度：0.1 偏向确定性/保守，适合"基于检索资料回答"的事实型场景，越低越不容易胡编
    max_tokens: int = int(_env("LLM_MAX_TOKENS", "2048"))  # 单次回答最大 token 数，防止失控输出/超长烧钱
    # 降级链：主 LLM 挂了，自动切备用（逗号分隔，可多个）
    # 配置示例：LLM_FALLBACK_PROVIDERS=qwen,zhipu
    #          LLM_FALLBACK_API_KEYS=sk-key1,sk-key2
    # 这四个字段都是"逗号分隔的列表字符串"，运行时再按逗号 split 成多个备用 provider。
    #   用字符串而非 list 存，是因为环境变量只能是字符串；解析逻辑放在使用方做。
    llm_fallback_providers: str = _env("LLM_FALLBACK_PROVIDERS", "")   # 备用厂商列表，如 "qwen,zhipu"
    llm_fallback_api_keys: str = _env("LLM_FALLBACK_API_KEYS", "")     # 各备用厂商的密钥，顺序与 providers 对应
    llm_fallback_models: str = _env("LLM_FALLBACK_MODELS", "")         # 各备用厂商的模型名
    llm_fallback_base_urls: str = _env("LLM_FALLBACK_BASE_URLS", "")   # 各备用厂商的 API 入口

    # ===== ⑤ 权限（JWT 认证）=====
    # JWT（JSON Web Token）：用户登录后签发一个 token，后续请求带上它即可证明身份，
    #   服务端用 jwt_secret 校验签名，无需存 session，适合无状态/分布式部署。
    # JWT 密钥（生产必须改！开发默认值仅为方便本地调试）
    #   —— 这个默认值是公开的，任何人都能伪造合法 token，所以上线前必须替换成随机长串。
    jwt_secret: str = _env("JWT_SECRET", "rag-dev-secret-change-me-in-production")
    # jwt_expire_hours：token 有效时长（小时）。24 = 一天，过期后需重新登录/刷新，
    #   平衡"用户免频繁登录"与"token 泄露后的风险窗口"。
    jwt_expire_hours: int = int(_env("JWT_EXPIRE_HOURS", "24"))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """从字典构造配置实例（工厂方法）。

        用途：当配置来自 JSON/YAML 文件或网络时，先解析成 dict，再用本方法
        转成 RAGConfig。cls(**config_dict) 会把字典的键值当作关键字参数
        传给 __init__，等价于 RAGConfig(data_path=..., chunk_size=...)。
        """
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """把配置转成字典（便于序列化、日志、持久化）。

        延迟导入 asdict：避免模块加载时就引入 dataclasses 子模块的额外开销，
        且符合"用到才导入"的习惯。
        """
        from dataclasses import asdict  # asdict 会递归把 dataclass 实例转成普通 dict
        return asdict(self)

    def validate(self) -> list:
        """校验必填项，返回缺失项列表（真实系统启动前必须自检）。

        【设计思想】不在校验失败时直接抛异常，而是把所有问题收集到一个列表返回。
        这样调用方能一次性看到全部缺失项，而不是修一个、重启、再发现下一个——
        体验更好，尤其在配置项较多的企业系统里。

        返回:
            list[str]：缺失/有问题的项；空列表表示全部通过。

        校验内容：
          - LLM_API_KEY：没有 key 根本无法生成回答，必须配；
          - data_path 存在性：目录都不存在，文档处理无从谈起。
        """
        missing = []  # 收集所有问题
        if not self.llm_api_key:  # 空串判定：未配置 LLM_API_KEY
            missing.append("LLM_API_KEY")
        if not Path(self.data_path).exists():  # 目录/文件不存在判定
            missing.append(f"data_path 不存在: {self.data_path}")
        if self.jwt_secret == "rag-dev-secret-change-me-in-production":
            logger.warning("⚠️ JWT_SECRET 仍为默认值——生产务必改成随机长串，否则 token 可伪造")
        return missing


# 默认配置实例
# —— 模块级直接 new 一个实例，其他模块 `from config import DEFAULT_CONFIG` 即可拿到
#   统一的配置对象（读取的是当前环境变量 + 默认值）。
#   注意：这是"单例式"约定，整个进程共享同一份配置；如需多套配置（如测试），
#   应自己 RAGConfig(...) 显式构造，而不是改这个全局对象。
DEFAULT_CONFIG = RAGConfig()
