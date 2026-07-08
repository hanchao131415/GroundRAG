"""
多级缓存 + 短路路由（性能优化）

解决两个真实问题（由 trace 发现）：
1. 高频问题重复问，每次都走完整链路（10秒）→ 多级缓存秒回
2. 明确问题也走 LLM 改写（6秒），但根本不需要 → 短路路由跳过

两级缓存：
  ① 精确匹配（query 哈希）—— 完全相同的问题，O(1) 命中，0ms
  ② 语义匹配（向量相似度）—— 语义相同的问题（"密码多久换"≈"密码更换周期"），命中

短路路由：
  明确问题（含具体名词、非口语）跳过 LLM 改写，直接用原 query 检索
  只有模糊/口语化问题才走 LLM 改写

对应《真实RAG全貌》⑦可观测 + ⑧服务运营。
"""

# —— 标准库导入，均为本模块所用：json 落盘/加载、re 正则、hashlib 算缓存 key、logging 打日志
import json
import re
import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

# 模块级 logger：随调用方日志配置统一输出，不在此处配 handler
logger = logging.getLogger(__name__)

# 预编译正则（热路径 _safe_to_cache_hit 每次调用都要用，避免重复编译）
# 注意：re.compile 只做一次，后续 .findall 直接用编译好的状态机，比每次现编译快很多
_RE_ENTITY = re.compile(r'[A-Za-z]+\d+', re.IGNORECASE)  # 匹配"字母+数字"实体，如 P1 / P2 / M2
_RE_DIGITS = re.compile(r'\d+')                            # 匹配所有数字串，用于校验数字是否一致（坑25）


class CacheService:
    """两级缓存：精确(哈希) + 语义(向量相似度)

    设计要点：
    - 精确层 O(1)：用 query 哈希做 key，完全相同问题直接秒回；
    - 语义层 O(n)：只在精确层未命中时退化算 cosine，用相似度阈值兜底；
    - 权限隔离：缓存 key 含 perm_sig，不同权限用户的缓存天然分开，防跨权限泄露（坑26）。
    """

    # __slots__：固定实例属性集合，禁止 Python 给每个实例生成 __dict__。
    # 好处：① 省内存（每个实例少一个 dict 对象）；② 访问属性更快（走 __slots__ 槽位，不走 __dict__ 哈希查找）。
    # 代价：不能再动态加未声明的属性。本类是热路径对象，值得用。
    __slots__ = ("cache_dir", "cache_file", "embeddings", "sem_threshold",
                 "_cache", "_hash_idx", "_lock")

    def __init__(self, cache_dir: str = "data/cache", embeddings=None, sem_threshold: float = 0.85):
        """
        初始化缓存服务。

        Args:
            cache_dir: 缓存目录（存放 qa_cache.jsonl）
            embeddings: embedding 模型（用于语义缓存），None 则只精确缓存
            sem_threshold: 语义缓存命中阈值（cosine 相似度，默认 0.85 较严，避免误命中）
                           真实坑(坑21)：0.78 太松，"p2工资"≈"p1和p2相差多少"(0.854)误命中答非所问
                           调到 0.85 + 加实体/长度校验，降低误命中率。
                           为什么不能用更低的阈值：阈值越低召回越高，但会把"相似但不同的问题"
                           也判为命中，直接返回错误的旧答案，用户体感是"答非所问"，比缓存 miss 更糟。
        """
        self.cache_dir = Path(cache_dir)             # 缓存目录，转成 Path 方便后续路径拼接
        self.cache_dir.mkdir(parents=True, exist_ok=True)  # 目录不存在则递归创建（exist_ok=True 避免已存在报错）
        self.cache_file = self.cache_dir / "qa_cache.jsonl"  # 缓存文件路径（JSONL：每行一条 JSON）
        self.embeddings = embeddings                  # 外部传入的 embedding 模型，None 表示禁用语义层
        self.sem_threshold = sem_threshold            # 语义相似度命中阈值（cosine），默认 0.85
        self._cache: List[Dict] = []  # 内存缓存：list of dict，每个 dict 是一条 {query, answer, embedding, ts}
        self._hash_idx: Dict[str, int] = {}  # hash → list index，O(1) 精确查找的索引表
        self._lock = threading.Lock()  # 线程安全锁：保护 put/get/clear 的并发访问
        self._load()                                  # 启动时把磁盘上的历史缓存加载进内存

    def _load(self):
        """从磁盘加载已有缓存。

        为什么逐行解析而非整体 json.load：缓存是 JSONL 格式（每行一条 JSON），
        天然支持追加写入（put 时 append 一行即可），且损坏一行不影响其余行。
        """
        if self.cache_file.exists():                    # 历史缓存文件存在才加载
            # splitlines 按行切；这里整文件读进内存，缓存量级不大时可接受
            for line in self.cache_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()                     # 去首尾空白
                if line:                                # 跳过空行
                    try:
                        self._cache.append(json.loads(line))  # 解析一行 JSON 追加进内存
                    except Exception:
                        pass                            # 单行解析失败（损坏数据）直接跳过，不让缓存服务起不来
            # 构建 O(1) 精确查找索引：遍历内存 list，把 hash → 下标登记进 dict
            # 之后精确命中走 self._hash_idx.get(qh)，避免每次线性扫 list
            for idx, item in enumerate(self._cache):
                self._hash_idx[item["hash"]] = idx
            logger.info(f"缓存加载 {len(self._cache)} 条")

    def _hash(self, query: str, perm_sig: str = "") -> str:
        """
        生成缓存 key 哈希（含权限签名）。

        关键修复（坑26）：缓存必须带权限维度，否则跨权限泄露。
        - HR 问"年假几天" 和 财务问"年假几天" 权限不同，必须是不同缓存条目；
        - 如果 key 只用 query，那么 HR 查过的答案会被财务直接命中，导致越权读到 HR 专属数据。
        - perm_sig = 用户可见部门集合的规范化字符串（如 "HR,公共"），作为 key 的一部分。

        Args:
            query: 用户原始查询
            perm_sig: 权限签名（见 _perm_signature）
        Returns:
            md5 十六进制摘要（32 位定长字符串，适合做 dict key）
        """
        # query 做规范化：strip 去首尾空格，lower 统一小写，让" P1 " / "p1" / "P1" 命中同一 key
        # 用 "|" 把 query 和 perm_sig 拼起来再 hash，避免不同组合意外碰撞
        key = f"{query.strip().lower()}|{perm_sig}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    @staticmethod
    def _perm_signature(departments) -> str:
        """权限签名：把部门列表规范化成稳定字符串（排序去重）。

        为什么要排序去重：同一用户每次进来部门集合顺序可能不一致
        （如 ["HR","公共"] vs ["公共","HR"]），规范化后才能保证 perm_sig 稳定，
        进而保证 _hash 出来的 key 一致、缓存可命中。
        """
        if not departments:                             # 没传部门 → 视为"全可见"（占位标记 ALL）
            return "ALL"
        # set 去重 → str 转字符串 → sorted 排序 → "," 拼接，得到稳定规范化签名
        return ",".join(sorted(set(str(d) for d in departments)))

    def get(self, query: str, departments=None) -> Optional[Dict]:
        """
        查缓存：先精确，再语义（带权限隔离）。

        两级查找顺序（性价比从高到低）：
          ① 精确匹配 O(1)：query 哈希直接 dict 查找，命中即返回，几乎 0 开销；
          ② 语义匹配 O(n)：只在前一层 miss 时才走，需算 embedding 并遍历计算 cosine，
             只在与当前请求 perm_sig 相同的条目里匹配，天然隔离权限。

        Args:
            query: 用户查询
            departments: 当前用户可见部门（权限维度，坑26），用于算 perm_sig
        Returns:
            命中返回 {"answer":..., "cache_type":"exact"|"semantic"}；未命中返回 None。
        """
        with self._lock:
            perm_sig = self._perm_signature(departments)   # 先算当前请求的权限签名
            qh = self._hash(query, perm_sig)                # 算带权限的查询哈希（精确层 key）

            # ① 精确匹配 O(1)（权限签名隔离，不同权限不串）
            # dict.get 是 O(1)，命中则直接取 list 里对应下标的条目
            idx = self._hash_idx.get(qh)
            if idx is not None:
                item = self._cache[idx]
                logger.info(f"  [缓存-精确命中 perm={perm_sig}] {query[:20]}")
                return {"answer": item["answer"], "cache_type": "exact"}

            # ② 语义匹配（仅在同权限签名内匹配，防跨权限泄露）
            # 前置条件：① 有 embedding 模型（否则语义层无意义）；② 缓存非空
            if self.embeddings and self._cache:
                q_vec = self.embeddings.embed_query(query)  # 把当前 query 编码成向量
                import numpy as np                          # 局部导入：只在真正走语义层时才加载 numpy，省启动开销
                q_vec = np.array(q_vec)                     # 转 ndarray 才能用向量运算
                best_sim, best_item = 0, None               # 记录最高相似度及对应条目
                for item in self._cache:                    # 线性遍历缓存（量级不大时可接受）
                    if "embedding" not in item:             # 没向量的老条目（如禁用 embedding 时存的）跳过
                        continue
                    # 权限隔离核心：只在同 perm_sig 的缓存条目里做语义匹配
                    # 不同权限的条目即便向量再像，也绝不参与匹配，从源头杜绝越权返回
                    if item.get("perm_sig", "") != perm_sig:
                        continue
                    v = np.array(item["embedding"])         # 取缓存条目的向量
                    # cosine 相似度 = 点积 / (模长×模长)，+1e-9 防止除零（零向量）
                    sim = float(np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v) + 1e-9))
                    if sim > best_sim:                      # 维护当前最高相似度条目
                        best_sim, best_item = sim, item

                # 相似度过了阈值，还要过 _safe_to_cache_hit 四重校验才算真命中（坑21/25）
                if best_sim >= self.sem_threshold and best_item:
                    cached_q = best_item["query"]
                    if not self._safe_to_cache_hit(query, cached_q, best_sim):
                        # 语义像但本质不同（实体/数字/意图不一致）→ 否决，宁可 miss 也不能答错
                        logger.info(f"  [缓存-语义命中被否决] sim={best_sim:.3f} 校验未过: {query[:20]}vs{cached_q[:20]}")
                        return None
                    logger.info(f"  [缓存-语义命中 perm={perm_sig}] sim={best_sim:.3f} {query[:20]}≈{cached_q[:20]}")
                    return {"answer": best_item["answer"], "cache_type": "semantic"}

            return None                                     # 两层都没命中，返回 None 让上层走完整链路

    @staticmethod
    def _safe_to_cache_hit(query: str, cached_q: str, sim: float) -> bool:
        """
        语义缓存命中的安全校验（防误命中，坑21/25，整个文件最核心的方法）。

        为什么需要它：单纯靠 cosine 相似度阈值并不可靠——两个问题可能"用词很像"
        但"问的根本不是一回事"。一旦误命中，用户拿到的是旧问题的错误答案，
        比缓存未命中更糟糕。所以这里再加四重规则校验，任何一条不过都否决。

        四重校验：
          ① 长度差异：两个 query 长度差超过 40%，大概率是不同问题；
          ② 实体匹配：新 query 的关键实体（如 P1/M2）必须在 cached_q 里也出现；
          ③ 意图词一致：含"比较/计算"意图词的状态必须一致（一个问相差、一个问单值 → 否决）；
          ④ 数字完全一致：所有数字必须一一对应（"满1年"≠"满3年"，坑25）。
        （相似度阈值 0.85 已在外层 sem_threshold 判断，本方法只做语义之外的规则校验。）

        真实案例：
          "p1 和 P2 工资相差多少" ≈ "p2 基本工资多少" (sim 0.854)
          → 实体不同(P1 vs 只P2)、意图不同(相差 vs 单值) → 否决，不命中

        Args:
            query: 当前用户的新查询
            cached_q: 候选命中条目的原始 query
            sim: 二者的 cosine 相似度（已 ≥ 阈值，本方法用不到，但保留签名便于扩展）
        Returns:
            True=可命中；False=否决（语义虽像但本质不同）
        """
        # ① 长度差异校验：长度差超过 40% 视为可能不同问题
        # 直觉：问题长度差太多，往往一个详尽一个简短，问的侧重点不同
        len_q, len_c = len(query), len(cached_q)
        if max(len_q, len_c) > 0:                       # 防止两个都为空时除零
            ratio = abs(len_q - len_c) / max(len_q, len_c)  # 用较长的那个做分母，得到"差异占比"
            if ratio > 0.4:                              # 差异 > 40% → 判定为可能不是同一问题
                return False

        # ② 实体匹配校验：提取新 query 的"实体标识"（字母数字组合如 P1/M2）
        #    新 query 独有的实体若不在 cached_q 里，说明问的不是同一个东西
        new_entities = set(_RE_ENTITY.findall(query))    # 如 {P1, P2, M2}，用 set 便于做集合运算
        cached_entities = set(_RE_ENTITY.findall(cached_q))
        if new_entities and cached_entities:
            # 新 query 有 cached_q 没有的实体 → 可能问不同对象
            # 例：新 query 问 P1，cached 只讲 P2 → 不是一回事，否决
            if new_entities - cached_entities:           # 差集非空说明新 query 多了实体
                return False
        elif new_entities and not cached_entities:
            # 新 query 有实体但 cached 没有 → 一定不同
            return False
        # 注：若两者都没实体（纯中文），本条放过，交给③④继续判

        # ③ 含"比较/计算"意图词的 query 不缓存命中（这类问题必须重新算）
        # 直觉：问"相差"是二元比较，问"多少"是单值查询，即便主体相同答案也完全不同
        #       → 二者的"比较意图"必须一致，一个有一个没有就否决
        compare_words = ["相差", "差多少", "比较", "哪个多", "哪个少", "比例", "百分比", "一共", "总共"]
        has_compare = any(w in query for w in compare_words)      # 新 query 是否含比较意图词
        cached_has_compare = any(w in cached_q for w in compare_words)  # cached 是否含比较意图词
        if has_compare != cached_has_compare:           # XOR：只有一方含比较词 → 意图不同，否决
            return False

        # ④ 数字必须完全一致（坑25：数字不同=不同问题）
        # 真实案例："工作满1年年假几天"≈"工作满3年年假几天"(sim 0.940)
        #   只差一个字，但答案完全不同（1年5天 vs 3年10天）→ 必须否决
        # 注：q_numbers != c_numbers 已隐含至少一方非空（空集 == 空集不会 !=），
        # 故无需再判 q_numbers or c_numbers，直接否决即可。
        q_numbers = set(_RE_DIGITS.findall(query))      # 提取新 query 所有数字串，如 {'1'}
        c_numbers = set(_RE_DIGITS.findall(cached_q))   # 提取 cached 所有数字串，如 {'3'}
        if q_numbers != c_numbers:                      # 数字集合不等 → 数字不一致，否决
            return False

        return True                                     # 四重校验全过 → 安全，可命中

    def put(self, query: str, answer: str, departments=None):
        """存缓存（带权限签名，坑26：缓存隔离权限，防跨权限泄露）。

        命中失败后由上层调用，把"问→答"对写进缓存，供下次复用。
        每条记录同时维护：内存 list + 精确索引 dict + JSONL 落盘。

        Args:
            query: 用户原始查询
            answer: 本次生成的答案
            departments: 当前用户可见部门（权限维度）
        """
        with self._lock:
            perm_sig = self._perm_signature(departments)   # 算权限签名，写入条目里
            item = {
                "hash": self._hash(query, perm_sig),  # 含权限签名的 key（精确层用）
                "perm_sig": perm_sig,                  # 权限签名（语义匹配时做隔离用）
                "query": query,
                "answer": answer,
                "ts": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),  # 时间戳，便于排查
            }
            if self.embeddings:                             # 有 embedding 模型才存向量，供语义层用
                item["embedding"] = self.embeddings.embed_query(query)
            idx = len(self._cache)                          # 新条目将落在 list 末尾的下标
            self._cache.append(item)                        # 内存 list 追加
            self._hash_idx[item["hash"]] = idx  # 维护 O(1) 索引：hash → 下标
            # 落盘（追加模式）：JSONL 每行一条 JSON，写入即可，无需重写整个文件
            with open(self.cache_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")  # ensure_ascii=False 保留中文可读

    def stats(self) -> Dict:
        """返回缓存统计信息（当前只暴露总数，可按需扩展命中率等指标）。"""
        return {"total": len(self._cache)}

    def clear(self):
        """清空缓存（增量索引检测到文档变化时调用，坑29：缓存一致性）。

        缓存一致性方案：当知识库文档被增删改，索引重建后，旧缓存里的答案可能已过时
        （甚至指向不存在的文档）。此时必须整体失效缓存，避免把过期答案返回给用户。
        做法：清空内存（list + 索引）并把磁盘 JSONL 截断为空。
        """
        with self._lock:
            n = len(self._cache)                            # 先记下数量用于日志
            self._cache.clear()                             # 清空内存 list
            self._hash_idx.clear()                          # 同步清空精确索引
            if self.cache_file.exists():                    # 磁盘文件存在则清空
                self.cache_file.write_text("", encoding="utf-8")  # 写空串 = 截断为 0 字节
            logger.info(f"🗑️ 缓存已清空（{n} 条），索引更新后缓存失效")


def is_simple_query(query: str) -> bool:
    """
    短路路由判断：明确问题跳过 LLM 改写（坑20/24）。

    背景：原本所有 query 都先过 LLM 改写（耗时约 6 秒），但很多问题本身已经很明确
    （含具体名词、非口语），改写纯属浪费。本函数快速判定是否"足够明确"可直接检索。

    判定"明确"（不需改写，return True）：
    - 长度≥4字；
    - 不含纯口语黑名单词；
    - （简单启发式）默认通过，由上层用原 query 直接检索。

    判定"模糊"（需改写，return False）：
    - 过短（<4字）：信息量不足，需补全；
    - 命中口语黑名单（"那个""怎么搞""帮我看看"等）：纯代词/口语，需 LLM 改写明确化。

    Args:
        query: 用户查询
    Returns:
        True=明确，可短路（跳过 LLM 改写）；False=模糊，需走 LLM 改写
    """
    q = query.strip()                                   # 去首尾空白后再判断长度
    # 过短：长度 < 4 视为信息量不足（如"年假""密码"），需要改写补全
    if len(q) < 4:
        return False
    # 纯口语黑名单：命中任一即判模糊，这些词没有明确意图，必须改写
    vague = ["那个", "这个", "怎么搞", "帮我看看", "咋办", "怎么办", "啥", "那个啥"]
    if any(v in q for v in vague):
        return False
    # 含具体实体（简单启发式：含≥2字中文词或≥3字英文）
    # 能走到这里说明长度够且不含口语词 → 默认视为明确，可短路
    return True
