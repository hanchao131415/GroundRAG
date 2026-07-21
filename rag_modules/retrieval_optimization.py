"""
检索优化模块 —— 企业 RAG 系统的"检索护城河"。

本模块实现完整的两阶段混合检索流水线，是决定"答得准不准"的关键：
    向量宽松召回(语义) → BM25 召回(关键词) → RRF 融合(排序)
    → MMR 去重(多样性) → Reranker 精排(相关性) → 阈值过滤(拒答负例)

为什么需要这么复杂？单一检索都有短板：
    - 纯向量：语义近但关键词对不上（"GPU" vs "显卡"能匹配，但"H100"这种专有名词容易漏）
    - 纯 BM25：关键词命中但语义不通（同义改写就废）
    - 融合 + 精排才能既不漏正例、又能把负例排下去。

权限层面（RBAC）：admin 走全库，普通用户只搜自己部门的"子索引"，做到权限隔离。
"""

# ---- 标准库导入（块注释一笔带过）----
import logging          # 日志，全程打详细检索轨迹便于排查
import os
import pickle           # BM25 索引持久化用，省去每次启动重建的开销
import hashlib          # 算 content_hash / chunk_id 兜底，用于去重
from pathlib import Path  # 跨平台路径，拼 bm25_index.pkl 位置
from typing import List, Dict, Any, Callable

# ---- 第三方库导入 ----
import jieba            # 中文分词，BM25 在中文上唯一的救命稻草
from langchain_community.vectorstores import FAISS          # 向量库（IP 索引，返回 cosine）
from langchain_community.retrievers import BM25Retriever     # 基于关键词的稀疏检索器
from langchain_core.documents import Document               # 文档块的统一结构（page_content + metadata）

# 本模块专用 logger，检索过程会打大量 info 级轨迹
logger = logging.getLogger(__name__)

# ============================================================================
# 坑34：英文技术专有名词注册表
# ============================================================================
# 为什么要这个大表？
#   BM25 是"按词匹配"的算法（见 chinese_tokenizer 的坑5），中文检索必须先 jieba 分词。
#   但 jieba 默认词典不懂企业里的英文技术术语——它会把 "H100" 切成 "H"、"100"，
#   "Node.js" 切成 "Node"、"."、"js"，"CI/CD" 切成乱七八糟的碎片。
#   一旦切碎，用户搜 "H100" 时 BM25 根本拼不回去 → 召回率断崖式下跌。
#
#   解法：启动时把这些词手动 add_word 进 jieba，让它当成"一个整体词"来识别。
#   这样 BM25 分词后能保留 "H100"、"Node.js"、"CI/CD" 这种完整 token，精确匹配才能成立。
#
#   重要性：这些词在企业文档（尤其《IDC相关IT基础知识》、GPU 选型手册）里高频出现，
#   不注册 = 关键词检索基本废掉，这是 P1 级别的检索质量修复。
_JIEBA_TECH_TERMS = [
    # === 开发/运维 ===
    "Redis", "MySQL", "PostgreSQL", "MongoDB", "Elasticsearch",
    "Docker", "Kubernetes", "Nginx", "Jenkins", "GitLab",
    "API", "HTTP", "HTTPS", "TCP", "DNS", "SSH", "VPN",
    "JSON", "XML", "CSV", "SQL", "NoSQL",
    "Kafka", "RabbitMQ", "CI/CD", "DevOps",
    # === 编程语言/框架 ===
    "Python", "Java", "JavaScript", "TypeScript", "HTML", "CSS",
    "React", "Vue", "Node.js", "Spring", "Django", "Flask", "FastAPI",
    # === IDC/数据中心（来自《IDC相关IT基础知识》）===
    "IDC", "CDN", "CPU", "GPU", "NVIDIA", "AMD",
    "InfiniBand", "RDMA", "SDN", "SXM",
    "ODF", "OLT", "KVM", "BIOS", "IPMI",
    "NAS", "SAN", "RAID", "UPS", "PDU",
    "RJ45", "STP", "UTP",
    "UDP", "Socket", "LAN", "WAN", "MAC", "SNMP",
    "PCI-E", "HDR", "EDR", "FDR", "QDR",
    "Mbps", "Gbps", "GHz", "MHz",
    # GPU 型号/架构
    "H800", "H100", "A100", "A800", "V100", "A40", "A30",
    "H200", "B200", "B100", "L40S", "L40", "L20", "L4", "T4",
    "H20", "RTX", "A6000", "A5000", "A4000",
    # GPU 架构代号
    "Hopper", "Ampere", "Ada", "Blackwell", "Pascal", "Volta", "Turing",
    # GPU 技术术语
    "FP16", "FP32", "FP64", "FP8", "INT8", "INT4",
    "TFLOPS", "NVLink", "NVSwitch", "显存", "HBM2", "HBM2e", "HBM3", "HBM3e",
    "TDP", "PCIe", "SXM", "NVL", "GDDR6", "GDDR6X",
    # 云计算
    "IaaS", "PaaS", "SaaS", "VPC", "CDN",
    # === 硬件/存储 ===
    "SSD", "HDD", "RAM", "ROM", "CPU",
    # === 企业办公/术语 ===
    "PDF", "DOCX", "XLSX", "PPT", "OCR",
    "HR", "IT", "OKR", "KPI", "CRM", "ERP", "OA",
    "BGP", "OSPF",
]
# 模块加载时一次性把所有术语注册进 jieba 自定义词典。
# 注意：这是"全局副作用"——import 本模块就会改 jieba 的内部词典，
# 但企业应用里这个改动是期望行为（整个进程都该用同一套分词规则）。
for _term in _JIEBA_TECH_TERMS:
    jieba.add_word(_term)


def chinese_tokenizer(text: str) -> List[str]:
    """
    中文分词器（jieba）—— BM25 在中文上必须分词，否则失效。

    【坑5：rank_bm25 默认按空格分词】
    真实坑：rank_bm25 底层实现里，对文档和查询都是用 Python 默认的
    str.split()（按空格切）。但中文书写没有空格——"显卡的功耗参数" 整句
    被当成"一个超长词"，BM25 的词频统计、IDF 计算全失效，几乎匹配不到任何东西。
    必须显式传入 preprocess_func=chinese_tokenizer，让 jieba 先把中文切成有意义的词。

    Args:
        text: 待分词的中文/中英混合文本

    Returns:
        分词后的 token 列表（已去空白和空串）
    """
    # jieba.cut 返回一个生成器，逐个产出切好的词（含标点/空白/停用词）
    # 过滤逻辑：strip() 去首尾空白 → 非空才保留
    # 注意这里没有做"去停用词"——BM25 的 IDF 本身会给停用词极低权重，不必额外处理
    tokens = [t.strip() for t in jieba.cut(text) if t.strip() and len(t.strip()) > 0]
    return tokens


def _fmt_chunks(docs: List[Document]) -> List[str]:
    """
    格式化 chunk 列表为可读形式（用于日志，区分"同一文件的不同块"）

    【真实痛点】日志只显示 source(文件名)，同一文件切多块时看着都一样，
    分不清是同一块还是不同块。加上 chunk_index + 类型 + 内容预览就清楚了。

    Args:
        docs: 文档块列表

    Returns:
        格式化后的字符串列表，形如 ["文件名#序号[类型]:内容预览", ...]
    """
    out = []
    append = out.append  # 微优化：把方法引用存到局部变量，循环里少一次属性查找
    for d in docs:
        md = d.metadata
        # 内容预览取前 25 字符，换行换成空格避免日志串行（多行日志难读）
        preview = d.page_content[:25].replace("\n", " ").strip()
        # 拼出"文件名#块序号[块类型]:内容预览"的紧凑格式
        # block_type 区分 text/table/heading 等，便于看出检索命中的是正文还是表格
        append(f"{md.get('source', '?')}#{md.get('chunk_index', '?')}[{md.get('block_type', 'text')}]:{preview}")
    return out


def _mmr_dedup(docs: List[Document], lam: float = 0.7, sim_threshold: float = 0.85) -> List[Document]:
    """
    MMR (Maximal Marginal Relevance) 去重 + 多样性重排（P1 修复）。

    【坑35：重复块污染检索结果】
    企业文档常有大量重复内容——每页的页眉页脚、模板套话、跨章节重复的术语说明。
    这些块都会命中检索，导致 top-k 被同质内容占满，真正多样的答案挤不进来。
    解法是 MMR：既看"相关性"，也看"和已选内容的差异度"，平衡两者。

    两步去重（由粗到细）：
      1. content_hash 精确去重：内容完全相同的块（不同 chunk_id 但字面一样）
         → 只保留分数最高的那个。这一步 O(n)，快且准。
      2. MMR 多样性去重：内容高度相似但不完全相同（如页眉略不同）
         → 用 3-gram Jaccard 算相似度，对相似块惩罚低分者。

    MMR 打分公式：
        score(d) = λ × relevance(d) − (1−λ) × max_similarity(d, 已选集合)
      即：相关性高的加分，但和已选内容太像的扣分。λ 权衡两者。

    Args:
        docs: 已按分数排序的候选文档（需有 rrf_score 或 vector_sim 元数据）
        lam: 相关性权重 λ，0~1。越大越偏重相关性（保留相似块），
             越小去重越激进。默认 0.7 = 偏相关性，适度去重。
        sim_threshold: Jaccard 相似度阈值，超过此值才视为"重复候选"参与惩罚。
                       默认 0.85 = 高度相似才算重复，避免误杀只是相关的内容。

    Returns:
        去重 + 重排后的文档列表
    """
    # 边界：0 或 1 个候选无需处理，直接返回（避免后续逻辑报错）
    if len(docs) <= 1:
        return docs

    # ------------------------------------------------------------------
    # 第一步：content_hash 精确去重
    # ------------------------------------------------------------------
    # seen_hashes: hash → 文档对象，记录已出现过的内容
    # deduped: 去重后的有序列表（保持原排序）
    seen_hashes = {}
    deduped = []
    for d in docs:
        # 优先用预计算的 content_hash；没有就现场算 md5（内容哈希，相同内容必同 hash）
        h = d.metadata.get("content_hash") or hashlib.md5(d.page_content.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            # 重复内容：保留分数更高的那个
            existing = seen_hashes[h]
            # 分数取值优先 rrf_score（融合后），否则退回 vector_sim（纯向量时）
            cur_score = d.metadata.get("rrf_score", d.metadata.get("vector_sim", 0))
            exist_score = existing.metadata.get("rrf_score", existing.metadata.get("vector_sim", 0))
            if cur_score > exist_score:
                # 用高分块替换低分块在列表中的位置（保持排序位置）
                deduped[deduped.index(existing)] = d
                seen_hashes[h] = d
            # 否则丢弃当前低分重复块（什么都不做）
        else:
            # 新内容：登记 hash 并加入结果
            seen_hashes[h] = d
            deduped.append(d)

    # 精确去重后若只剩 1 个，无需 MMR
    if len(deduped) <= 1:
        return deduped

    # ------------------------------------------------------------------
    # P1 性能保护：候选过多时只对 Top-15 做 MMR
    # ------------------------------------------------------------------
    # 为什么？MMR 核心是 O(n²) 的两两 Jaccard 比较（每个候选要和所有已选比）。
    # 候选 100 个时就是 ~5000 次集合运算，量大时明显拖慢检索。
    # 但尾部低分块（排名 15 之后）本就难进 top_k，重排它们性价比极低。
    # 所以策略：>20 个候选时，只对 Top-15 做 MMR，其余原样追加在后面。
    # 这样把 Jaccard 计算量封顶在 15×14/2 ≈ 100 次，性能可控。
    MMR_CAP = 15
    if len(deduped) > 20:
        top = deduped[:MMR_CAP]      # 高分区：做 MMR 重排
        rest = deduped[MMR_CAP:]     # 低分区：保持原序直接拼接
        top = _mmr_core(top, lam, sim_threshold)
        return top + rest

    # 候选不多（≤20）：全部进 MMR，质量最优
    return _mmr_core(deduped, lam, sim_threshold)


def _mmr_core(docs: List[Document], lam: float, sim_threshold: float) -> List[Document]:
    """
    MMR 核心算法：对候选做 Jaccard 相似度多样性去重。

    这是 _mmr_dedup 抽出来的内部函数——单独成函数既便于阅读，
    也让上面的性能保护逻辑能"只对 Top-N 调用"，避免重复写一遍 MMR 主体。

    算法步骤（贪心）：
      1. 第一个（分数最高的）直接入选
      2. 之后每轮：对每个剩余候选算 MMR 分（相关性 − 相似度惩罚），选最高的入选
      3. 直到所有候选都入选（顺序已被重排为"多样性优先"的次序）

    Args:
        docs: 候选文档（已按分数降序）
        lam: 相关性权重 λ
        sim_threshold: 相似度阈值，超过才施加惩罚

    Returns:
        按 MMR 分数重排后的文档列表（数量不变，只是顺序变了）
    """
    if len(docs) <= 1:
        return docs

    # ------------------------------------------------------------------
    # 工具函数：字符 n-gram 与 Jaccard 相似度
    # ------------------------------------------------------------------
    def _ngrams(text: str, n: int = 3) -> set:
        """
        字符级 n-gram 集合（中文友好，无需分词）。

        为什么用字符 n-gram 而不是词？中文分词本身有误差，且这里只是估算
        "两段文本像不像"，字符级 3-gram（连续 3 个字符的滑动窗口）对中文足够灵敏，
        还省了一次 jieba 调用。英文文档也能用，鲁棒性好。
        """
        text = text.replace("\n", " ").replace("\r", " ")  # 折行不影响相似度
        # 滑动窗口取每 3 个连续字符作为一个 gram，去重存入集合
        return {text[i:i + n] for i in range(len(text) - n + 1)}

    def _jaccard(a: set, b: set) -> float:
        """Jaccard 相似度 = 交集大小 / 并集大小，范围 0~1，越大越像。"""
        if not a or not b:
            return 0.0  # 空集边界，避免除零
        return len(a & b) / len(a | b)

    # ------------------------------------------------------------------
    # 预计算：每篇文档的 n-gram 集合和分数（避免循环里重复算）
    # ------------------------------------------------------------------
    ngram_sets = [_ngrams(d.page_content) for d in docs]
    scores = [d.metadata.get("rrf_score", d.metadata.get("vector_sim", 0)) for d in docs]

    # ------------------------------------------------------------------
    # 贪心选择主循环
    # ------------------------------------------------------------------
    selected = [0]          # 已选中文档的 index 列表，从分数最高的第 0 个开始
    candidates = list(range(1, len(docs)))  # 剩余待选 index

    while candidates:
        best_idx, best_score = None, -1.0  # 本轮最优候选及其 MMR 分
        for i in candidates:
            # 当前候选 i 与所有已选 s 的最大相似度（"最像的那个"）
            max_sim = max(_jaccard(ngram_sets[i], ngram_sets[s]) for s in selected)
            # 关键设计：只有"高度相似"（≥阈值）才施加惩罚。
            # 否则直接用原始相关性分数——避免误伤只是主题相关、内容不同的正常块。
            if max_sim >= sim_threshold:
                # MMR 公式：λ×相关性 − (1−λ)×相似度惩罚
                mmr = lam * scores[i] - (1 - lam) * max_sim
            else:
                # 不相似：不惩罚，纯按相关性走
                mmr = scores[i]
            # 跟踪本轮最高 MMR 分的候选
            if mmr > best_score:
                best_score, best_idx = mmr, i

        if best_idx is None:
            break  # 安全兜底（理论上不会触发）
        selected.append(best_idx)     # 最优者入选
        candidates.remove(best_idx)   # 移出待选池

    # 按 selected 的顺序（MMR 重排后的次序）返回文档
    return [docs[i] for i in selected]


class RetrievalOptimizationModule:
    """
    检索优化模块 - 整个 RAG 的检索核心。

    职责：把"用户问题"转成"喂给 LLM 的高质量上下文 chunk"。
    能力栈：混合检索(向量+BM25) + RRF融合 + MMR多样性去重
            + Reranker精排 + RBAC权限隔离。

    一个实例对应一份知识库；RBAC 场景下 admin 走全库、普通用户走部门子索引。
    """

    def __init__(self, vectorstore: FAISS, chunks: List[Document], reranker=None,
                 config=None, dept_indexes=None, index_save_dir: str = None):
        """
        初始化检索模块：接收依赖、解析参数、构建检索器。

        Args:
            vectorstore: 向量库（FAISS，IP 索引，similarity_search 返回 cosine）
            chunks: 全部文档块（用于 BM25 索引构建、RBAC 过滤的源数据）
            reranker: 可选的 Reranker 实例（bge-reranker）。传入则启用精排，
                      不传则只做 RRF+MMR，精排环节自动跳过。
            config: 可选 RAGConfig 对象——传入则检索参数(各阈值/k值)走 config，
                    不传则用硬编码默认值。这样老代码不传 config 也能跑（零破坏）。
            dept_indexes: 可选 Dict[部门名, FAISS]——RBAC 部门子索引。
                          传入后普通用户检索走"真·先过滤"子索引路径，性能和准确率更优。
            index_save_dir: BM25 索引持久化目录。默认优先级：
                            显式参数 > config.index_save_path > data/vector_index。
        """
        # ---- 保存核心依赖 ----
        self.vectorstore = vectorstore       # 向量库（语义检索用）
        self.chunks = chunks                 # 全量文档块（BM25 + RBAC 过滤源）
        self.reranker = reranker             # 精排器（可选）
        self.dept_indexes = dept_indexes     # 部门子索引（RBAC 加速用）

        # 解析检索参数到 self.xxx（见 _resolve_config）
        self._resolve_config(config)

        # ---- BM25 索引持久化路径 ----
        # 为什么持久化？BM25Retriever.from_documents 要对所有 chunk 建倒排索引，
        # 10 万级 chunk 时构建耗时数秒~十几秒。每次启动都重建太浪费，
        # 序列化成 bm25_index.pkl 后，下次启动直接 load，秒级就绪。
        # 路径优先级：显式参数 > config > 默认 data/vector_index（与 FAISS 同目录）
        save_dir = index_save_dir or (getattr(config, "index_save_path", None) if config else None) or "data/vector_index"
        self._bm25_path = Path(save_dir) / "bm25_index.pkl"

        # 构建检索器（向量检索器 + BM25 检索器）
        self.setup_retrievers()

    def _resolve_config(self, config):
        """
        解析检索参数：有 config 用 config，否则退回原硬编码默认（零破坏）。

        设计意图：让 RAGConfig 能集中调参（运维不用改代码），同时老调用方
        不传 config 也能用默认值正常工作。getattr(config, key, default) 模式
        天然支持"config 没这个字段就用默认"，非常稳。

        抽成独立方法的额外好处：单测时可直接 new 一个不带 vectorstore 的逻辑
        来测参数解析，不必拉起整个检索栈。

        各参数含义：
          - vector_search_k: 向量召回数量（候选池大小）
          - bm25_search_k: BM25 召回数量
          - rrf_k: RRF 融合的平滑参数（见 _rrf_rerank）
          - vector_score_threshold: 向量 cosine 阈值，低于则丢弃（宽松召回的"宽松"边界）
          - rerank_threshold: reranker 分数阈值，精排后低于则丢弃（拒答负例的关键）
        """
        # 每个参数都是"有 config 取 config，否则用硬编码默认"
        self.vector_search_k = getattr(config, "vector_search_k", 5) if config else 5
        self.bm25_search_k = getattr(config, "bm25_search_k", 5) if config else 5
        self.rrf_k = getattr(config, "rrf_k", 60) if config else 60
        self.vector_score_threshold = getattr(config, "vector_score_threshold", 0.3) if config else 0.3
        self.rerank_threshold = getattr(config, "rerank_threshold", 0.3) if config else 0.3

    def setup_retrievers(self):
        """
        构建向量检索器 + BM25 检索器。

        BM25 索引采用"懒加载 + 持久化"策略：
          1. 先看磁盘上有没有 bm25_index.pkl，有就直接 load（快）
          2. 没有就用 chunks 现场构建，构建完立刻 pickle 存盘（下次复用）

        向量检索器是轻量的——它只是 vectorstore 的一个检索视图，无需持久化。
        """
        logger.info("正在设置检索器（向量 + BM25中文分词）...")

        # ---- 向量检索器（语义匹配）----
        # as_retriever 把 FAISS 包装成 LangChain 检索器接口，k 是返回数量
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",                      # 纯相似度（不分阈值过滤）
            search_kwargs={"k": self.vector_search_k}      # 召回数量走 config
        )

        # ---- BM25 检索器：优先从磁盘加载，未命中再重建 ----
        if self._bm25_path.exists():
            try:
                with open(self._bm25_path, "rb") as f:
                    # pickle 反序列化恢复整个 BM25Retriever 对象（含倒排索引）
                    self.bm25_retriever = pickle.load(f)
                self.bm25_retriever.k = self.bm25_search_k
                logger.info(f"📂 BM25 索引从磁盘加载: {self._bm25_path} "
                            f"({len(self.chunks)} chunks)")
                return  # 加载成功，跳过重建
            except Exception as e:
                # 索引文件损坏/版本不兼容：不致命，重建即可
                logger.warning(f"BM25 索引加载失败，将重建: {e}")

        # 磁盘没有 → 现场构建。注意 preprocess_func=chinese_tokenizer 是中文救命设置
        # （坑5：不传这个，BM25 会按空格分词，中文整句变一个词，召回全废）
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            preprocess_func=chinese_tokenizer,   # ← 关键：jieba 中文分词
            k=self.bm25_search_k,
        )

        # ---- 新建后立刻持久化，下次启动免重建 ----
        try:
            # 确保父目录存在（首次运行时 data/vector_index 可能还没建）
            self._bm25_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._bm25_path, "wb") as f:
                pickle.dump(self.bm25_retriever, f)
            logger.info(f"💾 BM25 索引已持久化: {self._bm25_path}")
        except Exception as e:
            # 持久化失败不影响当次运行（内存里 retriever 已经建好），只影响下次启动
            logger.warning(f"BM25 索引持久化失败（不影响运行）: {e}")

        logger.info("检索器设置完成（BM25 已启用 jieba 中文分词）")
    
    def hybrid_search(self, query: str, top_k: int = 3, score_threshold: float = None,
                      rerank_threshold: float = None) -> List[Document]:
        """
        两阶段混合检索主流程（本模块最核心的方法）。

        设计哲学："宽进严出"——
          召回阶段尽量宽松（不漏正例），精排阶段严格过滤（拒掉负例）。
        这样既保证召回率（正例不丢），又保证精度（负例不混进 LLM 上下文）。

        完整流水线：
          ① 向量宽松召回 (cosine≥0.3)   —— 语义匹配，捞语义相近的候选
          ② BM25 召回                   —— 关键词匹配，捞专有名词命中的候选
          ③ RRF 融合                    —— 把两路结果按排名合并、去重
          ④ MMR 去重                    —— 干掉重复/同质内容，保证多样性
          ⑤ Reranker 精排 (如有)        —— bge-reranker 对 query-chunk 对做精细评分
          ⑥ 阈值过滤 + 截断 top_k       —— 低分负例被丢弃，只返回最优的几个

        Args:
            query: 查询文本
            top_k: 最终返回给 LLM 的 chunk 数量（默认 3）
            score_threshold: 向量 cosine 阈值。None 时用 self.vector_score_threshold。
                             默认 0.3 是"宽松"——多召回候选不漏正例，负例交给 rerank 去排。
            rerank_threshold: reranker 分数阈值。None 时用 self.rerank_threshold。
                              精排后低于此分的丢弃——这是"严出"的关键，负例在此被拒。

        Returns:
            排序后的文档列表（最多 top_k 个）；若向量路完全无候选则返回 [] 触发拒答。
        """
        # ---- 阈值默认从 self 取（接线：让 config 生效；显式传入仍可覆盖）----
        # 这样调用方既可以用 config 统一调参，也可以单次调用临时覆盖
        if score_threshold is None:
            score_threshold = self.vector_score_threshold
        if rerank_threshold is None:
            rerank_threshold = self.rerank_threshold

        logger.info(f"{'='*50}")
        logger.info(f"【检索问题】{query}")

        # ==================================================================
        # 阶段1：双路召回
        # ==================================================================

        # ---- 向量路：宽松召回 ----
        # similarity_search_with_score 在 IP 索引下返回的是 cosine 相似度（0~1，越大越相似）
        raw = self.vectorstore.similarity_search_with_score(query, k=self.vector_search_k)
        vector_docs = []
        for doc, sim in raw:
            # 把相似度塞进 metadata，后续 RRF/MMR/日志都要用
            doc.metadata["vector_sim"] = round(float(sim), 4)  # 保留 4 位小数，日志好看
            vector_docs.append(doc)
        # 阈值过滤：相似度低于阈值的丢弃。"宽松"就体现在默认阈值 0.3 很低——
        # 宁可多捞些边界候选交给精排，也不在召回阶段就把正例筛掉
        vector_docs = [d for d in vector_docs if d.metadata["vector_sim"] >= score_threshold]

        # ---- BM25 路：关键词召回 ----
        # invoke 内部会用 chinese_tokenizer 分词后做 BM25 匹配（见坑5）
        bm25_docs = self.bm25_retriever.invoke(query)

        # ---- 可观测：打印两路召回结果（带 chunk 区分，便于排查召回质量）----
        v_detail = [(d.metadata.get("source", "?") + f"#{d.metadata.get('chunk_index','?')}[{d.metadata.get('block_type','text')}]", d.metadata.get("vector_sim", 0)) for d in vector_docs]
        logger.info(f"  [向量召回] cosine≥{score_threshold}: {len(vector_docs)} 个:")
        for d in vector_docs:
            logger.info(f"      sim={d.metadata['vector_sim']:.3f}  {_fmt_chunks([d])[0]}")
        logger.info(f"  [BM25召回]: {_fmt_chunks(bm25_docs)}")

        # 向量路一个候选都没有 → 知识库里没有语义相关内容 → 拒答（返回空）
        # 注意：这里只看向量路不看 BM25 路，因为向量是主路；BM25 通常作为补充
        if not vector_docs:
            logger.info("  向量路无候选 → 拒答")
            return []

        # ==================================================================
        # 阶段2：RRF 融合
        # ==================================================================
        # 把向量路和 BM25 路的结果按"排名"融合成一份去重列表（见 _rrf_rerank）
        candidates = self._rrf_rerank(vector_docs, bm25_docs)
        logger.info(f"  [RRF融合] 候选 {len(candidates)} 个")

        return self._finalize_candidates(query, candidates, top_k, rerank_threshold)

    def _finalize_candidates(self, query: str, candidates: List[Document], top_k: int,
                             rerank_threshold: float = None) -> List[Document]:
        """Apply the shared MMR, rerank, and threshold tail."""
        candidates = _mmr_dedup(candidates)
        if not self.reranker:
            return candidates[:top_k]
        try:
            ranked = self.reranker.rerank(query, candidates)
        except Exception as e:
            logger.error(f"Reranker 执行失败，降级跳过精排: {e}")
            return candidates[:top_k]
        threshold = self.rerank_threshold if rerank_threshold is None else rerank_threshold
        return [
            doc for doc in ranked
            if doc.metadata.get("rerank_score", 0) >= threshold
        ][:top_k]
    
    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        带元数据过滤的检索（"先过滤后检索"模式）。

        【坑7：先检索后过滤 vs 先过滤后检索】
        原始实现（错误）："先在全库检索 top_k，再用 filters 过滤"。
          问题：全库 top_k 里可能大部分是无权限文档，过滤后剩下的寥寥无几甚至为空，
          导致用户查不到本该能查到的内容（召回不足）。
        本实现（正确）："先把全库 chunks 按权限过滤成子集，再在子集上检索"。
          这样 top_k 候选全部来自权限范围内，召回率有保障。

        对应《真实RAG全貌》⑤权限隔离。这是 dept_indexes 不可用时的回退路径
        （性能不如子索引，但逻辑正确、零破坏）。

        Args:
            query: 查询文本
            filters: 元数据过滤条件，支持单值或列表：
                     {"department": "HR"}              → 部门必须等于 HR
                     {"department": ["HR", "公共"]}     → 部门在 HR/公共 任一即可
            top_k: 返回数量

        Returns:
            过滤 + 检索后的文档列表；权限内无可见文档则返回 []
        """
        # ---- ① 先过滤 chunks 子集（圈定权限边界）----
        allowed_chunks = [
            doc for doc in self.chunks
            if self._match_filters(doc.metadata, filters)   # 逐个判断是否满足过滤条件
        ]
        if not allowed_chunks:
            # 权限范围内一个文档都没有 → 直接返回空，不让检索白跑
            logger.warning(f"权限过滤后无可见文档，filters={filters}")
            return []

        # ---- ② 在子集上重建临时检索器并检索 ----
        logger.info(f"权限过滤: 全库 {len(self.chunks)} → 可见 {len(allowed_chunks)} (filters={filters})")
        # 向量侧：复用全库 vectorstore（FAISS 不能轻易切片），多召回些(top_k*2)再用 filters 二次过滤
        # 注意：FAISS 索引是全库的，子集过滤只能"召回后筛"，所以这里还是要 _match_filters 再过一遍
        tmp_vector = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k * 2})
        # BM25 侧：直接在 allowed_chunks 上重建——BM25 是内存结构，重建快，且天然只含权限文档
        tmp_bm25 = BM25Retriever.from_documents(allowed_chunks, k=top_k * 2, preprocess_func=chinese_tokenizer)

        # 向量召回结果需要二次过滤（因为 FAISS 是全库索引，会召回到权限外的文档）
        vector_docs = [d for d in tmp_vector.invoke(query) if self._match_filters(d.metadata, filters)]
        bm25_docs = tmp_bm25.invoke(query)   # BM25 已在子集建，结果天然在权限内，无需再过滤

        # ---- 可观测：打印两路在权限内选了什么 ----
        logger.info(f"  [向量检索-权限内] 选了 {len(vector_docs)} 个:")
        for d in vector_docs:
            logger.info(f"      {_fmt_chunks([d])[0]}")
        logger.info(f"  [BM25检索-权限内] 选了 {len(bm25_docs)} 个: {_fmt_chunks(bm25_docs)}")

        # ---- 融合 + 去重 + 截断（与 hybrid_search 同样的后处理）----
        reranked = self._rrf_rerank(vector_docs, bm25_docs)
        reranked = _mmr_dedup(reranked)  # P1：MMR 去重
        logger.info(f"  [RRF融合-权限内] 最终返回 {len(reranked[:top_k])} 个:")
        for d in reranked[:top_k]:
            logger.info(f"      rrf={d.metadata.get('rrf_score',0):.4f}  {_fmt_chunks([d])[0]}")
        return reranked[:top_k]

    def _rbac_subindex_search(self, query: str, allowed_depts: List[str], top_k: int = 3) -> List[Document]:
        """
        真·先过滤的 RBAC 检索：只在 allowed_depts（已含"公共"）的部门子索引里搜。

        这是 RBAC 的"最优路径"——比 metadata_filtered_search 更快更准：
          - 向量侧：每个部门有独立的小 FAISS 索引（dept_indexes），只在授权部门里搜，
                    天然权限隔离，且召回的就是权限内文档，无需二次过滤。
          - BM25 侧：在 allowed_chunks 上重建（与 metadata_filtered_search 一致）。

        与 hybrid_search 的差异：
          - 候选池规模对齐：向量侧各子索引召回后全局排序取 top-vector_search_k，
            保证和 hybrid_search 的向量候选池等规模 → RRF 融合时两路权重平衡。
          - 尾部不含 reranker（与原 RBAC 路径一致；见 spec §11 顺带发现，可后续补）。

        Args:
            query: 查询文本
            allowed_depts: 用户可见部门列表（调用方已把"公共"加进去）
            top_k: 返回数量

        Returns:
            权限内检索结果；向量路无候选则返回 [] 拒答
        """
        # 用 set 做 O(1) 成员判断（BM25 过滤时用）
        allowed = set(allowed_depts)

        # ==================================================================
        # 向量侧：union 各部门子索引，按 cosine 全局排序
        # ==================================================================
        collected = []
        for dept in allowed_depts:
            # dept_indexes 是 Dict[部门, FAISS]；拿不到该部门的子索引就跳过
            sub = (self.dept_indexes or {}).get(dept)
            if sub is None:
                continue
            # 每个子索引各自召回 vector_search_k 个，带 cosine 分数
            for doc, sim in sub.similarity_search_with_score(query, k=self.vector_search_k):
                doc.metadata["vector_sim"] = round(float(sim), 4)
                collected.append(doc)
        # 关键：跨部门全局排序——把所有部门召回的混在一起按 cosine 排，取整体 top
        # 这样和 hybrid_search 的候选池规模一致，RRF 两路才平衡
        collected.sort(key=lambda d: d.metadata["vector_sim"], reverse=True)
        vector_docs = collected[: self.vector_search_k]
        # cosine 阈值过滤（与 hybrid_search 一致的"宽松召回"边界）
        vector_docs = [d for d in vector_docs if d.metadata["vector_sim"] >= self.vector_score_threshold]

        if not vector_docs:
            logger.info(f"  [RBAC子索引] allowed={allowed_depts} 向量路无候选 → 拒答")
            return []

        # ==================================================================
        # BM25 侧：在权限子集上重建（先过滤，与 metadata_filtered_search 一致）
        # ==================================================================
        allowed_chunks = [d for d in self.chunks if d.metadata.get("department") in allowed]
        if allowed_chunks:
            tmp_bm25 = BM25Retriever.from_documents(
                allowed_chunks, k=self.bm25_search_k, preprocess_func=chinese_tokenizer)
            bm25_docs = tmp_bm25.invoke(query)
        else:
            bm25_docs = []

        # ==================================================================
        # 尾部：RRF 融合 + MMR 去重 + 截断
        # ==================================================================
        candidates = self._rrf_rerank(vector_docs, bm25_docs)
        final = self._finalize_candidates(query, candidates, top_k)
        logger.info(f"  [RBAC子索引] allowed={allowed_depts} 向量{len(vector_docs)}+BM25{len(bm25_docs)} "
                    f"→ RRF{len(candidates)} → 返回{len(final)}")
        return final

    def permission_aware_search(self, query: str, user_departments: List[str], top_k: int = 3) -> List[Document]:
        """
        权限感知检索（RBAC）—— 对外暴露的 RBAC 检索唯一入口。

        用户只能看到自己有权限的部门文档。根据权限分两条路：
          - admin（'*'）/未登录 → 全库 hybrid_search：真·全局 cosine 排序，检索质量最优
          - 普通用户 → 部门子索引（dept_indexes 在时走 _rbac_subindex_search，真·先过滤）；
                      若子索引不可用，退回 metadata_filtered_search（全库后过滤，零破坏回退）。

        为什么 admin 不走子索引？因为 admin 能看全库，走全库全局排序才能拿到
        跨部门的真正最优结果；子索引是按部门物理隔离的，跨部门全局排序做不到。

        Args:
            query: 查询文本
            user_departments: 用户部门权限列表。'*' 表示管理员（全权限）。
            top_k: 返回数量

        Returns:
            权限范围内的检索结果
        """
        # ---- 管理员路径：全库混合检索 ----
        if "*" in user_departments:
            logger.info(f"【问答】{query}")
            logger.info("管理员权限，全库检索")
            return self.hybrid_search(query, top_k)

        # ---- 普通用户路径：部门隔离检索 ----
        # 普通用户可见 = 自己的部门 + "公共"（公共文档所有人都能看）
        # 用 set 去重：用户可能本身就在公共部门，避免重复
        allowed = list(set(user_departments + ["公共"]))
        logger.info(f"【问答】{query}")
        logger.info(f"用户权限检索: 仅可见部门 {allowed}")
        # 优先走子索引（更快更准）；子索引不可用时退回全库后过滤（保证可用）
        if self.dept_indexes is not None:
            return self._rbac_subindex_search(query, allowed, top_k)
        return self.metadata_filtered_search(query, {"department": allowed}, top_k)

    @staticmethod
    def _match_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        判断文档元数据是否匹配过滤条件（支持单值和列表）。

        用在 metadata_filtered_search 里做权限/元数据筛选。
        所有条件是 AND 关系（必须全部满足）；value 是列表时是 OR（命中任一即可）。

        Args:
            metadata: 文档的 metadata 字典
            filters: 过滤条件，如 {"department": "HR"} 或 {"department": ["HR","公共"]}

        Returns:
            True 表示文档满足所有过滤条件
        """
        # 遍历每个过滤条件，任一不满足就返回 False（AND 逻辑）
        for key, value in filters.items():
            doc_val = metadata.get(key)
            if isinstance(value, list):
                # 列表值 → OR：文档值需在列表里（如部门在 ["HR","公共"] 之一）
                if doc_val not in value:
                    return False
            else:
                # 单值 → 精确匹配
                if doc_val != value:
                    return False
        # 所有条件都满足
        return True

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = None) -> List[Document]:
        """
        RRF (Reciprocal Rank Fusion) 融合算法：把两路检索结果按"排名"合并去重。

        【为什么用 RRF？】
        向量和 BM25 的原始分数不可比——向量是 cosine(0~1)，BM25 是 TF-IDF 衍生分(无上界)。
        直接比分数没意义。RRF 的妙处是"只看排名不看分数"：
        在每一路里排名第 1 的都得固定的高分，排名第 2 的稍低……
        两路分数相加，"两路都靠前"的文档自然脱颖而出。

        【RRF 公式】
            score(d) = Σ_paths  1 / (k + rank_in_path)
        其中 rank 从 0 开始，公式里写成 1/(k+rank+1)。
        k 是平滑常数（默认 60）：
          - k 大（如 60）：各名次之间分差小，更"民主"，弱化第 1 名的统治力
          - k 小（如 1）：第 1 名分数远高于后面，更"精英"，头部名次权重极大
          - k=60 是业界经验值，对多数场景都够稳。

        【为什么用 chunk_id 去重？】
        向量路和 BM25 路很可能召回同一个 chunk（语义和关键词都命中）。
        用 chunk_id 作为文档唯一标识累加分数——两路都命中的文档分数会叠加，
        排名自然靠前，这正是我们想要的"双保险"文档。
        无 chunk_id 时降级用 md5(content)，保证仍有确定性标识。

        Args:
            vector_docs: 向量检索结果（按 cosine 排序）
            bm25_docs: BM25 检索结果（按 BM25 分数排序）
            k: RRF 平滑参数；None 时用 self.rrf_k（默认 60）

        Returns:
            融合去重后的文档列表，按 RRF 分数降序，rrf_score 写入 metadata
        """
        # k 默认从 self 取（接线 config）
        k = self.rrf_k if k is None else k
        doc_scores = {}    # doc_id → 累加的 RRF 分数
        doc_objects = {}   # doc_id → Document 对象（最后要用）

        # ==================================================================
        # 处理向量路：按排名累加 RRF 分数
        # ==================================================================
        for rank, doc in enumerate(vector_docs):
            # 唯一标识：优先 chunk_id（全局唯一，最稳）；没有就 md5(content) 兜底
            doc_id = doc.metadata.get("chunk_id") or hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc_objects[doc_id] = doc   # 记录对象（同 id 后来的会覆盖，但内容相同无妨）

            # RRF 核心：1 / (k + rank + 1)。rank=0 时分母最大但分子固定为 1 → 第 1 名分最高
            rrf_score = 1.0 / (k + rank + 1)
            # 累加：同一文档若两路都召回，分数会叠加（这正是我们要的"双保险"效果）
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # ==================================================================
        # 处理 BM25 路：同样的公式，与向量路分数累加到同一字典
        # ==================================================================
        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.metadata.get("chunk_id") or hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # ==================================================================
        # 按 RRF 总分降序排序（分数高的 = 两路都靠前 = 最相关）
        # ==================================================================
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # ==================================================================
        # 构建最终结果：把 RRF 分数写回 metadata，供后续 MMR/日志/精排使用
        # ==================================================================
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                doc.metadata['rrf_score'] = final_score   # 写回分数，下游 MMR 要用它当 relevance
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs


