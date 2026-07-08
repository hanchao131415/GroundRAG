"""
Reranker 精排模块（bge-reranker cross-encoder）

为什么需要 reranker（解决向量检索的精度/召回权衡）：
- 向量检索(bi-encoder)：query/doc 分别编码再算相似度，快但糙，语义相近但实际无关的会混入
- reranker(cross-encoder)：query/doc 拼一起编码，慢但准，能精确判断真实相关性
- 标准：向量+BM25 宽松召回(不漏) → RRF融合 → MMR去重 → rerank精排(去噪+排序) → topK

================= cross-encoder vs bi-encoder（核心原理） =================
【bi-encoder 双塔模型】如 bge-embedding、text2vec
  query ──encoder──> 向量 q
  doc   ──encoder──> 向量 d        # doc 向量可离线全部算好存进向量库
  相关性 = cosine(q, d)
  - 优点：doc 向量预计算 + FAISS 近似最近邻，全库检索毫秒级
  - 缺点：q、d 编码时彼此看不到，无法建模细粒度词级交互 → 精度有限
【cross-encoder 交叉模型】如 bge-reranker（本模块）
  输入 = "[CLS] query [SEP] doc [SEP]"，整体喂进一个 transformer
  query 和 doc 的 token 在 self-attention 中充分交互 → 输出一个相关性分数
  - 优点：精度高，能判断真实相关性（去噪能力远强于 bi-encoder）
  - 缺点：每个 (query,doc) 对都要单独 forward，无法预计算，O(N) 次推理
  ⇒ 只能用在【精排】：候选已被粗排缩到几十条，否则太慢
=========================================================================

v2: 使用 transformers 原生加载，绕过 FlagEmbedding 在 Windows CPU 上的 0xC0000005 崩溃。

为什么不直接用官方 FlagEmbedding 库：
  - FlagEmbedding 的 FlagReranker 底层同样是 transformers，但它会额外触发一些
    C 扩展/编译算子路径，在 Windows + 纯 CPU 环境下偶发 0xC0000005（内存访问违规），
    进程直接被操作系统杀死，Python 层根本捕获不到异常。
  - 绕过办法：直接用 transformers 的 AutoModelForSequenceClassification 加载
    同一个权重（bge-reranker 本质就是一个 num_labels=1 的 SequenceClassification 模型），
    调用方式等价，结果与 FlagReranker(normalize=True) 完全一致，但更稳。

对应《真实RAG全貌》③检索子系统。
"""

# logging：标准库日志，模块级 logger 以"模块名"为名，方便在全局日志配置里按模块过滤级别
import logging
# typing.List：类型提示，声明文档列表类型，便于 IDE 提示与静态检查
from typing import List

# LangChain 的 Document 类型：RAG 全链路统一用它承载数据（page_content + metadata）
from langchain_core.documents import Document

# 建一个本模块专属 logger；__name__ 形如 "rag_modules.reranker"，日志可按模块名精确控制
logger = logging.getLogger(__name__)


class Reranker:
    """bge-reranker cross-encoder 精排器（transformers 原生，CPU 版）。

    职责：给定一个 query 和若干候选 Document，输出按相关性重排后的 Document 列表，
    并把分数写入每个文档的 metadata['rerank_score'] 供下游/调试使用。

    说明：本类是"有状态的重型对象"——模型加载慢、占内存（约 1.1GB），因此应作为
    单例在应用启动时加载一次，之后多次复用（RAG pipeline 里共享同一个实例）。
    """

    # __slots__：固定实例允许的属性，禁止动态新增属性。
    # 好处：① 节省内存（不为每个实例建 __dict__）；② 拼错属性名会立刻报错，避免隐式 bug。
    __slots__ = ("tokenizer", "model", "_device")

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", llm=None):
        """加载 reranker 模型与 tokenizer 到 CPU。

        为什么默认 bge-reranker-base（而不是 large）：
          - base 参数量小、速度快，单 CPU 机器上几十条候选秒级即可排完，企业内部 RAG 够用；
          - large 精度略高但约 2.2GB 内存、推理慢一倍，性价比不如升级召回策略。

        Args:
            model_name: HuggingFace 模型 ID 或本地路径（默认官方 base 版，约 1.1GB 内存）。
            llm: 历史遗留的兼容参数，本类不使用 LLM（rerank 是模型自身出分，不调生成式 LLM）。
        """
        logger.info(f"加载 bge-reranker: {model_name} (transformers 原生, CPU)")

        # torch 与 transformers 在这里才 import（而非文件顶部），属于"延迟导入"：
        # ① 让本模块 import 时更轻、更快；② 调用方若没装 torch 也能 import 本模块（仅在实际实例化时报错）。
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # tokenizer：把文本切成模型能吃的 token id 序列。bge-reranker 用的是类 BERT 的 tokenizer。
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 加载模型：bge-reranker 本质是一个 num_labels=1 的 SequenceClassification 模型，
        # 对每个 (query, doc) 输出一个标量 logit = 该 pair 的相关性得分（未归一化）。
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # CPU 上用 float32 最稳（fp16 在 CPU 上支持差且慢）
            device_map="cpu",           # 强制放 CPU，避免在没有 GPU 的机器上报错
        )
        # 切到推理模式：关闭 dropout 等训练期行为，保证打分稳定可复现。
        self.model.eval()
        # 记录运行设备，后面把输入张量 .to(self._device) 时用，保持与模型所在设备一致。
        self._device = torch.device("cpu")

        logger.info("reranker 加载完成 (CPU, 约 1.1GB 内存)")

    def rerank(self, query: str, docs: List[Document], top_n: int = None) -> List[Document]:
        """对候选文档精排，按 cross-encoder 相关性分数降序返回。

        整体流程（4 步）：
          1. 构造 (query, doc) 文本对；
          2. 分批 tokenize + forward，得到每个 pair 的 logit；
          3. sigmoid 归一化为 [0,1] 的相关性概率分；
          4. 按分数降序排序，写回 metadata，截取 top_n 返回。

        Args:
            query: 用户查询文本。
            docs: 候选文档列表（通常来自上游召回+融合+去重，已收敛到几十条以内）。
            top_n: 只返回前 N 个；None 表示全部返回（顺序已排好）。

        Returns:
            按分数降序排列的 Document 列表；每条的 metadata['rerank_score'] 已写入分数。
        """
        # 边界：空候选直接返回，避免后续 tokenize 报错或返回脏数据。
        if not docs:
            return []

        import torch  # 延迟导入，理由同 __init__

        # 构造 cross-encoder 的输入：每个元素是 [query, doc正文] 文本对。
        # tokenizer 会把它们拼成 "[CLS] query [SEP] doc [SEP]" 让两个序列在模型内交互。
        # 用 page_content（纯文本正文）而非整个 Document，元数据不参与语义打分。
        pairs = [[query, d.page_content] for d in docs]

        # ---------------- 分批推理 ----------------
        # 为什么要分批：
        #   - 一次把几十条 512 长度的 pair 全塞进模型，CPU 内存/CACHE 顶不住（尤其大候选量时）；
        #   - batch_size=16 是经验值：CPU 上吞吐与内存占用的折中，单批约几百 MB 临时显存/内存，
        #     即使候选上百条也能稳定跑完；GPU 上可以调大（如 32/64）。
        scores = []
        batch_size = 16
        # range(start, end, step)：每次跳 batch_size 步，i 是每批的起始下标。
        for i in range(0, len(pairs), batch_size):
            # 切出当前这一批文本对（最多 batch_size 条）。
            batch = pairs[i:i + batch_size]
            # tokenizer 批量编码：返回 dict{input_ids, attention_mask, token_type_ids}
            inputs = self.tokenizer(
                batch,
                padding=True,        # 把本批内短序列 pad 到该批最长长度，便于组张量并行计算
                truncation=True,     # 超长截断（保护 max_length）
                return_tensors="pt", # 返回 PyTorch 张量（"pt" = pytorch）；也可 "tf"/"np"
                max_length=512,      # 截断上限 512，是 BERT 类模型的标准上下文窗口
            )
            # 把输入张量搬到模型所在设备（这里是 CPU，这一步保持代码与"换 GPU"兼容）。
            # inputs 是 dict，遍历每个 value（input_ids、attention_mask…）逐个 .to(device)。
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            # torch.no_grad()：推理时关闭梯度计算与计算图构建，省内存、提速。
            with torch.no_grad():
                # model(**inputs)：把 input_ids/attention_mask 解包成关键字参数喂给模型。
                # .logits：取分类头输出，形状 [batch, 1]（每个 pair 一个相关性标量）。
                # .squeeze(-1)：去掉末尾长度为 1 的维度 → [batch]，得到每个 pair 一个分数。
                logits = self.model(**inputs).logits.squeeze(-1)
                # sigmoid 归一化到 [0,1]，和 FlagReranker normalize=True 一致
                # logit 是任意实数，sigmoid 把它压成概率；分越大表示越相关，便于横向比较与阈值过滤。
                batch_scores = torch.sigmoid(logits).tolist()
            # tolist() 在 batch=1 时会返回单个 float 而非 list，这里统一成 list 方便 extend。
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            # 把本批分数累加进总分数列表（顺序与 pairs/docs 对齐）。
            scores.extend(batch_scores)

        # ---------------- 排序 ----------------
        # zip(docs, scores)：把文档和它的分数两两配对成 (doc, score) 元组列表。
        # key=lambda x: x[1]：按元组的第 2 项（分数）排序。
        # reverse=True：降序——最相关的排最前。
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        # 把分数写回每个文档的 metadata，供下游（如最终排序权重）或调试/可观测性使用。
        # round(...,4)：保留 4 位小数，够用且日志/传输更干净。
        for doc, score in ranked:
            doc.metadata["rerank_score"] = round(float(score), 4)

        # 打印 top5 的来源与分数，便于排查"为什么这条排前面/没排上"。source 可能缺失则显示 '?'。
        logger.info(f"  [Rerank] {len(docs)}候选 → 分数: "
                    f"{[(d.metadata.get('source', '?')[:12], round(s, 3)) for d, s in ranked[:5]]}")
        # ranked 是 (doc, score) 元组列表，这里只取 doc 部分，丢弃已写进 metadata 的分数。
        result = [d for d, _ in ranked]
        # top_n 截断：传了就返回前 N 个，没传(None)就返回全部（顺序仍是降序）。
        return result[:top_n] if top_n else result
