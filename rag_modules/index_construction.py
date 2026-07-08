"""
索引构建模块

职责：把文档块（Document）向量化，并构建/加载/增量更新 FAISS 向量索引。
在整个 RAG 数据流中的位置：

    [文档加载] → [切块] → 【本模块：向量化 + 建索引】 → [检索] → [LLM 生成]

本模块是"离线建库"环节，产出的索引是后续检索模块（retrieval）的输入。
线程/性能相关的坑都集中在模块顶部处理（见下方 OMP/MKL 设置）。
"""

import os
# —— 必须在 import torch / faiss 之前设置 ——
# OMP_NUM_THREADS 控制 OpenMP（PyTorch CPU 算子底层用的并行框架）的线程数；
# MKL_NUM_THREADS 控制 Intel MKL（numpy/scipy 的 BLAS 后端）的线程数。
#
# 为什么强制 =1？
# Windows 下 PyTorch 的 CPU 内存分配器在多线程并发分配时存在不稳定问题
# （表现为内存暴涨或偶发崩溃）。每个线程都会预占一份线程本地内存池，
# 线程越多峰值内存越高、越容易在批量 embedding 时 OOM。
# 单线程虽然吞吐略降，但内存占用稳定、可预测，对"离线建库"这种长任务更可靠。
# 用 `if ... not in os.environ` 是为了不覆盖用户显式设的值（尊重环境配置）。
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"

# —— 以下都是标准/三方库导入，样板代码，统一块注释 ——
import logging
from typing import List, Dict          # 类型注解用（List/Dict）
from pathlib import Path               # 跨平台路径操作（Windows/Linux 都安全）

from langchain_huggingface import HuggingFaceEmbeddings   # langchain 封装的 HF 嵌入模型
from langchain_community.vectorstores import FAISS        # langchain 对 FAISS 的封装
from langchain_core.documents import Document             # 文档块的标准数据结构

# 模块级 logger：每个模块独立日志，便于按模块过滤/定位问题
logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建。

    设计为一个有状态的服务对象：
      - 构造时即加载嵌入模型（embedding），避免重复加载（模型加载很慢，几百 MB）；
      - 持有当前 vectorstore（内存中的 FAISS 索引）；
      - 对外提供：建库 / 增量更新 / 保存 / 加载 / 检索 / 部门子索引。

    注意：本类**不负责切块**，输入的 chunks 已由上游模块（chunking）切好，
    这里只做"向量化 + 建索引"。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./vector_index"):
        """
        初始化索引构建模块

        Args:
            model_name: 嵌入模型名称，默认 BAAI/bge-small-zh-v1.5
                （智源开源中文嵌入模型，512 维，体积小、中文效果好，适合 CPU 部署）
            index_save_path: 索引保存路径（FAISS 会在此目录下生成 .faiss / .pkl 文件）
        """
        # 记录配置，后续 setup_embeddings / save_index 都要用
        self.model_name = model_name
        self.index_save_path = index_save_path
        # 占位：embedding 模型和索引在后续方法中才真正初始化
        self.embeddings = None
        self.vectorstore = None
        # 构造时立即把模型加载进内存——后续所有方法都依赖它，早加载早失败（fail-fast）
        self.setup_embeddings()
    
    def setup_embeddings(self):
        """初始化嵌入模型（embedding model）。

        职责：加载 sentence-transformers / HF 模型，产出一个可调用的 embeddings 对象，
        后续 FAISS.from_documents / add_documents 都会自动调用它把文本变成向量。

        模型选型说明（面试可讲）：
          - 默认 BAAI/bge-small-zh-v1.5：智源开源，针对中文优化，512 维。
          - 选 small 而非 base/large：体积小（~100MB）、CPU 推理够快，
            中文效果在企业知识库场景已够用；large 在 CPU 上太慢，性价比低。
          - 这是"离线建库 + 在线检索"都用同一个模型，所以一次加载、全程复用。
        """
        logger.info(f"正在初始化嵌入模型: {self.model_name}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            # 强制 CPU 推理：本项目部署环境无 GPU，且 bge-small 在 CPU 上够快；
            # 显式写 'cpu' 是为了即便机器有 GPU 也不误用（避免 GPU 显存竞争）
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                # 关键：归一化向量（L2-normalize）。
                # normalize 后每条向量模长=1，此时"内积(IP)"等价于"余弦相似度(cosine)"。
                # 这直接影响 build_vector_index 里能用 IndexFlatIP 当 cosine 用。
                'normalize_embeddings': True,
                'batch_size': 8,         # Windows CPU 内存分配器不稳定，降批次
                # batch_size=8：每次只 embed 8 条文本。默认是 32，
                # 但 Windows 上 PyTorch CPU 分配器批量推理时内存波动大，
                # 小批次可压低峰值内存，换来稳定性（与顶部 OMP=1 同一思路）。
            },
            # 新版 langchain_huggingface 把进度条参数提到类层级了，
            # 不能再放进 encode_kwargs（会和 _embed 内部固定传的
            # show_progress_bar 撞车，报 "multiple values for keyword argument"）
            #
            # 为什么用 show_progress 而不是 show_progress_bar？
            # 旧版（langchain_community）参数名是 show_progress_bar 且放 encode_kwargs；
            # 新版（langchain_huggingface 独立包）改成类参数 show_progress，且 encode_kwargs
            # 里如果再传 show_progress_bar 会与底层 sentence_transformers 固定传的同名参数
            # 冲突报错。升级依赖后这是个高频踩坑点，所以显式写 False 关掉进度条。
            show_progress=False,
        )
        
        logger.info("嵌入模型初始化完成")
    
    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        构建向量索引（内积 IP = cosine，归一化向量下等价）

        选型说明（面试可讲）：
        - bge 向量已归一化（normalize_embeddings=True）
        - 归一化下：内积IP = cosine相似度（0~1，越大越相似）
        - 这里用 IndexFlatIP 而非默认的 L2：
          优点：分数直观(0~1相似度)、阈值好定、性能略快于L2
        - 归一化向量下 L2 与 cosine 单调对应(L2=√(2-2cos))，排序等价，
          但 IP(cosine) 数值更直观、更适合做相似度阈值过滤。

        数据流位置：本方法是"全量建库"的入口，输入切块后的 chunks，
        输出可直接用于检索的 self.vectorstore。

        设计取舍（坑19：IP/cosine vs 默认 L2）：
          LangChain 的 FAISS.from_documents 默认建的是 IndexFlatL2（欧氏距离）。
          但本项目全程用 IP/cosine，所以这里采用"先建默认库，再换内核"的两步法
          （详见函数内注释）。代价是多一次 reconstruct_n + add，但建库是离线低频操作，
          可接受。

        Args:
            chunks: 文档块列表
        """
        import faiss
        logger.info("正在构建FAISS向量索引(内积IP=cosine)...")

        if not chunks:
            raise ValueError("文档块列表不能为空")

        # —— 两步法第一步：先用 LangChain 默认方式建库 ——
        # from_documents 内部会自动：1) 调 embeddings 把每个 chunk 的 page_content 转成向量；
        # 2) 建一个 IndexFlatL2（默认是 L2 距离）；3) 把向量 add 进去；4) 维护 docstore 映射。
        # 我们复用它主要是因为它封装好了 docstore / id 映射这些繁琐的元数据管理。
        # 唯一的问题是：默认建的是 L2 索引，不是我们要的 IP 索引，下一步要替换。
        store = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
        # .d 是向量维度（如 bge-small 是 512）。后面新建索引必须用同样的维度。
        dim = store.index.d
        # 重建为内积索引，并重新添加已归一化的向量
        # —— 第二步：把内部的 L2 索引换成 IP 索引（核心技巧）——
        # IndexFlatIP：暴力扫描的内积索引。"Flat" 表示不压缩、不近似，召回 100% 精确，
        # 适合中小型知识库（几十万条以内）。量级再大才需要考虑 IVF/HNSW 等近似索引。
        # 为什么不直接 new_index = ...; new_index.add(embeddings.embed_documents(...))？
        # 因为 from_documents 已经 embed 过一次了，这里直接 reconstruct 复用，省掉重复 embedding。
        new_index = faiss.IndexFlatIP(dim)
        # reconstruct_n(起始, 数量)：从已建好的 L2 索引里把全部向量读出来（不重新计算）。
        # 返回 numpy 数组 (n, dim)。前提是索引支持 reconstruct（flat 索引天然支持）。
        # 取出已有向量重建（向量已归一化，IP 即 cosine）
        vectors = store.index.reconstruct_n(0, store.index.ntotal)
        # 把归一化后的向量原样灌进 IP 索引。注意顺序必须与 docstore 的 id 映射保持一致，
        # 否则检索到的向量会和文档对不上号——这里 reconstruct 保持原顺序，所以安全。
        new_index.add(vectors)
        # 替换 store 内部的 index 引用：docstore / id 映射不动，只换"度量方式"。
        # 这是整个两步法的关键一步——用 IP 内核替换 L2 内核，外层包装（docstore 等）原样保留。
        store.index = new_index

        self.vectorstore = store
        logger.info(f"向量索引构建完成(IP/cosine)，包含 {len(chunks)} 个向量")
        return self.vectorstore
    
    def add_documents(self, new_chunks: List[Document]):
        """
        向现有索引追加新文档（只增不删）。

        ⚠️ 注意（坑19/P0 修复相关）：
          本方法依赖 FAISS 的 add，但 IndexFlatIP 这类 flat 索引**不支持删除单个向量**。
          所以本方法只适合"纯新增"场景；若文档有更新/删除（同一 chunk 内容变了），
          旧向量无法移除，会变成"僵尸向量"导致重复召回。这种场景请走 build_incremental
          （它内部用全量重建规避了删除问题）。

        Args:
            new_chunks: 新的文档块列表（仅追加，不涉及更新/删除）
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")
        
        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")
        # LangChain 的 add_documents：内部会调 embeddings 把新 chunk 向量化，再追加进 FAISS。
        # 追加后 docstore 的 id 映射也会同步更新。
        self.vectorstore.add_documents(new_chunks)
        logger.info("新文档添加完成")

    def build_incremental(self, chunks: List[Document]) -> FAISS:
        """
        增量索引构建（content_hash 对比，只更新变化的 chunk）

        核心逻辑（解决故障模式3：过时上下文）：
          1. 加载已存索引和 hash 记录
          2. 对比每个 chunk 的 content_hash
          3. hash 相同的跳过，hash 不同的（新增/修改）重新 embedding
          4. 只在有变化时才更新索引

        这解决了生产 RAG 最高频的故障：文档更新后索引没更新。
        8 大故障模式至此 100% 覆盖。

        数据流位置：本方法是"文档更新后重建索引"的入口，是离线运维流程的一环。
        它的存在让 RAG 系统能在不全量重跑、不丢 docstore 元数据的前提下跟进文档变更。

        Returns:
            (vectorstore, has_changes) — has_changes=True 时需清缓存(坑29:缓存一致性)
        """
        import hashlib, json
        # hash 记录单独存成一个 json 文件（与 FAISS 索引同目录），
        # 形如 {"源文件#chunk序号": "md5内容指纹"}。下次启动时读它做对比。
        hash_file = Path(self.index_save_path) / "chunk_hashes.json"

        # 1. 加载旧 hash 记录
        old_hashes = {}
        if hash_file.exists():
            old_hashes = json.loads(hash_file.read_text(encoding="utf-8"))
            logger.info(f"加载旧 hash 记录: {len(old_hashes)} 条")

        # 2. 加载已有索引（如果存在）
        # load_index 内部会处理"路径不存在/加载失败"的情况，返回 None 即首次建库。
        existing = self.load_index()

        # 3. 筛选变化的 chunk
        # 思路：遍历当前所有 chunk，逐个算/取内容指纹，与旧记录对比。
        # hash 相同 → 内容没变，复用旧向量；hash 不同 → 内容变了，需重新 embed。
        new_chunks = []
        unchanged = 0
        current_hashes = {}
        for c in chunks:
            # content_hash 优先用上游切块模块预算好的（省一次 md5 计算）；
            # 没有就现场算 md5——md5 对内容指纹足够快，且改动一字节就会变，能可靠检测更新。
            h = c.metadata.get("content_hash") or hashlib.md5(c.page_content.encode("utf-8")).hexdigest()
            # chunk_key：用 "源文件#chunk序号" 作为唯一标识。
            # 这是"逻辑 id"，与 FAISS 内部的物理 id 解耦——即使重建索引，逻辑标识也不变，
            # 所以 hash 记录可以跨重建复用。
            chunk_key = c.metadata.get("source", "?") + "#" + str(c.metadata.get("chunk_index", ""))
            current_hashes[chunk_key] = h
            # 三种情况算"没变"：1) 有旧索引；2) 旧记录里有这个 key；3) 新旧 hash 相等。
            # 只要有一条不满足（新增 / 修改 / 首次建库），就进 new_chunks 走重建。
            if existing and chunk_key in old_hashes and old_hashes[chunk_key] == h:
                unchanged += 1
            else:
                new_chunks.append(c)

        # 只要有一个 chunk 变了，就认为本次有变更（需重建 + 通知清缓存）
        has_changes = len(new_chunks) > 0

        # 4. 有变化时必须全量重建（P0 修复：IndexFlatIP 不支持删除向量，
        #    增量 add_documents 会让旧版本文件的僵尸向量永远留在索引里，
        #    导致检索返回重复内容——新旧版本同一 chunk 都参与召回）
        #
        # 为什么是"全量重建"而不是"只更新变化的 chunk"？
        #   理想增量 = 删除旧版本的向量 + 新增新版本的向量。
        #   但 IndexFlatIP（flat 精确索引）不支持删除单条向量（没有 remove API），
        #   所以无法删旧向量。若硬用 add_documents 追加新向量，旧版本会残留——
        #   这就是"僵尸向量"：内容已过时但还在索引里参与召回，导致同一信息被召回两次
        #   （一次新一次旧），严重降低回答质量。这是真实踩过的 P0 故障。
        #   解法：只要有任何变化，就拿【全部 chunks】重新 build_vector_index。
        #   代价是重复 embed 那些"没变"的 chunk，但保证索引绝对正确。
        #   全量重建对几十万 chunk 的库也就几分钟，离线运维完全可接受。
        if new_chunks:
            # 统计哪些源文件有变化，打印明细便于运维定位（哪个文档改了）
            changed_files = sorted(set(c.metadata.get("source", "?") for c in new_chunks))
            logger.info(f"增量更新: {unchanged} 块不变, {len(new_chunks)} 块需重新 embedding")
            logger.info(f"  📝 变化文件 ({len(changed_files)} 个):")
            for f in changed_files:
                n = sum(1 for c in new_chunks if c.metadata.get("source") == f)
                logger.info(f"     {f} ({n} 块)")
            # 注意：传的是【全部 chunks】而非 new_chunks——全量重建，确保无僵尸向量
            self.build_vector_index(chunks)
        else:
            # 完全没变化：复用已加载的索引，零开销。这是最常见的"日常跑一遍增量但啥也没变"分支。
            logger.info(f"增量更新: 全部 {len(chunks)} 块未变化,跳过重建")
            self.vectorstore = existing

        # 5. 保存当前 hash 记录
        # 无论是否重建，都要把最新的 hash 记录落盘——下次启动才能正确对比。
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        hash_file.write_text(json.dumps(current_hashes, ensure_ascii=False), encoding="utf-8")
        # 兜底：确保 self.vectorstore 一定有值（build_vector_index 没跑时用 existing）
        self.vectorstore = self.vectorstore or existing
        return self.vectorstore, has_changes

    def save_index(self):
        """
        保存向量索引到配置的路径（self.index_save_path）。

        LangChain 的 FAISS.save_local 会在目标目录下写两个文件：
          - index.faiss：二进制 FAISS 索引（向量 + 结构）
          - index.pkl：pickle 序列化的 docstore（id→Document 映射）和元数据
        下次 load_index 直接读这两个文件即可还原整个索引。
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        # 确保保存目录存在（parents=True 递归创建，exist_ok=True 已存在不报错）
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"向量索引已保存到: {self.index_save_path}")
    
    def load_index(self):
        """
        从配置的路径加载向量索引。

        设计为"宽容加载"：路径不存在 / 加载失败都不抛异常，而是返回 None，
        让调用方（如 build_incremental）能优雅地走"首次建库"分支，而不是崩。
        这样把"首次运行"和"日常增量"统一到同一条代码路径里。

        ⚠️ 安全注意（allow_dangerous_deserialization）：
          FAISS 索引的 docstore 部分是用 pickle 序列化的（index.pkl）。
          pickle 反序列化时会执行任意代码——如果这个 .pkl 文件被恶意篡改/替换，
          攻击者可在加载时执行任意 Python 代码（典型反序列化 RCE）。
          因此 langchain 社区版默认拒绝加载，必须显式传 allow_dangerous_deserialization=True。
          这里设 True 的前提：索引文件由本系统自己生成、存放在受控目录、不接受外部上传。
          如果你的索引文件可能来自不可信来源，绝对不要开这个开关，应改用更安全的存储格式。

        Returns:
            加载的向量存储对象，如果加载失败返回None
        """
        # 兜底：万一 embeddings 还没初始化（比如某些调用顺序），先补上
        if not self.embeddings:
            self.setup_embeddings()

        # 路径不存在 = 首次运行，返回 None 触发上游走建库分支
        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在: {self.index_save_path}，将构建新索引")
            return None

        try:
            # 加载索引：把 .faiss + .pkl 还原成完整的 vectorstore。
            # embeddings 参数是必须的——后续 similarity_search/add_documents 还要靠它把 query/新文本转向量。
            # allow_dangerous_deserialization=True 见上方 docstring 的安全说明。
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"向量索引已从 {self.index_save_path} 加载")
            return self.vectorstore
        except Exception as e:
            # 任何加载失败（文件损坏、版本不兼容、pickle 报错等）都降级为"重建"，
            # 保证服务可用性优先于数据——宁可重建也不让加载异常把整个流程卡死。
            logger.warning(f"加载向量索引失败: {e}，将构建新索引")
            return None
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        相似度搜索（本模块提供的便捷检索接口，主要供调试/自测用）。

        内部调用 LangChain FAISS 的 similarity_search：
          1) 把 query 转成向量（用构造时加载的 embeddings）；
          2) 在 IP 索引里找内积最大的 k 条（归一化下即 cosine 最大）；
          3) 通过 docstore 把命中的向量位置映射回 Document 返回。
        
        生产检索路径一般走 retrieval 模块（含部门 RBAC 过滤 + 重排），本方法只是裸检索。

        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似文档列表（按相似度从高到低）
        """
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")

        return self.vectorstore.similarity_search(query, k=k)

    def build_department_indexes(self, vectorstore) -> Dict[str, "FAISS"]:
        """构建按部门（含公共）的子索引，供 RBAC 真·先过滤检索用。

        背景（为什么需要部门子索引）：
          企业 RAG 有权限控制（RBAC）：不同部门只能检索自己部门 + 公共的文档。
          实现 RBAC 有两种方式：
            A) 后过滤（post-filter）：在全库检索 top-k，再按部门过滤。问题：如果某部门
               文档在全库里很少，top-k 可能全是别的部门的，过滤完剩 0 条——召回率崩。
            B) 先过滤（pre-filter）：只在该部门可见的子索引里检索，召回天然准确。
          本项目采用 B，所以需要为每个部门预建一个子索引。

        从全库 vectorstore 切片，零重复嵌入（详见 _slice_department_indexes）。
        复用 self.embeddings（已加载的 bge 单例）作为子库 embedding_function。

        设计取舍（零重复嵌入）：
          最朴素的实现是"每个部门单独 embed 一遍"——但这样 N 个部门就 embed N 次同一批
          文档，浪费且慢。本方法改为"建一次全库，再从全库切片"：向量只算一次，
          之后按部门元数据把向量搬运到子索引里，embedding 开销 = O(1) 次（而非 O(N) 次）。

        Returns:
            Dict[部门名, FAISS]；无文档的部门不在其中。
        """
        logger.info("正在构建按部门子索引（零重复嵌入，从全库切片）...")
        # 实际切片逻辑抽到模块级函数 _slice_department_indexes，保持本方法薄、易读。
        # 传入 self.embeddings 是为了让子库持有同一个 embedding_function，
        # 这样后续对子库做检索时，query 转向量用的还是同一个 bge 模型（必须一致！）。
        result = _slice_department_indexes(vectorstore, self.embeddings)
        logger.info(f"部门子索引构建完成: {len(result)} 个部门 → {list(result.keys())}")
        return result


def _slice_department_indexes(vectorstore, embeddings):
    """从全库 FAISS 切片出按部门（含公共）的子索引，零重复嵌入。

    本函数是 RBAC 部门子索引的实现核心，是整个模块最"巧"的一段代码。

    整体思路（3 步）：
      1. reconstruct：从全库 FAISS 索引里把全部向量一次性读出来（不重新算）；
      2. 分桶：按每个 Document 的 metadata["department"] 把向量归到不同部门；
      3. 建子库：每个部门建一个新的 IndexFlatIP，把对应向量灌进去，
         再建一个只含该部门文档的 docstore 和 id 映射。

    关键约束：子索引里向量的物理顺序、docstore 的 id 顺序、id_map 必须三者对齐，
    否则检索会张冠李戴（查到的向量和返回的文档对不上）。

    不重新 embed：reconstruct 全库向量 → 按 docstore 的 department 元数据分桶
    → 每桶建 IndexFlatIP 子库。依赖全库索引支持 reconstruct_n（flat 索引即可）。

    Args:
        vectorstore: 全库 LangChain FAISS；其 docstore 内 Document 须带 metadata["department"]。
        embeddings: 嵌入模型（仅作为子库 embedding_function 存起来，本函数不调用）。

    Returns:
        Dict[部门名, FAISS]；无文档的部门不出现在 dict 中。
    """
    import faiss
    from collections import defaultdict

    # 取出全库底层的 faiss 索引对象（IndexFlatIP，由 build_vector_index 建好）
    index = vectorstore.index
    # ntotal：索引里向量的总数（= 全库 chunk 数）
    n = index.ntotal
    if n == 0:
        return {}

    dim = index.d
    # —— 第 1 步：reconstruct，把全库向量读出来 ——
    # reconstruct_n(0, n)：从位置 0 开始读 n 条，返回 (n, dim) 的 numpy 数组。
    # 这一步不重新跑 embedding 模型，只是把已经存好的向量拷出来，所以极快。
    # 前提：索引支持 reconstruct——flat 索引（IndexFlatIP/IndexFlatL2）天然支持；
    # 压缩索引（如 IVFPQ）则不支持，会抛异常。所以本设计依赖全库用 flat 索引。
    vectors = index.reconstruct_n(0, n)            # (n, dim) ndarray
    # index_to_docstore_id：FAISS 内部"向量物理位置 → docstore 文档 id"的映射。
    # 物理位置是 0..n-1（向量在数组里的下标），docstore id 是文档的逻辑标识（字符串/数字）。
    pos_to_id = vectorstore.index_to_docstore_id   # {position: docstore_id}
    # 把每个物理位置翻译成 docstore id，得到与 vectors 行顺序一致的 id 列表。
    all_ids = [pos_to_id[pos] for pos in range(n)]
    # 全已知 id：直接按 position 从 _dict 取（等价于 search(all_ids)，且兼容
    # 部分版本 docstore.search 不接受 list 的情况）。
    # _dict 是 docstore 内部的 {id: Document} 字典。直接用它取文档比调 search 更稳
    # （不同 langchain 版本的 search 行为不一致，有的不接受 list 入参）。
    _dict = vectorstore.docstore._dict
    # 同样按物理位置取文档，保证 all_docs[k] 与 vectors[k]、all_ids[k] 三者对齐。
    all_docs = [_dict[pos_to_id[pos]] for pos in range(n)]
    # 防御性断言：如果 docstore 缺文档（数据不一致），立即崩，避免后面静默错位。
    assert len(all_docs) == n, "docstore 缺文档，无法对齐"

    # 取 docstore 的类型，建子库时复用同类型（保持元数据结构一致）
    DocstoreCls = type(vectorstore.docstore)

    # —— 第 2 步：按部门分桶 ——
    # by_dept: {部门名: [物理位置, ...]}。range(n) 保证每个桶内的位置天然升序，
    # 这点很重要——后面取 ds_ids/docs/vectors[positions] 时依赖这个顺序与子库内顺序一致。
    by_dept = defaultdict(list)  # 部门 -> [position, ...]（range(n) 保证 position 升序）
    for pos in range(n):
        # 缺省归入"公共"部门——公共部门所有用户都能看（RBAC 里是全员可见的兜底桶）。
        dept = all_docs[pos].metadata.get("department", "公共")
        by_dept[dept].append(pos)

    # —— 第 3 步：每个部门建一个独立子索引 ——
    result = {}
    for dept, positions in by_dept.items():
        # 该部门涉及到的 docstore id 列表（顺序与 positions 一致）
        ds_ids = [all_ids[p] for p in positions]
        # 该部门的文档列表（顺序与 positions 一致）
        docs = [all_docs[p] for p in positions]

        # 子库用与全库相同的 IndexFlatIP（保持度量一致：IP/cosine）。
        sub_index = faiss.IndexFlatIP(dim)
        # vectors[positions]：用列表索引从全库向量数组里"切片"出该部门的向量，
        # 顺序与 ds_ids/docs 完全对齐（都来自同一个 positions 列表）。
        # 这就是"零重复嵌入"的精髓——向量来自 reconstruct，不是重新算的。
        sub_index.add(vectors[positions])          # 顺序与 ds_ids/docs 对齐

        # 子 docstore：{id: Document}，只含本部门文档。
        # 用 dict(zip(ds_ids, docs)) 保证 id 与文档一一对应。
        sub_docstore = DocstoreCls(dict(zip(ds_ids, docs)))
        # 子 id_map：{子库内新物理位置(0..m-1): docstore_id}。
        # 子库里向量是重新从 0 编号的，所以这里重建 position→id 映射。
        sub_id_map = {new_pos: sid for new_pos, sid in enumerate(ds_ids)}
        # 用 FAISS 的 4 参数构造函数手工拼出子库：
        #   (embedding_function, index, docstore, index_to_docstore_id)
        # 这 4 个必须互相一致，否则检索错位。前面所有的"对齐"工作都是为了这步安全。
        result[dept] = FAISS(embeddings, sub_index, sub_docstore, sub_id_map)

    return result
