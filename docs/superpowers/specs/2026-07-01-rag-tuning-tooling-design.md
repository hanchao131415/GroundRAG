# RAG 参数调参工具化 — 设计文档

- **日期**：2026-07-01
- **状态**：已批准（2026-07-01）
- **关联文档**：`docs/RAG参数调参方法论.md`（方法论）、`docs/RAG调参落地手册.md`（实现后产出）

## 1. 背景与动机

`docs/RAG参数调参方法论.md` 提出「没有银弹，但有银色流程」。本设计把方法论里**最可落地**的两步做成可执行工具：

1. **接线修复**——审计发现 `RAG_VECTOR_K` / `RAG_BM25_K` / `RAG_RERANK_TOP_N` 三个配置项是死旋钮（定义于 `config.py` 但未接入检索热路径），两个最关键的阈值（向量召回阈值、rerank 拒答阈值）甚至没进 config。**调参前必须先让旋钮生效。**
2. **检索扫参脚本**——方法论里的「便宜循环」：在黄金集上扫 `chunk_size × 召回宽度`，算检索指标（Recall@K / MRR / neg_refuse），不跑 LLM、不跑 reranker，可跑上百组配置。

## 2. 目标与非目标

### 目标
- 让 `vector_search_k` / `bm25_search_k` / `rrf_k` / 向量阈值 / rerank 阈值真正可调（env 驱动）。
- 产出一个**今天就能跑、复用现有黄金集、无需新标注**的检索扫参脚本。
- 产出一篇「换公司也能照着做」的落地手册。

### 非目标（YAGNI，本轮不做）
- 生成层扫参（需 LLM，单独脚本，后续）。
- `ab_compare.py` 改造（rerank 阈值扫参，后续）。
- 在线回路 / Langfuse 接入 / Redis 缓存淘汰。
- 给黄金集补 chunk 级 `relevant_chunk_id` 标注——本设计用 answer-span 匹配规避了这步（见 §5.2）。

## 3. 关键设计决策（已确认）

| 决策 | 选择 | 理由 |
|---|---|---|
| `vector_search_k` 接线方式 | 改为 `k=vector_search_k`（替换 `top_k*2`） | 召回宽度应独立于最终 top_k，这是两阶段检索的意义；5≈6 影响可忽略，扫参会量化验证。**有意的行为变化，已批准。** |
| `rerank_top_n` | **废弃**（config 标注 deprecated，不接线） | 与 `top_k` 完全重叠；RRF 去重后候选 ~10 个，cap 无意义；多一个旋钮只会让调参矩阵更乱。 |
| 检索命中判定 | **answer-span 文件级匹配**（复用 `ground_truth` 的判别数字） | `chunk_id` 是每次切分重生的 uuid，扫 chunk_size 时不稳定；answer-span 用现有 `ground_truth`，无需新标注，且能区分 1年/3年/10年。 |
| 扫参脚本是否跑 LLM/reranker | **都不跑**（v1 只测向量+BM25+RRF 召回） | 便宜循环；rerank 是独立精排层，其阈值扫参属后续非目标。 |

## 4. Part A：接线修复

### 4.1 `config.py` 变更

新增两个环境变量驱动的阈值（默认值 = 当前硬编码行为）：

```python
# ===== ③ 检索 =====（节选，新增两项）
vector_search_k: int = int(_env("RAG_VECTOR_K", "5"))
bm25_search_k: int = int(_env("RAG_BM25_K", "5"))
rrf_k: int = int(_env("RAG_RRF_K", "60"))
# rerank_top_n: 已废弃，与 top_k 重叠，保留字段以向后兼容但不再使用
vector_score_threshold: float = float(_env("RAG_VECTOR_SCORE_THRESHOLD", "0.3"))   # 新增
rerank_threshold: float = float(_env("RAG_RERANK_THRESHOLD", "0.3"))                # 新增
```

`rerank_top_n` 字段保留（避免破坏读取该字段的代码），加注释标 deprecated。

### 4.2 `retrieval_optimization.py` 变更

**`__init__` 新增可选 `config` 参数**（行为保持的关键）：

```python
def __init__(self, vectorstore, chunks, reranker=None, config=None):
    self.vectorstore = vectorstore
    self.chunks = chunks
    self.reranker = reranker
    # 有 config 用 config，没有则退回当前硬编码默认（现有调用方零破坏）
    self.vector_search_k = getattr(config, "vector_search_k", 5) if config else 5
    self.bm25_search_k = getattr(config, "bm25_search_k", 5) if config else 5
    self.rrf_k = getattr(config, "rrf_k", 60) if config else 60
    self.vector_score_threshold = getattr(config, "vector_score_threshold", 0.3) if config else 0.3
    self.rerank_threshold = getattr(config, "rerank_threshold", 0.3) if config else 0.3
    self.setup_retrievers()
```

**`setup_retrievers`**：BM25 用 `self.bm25_search_k`（替 `:76` 硬编码 5）；向量 retriever 同步用 `self.vector_search_k`（保持一致，虽然该对象当前未被 hybrid_search 使用）。

**`hybrid_search`**：
- 签名阈值参数默认改为从 self 解析：`score_threshold=None`、`rerank_threshold=None`；函数内 `score_threshold = score_threshold if score_threshold is not None else self.vector_score_threshold`（rerank 同理）。
- 向量召回：`vs.similarity_search_with_score(query, k=self.vector_search_k)`（替 `top_k*2`）。
- `_rrf_rerank(vector_docs, bm25_docs)` 内部用 `self.rrf_k`（替硬编码 60 默认）。

**`metadata_filtered_search`**：同样把 `_rrf_rerank` 的 k 走 self.rrf_k。

### 4.3 调用方变更
- `main.py`：构造 `RetrievalOptimizationModule(..., config=self.config)`。
- `run_eval.py` / `ab_compare.py`：可选传 config（不传则行为同前）。本轮为最小改动，**仅 main.py 必改**，eval 脚本保持不传（用默认）以隔离变量。

## 5. Part B：扫参脚本 `evaluation/sweep_retrieval.py`

### 5.1 流程
```
for chunk_size in CHUNK_GRID:
    重切分文档(dp.chunk_documents with chunk_size) → chunks
    重建 FAISS 索引 → vs
    for vector_k in VK_GRID:
        for bm25_k in BK_GRID:
            构造 RetrievalOptimizationModule(vs, chunks, config=临时config(vector_k, bm25_k, ...))
            for case in eval_set:
                docs = ret.hybrid_search(case["question"], top_k=FIXED_TOP_K)  # 原始 query，不改写（隔离变量）
                算命中(见 5.2)
            汇总该配置的 Recall@K / MRR / neg_refuse / overall
    释放索引(下一个 chunk_size)
输出 sweep_result.json + 终端表格
```

### 5.2 命中判定（answer-span 文件级匹配）

```python
def is_hit(chunk, case):
    src = chunk.metadata.get("source", "")
    gt = case["ground_truth"]
    # ① 文件级：必须在正确文件里
    if case["source_doc"] and not (case["source_doc"] in src or src in case["source_doc"]):
        return False
    # ② 判别数字：gt 的所有数字必须出现在 chunk 里（区分 1年/3年/10年，坑25 思想）
    gt_nums = set(re.findall(r"\d+", gt))
    if gt_nums:
        chunk_nums = set(re.findall(r"\d+", chunk.page_content))
        if not gt_nums.issubset(chunk_nums):
            return False
    # ③ gt 无数字（如流程类问题）→ 文件级命中即可
    return True

def is_correct_refuse(docs, case):
    # 负例：无召回即正确拒答
    return case["source_doc"] == "" and len(docs) == 0
```

### 5.3 指标
- **Recall@K**：正例中 top-K 至少 1 个命中的比例。
- **MRR**：正例中首个命中 rank 的倒数均值（1/rank）。
- **neg_refuse**：负例正确拒答率。
- **overall = Recall@K × neg_refuse**（正负例都要好）。

> **结果解读注意**：v1 不接 reranker，`hybrid_search` 走无 rerank 分支（`candidates[:top_k]`），负例拒答仅靠向量 `vector_score_threshold` 过滤——但 BM25 无阈值，可能仍返回结果。因此 v1 的 `neg_refuse` 预期偏低，这恰好量化说明了「rerank 对拒答的必要性」（后续把 reranker 接进扫参即可看到 neg_refuse 跳升）。读结果时不要把 v1 的 neg_refuse 当成线上完整管道的拒答率。

### 5.4 默认网格（CLI 可覆盖）
- `CHUNK_GRID = [200, 350, 500, 700]`
- `VK_GRID = [3, 5, 8]`
- `BK_GRID = [3, 5, 8]`
- 固定：`top_k=3`、`vector_score_threshold=0.3`、`rerank_threshold=0.3`、`rrf_k=60`、`query_rewrite=关`（扫参时不改写，隔离变量）。

共 4×3×3 = 36 组配置。

### 5.5 性能与可移植性
- 小语料（本项目 ~6 文档）：每组切分+建索引秒级，36 组分钟级，可接受。
- 大语料：embedding 是瓶颈。脚本按 `content_hash` 缓存 embedding（chunk 文本 hash → 向量），同 chunk_size 内/跨配置复用。文档说明此优化点。
- 索引重建：每个 chunk_size 必须重建（chunk 变了），无法跨 chunk_size 复用。

### 5.6 输出
- `evaluation/sweep_result.json`：每配置一行 `{chunk_size, vector_k, bm25_k, recall, mrr, neg_refuse, overall}`。
- 终端表格 + 标注：`overall` 最高配置、Recall 平台点（knee：Recall 不再显著上升的最小 chunk_size）。

## 6. Part C：落地手册 `docs/RAG调参落地手册.md`

实现后产出。结构：
1. 一页纸：接线改了什么（diff 级）+ 为什么。
2. 跑扫参：环境准备 → 命令 → 怎么读结果表 → 怎么按业务约束（准确率地板/延迟/成本）选点。
3. **移植到别的公司**：换黄金集（自带 ground_truth + 负例）、换网格范围、重跑、读同样的指标。这是「可移植」的核心。
4. 进阶：把 reranker 接进扫参、生成层 top_k 扫参（指向后续工作）。
5. 面试话术。

## 7. 测试策略

- **接线回归**：新增测试验证 `config.vector_search_k/bm25_search_k/rrf_k/thresholds` 真的传到了检索（构造 config，检查 BM25Retriever 的 k、hybrid_search 行为）。`config=None` 时行为同旧（现有 22 个测试不能挂）。
- **命中判定单测**：`is_hit` 对 1年/3年/10年 三条 case 返回不同结果（关键回归）；负例 `is_correct_refuse`。
- **扫参脚本冒烟**：在极小网格（1×1×1）上跑通，产出合法 JSON。
- 复用现有 `tests/test_core.py` 风格，加一个 `TestRetrievalWiring` 类。

## 8. 风险与回滚
- **行为变化**：仅 `vector_search_k` 一处（6→5）。若 eval 显示 Recall 下降，把默认调回 6 即可（env `RAG_VECTOR_K=6`），无需改代码——这正是接线的价值。
- **config=None 兼容**：所有现有调用方不传 config 时行为完全不变，回滚成本为零。

## 9. 交付物清单
1. `config.py`：+2 阈值字段，rerank_top_n 标 deprecated。
2. `rag_modules/retrieval_optimization.py`：接线 5 个旋钮 + `config` 参数。
3. `main.py`：传 config。
4. `evaluation/sweep_retrieval.py`：新建扫参脚本。
5. `tests/test_core.py`：+`TestRetrievalWiring`。
6. `docs/RAG调参落地手册.md`：落地手册。
7. `README.md`：docs 表加一行。
