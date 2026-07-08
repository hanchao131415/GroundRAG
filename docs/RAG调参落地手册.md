# RAG 调参落地手册

> 7 节。`docs/RAG参数调参方法论.md` 的**可执行配套**：方法论讲"为什么这么调"，本手册讲"在本项目里怎么动手调、换公司怎么搬"。

---

## 0. 一句话定位

本手册是 `docs/RAG参数调参方法论.md` 的**可执行配套**：方法论讲"为什么这么调"，本手册讲"在本项目里怎么动手调、换公司怎么搬"。

如果你只读一篇文档就要上手，读这篇。

---

## 1. 接线改了什么（diff 级）

调参的前提是旋钮真的连着线。Task 1–3 把原来"改了不生效"的死旋钮接上，改动集中在三个文件：

**`config.py`** —— 新增两个阈值字段，全部环境变量可配：

| 字段 | 默认 | 环境变量 | 作用 |
|---|---|---|---|
| `vector_score_threshold` | `0.3` | `RAG_VECTOR_SCORE_THRESHOLD` | 向量召回 cosine 阈值，低于则丢弃 |
| `rerank_threshold` | `0.3` | `RAG_RERANK_THRESHOLD` | rerank 分数阈值，精排后低于则丢弃（负例拒答关键） |

同时把 `rerank_top_n` 标注为**废弃**：它与 `top_k` 重叠（reranker 给所有候选打分后，由 threshold + top_k 截断就够），字段保留仅为向后兼容，检索热路径不再读取。调参请改 `top_k`。

**`rag_modules/retrieval_optimization.py`** —— `RetrievalOptimizationModule.__init__` 增加可选 `config` 参数；5 个旋钮从 config 读取：

- `vector_search_k`（默认 5，env `RAG_VECTOR_K`）
- `bm25_search_k`（默认 5，env `RAG_BM25_K`）
- `rrf_k`（默认 60，env `RAG_RRF_K`）
- `vector_score_threshold`（默认 0.3）
- `rerank_threshold`（默认 0.3）

**`config=None` 时退回原硬编码默认（5/5/60/0.3/0.3），现有调用方零破坏**——这是有意设计的渐进式接线，避免一次性破坏所有依赖该模块的代码。

**`main.py`** —— 构造检索模块时传 `config=self.config`：

```python
self.retrieval_module = RetrievalOptimizationModule(
    vectorstore, chunks, reranker=self._init_reranker(), config=self.config)
```

这让 `.env` 里的检索参数在 CLI / API 真实路径上生效。改 `.env` 不再需要改代码。

**唯一有意的运行时行为变化**：向量召回宽度从原来的 `top_k*2` 改为 `vector_search_k`（默认 5）。
- **理由**：召回宽度应独立于最终 `top_k`——这正是两阶段检索（宽松召回 + 精排截断）存在的意义；把召回宽度绑死在 `top_k` 上会让两个旋钮耦合，没法独立扫。
- **影响**：默认场景下 5 ≈ 原 `top_k*2=6`，影响可忽略，但扫参时它变成可量化的独立维度。
- **回退**：若实测 Recall 下降，把 `RAG_VECTOR_K` 调大即可（如设 8/10），无需改代码。

**为什么调参前必须先接线**：死旋钮改了不生效，会让人误判"这个参数对效果没影响，不重要"。接线审计是扫参的第一步——先确认你扫的每个值都真的进入了检索路径，否则整张扫参表都是在读一个常数。

---

## 2. 跑扫参

**环境**：conda env `py312`（Python 3.12，**不要用 3.14**——见 `docs/环境配置.md`）。

**基础命令**（从 `project1-rag/` 目录）：

```bash
conda run -n py312 python evaluation/sweep_retrieval.py
```

**Windows 报 `UnicodeEncodeError`**（中文输出 + 默认 GBM 控制台编码），用：

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 conda run --no-capture-output -n py312 python evaluation/sweep_retrieval.py
```

**自定义网格**（按自己文档分布收窄/拓宽）：

```bash
python evaluation/sweep_retrieval.py --chunks 300,500,700 --vk 3,5,8 --bk 3,5,8 --top-k 3
```

完整参数：`--chunks`（chunk_size 网格）、`--vk`（vector_search_k）、`--bk`（bm25_search_k）、`--top-k`（最终 top_k，固定不扫）。

**输出**：
- **终端**：逐组打印 `recall / mrr / neg_refuse / overall` + 命中计数（如 `34/37`），最后一张 Top5 表。
- **文件**：`evaluation/sweep_result.json`——全部 36 组明细，按 `overall` 降序，可直接喂给脚本/表格二次分析。

**成本**：默认网格 4×3×3 = 36 组；脚本**只测检索层**（向量+BM25+RRF），不跑 LLM、不跑 reranker，所以非常便宜，秒级到分钟级跑完（嵌入模型只加载一次，靠 `idx.embeddings` 复用）。

**实测基线**（默认 chunk=500 / vk=5 / bk=5）：

| 指标 | 值 | 说明 |
|---|---|---|
| recall | ≈ 0.92 | 34/37 正例命中 |
| neg_refuse | 0 | v1 未接 reranker，负例全被召回（向量阈值 0.3 太松拦不住） |
| overall | 0 | `overall = recall × neg_refuse`，neg_refuse=0 直接归零 |

---

## 3. 怎么读结果表 + 选点

扫完拿到 36 行，怎么选？三步：

### 3.1 看 Recall 平台点（knee）

固定 `vk/bk`，把 `chunk_size` 从小到大排，看 recall 上升后**变平**的那个最小 chunk_size——就是性价比点。再大 recall 收益递减，且 chunk 太大会稀释信号（一个 chunk 塞太多信息，精排和 LLM 都更难定位答案）。

### 3.2 看 overall（= recall × neg_refuse）

v1 因为 **neg_refuse=0 导致 overall 全 0**，整列没区分度。所以 v1 阶段**主要看 recall + mrr** 来选检索参数；等接上 reranker 后，neg_refuse 会跳升，overall 才成为有效的综合指标。

> 不要因为 overall 全 0 就觉得扫参白做了——recall/mrr 的差异就是这一阶段要捕捉的信号。

### 3.3 按业务约束选点（三选一）

最优不是单点，是**业务约束下的 Pareto 拐点**：

| 你的约束 | 选点策略 |
|---|---|
| **准确率地板**（如"recall 必须 ≥0.90"） | 在满足 recall≥0.90 的行里，挑最小的 chunk_size / 最小的 K——最省成本 |
| **延迟预算** | rerank 候选数（vk+bk）决定精排延迟；在延迟上限内取 recall 最高的那组 |
| **成本预算** | top_k 决定喂给 LLM 的 token；同理在 token 预算内取 recall 最高的 |

### 3.4 诊断决策表（症状 → 旋钮）

直接对应方法论第 6 节。recall 上不去时按这个查：

| 症状 | 可能原因 | 该调的旋钮 |
|---|---|---|
| recall 低、调 chunk_size 无改善 | 召回宽度不够 / 阈值太严 | 调大 `RAG_VECTOR_K` / 调小 `vector_score_threshold` |
| recall 随 chunk_size 增大反而下降 | chunk 太大稀释信号 | 调小 chunk_size，或加大 chunk_overlap |
| recall 高但答案被淹没 | 排序问题，不是召回问题 | 调 `rrf_k`（小 k 放大头部权重），或接 reranker |
| neg_refuse=0（负例乱答） | 缺精排阈值过滤 | 接 reranker + 调 `rerank_threshold`（v1 暂时无法解决） |
| BM25 路完全没贡献 | 中文未分词（jieba 没生效） | 检查 `chinese_tokenizer` 是否传给 BM25Retriever |
| 换数据后 recall 暴跌 | 文档段落长度分布变了 | 重新量文档平均字数，重定 chunk_size 网格 |

---

## 4. 移植到别的公司（核心）

这一节是手册存在的主要理由。**流程不变，只换数据和范围**，4 步：

### Step 1：换黄金集

准备自己的 `evaluation/eval_dataset.jsonl`，每行一个 JSON：

```json
{"question": "年假有几天？", "ground_truth": "入职满1年5天，满3年10天", "source_doc": "HR/年假制度.pdf"}
{"question": "公司有免费午餐吗？", "ground_truth": "", "source_doc": ""}
```

字段约定：

| 字段 | 正例 | 负例（知识库没有的问题） |
|---|---|---|
| `question` | 真实问法 | 知识库答不了的问题 |
| `ground_truth` | 标准答案，**数字要写全**（如"5天/10天"） | 留空或随便填 |
| `source_doc` | 答案所在文件路径（或文件名片段） | **必须留空 `""`** |

关键：`ground_truth` 里的**数字会被当成判别指纹**——匹配逻辑用"gt 的全部数字 ⊆ chunk 文本"来判断命中。所以"1年/3年/10年"这种同文件不同档位的问题，必须把数字写全，否则 1 年和 3 年的题会被判成同一个命中。

### Step 2：换网格范围

按自己文档的长度分布定 `chunk_size` 上下界：

- 先量一下文档段落的平均字数（脚本或 `wc`）。
- 网格至少覆盖 `<平均字数`、`≈平均字数`、`>平均字数` 三档。
- `vk/bk` 从 `{3,5,8}` 起步（小语料可加 10/12，大语料从 5 开始够了）。

### Step 3：重跑

把你的文档放到 `data/docs/`（或改 `.env` 的 `RAG_DATA_PATH` 指到你的目录），然后：

```bash
python evaluation/sweep_retrieval.py
```

不需要改任何代码——`config.py` 默认就指向 `data/docs/`。

### Step 4：读同样的指标选点

recall / mrr / overall 的解读方式**完全不变**，照第 3 节来。

---

**命中的是 answer-span 文件级匹配**——换公司时只要黄金集带 `ground_truth` + `source_doc` + 负例，匹配逻辑自动适配，**无需改代码**。这是这套扫参流程可移植的关键：不依赖 chunk 级标注（那东西标起来极贵），只依赖文件级标注 + ground_truth 里本来就有的数字。

---

## 5. 结果解读注意（避坑）

1. **v1 不接 reranker → neg_refuse 偏低属正常**。BM25 没有阈值，对负例（知识库没有的问题）仍会召回最相关的几个 chunk，所以 neg_refuse≈0。这恰好**量化了"rerank 对拒答的必要性"**——接上 reranker 后再扫一次，neg_refuse 会明显跳升。

2. **不要把 v1 的 neg_refuse 当成线上完整管道的拒答率**。线上有 reranker 精排 + 阈值过滤，拒答率会高得多。v1 的 neg_refuse 只反映"向量阈值过滤"这一层的效果。

3. **answer-span 匹配的局限**：
   - `ground_truth` **无数字**时（流程类问题，如"年假怎么申请"）→ 退化为**纯文件级命中**（粒度变粗，只要来自正确文件就算命中，区分不出同文件的不同问题）。
   - `ground_truth` **含数字**时（如"5天/10天"）→ 按"gt 全部数字 ⊆ chunk"判，能区分 1年/3年/10年档位。

4. **chunk_size 不是越大越好**。recall 在某个点会平台化甚至下降（信号被稀释），扫参能直接看到这个拐点，凭感觉调容易调过头。

5. **mrr 比 recall 更敏感于排序**。recall 一样时，看 mrr 选——mrr 高意味着正确答案排更前，对后续 LLM 更友好。

---

## 6. 进阶（指向后续）

v1 只扫了检索层（向量+BM25+RRF）。要继续往下游扫：

### 6.1 把 reranker 接进扫参

在 `eval_one_config` 里给 `ret` 传 reranker（`RetrievalOptimizationModule(vs, chunks, reranker=..., config=sweep_cfg)`），再加扫 `rerank_threshold` 网格（如 `0.2,0.3,0.5`）。预期：`neg_refuse` 从 0 跳升到 0.7+，`overall` 不再恒 0，综合指标才有意义。

### 6.2 生成层扫 top_k

生成层（喂给 LLM 的 chunk 数）要 LLM 裁判，**贵**。所以只对检索层选出的 3–5 个 finalist 配置跑，不要全网格跑。典型现象：top_k 在 3–5 见顶，之后 lost-in-the-middle，忠实度/相关性反而下降。

### 6.3 大语料优化

- **按 `content_hash` 缓存 embedding**：同 chunk 文本跨配置复用向量，避免重复编码（本扫参脚本已用 `idx.embeddings` 单次加载嵌入模型，是同思想的简化版）。
- **增量索引**：文档变更时只重新编码变更部分，不全量重建 FAISS。

---

## 7. 面试话术（3–5 句背诵）

这套话术覆盖"流程 / 接线审计 / 分层扫参 / 命中判定 / Pareto"五个点，照背即可：

1. **"RAG 没银弹，但有通用流程：黄金集 → 分层扫参 → Pareto 选点 → 在线回路。"**

2. **"调参前我先审计了旋钮接线——发现 vector_k/bm25_k/rerank_top_n 是死旋钮，先接线再扫参。"**
   （死旋钮 = 改了值但代码里没读，扫出来的"最优"是假的。）

3. **"检索层用 Recall@K/MRR 扫，不跑 LLM（便宜）；生成层才用 LLM 裁判扫 top_k。"**
   （分层是因为沿管道因果序：检索不行生成一定不行，且检索层评估便宜得多。）

4. **"命中判定用 answer-span 文件级匹配，复用 ground_truth 的判别数字，无需 chunk 级标注就能区分 1年/3年档位。"**
   （chunk 级标注极贵；文件级 + 数字指纹是成本/粒度的甜点位。）

5. **"最优不是单点，是业务约束下的 Pareto 拐点。"**
   （recall/延迟/成本三角，按业务地板选点，而不是无脑追最高 recall。）
