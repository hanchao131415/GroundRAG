# RAG 参数调参工具化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 RAG 检索参数真正可调（接线死旋钮），并产出一个复用现有黄金集、不跑 LLM 的检索层扫参脚本。

**Architecture:** 三部分：① `config.py` + `retrieval_optimization.py` 接线（新增 `config` 入参，`config=None` 时行为同旧，零破坏）；② `evaluation/sweep_retrieval.py` 扫 `chunk_size × vector_k × bm25_k`，用 answer-span 文件级匹配算 Recall@K/MRR/neg_refuse；③ 落地手册文档。

**Tech Stack:** Python 3.12 (conda env `py312`)、FAISS、BM25 (rank_bm25)、jieba、LangChain、pytest。

## Global Constraints

- **Python 环境**：必须用 conda env `py312`（3.12.13）。pytest 命令统一写 `conda run -n py312 python -m pytest ...`。**不要用裸 `python`**（系统默认是 3.13，没装 pytest）。
- **提交策略（覆盖 skill 默认）**：`project1-rag` 当前整体 git 未跟踪。**未经用户明确要求，不要 `git add`/`git commit`。** 计划里每个 Task 末尾的 commit 步骤默认跳过；以"✅ Task N 完成（未提交）"收尾。需要提交时用户会说。
- **外科手术式改动**：只改任务指定行，匹配现有代码风格，不动无关代码。
- **行为保持**：`config=None` 路径必须让现有 22 个测试全绿。唯一有意行为变化：向量召回 `top_k*2` → `vector_search_k`（默认 5），已批准。
- **测试复用现有文件** `tests/test_core.py`，新测试类追加到末尾，沿用其 `sys.path.insert` 约定。

## File Structure

| 文件 | 责任 | 本轮改动 |
|---|---|---|
| `config.py` | 全局配置（env 驱动） | 新增 2 阈值字段，`rerank_top_n` 标废弃 |
| `rag_modules/retrieval_optimization.py` | 混合检索 + RRF + rerank + 权限 | 接线 5 旋钮，加 `config` 入参与 `_resolve_config` |
| `main.py` | 系统主入口 | 构造检索模块时传 `config` |
| `evaluation/sweep_retrieval.py` | 检索层扫参（新建） | 全新文件 |
| `tests/test_core.py` | 单元测试 | 追加 `TestRetrievalWiring`、`TestSweepHitCriterion` |
| `docs/RAG调参落地手册.md` | 落地手册（新建） | 全新文件 |
| `README.md` | 项目说明 | docs 表加一行 |

---

### Task 1: config.py — 接入两个阈值，废弃 rerank_top_n

**Files:**
- Modify: `project1-rag/config.py:36-42`
- Test: `project1-rag/tests/test_core.py`（追加 `TestConfigThresholds`）

**Interfaces:**
- Produces: `RAGConfig.vector_score_threshold: float`、`RAGConfig.rerank_threshold: float`（默认均 0.3）；`rerank_top_n` 保留。

- [ ] **Step 1: Write the failing test**

在 `tests/test_core.py` 末尾追加：

```python
# ===== 8. config 阈值接线（Task1）=====
class TestConfigThresholds:
    """验证两个阈值进入 config，且 rerank_top_n 向后兼容保留"""

    def test_thresholds_exist_with_defaults(self):
        from config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG.vector_score_threshold == 0.3
        assert DEFAULT_CONFIG.rerank_threshold == 0.3

    def test_thresholds_env_overridable(self, monkeypatch):
        monkeypatch.setenv("RAG_VECTOR_SCORE_THRESHOLD", "0.45")
        monkeypatch.setenv("RAG_RERANK_THRESHOLD", "0.6")
        from config import RAGConfig
        cfg = RAGConfig()
        assert cfg.vector_score_threshold == 0.45
        assert cfg.rerank_threshold == 0.6

    def test_rerank_top_n_kept_for_backcompat(self):
        from config import DEFAULT_CONFIG
        assert hasattr(DEFAULT_CONFIG, "rerank_top_n")  # 字段仍在（不破坏旧读取）
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py312 python -m pytest tests/test_core.py::TestConfigThresholds -v`
Expected: FAIL — `AttributeError: 'RAGConfig' object has no attribute 'vector_score_threshold'`

- [ ] **Step 3: Implement — modify config.py**

把 `config.py` 的检索段（`# ===== ③ 检索 =====` 那一块）替换为：

```python
    # ===== ③ 检索 =====
    embedding_model: str = _env("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    top_k: int = int(_env("RAG_TOP_K", "3"))
    vector_search_k: int = int(_env("RAG_VECTOR_K", "5"))    # 向量召回宽度（融合前候选数）
    bm25_search_k: int = int(_env("RAG_BM25_K", "5"))        # BM25 召回宽度
    rrf_k: int = int(_env("RAG_RRF_K", "60"))                # RRF 平滑参数
    # rerank_top_n 已废弃：与 top_k 重叠（rerank 给所有候选打分后由 threshold+top_k 截断）。
    # 保留字段仅为向后兼容，检索热路径不再读取。调参请改 top_k。
    rerank_top_n: int = int(_env("RAG_RERANK_TOP_N", "3"))
    vector_score_threshold: float = float(_env("RAG_VECTOR_SCORE_THRESHOLD", "0.3"))  # 向量召回 cosine 阈值
    rerank_threshold: float = float(_env("RAG_RERANK_THRESHOLD", "0.3"))              # rerank 拒答阈值
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py312 python -m pytest tests/test_core.py::TestConfigThresholds -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit** — 默认跳过（见 Global Constraints）。✅ Task 1 完成（未提交）。

---

### Task 2: retrieval_optimization.py — 接线 5 个旋钮 + config 入参

**Files:**
- Modify: `project1-rag/rag_modules/retrieval_optimization.py`（`__init__` `:51-61`、`setup_retrievers` `:63-80`、`hybrid_search` 签名与召回 `:82-100`、`_rrf_rerank` `:225`）
- Test: `project1-rag/tests/test_core.py`（追加 `TestRetrievalWiring`）

**Interfaces:**
- Consumes: `RAGConfig.vector_search_k / bm25_search_k / rrf_k / vector_score_threshold / rerank_threshold`（Task 1）
- Produces: `RetrievalOptimizationModule(vectorstore, chunks, reranker=None, config=None)`；新增纯方法 `_resolve_config(config)`；`hybrid_search(query, top_k=3, score_threshold=None, rerank_threshold=None)`。

- [ ] **Step 1: Write the failing tests**

在 `tests/test_core.py` 末尾追加：

```python
# ===== 9. 检索接线（Task2：死旋钮生效）=====
class TestRetrievalWiring:
    """验证 config 的检索参数真正传到检索器（修复死旋钮）"""

    def test_resolve_config_uses_config_values(self):
        from types import SimpleNamespace
        from rag_modules.retrieval_optimization import RetrievalOptimizationModule as R
        ret = R.__new__(R)
        cfg = SimpleNamespace(vector_search_k=8, bm25_search_k=7, rrf_k=30,
                              vector_score_threshold=0.4, rerank_threshold=0.5)
        ret._resolve_config(cfg)
        assert ret.vector_search_k == 8
        assert ret.bm25_search_k == 7
        assert ret.rrf_k == 30
        assert ret.vector_score_threshold == 0.4
        assert ret.rerank_threshold == 0.5

    def test_resolve_config_none_uses_old_defaults(self):
        """config=None 时退回原硬编码默认（零破坏，现有调用方不受影响）"""
        from rag_modules.retrieval_optimization import RetrievalOptimizationModule as R
        ret = R.__new__(R)
        ret._resolve_config(None)
        assert ret.vector_search_k == 5
        assert ret.bm25_search_k == 5
        assert ret.rrf_k == 60
        assert ret.vector_score_threshold == 0.3
        assert ret.rerank_threshold == 0.3

    def test_bm25_k_wired_end_to_end(self):
        """端到端：config.bm25_search_k 真的传到 BM25Retriever（不是硬编码 5）"""
        from types import SimpleNamespace
        from langchain_core.documents import Document
        from rag_modules.retrieval_optimization import RetrievalOptimizationModule as R

        chunks = [Document(page_content="年假5天", metadata={"source": "HR/x.md", "chunk_id": "1"}),
                  Document(page_content="报销500元", metadata={"source": "财务/y.md", "chunk_id": "2"})]

        class FakeVS:
            def as_retriever(self, **kw):
                return None  # 本测试不使用向量检索器

        cfg = SimpleNamespace(vector_search_k=8, bm25_search_k=7, rrf_k=60,
                              vector_score_threshold=0.3, rerank_threshold=0.3)
        ret = R(FakeVS(), chunks, config=cfg)
        assert ret.bm25_retriever.k == 7  # 接线生效；修复前恒为 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py312 python -m pytest tests/test_core.py::TestRetrievalWiring -v`
Expected: FAIL — `_resolve_config` 不存在 / `__init__` 不接受 `config`。

- [ ] **Step 3: Implement — 改 `__init__` 并抽出 `_resolve_config`**

把 `retrieval_optimization.py:51-61` 的 `__init__` 替换为：

```python
    def __init__(self, vectorstore: FAISS, chunks: List[Document], reranker=None, config=None):
        """
        Args:
            vectorstore: 向量库
            chunks: 文档块
            reranker: 可选的 Reranker 实例（传入则启用精排）
            config: 可选 RAGConfig——传入则检索参数走 config（让 env 真正生效）；
                    不传则退回原硬编码默认（config=None 时行为同旧，现有调用方零破坏）。
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.reranker = reranker
        self._resolve_config(config)
        self.setup_retrievers()

    def _resolve_config(self, config):
        """解析检索参数：有 config 用 config，否则退回原硬编码默认（零破坏）。

        抽成独立方法便于单测（无需真实 vectorstore）。
        """
        self.vector_search_k = getattr(config, "vector_search_k", 5) if config else 5
        self.bm25_search_k = getattr(config, "bm25_search_k", 5) if config else 5
        self.rrf_k = getattr(config, "rrf_k", 60) if config else 60
        self.vector_score_threshold = getattr(config, "vector_score_threshold", 0.3) if config else 0.3
        self.rerank_threshold = getattr(config, "rerank_threshold", 0.3) if config else 0.3
```

- [ ] **Step 4: Wire `setup_retrievers`（用 self 值替硬编码 5）**

把 `setup_retrievers`（`:63-80`）里两处 `k` 改为 `self.*`：

```python
    def setup_retrievers(self):
        """设置向量检索器(语义) + BM25检索器(关键词，中文分词)"""
        logger.info("正在设置检索器（向量 + BM25中文分词）...")

        # 向量检索器（语义匹配）
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.vector_search_k}
        )

        # BM25 检索器（关键词匹配，关键：传 chinese_tokenizer 做分词）
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=self.bm25_search_k,
            preprocess_func=chinese_tokenizer,  # ⭐ 中文必须分词，否则 BM25 失效
        )

        logger.info("检索器设置完成（BM25 已启用 jieba 中文分词）")
```

- [ ] **Step 5: Wire `hybrid_search` 阈值与向量召回宽度**

把 `hybrid_search` 签名（`:82-83`）改为：

```python
    def hybrid_search(self, query: str, top_k: int = 3, score_threshold: float = None,
                      rerank_threshold: float = None) -> List[Document]:
```

在 `hybrid_search` 的 docstring 结束后、`# 阶段1` 注释前，插入阈值解析：

```python
        # 阈值默认从 self 解析（接线：让 config 生效；显式传入仍可覆盖）
        if score_threshold is None:
            score_threshold = self.vector_score_threshold
        if rerank_threshold is None:
            rerank_threshold = self.rerank_threshold
```

把向量召回那行（原 `:100`）：

```python
        raw = self.vectorstore.similarity_search_with_score(query, k=top_k * 2)
```

改为：

```python
        raw = self.vectorstore.similarity_search_with_score(query, k=self.vector_search_k)
```

- [ ] **Step 6: Wire `_rrf_rerank` 的 k**

把 `_rrf_rerank` 签名（`:225`）与函数首行改为：

```python
    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = None) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名；None 时用 self.rrf_k（接线 config）

        Returns:
            重排后的文档列表
        """
        k = self.rrf_k if k is None else k
```

（函数体内原本就用 `k` 变量算 `1.0 / (k + rank + 1)`，无需再改。）

- [ ] **Step 7: Run tests to verify pass**

Run: `conda run -n py312 python -m pytest tests/test_core.py::TestRetrievalWiring -v`
Expected: PASS (3 passed)

- [ ] **Step 8: Run full suite — 确认零回归**

Run: `conda run -n py312 python -m pytest tests/test_core.py -v`
Expected: PASS — 之前 25 个（22 + Task1 的 3）+ 本 Task 的 3 = **28 passed**。

- [ ] **Step 9: Commit** — 默认跳过。✅ Task 2 完成（未提交）。

---

### Task 3: main.py — 构造检索模块时传 config

**Files:**
- Modify: `project1-rag/main.py:95`

**Interfaces:**
- Consumes: Task 2 的 `RetrievalOptimizationModule(..., config=)`。
- Produces: 生产路径（`main.py`、后续 `app/api.py` 经 `rag.retrieval_module`）的检索参数走 env。

- [ ] **Step 1: Modify main.py:95**

把：

```python
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks, reranker=self._init_reranker())
```

改为：

```python
        self.retrieval_module = RetrievalOptimizationModule(
            vectorstore, chunks, reranker=self._init_reranker(), config=self.config)
```

- [ ] **Step 2: Verify — 全量测试仍绿（main.py 不易单测，靠现有套件兜底回归）**

Run: `conda run -n py312 python -m pytest tests/test_core.py -v`
Expected: PASS (28 passed)。

- [ ] **Step 3: Commit** — 默认跳过。✅ Task 3 完成（未提交）。

---

### Task 4: sweep_retrieval.py — 命中判定纯函数

**Files:**
- Create: `project1-rag/evaluation/sweep_retrieval.py`（先只写头部 + 3 个纯函数）
- Test: `project1-rag/tests/test_core.py`（追加 `TestSweepHitCriterion`）

**Interfaces:**
- Produces: `_gt_numbers(text) -> Set[str]`、`is_hit(chunk: Document, case: dict) -> bool`、`is_correct_refuse(docs: list, case: dict) -> bool`。

- [ ] **Step 1: Write the failing tests**

在 `tests/test_core.py` 末尾追加：

```python
# ===== 10. 扫参命中判定（Task4：answer-span 文件级匹配）=====
class TestSweepHitCriterion:
    """验证 is_hit 用判别数字区分同文件不同档位（1年/3年/10年），且无需 chunk 级标注"""

    @staticmethod
    def _chunk(content, source):
        from langchain_core.documents import Document
        return Document(page_content=content, metadata={"source": source})

    def test_3yr_chunk_hits_3yr_case_only(self):
        from evaluation.sweep_retrieval import is_hit
        case_1 = {"ground_truth": "工作满1年不满3年年假5天", "source_doc": "HR/年假管理制度.md"}
        case_3 = {"ground_truth": "工作满3年不满5年年假10天", "source_doc": "HR/年假管理制度.md"}
        chunk_3yr = self._chunk("工作满3年不满5年的，年假10天。", "HR/年假管理制度.md")
        assert is_hit(chunk_3yr, case_3) is True   # 数字 {3,5,10} ⊆ {3,5,10}
        assert is_hit(chunk_3yr, case_1) is False   # case_1 要 {1,3,5}，chunk 缺 1 → 不命中

    def test_wrong_file_not_hit(self):
        from evaluation.sweep_retrieval import is_hit
        case = {"ground_truth": "工作满3年年假10天", "source_doc": "HR/年假管理制度.md"}
        chunk = self._chunk("工作满3年年假10天", "财务/报销.md")
        assert is_hit(chunk, case) is False

    def test_no_number_gt_falls_back_to_file_level(self):
        from evaluation.sweep_retrieval import is_hit
        case = {"ground_truth": "OA系统提交申请主管审批", "source_doc": "HR/考勤管理制度.md"}
        chunk = self._chunk("请假通过OA系统提交", "HR/考勤管理制度.md")
        assert is_hit(chunk, case) is True

    def test_correct_refuse(self):
        from evaluation.sweep_retrieval import is_correct_refuse
        neg = {"source_doc": ""}
        assert is_correct_refuse([], neg) is True
        assert is_correct_refuse([object()], neg) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py312 python -m pytest tests/test_core.py::TestSweepHitCriterion -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evaluation.sweep_retrieval'`。

- [ ] **Step 3: Create sweep_retrieval.py（头部 + 3 纯函数）**

新建 `project1-rag/evaluation/sweep_retrieval.py`：

```python
"""
检索层扫参脚本（便宜循环，不跑 LLM、不跑 reranker）

对应《RAG参数调参方法论》第 4 节：沿管道因果序调参，检索层用 Recall@K/MRR 测，
不需要 LLM。本脚本扫 chunk_size × vector_search_k × bm25_search_k。

命中判定：answer-span 文件级匹配——复用黄金集已有的 ground_truth 的判别数字，
无需 chunk 级标注，且能区分 1年/3年/10年（同文件不同档位）。

用法：
  python evaluation/sweep_retrieval.py                         # 默认网格
  python evaluation/sweep_retrieval.py --chunks 300,500 --vk 5,8 --bk 5
"""

import re
import logging

logger = logging.getLogger(__name__)


def _gt_numbers(text: str):
    """从 ground_truth 提取所有数字（判别用，如 1年/3年/10年、500元、90天）"""
    return set(re.findall(r"\d+", text or ""))


def is_hit(chunk, case: dict) -> bool:
    """answer-span 文件级命中：必须来自正确文件 且 含 ground_truth 的全部判别数字。

    - 文件级：chunk.source 与 case.source_doc 匹配（任一包含另一）。
    - 判别数字：gt 的所有数字必须出现在 chunk 文本里（坑25 思想：数字不同=不同问题）。
    - gt 无数字（如流程类）→ 仅文件级命中。
    """
    src = chunk.metadata.get("source", "")
    gt_src = case.get("source_doc", "")
    # ① 文件级
    if gt_src and not (gt_src in src or src in gt_src):
        return False
    # ② 判别数字
    nums = _gt_numbers(case.get("ground_truth", ""))
    if nums:
        chunk_nums = set(re.findall(r"\d+", chunk.page_content))
        if not nums.issubset(chunk_nums):
            return False
    return True


def is_correct_refuse(docs, case: dict) -> bool:
    """负例正确拒答：知识库无答案的问题，检索无召回。"""
    return case.get("source_doc", "") == "" and len(docs) == 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py312 python -m pytest tests/test_core.py::TestSweepHitCriterion -v`
Expected: PASS (4 passed)。

> 若报 `ModuleNotFoundError: No module named 'evaluation'`：在 `evaluation/` 下新建空 `__init__.py`（PEP 420 命名空间包本不需要，但部分环境/旧 pytest 需要显式标记）。先不加，失败再加。

- [ ] **Step 5: Commit** — 默认跳过。✅ Task 4 完成（未提交）。

---

### Task 5: sweep_retrieval.py — 指标计算 + 主循环 + CLI

**Files:**
- Modify: `project1-rag/evaluation/sweep_retrieval.py`（追加 `eval_one_config` + `main` + 入口）
- Test: `project1-rag/tests/test_core.py`（追加 `TestSweepMetrics`，用 fake retriever）

**Interfaces:**
- Consumes: Task 2 的 `RetrievalOptimizationModule(..., config=)`、`DataPreparationModule`、`IndexConstructionModule`、`RAGConfig.from_dict`、`evaluation/eval_dataset.jsonl`、Task 4 的 `is_hit/is_correct_refuse`。
- Produces: `eval_one_config(ret, eval_set, top_k) -> dict`、`main()`、CLI；写出 `evaluation/sweep_result.json`。

- [ ] **Step 1: Write the failing test for eval_one_config**

在 `tests/test_core.py` 末尾追加：

```python
# ===== 11. 扫参指标计算（Task5：fake retriever 单测）=====
class TestSweepMetrics:
    """eval_one_config 用 fake retriever 验证 Recall@K/MRR/neg_refuse 计算"""

    def test_metrics_from_known_mapping(self):
        from evaluation.sweep_retrieval import eval_one_config
        from langchain_core.documents import Document

        def chunk(content, source):
            return Document(page_content=content, metadata={"source": source})

        # 3 正例（同文件，靠数字区分）+ 1 负例
        eval_set = [
            {"question": "q3", "ground_truth": "10天", "source_doc": "HR/x.md"},
            {"question": "q5", "ground_truth": "500元", "source_doc": "财务/y.md"},
            {"question": "qmiss", "ground_truth": "999", "source_doc": "IT/z.md"},  # 召回错误内容
            {"question": "qneg", "ground_truth": "", "source_doc": ""},             # 负例
        ]
        mapping = {
            "q3": [chunk("年假10天", "HR/x.md")],          # rank1 命中
            "q5": [chunk("无关", "财务/y.md"), chunk("报销500元", "财务/y.md")],  # rank2 命中
            "qmiss": [chunk("别的9999", "IT/z.md")],        # 数字 999 不在 → 未命中
            "qneg": [chunk("xxx", "其他/w.md")],            # 有召回 → 负例拒答失败
        }

        class FakeRet:
            def hybrid_search(self, q, top_k=3):
                return mapping.get(q, [])[:top_k]

        m = eval_one_config(FakeRet(), eval_set, top_k=3)
        # 正例 3 个，命中 2 个 → recall = 2/3
        assert m["recall"] == round(2 / 3, 4)
        # MRR：q3 rank1=1.0，q5 rank2=0.5，qmiss=0 → (1.0+0.5+0)/3
        assert m["mrr"] == round((1.0 + 0.5 + 0.0) / 3, 4)
        # 负例 1 个，拒答失败 → neg_refuse = 0/1 = 0
        assert m["neg_refuse"] == 0.0
        assert m["overall"] == round(2 / 3, 4) * 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py312 python -m pytest tests/test_core.py::TestSweepMetrics -v`
Expected: FAIL — `ImportError: cannot import name 'eval_one_config'`。

- [ ] **Step 3: Append eval_one_config + main + CLI to sweep_retrieval.py**

在 `evaluation/sweep_retrieval.py` 末尾追加：

```python
import argparse
import json
import sys
from pathlib import Path
from itertools import product
from typing import List, Dict


EVAL_FILE = Path(__file__).parent / "eval_dataset.jsonl"


def load_eval_set() -> List[Dict]:
    data = []
    with open(EVAL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def eval_one_config(ret, eval_set: List[Dict], top_k: int = 3) -> Dict:
    """对一个配置跑全部评测样例，算 Recall@K / MRR / neg_refuse / overall。

    ret 只需有 hybrid_search(query, top_k) 方法（鸭子类型，便于用 fake 单测）。
    不跑 LLM、不跑 reranker——本函数只测向量+BM25+RRF 召回层。
    """
    pos_total = pos_hit = 0
    neg_total = neg_correct = 0
    reciprocal_ranks: List[float] = []

    for case in eval_set:
        docs = ret.hybrid_search(case["question"], top_k=top_k)
        if case.get("source_doc"):
            pos_total += 1
            hit_rank = None
            for rank, d in enumerate(docs, 1):
                if is_hit(d, case):
                    hit_rank = rank
                    break
            if hit_rank is not None:
                pos_hit += 1
                reciprocal_ranks.append(1.0 / hit_rank)
            else:
                reciprocal_ranks.append(0.0)
        else:
            neg_total += 1
            if is_correct_refuse(docs, case):
                neg_correct += 1

    recall = pos_hit / pos_total if pos_total else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    neg_refuse = neg_correct / neg_total if neg_total else 1.0
    overall = recall * neg_refuse
    return {
        "recall": round(recall, 4),
        "mrr": round(mrr, 4),
        "neg_refuse": round(neg_refuse, 4),
        "overall": round(overall, 4),
        "pos": f"{pos_hit}/{pos_total}",
        "neg": f"{neg_correct}/{neg_total}",
    }


def main():
    parser = argparse.ArgumentParser(description="RAG 检索层扫参（不跑 LLM）")
    parser.add_argument("--chunks", default="200,350,500,700", help="chunk_size 网格，逗号分隔")
    parser.add_argument("--vk", default="3,5,8", help="vector_search_k 网格")
    parser.add_argument("--bk", default="3,5,8", help="bm25_search_k 网格")
    parser.add_argument("--top-k", type=int, default=3, help="最终 top_k（固定，不扫）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    chunk_grid = [int(x) for x in args.chunks.split(",")]
    vk_grid = [int(x) for x in args.vk.split(",")]
    bk_grid = [int(x) for x in args.bk.split(",")]

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DEFAULT_CONFIG, RAGConfig
    from rag_modules import DataPreparationModule, IndexConstructionModule, RetrievalOptimizationModule

    cfg = DEFAULT_CONFIG
    eval_set = load_eval_set()
    print(f"评测集 {len(eval_set)} 条；网格 chunk{chunk_grid} × vk{vk_grid} × bk{bk_grid} "
          f"= {len(chunk_grid)*len(vk_grid)*len(bk_grid)} 组")

    # 嵌入模型只加载一次（最贵），后续 build_vector_index 复用 idx.embeddings
    idx = IndexConstructionModule(cfg.embedding_model, cfg.index_save_path)

    results = []
    for chunk_size in chunk_grid:
        dp = DataPreparationModule(cfg.data_path, chunk_size, cfg.chunk_overlap)
        dp.load_documents()
        chunks = dp.chunk_documents()
        vs = idx.build_vector_index(chunks)  # 内存索引，不落盘，不污染生产索引
        for vk, bk in product(vk_grid, bk_grid):
            sweep_cfg = RAGConfig.from_dict({**cfg.to_dict(),
                                             "vector_search_k": vk,
                                             "bm25_search_k": bk})
            ret = RetrievalOptimizationModule(vs, chunks, config=sweep_cfg)
            m = eval_one_config(ret, eval_set, args.top_k)
            row = {"chunk_size": chunk_size, "vector_k": vk, "bm25_k": bk, **m}
            results.append(row)
            print(f"  chunk={chunk_size} vk={vk} bk={bk} → "
                  f"recall={m['recall']} mrr={m['mrr']} neg_refuse={m['neg_refuse']} overall={m['overall']} "
                  f"({m['pos']} / {m['neg']})")

    results.sort(key=lambda r: r["overall"], reverse=True)
    out = Path(__file__).parent / "sweep_result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"🏆 Top 5 by overall（共 {len(results)} 组，明细 {out}）:")
    for r in results[:5]:
        print(f"  chunk={r['chunk_size']} vk={r['vector_k']} bk={r['bm25_k']} → "
              f"overall={r['overall']} recall={r['recall']} neg_refuse={r['neg_refuse']}")
    print("=" * 70)
    print("提示：v1 未接 reranker，neg_refuse 仅反映向量阈值过滤；读结果见《RAG调参落地手册》")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit test to verify it passes**

Run: `conda run -n py312 python -m pytest tests/test_core.py::TestSweepMetrics -v`
Expected: PASS (1 passed)。

- [ ] **Step 5: Smoke run — 1×1×1 网格跑通（加载嵌入模型，约 10–30 秒）**

Run: `conda run -n py312 python evaluation/sweep_retrieval.py --chunks 500 --vk 5 --bk 5`
Expected: 打印 1 组结果行 + Top 5（只有 1 组）+ 生成 `evaluation/sweep_result.json`，无异常退出。

- [ ] **Step 6: Run full suite — 确认零回归**

Run: `conda run -n py312 python -m pytest tests/test_core.py -v`
Expected: PASS — 累计 28 + 4（Task4）+ 1（Task5）= **33 passed**。

- [ ] **Step 7: Commit** — 默认跳过。✅ Task 5 完成（未提交）。

---

### Task 6: 落地手册 + README 登记

**Files:**
- Create: `project1-rag/docs/RAG调参落地手册.md`
- Modify: `project1-rag/README.md`（docs 表加一行）

**Interfaces:** 无代码接口；纯文档。

- [ ] **Step 1: 写 `docs/RAG调参落地手册.md`**

内容结构（每节都要写实，不要占位）：

1. **一页纸：接线改了什么** — 列出 Task 1–3 的 diff 要点（新增 2 阈值、接线 vector_k/bm25_k/rrf_k、废弃 rerank_top_n、`vector_search_k` 的 6→5 有意变化），并解释「为什么调参前必须先接线」。
2. **跑扫参** — 环境要求（conda py312）、命令（默认网格 + 自定义 `--chunks/--vk/--bk`）、输出文件 `sweep_result.json`、结果表怎么读（overall 最高、Recall 平台点/knee）。
3. **怎么选点** — 三种业务约束（准确率地板 / 延迟预算 / 成本预算）对应的选法；附诊断决策表（指标低→拧哪个旋钮）。
4. **移植到别的公司（核心）** — 4 步：① 换黄金集（自带 `question/ground_truth/source_doc` + 负例）；② 换网格范围（按自己文档长度分布定 chunk_size 上下界）；③ 重跑 `sweep_retrieval.py`；④ 读同样的指标选点。强调「流程不变，只换数据和范围」。
5. **结果解读注意** — v1 不接 reranker，neg_refuse 偏低属正常（量化 rerank 价值）；answer-span 匹配的局限（gt 无数字退化为文件级）。
6. **进阶（指向后续）** — 把 reranker 接进扫参、生成层 top_k 扫参（要 LLM）、大语料按 content_hash 缓存 embedding。
7. **面试话术** — 3–5 句背诵要点。

- [ ] **Step 2: README docs 表加一行**

在 `README.md` 现有 docs 表（`docs/RAG参数调参方法论.md` 那行之后）追加：

```markdown
| `docs/RAG调参落地手册.md` | 接线 diff + 扫参脚本用法 + 换公司移植 4 步 + 诊断决策表 |
```

- [ ] **Step 3: Commit** — 默认跳过。✅ Task 6 完成（未提交）。

---

## Self-Review（计划完成后自检）

**Spec 覆盖**：
- §4 接线（config 2 阈值 + rerank_top_n 废弃）→ Task 1 ✓
- §4 retrieval_optimization 接线（config 入参 + 5 旋钮 + _resolve_config）→ Task 2 ✓
- §4.3 main.py 传 config → Task 3 ✓
- §5 扫参脚本（命中判定/指标/网格/输出/性能）→ Task 4+5 ✓
- §7 测试（TestRetrievalWiring + 命中判定 + 冒烟）→ Task 2/4/5 ✓
- §6 落地手册 → Task 6 ✓
- §9 交付物 1–7 全覆盖 ✓

**占位符扫描**：无 TBD/TODO；每个代码步骤都给了完整代码。✓

**类型/命名一致性**：`_resolve_config`、`is_hit`、`is_correct_refuse`、`eval_one_config`、`_gt_numbers` 在定义任务与测试任务中拼写一致；`hybrid_search(query, top_k, score_threshold=None, rerank_threshold=None)` 签名贯穿。✓
