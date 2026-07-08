# RBAC 按部门子索引（真·先过滤）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 RBAC 检索的向量路径从「全库 `top_k*2` 再 Python 过滤」（先检索后过滤，A1 缺陷）改成「只在权限部门子索引里搜」（真·先过滤后检索），admin/未登录路径完全不变。

**Architecture:** 外科手术式——admin/未登录仍走全库 `hybrid_search`（全局 cosine 排序最优，不动）；普通用户改走按部门子索引。子索引从全库 FAISS 向量 `reconstruct_n` **切片**而来，零重复嵌入。新增 `IndexConstructionModule.build_department_indexes` 产 `Dict[部门, FAISS]`；`RetrievalOptimizationModule` 增 `dept_indexes` 参 + `_rbac_subindex_search` + `permission_aware_search` 路由；`main.py` 透传。

**Tech Stack:** FAISS（`IndexFlatIP`、`reconstruct_n`）、LangChain（`FAISS`/`InMemoryDocstore`/`BM25Retriever`）、jieba、pytest、conda env `py312`。

## Global Constraints

- **pytest 必须用** `conda run -n py312 python -m pytest ...`（裸 `python` 是 3.13，无 pytest）。
- **Windows 中文输出**报 `UnicodeEncodeError`（GBK 终端）时用：`PYTHONUTF8=1 PYTHONIOENCODING=utf-8 conda run --no-capture-output -n py312 python -m pytest ...`。
- **不提交**：未经用户明确要求，**不做 `git add`/`commit`**（用户既定规则）。每个 Task 以「跑全量测试确认绿」收尾，**替代** commit 步骤。
- **外科手术**：只动 `rag_modules/index_construction.py`、`rag_modules/retrieval_optimization.py`、`main.py`、新测试文件 `tests/test_rbac_subindex.py`。不重构相邻代码，不改 admin/未登录的 `hybrid_search`。
- **零破坏契约**：`dept_indexes=None` 退回旧 `metadata_filtered_search`；`config=None` 退回硬编码默认（延续既有接线模式）。构造函数新增参数均有默认值，现有 32 条测试不破。
- **测试不加载真 bge**：用确定性 fake embeddings + 真 FAISS（测切片），或鸭子类型 fake（测 RBAC 路由），保持套件快（既有 32 测试同样避免重模型）。
- **测试文件**：新建 `project1-rag/tests/test_rbac_subindex.py`（隔离新测试与共享 helper，避免 test_core.py 继续膨胀）。
- **与 spec 的一处细化**：spec §2/§10 把 Bug 复现测试写成「旧路径返空」。实测中 BM25 路径会对权限子集返回结果，可能掩盖「向量侧返空」。故本计划的 A1 证明测试改用更鲁棒的判定——**「新路径向量检索不碰全库 `vectorstore`（`as_retriever` 无额外调用）且能召回权限内文档」+ 对照「旧路径确实搜了全库」**。意图（先过滤胜过后过滤）与 spec 一致，仅判定方式更稳。

---

## File Structure

| 文件 | 责任 | 改动 |
|---|---|---|
| `rag_modules/index_construction.py` | 索引构建；新增部门子索引切片 | 新增模块级 `_slice_department_indexes` + 类方法 `build_department_indexes`；`Dict` 加入 typing import |
| `rag_modules/retrieval_optimization.py` | 检索；RBAC 路径接子索引 | `__init__` 增 `dept_indexes` 参；新增 `_rbac_subindex_search`；`permission_aware_search` 加路由 |
| `main.py` | 启动装配 | `build_knowledge_base` 构建并透传 `dept_indexes` |
| `tests/test_rbac_subindex.py`（新建） | 切片器 + RBAC 路由的测试与共享 fake | 全部新增 |

---

## Task 1: 部门子索引切片器（零重复嵌入）

**Files:**
- Modify: `project1-rag/rag_modules/index_construction.py`（顶部 `from typing import List` 改为 `from typing import List, Dict`；文件末尾新增模块级 `_slice_department_indexes`；类内新增方法 `build_department_indexes`）
- Test: `project1-rag/tests/test_rbac_subindex.py`（新建）

**Interfaces:**
- Produces（供后续 Task 用）:
  - 模块级 `_slice_department_indexes(vectorstore, embeddings) -> Dict[str, FAISS]`
  - `IndexConstructionModule.build_department_indexes(self, vectorstore) -> Dict[str, FAISS]`
- Consumes: 全库 `vectorstore`（LangChain FAISS，其 docstore 内 `Document` 带 `metadata["department"]`，由 `data_preparation._enhance_metadata` 保证）；`self.embeddings`（bge 单例，**仅作为子库 `embedding_function` 存起来，本函数不调用它**）。

- [ ] **Step 1: 新建测试文件，写 3 个失败测试 + 共享 fake**

创建 `project1-rag/tests/test_rbac_subindex.py`：

```python
"""RBAC 按部门子索引测试。

不加载真 bge：切片测试用确定性 fake embeddings + 真 FAISS；
RBAC 路由测试用鸭子类型 fake（见 Task 2 追加）。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class _FakeEmb(Embeddings):
    """确定性 fake 嵌入（仅供测试）。基于 sha256 的固定 8 维归一化向量。"""
    dim = 8

    @staticmethod
    def _vec(text):
        b = hashlib.sha256(text.encode("utf-8")).digest()
        vals = [(b[i % len(b)] / 255.0 - 0.5) for i in range(_FakeEmb.dim)]
        norm = sum(v * v for v in vals) ** 0.5 or 1.0
        return [v / norm for v in vals]

    def embed_query(self, text):
        return self._vec(text)

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]


def _make_chunks(dept_docs):
    """{部门: [文本,...]} → List[Document]（带 department/chunk_id/source 元数据）。"""
    out = []
    for dept, texts in dept_docs.items():
        for i, t in enumerate(texts):
            out.append(Document(page_content=t,
                                metadata={"department": dept,
                                          "chunk_id": f"{dept}{i}",
                                          "source": f"{dept}/d{i}"}))
    return out


def _make_vectorstore(chunks):
    """用 fake emb 建一个真 FAISS 全库（默认 IndexFlatL2，支持 reconstruct_n）。"""
    from langchain_community.vectorstores import FAISS
    return FAISS.from_documents(chunks, _FakeEmb())


# ===== Task 1：切片器 =====
from rag_modules.index_construction import _slice_department_indexes  # noqa: E402


def test_slice_partitions_by_department():
    chunks = _make_chunks({"HR": ["年假5天"], "财务": ["报销1000"], "公共": ["公司地址"]})
    vs = _make_vectorstore(chunks)
    result = _slice_department_indexes(vs, _FakeEmb())
    assert set(result.keys()) == {"HR", "财务", "公共"}
    for dept, sub in result.items():
        docs = list(sub.docstore._dict.values())
        assert all(d.metadata["department"] == dept for d in docs), f"{dept} 子库混入其他部门"


def test_slice_vectors_match_full_index():
    """零重复嵌入证明：子库向量 == 全库向量（直接搬运，没重算 embedding）。"""
    import numpy as np
    chunks = _make_chunks({"HR": ["a"], "财务": ["b", "c"]})
    vs = _make_vectorstore(chunks)
    result = _slice_department_indexes(vs, _FakeEmb())

    full_vecs = vs.index.reconstruct_n(0, vs.index.ntotal)
    pos_to_id = vs.index_to_docstore_id          # {position: docstore_id}
    id2doc = vs.docstore._dict                    # {docstore_id: Document}
    fin_positions = sorted(p for p, did in pos_to_id.items()
                           if id2doc[did].metadata["department"] == "财务")
    sub_fin = result["财务"].index.reconstruct_n(0, result["财务"].index.ntotal)
    expected = np.array([full_vecs[p] for p in fin_positions])
    assert np.allclose(sub_fin, expected), "子库向量与全库不一致（疑似重新 embed 了）"


def test_slice_no_cross_department_leak():
    chunks = _make_chunks({"HR": ["年假"], "财务": ["报销"]})
    vs = _make_vectorstore(chunks)
    result = _slice_department_indexes(vs, _FakeEmb())
    hr_results = result["HR"].similarity_search("anything", k=10)
    assert len(hr_results) > 0
    assert all(d.metadata["department"] == "HR" for d in hr_results), "HR 子库泄露了其他部门"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n py312 python -m pytest project1-rag/tests/test_rbac_subindex.py -v`
Expected: FAIL，`ImportError: cannot import name '_slice_department_indexes'`（函数还没实现）。

- [ ] **Step 3: 实现切片器 + 包装方法**

在 `project1-rag/rag_modules/index_construction.py`：

(a) 顶部 typing import 改为：
```python
from typing import List, Dict
```

(b) 文件末尾（`IndexConstructionModule` 类之外）新增模块级函数：
```python
def _slice_department_indexes(vectorstore, embeddings):
    """从全库 FAISS 切片出按部门（含公共）的子索引，零重复嵌入。

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

    index = vectorstore.index
    n = index.ntotal
    if n == 0:
        return {}

    dim = index.d
    vectors = index.reconstruct_n(0, n)            # (n, dim) ndarray
    pos_to_id = vectorstore.index_to_docstore_id   # {position: docstore_id}
    all_ids = [pos_to_id[pos] for pos in range(n)]
    all_docs = vectorstore.docstore.search(all_ids)  # 全已知 → 长度 n、保序
    assert len(all_docs) == n, "docstore 缺文档，无法对齐"

    DocstoreCls = type(vectorstore.docstore)

    by_dept = defaultdict(list)  # 部门 -> [position, ...]（range(n) 保证 position 升序）
    for pos in range(n):
        dept = all_docs[pos].metadata.get("department", "公共")
        by_dept[dept].append(pos)

    result = {}
    for dept, positions in by_dept.items():
        ds_ids = [all_ids[p] for p in positions]
        docs = [all_docs[p] for p in positions]

        sub_index = faiss.IndexFlatIP(dim)
        sub_index.add(vectors[positions])          # 顺序与 ds_ids/docs 对齐

        sub_docstore = DocstoreCls(dict(zip(ds_ids, docs)))
        sub_id_map = {new_pos: sid for new_pos, sid in enumerate(ds_ids)}
        result[dept] = FAISS(embeddings, sub_index, sub_docstore, sub_id_map)

    return result
```

(c) 在 `IndexConstructionModule` 类内（建议放 `similarity_search` 方法后）新增方法：
```python
    def build_department_indexes(self, vectorstore) -> Dict[str, "FAISS"]:
        """构建按部门（含公共）的子索引，供 RBAC 真·先过滤检索用。

        从全库 vectorstore 切片，零重复嵌入（详见 _slice_department_indexes）。
        复用 self.embeddings（已加载的 bge 单例）作为子库 embedding_function。

        Returns:
            Dict[部门名, FAISS]；无文档的部门不在其中。
        """
        logger.info("正在构建按部门子索引（零重复嵌入，从全库切片）...")
        result = _slice_department_indexes(vectorstore, self.embeddings)
        logger.info(f"部门子索引构建完成: {len(result)} 个部门 → {list(result.keys())}")
        return result
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n py312 python -m pytest project1-rag/tests/test_rbac_subindex.py -v`
Expected: PASS（3 条）。若 `docstore.search` 行为异常（个别版本过滤未知 id），改用 `vectorstore.docstore._dict[pos_to_id[pos]]` 直接取（全已知等价）。

- [ ] **Step 5: 跑全量测试确认无回归**

Run: `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 conda run --no-capture-output -n py312 python -m pytest project1-rag/tests/ -v`
Expected: PASS（既有 32 条 + 新增 3 条 = 35 条）。**不提交。**

---

## Task 2: RetrievalOptimizationModule 接子索引（RBAC 真·先过滤）

**Files:**
- Modify: `project1-rag/rag_modules/retrieval_optimization.py`（`__init__` 增 `dept_indexes` 参；新增 `_rbac_subindex_search`；`permission_aware_search` 加路由分支）
- Test: `project1-rag/tests/test_rbac_subindex.py`（追加 fake helper + 5 条测试）

**Interfaces:**
- Consumes（来自 Task 1）: `dept_indexes: Dict[str, FAISS]`；既有 config 字段 `vector_search_k`/`bm25_search_k`/`vector_score_threshold`/`rrf_k`（由 `_resolve_config` 设置）。
- Produces: `_rbac_subindex_search(self, query, allowed_depts, top_k) -> List[Document]`；`permission_aware_search` 在 `dept_indexes` 非空时改走该方法。

- [ ] **Step 1: 追加 fake helper + 5 个失败测试**

在 `project1-rag/tests/test_rbac_subindex.py` 末尾追加：

```python
# ===== Task 2：RBAC 路由 =====
import pytest
from rag_modules.retrieval_optimization import RetrievalOptimizationModule


class _Cfg:
    """最小 config（仅检索字段），避免 RAGConfig 的 .env 耦合。"""
    vector_search_k = 5
    bm25_search_k = 5
    rrf_k = 60
    vector_score_threshold = 0.0   # 测试不卡阈值
    rerank_threshold = 0.3


class _FakeSub:
    """鸭子类型子索引：similarity_search_with_score(query, k) -> [(doc, sim), ...]。"""
    def __init__(self, scored):
        self._scored = scored

    def similarity_search_with_score(self, query, k):
        return list(self._scored)[:k]


class _FakeVS:
    """鸭子类型全库 vectorstore：as_retriever 返回固定 docs，统计调用次数。"""
    def __init__(self, invoke_returns=None):
        self._invoke_returns = invoke_returns or []
        self.as_retriever_calls = 0

    def as_retriever(self, **kwargs):
        self.as_retriever_calls += 1
        vs = self

        class _R:
            def invoke(_self, q):
                return list(vs._invoke_returns)

        return _R()


def _build_ret(dept_indexes=None, fake_vs=None, chunks=None):
    return RetrievalOptimizationModule(
        fake_vs or _FakeVS(), chunks or _make_chunks({"HR": ["x"]}),
        config=_Cfg(), dept_indexes=dept_indexes)


def test_rbac_subindex_unions_allowed_and_public_no_leak():
    hr = Document("年假", metadata={"department": "HR", "chunk_id": "h1", "source": "HR/a"})
    pub = Document("地址", metadata={"department": "公共", "chunk_id": "p1", "source": "公共/b"})
    fin = Document("报销", metadata={"department": "财务", "chunk_id": "f1", "source": "财务/c"})
    dept_indexes = {
        "HR": _FakeSub([(hr, 0.9)]),
        "公共": _FakeSub([(pub, 0.8)]),
        "财务": _FakeSub([(fin, 0.95)]),   # 不在 allowed，不应出现
    }
    ret = _build_ret(dept_indexes=dept_indexes, chunks=[hr, pub, fin])
    results = ret.permission_aware_search("年假", ["HR"], top_k=3)
    ids = {d.metadata["chunk_id"] for d in results}
    assert "f1" not in ids        # 无财务泄露
    assert "h1" in ids            # HR 命中
    assert "p1" in ids            # 公共命中


def test_rbac_path_does_not_search_full_index():
    """A1 修复证明：dept_indexes 在时，RBAC 向量检索走子索引，不碰全库 vectorstore。
    全库 rigged 只返回财务，但 HR 用户仍拿到 HR 文档——因为向量根本没搜全库。"""
    hr = Document("年假5天", metadata={"department": "HR", "chunk_id": "h1", "source": "HR/a"})
    fin = Document("报销", metadata={"department": "财务", "chunk_id": "f1", "source": "财务/c"})
    rigged_vs = _FakeVS(invoke_returns=[fin] * 6)   # 模拟 HR chunk 被财务挤出
    dept_indexes = {"HR": _FakeSub([(hr, 0.9)]), "公共": _FakeSub([])}
    ret = _build_ret(dept_indexes=dept_indexes, fake_vs=rigged_vs,
                     chunks=[hr] + [fin] * 6)
    calls_after_init = rigged_vs.as_retriever_calls   # __init__ 的 setup_retrievers 已调过 1 次
    results = ret.permission_aware_search("年假", ["HR"], top_k=3)
    assert rigged_vs.as_retriever_calls == calls_after_init   # 没额外调用 → 没搜全库
    assert any(d.metadata["chunk_id"] == "h1" for d in results)  # HR 文档被召回


def test_rbac_path_searches_full_index_when_no_dept_indexes():
    """对照（A1 病灶）：dept_indexes=None（旧路径）会搜全库 vectorstore。"""
    fin = Document("报销", metadata={"department": "财务", "chunk_id": "f1", "source": "财务/c"})
    rigged_vs = _FakeVS(invoke_returns=[fin])
    ret = _build_ret(dept_indexes=None, fake_vs=rigged_vs, chunks=[fin])
    ret.permission_aware_search("报销", ["财务"], top_k=3)
    assert rigged_vs.as_retriever_calls >= 2   # __init__ 1 次 + metadata_filtered_search 又搜 1 次


def test_fallback_uses_metadata_filtered_search_when_no_dept_indexes(monkeypatch):
    """dept_indexes=None → 走旧 metadata_filtered_search（零破坏回退）。"""
    called = {"meta": False}
    ret = _build_ret(dept_indexes=None, chunks=_make_chunks({"HR": ["x"]}))

    def _spy(query, filters, top_k):
        called["meta"] = True
        return []

    monkeypatch.setattr(ret, "metadata_filtered_search", _spy)
    ret.permission_aware_search("x", ["HR"], top_k=3)
    assert called["meta"] is True


def test_admin_star_uses_hybrid_search(monkeypatch):
    """admin '*' → hybrid_search 全库，不碰子索引/旧过滤。"""
    called = {"hybrid": False, "meta": False}
    ret = _build_ret(dept_indexes={"HR": _FakeSub([])}, chunks=_make_chunks({"HR": ["x"]}))
    monkeypatch.setattr(ret, "hybrid_search",
                        lambda q, tk: called.__setitem__("hybrid", True) or [])
    monkeypatch.setattr(ret, "metadata_filtered_search",
                        lambda q, f, tk: called.__setitem__("meta", True) or [])
    ret.permission_aware_search("x", ["*"], top_k=3)
    assert called["hybrid"] is True
    assert called["meta"] is False


def test_multi_dept_union():
    hr = Document("a", metadata={"department": "HR", "chunk_id": "h1", "source": "HR/a"})
    fin = Document("b", metadata={"department": "财务", "chunk_id": "f1", "source": "财务/b"})
    pub = Document("c", metadata={"department": "公共", "chunk_id": "p1", "source": "公共/c"})
    dept_indexes = {"HR": _FakeSub([(hr, 0.9)]),
                    "财务": _FakeSub([(fin, 0.85)]),
                    "公共": _FakeSub([(pub, 0.7)])}
    ret = _build_ret(dept_indexes=dept_indexes, chunks=[hr, fin, pub])
    results = ret.permission_aware_search("q", ["HR", "财务"], top_k=5)
    ids = {d.metadata["chunk_id"] for d in results}
    assert {"h1", "f1", "p1"} <= ids   # HR + 财务 + 公共 都在
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n py312 python -m pytest project1-rag/tests/test_rbac_subindex.py -v`
Expected: 6 条 FAIL（`_rbac_subindex_search` 不存在；`dept_indexes` 参不存在）。

- [ ] **Step 3: 实现 `dept_indexes` 参 + `_rbac_subindex_search` + 路由**

在 `project1-rag/rag_modules/retrieval_optimization.py`：

(a) `__init__` 增 `dept_indexes` 参（在 `reranker` 之后、`config` 之前插入，保持 `config` 位置不变以减少 diff）：
```python
    def __init__(self, vectorstore: FAISS, chunks: List[Document], reranker=None,
                 config=None, dept_indexes=None):
        """
        Args:
            vectorstore: 向量库
            chunks: 文档块
            reranker: 可选的 Reranker 实例（传入则启用精排）
            config: 可选 RAGConfig——传入则检索参数走 config（让 env 真正生效）；
                    不传则退回原硬编码默认（config=None 时行为同旧，现有调用方零破坏）。
            dept_indexes: 可选 Dict[部门, FAISS]——传入则 RBAC 路径走真·先过滤（部门子索引）；
                    不传则退回旧 metadata_filtered_search（全库后过滤）。默认 None 零破坏。
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.reranker = reranker
        self.dept_indexes = dept_indexes
        self._resolve_config(config)
        self.setup_retrievers()
```

(b) 新增 `_rbac_subindex_search` 方法（建议放在 `metadata_filtered_search` 之后、`permission_aware_search` 之前）：
```python
    def _rbac_subindex_search(self, query: str, allowed_depts: List[str], top_k: int = 3) -> List[Document]:
        """真·先过滤：只在 allowed_depts（已含公共）的部门子索引里搜。

        向量侧：各子索引 similarity_search_with_score(vector_search_k) → 按 cosine 全局排序
                取 top-vector_search_k（与 hybrid_search 向量候选池等规模，保 RRF 平衡）
                → cosine>=vector_score_threshold 过滤。
        BM25 侧：在 allowed_chunks 重建（与现 metadata_filtered_search 一致，本就是先过滤）。
        尾部：_rrf_rerank → [:top_k]（与现 RBAC 路径一致，不含 reranker——见 spec §11 顺带发现）。
        """
        allowed = set(allowed_depts)

        # 向量侧：union 各子索引，按 cosine 全局排序
        collected = []
        for dept in allowed_depts:
            sub = (self.dept_indexes or {}).get(dept)
            if sub is None:
                continue
            for doc, sim in sub.similarity_search_with_score(query, k=self.vector_search_k):
                doc.metadata["vector_sim"] = round(float(sim), 4)
                collected.append(doc)
        collected.sort(key=lambda d: d.metadata["vector_sim"], reverse=True)
        vector_docs = collected[: self.vector_search_k]
        vector_docs = [d for d in vector_docs if d.metadata["vector_sim"] >= self.vector_score_threshold]

        if not vector_docs:
            logger.info(f"  [RBAC子索引] allowed={allowed_depts} 向量路无候选 → 拒答")
            return []

        # BM25 侧：在权限子集重建（先过滤）
        allowed_chunks = [d for d in self.chunks if d.metadata.get("department") in allowed]
        if allowed_chunks:
            tmp_bm25 = BM25Retriever.from_documents(
                allowed_chunks, k=self.bm25_search_k, preprocess_func=chinese_tokenizer)
            bm25_docs = tmp_bm25.invoke(query)
        else:
            bm25_docs = []

        candidates = self._rrf_rerank(vector_docs, bm25_docs)
        final = candidates[:top_k]
        logger.info(f"  [RBAC子索引] allowed={allowed_depts} 向量{len(vector_docs)}+BM25{len(bm25_docs)} "
                    f"→ RRF{len(candidates)} → 返回{len(final)}")
        return final
```

(c) `permission_aware_search` 加路由分支。把方法体改为（保留 admin 分支与 docstring 风格，仅普通用户分支加 `dept_indexes` 判断）：
```python
    def permission_aware_search(self, query: str, user_departments: List[str], top_k: int = 3) -> List[Document]:
        """
        权限感知检索（RBAC）：用户只能看到自己有权限的部门文档。

        admin '*' / 未登录 → 全库 hybrid_search（真·全局 cosine 排序，最优）；
        普通用户 → 部门子索引真·先过滤（dept_indexes 在时），
                  否则退回 metadata_filtered_search（全库后过滤，零破坏回退）。
        """
        if "*" in user_departments:
            # 管理员：全库
            logger.info("管理员权限，全库检索")
            return self.hybrid_search(query, top_k)

        # 普通用户：只看授权部门 + 公共
        allowed = list(set(user_departments + ["公共"]))
        logger.info(f"用户权限检索: 仅可见部门 {allowed}")
        if self.dept_indexes is not None:
            return self._rbac_subindex_search(query, allowed, top_k)
        return self.metadata_filtered_search(query, {"department": allowed}, top_k)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n py312 python -m pytest project1-rag/tests/test_rbac_subindex.py -v`
Expected: PASS（6 条）。

- [ ] **Step 5: 跑全量测试确认无回归**

Run: `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 conda run --no-capture-output -n py312 python -m pytest project1-rag/tests/ -v`
Expected: PASS（35 + 本 Task 6 = 41 条）。**不提交。**

---

## Task 3: main.py 装配子索引（集成 + 回归）

**Files:**
- Modify: `project1-rag/main.py:84-98`（`build_knowledge_base`：构建并透传 `dept_indexes`）
- Test: 全量回归（无新增单测——装配是 2 行调用 Task 1/2 已测方法；`app/api.py` 复用同一 `retrieval_module`，无需改）

**说明**：本 Task 不新增单测。理由：`build_department_indexes`（Task 1）与 `RetrievalOptimizationModule` 接子索引（Task 2）均已单测覆盖；`main.py` 仅是把前者产物透传给后者（纯装配，2 行）。验收 = 全量套件绿 + 手动冒烟（可选）。这与既有「main.py 透传 config」Task 的验收方式一致（review by inspection + 套件绿）。

- [ ] **Step 1: 修改 `build_knowledge_base`**

把 `project1-rag/main.py` 的 `build_knowledge_base` 方法（当前约 84-98 行）整体替换为：
```python
    def build_knowledge_base(self):
        """构建知识库：增量索引，文档变化时自动清缓存"""
        logger.info("📚 构建知识库（增量模式）...")
        self.data_module.load_documents()
        chunks = self.data_module.chunk_documents()
        # 增量构建：content_hash 对比，只更新变化的 chunk
        vectorstore, has_changes = self.index_module.build_incremental(chunks)
        self.index_module.save_index()
        # 按部门子索引（RBAC 真·先过滤用；从全库切片，零重复嵌入）
        dept_indexes = self.index_module.build_department_indexes(vectorstore)
        # 坑29：文档变了 → 缓存必须清空，否则返回旧答案（缓存一致性）
        if has_changes and self.cache:
            self.cache.clear()
        self.retrieval_module = RetrievalOptimizationModule(
            vectorstore, chunks, reranker=self._init_reranker(),
            config=self.config, dept_indexes=dept_indexes)
        stats = self.data_module.get_statistics()
        logger.info(f"📊 知识库统计: {stats}")
```

- [ ] **Step 2: 跑全量测试确认无回归**

Run: `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 conda run --no-capture-output -n py312 python -m pytest project1-rag/tests/ -v`
Expected: PASS（41 条全绿）。

- [ ] **Step 3（可选冒烟）: 手动验证 RBAC 路径**

若环境有 `LLM_API_KEY`，可手动跑（会加载真 bge，较慢）：
Run: `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 conda run --no-capture-output -n py312 python project1-rag/main.py rbac`
Expected: 4 个场景正常输出；日志可见 `部门子索引构建完成: N 个部门` 与 `[RBAC子索引] allowed=...`。无 `git` 操作。**不提交。**

---

## Self-Review（plan 作者自检，已做）

1. **Spec 覆盖**：§6.1→Task1；§6.2→Task2；§6.3→Task3；§10 的 9 条测试全部分配（切片 3 条 T1；union/no-leak/A1证明/对照/fallback/admin/multi-dept 6 条 T2）。✓
2. **占位符扫描**：无 TBD/TODO；所有代码步骤含完整代码。✓
3. **类型一致**：`_slice_department_indexes(vectorstore, embeddings)`、`build_department_indexes(self, vectorstore)`、`dept_indexes` 参名、`_rbac_subindex_search(query, allowed_depts, top_k)` 跨 Task 一致。✓
4. **细化已标注**：A1 证明测试从 spec 的「返空」改为「不搜全库 + 对照搜全库」（Global Constraints 末段已声明，意图与 spec 一致）。✓

---

## 执行交接

计划已保存至 `project1-rag/docs/superpowers/plans/2026-07-01-rbac-sub-index.md`。两种执行方式：

1. **Subagent-Driven（推荐）** — 每 Task 派一个全新 implementer 子代理 + 每 Task 后审查 + 末尾整支审查。
2. **Inline 执行** — 在本会话用 executing-plans 批量执行，带检查点。

选哪种？
