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


def test_slice_defaults_missing_department_to_public():
    """缺 department 元数据的文档默认归「公共」（Task 2 RBAC 路由依赖此契约）。"""
    chunks = _make_chunks({"HR": ["年假"]})
    chunks.append(Document(page_content="无部门", metadata={"chunk_id": "x"}))  # 无 department
    vs = _make_vectorstore(chunks)
    result = _slice_department_indexes(vs, _FakeEmb())
    assert "公共" in result
    assert all(d.metadata.get("department", "公共") == "公共"
               for d in result["公共"].docstore._dict.values())


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
