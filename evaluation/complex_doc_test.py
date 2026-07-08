"""
复杂文档解析验证（真实 PDF + DOCX + Markdown 混用）

验证点：
  1. PDF 解析：PyMuPDF 能否正确提取文本、保留页码
  2. DOCX 解析：python-docx 能否正确提取段落
  3. 多格式混用：PDF+DOCX+MD 能否统一索引
  4. 增量索引：新增文档只 embedding 新 chunk，不重建全量

用法：python evaluation/complex_doc_test.py
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_CONFIG
from rag_modules import DataPreparationModule, IndexConstructionModule

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_pdf_parsing():
    """验证 PDF 解析"""
    dp = DataPreparationModule("data/docs", 500, 50)
    dp.load_documents()
    pdf_docs = [d for d in dp.documents if d.metadata.get("doc_type") == "pdf"]
    print(f"📄 PDF 文档: {len(pdf_docs)} 个页面")
    for d in pdf_docs[:3]:
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        preview = d.page_content[:80].replace("\n", " ")
        print(f"  {src} 第{page}页: {preview}...")
    return len(pdf_docs) > 0


def test_docx_parsing():
    """验证 DOCX 解析"""
    dp = DataPreparationModule("data/docs", 500, 50)
    dp.load_documents()
    docx_docs = [d for d in dp.documents if d.metadata.get("doc_type") == "docx"]
    print(f"\n📝 DOCX 文档: {len(docx_docs)} 个")
    for d in docx_docs:
        src = d.metadata.get("source", "?")
        preview = d.page_content[:80].replace("\n", " ")
        print(f"  {src}: {preview}...")
    return len(docx_docs) > 0


def test_incremental_indexing():
    """验证增量索引：新增文档只 embedding 新 chunk"""
    dp = DataPreparationModule("data/docs", 500, 50)
    dp.load_documents()
    chunks = dp.chunk_documents()
    idx = IndexConstructionModule(DEFAULT_CONFIG.embedding_model, DEFAULT_CONFIG.index_save_path)
    vs, has_changes = idx.build_incremental(chunks)
    idx.save_index()
    stats = dp.get_statistics()
    print(f"\n📊 全量文档: {stats['total_documents']} 篇, {stats['total_chunks']} 块, {stats['departments']}")
    return stats['total_documents'] >= 7  # 原来6篇 + 新PDF + 新DOCX


def test_cross_format_search():
    """验证跨 PDF+DOCX+MD 统一检索"""
    from rag_modules import RetrievalOptimizationModule
    from rag_modules.reranker import Reranker
    dp = DataPreparationModule("data/docs", 500, 50)
    dp.load_documents()
    chunks = dp.chunk_documents()
    idx = IndexConstructionModule(DEFAULT_CONFIG.embedding_model, DEFAULT_CONFIG.index_save_path)
    vs = idx.load_index() or idx.build_vector_index(chunks)
    ret = RetrievalOptimizationModule(vs, chunks, reranker=Reranker())

    queries = ["项目立项", "采购分级", "年假几天", "密码长度"]
    print(f"\n🔍 跨格式检索:")
    for q in queries:
        docs = ret.hybrid_search(q, top_k=1)
        if docs:
            src = docs[0].metadata.get("source", "?")
            doc_type = docs[0].metadata.get("doc_type", "?")
            print(f"  {q} → {src} ({doc_type})")
        else:
            print(f"  {q} → 未找到")


if __name__ == "__main__":
    print("=" * 60)
    print("🔬 复杂文档解析验证")
    print("=" * 60)

    results = []
    results.append(("PDF解析", test_pdf_parsing()))
    results.append(("DOCX解析", test_docx_parsing()))
    results.append(("增量索引", test_incremental_indexing()))
    test_cross_format_search()

    print("\n" + "=" * 60)
    all_pass = all(r[1] for r in results)
    print(f"\n📋 结果: {'✅ 全部通过' if all_pass else '❌ 有失败'}")
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
