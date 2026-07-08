"""
A/B 对比实验：纯向量 vs 混合检索 vs 混合+rerank

目的：用真实数据证明 rerank 的价值，产出简历可用的量化对比表。
对应《踩坑实录》坑8/9/10——这是"为什么用rerank"的最硬证据。

策略：
  A 纯向量     : 只向量检索(cosine)，无 BM25、无 rerank
  B 混合检索   : 向量+BM25+RRF融合，无 rerank
  C 混合+rerank: 向量+BM25+RRF+bge-reranker 精排

指标：
  - context_recall: 正例中，召回了正确文档的比例（越高越好）
  - 负例拒答率   : 负例中，正确拒答(无召回)的比例（越高越好）
  - 综合准确率   : 正例召回 AND 负例拒答 的总体表现

用法：python evaluation/ab_compare.py
"""

import json
import sys
import logging
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_CONFIG
from rag_modules import (
    DataPreparationModule, IndexConstructionModule,
    RetrievalOptimizationModule, GenerationIntegrationModule,
)
from langchain_core.documents import Document

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVAL_FILE = Path(__file__).parent / "eval_dataset.jsonl"


def load_eval_set() -> List[Dict]:
    data = []
    with open(EVAL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def strategy_pure_vector(query: str, vs, top_k: int = 3, threshold: float = 0.3) -> List[Document]:
    """策略A：纯向量检索（cosine 阈值过滤）"""
    raw = vs.similarity_search_with_score(query, k=top_k * 2)
    docs = []
    for doc, sim in raw:
        doc.metadata["vector_sim"] = round(float(sim), 4)
        docs.append(doc)
    return [d for d in docs if d.metadata["vector_sim"] >= threshold][:top_k]


def strategy_hybrid(query: str, ret, top_k: int = 3, use_rerank: bool = False) -> List[Document]:
    """策略B/C：混合检索（B无rerank，C有rerank）。通过临时控制 reranker 开关。"""
    saved = ret.reranker
    ret.reranker = saved if use_rerank else None
    try:
        docs = ret.hybrid_search(query, top_k=top_k)
    finally:
        ret.reranker = saved
    return docs


def eval_strategy(name: str, search_fn, eval_set, **kwargs) -> Dict:
    """评测一个策略：正例看召回，负例看拒答"""
    pos_correct = 0  # 正例正确召回数
    pos_total = 0
    neg_correct = 0  # 负例正确拒答数
    neg_total = 0

    for case in eval_set:
        q = case["question"]
        gt_doc = case["source_doc"]
        docs = search_fn(q, **kwargs)
        hit_sources = {d.metadata.get("source", "?") for d in docs}

        if gt_doc:  # 正例（有标准答案文档）
            pos_total += 1
            # 判定：是否召回了正确文档（source 匹配）
            if any(gt_doc in s or s in gt_doc for s in hit_sources):
                pos_correct += 1
        else:  # 负例（知识库无答案）
            neg_total += 1
            if not docs:  # 正确拒答（无召回）
                neg_correct += 1

    recall = pos_correct / pos_total if pos_total else 0
    refuse = neg_correct / neg_total if neg_total else 0
    # 综合分：正例召回率 × 负例拒答率（两者都重要，乘积代表综合）
    overall = recall * refuse if (pos_total and neg_total) else recall

    return {
        "strategy": name,
        "pos_recall": round(recall, 4),
        "neg_refuse": round(refuse, 4),
        "overall": round(overall, 4),
        "pos_detail": f"{pos_correct}/{pos_total}",
        "neg_detail": f"{neg_correct}/{neg_total}",
    }


def main():
    cfg = DEFAULT_CONFIG
    cfg.validate()

    # 初始化
    dp = DataPreparationModule(cfg.data_path, cfg.chunk_size, cfg.chunk_overlap)
    dp.load_documents()
    chunks = dp.chunk_documents()
    idx = IndexConstructionModule(cfg.embedding_model, cfg.index_save_path)
    vs = idx.load_index() or idx.build_vector_index(chunks)

    from rag_modules.reranker import Reranker
    reranker = Reranker()
    ret = RetrievalOptimizationModule(vs, chunks, reranker=reranker)

    eval_set = load_eval_set()
    logger.info(f"评测集 {len(eval_set)} 条")

    # 跑三种策略
    print("\n" + "=" * 70)
    print("🔬 A/B 对比实验：纯向量 vs 混合检索 vs 混合+rerank")
    print("=" * 70)
    print(f"评测集: {len(eval_set)} 条（正例 {sum(1 for c in eval_set if c['source_doc'])} + 负例 {sum(1 for c in eval_set if not c['source_doc'])}）\n")

    results = []
    results.append(eval_strategy("A 纯向量", strategy_pure_vector, eval_set, vs=vs))
    results.append(eval_strategy("B 混合检索", strategy_hybrid, eval_set, ret=ret, use_rerank=False))
    results.append(eval_strategy("C 混合+rerank", strategy_hybrid, eval_set, ret=ret, use_rerank=True))

    # 输出对比表
    print(f"{'策略':<14} | {'正例召回率':>10} | {'负例拒答率':>10} | {'综合分':>8} | {'正例':>6} {'负例':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['strategy']:<14} | {r['pos_recall']:>10.2%} | {r['neg_refuse']:>10.2%} | {r['overall']:>8.2%} | {r['pos_detail']:>6} {r['neg_detail']:>6}")
    print("-" * 70)

    # 结论
    best = max(results, key=lambda x: x["overall"])
    print(f"\n🏆 最优策略: {best['strategy']}（综合分 {best['overall']:.2%}）")

    # rerank 增量价值
    b = results[1]
    c = results[2]
    print(f"\n📊 rerank 的增量价值（B→C）:")
    print(f"  正例召回率: {b['pos_recall']:.2%} → {c['pos_recall']:.2%}（{'+' if c['pos_recall']>=b['pos_recall'] else ''}{(c['pos_recall']-b['pos_recall']):.2%}）")
    print(f"  负例拒答率: {b['neg_refuse']:.2%} → {c['neg_refuse']:.2%}（{'+' if c['neg_refuse']>=b['neg_refuse'] else ''}{(c['neg_refuse']-b['neg_refuse']):.2%}）")

    # 保存
    out = Path(__file__).parent / "ab_result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"结果已保存: {out}")


if __name__ == "__main__":
    main()
