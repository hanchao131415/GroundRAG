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
