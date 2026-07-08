"""
RAG 评测脚本（自研轻量版，不依赖 ragas）

为什么不用 ragas：
- ragas 依赖重（scikit-network 等），在 Python 3.14 编译失败
- ragas 是黑箱，自研能讲清每个指标怎么算（面试加分）
- 核心指标用 LLM-as-Judge 可直接算

评测三维度（对标 ragas，但自研透明）：
  ① 检索召回：该召回的文档召回了没？（context_recall）
  ② 答案忠实：答案基于检索内容，没瞎编？（faithfulness）
  ③ 答案相关：答案正面回答了问题？（answer_relevancy）

用法：python evaluation/run_eval.py
"""

import json
import sys
import logging
from pathlib import Path
from typing import List, Dict

# 让脚本能 import 项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_CONFIG
from rag_modules import (
    DataPreparationModule, IndexConstructionModule,
    RetrievalOptimizationModule, GenerationIntegrationModule,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

EVAL_FILE = Path(__file__).parent / "eval_dataset.jsonl"


def load_eval_set() -> List[Dict]:
    """加载评测集"""
    data = []
    with open(EVAL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info(f"加载评测集 {len(data)} 条")
    return data


def make_judge_prompt(question, ground_truth, retrieved_context, answer) -> str:
    """构造 LLM 裁判 prompt：让 LLM 判断 3 个维度"""
    return f"""你是一个严谨的 RAG 系统评测裁判。请对以下问答结果打分。

【用户问题】{question}
【标准答案】{ground_truth}
【检索到的上下文】
{retrieved_context}
【系统回答】{answer}

请判断以下 3 个维度，每个给 0 或 1（0=不合格，1=合格），严格按 JSON 格式输出，不要其他内容：

1. "context_recall"：检索到的上下文是否包含了回答问题所需的关键信息？（标准答案的信息在上下文里吗？）
2. "faithfulness"：系统回答是否忠实于检索到的上下文，没有编造上下文之外的内容？
3. "answer_relevancy"：系统回答是否正面回答了用户问题？（拒答类问题，若确实无相关信息且回答"未找到"也算合格）

只输出 JSON，例：
{{"context_recall": 1, "faithfulness": 1, "answer_relevancy": 1}}"""


def parse_judge(text: str) -> Dict[str, int]:
    """解析裁判返回的 JSON"""
    try:
        # 提取第一个 {...}
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {"context_recall": 0, "faithfulness": 0, "answer_relevancy": 0}


def run_eval():
    """跑评测：对每条 case 检索+生成+打分，汇总三维度"""
    cfg = DEFAULT_CONFIG
    cfg.validate()

    # 初始化系统
    dp = DataPreparationModule(cfg.data_path, cfg.chunk_size, cfg.chunk_overlap)
    dp.load_documents()
    chunks = dp.chunk_documents()
    idx = IndexConstructionModule(cfg.embedding_model, cfg.index_save_path)
    vs = idx.load_index() or idx.build_vector_index(chunks)
    gen = GenerationIntegrationModule(
        cfg.llm_provider, cfg.llm_base_url, cfg.llm_api_key,
        cfg.llm_model, cfg.temperature, cfg.max_tokens,
    )

    # bge-reranker 精排（本地，快且准）
    from rag_modules.reranker import Reranker
    reranker = Reranker()
    ret = RetrievalOptimizationModule(vs, chunks, reranker=reranker)

    eval_set = load_eval_set()
    results = []
    # 累加三维度（算平均分）
    totals = {"context_recall": 0, "faithfulness": 0, "answer_relevancy": 0}
    n = len(eval_set)

    for i, case in enumerate(eval_set, 1):
        q = case["question"]
        gt = case["ground_truth"]
        logger.info(f"[{i}/{n}] 评测: {q}")

        # 检索（管理员权限，评测时不限权）
        chunks_hit = ret.hybrid_search(gen.query_rewrite(q), top_k=cfg.top_k)
        context = gen._build_context(chunks_hit)
        # 生成
        answer = gen.generate_answer(q, chunks_hit)

        # LLM 裁判打分
        judge = gen.llm.invoke(make_judge_prompt(q, gt, context, answer))
        scores = parse_judge(judge.content if hasattr(judge, "content") else str(judge))

        for k in totals:
            totals[k] += scores.get(k, 0)

        results.append({
            "question": q, "ground_truth": gt, "answer": answer,
            "scores": scores,
            "hit_source": [c.metadata.get("source") for c in chunks_hit],
        })
        logger.info(f"   → 检索:{results[-1]['hit_source']}  分数:{scores}")

    # 汇总
    summary = {k: round(v / n, 4) for k, v in totals.items()}
    print("\n" + "=" * 60)
    print("📊 RAG 评测结果（三维度，0~1，越高越好）")
    print("=" * 60)
    print(f"  ① 检索召回 context_recall : {summary['context_recall']}")
    print(f"  ② 答案忠实 faithfulness    : {summary['faithfulness']}")
    print(f"  ③ 答案相关 answer_relevancy: {summary['answer_relevancy']}")
    print("=" * 60)

    # 保存明细
    out = Path(__file__).parent / "eval_result.json"
    out.write_text(json.dumps({"summary": summary, "details": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"明细已保存: {out}")

    # 列出 bad case（任一维度 0 分）
    print("\n❌ Bad case（任一维度未达标）:")
    bad = [r for r in results if any(v == 0 for v in r["scores"].values())]
    if not bad:
        print("  无（全部达标）")
    for r in bad:
        print(f"  - {r['question']}  分数:{r['scores']}")


if __name__ == "__main__":
    run_eval()
