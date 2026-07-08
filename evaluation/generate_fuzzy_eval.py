"""
用 LLM 合成模糊/口语化评测查询

目的：自编评测集(41条)全是开发者视角的标准问法，真实用户问法是模糊/口语/带错别字的。
用 LLM 把标准查询改成真实用户的问法，生成一批"脏"评测集，测你的系统在真实场景下还扛不扛得住。

用法：python evaluation/generate_fuzzy_eval.py
"""

import sys, json, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_CONFIG
from rag_modules.generation_integration import GenerationIntegrationModule

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 从标准评测集中取几条正例，让 LLM 改写成 3 种口语变体
SEEDS = [
    "工作满3年年假几天",
    "住宿费一线城市报销上限多少",
    "密码多少天更换一次",
    "机票报销流程是什么",
    "病假工资怎么发",
]

PROMPT = """把下面的企业知识库问题改写成 3 种真实用户可能的问法。要求：
1. 口语化、不正式（如"那个啥""怎么搞""多少天来着"）
2. 可以含错别字或模糊表述（如"坐速费""密马"）
3. 保持原意不变
4. 每行一个，只输出问题本身，不要编号和解释

原问题：{question}
口语化改写："""


def generate():
    cfg = DEFAULT_CONFIG
    cfg.validate()
    gen = GenerationIntegrationModule(
        cfg.llm_provider, cfg.llm_base_url, cfg.llm_api_key,
        cfg.llm_model, cfg.temperature, cfg.max_tokens,
    )

    results = []
    # 加载已有的标准评测集（取正例 source_doc 作 golden）
    eval_file = Path(__file__).parent / "eval_dataset.jsonl"
    golden_map = {}
    if eval_file.exists():
        for line in eval_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("source_doc"):
                    golden_map[d["question"]] = d

    for q in SEEDS:
        resp = gen.llm.invoke(PROMPT.format(question=q))
        variants = [v.strip() for v in resp.content.split("\n") if v.strip()]
        for v in variants:
            entry = {"question": v, "ground_truth": golden_map.get(q, {}).get("ground_truth", ""),
                     "source_doc": golden_map.get(q, {}).get("source_doc", ""), "fuzzy_of": q}
            results.append(entry)
            print(f"  {q} → {v}")

    out = Path(__file__).parent / "fuzzy_eval.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"生成 {len(results)} 条模糊评测 → {out}")


if __name__ == "__main__":
    generate()
