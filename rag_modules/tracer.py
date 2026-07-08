"""
轻量全链路 Trace（可观测性，自研）

为什么自研而非 Langfuse：
- Langfuse 自托管需起 Docker+Postgres，本地开发偏重
  （Langfuse 是流行的开源 LLM 可观测平台，但它的自托管版必须跑起 Docker Compose
   一整套服务：Postgres + 后端 + 前端 + worker，对一台开发机来说太重了）
- 核心需求"每次问答看清各环节耗时/输入输出/成本"用自研即可达成
  （自研方案只需一个 Python 文件 + 一个 JSONL 文件，零依赖、零运维）
- 自研透明，能讲清每步；等需多租户/团队协作再上 Langfuse
  （"先自研够用，再按需上重型方案"是务实的工程演进路径）

记录每次问答的：
  ① 各环节耗时（意图/改写/检索/生成）
  ② 各环节输入输出
  ③ token 用量 + 成本估算
  ④ 失败定位（哪一步挂了）

存成 JSONL，可后续分析/出报表。
对应《真实 RAG 全貌》⑦可观测子系统。

---
【设计要点】本模块贯穿"数据采集 → 累加统计 → 成本估算 → 落盘"四步：
  - Tracer.start()    : 创建一条空的 trace 记录（拿到 trace_id）
  - Tracer.step()     : 每完成一个环节就追加一条 step（耗时/IO/token 累加）
  - Tracer.finish()   : 汇总、按 provider 算钱、写一行 JSONL
  - Tracer.print_summary() : 给人看的终端摘要
"""

# ===== 标准库导入（全部来自标准库，因此本模块零第三方依赖，便于部署）=====
import json          # 序列化 trace 字典 → 一行 JSON 文本
import time          # perf_counter 高精度计时（计耗时）
import uuid          # 生成 trace_id（唯一标识本次问答）
import logging       # 模块级 logger，便于将来接日志系统
from datetime import datetime   # 记录时间戳 ts
from pathlib import Path        # 跨平台路径处理（trace_dir）
from typing import Any, Dict, Optional  # 类型注解，提升可读性

# 本模块自己的 logger（命名空间为模块名），方便上层统一配置日志级别
logger = logging.getLogger(__name__)


# 模块级常量：用"─"字符拼 50 个作为分隔线。
# 放模块级（而非每次 print 时再拼）是为了避免重复创建字符串，是个小优化。
_SEP = "─" * 50
# _truncate 截断后的统一后缀，定义为常量便于全局统一修改
_TRUNC_SUFFIX = "...(截断)"


class Tracer:
    """单次问答的全链路 trace 收集器。

    典型用法（贯穿一次问答的完整生命周期）::

        tracer = Tracer()
        trace = tracer.start(question, user)
        # ... 执行环节 A
        tracer.step(trace, "检索", q_in, docs, elapsed_ms(t0))
        # ... 执行环节 B
        tracer.step(trace, "生成", prompt, answer, elapsed_ms(t1), tokens)
        # 收尾：算钱 + 落盘
        tracer.finish(trace, answer, provider="deepseek")
        tracer.print_summary(trace)

    说明：Tracer 实例本身是"无状态"的——每次问答调用 start() 拿到一个独立的
    trace 字典，trace 与 trace 之间互不干扰。Tracer 只是提供方法 + 落盘路径。
    """

    # __slots__ 显式声明实例属性，禁止动态新增属性。
    # 好处：① 节省内存（无 __dict__）② 防止拼写错属性名（如 self.out_path）
    __slots__ = ("trace_dir", "_out_path")

    def __init__(self, trace_dir: str = "data/traces"):
        """初始化 Tracer。

        :param trace_dir: trace 文件存放目录，默认 data/traces
        """
        self.trace_dir = Path(trace_dir)             # 转成 Path 对象，跨平台
        # 目录不存在则创建（含父目录），存在则不报错（exist_ok=True）
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        # 关键优化：把最终输出路径在初始化时就算好缓存起来，
        # 避免每次 finish() 都重新做一次路径拼接（热点路径少做点活）
        self._out_path = self.trace_dir / "trace.jsonl"  # 预计算路径，避免每次 join

    def start(self, question: str, user: Optional[str] = None) -> Dict[str, Any]:
        """开始一次 trace，返回 trace 记录。

        这里返回的是一个普通 dict（"记录"），后续 step/finish 都就地修改它，
        因此调用方拿到引用后传到各环节即可，无需手动回传。

        :param question: 用户原始问题
        :param user: 用户标识，可选（用于后续按用户筛选 trace）
        :return: 一个空的 trace 字典，含本次问答的骨架字段
        """
        return {
            # uuid4 生成随机串，取 .hex（32 位无横线）前 12 位作 trace_id。
            # 12 位碰撞概率足够低，且短了方便日志/终端里人眼比对。
            "trace_id": uuid.uuid4().hex[:12],
            # 时间戳精确到秒（timespec="seconds"），够用又不太长
            "ts": datetime.now().isoformat(timespec="seconds"),
            "user": user,
            "question": question,
            "steps": [],            # 各环节 step 的列表，由 step() 追加
            "total_ms": 0,          # 累计耗时（ms），由 step() 累加
            # token 三件套：prompt(输入) / completion(输出) / total(合计)
            # 由 step() 在记录带 token 的环节时累加
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "cost_usd": 0.0,        # 本次估算成本（USD），finish() 时计算
            "status": "ok",         # 默认成功，失败时 finish() 改为 "error"
            "error": None,          # 错误信息，失败时填，成功则保持 None
        }

    def step(self, trace: Dict, name: str, input_data: Any, output_data: Any,
             elapsed_ms: float, tokens: Optional[Dict] = None):
        """记录一个环节。

        一个 trace 由多个 step 组成（意图识别 / 查询改写 / 检索 / 生成 ...）。
        每完成一个环节就调一次本方法，把该环节的 IO/耗时/（可选）token 追加进去。

        :param trace: start() 返回的 trace 字典（就地修改）
        :param name: 环节名，如 "检索"、"生成"
        :param input_data: 该环节的输入（任意类型，会 str() 化并截断）
        :param output_data: 该环节的输出
        :param elapsed_ms: 该环节耗时（毫秒），一般用 elapsed_ms(t0) 算出
        :param tokens: 可选 token 统计 dict，形如
                       {"prompt": N, "completion": M, "total": N+M}。
                       只有大模型调用环节才需要传（检索这种无 token 的环节可省略）。
        """
        # 组装单个 step 记录。
        # 注意：input/output 用 str() 转 + _truncate 截断到 500 字符，
        # 这是为了控制 JSONL 单行体积——长文本（如检索到的文档）整段存进去
        # 会让 trace 文件迅速膨胀，截断后既能定位问题又不至于过大。
        step = {
            "name": name,
            "input": _truncate(str(input_data), 500),
            "output": _truncate(str(output_data), 500),
            "ms": round(elapsed_ms, 1),   # 保留 1 位小数，够精确又不冗长
        }
        # 仅当该环节是大模型调用（带 token）时，记录 token 并累加到 trace 汇总。
        # 检索/向量召回这类本地计算没有 token，不传即可——这是 tokens 设为可选的原因。
        if tokens:
            step["tokens"] = tokens
            # 把本环节的 token 累加进 trace 顶层汇总（prompt/completion/total 三个键）
            for k in ("prompt", "completion", "total"):
                trace["tokens"][k] += tokens.get(k, 0)
        # 把本 step 追加进 steps 列表（trace["steps"] 是就地修改的 list）
        trace["steps"].append(step)
        # 累加耗时到 total_ms。注意这是"各环节耗时之和"，并非端到端墙钟时间
        # （二者通常接近，但若存在并发/等待则可能不同——这里取累加值即可）。
        trace["total_ms"] = round(trace["total_ms"] + elapsed_ms, 1)

    # ============================================================
    # 【坑37：token 成本估算】各供应商的 token 单价表
    # ============================================================
    # 关键设计：按 provider 动态定价。不同供应商单价差异巨大（如 deepseek 输入
    # 仅 $0.14/M，而 openai gpt-4o-mini 输入 $2.5/M，相差 ~18 倍），
    # 如果不区分 provider 而用统一价，成本估算会严重失真。
    #
    # 为什么单位是"USD / 1 token"而不是"USD / 1M token"？
    #   ——因为后面算钱时直接 tokens * price 即可，省去 /1_000_000 的除法，
    #     也避免浮点精度问题。所以这里的数值看起来都很小（1e-6 量级）。
    #
    # 这是一个【类变量】（定义在方法外、无 self 前缀），所有 Tracer 实例共享同一份，
    # 既省内存也便于把"价格表"当配置看。每个 value 是 (输入价, 输出价) 二元组。
    #
    # 各供应商的 token 单价（USD / 1 token），方便面试/运维时一眼看清成本
    # 价格为 2025-2026 公开定价，实际以供应商官网为准
    _PRICE_PER_TOKEN = {
        "deepseek":   (0.00000014, 0.00000028),   # $0.14/M in, $0.28/M out
        "zhipu":      (0.00000007, 0.00000029),   # ¥0.5/M in, ¥2/M out → 折 USD
        "zai":        (0.00000100, 0.00000400),   # 约 $1/M in, $4/M out
        "qwen":       (0.00000050, 0.00000200),   # 约 $0.5/M in, $2/M out
        "openai":     (0.00000250, 0.00001000),   # gpt-4o-mini: $2.5/M in, $10/M out
        "local":      (0.0, 0.0),                 # 本地模型免费
    }

    def finish(self, trace: Dict, answer: str, status: str = "ok", error: str = None,
               provider: str = ""):
        """结束 trace，估算成本（按 provider 动态定价），落盘。

        这是 trace 生命周期的收尾，做三件事：
          1) 回填最终答案/状态/错误；
          2) 按指定 provider 查价格表，算出本次成本（坑37）；
          3) 把整条 trace 序列化为一行 JSON，追加到 JSONL 文件。

        :param trace: start() 返回的 trace 字典
        :param answer: 最终给用户的答案（截断到 800 字符存档）
        :param status: "ok" 或 "error"，默认 "ok"
        :param error: 失败原因（成功时为 None）
        :param provider: 供应商名，用于查 _PRICE_PER_TOKEN 算钱。
                         未知 provider 会落到默认价（按智谱估算，偏保守）。
        :return: 同一个 trace 字典（已更新并落盘），便于调用方继续打印
        """
        # ① 回填最终结果字段。answer 截断到 800（比 step 的 500 略宽，因为它是终态）
        trace["answer"] = _truncate(answer, 800)
        trace["status"] = status
        trace["error"] = error

        # ② 成本估算（核心：按 provider 查价）
        # 取出累计的输入/输出 token 数
        pt, ct = trace["tokens"]["prompt"], trace["tokens"]["completion"]
        # 按供应商查价；若 provider 不在表中（如空串/未识别），用一个保守的默认价
        # （这里默认沿用 zhipu 的价，是个"够用且不至于严重低估"的兜底）
        price_in, price_out = self._PRICE_PER_TOKEN.get(provider, (0.00000007, 0.00000029))
        # 成本 = 输入token*输入价 + 输出token*输出价；保留 6 位小数（$ 级别已足够细）
        trace["cost_usd"] = round(pt * price_in + ct * price_out, 6)

        # ③ 落盘（JSONL，一行一条）
        # 【为什么要 JSONL 而不是普通 JSON 大数组？】
        #   - 追加写（mode="a"）极轻量，无需读旧文件、无需把整个数组加载进内存；
        #   - 每行独立，一行挂了不影响其他行；
        #   - 分析时天然适合流式处理：grep / jq / pandas.read_json(lines=True)。
        # ensure_ascii=False 让中文原样写入，避免 \uXXXX 转义，可读性更好。
        with open(self._out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        return trace

    def print_summary(self, trace: Dict):
        """打印人类可读的 trace 摘要。

        与 finish() 的 JSONL（给机器/后续分析用）互为补充——这个方法面向终端用户，
        用 emoji + 对齐排版，让人一眼看清"哪一步慢、花了多少 token、估了多少钱"。
        仅做打印，不修改 trace 也不落盘。
        """
        print("\n" + _SEP)   # 顶部空一行加分隔线，视觉上把多次问答的摘要隔开
        # 头部：trace_id / 用户 / 状态。user 用 .get("...","-") 兜底（None 时显示 -）
        print(f"📋 TRACE {trace['trace_id']}  用户:{trace.get('user','-')}  状态:{trace['status']}")
        print(f"❓ {trace['question']}")    # 原始问题
        # 逐行打印每个环节：名称、耗时、（若有）token 总数
        # 格式串中 {s['name']:12} 左对齐占 12 列，{s['ms']:>7.1f} 右对齐占 7 列，
        # 这样多次问答的耗时数字能上下对齐，方便人眼比对哪步慢。
        for s in trace["steps"]:
            # 只在该 step 记录过 token 时才显示 token 数；检索等无 token 环节则省略
            tok = f" tok:{s.get('tokens',{}).get('total','-')}" if "tokens" in s else ""
            print(f"  └ {s['name']:12} {s['ms']:>7.1f}ms{tok}")
        # 汇总行：总耗时 + token 三件套 + 估算成本
        t = trace["tokens"]
        print(f"⏱ 总耗时 {trace['total_ms']}ms | tokens:输入{t['prompt']} 输出{t['completion']} 共{t['total']} | ≈${trace['cost_usd']:.6f}")
        # 若有错误，单独打印一行，便于快速定位失败原因
        if trace.get("error"):
            print(f"❌ 错误: {trace['error']}")
        print(_SEP)   # 收尾分隔线


def _truncate(s: str, n: int) -> str:
    """把字符串截断到 n 个字符，超出则加 "...(截断)" 后缀。

    模块级私有函数（下划线前缀），供 step/finish 存档长文本时压缩体积用。
    注意：是按【字符数】而非字节数截断，对中文友好（1 个中文算 1 字符）。
    """
    # 不超长就原样返回（避免无谓的字符串拷贝）；超长则取前 n 个 + 截断后缀
    return s if len(s) <= n else s[:n] + _TRUNC_SUFFIX


def time_it():
    """简易计时器：t0=time_it(); ...; ms=time_it(t0)

    返回一个高精度时间戳（time.perf_counter()）。
    配合 elapsed_ms(t0) 计算耗时。perf_counter 比 time.time() 更适合计时间间隔
    （它单调递增、精度更高，不受系统时钟回拨影响）。
    """
    return time.perf_counter()


def elapsed_ms(t0: float) -> float:
    """计算从 t0（time_it() 返回值）到现在的耗时，单位毫秒。

    perf_counter 差值是秒，乘 1000 转毫秒。返回浮点数，调用方一般会 round 一下。
    """
    return (time.perf_counter() - t0) * 1000
