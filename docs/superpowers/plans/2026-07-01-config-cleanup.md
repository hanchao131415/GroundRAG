# Config 死字段清理 + 性能优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 清理 config.py 中 7 个从未被读取的死字段，修正 observability.py 绕过 config 直接读 os.getenv 的不一致，调整 chunk_overlap 默认值减少索引膨胀。

**Architecture:** 外科手术式——只动 4 个文件，不动检索/生成热路径。删除即优化（每实例化少 7 次 env 读取 + 7 次 dataclass 字段初始化）。

**Tech Stack:** Python 3.12+, dataclasses, pytest.

## Global Constraints

- **不提交**：未经用户明确要求，不做 `git add`/`commit`。
- **外科手术**：只改 `config.py`、`observability.py`、`.env.example`、`tests/test_core.py`。不动检索/生成热路径。
- **零破坏**：现有 42 条测试必须全绿。删除字段后，所有 `hasattr(DEFAULT_CONFIG, ...)` 相关测试需同步更新。
- **Python 环境**：当前系统 Python 3.13，pytest 直接 `python -m pytest tests/ -v`。

---

## File Structure

| 文件 | 责任 | 本轮改动 |
|---|---|---|
| `config.py` | 全局配置（env 驱动） | 删除 7 死字段，chunk_overlap 默认 50→25 |
| `rag_modules/observability.py` | Langfuse 可观测 | 从 DEFAULT_CONFIG 读 langfuse 配置（替代直接 os.getenv） |
| `.env.example` | 环境变量模板 | 删除 `RAG_RERANK_TOP_N=3` |
| `tests/test_core.py` | 单元测试 | 删除 `test_rerank_top_n_kept_for_backcompat`，追加 `TestConfigDeadFields` |

---

### Task 1: config.py — 删除 7 个死字段 + chunk_overlap 默认调优

**Files:**
- Modify: `project1-rag/config.py`
- Test: `project1-rag/tests/test_core.py`（删除旧测试 + 追加新测试）

**Interfaces:**
- Removes: `RAGConfig.rerank_top_n`, `enable_rbac`, `eval_dataset_path`, `redis_url`, `langfuse_host`, `langfuse_public_key`, `langfuse_secret_key`
- Changes: `RAGConfig.chunk_overlap` 默认值 50 → 25
- Produces: 精简后的 `RAGConfig`（35→23 行配置字段）

- [ ] **Step 1: 更新测试文件**

删除 `test_rerank_top_n_kept_for_backcompat`，追加 `TestConfigDeadFields` 验证死字段已移除：

在 `tests/test_core.py` 中：

```python
# 删除 TestConfigThresholds 中的 test_rerank_top_n_kept_for_backcompat 方法（约 lines 328-330）

# 追加到文件末尾：
# ===== 12. config 死字段清理（Task1）=====
class TestConfigDeadFields:
    """验证 7 个死字段已从 config 移除"""

    def test_dead_fields_removed(self):
        from config import DEFAULT_CONFIG
        dead = ["rerank_top_n", "enable_rbac", "eval_dataset_path", "redis_url",
                "langfuse_host", "langfuse_public_key", "langfuse_secret_key"]
        for field in dead:
            assert not hasattr(DEFAULT_CONFIG, field), f"死字段 {field} 应已移除"

    def test_chunk_overlap_default_is_25(self):
        from config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG.chunk_overlap == 25
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python -m pytest tests/test_core.py::TestConfigDeadFields -v
```
预期: 2 个 FAIL（死字段仍存在、chunk_overlap 仍是 50）

- [ ] **Step 3: 修改 config.py**

删除以下内容：

```python
# 删除 lines 42-44（rerank_top_n 废弃注释 + 字段定义）:
    # rerank_top_n 已废弃：与 top_k 重叠（rerank 给所有候选打分后由 threshold+top_k 截断）。
    # 保留字段仅为向后兼容，检索热路径不再读取。调参请改 top_k。
    rerank_top_n: int = int(_env("RAG_RERANK_TOP_N", "3"))

# 删除 lines 63-64（enable_rbac）:
    # ===== ⑤ 权限（JWT 认证 + RBAC 授权）=====
    enable_rbac: bool = _env("RAG_ENABLE_RBAC", "false").lower() == "true"

# 删除 lines 69-70（eval_dataset_path）:
    # ===== ⑥ 评测 =====
    eval_dataset_path: str = _env("RAG_EVAL_DATASET", str(Path(__file__).parent / "evaluation" / "eval_dataset.jsonl"))

# 删除 lines 72-75（langfuse 3 字段 + 注释）:
    # ===== ⑦ 可观测（Langfuse，预留）=====
    langfuse_host: str = _env("LANGFUSE_HOST", "")
    langfuse_public_key: str = _env("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = _env("LANGFUSE_SECRET_KEY", "")

# 删除 lines 77-78（redis_url）:
    # ===== ⑧ 服务运营（预留）=====
    redis_url: str = _env("REDIS_URL", "redis://localhost:6379/0")
```

修改 chunk_overlap 默认值：

```python
# line 34: 50 → 25
chunk_overlap: int = int(_env("RAG_CHUNK_OVERLAP", "25"))
```

修改后各节编号需调整：
- ⑤ 权限部分保留 `jwt_secret` 和 `jwt_expire_hours`，注释从 "⑤ 权限（JWT 认证 + RBAC 授权）" 改为 "⑤ 权限（JWT 认证）"
- ⑥⑦⑧ 节删除，`validate()` 方法保持不变

- [ ] **Step 4: 跑全量测试确认**

```bash
python -m pytest tests/ -v
```
预期: 42 passed（删除 1 旧测试 + 新增 2 测试 = 43 tests，但 TestConfigThresholds 少 1 个方法 = 41 + 2 = ... 实际需确认最终数量）

核心要求：全绿，无 regression。

---

### Task 2: observability.py — 从 config 读 langfuse 配置（替代直接 os.getenv）

**Files:**
- Modify: `project1-rag/rag_modules/observability.py:25-27`

**Interfaces:**
- Consumes: `DEFAULT_CONFIG.langfuse_*`（但 Task 1 删了这些字段！）

**重要：** 因为 Task 1 删除了 config 中的 langfuse 字段，observability.py 不能改为从 config 读取。本 Task 改为：**保持 observability.py 直接用 os.getenv（这是正确的设计——observability 是基础设施层，不应依赖 config 模块）**。

那 Task 2 实际做什么？—— 加注释说明为什么直接读 env 而不走 config：

- [ ] **Step 1: 修改 observability.py:23-27**

```python
def get_langfuse_handler():
    """
    获取 Langfuse CallbackHandler（如果配置了 key）。
    没配 key / 加载失败 → 返回 None，主流程降级为只用本地 trace。

    设计说明：直接读 os.getenv 而非从 config 模块导入——
    observability 是基础设施层，应零依赖业务 config；
    且 langfuse key 可能由平台注入（K8s secret / CI env），不经过 .env。

    Returns:
        CallbackHandler 实例 或 None
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
```

只是更新 docstring，代码逻辑不变。

- [ ] **Step 2: 确认无回归**

```bash
python -m pytest tests/ -v
```
预期: 全绿。

---

### Task 3: .env.example — 删除死字段 RAG_RERANK_TOP_N

**Files:**
- Modify: `project1-rag/.env.example`

- [ ] **Step 1: 删除 RAG_RERANK_TOP_N 行**

删除 `.env.example` 中的：
```
RAG_RERANK_TOP_N=3
```

位于检索配置段（约 line 59）。

- [ ] **Step 2: 验证 .env.example 仍有效**

```bash
python -c "from dotenv import load_dotenv; load_dotenv('.env.example'); import os; print('OK')"
```
预期: OK

---

### Task 4: 全量回归 + 清理收尾

- [ ] **Step 1: 跑全量测试**

```bash
python -m pytest tests/ -v --tb=short
```
预期: 全部通过（41 tests 或更新后的数量）。

- [ ] **Step 2: 验证 import 正常**

```bash
python -c "from config import DEFAULT_CONFIG; print(f'fields={len([f for f in dir(DEFAULT_CONFIG) if not f.startswith(\"_\")])}')" 
python -c "from main import EnterpriseRAGSystem; print('main OK')"
python -c "from app.api import app; print('api OK')"
```
预期: 无 import 错误。

---

## Self-Review

**1. Spec coverage:**
- [x] 删除 rerank_top_n — Task 1
- [x] 删除 enable_rbac — Task 1
- [x] 删除 eval_dataset_path — Task 1
- [x] 删除 redis_url — Task 1
- [x] 删除 langfuse_* 三个字段 — Task 1
- [x] chunk_overlap 默认 50→25 — Task 1
- [x] observability 修正 — Task 2（改为加注释说明设计，而非改从 config 读——因为死字段已删，直接从 config 读反而会崩）
- [x] .env.example 清理 — Task 3
- [x] 全量回归 — Task 4

**2. Placeholder scan:** 无 TBD/TODO。每个 step 有具体代码和命令。

**3. Type consistency:** 字段删除后，相关引用同步清理（测试更新 + .env.example 更新）。

**4. 修正：** 原分析建议 observability.py 从 config 读 langfuse 配置，但既然 config 中这些字段是死的且要被删除，observability.py 直接读 os.getenv 是正确设计。Task 2 改为加注释说明设计意图。
