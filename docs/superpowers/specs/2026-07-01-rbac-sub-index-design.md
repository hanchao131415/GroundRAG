# RBAC 检索：按部门子索引实现真·先过滤后检索（修 A1）

> **状态**：设计待评审
> **日期**：2026-07-01
> **关联**：`docs/源码精讲与面试深度拷问.md` 找茬#1 / Q3（杀手锏问题）；`rag_modules/retrieval_optimization.py`、`rag_modules/index_construction.py`、`main.py`
> **前置**：已完成 RAG 调参工具化（接线 5 旋钮 + 扫参脚本），32 测试通过。

---

## 1. 背景与问题（A1 精确诊断）

`permission_aware_search` 是 RBAC 检索的唯一入口，CLI（`main.py:188`）与 API（`app/api.py:167,196`）都经它。普通用户路径最终走 `metadata_filtered_search`，该方法注释声称「先过滤后检索」，但**两条召回路径不对称**：

- **BM25 路径**（`retrieval_optimization.py:189`）：`BM25Retriever.from_documents(allowed_chunks, ...)` —— 在权限子集上重建索引，**真·先过滤** ✓
- **向量路径**（`retrieval_optimization.py:188,191`）：`self.vectorstore.as_retriever(search_kwargs={"k": top_k*2})` 拿的是**全库** vectorstore，从全库召回 `top_k*2` 个再 `_match_filters` 在 Python 里过滤 —— **先检索后过滤** ✗

**故障模式**：若别的部门有 ≥ `top_k*2` 个 chunk 语义上比目标部门的 chunk 更近，名额被占满 → 过滤后为空 → 普通用户对自己有权访问的文档搜不到。变量名 `tmp_vector` 有误导性（它不是子集索引，就是全库）。

admin（`"*"`）/未登录走 `hybrid_search`（全库），无需过滤，**正确，不在本次修复范围**。

项目文档 `docs/源码精讲与面试深度拷问.md` 已把此矛盾标为面试官的「杀手锏问题」（Q3 / 找茬#1）。本设计把它从「承认弱点」变成「已修复 + 可讲清原理」。

---

## 2. 目标与成功判据

**目标**：普通用户的向量召回从「全库 `top_k*2` 再 Python 过滤」（后过滤）改为「只在权限部门子索引里搜」（真·先过滤后检索），且 admin/未登录路径完全不变。

**成功判据（可验证）**：
1. **Bug 复现测试通过**：构造场景——HR 用户的目标 HR chunk 在全库 cosine 排名 > `top_k*2`（被财务 chunk 挤出）。旧路径（全库后过滤）对 HR 用户返回空；新路径（子索引）返回该 HR chunk。
2. **无跨部门泄露**：HR 用户的检索结果永不包含 `department != HR/公共` 的 chunk（结构保证：子索引只含本部门向量，从不搜索其他部门的向量）。
3. **零回归**：admin `"*"` 与未登录路径行为不变（仍 `hybrid_search` 全库）；现有 32 条测试不破。
4. **零破坏契约**：`dept_indexes=None` 时退回旧的 `metadata_filtered_search` 行为（延续调参工具化 Task 2 的 `config=None` 渐进式接线模式）。

---

## 3. 非目标（明确范围外）

- ❌ 子索引单独持久化到磁盘（启动时从全库切片，秒级；权威持久化仍由全库索引承担）。后续要「重启免切片」再加 per-dept `save_local`/`load_local`。
- ❌ 给 RBAC 路径加 reranker + 阈值过滤（见 §11 顺带发现，独立问题，本轮只 surface 不修）。
- ❌ 预建 per-dept BM25（当前每次查询在 `allowed_chunks` 重建，O(N)，是已知 Minor；本轮保持现状）。
- ❌ 改 admin/未登录的 `hybrid_search`。
- ❌ 改扫参脚本（它只调 `hybrid_search`）。

---

## 4. 架构决策：外科手术式

在两种范围里选了**外科手术式**（保留全库索引给 admin/未登录，只为 RBAC 路径加子索引），而非「整体替换」（只留子索引、admin=union 全部子索引）。

**理由**：
- **admin 检索质量**：单一全库 `IndexFlatIP` 给出真·全局 cosine 排序，质量最优。「整体替换」需对每个子索引取 top-k 再合并，当某部门独占答案时，per-partition top-k 截断会漏检 → admin 召回轻微下降。外科手术式让 admin 保持最优。
- **两种正确性需求**：admin 要「全局最优排序」；普通用户要「硬权限隔离」。两种不同需求 → 两套结构各有其理，面试可讲清。
- **零嵌入冗余**：§5 的「从全库切片」技巧让外科手术式不再有 2× 嵌入成本，砍掉了它原本的主要缺点。
- **符合「外科手术、只动必须动的」**：admin 路径没坏，不动。

---

## 5. 核心技术：零重复嵌入的子索引切片

子索引**不重新 embed**。全库 `IndexFlatIP` 已有全部向量，直接复用：

1. `vectors = vectorstore.index.reconstruct_n(0, ntotal)` —— 取出全库所有向量（`IndexFlatIP` 是 flat 索引，支持精确 reconstruct；项目正是用它，见 `index_construction.py:69`）。
2. 用 `vectorstore.index_to_docstore_id`（position→docstore_id）+ `vectorstore.docstore`（docstore_id→Document）建立 position→Document 映射。
3. 按 `doc.metadata["department"]` 把 position 分桶（缺省归 `公共`，与 `data_preparation._enhance_metadata` 一致）。
4. 每个部门桶：`faiss.IndexFlatIP(dim).add(该桶向量)`，配一个只含该桶文档的 `InMemoryDocstore` 与 `{新position: docstore_id}` 映射，构造一个 LangChain `FAISS` 子库。**关键不变量**：`add` 的向量顺序 == id_map 的 position 顺序 == docstore 的文档顺序，三者对齐。
5. 返回 `Dict[部门, FAISS]`，含 `公共` 桶（若无根级文件则无 `公共` 键，查询时 `get` 返回 None 自然跳过）。

复用 `vectorstore.embeddings`（已加载的 bge 单例），整个切片过程**零次 embedding 调用**——只是向量数组搬运。这是面试亮点：「子索引从全库向量切片而来，零冗余嵌入」。

---

## 6. 组件改动（3 处）

### 6.1 `IndexConstructionModule`（新增方法）

```python
def build_department_indexes(self, vectorstore, chunks) -> Dict[str, "FAISS"]:
    """从全库索引切片出按部门（含公共）的子索引，零重复嵌入。

    Returns: {部门名: FAISS子库}。无文档的部门不出现在 dict 中。
    依赖: vectorstore.index 必须支持 reconstruct_n（项目用 IndexFlatIP，满足）。
    """
```

**算法**：见 §5 五步。边界：`ntotal==0` → 返回 `{}`（实际不会发生，`build_vector_index` 空集已抛错）；单文档部门正常处理；`公共` 桶可选。

### 6.2 `RetrievalOptimizationModule`（3 处改）

**(a) 构造函数增参**：
```python
def __init__(self, vectorstore, chunks, reranker=None, config=None, dept_indexes=None):
```
`dept_indexes` 存为 `self.dept_indexes`。默认 `None` = 旧行为（零破坏）。

**(b) 新增 `_rbac_subindex_search`**（镜像 `metadata_filtered_search` 结构，仅向量侧换成子索引）：
```python
def _rbac_subindex_search(self, query, allowed_depts, top_k) -> List[Document]:
    """真·先过滤：只在 allowed_depts（已含公共）的子索引里搜。
    流程：各子索引 similarity_search_with_score(vector_search_k) → 按 cosine 全局排序
          取 top-vector_search_k（与 hybrid_search 向量候选池等规模，保 RRF 平衡）
          → cosine≥vector_score_threshold 过滤 → BM25(allowed_chunks 重建) → _rrf_rerank → [:top_k]
    """
```
- 向量候选池规模 = `vector_search_k`（全局 top，按 cosine）。因为所有子索引共享同一归一化嵌入空间，cosine 跨子索引可比，全局 top-vector_search_k of (union) = 「权限域内的全局 top-vector_search_k」，与 `hybrid_search` 的向量召回语义等价（只是预过滤了）。
- BM25 仍在 `allowed_chunks` 上重建（`bm25_search_k`），与现 `metadata_filtered_search` 一致——本来就是对的先过滤，不动。
- 尾部接 `_rrf_rerank` → `[:top_k]`，**不含 reranker**（与现 RBAC 路径一致，见 §11）。

**(c) `permission_aware_search` 路由**：
```python
# admin "*" / 未登录 → 仍 hybrid_search（不变）
if "*" in user_departments:
    return self.hybrid_search(query, top_k)
allowed = list(set(user_departments + ["公共"]))
if self.dept_indexes is not None:
    return self._rbac_subindex_search(query, allowed, top_k)   # 新：真·先过滤
return self.metadata_filtered_search(query, {"department": allowed}, top_k)  # 回退
```

### 6.3 `main.py`（接 1 行）

`build_knowledge_base` 在 `build_incremental` 后：
```python
vectorstore, has_changes = self.index_module.build_incremental(chunks)
self.index_module.save_index()
dept_indexes = self.index_module.build_department_indexes(vectorstore, chunks)   # 新增
...
self.retrieval_module = RetrievalOptimizationModule(
    vectorstore, chunks, reranker=self._init_reranker(),
    config=self.config, dept_indexes=dept_indexes)                              # 透传
```
**API（`app/api.py`）无需改**——它复用同一个 `rag.retrieval_module`，自动获得修复。

---

## 7. 数据流

**普通用户**（如 zhangsan, depts=[HR]）：
```
permission_aware_search(q, [HR])
  → allowed = [HR, 公共]
  → _rbac_subindex_search:
      vector = union(dept_indexes[HR].search, dept_indexes[公共].search) 按 cosine 取 top-vector_search_k
      bm25   = BM25Retriever(allowed_chunks).invoke   # 仅 HR+公共 chunk
      → _rrf_rerank(vector, bm25) → [:top_k]
```
财务向量**从不被搜索**（财务子索引根本没参与）→ 真·先过滤。

**admin `"*"` / 未登录**：`hybrid_search` 全库，完全不变。

---

## 8. 公共与多部门

- **公共**：独立子索引，查询时 union 进权限域。无重复存储（不像「把公共塞进每个部门子索引」的反范式/denormalize 方案）；公共文档更新只动 1 个子索引。
- **多部门用户**（schema 支持列表，如 `[HR, 财务]`）：`allowed = [HR, 财务, 公共]`，union 三个子索引，天然支持。
- **空权限域**：若用户所有 allowed 部门都无子索引（含公共）→ 向量候选为空 → 返回 `[]`（正确拒答，与现「权限过滤后无可见文档 → return []」一致）。

---

## 9. 向后兼容

- `dept_indexes=None`（默认）→ `permission_aware_search` 走旧 `metadata_filtered_search`，行为同今。
- 扫参脚本构造 `RetrievalOptimizationModule(vs, chunks, config=sweep_cfg)` 不传 `dept_indexes`，且只调 `hybrid_search` → 不受影响。
- 现有 32 条测试不破（构造函数新增参数有默认值）。

---

## 10. 测试策略（TDD，目标驱动）

新增测试（建议入 `tests/test_core.py` 或新建 `tests/test_rbac_subindex.py`）：

| 测试 | 目的 | 手段 |
|---|---|---|
| `test_build_dept_indexes_partitions` | 分桶正确 | 用确定性 fake embeddings + 真 FAISS 建全库；断言 dict 键 = 出现的部门；每子库 docstore 只含本部门 chunk |
| `test_subindex_vectors_match_full` | **证明零重复嵌入** | 同一 chunk 在全库 `reconstruct` 与在子索引 `reconstruct` 的向量完全相等 |
| `test_subindex_no_cross_dept_leak` | 子索引结构隔离 | HR 子库 search 结果 `department` 全 ∈ {HR} |
| `test_rbac_subindex_unions_allowed_and_public` | 并集+排序 | mock 子索引（鸭子类型，实现 `similarity_search_with_score`）；allowed=[HR] → 返回 HR+公共、按 sim 降序、无财务 |
| `test_permission_aware_uses_subindexes` | 端到端 RBAC | dept_indexes 在：HR 用户得 HR+公共、永不得财务 |
| **`test_a1_subindex_finds_what_postfilter_missed`** | **Bug 复现（核心）** | 同一 rigged 场景对比两路径：① `dept_indexes=None`（旧全库后过滤）—— mock 全库 `as_retriever` 只返回财务 top-6（HR chunk 被挤出）→ `permission_aware_search([HR])` 返空；② `dept_indexes` 在（新子索引）—— HR 子索引返回 HR chunk → 返回该 HR chunk。证明先过滤胜过先检索后过滤 |
| `test_dept_indexes_none_falls_back` | 零破坏 | `dept_indexes=None` → 走 `metadata_filtered_search`，不崩，仍按 department 过滤 |
| `test_admin_star_uses_hybrid` | admin 不变 | user=[`"*"`] → 命中 `hybrid_search` 路径，不碰子索引 |
| `test_multi_dept_user_union` | 多部门 | user=[HR, 财务] → 得 HR+财务+公共 |

**测试基建**：`build_department_indexes` 与零重复嵌入证明用「确定性 fake embeddings + 真 FAISS」（快、可重现）；`_rbac_subindex_search` 单测用鸭子类型 fake 子索引（不依赖 FAISS）。

---

## 11. 权衡与范围外（含顺带发现）

- **子索引不持久化**：启动从全库切片（秒级）。权威持久化 = 全库索引（`build_incremental` + `save_index` 不变）。要重启免切片再加 per-dept 持久化。
- **顺带发现（独立问题，本轮不修，仅记录）**：当前 `metadata_filtered_search` 尾部**无 reranker + 阈值过滤**（对比 `hybrid_search` 有）——RBAC 路径的负例拒答能力本就弱于非 RBAC 路径。新 `_rbac_subindex_search` 保持与现 RBAC 路径一致（不加 reranker），避免混入未要求的改动。是否后续单独修（让 RBAC 路径也走 reranker），由用户决定。
- **BM25 每查询重建**（`docs/源码精讲与面试深度拷问.md:285` 已记）：O(N) per call，已知 Minor，本轮保持。
- **`reconstruct_n` 前置条件**：依赖全库索引是 flat（支持精确 reconstruct）。项目用 `IndexFlatIP` 满足；若未来换压缩索引（IVF_PQ 等），reconstruct 变近似，本切片法需改用重新 embed——届时再评估。

---

## 12. 面试话术（修复后版本）

> 「`permission_aware_search` 原本声称先过滤后检索，但向量路径是全库 `as_retriever` 再 Python 过滤——先检索后过滤，top-k 名额会被别的部门占满导致本部门结果被挤掉。我用**按部门子索引**修成真·先过滤：admin 仍走全库保全局最优排序，普通用户只在权限域子索引里搜，财务向量从不被碰到。关键是子索引**从全库向量 `reconstruct_n` 切片而来，零重复嵌入**，所以没有额外 embedding 成本，也没引入 Milvus 这类支持原生过滤的向量库——FAISS 自己就够。」

---

## 实施交接

本 spec 评审通过后，进入 `superpowers:writing-plans` 生成任务级实施计划（TDD，含完整代码的 bite-sized 任务）。
