# GroundRAG Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing GroundRAG search and chat workflow deterministic, consistently reranked, failover-capable, and fully covered by standard backend and frontend test commands.

**Architecture:** Preserve the public API and existing single-process design. Consolidate retrieval finalization behind one private helper, compare complete deterministic chunk manifests before rebuilding FAISS, and make tests consume the same typed stream contract as production.

**Tech Stack:** Python 3.12+, pytest, FastAPI, SlowAPI, LangChain FAISS/BM25, React 19, TypeScript, Vitest, Vite.

---

### Task 1: Repair LLM failover ordering

**Files:**
- Modify: `rag_modules/llm_fallback.py:79-145`
- Test: `tests/test_core.py`

- [ ] **Step 1: Extend the failing scheduler tests**

Add a regression proving a cooled primary is skipped in favor of the healthy backup and retain the existing primary-failure and all-failure cases:

```python
def test_cooling_primary_is_skipped(self):
    primary = RecordingLLM("primary", error=RuntimeError("down"))
    backup = RecordingLLM("backup", result="ok")
    llm = FallbackLLM([primary, backup], cooldown_seconds=60)
    assert llm.invoke("first") == "ok"
    assert llm.invoke("second") == "ok"
    assert primary.calls == 1
    assert backup.calls == 2
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_core.py::TestFallbackLLM -q`

Expected: failure at `rag_modules/llm_fallback.py` because `_ordered_llms()` references undefined `lls`.

- [ ] **Step 3: Implement the minimal scheduler correction**

Iterate the local `llms` alias in both healthy and cooling passes. Do not change the public constructor or exception contract.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_core.py::TestFallbackLLM -q`

Expected: all fallback tests pass.

### Task 2: Align API tests with typed streaming and isolate limits

**Files:**
- Modify: `tests/test_core.py`
- Modify: `tests/test_api_streaming.py`
- Modify: `app/api.py` only if a production defect is exposed by the corrected tests

- [ ] **Step 1: Replace obsolete `ask()` fakes with typed `ask_stream()` fakes**

Each fake records request identity and yields the production event sequence:

```python
def ask_stream(self, question, user_departments=None, user_id=None):
    recorded.append({"user_id": user_id, "user_departments": user_departments})
    yield {"type": "sources", "items": []}
    yield {"type": "token", "text": "ok"}
    yield {"type": "trace", "trace": {"trace_id": "test"}}
    yield {"type": "done"}
```

- [ ] **Step 2: Add an autouse limiter reset fixture**

Clear SlowAPI's in-memory storage before and after each API test so login calls cannot leak across clients.

- [ ] **Step 3: Verify the original failures are GREEN**

Run: `python -m pytest tests/test_core.py::TestAPIConcurrencyFix tests/test_api_streaming.py -q`

Expected: all selected tests pass without `AttributeError` or `429`.

### Task 3: Apply reranking consistently to RBAC retrieval

**Files:**
- Modify: `rag_modules/retrieval_optimization.py:430-541`
- Modify: `rag_modules/retrieval_optimization.py:603-671`
- Test: `tests/test_rbac_subindex.py`

- [ ] **Step 1: Add failing RBAC reranker tests**

Use a fake reranker that records candidates, assigns deterministic scores, and returns a reordered list. Assert that only authorized candidates reach it and that candidates below `rerank_threshold` are removed.

```python
assert reranker.seen_departments <= {"HR", "公共"}
assert [doc.metadata["rerank_score"] for doc in result] == [0.9]
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_rbac_subindex.py -q`

Expected: RBAC results retain RRF order and low-scored candidates because `_rbac_subindex_search()` bypasses the reranker.

- [ ] **Step 3: Extract one retrieval finalization helper**

Create `_finalize_candidates(query, candidates, top_k, apply_threshold=True)` that performs MMR, invokes the configured reranker, applies `self.rerank_threshold`, and falls back to the fused candidates when reranking raises.

- [ ] **Step 4: Use the helper from global and RBAC paths**

Keep authorization before candidate construction. Do not pass unauthorized chunks to the helper.

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/test_rbac_subindex.py tests/test_core.py::TestRetrievalWiring -q`

Expected: selected retrieval tests pass.

### Task 4: Make index change detection complete and deterministic

**Files:**
- Modify: `rag_modules/index_construction.py:198-289`
- Test: `tests/test_core.py` or create `tests/test_index_incremental.py`

- [ ] **Step 1: Add failing unchanged-run and deletion tests**

Use a temporary index directory and a lightweight fake vectorstore/build method. Assert:

```python
_, first_changed = module.build_incremental(chunks)
_, second_changed = module.build_incremental(equivalent_chunks)
assert first_changed is True
assert second_changed is False

_, deleted_changed = module.build_incremental(chunks[:-1])
assert deleted_changed is True
```

Also assert the manifest contains one entry per input chunk, including PDF/XLSX chunks whose source and `chunk_index` collide.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_index_incremental.py -q`

Expected: duplicate `source#chunk_index` keys collapse and deletion-only changes are missed.

- [ ] **Step 3: Build deterministic unique manifest keys**

Normalize source separators and assign a per-source occurrence counter around the declared `chunk_index`, producing keys that remain stable for equivalent ordered chunks while preserving duplicate pages/rows.

- [ ] **Step 4: Detect additions, modifications, and deletions**

Set `has_changes` when `current_hashes != old_hashes`. Rebuild with all current chunks for any manifest difference, including deletion-only changes. Persist the complete manifest atomically after a successful build/reuse decision.

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/test_index_incremental.py -q`

Expected: unchanged, duplicate-key, and deletion cases pass.

### Task 5: Add the standard frontend test entrypoint

**Files:**
- Modify: `web/package.json:6-11`

- [ ] **Step 1: Verify the missing command**

Run: `npm test -- --run` from `web/`.

Expected: npm reports `Missing script: "test"`.

- [ ] **Step 2: Add the script**

```json
"test": "vitest run"
```

- [ ] **Step 3: Verify GREEN**

Run: `npm test` from `web/`.

Expected: one test file and five tests pass.

### Task 6: Full verification

**Files:**
- Review all modified files

- [ ] **Step 1: Run backend tests**

Run: `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 2: Run frontend tests, lint, and production build**

Run from `web/`: `npm test`, `npm run lint`, and `npm run build`.

Expected: tests and build exit zero; lint has no errors.

- [ ] **Step 3: Inspect the patch**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors and only planned files changed.
