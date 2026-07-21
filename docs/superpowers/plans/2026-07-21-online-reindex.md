# Online Reindex Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep an existing GroundRAG instance queryable during background reindexing, expose degraded failures without discarding the active index, and provide an idempotent local demo-data command.

**Architecture:** Preserve `_rag` as the active immutable reference while a replacement `EnterpriseRAGSystem` is constructed in `asyncio.to_thread`. Status and readiness distinguish active-service health from the background rebuild result. The existing sample-document generator remains the single source of demo data, with a small wrapper command and Make target for reviewer ergonomics.

**Tech Stack:** Python 3.12, FastAPI, asyncio, pytest, React 19, TypeScript, Vitest, GNU Make.

---

### Task 1: Online Reindex State Semantics

**Files:**
- Modify: `tests/test_api_runtime.py`
- Modify: `app/api.py`

- [ ] **Step 1: Write failing readiness and failure-preservation tests**

Add tests which set `_rag` to a sentinel while `_rag_status` is `reindexing` or `degraded`, assert `/ready` returns 200 with `serving: true`, and patch `_build_rag` to raise during `_rebuild_rag()` while asserting the sentinel remains active and status becomes `degraded`.

- [ ] **Step 2: Verify the new tests fail for the intended behavior**

Run: `python -m pytest tests/test_api_runtime.py -q`

Expected: readiness returns 503 for the new states and rebuild failure reports `error` instead of `degraded`.

- [ ] **Step 3: Implement the minimal state transition changes**

Change rebuild state from `indexing` to `reindexing`. Make `_rebuild_rag()` assign the replacement only after successful construction and select `degraded` on failure when `_rag` is still available. Make `/ready` depend on whether `_rag` exists, returning the detailed status plus `serving: true` when available.

- [ ] **Step 4: Run focused and document API tests**

Run: `python -m pytest tests/test_api_runtime.py tests/test_documents_api.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit backend lifecycle behavior**

```bash
git add app/api.py tests/test_api_runtime.py tests/test_documents_api.py
git commit -m "feat: keep active RAG online during reindex"
```

### Task 2: Knowledge Base Degraded UI

**Files:**
- Modify: `web/src/i18n/index.tsx`
- Modify: `web/src/views/KnowledgeBaseView.tsx`
- Create: `web/src/views/__tests__/KnowledgeBaseView.test.tsx`

- [ ] **Step 1: Write a failing degraded-state component test**

Render `KnowledgeBaseView` with mocked API responses `{status: "degraded", error: "rebuild failed"}` and an administrator user. Assert that the error is visible and the retry button remains enabled.

- [ ] **Step 2: Verify the component test fails**

Run: `npm test -- src/views/__tests__/KnowledgeBaseView.test.tsx`

Expected: FAIL because degraded status/error rendering is absent.

- [ ] **Step 3: Implement explicit state presentation**

Add translated labels for `reindexing` and `degraded`, render the last error for degraded state, disable mutations only for `reindexing`/`initializing`, and leave admin retry enabled for degraded state.

- [ ] **Step 4: Run frontend tests and production build**

Run: `npm test && npm run build`

Expected: all tests pass and Vite produces `dist/`.

- [ ] **Step 5: Commit UI state handling**

```bash
git add web/src/i18n/index.tsx web/src/views/KnowledgeBaseView.tsx web/src/views/__tests__/KnowledgeBaseView.test.tsx
git commit -m "feat: show degraded index status"
```

### Task 3: Idempotent Demo Command

**Files:**
- Create: `tests/test_demo_setup.py`
- Create: `scripts/demo_setup.py`
- Modify: `Makefile`
- Modify: `README.md`
- Modify: `README.zh-CN.md`

- [ ] **Step 1: Write a failing demo command test**

Test `prepare_demo(root)` against a temporary directory. Assert two calls succeed, generated sample documents exist, and returned prompts include HR, finance, IT, and administrator walkthrough entries.

- [ ] **Step 2: Verify the demo test fails**

Run: `python -m pytest tests/test_demo_setup.py -q`

Expected: collection fails because `scripts.demo_setup` does not exist.

- [ ] **Step 3: Implement the demo wrapper**

Expose `prepare_demo(root)` which invokes the existing generator with the supplied output directory and returns a fixed reviewer checklist. Its CLI writes to configured `data/docs`, prints created/existing counts, then prints the four walkthrough prompts.

- [ ] **Step 4: Wire and document the command**

Add `make demo` and a Quick Demo section to both READMEs showing `make setup`, `make demo`, `make run`, user selection, and expected RBAC observations.

- [ ] **Step 5: Verify the command and docs**

Run: `python -m pytest tests/test_demo_setup.py -q`, `python scripts/demo_setup.py --help`, and `git diff --check`.

Expected: test passes, CLI help exits 0, and no whitespace errors are reported.

- [ ] **Step 6: Commit the demo workflow**

```bash
git add scripts/demo_setup.py tests/test_demo_setup.py Makefile README.md README.zh-CN.md
git commit -m "feat: add one-command demo setup"
```

### Task 4: Full Verification And Live Showcase QA

**Files:**
- Modify only if verification exposes a scoped defect.

- [ ] **Step 1: Run backend verification**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run frontend verification**

Run: `npm test`, `npm run build`, and `npm run lint` from `web/`.

Expected: tests and build pass; lint has no errors.

- [ ] **Step 3: Verify repository hygiene**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors and no unintended files.

- [ ] **Step 4: Run live browser checks**

Start the FastAPI showcase server, open Chat and Knowledge Base, verify user switching, status rendering, and absence of browser console errors. Exercise a rebuild while confirming `/ready` continues returning 200 when an active RAG exists.

- [ ] **Step 5: Record the verified checkpoint**

Commit only scoped fixes found during verification, then report commands, counts, remaining lint warnings, branch, commits, and local URL.
