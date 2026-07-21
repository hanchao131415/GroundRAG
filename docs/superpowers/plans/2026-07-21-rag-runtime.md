# GroundRAG Runtime Experience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make startup, readiness, streaming cancellation, and frontend failure states predictable without changing the public chat event schema.

**Architecture:** Background initialization feeds a small process-level state machine. RAG routes share one readiness guard; SSE uses a bounded cross-thread queue and cancellation event; frontend errors use a typed `ApiError`.

**Tech Stack:** FastAPI, asyncio, threading, queue, pytest, React, TypeScript, Vitest.

---

### Task 1: Runtime readiness

- [ ] Add failing tests for lightweight `/health`, `503 /ready`, and a protected route while not ready.
- [ ] Implement background lifespan initialization and a shared readiness guard.
- [ ] Run `python -m pytest tests/test_api_runtime.py tests/test_api_streaming.py -q`.

### Task 2: Bounded cancellable SSE

- [ ] Add a failing unit test for fixed queue capacity and cancellation propagation.
- [ ] Extract the producer bridge and use it from `/api/v1/chat`.
- [ ] Run focused streaming tests.

### Task 3: Typed frontend errors and Stop

- [ ] Add failing Vitest cases for backend detail/status and `401` token clearing.
- [ ] Implement `ApiError`, response parsing, abort-aware hook behavior, and Stop control.
- [ ] Run `npm test`, `npm run lint`, and `npm run build`.

### Task 4: Full verification

- [ ] Run `python -m pytest -q`.
- [ ] Run all frontend verification commands.
- [ ] Run `git diff --check` and review the final patch.
