# GroundRAG Stability Design

## Goal

Make the existing RAG workflow reliable for a portfolio demonstration without expanding the product surface. A successful change leaves search and chat behavior consistent across RBAC roles, avoids unnecessary index rebuilds, provides working LLM failover, and gives the repository a repeatable green test command for both backend and frontend.

## Scope

This iteration contains five changes:

1. Repair the fallback LLM scheduler and cover primary failure, total failure, and ordering behavior.
2. Apply the existing cross-encoder reranker and relevance threshold to RBAC sub-index retrieval, matching the global retrieval tail.
3. Make chunk identity deterministic so an unchanged second startup performs zero embeddings and does not clear the answer cache.
4. Align API test doubles with the current typed event stream contract and isolate SlowAPI rate-limit state between tests.
5. Add the missing frontend `test` script and keep the existing SSE parser tests in the standard verification path.

Document upload, background indexing, health endpoint redesign, SSE cancellation/backpressure, and new UI screens are explicitly deferred to later iterations.

## Architecture

### LLM fallback

`FallbackLLM` remains the single failover boundary. `_ordered_llms()` yields the last successful healthy provider first, then other healthy providers, then cooling providers. Both `invoke()` and `stream()` consume that ordering. A provider failure updates only that provider's cooldown and never prevents the scheduler from visiting the remaining providers.

### Retrieval

Authorization continues to happen before retrieval through per-department FAISS indexes. The RBAC path will use the same post-retrieval sequence as global search:

```text
authorized vector candidates + authorized BM25 candidates
  -> RRF fusion
  -> MMR deduplication
  -> cross-encoder reranking
  -> configured relevance threshold
  -> top-k
```

The implementation will share a small private finalization helper so global and RBAC paths cannot drift again. If the reranker raises an operational error, retrieval degrades to the fused candidates instead of failing the request, preserving current behavior.

### Index identity

Every current chunk receives a stable key derived from normalized source path plus a deterministic per-source chunk ordinal. The persisted manifest contains one entry per chunk and compares normalized text content hashes. Random `parent_id` and `chunk_id` metadata are excluded from change detection.

Deleted keys count as changes as well as added or modified keys. Any real change still triggers a full FAISS rebuild because the current `IndexFlatIP` strategy does not safely delete stale vectors. An unchanged run reuses the saved index, preserves cache entries, and writes no new embeddings.

### API and frontend test contract

API test doubles implement `ask_stream()` and emit `sources`, `token`, `trace`, and `done`, matching production. Rate limiter state is reset for each API test so test ordering cannot cause `429` responses.

The frontend exposes `npm test` as a non-watch Vitest run. No visual behavior changes are included in this iteration.

## Error Handling

- Fallback providers retain their original exception as the cause of the final aggregate failure.
- Reranker failures are logged and return fused authorized candidates.
- Missing or corrupt index manifests cause a safe rebuild rather than partial index reuse.
- API streaming errors remain typed `error` events followed by `trace` and `done`.

## Verification

The implementation is accepted when all of the following hold:

- A failing primary LLM successfully switches to a healthy backup.
- Two failing providers raise the aggregate `RuntimeError`, not `NameError`.
- An HR query is reranked only among HR and public candidates.
- RBAC retrieval applies the configured rerank threshold.
- Running incremental index construction twice with identical chunks performs zero embeddings on the second run.
- Deleting a chunk triggers a rebuild and removes it from the persisted index.
- API tests do not depend on execution order or shared login-rate state.
- `python -m pytest -q` passes.
- `npm test`, `npm run lint`, and `npm run build` pass from `web/`.

## Compatibility

Public HTTP routes and JSON/SSE event shapes remain unchanged. Existing environment variables remain valid. Existing FAISS indexes without a complete manifest rebuild once and then enter the stable no-change path.
