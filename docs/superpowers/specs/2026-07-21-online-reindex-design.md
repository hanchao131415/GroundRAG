# GroundRAG Online Reindex Design

## Goal

Keep the showcase query path available while documents are reindexed, without adding Redis, Celery, a database, or another deployable service. A failed rebuild must not replace a working RAG instance.

## Scope

This iteration covers the in-process index lifecycle and a deterministic demo-data command. It does not provide durable job recovery across process restarts, distributed locking, multi-process index coordination, or a production authentication system.

## Runtime State

The API exposes four states:

- `initializing`: no usable RAG instance exists and startup is still loading.
- `ready`: the active RAG instance is current.
- `reindexing`: an active RAG instance remains queryable while a replacement is built.
- `degraded`: the active RAG instance remains queryable, but the most recent rebuild failed.
- `error`: startup failed and no usable RAG instance exists.

`/health` remains a lightweight process liveness probe. `/ready` returns 200 whenever an active RAG instance exists, including `reindexing` and `degraded`, and returns 503 only when no usable instance exists. `/api/v1/index-status` reports the detailed state and last error for the UI.

## Reindex Flow

Upload and delete endpoints validate authorization and mutate the document filesystem before scheduling one background rebuild. A process-local task guard rejects overlapping mutations with `409 INDEX_BUSY`.

The rebuild constructs a new `EnterpriseRAGSystem` in a worker thread. The current `_rag` reference is never cleared. After construction succeeds, assignment of the new instance replaces the old reference in one event-loop step. If construction fails, the old reference remains active and the status becomes `degraded`. If no old instance exists, failure produces `error`.

The implementation stays deliberately single-process. Deployment documentation must continue to run one FastAPI worker because independent workers would have independent task guards and RAG instances.

## Demo Data

An idempotent command prepares the repository's existing sample documents and prints a short demonstration checklist covering HR, finance, IT, and administrator visibility. The command reuses existing sample-data generation rather than introducing a second document fixture format. `make demo` invokes this command and documents the next startup step.

## API And UI Behavior

- Chat and search continue to use the active RAG instance during `reindexing` and `degraded`.
- Knowledge Base displays `reindexing` distinctly from initial loading.
- A degraded status shows the last rebuild error and allows an administrator to retry.
- Upload, delete, and manual reindex remain disabled while a rebuild is active.
- No new external service or browser-visible setup step is introduced.

## Failure Handling

- Replacement construction failure preserves the prior RAG object.
- Concurrent rebuild requests return structured `409 INDEX_BUSY`.
- Startup failure without a prior RAG object remains unavailable and reports `error`.
- Process termination may interrupt a rebuild; after restart, normal startup reconstructs from the document directory.

## Verification

Backend regression tests must first demonstrate that the current implementation makes `/ready` unavailable during indexing and loses the correct status after rebuild failure. The implementation then makes these tests pass and retains existing API, retrieval, and document tests.

Frontend tests cover status rendering and retry availability. Final verification runs the full pytest suite, frontend unit tests, TypeScript production build, lint, `git diff --check`, and live browser checks of chat and Knowledge Base views.
