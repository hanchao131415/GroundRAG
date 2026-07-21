# GroundRAG Runtime Experience Design

## Goal

Keep the HTTP process responsive while heavyweight RAG components initialize, expose explicit readiness, bound SSE buffering, propagate client cancellation, and show actionable frontend errors.

## Runtime State

FastAPI lifespan starts RAG initialization in a background task backed by `asyncio.to_thread`. State is one of `initializing`, `ready`, or `error`, with a sanitized error message. `/health` is a liveness endpoint and never initializes RAG. `/ready` returns `200` only for `ready` and `503` otherwise.

Routes that require `user_service`, retrieval, or generation call a single readiness guard. A test-injected `_rag` remains accepted to preserve isolated API tests. During initialization or after failure, the guard returns structured `503` JSON instead of blocking the request.

## SSE Bridge

The synchronous `ask_stream()` producer runs in a daemon thread and writes to `queue.Queue(maxsize=32)`. Blocking `put` supplies backpressure. A `threading.Event` is checked by the producer and set when the async response is cancelled or the client disconnects. The async generator reads through `asyncio.to_thread`, emits the existing event protocol, and always releases the producer in `finally`.

## Frontend

The API client raises `ApiError` containing HTTP status and backend detail. A `401` clears the stored token. Status-specific messages distinguish authentication, rate limiting, readiness, and generic server failure.

`useSSEChat` exposes `stop()`. Aborting is treated as an intentional stop rather than an error. The chat command changes from Send to Stop while generation is active, and the assistant retains any partial answer.

## Verification

- `/health` returns immediately without calling RAG initialization.
- `/ready` and protected RAG-dependent routes return structured `503` while initializing or failed.
- A ready injected RAG preserves existing login/search/chat behavior.
- The SSE queue has a fixed capacity and cancellation signals the producer.
- Frontend API errors retain status/detail and clear auth on `401`.
- Stop aborts the active request without adding an error message.
- Backend tests, frontend tests, lint, and production build pass.
