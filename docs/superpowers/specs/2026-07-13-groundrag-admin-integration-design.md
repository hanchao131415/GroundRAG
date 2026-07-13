# GroundRAG Admin Integration Design

## 1. Goal

Build a new, independently deployable portfolio project that combines the management capabilities of `vue-fastapi-admin` with the RAG pipeline from `GroundRAG`.

The target is a production-oriented demonstration deployment for a single Linux server using Docker Compose. It should reliably support 1-5 concurrent users without introducing infrastructure that is unnecessary at this scale.

The new project will use:

- `vue-fastapi-admin` as the FastAPI application and Vue 3 administration shell.
- `GroundRAG` as a RAG domain module inside the same FastAPI process.
- SQLite for users, authorization, knowledge metadata, jobs, and audit metadata.
- FAISS and filesystem volumes for vector indexes, source documents, cache data, and local traces.
- A single Vue 3 + Naive UI frontend for both administration and RAG user workflows.

The two source repositories remain unchanged and serve as migration sources. The integrated application is created as a separate repository, provisionally named `groundrag-admin` under `F:/code/python/groundrag-admin`.

## 2. Scope

### Included

- Password login, JWT authentication, users, roles, departments, menus, API permissions, and audit metadata.
- Vue pages for chat, search, knowledge management, indexing jobs, and RAG statistics.
- GroundRAG document parsing, hybrid retrieval, reranking, generation, caching, tracing, and evaluation logic.
- Department-aware knowledge access and multi-department users.
- SQLite migrations, durable Docker volumes, health checks, backup and restore commands.
- In-process indexing jobs with durable job state and atomic index activation.
- Request timeouts, bounded retries, RAG concurrency limits, and predictable overload responses.
- Backend, frontend, integration, deployment, and small-load verification.

### Excluded from the first release

- PostgreSQL, Redis, Celery, Kubernetes, and multi-node deployment.
- Object storage and distributed vector databases.
- OAuth, LDAP, SSO, and refresh-token rotation.
- Visual workflow builders, agents, web search, and plugin marketplaces.
- Real-time multi-user collaboration.

These exclusions keep the application credible for its expected load while preserving clear upgrade paths.

## 3. Product Positioning

The project does not attempt to match the breadth of Dify, RAGFlow, AnythingLLM, or Open WebUI.

- Dify uses multiple application, worker, database, cache, proxy, sandbox, and plugin services to provide a general AI application platform.
- RAGFlow emphasizes complex document understanding and large RAG workflows, with materially higher self-hosting requirements.
- AnythingLLM and Open WebUI emphasize easy self-hosting, multi-user access, persistent configuration, provider flexibility, and administration.

GroundRAG Admin will instead demonstrate a smaller, explainable architecture: transparent hybrid retrieval, grounded citations, department isolation, complete administration, and reliable single-server operation.

## 4. Architecture

One FastAPI application owns authentication, administration, and RAG APIs. It loads the active RAG runtime once per process and exposes it through an application service boundary.

```text
Browser
  |
  v
Caddy
  |-- /            Vue 3 application
  `-- /api/*       FastAPI application
                     |-- administration APIs
                     |-- authentication and RBAC
                     `-- RAG APIs
                           |-- ingestion and indexing
                           |-- hybrid retrieval and reranking
                           `-- streaming generation

Persistent data
  |-- SQLite: identity, RBAC, metadata, jobs, audit summaries
  |-- documents volume: uploaded source files
  |-- indexes volume: versioned FAISS indexes
  |-- cache volume: answer cache
  `-- traces volume: local diagnostic traces
```

### Backend module boundaries

- `app/api`, `app/models`, `app/controllers`, and `app/core` retain the management template responsibilities.
- `app/rag/api` contains HTTP schemas and routers only.
- `app/rag/services` coordinates authorization, runtime access, indexing jobs, and response mapping.
- `app/rag/modules` contains migrated GroundRAG parsing, indexing, retrieval, reranking, generation, caching, and tracing code.
- RAG modules do not import FastAPI request objects or management controllers.
- The service layer translates authenticated users and database records into department filters understood by retrieval.

### Frontend boundaries

The React application is not migrated. Its user-visible behavior is reimplemented in the existing Vue 3 + Naive UI application.

Dynamic menus provide these views:

- Intelligent chat
- Knowledge search
- Knowledge bases and documents
- Indexing jobs
- RAG usage and status
- Existing system management views

Ordinary users see chat and search. Authorized knowledge managers see knowledge and indexing views. Administrators also see system management and audit views.

## 5. Data Model

Existing management models are retained: `User`, `Role`, `Menu`, `Api`, `Dept`, and `AuditLog`.

### Authorization changes

- Replace the single `User.dept_id` relationship with a many-to-many user-to-department relationship.
- Preserve role-to-menu and role-to-API relationships.
- Grant knowledge visibility from the authenticated user's department set.
- Treat a superuser as globally authorized without using wildcard department rows.
- Remove all development-token and default-password bypasses.

### New models

`KnowledgeBase`

- Name and description
- Enabled state
- Active index version
- Last successful indexing time
- Created and updated timestamps

`KnowledgeDocument`

- Knowledge base reference
- Owning department reference
- Original filename, storage key, media type, and byte size
- SHA-256 content hash
- Parsing/indexing status and sanitized error summary
- Created and updated timestamps

`IndexJob`

- Knowledge base reference
- Requested-by user reference
- State: queued, running, succeeded, or failed
- Target index version
- Progress counters and sanitized error summary
- Queued, started, and completed timestamps

`RAGQueryLog`

- User reference and trace ID
- Question hash and short sanitized summary
- Duration, token counts, estimated cost, source hit count, and status
- Created timestamp

Full answers, retrieved document text, passwords, JWTs, API keys, and SSE response bodies are never stored in audit tables.

## 6. Authentication and Authorization

- Login uses the template password hashing implementation after its dependencies and parameters are reviewed and locked.
- JWTs are sent through the standard `Authorization: Bearer` header.
- Production startup rejects a missing, default, or weak signing secret.
- Access tokens use a short configurable lifetime. The first release does not implement refresh tokens.
- Inactive users are rejected during authentication and on protected requests.
- API authorization matches normalized route templates and methods, not raw URLs with literal path parameters.
- RAG access requires both API permission and knowledge department permission.
- Login and expensive RAG endpoints receive separate rate limits.

## 7. RAG Runtime and Request Flow

At startup, the application initializes SQLite, verifies migration state, and attempts to load embedding, reranker, and the active FAISS index. Management endpoints remain available if RAG initialization fails. RAG readiness reports the failure and RAG endpoints return `503` until recovery.

A chat request follows this sequence:

1. Validate the JWT and load the active user.
2. Verify route-level API permission.
3. Resolve the user's authorized departments.
4. Acquire a bounded RAG concurrency slot.
5. Retrieve only authorized content, rerank it, and emit source metadata.
6. Stream generation events to the client.
7. Cancel generation if the client disconnects.
8. Record sanitized query metrics and release the slot.

The application applies separate connection, first-token, and total request deadlines to external LLM calls. It retries only explicitly transient network errors and upstream rate limiting, using a small bounded retry policy. Capacity exhaustion returns `429` with `Retry-After`; unavailable RAG dependencies return `503`.

## 8. Document Ingestion and Index Lifecycle

The server validates filename, media type, extension, size, and content hash. Files are stored under generated storage keys rather than user-supplied paths.

Index building uses a durable `IndexJob` record and one in-process worker task. Only one indexing job runs at a time. This is sufficient for the single-process, single-server target and avoids Redis or Celery.

The worker builds a complete candidate index in a versioned temporary directory. On success it:

1. Flushes and validates all candidate index files.
2. Atomically promotes the candidate directory to a versioned index directory.
3. Updates the active version in a short SQLite transaction.
4. Swaps the in-memory runtime under a synchronization boundary.

On failure, the previous index remains active. Jobs left in `running` state after process restart are marked failed with an interruption reason and can be retried by an administrator.

## 9. SQLite Operations

- Use Tortoise ORM and Aerich from the management template.
- Enable WAL, foreign keys, and a configured busy timeout for every connection.
- Keep write transactions short and never hold a database transaction while parsing documents or calling an LLM.
- Run a single Uvicorn worker because the local model runtime and in-process indexing worker are process-local.
- Store the database on a dedicated Docker volume.
- Back up with SQLite's online backup mechanism rather than copying live database, WAL, and shared-memory files independently.
- Provide documented backup and restore commands covering SQLite, documents, and active index files.

## 10. Security Corrections to the Template

The integration must not inherit the template unchanged. It will:

- Remove the `token == "dev"` authentication bypass.
- Remove automatic `admin / 123456` creation.
- Require explicit bootstrap administrator credentials or a one-time initialization command.
- Stop returning internal exception representations to clients.
- Redact secrets and RAG content from audit logs.
- Replace broad CORS defaults with an explicit deployment origin.
- Add request body limits, upload limits, and security response headers at Caddy and application boundaries.
- Run the container as non-root with only required writable volumes.
- Disable or protect interactive API documentation in production.

## 11. Deployment

Docker Compose contains two services:

- `app`: FastAPI plus the built Vue static assets and local RAG models.
- `caddy`: TLS termination, reverse proxy, compression, request limits, and security headers.

The application container has health checks, a restart policy, memory and CPU constraints, log rotation, and dedicated volumes for each persistent data class. It exposes liveness separately from readiness so a broken RAG index does not create a restart loop that blocks administration.

## 12. Observability and Failure Handling

- Emit structured JSON logs to stdout with request ID, trace ID, route, status, and duration.
- Preserve GroundRAG step timing and token metrics without logging complete sensitive prompts or document chunks.
- Store compact query summaries in SQLite and detailed local traces in a bounded, rotated volume.
- Return stable error codes and user-safe messages for authentication, authorization, overload, upstream timeout, RAG unavailability, and indexing failure.
- Expose `/health/live` for process health and `/health/ready` for database and RAG readiness.

Langfuse remains optional. Its failure never blocks RAG requests.

## 13. Testing

### Unit tests

- Password, JWT, inactive user, and secret validation behavior
- Route-template API authorization
- User-to-department permission translation
- Upload validation and content hashing
- Retry classification, concurrency limiting, and error mapping
- Index version promotion and rollback

### API tests

- Login and management RBAC
- Chat, search, stats, and knowledge endpoints
- Department isolation and multi-department access
- SSE event order, client disconnect cancellation, and sanitization
- `429`, `Retry-After`, `503`, and timeout behavior

### Integration tests

- SQLite, Tortoise, and Aerich migrations
- A small real FAISS index and department-filtered retrieval
- Successful index activation, failed build rollback, and interrupted-job recovery
- Persistent data after application restart

### Frontend tests

- Login and dynamic menus
- Streaming chat and source citations
- Search results and permission-dependent navigation
- Upload validation, index job status, and failure presentation

### Deployment verification

- Reproducible Docker Compose build and startup
- Liveness/readiness behavior
- Backup and restore drill
- Locust smoke test with 1-5 concurrent users
- Dependency, lint, type, test, and build checks in CI

## 14. Acceptance Criteria

- A documented Docker Compose command starts the application on a clean Linux server.
- No development token, default password, or default signing secret permits access.
- Users, documents, job records, and the active index survive container restart.
- Unauthorized departments never appear in retrieval results or citations.
- Five concurrent demonstration users cannot crash the application; excess expensive work is rejected predictably.
- A failed index build leaves the last successful index available.
- RAG failure does not prevent administrators from logging in and repairing the knowledge base.
- Backend tests, frontend tests, static checks, image build, health checks, and the small-load smoke test pass.

## 15. Migration Strategy

Implementation proceeds in vertical increments:

1. Create the new repository from a clean copy of the management template and establish dependency compatibility.
2. Apply mandatory template security corrections and baseline tests.
3. Add RAG data models, migrations, permissions, and menu seeds.
4. Migrate GroundRAG modules behind a framework-independent runtime service.
5. Add protected RAG APIs and department-aware retrieval.
6. Rebuild chat, search, status, knowledge, and job views in Vue.
7. Add durable indexing, atomic activation, backup, health checks, and production Compose configuration.
8. Complete integration, security, restart, and 1-5 user load verification.

Each increment must remain runnable and independently testable. The source repositories are not modified during migration.
