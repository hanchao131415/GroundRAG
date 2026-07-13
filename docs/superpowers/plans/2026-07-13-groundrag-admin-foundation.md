# GroundRAG Admin Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `F:/code/python/groundrag-admin` as a secure, tested Python 3.12 + FastAPI 0.115.x + Vue 3 administration application that supports local hot-reload development and a two-container Caddy production deployment.

**Architecture:** Start from the local `vue-fastapi-admin` repository and preserve its Tortoise/Aerich RBAC model and Vue management UI. Upgrade and secure the application before migrating any GroundRAG code. Local development runs Vite and Uvicorn directly; production serves Vue assets from Caddy and reverse-proxies `/api/*` to a private FastAPI container.

**Tech Stack:** Python 3.12, FastAPI 0.115.x, Pydantic Settings, Tortoise ORM 0.23.x, Aerich, SQLite WAL, PyJWT, Passlib/Argon2id, pytest, HTTPX, Vue 3, Vite, pnpm, Caddy 2, Docker Compose

---

## Scope and Follow-up Plans

This is plan 1 of 5. It produces a runnable and deployable management foundation without RAG behavior.

1. **This plan:** repository, dependency, security, development, deployment, and CI foundation.
2. **Plan 2:** RAG database models, Aerich migrations, multi-department authorization, and menu seeds.
3. **Plan 3:** migrate GroundRAG runtime, bounded executors, protected APIs, SSE, and atomic index activation.
4. **Plan 4:** Vue chat, search, knowledge, indexing-job, and statistics views.
5. **Plan 5:** backup/restore, observability, restart tests, security checks, and 1-5 user load validation.

Do not copy `GroundRAG/rag_modules`, add RAG dependencies, or create RAG routes during this plan.

## Target File Map

Files are relative to `F:/code/python/groundrag-admin`.

- `pyproject.toml`: direct backend dependencies, Python 3.12 requirement, pytest and Ruff configuration.
- `uv.lock`: reproducible resolved backend environment.
- `.env.example`: non-secret development and production configuration contract.
- `app/settings/config.py`: typed settings and production fail-fast validation.
- `app/core/database.py`: SQLite connection initialization and required pragmas.
- `app/core/dependency.py`: Bearer JWT authentication and current-user loading.
- `app/core/init_app.py`: application middleware, router, bootstrap, and initialization wiring.
- `app/core/middlewares.py`: bounded/redacted audit behavior and streaming exclusions.
- `app/core/exceptions.py`: stable client-safe exception responses.
- `app/api/v1/base/base.py`: login, user information, and password change endpoints.
- `app/api/v1/health.py`: liveness and database readiness endpoints.
- `app/commands/create_admin.py`: explicit initial-administrator command.
- `web/src/utils/http/interceptors.js`: standard Bearer header injection.
- `web/vite.config.js`: local `/api` proxy.
- `Makefile`: reproducible local and production commands.
- `Dockerfile.app`: Python application image only.
- `Dockerfile.caddy`: Vue build plus Caddy runtime image.
- `deploy/Caddyfile`: static SPA hosting and `/api/*` reverse proxy.
- `docker-compose.yml`: private app service, public Caddy service, health checks, limits, and volumes.
- `tests/`: backend unit/API tests introduced before each behavior change.
- `.github/workflows/ci.yml`: backend checks, frontend checks, and image build.

## Task 1: Create the Independent Repository

**Files:**
- Create repository: `F:/code/python/groundrag-admin`
- Create: `docs/design/groundrag-admin-integration.md`
- Modify: `README.md`

- [ ] **Step 1: Verify both source repositories are clean enough to use as read-only inputs**

Run:

```powershell
git -C F:\code\python\vue-fastapi-admin status --short
git -C F:\code\python\GroundRAG status --short
```

Expected: no uncommitted changes that need to be copied into the new repository. Do not clean or reset either source repository.

- [ ] **Step 2: Clone the management template locally and retain its origin as `template`**

Run:

```powershell
git clone F:\code\python\vue-fastapi-admin F:\code\python\groundrag-admin
git -C F:\code\python\groundrag-admin remote rename origin template
git -C F:\code\python\groundrag-admin switch -c codex/integration-foundation
```

Expected: the new repository is on `codex/integration-foundation`; both source repositories are unchanged.

- [ ] **Step 3: Copy the approved design into the new repository**

Run:

```powershell
New-Item -ItemType Directory -Force F:\code\python\groundrag-admin\docs\design
Copy-Item F:\code\python\GroundRAG\docs\superpowers\specs\2026-07-13-groundrag-admin-integration-design.md F:\code\python\groundrag-admin\docs\design\groundrag-admin-integration.md
```

Expected: the approved architecture is available inside the repository that will implement it.

- [ ] **Step 4: Replace the README opening and add explicit phase status**

Use this opening in `README.md`:

```markdown
# GroundRAG Admin

GroundRAG Admin combines a Vue 3 RBAC management application with an explainable,
department-isolated RAG pipeline. The current branch contains the secured management
foundation; RAG migration is implemented in later vertical phases.

## Development Status

- [x] Management template imported as an independent repository
- [ ] Production configuration and authentication baseline
- [ ] GroundRAG data and runtime integration
- [ ] Vue RAG workflows
- [ ] Deployment and load validation
```

- [ ] **Step 5: Commit the repository boundary**

Run:

```powershell
git add README.md docs/design/groundrag-admin-integration.md
git commit -m "chore: initialize GroundRAG Admin repository"
```

Expected: one commit containing only repository identity and the approved design.

## Task 2: Establish the Python 3.12 Dependency and Test Baseline

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`
- Create: `tests/test_app_smoke.py`
- Regenerate: `uv.lock`

- [ ] **Step 1: Write the application smoke test**

Create `tests/test_app_smoke.py`:

```python
from fastapi import FastAPI


def test_application_imports():
    from app import app

    assert isinstance(app, FastAPI)
    assert app.openapi_url == "/openapi.json"
```

Create `tests/conftest.py`:

```python
import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("SECRET_KEY", "test-only-secret-key-with-at-least-32-bytes")
os.environ.setdefault("DATABASE_PATH", ":memory:")
```

- [ ] **Step 2: Run the test with Python 3.12 and record the dependency failure**

Run:

```powershell
cd F:\code\python\groundrag-admin
uv python pin 3.12
uv run --python 3.12 pytest tests/test_app_smoke.py -v
```

Expected before dependency cleanup: FAIL during environment resolution, import, or startup. Preserve the exact failure in the task notes.

- [ ] **Step 3: Replace transitive dependency pins with a direct dependency set**

Set these `pyproject.toml` sections; keep existing formatter settings unless contradicted below:

```toml
[project]
name = "groundrag-admin"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "aerich>=0.8.1,<0.9",
    "aiosqlite>=0.20,<0.22",
    "argon2-cffi>=23.1,<26",
    "fastapi>=0.115,<0.116",
    "loguru>=0.7,<0.8",
    "passlib>=1.7.4,<2",
    "pydantic-settings>=2.7,<3",
    "pyjwt>=2.10,<3",
    "python-multipart>=0.0.20,<0.1",
    "tortoise-orm>=0.23,<0.24",
    "uvicorn[standard]>=0.34,<0.35",
]

[dependency-groups]
dev = [
    "httpx>=0.28,<0.29",
    "pytest>=8.3,<9",
    "pytest-asyncio>=0.25,<0.26",
    "ruff>=0.9,<0.10",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 120
target-version = "py312"
```

- [ ] **Step 4: Lock and verify the resolved compatibility set**

Run:

```powershell
uv lock --python 3.12
uv run python -c "import fastapi, starlette, tortoise, pydantic; print(fastapi.__version__, starlette.__version__, tortoise.__version__, pydantic.__version__)"
uv run pytest tests/test_app_smoke.py -v
uv run ruff check app tests
```

Expected: dependency resolution succeeds, the smoke test passes, and Ruff reports no errors. If Tortoise 0.23 is incompatible with the resolved FastAPI/Pydantic set, stop and resolve the compatibility set here rather than loosening upper bounds.

- [ ] **Step 5: Commit the dependency baseline**

```powershell
git add pyproject.toml uv.lock tests
git commit -m "build: establish Python 3.12 application baseline"
```

## Task 3: Add Typed Production Configuration and SQLite Pragmas

**Files:**
- Modify: `app/settings/config.py`
- Modify: `app/settings/__init__.py`
- Create: `app/core/database.py`
- Create: `.env.example`
- Create: `tests/test_settings.py`
- Create: `tests/test_database.py`

- [ ] **Step 1: Write failing settings and database tests**

Create `tests/test_settings.py`:

```python
import pytest
from pydantic import ValidationError

from app.settings.config import Settings


def test_production_rejects_default_secret(tmp_path):
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        Settings(APP_ENV="production", SECRET_KEY="change-me", DATABASE_PATH=tmp_path / "app.db")


def test_production_rejects_wildcard_cors(tmp_path):
    with pytest.raises(ValidationError, match="CORS_ORIGINS"):
        Settings(
            APP_ENV="production",
            SECRET_KEY="a" * 32,
            DATABASE_PATH=tmp_path / "app.db",
            CORS_ORIGINS=["*"],
        )
```

Create `tests/test_database.py`:

```python
import sqlite3

import pytest

from app.core.database import configure_sqlite_connection, configure_tortoise_sqlite


def test_sqlite_connection_pragmas(tmp_path):
    connection = sqlite3.connect(tmp_path / "test.db")
    configure_sqlite_connection(connection)

    assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


@pytest.mark.asyncio
async def test_tortoise_sqlite_pragmas(monkeypatch):
    statements = []

    class FakeConnection:
        async def execute_query(self, sql):
            statements.append(sql)

    monkeypatch.setattr("app.core.database.connections.get", lambda _: FakeConnection())
    await configure_tortoise_sqlite()

    assert statements == [
        "PRAGMA journal_mode=WAL",
        "PRAGMA foreign_keys=ON",
        "PRAGMA busy_timeout=5000",
    ]
```

- [ ] **Step 2: Verify both tests fail**

Run:

```powershell
uv run pytest tests/test_settings.py tests/test_database.py -v
```

Expected: FAIL because production validators and `configure_sqlite_connection` do not exist.

- [ ] **Step 3: Implement the settings contract**

Define these fields and validation rules in `app/settings/config.py` while preserving the existing `TORTOISE_ORM` model registration:

```python
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_ENV: Literal["development", "test", "production"] = "development"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, ge=5, le=480)
    DATABASE_PATH: Path = Path("data/db.sqlite3")
    CORS_ORIGINS: list[str] = ["http://localhost:3100"]

    @model_validator(mode="after")
    def validate_production(self):
        if self.APP_ENV == "production":
            if self.SECRET_KEY == "change-me" or len(self.SECRET_KEY) < 32:
                raise ValueError("SECRET_KEY must contain at least 32 non-default characters")
            if "*" in self.CORS_ORIGINS:
                raise ValueError("CORS_ORIGINS cannot contain '*' in production")
        return self
```

Build the SQLite Tortoise connection path from `DATABASE_PATH`; do not retain the source template's hard-coded repository-root database path.

- [ ] **Step 4: Implement SQLite connection configuration**

Create `app/core/database.py`:

```python
import sqlite3

from tortoise import connections


def configure_sqlite_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")


async def configure_tortoise_sqlite(connection_name: str = "default") -> None:
    connection = connections.get(connection_name)
    for statement in (
        "PRAGMA journal_mode=WAL",
        "PRAGMA foreign_keys=ON",
        "PRAGMA busy_timeout=5000",
    ):
        await connection.execute_query(statement)
```

Name the Tortoise connection `default` in `TORTOISE_ORM`. Call `await configure_tortoise_sqlite()` immediately after `await command.init()` in `init_data()` and before any seed or request work.

- [ ] **Step 5: Add the environment template**

Create `.env.example`:

```dotenv
APP_ENV=development
DEBUG=false
SECRET_KEY=change-me
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
DATABASE_PATH=data/db.sqlite3
CORS_ORIGINS=["http://localhost:3100"]
```

- [ ] **Step 6: Run focused and full baseline checks**

```powershell
uv run pytest tests/test_settings.py tests/test_database.py -v
uv run pytest -v
uv run ruff check app tests
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 7: Commit configuration and database behavior**

```powershell
git add .env.example app/settings app/core/database.py tests/test_settings.py tests/test_database.py
git commit -m "feat: add production settings and SQLite safeguards"
```

## Task 4: Migrate Authentication to Standard Bearer JWT

**Files:**
- Modify: `app/core/dependency.py`
- Modify: `app/utils/jwt_utils.py`
- Modify: `app/api/v1/base/base.py`
- Modify: `app/core/middlewares.py`
- Modify: `web/src/utils/http/interceptors.js`
- Create: `tests/test_auth.py`

- [ ] **Step 1: Write failing token tests**

Create `tests/test_auth.py`:

```python
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import HTTPException

from app.core.dependency import decode_access_token


def make_token(**overrides):
    payload = {
        "sub": "1",
        "username": "admin",
        "is_superuser": True,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        **overrides,
    }
    return jwt.encode(payload, "test-secret", algorithm="HS256")


def test_decode_access_token_returns_subject():
    payload = decode_access_token(make_token(), secret="test-secret", algorithm="HS256")
    assert payload["sub"] == "1"


def test_decode_access_token_rejects_invalid_token():
    with pytest.raises(HTTPException) as exc:
        decode_access_token("not-a-token", secret="test-secret", algorithm="HS256")
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid or expired access token"


def test_dev_token_is_never_accepted():
    with pytest.raises(HTTPException) as exc:
        decode_access_token("dev", secret="test-secret", algorithm="HS256")
    assert exc.value.status_code == 401
```

- [ ] **Step 2: Run tests and verify the existing bypass/decoder contract fails**

```powershell
uv run pytest tests/test_auth.py -v
```

Expected: FAIL because `decode_access_token` is not defined and the old dependency contains a `dev` bypass.

- [ ] **Step 3: Implement one safe decoder and HTTPBearer dependency**

Use this decoder in `app/core/dependency.py`:

```python
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

bearer = HTTPBearer(auto_error=False)


def decode_access_token(token: str, *, secret: str, algorithm: str) -> dict:
    try:
        return jwt.decode(token, secret, algorithms=[algorithm], options={"require": ["sub", "exp"]})
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired access token") from exc


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Bearer token required")
    payload = decode_access_token(
        credentials.credentials,
        secret=settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    user = await User.filter(id=int(payload["sub"]), is_active=True).first()
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired access token")
    CTX_USER_ID.set(user.id)
    return user
```

Remove `AuthControl.is_authed` only after all dependencies and middleware use `get_current_user` or the pure decoder. Do not accept the old `token` header as a fallback.

- [ ] **Step 4: Issue JWTs with standard registered claims**

Update the login payload to include string `sub`, `iat`, and `exp`. Keep `username` and `is_superuser` as display/optimization claims, but always reload the active user from SQLite on protected requests.

- [ ] **Step 5: Change the Vue interceptor**

Replace custom header injection in `web/src/utils/http/interceptors.js` with:

```javascript
const token = getToken()
if (token) {
  config.headers.Authorization = `Bearer ${token}`
}
```

- [ ] **Step 6: Run backend and frontend verification**

```powershell
uv run pytest tests/test_auth.py -v
uv run pytest -v
uv run ruff check app tests
cd web
pnpm install --frozen-lockfile
pnpm lint
pnpm build
```

Expected: token tests and the full backend suite pass; Vue lint and build pass.

- [ ] **Step 7: Commit the header migration atomically**

```powershell
git add app/core/dependency.py app/utils/jwt_utils.py app/api/v1/base/base.py app/core/middlewares.py web/src/utils/http/interceptors.js tests/test_auth.py
git commit -m "fix: enforce standard bearer authentication"
```

## Task 5: Replace Default Administrator Creation with an Explicit Command

**Files:**
- Modify: `app/core/init_app.py`
- Create: `app/commands/__init__.py`
- Create: `app/commands/create_admin.py`
- Create: `tests/test_admin_bootstrap.py`

- [ ] **Step 1: Write failing bootstrap tests**

Create `tests/test_admin_bootstrap.py`:

```python
import pytest

from app.commands.create_admin import validate_admin_input


def test_bootstrap_rejects_default_password():
    with pytest.raises(ValueError, match="12 characters"):
        validate_admin_input("admin", "admin@example.com", "123456")


def test_bootstrap_accepts_explicit_strong_credentials():
    validate_admin_input("owner", "owner@example.com", "correct-horse-42")
```

- [ ] **Step 2: Verify the command does not exist**

```powershell
uv run pytest tests/test_admin_bootstrap.py -v
```

Expected: FAIL importing `app.commands.create_admin`.

- [ ] **Step 3: Remove automatic weak-user initialization**

Delete `init_superuser()` and its invocation from `app/core/init_app.py`. Startup must never create credentials.

- [ ] **Step 4: Implement the explicit command**

Create `app/commands/create_admin.py`:

```python
import argparse
import asyncio
import getpass

from tortoise import Tortoise

from app.models import User
from app.settings import TORTOISE_ORM
from app.utils.password import get_password_hash


def validate_admin_input(username: str, email: str, password: str) -> None:
    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters")
    forbidden = {"123456", "password", username.lower()}
    if password.lower() in forbidden:
        raise ValueError("Password is not allowed")
    if "@" not in email:
        raise ValueError("Email is invalid")


async def create_admin(username: str, email: str, password: str) -> None:
    await Tortoise.init(config=TORTOISE_ORM)
    try:
        if await User.filter(username=username).exists():
            raise ValueError("Username already exists")
        if await User.filter(email=email).exists():
            raise ValueError("Email already exists")
        await User.create(
            username=username,
            email=email,
            password=get_password_hash(password),
            is_active=True,
            is_superuser=True,
        )
    finally:
        await Tortoise.close_connections()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--email", required=True)
    args = parser.parse_args()
    password = getpass.getpass("Password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match")
    validate_admin_input(args.username, args.email, password)
    asyncio.run(create_admin(args.username, args.email, password))


if __name__ == "__main__":
    main()
```

Expose it as:

```powershell
uv run python -m app.commands.create_admin --username owner --email owner@example.com
```

- [ ] **Step 5: Run tests and a temporary-database command smoke test**

```powershell
uv run pytest tests/test_admin_bootstrap.py -v
$env:DATABASE_PATH = "$env:TEMP\groundrag-admin-bootstrap.db"
uv run aerich upgrade
uv run python -m app.commands.create_admin --username owner --email owner@example.com
```

Expected: tests pass; the command prompts twice and creates one user without printing the password.

- [ ] **Step 6: Commit the bootstrap boundary**

```powershell
git add app/core/init_app.py app/commands tests/test_admin_bootstrap.py
git commit -m "fix: require explicit administrator bootstrap"
```

## Task 6: Make Audit Logging Safe for Secrets and Future SSE

**Files:**
- Modify: `app/core/middlewares.py`
- Modify: `app/core/init_app.py`
- Create: `tests/test_audit_middleware.py`

- [ ] **Step 1: Write failing audit-policy tests**

Create `tests/test_audit_middleware.py`:

```python
from app.core.middlewares import AuditPolicy


def test_streaming_chat_is_excluded_before_body_read():
    policy = AuditPolicy(excluded_prefixes=("/api/v1/rag/chat",))
    assert policy.should_skip("POST", "/api/v1/rag/chat") is True


def test_secret_fields_are_redacted_recursively():
    policy = AuditPolicy(excluded_prefixes=())
    value = {"password": "secret", "nested": {"access_token": "token", "name": "ok"}}
    assert policy.redact(value) == {
        "password": "[REDACTED]",
        "nested": {"access_token": "[REDACTED]", "name": "ok"},
    }
```

- [ ] **Step 2: Verify policy tests fail**

```powershell
uv run pytest tests/test_audit_middleware.py -v
```

Expected: FAIL because `AuditPolicy` does not exist.

- [ ] **Step 3: Implement a policy checked before request or response consumption**

Add a small `AuditPolicy` class with:

```python
SENSITIVE_FIELDS = frozenset({"password", "old_password", "new_password", "token", "access_token", "secret", "api_key"})


def should_skip(self, method: str, path: str) -> bool:
    return any(path.startswith(prefix) for prefix in self.excluded_prefixes)
```

Call `should_skip()` at the first line of middleware dispatch. When true, immediately return `await call_next(request)` without reading the request body or iterating the response body. Redact dictionaries and lists recursively before `AuditLog.create`.

- [ ] **Step 4: Register explicit exclusions**

Configure exclusions for login, OpenAPI, health, file upload, and the future `/api/v1/rag/chat` stream. Upload endpoints log metadata through their own service later.

- [ ] **Step 5: Verify audit behavior**

```powershell
uv run pytest tests/test_audit_middleware.py -v
uv run pytest -v
uv run ruff check app tests
```

Expected: policy tests and full suite pass; Ruff reports no errors.

- [ ] **Step 6: Commit audit safety**

```powershell
git add app/core/middlewares.py app/core/init_app.py tests/test_audit_middleware.py
git commit -m "fix: redact audit data and bypass streaming routes"
```

## Task 7: Separate Local Development from Production Serving

**Files:**
- Modify: `web/vite.config.js`
- Create: `web/.env.development`
- Modify: `Makefile`
- Modify: `README.md`

- [ ] **Step 1: Add explicit development environment values**

Create `web/.env.development`:

```dotenv
VITE_TITLE=GroundRAG Admin
VITE_PORT=3100
VITE_PUBLIC_PATH=/
VITE_BASE_API=/api/v1
VITE_USE_PROXY=true
VITE_USE_COMPRESS=false
```

Set the existing proxy target for `/api/v1` to `http://127.0.0.1:8000` in the Vite proxy configuration. Keep the API path relative in application code.

- [ ] **Step 2: Add deterministic Make targets**

Add:

```make
.PHONY: dev-api dev-web test test-web build-web up down

dev-api: ## Run FastAPI with reload on :8000
	uv run uvicorn app:app --reload --host 127.0.0.1 --port 8000

dev-web: ## Run Vue/Vite with HMR on :3100
	cd web && pnpm dev

test: ## Run backend tests
	uv run pytest -v

test-web: ## Run frontend checks
	cd web && pnpm lint && pnpm build

build-web: ## Build production Vue assets
	cd web && pnpm build

up: ## Build and start the production topology
	docker compose up -d --build

down: ## Stop the production topology
	docker compose down
```

- [ ] **Step 3: Document the two-terminal development workflow**

Add to `README.md`:

```markdown
## Local Development

Run `make dev-api` and `make dev-web` in separate terminals, then open
http://localhost:3100. Vite proxies `/api/v1` to FastAPI on port 8000.
Caddy is not used during local development.
```

- [ ] **Step 4: Verify local builds and proxy configuration**

Run:

```powershell
uv run pytest -v
cd web
pnpm lint
pnpm build
```

Expected: backend tests pass and Vue produces `web/dist` successfully.

- [ ] **Step 5: Commit the development workflow**

```powershell
git add web/.env.development web/vite.config.js Makefile README.md
git commit -m "dev: separate Vite and FastAPI workflows"
```

## Task 8: Add Health Endpoints and Two-Container Production Deployment

**Files:**
- Create: `app/api/v1/health.py`
- Modify: `app/api/v1/__init__.py`
- Create: `tests/test_health.py`
- Create: `Dockerfile.app`
- Create: `Dockerfile.caddy`
- Create: `deploy/Caddyfile`
- Create: `docker-compose.yml`
- Modify: `.dockerignore`
- Remove after replacement: `Dockerfile`

- [ ] **Step 1: Write failing health tests**

Create `tests/test_health.py`:

```python
from fastapi.testclient import TestClient

from app import app


def test_liveness_does_not_require_database():
    with TestClient(app) as client:
        response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json()["data"]["status"] == "alive"


def test_readiness_reports_database_state(monkeypatch):
    async def database_ready():
        return True

    monkeypatch.setattr("app.api.v1.health.database_ready", database_ready)
    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    assert response.json()["data"]["database"] == "ready"
```

- [ ] **Step 2: Verify health tests fail**

```powershell
uv run pytest tests/test_health.py -v
```

Expected: FAIL with `404` because the health router is absent.

- [ ] **Step 3: Implement and register the health router**

Implement unauthenticated `/health/live` and `/health/ready` under the existing `/api/v1` prefix. Liveness must not touch SQLite. Readiness executes `SELECT 1`; it returns `503` with `{database: "unavailable"}` on failure. RAG readiness is added in plan 3.

- [ ] **Step 4: Create the FastAPI image**

Create `Dockerfile.app`:

```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY migrations ./migrations
COPY pyproject.toml ./

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data \
    && chown -R app:app /app

USER app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

- [ ] **Step 5: Create the Vue/Caddy image**

Create `Dockerfile.caddy`:

```dockerfile
FROM node:20-alpine AS web-builder
WORKDIR /src/web
RUN corepack enable
COPY web/package.json web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY web/ ./
RUN pnpm build

FROM caddy:2-alpine
COPY --from=web-builder /src/web/dist /srv
COPY deploy/Caddyfile /etc/caddy/Caddyfile
```

Create `deploy/Caddyfile`:

```caddyfile
{$SITE_ADDRESS:http://localhost} {
    encode zstd gzip

    handle /api/* {
        reverse_proxy app:8000
    }

    handle {
        root * /srv
        try_files {path} /index.html
        file_server
    }

    header {
        X-Content-Type-Options nosniff
        Referrer-Policy strict-origin-when-cross-origin
        X-Frame-Options DENY
        -Server
    }
}
```

- [ ] **Step 6: Create Compose with one public service**

Create `docker-compose.yml`:

```yaml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.app
    env_file: .env
    environment:
      APP_ENV: production
      DATABASE_PATH: /app/data/db.sqlite3
    expose:
      - "8000"
    volumes:
      - app-data:/app/data
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health/ready', timeout=3)"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 20s
    restart: unless-stopped
    mem_limit: 1g
    cpus: 1.0
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  caddy:
    build:
      context: .
      dockerfile: Dockerfile.caddy
    environment:
      SITE_ADDRESS: ${SITE_ADDRESS:-http://localhost}
    depends_on:
      app:
        condition: service_healthy
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - caddy-data:/data
      - caddy-config:/config
    restart: unless-stopped
    mem_limit: 256m
    cpus: 0.5
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  app-data:
  caddy-data:
  caddy-config:
```

- [ ] **Step 7: Verify images, routing, and persistence**

Run:

```powershell
uv run pytest tests/test_health.py -v
docker compose build
docker compose up -d
curl.exe -f http://localhost/api/v1/health/live
curl.exe -f http://localhost/
docker compose restart app
curl.exe -f http://localhost/api/v1/health/ready
docker compose down
```

Expected: both images build; Caddy returns Vue at `/`; API health succeeds only through Caddy; restart preserves SQLite under `app-data`.

- [ ] **Step 8: Commit production deployment**

```powershell
git add app/api/v1 tests/test_health.py Dockerfile.app Dockerfile.caddy deploy/Caddyfile docker-compose.yml .dockerignore
git rm Dockerfile
git commit -m "build: add Caddy production topology"
```

## Task 9: Add Continuous Integration and Final Foundation Verification

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `README.md`

- [ ] **Step 1: Create CI with independent backend, frontend, and container jobs**

Create `.github/workflows/ci.yml`:

```yaml
name: ci

on:
  push:
  pull_request:

permissions:
  contents: read

jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.12"
          enable-cache: true
      - run: uv sync --frozen
      - run: uv run ruff check app tests
      - run: uv run pytest -v

  frontend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: web
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
        with:
          version: 9
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: pnpm
          cache-dependency-path: web/pnpm-lock.yaml
      - run: pnpm install --frozen-lockfile
      - run: pnpm lint
      - run: pnpm build

  containers:
    runs-on: ubuntu-latest
    needs: [backend, frontend]
    steps:
      - uses: actions/checkout@v4
      - run: cp .env.example .env
      - run: docker compose build
```

- [ ] **Step 2: Run the entire local evidence suite**

Run from `F:/code/python/groundrag-admin`:

```powershell
uv sync --frozen
uv run ruff check app tests
uv run pytest -v
cd web
pnpm install --frozen-lockfile
pnpm lint
pnpm build
cd ..
docker compose build
docker compose up -d
curl.exe -f http://localhost/api/v1/health/live
curl.exe -f http://localhost/
docker compose down
git status --short
```

Expected: all commands exit `0`; health and Vue return successful responses; `git status --short` shows only the intended CI/README changes before commit.

- [ ] **Step 3: Update phase status and operational instructions**

Mark the management foundation complete in `README.md`. Document:

- local commands `make dev-api` and `make dev-web`;
- production command `make up`;
- administrator command;
- required production environment variables;
- confirmation that Caddy is production-only and installed through Compose.

- [ ] **Step 4: Commit CI and the verified foundation**

```powershell
git add .github/workflows/ci.yml README.md
git commit -m "ci: verify GroundRAG Admin foundation"
```

- [ ] **Step 5: Record final state for the next plan**

Run:

```powershell
git log --oneline --decorate -10
git status --short
```

Expected: nine focused task commits or fewer only where a task required no source change; clean worktree; current branch `codex/integration-foundation`.

## Foundation Exit Criteria

- The new repository exists independently and neither source repository was modified.
- Python 3.12 dependencies resolve from one lock file.
- Template regression tests exist and pass on FastAPI 0.115.x.
- Production rejects weak secrets and wildcard CORS.
- SQLite uses the configured durable path, WAL, foreign keys, and a busy timeout.
- The `dev` token and automatic `admin / 123456` path no longer exist.
- Client, server, tests, and audit identity lookup use only `Authorization: Bearer`.
- Audit logging redacts secrets and bypasses future streaming routes before body inspection.
- Local development uses Vite on `3100` and FastAPI on `8000` without Caddy.
- Production uses Caddy as the only public service and FastAPI only for `/api/*`.
- Backend tests, frontend lint/build, Compose build, health checks, and restart persistence verification pass.

Do not begin plan 2 until every exit criterion has current command evidence.
