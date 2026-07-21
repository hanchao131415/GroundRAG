.PHONY: setup dev build run docs demo test test-web docker-build docker-up docker-down clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---- Local Development ----

setup: ## Install Python + Node dependencies
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1 && pip install -r requirements.txt
	cd web && npm install

dev: ## Start backend (uvicorn) + frontend (Vite dev server)
	@echo "Starting backend at http://localhost:8000 ..."
	.\.venv\Scripts\Activate.ps1 && uvicorn app.api:app --reload --host 0.0.0.0 --port 8000 &
	@echo "Starting frontend at http://localhost:5173 ..."
	cd web && npm run dev

build: ## Build frontend for production (output: web/dist/)
	cd web && npm run build

run: build ## Build frontend + start FastAPI serving everything at :8000
	@echo "Serving GroundRAG at http://localhost:8000"
	@echo "API docs at http://localhost:8000/docs"
	.\.venv\Scripts\Activate.ps1 && uvicorn app.api:app --host 0.0.0.0 --port 8000

# ---- Data ----

docs: ## Generate synthetic sample documents
	.\.venv\Scripts\Activate.ps1 && python scripts/make_sample_docs.py

demo: ## Prepare sample documents and print the reviewer walkthrough
	.\.venv\Scripts\Activate.ps1 && python scripts/demo_setup.py

# ---- Testing ----

test: ## Run backend tests
	.\.venv\Scripts\Activate.ps1 && python -m pytest tests/ -v

test-web: ## Run frontend tests
	cd web && npx vitest run

# ---- Docker ----

docker-build: ## Build Docker image
	docker build -t groundrag:latest .

docker-up: ## Start with docker compose
	docker compose up --build

docker-down: ## Stop docker compose
	docker compose down

# ---- Cleanup ----

clean: ## Remove build artifacts and runtime data
	rm -rf web/dist/
	rm -rf data/cache/ data/traces/ data/vector_index/
	rm -rf .pytest_cache/ .venv/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
