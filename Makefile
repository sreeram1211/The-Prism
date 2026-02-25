.PHONY: help install install-dev dev test lint format clean docker-up docker-down

PYTHON := python3
PIP := pip3
UVICORN := uvicorn
BACKEND_DIR := backend

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install backend production dependencies
	cd $(BACKEND_DIR) && $(PIP) install -e .

install-dev: ## Install backend dev dependencies
	cd $(BACKEND_DIR) && $(PIP) install -e ".[dev]"

dev: ## Run FastAPI dev server with hot reload
	cd $(BACKEND_DIR) && $(UVICORN) prism.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run backend test suite
	cd $(BACKEND_DIR) && $(PYTHON) -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	cd $(BACKEND_DIR) && $(PYTHON) -m pytest tests/ -v --cov=prism --cov-report=term-missing

lint: ## Lint backend with ruff
	cd $(BACKEND_DIR) && $(PYTHON) -m ruff check prism/ tests/

format: ## Format backend with ruff
	cd $(BACKEND_DIR) && $(PYTHON) -m ruff format prism/ tests/

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf $(BACKEND_DIR)/.pytest_cache $(BACKEND_DIR)/.ruff_cache || true

docker-up: ## Start services via docker-compose
	docker compose up -d --build

docker-down: ## Stop docker-compose services
	docker compose down

# Phase 3 targets (Next.js)
frontend-install: ## Install frontend dependencies
	cd frontend && npm install

frontend-dev: ## Run Next.js dev server
	cd frontend && npm run dev

frontend-build: ## Build Next.js for production
	cd frontend && npm run build
