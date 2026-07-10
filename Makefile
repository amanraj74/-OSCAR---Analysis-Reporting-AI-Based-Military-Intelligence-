.PHONY: help install install-dev lint format test test-unit test-cov refresh train dashboard seed-demo clean docker-build docker-up clean-data

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

.DEFAULT: help

help: ## Show this help.
	@$(PYTHON) scripts/dev.py --help

install: ## Install runtime dependencies.
	$(PIP) install -r requirements.txt

install-dev: ## Install all dependencies (runtime + dev).
	$(PIP) install -r requirements-dev.txt
	$(PYTHON) scripts/dev.py install-hooks

lint: ## Run all linters (ruff, black --check, isort --check, bandit, mypy).
	$(PYTHON) scripts/dev.py lint

format: ## Auto-format code (black, isort, ruff --fix).
	$(PYTHON) scripts/dev.py format

test: ## Run all tests with coverage.
	$(PYTHON) scripts/dev.py test

test-unit: ## Run unit tests only.
	$(PYTHON) scripts/dev.py test-unit

test-cov: ## Run tests with detailed coverage report.
	$(PYTHON) scripts/dev.py test-coverage

refresh: ## Refresh all data feeds.
	$(PYTHON) scripts/dev.py refresh

train: ## Train/retrain all models.
	$(PYTHON) scripts/dev.py train

dashboard: ## Launch Streamlit dashboard.
	$(PYTHON) scripts/dev.py dashboard

seed-demo: ## Seed demo dataset (30-day snapshot).
	$(PYTHON) scripts/dev.py seed-demo

clean: ## Remove caches, pyc files, build artifacts.
	$(PYTHON) scripts/dev.py clean

clean-data: ## Remove data/processed and SQLite DB (DANGEROUS).
	$(PYTHON) scripts/dev.py clean-data

docker-build: ## Build Docker image.
	docker build -t oscar:latest .

docker-up: ## Run full stack with docker-compose.
	docker compose up --build

docker-down: ## Stop docker-compose stack.
	docker compose down
