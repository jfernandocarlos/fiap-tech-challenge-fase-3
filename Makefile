.PHONY: help install validate lint format test download-data train benchmark \
        api-up stack-up stack-down docker-build clean

PYTHON := python

help: ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instala dependências com Poetry
	poetry install
	poetry run pre-commit install || true

lint: ## Lint com ruff
	poetry run ruff check src tests scripts airflow
	poetry run ruff format --check src tests scripts airflow

test: ## Executa testes
	poetry run pytest

clean: ## Remove caches locais
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
