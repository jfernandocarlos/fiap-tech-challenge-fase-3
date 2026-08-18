.PHONY: help install lint format test download-data train api-up clean

PYTHON := python

help: ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instala dependências com Poetry
	poetry install
	poetry run pre-commit install || true

lint: ## Lint com ruff
	poetry run ruff check src tests scripts
	poetry run ruff format --check src tests scripts

format: ## Formata código
	poetry run ruff format src tests scripts
	poetry run ruff check --fix src tests scripts

test: ## Executa testes
	poetry run pytest

download-data: ## Baixa Medical Abstracts TC Corpus
	poetry run $(PYTHON) -m scripts.download_data

train: ## Treina classificador e salva joblib
	poetry run $(PYTHON) -m scripts.train

api-up: ## Sobe apenas a API localmente
	poetry run uvicorn src.triage.api.app:app --host 0.0.0.0 --port 8000

clean: ## Remove caches locais
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
