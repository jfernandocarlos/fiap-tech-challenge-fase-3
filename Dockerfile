FROM python:3.11-slim AS builder

WORKDIR /app

ENV POETRY_VERSION=1.8.5 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-ansi --no-root

COPY src/ ./src/
COPY scripts/ ./scripts/
RUN poetry install --only main --no-ansi

FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/
COPY scripts/ ./scripts/

# Artefatos devem existir antes do build (make train && make docker-build)
COPY models/ ./models/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INFERENCE_BACKEND=sklearn

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health | grep -q '"model_loaded": true'

CMD ["uvicorn", "src.triage.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
