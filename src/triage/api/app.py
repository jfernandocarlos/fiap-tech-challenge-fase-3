"""API FastAPI de triagem de laudos médicos."""

import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from src.triage import __version__
from src.triage.api.metrics import PREDICTION_COUNT, PREDICTION_LATENCY
from src.triage.api.middleware import MetricsMiddleware
from src.triage.api.schemas import HealthResponse, PredictionResponse, ReportInput
from src.triage.config import settings
from src.triage.data.schema import UrgencyLevel
from src.triage.logging_config import get_logger, setup_logging
from src.triage.models.inference import ModelRegistry
from starlette.responses import Response

logger = get_logger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

registry = ModelRegistry()

RECOMMENDATIONS = {
    UrgencyLevel.NORMAL: "Monitoramento ambulatorial. Retorno conforme protocolo.",
    UrgencyLevel.ATENCAO: "Avaliação prioritária em até 24 horas.",
    UrgencyLevel.URGENTE: "Encaminhamento imediato para atendimento de emergência.",
}


def verify_api_key(api_key: str | None = Security(api_key_header)) -> None:
    """Valida chave de API quando configurada."""
    if settings.api_key and api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="API key inválida ou ausente")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    logger.info("iniciando API de triagem", version=__version__, backend=settings.inference_backend)
    registry.backend_name = settings.inference_backend.lower()
    registry.load()
    yield
    logger.info("encerrando API")


app = FastAPI(
    title="Medical Triage API",
    description="Classificação de urgência em laudos médicos (NLP leve + ONNX).",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(MetricsMiddleware)


@app.get("/health", response_model=HealthResponse, tags=["Monitoramento"])
async def health_check() -> HealthResponse:
    """Health check para containers e ALB."""
    return HealthResponse(
        status="healthy",
        model_loaded=registry.model_loaded,
        backend=registry.backend_name,
        version=__version__,
        artifacts=registry.artifact_status(),
    )


@app.get("/metrics", tags=["Monitoramento"])
async def metrics() -> Response:
    """Endpoint Prometheus."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predição"],
    dependencies=[Depends(verify_api_key)],
)
async def predict(report: ReportInput) -> PredictionResponse:
    """Classifica urgência de um laudo médico."""
    if not registry.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute 'make train' e reinicie a API.",
        )

    start = time.perf_counter()
    try:
        label, probabilities = registry.predict(report.text)
    except (RuntimeError, ValueError) as exc:
        logger.exception("erro na predição")
        raise HTTPException(status_code=500, detail="Erro interno na predição") from exc
    finally:
        PREDICTION_LATENCY.labels(backend=registry.backend_name).observe(
            time.perf_counter() - start
        )

    urgency = UrgencyLevel(label)
    confidence = probabilities.get(label, 0.0)
    typed_probabilities = {
        UrgencyLevel(key): value
        for key, value in probabilities.items()
        if key in UrgencyLevel._value2member_map_
    }

    PREDICTION_COUNT.labels(urgency=urgency.value).inc()
    logger.info("predição realizada", urgency=urgency.value, confidence=round(confidence, 4))

    return PredictionResponse(
        urgency=urgency,
        confidence=round(confidence, 4),
        probabilities=typed_probabilities,
        backend=registry.backend_name,  # type: ignore[arg-type]
        recommendation=RECOMMENDATIONS[urgency],
    )


def start_server() -> None:
    uvicorn.run(
        "src.triage.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    start_server()
