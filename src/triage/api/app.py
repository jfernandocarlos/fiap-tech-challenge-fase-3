"""API FastAPI de triagem de laudos médicos."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from src.triage import __version__
from src.triage.api.schemas import HealthResponse, PredictionResponse, ReportInput
from src.triage.config import settings
from src.triage.data.schema import UrgencyLevel
from src.triage.logging_config import get_logger, setup_logging
from src.triage.models.inference import ModelRegistry

logger = get_logger(__name__)
registry = ModelRegistry()

RECOMMENDATIONS = {
    UrgencyLevel.NORMAL: "Monitoramento ambulatorial. Retorno conforme protocolo.",
    UrgencyLevel.ATENCAO: "Avaliação prioritária em até 24 horas.",
    UrgencyLevel.URGENTE: "Encaminhamento imediato para atendimento de emergência.",
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    logger.info("iniciando API de triagem", version=__version__)
    registry.load()
    yield
    logger.info("encerrando API")


app = FastAPI(
    title="Medical Triage API",
    description="Classificação de urgência em laudos médicos (NLP leve).",
    version=__version__,
    lifespan=lifespan,
)


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


@app.post("/predict", response_model=PredictionResponse, tags=["Predição"])
async def predict(report: ReportInput) -> PredictionResponse:
    """Classifica urgência de um laudo médico."""
    if not registry.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute 'make train' e reinicie a API.",
        )

    try:
        label, probabilities = registry.predict(report.text)
    except (RuntimeError, ValueError) as exc:
        logger.exception("erro na predição")
        raise HTTPException(status_code=500, detail="Erro interno na predição") from exc

    urgency = UrgencyLevel(label)
    confidence = probabilities.get(label, 0.0)
    typed_probabilities = {
        UrgencyLevel(key): value
        for key, value in probabilities.items()
        if key in UrgencyLevel._value2member_map_
    }

    logger.info("predição realizada", urgency=urgency.value, confidence=round(confidence, 4))

    return PredictionResponse(
        urgency=urgency,
        confidence=round(confidence, 4),
        probabilities=typed_probabilities,
        backend="sklearn",
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
