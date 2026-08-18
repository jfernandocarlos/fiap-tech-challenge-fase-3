"""Schemas Pydantic da API."""

from typing import Literal

from pydantic import BaseModel, Field
from src.triage.data.schema import UrgencyLevel


class ReportInput(BaseModel):
    """Entrada de laudo médico para triagem."""

    text: str = Field(
        ...,
        min_length=20,
        max_length=10000,
        description="Texto do laudo ou resumo clínico",
        examples=["Patient with acute chest pain, ST elevation on ECG and elevated troponin."],
    )


class PredictionResponse(BaseModel):
    """Resposta de classificação de urgência."""

    urgency: UrgencyLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[UrgencyLevel, float]
    backend: Literal["sklearn", "onnx"]
    recommendation: str


class HealthResponse(BaseModel):
    """Status da API."""

    status: Literal["healthy"]
    model_loaded: bool
    backend: str
    version: str
    artifacts: dict[str, bool]
