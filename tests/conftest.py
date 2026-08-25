"""Fixtures compartilhadas de teste."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from src.triage.api.app import app
from src.triage.config import settings
from src.triage.models.inference import ModelRegistry


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def api_headers() -> dict[str, str]:
    return {"X-API-Key": settings.api_key}


@pytest.fixture()
def sample_report() -> dict[str, str]:
    return {
        "text": (
            "Patient with acute chest pain radiating to left arm, "
            "ST elevation on ECG and elevated troponin levels."
        )
    }


@pytest.fixture()
def trained_registry(monkeypatch: pytest.MonkeyPatch) -> Generator[ModelRegistry, None, None]:
    """Treina pipeline mínimo em memória para testes de integração locais."""
    import pandas as pd
    from src.triage.config import MODELS_DIR, ONNX_MODEL_PATH, SKLEARN_MODEL_PATH
    from src.triage.models.training import train_model

    sample_df = pd.DataFrame(
        {
            "text": [
                "acute myocardial infarction with chest pain and ST elevation",
                "mild gastritis with abdominal discomfort after meals",
                "routine annual checkup without acute findings",
                "severe stroke symptoms with hemiparesis and aphasia",
                "chronic back pain without neurological deficit",
            ]
            * 500,
            "urgency": ["urgente", "atencao", "normal", "urgente", "atencao"] * 500,
        }
    )

    sample_path = MODELS_DIR / "test_sample.csv"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(sample_path, index=False)

    monkeypatch.setattr(
        "src.triage.data.loader.load_dataset",
        lambda source="raw": sample_df,
    )
    monkeypatch.setattr(
        "src.triage.data.loader.validate_dataset",
        lambda df: None,
    )

    pipeline, _ = train_model(source="raw")
    registry = ModelRegistry()
    registry.backend_name = "onnx"
    registry.load()

    import src.triage.api.app as api_module

    api_module.registry = registry

    yield registry

    for path in (SKLEARN_MODEL_PATH, ONNX_MODEL_PATH):
        if path.exists():
            path.unlink()
