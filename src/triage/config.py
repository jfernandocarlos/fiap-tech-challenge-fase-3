"""Configuração centralizada do projeto.

Duas fontes de verdade, com responsabilidades separadas:

* ``Settings`` (Pydantic + ``.env``): infraestrutura e runtime
  (host/porta da API, chave de autenticação, backend de inferência).
* ``params.yaml`` (lido por :func:`load_params`): hiperparâmetros de ML
  rastreados de forma explícita e versionados no repositório.
"""

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
PARAMS_FILE = PROJECT_ROOT / "params.yaml"

SKLEARN_MODEL_PATH = MODELS_DIR / "triage_pipeline.joblib"
ONNX_MODEL_PATH = MODELS_DIR / "triage_pipeline.onnx"
METRICS_PATH = MODELS_DIR / "metrics.json"
LATENCY_REPORT_PATH = MODELS_DIR / "latency_report.json"
LATENCY_COMPARISON_PATH = PROJECT_ROOT / "docs" / "latency_comparison.md"


class Settings(BaseSettings):
    """Configurações de infraestrutura lidas do ``.env``."""

    text_column: str = Field(default="text", description="Coluna de texto do laudo")
    target_column: str = Field(default="urgency", description="Coluna alvo de urgência")

    api_host: str = Field(default="0.0.0.0", description="Host da API")
    api_port: int = Field(default=8000, description="Porta da API")
    api_key: str = Field(default="dev-api-key-change-me", description="Chave de API")
    inference_backend: str = Field(
        default="onnx",
        description="Backend de inferência: sklearn ou onnx",
    )

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}


def load_params(path: Path = PARAMS_FILE) -> dict[str, Any]:
    """Carrega hiperparâmetros do ``params.yaml``.

    Args:
        path: Caminho do arquivo de parâmetros.

    Returns:
        Dicionário com as seções ``data``, ``model``, ``train`` e ``benchmark``.
    """
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


settings = Settings()
