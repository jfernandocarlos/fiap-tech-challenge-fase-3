"""Classificador TF-IDF — camada fina sobre a Factory.

Mantido para compatibilidade; a construção real está em :mod:`factory`.
"""

from sklearn.pipeline import Pipeline
from src.triage.config import load_params
from src.triage.models.factory import create_baseline_pipeline, create_production_pipeline


def build_production_pipeline() -> Pipeline:
    """Cria pipeline de produção a partir do ``params.yaml``.

    Returns:
        Pipeline TF-IDF + LogisticRegression.
    """
    params = load_params()
    return create_production_pipeline(params["model"], params["train"]["random_seed"])


def build_baseline_pipeline() -> Pipeline:
    """Cria baseline Random Forest a partir do ``params.yaml``.

    Returns:
        Pipeline TF-IDF + RandomForestClassifier.
    """
    params = load_params()
    return create_baseline_pipeline(params["model"], params["train"]["random_seed"])
