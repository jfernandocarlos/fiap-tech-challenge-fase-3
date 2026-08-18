"""Testes da Factory de modelos."""

from src.triage.config import load_params
from src.triage.models.factory import create_baseline_pipeline, create_production_pipeline


def test_create_production_pipeline_returns_steps() -> None:
    params = load_params()
    pipeline = create_production_pipeline(params["model"], params["train"]["random_seed"])
    assert list(pipeline.named_steps.keys()) == ["tfidf", "classifier"]


def test_create_baseline_pipeline_returns_steps() -> None:
    params = load_params()
    pipeline = create_baseline_pipeline(params["model"], params["train"]["random_seed"])
    assert list(pipeline.named_steps.keys()) == ["tfidf", "classifier"]
