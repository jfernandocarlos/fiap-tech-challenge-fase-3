"""Testes do classificador."""

from src.triage.models.classifier import build_production_pipeline


def test_pipeline_has_expected_steps() -> None:
    pipeline = build_production_pipeline()
    assert list(pipeline.named_steps.keys()) == ["tfidf", "classifier"]
