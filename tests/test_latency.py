"""Testes de latência."""

import pytest
from src.triage.config import LATENCY_COMPARISON_PATH, LATENCY_REPORT_PATH
from src.triage.evaluation.latency import run_benchmark


@pytest.mark.integration
def test_benchmark_generates_report(trained_registry: object) -> None:
    report = run_benchmark(n_requests=20)
    assert report["sklearn"]["mean_ms"] > 0
    assert report["onnx"]["mean_ms"] > 0
    assert LATENCY_REPORT_PATH.exists()
    assert LATENCY_COMPARISON_PATH.exists()
    assert "Comparativo de Latência" in LATENCY_COMPARISON_PATH.read_text(encoding="utf-8")
