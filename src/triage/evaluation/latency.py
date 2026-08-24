"""Benchmark de latência sklearn vs ONNX."""

import json
import statistics
import time
from typing import Literal

from src.triage.config import LATENCY_COMPARISON_PATH, LATENCY_REPORT_PATH, load_params
from src.triage.models.backends import get_inference_backend

SAMPLE_TEXT = (
    "Patient presents with acute chest pain radiating to the left arm, "
    "shortness of breath and diaphoresis. ECG shows ST elevation."
)


def _measure(
    backend: Literal["sklearn", "onnx"],
    n_requests: int = 200,
    warmup: int = 20,
) -> dict[str, float]:
    """Mede latência em ms para um backend.

    Args:
        backend: Nome do backend (``sklearn`` ou ``onnx``).
        n_requests: Requisições medidas após warmup.
        warmup: Requisições descartadas antes da medição.

    Returns:
        Estatísticas de latência em milissegundos.
    """
    latencies: list[float] = []
    strategy = get_inference_backend(backend)
    if not strategy.load():
        msg = f"Backend {backend} indisponível — execute 'make train' antes."
        raise FileNotFoundError(msg)

    def predict_fn(text: str) -> tuple[str, dict[str, float]]:
        return strategy.predict(text)

    for _ in range(warmup):
        predict_fn(SAMPLE_TEXT)

    for _ in range(n_requests):
        start = time.perf_counter()
        predict_fn(SAMPLE_TEXT)
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "p50_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 3),
        "mean_ms": round(statistics.mean(latencies), 3),
        "n_requests": float(n_requests),
    }


def _write_latency_comparison(
    sklearn_stats: dict[str, float],
    onnx_stats: dict[str, float],
    improvement: float,
) -> None:
    """Gera ``docs/latency_comparison.md`` a partir do relatório medido."""
    content = (
        "# Comparativo de Latência\n\n"
        "| Backend | Média (ms) | p95 (ms) |\n"
        "|---------|------------|----------|\n"
        f"| sklearn | {sklearn_stats['mean_ms']} | {sklearn_stats['p95_ms']} |\n"
        f"| **ONNX** | **{onnx_stats['mean_ms']}** | **{onnx_stats['p95_ms']}** |\n\n"
        f"Melhoria: **~{improvement:.0f}%** — gerado por `make benchmark`.\n"
    )
    LATENCY_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATENCY_COMPARISON_PATH.write_text(content, encoding="utf-8")


def run_benchmark(n_requests: int | None = None) -> dict[str, object]:
    """Compara latência sklearn vs ONNX e salva relatório.

    Args:
        n_requests: Sobrescreve valor do ``params.yaml`` se informado.

    Returns:
        Relatório com estatísticas de ambos os backends.
    """
    params = load_params()["benchmark"]
    n = n_requests or params["n_requests"]
    warmup = params["warmup"]

    sklearn_stats = _measure("sklearn", n_requests=n, warmup=warmup)
    onnx_stats = _measure("onnx", n_requests=n, warmup=warmup)

    improvement = round(
        (sklearn_stats["mean_ms"] - onnx_stats["mean_ms"]) / sklearn_stats["mean_ms"] * 100,
        2,
    )

    report = {
        "sample_text_chars": len(SAMPLE_TEXT),
        "sklearn": sklearn_stats,
        "onnx": onnx_stats,
        "mean_improvement_percent": improvement,
    }

    LATENCY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATENCY_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_latency_comparison(sklearn_stats, onnx_stats, improvement)
    return report
