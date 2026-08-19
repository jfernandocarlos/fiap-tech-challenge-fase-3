"""Métricas Prometheus expostas pela API."""

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "triage_requests_total",
    "Total de requisições HTTP",
    ["method", "endpoint", "status"],
)

PREDICTION_COUNT = Counter(
    "triage_predictions_total",
    "Total de predições por urgência",
    ["urgency"],
)

REQUEST_LATENCY = Histogram(
    "triage_request_latency_seconds",
    "Latência das requisições HTTP",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

PREDICTION_LATENCY = Histogram(
    "triage_prediction_latency_seconds",
    "Latência da inferência",
    ["backend"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

ERROR_COUNT = Counter(
    "triage_errors_total",
    "Total de erros HTTP",
    ["endpoint", "status"],
)
