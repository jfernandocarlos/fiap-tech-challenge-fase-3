"""Middleware de latência e contagem de requisições."""

import time

from src.triage.api.metrics import ERROR_COUNT, REQUEST_COUNT, REQUEST_LATENCY
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Registra latência e contadores Prometheus por requisição."""

    async def dispatch(self, request: Request, call_next: object) -> Response:
        start = time.perf_counter()
        endpoint = request.url.path
        status_code = 500
        response: Response | None = None

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - start
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status=str(status_code),
            ).inc()
            if status_code >= 400:
                ERROR_COUNT.labels(endpoint=endpoint, status=str(status_code)).inc()
            if response is not None:
                response.headers["X-Process-Time"] = f"{elapsed:.4f}"
