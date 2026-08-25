"""Testes da API."""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "artifacts" in data


class TestPredictEndpoint:
    def test_predict_without_api_key_returns_401(
        self, client: TestClient, sample_report: dict[str, str]
    ) -> None:
        response = client.post("/predict", json=sample_report)
        assert response.status_code == 401

    def test_predict_invalid_payload_returns_422(
        self, client: TestClient, api_headers: dict[str, str]
    ) -> None:
        response = client.post("/predict", json={"text": "curto"}, headers=api_headers)
        assert response.status_code == 422

    @pytest.mark.integration
    def test_predict_with_model_returns_200(
        self,
        client: TestClient,
        api_headers: dict[str, str],
        sample_report: dict[str, str],
        trained_registry: object,
    ) -> None:
        response = client.post("/predict", json=sample_report, headers=api_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["urgency"] in ("normal", "atencao", "urgente")
        assert 0 <= data["confidence"] <= 1
        assert data["backend"] == "onnx"
        assert "recommendation" in data


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "triage_requests_total" in response.text
