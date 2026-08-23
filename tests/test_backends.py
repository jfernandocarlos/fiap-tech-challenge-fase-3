"""Testes dos backends de inferência (padrão Strategy)."""

import pytest
from src.triage.models.backends import (
    OnnxBackend,
    SklearnBackend,
    get_inference_backend,
)


def test_get_inference_backend_returns_correct_type() -> None:
    assert isinstance(get_inference_backend("sklearn"), SklearnBackend)
    assert isinstance(get_inference_backend("onnx"), OnnxBackend)


def test_get_inference_backend_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Backend desconhecido"):
        get_inference_backend("tensorflow")
