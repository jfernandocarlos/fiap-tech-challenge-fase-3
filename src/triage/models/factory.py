"""Factory para criação de pipelines de classificação (padrão Factory Method).

Centraliza a construção dos modelos a partir do ``params.yaml``, isolando o
resto do código dos detalhes de instanciação.
"""

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def _build_tfidf(model_params: dict[str, Any]) -> TfidfVectorizer:
    """Cria vetorizador TF-IDF a partir dos parâmetros do modelo.

    Args:
        model_params: Seção ``model`` do ``params.yaml``.

    Returns:
        Instância configurada de :class:`TfidfVectorizer`.
    """
    ngram_min, ngram_max = model_params["ngram_range"]
    return TfidfVectorizer(
        max_features=model_params["max_features"],
        ngram_range=(ngram_min, ngram_max),
        stop_words="english",
        sublinear_tf=True,
    )


def create_production_pipeline(model_params: dict[str, Any], seed: int) -> Pipeline:
    """Cria pipeline de produção: TF-IDF + Regressão Logística (exportável ONNX).

    Args:
        model_params: Seção ``model`` do ``params.yaml``.
        seed: Semente para reprodutibilidade.

    Returns:
        Pipeline sklearn pronto para ``fit``.
    """
    return Pipeline(
        steps=[
            ("tfidf", _build_tfidf(model_params)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=model_params["max_iter"],
                    random_state=seed,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def create_baseline_pipeline(model_params: dict[str, Any], seed: int) -> Pipeline:
    """Cria baseline Random Forest para comparação offline.

    Args:
        model_params: Seção ``model`` do ``params.yaml``.
        seed: Semente para reprodutibilidade.

    Returns:
        Pipeline sklearn de baseline (sem export ONNX confiável).
    """
    return Pipeline(
        steps=[
            ("tfidf", _build_tfidf(model_params)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=model_params["n_estimators_baseline"],
                    random_state=seed,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )
