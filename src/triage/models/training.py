"""Treinamento, persistência e exportação ONNX do classificador."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.triage.config import (
    METRICS_PATH,
    MODELS_DIR,
    ONNX_MODEL_PATH,
    SKLEARN_MODEL_PATH,
    load_params,
    settings,
)
from src.triage.data.loader import load_dataset, validate_dataset
from src.triage.logging_config import get_logger
from src.triage.models.factory import create_baseline_pipeline, create_production_pipeline

logger = get_logger(__name__)


def train_model(source: str = "raw") -> tuple[Pipeline, dict[str, object]]:
    """Treina o pipeline, salva artefatos e retorna métricas.

    Args:
        source: Origem dos dados (``raw``, ``processed`` ou ``sample``).

    Returns:
        Tupla ``(pipeline_treinado, métricas)``.
    """
    params = load_params()
    df = load_dataset(source=source)
    validate_dataset(df)

    x = df[settings.text_column]
    y = df[settings.target_column]
    seed = params["train"]["random_seed"]
    test_size = params["data"]["test_size"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    model_params = params["model"]
    pipeline = create_production_pipeline(model_params, seed)
    pipeline.fit(x_train, y_train)

    baseline = create_baseline_pipeline(model_params, seed)
    baseline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    baseline_pred = baseline.predict(x_test)
    metrics: dict[str, object] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(df),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "production_model": "tfidf_logistic_regression",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "baseline_random_forest_f1_macro": float(f1_score(y_test, baseline_pred, average="macro")),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "hyperparameters": {
            "max_features": model_params["max_features"],
            "n_estimators_baseline": model_params["n_estimators_baseline"],
            "random_seed": seed,
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, SKLEARN_MODEL_PATH)
    export_to_onnx(pipeline)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("modelo treinado", f1_macro=metrics["f1_macro"], path=str(SKLEARN_MODEL_PATH))
    return pipeline, metrics


def export_to_onnx(pipeline: Pipeline, output_path: Path | None = None) -> Path:
    """Exporta pipeline sklearn para ONNX.

    Args:
        pipeline: Pipeline treinado (TF-IDF + classificador).
        output_path: Destino opcional; padrão ``models/triage_pipeline.onnx``.

    Returns:
        Caminho do arquivo ONNX gerado.
    """
    target = output_path or ONNX_MODEL_PATH
    initial_type = [("input", StringTensorType([None, 1]))]
    options = {
        LogisticRegression: {"zipmap": False},
        RandomForestClassifier: {"zipmap": False},
    }
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=15,
        options=options,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as file:
        file.write(onnx_model.SerializeToString())
    logger.info("modelo exportado para ONNX", path=str(target))
    return target
