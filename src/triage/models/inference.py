"""Registro de artefatos sklearn carregados no startup da API."""

import joblib
from sklearn.pipeline import Pipeline
from src.triage.config import ONNX_MODEL_PATH, SKLEARN_MODEL_PATH, settings


class ModelRegistry:
    """Carrega o pipeline sklearn e expõe predict."""

    def __init__(self) -> None:
        self.backend_name = settings.inference_backend.lower()
        self.pipeline: Pipeline | None = None
        self.model_loaded = False

    def load(self) -> None:
        """Carrega ``triage_pipeline.joblib`` se existir."""
        if not SKLEARN_MODEL_PATH.exists():
            self.model_loaded = False
            return
        self.pipeline = joblib.load(SKLEARN_MODEL_PATH)
        self.backend_name = "sklearn"
        self.model_loaded = True

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Prediz urgência com sklearn.

        Args:
            text: Texto do laudo médico.

        Returns:
            Tupla ``(urgência, probabilidades)``.
        """
        if not self.model_loaded or self.pipeline is None:
            msg = "Modelo não carregado"
            raise RuntimeError(msg)

        label = str(self.pipeline.predict([text])[0])
        probas = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        probabilities = {cls: float(prob) for cls, prob in zip(classes, probas, strict=True)}
        return label, probabilities

    def artifact_status(self) -> dict[str, bool]:
        """Informa quais artefatos existem no disco."""
        return {
            "sklearn": SKLEARN_MODEL_PATH.exists(),
            "onnx": ONNX_MODEL_PATH.exists(),
        }
