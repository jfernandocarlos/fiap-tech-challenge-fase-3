"""Registro de artefatos e backend de inferência ativo."""

from src.triage.config import ONNX_MODEL_PATH, SKLEARN_MODEL_PATH, settings
from src.triage.models.backends import InferenceBackend, get_inference_backend


class ModelRegistry:
    """Registro de artefatos carregados no startup da API.

    Delega a predição ao backend configurado (padrão Strategy).
    """

    def __init__(self) -> None:
        self.backend_name = settings.inference_backend.lower()
        self.backend: InferenceBackend | None = None
        self.model_loaded = False

    def load(self) -> None:
        """Carrega o backend de inferência conforme ``INFERENCE_BACKEND``."""
        self.backend = get_inference_backend(self.backend_name)
        self.model_loaded = self.backend.load()

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Executa predição no backend ativo.

        Args:
            text: Texto do laudo médico.

        Returns:
            Tupla ``(urgência, probabilidades)``.

        Raises:
            RuntimeError: Se o modelo não estiver carregado.
        """
        if not self.model_loaded or self.backend is None:
            msg = "Modelo não carregado"
            raise RuntimeError(msg)
        return self.backend.predict(text)

    def artifact_status(self) -> dict[str, bool]:
        """Informa quais artefatos existem no disco.

        Returns:
            Mapa ``sklearn`` / ``onnx`` → arquivo presente.
        """
        return {
            "sklearn": SKLEARN_MODEL_PATH.exists(),
            "onnx": ONNX_MODEL_PATH.exists(),
        }
