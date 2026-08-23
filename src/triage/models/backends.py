"""Backends de inferência (padrão Strategy).

Permite trocar sklearn ↔ ONNX sem alterar a API ou o ``ModelRegistry``.
"""

from abc import ABC, abstractmethod

import joblib
import numpy as np
import onnxruntime as ort
from sklearn.pipeline import Pipeline
from src.triage.config import ONNX_MODEL_PATH, SKLEARN_MODEL_PATH


class InferenceBackend(ABC):
    """Interface comum para backends de predição."""

    name: str

    @abstractmethod
    def load(self) -> bool:
        """Carrega artefatos do disco.

        Returns:
            ``True`` se o backend ficou pronto para inferência.
        """

    @abstractmethod
    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Classifica um laudo e retorna rótulo + probabilidades.

        Args:
            text: Texto do laudo médico.

        Returns:
            Tupla ``(urgência, mapa_de_probabilidades)``.
        """


class SklearnBackend(InferenceBackend):
    """Inferência via pipeline sklearn serializado (joblib)."""

    name = "sklearn"

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.label_encoder: dict[int, str] = {}

    def load(self) -> bool:
        """Carrega ``triage_pipeline.joblib``."""
        if not SKLEARN_MODEL_PATH.exists():
            return False
        self.pipeline = joblib.load(SKLEARN_MODEL_PATH)
        classes = list(self.pipeline.classes_)
        self.label_encoder = {idx: label for idx, label in enumerate(classes)}
        return True

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Prediz com ``predict`` + ``predict_proba`` do sklearn."""
        if self.pipeline is None:
            msg = "Pipeline sklearn não carregado"
            raise RuntimeError(msg)

        label = str(self.pipeline.predict([text])[0])
        probas = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        probabilities = {cls: float(prob) for cls, prob in zip(classes, probas, strict=True)}
        return label, probabilities


class OnnxBackend(InferenceBackend):
    """Inferência via ONNX Runtime (baixa latência)."""

    name = "onnx"

    def __init__(self) -> None:
        self.session: ort.InferenceSession | None = None
        self.label_encoder: dict[int, str] = {}
        self._sklearn_for_labels = SklearnBackend()

    def load(self) -> bool:
        """Carrega sessão ONNX; usa joblib só para mapear índices → rótulos."""
        if not ONNX_MODEL_PATH.exists():
            return False

        self.session = ort.InferenceSession(
            str(ONNX_MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

        # Labels vêm do pipeline sklearn (mesma ordem de classes)
        if self._sklearn_for_labels.load():
            self.label_encoder = self._sklearn_for_labels.label_encoder
        return True

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Prediz com ONNX Runtime."""
        if self.session is None:
            msg = "Sessão ONNX não carregada"
            raise RuntimeError(msg)

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: np.array([[text]], dtype=object)})

        raw_label = outputs[0].ravel()[0]
        if isinstance(raw_label, bytes | np.bytes_):
            label = raw_label.decode("utf-8")
        else:
            label = str(raw_label)

        probabilities: dict[str, float] = {}
        if len(outputs) > 1:
            prob_vector = np.asarray(outputs[1]).ravel()
            if prob_vector.size == len(self.label_encoder):
                for idx, prob in enumerate(prob_vector):
                    probabilities[self.label_encoder[idx]] = float(prob)
            else:
                probabilities[label] = 1.0
        else:
            probabilities[label] = 1.0

        return label, probabilities


_BACKENDS: dict[str, type[InferenceBackend]] = {
    "sklearn": SklearnBackend,
    "onnx": OnnxBackend,
}


def get_inference_backend(name: str) -> InferenceBackend:
    """Factory simples que resolve o backend pelo nome.

    Args:
        name: ``sklearn`` ou ``onnx``.

    Returns:
        Instância do backend solicitado.

    Raises:
        ValueError: Se o nome não for reconhecido.
    """
    key = name.lower()
    if key not in _BACKENDS:
        msg = f"Backend desconhecido: {name!r}. Opções: {sorted(_BACKENDS)}"
        raise ValueError(msg)
    return _BACKENDS[key]()
