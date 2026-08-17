"""Schema e constantes de dados."""

from enum import Enum


class UrgencyLevel(str, Enum):
    """Níveis de urgência clínica para triagem."""

    NORMAL = "normal"
    ATENCAO = "atencao"
    URGENTE = "urgente"


# Mapeamento do Medical Abstracts TC Corpus (condition_label) → urgência simulada
CONDITION_TO_URGENCY: dict[int, UrgencyLevel] = {
    1: UrgencyLevel.URGENTE,  # neoplasms
    2: UrgencyLevel.ATENCAO,  # digestive system diseases
    3: UrgencyLevel.ATENCAO,  # nervous system diseases
    4: UrgencyLevel.URGENTE,  # cardiovascular diseases
    5: UrgencyLevel.NORMAL,  # general pathological conditions
}

URGENCY_LABELS: list[str] = [level.value for level in UrgencyLevel]
