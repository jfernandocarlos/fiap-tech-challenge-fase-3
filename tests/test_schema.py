"""Testes de schema e mapeamento."""

from src.triage.data.schema import CONDITION_TO_URGENCY, UrgencyLevel


def test_condition_mapping_covers_all_labels() -> None:
    assert set(CONDITION_TO_URGENCY) == {1, 2, 3, 4, 5}


def test_urgency_enum_values() -> None:
    assert UrgencyLevel.NORMAL.value == "normal"
    assert UrgencyLevel.ATENCAO.value == "atencao"
    assert UrgencyLevel.URGENTE.value == "urgente"
