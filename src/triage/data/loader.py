"""Carregamento e preparação dos dados de laudos médicos."""

from pathlib import Path

import pandas as pd
from src.triage.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DATA_SAMPLE_DIR,
    load_params,
    settings,
)
from src.triage.data.schema import CONDITION_TO_URGENCY, URGENCY_LABELS


def map_condition_to_urgency(condition_label: int) -> str:
    """Converte rótulo original do corpus para urgência clínica."""
    urgency = CONDITION_TO_URGENCY.get(int(condition_label))
    if urgency is None:
        msg = f"condition_label inválido: {condition_label}"
        raise ValueError(msg)
    return urgency.value


def load_raw_corpus(raw_dir: Path | None = None) -> pd.DataFrame:
    """Carrega CSVs brutos (train + test) e unifica em um DataFrame."""
    base = raw_dir or DATA_RAW_DIR
    frames: list[pd.DataFrame] = []

    for filename in ("medical_tc_train.csv", "medical_tc_test.csv"):
        path = base / filename
        if not path.exists():
            msg = (
                f"Arquivo não encontrado: {path}. " "Execute 'make download-data' antes do treino."
            )
            raise FileNotFoundError(msg)
        frames.append(pd.read_csv(path))

    return pd.concat(frames, ignore_index=True)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza colunas e aplica mapeamento de urgência."""
    required = {"condition_label", "medical_abstract"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Colunas obrigatórias ausentes: {sorted(missing)}"
        raise ValueError(msg)

    prepared = df[["medical_abstract", "condition_label"]].copy()
    prepared = prepared.rename(
        columns={
            "medical_abstract": settings.text_column,
            "condition_label": "condition_label",
        }
    )
    prepared[settings.text_column] = prepared[settings.text_column].astype(str).str.strip()
    min_len = load_params()["data"]["min_text_length"]
    prepared = prepared[prepared[settings.text_column].str.len() > min_len]
    prepared[settings.target_column] = prepared["condition_label"].map(
        lambda value: map_condition_to_urgency(int(value))
    )
    prepared = prepared.dropna(subset=[settings.target_column])
    return prepared[[settings.text_column, settings.target_column]]


def load_dataset(source: str = "raw") -> pd.DataFrame:
    """Carrega dataset processado ou reconstrói a partir dos CSVs brutos."""
    if source == "sample":
        sample_path = DATA_SAMPLE_DIR / "triage_sample.csv"
        if not sample_path.exists():
            msg = f"Amostra não encontrada: {sample_path}"
            raise FileNotFoundError(msg)
        df = pd.read_csv(sample_path)
        return df[[settings.text_column, settings.target_column]]

    processed_path = DATA_PROCESSED_DIR / "triage_dataset.csv"
    if processed_path.exists() and source == "processed":
        return pd.read_csv(processed_path)

    raw_df = load_raw_corpus()
    prepared = prepare_dataset(raw_df)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(processed_path, index=False)
    return prepared


def validate_dataset(df: pd.DataFrame) -> None:
    """Valida tamanho mínimo e rótulos permitidos."""
    if len(df) < 2000:
        msg = f"Dataset insuficiente: {len(df)} linhas (mínimo 2000)."
        raise ValueError(msg)

    invalid = set(df[settings.target_column].unique()) - set(URGENCY_LABELS)
    if invalid:
        msg = f"Rótulos inválidos encontrados: {sorted(invalid)}"
        raise ValueError(msg)
