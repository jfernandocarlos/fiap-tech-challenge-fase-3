"""DAG Airflow: ingestão → treino → persistência do modelo."""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_data_task() -> str:
    from src.triage.data.loader import load_dataset

    df = load_dataset(source="raw")
    output = PROJECT_ROOT / "data" / "processed" / "triage_dataset.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return f"{len(df)} registros processados em {output}"


def train_model_task() -> str:
    from src.triage.models.training import train_model

    _, metrics = train_model(source="processed")
    return f"F1 macro: {metrics['f1_macro']:.4f}"


default_args = {
    "owner": "medical-triage",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="triage_training_pipeline",
    description="Pipeline de treino/retreino do classificador de triagem",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["triage", "nlp", "training"],
) as dag:
    load_data = PythonOperator(
        task_id="load_and_prepare_data",
        python_callable=load_data_task,
    )

    train_model = PythonOperator(
        task_id="train_and_save_model",
        python_callable=train_model_task,
    )

    load_data >> train_model
