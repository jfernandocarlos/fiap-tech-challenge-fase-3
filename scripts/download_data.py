"""Download do Medical Abstracts TC Corpus (GitHub público)."""

import sys
from pathlib import Path
from urllib.request import urlretrieve

from src.triage.config import DATA_RAW_DIR

BASE_URL = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main"
FILES = ("medical_tc_train.csv", "medical_tc_test.csv")


def download_file(url: str, destination: Path) -> None:
    """Baixa um arquivo para o destino informado."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {url} -> {destination}")
    urlretrieve(url, destination)


def main() -> None:
    for filename in FILES:
        url = f"{BASE_URL}/{filename}"
        destination = DATA_RAW_DIR / filename
        download_file(url, destination)
    print(f"Dataset salvo em {DATA_RAW_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro no download: {exc}", file=sys.stderr)
        sys.exit(1)
