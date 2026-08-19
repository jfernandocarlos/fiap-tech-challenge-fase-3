import importlib
import sys
from pathlib import Path

# Só o que já foi instalado até o commit 9
REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "sklearn",
    "fastapi",
    "uvicorn",
    "structlog",
    "pydantic_settings",
    "yaml",
    "joblib",
    "prometheus_client",
]
MIN_PYTHON = (3, 10)


def _check_python() -> list[str]:
    if sys.version_info < MIN_PYTHON:
        return [
            (
                f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ requerido "
                f"(atual: {sys.version.split()[0]})"
            )
        ]
    return []


def _check_packages() -> list[str]:
    problems: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            problems.append(f"pacote ausente: {package}")
    return problems


def _check_files() -> list[str]:
    root = Path(__file__).resolve().parents[1]
    problems: list[str] = []
    if not (root / "params.yaml").exists():
        problems.append("params.yaml ausente")
    if not (root / ".env").exists() and not (root / ".env.example").exists():
        problems.append(".env / .env.example ausente")
    return problems


def main() -> int:
    problems = _check_python() + _check_packages() + _check_files()
    if problems:
        print("Ambiente INVÁLIDO:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("Ambiente OK: Python, dependências e configs validados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
