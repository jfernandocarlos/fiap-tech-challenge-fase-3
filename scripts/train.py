"""Script CLI de treinamento."""

import argparse

from src.triage.models.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina classificador de triagem")
    parser.add_argument(
        "--source",
        choices=["raw", "processed", "sample"],
        default="raw",
        help="Origem dos dados",
    )
    args = parser.parse_args()
    _, metrics = train_model(source=args.source)
    print(f"Treino concluído — F1 macro: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
