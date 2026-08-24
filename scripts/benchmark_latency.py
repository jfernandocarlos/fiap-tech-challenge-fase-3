"""Benchmark de latência sklearn vs ONNX."""

from src.triage.evaluation.latency import run_benchmark


def main() -> None:
    report = run_benchmark()
    print("Benchmark concluído:")
    print(f"  sklearn mean: {report['sklearn']['mean_ms']} ms")
    print(f"  onnx mean:    {report['onnx']['mean_ms']} ms")
    print(f"  melhoria:     {report['mean_improvement_percent']}%")
    print("  artefatos:    models/latency_report.json, docs/latency_comparison.md")


if __name__ == "__main__":
    main()
