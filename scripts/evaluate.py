import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from renewcast.pipeline import evaluate_pipeline  # noqa: E402


if __name__ == "__main__":
    try:
        metrics, predictions = evaluate_pipeline()
    except (OSError, ValueError, KeyError) as exc:
        sys.exit(f"Evaluation failed: {exc}")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Test predictions: {len(predictions):,}; drift warnings: {int(predictions['drift_warning'].sum())}")
    print("Saved artifacts/metrics.csv and artifacts/predictions.csv")
