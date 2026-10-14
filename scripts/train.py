import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from renewcast.pipeline import train_pipeline  # noqa: E402


if __name__ == "__main__":
    try:
        metadata = train_pipeline()
    except (OSError, ValueError) as exc:
        sys.exit(f"Training failed: {exc}")
    print(f"Trained on {metadata['train_samples']:,} hourly examples")
    print(f"Validation: {metadata['validation_start']} through {metadata['validation_end']}")
    print(f"Untouched test: {metadata['test_start']} through {metadata['test_end']}")
    for name, scores in metadata["validation_scores"].items():
        print(f"Validation {name}: MAE={scores['mae_mw']:.2f} MW; RMSE={scores['rmse_mw']:.2f} MW")
    print(f"Selected by validation MAE: {metadata['selected_model']}")
    print("Saved artifacts/model.joblib and artifacts/metadata.json")
