import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import (
    ARTIFACT_DIR,
    CLEAN_CSV,
    DRIFT_CONSECUTIVE_HOURS,
    DRIFT_MULTIPLIER,
    DRIFT_WINDOW_HOURS,
    FORECAST_HOURS,
    SOURCE_URL,
)
from .data import load_clean_csv
from .drift import monitor_errors
from .evaluation import chronological_split, score_forecast
from .features import FEATURE_COLUMNS, usable_samples
from .models import BASELINE_NAME, BOOSTING_NAME, fit_boosting, predict_models


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data_integrity(clean_csv: Path, metadata: dict) -> None:
    if not clean_csv.is_file():
        raise FileNotFoundError(
            f"Clean CSV not found: {clean_csv}. Run 'python scripts/download_data.py' first."
        )
    if _file_sha256(clean_csv) != metadata["data_sha256"]:
        raise ValueError(
            "Clean dataset changed since training. Re-run train.py and evaluate.py."
        )


def _prepare_data(clean_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable = usable_samples(load_clean_csv(clean_csv))
    return chronological_split(usable)


def train_pipeline(
    clean_csv: Path = CLEAN_CSV,
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict:
    train, validation, test = _prepare_data(clean_csv)
    training_mean = float(train["power_mw"].mean())
    if training_mean <= 0:
        raise ValueError("Training mean must be positive for normalized evaluation.")
    model = fit_boosting(train)
    predictions = predict_models(validation, model)
    validation_scores = {
        name: score_forecast(validation["power_mw"], values, training_mean)
        for name, values in predictions.items()
    }
    # Test labels are not used for fitting, hyperparameter tuning or model selection.
    selected = min(validation_scores, key=lambda name: validation_scores[name]["mae_mw"])
    metadata = {
        "data_sha256": _file_sha256(clean_csv),
        "data_path": str(clean_csv.resolve()),
        "default_source_url": SOURCE_URL,
        "forecast_horizon_hours": FORECAST_HOURS,
        "forecast_origin": "End of hour t-1, once the actual for t-1 is available",
        "feature_columns": list(FEATURE_COLUMNS),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "train_end": train.index[-1].isoformat(),
        "validation_start": validation.index[0].isoformat(),
        "validation_end": validation.index[-1].isoformat(),
        "test_start": test.index[0].isoformat(),
        "test_end": test.index[-1].isoformat(),
        "training_mean_mw": training_mean,
        "selected_model": selected,
        "validation_scores": validation_scores,
        "drift_window_hours": DRIFT_WINDOW_HOURS,
        "drift_multiplier": DRIFT_MULTIPLIER,
        "drift_consecutive_hours": DRIFT_CONSECUTIVE_HOURS,
        "drift_reference_mae_mw": validation_scores[selected]["mae_mw"],
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in ("metrics.csv", "predictions.csv"):
        (artifact_dir / stale_file).unlink(missing_ok=True)
    joblib.dump(model, artifact_dir / "model.joblib")
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def load_artifacts(artifact_dir: Path = ARTIFACT_DIR) -> tuple[object, dict]:
    model_file = artifact_dir / "model.joblib"
    metadata_file = artifact_dir / "metadata.json"
    if not model_file.is_file() or not metadata_file.is_file():
        raise FileNotFoundError(
            f"Missing trained artifacts in {artifact_dir}. Run 'python scripts/train.py' first."
        )
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata["feature_columns"] != list(FEATURE_COLUMNS):
        raise ValueError("Saved feature configuration does not match the current code.")
    # joblib should only be loaded from locally generated, trusted artifact files.
    model = joblib.load(model_file)
    return model, metadata


def evaluate_pipeline(
    clean_csv: Path = CLEAN_CSV,
    artifact_dir: Path = ARTIFACT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model, metadata = load_artifacts(artifact_dir)
    verify_data_integrity(clean_csv, metadata)
    train, validation, test = _prepare_data(clean_csv)
    if (test.index[0].isoformat() != metadata["test_start"]
            or test.index[-1].isoformat() != metadata["test_end"]
            or len(test) != metadata["test_samples"]
            or train.index[-1].isoformat() != metadata["train_end"]
            or validation.index[-1].isoformat() != metadata["validation_end"]):
        raise ValueError("Split boundaries differ from the saved training run.")

    predictions = predict_models(test, model, metadata["feature_columns"])
    metrics = pd.DataFrame([
        {"model": name, **score_forecast(
            test["power_mw"], values, metadata["training_mean_mw"]
        )}
        for name, values in predictions.items()
    ])
    results = pd.DataFrame({
        "actual_mw": test["power_mw"],
        f"{BASELINE_NAME}_mw": predictions[BASELINE_NAME],
        f"{BOOSTING_NAME}_mw": predictions[BOOSTING_NAME],
    }, index=test.index)
    selected = metadata["selected_model"]
    results["selected_prediction_mw"] = results[f"{selected}_mw"]
    monitor = monitor_errors(
        results["actual_mw"], results["selected_prediction_mw"],
        metadata["drift_reference_mae_mw"],
        metadata["drift_window_hours"],
        metadata["drift_multiplier"],
        metadata["drift_consecutive_hours"],
    )
    results = results.join(monitor)
    results.index.name = "timestamp"
    metrics.to_csv(artifact_dir / "metrics.csv", index=False)
    results.to_csv(artifact_dir / "predictions.csv", date_format="%Y-%m-%dT%H:%M:%SZ")
    return metrics, results
