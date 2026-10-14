import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import MIN_SAMPLES_PER_SPLIT, TRAIN_FRACTION, VALIDATION_FRACTION


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split valid supervised examples by timestamp, without shuffling."""
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Samples must have unique, sorted timestamps.")
    size = len(frame)
    train_end = int(size * TRAIN_FRACTION)
    val_end = int(size * (TRAIN_FRACTION + VALIDATION_FRACTION))
    if min(train_end, val_end - train_end, size - val_end) < MIN_SAMPLES_PER_SPLIT:
        raise ValueError(
            "Not enough usable hourly samples for three chronological splits "
            f"of at least {MIN_SAMPLES_PER_SPLIT} samples each."
        )
    return frame.iloc[:train_end], frame.iloc[train_end:val_end], frame.iloc[val_end:]


def score_forecast(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    training_mean_mw: float,
) -> dict[str, float]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if (actual_values.ndim != 1 or actual_values.size == 0
            or actual_values.shape != predicted_values.shape):
        raise ValueError("Actual and predicted must be equally sized, nonempty 1D arrays.")
    if not (np.isfinite(actual_values).all() and np.isfinite(predicted_values).all()):
        raise ValueError("Evaluation values must be finite.")
    if not np.isfinite(training_mean_mw) or training_mean_mw <= 0:
        raise ValueError("Training mean must be positive for normalized errors.")
    mae = float(mean_absolute_error(actual_values, predicted_values))
    rmse = float(np.sqrt(mean_squared_error(actual_values, predicted_values)))
    return {
        "mae_mw": mae,
        "rmse_mw": rmse,
        "nmae_pct": 100.0 * mae / training_mean_mw,
    }
