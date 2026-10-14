import numpy as np
import pandas as pd

from .config import DRIFT_CONSECUTIVE_HOURS, DRIFT_MULTIPLIER, DRIFT_WINDOW_HOURS


def monitor_errors(
    actual: pd.Series,
    predicted: pd.Series,
    validation_mae_mw: float,
    window_hours: int = DRIFT_WINDOW_HOURS,
    multiplier: float = DRIFT_MULTIPLIER,
    consecutive_hours: int = DRIFT_CONSECUTIVE_HOURS,
) -> pd.DataFrame:
    """Flag sustained test rolling-MAE deterioration relative to validation MAE."""
    if (not np.isfinite(validation_mae_mw) or validation_mae_mw < 0
            or window_hours < 1 or multiplier <= 0 or consecutive_hours < 1):
        raise ValueError("Invalid validation reference or drift rule parameters.")
    if not isinstance(actual.index, pd.DatetimeIndex) or not actual.index.equals(predicted.index):
        raise ValueError("Actual and predicted series must share a DatetimeIndex.")
    if actual.empty or not actual.index.is_monotonic_increasing or actual.index.has_duplicates:
        raise ValueError("The time index must be nonempty, sorted, and unique.")
    if actual.index.tz is None:
        raise ValueError("Timestamps must be timezone-aware.")

    error = (actual.astype(float) - predicted.astype(float)).abs().asfreq("h")
    rolling = error.rolling(window_hours, min_periods=window_hours).mean()
    threshold = validation_mae_mw * multiplier
    breach = rolling.gt(threshold)
    consecutive = breach.astype(int).rolling(
        consecutive_hours, min_periods=consecutive_hours
    ).sum()
    return pd.DataFrame({
        "absolute_error_mw": error,
        "rolling_mae_mw": rolling,
        "drift_breach": breach,
        "drift_warning": consecutive.eq(consecutive_hours),
    })
