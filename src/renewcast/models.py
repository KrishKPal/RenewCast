from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import LONGEST_LOOKBACK_HOURS
from .features import FEATURE_COLUMNS, make_features

BASELINE_NAME = "seasonal_naive_24h"
BOOSTING_NAME = "gradient_boosting"


def fit_boosting(train: pd.DataFrame) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        learning_rate=0.07,
        max_iter=150,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(train[list(FEATURE_COLUMNS)], train["power_mw"])
    return model


def predict_models(
    frame: pd.DataFrame,
    model: HistGradientBoostingRegressor,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> dict[str, np.ndarray]:
    return {
        BASELINE_NAME: frame["lag_24h"].to_numpy(dtype=float),
        BOOSTING_NAME: np.maximum(0.0, model.predict(frame[list(feature_columns)])),
    }


def predict_next_hour(
    history: pd.Series,
    forecast_time: pd.Timestamp,
    model: HistGradientBoostingRegressor | None,
    model_name: str = BOOSTING_NAME,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> float:
    """Forecast t when actual observations through t-1 have become available."""
    forecast_time = pd.Timestamp(forecast_time)
    if forecast_time.tzinfo is None or str(forecast_time.tz_convert("UTC").tz) != "UTC":
        raise ValueError("forecast_time must be timezone-aware.")
    forecast_time = forecast_time.tz_convert("UTC")
    if history.empty or not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("history must be a UTC-indexed series of past actual generation.")
    if history.index.max() != forecast_time - pd.Timedelta(hours=1):
        raise ValueError("History must end exactly one hour before the forecast time.")
    if len(history) < LONGEST_LOOKBACK_HOURS:
        raise ValueError(f"At least {LONGEST_LOOKBACK_HOURS} previous hours are required.")
    recent = history.iloc[-LONGEST_LOOKBACK_HOURS:]
    if not np.isfinite(recent.to_numpy(dtype=float)).all() or (recent < 0).any():
        raise ValueError("History must contain finite, nonnegative observed generation.")
    extended = pd.concat([
        recent.rename("power_mw"),
        pd.Series([np.nan], index=pd.DatetimeIndex([forecast_time]), name="power_mw"),
    ])
    row = make_features(extended.to_frame()).loc[[forecast_time]]
    if row[list(feature_columns)].isna().any().any():
        raise ValueError("Incomplete historical observations in the lookback window.")
    if model_name == BASELINE_NAME:
        return float(row["lag_24h"].iloc[0])
    if model_name != BOOSTING_NAME or model is None:
        raise ValueError(f"Unknown or unavailable forecasting model: {model_name}")
    return float(max(0, model.predict(row[list(feature_columns)])[0]))
