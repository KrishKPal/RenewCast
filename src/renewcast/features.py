import numpy as np
import pandas as pd

from .config import LONGEST_LOOKBACK_HOURS

FEATURE_COLUMNS = (
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "mean_previous_24h",
    "mean_previous_168h",
    "std_previous_24h",
    "hour_sin",
    "hour_cos",
    "year_day_sin",
    "year_day_cos",
    "day_of_week",
)


def make_features(generation: pd.DataFrame) -> pd.DataFrame:
    """Features for hour t use only actual generation from hours earlier than t."""
    if not isinstance(generation.index, pd.DatetimeIndex):
        raise ValueError("Generation must have a DatetimeIndex.")
    if generation.index.tz is None or not generation.index.is_monotonic_increasing:
        raise ValueError("Generation index must be sorted and timezone-aware.")
    if generation.index.has_duplicates or not generation.index.equals(
        pd.date_range(generation.index.min(), generation.index.max(), freq="h")
    ):
        raise ValueError("Generation index must be a unique, continuous hourly UTC grid.")
    if str(generation.index.tz) != "UTC":
        raise ValueError("Generation timestamps must be in UTC.")

    previous = generation["power_mw"].shift(1)
    index = generation.index
    features = pd.DataFrame(index=index)
    features["lag_1h"] = previous
    features["lag_24h"] = generation["power_mw"].shift(24)
    features["lag_168h"] = generation["power_mw"].shift(LONGEST_LOOKBACK_HOURS)
    features["mean_previous_24h"] = previous.rolling(24, min_periods=24).mean()
    features["mean_previous_168h"] = previous.rolling(168, min_periods=168).mean()
    features["std_previous_24h"] = previous.rolling(24, min_periods=24).std()
    features["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
    features["year_day_sin"] = np.sin(2 * np.pi * index.dayofyear / 366)
    features["year_day_cos"] = np.cos(2 * np.pi * index.dayofyear / 366)
    features["day_of_week"] = index.dayofweek
    features["power_mw"] = generation["power_mw"]
    return features


def usable_samples(generation: pd.DataFrame) -> pd.DataFrame:
    frame = make_features(generation)
    return frame.dropna(subset=[*FEATURE_COLUMNS, "power_mw"])
