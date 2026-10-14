import numpy as np
import pandas as pd
import pytest

from renewcast.data import _validate_hourly_values
from renewcast.features import make_features
from renewcast.models import BASELINE_NAME, predict_next_hour


def sample_generation(n: int = 250) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"power_mw": np.arange(n, dtype=float)}, index=index)


def test_lags_rolling_and_no_current_target_leakage():
    frame = sample_generation()
    features = make_features(frame)
    target_time = frame.index[200]
    assert features.loc[target_time, "lag_1h"] == 199
    assert features.loc[target_time, "lag_24h"] == 176
    assert features.loc[target_time, "lag_168h"] == 32
    assert features.loc[target_time, "mean_previous_24h"] == pytest.approx(187.5)
    assert features.loc[target_time, "mean_previous_168h"] == pytest.approx(115.5)

    changed = frame.copy()
    changed.loc[target_time, "power_mw"] = 100000
    changed_features = make_features(changed)
    pd.testing.assert_series_equal(
        features.loc[target_time].drop("power_mw"),
        changed_features.loc[target_time].drop("power_mw"),
    )
    assert changed_features.loc[frame.index[201], "lag_1h"] == 100000


def test_next_hour_baseline_is_seasonal_lag():
    frame = sample_generation()
    target_time = frame.index[-1]
    prediction = predict_next_hour(
        frame["power_mw"].iloc[:-1], target_time, None, BASELINE_NAME
    )
    assert prediction == frame["power_mw"].iloc[-25]
    with pytest.raises(ValueError, match="end exactly"):
        predict_next_hour(frame["power_mw"], target_time, None, BASELINE_NAME)


def test_missing_hour_is_not_filled_from_the_future():
    frame = sample_generation()
    frame.loc[frame.index[170], "power_mw"] = np.nan
    features = make_features(frame)
    assert np.isnan(features.loc[frame.index[171], "lag_1h"])
    assert np.isnan(features.loc[frame.index[180], "mean_previous_24h"])


def test_raw_validation_rejects_duplicates_negative_and_bad_timestamps():
    raw = pd.DataFrame({
        "timestamp": ["2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z"],
        "power_mw": [1, 2],
    })
    with pytest.raises(ValueError, match="Duplicate"):
        _validate_hourly_values(raw)
    raw.loc[1, "timestamp"] = "2020-01-01T01:00:00Z"
    raw.loc[1, "power_mw"] = -1
    with pytest.raises(ValueError, match="Negative"):
        _validate_hourly_values(raw)
    raw.loc[1, "power_mw"] = 1
    raw.loc[1, "timestamp"] = "bad-timestamp"
    with pytest.raises(ValueError, match="Invalid"):
        _validate_hourly_values(raw)
