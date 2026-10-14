import numpy as np
import pandas as pd
import pytest

from renewcast.drift import monitor_errors


def test_warning_needs_complete_windows_and_consecutive_breaches():
    index = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    actual = pd.Series(np.array([0] * 10, dtype=float), index=index)
    predicted = pd.Series(np.array([0, 0, 0, 2, 2, 2, 2, 2, 2, 2], dtype=float), index=index)
    result = monitor_errors(actual, predicted, validation_mae_mw=1,
                            window_hours=3, multiplier=1.5, consecutive_hours=2)
    assert not result["drift_warning"].iloc[:6].any()
    assert result["drift_warning"].iloc[6]
    assert result["rolling_mae_mw"].iloc[6] == pytest.approx(2)


def test_missing_hour_breaks_rolling_window():
    full_index = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    index = full_index.delete(5)
    actual = pd.Series(0.0, index=index)
    predicted = pd.Series(10.0, index=index)
    result = monitor_errors(actual, predicted, 1, window_hours=3,
                            consecutive_hours=1)
    assert not result.loc[full_index[6], "drift_warning"]
    assert result.loc[full_index[8], "drift_warning"]


def test_zero_reference_can_flag_positive_error():
    index = pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")
    result = monitor_errors(pd.Series(1.0, index=index),
                            pd.Series(0.0, index=index), 0,
                            window_hours=2, consecutive_hours=1)
    assert result["drift_warning"].iloc[1]
