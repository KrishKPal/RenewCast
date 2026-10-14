import numpy as np
import pandas as pd
import pytest

from renewcast.evaluation import chronological_split, score_forecast


def test_error_metrics_exact():
    result = score_forecast(np.array([10, 20]), np.array([12, 16]), 15)
    assert result["mae_mw"] == pytest.approx(3)
    assert result["rmse_mw"] == pytest.approx(np.sqrt(10))
    assert result["nmae_pct"] == pytest.approx(20)


def test_chronological_split_never_shuffles():
    index = pd.date_range("2020-01-01", periods=200, freq="h", tz="UTC")
    samples = pd.DataFrame({"power_mw": np.arange(200)}, index=index)
    train, validation, test = chronological_split(samples)
    assert (len(train), len(validation), len(test)) == (140, 30, 30)
    assert train.index[-1] < validation.index[0] < validation.index[-1] < test.index[0]


def test_too_few_samples_rejected():
    index = pd.date_range("2020-01-01", periods=40, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="Not enough"):
        chronological_split(pd.DataFrame(index=index))


def test_invalid_normalization_rejected():
    with pytest.raises(ValueError, match="positive"):
        score_forecast([1, 2], [1, 2], 0)
