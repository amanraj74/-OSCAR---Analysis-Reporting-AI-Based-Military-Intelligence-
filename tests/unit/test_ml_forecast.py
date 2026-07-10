"""Tests for the forecaster."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_series() -> pd.Series:
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    import numpy as np

    values = np.linspace(10, 30, 30) + np.random.default_rng(0).normal(0, 2, 30)
    return pd.Series(values, index=dates, name="event_count")


def test_forecaster_initializes() -> None:
    from src.ml.forecast import Forecaster

    fc = Forecaster(force_mode="linear")
    assert fc.mode == "linear"


def test_forecaster_fit_predict(sample_series) -> None:
    from src.ml.forecast import Forecaster

    fc = Forecaster(force_mode="linear")
    fc.fit(sample_series, region="TEST")
    pred = fc.predict(periods=7)
    assert len(pred) == 7
    assert "ds" in pred.columns
    assert "yhat" in pred.columns
    assert "yhat_lower" in pred.columns
    assert "yhat_upper" in pred.columns


def test_forecaster_score(sample_series) -> None:
    from src.ml.forecast import Forecaster

    fc = Forecaster(force_mode="linear")
    fc.fit(sample_series, region="TEST")
    m = fc.score()
    assert "mae" in m
    assert "rmse" in m
    assert "mape" in m
    assert m["mae"] >= 0


def test_forecaster_forecast_result(sample_series) -> None:
    from src.ml.forecast import Forecaster

    fc = Forecaster(force_mode="linear")
    fc.fit(sample_series, region="UKR")
    result = fc.forecast(periods=7)
    assert result.region == "UKR"
    assert result.horizon_days == 7
    assert result.model == "linear"
    assert len(result.forecast) == 7


def test_forecaster_unfitted_raises() -> None:
    from src.ml.forecast import Forecaster

    fc = Forecaster(force_mode="linear")
    with pytest.raises(RuntimeError):
        fc.predict(periods=7)


def test_linear_backend_basic() -> None:
    import numpy as np

    from src.ml.forecast import _LinearRegressionBackend

    backend = _LinearRegressionBackend()
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    backend.fit(y)
    yhat, lo, hi = backend.predict(3)
    assert len(yhat) == 3
    assert all(lo[i] <= yhat[i] <= hi[i] for i in range(3))


def test_linear_backend_short_series() -> None:
    import numpy as np

    from src.ml.forecast import _LinearRegressionBackend

    backend = _LinearRegressionBackend()
    backend.fit(np.array([5.0]))
    yhat, lo, hi = backend.predict(3)
    assert all(yhat == 5.0)
    assert all(lo == hi)


def test_fit_per_region() -> None:
    import numpy as np

    from src.ml.forecast import fit_per_region

    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    rng = np.random.default_rng(0)
    rows = []
    for region in ["USA", "RUS"]:
        vals = rng.integers(10, 50, 30).tolist()
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "actor1_country_code": region,
                    "event_count": vals[i],
                    "conflict_count": int(vals[i] * 0.3),
                }
            )
    df = pd.DataFrame(rows)
    out = fit_per_region(df, target_col="event_count", periods=5, force_mode="linear")
    assert len(out) == 2
    for f in out:
        assert len(f.forecast) == 5
        assert f.model == "linear"


def test_forecaster_region_property(sample_series) -> None:
    from src.ml.forecast import Forecaster

    fc = Forecaster(force_mode="linear")
    assert fc.region is None
    fc.fit(sample_series, region="UKR")
    assert fc.region == "UKR"
