"""Time-series forecaster for OSCAR.

Prophet primary backend (when installed) with linear regression fallback.
Forecasts event volume + conflict events per region with confidence intervals.

Public API
----------
    Forecaster().fit(series, region)         -> self
    Forecaster().predict(periods=7)          -> pd.DataFrame (ds, yhat, yhat_lower, yhat_upper)
    Forecaster().score()                     -> dict[str, float]
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.ml import HAS_PROPHET, logger


@dataclass
class ForecastResult:
    """A forecast for one region."""

    region: str
    horizon_days: int
    forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict[str, float] = field(default_factory=dict)
    model: str = ""
    in_sample_mae: float = 0.0
    in_sample_rmse: float = 0.0
    in_sample_mape: float = 0.0


class _LinearRegressionBackend:
    """Linear regression forecaster (always-available, no extra deps).

    Fits y = a + b*t to the series and projects forward.
    Confidence interval computed from residual std.
    """

    def __init__(self) -> None:
        self._coef_a: float = 0.0
        self._coef_b: float = 0.0
        self._residual_std: float = 0.0
        self._fitted: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> None:
        n = len(y)
        if n < 2:
            self._coef_a = float(y[0]) if n == 1 else 0.0
            self._coef_b = 0.0
            self._residual_std = 0.0
            self._fitted = y.astype(float).copy()
            return
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            self._coef_a = float(y_mean)
            self._coef_b = 0.0
        else:
            self._coef_b = float(((x - x_mean) * (y - y_mean)).sum() / denom)
            self._coef_a = float(y_mean - self._coef_b * x_mean)
        self._fitted = self._coef_a + self._coef_b * x
        residuals = y - self._fitted
        self._residual_std = float(residuals.std()) if len(residuals) > 1 else 0.0

    def predict(self, n_periods: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(self._fitted) if self._fitted is not None else 0
        x_future = np.arange(n, n + n_periods, dtype=float)
        yhat = self._coef_a + self._coef_b * x_future
        band = 1.96 * self._residual_std
        return yhat, yhat - band, yhat + band


class _ProphetBackend:
    """Prophet forecaster. Requires `prophet` package."""

    def __init__(self, weekly_seasonality: bool = True, yearly_seasonality: bool = False) -> None:
        from prophet import Prophet

        self._model = Prophet(
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            interval_width=0.95,
            daily_seasonality=False,
        )

    def fit(self, df: pd.DataFrame) -> None:
        self._model.fit(df)

    def predict(self, periods: int) -> pd.DataFrame:
        future = self._model.make_future_dataframe(periods=periods)
        return self._model.predict(future).tail(periods).reset_index(drop=True)


class Forecaster:
    """Prophet primary + linear regression fallback."""

    def __init__(self, force_mode: str | None = None) -> None:
        self._backend: Any = None
        self._mode = "prophet" if (HAS_PROPHET and (force_mode in (None, "prophet"))) else "linear"
        if force_mode == "linear":
            self._mode = "linear"
        self._region: str | None = None
        self._series: pd.Series | None = None
        self._fitted = False

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def region(self) -> str | None:
        return self._region

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return
        if self._mode == "prophet" and HAS_PROPHET:
            try:
                self._backend = _ProphetBackend()
            except Exception as e:
                logger.warning("prophet_fallback_to_linear", error=str(e))
                self._mode = "linear"
        if self._mode == "linear":
            self._backend = _LinearRegressionBackend()

    def fit(self, series: pd.Series, region: str) -> Forecaster:
        """Fit on a univariate time series indexed by date.

        Args:
            series: pd.Series with DatetimeIndex, daily granularity.
            region: region label (for result metadata).
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("series must have a DatetimeIndex")
        series = series.sort_index().asfreq("D").fillna(0.0)

        self._region = region
        self._series = series.astype(float)
        self._ensure_backend()

        if self._mode == "prophet":
            df = pd.DataFrame({"ds": series.index, "y": series.values})
            self._backend.fit(df)
        else:
            self._backend.fit(series.values.astype(float))
        self._fitted = True
        return self

    def predict(self, periods: int = 7) -> pd.DataFrame:
        if not self._fitted or self._backend is None:
            raise RuntimeError("Forecaster not fitted; call .fit() first")
        if self._mode == "prophet":
            future = self._backend.predict(periods)
            return future[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        yhat, lower, upper = self._backend.predict(periods)
        last_date = self._series.index[-1] if self._series is not None else pd.Timestamp.today()
        future_idx = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=periods, freq="D"
        )
        return pd.DataFrame(
            {"ds": future_idx, "yhat": yhat, "yhat_lower": lower, "yhat_upper": upper}
        )

    def score(self) -> dict[str, float]:
        """In-sample fit metrics."""
        if not self._fitted or self._backend is None or self._series is None:
            return {}
        y_true = self._series.values
        if self._mode == "prophet":
            fitted = self._backend.predict(-len(y_true))
            y_pred = fitted["yhat"].values
        else:
            y_pred = self._backend._fitted  # type: ignore[attr-defined]
        if y_pred is None:
            return {}
        err = y_true - y_pred
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        nonzero = np.abs(y_true) > 1e-6
        mape = (
            float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100.0) if nonzero.any() else 0.0
        )
        return {"mae": mae, "rmse": rmse, "mape": mape}

    def forecast(self, periods: int = 7) -> ForecastResult:
        if self._region is None:
            raise RuntimeError("Forecaster not fitted; call .fit() first")
        forecast_df = self.predict(periods=periods)
        return ForecastResult(
            region=self._region,
            horizon_days=periods,
            forecast=forecast_df,
            metrics=self.score(),
            model=self._mode,
        )


def fit_per_region(
    features_df: pd.DataFrame,
    target_col: str = "event_count",
    regions: Iterable[str] | None = None,
    periods: int = 7,
    force_mode: str | None = None,
) -> list[ForecastResult]:
    """Fit one Forecaster per region; return forecasts."""
    if "actor1_country_code" not in features_df.columns or "date" not in features_df.columns:
        raise ValueError("features_df must have 'date' and 'actor1_country_code' columns")

    df = features_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["actor1_country_code", "date"])

    regions_list = (
        list(regions) if regions is not None else df["actor1_country_code"].unique().tolist()
    )
    out: list[ForecastResult] = []
    for region in regions_list:
        sub = df[df["actor1_country_code"] == region]
        if len(sub) < 5:
            continue
        series = pd.Series(sub[target_col].values, index=sub["date"])
        fc = Forecaster(force_mode=force_mode)
        try:
            fc.fit(series, region=region)
            out.append(fc.forecast(periods=periods))
        except Exception as e:
            logger.warning("forecast_failed", region=region, error=str(e))
    return out


__all__ = ["Forecaster", "ForecastResult", "fit_per_region"]
