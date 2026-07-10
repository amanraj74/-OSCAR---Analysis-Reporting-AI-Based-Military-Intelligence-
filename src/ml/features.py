"""Feature engineering for ML models.

Builds a per-(date, country) feature matrix from the silver `events_per_country_day`
Parquet table. Adds lag features, rolling statistics, and calendar features.

Public API
----------
    build_feature_matrix(silver_path, horizons=(1, 3, 7)) -> tuple[pd.DataFrame, list[str]]
    MaterializedFeatureMatrix   : dataclass holding X (features), y_dict (per-horizon targets)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_settings
from src.observability import get_logger

logger = get_logger("ml.features")


_REQUIRED_COLS = ["date", "actor1_country_code", "event_count", "conflict_count"]
_OPTIONAL_COLS = ["avg_goldstein", "avg_tone", "total_mentions", "total_articles", "conflict_ratio"]

_LAG_DAYS = [1, 3, 7]
_ROLLING_WINDOWS = [3, 7, 14]

_CALENDAR_COLS = ["day_of_week", "month", "is_weekend"]


@dataclass
class MaterializedFeatureMatrix:
    """Holds engineered features and per-horizon targets."""

    features: pd.DataFrame
    target_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    horizon_days: tuple[int, ...] = ()

    def get_target(self, horizon: int) -> pd.Series | None:
        col = f"target_h{horizon}"
        if col in self.features.columns:
            return self.features[col]
        return None

    def to_xy(self, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
        if f"target_h{horizon}" not in self.features.columns:
            raise ValueError(f"horizon={horizon} target not found in features")
        y = self.features[f"target_h{horizon}"]
        X = self.features[self.feature_columns].copy()
        return X, y


def _load_silver(path: Path) -> pd.DataFrame:
    """Load and validate silver Parquet."""
    if not path.exists():
        raise FileNotFoundError(f"Silver table not found: {path}")
    df = pd.read_parquet(path)
    for c in _REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Silver table missing required column: {c}")
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def _add_lag_features(
    df: pd.DataFrame,
    group_col: str,
    value_cols: list[str],
    lag_days: list[int],
) -> pd.DataFrame:
    df = df.sort_values([group_col, "date"]).reset_index(drop=True)
    for col in value_cols:
        if col not in df.columns:
            continue
        for lag in lag_days:
            df[f"{col}_lag{lag}d"] = df.groupby(group_col)[col].shift(lag).astype("float32")
    return df


def _add_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    value_cols: list[str],
    windows: list[int],
) -> pd.DataFrame:
    for col in value_cols:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_roll{w}d_mean"] = (
                df.groupby(group_col)[col]
                .transform(lambda s: s.rolling(w, min_periods=1).mean())
                .astype("float32")
            )
            df[f"{col}_roll{w}d_std"] = (
                df.groupby(group_col)[col]
                .transform(lambda s: s.rolling(w, min_periods=2).std())
                .astype("float32")
            )
    return df


def _add_targets(
    df: pd.DataFrame,
    horizons: tuple[int, ...],
    escalation_threshold: float,
) -> pd.DataFrame:
    """Compute binary target: will `event_count AND conflict_count` jump in `h` days?

    Definition of "escalation" at horizon h:
        conflict_count(t+h) > mean(conflict_count of last 7d at t) + 1 standard deviation
    """
    for h in horizons:
        future_conflict = df.groupby("actor1_country_code")["conflict_count"].shift(-h)
        baseline_mean = df.groupby("actor1_country_code")["conflict_count"].transform(
            lambda s: s.rolling(7, min_periods=1).mean()
        )
        baseline_std = df.groupby("actor1_country_code")["conflict_count"].transform(
            lambda s: s.rolling(7, min_periods=2).std()
        )
        threshold_value = baseline_mean + escalation_threshold * baseline_std.fillna(0.0)
        df[f"target_h{h}"] = (future_conflict > threshold_value).astype(int)

    return df


def build_feature_matrix(
    silver_path: Path | None = None,
    horizons: tuple[int, ...] = (1, 3, 7),
    escalation_threshold_sigma: float = 1.0,
) -> MaterializedFeatureMatrix:
    """Build ML-ready feature matrix from silver events table.

    Args:
        silver_path: Path to events_per_country_day.parquet (default from settings).
        horizons: Forecast horizons in days for target labels.
        escalation_threshold_sigma: Sigma multiplier for escalation threshold.

    Returns:
        MaterializedFeatureMatrix with features, target_columns, feature_columns, horizon_days.
    """
    settings = get_settings()
    if silver_path is None:
        silver_path = settings.processed_data_dir / "silver" / "events_per_country_day.parquet"

    df = _load_silver(silver_path)
    df = df.rename(columns={"sql_date": "date"}) if "sql_date" in df.columns else df
    df["date"] = pd.to_datetime(df["date"])

    base_cols = ["event_count", "conflict_count", "conflict_ratio"]
    all_value_cols = base_cols + _OPTIONAL_COLS
    for c in all_value_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].astype("float32").fillna(0.0)

    df = df.sort_values(["actor1_country_code", "date"]).reset_index(drop=True)
    df = _add_lag_features(df, "actor1_country_code", base_cols, _LAG_DAYS)
    df = _add_rolling_features(df, "actor1_country_code", base_cols, _ROLLING_WINDOWS)
    df = _add_calendar_features(df)
    df = _add_targets(df, horizons, escalation_threshold_sigma)

    target_cols = [f"target_h{h}" for h in horizons]
    feature_cols: list[str] = []
    seen: set[str] = set()
    for col in all_value_cols:
        for lag in _LAG_DAYS:
            fc = f"{col}_lag{lag}d"
            if fc in df.columns and fc not in seen:
                feature_cols.append(fc)
                seen.add(fc)
    for col in all_value_cols:
        for w in _ROLLING_WINDOWS:
            for stat in ("mean", "std"):
                fc = f"{col}_roll{w}d_{stat}"
                if fc in df.columns and fc not in seen:
                    feature_cols.append(fc)
                    seen.add(fc)
    feature_cols.extend(_CALENDAR_COLS)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    logger.info(
        "feature_matrix_built",
        rows=len(df),
        n_features=len(feature_cols),
        horizons=horizons,
    )

    return MaterializedFeatureMatrix(
        features=df,
        target_columns=target_cols,
        feature_columns=feature_cols,
        horizon_days=horizons,
    )


def featurize_single_observation(observation: dict[str, float]) -> dict[str, float]:
    """Convert a single (country, day) observation dict to a flat feature dict.

    Used for inference-time scoring when only the latest values are known.
    """
    out: dict[str, float] = {}
    for k, v in observation.items():
        out[str(k)] = float(v) if v is not None else 0.0
    return out


__all__ = [
    "MaterializedFeatureMatrix",
    "build_feature_matrix",
    "featurize_single_observation",
]
