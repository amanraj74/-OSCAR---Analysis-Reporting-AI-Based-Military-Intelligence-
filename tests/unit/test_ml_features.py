"""Tests for ML feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def silver_features() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    regions = ["USA", "RUS", "UKR"]
    rows = []
    rng = np.random.default_rng(42)
    for region in regions:
        base = rng.integers(5, 50, size=len(dates))
        for i, d in enumerate(dates):
            conflict = int(base[i] * 0.3 + rng.integers(0, 5))
            rows.append(
                {
                    "date": d,
                    "actor1_country_code": region,
                    "event_count": int(base[i]),
                    "conflict_count": conflict,
                    "conflict_ratio": conflict / max(int(base[i]), 1),
                    "avg_goldstein": float(rng.uniform(-8, 0)),
                    "avg_tone": float(rng.uniform(-5, 5)),
                    "total_mentions": int(base[i] * 10),
                    "total_articles": int(base[i] * 3),
                }
            )
    return pd.DataFrame(rows)


def test_build_feature_matrix(silver_features, tmp_path) -> None:
    from src.ml.features import build_feature_matrix

    silver_path = tmp_path / "events_per_country_day.parquet"
    silver_features.to_parquet(silver_path, index=False)
    matrix = build_feature_matrix(silver_path=silver_path, horizons=(1, 7))
    assert len(matrix.features) == len(silver_features)
    assert len(matrix.feature_columns) > 0
    assert "target_h1" in matrix.features.columns
    assert "target_h7" in matrix.features.columns


def test_feature_matrix_calendar_features(silver_features, tmp_path) -> None:
    from src.ml.features import build_feature_matrix

    silver_path = tmp_path / "events.parquet"
    silver_features.to_parquet(silver_path, index=False)
    matrix = build_feature_matrix(silver_path=silver_path, horizons=(3,))
    assert "day_of_week" in matrix.features.columns
    assert "month" in matrix.features.columns
    assert "is_weekend" in matrix.features.columns


def test_feature_matrix_lag_features(silver_features, tmp_path) -> None:
    from src.ml.features import build_feature_matrix

    silver_path = tmp_path / "events.parquet"
    silver_features.to_parquet(silver_path, index=False)
    matrix = build_feature_matrix(silver_path=silver_path, horizons=(1,))
    lag_cols = [c for c in matrix.feature_columns if "_lag" in c]
    assert len(lag_cols) > 0
    assert any("lag1d" in c for c in lag_cols)
    assert any("lag3d" in c for c in lag_cols)
    assert any("lag7d" in c for c in lag_cols)


def test_feature_matrix_rolling_features(silver_features, tmp_path) -> None:
    from src.ml.features import build_feature_matrix

    silver_path = tmp_path / "events.parquet"
    silver_features.to_parquet(silver_path, index=False)
    matrix = build_feature_matrix(silver_path=silver_path, horizons=(1,))
    roll_cols = [c for c in matrix.feature_columns if "_roll" in c]
    assert len(roll_cols) > 0
    assert any("roll3d_mean" in c for c in roll_cols)
    assert any("roll14d_mean" in c for c in roll_cols)


def test_to_xy_returns_features_and_target(silver_features, tmp_path) -> None:
    from src.ml.features import build_feature_matrix

    silver_path = tmp_path / "events.parquet"
    silver_features.to_parquet(silver_path, index=False)
    matrix = build_feature_matrix(silver_path=silver_path, horizons=(1, 3, 7))
    X, y = matrix.to_xy(7)
    assert len(X) == len(y)
    assert all(c in matrix.feature_columns for c in X.columns)


def test_get_target_returns_series(silver_features, tmp_path) -> None:
    from src.ml.features import build_feature_matrix

    silver_path = tmp_path / "events.parquet"
    silver_features.to_parquet(silver_path, index=False)
    matrix = build_feature_matrix(silver_path=silver_path, horizons=(1, 7))
    t = matrix.get_target(7)
    assert t is not None
    assert len(t) == len(matrix.features)


def test_feature_matrix_handles_missing_optional_cols(tmp_path) -> None:
    from src.ml.features import build_feature_matrix

    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "actor1_country_code": ["USA"] * 30 + ["RUS"] * 30,
            "event_count": np.random.default_rng(0).integers(1, 20, 60).tolist(),
            "conflict_count": np.random.default_rng(1).integers(0, 5, 60).tolist(),
        }
    )
    p = tmp_path / "events.parquet"
    df.to_parquet(p, index=False)
    matrix = build_feature_matrix(silver_path=p, horizons=(1,))
    assert len(matrix.features) == 60
    assert (
        "avg_goldstein" not in matrix.features.columns
        or matrix.features["avg_goldstein"].sum() == 0
    )


def test_featurize_single_observation() -> None:
    from src.ml.features import featurize_single_observation

    out = featurize_single_observation({"a": 1, "b": None, "c": "3.5"})
    assert out["a"] == 1.0
    assert out["b"] == 0.0
    assert out["c"] == 3.5
