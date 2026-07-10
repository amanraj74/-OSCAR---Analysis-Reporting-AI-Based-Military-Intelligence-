"""Tests for anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_features() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    rows = []
    rng = np.random.default_rng(42)
    for region in ["USA", "RUS"]:
        base = rng.integers(10, 50, size=60).tolist()
        for i, d in enumerate(dates):
            if region == "RUS" and 40 < i < 50:
                base[i] = 200
            rows.append(
                {
                    "date": d,
                    "actor1_country_code": region,
                    "event_count": base[i],
                    "conflict_count": int(base[i] * 0.3),
                    "avg_tone": float(rng.uniform(-5, 0)),
                }
            )
    return pd.DataFrame(rows)


def test_zscore_detector_detects_spike(sample_features) -> None:
    from src.ml.anomaly import ZScoreDetector

    detector = ZScoreDetector(window=14, threshold=2.5)
    anomalies = detector.detect(
        sample_features,
        feature_cols=["event_count"],
        anomaly_type="zscore",
    )
    assert any(a.region == "RUS" for a in anomalies)


def test_zscore_detector_returns_anomalies(sample_features) -> None:
    from src.ml.anomaly import ZScoreDetector

    detector = ZScoreDetector(window=14, threshold=2.5)
    anomalies = detector.detect(sample_features, feature_cols=["event_count"])
    for a in anomalies:
        assert a.region in {"USA", "RUS"}
        assert a.severity > 0
        assert a.score != 0


def test_zscore_detector_handles_short_series() -> None:
    from src.ml.anomaly import ZScoreDetector

    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5),
            "actor1_country_code": ["A"] * 5,
            "event_count": [1, 2, 3, 4, 5],
        }
    )
    detector = ZScoreDetector(window=3, threshold=2.5)
    anomalies = detector.detect(df, feature_cols=["event_count"])
    assert isinstance(anomalies, list)


def test_iforest_detector_basic(sample_features) -> None:
    from src.ml.anomaly import IsolationForestDetector

    detector = IsolationForestDetector(contamination=0.05)
    anomalies = detector.detect(sample_features, feature_cols=["event_count", "conflict_count"])
    assert isinstance(anomalies, list)


def test_anomaly_detector_routes_to_iforest(sample_features) -> None:
    from src.ml.anomaly import AnomalyDetector

    detector = AnomalyDetector(window=14, threshold=2.5, force_mode="iforest")
    anomalies = detector.detect(sample_features, feature_cols=["event_count", "conflict_count"])
    assert isinstance(anomalies, list)
    assert all(a.anomaly_type.startswith("iForest") for a in anomalies)


def test_anomaly_detector_routes_to_zscore(sample_features) -> None:
    from src.ml.anomaly import AnomalyDetector

    detector = AnomalyDetector(window=14, threshold=2.0, force_mode="zscore")
    anomalies = detector.detect(sample_features, feature_cols=["event_count"])
    assert all(a.anomaly_type.startswith("zscore") for a in anomalies)


def test_anomaly_detector_per_region(sample_features) -> None:
    from src.ml.anomaly import AnomalyDetector

    detector = AnomalyDetector(window=14, threshold=2.0, force_mode="zscore")
    by_region = detector.detect_per_region(sample_features, feature_cols=["event_count"])
    assert isinstance(by_region, dict)
    for region, anoms in by_region.items():
        assert region in {"USA", "RUS"}
        assert all(a.region == region for a in anoms)


def test_anomaly_to_db_row() -> None:
    from src.ml.anomaly import Anomaly

    a = Anomaly(
        region="RUS",
        date="2025-02-15",
        anomaly_type="zscore:event_count",
        severity=0.8,
        score=3.5,
        description="spike",
        context={"value": 200.0},
    )
    row = a.to_db_row()
    assert row["region"] == "RUS"
    assert row["anomaly_type"] == "zscore:event_count"
    assert row["severity"] == 0.8
