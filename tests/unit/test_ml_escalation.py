"""Tests for the escalation classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_matrix(tmp_path):
    from src.ml.features import build_feature_matrix

    dates = pd.date_range("2025-01-01", periods=80, freq="D")
    regions = ["USA", "RUS", "UKR", "IRN", "ISR"]
    rng = np.random.default_rng(123)
    rows = []
    for region in regions:
        base = rng.integers(10, 60, size=len(dates)).tolist()
        conflict = [int(b * 0.3 + rng.integers(0, 5)) for b in base]
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "actor1_country_code": region,
                    "event_count": base[i],
                    "conflict_count": conflict[i],
                    "conflict_ratio": conflict[i] / max(base[i], 1),
                    "avg_goldstein": float(rng.uniform(-8, 0)),
                    "avg_tone": float(rng.uniform(-5, 5)),
                    "total_mentions": base[i] * 10,
                    "total_articles": base[i] * 3,
                }
            )
    df = pd.DataFrame(rows)
    p = tmp_path / "events.parquet"
    df.to_parquet(p, index=False)
    return build_feature_matrix(silver_path=p, horizons=(1, 3, 7))


def test_escalation_initializes_with_force_mode() -> None:
    from src.ml.escalation import EscalationClassifier

    clf = EscalationClassifier(force_mode="sklearn")
    assert clf.mode == "sklearn"


def test_escalation_fit_predict(synthetic_matrix) -> None:
    from src.ml.escalation import EscalationClassifier

    clf = EscalationClassifier(horizon_days=7, force_mode="sklearn")
    clf.fit(synthetic_matrix, horizon=7)
    assert clf.is_fitted

    X = synthetic_matrix.features[synthetic_matrix.feature_columns]
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X),)
    assert (proba >= 0).all()
    assert (proba <= 1).all()


def test_escalation_score(synthetic_matrix) -> None:
    from src.ml.escalation import EscalationClassifier

    clf = EscalationClassifier(horizon_days=7, force_mode="sklearn")
    clf.fit(synthetic_matrix, horizon=7)
    X = synthetic_matrix.features[synthetic_matrix.feature_columns]
    y = synthetic_matrix.features["target_h7"].astype(int)
    metrics = clf.score(X, y)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert metrics.n_samples == len(X)
    assert "confusion_matrix" in metrics.to_dict()


def test_escalation_predict_for_region(synthetic_matrix) -> None:
    from src.ml.escalation import EscalationClassifier

    clf = EscalationClassifier(horizon_days=7, force_mode="sklearn")
    clf.fit(synthetic_matrix, horizon=7)
    out = clf.predict_for_region(synthetic_matrix, "USA")
    assert "escalation_probability" in out.columns
    assert (out["actor1_country_code"] == "USA").all()


def test_escalation_feature_importance(synthetic_matrix) -> None:
    from src.ml.escalation import EscalationClassifier

    clf = EscalationClassifier(horizon_days=7, force_mode="sklearn")
    clf.fit(synthetic_matrix, horizon=7)
    fi = clf.feature_importance()
    assert len(fi) > 0
    assert "feature" in fi.columns
    assert "importance" in fi.columns


def test_escalation_serialization_roundtrip(synthetic_matrix, tmp_path) -> None:
    from src.ml.escalation import EscalationClassifier

    clf = EscalationClassifier(horizon_days=7, force_mode="sklearn")
    clf.fit(synthetic_matrix, horizon=7)
    blob = clf.to_bytes()
    restored = EscalationClassifier.from_bytes(blob)
    assert restored.is_fitted
    assert restored.horizon_days == clf.horizon_days
    assert restored._feature_columns == clf._feature_columns


def test_escalation_cross_validate(synthetic_matrix) -> None:
    from src.ml.escalation import cross_validate

    cv = cross_validate(synthetic_matrix, horizon=7, n_splits=3, force_mode="sklearn")
    assert "f1_mean" in cv
    assert "pr_auc_mean" in cv


def test_escalation_unfitted_raises(synthetic_matrix) -> None:
    from src.ml.escalation import EscalationClassifier

    clf = EscalationClassifier(force_mode="sklearn")
    X = synthetic_matrix.features[synthetic_matrix.feature_columns]
    with pytest.raises(RuntimeError):
        clf.predict_proba(X)


def test_metrics_to_dict() -> None:
    from src.ml.escalation import ClassifierMetrics

    m = ClassifierMetrics(
        accuracy=0.8, precision=0.7, recall=0.6, f1=0.65, pr_auc=0.75, n_samples=100
    )
    d = m.to_dict()
    assert d["accuracy"] == 0.8
    assert d["n_samples"] == 100
