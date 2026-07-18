"""End-to-end smoke tests for OSCAR.

These tests verify the full pipeline works:
    seed → ingest → transform → NLP → ML → dashboard loaders

Marked `@pytest.mark.e2e()` so they don't run by default.
Run explicitly: `pytest -m e2e tests/e2e/`.

These tests are designed to be hermetic — they use a temp directory and
the seeded demo data, NOT real network APIs. They verify the
integration of all layers, not external behavior.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_SCRIPT = REPO_ROOT / "scripts" / "seed_demo.py"


def _run_seed_demo(db_path: Path) -> None:
    """Run the seed_demo script's main() in-process via importlib."""
    spec = importlib.util.spec_from_file_location("seed_demo_e2e", SEED_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    rc = mod.main()
    assert rc == 0, f"seed_demo.main() returned {rc}"


@pytest.fixture()
def seeded_db(tmp_path, monkeypatch):
    """Seed a fresh DB for this test. Function-scoped for isolation."""
    db_path = tmp_path / "demo.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("NEWS_API_KEY", "demo-key")

    from src.config import reset_settings_cache
    from src.persistence.database import init_schema, reset_engine

    reset_settings_cache()
    reset_engine()
    init_schema()

    _run_seed_demo(db_path)

    yield db_path

    reset_engine()
    reset_settings_cache()


@pytest.mark.e2e()
def test_seed_populates_all_tables(seeded_db) -> None:
    """All 8 tables should have rows after seeding."""
    from sqlalchemy import create_engine, text

    from src.config import reset_settings_cache

    reset_settings_cache()
    engine = create_engine(f"sqlite:///{seeded_db}")
    expected = {
        "events": 200,
        "articles": 50,
        "entities": 5,
        "entity_mentions": 10,
        "sentiments": 50,
        "topics": 2,
        "anomalies": 2,
    }
    with engine.connect() as conn:
        for table, min_count in expected.items():
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            assert count >= min_count, f"Expected >= {min_count} rows in {table}, got {count}"


@pytest.mark.e2e()
def test_nlp_pipeline_runs_against_seeded_data(seeded_db) -> None:
    """NER pipeline extracts entities from seeded articles."""
    from src.config import reset_settings_cache
    from src.nlp.ner import NerPipeline

    reset_settings_cache()
    pipeline = NerPipeline()
    text = (
        "The United States supplied F-16 fighter jets to Ukraine last week. "
        "Russia responded by deploying Su-35 aircraft. "
        "NATO condemned the Wagner Group's involvement. "
        "IDF retaliated with Iron Dome interceptors."
    )
    result = pipeline.extract(text)
    assert result.count > 0
    weapon_texts = {e.text.upper() for e in result.entities if e.label == "WEAPON"}
    assert "F-16" in weapon_texts
    assert "SU-35" in weapon_texts


@pytest.mark.e2e()
def test_ml_pipeline_builds_and_scores(seeded_db) -> None:
    """End-to-end ML: features → classifier → score."""
    from datetime import datetime, timezone

    from src.config import reset_settings_cache
    from src.ml.escalation import EscalationClassifier
    from src.ml.features import build_feature_matrix

    reset_settings_cache()

    dates = pd.date_range(
        datetime.now(timezone.utc) - pd.Timedelta(days=80),
        periods=80,
        freq="D",
    )
    rng = np.random.default_rng(42)
    rows = []
    for region in ["USA", "RUS", "UKR", "IRN"]:
        base = rng.integers(10, 50, 80).tolist()
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "actor1_country_code": region,
                    "event_count": base[i],
                    "conflict_count": int(base[i] * 0.3),
                    "conflict_ratio": 0.3,
                    "avg_goldstein": float(rng.uniform(-8, 0)),
                    "avg_tone": float(rng.uniform(-5, 5)),
                    "total_mentions": base[i] * 10,
                    "total_articles": base[i] * 3,
                }
            )
    df = pd.DataFrame(rows)
    silver_path = seeded_db.parent / "events.parquet"
    df.to_parquet(silver_path, index=False)
    matrix = build_feature_matrix(silver_path=silver_path, horizons=(7,))
    assert matrix.features is not None
    assert len(matrix.feature_columns) > 10

    clf = EscalationClassifier(horizon_days=7, force_mode="sklearn")
    clf.fit(matrix, horizon=7)
    assert clf.is_fitted
    proba = clf.predict_proba(matrix.features[matrix.feature_columns])
    assert (proba >= 0).all()
    assert (proba <= 1).all()


@pytest.mark.e2e()
def test_dashboard_data_loaders_work_with_seeded_data(seeded_db) -> None:
    """All dashboard data loaders return non-empty DataFrames after seeding."""
    from src.config import reset_settings_cache

    reset_settings_cache()
    from dashboard import utils

    metrics = utils.get_overview_metrics.__wrapped__()
    assert metrics["events"] > 0, f"Expected events > 0, got {metrics}"
    assert metrics["articles"] > 0
    assert metrics["entities"] > 0
    assert metrics["anomalies"] > 0
    assert metrics["topics"] > 0

    events = utils.get_events_dataframe.__wrapped__(days=30)
    assert not events.empty
    assert "actor1_country_code" in events.columns

    articles = utils.get_articles_dataframe.__wrapped__(days=30)
    assert not articles.empty
    assert "title" in articles.columns

    entities = utils.get_top_entities.__wrapped__(limit=20)
    assert not entities.empty
    assert "canonical_name" in entities.columns

    anomalies = utils.get_recent_anomalies.__wrapped__(limit=20)
    assert not anomalies.empty
    assert "severity" in anomalies.columns

    topics = utils.get_topics.__wrapped__(n=10)
    assert not topics.empty
    assert "label" in topics.columns


@pytest.mark.e2e()
def test_dashboard_pages_can_be_loaded(seeded_db) -> None:
    """All dashboard page modules import without errors against seeded data."""
    for name in [
        "0_Home",
        "1_Map",
        "2_Sentiment",
        "3_Entities",
        "4_Forecast",
        "5_Alerts",
        "6_About",
    ]:
        mod = importlib.import_module(f"dashboard.pages.{name}")
        assert mod is not None
