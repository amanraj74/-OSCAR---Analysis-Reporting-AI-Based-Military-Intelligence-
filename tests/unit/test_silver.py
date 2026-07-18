"""Tests for the silver transform layer."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.config import reset_settings_cache
from src.persistence.database import init_schema, session_scope
from src.persistence.models import Article, Event
from src.transform.silver import build_all_silver, build_articles_silver, build_events_silver


@pytest.fixture()
def seeded_db(fresh_db):
    """Seed the DB with a small known set of events + articles."""
    events = [
        Event(
            global_event_id=i,
            sql_date="20250705",
            year=2025,
            actor1_name="UNITED STATES",
            actor1_country_code="USA",
            actor2_name="RUSSIA",
            actor2_country_code="RUS",
            event_code="190",
            event_root_code="19",
            goldstein_scale=-7.0,
            num_mentions=10,
            num_articles=5,
            avg_tone=-2.5,
            action_geo_fullname="Kyiv, Ukraine",
            action_geo_country_code="UKR",
            action_geo_lat=50.45,
            action_geo_long=30.52,
            source_url=None,
        )
        for i in range(1, 4)
    ]
    articles = [
        Article(
            external_id=f"a{i}",
            source="newsapi",
            title=f"Headline {i}",
            description="desc",
            content=None,
            url=f"https://example.com/a{i}",
            author=None,
            image_url=None,
            language="en",
            published_at=datetime(2025, 7, 5, 12, 0, 0, tzinfo=timezone.utc),
        )
        for i in range(1, 3)
    ]
    with session_scope() as s:
        s.add_all(events)
        s.add_all(articles)

    return fresh_db


def test_build_events_silver_creates_parquet(seeded_db) -> None:  # noqa: ARG001
    reset_settings_cache()
    init_schema()
    with session_scope() as s:
        path = build_events_silver(s)
    assert path.exists()
    df = pd.read_parquet(path)
    assert "event_count" in df.columns
    assert "conflict_count" in df.columns
    assert len(df) >= 1
    row = df.iloc[0]
    assert row["event_count"] >= 3
    assert row["conflict_count"] >= 3
    assert row["actor1_country_code"] == "UKR"
    assert 0.0 <= row["conflict_ratio"] <= 1.0


def test_build_articles_silver_creates_parquet(seeded_db) -> None:  # noqa: ARG001
    reset_settings_cache()
    init_schema()
    with session_scope() as s:
        path = build_articles_silver(s)
    assert path.exists()
    df = pd.read_parquet(path)
    assert "article_count" in df.columns
    assert len(df) >= 1
    assert df.iloc[0]["article_count"] >= 2


def test_build_all_silver_returns_paths(seeded_db) -> None:  # noqa: ARG001
    reset_settings_cache()
    out = build_all_silver()
    assert "events_per_country_day" in out
    assert "articles_per_source_day" in out
    assert out["events_per_country_day"].exists()
    assert out["articles_per_source_day"].exists()
