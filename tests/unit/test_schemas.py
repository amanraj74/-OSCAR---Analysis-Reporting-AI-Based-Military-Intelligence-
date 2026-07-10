"""Tests for domain schemas."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.domain.schemas import GdeltEventSchema, IngestionSummary, RedditPostSchema


def test_gdelt_event_schema_valid() -> None:
    s = GdeltEventSchema(
        global_event_id=1,
        sql_date="20250705",
        year=2025,
        actor1_name="USA",
        actor2_name="RUS",
        event_root_code="19",
        goldstein_scale=-5.0,
        avg_tone=-2.0,
    )
    assert s.global_event_id == 1
    assert s.year == 2025


def test_gdelt_event_schema_bad_date_rejected() -> None:
    with pytest.raises(ValidationError):
        GdeltEventSchema(
            global_event_id=1,
            sql_date="2025-07-05",
        )


def test_gdelt_event_schema_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        GdeltEventSchema(
            global_event_id=1,
            sql_date="20250705",
            goldstein_scale=-100.0,
        )


def test_reddit_post_schema_valid() -> None:
    s = RedditPostSchema(
        external_id="abc",
        subreddit="x",
        title="t",
        url="https://reddit.com/r/x/comments/abc/t/",
        published_at=datetime(2025, 7, 5, tzinfo=timezone.utc),
    )
    assert s.source == "reddit"


def test_ingestion_summary_to_dict() -> None:
    summary = IngestionSummary(
        source="gdelt",
        fetched=10,
        parsed=8,
        persisted=8,
        success=True,
    )
    d = summary.to_dict()
    assert d["source"] == "gdelt"
    assert d["fetched"] == 10
    assert "started_at" in d
    assert "finished_at" in d
