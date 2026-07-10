"""Tests for NewsAPI ingestor and Pydantic schemas."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from src.config import reset_settings_cache
from src.domain.schemas import NewsArticleSchema
from src.ingestion.newsapi import NewsApiIngestor, NewsArticle, _parse_newsapi_response, _stable_id
from src.persistence.database import session_scope
from src.persistence.models import Article


def test_news_article_schema_valid() -> None:
    raw = {
        "source": {"id": "bbc", "name": "BBC"},
        "author": "Test Author",
        "title": "Some Title",
        "description": "Some description",
        "url": "https://example.com/article/1",
        "urlToImage": "https://example.com/img.jpg",
        "publishedAt": "2025-07-05T12:34:56Z",
        "content": "Some content...",
    }
    s = NewsArticleSchema.model_validate({**raw, "source": raw["source"]["name"]})
    assert s.title == "Some Title"
    assert s.published_at.tzinfo is not None
    assert s.source == "BBC"


def test_news_article_schema_rejects_bad_url() -> None:
    raw = {
        "source": "BBC",
        "title": "Title",
        "url": "not-a-url",
        "publishedAt": "2025-07-05T12:34:56Z",
    }
    with pytest.raises(ValidationError):
        NewsArticleSchema.model_validate(raw)


def test_news_article_schema_rejects_bad_date() -> None:
    raw = {
        "source": "BBC",
        "title": "Title",
        "url": "https://example.com/a",
        "publishedAt": "not-a-date",
    }
    with pytest.raises(ValidationError):
        NewsArticleSchema.model_validate(raw)


def test_parse_newsapi_response_skips_invalid() -> None:
    payload = {
        "status": "ok",
        "articles": [
            {
                "source": {"name": "BBC"},
                "title": "Good Article",
                "url": "https://example.com/good",
                "publishedAt": "2025-07-05T12:00:00Z",
            },
            {
                "source": {"name": "X"},
                "title": "Bad URL",
                "url": "not-a-url",
                "publishedAt": "2025-07-05T12:00:00Z",
            },
            {
                "source": {"name": "Y"},
                "title": "Bad Date",
                "url": "https://example.com/y",
                "publishedAt": "garbage",
            },
        ],
    }
    out = _parse_newsapi_response(payload)
    assert len(out) == 1
    assert out[0].title == "Good Article"


def test_parse_newsapi_response_dedupes() -> None:
    payload = {
        "status": "ok",
        "articles": [
            {
                "source": "BBC",
                "title": "Same",
                "url": "https://example.com/same",
                "publishedAt": "2025-07-05T12:00:00Z",
            },
            {
                "source": "BBC",
                "title": "Same",
                "url": "https://example.com/same",
                "publishedAt": "2025-07-05T12:00:00Z",
            },
        ],
    }
    out = _parse_newsapi_response(payload)
    assert len(out) == 1


def test_newsapi_ingestor_no_key_returns_empty(tmp_db_path, monkeypatch) -> None:
    monkeypatch.setenv("NEWS_API_KEY", "")
    reset_settings_cache()
    ingestor = NewsApiIngestor(query="Ukraine")
    result = ingestor.run()
    assert result.success is True
    assert result.count == 0


def test_newsapi_ingestor_persist_idempotent(fresh_db) -> None:
    art = NewsArticle(
        external_id="test-ext-1",
        source="BBC",
        title="Test",
        description=None,
        url="https://example.com/1",
        image_url=None,
        published_at=datetime(2025, 7, 5, 12, 0, 0, tzinfo=timezone.utc),
        author="X",
        content=None,
        language="en",
    )
    ingestor = NewsApiIngestor()
    n1 = ingestor.persist([art])
    assert n1 == 1
    with session_scope() as s:
        all_a = s.execute(select(Article)).scalars().all()
        assert len(all_a) == 1
    n2 = ingestor.persist([art])
    assert n2 == 1
    with session_scope() as s:
        all_a = s.execute(select(Article)).scalars().all()
        assert len(all_a) == 1


def test_stable_id_is_deterministic() -> None:
    a = _stable_id("https://x.com/1", "title1")
    b = _stable_id("https://x.com/1", "title1")
    c = _stable_id("https://x.com/2", "title1")
    assert a == b
    assert a != c
    assert len(a) == 32
