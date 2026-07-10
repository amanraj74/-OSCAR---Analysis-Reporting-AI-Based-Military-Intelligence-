"""Tests for the sentiment scorer."""

from __future__ import annotations

import pytest

from src.nlp.sentiment import SentimentResult, SentimentScorer, aggregate_sentiment

POSITIVE_TEXT = "Peace deal signed; ceasefire holds; aid reaches civilians."
NEGATIVE_TEXT = "Brutal attacks kill dozens; airstrikes destroy hospitals; atrocities reported."
NEUTRAL_TEXT = "The meeting is scheduled for next Tuesday in Brussels."


def _vader_scorer() -> SentimentScorer:
    """Construct a scorer forced to VADER backend (deterministic, offline)."""
    pytest.importorskip("vaderSentiment")
    return SentimentScorer(force_mode="vader")


def test_scorer_initializes() -> None:
    scorer = SentimentScorer()
    assert scorer.mode in {"transformer", "vader", "noop"}


def test_force_mode_vader() -> None:
    pytest.importorskip("vaderSentiment")
    scorer = SentimentScorer(force_mode="vader")
    assert scorer.mode == "vader"


def test_score_returns_result() -> None:
    scorer = _vader_scorer()
    out = scorer.score(POSITIVE_TEXT)
    assert isinstance(out, SentimentResult)
    assert -1.0 <= out.score <= 1.0
    assert 0.0 <= out.positive <= 1.0
    assert 0.0 <= out.negative <= 1.0
    assert 0.0 <= out.neutral <= 1.0


def test_score_positive_text_vader() -> None:
    pytest.importorskip("vaderSentiment")
    from src.nlp.sentiment import _VaderBackend

    backend = _VaderBackend()
    out = backend.score(POSITIVE_TEXT)
    assert out.label == "positive"
    assert out.score > 0


def test_score_negative_text_vader() -> None:
    pytest.importorskip("vaderSentiment")
    from src.nlp.sentiment import _VaderBackend

    backend = _VaderBackend()
    out = backend.score(NEGATIVE_TEXT)
    assert out.label == "negative"
    assert out.score < -0.2


def test_score_neutral_text_vader() -> None:
    pytest.importorskip("vaderSentiment")
    from src.nlp.sentiment import _VaderBackend

    backend = _VaderBackend()
    out = backend.score(NEUTRAL_TEXT)
    assert out.label in {"neutral", "positive", "negative"}
    assert abs(out.score) < 0.6


def test_score_empty_text_returns_neutral() -> None:
    scorer = _vader_scorer()
    out = scorer.score("")
    assert out.label == "neutral"
    assert out.score == 0.0


def test_score_batch() -> None:
    scorer = _vader_scorer()
    out = scorer.score_batch([POSITIVE_TEXT, NEGATIVE_TEXT, NEUTRAL_TEXT])
    assert len(out) == 3
    assert all(isinstance(r, SentimentResult) for r in out)


def test_aggregate_sentiment() -> None:
    results = [
        SentimentResult("positive", 0.8, 0.9, 0.1, 0.0, "x"),
        SentimentResult("negative", -0.6, 0.0, 0.2, 0.8, "x"),
    ]
    agg = aggregate_sentiment(results)
    assert agg.score == pytest.approx(0.1, abs=0.01)


def test_score_and_persist(fresh_db) -> None:
    from sqlalchemy import select

    from src.persistence.models import Sentiment

    scorer = _vader_scorer()
    result = scorer.score_and_persist("article", 1, NEGATIVE_TEXT)
    assert result is not None
    with __import__("src.persistence.database", fromlist=["session_scope"]).session_scope() as s:
        rows = s.execute(select(Sentiment)).scalars().all()
        assert len(rows) >= 1
        assert rows[0].source_id == 1
        assert rows[0].source_type == "article"


def test_score_idempotent(fresh_db) -> None:
    from sqlalchemy import select

    from src.persistence.models import Sentiment

    scorer = _vader_scorer()
    scorer.score_and_persist("article", 1, NEGATIVE_TEXT)
    scorer.score_and_persist("article", 1, NEGATIVE_TEXT)
    with __import__("src.persistence.database", fromlist=["session_scope"]).session_scope() as s:
        rows = s.execute(select(Sentiment)).scalars().all()
        assert len(rows) == 1


def test_result_is_negative_property() -> None:
    r = SentimentResult("negative", -0.7, 0.0, 0.0, 1.0, "x")
    assert r.is_negative is True
    assert r.is_positive is False


def test_result_is_positive_property() -> None:
    r = SentimentResult("positive", 0.6, 1.0, 0.0, 0.0, "x")
    assert r.is_positive is True
    assert r.is_negative is False


@pytest.mark.integration
def test_transformer_backend_integration() -> None:
    """Integration test for DistilBERT. Requires model download; slow.

    Skip unless explicitly requested: `pytest -m integration`.
    """
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    has_cached = cache_dir.exists() and any(cache_dir.glob("models--distilbert*"))
    if not has_cached:
        pytest.skip(
            "DistilBERT model not cached; run `python -c \"from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')\"` to cache, then re-run."
        )

    from src.nlp.sentiment import SentimentScorer

    scorer = SentimentScorer(force_mode="transformer")
    out = scorer.score(POSITIVE_TEXT)
    assert out.model == "distilbert-sst2"
    assert out.label in {"positive", "negative", "neutral"}
