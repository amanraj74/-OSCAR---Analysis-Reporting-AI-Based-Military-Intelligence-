"""Sentiment analysis for OSCAR.

Uses HuggingFace `distilbert-base-uncased-finetuned-sst-2-english` when
available, with VADER (vaderSentiment) as a fast fallback. Either path
produces a `SentimentResult` with positive/neutral/negative probabilities,
a label, and a continuous score in [-1.0, +1.0].

Public API
----------
    SentimentScorer().score(text)              -> SentimentResult
    SentimentScorer().score_batch(texts)       -> list[SentimentResult]
    SentimentScorer().score_and_persist(...)   -> int (rows written)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.config import get_settings
from src.nlp import HAS_TRANSFORMERS, HAS_VADER, logger
from src.persistence.database import session_scope
from src.persistence.models import Sentiment

_DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
_MAX_CHARS = 1500


@dataclass
class SentimentResult:
    """A single sentiment score."""

    label: str
    score: float
    positive: float
    neutral: float
    negative: float
    model: str

    @property
    def is_negative(self) -> bool:
        return self.label.lower() == "negative" or self.score < -0.2

    @property
    def is_positive(self) -> bool:
        return self.label.lower() == "positive" or self.score > 0.2

    def to_db_row(self, source_type: str, source_id: int) -> dict[str, Any]:
        return {
            "source_type": source_type,
            "source_id": source_id,
            "positive": self.positive,
            "neutral": self.neutral,
            "negative": self.negative,
            "label": self.label,
            "score": self.score,
            "model": self.model,
        }


class _VaderBackend:
    """VADER lexicon-based fallback. No model download required."""

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> SentimentResult:
        s = self._analyzer.polarity_scores(text or "")
        compound = float(s.get("compound", 0.0))
        pos = float(s.get("pos", 0.0))
        neu = float(s.get("neu", 1.0))
        neg = float(s.get("neg", 0.0))
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return SentimentResult(
            label=label,
            score=compound,
            positive=pos,
            neutral=neu,
            negative=neg,
            model="vader",
        )


class _TransformerBackend:
    """DistilBERT SST-2 backend. ~250 MB download on first run."""

    def __init__(self, model_name: str, device: int) -> None:
        from transformers import pipeline

        logger.info("sentiment_loading_transformer", model=model_name, device=device)
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            truncation=True,
            max_length=512,
        )

    def score(self, text: str) -> SentimentResult:
        text = (text or "")[:_MAX_CHARS]
        out = self._pipeline(text)[0]
        label = out["label"].lower()
        confidence = float(out["score"])

        if label == "positive":
            pos = confidence
            neg = 1.0 - confidence
            score = confidence
        elif label == "negative":
            neg = confidence
            pos = 1.0 - confidence
            score = -confidence
        else:
            pos = neg = 0.5 * confidence
            score = 0.0
        return SentimentResult(
            label=label,
            score=score,
            positive=pos,
            neutral=max(0.0, 1.0 - pos - neg),
            negative=neg,
            model="distilbert-sst2",
        )


class SentimentScorer:
    """DistilBERT sentiment scorer with VADER fallback."""

    def __init__(
        self,
        model_name: str | None = None,
        device: int | None = None,
        force_mode: str | None = None,
    ) -> None:
        """Initialize scorer.

        Args:
            model_name: HuggingFace model name (default from settings).
            device: -1 for CPU, 0+ for GPU.
            force_mode: Force a specific backend ("transformer" / "vader" / "noop").
                        If None, auto-selects based on availability.
        """
        settings = get_settings()
        self._model_name = model_name or settings.nlp.sentiment_model
        self._device = device if device is not None else (-1 if settings.nlp.device == "cpu" else 0)
        self._backend: Any = None
        if force_mode is not None:
            self._mode = force_mode
        else:
            self._mode = "transformer" if HAS_TRANSFORMERS else "vader" if HAS_VADER else "noop"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def model_name(self) -> str:
        return self._model_name if self._backend is not None else "noop"

    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return
        if self._mode == "transformer" and HAS_TRANSFORMERS:
            try:
                self._backend = _TransformerBackend(self._model_name, self._device)
            except Exception as e:
                logger.warning("sentiment_transformer_failed", error=str(e))
                self._mode = "vader"
                if HAS_VADER:
                    self._backend = _VaderBackend()
                else:
                    self._mode = "noop"
        elif self._mode in {"vader", "transformer"} and HAS_VADER:
            self._backend = _VaderBackend()
            self._mode = "vader"
        else:
            self._mode = "noop"
            logger.warning("sentiment_no_backend")

    def score(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(
                label="neutral",
                score=0.0,
                positive=0.0,
                neutral=1.0,
                negative=0.0,
                model="noop",
            )
        self._ensure_loaded()
        if self._backend is None:
            return SentimentResult(
                label="neutral",
                score=0.0,
                positive=0.0,
                neutral=1.0,
                negative=0.0,
                model="noop",
            )
        return self._backend.score(text)

    def score_batch(self, texts: Iterable[str]) -> list[SentimentResult]:
        return [self.score(t) for t in texts]

    def score_and_persist(
        self,
        source_type: str,
        source_id: int,
        text: str,
    ) -> SentimentResult:
        result = self.score(text)
        with session_scope() as session:
            row = result.to_db_row(source_type=source_type, source_id=source_id)
            stmt = sqlite_insert(Sentiment).values(row)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Sentiment.source_type, Sentiment.source_id, Sentiment.model],
                set_={
                    "positive": stmt.excluded.positive,
                    "neutral": stmt.excluded.neutral,
                    "negative": stmt.excluded.negative,
                    "label": stmt.excluded.label,
                    "score": stmt.excluded.score,
                    "scored_at": stmt.excluded.scored_at,
                },
            )
            session.execute(stmt)
        return result


def aggregate_sentiment(results: Iterable[SentimentResult]) -> SentimentResult:
    """Aggregate multiple sentiment results into one (mean)."""
    results = list(results)
    if not results:
        return SentimentResult("neutral", 0.0, 0.0, 1.0, 0.0, "agg")
    n = len(results)
    return SentimentResult(
        label="neutral",
        score=sum(r.score for r in results) / n,
        positive=sum(r.positive for r in results) / n,
        neutral=sum(r.neutral for r in results) / n,
        negative=sum(r.negative for r in results) / n,
        model="agg",
    )


__all__ = ["SentimentScorer", "SentimentResult", "aggregate_sentiment"]
