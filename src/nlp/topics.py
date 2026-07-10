"""Topic discovery for OSCAR.

Uses BERTopic when available (best quality); falls back to TF-IDF +
KMeans + top-K keywords (always-available, no extra downloads).

Public API
----------
    TopicDiscoverer().fit(documents)            -> list[TopicInfo]
    TopicDiscoverer().fit_and_persist(...)      -> int
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.nlp import HAS_BERTOPIC, logger
from src.persistence.database import session_scope
from src.persistence.models import Topic

_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "to",
    "for",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "should",
    "could",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "as",
    "by",
    "from",
    "with",
    "at",
    "they",
    "their",
    "we",
    "our",
    "you",
    "your",
    "i",
    "he",
    "she",
    "his",
    "her",
    "them",
    "us",
    "about",
    "after",
    "before",
    "over",
    "under",
    "than",
    "then",
    "so",
    "if",
    "no",
    "not",
    "also",
}


@dataclass
class TopicInfo:
    topic_id: int
    label: str
    keywords: list[str] = field(default_factory=list)
    article_count: int = 0
    representation: dict[str, Any] = field(default_factory=dict)


def _tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t and t not in _STOPWORDS and len(t) > 3]


class _SklearnBackend:
    """TF-IDF + KMeans topic discovery (lightweight, no GPU, no model downloads)."""

    def __init__(self, n_topics: int = 8) -> None:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            max_df=0.7,
            min_df=2,
            max_features=2000,
            ngram_range=(1, 2),
        )
        self._n_topics = n_topics
        self._kmeans = KMeans(n_clusters=n_topics, n_init=10, random_state=42)

    def fit(self, documents: list[str]) -> list[TopicInfo]:
        if not documents:
            return []
        try:
            tfidf = self._vectorizer.fit_transform(documents)
        except ValueError:
            return []
        if tfidf.shape[1] == 0:
            return []

        n_clusters = min(self._n_topics, max(2, tfidf.shape[0] // 3))
        self._kmeans.n_clusters = n_clusters
        labels = self._kmeans.fit_predict(tfidf)

        feature_names = self._vectorizer.get_feature_names_out()
        centers = self._kmeans.cluster_centers_
        counts = Counter(labels.tolist())

        topics: list[TopicInfo] = []
        for cid in range(n_clusters):
            top_idx = centers[cid].argsort()[::-1][:10]
            keywords = [str(feature_names[i]) for i in top_idx if i < len(feature_names)]
            label = ", ".join(keywords[:3]) if keywords else f"topic_{cid}"
            topics.append(
                TopicInfo(
                    topic_id=cid,
                    label=label,
                    keywords=keywords,
                    article_count=counts.get(cid, 0),
                    representation={"backend": "sklearn", "n_docs": len(documents)},
                )
            )
        return topics


class _BertopicBackend:
    """BERTopic backend. Higher quality; requires `bertopic` package + embeddings."""

    def __init__(self, embedding_model: str, n_topics: int | None) -> None:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        logger.info("topics_loading_bertopic", embedding_model=embedding_model)
        emb_model = SentenceTransformer(embedding_model, device="cpu")
        self._model = BERTopic(embedding_model=emb_model, nr_topics=n_topics, verbose=False)

    def fit(self, documents: list[str]) -> list[TopicInfo]:
        if not documents:
            return []
        topics, probs = self._model.fit_transform(documents)
        info = self._model.get_topic_info()
        out: list[TopicInfo] = []
        counts = Counter(topics)
        for row in info.itertuples(index=False):
            tid = int(row.Topic)
            if tid == -1:
                continue
            keywords = [w for w, _ in self._model.get_topic(tid)[:10]]
            out.append(
                TopicInfo(
                    topic_id=tid,
                    label=", ".join(keywords[:3]) if keywords else f"topic_{tid}",
                    keywords=keywords,
                    article_count=counts.get(tid, 0),
                    representation={"backend": "bertopic", "n_docs": len(documents)},
                )
            )
        return out


class TopicDiscoverer:
    """BERTopic primary + sklearn fallback topic discovery."""

    def __init__(
        self,
        n_topics: int | None = 8,
        embedding_model: str = "all-MiniLM-L6-v2",
        force_mode: str | None = None,
    ) -> None:
        self._n_topics = n_topics
        self._embedding_model = embedding_model
        self._backend: Any = None
        if force_mode is not None:
            self._mode = force_mode
        else:
            self._mode = "bertopic" if HAS_BERTOPIC else "sklearn"

    @property
    def mode(self) -> str:
        return self._mode

    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return
        if self._mode == "bertopic" and HAS_BERTOPIC:
            try:
                self._backend = _BertopicBackend(self._embedding_model, self._n_topics)
            except Exception as e:
                logger.warning("bertopic_fallback_to_sklearn", error=str(e))
                self._mode = "sklearn"
                self._backend = _SklearnBackend(self._n_topics or 8)
        else:
            self._backend = _SklearnBackend(self._n_topics or 8)
            self._mode = "sklearn"

    def fit(self, documents: list[str]) -> list[TopicInfo]:
        if not documents:
            return []
        self._ensure_loaded()
        return self._backend.fit(documents)

    def fit_and_persist(
        self,
        documents: list[str],
        min_article_count: int = 1,
    ) -> int:
        topics = self.fit(documents)
        if not topics:
            return 0

        with session_scope() as session:
            for t in topics:
                if t.article_count < min_article_count:
                    continue
                row = {
                    "topic_id": t.topic_id,
                    "label": t.label,
                    "keywords": t.keywords,
                    "article_count": t.article_count,
                    "representation": t.representation,
                }
                stmt = sqlite_insert(Topic).values(row)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[Topic.topic_id, Topic.label],
                    set_={
                        "keywords": stmt.excluded.keywords,
                        "article_count": stmt.excluded.article_count,
                        "representation": stmt.excluded.representation,
                    },
                )
                session.execute(stmt)
        return len([t for t in topics if t.article_count >= min_article_count])

    def get_top_topics(self, n: int = 10) -> list[TopicInfo]:
        with session_scope() as session:
            rows = (
                session.execute(select(Topic).order_by(Topic.article_count.desc()).limit(n))
                .scalars()
                .all()
            )
        return [
            TopicInfo(
                topic_id=r.topic_id,
                label=r.label,
                keywords=list(r.keywords or []),
                article_count=r.article_count,
                representation=dict(r.representation or {}),
            )
            for r in rows
        ]


__all__ = [
    "TopicDiscoverer",
    "TopicInfo",
    "_SklearnBackend",
    "_BertopicBackend",
]
