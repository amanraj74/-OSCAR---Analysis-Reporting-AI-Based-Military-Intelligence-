"""SQLAlchemy ORM models for OSCAR.

Tables:
    events             — GDELT global events (per-event raw data).
    articles           — News + Reddit articles/posts.
    entities           — Named entities (ORG, GPE, PERSON, WEAPON, ...).
    entity_mentions    — Many-to-many: entities ↔ articles/events.
    sentiments         — Sentiment scores per article/event.
    topics             — Discovered themes via BERTopic.
    risk_scores        — Escalation risk scores per region/date.
    anomalies          — Detected anomalies per region/date.

All tables include `ingested_at` for auditability.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.persistence.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Event(Base):
    """A single GDELT 2.0 event.

    Stored with the global event ID as a unique key for idempotent ingest.
    """

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    global_event_id: Mapped[int] = mapped_column(
        BigInteger, unique=True, nullable=False, index=True
    )
    sql_date: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)

    actor1_name: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    actor1_country_code: Mapped[str | None] = mapped_column(String(3), nullable=True, index=True)
    actor2_name: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    actor2_country_code: Mapped[str | None] = mapped_column(String(3), nullable=True, index=True)

    event_code: Mapped[str | None] = mapped_column(String(4), nullable=True, index=True)
    event_root_code: Mapped[str | None] = mapped_column(String(2), nullable=True, index=True)

    goldstein_scale: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_mentions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    num_articles: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    avg_tone: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)

    action_geo_fullname: Mapped[str | None] = mapped_column(String(255), nullable=True)
    action_geo_country_code: Mapped[str | None] = mapped_column(
        String(3), nullable=True, index=True
    )
    action_geo_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    action_geo_long: Mapped[float | None] = mapped_column(Float, nullable=True)

    source_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_events_date_country", "sql_date", "actor1_country_code"),
        Index("ix_events_date_event", "sql_date", "event_root_code"),
    )

    def __repr__(self) -> str:
        return f"<Event id={self.global_event_id} date={self.sql_date} actor1={self.actor1_name!r}>"


class Article(Base):
    """A news article (NewsAPI) or social post (Reddit)."""

    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(String(1024), nullable=False)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_articles_source_external"),
        Index("ix_articles_source_published", "source", "published_at"),
    )

    def __repr__(self) -> str:
        return f"<Article id={self.id} source={self.source!r} title={self.title[:40]!r}>"


class Entity(Base):
    """A named entity extracted from text."""

    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    mention_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    first_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    last_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("canonical_name", "entity_type", name="uq_entities_canonical_type"),
    )

    mentions: Mapped[list[EntityMention]] = relationship(
        back_populates="entity", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Entity {self.entity_type} {self.canonical_name!r}>"


class EntityMention(Base):
    """A specific occurrence of an entity in an article or event."""

    __tablename__ = "entity_mentions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_type: Mapped[str] = mapped_column(String(16), nullable=False)
    source_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    sentiment_label: Mapped[str | None] = mapped_column(String(16), nullable=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mentioned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    entity: Mapped[Entity] = relationship(back_populates="mentions")

    __table_args__ = (
        Index("ix_mentions_source", "source_type", "source_id"),
        UniqueConstraint(
            "entity_id",
            "source_type",
            "source_id",
            name="uq_mentions_entity_source",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<EntityMention entity_id={self.entity_id} source={self.source_type}#{self.source_id}>"
        )


class Sentiment(Base):
    """A sentiment score for an article or event."""

    __tablename__ = "sentiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    source_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    positive: Mapped[float] = mapped_column(Float, nullable=False)
    neutral: Mapped[float] = mapped_column(Float, nullable=False)
    negative: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str] = mapped_column(String(16), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("source_type", "source_id", "model", name="uq_sentiments_source_model"),
    )

    def __repr__(self) -> str:
        return f"<Sentiment {self.label} {self.score:.2f} on {self.source_type}#{self.source_id}>"


class Topic(Base):
    """A theme discovered via BERTopic."""

    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic_id: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    keywords: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    article_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    representation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (UniqueConstraint("topic_id", "label", name="uq_topics_id_label"),)


class RiskScore(Base):
    """An escalation risk score for a region and date."""

    __tablename__ = "risk_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    region: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    date: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    features: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "region", "date", "horizon_days", "model", name="uq_risk_region_date_model"
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<RiskScore {self.region} {self.date} h={self.horizon_days}d p={self.probability:.2f}>"
        )


class Anomaly(Base):
    """A detected anomaly (Isolation Forest) for a region/date."""

    __tablename__ = "anomalies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    region: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    date: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    anomaly_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<Anomaly {self.region} {self.date} {self.anomaly_type} sev={self.severity:.2f}>"


__all__ = [
    "Event",
    "Article",
    "Entity",
    "EntityMention",
    "Sentiment",
    "Topic",
    "RiskScore",
    "Anomaly",
]
