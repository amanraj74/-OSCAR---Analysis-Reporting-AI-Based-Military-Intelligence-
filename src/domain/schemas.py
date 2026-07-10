"""Domain layer: typed schemas for OSCAR entities.

Pure-Python Pydantic models used at the boundary between
ingestion/parsing and persistence. Persistence layer maps to ORM
via `to_orm_kwargs()` helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GdeltEventSchema(BaseModel):
    """Validated schema for a parsed GDELT 2.0 event."""

    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    global_event_id: int = Field(ge=1)
    sql_date: str = Field(min_length=8, max_length=8, pattern=r"^\d{8}$")
    year: int | None = Field(default=None, ge=1900, le=2100)

    actor1_name: str | None = Field(default=None, max_length=255)
    actor1_country_code: str | None = Field(default=None, max_length=3)
    actor2_name: str | None = Field(default=None, max_length=255)
    actor2_country_code: str | None = Field(default=None, max_length=3)

    event_code: str | None = Field(default=None, max_length=4)
    event_root_code: str | None = Field(default=None, max_length=2)
    event_root_label: str | None = Field(default=None, max_length=64)

    goldstein_scale: float | None = Field(default=None, ge=-10.0, le=10.0)
    num_mentions: int = Field(default=0, ge=0)
    num_articles: int = Field(default=0, ge=0)
    avg_tone: float | None = Field(default=None, ge=-100.0, le=100.0)

    action_geo_fullname: str | None = Field(default=None, max_length=255)
    action_geo_country_code: str | None = Field(default=None, max_length=3)
    action_geo_lat: float | None = Field(default=None, ge=-90.0, le=90.0)
    action_geo_long: float | None = Field(default=None, ge=-180.0, le=180.0)

    source_url: str | None = Field(default=None, max_length=512)

    @field_validator("sql_date")
    @classmethod
    def _validate_sql_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y%m%d")
        except ValueError as exc:
            raise ValueError(f"sql_date must be valid YYYYMMDD, got {v!r}") from exc
        return v


class NewsArticleSchema(BaseModel):
    """Validated schema for a NewsAPI article."""

    model_config = {"extra": "ignore", "str_strip_whitespace": True}

    source: str = Field(min_length=1, max_length=128)
    author: str | None = Field(default=None, max_length=255)
    title: str = Field(min_length=1, max_length=1024)
    description: str | None = Field(default=None, max_length=4096)
    url: HttpUrl
    image_url: HttpUrl | None = Field(default=None, alias="urlToImage")
    published_at: datetime = Field(alias="publishedAt")
    content: str | None = Field(default=None, max_length=8192)

    external_id: str | None = Field(default=None, max_length=255)

    @field_validator("published_at")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)

    @field_validator("external_id", mode="before")
    @classmethod
    def _default_external_id(cls, v: Any) -> Any:
        return v or None


class RedditPostSchema(BaseModel):
    """Validated schema for a parsed Reddit RSS post."""

    model_config = {"extra": "ignore", "str_strip_whitespace": True}

    source: Literal["reddit"] = "reddit"
    external_id: str = Field(min_length=1, max_length=64)
    subreddit: str = Field(min_length=1, max_length=64)
    author: str | None = Field(default=None, max_length=128)
    title: str = Field(min_length=1, max_length=512)
    description: str | None = Field(default=None, max_length=8192)
    url: HttpUrl
    published_at: datetime
    score: int = Field(default=0)
    num_comments: int = Field(default=0, ge=0)
    language: str | None = Field(default=None, max_length=8)

    @field_validator("published_at")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


class IngestionSummary(BaseModel):
    """Summary returned at the end of an ingestion run."""

    model_config = {"extra": "forbid"}

    source: str
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    fetched: int = 0
    parsed: int = 0
    persisted: int = 0
    skipped: int = 0
    errors: list[str] = Field(default_factory=list)
    success: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "fetched": self.fetched,
            "parsed": self.parsed,
            "persisted": self.persisted,
            "skipped": self.skipped,
            "errors": self.errors,
            "success": self.success,
            "metadata": self.metadata,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


__all__ = [
    "GdeltEventSchema",
    "NewsArticleSchema",
    "RedditPostSchema",
    "IngestionSummary",
]
