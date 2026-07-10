"""Reddit RSS ingestor.

Pulls public subreddit feeds via `.rss` endpoints. No authentication
required; respects Reddit's public-access ToS by polling politely.

Reference:
    https://www.reddit.com/r/<subreddit>/.rss
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import feedparser
from pydantic import ValidationError
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.config import get_settings
from src.domain.schemas import RedditPostSchema
from src.ingestion.base import BaseIngestor
from src.observability import get_logger
from src.persistence.database import session_scope
from src.persistence.models import Article


@dataclass
class RedditPost:
    """A parsed Reddit post, normalized for OSCAR."""

    external_id: str
    source: str
    title: str
    description: str | None
    url: str
    published_at: datetime
    author: str | None
    subreddit: str
    score: int
    num_comments: int

    def to_db_row(self) -> dict[str, Any]:
        return {
            "external_id": self.external_id,
            "source": "reddit",
            "title": self.title,
            "description": self.description,
            "url": self.url,
            "author": self.author,
            "published_at": self.published_at,
        }


_PERMALINK_RE = re.compile(r"/r/(?P<sub>[^/]+)/comments/(?P<id>[a-z0-9]+)/", re.IGNORECASE)


def _extract_id_and_subreddit(entry: dict[str, Any], fallback_url: str) -> tuple[str, str]:
    """Extract (post_id, subreddit) from an entry."""
    permalink = entry.get("link") or fallback_url
    m = _PERMALINK_RE.search(permalink)
    if m:
        return m.group("id"), m.group("sub")

    guid = entry.get("id") or entry.get("guid") or permalink
    return str(guid).split("/")[-1].split("?")[0] or "unknown", "unknown"


def _coerce_dt(value: Any) -> datetime | None:
    """Parse various datetime formats into timezone-aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    try:
        dt = datetime(*value[:6], tzinfo=timezone.utc)  # type: ignore[misc]
        return dt
    except (TypeError, ValueError):
        pass

    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(str(value))
        if dt:
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        pass

    return None


def _parse_entry(entry: dict[str, Any], subreddit: str) -> RedditPost | None:
    """Convert a single feedparser entry into a RedditPost."""
    permalink = entry.get("link") or ""
    external_id, detected_sub = _extract_id_and_subreddit(entry, permalink)
    sub = subreddit or detected_sub

    title = (entry.get("title") or "").strip()
    if not title:
        return None

    published_at = (
        _coerce_dt(entry.get("published_parsed"))
        or _coerce_dt(entry.get("published"))
        or _coerce_dt(entry.get("updated_parsed"))
        or _coerce_dt(entry.get("updated"))
        or datetime.now(timezone.utc)
    )

    summary = entry.get("summary") or entry.get("description") or ""

    author = entry.get("author") or entry.get("dc_creator") or None
    if author and "/" in author:
        author = author.split("/")[-1]

    try:
        schema = RedditPostSchema.model_validate(
            {
                "external_id": external_id,
                "subreddit": sub,
                "title": title[:500],
                "description": summary[:8000] or None,
                "url": permalink or "https://www.reddit.com/r/" + sub + "/",
                "published_at": published_at,
                "author": author,
                "score": 0,
                "num_comments": 0,
            }
        )
    except ValidationError:
        return None

    return RedditPost(
        external_id=schema.external_id,
        source="reddit",
        title=schema.title,
        description=schema.description,
        url=str(schema.url),
        published_at=schema.published_at,
        author=schema.author,
        subreddit=schema.subreddit,
        score=0,
        num_comments=0,
    )


def parse_reddit_feed(payload: bytes, subreddit: str) -> list[RedditPost]:
    """Parse a Reddit RSS payload into RedditPost list."""
    feed = feedparser.parse(payload)
    if getattr(feed, "bozo", 0) and not feed.entries:
        return []
    return [p for p in (_parse_entry(e, subreddit) for e in feed.entries) if p is not None]


class RedditIngestor(BaseIngestor[RedditPost]):
    """Reddit RSS ingestor.

    Usage:
        ingestor = RedditIngestor(subreddits=["worldnews", "geopolitics"])
        result = ingestor.run()
    """

    source_name = "reddit"

    def __init__(
        self,
        subreddits: list[str] | None = None,
        max_posts_per_sub: int | None = None,
        **kwargs: Any,
    ) -> None:
        rate_limit = kwargs.pop(
            "rate_limit_per_minute",
            get_settings().reddit.rate_limit_per_minute,
        )
        super().__init__(rate_limit_per_minute=rate_limit, **kwargs)
        cfg = get_settings().reddit
        self.subreddits = subreddits or cfg.subreddits
        self.max_posts_per_sub = max_posts_per_sub or cfg.max_posts_per_sub
        self.logger = get_logger(f"ingestion.{self.source_name}")

    def _feed_url(self, subreddit: str) -> str:
        return f"{self.settings.reddit.base_url}r/{subreddit}/.rss"

    def fetch_raw(self) -> list[bytes]:
        """Fetch RSS feeds for each configured subreddit."""
        out: list[bytes] = []
        for sub in self.subreddits:
            url = self._feed_url(sub)
            try:
                data = self._http_get(url)
                out.append(data)
            except Exception as e:
                self.logger.warning("reddit_fetch_failed", subreddit=sub, error=str(e))
        return out

    def parse(self, raw: list[bytes]) -> list[RedditPost]:
        """Parse all subreddit feeds; cap per subreddit."""
        all_posts: list[RedditPost] = []
        for payload, sub in zip(raw, self.subreddits, strict=False):
            posts = parse_reddit_feed(payload, subreddit=sub)
            all_posts.extend(posts[: self.max_posts_per_sub])

        deduped: dict[str, RedditPost] = {}
        for p in all_posts:
            deduped.setdefault(p.external_id, p)
        return list(deduped.values())

    def persist(self, items: list[RedditPost]) -> int:
        """Bulk-insert Reddit posts; idempotent on (source, external_id)."""
        if not items:
            return 0

        rows = [p.to_db_row() for p in items]
        with session_scope() as session:
            stmt = sqlite_insert(Article).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Article.source, Article.external_id],
                set_={
                    "title": stmt.excluded.title,
                    "description": stmt.excluded.description,
                    "url": stmt.excluded.url,
                    "author": stmt.excluded.author,
                    "published_at": stmt.excluded.published_at,
                },
            )
            session.execute(stmt)
        return len(rows)


__all__ = ["RedditPost", "RedditIngestor", "parse_reddit_feed"]
