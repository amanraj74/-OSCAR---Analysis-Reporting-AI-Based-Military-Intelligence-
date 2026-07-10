"""NewsAPI.org news ingestor.

Pulls articles from `/v2/everything` (search) and `/v2/top-headlines`
by source / country. Free tier: 100 requests/day.

Reference:
    https://newsapi.org/docs/endpoints/everything
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import ValidationError
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.config import get_settings
from src.domain.schemas import NewsArticleSchema
from src.ingestion.base import BaseIngestor
from src.observability import get_logger
from src.persistence.database import session_scope
from src.persistence.models import Article


@dataclass
class NewsArticle:
    """A single validated NewsAPI article, normalized for OSCAR."""

    external_id: str
    source: str
    title: str
    description: str | None
    url: str
    image_url: str | None
    published_at: datetime
    author: str | None
    content: str | None
    language: str | None

    def to_db_row(self) -> dict[str, Any]:
        return {
            "external_id": self.external_id,
            "source": "newsapi",
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "url": self.url,
            "author": self.author,
            "image_url": self.image_url,
            "language": self.language,
            "published_at": self.published_at,
        }


def _stable_id(url: str, title: str) -> str:
    """Generate a stable ID for an article when one isn't supplied."""
    h = hashlib.sha256()
    h.update(url.encode("utf-8"))
    h.update(b"|")
    h.update(title.encode("utf-8"))
    return h.hexdigest()[:32]


def _parse_newsapi_response(payload: dict[str, Any]) -> list[NewsArticle]:
    """Parse a NewsAPI JSON response into validated articles.

    Deduplicates on (url, title) — multiple carriers sometimes repeat.
    """
    deduped: dict[str, NewsArticle] = {}
    for raw in payload.get("articles", []) or []:
        item = dict(raw)
        src = item.get("source")
        if isinstance(src, dict):
            item["source"] = src.get("name") or src.get("id") or ""
        if not item["source"]:
            continue
        try:
            schema = NewsArticleSchema.model_validate(item)
        except ValidationError:
            continue

        article = NewsArticle(
            external_id=schema.external_id or _stable_id(str(schema.url), schema.title),
            source=schema.source,
            title=schema.title,
            description=schema.description,
            url=str(schema.url),
            image_url=str(schema.image_url) if schema.image_url else None,
            published_at=schema.published_at,
            author=schema.author,
            content=schema.content,
            language=None,
        )
        deduped.setdefault(article.external_id, article)
    return list(deduped.values())


class NewsApiIngestor(BaseIngestor[NewsArticle]):
    """NewsAPI ingestor.

    Usage:
        ingestor = NewsApiIngestor(query="Ukraine", page_size=50)
        result = ingestor.run()

    Requires `NEWS_API_KEY` env var (free at newsapi.org/register).
    """

    source_name = "newsapi"

    def __init__(
        self,
        query: str | None = None,
        sources: str | None = None,
        country: str | None = None,
        category: str | None = None,
        page_size: int = 50,
        language: str = "en",
        **kwargs: Any,
    ) -> None:
        rate_limit = kwargs.pop(
            "rate_limit_per_minute",
            get_settings().newsapi.rate_limit_per_minute,
        )
        super().__init__(rate_limit_per_minute=rate_limit, **kwargs)

        self.query = query
        self.sources = sources
        self.country = country
        self.category = category
        self.page_size = min(max(page_size, 1), 100)
        self.language = language
        self.logger = get_logger(f"ingestion.{self.source_name}")

    def _endpoint(self) -> str:
        if self.query:
            return f"{self.settings.newsapi.base_url}everything"
        return f"{self.settings.newsapi.base_url}top-headlines"

    def _params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "pageSize": self.page_size,
            "language": self.language,
        }
        if self.query:
            params["q"] = self.query
            params["sortBy"] = "publishedAt"
        if self.sources:
            params["sources"] = self.sources
        if self.country:
            params["country"] = self.country
        if self.category:
            params["category"] = self.category
        return params

    def fetch_raw(self) -> list[bytes]:
        """Fetch articles from NewsAPI."""
        api_key = self.settings.newsapi.api_key
        if not api_key:
            self.logger.warning("newsapi_no_key", hint="Set NEWS_API_KEY in .env")
            return []

        url = self._endpoint()
        params = self._params()
        headers = {"X-Api-Key": api_key, "User-Agent": self.session.headers.get("User-Agent", "")}

        self.rate_limiter.wait()
        response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()

        if response.status_code == 429:
            self.logger.error("newsapi_rate_limited")
            return []

        return [response.content]

    def parse(self, raw: list[bytes]) -> list[NewsArticle]:
        """Parse NewsAPI JSON responses into validated articles."""
        articles: list[NewsArticle] = []
        for payload in raw:
            try:
                data = json.loads(payload.decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.logger.warning("newsapi_decode_failed")
                continue

            if data.get("status") != "ok":
                self.logger.warning(
                    "newsapi_status_not_ok",
                    code=data.get("code"),
                    msg=data.get("message"),
                )
                continue

            articles.extend(_parse_newsapi_response(data))

        return articles

    def persist(self, items: list[NewsArticle]) -> int:
        """Bulk-insert articles; idempotent on (source, external_id)."""
        if not items:
            return 0

        rows = [a.to_db_row() for a in items]
        with session_scope() as session:
            stmt = sqlite_insert(Article).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Article.source, Article.external_id],
                set_={
                    "title": stmt.excluded.title,
                    "description": stmt.excluded.description,
                    "content": stmt.excluded.content,
                    "url": stmt.excluded.url,
                    "author": stmt.excluded.author,
                    "image_url": stmt.excluded.image_url,
                    "published_at": stmt.excluded.published_at,
                },
            )
            session.execute(stmt)
        return len(rows)


__all__ = ["NewsArticle", "NewsApiIngestor"]
