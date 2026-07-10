"""Wikipedia API client (no auth, no key, public).

Uses the Wikimedia REST API to fetch:
- Page summaries (lead paragraph + thumbnail)
- Pageview counts (proxy for topic activity / interest)

Reference:
    https://en.wikipedia.org/api/rest_v1/
    https://wikimedia.org/api/rest_v1/

The REST API requires no authentication and is freely usable.
We send a custom User-Agent per Wikimedia's API etiquette.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

USER_AGENT = "OSCAR-Dashboard/0.5 (educational; contact: aman.raj.intern@bserc.org)"

_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/"
    "per-article/{project}/all-access/all-agents/{title}/daily/{start}/{end}"
)

_DEFAULT_TIMEOUT = 10
_MAX_TITLE_LEN = 240


class WikipediaClient:
    """Lightweight Wikipedia REST API client."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT

    @staticmethod
    def normalize_title(title: str) -> str:
        """Normalize a free-text entity name to a likely Wikipedia article title.

        'Wagner Group' -> 'Wagner_Group'
        'F-16'         -> 'F-16'
        'Kyiv, Ukraine' -> 'Kyiv'
        """
        if not title:
            return ""
        # Strip parentheticals / qualifiers
        t = re.sub(r"\s*\([^)]*\)", "", title).strip()
        # Drop trailing commas and noise
        t = t.split(",")[0].strip()
        # Keep technical names with alnum and dashes
        t = re.sub(r"[^\w\s-]", "", t)
        # Collapse spaces, convert to underscores
        t = re.sub(r"\s+", "_", t).strip("_")
        return t[:_MAX_TITLE_LEN]

    def get_summary(self, title: str) -> dict[str, Any] | None:
        """Fetch Wikipedia page summary. Returns None if not found."""
        normalized = self.normalize_title(title)
        if not normalized:
            return None
        url = _SUMMARY_URL.format(title=normalized)
        try:
            r = self._session.get(url, timeout=_DEFAULT_TIMEOUT)
        except requests.RequestException:
            return None
        if r.status_code != 200:
            return None
        try:
            data = r.json()
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None

        def _digested() -> dict[str, Any]:
            content_urls = data.get("content_urls") or {}
            desktop = content_urls.get("desktop") or {}
            thumb = data.get("thumbnail") or {}
            return {
                "title": data.get("title", normalized),
                "extract": data.get("extract", ""),
                "url": desktop.get("page", ""),
                "thumbnail": thumb.get("source"),
                "description": data.get("description", ""),
            }

        return _digested()

    def get_pageviews(self, title: str, days: int = 30) -> int:
        """Total pageviews over the last N days. Returns 0 if unavailable."""
        normalized = self.normalize_title(title)
        if not normalized:
            return 0
        end = datetime.now(timezone.utc).strftime("%Y%m%d")
        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y%m%d")
        url = _PAGEVIEWS_URL.format(
            project="en.wikipedia",
            title=normalized,
            start=start,
            end=end,
        )
        try:
            r = self._session.get(url, timeout=_DEFAULT_TIMEOUT)
        except requests.RequestException:
            return 0
        if r.status_code != 200:
            return 0
        try:
            data = r.json()
        except ValueError:
            return 0
        if not isinstance(data, dict):
            return 0

        # Wikipedia returns "items" as a LIST of {project, article, granularity, timestamp, accesses}
        items = data.get("items") or []
        total = 0
        if isinstance(items, dict):
            # Some endpoints return a dict keyed by date
            for v in items.values():
                try:
                    total += int(v or 0)
                except (TypeError, ValueError):
                    continue
        elif isinstance(items, list):
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                try:
                    total += int(entry.get("accesses") or 0)
                except (TypeError, ValueError):
                    continue
        return total


__all__ = ["WikipediaClient"]


def get_summary_cached(title: str) -> dict[str, Any] | None:
    """Convenience function: cached, single-instance client."""
    return _CLIENT.get_summary(title)


_CLIENT = WikipediaClient()
