"""Abstract base class for all OSCAR ingestors.

Provides:
    - Retry with exponential backoff (Tenacity).
    - Rate limiting (token-bucket style per minute).
    - Structured logging via Loguru.
    - Standardized `IngestionResult` return value.
    - Optional HTTP-cache to avoid re-fetching same URL.

Subclasses must implement `fetch_raw()` and `parse()`.
`persist()` has a default implementation that no-ops (subclasses
override to write to DB / Parquet).
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, TypeVar

import requests
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.observability import get_logger

T = TypeVar("T")


class SourceNotAvailableError(RuntimeError):
    """Raised when a source cannot be reached after all retries."""


@dataclass
class IngestionResult(Generic[T]):
    """Standardized result returned by `BaseIngestor.run()`."""

    source: str
    items: list[T] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    success: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "count": self.count,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


class _RateLimiter:
    """Simple per-minute token bucket."""

    def __init__(self, calls_per_minute: int) -> None:
        self._interval = 60.0 / max(calls_per_minute, 1)
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_call = time.monotonic()


class BaseIngestor(ABC, Generic[T]):
    """Abstract base for OSCAR data ingestors."""

    source_name: str = "base"

    def __init__(
        self,
        rate_limit_per_minute: int = 30,
        max_retries: int = 3,
        timeout: int = 30,
        cache_dir: Path | None = None,
        http_session: requests.Session | None = None,
    ) -> None:
        self.settings = get_settings()
        self.rate_limiter = _RateLimiter(rate_limit_per_minute)
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = get_logger(f"ingestion.{self.source_name}")

        self.cache_dir = cache_dir or (self.settings.raw_data_dir / self.source_name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = http_session or requests.Session()
        self.session.headers.setdefault(
            "User-Agent",
            f"OSCAR/{self.settings.app_version} (+https://github.com/amanraj74/-OSCAR---Analysis-Reporting-AI-Based-Military-Intelligence-)",
        )

    @abstractmethod
    def fetch_raw(self) -> list[bytes]:
        """Fetch raw payloads from the source.

        Returns:
            List of raw bytes payloads (each typically a file).
        """

    @abstractmethod
    def parse(self, raw: list[bytes]) -> list[T]:
        """Parse raw payloads into domain items.

        Args:
            raw: Output of `fetch_raw()`.

        Returns:
            Parsed domain items.
        """

    def persist(self, items: list[T]) -> int:
        """Persist items (override in subclasses). Default: no-op.

        Returns:
            Number of items successfully persisted.
        """
        return len(items)

    def run(self) -> IngestionResult[T]:
        """Execute the full fetch → parse → persist pipeline."""
        result = IngestionResult[T](source=self.source_name)
        try:
            self.logger.info("ingestion_start")
            raw = self._fetch_with_retries()
            result.metadata["raw_count"] = len(raw)
            self.logger.info("raw_fetched", count=len(raw))

            items = self.parse(raw)
            result.items = items
            result.metadata["parsed_count"] = len(items)
            self.logger.info("parsed", count=len(items))

            persisted = self.persist(items)
            result.metadata["persisted_count"] = persisted
            result.success = True
            self.logger.info("ingestion_done", count=persisted)

        except Exception as e:
            result.success = False
            result.error = f"{type(e).__name__}: {e}"
            self.logger.exception("ingestion_failed", error=result.error)

        result.finished_at = datetime.now(timezone.utc)
        return result

    def _fetch_with_retries(self) -> list[bytes]:
        """Fetch with retries + rate-limit."""
        retryer = Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=20),
            retry=retry_if_exception_type((requests.RequestException, TimeoutError)),
            reraise=True,
        )
        for attempt in retryer:
            with attempt:
                self.rate_limiter.wait()
                return self.fetch_raw()
        raise SourceNotAvailableError(f"Failed to fetch from {self.source_name}")

    def _http_get(self, url: str) -> bytes:
        """GET request with retries and rate-limit (helper for subclasses)."""
        self.rate_limiter.wait()
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.content

    def _cache_key(self, url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, url: str, suffix: str = ".bin") -> Path:
        return self.cache_dir / f"{self._cache_key(url)}{suffix}"

    def get_cached_or_fetch(self, url: str, suffix: str = ".bin", use_cache: bool = True) -> bytes:
        """Return cached payload if present, else fetch + cache."""
        path = self._cache_path(url, suffix)
        if use_cache and path.exists():
            self.logger.debug("cache_hit", url=url)
            return path.read_bytes()
        data = self._http_get(url)
        path.write_bytes(data)
        return data


__all__ = ["BaseIngestor", "IngestionResult", "SourceNotAvailableError"]
