"""GDELT Project 2.0 events ingestor.

GDELT 2.0 publishes pipe-delimited event files every 15 minutes. Each
file is a `.export.CSV.zip` containing rows of 61 columns. We extract a
focused subset of columns most useful for OSCAR's escalation, sentiment,
and geo visualizations.

Reference:
    https://www.gdeltproject.org/data.html#documentation
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.ingestion.base import BaseIngestor
from src.persistence.database import session_scope
from src.persistence.models import Event

CAMEO_ROOT_CODES: dict[str, str] = {
    "01": "MAKE PUBLIC STATEMENT",
    "02": "APPEAL",
    "03": "EXPRESS INTENT TO COOPERATE",
    "04": "CONSULT",
    "05": "ENGAGE IN DIPLOMATIC COOPERATION",
    "06": "ENGAGE IN MATERIAL COOPERATION",
    "07": "PROVIDE AID",
    "08": "YIELD",
    "09": "INVESTIGATE",
    "10": "DEMAND",
    "11": "DISAPPROVE",
    "12": "REJECT",
    "13": "THREATEN",
    "14": "PROTEST",
    "15": "EXHIBIT FORCE POSTURE",
    "16": "REDUCE RELATIONS",
    "17": "COERCE",
    "18": "ASSAULT",
    "19": "FIGHT",
    "20": "USE UNCONVENTIONAL MASS VIOLENCE",
}


@dataclass
class GdeltEvent:
    """A single GDELT 2.0 event, normalized for OSCAR."""

    global_event_id: int
    sql_date: str
    year: int | None
    actor1_name: str | None
    actor1_country_code: str | None
    actor2_name: str | None
    actor2_country_code: str | None
    event_code: str | None
    event_root_code: str | None
    event_root_label: str | None
    goldstein_scale: float | None
    num_mentions: int
    num_articles: int
    avg_tone: float | None
    action_geo_fullname: str | None
    action_geo_country_code: str | None
    action_geo_lat: float | None
    action_geo_long: float | None
    source_url: str | None

    def to_db_row(self) -> dict[str, Any]:
        return {
            "global_event_id": self.global_event_id,
            "sql_date": self.sql_date,
            "year": self.year,
            "actor1_name": self.actor1_name,
            "actor1_country_code": self.actor1_country_code,
            "actor2_name": self.actor2_name,
            "actor2_country_code": self.actor2_country_code,
            "event_code": self.event_code,
            "event_root_code": self.event_root_code,
            "goldstein_scale": self.goldstein_scale,
            "num_mentions": self.num_mentions,
            "num_articles": self.num_articles,
            "avg_tone": self.avg_tone,
            "action_geo_fullname": self.action_geo_fullname,
            "action_geo_country_code": self.action_geo_country_code,
            "action_geo_lat": self.action_geo_lat,
            "action_geo_long": self.action_geo_long,
            "source_url": self.source_url,
        }

    @property
    def is_conflict(self) -> bool:
        """True if event_root_code is in the conflict range (14-20)."""
        if not self.event_root_code:
            return False
        return self.event_root_code in {"14", "15", "16", "17", "18", "19", "20"}

    @property
    def has_geo(self) -> bool:
        return self.action_geo_lat is not None and self.action_geo_long is not None


_LAST_UPDATE_LINE_RE = re.compile(r"^\S+\s+\S+\s+(https?://\S+)$")


def _parse_last_update(text: str) -> list[tuple[str, str]]:
    """Parse GDELT's lastupdate.txt into (filename, url) tuples.

    GDELT 2.0 lastupdate.txt format:  `<size> <md5> <url>`  (3 columns)
    """
    out: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _LAST_UPDATE_LINE_RE.match(line)
        if m:
            url = m.group(1)
            filename = url.rsplit("/", 1)[-1]
            out.append((filename, url))
    return out


def _coerce_int(v: str | None) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _coerce_float(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def parse_gdelt_line(line: str) -> GdeltEvent | None:
    """Parse a single GDELT 2.0 tab-delimited event line.

    Returns None for malformed lines. Skipped silently in batch processing.
    """
    parts = line.rstrip("\n\r").split("\t")
    if len(parts) < 58:
        return None

    event_id = _coerce_int(parts[0])
    sql_date = parts[1] or ""
    if event_id is None or not sql_date or len(sql_date) != 8:
        return None

    event_root_code = parts[27] or None
    return GdeltEvent(
        global_event_id=event_id,
        sql_date=sql_date,
        year=_coerce_int(parts[3]),
        actor1_name=parts[6] or None,
        actor1_country_code=parts[7] or None,
        actor2_name=parts[16] or None,
        actor2_country_code=parts[17] or None,
        event_code=parts[26] or None,
        event_root_code=event_root_code,
        event_root_label=CAMEO_ROOT_CODES.get(event_root_code) if event_root_code else None,
        goldstein_scale=_coerce_float(parts[29]),
        num_mentions=_coerce_int(parts[31]) or 0,
        num_articles=_coerce_int(parts[33]) or 0,
        avg_tone=_coerce_float(parts[30]),
        action_geo_fullname=parts[52] or None if len(parts) > 52 else None,
        action_geo_country_code=parts[53] or None if len(parts) > 53 else None,
        action_geo_lat=_coerce_float(parts[49]) if len(parts) > 49 else None,
        action_geo_long=_coerce_float(parts[50]) if len(parts) > 50 else None,
        source_url=None,
    )


def parse_gdelt_bytes(payload: bytes, source_url: str | None = None) -> list[GdeltEvent]:
    """Parse a GDELT export file (raw, gzipped, or zip).

    Args:
        payload: Raw file bytes.
        source_url: Original URL (recorded for audit).

    Returns:
        List of parsed events.
    """
    text: str | None = None

    try:
        if payload[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as fh:
                    text = fh.read().decode("utf-8", errors="replace")
        elif payload[:2] == b"\x1f\x8b":
            text = __import__("gzip").decompress(payload).decode("utf-8", errors="replace")
        else:
            text = payload.decode("utf-8", errors="replace")
    except Exception:
        return []

    if text is None:
        return []

    out: list[GdeltEvent] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        ev = parse_gdelt_line(line)
        if ev is None:
            continue
        ev.source_url = source_url
        out.append(ev)
    return out


class GdeltIngestor(BaseIngestor[GdeltEvent]):
    """GDELT 2.0 events ingestor.

    Pulls the most recent export files via the `lastupdate.txt` index.
    Filters events to the last `batch_hours_back` window.
    """

    source_name = "gdelt"

    def __init__(
        self,
        max_files: int = 5,
        filter_window_hours: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rate_limit_per_minute=kwargs.pop(
                "rate_limit_per_minute",
                get_settings_safe("rate_limit_per_minute", 30),
            ),
            **kwargs,
        )
        self.max_files = max_files
        self.filter_window_hours = filter_window_hours or self.settings.gdelt.batch_hours_back
        self._last_update_cache: list[tuple[str, str]] | None = None

    def _fetch_last_update(self) -> list[tuple[str, str]]:
        """Fetch & cache the GDELT last-update index."""
        if self._last_update_cache is not None:
            return self._last_update_cache
        url = self.settings.gdelt.last_update_url
        payload = self._http_get(url)
        self._last_update_cache = _parse_last_update(payload.decode("utf-8", errors="replace"))
        self.logger.info("gdelt_last_update", files=len(self._last_update_cache))
        return self._last_update_cache

    def fetch_raw(self) -> list[bytes]:
        """Download the most recent GDELT export files."""
        entries = self._fetch_last_update()
        if not entries:
            return []
        entries = entries[-self.max_files :]
        out: list[bytes] = []
        for _filename, url in entries:
            try:
                data = self._http_get(url)
                out.append(data)
            except requests.RequestException as e:
                self.logger.warning("gdelt_file_failed", url=url, error=str(e))
        return out

    def parse(self, raw: list[bytes]) -> list[GdeltEvent]:
        """Parse all downloaded GDELT files, filter to window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.filter_window_hours)
        cutoff_date = cutoff.strftime("%Y%m%d")
        cutoff_dt = int(cutoff_date)

        all_events: list[GdeltEvent] = []
        for payload in raw:
            events = parse_gdelt_bytes(payload, source_url="gdelt://lastupdate")
            for ev in events:
                ev_date_int = _coerce_int(ev.sql_date) or 0
                if ev_date_int >= cutoff_dt:
                    all_events.append(ev)

        deduped: dict[int, GdeltEvent] = {}
        for ev in all_events:
            deduped.setdefault(ev.global_event_id, ev)

        result = list(deduped.values())
        self.logger.info(
            "gdelt_parsed",
            total=len(all_events),
            unique=len(result),
            window_hours=self.filter_window_hours,
        )
        return result

    def persist(self, items: list[GdeltEvent]) -> int:
        """Bulk-insert events into SQLite, skipping duplicates."""
        if not items:
            return 0

        rows = [ev.to_db_row() for ev in items]
        with session_scope() as session:
            stmt = sqlite_insert(Event).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Event.global_event_id],
                set_={
                    "num_mentions": stmt.excluded.num_mentions,
                    "num_articles": stmt.excluded.num_articles,
                    "avg_tone": stmt.excluded.avg_tone,
                    "ingested_at": stmt.excluded.ingested_at,
                },
            )
            session.execute(stmt)
        return len(rows)


def get_settings_safe(attr: str, default: Any) -> Any:
    """Lazy access to settings to avoid circular import at module load."""
    try:
        from src.config import get_settings as _gs

        cfg = _gs()
        return getattr(cfg.gdelt, attr, default)
    except Exception:  # noqa: BLE001
        return default


__all__ = [
    "GdeltEvent",
    "GdeltIngestor",
    "parse_gdelt_line",
    "parse_gdelt_bytes",
    "CAMEO_ROOT_CODES",
    "_canonicalize",
]


def _canonicalize(name: str) -> str:
    """Lowercase, strip whitespace, basic punctuation normalization."""
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"^[\W_]+|[\W_]+$", "", name)
    return name or "_unknown_"
