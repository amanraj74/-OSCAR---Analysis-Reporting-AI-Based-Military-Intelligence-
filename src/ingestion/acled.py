"""ACLED (Armed Conflict Location & Event Data) ingestor.

Uses OAuth 2.0 password grant for authentication. ACLED returns rich
conflict event data with fatalities, actor types, civilian targeting,
and more — far richer than GDELT.

Reference:
    https://acleddata.com/api-documentation/getting-started
    https://acleddata.com/api-documentation/acled

ACLED records arrive with columns like:
    event_id_cnty, event_date, year, event_type, sub_event_type, actor1,
    actor2, country, iso, region, admin1, admin2, location, latitude,
    longitude, fatalities, source, source_scale, notes, ...

We map them to OSCAR's `Event` model. For ACLED's CAMEO-like event_type
we use `event_root_code` set to a 2-digit derivation (e.g. "19" for
violence against civilians).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import requests
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.config import get_settings, reset_settings_cache
from src.ingestion.base import BaseIngestor
from src.observability import get_logger
from src.persistence.database import session_scope
from src.persistence.models import Event

logger = get_logger("ingestion.acled")


# ACLED's 8 disorder types → OSCAR's 2-digit event root code (CAMEO-like)
_DISORDER_TO_ROOT = {
    "Battles": "19",
    "Explosions/Remote violence": "19",
    "Violence against civilians": "18",
    "Protests": "14",
    "Riots": "16",
    "Strategic developments": "03",
    "Riots; Strategic developments": "16",
}

# ACLED event_type → simplified CAMEO root code
_EVENT_TYPE_TO_ROOT = {
    "Battle": "19",
    "Explosion/Remote violence": "19",
    "Violence against civilians": "18",
    "Mob violence": "16",
    "Protests": "14",
    "Riots": "16",
    "Strategic development": "03",
}


class AcledIngestor(BaseIngestor):
    """Ingestor for ACLED conflict events.

    Requires `ACLED_USERNAME` and `ACLED_PASSWORD` in env. Register free at
    https://acleddata.com/registration/ (academic email gets instant approval).
    """

    source_name = "acled"

    def __init__(
        self,
        country: str | None = None,
        year: int | None = None,
        event_date_from: str | None = None,
        event_date_to: str | None = None,
        max_rows: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rate_limit_per_minute=kwargs.pop("rate_limit_per_minute", 30),
            **kwargs,
        )
        self.country = country
        self.year = year
        self.event_date_from = event_date_from
        self.event_date_to = event_date_to
        self.max_rows = max_rows or 50000
        self._access_token: str | None = None
        self._token_expiry: float = 0.0

    def _get_access_token(self) -> str:
        """Fetch (or reuse) the OAuth 2.0 access token."""
        reset_settings_cache()
        cfg = get_settings()
        if not cfg.acled.username or not cfg.acled.password:
            raise RuntimeError(
                "ACLED credentials missing. Set ACLED_USERNAME and ACLED_PASSWORD in .env."
            )

        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token

        logger.info("acled_requesting_token")
        r = requests.post(
            cfg.acled.token_url,
            data={
                "username": cfg.acled.username,
                "password": cfg.acled.password,
                "grant_type": "password",
                "client_id": "acled",
                "scope": "authenticated",
            },
            timeout=15,
        )
        if r.status_code != 200:
            raise RuntimeError(f"ACLED auth failed: {r.status_code} - {r.text[:200]}")
        data = r.json()
        self._access_token = data["access_token"]
        self._token_expiry = time.time() + int(data.get("expires_in", 86400))
        logger.info("acled_token_ok", expires_in=data.get("expires_in"))
        return self._access_token

    def fetch_raw(self) -> list[bytes]:
        """Fetch ACLED events, paginated."""
        reset_settings_cache()
        cfg = get_settings()

        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json",
        }
        params: dict[str, Any] = {
            "_format": "json",
            "limit": cfg.acled.page_size,
        }
        if self.country:
            params["country"] = self.country
        if self.year:
            params["year"] = self.year
        if self.event_date_from and self.event_date_to:
            params["event_date"] = f"{self.event_date_from}|{self.event_date_to}"
            params["event_date_where"] = "BETWEEN"
        elif self.event_date_from:
            params["event_date"] = f"{self.event_date_from}|2099-12-31"
            params["event_date_where"] = "BETWEEN"

        all_events: list[dict[str, Any]] = []
        offset = 0
        while True:
            params["offset"] = offset
            logger.info(
                "acled_fetching",
                offset=offset,
                accumulated=len(all_events),
            )
            r = requests.get(
                f"{cfg.acled.base_url}/read",
                params=params,
                headers=headers,
                timeout=30,
            )
            if r.status_code != 200:
                logger.warning(
                    "acled_fetch_failed",
                    status=r.status_code,
                    body=r.text[:200],
                )
                break
            payload = r.json()
            if not isinstance(payload, dict) or payload.get("status") != 200:
                logger.warning("acled_bad_response", payload=str(payload)[:200])
                break
            chunk = payload.get("data", [])
            all_events.extend(chunk)
            if not chunk or len(all_events) >= self.max_rows:
                break
            offset += len(chunk)
            if len(chunk) < cfg.acled.page_size:
                break

        logger.info("acled_fetched", total=len(all_events))
        return [json.dumps(e, ensure_ascii=False).encode("utf-8") for e in all_events]

    def parse(self, raw: list[bytes]) -> list[dict[str, Any]]:
        """Parse ACLED JSON rows to dicts ready for DB insert."""
        out: list[dict[str, Any]] = []
        for blob in raw:
            try:
                rec = json.loads(blob.decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(rec, dict) or "event_id_cnty" not in rec:
                continue
            out.append(_map_acled_to_event(rec))
        return out

    def persist(self, items: list[dict[str, Any]]) -> int:
        """Upsert ACLED events into the Event table."""
        if not items:
            return 0
        with session_scope() as session:
            stmt = sqlite_insert(Event).values(items)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Event.global_event_id],
                set_={
                    "actor1_name": stmt.excluded.actor1_name,
                    "actor2_name": stmt.excluded.actor2_name,
                    "actor1_country_code": stmt.excluded.actor1_country_code,
                    "actor2_country_code": stmt.excluded.actor2_country_code,
                    "event_code": stmt.excluded.event_code,
                    "event_root_code": stmt.excluded.event_root_code,
                    "goldstein_scale": stmt.excluded.goldstein_scale,
                    "num_articles": stmt.excluded.num_articles,
                    "avg_tone": stmt.excluded.avg_tone,
                    "action_geo_fullname": stmt.excluded.action_geo_fullname,
                    "action_geo_country_code": stmt.excluded.action_geo_country_code,
                    "action_geo_lat": stmt.excluded.action_geo_lat,
                    "action_geo_long": stmt.excluded.action_geo_long,
                    "source_url": stmt.excluded.source_url,
                    "ingested_at": stmt.excluded.ingested_at,
                },
            )
            session.execute(stmt)
        return len(items)


def _map_acled_to_event(rec: dict[str, Any]) -> dict[str, Any]:
    """Map an ACLED JSON record to OSCAR's Event row dict."""
    raw_eid = str(rec.get("event_id_cnty", "")).strip() or "0"

    if "-" in raw_eid:
        # Format like "USA-12345-1" — take the first numeric segment
        int_candidate = next((seg for seg in raw_eid.split("-") if seg.isdigit()), None)
    else:
        # Format like "USA12345" — extract digit run
        int_candidate = "".join(c for c in raw_eid if c.isdigit())

    try:
        global_event_id = (
            int(int_candidate) if int_candidate else abs(hash(raw_eid)) % 2_000_000_000
        )
    except ValueError:
        global_event_id = abs(hash(raw_eid)) % 2_000_000_000

    raw_date = rec.get("event_date")
    if isinstance(raw_date, str) and len(raw_date) >= 8:
        sql_date = raw_date.replace("-", "")[:8]
    else:
        sql_date = "19700101"

    iso = rec.get("iso", "")
    actor1_cc = iso if len(iso) == 3 else (iso[:3].upper() if iso else None)
    country = rec.get("country", "")
    actor2 = rec.get("actor2", "")
    actor2_cc = None
    if " (" in actor2 and actor2.endswith(")"):
        # Extract ISO code from "Actor (XXX)" format
        inside = actor2.split(" (")[-1].rstrip(")")
        if len(inside) == 3:
            actor2_cc = inside

    ev_type = rec.get("event_type", "")
    disorder = rec.get("disorder_type", "")
    event_root = _EVENT_TYPE_TO_ROOT.get(ev_type) or _DISORDER_TO_ROOT.get(disorder)

    fatalities = rec.get("fatalities")
    try:
        fatalities = int(fatalities) if fatalities is not None else None
    except (ValueError, TypeError):
        fatalities = None

    return {
        "global_event_id": global_event_id,
        "sql_date": sql_date,
        "year": int(rec["year"]) if rec.get("year") else None,
        "actor1_name": rec.get("actor1") or None,
        "actor1_country_code": actor1_cc,
        "actor2_name": rec.get("actor2") or None,
        "actor2_country_code": actor2_cc,
        "event_code": (ev_type[:4] if ev_type else None),
        "event_root_code": event_root,
        "goldstein_scale": None,
        "num_mentions": 1,
        "num_articles": fatalities or 0,
        "avg_tone": None,
        "action_geo_fullname": rec.get("location") or None,
        "action_geo_country_code": actor1_cc,
        "action_geo_lat": _safe_float(rec.get("latitude")),
        "action_geo_long": _safe_float(rec.get("longitude")),
        "source_url": rec.get("source") or None,
        "ingested_at": datetime.now(timezone.utc),
    }


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


__all__ = ["AcledIngestor"]
