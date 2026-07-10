"""Shared pytest fixtures for OSCAR tests."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.config import reset_settings_cache
from src.persistence import models  # noqa: F401
from src.persistence.database import init_schema, reset_engine


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Force a clean, isolated environment for every test."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("NEWS_API_KEY", "test-news-key")
    monkeypatch.setenv("GDELT_BATCH_HOURS_BACK", "48")
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture()
def tmp_db_path(tmp_path: Path) -> Path:
    """Return a unique tmp DB path."""
    return tmp_path / "test.db"


@pytest.fixture()
def fresh_db(tmp_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Create a fresh OSCAR schema in a tmp SQLite DB.

    Resets the global engine so `get_engine()` returns an engine
    pointing at the same tmp DB used by ingestors/persistence code.
    """
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_db_path}")
    reset_settings_cache()

    reset_engine()
    init_schema()
    yield tmp_db_path
    reset_engine()
    reset_settings_cache()


@pytest.fixture()
def sample_gdelt_line() -> str:
    """A single valid GDELT 2.0 tab-delimited event line (61 columns, dated today UTC).

    Column indices (0-based) follow GDELT 2.0 spec:
        0  GLOBALEVENTID
        1  SQLDATE
        3  Year
        6  Actor1Name         7  Actor1CountryCode
        16 Actor2Name         17 Actor2CountryCode
        25 IsRootEvent         26 EventCode          27 EventRootCode
        29 GoldsteinScale     30 AvgTone            31 NumMentions
        32 NumSources         33 NumArticles
        49 ActionGeo_Lat      50 ActionGeo_Long
        52 ActionGeo_FullName 53 ActionGeo_CountryCode
    """
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    year = datetime.now(timezone.utc).strftime("%Y")
    cols = [""] * 61
    cols[0] = "123456789"
    cols[1] = today
    cols[3] = year
    cols[6] = "UNITED STATES"
    cols[7] = "USA"
    cols[16] = "RUSSIA"
    cols[17] = "RUS"
    cols[25] = "1"
    cols[26] = "190"
    cols[27] = "19"
    cols[29] = "-7.5"
    cols[30] = "-3.21"
    cols[31] = "42"
    cols[33] = "12"
    cols[52] = "Kyiv, Ukraine"
    cols[53] = "UKR"
    cols[49] = "50.4501"
    cols[50] = "30.5234"
    return "\t".join(cols)


@pytest.fixture()
def sample_gdelt_bytes(sample_gdelt_line: str) -> bytes:
    """Raw GDELT payload (uncompressed tab-delimited text)."""
    return sample_gdelt_line.encode("utf-8")
