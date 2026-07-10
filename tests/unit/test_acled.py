"""Tests for the ACLED ingestor (without real network calls)."""

from __future__ import annotations

import json

import pytest

SAMPLE_ACLED_RECORDS = [
    {
        "event_id_cnty": "USA12345",
        "event_date": "2025-07-01",
        "year": 2025,
        "event_type": "Violence against civilians",
        "sub_event_type": "Attack",
        "disorder_type": "Violence against civilians",
        "actor1": "Government of USA",
        "actor2": "Protesters (USA)",
        "country": "United States",
        "iso": "USA",
        "region": "Northern America",
        "admin1": "California",
        "location": "Los Angeles",
        "latitude": "34.05",
        "longitude": "-118.24",
        "fatalities": "2",
        "source": "Reuters",
    },
    {
        "event_id_cnty": "UKR98765",
        "event_date": "2025-07-02",
        "year": 2025,
        "event_type": "Battle",
        "disorder_type": "Battles",
        "actor1": "Military of Russia (RUS)",
        "actor2": "Military of Ukraine (UKR)",
        "country": "Ukraine",
        "iso": "UKR",
        "location": "Kharkiv",
        "latitude": "49.99",
        "longitude": "36.23",
        "fatalities": "5",
    },
]


def test_acled_mapping_basic():
    from src.ingestion.acled import _map_acled_to_event

    rec = SAMPLE_ACLED_RECORDS[0]
    row = _map_acled_to_event(rec)
    assert row["global_event_id"] == 12345
    assert row["sql_date"] == "20250701"
    assert row["year"] == 2025
    assert row["actor1_name"] == "Government of USA"
    assert row["actor1_country_code"] == "USA"
    assert row["actor2_name"] == "Protesters (USA)"
    assert row["action_geo_fullname"] == "Los Angeles"
    assert row["action_geo_lat"] == 34.05
    assert row["action_geo_long"] == -118.24
    assert row["num_articles"] == 2  # fatalities mapped to num_articles
    assert row["event_root_code"] == "18"  # Violence against civilians


def test_acled_mapping_battle():
    from src.ingestion.acled import _map_acled_to_event

    rec = SAMPLE_ACLED_RECORDS[1]
    row = _map_acled_to_event(rec)
    assert row["event_root_code"] == "19"  # Battle
    assert row["actor1_country_code"] == "UKR"
    assert row["actor2_name"] == "Military of Ukraine (UKR)"


def test_acled_mapping_actor2_iso_extraction():
    from src.ingestion.acled import _map_acled_to_event

    rec = {
        "event_id_cnty": "X1",
        "event_date": "2025-01-01",
        "year": 2025,
        "iso": "RUS",
        "actor1": "Government",
        "actor2": "Protesters (UKR)",
        "country": "Russia",
        "event_type": "Riots",
        "location": "Moscow",
    }
    row = _map_acled_to_event(rec)
    assert row["actor2_country_code"] == "UKR"


def test_acled_ingestor_parse():
    from src.ingestion.acled import AcledIngestor

    ingestor = AcledIngestor()
    raw = [json.dumps(r).encode("utf-8") for r in SAMPLE_ACLED_RECORDS]
    events = ingestor.parse(raw)
    assert len(events) == 2
    assert events[0]["global_event_id"] == 12345
    assert events[1]["actor2_country_code"] == "UKR"


def test_acled_ingestor_requires_credentials():
    from src.ingestion.acled import AcledIngestor

    ingestor = AcledIngestor()
    with pytest.raises(RuntimeError, match="ACLED credentials"):
        ingestor._get_access_token()


def test_acled_ingestor_parse_handles_bad_json():
    from src.ingestion.acled import AcledIngestor

    ingestor = AcledIngestor()
    raw = [b"not json", b'{"valid": 1}']  # Second is valid JSON but NOT an ACLED event
    events = ingestor.parse(raw)
    # Both are filtered out: not JSON, and not an ACLED event (no event_id_cnty)
    assert len(events) == 0

    # Test that a malformed JSON with ACLED shape is also filtered
    raw_bad = [b"this is not json"]
    assert len(ingestor.parse(raw_bad)) == 0


def test_acled_safe_float():
    from src.ingestion.acled import _safe_float

    assert _safe_float("1.5") == 1.5
    assert _safe_float(None) is None
    assert _safe_float("") is None
    assert _safe_float("abc") is None


def test_acled_event_id_handles_dash_format():
    from src.ingestion.acled import _map_acled_to_event

    rec = {"event_id_cnty": "USA-12345-1", "iso": "USA", "location": "X"}
    row = _map_acled_to_event(rec)
    # 12345 is the integer portion before the dash
    assert row["global_event_id"] == 12345


def test_acled_settings_in_config():
    from src.config import get_settings, reset_settings_cache

    reset_settings_cache()
    cfg = get_settings()
    assert hasattr(cfg, "acled")
    assert hasattr(cfg.acled, "username")
    assert hasattr(cfg.acled, "password")
    assert hasattr(cfg.acled, "base_url")
    assert cfg.acled.base_url == "https://acleddata.com/api/acled"
    assert cfg.acled.token_url == "https://acleddata.com/oauth/token"
