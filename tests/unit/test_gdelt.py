"""Tests for GDELT parser and ingestor."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from src.ingestion.gdelt import CAMEO_ROOT_CODES, GdeltIngestor, parse_gdelt_bytes, parse_gdelt_line
from src.persistence.database import session_scope
from src.persistence.models import Event


def test_parse_gdelt_line_valid(sample_gdelt_line: str) -> None:
    ev = parse_gdelt_line(sample_gdelt_line)
    assert ev is not None
    assert ev.global_event_id == 123456789
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    assert ev.sql_date == today
    assert ev.actor1_name == "UNITED STATES"
    assert ev.actor1_country_code == "USA"
    assert ev.actor2_name == "RUSSIA"
    assert ev.actor2_country_code == "RUS"
    assert ev.event_code == "190"
    assert ev.event_root_code == "19"
    assert ev.goldstein_scale == pytest.approx(-7.5)
    assert ev.num_mentions == 42
    assert ev.num_articles == 12
    assert ev.avg_tone == pytest.approx(-3.21)
    assert ev.action_geo_fullname == "Kyiv, Ukraine"
    assert ev.action_geo_country_code == "UKR"
    assert ev.action_geo_lat == pytest.approx(50.4501)
    assert ev.action_geo_long == pytest.approx(30.5234)


def test_parse_gdelt_line_malformed() -> None:
    assert parse_gdelt_line("") is None
    assert parse_gdelt_line("only\tfour\tcols") is None
    assert parse_gdelt_line("not_a_number\t20250705\t" + "\t".join([""] * 60)) is None


def test_parse_gdelt_line_short_date() -> None:
    cols = ["999"] + [""] * 60
    cols[1] = "2025"
    line = "\t".join(cols)
    assert parse_gdelt_line(line) is None


def test_event_root_label() -> None:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    year = datetime.now(timezone.utc).strftime("%Y")
    line_parts = ["1", today, "", year] + [""] * 60
    while len(line_parts) < 61:
        line_parts.append("")
    line_parts[27] = "19"
    line = "\t".join(line_parts[:61])

    ev = parse_gdelt_line(line)
    assert ev is not None
    assert ev.event_root_label == "FIGHT"
    assert ev.is_conflict is True


def test_parse_gdelt_bytes_plain() -> None:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    year = datetime.now(timezone.utc).strftime("%Y")
    line_parts = ["1", today, "", year] + [""] * 60
    while len(line_parts) < 61:
        line_parts.append("")
    line_parts[27] = "19"
    line = "\t".join(line_parts[:61])

    payload = line.encode("utf-8") + b"\n"
    events = parse_gdelt_bytes(payload)
    assert len(events) == 1
    assert events[0].global_event_id == 1


def test_parse_gdelt_bytes_zip() -> None:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    year = datetime.now(timezone.utc).strftime("%Y")
    line_parts = ["2", today, "", year] + [""] * 60
    while len(line_parts) < 61:
        line_parts.append("")
    line_parts[27] = "14"
    line = "\t".join(line_parts[:61]).encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("events.csv", line + b"\n")
    payload = buf.getvalue()

    events = parse_gdelt_bytes(payload, source_url="http://test/gdelt.zip")
    assert len(events) == 1
    assert events[0].global_event_id == 2
    assert events[0].source_url == "http://test/gdelt.zip"
    assert events[0].is_conflict is True


def test_gdelt_ingestor_parses_with_window_filter(sample_gdelt_line: str) -> None:  # noqa: ARG001

    ingestor = GdeltIngestor()
    events = ingestor.parse([sample_gdelt_line.encode("utf-8")])
    assert len(events) == 1
    assert events[0].global_event_id == 123456789


def test_gdelt_ingestor_dedupes_events(sample_gdelt_line: str) -> None:  # noqa: ARG001

    ingestor = GdeltIngestor()
    payload = (sample_gdelt_line + "\n" + sample_gdelt_line).encode("utf-8")
    events = ingestor.parse([payload])
    assert len(events) == 1


def test_gdelt_event_to_db_row() -> None:
    from src.ingestion.gdelt import parse_gdelt_line

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    year = datetime.now(timezone.utc).strftime("%Y")
    cols = ["12345", today, "", year] + [""] * 60
    while len(cols) < 61:
        cols.append("")
    cols[6] = "A"
    cols[7] = "AAA"
    cols[16] = "B"
    cols[17] = "BBB"
    cols[27] = "19"
    cols[29] = "-5.0"
    cols[30] = "-1.0"
    cols[31] = "1"
    cols[33] = "1"
    cols[49] = "10.0"
    cols[50] = "20.0"
    line = "\t".join(cols[:61])

    ev = parse_gdelt_line(line)
    assert ev is not None
    row = ev.to_db_row()
    assert row["global_event_id"] == 12345
    assert row["actor1_country_code"] == "AAA"
    assert row["action_geo_lat"] == 10.0


def test_canonicalize_basic() -> None:
    from src.ingestion.gdelt import _canonicalize

    assert _canonicalize("Wagner Group") == "wagner group"
    assert _canonicalize("  F-16  ") == "f-16"
    assert _canonicalize("PMC Wagner!!") == "pmc wagner"
    assert _canonicalize("") == "_unknown_"


def test_cameo_conflict_detection() -> None:
    for code in ["14", "15", "16", "17", "18", "19", "20"]:
        assert code in CAMEO_ROOT_CODES


def test_ingestor_persist_idempotent(fresh_db, sample_gdelt_line: str) -> None:

    ingestor = GdeltIngestor()
    first = ingestor.persist([ingestor.parse([sample_gdelt_line.encode("utf-8")])[0]])
    assert first == 1

    with session_scope() as s:
        all_events = s.execute(select(Event)).scalars().all()
        assert len(all_events) == 1

    second = ingestor.persist([ingestor.parse([sample_gdelt_line.encode("utf-8")])[0]])
    assert second == 1

    with session_scope() as s:
        all_events = s.execute(select(Event)).scalars().all()
        assert len(all_events) == 1
