"""Tests for the Streamlit dashboard.

These tests verify that:
- All page modules import cleanly.
- All utility functions are importable and have correct signatures.
- Cached data loaders work end-to-end with a fresh DB.

Streamlit rendering itself is NOT tested (would require Playwright/Streamlit's
testing framework, which is out of scope here). We instead test the data +
helper layer that backs the pages.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

PAGES_DIR = Path("dashboard/pages")


def test_dashboard_app_module_imports() -> None:
    importlib.import_module("dashboard.app")


def test_dashboard_utils_module_imports() -> None:
    importlib.import_module("dashboard.utils")


def _list_pages() -> list[str]:
    return sorted(p.stem for p in PAGES_DIR.glob("*.py") if not p.name.startswith("_"))


@pytest.mark.parametrize("page_name", _list_pages())
def test_page_module_imports(page_name: str, fresh_db) -> None:  # noqa: ARG001
    importlib.import_module(f"dashboard.pages.{page_name}")


def test_css_file_exists() -> None:
    assert Path("dashboard/assets/style.css").exists()
    content = Path("dashboard/assets/style.css").read_text(encoding="utf-8")
    assert "OSCAR" in content or "--primary" in content


def test_format_int() -> None:
    from dashboard.utils import format_int

    assert format_int(0) == "0"
    assert format_int(999) == "999"
    assert format_int(1500).endswith("K")
    assert format_int(2_500_000).endswith("M")
    assert format_int(None) == "—"


def test_format_pct() -> None:
    from dashboard.utils import format_pct

    assert format_pct(0.5) == "50.0%"
    assert format_pct(None) == "—"


def test_sentiment_color() -> None:
    from dashboard.utils import sentiment_color

    assert sentiment_color(0.5) == "#16a34a"
    assert sentiment_color(-0.5) == "#dc2626"
    assert sentiment_color(0.0) == "#f59e0b"
    assert sentiment_color(None) == "#9ca3af"


def test_severity_color() -> None:
    from dashboard.utils import severity_color

    assert severity_color(0.9) == "#dc2626"
    assert severity_color(0.5) == "#f59e0b"
    assert severity_color(0.2) == "#3b82f6"
    assert severity_color(None) == "#9ca3af"


def test_country_code_to_iso3_valid_alpha2() -> None:
    from dashboard.utils import country_code_to_iso3

    assert country_code_to_iso3("US") == "USA"
    assert country_code_to_iso3("GB") == "GBR"
    assert country_code_to_iso3("UA") == "UKR"


def test_country_code_to_iso3_handles_alpha3() -> None:
    from dashboard.utils import country_code_to_iso3

    assert country_code_to_iso3("USA") == "USA"
    assert country_code_to_iso3("GBR") == "GBR"


def test_country_code_to_iso3_handles_invalid() -> None:
    from dashboard.utils import country_code_to_iso3

    assert country_code_to_iso3("XX") == "XX"
    assert country_code_to_iso3("") == ""
    assert country_code_to_iso3(None) == ""


def test_get_overview_metrics_returns_dict(fresh_db) -> None:  # noqa: ARG001
    from dashboard import utils

    metrics = utils.get_overview_metrics.__wrapped__()
    assert isinstance(metrics, dict)
    for key in (
        "events",
        "articles",
        "entities",
        "sentiments",
        "topics",
        "anomalies",
        "risk_scores",
    ):
        assert key in metrics


def test_get_events_dataframe_empty(fresh_db) -> None:  # noqa: ARG001
    from dashboard import utils

    df = utils.get_events_dataframe.__wrapped__(days=30)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_articles_dataframe_empty(fresh_db) -> None:  # noqa: ARG001
    from dashboard import utils

    df = utils.get_articles_dataframe.__wrapped__(days=30)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_top_entities_empty(fresh_db) -> None:  # noqa: ARG001
    from dashboard import utils

    df = utils.get_top_entities.__wrapped__(limit=10)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_recent_anomalies_empty(fresh_db) -> None:  # noqa: ARG001
    from dashboard import utils

    df = utils.get_recent_anomalies.__wrapped__(limit=10)
    assert isinstance(df, pd.DataFrame)


def test_get_silver_tables_empty(fresh_db, monkeypatch) -> None:  # noqa: ARG001
    """Silver parquet should be empty if it doesn't exist on disk."""
    import shutil

    from dashboard import utils
    from src.config import get_settings, reset_settings_cache

    monkeypatch.setenv("APP_ENV", "test")
    reset_settings_cache()

    settings = get_settings()
    silver_dir = settings.processed_data_dir / "silver"
    backup = silver_dir.with_suffix(".bak")
    if silver_dir.exists():
        if backup.exists():
            shutil.rmtree(backup)
        silver_dir.rename(backup)
    try:
        e = utils.get_silver_events_per_country_day.__wrapped__()
        a = utils.get_silver_articles_per_source_day.__wrapped__()
        assert isinstance(e, pd.DataFrame)
        assert e.empty
        assert isinstance(a, pd.DataFrame)
        assert a.empty
    finally:
        if backup.exists():
            if silver_dir.exists():
                shutil.rmtree(silver_dir)
            backup.rename(silver_dir)


def test_get_forecast_for_region_empty(fresh_db) -> None:  # noqa: ARG001
    from dashboard import utils

    df = utils.get_forecast_for_region.__wrapped__("XYZ")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_topics_empty(fresh_db) -> None:  # noqa: ARG001
    from dashboard import utils

    df = utils.get_topics.__wrapped__(n=10)
    assert isinstance(df, pd.DataFrame)


def test_get_production_model_metrics_empty(fresh_db) -> None:  # noqa: ARG001
    from dashboard import utils

    metrics = utils.get_production_model_metrics.__wrapped__()
    assert isinstance(metrics, dict)
    assert "escalation_h7" in metrics
