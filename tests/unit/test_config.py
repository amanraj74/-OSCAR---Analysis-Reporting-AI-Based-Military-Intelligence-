"""Tests for the centralized Settings system."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_settings_load_with_defaults() -> None:
    from src.config import get_settings

    settings = get_settings()
    assert settings.app_name == "oscar"
    assert settings.app_env == "test"
    assert settings.app_version == "0.1.1"
    assert settings.log_level == "WARNING"


def test_settings_typed_nested_models() -> None:
    from src.config import get_settings

    settings = get_settings()
    assert hasattr(settings, "gdelt")
    assert hasattr(settings, "newsapi")
    assert hasattr(settings, "reddit")
    assert hasattr(settings, "nlp")
    assert hasattr(settings, "ml")
    assert hasattr(settings, "dashboard")


def test_settings_path_resolution() -> None:
    from src.config import get_settings

    settings = get_settings()
    assert isinstance(settings.project_root, Path)
    assert settings.processed_data_dir.is_absolute()
    assert (settings.processed_data_dir).name == "processed"


def test_invalid_log_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import ValidationError

    from src.config import Settings, reset_settings_cache

    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
    reset_settings_cache()
    with pytest.raises(ValidationError):
        Settings()


def test_settings_yaml_overrides() -> None:
    from src.config import get_settings

    settings = get_settings()
    assert settings.gdelt.batch_hours_back == 48
    assert "worldnews" in settings.reddit.subreddits


def test_news_api_key_loaded_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import get_settings, reset_settings_cache

    monkeypatch.setenv("NEWS_API_KEY", "abc-123")
    reset_settings_cache()
    assert get_settings().newsapi.api_key == "abc-123"


def test_horizon_days_parsed() -> None:
    from src.config import get_settings

    settings = get_settings()
    assert 1 in settings.ml.escalation_horizon_days
    assert settings.ml.forecast_horizon_days >= 1
    assert 0.0 < settings.ml.test_size < 1.0
