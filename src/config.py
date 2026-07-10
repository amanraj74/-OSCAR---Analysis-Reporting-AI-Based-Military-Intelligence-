"""Centralized typed configuration for OSCAR.

Loads from environment variables (via Pydantic Settings) and from
`configs/settings.yaml`. Validated at startup; raises on invalid values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _csv_or_list(v: Any) -> list[str]:
    """Accept a list, a JSON list, or a comma-separated string; return list[str]."""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        if s.startswith("["):
            import json

            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                pass
        return [x.strip() for x in s.split(",") if x.strip()]
    return [str(v)]


class GdeltSettings(BaseSettings):
    """GDELT Project 2.0 settings."""

    base_url: str = "http://data.gdeltproject.org/gdeltv2/"
    last_update_url: str = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
    batch_hours_back: int = Field(default=24, ge=1, le=168)
    rate_limit_per_minute: int = Field(default=30, ge=1, le=600)

    model_config = SettingsConfigDict(env_prefix="GDELT_", extra="ignore")


class NewsApiSettings(BaseSettings):
    """NewsAPI.org settings."""

    api_key: str = Field(default="", alias="NEWS_API_KEY")
    base_url: str = "https://newsapi.org/v2/"
    requests_per_day: int = Field(default=100, ge=1, le=100000)
    rate_limit_per_minute: int = Field(default=5, ge=1, le=600)

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")


class AcledSettings(BaseSettings):
    """ACLED (Armed Conflict Location & Event Data) settings.

    Uses OAuth 2.0 password grant. Set ACLED_USERNAME and ACLED_PASSWORD
    in .env (or env vars). Register at https://acleddata.com/registration/.
    """

    username: str = Field(default="", alias="ACLED_USERNAME")
    password: str = Field(default="", alias="ACLED_PASSWORD")
    base_url: str = "https://acleddata.com/api/acled"
    token_url: str = "https://acleddata.com/oauth/token"
    page_size: int = Field(default=5000, ge=100, le=5000)
    rate_limit_per_minute: int = Field(default=30, ge=1, le=120)

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")


class OpenWeatherSettings(BaseSettings):
    """OpenWeather API settings.

    Register free at https://home.openweathermap.org/users/sign_up.
    Free tier: 60 calls/min, 1M calls/month.
    """

    api_key: str = Field(default="", alias="OPENWEATHER_API_KEY")
    base_url: str = "https://api.openweathermap.org/data/2.5"
    geocoding_url: str = "http://api.openweathermap.org/geo/1.0/direct"
    units: str = "metric"
    rate_limit_per_minute: int = Field(default=55, ge=1, le=60)
    cache_ttl_seconds: int = Field(default=600, ge=60, le=3600)

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")


class AlphaVantageSettings(BaseSettings):
    """Alpha Vantage (market data: stocks, forex, crypto) settings.

    Register free at https://www.alphavantage.co/support/#api-key.
    Free tier: 25 requests/day (no credit card needed).
    """

    api_key: str = Field(default="", alias="ALPHA_VANTAGE_API_KEY")
    base_url: str = "https://www.alphavantage.co/query"
    rate_limit_per_minute: int = Field(default=25, ge=1, le=75)
    cache_ttl_seconds: int = Field(default=3600, ge=300, le=86400)

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")


class RedditSettings(BaseSettings):
    """Reddit RSS settings."""

    subreddits_raw: str = Field(
        default="worldnews,geopolitics,ukraine,IsraelPalestine,worldevents",
        alias="subreddits",
    )
    max_posts_per_sub: int = Field(default=50, ge=1, le=500)
    rate_limit_per_minute: int = Field(default=10, ge=1, le=600)

    model_config = SettingsConfigDict(env_prefix="REDDIT_", extra="ignore", populate_by_name=True)

    @property
    def subreddits(self) -> list[str]:
        return _csv_or_list(self.subreddits_raw)


class NlpSettings(BaseSettings):
    """NLP model settings."""

    spacy_model: str = "en_core_web_sm"
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    embedding_model: str = "all-MiniLM-L6-v2"
    device: Literal["cpu", "cuda"] = "cpu"

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")


class MlSettings(BaseSettings):
    """Machine learning settings."""

    escalation_horizon_days: list[int] = Field(default_factory=lambda: [1, 3, 7])
    forecast_horizon_days: int = Field(default=7, ge=1, le=30)
    anomaly_contamination: float = Field(default=0.05, ge=0.001, le=0.5)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = 42

    model_config = SettingsConfigDict(env_prefix="ML_", extra="ignore")


class DashboardSettings(BaseSettings):
    """Streamlit dashboard settings."""

    theme_base: Literal["light", "dark"] = "light"
    cache_ttl_seconds: int = Field(default=300, ge=10, le=3600)
    default_map_zoom: int = Field(default=2, ge=0, le=10)

    model_config = SettingsConfigDict(env_prefix="STREAMLIT_", extra="ignore")


class Settings(BaseSettings):
    """Top-level OSCAR settings."""

    app_name: str = "oscar"
    app_version: str = "0.1.1"
    app_env: Literal["dev", "staging", "prod", "ci", "test"] = "dev"
    debug: bool = False
    log_level: str = "INFO"
    database_url: str = "sqlite:///data/oscar.db"

    gdelt: GdeltSettings = Field(default_factory=GdeltSettings)
    newsapi: NewsApiSettings = Field(default_factory=NewsApiSettings)
    acled: AcledSettings = Field(default_factory=AcledSettings)
    openweather: OpenWeatherSettings = Field(default_factory=OpenWeatherSettings)
    alphavantage: AlphaVantageSettings = Field(default_factory=AlphaVantageSettings)
    reddit: RedditSettings = Field(default_factory=RedditSettings)
    nlp: NlpSettings = Field(default_factory=NlpSettings)
    ml: MlSettings = Field(default_factory=MlSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)

    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    external_data_dir: Path = PROJECT_ROOT / "data" / "external"
    models_dir: Path = PROJECT_ROOT / "models"
    logs_dir: Path = PROJECT_ROOT / "logs"
    config_file: Path = PROJECT_ROOT / "configs" / "settings.yaml"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return v_upper

    def load_yaml_overrides(self) -> None:
        """Layer `configs/settings.yaml` non-secret values onto settings.

        YAML values are used only when the corresponding env var did NOT
        set a value (i.e. the field is still at its Pydantic default).
        """
        if not self.config_file.exists():
            return
        with self.config_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        def _set_if_default(sub_model: Any, field_name: str, yaml_value: Any) -> None:
            default_value = type(sub_model).model_fields[field_name].default
            current = getattr(sub_model, field_name)
            if current == default_value:
                setattr(sub_model, field_name, yaml_value)

        gdelt = data.get("data", {}).get("gdelt", {})
        for key in ("base_url", "last_update_url"):
            if key in gdelt:
                _set_if_default(self.gdelt, key, gdelt[key])
        for key in ("batch_hours_back", "rate_limit_per_minute"):
            if key in gdelt:
                _set_if_default(self.gdelt, key, int(gdelt[key]))

        reddit = data.get("data", {}).get("reddit", {})
        if "subreddits" in reddit:
            _set_if_default(self.reddit, "subreddits_raw", ",".join(reddit["subreddits"]))
        if "max_posts_per_sub" in reddit:
            _set_if_default(self.reddit, "max_posts_per_sub", int(reddit["max_posts_per_sub"]))
        if "rate_limit_per_minute" in reddit:
            _set_if_default(
                self.reddit, "rate_limit_per_minute", int(reddit["rate_limit_per_minute"])
            )

        nlp = data.get("nlp", {})
        for key in ("spacy_model", "sentiment_model", "embedding_model", "device"):
            if key in nlp:
                _set_if_default(self.nlp, key, nlp[key])

        ml = data.get("ml", {})
        for key in (
            "escalation_horizon_days",
            "forecast_horizon_days",
            "anomaly_contamination",
            "test_size",
            "random_state",
        ):
            if key in ml:
                _set_if_default(self.ml, key, ml[key])

        dashboard = data.get("dashboard", {})
        for key in ("theme_base", "cache_ttl_seconds", "default_map_zoom"):
            if key in dashboard:
                _set_if_default(self.dashboard, key, dashboard[key])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton (loads env + YAML)."""
    s = Settings()
    s.load_yaml_overrides()
    return s


def reset_settings_cache() -> None:
    """Clear the settings cache (useful in tests)."""
    get_settings.cache_clear()


if __name__ == "__main__":
    cfg = get_settings()
    print(f"OSCAR v{cfg.app_version} | env={cfg.app_env} | db={cfg.database_url}")
