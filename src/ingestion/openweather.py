"""OpenWeather API client (free tier).

Endpoints used:
    - Current weather:  /data/2.5/weather
    - 5-day/3-hour forecast: /data/2.5/forecast
    - Geocoding:  /geo/1.0/direct

Reference:
    https://openweathermap.org/current
    https://openweathermap.org/forecast5
    https://openweathermap.org/api/geocoding-api

The class is designed to be used both as a one-shot client and as a
module-level cached singleton (via `get_weather_cached`).
"""

from __future__ import annotations

import time
from typing import Any

import requests

USER_AGENT = "OSCAR-Dashboard/0.5 (educational; contact: aman.raj.intern@bserc.org)"

_DEFAULT_TIMEOUT = 10
_MAX_CITY_LEN = 100


class OpenWeatherClient:
    """Lightweight OpenWeather REST API client."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT
        self._last_call: float = 0.0
        self._min_interval: float = 0.0

    @staticmethod
    def _clean(s: str) -> str:
        return (s or "").strip()[:_MAX_CITY_LEN]

    def _throttle(self) -> None:
        if self._min_interval > 0:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def _geocode(self, city: str, country_code: str | None = None) -> dict[str, Any] | None:
        from src.config import get_settings

        cfg = get_settings().openweather
        params = {"q": city, "limit": 1, "appid": cfg.api_key}
        if country_code:
            params["q"] = f"{city},{country_code}"
        self._throttle()
        try:
            r = self._session.get(cfg.geocoding_url, params=params, timeout=_DEFAULT_TIMEOUT)
        except requests.RequestException:
            return None
        if r.status_code != 200:
            return None
        try:
            data = r.json()
        except ValueError:
            return None
        if not isinstance(data, list) or not data:
            return None
        return data[0]

    def get_current(self, city: str, country_code: str | None = None) -> dict[str, Any] | None:
        """Fetch current weather for a city."""
        city = self._clean(city)
        if not city:
            return None
        from src.config import get_settings

        cfg = get_settings().openweather
        if not cfg.api_key:
            return None
        geo = self._geocode(city, country_code)
        if not geo:
            return None
        params = {
            "lat": geo.get("lat"),
            "lon": geo.get("lon"),
            "appid": cfg.api_key,
            "units": cfg.units,
        }
        self._throttle()
        try:
            r = self._session.get(
                f"{cfg.base_url}/weather", params=params, timeout=_DEFAULT_TIMEOUT
            )
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
        return {
            "city": geo.get("name") or geo.get("local_names", {}).get("en") or city,
            "country": geo.get("country"),
            "coord": data.get("coord", {}),
            "weather": data.get("weather", [{}])[0] if data.get("weather") else {},
            "main": data.get("main", {}),
            "wind": data.get("wind", {}),
            "clouds": data.get("clouds", {}),
            "rain": data.get("rain", {}),
            "snow": data.get("snow", {}),
            "dt": data.get("dt"),
            "timezone": data.get("timezone"),
        }

    def get_forecast(self, city: str, country_code: str | None = None) -> list[dict[str, Any]]:
        """Fetch 5-day/3-hour forecast for a city. Returns list of 3-hour entries."""
        city = self._clean(city)
        if not city:
            return []
        from src.config import get_settings

        cfg = get_settings().openweather
        if not cfg.api_key:
            return []
        geo = self._geocode(city, country_code)
        if not geo:
            return []
        params = {
            "lat": geo.get("lat"),
            "lon": geo.get("lon"),
            "appid": cfg.api_key,
            "units": cfg.units,
        }
        self._throttle()
        try:
            r = self._session.get(
                f"{cfg.base_url}/forecast", params=params, timeout=_DEFAULT_TIMEOUT
            )
        except requests.RequestException:
            return []
        if r.status_code != 200:
            return []
        try:
            data = r.json()
        except ValueError:
            return []
        if not isinstance(data, dict):
            return []
        out: list[dict[str, Any]] = []
        for entry in data.get("list", []):
            if not isinstance(entry, dict):
                continue
            out.append(
                {
                    "dt": entry.get("dt"),
                    "temp": entry.get("main", {}).get("temp"),
                    "feels_like": entry.get("main", {}).get("feels_like"),
                    "temp_min": entry.get("main", {}).get("temp_min"),
                    "temp_max": entry.get("main", {}).get("temp_max"),
                    "humidity": entry.get("main", {}).get("humidity"),
                    "weather": entry.get("weather", [{}])[0] if entry.get("weather") else {},
                    "wind_speed": entry.get("wind", {}).get("speed"),
                    "wind_deg": entry.get("wind", {}).get("deg"),
                    "clouds": entry.get("clouds", {}).get("all"),
                    "pop": entry.get("pop"),
                    "rain_3h": entry.get("rain", {}).get("3h", 0),
                    "dt_txt": entry.get("dt_txt"),
                }
            )
        return out


__all__ = ["OpenWeatherClient"]


_CLIENT: OpenWeatherClient | None = None


def get_client() -> OpenWeatherClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenWeatherClient()
    return _CLIENT
