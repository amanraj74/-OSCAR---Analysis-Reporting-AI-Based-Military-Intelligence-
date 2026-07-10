"""Tests for the OpenWeather client (no live HTTP)."""

from __future__ import annotations

import os

os.environ.setdefault("OPENWEATHER_API_KEY", "test_key_for_unit_tests")


def _mock_response(status, json_data):
    from unittest.mock import MagicMock

    m = MagicMock()
    m.status_code = status
    m.json.return_value = json_data
    return m


def test_current_handles_empty_city():
    from src.ingestion.openweather import OpenWeatherClient

    client = OpenWeatherClient()
    assert client.get_current("") is None
    assert client.get_current("   ") is None


def test_current_handles_401():
    from src.ingestion.openweather import OpenWeatherClient

    client = OpenWeatherClient()
    client._session.get = lambda *_args, **_kw: _mock_response(
        401, {"cod": 401, "message": "bad key"}
    )
    assert client.get_current("Moscow") is None


def test_current_parses_response():
    from unittest.mock import MagicMock

    from src.ingestion.openweather import OpenWeatherClient

    client = OpenWeatherClient()
    geo_resp = _mock_response(
        200, [{"name": "Moscow", "country": "RU", "lat": 55.75, "lon": 37.62, "local_names": {}}]
    )
    wx_resp = _mock_response(
        200,
        {
            "coord": {"lon": 37.62, "lat": 55.75},
            "weather": [{"id": 800, "main": "Clear", "description": "clear sky"}],
            "main": {"temp": 5.2, "feels_like": 1.0, "humidity": 65},
            "wind": {"speed": 4.5, "deg": 180},
            "clouds": {"all": 0},
            "dt": 1700000000,
            "timezone": 10800,
        },
    )
    client._session.get = MagicMock(side_effect=[geo_resp, wx_resp])
    result = client.get_current("Moscow", "RU")
    assert result is not None
    assert result["city"] == "Moscow"
    assert result["country"] == "RU"
    assert result["main"]["temp"] == 5.2
    assert result["weather"]["main"] == "Clear"


def test_forecast_parses_list():
    from unittest.mock import MagicMock

    from src.ingestion.openweather import OpenWeatherClient

    client = OpenWeatherClient()
    geo_resp = _mock_response(200, [{"name": "Kyiv", "country": "UA", "lat": 50.45, "lon": 30.52}])
    fc_resp = _mock_response(
        200,
        {
            "list": [
                {
                    "dt": 1700000000,
                    "main": {
                        "temp": 0.0,
                        "feels_like": -3.0,
                        "temp_min": -1.0,
                        "temp_max": 1.0,
                        "humidity": 80,
                    },
                    "weather": [{"main": "Snow"}],
                    "wind": {"speed": 3.0, "deg": 90},
                    "clouds": {"all": 90},
                    "pop": 0.5,
                    "rain": {"3h": 0.3},
                    "dt_txt": "2024-01-01 00:00:00",
                }
            ]
        },
    )
    client._session.get = MagicMock(side_effect=[geo_resp, fc_resp])
    result = client.get_forecast("Kyiv", "UA")
    assert len(result) == 1
    assert result[0]["temp"] == 0.0
    assert result[0]["pop"] == 0.5
    assert result[0]["rain_3h"] == 0.3


def test_forecast_handles_empty_list():
    from src.ingestion.openweather import OpenWeatherClient

    client = OpenWeatherClient()
    client._session.get = lambda *_a, **_k: _mock_response(200, {"list": []})
    assert client.get_forecast("Nowhere") == []


def test_forecast_handles_error_status():
    from src.ingestion.openweather import OpenWeatherClient

    client = OpenWeatherClient()
    client._session.get = lambda *_a, **_k: _mock_response(500, {"cod": 500})
    assert client.get_forecast("Nowhere") == []


def test_geocode_uses_country_code_suffix():
    from unittest.mock import MagicMock

    from src.ingestion.openweather import OpenWeatherClient

    client = OpenWeatherClient()
    geo_resp = _mock_response(200, [{"name": "X", "country": "X", "lat": 0, "lon": 0}])
    wx_resp = _mock_response(
        200, {"coord": {}, "weather": [], "main": {}, "wind": {}, "clouds": {}}
    )
    mock_get = MagicMock(side_effect=[geo_resp, wx_resp])
    client._session.get = mock_get

    client.get_current("X", "YY")
    args, kwargs = mock_get.call_args_list[0]
    assert kwargs["params"]["q"] == "X,YY"


def test_init_sets_user_agent():
    from src.ingestion.openweather import OpenWeatherClient

    client = OpenWeatherClient()
    assert "OSCAR" in client._session.headers["User-Agent"]


def test_settings_has_openweather():
    import os

    from src.config import get_settings, reset_settings_cache

    os.environ["OPENWEATHER_API_KEY"] = "test_key"
    reset_settings_cache()
    cfg = get_settings()
    assert cfg.openweather.api_key == "test_key"
    assert cfg.openweather.base_url.startswith("https://")
