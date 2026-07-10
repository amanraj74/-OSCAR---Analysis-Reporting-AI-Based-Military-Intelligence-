"""Tests for the Alpha Vantage client (no live HTTP)."""

from __future__ import annotations

import os

os.environ.setdefault("ALPHA_VANTAGE_API_KEY", "test_key_for_unit_tests")


def _mock_response(json_data, status: int = 200):
    from unittest.mock import MagicMock

    m = MagicMock()
    m.status_code = status
    m.json.return_value = json_data
    return m


def test_quote_handles_empty_symbol():
    from src.ingestion.alphavantage import AlphaVantageClient

    client = AlphaVantageClient()
    assert client.get_quote("") is None
    assert client.get_quote(None) is None


def test_quote_handles_missing_data():
    from src.ingestion.alphavantage import AlphaVantageClient

    client = AlphaVantageClient()
    client._session.get = lambda *_a, **_k: _mock_response({})
    assert client.get_quote("SPY") is None


def test_quote_handles_rate_limit_message():
    from src.ingestion.alphavantage import AlphaVantageClient

    client = AlphaVantageClient()
    client._session.get = lambda *_a, **_k: _mock_response({"Note": "API rate limit reached"})
    assert client.get_quote("SPY") is None


def test_quote_parses_response():
    from unittest.mock import MagicMock

    from src.ingestion.alphavantage import AlphaVantageClient

    client = AlphaVantageClient()
    mock = _mock_response(
        {
            "Global Quote": {
                "01. symbol": "SPY",
                "02. open": "520.00",
                "03. high": "525.00",
                "04. low": "518.00",
                "05. price": "523.50",
                "06. volume": "1000000",
                "07. latest trading day": "2024-01-15",
                "08. previous close": "520.00",
                "09. change": "3.50",
                "10. change percent": "0.673%",
            }
        }
    )
    client._session.get = MagicMock(return_value=mock)

    quote = client.get_quote("SPY")
    assert quote is not None
    assert quote["symbol"] == "SPY"
    assert quote["price"] == 523.50
    assert quote["volume"] == 1000000
    assert quote["change_pct"] == "0.673%"


def test_daily_handles_empty_response():
    from src.ingestion.alphavantage import AlphaVantageClient

    client = AlphaVantageClient()
    client._session.get = lambda *_a, **_k: _mock_response({})
    assert client.get_daily("SPY") == []


def test_daily_parses_series():
    from unittest.mock import MagicMock

    from src.ingestion.alphavantage import AlphaVantageClient

    client = AlphaVantageClient()
    mock = _mock_response(
        {
            "Time Series (Daily)": {
                "2024-01-15": {
                    "1. open": "100",
                    "2. high": "105",
                    "3. low": "99",
                    "4. close": "103",
                    "5. volume": "1000",
                },
                "2024-01-14": {
                    "1. open": "98",
                    "2. high": "101",
                    "3. low": "97",
                    "4. close": "100",
                    "5. volume": "2000",
                },
            }
        }
    )
    client._session.get = MagicMock(return_value=mock)

    series = client.get_daily("SPY", days=30)
    assert len(series) == 2
    # Newest first
    assert series[0]["date"] == "2024-01-15"
    assert series[0]["close"] == 103.0
    assert series[0]["volume"] == 1000
    assert series[1]["date"] == "2024-01-14"


def test_fx_rate_handles_empty():
    from src.ingestion.alphavantage import AlphaVantageClient

    client = AlphaVantageClient()
    client._session.get = lambda *_a, **_k: _mock_response({})
    assert client.get_fx_rate("USD", "RUB") is None


def test_fx_rate_parses():
    from unittest.mock import MagicMock

    from src.ingestion.alphavantage import AlphaVantageClient

    client = AlphaVantageClient()
    mock = _mock_response(
        {
            "Realtime Currency Exchange Rate": {
                "1. From_Currency Code": "USD",
                "3. To_Currency Code": "RUB",
                "5. Exchange Rate": "92.4500",
                "6. Last Refreshed": "2024-01-15 12:00:00",
                "8. Bid Price": "92.4400",
                "9. Ask Price": "92.4600",
            }
        }
    )
    client._session.get = MagicMock(return_value=mock)

    fx = client.get_fx_rate("USD", "RUB")
    assert fx is not None
    assert fx["from"] == "USD"
    assert fx["to"] == "RUB"
    assert fx["rate"] == 92.45
    assert fx["bid"] == 92.44
    assert fx["ask"] == 92.46


def test_handles_error_message():
    from src.ingestion.alphavantage import AlphaVantageClient

    client = AlphaVantageClient()
    client._session.get = lambda *_a, **_k: _mock_response({"Error Message": "Invalid API call"})
    assert client.get_quote("SPY") is None
    assert client.get_daily("SPY") == []


def test_handles_non_200():
    from src.ingestion.alphavantage import AlphaVantageClient

    client = AlphaVantageClient()
    client._session.get = lambda *_a, **_k: _mock_response({}, status=500)
    assert client.get_quote("SPY") is None
    assert client.get_daily("SPY") == []


def test_settings_has_alphavantage():
    import os

    from src.config import get_settings, reset_settings_cache

    os.environ["ALPHA_VANTAGE_API_KEY"] = "test_key"
    reset_settings_cache()
    cfg = get_settings()
    assert cfg.alphavantage.api_key == "test_key"
    assert cfg.alphavantage.base_url.startswith("https://")
