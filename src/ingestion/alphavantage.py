"""Alpha Vantage (market data: stocks, forex, crypto) client.

Reference:
    https://www.alphavantage.co/documentation/

The free tier is 25 requests/day and 75 requests/minute. All responses
are JSON.
"""

from __future__ import annotations

import time
from typing import Any

import requests

USER_AGENT = "OSCAR-Dashboard/0.5 (educational; contact: aman.raj.intern@bserc.org)"

_DEFAULT_TIMEOUT = 10
_MAX_RETRIES = 1


class AlphaVantageClient:
    """Lightweight Alpha Vantage REST client.

    Wraps the free tier endpoints (25 req/day). Supports:
    - get_quote (GLOBAL_QUOTE)
    - get_daily (TIME_SERIES_DAILY)
    - get_fx_rate (CURRENCY_EXCHANGE_RATE)
    """

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT
        self._last_call: float = 0.0
        self._min_interval: float = 0.8  # 75 req/min safe

    def _request(self, params: dict[str, str]) -> dict[str, Any]:
        from src.config import get_settings, reset_settings_cache

        reset_settings_cache()
        cfg = get_settings().alphavantage
        if not cfg.api_key:
            return {}

        params = {**params, "apikey": cfg.api_key}
        self._throttle()
        for attempt in range(_MAX_RETRIES + 1):
            try:
                r = self._session.get(cfg.base_url, params=params, timeout=_DEFAULT_TIMEOUT)
            except requests.RequestException:
                if attempt < _MAX_RETRIES:
                    time.sleep(1.0)
                    continue
                return {}
            if r.status_code == 429:
                if attempt < _MAX_RETRIES:
                    time.sleep(2.0)
                    continue
                return {}
            if r.status_code != 200:
                return {}
            try:
                data = r.json()
            except ValueError:
                return {}
            if "Note" in data or "Information" in data:
                # Free tier rate limit message; treat as empty
                return {}
            if "Error Message" in data:
                return {}
            return data
        return {}

    def _throttle(self) -> None:
        if self._min_interval > 0:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def get_quote(self, symbol: str) -> dict[str, Any] | None:
        """Real-time quote for a stock/ETF/FX/crypto symbol."""
        if not symbol:
            return None
        data = self._request({"function": "GLOBAL_QUOTE", "symbol": symbol.upper()})
        q = data.get("Global Quote") or {}
        if not q:
            return None
        try:
            return {
                "symbol": q.get("01. symbol"),
                "price": float(q.get("05. price", 0)) if q.get("05. price") else None,
                "open": float(q.get("02. open", 0)) if q.get("02. open") else None,
                "high": float(q.get("03. high", 0)) if q.get("03. high") else None,
                "low": float(q.get("04. low", 0)) if q.get("04. low") else None,
                "volume": int(q.get("06. volume", 0)) if q.get("06. volume") else None,
                "latest_day": q.get("07. latest trading day"),
                "previous_close": (
                    float(q.get("08. previous close", 0)) if q.get("08. previous close") else None
                ),
                "change": q.get("09. change"),
                "change_pct": q.get("10. change percent"),
            }
        except (TypeError, ValueError):
            return None

    def get_daily(self, symbol: str, days: int = 30) -> list[dict[str, Any]]:
        """Last N days of daily OHLCV data for a symbol."""
        if not symbol:
            return []
        data = self._request(
            {"function": "TIME_SERIES_DAILY", "symbol": symbol.upper(), "outputsize": "compact"}
        )
        series = data.get("Time Series (Daily)") or {}
        out: list[dict[str, Any]] = []
        for date_str in sorted(series.keys(), reverse=True)[:days]:
            row = series[date_str]
            try:
                out.append(
                    {
                        "date": date_str,
                        "open": float(row.get("1. open", 0)),
                        "high": float(row.get("2. high", 0)),
                        "low": float(row.get("3. low", 0)),
                        "close": float(row.get("4. close", 0)),
                        "volume": int(row.get("5. volume", 0)),
                    }
                )
            except (TypeError, ValueError):
                continue
        return out

    def get_fx_rate(self, from_currency: str, to_currency: str) -> dict[str, Any] | None:
        """Real-time FX rate between two currencies."""
        if not from_currency or not to_currency:
            return None
        data = self._request(
            {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": from_currency.upper(),
                "to_currency": to_currency.upper(),
            }
        )
        if not data:
            return None
        rate_info = data.get("Realtime Currency Exchange Rate") or {}
        if not rate_info:
            return None
        try:
            return {
                "from": rate_info.get("1. From_Currency Code"),
                "to": rate_info.get("3. To_Currency Code"),
                "rate": float(rate_info.get("5. Exchange Rate", 0)),
                "last_refreshed": rate_info.get("6. Last Refreshed"),
                "bid": float(rate_info.get("8. Bid Price", 0)),
                "ask": float(rate_info.get("9. Ask Price", 0)),
            }
        except (TypeError, ValueError):
            return None


__all__ = ["AlphaVantageClient"]


_CLIENT: AlphaVantageClient | None = None


def get_client() -> AlphaVantageClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = AlphaVantageClient()
    return _CLIENT
