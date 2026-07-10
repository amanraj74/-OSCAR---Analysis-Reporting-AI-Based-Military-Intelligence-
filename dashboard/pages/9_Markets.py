"""Market Impact - Live market data correlated with conflict events.

Country keys are FIPS 2-letter codes (matches DB). Each entry maps to
an Alpha Vantage symbol and a currency.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))  # noqa: E402 bootstrap sys.path

# noqa: E402 - imports below must come after sys.path bootstrap
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.utils import (  # noqa: E402
    empty_state,
    page_header,
    sidebar_status_panel,
)

st.set_page_config(page_title="OSCAR . Markets", page_icon="\U0001f4c8", layout="wide")
sidebar_status_panel()

page_header(
    "Markets",
    "Live market data from Alpha Vantage overlaid with recent conflict events.",
)

# FIPS 2-letter code -> (alpha_vantage_symbol, currency, currency_name, friendly_country_name, flag_emoji)
# FIPS comes from GDELT action_geo_country_code (2-letter codes, not ISO-3).
# Some markets use ETF proxies where direct indices are unavailable.
FIPS_MARKETS = {
    "US": ("SPY", "USD", "US Dollar", "United States", "\U0001f1fa\U0001f1f8"),
    "RU": ("MCX", "RUB", "Russian Ruble", "Russia", "\U0001f1f7\U0001f1fa"),
    "UP": ("UXA", "UAH", "Ukrainian Hryvnia", "Ukraine", "\U0001f1fa\U0001f1e6"),
    "UK": ("ISF.L", "GBP", "British Pound", "United Kingdom", "\U0001f1ec\U0001f1e7"),
    "IN": ("INDA", "INR", "Indian Rupee", "India", "\U0001f1ee\U0001f1f3"),
    "IR": ("TUR", "IRR", "Iranian Rial", "Iran", "\U0001f1ee\U0001f1f7"),
    "IS": ("EIMI", "ILS", "Israeli Shekel", "Israel", "\U0001f1ee\U0001f1f1"),
    "CA": ("EWC", "CAD", "Canadian Dollar", "Canada", "\U0001f1e8\U0001f1e6"),
    "CH": ("EWL", "CHF", "Swiss Franc", "Switzerland", "\U0001f1e8\U0001f1ed"),
    "TU": ("XU100", "TRY", "Turkish Lira", "Turkey", "\U0001f1f9\U0001f1f7"),
    "RS": ("ERUS", "RSD", "Serbian Dinar", "Serbia", "\U0001f1f7\U0001f1f8"),
    "PK": ("KSE100", "PKR", "Pakistani Rupee", "Pakistan", "\U0001f1f5\U0001f1f0"),
    "SF": ("EZA", "ZAR", "South African Rand", "South Africa", "\U0001f1ff\U0001f1e6"),
    "FR": ("EWQ", "EUR", "Euro", "France", "\U0001f1eb\U0001f1f7"),
    "ID": ("EIDO", "IDR", "Indonesian Rupiah", "Indonesia", "\U0001f1ee\U0001f1e9"),
    "EI": ("EIRL", "EUR", "Euro", "Ireland", "\U0001f1ee\U0001f1fa"),
    "SY": ("EIS", "SYP", "Syrian Pound", "Syria", "\U0001f1f8\U0001f1fe"),
    "GM": ("EWG", "EUR", "Euro", "Germany", "\U0001f1e9\U0001f1ea"),
    "RP": ("EPHE", "PHP", "Philippine Peso", "Philippines", "\U0001f1f5\U0001f1ed"),
    "JA": ("EWJ", "JPY", "Japanese Yen", "Japan", "\U0001f1ef\U0001f1f5"),
    "IT": ("EWI", "EUR", "Euro", "Italy", "\U0001f1ee\U0001f1f9"),
    "NZ": ("ENZ", "NZD", "New Zealand Dollar", "New Zealand", "\U0001f1f3\U0001f1ff"),
    "KS": ("EWS", "SAR", "Saudi Riyal", "Saudi Arabia", "\U0001f1f8\U0001f1e6"),
    "SP": ("EWP", "EUR", "Euro", "Spain", "\U0001f1ea\U0001f1f8"),
    "KE": ("AFK", "KES", "Kenyan Shilling", "Kenya", "\U0001f1f0\U0001f1ea"),
    "LE": ("AFK", "LBP", "Lebanese Pound", "Lebanon", "\U0001f1f1\U0001f1e7"),
    "GH": ("AFK", "GHS", "Ghanaian Cedi", "Ghana", "\U0001f1ec\U0001f1ed"),
    "IZ": ("TUR", "IQD", "Iraqi Dinar", "Iraq", "\U0001f1ee\U0001f1f6"),
    "AF": ("AFK", "AFN", "Afghan Afghani", "Afghanistan", "\U0001f1e6\U0001f1eb"),
    "AS": ("EWS", "AUD", "Australian Dollar", "American Samoa", "\U0001f1e6\U0001f1f8"),
    "NI": ("EIDO", "NGN", "Nigerian Naira", "Nigeria", "\U0001f1f3\U0001f1ec"),
    "EG": ("EGPT", "EGP", "Egyptian Pound", "Egypt", "\U0001f1ea\U0001f1ec"),
    "PL": ("EPOL", "PLN", "Polish Zloty", "Poland", "\U0001f1f5\U0001f1f1"),
    "RO": ("GREK", "RON", "Romanian Leu", "Romania", "\U0001f1f7\U0001f1f4"),
    "TH": ("THD", "THB", "Thai Baht", "Thailand", "\U0001f1f9\U0001f1ed"),
    "DA": ("EWG", "EUR", "Euro", "Germany", "\U0001f1e9\U0001f1ea"),
    "EZ": ("EWN", "CZK", "Czech Koruna", "Czech Republic", "\U0001f1e8\U0001f1ff"),
    "BO": ("EWG", "BYN", "Belarusian Ruble", "Belarus", "\U0001f1e7\U0001f1fe"),
    "AL": ("EWG", "ALL", "Albanian Lek", "Albania", "\U0001f1e6\U0001f1f1"),
    "GR": ("EWG", "EUR", "Euro", "Greece", "\U0001f1ec\U0001f1f7"),
    "ME": ("EWG", "EUR", "Euro", "Montenegro", "\U0001f1f2\U0001f1ea"),
    "HR": ("EWG", "EUR", "Euro", "Croatia", "\U0001f1ed\U0001f1f7"),
    "BA": ("EWG", "BAM", "Bosnia Mark", "Bosnia", "\U0001f1e7\U0001f1e6"),
    "MD": ("EWG", "MDL", "Moldovan Leu", "Moldova", "\U0001f1f2\U0001f1e9"),
    "NL": ("EWN", "EUR", "Euro", "Netherlands", "\U0001f1f3\U0001f1f1"),
    "BE": ("EWK", "EUR", "Euro", "Belgium", "\U0001f1e7\U0001f1ea"),
    "AT": ("EWO", "EUR", "Euro", "Austria", "\U0001f1e6\U0001f1f9"),
    "SE": ("EWD", "SEK", "Swedish Krona", "Sweden", "\U0001f1f8\U0001f1ea"),
    "NO": ("ENOR", "NOK", "Norwegian Krone", "Norway", "\U0001f1f3\U0001f1f4"),
    "FI": ("EFNL", "EUR", "Euro", "Finland", "\U0001f1eb\U0001f1ee"),
    "DK": ("EDEN", "DKK", "Danish Krone", "Denmark", "\U0001f1e9\U0001f1f0"),
    "AU": ("EWA", "AUD", "Australian Dollar", "Australia", "\U0001f1e6\U0001f1fa"),
}


@st.cache_data(ttl=3600, show_spinner=False)
def _quote(symbol: str) -> dict[str, Any] | None:
    from src.ingestion.alphavantage import get_client

    return get_client().get_quote(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def _daily(symbol: str) -> list[dict[str, Any]]:
    from src.ingestion.alphavantage import get_client

    return get_client().get_daily(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def _fx_rate(from_cur: str, to_cur: str) -> dict[str, Any] | None:
    from src.ingestion.alphavantage import get_client

    return get_client().get_fx_rate(from_cur, to_cur)


@st.cache_data(ttl=60, show_spinner=False)
def _country_list() -> list[str]:
    from sqlalchemy import distinct, select

    from src.persistence.database import session_scope
    from src.persistence.models import Event

    with session_scope() as s:
        rows = s.execute(
            select(distinct(Event.action_geo_country_code))
            .where(Event.action_geo_country_code.isnot(None))
            .order_by(Event.action_geo_country_code)
        ).all()
    return [r[0] for r in rows if r[0] and r[0] != "UNK"]


@st.cache_data(ttl=60, show_spinner=False)
def _country_recent_events(country_code: str, days: int = 30) -> list[dict[str, Any]]:
    from sqlalchemy import func, select

    from src.persistence.database import session_scope
    from src.persistence.models import Event

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y%m%d")
    with session_scope() as s:
        rows = s.execute(
            select(Event.sql_date, func.count().label("n"))
            .where(Event.action_geo_country_code == country_code)
            .where(Event.sql_date >= cutoff)
            .group_by(Event.sql_date)
            .order_by(Event.sql_date)
        ).all()
    return [{"date": r[0], "events": r[1]} for r in rows]


def _safe_float(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def render() -> None:
    available = _country_list()
    supported_codes = [c for c in available if c in FIPS_MARKETS]

    if not supported_codes:
        empty_state(
            "No supported markets yet.",
            "Try a country that has live markets tracked (e.g., USA, RUS, CHN, ISR).",
        )
        return

    # Build options - also track label -> fips code mapping for later use
    options = []
    label_to_code = {}
    code_to_entry = {}
    for code in supported_codes:
        entry = FIPS_MARKETS[code]
        symbol, currency, currency_name, name, flag = entry
        label = f"{flag} {name} ({code})"
        options.append(label)
        label_to_code[label] = code
        code_to_entry[label] = entry

    selected = st.selectbox(
        "Country",
        options=options,
        index=0,
        key="markets_country",
    )
    country_code = label_to_code[selected]
    symbol, currency, currency_name, country_name, flag_emoji = code_to_entry[selected]

    st.markdown(f"### {flag_emoji} {country_name} - {symbol} (vs {currency} {currency_name})")
    with st.spinner(f"Fetching quote for {symbol}..."):
        quote = _quote(symbol)
    fx = _fx_rate(currency, "USD")

    if not quote:
        st.warning(
            f"Could not fetch quote for {symbol}. Free tier rate limit may have been hit. "
            f"Free tier: 25 requests/day."
        )
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            price = _safe_float(quote.get("price"))
            st.metric("Current price", f"{price:.2f}" if price is not None else "-")
        with c2:
            chg = _safe_float(quote.get("change"))
            st.metric("Change", f"{chg:+.2f}" if chg is not None else "-")
        with c3:
            pct = quote.get("change_pct")
            st.metric("Change %", pct if pct else "-")
        with c4:
            high = _safe_float(quote.get("high"))
            st.metric("Day high", f"{high:.2f}" if high is not None else "-")
        with c5:
            vol = quote.get("volume")
            st.metric("Volume", f"{vol:,}" if vol else "-")
        st.caption(f"Trading day: {quote.get('latest_day', '-')}")

    st.markdown("---")
    st.markdown(f"### 30-day price history for {symbol}")

    if quote:
        series = _daily(symbol)
        if series:
            df = pd.DataFrame(series)
            df["date"] = pd.to_datetime(df["date"])
            fig = px.line(
                df,
                x="date",
                y="close",
                title=f"{symbol} close price (last 30 days)",
            )
            fig.update_layout(height=320, margin={"l": 0, "r": 0, "t": 30, "b": 0})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption(f"No daily data available for {symbol}.")
    else:
        st.caption("Skipped history (no current quote available).")

    st.markdown("---")
    st.markdown(f"### {currency} vs USD")
    if fx:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Current rate", f"{fx.get('rate', 0):.4f}")
        with c2:
            st.metric("Bid", f"{fx.get('bid', 0):.4f}")
        with c3:
            st.metric("Ask", f"{fx.get('ask', 0):.4f}")
        st.caption(f"Last refreshed: {fx.get('last_refreshed', '-')}")
    else:
        st.caption("FX rate not available (free tier limit).")

    st.markdown("---")
    st.markdown("### Conflict overlay (last 30 days)")
    events = _country_recent_events(country_code, days=30)
    if not events:
        st.caption(f"No recent events in {country_code}.")
    else:
        df_e = pd.DataFrame(events)
        df_e["date"] = pd.to_datetime(df_e["date"])

        if quote and series:
            df_price = pd.DataFrame(series)
            df_price["date"] = pd.to_datetime(df_price["date"])
            df_combined = df_price.merge(df_e, on="date", how="outer").sort_values("date")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df_combined["date"],
                    y=df_combined["close"],
                    mode="lines",
                    name=f"{symbol} close",
                    line={"color": "#1e3a8a"},
                )
            )
            fig.add_trace(
                go.Bar(
                    x=df_combined["date"],
                    y=df_combined["events"].fillna(0),
                    name="GDELT events",
                    marker_color="#ef4444",
                    opacity=0.4,
                    yaxis="y2",
                )
            )
            fig.update_layout(
                title=f"{symbol} price vs conflict events in {country_name}",
                height=360,
                yaxis={"title": f"{symbol} price"},
                yaxis2={"title": "Events", "overlaying": "y", "side": "right"},
                margin={"l": 0, "r": 0, "t": 40, "b": 0},
                legend={"orientation": "h", "y": -0.15},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.bar(df_e, x="date", y="events", title=f"Conflict events in {country_name}")
            st.plotly_chart(fig, use_container_width=True)

        n = int(df_e["events"].sum())
        st.metric("Total events (30d)", n)


render()
