"""Weather × Conflict — Live weather from OpenWeather + conflict correlation.

Pick a country (auto-filled from entity database) and a city, then
view:
- Current weather conditions
- 5-day forecast (every 3 hours)
- Recent conflict events in that country (from GDELT/ACLED)
- A risk indicator correlating weather anomalies with unrest
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))  # noqa: F401  bootstrap sys.path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.utils import empty_state, fmt_dt, page_header, sidebar_status_panel

st.set_page_config(page_title="OSCAR · Weather", page_icon="🌤️", layout="wide")
sidebar_status_panel()

page_header(
    "Weather × Conflict",
    "Live weather data from OpenWeather correlated with recent conflict events.",
)


@st.cache_data(ttl=600, show_spinner=False)
def _current(city: str, country_code: str | None) -> dict[str, Any] | None:
    from src.ingestion.openweather import get_client

    return get_client().get_current(city, country_code)


@st.cache_data(ttl=600, show_spinner=False)
def _forecast(city: str, country_code: str | None) -> list[dict[str, Any]]:
    from src.ingestion.openweather import get_client

    return get_client().get_forecast(city, country_code)


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
    from datetime import timedelta

    from sqlalchemy import select

    from src.persistence.database import session_scope
    from src.persistence.models import Event

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y%m%d")
    with session_scope() as s:
        rows = s.execute(
            select(Event.sql_date, Event.event_root_code, Event.goldstein_scale, Event.avg_tone)
            .where(Event.action_geo_country_code == country_code)
            .where(Event.sql_date >= cutoff)
            .limit(500)
        ).all()
    return [{"date": r[0], "root": r[1], "goldstein": r[2], "tone": r[3]} for r in rows]


CITY_SUGGESTIONS = {
    "USA": "New York",
    "RUS": "Moscow",
    "UKR": "Kyiv",
    "CHN": "Beijing",
    "IRN": "Tehran",
    "ISR": "Tel Aviv",
    "PSE": "Gaza",
    "SYR": "Damascus",
    "AFG": "Kabul",
    "IRQ": "Baghdad",
    "YEM": "Sanaa",
    "SOM": "Mogadishu",
    "SSD": "Juba",
    "SDN": "Khartoum",
    "LBY": "Tripoli",
    "IND": "New Delhi",
    "PAK": "Islamabad",
    "BGD": "Dhaka",
    "MMR": "Yangon",
    "TWN": "Taipei",
    "PRK": "Pyongyang",
}


def _temp_to_color(t: float | None) -> str:
    if t is None:
        return "gray"
    if t < -10:
        return "darkblue"
    if t < 0:
        return "blue"
    if t < 15:
        return "lightblue"
    if t < 25:
        return "yellow"
    if t < 35:
        return "orange"
    return "red"


def render() -> None:
    countries = _country_list()
    if not countries:
        empty_state("No countries in the database yet.")
        return

    cfg_col, city_col = st.columns([1, 2], gap="medium")
    with cfg_col:
        country_code = st.selectbox(
            "Country",
            options=countries,
            index=countries.index("USA") if "USA" in countries else 0,
            key="weather_country",
        )
    with city_col:
        default_city = CITY_SUGGESTIONS.get(country_code, "")
        city = st.text_input(
            "City",
            value=default_city,
            placeholder="e.g., Moscow, Kyiv, Tehran",
            key="weather_city",
        )

    if not city:
        empty_state("Enter a city to see weather.", "Then click Refresh.")
        return

    with st.spinner(f"Fetching weather for {city}, {country_code}..."):
        current = _current(city, country_code)
        forecast = _forecast(city, country_code)

    if not current:
        st.error(
            f"Could not fetch weather for **{city}, {country_code}**. "
            f"Check the city name or try a different one."
        )
        return

    # ── Current weather ──
    st.markdown(f"### {current['city']}, {current['country']}")
    main = current.get("main", {})
    weather = current.get("weather", {})
    wind = current.get("wind", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "Temperature", f"{main.get('temp', '—'):.0f}°C" if main.get("temp") is not None else "—"
        )
    with c2:
        st.metric(
            "Feels like",
            f"{main.get('feels_like', '—'):.0f}°C" if main.get("feels_like") is not None else "—",
        )
    with c3:
        st.metric(
            "Humidity", f"{main.get('humidity', '—')}%" if main.get("humidity") is not None else "—"
        )
    with c4:
        st.metric(
            "Wind", f"{wind.get('speed', '—'):.1f} m/s" if wind.get("speed") is not None else "—"
        )
    with c5:
        st.metric("Conditions", weather.get("main", "—"))

    st.caption(
        f"Last updated: {fmt_dt(datetime.fromtimestamp(current['dt'], tz=timezone.utc)) if current.get('dt') else '—'}"
    )

    # ── Forecast ──
    if forecast:
        st.markdown("---")
        st.markdown("### 5-day forecast (every 3 hours)")

        df_fc = pd.DataFrame(forecast)
        df_fc["datetime"] = pd.to_datetime(df_fc["dt_txt"])
        df_fc["condition"] = df_fc["weather"].apply(
            lambda w: w.get("main", "—") if isinstance(w, dict) else "—"
        )

        # Temperature chart
        fig = px.line(
            df_fc,
            x="datetime",
            y="temp",
            color="condition",
            markers=True,
            title="Temperature forecast",
        )
        fig.update_layout(height=320, margin={"l": 0, "r": 0, "t": 30, "b": 0})
        st.plotly_chart(fig, use_container_width=True)

        # Precipitation + wind
        c1, c2 = st.columns(2)
        with c1:
            if "rain_3h" in df_fc.columns and df_fc["rain_3h"].sum() > 0:
                fig_p = px.bar(df_fc, x="datetime", y="rain_3h", title="Precipitation (mm / 3h)")
                fig_p.update_layout(height=240, margin={"l": 0, "r": 0, "t": 30, "b": 0})
                st.plotly_chart(fig_p, use_container_width=True)
        with c2:
            if "wind_speed" in df_fc.columns:
                fig_w = px.line(df_fc, x="datetime", y="wind_speed", title="Wind speed (m/s)")
                fig_w.update_layout(height=240, margin={"l": 0, "r": 0, "t": 30, "b": 0})
                st.plotly_chart(fig_w, use_container_width=True)

    # ── Conflict correlation ──
    st.markdown("---")
    st.markdown("### Conflict correlation")
    events = _country_recent_events(country_code, days=30)
    if not events:
        st.caption(f"No recent events in {country_code}.")
    else:
        n = len(events)
        avg_tone = sum(e["tone"] or 0 for e in events) / n if n else 0
        roots = Counter(e["root"] for e in events if e["root"])
        top_root = roots.most_common(1)[0] if roots else "—"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Recent events (30d)", n)
        with c2:
            st.metric("Avg tone", f"{avg_tone:+.2f}")
        with c3:
            st.metric("Top event type", f"{top_root} ({roots[top_root]})")

        # Weather risk indicator
        if main.get("temp") is not None:
            temp = main["temp"]
            if temp < -10:
                risk = "❄️ Cold stress: very low temps may correlate with civil unrest"
                risk_color = "info"
            elif temp > 35:
                risk = "🔥 Heat wave: extreme temps may correlate with unrest"
                risk_color = "warning"
            elif 5 < temp < 18:
                risk = "🌡️ Moderate temps: standard risk baseline"
                risk_color = "info"
            else:
                risk = "✅ Comfortable temps: minimal weather-driven unrest risk"
                risk_color = "success"
            getattr(st, risk_color)(risk)


render()
