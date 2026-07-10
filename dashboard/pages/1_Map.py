"""Map — World choropleth of sentiment + event density overlay."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))  # noqa: F401  bootstrap sys.path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils import (
    country_code_to_iso3,
    empty_state,
    get_articles_dataframe,
    get_events_dataframe,
    get_sentiment_for_articles,
    page_header,
    sidebar_status_panel,
)

st.set_page_config(page_title="OSCAR · Map", page_icon="🗺️", layout="wide")
sidebar_status_panel()

page_header(
    "Map",
    "Country-level sentiment choropleth with geo-located GDELT event bubbles.",
)

with st.sidebar:
    st.markdown("### Filters")
    days = st.slider("Window (days)", min_value=1, max_value=90, value=14, key="map_days")
    color_metric = st.radio(
        "Color by",
        ["avg_sentiment", "event_count", "conflict_count"],
        format_func={
            "avg_sentiment": "Sentiment",
            "event_count": "Events",
            "conflict_count": "Conflicts",
        }.get,
        horizontal=False,
    )

events_df = get_events_dataframe(days=days)
articles_df = get_articles_dataframe(days=days, limit=1000)

if events_df.empty and articles_df.empty:
    empty_state(
        f"No data for the last {days} days.",
        "Run ingestors or widen the window using the sidebar.",
    )
    st.stop()

if not articles_df.empty:
    article_ids = tuple(int(x) for x in articles_df["id"].tolist())
    sentiment_df = get_sentiment_for_articles(article_ids)
    if not sentiment_df.empty:
        articles_with_sent = articles_df.merge(
            sentiment_df[["source_id", "score", "label"]],
            left_on="id",
            right_on="source_id",
            how="inner",
        )
    else:
        articles_with_sent = pd.DataFrame()
else:
    articles_with_sent = pd.DataFrame()

if not articles_with_sent.empty:
    articles_with_sent["iso3"] = articles_with_sent["url"].apply(
        lambda u: (
            country_code_to_iso3(u.split(".")[-2].split("/")[-1])
            if isinstance(u, str) and len(u.split(".")[-2].split("/")[-1]) == 2
            else None
        )
    )
    by_country = (
        articles_with_sent.dropna(subset=["iso3"])
        .groupby("iso3")
        .agg(
            article_count=("id", "count"),
            avg_sentiment=("score", "mean"),
        )
        .reset_index()
    )
else:
    by_country = pd.DataFrame(columns=["iso3", "article_count", "avg_sentiment"])

if not events_df.empty:
    events_df["iso3_actor1"] = events_df["actor1_country_code"].apply(
        lambda c: country_code_to_iso3(c) if isinstance(c, str) and len(c) == 2 else None
    )
    event_agg = (
        events_df.dropna(subset=["iso3_actor1"])
        .groupby("iso3_actor1")
        .agg(
            event_count=("global_event_id", "count"),
            conflict_count=(
                "event_root_code",
                lambda s: int(((s >= "14") & (s <= "20")).sum()),
            ),
            avg_goldstein=("goldstein_scale", "mean"),
            avg_tone=("avg_tone", "mean"),
        )
        .reset_index()
        .rename(columns={"iso3_actor1": "iso3"})
    )
else:
    event_agg = pd.DataFrame(columns=["iso3", "event_count", "conflict_count"])

if not by_country.empty and not event_agg.empty:
    combined = by_country.merge(event_agg, on="iso3", how="outer").fillna(0)
elif not by_country.empty:
    combined = by_country.copy()
    combined["event_count"] = 0
    combined["conflict_count"] = 0
elif not event_agg.empty:
    combined = event_agg.copy()
    combined["article_count"] = 0
    combined["avg_sentiment"] = 0.0
else:
    combined = pd.DataFrame()

color_label = {
    "avg_sentiment": "Avg sentiment (-1 to +1)",
    "event_count": "Event count",
    "conflict_count": "Conflict events (CAMEO 14-20)",
}[color_metric]
color_scale = "RdYlGn" if color_metric == "avg_sentiment" else "Reds"
range_color = (-1.0, 1.0) if color_metric == "avg_sentiment" else None

if combined.empty:
    empty_state("No country-level data to plot yet.")
else:
    fig = px.choropleth(
        combined,
        locations="iso3",
        color=color_metric,
        hover_name="iso3",
        hover_data={
            "iso3": True,
            "article_count": True,
            "event_count": True,
            "conflict_count": True,
            "avg_sentiment": ":.3f",
            "avg_tone": ":.3f",
        },
        color_continuous_scale=color_scale,
        range_color=range_color,
        labels={color_metric: color_label},
        title=f"Countries by {color_label} · last {days} days",
    )
    fig.update_geos(
        showcountries=True,
        showcoastlines=True,
        projection_type="natural earth",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        height=520,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "-apple-system, Segoe UI, Inter, sans-serif"},
    )
    st.plotly_chart(fig, use_container_width=True)

# Top countries table
st.markdown("")
st.markdown("### Top countries")
if combined.empty:
    st.caption("No country aggregates yet.")
else:
    rank_metric = st.selectbox(
        "Rank by",
        ["event_count", "conflict_count", "article_count", "avg_sentiment"],
        index=0,
        key="map_rank",
    )
    top = combined.nlargest(10, rank_metric)[
        ["iso3", "event_count", "conflict_count", "article_count", "avg_sentiment"]
    ]
    st.dataframe(
        top.reset_index(drop=True),
        hide_index=True,
        use_container_width=True,
        column_config={
            "iso3": st.column_config.TextColumn("Country"),
            "event_count": st.column_config.NumberColumn("Events", format="%d"),
            "conflict_count": st.column_config.NumberColumn("Conflicts", format="%d"),
            "article_count": st.column_config.NumberColumn("Articles", format="%d"),
            "avg_sentiment": st.column_config.NumberColumn("Avg sentiment", format="%.3f"),
        },
    )

st.markdown("---")
st.markdown("### Geo-located events")

if not events_df.empty and {"action_geo_lat", "action_geo_long"}.issubset(events_df.columns):
    geo = events_df.dropna(subset=["action_geo_lat", "action_geo_long"])
    if geo.empty:
        empty_state("No geo-tagged events in this window.")
    else:
        sample = geo.sample(min(1500, len(geo)), random_state=42) if len(geo) > 1500 else geo
        fig2 = go.Figure(
            go.Scattergeo(
                lon=sample["action_geo_long"],
                lat=sample["action_geo_lat"],
                mode="markers",
                marker={
                    "size": 5,
                    "color": sample["goldstein_scale"],
                    "colorscale": "RdYlGn",
                    "cmin": -10,
                    "cmax": 10,
                    "colorbar": {"title": "Goldstein", "thickness": 12},
                    "opacity": 0.7,
                    "line": {"width": 0},
                },
                text=sample.get("actor1_name", ""),
                hovertemplate="<b>%{text}</b><br>lat %{lat:.2f}<br>lon %{lon:.2f}<extra></extra>",
            )
        )
        fig2.update_geos(
            showcountries=True,
            showcoastlines=True,
            projection_type="natural earth",
            bgcolor="rgba(0,0,0,0)",
        )
        fig2.update_layout(
            height=420,
            margin={"l": 0, "r": 0, "t": 20, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)
else:
    empty_state("No geo-tagged events available.")
