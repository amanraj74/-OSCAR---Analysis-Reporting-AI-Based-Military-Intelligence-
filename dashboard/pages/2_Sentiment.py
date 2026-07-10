"""Sentiment — Time-series trends + top positive/negative articles."""

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
    empty_state,
    get_articles_dataframe,
    get_events_dataframe,
    get_sentiment_for_articles,
    page_header,
    sentiment_color,
    sidebar_status_panel,
)

st.set_page_config(page_title="OSCAR · Sentiment", page_icon="💬", layout="wide")
sidebar_status_panel()


def _render_article(row) -> None:
    """Render one article row in the sentiment top-lists."""
    color = sentiment_color(row["score"])
    source = row["source"] if row["source"] else "unknown"
    title = row["title"]
    if len(title) > 90:
        title = title[:90] + "…"
    st.markdown(
        f'<div style="padding: 0.5rem 0; border-bottom: 1px solid var(--border);">'
        f'<a href="{row["url"]}" target="_blank" '
        f'style="color: var(--text); text-decoration: none; font-weight: 500;">'
        f"{title}</a>"
        f'<div style="display: flex; gap: 0.75rem; margin-top: 0.25rem; '
        f'font-size: 0.78rem; color: var(--text-mute);">'
        f'<span style="color: {color}; font-weight: 600;">{row["score"]:+.2f}</span>'
        f"<span>·</span><span>{source}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


page_header(
    "Sentiment",
    "Sentiment trends across articles + GDELT average tone.",
)

with st.sidebar:
    st.markdown("### Filters")
    days = st.slider("Window (days)", 1, 90, 30, key="sent_days")

articles_df = get_articles_dataframe(days=days, limit=2000)
events_df = get_events_dataframe(days=days)

if articles_df.empty and events_df.empty:
    empty_state("No data ingested for this window yet.")
    st.stop()

# ── Article sentiment over time ─────────────────────────
st.markdown("### Sentiment over time · Articles")

if not articles_df.empty:
    article_ids = tuple(int(x) for x in articles_df["id"].tolist())
    sentiment_df = get_sentiment_for_articles(article_ids)
    if not sentiment_df.empty:
        merged = articles_df[["id", "published_at", "title", "url", "source"]].merge(
            sentiment_df, left_on="id", right_on="source_id", how="inner"
        )
        merged["date"] = pd.to_datetime(merged["published_at"]).dt.date
        daily = (
            merged.groupby("date")
            .agg(
                mean_score=("score", "mean"),
                median_score=("score", "median"),
                positive_share=("label", lambda s: float((s == "positive").mean())),
                negative_share=("label", lambda s: float((s == "negative").mean())),
                n=("id", "count"),
            )
            .reset_index()
        )

        if daily.empty:
            empty_state(
                "No sentiment scored in this window.",
                "Run `python -m src.nlp.cli sentiment-score` first.",
            )
        else:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("#### Mean & median score")
                fig = px.line(
                    daily,
                    x="date",
                    y=["mean_score", "median_score"],
                    labels={"value": "Avg compound (-1..+1)", "variable": ""},
                )
                fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1)
                fig.update_layout(
                    height=320,
                    margin={"l": 0, "r": 0, "t": 10, "b": 0},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend={"orientation": "h", "y": -0.15, "yanchor": "top"},
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("#### Positive vs Negative share")
                fig2 = px.area(
                    daily,
                    x="date",
                    y=["positive_share", "negative_share"],
                    labels={"value": "Share of articles", "variable": ""},
                    color_discrete_map={"positive_share": "#10b981", "negative_share": "#ef4444"},
                )
                fig2.update_layout(
                    height=320,
                    margin={"l": 0, "r": 0, "t": 10, "b": 0},
                    yaxis_tickformat=".0%",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend={"orientation": "h", "y": -0.15, "yanchor": "top"},
                    hovermode="x unified",
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")

            # Top articles grid
            st.markdown("### Top articles")

            top_n = st.slider("Show top N per category", 3, 15, 8, key="sent_topn")
            c1, c2 = st.columns(2, gap="large")

            with c1:
                st.markdown("#### Most positive")
                positives = merged.nlargest(top_n, "score")[
                    ["title", "score", "url", "published_at", "source"]
                ]
                for _, row in positives.iterrows():
                    _render_article(row)

            with c2:
                st.markdown("#### Most negative")
                negatives = merged.nsmallest(top_n, "score")[
                    ["title", "score", "url", "published_at", "source"]
                ]
                for _, row in negatives.iterrows():
                    _render_article(row)
    else:
        empty_state(
            "No sentiment scores yet.",
            "Run `python -m src.nlp.cli sentiment-score` to score articles.",
        )
else:
    empty_state("No articles ingested in this window.")

# ── GDELT tone ──────────────────────────────────────────
st.markdown("---")
st.markdown("### GDELT average tone")

if not events_df.empty and "avg_tone" in events_df.columns:
    events_df["date"] = pd.to_datetime(events_df["sql_date"], format="%Y%m%d", errors="coerce")
    tone = (
        events_df.dropna(subset=["avg_tone", "date"])
        .groupby("date")
        .agg(mean_tone=("avg_tone", "mean"), n=("global_event_id", "count"))
        .reset_index()
    )
    if tone.empty:
        empty_state("No GDELT events with avg_tone.")
    else:
        fig3 = go.Figure(
            go.Scatter(
                x=tone["date"],
                y=tone["mean_tone"],
                mode="lines+markers",
                marker={
                    "size": tone["n"],
                    "sizemode": "area",
                    "sizeref": 2.0 * max(tone["n"]) / (40.0**2),
                    "color": "#3b82f6",
                },
                line={"color": "#3b82f6", "width": 2},
            )
        )
        fig3.update_layout(
            height=320,
            yaxis_title="Avg tone (-100 .. +100)",
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig3.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1)
        st.plotly_chart(fig3, use_container_width=True)
else:
    empty_state("No GDELT events to plot.")
