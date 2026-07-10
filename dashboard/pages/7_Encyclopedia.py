"""Encyclopedia — Live Wikipedia context for any entity in the DB.

Browse any entity (Wagner Group, F-16, Tehran, ...) and get:
- Wikipedia summary (lead paragraph)
- Thumbnail (if available)
- 30-day pageview count (public interest proxy)
- Direct link to the full article
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))  # noqa: F401  bootstrap sys.path

import streamlit as st

from dashboard.utils import empty_state, get_top_entities, page_header, sidebar_status_panel

st.set_page_config(page_title="OSCAR · Encyclopedia", page_icon="📚", layout="wide")
sidebar_status_panel()

page_header(
    "Encyclopedia",
    "Live Wikipedia context for any entity in the database. "
    "Choose an entity below to fetch its summary, recent pageviews, "
    "and a direct link to the full article.",
)


@st.cache_data(ttl=3600, show_spinner=False)
def _wiki_summary(name: str) -> dict[str, Any] | None:
    from src.ingestion.wikipedia import WikipediaClient

    return WikipediaClient().get_summary(name)


@st.cache_data(ttl=3600, show_spinner=False)
def _wiki_pageviews(name: str) -> int:
    from src.ingestion.wikipedia import WikipediaClient

    return WikipediaClient().get_pageviews(name, days=30)


@st.cache_data(ttl=3600, show_spinner=False)
def _top_entities(limit: int = 200) -> list[str]:
    df = get_top_entities(limit=limit)
    if df.empty:
        return []
    return df["name"].drop_duplicates().sort_values().tolist()


entity_options = _top_entities(200)
if not entity_options:
    empty_state(
        "No entities in the database yet.",
        "Run `python -m src.nlp.cli ner-process` after ingesting data.",
    )
else:
    col_select, col_metric = st.columns([3, 1], gap="medium")
    with col_select:
        entity_choice = st.selectbox(
            "Choose an entity",
            options=entity_options,
            key="encyclopedia_entity_pick",
        )
    with col_metric:
        st.metric("Entities available", len(entity_options))

    if entity_choice:
        c1, c2 = st.columns([3, 1], gap="large")
        with c1:
            with st.spinner(f"Fetching Wikipedia summary for {entity_choice}..."):
                summary = _wiki_summary(entity_choice)
            if summary is None:
                empty_state(
                    f"No Wikipedia article for '{entity_choice}'.",
                    "Try a different name (e.g., 'Wagner Group' not 'wagner').",
                )
            else:
                title = summary["title"]
                extract = summary["extract"]
                url = summary["url"]
                thumb = summary.get("thumbnail")
                if thumb:
                    st.image(thumb, width=200)
                st.markdown(f"### {title}")
                st.caption("Source: Wikipedia")
                st.markdown(extract if extract else "_(no summary available)_")
                if url:
                    st.markdown(f"[Read full article on Wikipedia]({url})")
        with c2:
            with st.spinner("Fetching 30-day pageviews..."):
                pv = _wiki_pageviews(entity_choice)
            st.metric("Pageviews (last 30 days)", f"{pv:,}")
            st.caption("Higher pageviews = more public interest in this topic.")
            if pv > 50000:
                st.success("Highly watched topic")
            elif pv > 10000:
                st.info("Moderately watched")
            elif pv > 0:
                st.caption("Low activity")
            else:
                st.caption("No recent activity")
