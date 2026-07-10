"""Entities — Trending actors, weapons, locations + co-occurrence."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))  # noqa: F401  bootstrap sys.path
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.utils import (
    empty_state,
    get_recent_anomalies,
    get_top_entities,
    page_header,
    sidebar_status_panel,
)

st.set_page_config(page_title="OSCAR · Entities", page_icon="🏷️", layout="wide")
sidebar_status_panel()

page_header(
    "Entities",
    "Most-mentioned orgs, weapons, locations, persons — driven by NLP extraction.",
)

with st.sidebar:
    st.markdown("### Filters")
    limit = st.slider("Top N per type", 10, 100, 30, key="entities_limit")

tabs = st.tabs(["Organizations", "Weapons", "Locations", "Persons", "Other"])

ORG_TYPES = ("ORG", "MILITARY_ORG")
LOCATION_TYPES = ("GPE", "LOC")
OTHER_TYPES = ("NORP", "MISC")

ORG_TYPES = ("ORG", "MILITARY_ORG")
LOCATION_TYPES = ("GPE", "LOC")
OTHER_TYPES = ("NORP", "MISC")


def _render_table(entity_type: str | tuple[str, ...], label: str) -> None:
    if isinstance(entity_type, tuple):
        df = pd.concat(
            [get_top_entities(limit=limit, entity_type=t) for t in entity_type],
            ignore_index=True,
        )
    else:
        df = get_top_entities(limit=limit, entity_type=entity_type)

    if df.empty:
        empty_state(
            f"No {label} extracted yet.",
            "Run `python -m src.nlp.cli ner-process` to populate.",
        )
        return

    df = df.sort_values("mention_count", ascending=False).head(limit)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.dataframe(
            df[["name", "canonical_name", "mention_count", "last_seen"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Name"),
                "canonical_name": st.column_config.TextColumn("Canonical"),
                "mention_count": st.column_config.ProgressColumn(
                    "Mentions",
                    min_value=0,
                    max_value=int(df["mention_count"].max()),
                ),
                "last_seen": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
            },
        )
    with c2:
        st.markdown(f"#### Top 10 {label}")
        top10 = df.head(10).sort_values("mention_count")
        fig = px.bar(
            top10,
            x="mention_count",
            y="name",
            orientation="h",
            labels={"mention_count": "Mentions", "name": ""},
        )
        fig.update_layout(
            yaxis={"autorange": "reversed", "categoryorder": "total ascending"},
            height=420,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


with tabs[0]:
    _render_table(ORG_TYPES, "Organizations")

with tabs[1]:
    _render_table("WEAPON", "Weapons")

with tabs[2]:
    _render_table(LOCATION_TYPES, "Locations")

with tabs[3]:
    _render_table("PERSON", "Persons")

with tabs[4]:
    _render_table(OTHER_TYPES, "Other")

# ── Co-occurrence ───────────────────────────────────────
st.markdown("---")
st.markdown("### Co-occurrence (lightweight)")
df_all = get_top_entities(limit=500)
if df_all.empty:
    empty_state("No entities available yet.")
else:
    st.caption("Top 20 entities by mention count, deduped by canonical_name.")
    top20 = df_all.drop_duplicates(subset=["canonical_name"]).head(20)
    if top20.empty:
        empty_state("Not enough entities.")
    else:
        pivot = pd.crosstab(top20["entity_type"], top20["canonical_name"]).reindex(
            columns=top20.sort_values("mention_count", ascending=False)["canonical_name"]
        )
        st.dataframe(pivot, use_container_width=True)

# ── Anomalies referencing entities ────────────────────────
st.markdown("---")
st.markdown("### Recent anomalies")
anoms = get_recent_anomalies(limit=20)
if anoms.empty:
    empty_state("No anomalies detected yet.")
else:
    st.dataframe(
        anoms[["region", "date", "anomaly_type", "severity", "description", "detected_at"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "severity": st.column_config.ProgressColumn(
                "Severity", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "detected_at": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        },
    )
