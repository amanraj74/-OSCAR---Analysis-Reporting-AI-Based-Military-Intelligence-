"""Home — landing dashboard with status metrics."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))  # noqa: F401  bootstrap sys.path
import streamlit as st

from dashboard.utils import fmt_dt, fmt_int, get_overview_metrics, page_header, sidebar_status_panel

st.set_page_config(page_title="OSCAR · Home", page_icon="🏠", layout="wide")
sidebar_status_panel()

page_header(
    "Home",
    "Live status of the data pipeline. Refresh from the sidebar to clear cache.",
)

metrics = get_overview_metrics()

# ── KPI strip ──────────────────────────────────────────────
st.markdown(
    '<p style="font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; '
    'text-transform: uppercase; color: var(--text-mute); margin-bottom: 0.5rem;">'
    "Data pipeline</p>",
    unsafe_allow_html=True,
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Articles ingested", fmt_int(metrics["articles"]), help="News + Reddit posts")
c2.metric("GDELT events", fmt_int(metrics["events"]), help="Global events ingested")
c3.metric(
    "Entities extracted",
    fmt_int(metrics["entities"]),
    help="ORG / GPE / PERSON / WEAPON / MILITARY_ORG",
)
c4.metric("Anomalies", fmt_int(metrics["anomalies"]), help="Active anomaly flags")

st.markdown("")

# ── Activity / Setup split ──────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("### Recent activity")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.metric(
            "Last GDELT event ingested",
            fmt_dt(metrics["latest_event_at"], default="No data yet"),
        )
    with rc2:
        st.metric(
            "Last article ingested",
            fmt_dt(metrics["latest_article_at"], default="No data yet"),
        )

    st.markdown("---")

    st.markdown("### Quick navigation")
    st.markdown("""
| Page | What it shows |
|------|---------------|
| **Map** | Choropleth of sentiment + geo-located GDELT events |
| **Sentiment** | Time-series trends + top positive / negative articles |
| **Entities** | Trending orgs / weapons / locations + co-occurrence |
| **Forecast** | 7-day risk forecasts per region with confidence bands |
| **Alerts** | Live anomaly feed with severity scoring |
| **About** | Methodology, data sources, ethics |
        """)

    if metrics["events"] == 0 and metrics["articles"] == 0:
        st.markdown("---")
        st.info(
            "**No data ingested yet.** Run the ingestion pipeline to populate the dashboard:",
            icon=None,
        )
        st.code(
            "# 1. Pull data\n"
            "python -m src.ingestion.cli refresh --source gdelt\n"
            "python -m src.ingestion.cli refresh --source newsapi\n"
            "python -m src.ingestion.cli refresh --source reddit\n"
            "\n"
            "# 2. Build silver + run NLP + ML\n"
            "python -m src.ingestion.cli transform\n"
            "python -m src.nlp.cli nlp-process\n"
            "python -m src.ml.cli ml-train-all",
            language="bash",
        )

with right:
    st.markdown("### Pipeline health")
    st.json(
        {
            "events": metrics["events"],
            "articles": metrics["articles"],
            "entities": metrics["entities"],
            "sentiments": metrics["sentiments"],
            "topics": metrics["topics"],
            "anomalies": metrics["anomalies"],
            "risk_scores": metrics["risk_scores"],
            "latest_event_at": (
                str(metrics["latest_event_at"]) if metrics["latest_event_at"] else None
            ),
            "latest_article_at": (
                str(metrics["latest_article_at"]) if metrics["latest_article_at"] else None
            ),
        }
    )
