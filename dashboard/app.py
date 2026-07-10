"""OSCAR Streamlit dashboard — landing page."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        env_path = ROOT_DIR.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
    except ImportError:
        pass


_load_env()

from dashboard.utils import fmt_int, get_overview_metrics, sidebar_status_panel
from src import __version__
from src.config import get_settings
from src.persistence.database import init_schema

ROOT_DIR = Path(__file__).resolve().parent
CSS_PATH = ROOT_DIR / "assets" / "style.css"


def _load_css() -> None:
    if CSS_PATH.exists():
        st.markdown(
            f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True
        )


def main() -> None:
    st.set_page_config(
        page_title="OSCAR · OSINT Dashboard",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _load_css()
    cfg = get_settings()
    init_schema()
    sidebar_status_panel()

    metrics = get_overview_metrics()

    # Hero
    st.markdown(
        """
        <div style="padding: 2rem 0 1rem 0; border-bottom: 1px solid var(--border); margin-bottom: 2rem;">
            <p style="font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
                    color: var(--accent); margin-bottom: 0.5rem;">BSERC Def-Space · Summer Internship 2026</p>
            <h1 style="margin-bottom: 0.5rem;">OSCAR — Threat Intelligence Dashboard</h1>
            <p style="color: var(--text-mute); font-size: 1.05rem; margin: 0;">
                Open-Source Conflict Analysis &amp; Reporting.
                Real-time OSINT signals from GDELT, news, and public social feeds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI strip
    st.markdown(
        '<p style="font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; '
        'color: var(--text-mute); margin-bottom: 0.75rem;">Pipeline status</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GDELT events", fmt_int(metrics["events"]), help="Global events ingested")
    c2.metric("Articles", fmt_int(metrics["articles"]), help="News + Reddit posts")
    c3.metric("Entities", fmt_int(metrics["entities"]), help="Named entities extracted")
    c4.metric("Anomalies", fmt_int(metrics["anomalies"]), help="Active anomaly flags", delta=None)

    st.markdown("")

    # Two-column overview
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("### About")
        st.markdown("""
OSCAR fuses **three open data sources** with **NLP** and **machine learning**
to surface defense-relevant signals across 50+ countries:

| Source | Type | Cadence |
|---|---|---|
| GDELT Project 2.0 | Global event database | 15-min refresh |
| NewsAPI.org | News headlines + search | On-demand |
| Reddit RSS | Public subreddit feeds | On-demand |

All processing is **local**, **open-source**, and **reproducible**.
            """)

        st.markdown("")
        st.markdown("### Pages")
        st.markdown("""
- **Map** — World choropleth of sentiment + geo-located events
- **Sentiment** — Time-series trends + top positive / negative articles
- **Entities** — Trending orgs, weapons, and locations
- **Forecast** — 7-day risk forecasts per region with confidence bands
- **Alerts** — Live anomaly feed with severity scoring
            """)

    with right:
        st.markdown("### System")
        st.json(
            {
                "version": __version__,
                "environment": cfg.app_env,
                "log_level": cfg.log_level,
                "database": (
                    cfg.database_url.split(":///")[-1] if ":///" in cfg.database_url else "—"
                ),
                "ingestion_window_hours": cfg.gdelt.batch_hours_back,
                "sentiment_model": cfg.nlp.sentiment_model,
                "spacy_model": cfg.nlp.spacy_model,
                "embedding_model": cfg.nlp.embedding_model,
            }
        )

        st.markdown("### Setup")
        st.code(
            "# Ingest real data\n"
            "python -m src.ingestion.cli refresh --source gdelt\n"
            "python -m src.ingestion.cli refresh --source newsapi\n"
            "python -m src.ingestion.cli refresh --source reddit\n"
            "\n"
            "# Build silver + train\n"
            "python -m src.ingestion.cli transform\n"
            "python -m src.ml.cli ml-train-all",
            language="bash",
        )


if __name__ == "__main__":
    main()
