"""Dashboard utility helpers — cached data loaders, shared UI primitives.

All `dashboard/utils.py` consumers get the same:
    - cached data loaders (5 min TTL)
    - safe fallbacks when DB / parquet missing
    - consistent header / sidebar / metric primitives
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src import __version__
from src.config import get_settings
from src.persistence.database import session_scope
from src.persistence.models import Anomaly, Article, Entity, Event, RiskScore, Sentiment, Topic

_TTL = 300


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _empty_events_df() -> pd.DataFrame:
    return pd.DataFrame()


def _empty_articles_df() -> pd.DataFrame:
    return pd.DataFrame()


def _empty_entities_df() -> pd.DataFrame:
    return pd.DataFrame()


def _empty_sentiments_df() -> pd.DataFrame:
    return pd.DataFrame()


def _empty_anomalies_df() -> pd.DataFrame:
    return pd.DataFrame()


def _empty_topics_df() -> pd.DataFrame:
    return pd.DataFrame()


def _empty_overview() -> dict[str, Any]:
    return {
        "events": 0,
        "articles": 0,
        "entities": 0,
        "sentiments": 0,
        "topics": 0,
        "anomalies": 0,
        "risk_scores": 0,
        "latest_event_at": None,
        "latest_article_at": None,
    }


@st.cache_data(ttl=_TTL, show_spinner=False)
def get_overview_metrics() -> dict[str, Any]:
    try:
        with session_scope() as session:
            return {
                "events": session.query(Event).count(),
                "articles": session.query(Article).count(),
                "entities": session.query(Entity).count(),
                "sentiments": session.query(Sentiment).count(),
                "topics": session.query(Topic).count(),
                "anomalies": session.query(Anomaly).count(),
                "risk_scores": session.query(RiskScore).count(),
                "latest_event_at": _latest(Event, "ingested_at"),
                "latest_article_at": _latest(Article, "ingested_at"),
            }
    except Exception:
        return _empty_overview()


def _latest(model: type, col: str) -> datetime | None:
    with session_scope() as session:
        obj = session.query(model).order_by(getattr(model, col).desc()).first()
    return getattr(obj, col, None) if obj else None


@st.cache_data(ttl=_TTL, show_spinner=False)
def get_events_dataframe(days: int = 30) -> pd.DataFrame:
    try:
        cutoff = (_utcnow() - timedelta(days=days)).strftime("%Y%m%d")
        with session_scope() as session:
            rows = session.query(Event).filter(Event.sql_date >= cutoff).all()
        if not rows:
            return _empty_events_df()
        return pd.DataFrame(
            [
                {
                    "global_event_id": r.global_event_id,
                    "sql_date": r.sql_date,
                    "actor1_country_code": r.actor1_country_code,
                    "actor2_country_code": r.actor2_country_code,
                    "event_root_code": r.event_root_code,
                    "goldstein_scale": r.goldstein_scale,
                    "avg_tone": r.avg_tone,
                    "num_articles": r.num_articles,
                    "action_geo_country_code": r.action_geo_country_code,
                    "action_geo_lat": r.action_geo_lat,
                    "action_geo_long": r.action_geo_long,
                }
                for r in rows
            ]
        )
    except Exception:
        return _empty_events_df()


@st.cache_data(ttl=_TTL, show_spinner=False)
def get_articles_dataframe(days: int = 30, limit: int = 500) -> pd.DataFrame:
    try:
        cutoff = _utcnow() - timedelta(days=days)
        with session_scope() as session:
            rows = (
                session.query(Article)
                .filter(Article.published_at >= cutoff)
                .order_by(Article.published_at.desc())
                .limit(limit)
                .all()
            )
        if not rows:
            return _empty_articles_df()
        return pd.DataFrame(
            [
                {
                    "id": r.id,
                    "source": r.source,
                    "title": r.title,
                    "description": r.description,
                    "url": r.url,
                    "author": r.author,
                    "published_at": r.published_at,
                    "language": r.language,
                }
                for r in rows
            ]
        )
    except Exception:
        return _empty_articles_df()


@st.cache_data(ttl=_TTL, show_spinner=False)
def get_sentiment_for_articles(article_ids: tuple[int, ...]) -> pd.DataFrame:
    if not article_ids:
        return _empty_sentiments_df()
    try:
        with session_scope() as session:
            rows = (
                session.query(Sentiment)
                .filter(Sentiment.source_type == "article", Sentiment.source_id.in_(article_ids))
                .all()
            )
        return pd.DataFrame(
            [
                {
                    "source_id": r.source_id,
                    "label": r.label,
                    "score": r.score,
                    "positive": r.positive,
                    "neutral": r.neutral,
                    "negative": r.negative,
                    "model": r.model,
                }
                for r in rows
            ]
        )
    except Exception:
        return _empty_sentiments_df()


@st.cache_data(ttl=_TTL, show_spinner=False)
def get_top_entities(limit: int = 50, entity_type: str | None = None) -> pd.DataFrame:
    try:
        with session_scope() as session:
            q = session.query(Entity)
            if entity_type:
                q = q.filter(Entity.entity_type == entity_type)
            rows = q.order_by(Entity.mention_count.desc()).limit(limit).all()
        return pd.DataFrame(
            [
                {
                    "id": r.id,
                    "name": r.name,
                    "canonical_name": r.canonical_name,
                    "entity_type": r.entity_type,
                    "mention_count": r.mention_count,
                    "first_seen": r.first_seen,
                    "last_seen": r.last_seen,
                }
                for r in rows
            ]
        )
    except Exception:
        return _empty_entities_df()


@st.cache_data(ttl=_TTL, show_spinner=False)
def get_recent_anomalies(limit: int = 50) -> pd.DataFrame:
    try:
        with session_scope() as session:
            rows = session.query(Anomaly).order_by(Anomaly.detected_at.desc()).limit(limit).all()
        if not rows:
            return _empty_anomalies_df()
        return pd.DataFrame(
            [
                {
                    "id": r.id,
                    "region": r.region,
                    "date": r.date,
                    "anomaly_type": r.anomaly_type,
                    "severity": r.severity,
                    "score": r.score,
                    "description": r.description,
                    "context": r.context,
                    "detected_at": r.detected_at,
                }
                for r in rows
            ]
        )
    except Exception:
        return _empty_anomalies_df()


@st.cache_data(ttl=_TTL, show_spinner=False)
def get_topics(n: int = 20) -> pd.DataFrame:
    try:
        with session_scope() as session:
            rows = session.query(Topic).order_by(Topic.article_count.desc()).limit(n).all()
        return pd.DataFrame(
            [
                {
                    "topic_id": r.topic_id,
                    "label": r.label,
                    "keywords": r.keywords,
                    "article_count": r.article_count,
                    "representation": r.representation,
                }
                for r in rows
            ]
        )
    except Exception:
        return _empty_topics_df()


@st.cache_data(ttl=_TTL, show_spinner=False)
def get_silver_events_per_country_day() -> pd.DataFrame:
    try:
        path = get_settings().processed_data_dir / "silver" / "events_per_country_day.parquet"
        if not path.exists():
            return _empty_events_df()
        return pd.read_parquet(path)
    except Exception:
        return _empty_events_df()


@st.cache_data(ttl=_TTL, show_spinner=False)
def get_silver_articles_per_source_day() -> pd.DataFrame:
    try:
        path = get_settings().processed_data_dir / "silver" / "articles_per_source_day.parquet"
        if not path.exists():
            return _empty_articles_df()
        return pd.read_parquet(path)
    except Exception:
        return _empty_articles_df()


@st.cache_data(ttl=_TTL, show_spinner=False)
def get_forecast_for_region(region: str) -> pd.DataFrame:
    try:
        path = get_settings().processed_data_dir / "forecasts" / f"forecast_{region}.parquet"
        if not path.exists():
            return _empty_events_df()
        return pd.read_parquet(path)
    except Exception:
        return _empty_events_df()


@st.cache_data(ttl=_TTL, show_spinner=False)
def get_production_model_metrics() -> dict[str, Any]:
    try:
        from src.ml.registry import Registry

        reg = Registry()
        out: dict[str, Any] = {}
        for name in ["escalation_h1", "escalation_h3", "escalation_h7"]:
            prod = reg.get_production(name)
            out[name] = prod.to_dict() if prod is not None else None
        return out
    except Exception:
        return dict.fromkeys(["escalation_h1", "escalation_h3", "escalation_h7"])


def country_code_to_iso3(alpha2_or_fips: str) -> str:
    if not alpha2_or_fips or len(alpha2_or_fips) != 2:
        return alpha2_or_fips or ""
    try:
        import pycountry

        c = pycountry.countries.get(alpha_2=alpha2_or_fips)
        if c:
            return c.alpha_3
    except Exception:
        pass
    return alpha2_or_fips


def fmt_int(n: int | float | None) -> str:
    if n is None:
        return "—"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n)) if isinstance(n, float) and n.is_integer() else str(n)


def fmt_pct(p: float | None) -> str:
    return "—" if p is None else f"{p * 100:.1f}%"


def fmt_dt(d: datetime | None, default: str = "—") -> str:
    return default if d is None else d.strftime("%Y-%m-%d %H:%M UTC")


def sentiment_color(score: float | None) -> str:
    if score is None:
        return "#9ca3af"
    if score > 0.2:
        return "#16a34a"
    if score < -0.2:
        return "#dc2626"
    return "#f59e0b"


def severity_color(severity: float | None) -> str:
    if severity is None:
        return "#9ca3af"
    if severity >= 0.7:
        return "#dc2626"
    if severity >= 0.4:
        return "#f59e0b"
    return "#3b82f6"


def page_header(title: str, subtitle: str | None = None) -> None:
    """Standard page header. Clean, two-line."""
    st.title(title)
    if subtitle:
        st.markdown(
            f'<p style="color: var(--text-mute); font-size: 0.95rem; margin-top: -0.75rem; margin-bottom: 1.5rem;">{subtitle}</p>',
            unsafe_allow_html=True,
        )
    st.markdown("---")


def sidebar_status_panel() -> None:
    """Sidebar status panel — clean, minimal, no emoji clutter."""
    with st.sidebar:
        st.markdown(
            '<p style="font-size: 1.1rem; font-weight: 600; color: var(--primary); margin-bottom: 0.25rem;">OSCAR</p>'
            f'<p style="font-size: 0.78rem; color: var(--text-mute); margin-bottom: 1rem;">v{__version__} · OSINT Threat Dashboard</p>',
            unsafe_allow_html=True,
        )

        try:
            metrics = get_overview_metrics()
            st.markdown(
                '<p style="font-size: 0.7rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; '
                'color: var(--text-mute); margin-top: 1.25rem; margin-bottom: 0.5rem;">System</p>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Events", fmt_int(metrics["events"]))
                st.metric("Entities", fmt_int(metrics["entities"]))
            with c2:
                st.metric("Articles", fmt_int(metrics["articles"]))
                st.metric("Anomalies", fmt_int(metrics["anomalies"]))
        except Exception:
            pass

        st.markdown("---")

        if st.button("Refresh data", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()


def empty_state(title: str, hint: str | None = None, icon: str | None = None) -> None:
    """Show a friendly empty-state block when there's no data."""
    parts = [f"### {title}"]
    if hint:
        parts.append(hint)
    st.markdown("\n\n".join(parts))


__all__ = [
    "get_overview_metrics",
    "get_events_dataframe",
    "get_articles_dataframe",
    "get_sentiment_for_articles",
    "get_top_entities",
    "get_recent_anomalies",
    "get_topics",
    "get_silver_events_per_country_day",
    "get_silver_articles_per_source_day",
    "get_forecast_for_region",
    "get_production_model_metrics",
    "country_code_to_iso3",
    "fmt_int",
    "fmt_pct",
    "fmt_dt",
    "format_int",  # alias
    "format_pct",  # alias
    "format_dt",  # alias
    "sentiment_color",
    "severity_color",
    "sidebar_status_panel",
    "page_header",
    "empty_state",
]


def format_int(n: int | float | None) -> str:
    """Alias for fmt_int."""
    return fmt_int(n)


def format_pct(p: float | None) -> str:
    """Alias for fmt_pct."""
    return fmt_pct(p)


def format_dt(d: datetime | None, default: str = "—") -> str:
    """Alias for fmt_dt."""
    return fmt_dt(d, default)


def _unused_import_marker() -> str:
    return Path.__name__
