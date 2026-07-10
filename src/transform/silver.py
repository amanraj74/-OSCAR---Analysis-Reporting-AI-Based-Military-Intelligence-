"""Transform layer: convert raw/bronze data to cleaned silver tables.

Reads from `events`, `articles`, and produces aggregated silver tables:
    silver_events_per_country_day
    silver_articles_per_source_day

Silver tables are Parquet files in `data/processed/silver/`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import select

from src.config import get_settings
from src.observability import get_logger
from src.persistence.database import session_scope
from src.persistence.models import Article, Event

logger = get_logger("transform.silver")


def _ensure_silver_dir() -> Path:
    settings = get_settings()
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    silver_dir = settings.processed_data_dir / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)
    return silver_dir


def _read_events(session) -> pd.DataFrame:
    stmt = select(
        Event.sql_date,
        Event.action_geo_country_code,
        Event.actor1_country_code,
        Event.actor2_country_code,
        Event.event_root_code,
        Event.goldstein_scale,
        Event.num_mentions,
        Event.num_articles,
        Event.avg_tone,
    )
    df = pd.read_sql_query(stmt, session.bind)
    df["sql_date"] = pd.to_datetime(df["sql_date"], format="%Y%m%d", errors="coerce")
    df["action_geo_country_code"] = df["action_geo_country_code"].fillna("UNK")
    df["actor1_country_code"] = df["actor1_country_code"].fillna("UNK")
    df["actor2_country_code"] = df["actor2_country_code"].fillna("UNK")
    return df.dropna(subset=["sql_date"])


def _read_articles(session) -> pd.DataFrame:
    stmt = select(
        Article.source,
        Article.published_at,
        Article.title,
        Article.language,
    )
    df = pd.read_sql_query(stmt, session.bind)
    df["published_date"] = pd.to_datetime(df["published_at"]).dt.date
    return df


def build_events_silver(session) -> Path:
    """Aggregate events to (date, country) silver table."""
    silver_dir = _ensure_silver_dir()
    out_path = silver_dir / "events_per_country_day.parquet"

    df = _read_events(session)
    if df.empty:
        df.to_parquet(out_path)
        logger.info("silver_events_empty", path=str(out_path))
        return out_path

    def _conflict_count(series: pd.Series) -> int:
        return int(((series >= "14") & (series <= "20")).sum())

    out = (
        df.groupby(["sql_date", "action_geo_country_code"])
        .agg(
            event_count=("event_root_code", "count"),
            avg_goldstein=("goldstein_scale", "mean"),
            avg_tone=("avg_tone", "mean"),
            total_mentions=("num_mentions", "sum"),
            total_articles=("num_articles", "sum"),
            conflict_count=("event_root_code", _conflict_count),
        )
        .reset_index()
        .sort_values("sql_date")
    )
    out = out.rename(columns={"sql_date": "date", "action_geo_country_code": "actor1_country_code"})
    out["conflict_ratio"] = (out["conflict_count"] / out["event_count"]).fillna(0.0)

    out.to_parquet(out_path, index=False)
    logger.info(
        "silver_events_built",
        rows=len(out),
        path=str(out_path),
        countries=out["actor1_country_code"].nunique(),
    )
    return out_path


def build_articles_silver(session) -> Path:
    """Aggregate articles to (date, source) silver table."""
    silver_dir = _ensure_silver_dir()
    out_path = silver_dir / "articles_per_source_day.parquet"

    df = _read_articles(session)
    if df.empty:
        df.to_parquet(out_path)
        logger.info("silver_articles_empty", path=str(out_path))
        return out_path

    out = (
        df.groupby(["published_date", "source"])
        .agg(article_count=("title", "count"))
        .reset_index()
        .sort_values("published_date")
    )
    out.to_parquet(out_path, index=False)
    logger.info("silver_articles_built", rows=len(out), path=str(out_path))
    return out_path


def build_all_silver() -> dict[str, Path]:
    """Build all silver tables. Returns mapping of table name → parquet path."""
    with session_scope() as session:
        events_path = build_events_silver(session)
        articles_path = build_articles_silver(session)

    return {
        "events_per_country_day": events_path,
        "articles_per_source_day": articles_path,
    }


__all__ = [
    "build_all_silver",
    "build_events_silver",
    "build_articles_silver",
]
