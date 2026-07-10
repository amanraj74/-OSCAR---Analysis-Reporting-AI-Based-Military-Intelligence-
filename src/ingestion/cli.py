"""CLI entry-point for OSCAR ingestion commands.

Usage:
    python -m src.ingestion.cli refresh --source gdelt
    python -m src.ingestion.cli refresh --source newsapi --query "Ukraine"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
except ImportError:
    pass

from src.config import get_settings
from src.ingestion.acled import AcledIngestor
from src.ingestion.gdelt import GdeltIngestor
from src.ingestion.newsapi import NewsApiIngestor
from src.ingestion.reddit import RedditIngestor
from src.observability import configure_logging, get_logger
from src.transform import build_all_silver

_INGESTORS = {
    "acled": AcledIngestor,
    "gdelt": GdeltIngestor,
    "newsapi": NewsApiIngestor,
    "reddit": RedditIngestor,
}


def _build_ingestor(source: str, **kwargs: Any) -> Any:
    if source not in _INGESTORS:
        raise SystemExit(f"Unknown source: {source!r}")
    if source == "reddit":
        subreddits = kwargs.pop("subreddit", None)
        if subreddits:
            kwargs["subreddits"] = subreddits
    return _INGESTORS[source](**kwargs)


def cmd_refresh(args: argparse.Namespace) -> int:
    logger = get_logger("ingestion.cli")

    kwargs: dict[str, Any] = {}
    if args.query:
        kwargs["query"] = args.query
    if args.max_files:
        kwargs["max_files"] = args.max_files

    ingestor = _build_ingestor(args.source, **kwargs)
    result = ingestor.run()

    logger.info(
        "ingestion_summary",
        source=result.source,
        count=result.count,
        success=result.success,
        error=result.error,
        metadata=result.metadata,
    )

    print(
        f"[{result.source}] success={result.success} "
        f"count={result.count} "
        f"elapsed={(result.finished_at - result.started_at).total_seconds():.2f}s"
    )
    return 0 if result.success else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oscar-ingest", description="OSCAR ingestion CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    refresh = sub.add_parser("refresh", help="Run an ingestor end-to-end.")
    refresh.add_argument(
        "--source",
        required=True,
        choices=["acled", "gdelt", "newsapi", "reddit"],
        help="Data source to ingest.",
    )
    refresh.add_argument("--query", help="Optional search query (newsapi).")
    refresh.add_argument("--max-files", type=int, help="Limit files (gdelt).")
    refresh.add_argument("--subreddit", action="append", help="Subreddit to ingest (repeatable).")
    refresh.set_defaults(func=cmd_refresh)

    transform = sub.add_parser("transform", help="Build silver Parquet tables.")
    transform.set_defaults(func=cmd_transform)

    return parser


def cmd_transform(args: argparse.Namespace) -> int:  # noqa: ARG001
    out = build_all_silver()
    for name, path in out.items():
        print(f"[transform] {name}: {path}")
    return 0


def main() -> int:
    cfg = get_settings()
    configure_logging(level=cfg.log_level, json_format=False)
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
