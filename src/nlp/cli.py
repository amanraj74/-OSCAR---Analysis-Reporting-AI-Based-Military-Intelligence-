"""CLI entry-point for OSCAR NLP commands.

Usage:
    python -m src.nlp.cli ner-process --source articles
    python -m src.nlp.cli sentiment-score --source articles
    python -m src.nlp.cli topics-discover --source articles --n-topics 8
    python -m src.nlp.cli entities-normalize
    python -m src.nlp.cli nlp-process --source articles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
except ImportError:
    pass

from sqlalchemy import select

from src.config import get_settings
from src.observability import configure_logging, get_logger
from src.persistence.database import session_scope
from src.persistence.models import Article, Entity


def _fetch_articles(limit: int | None = None) -> list[Article]:
    with session_scope() as session:
        stmt = select(Article).order_by(Article.id)
        if limit:
            stmt = stmt.limit(limit)
        return list(session.execute(stmt).scalars().all())


def cmd_ner_process(args: argparse.Namespace) -> int:
    logger = get_logger("nlp.cli")
    from src.nlp.ner import NerPipeline

    pipeline = NerPipeline()
    articles = _fetch_articles(limit=args.limit)
    logger.info("ner_start", count=len(articles), mode=pipeline.mode)

    total = 0
    for art in articles:
        text = " ".join(filter(None, [art.title, art.description, art.content]))
        n = pipeline.extract_and_persist("article", art.id, text)
        total += n
        logger.info("ner_extracted", article_id=art.id, entities=n)

    print(
        f"[ner] processed {len(articles)} articles, extracted {total} entity mentions (mode={pipeline.mode})"
    )
    return 0


def cmd_sentiment_score(args: argparse.Namespace) -> int:
    logger = get_logger("nlp.cli")
    from src.nlp.sentiment import SentimentScorer

    scorer = SentimentScorer()
    articles = _fetch_articles(limit=args.limit)
    logger.info("sentiment_start", count=len(articles), mode=scorer.mode)

    scored = 0
    for art in articles:
        text = " ".join(filter(None, [art.title, art.description, art.content]))
        scorer.score_and_persist("article", art.id, text)
        scored += 1

    print(f"[sentiment] scored {scored} articles (mode={scorer.mode})")
    return 0


def cmd_topics_discover(args: argparse.Namespace) -> int:
    logger = get_logger("nlp.cli")
    from src.nlp.topics import TopicDiscoverer

    discoverer = TopicDiscoverer(n_topics=args.n_topics)
    articles = _fetch_articles(limit=args.limit)
    docs = [" ".join(filter(None, [a.title, a.description, a.content])) for a in articles]
    docs = [d for d in docs if d.strip()]
    logger.info("topics_start", count=len(docs), mode=discoverer.mode)

    n = discoverer.fit_and_persist(docs, min_article_count=args.min_articles)
    print(f"[topics] discovered {n} themes (mode={discoverer.mode})")
    return 0


def cmd_entities_normalize(args: argparse.Namespace) -> int:
    logger = get_logger("nlp.cli")
    from src.nlp.normalize import EntityNormalizer

    normalizer = EntityNormalizer(threshold=args.threshold)
    with session_scope() as session:
        entities = list(session.execute(select(Entity)).scalars().all())
    aliases = normalizer.compute_aliases(entities, threshold=args.threshold)
    logger.info("entities_aliases_found", count=len(aliases))
    n_updated = normalizer.update_canonical_names(threshold=args.threshold)
    print(
        f"[normalize] found {len(aliases)} aliases, updated {n_updated} canonical names (mode={normalizer.mode})"
    )
    return 0


def cmd_nlp_process(args: argparse.Namespace) -> int:
    """Run all NLP stages in sequence."""
    for sub in (cmd_ner_process, cmd_sentiment_score, cmd_topics_discover, cmd_entities_normalize):
        rc = sub(args)
        if rc != 0:
            return rc
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oscar-nlp", description="OSCAR NLP CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--limit", type=int, help="Limit rows processed.")

    ner = sub.add_parser("ner-process", parents=[common], help="Extract entities from articles.")
    ner.set_defaults(func=cmd_ner_process)

    sentiment = sub.add_parser(
        "sentiment-score", parents=[common], help="Score sentiment of articles."
    )
    sentiment.set_defaults(func=cmd_sentiment_score)

    topics = sub.add_parser("topics-discover", parents=[common], help="Discover themes.")
    topics.add_argument("--n-topics", type=int, default=8, help="Number of topics (default: 8).")
    topics.add_argument("--min-articles", type=int, default=1, help="Min article count to persist.")
    topics.set_defaults(func=cmd_topics_discover)

    entities = sub.add_parser("entities-normalize", help="Merge entity aliases.")
    entities.add_argument(
        "--threshold", type=float, default=0.8, help="Similarity threshold (default: 0.8)."
    )
    entities.set_defaults(func=cmd_entities_normalize)

    all_cmd = sub.add_parser(
        "nlp-process",
        parents=[common],
        help="Run all NLP stages.",
    )
    all_cmd.set_defaults(func=cmd_nlp_process)

    return parser


def main() -> int:
    cfg = get_settings()
    configure_logging(level=cfg.log_level, json_format=False)
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
