"""Seed OSCAR with a realistic 30-day demo dataset.

Generates synthetic but plausible:
    - GDELT events across 12 countries with realistic event codes
    - News + Reddit articles about the same events
    - NER entities (weapons, orgs, locations)
    - Sentiment scores (positive/negative distribution)
    - Forecasts (linear trend)
    - Anomalies (random spikes)

This is what `python scripts/seed_demo.py` produces — enough for the
dashboard to render all pages meaningfully.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from src.config import get_settings
from src.observability import configure_logging, get_logger
from src.persistence.database import init_schema, session_scope
from src.persistence.models import Anomaly, Article, Entity, EntityMention, Event, Sentiment, Topic

SAMPLE_COUNTRIES = [
    "USA",
    "RUS",
    "UKR",
    "ISR",
    "IRN",
    "CHN",
    "TWN",
    "PRK",
    "SYR",
    "YEM",
    "SDN",
    "MMR",
]
SAMPLE_ORGS = ["Wagner Group", "IDF", "NATO", "Hamas", "Hezbollah", "Houthis", "Taliban"]
SAMPLE_WEAPONS = ["F-16", "Su-35", "ATACMS", "HIMARS", "Bayraktar TB2", "Javelin", "Iron Dome"]
SAMPLE_LOCATIONS = ["Kyiv", "Moscow", "Tel Aviv", "Tehran", "Gaza", "Donetsk", "Kharkiv", "Sanaa"]
SAMPLE_DOMAINS = ["Reuters", "AP", "BBC", "CNN", "Al Jazeera", "Bloomberg", "Washington Post"]
SAMPLE_SUBREDDITS = ["worldnews", "geopolitics", "ukraine"]


def _random_event_code(rng: random.Random) -> tuple[str, str]:
    """Return (CAMEO_code, CAMEO_root_code) with realistic distribution."""
    buckets: list[tuple[str, str, int]] = [
        ("010", "01", 5),
        ("020", "02", 8),
        ("040", "04", 10),
        ("080", "08", 5),
        ("100", "10", 8),
        ("130", "13", 12),
        ("140", "14", 10),
        ("150", "15", 8),
        ("170", "17", 7),
        ("180", "18", 8),
        ("190", "19", 12),
        ("200", "20", 7),
    ]
    codes = []
    weights = []
    for c, r, w in buckets:
        codes.append((c, r))
        weights.append(w)
    return rng.choices(codes, weights=weights, k=1)[0]


def _seed_events(rng: random.Random, days: int = 30) -> list[Event]:
    """Generate GDELT-style events for the last N days."""
    end = datetime.now(timezone.utc)
    end - timedelta(days=days)
    out: list[Event] = []
    event_id_base = rng.randint(100_000_000, 999_999_999)

    for day_offset in range(days):
        d = end - timedelta(days=day_offset)
        for country in SAMPLE_COUNTRIES:
            base_count = rng.randint(2, 12)
            for _ in range(base_count):
                code, root = _random_event_code(rng)
                num_mentions = rng.randint(1, 30)
                num_articles = max(1, num_mentions // rng.randint(2, 5))
                ev = Event(
                    global_event_id=event_id_base,
                    sql_date=d.strftime("%Y%m%d"),
                    year=d.year,
                    actor1_name=country,
                    actor1_country_code=country,
                    actor2_name=rng.choice(SAMPLE_COUNTRIES),
                    actor2_country_code=None,
                    event_code=code,
                    event_root_code=root,
                    goldstein_scale=rng.uniform(-9, 5),
                    num_mentions=num_mentions,
                    num_articles=num_articles,
                    avg_tone=rng.uniform(-8, 5),
                    action_geo_fullname=f"{rng.choice(['Capital', 'City', 'Region'])}, {country}",
                    action_geo_country_code=country,
                    action_geo_lat=rng.uniform(-50, 70),
                    action_geo_long=rng.uniform(-180, 180),
                    source_url="https://www.gdeltproject.org/",
                )
                out.append(ev)
                event_id_base += rng.randint(1, 50)
    return out


def _seed_articles(rng: random.Random, events: list[Event], max_n: int = 200) -> list[Article]:
    """Generate articles referencing the events."""
    out: list[Article] = []
    titles_pos = [
        "Ceasefire holds in {loc}",
        "Aid reaches civilians in {loc}",
        "Diplomatic talks resume between {c1} and {c2}",
        "{c1} announces humanitarian corridor",
    ]
    titles_neg = [
        "Escalation feared as {c1} strikes {loc}",
        "{c1} launches major offensive near {loc}",
        "Dozens killed in {loc} attack",
        "{c1} condemns {c2} for {loc} violence",
    ]
    titles_neutral = [
        "Analysis: {c1} policy on {loc} unchanged",
        "Press briefing at {c1} ministry",
        "{c1} delegation visits {loc}",
    ]

    sample = rng.sample(events, min(len(events), max_n))
    for i, ev in enumerate(sample):
        is_pos = rng.random() < 0.3
        is_neg = rng.random() < 0.55
        if is_pos and not is_neg:
            title = rng.choice(titles_pos).format(
                c1=ev.actor1_name,
                c2=ev.actor2_name or "neighbor",
                loc=ev.action_geo_fullname.split(",")[0],
            )
            rng.uniform(0.2, 0.8)
        elif is_neg and not is_pos:
            title = rng.choice(titles_neg).format(
                c1=ev.actor1_name,
                c2=ev.actor2_name or "neighbor",
                loc=ev.action_geo_fullname.split(",")[0],
            )
            rng.uniform(-0.8, -0.2)
        else:
            title = rng.choice(titles_neutral).format(
                c1=ev.actor1_name, loc=ev.action_geo_fullname.split(",")[0]
            )
            rng.uniform(-0.1, 0.1)

        article = Article(
            external_id=f"demo-{i:04d}",
            source=rng.choice(SAMPLE_DOMAINS + SAMPLE_SUBREDDITS),
            title=title,
            description=f"Demo article about {ev.actor1_name} activity.",
            content=title + ". " * 5,
            url=f"https://example.com/article/{i}",
            author=rng.choice(["Reporter A", "Reporter B", "Anonymous", None]),
            image_url=None,
            language="en",
            published_at=ev.ingested_at,
        )
        out.append(article)
    return out


def _seed_entities_and_mentions(
    rng: random.Random, articles: list[Article]
) -> tuple[list[Entity], list[EntityMention]]:
    """Generate entities (orgs, weapons, locations) and link them to articles.

    Returns (entities, mentions) with mentions properly linked to entities
    via stable per-(article, etype, name) keys.
    """
    entities: list[Entity] = []
    entity_by_key: dict[tuple[str, str], Entity] = {}

    def _add_or_get(name: str, etype: str) -> Entity:
        canonical = name.lower().strip()
        k = (canonical, etype)
        if k in entity_by_key:
            return entity_by_key[k]
        e = Entity(
            name=name,
            canonical_name=canonical,
            entity_type=etype,
            mention_count=0,
        )
        entities.append(e)
        entity_by_key[k] = e
        return e

    mentions: list[EntityMention] = []
    seen_per_article: set[tuple[int, tuple[str, str]]] = set()
    for i, art in enumerate(articles):
        for _ in range(rng.randint(2, 6)):
            etype = rng.choices(
                ["ORG", "MILITARY_ORG", "GPE", "WEAPON", "PERSON"],
                weights=[2, 2, 5, 2, 1],
            )[0]
            if etype == "WEAPON":
                name = rng.choice(SAMPLE_WEAPONS)
            elif etype in {"ORG", "MILITARY_ORG"}:
                name = rng.choice(SAMPLE_ORGS)
            elif etype == "GPE":
                name = rng.choice(SAMPLE_LOCATIONS + SAMPLE_COUNTRIES)
            else:
                name = rng.choice(["Leader X", "Minister Y", "General Z"])
            canonical = name.lower().strip()
            k = (canonical, etype)
            seen_key = (i + 1, k)
            if seen_key in seen_per_article:
                continue
            seen_per_article.add(seen_key)
            e = _add_or_get(name, etype)
            e.mention_count += 1
            mentions.append(
                EntityMention(
                    entity_id=0,  # patched below
                    source_type="article",
                    source_id=i + 1,
                    context=art.title,
                    sentiment_label="neutral",
                    sentiment_score=0.0,
                )
            )
            mentions[-1]._entity_key = k  # type: ignore[attr-defined]

    for idx, e in enumerate(entities, start=1):
        e.id = idx

    for m in mentions:
        m.entity_id = entity_by_key[m._entity_key].id  # type: ignore[attr-defined]

    return entities, mentions


def _seed_sentiments(rng: random.Random, articles: list[Article]) -> list[Sentiment]:
    out: list[Sentiment] = []
    for i, _art in enumerate(articles):
        score = rng.uniform(-0.8, 0.8)
        if score > 0.2:
            label, pos, neg = "positive", 0.7, 0.1
        elif score < -0.2:
            label, pos, neg = "negative", 0.1, 0.7
        else:
            label, pos, neg = "neutral", 0.4, 0.3
        out.append(
            Sentiment(
                source_type="article",
                source_id=i + 1,
                positive=pos,
                neutral=1.0 - pos - neg,
                negative=neg,
                label=label,
                score=score,
                model="vader",
            )
        )
    return out


def _seed_topics(rng: random.Random) -> list[Topic]:
    return [
        Topic(
            topic_id=0,
            label="ukraine, russia, conflict",
            keywords=["ukraine", "russia", "kyiv", "donetsk", "kharkiv"],
            article_count=rng.randint(20, 60),
            representation={"backend": "sklearn"},
        ),
        Topic(
            topic_id=1,
            label="israel, gaza, hamas",
            keywords=["israel", "gaza", "hamas", "tel aviv"],
            article_count=rng.randint(15, 50),
            representation={"backend": "sklearn"},
        ),
        Topic(
            topic_id=2,
            label="china, taiwan, military",
            keywords=["china", "taiwan", "pla", "strait"],
            article_count=rng.randint(10, 30),
            representation={"backend": "sklearn"},
        ),
        Topic(
            topic_id=3,
            label="iran, proxies, sanctions",
            keywords=["iran", "hezbollah", "houthis", "sanctions"],
            article_count=rng.randint(8, 25),
            representation={"backend": "sklearn"},
        ),
    ]


def _seed_anomalies(rng: random.Random) -> list[Anomaly]:
    out: list[Anomaly] = []
    end = datetime.now(timezone.utc)
    anomaly_specs = [
        ("RUS", "event_count", 0.85, 4.2, "Sudden 5x increase in event volume"),
        ("UKR", "avg_tone", 0.78, -3.8, "Sentiment dropped sharply"),
        ("ISR", "conflict_count", 0.92, 5.1, "Conflict events spiked 6x"),
        ("IRN", "event_count", 0.71, 3.5, "Activity well above baseline"),
        ("YEM", "conflict_count", 0.65, 2.9, "Maritime incidents increasing"),
    ]
    for region, atype, sev, score, desc in anomaly_specs:
        out.append(
            Anomaly(
                region=region,
                date=(end - timedelta(days=rng.randint(0, 14))).strftime("%Y%m%d"),
                anomaly_type=f"zscore:{atype}",
                severity=sev,
                score=score,
                description=desc,
                context={"column": atype, "z_value": score},
            )
        )
    return out


def main() -> int:
    cfg = get_settings()
    configure_logging(level="INFO", json_format=False)
    logger = get_logger("seed")

    rng = random.Random(42)
    days = 30

    logger.info("seed_init", db=cfg.database_url, days=days)
    init_schema()

    logger.info("seed_events_start")
    events = _seed_events(rng, days=days)
    with session_scope() as session:
        for ev in events:
            session.add(ev)
        session.commit()
    logger.info("seed_events_done", count=len(events))

    logger.info("seed_articles_start")
    articles = _seed_articles(rng, events)
    with session_scope() as session:
        for art in articles:
            session.add(art)
        session.commit()
    logger.info("seed_articles_done", count=len(articles))

    logger.info("seed_entities_start")
    entities, mentions = _seed_entities_and_mentions(rng, articles)
    with session_scope() as session:
        for ent in entities:
            session.add(ent)
        session.flush()
        for m in mentions:
            session.add(m)
        session.commit()
    logger.info("seed_entities_done", entities=len(entities), mentions=len(mentions))

    logger.info("seed_sentiments_start")
    sentiments = _seed_sentiments(rng, articles)
    with session_scope() as session:
        for s in sentiments:
            session.add(s)
        session.commit()
    logger.info("seed_sentiments_done", count=len(sentiments))

    logger.info("seed_topics_start")
    topics = _seed_topics(rng)
    with session_scope() as session:
        for t in topics:
            session.add(t)
        session.commit()
    logger.info("seed_topics_done", count=len(topics))

    logger.info("seed_anomalies_start")
    anomalies = _seed_anomalies(rng)
    with session_scope() as session:
        for a in anomalies:
            session.add(a)
        session.commit()
    logger.info("seed_anomalies_done", count=len(anomalies))

    print("\n[OK] Demo data seeded.")
    print(f"      Events: {len(events)}")
    print(f"      Articles: {len(articles)}")
    print(f"      Entities: {len(entities)}")
    print(f"      Sentiments: {len(sentiments)}")
    print(f"      Topics: {len(topics)}")
    print(f"      Anomalies: {len(anomalies)}")
    print("\nRun `python -m scripts.dev dashboard` to explore.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
