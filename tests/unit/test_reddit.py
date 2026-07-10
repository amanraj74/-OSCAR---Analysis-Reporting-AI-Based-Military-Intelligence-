"""Tests for Reddit RSS ingestor."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from src.ingestion.reddit import (
    RedditIngestor,
    RedditPost,
    _coerce_dt,
    _extract_id_and_subreddit,
    parse_reddit_feed,
)
from src.persistence.database import session_scope
from src.persistence.models import Article

RSS_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:atom="http://www.w3.org/2005/Atom" version="2.0">
  <channel>
    <title>r/{subreddit}</title>
    <link>https://www.reddit.com/r/{subreddit}/</link>
    <description>Test feed</description>
    <item>
      <title>Test Post Title</title>
      <link>https://www.reddit.com/r/{subreddit}/comments/{post_id}/test_post/</link>
      <guid isPermaLink="true">https://www.reddit.com/r/{subreddit}/comments/{post_id}/test_post/</guid>
      <pubDate>Sun, 05 Jul 2025 12:00:00 +0000</pubDate>
      <description><![CDATA[Test description here.]]></description>
      <dc:creator xmlns:dc="http://purl.org/dc/elements/1.1/">/u/testuser</dc:creator>
    </item>
    <item>
      <title>Second Post</title>
      <link>https://www.reddit.com/r/{subreddit}/comments/{post_id2}/second/</link>
      <guid>https://www.reddit.com/r/{subreddit}/comments/{post_id2}/second/</guid>
      <pubDate>Sun, 05 Jul 2025 13:00:00 +0000</pubDate>
      <description>Plain description</description>
    </item>
  </channel>
</rss>
"""


def _feed(sub: str = "worldnews", post_id: str = "abc123", post_id2: str = "def456") -> bytes:
    return RSS_TEMPLATE.format(subreddit=sub, post_id=post_id, post_id2=post_id2).encode("utf-8")


def test_parse_reddit_feed_extracts_items() -> None:
    posts = parse_reddit_feed(_feed(), subreddit="worldnews")
    assert len(posts) == 2
    assert posts[0].title == "Test Post Title"
    assert posts[0].subreddit == "worldnews"
    assert posts[0].external_id == "abc123"
    assert posts[1].title == "Second Post"
    assert posts[1].external_id == "def456"


def test_parse_reddit_feed_empty() -> None:
    posts = parse_reddit_feed(b"<?xml version='1.0'?><rss><channel></channel></rss>", subreddit="x")
    assert posts == []


def test_parse_reddit_feed_skips_title_less() -> None:
    bad = b"""<?xml version="1.0"?>
    <rss><channel>
      <item><link>https://reddit.com/r/x/comments/123/abc/</link><pubDate>Sun, 05 Jul 2025 12:00:00 +0000</pubDate></item>
      <item><title>Good</title><link>https://reddit.com/r/x/comments/456/ghi/</link><pubDate>Sun, 05 Jul 2025 12:00:00 +0000</pubDate></item>
    </channel></rss>
    """
    posts = parse_reddit_feed(bad, subreddit="x")
    assert len(posts) == 1
    assert posts[0].title == "Good"


def test_reddit_post_to_db_row() -> None:
    p = RedditPost(
        external_id="abc",
        source="reddit",
        title="Title",
        description="desc",
        url="https://reddit.com/r/x/comments/abc/test/",
        published_at=datetime(2025, 7, 5, tzinfo=timezone.utc),
        author="u/test",
        subreddit="x",
        score=0,
        num_comments=0,
    )
    row = p.to_db_row()
    assert row["external_id"] == "abc"
    assert row["source"] == "reddit"


def test_reddit_ingestor_persist_idempotent(fresh_db) -> None:
    p = RedditPost(
        external_id="abc",
        source="reddit",
        title="Title",
        description="desc",
        url="https://reddit.com/r/x/comments/abc/test/",
        published_at=datetime(2025, 7, 5, tzinfo=timezone.utc),
        author="u/test",
        subreddit="x",
        score=0,
        num_comments=0,
    )
    ingestor = RedditIngestor()
    n1 = ingestor.persist([p])
    assert n1 == 1
    with session_scope() as s:
        all_a = s.execute(select(Article)).scalars().all()
        assert len(all_a) == 1
    n2 = ingestor.persist([p])
    assert n2 == 1
    with session_scope() as s:
        all_a = s.execute(select(Article)).scalars().all()
        assert len(all_a) == 1


def test_extract_id_and_subreddit() -> None:
    entry = {"link": "https://www.reddit.com/r/worldnews/comments/abc123/abc/"}
    post_id, sub = _extract_id_and_subreddit(entry, entry["link"])
    assert post_id == "abc123"
    assert sub == "worldnews"


def test_coerce_dt_parses_rss_date() -> None:
    dt = _coerce_dt("Sun, 05 Jul 2025 12:00:00 +0000")
    assert dt is not None
    assert dt.year == 2025
    assert dt.tzinfo is not None


def test_coerce_dt_handles_none() -> None:
    assert _coerce_dt(None) is None


def test_reddit_ingestor_parse_dedupes() -> None:
    ingestor = RedditIngestor()
    feed = _feed()
    posts = ingestor.parse([feed, feed])
    assert len(posts) == 2
