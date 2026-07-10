"""Tests for topic discovery."""

from __future__ import annotations

import pytest

from src.nlp.topics import TopicDiscoverer, TopicInfo, _tokenize

SAMPLE_DOCS = [
    "Russian forces launched a major offensive in eastern Ukraine, targeting Kharkiv with missiles.",
    "NATO summit discusses military aid to Ukraine, focusing on F-16 fighter jets.",
    "Hezbollah fires rockets into northern Israel; IDF conducts retaliatory airstrikes.",
    "Iran nuclear talks resume in Vienna with European mediators.",
    "Sudan civil war escalates as RSF captures key city; humanitarian crisis deepens.",
    "Myanmar military junta extends emergency rule amid ongoing resistance.",
    "Taiwan reports Chinese military aircraft incursions near the island.",
    "Yemen Houthi rebels target commercial shipping in the Red Sea with drones.",
    "North Korea tests new ballistic missile; US condemns provocation.",
    "Israel-Gaza ceasefire negotiations stall as humanitarian crisis worsens.",
    "Russia Ukraine war enters third year with continued artillery duels.",
    "China Taiwan tensions rise after Pelosi's visit to Taipei.",
    "Iran proxies attack US bases in Syria and Iraq; tensions escalate.",
    "Sudan conflict displaces millions as aid agencies struggle to respond.",
    "Myanmar ethnic armed groups gain territory amid junta infighting.",
]


def test_tokenize_basic() -> None:
    tokens = _tokenize("The Israeli military retaliated with F-16 fighter jets!")
    assert "israeli" in tokens
    assert "military" in tokens
    assert "fighter" in tokens
    assert "jets" in tokens
    assert "retaliated" in tokens
    assert "the" not in tokens


def test_tokenize_handles_empty() -> None:
    assert _tokenize("") == []
    assert _tokenize(None) == []


def test_discoverer_initializes() -> None:
    td = TopicDiscoverer(n_topics=5)
    assert td.mode in {"bertopic", "sklearn"}


def test_discoverer_force_mode() -> None:
    td = TopicDiscoverer(n_topics=5, force_mode="sklearn")
    assert td.mode == "sklearn"


def test_discoverer_empty_docs() -> None:
    td = TopicDiscoverer(force_mode="sklearn")
    out = td.fit([])
    assert out == []


def test_discoverer_sklearn_finds_topics() -> None:
    pytest.importorskip("sklearn")
    td = TopicDiscoverer(n_topics=3, force_mode="sklearn")
    out = td.fit(SAMPLE_DOCS)
    assert isinstance(out, list)
    assert all(isinstance(t, TopicInfo) for t in out)
    assert sum(t.article_count for t in out) <= len(SAMPLE_DOCS)


def test_discoverer_fit_and_persist(fresh_db) -> None:
    from sqlalchemy import select

    from src.persistence.models import Topic

    td = TopicDiscoverer(n_topics=4, force_mode="sklearn")
    n = td.fit_and_persist(SAMPLE_DOCS, min_article_count=1)
    assert n > 0

    with __import__("src.persistence.database", fromlist=["session_scope"]).session_scope() as s:
        rows = s.execute(select(Topic)).scalars().all()
        assert len(rows) > 0
        assert all(r.topic_id >= 0 for r in rows)
        assert all(len(r.keywords or []) > 0 for r in rows)


def test_discoverer_get_top_topics(fresh_db) -> None:
    td = TopicDiscoverer(n_topics=4, force_mode="sklearn")
    td.fit_and_persist(SAMPLE_DOCS, min_article_count=1)
    top = td.get_top_topics(n=5)
    assert isinstance(top, list)
    if top:
        assert all(isinstance(t, TopicInfo) for t in top)


def test_topic_info_dataclass() -> None:
    t = TopicInfo(topic_id=0, label="ukraine, russia, war", keywords=["ukraine", "russia"])
    assert t.topic_id == 0
    assert t.article_count == 0
    assert len(t.keywords) == 2
