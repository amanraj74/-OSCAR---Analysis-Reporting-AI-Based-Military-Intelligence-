"""Tests for the NER pipeline."""

from __future__ import annotations

from src.nlp.ner import (
    MILITARY_ORG_PATTERNS,
    WEAPON_PATTERNS,
    NerPipeline,
    NerResult,
    _canonicalize,
)

SAMPLE_CONFLICT_TEXT = (
    "The United States supplied F-16 fighter jets to Ukraine last week. "
    "Russia responded by deploying Su-35 aircraft to the region. "
    "NATO condemned the Wagner Group's involvement. "
    "Hezbollah launched ATACMS missiles from Lebanon towards Israel. "
    "The IDF retaliated with Iron Dome interceptors. "
    "Bayraktar TB2 drones operated by Ukraine struck Russian positions near Kharkiv."
)


def test_ner_extract_returns_result() -> None:
    pipeline = NerPipeline()
    result = pipeline.extract(SAMPLE_CONFLICT_TEXT)
    assert isinstance(result, NerResult)
    assert result.count > 0


def test_ner_extracts_weapons() -> None:
    pipeline = NerPipeline()
    result = pipeline.extract(SAMPLE_CONFLICT_TEXT)
    weapons = [e for e in result.entities if e.label == "WEAPON"]
    weapon_texts = {w.text.upper() for w in weapons}
    assert "F-16" in weapon_texts or "F16" in weapon_texts
    assert "SU-35" in weapon_texts or "SU35" in weapon_texts
    assert "ATACMS" in weapon_texts


def test_ner_extracts_military_orgs() -> None:
    pipeline = NerPipeline()
    result = pipeline.extract(SAMPLE_CONFLICT_TEXT)
    orgs = [e for e in result.entities if e.label == "MILITARY_ORG"]
    org_texts = {o.text.lower() for o in orgs}
    assert any("wagner" in t for t in org_texts)


def test_ner_extracts_gpes() -> None:
    pipeline = NerPipeline()
    result = pipeline.extract(SAMPLE_CONFLICT_TEXT)
    gpes = [e for e in result.entities if e.label == "GPE"]
    gpe_texts = {g.text for g in gpes}
    assert "Ukraine" in gpe_texts
    assert "Israel" in gpe_texts or "Israel" in " ".join(e.text for e in result.entities)


def test_ner_empty_text() -> None:
    pipeline = NerPipeline()
    assert pipeline.extract("").count == 0
    assert pipeline.extract("   ").count == 0


def test_ner_unique_dedupes() -> None:
    pipeline = NerPipeline()
    result = pipeline.extract("F-16 F-16 F-16 aircraft")
    unique = result.unique()
    assert len(unique) == len({(e.canonical_name, e.label) for e in unique})


def test_ner_by_type() -> None:
    pipeline = NerPipeline()
    result = pipeline.extract(SAMPLE_CONFLICT_TEXT)
    by_type = result.by_type()
    assert isinstance(by_type, dict)
    assert sum(by_type.values()) == result.count


def test_ner_persists_to_db(fresh_db) -> None:
    from sqlalchemy import select

    from src.persistence.models import Entity, EntityMention

    pipeline = NerPipeline()
    article_id = 1
    n = pipeline.extract_and_persist(
        source_type="article",
        source_id=article_id,
        text=SAMPLE_CONFLICT_TEXT,
    )
    assert n > 0

    with __import__("src.persistence.database", fromlist=["session_scope"]).session_scope() as s:
        ents = s.execute(select(Entity)).scalars().all()
        mentions = s.execute(select(EntityMention)).scalars().all()
        assert len(ents) > 0
        assert len(mentions) > 0
        assert all(m.source_type == "article" for m in mentions)
        assert all(m.source_id == article_id for m in mentions)


def test_canonicalize_basic() -> None:
    assert _canonicalize("Wagner Group") == "wagner group"
    assert _canonicalize("  F-16  ") == "f-16"
    assert _canonicalize("PMC Wagner!!") == "pmc wagner"
    assert _canonicalize("") == "_unknown_"


def test_ner_idempotent_persistence(fresh_db) -> None:
    from sqlalchemy import select

    from src.persistence.models import Entity, EntityMention

    pipeline = NerPipeline()
    text = "F-16 aircraft hit a Russian position. Wagner Group claimed responsibility."

    n1 = pipeline.extract_and_persist("article", 1, text)
    assert n1 > 0
    n2 = pipeline.extract_and_persist("article", 1, text)
    assert n2 > 0

    with __import__("src.persistence.database", fromlist=["session_scope"]).session_scope() as s:
        ents = s.execute(select(Entity)).scalars().all()
        mentions = s.execute(select(EntityMention)).scalars().all()
        assert len(ents) > 0
        assert len(mentions) == len(ents)


def test_weapon_patterns_have_required_keys() -> None:
    for p in WEAPON_PATTERNS:
        assert "label" in p
        assert p["label"] == "WEAPON"
        assert "pattern" in p
        assert len(p["pattern"]) >= 1


def test_military_org_patterns_have_required_keys() -> None:
    for p in MILITARY_ORG_PATTERNS:
        assert "label" in p
        assert "pattern" in p
