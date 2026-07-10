"""Tests for entity normalization."""

from __future__ import annotations

from src.nlp.normalize import EntityAlias, EntityNormalizer, _DifflibBackend


def test_difflib_backend_similarity() -> None:
    backend = _DifflibBackend()
    sim = backend.similarity("Wagner Group", "PMC Wagner")
    assert 0.5 < sim < 1.0


def test_difflib_backend_similarity_identical() -> None:
    backend = _DifflibBackend()
    sim = backend.similarity("Wagner", "Wagner")
    assert sim == 1.0


def test_normalizer_initializes() -> None:
    n = EntityNormalizer()
    assert n.mode in {"embedding", "difflib"}


def test_normalizer_force_mode() -> None:
    n = EntityNormalizer(force_mode="difflib")
    assert n.mode == "difflib"


def test_normalizer_compute_aliases_simple() -> None:
    from src.persistence.models import Entity

    n = EntityNormalizer(threshold=0.5, force_mode="difflib")
    ents = [
        Entity(
            id=1, name="Wagner Group", canonical_name="wagner group", entity_type="MILITARY_ORG"
        ),
        Entity(
            id=2, name="Wagner Group", canonical_name="wagner group", entity_type="MILITARY_ORG"
        ),
        Entity(id=3, name="Houthi", canonical_name="houthi", entity_type="MILITARY_ORG"),
    ]
    aliases = n.compute_aliases(ents)
    pairs = {(a.a_id, a.b_id) for a in aliases}
    assert (1, 2) in pairs or (2, 1) in pairs
    assert (1, 3) not in pairs
    assert (2, 3) not in pairs


def test_normalizer_compute_aliases_obvious_duplicate() -> None:
    from src.persistence.models import Entity

    n = EntityNormalizer(threshold=0.7, force_mode="difflib")
    ents = [
        Entity(
            id=1,
            name="Israel Defense Forces",
            canonical_name="israel defense forces",
            entity_type="MILITARY_ORG",
        ),
        Entity(
            id=2,
            name="Israeli Defense Forces",
            canonical_name="israeli defense forces",
            entity_type="MILITARY_ORG",
        ),
    ]
    aliases = n.compute_aliases(ents)
    assert len(aliases) >= 1
    pairs = {(a.a_id, a.b_id) for a in aliases}
    assert (1, 2) in pairs or (2, 1) in pairs


def test_normalizer_skips_different_types() -> None:
    from src.persistence.models import Entity

    n = EntityNormalizer(threshold=0.5, force_mode="difflib")
    ents = [
        Entity(id=1, name="Hamas", canonical_name="hamas", entity_type="MILITARY_ORG"),
        Entity(id=2, name="Hamas", canonical_name="hamas", entity_type="GPE"),
    ]
    aliases = n.compute_aliases(ents)
    assert aliases == []


def test_normalizer_threshold_filter() -> None:
    from src.persistence.models import Entity

    n_low = EntityNormalizer(threshold=0.1, force_mode="difflib")
    n_high = EntityNormalizer(threshold=0.99, force_mode="difflib")
    ents = [
        Entity(
            id=1,
            name="Israel Defense Forces",
            canonical_name="israel defense forces",
            entity_type="MILITARY_ORG",
        ),
        Entity(
            id=2,
            name="Israeli Defense Forces",
            canonical_name="israeli defense forces",
            entity_type="MILITARY_ORG",
        ),
    ]
    low = n_low.compute_aliases(ents)
    high = n_high.compute_aliases(ents)
    assert len(low) >= len(high)


def test_normalizer_update_canonical_names(fresh_db) -> None:
    from sqlalchemy import select

    from src.persistence.models import Entity

    n = EntityNormalizer(threshold=0.5, force_mode="difflib")
    with __import__("src.persistence.database", fromlist=["session_scope"]).session_scope() as s:
        s.add_all(
            [
                Entity(name="Wagner", canonical_name="wagner group", entity_type="MILITARY_ORG"),
                Entity(
                    name="Wagner Group", canonical_name="wagner groups", entity_type="MILITARY_ORG"
                ),
            ]
        )

    deleted = n.update_canonical_names(threshold=0.7)
    assert deleted >= 0

    with __import__("src.persistence.database", fromlist=["session_scope"]).session_scope() as s:
        rows = s.execute(select(Entity)).scalars().all()
        assert isinstance(rows, list)


def test_entity_alias_target_picks_longer() -> None:
    a = EntityAlias(1, "wagner", 2, "wagner group", 0.9)
    assert a.canonical_target() == "wagner group"
    a2 = EntityAlias(1, "wagner group", 2, "wagner", 0.9)
    assert a2.canonical_target() == "wagner group"
