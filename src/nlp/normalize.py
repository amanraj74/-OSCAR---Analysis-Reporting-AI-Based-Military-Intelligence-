"""Entity normalization: aliasing via sentence-transformers + diff fallback.

Detects "Wagner Group" ≡ "PMC Wagner" ≡ "Wagner" and merges them into a
single canonical entity. Uses `all-MiniLM-L6-v2` for embeddings when
available; otherwise falls back to rapidfuzz / difflib string similarity.

Public API
----------
    EntityNormalizer().compute_aliases(entities) -> list[(a, b, sim)]
    EntityNormalizer().update_canonical_names()  -> int
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from src.config import get_settings
from src.nlp import HAS_SENTENCE_TRANSFORMERS, logger
from src.persistence.database import session_scope
from src.persistence.models import Entity

_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_THRESHOLD = 0.80


@dataclass
class EntityAlias:
    """Two entities that should be merged."""

    a_id: int
    a_canonical: str
    b_id: int
    b_canonical: str
    similarity: float

    def canonical_target(self) -> str:
        """Pick the longer / more descriptive name as the merge target."""
        return (
            self.a_canonical if len(self.a_canonical) >= len(self.b_canonical) else self.b_canonical
        )


class _DifflibBackend:
    """String similarity fallback using difflib.SequenceMatcher."""

    @staticmethod
    def similarity(a: str, b: str) -> float:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class _EmbeddingBackend:
    """Sentence-transformers cosine-similarity backend."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("normalize_loading_st", model=model_name)
        self._model = SentenceTransformer(model_name, device="cpu")
        self._cache: dict[str, Any] = {}

    def encode(self, texts: list[str]) -> Any:
        import numpy as np
        from sentence_transformers import SentenceTransformer  # noqa: F401

        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            new_emb = self._model.encode(
                uncached, normalize_embeddings=True, show_progress_bar=False
            )
            for t, e in zip(uncached, new_emb, strict=False):
                self._cache[t] = e
        return np.stack([self._cache[t] for t in texts])

    @staticmethod
    def cosine_sim_matrix(a: Any, b: Any) -> Any:

        return a @ b.T

    def similarity(self, a: str, b: str) -> float:
        embs = self.encode([a, b])
        sim = (embs[0] @ embs[1]).item()
        return float(max(0.0, min(1.0, sim)))


class EntityNormalizer:
    """Detect and merge entity aliases."""

    def __init__(
        self,
        embedding_model: str | None = None,
        threshold: float | None = None,
        force_mode: str | None = None,
    ) -> None:
        settings = get_settings()
        self._model_name = embedding_model or settings.nlp.embedding_model
        self._threshold = threshold or _DEFAULT_THRESHOLD
        self._backend: Any = None
        if force_mode is not None:
            self._mode = force_mode
        else:
            self._mode = "embedding" if HAS_SENTENCE_TRANSFORMERS else "difflib"

    @property
    def mode(self) -> str:
        return self._mode

    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return
        if self._mode == "embedding" and HAS_SENTENCE_TRANSFORMERS:
            try:
                self._backend = _EmbeddingBackend(self._model_name)
            except Exception as e:
                logger.warning("normalize_st_failed", error=str(e))
                self._mode = "difflib"
                self._backend = _DifflibBackend()
        else:
            self._backend = _DifflibBackend()

    def _similarity(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        if self._backend is None:
            self._ensure_loaded()
        return self._backend.similarity(a, b)

    def compute_aliases(
        self,
        entities: Iterable[Entity],
        threshold: float | None = None,
    ) -> list[EntityAlias]:
        """Find pairs of entities whose canonical names are above the similarity threshold."""
        ents = list(entities)
        if not ents:
            return []

        th = threshold or self._threshold
        self._ensure_loaded()

        out: list[EntityAlias] = []
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                a, b = ents[i], ents[j]
                if a.entity_type != b.entity_type:
                    continue
                sim = self._similarity(a.canonical_name, b.canonical_name)
                if sim >= th:
                    out.append(
                        EntityAlias(
                            a_id=a.id,
                            a_canonical=a.canonical_name,
                            b_id=b.id,
                            b_canonical=b.canonical_name,
                            similarity=sim,
                        )
                    )
        return out

    def update_canonical_names(self, threshold: float | None = None) -> int:
        """Merge aliased entities: re-assign mentions to the canonical, delete dupes.

        Returns the number of entities deleted (merged into another).
        """
        with session_scope() as session:
            all_entities = list(session.execute(select(Entity)).scalars().all())
        if not all_entities:
            return 0

        aliases = self.compute_aliases(all_entities, threshold=threshold)
        if not aliases:
            return 0

        parent: dict[int, int] = {a.a_id: a.a_id for a in aliases}
        parent.update({a.b_id: a.b_id for a in aliases})
        for a in aliases:
            ra, rb = parent[a.a_id], parent[a.b_id]
            if ra == rb:
                continue
            keep = ra if a.canonical_target() == a.a_canonical else rb
            other = rb if keep == ra else ra
            parent[other] = keep

        root_by_id: dict[int, int] = {}
        for ent in all_entities:
            root = ent.id
            seen: set[int] = set()
            while parent.get(root, root) != root and root not in seen:
                seen.add(root)
                root = parent[root]
            root_by_id[ent.id] = root

        from src.persistence.models import EntityMention

        deletes = 0
        with session_scope() as session:
            for ent_id, root_id in root_by_id.items():
                if ent_id == root_id:
                    continue
                loser = session.get(Entity, ent_id)
                winner = session.get(Entity, root_id)
                if loser is None or winner is None:
                    continue
                session.query(EntityMention).filter(EntityMention.entity_id == loser.id).update(
                    {EntityMention.entity_id: winner.id},
                    synchronize_session=False,
                )
                session.delete(loser)
                winner.mention_count = (winner.mention_count or 0) + (loser.mention_count or 0)
                deletes += 1
        return deletes


__all__ = [
    "EntityAlias",
    "EntityNormalizer",
    "_DifflibBackend",
    "_EmbeddingBackend",
]
