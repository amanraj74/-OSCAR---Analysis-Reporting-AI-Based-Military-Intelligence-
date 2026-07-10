"""Named Entity Recognition pipeline for OSCAR.

Extracts entities (ORG, GPE, LOC, PERSON, DATE) using spaCy, plus a
custom `WEAPON` entity type via spaCy's EntityRuler patterns.

If spaCy is unavailable, falls back to a regex-only extractor that
still recognizes weapons, military orgs, and capitalized noun phrases.

Public API
----------
    NerPipeline().extract(text)               -> list[EntitySpan]
    NerPipeline().extract_and_persist(...)    -> int  (rows written)
    NerPipeline().score_entities()            -> {entity_type: count}

The pipeline is lazy-loaded: model + patterns are built on first call.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.nlp import HAS_SPACY, logger
from src.persistence.database import session_scope
from src.persistence.models import Entity, EntityMention

_DEFAULT_SPACY_MODEL = "en_core_web_sm"


WEAPON_PATTERNS: list[dict[str, Any]] = [
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^F[-]?16$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^F[-]?35$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^F[-]?18$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^Su[-]?24$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^Su[-]?25$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^Su[-]?34$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^Su[-]?35$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^Su[-]?57$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^MiG[-]?29$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^MiG[-]?31$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^T[-]?72$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^T[-]?80$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^T[-]?90$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^T[-]?14$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^T[-]?64$"}}]},
    {"label": "WEAPON", "pattern": [{"LOWER": "abrams"}]},
    {"label": "WEAPON", "pattern": [{"LOWER": "leopard"}, {"TEXT": {"REGEX": r"^[12]$"}}]},
    {"label": "WEAPON", "pattern": [{"LOWER": "challenger"}, {"TEXT": "2"}]},
    {"label": "WEAPON", "pattern": [{"TEXT": "ATACMS"}]},
    {"label": "WEAPON", "pattern": [{"TEXT": "HIMARS"}]},
    {"label": "WEAPON", "pattern": [{"TEXT": "Patriot"}]},
    {"label": "WEAPON", "pattern": [{"TEXT": "NASAMS"}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^S[-]?300$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^S[-]?400$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^S[-]?500$"}}]},
    {"label": "WEAPON", "pattern": [{"TEXT": "Bayraktar"}]},
    {"label": "WEAPON", "pattern": [{"TEXT": "TB2"}]},
    {"label": "WEAPON", "pattern": [{"TEXT": "Switchblade"}]},
    {"label": "WEAPON", "pattern": [{"LOWER": "shahed"}]},
    {"label": "WEAPON", "pattern": [{"TEXT": {"REGEX": r"^MQ[-]?9$"}}]},
    {"label": "WEAPON", "pattern": [{"LOWER": "reaper"}]},
    {"label": "WEAPON", "pattern": [{"LOWER": "predator"}]},
    {"label": "WEAPON", "pattern": [{"LOWER": "starlink"}]},
    {"label": "WEAPON", "pattern": [{"TEXT": "Javelin"}]},
    {"label": "WEAPON", "pattern": [{"TEXT": "NLAW"}]},
    {"label": "WEAPON", "pattern": [{"TEXT": "Stinger"}]},
    {"label": "WEAPON", "pattern": [{"LOWER": "iron"}, {"LOWER": "dome"}]},
    {"label": "WEAPON", "pattern": [{"TEXT": "GLSDB"}]},
    {"label": "WEAPON", "pattern": [{"TEXT": "Storm", "OP": "?"}, {"TEXT": "Shadow"}]},
]


MILITARY_ORG_PATTERNS: list[dict[str, Any]] = [
    {"label": "MILITARY_ORG", "pattern": [{"LOWER": "wagner"}, {"LOWER": "group"}]},
    {"label": "MILITARY_ORG", "pattern": [{"LOWER": "wagner"}]},
    {"label": "MILITARY_ORG", "pattern": [{"LOWER": "idf"}]},
    {"label": "MILITARY_ORG", "pattern": [{"LOWER": "hamas"}]},
    {"label": "MILITARY_ORG", "pattern": [{"LOWER": "hezbollah"}]},
    {"label": "MILITARY_ORG", "pattern": [{"LOWER": "houthi"}, {"LOWER": "rebels"}]},
    {"label": "MILITARY_ORG", "pattern": [{"LOWER": "houthis"}]},
    {"label": "MILITARY_ORG", "pattern": [{"LOWER": "isis"}]},
    {"label": "MILITARY_ORG", "pattern": [{"LOWER": "isil"}]},
    {"label": "MILITARY_ORG", "pattern": [{"LOWER": "taliban"}]},
    {"label": "MILITARY_ORG", "pattern": [{"LOWER": "nato"}]},
    {"label": "MILITARY_ORG", "pattern": [{"LOWER": "pmc"}, {"LOWER": "wagner"}]},
    {
        "label": "MILITARY_ORG",
        "pattern": [{"LOWER": "russian"}, {"LOWER": "armed"}, {"LOWER": "forces"}],
    },
]


@dataclass
class EntitySpan:
    """A single entity mention extracted from text."""

    text: str
    label: str
    start: int
    end: int
    canonical_name: str = ""

    def __post_init__(self) -> None:
        if not self.canonical_name:
            self.canonical_name = self.text.strip().lower()


@dataclass
class NerResult:
    """Aggregated NER extraction result for one document."""

    entities: list[EntitySpan] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.entities)

    def by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entities:
            out[e.label] = out.get(e.label, 0) + 1
        return out

    def unique(self) -> list[EntitySpan]:
        seen: set[tuple[str, str]] = set()
        out: list[EntitySpan] = []
        for e in self.entities:
            k = (e.canonical_name, e.label)
            if k in seen:
                continue
            seen.add(k)
            out.append(e)
        return out


_CAP_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-zA-Z'-]+)(?:\s+[A-Z][a-zA-Z'-]+){0,3}\b")


class _RegexEntityExtractor:
    """Lightweight fallback when spaCy is not installed."""

    ORG_KEYWORDS = {
        "UN",
        "NATO",
        "EU",
        "UNHCR",
        "WHO",
        "UNSC",
        "IAEA",
        "OPEC",
        "G7",
        "G20",
        "Kremlin",
        "Pentagon",
        "White House",
        "State Department",
        "MOD",
        "CIA",
        "FSB",
        "Mossad",
        "MI6",
        "GRU",
    }

    GPE_KEYWORDS = {
        "Ukraine",
        "Russia",
        "USA",
        "United States",
        "China",
        "Israel",
        "Iran",
        "Gaza",
        "Syria",
        "Iraq",
        "Afghanistan",
        "Yemen",
        "Lebanon",
        "Sudan",
        "Palestine",
        "Taiwan",
        "Korea",
        "Japan",
        "Germany",
        "France",
        "UK",
        "India",
        "Pakistan",
        "Turkey",
        "Egypt",
        "Libya",
        "Ethiopia",
        "Myanmar",
        "Philippines",
        "Mexico",
        "Brazil",
    }

    WEAPON_KEYWORDS = {
        "F-16",
        "F-35",
        "F-18",
        "Su-25",
        "Su-34",
        "Su-35",
        "Su-57",
        "MiG-29",
        "MiG-31",
        "T-72",
        "T-80",
        "T-90",
        "T-14",
        "T-64",
        "Abrams",
        "Leopard 2",
        "Challenger 2",
        "ATACMS",
        "HIMARS",
        "Patriot",
        "NASAMS",
        "S-300",
        "S-400",
        "S-500",
        "Bayraktar",
        "TB2",
        "Switchblade",
        "Shahed",
        "MQ-9 Reaper",
        "Predator",
        "Javelin",
        "NLAW",
        "Stinger",
        "Iron Dome",
        "GLSDB",
        "Storm Shadow",
        "Starlink",
    }

    def extract(self, text: str) -> list[EntitySpan]:
        out: list[EntitySpan] = []
        for kw in self.WEAPON_KEYWORDS:
            for m in re.finditer(re.escape(kw), text):
                out.append(EntitySpan(text=m.group(), label="WEAPON", start=m.start(), end=m.end()))
        for kw in self.MILITARY_ORG_PATTERNS:
            for entry in kw["pattern"]:
                tokens = entry if isinstance(entry, list) else []
                if not tokens:
                    continue
                pattern = (
                    r"\b"
                    + r"\s+".join(re.escape(t.get("LOWER", t.get("TEXT", ""))) for t in tokens)
                    + r"\b"
                )
                label = (
                    entry.get("label", "MILITARY_ORG")
                    if isinstance(entry, dict)
                    else "MILITARY_ORG"
                )
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    out.append(
                        EntitySpan(text=m.group(), label=label, start=m.start(), end=m.end())
                    )
        for kw in self.GPE_KEYWORDS:
            for m in re.finditer(r"\b" + re.escape(kw) + r"\b", text):
                out.append(EntitySpan(text=m.group(), label="GPE", start=m.start(), end=m.end()))
        for kw in self.ORG_KEYWORDS:
            for m in re.finditer(r"\b" + re.escape(kw) + r"\b", text):
                out.append(EntitySpan(text=m.group(), label="ORG", start=m.start(), end=m.end()))
        for m in _CAP_PHRASE_RE.finditer(text):
            phrase = m.group()
            if len(phrase) < 4:
                continue
            out.append(EntitySpan(text=phrase, label="MISC", start=m.start(), end=m.end()))
        return out


class NerPipeline:
    """NER pipeline with spaCy primary + regex fallback."""

    def __init__(self, spacy_model: str | None = None) -> None:
        self._model_name = spacy_model or _DEFAULT_SPACY_MODEL
        self._nlp: Any | None = None
        self._fallback = _RegexEntityExtractor()
        self._mode = "spacy" if HAS_SPACY else "regex"

    @property
    def mode(self) -> str:
        return self._mode

    def _ensure_loaded(self) -> None:
        if self._nlp is not None:
            return
        if not HAS_SPACY:
            logger.info("ner_using_regex_fallback")
            return
        try:
            import spacy

            logger.info("ner_loading_spacy", model=self._model_name)
            self._nlp = spacy.load(self._model_name)
            if "entity_ruler" not in self._nlp.pipe_names:
                ruler = self._nlp.add_pipe("entity_ruler", before="ner")
            else:
                ruler = self._nlp.get_pipe("entity_ruler")
            ruler.add_patterns(WEAPON_PATTERNS + MILITARY_ORG_PATTERNS)
            logger.info("ner_ready", pipes=self._nlp.pipe_names)
        except OSError:
            logger.warning(
                "ner_spacy_model_missing",
                model=self._model_name,
                hint="python -m spacy download " + self._model_name,
            )
            self._mode = "regex"

    def extract(self, text: str) -> NerResult:
        if not text or not text.strip():
            return NerResult()
        self._ensure_loaded()

        if self._nlp is None:
            return NerResult(entities=self._fallback.extract(text))

        doc = self._nlp(text)
        ents: list[EntitySpan] = []
        for ent in doc.ents:
            label = ent.label_
            if label in {"GPE", "LOC", "ORG", "PERSON", "DATE", "TIME", "NORP"}:
                ents.append(
                    EntitySpan(
                        text=ent.text,
                        label=label,
                        start=ent.start_char,
                        end=ent.end_char,
                    )
                )
            elif label == "WEAPON":
                ents.append(
                    EntitySpan(
                        text=ent.text,
                        label="WEAPON",
                        start=ent.start_char,
                        end=ent.end_char,
                    )
                )
            elif label == "MILITARY_ORG":
                ents.append(
                    EntitySpan(
                        text=ent.text,
                        label="MILITARY_ORG",
                        start=ent.start_char,
                        end=ent.end_char,
                    )
                )
        return NerResult(entities=ents)

    def extract_batch(self, texts: Iterable[str]) -> list[NerResult]:
        return [self.extract(t) for t in texts]

    def extract_and_persist(
        self,
        source_type: str,
        source_id: int,
        text: str,
        context_window: int = 120,
    ) -> int:
        """Extract entities from text and persist to entities + entity_mentions."""
        result = self.extract(text)
        if result.count == 0:
            return 0

        unique = result.unique()
        with session_scope() as session:
            entity_ids: dict[tuple[str, str], int] = {}
            for span in unique:
                canonical = _canonicalize(span.canonical_name)
                key = (canonical, span.label)
                if key in entity_ids:
                    continue
                stmt = sqlite_insert(Entity).values(
                    name=span.text,
                    canonical_name=canonical,
                    entity_type=span.label,
                    mention_count=0,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[Entity.canonical_name, Entity.entity_type],
                    set_={
                        "mention_count": Entity.mention_count + 1,
                        "last_seen": Entity.last_seen,
                    },
                )
                session.execute(stmt)
                row = session.execute(
                    __import__("sqlalchemy")
                    .select(Entity)
                    .where(
                        Entity.canonical_name == canonical,
                        Entity.entity_type == span.label,
                    )
                ).scalar_one()
                entity_ids[key] = row.id

            mention_rows: list[dict[str, Any]] = []
            for span in unique:
                canonical = _canonicalize(span.canonical_name)
                key = (canonical, span.label)
                eid = entity_ids[key]
                start = max(0, span.start - context_window // 2)
                end = min(len(text), span.end + context_window // 2)
                mention_rows.append(
                    {
                        "entity_id": eid,
                        "source_type": source_type,
                        "source_id": source_id,
                        "context": text[start:end],
                    }
                )

            if mention_rows:
                stmt = sqlite_insert(EntityMention).values(mention_rows)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[
                        EntityMention.entity_id,
                        EntityMention.source_type,
                        EntityMention.source_id,
                    ],
                    set_={
                        "context": stmt.excluded.context,
                        "mentioned_at": stmt.excluded.mentioned_at,
                    },
                )
                session.execute(stmt)

        return len(mention_rows)


def _canonicalize(name: str) -> str:
    """Lowercase, strip whitespace, basic punctuation normalization."""
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"^[\W_]+|[\W_]+$", "", name)
    return name or "_unknown_"


def batch_score(articles: Iterable[dict[str, Any]]) -> dict[int, NerResult]:
    """Run NER over many articles. Returns mapping article_id -> result."""
    pipeline = NerPipeline()
    out: dict[int, NerResult] = {}
    for art in articles:
        text = " ".join(
            filter(None, [art.get("title"), art.get("description"), art.get("content")])
        )
        out[art["id"]] = pipeline.extract(text)
    return out


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "EntitySpan",
    "NerPipeline",
    "NerResult",
    "WEAPON_PATTERNS",
    "MILITARY_ORG_PATTERNS",
    "batch_score",
]
