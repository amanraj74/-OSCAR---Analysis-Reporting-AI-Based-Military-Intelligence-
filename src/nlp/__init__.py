"""Natural Language Processing layer for OSCAR.

Modules:
    ner        — spaCy NER pipeline + custom weapon/org patterns.
    sentiment  — DistilBERT scorer with VADER fallback.
    topics     — BERTopic theme discovery with sklearn fallback.
    normalize  — sentence-transformers entity aliasing + diff fallback.

All models are lazy-loaded and optional. If a heavy dep (spaCy,
transformers, sentence-transformers, bertopic) is not installed,
the corresponding module falls back to a lightweight heuristic
implementation so the pipeline remains functional on minimal envs.
"""

from __future__ import annotations

import importlib
from typing import Any

from src.observability import get_logger

logger = get_logger("nlp")


def is_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


HAS_SPACY = is_available("spacy")
HAS_TRANSFORMERS = is_available("transformers")
HAS_SENTENCE_TRANSFORMERS = is_available("sentence_transformers")
HAS_BERTOPIC = is_available("bertopic")
HAS_VADER = is_available("vaderSentiment")
HAS_TORCH = is_available("torch")


__all__ = [
    "is_available",
    "HAS_SPACY",
    "HAS_TRANSFORMERS",
    "HAS_SENTENCE_TRANSFORMERS",
    "HAS_BERTOPIC",
    "HAS_VADER",
    "HAS_TORCH",
]


def __getattr__(name: str) -> Any:
    if name in {"ner", "sentiment", "topics", "normalize"}:
        return importlib.import_module(f"src.nlp.{name}")
    raise AttributeError(f"module 'src.nlp' has no attribute {name!r}")
