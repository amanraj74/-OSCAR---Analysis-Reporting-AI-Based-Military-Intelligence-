"""Data ingestion package for OSCAR.

Provides a common abstract base class (`BaseIngestor`) and concrete
implementations for each data source:

- GDELT Project 2.0 events
- NewsAPI news articles (Sprint 1)
- Reddit RSS posts (Sprint 1)
"""

from .base import BaseIngestor, IngestionResult, SourceNotAvailableError

__all__ = [
    "BaseIngestor",
    "IngestionResult",
    "SourceNotAvailableError",
]
