"""Domain layer: OSCAR business entities.

Pure-Python value objects used by application services. Independent of
ORM / DB. Persistence layer maps these to/from ORM models.
"""

from .schemas import GdeltEventSchema, IngestionSummary, NewsArticleSchema, RedditPostSchema

__all__ = [
    "GdeltEventSchema",
    "IngestionSummary",
    "NewsArticleSchema",
    "RedditPostSchema",
]
