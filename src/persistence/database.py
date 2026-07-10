"""SQLAlchemy engine, session, and declarative base for OSCAR.

The database URL is read from the `DATABASE_URL` environment variable
(defaults to `sqlite:///data/oscar.db`). The data directory is created
on first connect when SQLite is used.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import get_settings
from src.observability import get_logger
from src.persistence import models  # noqa: F401  ensure models registered

logger = get_logger("persistence.db")


class Base(DeclarativeBase):
    """Declarative base for all OSCAR ORM models."""


def _sqlite_path_from_url(url: str) -> Path | None:
    """Extract the filesystem path from a sqlite URL, if applicable."""
    if not url.startswith("sqlite:///"):
        return None
    return Path(url.replace("sqlite:///", "", 1))


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _build_engine(url: str) -> Engine:
    """Construct an Engine with sensible defaults."""
    connect_args: dict[str, Any] = {}
    engine_kwargs: dict[str, Any] = {
        "pool_pre_ping": True,
        "future": True,
    }

    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        engine_kwargs["connect_args"] = connect_args

    if url.startswith("postgresql"):
        engine_kwargs.setdefault("pool_size", 5)
        engine_kwargs.setdefault("max_overflow", 10)

    return create_engine(url, **engine_kwargs)


def get_engine() -> Engine:
    """Return the lazily-initialized global Engine."""
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine

    settings = get_settings()
    url = settings.database_url

    db_path = _sqlite_path_from_url(url)
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("sqlite_db_initialized", path=str(db_path))

    _engine = _build_engine(url)
    _SessionLocal = sessionmaker(
        bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False
    )
    return _engine


def init_schema() -> None:
    """Create all tables defined on `Base.metadata`.

    For production, prefer Alembic migrations (see `migrations/`).
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("schema_ready")


def get_session() -> Session:
    """Return a new Session (caller is responsible for close)."""
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope.

    Commits on success, rolls back on exception, always closes.
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Dispose the engine and clear the session factory (test helper)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


# Compatibility alias for the public API name.
engine = property(lambda _: get_engine())  # type: ignore[assignment]
