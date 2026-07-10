"""OSCAR entry-points — auto-load .env for any CLI / script invocation.

Import this module BEFORE importing src.* modules that depend on
Pydantic Settings, so that environment variables from `.env` are visible
at module-import time.

Usage::

    from scripts._bootstrap import PROJECT_ROOT
    # ... now src.config.get_settings() works as expected
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Load the project `.env` file into os.environ (idempotent)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


# Auto-load on import.
load_env()
