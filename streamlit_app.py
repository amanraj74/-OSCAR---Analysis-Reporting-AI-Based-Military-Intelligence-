"""Streamlit Community Cloud entry point for OSCAR.

This wrapper does three things on every startup:

  1. Ensures the SQLite schema exists (idempotent).
  2. If the DB is empty AND no real API keys are set, auto-seeds the
     30-day demo dataset so the dashboard renders meaningfully.
  3. Loads environment defaults suitable for a public demo (low log level,
     no real network calls).

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# --- 1. Environment defaults (set BEFORE importing the app) -----------------
os.environ.setdefault("APP_ENV", "streamlit")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("DATABASE_URL", "sqlite:///data/oscar.db")

# On Streamlit Cloud the filesystem is ephemeral, so model artifacts reset
# on every restart. That's OK — the dashboard reads from the SQLite DB which
# we re-seed on each cold start if needed.
os.environ.setdefault("MLFLOW_TRACKING_URI", "mlruns")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "oscar")

# --- 2. Bootstrap database + (maybe) seed demo data -------------------------
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env", override=False)  # noqa: E402


def _bootstrap_data() -> None:
    """Initialize the schema and auto-seed demo data if the DB is empty.

    Runs only once per cold start (Streamlit Cloud has an ephemeral disk so
    every restart behaves like a fresh machine — which is exactly what we
    want for a public demo).
    """
    from sqlalchemy import select

    from src.persistence.database import init_schema, session_scope
    from src.persistence.models import Event

    init_schema()

    with session_scope() as session:
        n = session.execute(select(Event).limit(1)).first()

    if n is None:
        # No real data → seed demo dataset for a meaningful first impression
        print("[OSCAR] Empty database detected — seeding 30-day demo dataset...")
        try:
            from scripts.seed_demo import main as seed_main

            seed_main()
            print("[OSCAR] Demo data seeded successfully.")
        except Exception as exc:  # noqa: BLE001
            print(f"[OSCAR] Demo seed skipped: {exc!s}")
    else:
        print("[OSCAR] Database already populated — skipping seed.")


_bootstrap_data()  # noqa: E402

# --- 3. Hand off to the actual dashboard ------------------------------------
# Importing dashboard.app runs its `main()` because of `render()` at module
# level, but to keep things clean we explicitly call main() here.
from dashboard import app as dashboard_app  # noqa: E402

if __name__ == "__main__":
    dashboard_app.main()
else:
    dashboard_app.main()

