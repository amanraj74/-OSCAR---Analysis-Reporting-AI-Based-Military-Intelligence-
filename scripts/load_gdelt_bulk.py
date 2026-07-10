"""Load GDELT bulk daily exports into the OSCAR events table.

Reads all CSV files in a directory (e.g., from scripts/fetch_gdelt_bulk.py)
and inserts events into the events table using the same format as the
regular GDELT ingestor.

Usage:
    python scripts/load_gdelt_bulk.py --dir data/external/gdelt_bulk
    python scripts/load_gdelt_bulk.py --file data/external/gdelt_bulk/gdelt_20260708.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)

import pandas as pd
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.config import reset_settings_cache
from src.ingestion.gdelt import parse_gdelt_line
from src.persistence.database import init_schema, session_scope
from src.persistence.models import Event

DATETIME_UTC = __import__("datetime").datetime
TIMEDELTA = __import__("datetime").timedelta
TIMEZONE_UTC = __import__("datetime").timezone


def load_file(path: Path, batch_size: int = 1000) -> int:
    """Load a single GDELT CSV into the events table."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        dtype=str,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
        nrows=0,  # set below if column count differs
        low_memory=False,
    )
    # GDELT 2.0 events has 58 columns (reduced format). Re-read with the right count.
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        dtype=str,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )

    if df.empty:
        print("  (empty file)")
        return 0

    # The 58 columns match the existing parse_gdelt_line format
    # Apply parser
    rows = []
    for _, row in df.iterrows():
        line = "\t".join(str(v) for v in row.tolist())
        ev = parse_gdelt_line(line)
        if ev is None:
            continue
        rows.append(ev.to_db_row())

    if not rows:
        print("  (no valid events after parsing)")
        return 0

    # Batch insert
    total = 0
    with session_scope() as session:
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            stmt = sqlite_insert(Event).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Event.global_event_id],
                set_={
                    "num_mentions": stmt.excluded.num_mentions,
                    "num_articles": stmt.excluded.num_articles,
                    "avg_tone": stmt.excluded.avg_tone,
                    "ingested_at": stmt.excluded.ingested_at,
                },
            )
            session.execute(stmt)
            total += len(chunk)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Load GDELT bulk CSV files into events table.")
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory of CSV files to load",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Single CSV file to load",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Insert batch size (default: 1000)",
    )
    args = parser.parse_args()

    if not args.dir and not args.file:
        print("ERROR: Provide --dir or --file")
        return 1

    reset_settings_cache()
    init_schema()

    paths = [args.file] if args.file else sorted(args.dir.glob("*.csv"))

    if not paths:
        print("No CSV files found")
        return 1

    print(f"Found {len(paths)} file(s)")
    grand_total = 0
    for path in paths:
        print(f"Loading {path.name}...", end=" ", flush=True)
        try:
            count = load_file(path, batch_size=args.batch_size)
            print(f"{count:,} events")
            grand_total += count
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nTotal loaded: {grand_total:,} events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
