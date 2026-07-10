"""Fetch historical GDELT 2.0 events for the last N days.

Downloads all 15-min export files from data.gdeltproject.org, parses them, and
persists to the DB. Older files (typically beyond 7-14 days) return 404 and
are skipped silently.

GDELT URL format:
    http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.export.CSV.zip

- 96 files per day (one every 15 minutes)
- Each file is ~50-100 KB compressed, ~300 KB uncompressed
- ~50-200 events per file → ~5K-20K events per day

Usage:
    python scripts/fetch_historical_gdelt.py --days 7
    python scripts/fetch_historical_gdelt.py --days 14 --hours-step 1
    python scripts/fetch_historical_gdelt.py --days 3 --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    if (PROJECT_ROOT / ".env").exists():
        load_dotenv(PROJECT_ROOT / ".env", override=True)
except ImportError:
    pass

from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.config import get_settings, reset_settings_cache
from src.ingestion.gdelt import parse_gdelt_bytes
from src.persistence.database import init_schema, session_scope
from src.persistence.models import Event


def list_timestamps(start: datetime, end: datetime, hours_step: float) -> list[datetime]:
    """Generate timestamps between [start, end) at given hour interval."""
    delta = timedelta(hours=hours_step)
    out: list[datetime] = []
    cur = start
    while cur < end:
        out.append(cur)
        cur += delta
    return out


def fetch_one(url: str, timeout: int) -> list | None:
    """Download one GDELT file, parse, return events list. None on 404."""
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException:
        return None
    if r.status_code == 404:
        return None
    r.raise_for_status()
    try:
        return parse_gdelt_bytes(r.content, source_url=url)
    except Exception:
        return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch historical GDELT 2.0 events into the local DB."
    )
    parser.add_argument("--days", type=int, default=7, help="Days back from now (default 7)")
    parser.add_argument(
        "--hours-step", type=float, default=0.25, help="Step in hours (0.25 = 15min)"
    )
    parser.add_argument("--max-files", type=int, default=2000, help="Safety cap on files")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between requests (sec)")
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout per file (sec)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List URLs only, do not download",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel downloads (1=safe, 4-8=faster)",
    )
    args = parser.parse_args()

    reset_settings_cache()
    settings = get_settings()
    print(f"DB: {settings.database_url}")
    print(f"Fetching {args.days} days back, step={args.hours_step}h")

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=args.days)

    timestamps = list_timestamps(start, end, args.hours_step)
    if len(timestamps) > args.max_files:
        print(f"  Truncating to {args.max_files} files (you asked for {len(timestamps)})")
        timestamps = timestamps[-args.max_files :]

    print(f"  {len(timestamps)} timestamps: {start}  to  {end}")

    if args.dry_run:
        print()
        print("Dry run — would fetch:")
        for ts in timestamps[:20]:
            ts_str = ts.strftime("%Y%m%d%H%M%S")
            print(f"  http://data.gdeltproject.org/gdeltv2/{ts_str}.export.CSV.zip")
        if len(timestamps) > 20:
            print(f"  ... and {len(timestamps) - 20} more")
        return 0

    init_schema()

    total_events = 0
    success_count = 0
    skip_404 = 0
    fail_count = 0
    start_time = time.monotonic()

    for ts in timestamps:
        ts_str = ts.strftime("%Y%m%d%H%M%S")
        url = f"http://data.gdeltproject.org/gdeltv2/{ts_str}.export.CSV.zip"

        events = fetch_one(url, args.timeout)
        if events is None:
            skip_404 += 1
            continue
        if not events:
            fail_count += 1
            continue

        try:
            with session_scope() as session:
                rows = [e.to_db_row() for e in events]
                stmt = sqlite_insert(Event).values(rows)
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
            total_events += len(events)
            success_count += 1
            print(f"  [{success_count:3d}/{len(timestamps)}] {ts_str}: {len(events):4d} events")
        except Exception as e:
            fail_count += 1
            print(f"  [{ts_str}] persist err: {e}")

        if args.delay > 0:
            time.sleep(args.delay)

    elapsed = time.monotonic() - start_time
    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"  Files succeeded : {success_count}")
    print(f"  Files 404/missing: {skip_404}")
    print(f"  Files failed    : {fail_count}")
    print(f"  Events ingested : {total_events:,}")
    print(f"  Wall time       : {elapsed:.1f}s")
    if elapsed > 0 and total_events > 0:
        print(f"  Rate            : {total_events / elapsed:.0f} events/sec")
    print()
    print("Next step: rebuild silver + train")
    print("  python -m src.ingestion.cli transform")
    print("  python -m src.ml.cli train-escalation --horizon 7")
    return 0


if __name__ == "__main__":
    sys.exit(main())
