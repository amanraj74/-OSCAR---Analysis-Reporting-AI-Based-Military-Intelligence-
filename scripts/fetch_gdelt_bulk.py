"""Bulk-download GDELT 2.0 events for any date range.

GDELT 2.0 publishes daily full exports at:
    http://data.gdeltproject.org/events/YYYYMMDD.export.csv
    http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip

Each file has all events for that 24h period. Larger than the
`lastupdate.txt` (which only has the last 24h's last update).

Usage:
    python scripts/fetch_gdelt_bulk.py --start 20250701 --end 20250709
    python scripts/fetch_gdelt_bulk.py --days 7
    python scripts/fetch_gdelt_bulk.py --days 3 --no-zip
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

USER_AGENT = "OSCAR-Bulk-Fetch/0.5 (educational; contact: aman.raj.intern@bserc.org)"

DEFAULT_DIR = Path(
    r"D:\my work\Bserc_projects\AI-Based Military Intelligence  ML and DA\data\external\gdelt_bulk"
)


def fetch_day(date_str: str, use_zip: bool = True) -> bytes:
    """Fetch a single day's GDELT export."""
    if use_zip:
        url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"
    else:
        url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV"
    r = requests.get(url, timeout=120, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.content


def extract_zip(content: bytes) -> bytes:
    """Extract the inner CSV from a GDELT zip."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = zf.namelist()[0]
        return zf.read(name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk-download GDELT 2.0 events for a date range.")
    parser.add_argument(
        "--start",
        help="Start date YYYYMMDD (default: today - days)",
    )
    parser.add_argument(
        "--end",
        help="End date YYYYMMDD (default: yesterday)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days back from end (default: 7)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Output directory (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Use uncompressed CSV (faster but bigger)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip dates that already have files (default: true)",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    end_dt = (
        datetime.strptime(args.end, "%Y%m%d")
        if args.end
        else (datetime.now(timezone.utc) - timedelta(days=1))
    )
    start_dt = (
        datetime.strptime(args.start, "%Y%m%d")
        if args.start
        else (end_dt - timedelta(days=args.days - 1))
    )

    print(f"Downloading GDELT events from {start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}")
    print(f"Output: {args.output}")
    print(f"Format: {'zip' if not args.no_zip else 'uncompressed CSV'}")

    total_bytes = 0
    success = 0
    skipped = 0
    failed = 0
    cur = start_dt
    while cur <= end_dt:
        date_str = cur.strftime("%Y%m%d")
        out_path = args.output / f"gdelt_{date_str}.csv"
        if args.skip_existing and out_path.exists():
            print(f"  {date_str}: SKIP (already exists)")
            skipped += 1
            cur += timedelta(days=1)
            continue
        try:
            print(f"  {date_str}: downloading...", end="", flush=True)
            content = fetch_day(date_str, use_zip=not args.no_zip)
            if not args.no_zip:
                content = extract_zip(content)
            out_path.write_bytes(content)
            size_kb = len(content) / 1024
            print(f" OK ({size_kb:.0f} KB)")
            total_bytes += len(content)
            success += 1
        except Exception as e:
            print(f" FAIL: {e}")
            failed += 1
        cur += timedelta(days=1)
        time.sleep(0.5)

    print()
    print(f"Done. Downloaded: {success}, Skipped: {skipped}, Failed: {failed}")
    print(f"Total size: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"Files in: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
