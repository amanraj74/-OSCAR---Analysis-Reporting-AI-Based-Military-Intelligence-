"""Debug pydantic settings load."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

print("OS env vars:")
for k in ["NEWS_API_KEY", "DATABASE_URL", "REDDIT_SUBREDDITS", "GDELT_BATCH_HOURS_BACK"]:
    print(f"  {k} = {os.environ.get(k, 'MISSING')[:50]}")

print("\nNow loading settings:")
try:
    from src.config import get_settings, reset_settings_cache

    reset_settings_cache()
    cfg = get_settings()
    print(f"  newsapi.api_key set: {bool(cfg.newsapi.api_key)}")
    print(f"  database_url: {cfg.database_url}")
    print(f"  reddit.subreddits: {cfg.reddit.subreddits}")
    print(f"  gdelt.batch_hours_back: {cfg.gdelt.batch_hours_back}")
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {e}")
