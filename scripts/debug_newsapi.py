"""Debug NewsAPI ingestor end-to-end."""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.newsapi import NewsApiIngestor

ing = NewsApiIngestor(query="Ukraine", page_size=20)
print(f"api_key set: {bool(ing.settings.newsapi.api_key)}")
print(f"query: {ing.query}")
print(f"endpoint: {ing._endpoint()}")
print(f"params: {ing._params()}")

raw = ing._fetch_with_retries()
print(f"raw bytes count: {len(raw)}")
if raw:
    d = json.loads(raw[0])
    print(f"  status: {d.get('status')}, totalResults: {d.get('totalResults')}")
    print(f"  articles in response: {len(d.get('articles', []))}")
    if d.get("articles"):
        print(f"  first title: {d['articles'][0]['title'][:80]}")

items = ing.parse(raw)
print(f"parsed items: {len(items)}")
if items:
    print(f"  first: {items[0].title[:60]}")

if items:
    persisted = ing.persist(items)
    print(f"persisted: {persisted}")
