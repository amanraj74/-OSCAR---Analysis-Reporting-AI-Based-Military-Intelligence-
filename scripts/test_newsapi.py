"""Test NewsAPI key works (after .env is loaded)."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

api_key = os.environ.get("NEWS_API_KEY")
if not api_key:
    print("ERROR: NEWS_API_KEY not in env")
    sys.exit(1)

masked = api_key[:8] + "..." + api_key[-4:]
print(f"Using key: {masked}")

import requests

url = "https://newsapi.org/v2/everything"
params = {
    "q": "Ukraine",
    "pageSize": 5,
    "language": "en",
    "sortBy": "publishedAt",
    "apiKey": api_key,
}
r = requests.get(url, params=params, timeout=10)
print(f"Status: {r.status_code}")
data = r.json()
if r.status_code == 200:
    total = data.get("totalResults", 0)
    articles = data.get("articles", [])
    print(f"  totalResults: {total}")
    print(f"  returned: {len(articles)}")
    if articles:
        a = articles[0]
        title = a.get("title") or ""
        src = (a.get("source") or {}).get("name", "")
        print(f"  sample: {title[:80]}")
        print(f"  source: {src}")
    print("  KEY WORKS OK")
else:
    code = data.get("code", "")
    msg = data.get("message", "")[:200]
    print(f"  Error: {code} - {msg}")
