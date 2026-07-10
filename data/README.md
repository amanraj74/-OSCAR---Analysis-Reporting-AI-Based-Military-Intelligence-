# Data Card — OSCAR

> **Sources, schemas, refresh policy, and licensing for every dataset OSCAR consumes.**

---

## Sources

| Source | Type | Endpoint | Auth | Cadence | License |
|---|---|---|---|---|---|
| **GDELT Project 2.0** | Event stream | `http://data.gdeltproject.org/gdeltv2/lastupdate.txt` | None | 15 min | CC0 / public domain |
| **NewsAPI.org** | News articles | `https://newsapi.org/v2/` | API key (free) | On-demand | Free tier ToS |
| **Reddit RSS** | Social posts | `https://www.reddit.com/r/<sub>/.rss` | None | On-demand | Reddit ToS |

---

## Schemas

### `events` (GDELT)

Mirrors a focused subset of GDELT 2.0 columns most useful for OSCAR:

| Column | Type | Source (GDELT col) | Notes |
|---|---|---|---|
| `global_event_id` | BIGINT (unique) | col 1 | Stable global ID |
| `sql_date` | STRING(8) | col 2 | `YYYYMMDD` |
| `year` | INT | col 4 | |
| `actor1_name` | STRING | col 8 | Source actor |
| `actor1_country_code` | STRING(3) | col 9 | FIPS/country code |
| `actor2_name` | STRING | col 18 | Target actor |
| `actor2_country_code` | STRING(3) | col 19 | FIPS/country code |
| `event_code` | STRING(4) | col 26 | CAMEO code (e.g. `190`) |
| `event_root_code` | STRING(2) | col 27 | CAMEO root (e.g. `19`) |
| `goldstein_scale` | FLOAT | col 30 | `[-10, +10]`: cooperation ↔ conflict |
| `num_mentions` | INT | col 31 | Source volume |
| `num_articles` | INT | col 33 | Article volume |
| `avg_tone` | FLOAT | col 34 | `[-100, +100]` |
| `action_geo_fullname` | STRING | col 52 | e.g. `"Kiev, Ukraine"` |
| `action_geo_country_code` | STRING(3) | col 53 | |
| `action_geo_lat` | FLOAT | col 56 | |
| `action_geo_long` | FLOAT | col 57 | |

CAMEO root codes we treat as **conflict events**: `14`-`20` (PROTEST → MASS VIOLENCE).

### `articles` (NewsAPI / Reddit)

| Column | Type | Notes |
|---|---|---|
| `id` | INT PK | |
| `external_id` | STRING | Source-native ID |
| `source` | STRING(32) | `newsapi` / `reddit` |
| `title` | TEXT | |
| `description` | TEXT? | |
| `content` | TEXT? | |
| `url` | STRING | Canonical URL |
| `author` | STRING? | |
| `image_url` | STRING? | |
| `language` | STRING? | ISO-639-1 |
| `published_at` | DATETIME | UTC |
| `ingested_at` | DATETIME | UTC, set by OSCAR |

---

## Refresh & Immutability

- **`raw/`** — immutable raw dumps (`.bin` / `.json`). Never edited.
- **`processed/`** — silver layer (Parquet). Idempotent re-runs.
- **`external/`** — third-party reference data.
- **`oscar.db`** — SQLite, gitignored.

File-naming convention:
```
<source>__<topic>__<YYYYMMDDTHHMMSSZ>.<ext>
```

---

## Hashing & Integrity

Every raw payload is hashed (SHA-256) before persistence. The first 16 hex chars are used as the cache key. Full hashes are stored in `metadata.json` per fetch.

---

## Rate Limits & Etiquette

| Source | Limit | Strategy |
|---|---|---|
| GDELT | None official, recommend ≤ 30 req/min | Sequential fetches |
| NewsAPI | 100 req/day (free) | Cache 1h; daily quota budget |
| Reddit | Public RSS, soft ~10 req/min | Polite polling; cache 15 min |

---

## Licensing & Ethics

- **GDELT**: public domain; cite as "GDELT Project".
- **NewsAPI**: free tier is for development/personal use; commercial use requires paid plan. OSCAR non-commercial.
- **Reddit**: public RSS; OSCAR aggregates only — no user-level scraping.

OSCAR stores **aggregate / regional / event-level** information only. No individual-level data, no targeting.

---

## Adding a New Source

1. Add endpoint to `.env.example` and `configs/settings.yaml`.
2. Add a `<NAME>Settings` to `src/config.py`.
3. Add ORM table to `src/persistence/models.py`.
4. Create `src/ingestion/<source>.py` with a `<Source>Ingestor(BaseIngestor)` class.
5. Add unit + integration tests under `tests/`.
6. Update this file with the new row.

---

**Last Updated:** 2026-07-05 (Sprint 0)