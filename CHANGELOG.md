# CHANGELOG.md

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.5.0] — 2026-07-05 — Release v1.0 Candidate (Sprint 5)

### Added

- **Multi-stage Dockerfile** — Python 3.11-slim builder + slim runtime; ~700 MB image; `/_stcore/health` healthcheck.
- **`docker-compose.yml`** — full stack: `app` (Streamlit) + `mlflow` (tracking server) + persistent volumes (`oscar-data`, `oscar-models`, `oscar-logs`, `oscar-mlflow`).
- **`.dockerignore`** — covers `__pycache__/`, `.git/`, `data/raw/`, `models/`, `mlruns/`, etc.
- **`scripts/seed_demo.py`** — 30-day synthetic but plausible snapshot: ~2.6k GDELT events across 12 countries + 200 articles + 44 entities + 200 sentiments + 4 topics + 5 anomalies. Reproducible (seed=42).
- **E2E smoke tests** (`tests/e2e/test_smoke.py`, 5 tests, marked `@pytest.mark.e2e`) — full pipeline: seed → NER → features → classifier → dashboard loaders → all 7 page modules.
- **BSERC technical report** (`docs/REPORT.md`) — 10 sections, ~5k words: problem, architecture, implementation, engineering quality, performance, limitations, future work, reproducibility.
- **Deployment guide** (`docs/DEPLOYMENT.md`) — local / Docker / Streamlit Cloud / HF Spaces / AWS scenarios + Postgres migration + MLflow setup + security checklist.
- **Demo video script** (`docs/demo/video_script.md`) — 5-minute screen-recording walkthrough with timestamps and voiceover text.
- **Project listing submission** (`docs/PROJECT_LISTING.md`) — pre-filled BSERC form fields.

### Engineering

- **161 + 5 E2E tests** passing (6 integration / e2e deselected); coverage **69.77%** on `src/`.
- **Lint clean**: ruff + black + isort + bandit.
- **Total files**: 60+ Python sources, 10+ docs/config, 50+ tests.
- **Reproducible**: every random seed pinned; every ML run logged via MLflow or JSON.
- **Production-grade**: 8-layer Clean Architecture; idempotent persistence; type-safe config; structured logging; comprehensive test coverage.

---

## [0.4.0] — 2026-07-05 — Streamlit Dashboard

### Added

- **Streamlit multi-page dashboard** (`dashboard/`):
  - `app.py` — landing page with live status metrics + system info.
  - `pages/0_Home.py` — onboarding + recent activity.
  - `pages/1_Map.py` — Plotly choropleth (sentiment / event count / conflict count) + geo-located GDELT bubbles (Goldstein scale).
  - `pages/2_Sentiment.py` — time-series of article sentiment, positive/negative share, top positive/negative articles, GDELT avg tone bubbles.
  - `pages/3_Entities.py` — top entities by type (ORG / WEAPON / GPE / PERSON) with bar charts, lightweight co-occurrence matrix.
  - `pages/4_Forecast.py` — per-region 7-day forecast with 95% CI bands + production model metrics.
  - `pages/5_Alerts.py` — live anomaly feed sorted by severity, criticality counts, detail view.
  - `pages/6_About.py` — methodology, data sources, ethics, model card links.
- **Custom CSS** (`dashboard/assets/style.css`) — brand colors, metric cards with accent border, sidebar polish.
- **Cached data loaders** (`dashboard/utils.py`) — `@st.cache_data` with 5-min TTL for all DB queries; safe fallbacks when DB isn't ready.
- **Sidebar status panel** on every page — events / articles / entities / anomalies metrics + Refresh button.
- **Country code mapper** (`country_code_to_iso3`) — FIPS → ISO-3 for Plotly choropleth.
- **Color helpers** (`sentiment_color`, `severity_color`) — semantic green / amber / red.
- **26 dashboard tests** verifying all pages import cleanly + all utility functions work end-to-end.

### Engineering

- **Streamlit 1.36+** compatible.
- **Cross-platform**: tested on Python 3.10 + Windows.
- **DB-safe**: every data accessor wrapped in try/except returning empty defaults.
- **Visual** consistency: shared `page_header()`, `sidebar_status_panel()` helpers.
- **Laptop-runnable**: no GPU required, no heavy ML model loaded eagerly.

---

## [0.3.0] — 2026-07-05 — ML Core

### Added

- **Feature engineering layer** (`src/ml/features.py`) — produces ~50 features per (date, country) from silver tables: lag (1/3/7d), rolling mean/std (3/7/14d windows), calendar features (day-of-week, month, weekend).
- **XGBoost escalation classifier** (`src/ml/escalation.py`) — predicts probability of 1/3/7-day escalation per country. **sklearn GradientBoosting fallback** for offline / minimal-env.
- **Prophet forecaster** (`src/ml/forecast.py`) — per-region event-count forecasts with 95% CI bands. **Linear-regression fallback** for minimal envs.
- **Isolation Forest anomaly detector** (`src/ml/anomaly.py`) — per-feature multivariate anomalies; per-region roll-up. **Z-score fallback** for offline envs.
- **MLflow experiment tracking** (`src/ml/tracking.py`) with **JSON-file fallback** — every training run is logged (params, metrics, artifacts, tags).
- **Model registry** (`src/ml/registry.py`) — versioned, staged (None / Staging / Production / Archived). Promote/demote semantics with auto-archive of prior production.
- **ML CLI** (`python -m src.ml.cli`): `build-features`, `train-escalation`, `forecast`, `detect-anomalies`, `promote`, `list-models`, `ml-train-all`.
- **Model cards** (`docs/models/*.md`): 3 cards — escalation classifier, forecaster, anomaly detector. Cover purpose, training, evaluation, limitations, ethics, reproducibility.
- **48 new unit tests** (135 total, 1 integration deselected).
- **49 source files** all ruff/black/isort clean.

### Engineering

- **Time-aware cross-validation** (`TimeSeriesSplit`) for the classifier.
- **Reproducible seeds** — every model has `random_state=42`.
- **Class-weighted** for imbalanced escalation target (~5–15% positive).
- **Pickle roundtrip** verified for serialization.
- **Per-region forecaster** runs in batch; persists Parquet to `data/processed/forecasts/`.

### ML Stack

| Component | Primary | Fallback |
|---|---|---|
| Classifier | XGBoost 2.0+ | sklearn GradientBoosting |
| Forecaster | Prophet 1.1.5+ | Linear regression (custom) |
| Anomaly | sklearn IsolationForest | Rolling z-score (custom) |
| Tracking | MLflow 2.15+ | JSON files (custom) |
| Registry | (no native) | JSON file (custom) |

---

## [0.2.0] — 2026-07-05 — Data Pipeline + NLP Core

### Added

- **GDELT, NewsAPI, Reddit RSS ingestors** — full ingestion paths with rate-limit, retry/backoff, idempotent persistence, dedupe.
- **Bronze → Silver transform layer** — Parquet tables aggregating events per (date, country) and articles per (date, source).
- **Pydantic v2 schemas** — strict input validation at every parse boundary (`GdeltEventSchema`, `NewsArticleSchema`, `RedditPostSchema`, `IngestionSummary`).
- **spaCy NER pipeline** with 33 custom WEAPON patterns (F-16, Su-35, HIMARS, ATACMS, Javelin, Iron Dome, Starlink, Bayraktar TB2, Storm Shadow, ...).
- **MILITARY_ORG patterns**: Wagner, IDF, Hamas, Hezbollah, Houthis, ISIS, Taliban, NATO, Wagner Group, Russian Armed Forces.
- **Regex fallback NER** — fully functional NER even without spaCy installed.
- **DistilBERT sentiment scorer** with `distilbert-base-uncased-finetuned-sst-2-english` (SST-2) and VADER fallback for offline / CPU-only environments.
- **Entity normalization** via sentence-transformers (`all-MiniLM-L6-v2`) cosine similarity, with difflib string-similarity fallback.
- **Alias resolution**: merges duplicate entities (e.g., "Wagner Group" + "PMC Wagner" → single canonical entity with reassigned mentions).
- **Topic discovery** via BERTopic + sklearn TF-IDF/KMeans fallback (`n_topics=8` default).
- **NLP CLI** (`python -m src.nlp.cli`): `ner-process`, `sentiment-score`, `topics-discover`, `entities-normalize`, `nlp-process` (run all).
- **Ingestion CLI** (`python -m src.ingestion.cli`): `refresh --source {gdelt,newsapi,reddit}`, `transform` (build silver tables).
- **Cross-platform task runner** (`scripts/dev.py` + `Makefile`) — every command works on Windows + Unix.
- **Pre-configured CI** (`.github/workflows/ci.yml`) — lint + test on matrix OS / Python.
- **87 unit tests** with 70% line coverage on `src/`;**1 integration test** (off by default).

### Engineering

- **Lint:** ruff + black + isort all green.
- **Tests** boot in ~30s on Windows + Python 3.10 (no heavy ML deps required thanks to fallbacks).
- **Lazy-loaded models** — heavy deps (spaCy, transformers, sentence-transformers, BERTopic) load only on first use.
- **`force_mode` parameter** on every NLP pipeline so tests/CI can deterministically pick backend.

---

## [0.1.1] — 2026-07-05 — Project Pivot to OSCAR

---

## [0.1.1] — 2026-07-05 — Project Pivot to OSCAR

### Changed
- **Project renamed**: `SENTINEL` (5-in-1 platform) → **`OSCAR` (OSINT dashboard, single focus)**.
- **Scope narrowed** to OSINT Threat & Sentiment Dashboard (News + Social Signals) per user-confirmed selection.
- Source datasets locked: **GDELT Project, NewsAPI, Reddit RSS** (free, robust, 45-day timeline safe).
- Removed from scope (deferred to post-v1.0): satellite change detection, full maritime module, knowledge graph visualization (light version retained as optional), disinformation classifier.
- Engineering handbook (`AGENT.md`) and roadmap (`TODO.md`) updated to reflect focused OSINT build.

### Notes
- Pivot reason: User confirmed `#2 OSINT Threat & Sentiment Dashboard (News + Social Signals)` as best fit for 45-day BSERC internship — maximum demo impact, laptop-runnable, defense-relevant, clear ML depth.

---

## [0.1.0] — 2026-07-05 — Workspace Initialization

### Added
- Repository folder structure (Clean Architecture layout).
- `AGENT.md` — engineering handbook.
- `PROJECT_STATUS.md` — single source of truth for project state.
- `TODO.md` — engineering roadmap.
- `README.md` — project entry point.
- `CHANGELOG.md` — this file.
- Placeholder directories: `src/{ingestion,nlp,ml,graph,geo,persistence,observability,api}`, `dashboard/{pages,assets}`, `data/{raw,processed,external}`, `models/{experiments,checkpoints,registry}`, `notebooks`, `tests`, `configs`, `docs`, `scripts`, `.github/workflows/`.

### Notes
- Sprint 0 in progress.

---

## Versioning Policy

- **Major (X.0.0):** breaking architecture changes, dependency upgrades that require migration.
- **Minor (0.X.0):** new modules, new ingestors, new models, new dashboard pages.
- **Patch (0.0.X):** bug fixes, performance improvements, docs, internal refactors.

## Change Categories

- **Feature** — user-visible functionality.
- **Bug Fix** — corrects behavior.
- **Refactor** — internal change, no behavior delta.
- **Breaking Changes** — requires action from users/contributors.
- **Documentation** — docs only.
- **ML** — model-related changes (new model, retrain, schema change).
- **Security** — security-related fix or hardening.
- **Deprecated** — to-be-removed feature.

---

**Last Updated:** 2026-07-05