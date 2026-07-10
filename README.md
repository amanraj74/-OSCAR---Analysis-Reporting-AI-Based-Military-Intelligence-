# OSCAR

**Open-Source Conflict Analysis & Reporting**

An OSINT-powered threat and sentiment intelligence dashboard. OSCAR pulls open data — global events, news, social signals — and turns it into actionable insight with NLP and machine learning. Everything runs locally on a laptop. No paid APIs required to get started.

![status](https://img.shields.io/badge/status-v0.1.1-blue) ![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![tests](https://img.shields.io/badge/tests-213%20passed-brightgreen) ![coverage](https://img.shields.io/badge/coverage-68%25-yellow)

---

## What it does

OSCAR watches the world so you don't have to. It:

- Pulls **global events** from GDELT (every 15 minutes), news from NewsAPI, public Reddit threads, weather, markets, and encyclopedia lookups
- Extracts **who, what, where** from unstructured text — organisations, places, persons, weapons — with spaCy + custom patterns
- Scores **sentiment** (DistilBERT or VADER), **escalation risk** (XGBoost / sklearn), and **anomalies** (Isolation Forest)
- Forecasts **7-day conflict intensity** per region (Prophet / linear baseline)
- Ships a **10-page interactive dashboard** (Streamlit + Plotly + Folium/PyDeck) where you can filter by country, day, actor, or topic

It is built for analysts who want defence-relevant signals across 50+ countries without spinning up a paid intelligence platform.

---

## Screenshots

> Run `streamlit run dashboard/app.py` and open `http://localhost:8501`. The four flagship views are the **Map**, **Forecast**, **Alerts**, and **Entities** pages.

---

## Quick start

### 1. Clone and set up

```bash
git clone https://github.com/amanraj74/-OSCAR---Analysis-Reporting-AI-Based-Military-Intelligence-.git
cd -OSCAR---Analysis-Reporting-AI-Based-Military-Intelligence-
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your API keys (NewsAPI is the only one needed for the demo)
```

Keys you can set (all optional except `NEWS_API_KEY` for live news):

| Variable | What it does | Where to get it |
|---|---|---|
| `NEWS_API_KEY` | News headlines | [newsapi.org](https://newsapi.org/register) (free) |
| `OPENWEATHER_API_KEY` | Weather for cities | [openweathermap.org](https://home.openweathermap.org/users/sign_up) (free) |
| `ALPHA_VANTAGE_API_KEY` | Market data | [alphavantage.co](https://www.alphavantage.co/support/#api-key) (free) |
| `ACLED_USERNAME` / `ACLED_PASSWORD` | Armed conflict data | [acleddata.com](https://acleddata.com/registration/) (free academic) |

Without any keys you can still run the full pipeline on **seeded demo data**.

### 3. Try it (no real API needed)

```bash
# Seed a 30-day demo dataset (no network calls)
python scripts/seed_demo.py

# Build the silver layer (parquet tables from DB)
python -m src.cli ingest transform

# Run NLP (NER + sentiment + topics)
python -m src.cli nlp nlp-process

# Run ML (features + escalation + forecasts + anomalies)
python -m src.cli ml ml-train-all

# Launch the dashboard
python -m src.cli dashboard
# or: streamlit run dashboard/app.py
```

Open **http://localhost:8501** and explore.

### 4. Try it with real APIs

```bash
# Pull fresh data
python -m src.cli ingest refresh --source gdelt
python -m src.cli ingest refresh --source newsapi
python -m src.cli ingest refresh --source reddit

# Then build silver + NLP + ML
python -m src.cli ingest transform
python -m src.cli nlp nlp-process
python -m src.cli ml ml-train-all
```

---

## The dashboard

Ten pages, all keyboard-navigable, all cached for 5 minutes.

| Page | What it shows |
|---|---|
| **Home** | KPI strip (events, articles, entities, anomalies) and quick links |
| **Map** | World choropleth of sentiment + geo-located GDELT events |
| **Sentiment** | Daily trends + top positive / negative articles |
| **Entities** | Trending orgs, weapons, locations, persons |
| **Forecast** | 7-day risk forecasts per region with confidence bands |
| **Alerts** | Live anomaly feed with severity scoring |
| **Encyclopedia** | Wikipedia summaries for trending topics |
| **Weather** | Current weather and 5-day forecast for capital cities |
| **Markets** | Live equity / FX data with conflict overlay |
| **About** | System info, version, and quick setup instructions |

---

## Architecture

Clean architecture: eight layers, each one replaceable. Config → Ingestion → Transform (silver) → NLP → Features → ML → Tracking → Dashboard.

```
┌──────────────────────────────────────────────────────────────┐
│                  DASHBOARD (Streamlit)                       │
│   src/cli.py  ─►  oscar … dashboard                          │
├──────────────────────────────────────────────────────────────┤
│                          ML                                  │
│   features · escalation · forecast · anomaly · registry       │
│   tracking (MLflow or JSON)                                  │
├──────────────────────────────────────────────────────────────┤
│                          NLP                                 │
│   ner (spaCy) · sentiment (DistilBERT/VADER)                 │
│   topics (BERTopic) · normalize                              │
├──────────────────────────────────────────────────────────────┤
│                       TRANSFORM                              │
│   silver parquet tables (events_per_country_day, …)          │
├──────────────────────────────────────────────────────────────┤
│                      INGESTION                               │
│   gdelt · newsapi · reddit · acled · wikipedia               │
│   openweather · alphavantage · base                          │
├──────────────────────────────────────────────────────────────┤
│                     PERSISTENCE                              │
│   SQLAlchemy + SQLite (Postgres-compatible)                  │
│   8 tables: events · articles · entities · …                 │
├──────────────────────────────────────────────────────────────┤
│                       CONFIG                                 │
│   Pydantic v2 settings · YAML overrides · .env               │
└──────────────────────────────────────────────────────────────┘
```

### Key design choices

- **Pydantic v2 schemas** at every parser boundary — invalid data fails fast and loud
- **Force-mode pattern** for every ML model (`sklearn`, `xgboost`, `prophet`, `iforest`, `zscore`, `linear`) so you can always fall back to a lightweight option
- **Idempotent persistence** — re-ingesting the same data updates rather than duplicates
- **Reproducible** — every random seed pinned, every ML run logged to MLflow *or* a JSON registry (no MLflow server required)
- **Graceful degradation** — if a model can't load, the dashboard still works

---

## Tech stack

| Layer | Library |
|---|---|
| Web framework | Streamlit ≥ 1.36 |
| Charts | Plotly, Folium, PyDeck, Matplotlib, Seaborn |
| Data | Pandas, NumPy, PyArrow |
| NLP | spaCy (en_core_web_sm), DistilBERT via HuggingFace Transformers, BERTopic, NLTK VADER |
| ML | scikit-learn, XGBoost, Prophet (optional), statsmodels |
| Deep learning | PyTorch (≥ 2.3), sentence-transformers |
| Persistence | SQLAlchemy 2.0, SQLite (default) / Postgres (compatible) |
| Tracking | MLflow (optional, JSON fallback included) |
| HTTP | httpx, requests, feedparser, BeautifulSoup4, lxml |
| Config | Pydantic v2, pydantic-settings, PyYAML, python-dotenv |

---

## Project layout

```
.
├── README.md                  ← you are here
├── LICENSE                    ← MIT
├── CHANGELOG.md
├── pyproject.toml             ← Python 3.10–3.12, all deps pinned
├── requirements.txt           ← runtime
├── requirements-dev.txt       ← dev + test
├── Makefile                   ← make dashboard, make test, …
├── scripts/dev.py             ← cross-platform wrapper for Make targets
│
├── src/
│   ├── cli.py                 ← `oscar` unified CLI entry point
│   ├── config.py              ← typed settings (Pydantic v2)
│   ├── domain/                ← pydantic schemas
│   ├── persistence/           ← SQLAlchemy models + DB engine
│   ├── ingestion/             ← 7 source-specific ingestors
│   ├── transform/             ← silver-layer builders (parquet)
│   ├── nlp/                   ← NER, sentiment, topics, normalize
│   ├── ml/                    ← features, escalation, forecast, anomaly
│   ├── observability/         ← structured logging
│   └── models/                ← registry + MLflow JSON fallback
│
├── dashboard/
│   ├── app.py                 ← landing page
│   ├── utils.py               ← cached data loaders, UI primitives
│   ├── assets/style.css
│   └── pages/                 ← 10 Streamlit pages
│       ├── 0_Home.py
│       ├── 1_Map.py
│       ├── 2_Sentiment.py
│       └── …
│
├── tests/
│   ├── unit/                  ← 21 files, 208 tests
│   ├── e2e/                   ← 1 file, 5 smoke tests
│   └── conftest.py
│
├── configs/                   ← settings.yaml, logging.yaml
├── docs/                      ← REPORT.md, DEPLOYMENT.md, …
├── data/                      ← SQLite DB + parquet (gitignored)
├── models/                    ← trained checkpoints + experiments
├── logs/
└── notebooks/
```

---

## Testing

```bash
# All unit + e2e tests (fast, ~1 min)
python -m pytest tests/ -m "e2e or (not slow and not integration and not e2e)"

# With coverage report
python -m pytest --cov=src --cov-report=term-missing

# Run a single test
python -m pytest tests/unit/test_gdelt.py -v
```

The configured threshold is 20% coverage but the actual is **68.59%** with 213 passing tests.

---

## CLI

OSCAR ships a single `oscar` command:

```bash
oscar --version                 # show version
oscar --help                    # show top-level commands
oscar ingest refresh --source gdelt
oscar ingest transform
oscar nlp ner-process
oscar nlp sentiment-score
oscar nlp topics-discover --n-topics 8
oscar nlp nlp-process           # all NLP in one shot
oscar ml build-features
oscar ml train-escalation --horizon 7
oscar ml forecast --regions UKR,RUS --periods 7
oscar ml detect-anomalies --window 14
oscar ml promote --name escalation_h7 --version 1
oscar ml list-models
oscar ml ml-train-all           # everything in one shot
oscar dashboard                 # launch Streamlit
```

Or use the included `Makefile` on Linux/macOS — `make dashboard`, `make test`, `make lint`, etc. On Windows, `python scripts/dev.py dashboard` does the same thing.

---

## Deployment

Three options, documented in `docs/DEPLOYMENT.md`:

1. **Streamlit Community Cloud** — push to GitHub, click "Deploy", done
2. **Docker Compose** — `docker compose up --build` brings up the full stack (app + MLflow)
3. **Hugging Face Spaces** — upload `Dockerfile`, point at port 8501

Postgres is a drop-in replacement for SQLite: set `DATABASE_URL=postgresql://user:pass@host:5432/oscar`.

---

## Contributing

Issues and pull requests welcome. The contract for code style, test coverage, and architecture is in `AGENT.md`. The full engineering handbook.

```bash
# Lint + format
python scripts/dev.py lint
python scripts/dev.py format
```

---

## License

MIT — see `LICENSE`.

This is a BSERC Def-Space Summer Internship 2026 deliverable. The codebase, trained models, and documentation are released for collaboration and adoption.

---

## Credits

- **GDELT Project** — global event database
- **NewsAPI** — news headlines
- **Reddit RSS** — public subreddit feeds
- **Wikipedia REST API** — encyclopedia lookups
- **spaCy · HuggingFace · BERTopic** — open NLP models
- **scikit-learn · XGBoost · Prophet · PyTorch** — open ML stack

---

**Contact:** See `docs/PROJECT_LISTING.md` for submission details, or open an issue on GitHub.
