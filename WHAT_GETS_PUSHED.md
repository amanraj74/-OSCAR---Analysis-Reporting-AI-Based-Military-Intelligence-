# What gets pushed to GitHub — and what doesn't

## ✅ PUSHED (the public repo contains only these)

### Top-level files (16)
```
.dockerignore
.env.example          ← template, no real keys
.gitignore
.pre-commit-config.yaml
AGENT.md              ← engineering handbook
CHANGELOG.md
Dockerfile
LICENSE               ← MIT
Makefile
README.md             ← perfect, human, professional
docker-compose.yml
pyproject.toml
requirements-dev.txt
requirements.txt
```

### Source code (29 files)
```
src/
├── cli.py                         ← unified `oscar` CLI
├── config.py                      ← typed settings
├── __init__.py
├── domain/__init__.py + schemas.py
├── geo/         (empty, gitignored placeholder)
├── graph/       (empty, gitignored placeholder)
├── data/        (empty, gitignored placeholder)
├── ingestion/   base, gdelt, newsapi, reddit, acled, wikipedia, openweather, alphavantage, cli, __init__
├── ml/          features, escalation, forecast, anomaly, tracking, registry, cli, __init__
├── models/      ← gitignored (MLflow experiments + checkpoints)
├── nlp/         ner, sentiment, topics, normalize, cli, __init__
├── observability/  logging, __init__
├── persistence/    database, models, __init__
└── transform/      silver, __init__
```

### Dashboard (15 files)
```
dashboard/
├── app.py                  ← landing page
├── utils.py                ← cached loaders, UI primitives
├── __init__.py
├── assets/style.css
└── pages/  0_Home, 1_Map, 2_Sentiment, 3_Entities, 4_Forecast,
            5_Alerts, 6_About, 7_Encyclopedia, 8_Weather, 9_Markets
```

### Tests (24 files)
```
tests/
├── conftest.py
├── __init__.py
├── unit/   21 test files, 208 tests
└── e2e/    1 file, 5 tests
```

### Configs / docs / scripts (15 files)
```
configs/settings.yaml
configs/logging.yaml
docs/REPORT.md
docs/DEPLOYMENT.md
docs/PROJECT_LISTING.md
docs/BSERC_FORM_ANSWERS.md
docs/CHANGELOG.md          (no — CHANGELOG is at root)
docs/demo/video_script.md
docs/models/{anomaly_detector,escalation_classifier,forecaster}.md
scripts/  dev.py, seed_demo.py, run_full_training.py, run_final_checks.py,
          debug_alerts.py, debug_newsapi.py, debug_settings.py,
          download_gdelt_kaggle.py, fetch_gdelt_bulk.py, fetch_historical_gdelt.py,
          load_gdelt_bulk.py, test_newsapi.py, _bootstrap.py
notebooks/README.md
.github/workflows/ci.yml
```

### Total size on disk
~**5–10 MB** of pure source + docs. Pushes in seconds.

---

## ❌ NOT PUSHED (gitignored)

| What | Why | Size |
|---|---|---|
| `.venv/` | Python virtualenv | ~2 GB |
| `data/oscar.db` | SQLite database with 370k events | 135 MB |
| `data/processed/*.parquet` | Generated silver tables | < 1 MB |
| `data/external/gdelt_bulk/*.csv` | Raw GDELT bulk files | ~120 MB |
| `data/test_e2e*.db` | E2E test artifacts | < 1 MB |
| `models/checkpoints/*.pkl` | Trained model artifacts | < 1 MB |
| `models/experiments/` | MLflow runs | < 5 MB |
| `src/models/{checkpoints,experiments,registry}/` | Local MLflow store | < 5 MB |
| `mlruns/`, `mlflow.db` | Local MLflow | varies |
| `logs/` | Runtime logs | varies |
| `.env` | **Real API keys** — never push secrets! | 1 KB |
| `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` | Build cache | varies |
| `htmlcov/`, `.coverage`, `coverage.xml` | Coverage reports | varies |
| `.idea/`, `.vscode/` | Editor state | varies |
| `*.log` | Runtime logs | varies |

Anyone who clones your repo will:
- Get a working pipeline on day 1
- Run `python scripts/seed_demo.py` to populate a demo DB
- Then run `python -m src.cli dashboard` to launch the UI

No secrets, no gigabytes of data, no model binaries — just clean, runnable code.