"""About — Methodology, data sources, ethics, model cards."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))  # noqa: F401  bootstrap sys.path
import streamlit as st

from dashboard.utils import page_header, sidebar_status_panel

st.set_page_config(page_title="OSCAR · About", page_icon="ℹ️", layout="wide")
sidebar_status_panel()

page_header(
    "About",
    "Methodology, data sources, ethics, and model cards.",
)

st.markdown("""
## What is OSCAR?

**OSCAR** (**O**pen-**S**ource **C**onflict **A**nalysis & **R**eporting) is an
OSINT-Powered Threat & Sentiment Intelligence Dashboard built as part of the
**BSERC Def-Space Summer Internship 2026**.

It fuses open data — global event streams, news, public social signals — with
NLP and machine learning to surface **what's happening, where, and how the
discourse is shifting** — for defense and intelligence analysts.

---

## Data sources

| Source | Type | Auth | Cadence | License |
|---|---|---|---|---|
| GDELT Project 2.0 | Global event database | None | 15 min | Public domain |
| NewsAPI.org | News headlines + search | Free key | On-demand | Free-tier ToS |
| Reddit RSS | Public subreddit feeds | None | On-demand | Reddit ToS |

All sources are **open and free**. OSCAR stores aggregate / regional /
event-level information only. No individual-level data, no targeting.

---

## Architecture

```
src/
├── ingestion/   GDELT, NewsAPI, Reddit ingestors + idempotent persistence
├── nlp/         spaCy NER, DistilBERT sentiment, BERTopic, entity normalization
├── ml/          feature engineering, XGBoost classifier, Prophet forecaster,
│               Isolation Forest anomaly, MLflow tracking, model registry
├── transform/   Bronze → Silver (Parquet) data layer
└── persistence/ SQLAlchemy ORM (8 tables) + SQLite

dashboard/      Streamlit multi-page app (Map · Sentiment · Entities · Forecast · Alerts · About)
```

Each layer follows **Clean Architecture**: domain → application → infrastructure.

---

## Methodology

### Ingestion
Idempotent persistence via `(source, external_id)` upserts. Rate-limited,
retried, validated.

### NLP
- **NER** — spaCy `en_core_web_sm` + 33 custom WEAPON patterns + 13 MILITARY_ORG
  patterns. Regex fallback for offline use.
- **Sentiment** — `distilbert-base-uncased-finetuned-sst-2-english` with
  VADER fallback.
- **Topics** — BERTopic with sklearn TF-IDF + KMeans fallback.
- **Entity aliasing** — sentence-transformers cosine similarity,
  with difflib fallback.

### ML
- **Features** — lag (1/3/7d), rolling mean/std (3/7/14d), calendar.
- **Classifier** — XGBoost (or sklearn GradientBoosting fallback).
  Time-aware `TimeSeriesSplit` cross-validation.
- **Forecaster** — Prophet (or linear regression fallback). Per-region.
- **Anomaly** — sklearn IsolationForest (or rolling z-score fallback).

### Tracking & Registry
- **MLflow** for experiment tracking (or JSON file fallback).
- **Custom registry** — versioned, staged
  (None / Staging / Production / Archived). Promote/demote semantics with
  auto-archive of prior production.

---

## Ethics & limitations

OSCAR is built with strict ethical guardrails:

- Aggregate only — no individual-level targeting or surveillance.
- Reproducible — pinned seeds, hashes, deterministic training.
- Documented — every model has a card
  (purpose / metrics / limits / ethics).
- Open data — no classified feeds; aggregate / regional level.

**Coverage gaps** — GDELT is news-derived; non-Western conflicts under-represented.

**Concept drift** — geopolitical dynamics change; retrain quarterly.

**Class imbalance** — most country-days have `escalation=0`;
PR-AUC is more meaningful than accuracy.

**Out of scope** — no action recommendations for kinetic operations;
no targeting outputs.

OSCAR outputs are **advisory**. They support — not replace — human analyst
judgment.

---

## Model cards

| Model | Backend | Purpose | Card |
|-------|---------|---------|------|
| Escalation Classifier | XGBoost | Per-region 7-day escalation risk | [`docs/models/escalation_classifier.md`](https://github.com/amanraj74/-OSCAR---Analysis-Reporting-AI-Based-Military-Intelligence-/blob/main/docs/models/escalation_classifier.md) |
| Forecaster | Prophet | Per-region event-volume forecasts | [`docs/models/forecaster.md`](https://github.com/amanraj74/-OSCAR---Analysis-Reporting-AI-Based-Military-Intelligence-/blob/main/docs/models/forecaster.md) |
| Anomaly Detector | IsolationForest | Multi-feature regional anomalies | [`docs/models/anomaly_detector.md`](https://github.com/amanraj74/-OSCAR---Analysis-Reporting-AI-Based-Military-Intelligence-/blob/main/docs/models/anomaly_detector.md) |

---

## Reproducibility

```bash
# Recompute everything
python -m src.ingestion.cli refresh --source gdelt
python -m src.ingestion.cli refresh --source newsapi
python -m src.ingestion.cli refresh --source reddit
python -m src.ingestion.cli transform
python -m src.nlp.cli nlp-process
python -m src.ml.cli ml-train-all
```

All seeds are pinned (`random_state=42`). Every training run is logged via
MLflow (or JSON fallback).

---

## License

**MIT License.** See `LICENSE`.

---

## Acknowledgments

- **BSERC** (Bharat Space Education Research Centre) — Def-Space Summer Internship 2026.
- **GDELT Project**, **NewsAPI**, **Reddit** — open data providers.
- **Open-source community** — every library in `requirements.txt`.
""")

st.divider()
st.caption("Built for BSERC Def-Space Summer Internship 2026.")
