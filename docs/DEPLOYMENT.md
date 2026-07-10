# OSCAR — Deployment Guide

This guide covers three deployment scenarios:

1. **Local development** — `streamlit run dashboard/app.py`
2. **Docker local stack** — `docker compose up`
3. **Public cloud deployment** — Streamlit Cloud / HuggingFace Spaces / AWS

---

## 1. Local Development (laptop)

```bash
# Setup
git clone <repo> oscar
cd oscar
python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell on Windows
# source .venv/bin/activate        # bash on Linux/macOS

pip install -r requirements-dev.txt
pre-commit install

# Configure
cp .env.example .env
# Edit .env: NEWS_API_KEY=<your free key from https://newsapi.org/register>

# Seed demo data (30-day snapshot, ~5K rows total)
python scripts/seed_demo.py

# Run dashboard
streamlit run dashboard/app.py
# Open http://localhost:8501
```

---

## 2. Docker Local Stack

```bash
# Build + run full stack (app + MLflow)
docker compose up --build

# App:        http://localhost:8501
# MLflow UI:  http://localhost:5000

# View logs
docker compose logs -f app

# Stop
docker compose down
```

The stack includes:
- **oscar-app** — Streamlit dashboard (port 8501)
- **oscar-mlflow** — MLflow tracking server (port 5000)
- **Volumes**: oscar-data (SQLite + Parquet), oscar-models, oscar-logs (persistent across rebuilds)

Health-check endpoint:
```bash
curl http://localhost:8501/_stcore/health
```

---

## 3. Public Cloud Deployment

### Option A: Streamlit Cloud (easiest, free hobby tier)

1. Push your fork to GitHub
2. Sign in at https://share.streamlit.io with GitHub
3. Click "New app" → select your repo
4. Set main file path: `dashboard/app.py`
5. Add secrets in "Advanced settings":
   - `NEWS_API_KEY`: your NewsAPI key
   - `DATABASE_URL`: leave default (SQLite persistent in app)
6. Deploy

Streamlit Cloud auto-installs from `requirements.txt` and runs your app.

### Option B: HuggingFace Spaces (better for ML demos)

1. Create a new Space at https://huggingface.co/spaces
2. Select "Streamlit" as SDK
3. Push your repo (or specific files)
4. Add secrets in Space settings
5. App is live at `https://huggingface.co/spaces/<user>/<space>`

### Option C: AWS / GCP / Azure (production)

For production-grade deployment:

```yaml
# docker-compose.prod.yml
services:
  app:
    image: ghcr.io/yourname/oscar:latest
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    environment:
      DATABASE_URL: postgresql://user:pass@db:5432/oscar
      NEWS_API_KEY: ${NEWS_API_KEY}
      MLFLOW_TRACKING_URI: http://mlflow:5000
    depends_on:
      - db
      - mlflow

  db:
    image: postgres:16
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: oscar
      POSTGRES_USER: oscar
      POSTGRES_PASSWORD: ${DB_PASSWORD}

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    volumes:
      - mlflow:/mlflow
```

Then push to ECR / GCR / ACR and deploy via ECS / Cloud Run / AKS.

---

## 4. Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NEWS_API_KEY` | For NewsAPI | Free key at https://newsapi.org/register |
| `DATABASE_URL` | No (default sqlite) | SQLAlchemy URL. e.g. `postgresql://user:pass@host:5432/oscar` |
| `APP_ENV` | No | `dev` / `staging` / `prod` / `test` / `ci` |
| `LOG_LEVEL` | No | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `STREAMLIT_THEME_BASE` | No | `light` / `dark` |
| `GDELT_BATCH_HOURS_BACK` | No | Hours of GDELT history to ingest (default 24) |
| `MLFLOW_TRACKING_URI` | No | MLflow server (e.g. `http://mlflow:5000`) |
| `MLFLOW_EXPERIMENT_NAME` | No | Default `oscar` |
| `HF_HOME` | For HF Spaces | HuggingFace cache dir |

---

## 5. Database Backends

### SQLite (default, laptop / small deployments)

```bash
DATABASE_URL=sqlite:///data/oscar.db
```

- Zero setup
- Single-file DB
- Suitable for ~100k events
- Backups: just copy the file

### PostgreSQL (production / multi-user)

```bash
# Add to docker-compose.yml:
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: oscar
      POSTGRES_USER: oscar
      POSTGRES_PASSWORD: secret
    volumes:
      - pgdata:/var/lib/postgresql/data

  app:
    environment:
      DATABASE_URL: postgresql://oscar:secret@db:5432/oscar
```

The SQLAlchemy ORM is portable; the existing schema is generated automatically. Production use should add Alembic migrations (see `docs/adr/` for the planned ADRs).

---

## 6. MLflow Setup (production)

For local: the JSON file backend works out of the box.

For production:

```bash
# Start MLflow server
docker run -d --name oscar-mlflow \
  -p 5000:5000 \
  -v mlflow-data:/mlflow \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server \
    --host 0.0.0.0 \
    --backend-store-uri sqlite:///mlflow/mlflow.db \
    --default-artifact-root /mlflow/artifacts

# Set in app env:
MLFLOW_TRACKING_URI=http://oscar-mlflow:5000
```

---

## 7. CI/CD (GitHub Actions example)

`.github/workflows/cd.yml` is configured to deploy on tagged releases:

```yaml
on:
  push:
    tags: [ 'v*' ]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build & push image
        run: docker build -t ghcr.io/${{ github.repository }}:${{ github.ref_name }} .
      - name: Deploy
        run: echo "Deploy to your platform here"
```

For Streamlit Cloud, the auto-deploy is built-in.

---

## 8. Monitoring

### Health check
Streamlit exposes `/_stcore/health` for load balancer probes:

```bash
curl http://localhost:8501/_stcore/health
# {"status": "ok"}
```

### Logs
- **App logs**: structured JSON to stdout (Loguru) + `logs/oscar.log`
- **MLflow**: tracking UI at port 5000
- **DB**: query `events`, `articles` row counts via dashboard

### Metrics to monitor
- Ingestion rate (events / min)
- Dashboard p99 latency
- Model inference time
- Anomaly count drift

---

## 9. Backup & Recovery

```bash
# Backup everything
docker compose exec app tar czf /tmp/oscar-backup.tar.gz /data /app/models

# Restore
docker compose cp /tmp/oscar-backup.tar.gz app:/tmp/
docker compose exec app tar xzf /tmp/oscar-backup.tar.gz -C /
```

Schedule nightly backups via cron:

```cron
0 3 * * * /usr/local/bin/oscar-backup.sh
```

---

## 10. Security Checklist

- [ ] All secrets in environment variables (never committed)
- [ ] HTTPS enabled at the load balancer / API gateway
- [ ] Database credentials rotated quarterly
- [ ] Container images scanned with Trivy
- [ ] Dependencies scanned with `pip-audit` (in CI)
- [ ] NewsAPI key scoped to read-only
- [ ] Rate limits on dashboard (Streamlit proxies)
- [ ] Read-only filesystem where possible
- [ ] Non-root user in Dockerfile (TODO: add USER directive)

---

## 11. Troubleshooting

### "no such table: events"
→ Run `python scripts/seed_demo.py` or `python -m src.ingestion.cli refresh --source gdelt`

### "401 Unauthorized" from NewsAPI
→ Verify `NEWS_API_KEY` is set correctly in `.env`

### "module not found"
→ Run `pip install -r requirements-dev.txt`

### Dashboard is blank
→ Check `logs/oscar.log` for errors; verify DB has data via Home page metrics

### docker compose fails
→ Check `docker compose logs app`; verify `/data` volume is writable

---

**Need help?** Open an issue at https://github.com/amanraj74/-OSCAR---Analysis-Reporting-AI-Based-Military-Intelligence-/issues