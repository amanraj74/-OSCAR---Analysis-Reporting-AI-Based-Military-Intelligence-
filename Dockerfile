# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────
# OSCAR — AI-Based Military Intelligence Dashboard
# Multi-stage Dockerfile: slim runtime image, no build tools.
# ─────────────────────────────────────────────────────────────────────

# ─── Stage 1: builder — install full deps for compilation wheels ───
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# System deps for compiling wheels (numpy, pandas, scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# ─── Stage 2: runtime — minimal image, copy from builder ─────────
FROM python:3.11-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    LOG_LEVEL=INFO \
    DATABASE_URL=sqlite:////data/oscar.db

LABEL org.opencontainers.image.title="OSCAR" \
      org.opencontainers.image.description="OSINT-Powered Threat & Sentiment Intelligence Dashboard" \
      org.opencontainers.image.source="https://github.com/amanraj74/-OSCAR---Analysis-Reporting-AI-Based-Military-Intelligence-" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Minimal runtime libs (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY pyproject.toml README.md AGENT.md LICENSE ./
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY docs/ ./docs/

# Pre-create data + logs + models dirs (mount as volumes in compose)
RUN mkdir -p /data /logs /app/models

# Healthcheck via Streamlit's _stcore/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

# Default: launch Streamlit dashboard
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]