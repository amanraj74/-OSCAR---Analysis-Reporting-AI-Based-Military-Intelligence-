"""Machine Learning layer for OSCAR.

Submodules:
    features    — Feature engineering from silver Parquet tables.
    escalation  — XGBoost escalation classifier with sklearn fallback.
    forecast    — Prophet forecaster with linear-regression fallback.
    anomaly     — Isolation Forest detector with z-score fallback.
    tracking    — MLflow experiment tracking with JSON fallback.
    registry    — Model registry (promote/demote, versioned).

All ML models are lazy-loaded. Heavy deps (xgboost, prophet) have
lightweight sklearn / statsmodels / custom fallbacks so the pipeline
remains functional on minimal environments.

Usage::

    from src.ml import EscalationClassifier, Forecaster, AnomalyDetector
"""

from __future__ import annotations

import importlib
from typing import Any

from src.observability import get_logger

logger = get_logger("ml")


def is_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


HAS_XGBOOST = is_available("xgboost")
HAS_PROPHET = is_available("prophet")
HAS_MLFLOW = is_available("mlflow")


__all__ = ["is_available", "HAS_XGBOOST", "HAS_PROPHET", "HAS_MLFLOW"]


def __getattr__(name: str) -> Any:
    if name in {"features", "escalation", "forecast", "anomaly", "tracking", "registry"}:
        return importlib.import_module(f"src.ml.{name}")
    raise AttributeError(f"module 'src.ml' has no attribute {name!r}")
