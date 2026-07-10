"""Anomaly detection for OSCAR.

Isolation Forest primary backend (sklearn) with rolling z-score fallback.
Detects per-region anomalies in event volume, conflict events, sentiment shifts,
and entity mentions.

Public API
----------
    AnomalyDetector().detect(series, region, ...) -> list[Anomaly]
    ZScoreDetector().detect(...)                  -> list[Anomaly] (fallback)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_settings
from src.observability import get_logger

logger = get_logger("ml.anomaly")


@dataclass
class Anomaly:
    """A single detected anomaly."""

    region: str
    date: str
    anomaly_type: str
    severity: float
    score: float
    description: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime | None = None

    def to_db_row(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "date": self.date,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "score": self.score,
            "description": self.description,
            "context": self.context,
        }


class IsolationForestDetector:
    """Sklearn Isolation Forest detector."""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        from sklearn.ensemble import IsolationForest

        self.contamination = contamination
        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X: np.ndarray) -> None:
        if len(X) < 5:
            return
        self._model.fit(X)
        self._fitted = True

    def detect(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        region_col: str = "actor1_country_code",
        date_col: str = "date",
        anomaly_type: str = "iForest",
    ) -> list[Anomaly]:
        if not self._fitted:
            self.fit(df[feature_cols].values)
        scores = -self._model.score_samples(df[feature_cols].values)
        preds = self._model.predict(df[feature_cols].values)

        out: list[Anomaly] = []
        for i, (pred, score) in enumerate(zip(preds.tolist(), scores.tolist(), strict=False)):
            if pred == -1:
                out.append(
                    Anomaly(
                        region=str(df.iloc[i][region_col]),
                        date=str(df.iloc[i][date_col])[:10],
                        anomaly_type=anomaly_type,
                        severity=min(1.0, abs(float(score)) / 2.0),
                        score=float(score),
                        description=f"Isolation Forest flagged (score={score:.2f})",
                        context={
                            col: float(df.iloc[i][col]) for col in feature_cols if col in df.columns
                        },
                    )
                )
        return out


class ZScoreDetector:
    """Rolling z-score detector (always-available)."""

    def __init__(self, window: int = 14, threshold: float = 2.5) -> None:
        self.window = window
        self.threshold = threshold

    def detect(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        region_col: str = "actor1_country_code",
        date_col: str = "date",
        anomaly_type: str = "zscore",
    ) -> list[Anomaly]:
        out: list[Anomaly] = []
        for col in feature_cols:
            if col not in df.columns:
                continue
            df_sorted = df.sort_values([region_col, date_col]).copy()
            grouped = df_sorted.groupby(region_col)[col]
            roll_mean = grouped.transform(lambda s: s.rolling(self.window, min_periods=1).mean())
            roll_std = grouped.transform(lambda s: s.rolling(self.window, min_periods=2).std())
            z = ((df_sorted[col] - roll_mean) / roll_std.replace(0, np.nan)).fillna(0.0)

            for i, score in enumerate(z.values):
                if abs(float(score)) >= self.threshold:
                    out.append(
                        Anomaly(
                            region=str(df_sorted.iloc[i][region_col]),
                            date=str(df_sorted.iloc[i][date_col])[:10],
                            anomaly_type=f"{anomaly_type}:{col}",
                            severity=min(1.0, abs(float(score)) / 5.0),
                            score=float(score),
                            description=f"{col} z-score={score:.2f} (>{self.threshold}σ)",
                            context={"column": col, "value": float(df_sorted.iloc[i][col])},
                        )
                    )
        return out


class AnomalyDetector:
    """Unified anomaly detector. Isolation Forest primary + Z-score fallback."""

    def __init__(
        self,
        contamination: float | None = None,
        window: int = 14,
        threshold: float = 2.5,
        force_mode: str | None = None,
    ) -> None:
        if contamination is None:
            try:
                contamination = get_settings().ml.anomaly_contamination
            except Exception:
                contamination = 0.05
        self.contamination = contamination
        self.window = window
        self.threshold = threshold
        self._mode = force_mode or "iforest"
        self._iforest: IsolationForestDetector | None = None
        self._zscore: ZScoreDetector | None = None

    @property
    def mode(self) -> str:
        return self._mode

    def _ensure(self) -> None:
        if self._mode == "iforest" and self._iforest is None:
            try:
                self._iforest = IsolationForestDetector(contamination=self.contamination)
            except Exception as e:
                logger.warning("iforest_unavailable", error=str(e))
                self._mode = "zscore"
        if self._zscore is None:
            self._zscore = ZScoreDetector(window=self.window, threshold=self.threshold)

    def detect(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        region_col: str = "actor1_country_code",
        date_col: str = "date",
        anomaly_type: str | None = None,
    ) -> list[Anomaly]:
        self._ensure()
        if self._mode == "iforest" and self._iforest is not None:
            return self._iforest.detect(
                df, feature_cols, region_col, date_col, anomaly_type or "iForest"
            )
        return self._zscore.detect(  # type: ignore[union-attr]
            df, feature_cols, region_col, date_col, anomaly_type or "zscore"
        )

    def detect_per_region(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        region_col: str = "actor1_country_code",
        date_col: str = "date",
    ) -> dict[str, list[Anomaly]]:
        self._ensure()
        by_region: dict[str, list[Anomaly]] = {}
        for region, sub in df.groupby(region_col):
            sub_sorted = sub.sort_values(date_col).reset_index(drop=True)
            anoms = self.detect(sub_sorted, feature_cols, region_col=region_col, date_col=date_col)
            if anoms:
                by_region[str(region)] = anoms
        return by_region


__all__ = [
    "Anomaly",
    "AnomalyDetector",
    "IsolationForestDetector",
    "ZScoreDetector",
]
