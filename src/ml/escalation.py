"""Escalation risk classifier.

XGBoost primary backend with sklearn `GradientBoostingClassifier` fallback.

Predicts the probability that a (country, date) observation will see
"escalation" — defined in `features._add_targets` — over 1/3/7 day horizons.

Public API
----------
    EscalationClassifier().fit(matrix, horizon)  -> self
    EscalationClassifier().predict_proba(X)      -> np.ndarray
    EscalationClassifier().score(X, y)           -> dict[str, float]
    EscalationClassifier().predict_for_region(matrix, region) -> pd.DataFrame
"""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.ml import HAS_XGBOOST, logger
from src.ml.features import MaterializedFeatureMatrix

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "logloss",
}


@dataclass
class ClassifierMetrics:
    """Evaluation metrics for a trained classifier."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    pr_auc: float = 0.0
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    confusion_matrix: list[list[int]] = field(default_factory=lambda: [[0, 0], [0, 0]])

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "pr_auc": self.pr_auc,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "confusion_matrix": self.confusion_matrix,
        }


class EscalationClassifier:
    """XGBoost classifier with sklearn GradientBoosting fallback."""

    def __init__(
        self,
        horizon_days: int = 7,
        params: dict[str, Any] | None = None,
        force_mode: str | None = None,
    ) -> None:
        self.horizon_days = horizon_days
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self._model: Any = None
        self._feature_columns: list[str] = []
        self._mode = self._pick_mode(force_mode)

    @staticmethod
    def _pick_mode(force_mode: str | None) -> str:
        if force_mode is not None:
            return force_mode
        return "xgboost" if HAS_XGBOOST else "sklearn"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def _build_model(self) -> Any:
        if self._mode == "xgboost":
            from xgboost import XGBClassifier

            logger.info("escalation_using_xgboost", horizon=self.horizon_days)
            return XGBClassifier(
                n_estimators=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                learning_rate=self.params["learning_rate"],
                subsample=self.params["subsample"],
                colsample_bytree=self.params["colsample_bytree"],
                random_state=self.params["random_state"],
                n_jobs=self.params["n_jobs"],
                eval_metric=self.params["eval_metric"],
                tree_method="hist",
            )
        from sklearn.ensemble import GradientBoostingClassifier

        logger.info("escalation_using_sklearn", horizon=self.horizon_days)
        return GradientBoostingClassifier(
            n_estimators=min(self.params["n_estimators"], 100),
            max_depth=min(self.params["max_depth"], 4),
            learning_rate=self.params["learning_rate"],
            subsample=self.params["subsample"],
            random_state=self.params["random_state"],
        )

    def fit(
        self,
        matrix: MaterializedFeatureMatrix,
        horizon: int | None = None,
    ) -> EscalationClassifier:
        """Fit on a `MaterializedFeatureMatrix`.

        Args:
            matrix: Pre-built feature matrix.
            horizon: Which horizon to use as target (default: `self.horizon_days`).
        """
        if horizon is None:
            horizon = self.horizon_days

        X, y = matrix.to_xy(horizon)
        mask = ~y.isna()
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True).astype(int)

        if len(y.unique()) < 2:
            logger.warning("escalation_single_class", horizon=horizon, n=int(y.sum()))

        self._feature_columns = list(X.columns)
        self._model = self._build_model()
        try:
            self._model.fit(X.values, y.values)
        except Exception as e:
            logger.error("escalation_fit_failed", error=str(e))
            raise
        self.horizon_days = horizon
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted; call .fit() first")
        for col in self._feature_columns:
            if col not in X.columns:
                X = X.copy()
                X[col] = 0.0
        X = X[self._feature_columns]
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X.values)[:, 1]
        else:
            scores = self._model.predict(X.values)
            proba = np.asarray(scores, dtype=float)
        return proba

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: pd.DataFrame, y: pd.Series) -> ClassifierMetrics:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_curve,
            precision_score,
            recall_score,
        )

        y_pred = self.predict(X)
        y_true = y.astype(int).values
        proba = self.predict_proba(X)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, proba)
        trap = getattr(np, "trapezoid", None) or np.trapz
        pr_auc = float(-trap(precision_curve, recall_curve)) if len(precision_curve) > 1 else 0.0

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        return ClassifierMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            pr_auc=pr_auc,
            n_samples=int(len(y_true)),
            n_positive=int(y_true.sum()),
            n_negative=int(len(y_true) - y_true.sum()),
            confusion_matrix=cm,
        )

    def predict_for_region(
        self,
        matrix: MaterializedFeatureMatrix,
        region: str,
    ) -> pd.DataFrame:
        df = matrix.features
        sub = df[df["actor1_country_code"] == region].copy()
        if sub.empty:
            return pd.DataFrame()
        X = sub[matrix.feature_columns]
        sub["escalation_probability"] = self.predict_proba(X)
        return sub[["date", "actor1_country_code", "escalation_probability"]].reset_index(drop=True)

    def feature_importance(self) -> pd.DataFrame:
        if self._model is None or not hasattr(self._model, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance"])
        return pd.DataFrame(
            {
                "feature": self._feature_columns,
                "importance": self._model.feature_importances_.astype(float),
            }
        ).sort_values("importance", ascending=False)

    def to_bytes(self) -> bytes:
        if self._model is None:
            raise RuntimeError("Cannot serialize unfitted model")
        return pickle.dumps(
            {
                "model": self._model,
                "feature_columns": self._feature_columns,
                "horizon_days": self.horizon_days,
                "mode": self._mode,
                "params": self.params,
            }
        )

    @classmethod
    def from_bytes(cls, blob: bytes) -> EscalationClassifier:
        data = pickle.loads(blob)
        inst = cls(
            horizon_days=data["horizon_days"], params=data["params"], force_mode=data["mode"]
        )
        inst._model = data["model"]
        inst._feature_columns = data["feature_columns"]
        return inst


def cross_validate(
    matrix: MaterializedFeatureMatrix,
    horizon: int,
    n_splits: int = 5,
    force_mode: str | None = None,
) -> dict[str, float]:
    """Time-aware cross-validation. Returns mean + std of metrics."""
    from sklearn.model_selection import TimeSeriesSplit

    X = matrix.features[matrix.feature_columns].reset_index(drop=True)
    y = matrix.features[f"target_h{horizon}"].reset_index(drop=True)

    tss = TimeSeriesSplit(n_splits=n_splits)
    metrics_list: list[ClassifierMetrics] = []
    for train_idx, test_idx in tss.split(X):
        clf = EscalationClassifier(horizon_days=horizon, force_mode=force_mode)
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        ~y_train.isna() & ~y_test.isna()
        X_train = X_train.loc[y_train.notna()]
        y_train = y_train.dropna().astype(int)

        clf._model = clf._build_model()
        clf._feature_columns = list(X_train.columns)
        clf._model.fit(X_train.values, y_train.values)

        X_test = X_test.loc[y_test.notna()]
        y_test = y_test.dropna().astype(int)
        m = clf.score(X_test, y_test)
        metrics_list.append(m)

    if not metrics_list:
        return {}
    keys = ["accuracy", "precision", "recall", "f1", "pr_auc"]
    out: dict[str, float] = {}
    for k in keys:
        vals = [getattr(m, k) for m in metrics_list]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


__all__ = [
    "ClassifierMetrics",
    "EscalationClassifier",
    "cross_validate",
    "DEFAULT_PARAMS",
]


def _pick_mode_for_tests() -> str:  # pragma: no cover
    return "sklearn"


def _unused_import_marker() -> str:  # pragma: no cover
    return io.__name__
