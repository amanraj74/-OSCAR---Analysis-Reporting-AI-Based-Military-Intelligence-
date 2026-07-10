"""Experiment tracking for OSCAR.

MLflow primary backend with local JSON file fallback. Every training run
gets an `EXP-<yyyymmdd>-<slug>` ID, with parameters, metrics, and artifacts
logged.

Public API
----------
    Tracker.start_run(experiment_name, run_name) -> RunContext
    RunContext.log_param(key, value) / .log_metric(key, value)
    RunContext.log_artifact(local_path, artifact_path)
"""

from __future__ import annotations

import json
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.ml import HAS_MLFLOW, logger

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _make_run_id(slug: str | None = None) -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    s = (slug or "run").lower()
    s = _SLUG_RE.sub("_", s).strip("_") or "run"
    return f"EXP-{date}-{s}-{uuid.uuid4().hex[:6]}"


@dataclass
class RunContext:
    """Active experiment run."""

    run_id: str
    experiment_name: str
    run_name: str
    started_at: datetime
    finished_at: datetime | None = None
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    artifacts: list[dict[str, str]] = field(default_factory=list)
    backend: str = "json"
    _run_dir: Path | None = None

    def log_param(self, key: str, value: Any) -> None:
        self.params[str(key)] = value if not isinstance(value, str | int | float | bool) else value

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if isinstance(value, int | float) and not isinstance(value, bool):
            self.metrics[str(key)] = float(value)
        elif isinstance(value, int | float | bool | str):
            self.metrics[str(key)] = value
        # else: skip non-scalar (lists, dicts) — they go in `tags` or as artifacts

    def log_metrics(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        src = Path(local_path)
        if not src.exists():
            logger.warning("artifact_missing", path=str(src))
            return
        self.artifacts.append({"source": str(src), "artifact_path": artifact_path or src.name})

    def set_tag(self, key: str, value: str) -> None:
        self.tags[str(key)] = str(value)

    def finish(self, status: str = "FINISHED") -> dict[str, Any]:
        self.finished_at = datetime.now(timezone.utc)
        self.tags.setdefault("status", status)
        return {
            "run_id": self.run_id,
            "experiment": self.experiment_name,
            "run_name": self.run_name,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "duration_seconds": (self.finished_at - self.started_at).total_seconds(),
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
            "artifacts": self.artifacts,
            "backend": self.backend,
        }


class _JsonBackend:
    """JSON-file based experiment tracker (always-available)."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def start(self, ctx: RunContext) -> None:
        run_dir = self.root_dir / ctx.experiment_name / ctx.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ctx._run_dir = run_dir

    def finish(self, ctx: RunContext) -> None:
        if ctx._run_dir is None:
            return
        meta = {
            "run_id": ctx.run_id,
            "experiment": ctx.experiment_name,
            "run_name": ctx.run_name,
            "started_at": ctx.started_at.isoformat(),
            "finished_at": ctx.finished_at.isoformat() if ctx.finished_at else None,
            "params": ctx.params,
            "metrics": ctx.metrics,
            "tags": ctx.tags,
            "artifacts": ctx.artifacts,
        }
        (ctx._run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        for name, value in ctx.metrics.items():
            (ctx._run_dir / f"metric_{name}.json").write_text(
                json.dumps({"name": name, "value": value}), encoding="utf-8"
            )

    def list_runs(self, experiment_name: str) -> list[Path]:
        exp_dir = self.root_dir / experiment_name
        if not exp_dir.exists():
            return []
        return sorted(p for p in exp_dir.iterdir() if p.is_dir())

    def load(self, experiment_name: str, run_id: str) -> dict[str, Any] | None:
        path = self.root_dir / experiment_name / run_id / "meta.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


class _MlflowBackend:
    """MLflow-based tracker (delegates to MLflow tracking API)."""

    def __init__(self, tracking_uri: str) -> None:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        self._mlflow = mlflow

    def start(self, ctx: RunContext) -> None:
        self._mlflow.set_experiment(ctx.experiment_name)
        self._mlflow.start_run(run_name=ctx.run_id)
        for k, v in ctx.params.items():
            self._mlflow.log_param(k, v)
        for k, v in ctx.tags.items():
            self._mlflow.set_tag(k, v)

    def finish(self, ctx: RunContext) -> None:
        for k, v in ctx.metrics.items():
            self._mlflow.log_metric(k, v)
        for art in ctx.artifacts:
            self._mlflow.log_artifact(art["source"], artifact_path=art.get("artifact_path"))
        self._mlflow.end_run()


class Tracker:
    """Unified experiment tracker with MLflow / JSON fallback."""

    def __init__(self, force_mode: str | None = None, root_dir: Path | None = None) -> None:
        settings = get_settings()
        if root_dir is None:
            root_dir = settings.models_dir / "experiments"
        self.root_dir = root_dir
        self._mode = force_mode or ("mlflow" if HAS_MLFLOW else "json")
        self._backend: _JsonBackend | _MlflowBackend | None = None

    @property
    def mode(self) -> str:
        return self._mode

    def _ensure_backend(self) -> _JsonBackend | _MlflowBackend:
        if self._backend is not None:
            return self._backend
        if self._mode == "mlflow":
            try:
                uri = (
                    getattr(get_settings().mlflow, "tracking_uri", "mlruns/")
                    if hasattr(get_settings(), "mlflow")
                    else "mlruns/"
                )
                self._backend = _MlflowBackend(uri)
                return self._backend
            except Exception as e:
                logger.warning("mlflow_unavailable", error=str(e))
                self._mode = "json"
        self._backend = _JsonBackend(self.root_dir)
        return self._backend

    @contextmanager
    def start_run(
        self,
        experiment_name: str,
        run_name: str | None = None,
        params: dict[str, Any] | None = None,
    ):
        ctx = RunContext(
            run_id=_make_run_id(run_name),
            experiment_name=experiment_name,
            run_name=run_name or uuid.uuid4().hex[:8],
            started_at=datetime.now(timezone.utc),
            backend=self._mode,
        )
        if params:
            ctx.params.update(params)
        backend = self._ensure_backend()
        backend.start(ctx)
        try:
            yield ctx
        finally:
            ctx.finish()
            if hasattr(backend, "finish"):
                backend.finish(ctx)

    def list_runs(self, experiment_name: str) -> list[Path]:
        backend = self._ensure_backend()
        if hasattr(backend, "list_runs"):
            return backend.list_runs(experiment_name)  # type: ignore[union-attr]
        return []

    def load(self, experiment_name: str, run_id: str) -> dict[str, Any] | None:
        backend = self._ensure_backend()
        if hasattr(backend, "load"):
            return backend.load(experiment_name, run_id)  # type: ignore[union-attr]
        return None


__all__ = ["Tracker", "RunContext", "_make_run_id"]


def _unused_import_marker() -> str:  # pragma: no cover
    return time.__name__
