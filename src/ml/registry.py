"""Model registry for OSCAR.

Promote/demote trained models, version them, and serve the latest
production model for inference.

Public API
----------
    Registry().register(name, version, metrics, artifact_path) -> dict
    Registry().promote(name, version)                          -> dict
    Registry().demote(name, version)                            -> dict
    Registry().get_production(name)                             -> dict | None
    Registry().load_production(name, deserializer)              -> Any
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.observability import get_logger

logger = get_logger("ml.registry")


_STAGE_DEFAULT = "None"
_STAGE_STAGING = "Staging"
_STAGE_PRODUCTION = "Production"
_STAGE_ARCHIVED = "Archived"


@dataclass
class ModelVersion:
    """Single model version entry."""

    name: str
    version: int
    stage: str = _STAGE_DEFAULT
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    run_id: str | None = None
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    promoted_at: datetime | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "stage": self.stage,
            "metrics": self.metrics,
            "params": self.params,
            "artifact_path": self.artifact_path,
            "run_id": self.run_id,
            "registered_at": self.registered_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "description": self.description,
        }


class Registry:
    """JSON-backed model registry with promote/demote semantics."""

    def __init__(self, registry_dir: Path | None = None) -> None:
        settings = get_settings()
        self.registry_dir = registry_dir or (settings.models_dir / "registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.registry_dir / "registry.json"

    def _load_index(self) -> dict[str, dict[str, ModelVersion]]:
        if not self.index_path.exists():
            return defaultdict(dict)
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        out: dict[str, dict[str, ModelVersion]] = defaultdict(dict)
        for name, versions in data.items():
            for vid, vd in versions.items():
                mv = ModelVersion(
                    name=vd["name"],
                    version=vd["version"],
                    stage=vd["stage"],
                    metrics=vd.get("metrics", {}),
                    params=vd.get("params", {}),
                    artifact_path=vd.get("artifact_path"),
                    run_id=vd.get("run_id"),
                    description=vd.get("description", ""),
                )
                out[name][vid] = mv
        return out

    def _save_index(self, index: dict[str, dict[str, ModelVersion]]) -> None:
        serialized: dict[str, dict[str, dict[str, Any]]] = {}
        for name, versions in index.items():
            serialized[name] = {str(v.version): v.to_dict() for v in versions.values()}
        self.index_path.write_text(
            json.dumps(serialized, indent=2, default=str),
            encoding="utf-8",
        )

    def register(
        self,
        name: str,
        metrics: dict[str, float] | None = None,
        params: dict[str, Any] | None = None,
        artifact_path: str | None = None,
        run_id: str | None = None,
        description: str = "",
    ) -> ModelVersion:
        index = self._load_index()
        existing = list(index[name].values()) if name in index else []
        next_version = max((m.version for m in existing), default=0) + 1

        mv = ModelVersion(
            name=name,
            version=next_version,
            stage=_STAGE_DEFAULT,
            metrics=metrics or {},
            params=params or {},
            artifact_path=artifact_path,
            run_id=run_id,
            description=description,
        )
        index[name][str(next_version)] = mv
        self._save_index(index)
        logger.info(
            "model_registered",
            name=name,
            version=next_version,
            run_id=run_id,
            metrics=metrics or {},
        )
        return mv

    def list_versions(self, name: str) -> list[ModelVersion]:
        index = self._load_index()
        return sorted(index.get(name, {}).values(), key=lambda m: m.version)

    def promote(self, name: str, version: int, stage: str = _STAGE_PRODUCTION) -> ModelVersion:
        index = self._load_index()
        if name not in index or str(version) not in index[name]:
            raise KeyError(f"{name} v{version} not registered")
        mv = index[name][str(version)]
        prev_stage = mv.stage
        mv.stage = stage
        mv.promoted_at = datetime.now(timezone.utc)

        if stage == _STAGE_PRODUCTION:
            for other in index[name].values():
                if other.version != version and other.stage == _STAGE_PRODUCTION:
                    other.stage = _STAGE_ARCHIVED
                    other.promoted_at = datetime.now(timezone.utc)

        self._save_index(index)
        logger.info(
            "model_stage_changed",
            name=name,
            version=version,
            from_stage=prev_stage,
            to_stage=stage,
        )
        return mv

    def demote(self, name: str, version: int, stage: str = _STAGE_ARCHIVED) -> ModelVersion:
        return self.promote(name, version, stage=stage)

    def get_production(self, name: str) -> ModelVersion | None:
        for mv in self.list_versions(name):
            if mv.stage == _STAGE_PRODUCTION:
                return mv
        for mv in self.list_versions(name):
            if mv.stage == _STAGE_STAGING:
                return mv
        return None

    def get_latest(self, name: str) -> ModelVersion | None:
        versions = self.list_versions(name)
        return versions[-1] if versions else None

    def load_artifact(
        self, name: str, version: int, deserializer: Callable[[bytes], Any]
    ) -> Any | None:
        mv = self.list_versions(name)
        target = next((m for m in mv if m.version == version), None)
        if target is None or target.artifact_path is None:
            return None
        path = Path(target.artifact_path)
        if not path.exists():
            logger.warning("artifact_missing", path=str(path))
            return None
        return deserializer(path.read_bytes())

    def load_production(
        self,
        name: str,
        deserializer: Callable[[bytes], Any],
    ) -> tuple[ModelVersion | None, Any | None]:
        mv = self.get_production(name)
        if mv is None or mv.artifact_path is None:
            return mv, None
        path = Path(mv.artifact_path)
        if not path.exists():
            return mv, None
        return mv, deserializer(path.read_bytes())


_STAGE_LABELS = {
    _STAGE_DEFAULT: "registered but not deployed",
    _STAGE_STAGING: "validated, ready to test",
    _STAGE_PRODUCTION: "serving live traffic",
    _STAGE_ARCHIVED: "superseded or deprecated",
}


__all__ = [
    "Registry",
    "ModelVersion",
    "_STAGE_DEFAULT",
    "_STAGE_STAGING",
    "_STAGE_PRODUCTION",
    "_STAGE_ARCHIVED",
]


def _unused_import_marker() -> str:  # pragma: no cover
    return time.__name__
