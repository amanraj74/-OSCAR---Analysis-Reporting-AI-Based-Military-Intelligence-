"""Tests for ML tracking and registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_registry(tmp_path) -> Path:
    return tmp_path


def test_tracker_initializes(tmp_registry) -> None:
    from src.ml.tracking import Tracker

    tracker = Tracker(force_mode="json", root_dir=tmp_registry)
    assert tracker.mode == "json"


def test_tracker_start_run_logs_params_and_metrics(tmp_registry) -> None:
    from src.ml.tracking import Tracker

    tracker = Tracker(force_mode="json", root_dir=tmp_registry)
    with tracker.start_run(
        experiment_name="test_exp",
        run_name="test_run",
        params={"lr": 0.01, "epochs": 10},
    ) as run:
        run.log_metric("f1", 0.85)
        run.log_metric("accuracy", 0.91)
        run.log_metrics({"precision": 0.83, "recall": 0.88})
        run.set_tag("dataset", "synthetic")

    run_dirs = tracker.list_runs("test_exp")
    assert len(run_dirs) == 1
    meta_file = run_dirs[0] / "meta.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text())
    assert meta["params"]["lr"] == 0.01
    assert meta["metrics"]["f1"] == 0.85
    assert meta["tags"]["dataset"] == "synthetic"


def test_tracker_persists_artifact(tmp_registry) -> None:
    from src.ml.tracking import Tracker

    artifact = tmp_registry / "model.pkl"
    artifact.write_bytes(b"fake-model-bytes")

    tracker = Tracker(force_mode="json", root_dir=tmp_registry)
    with tracker.start_run(experiment_name="exp", run_name="run") as run:
        run.log_artifact(str(artifact))

    run_dirs = tracker.list_runs("exp")
    meta = json.loads((run_dirs[0] / "meta.json").read_text())
    assert meta["artifacts"][0]["source"] == str(artifact)


def test_make_run_id_format() -> None:
    from src.ml.tracking import _make_run_id

    rid = _make_run_id("test slug with spaces & symbols!")
    assert rid.startswith("EXP-")
    assert "test_slug_with" in rid


def test_registry_register(tmp_registry) -> None:
    from src.ml.registry import Registry

    reg = Registry(registry_dir=tmp_registry)
    mv = reg.register(name="escalation_h7", metrics={"f1": 0.85})
    assert mv.name == "escalation_h7"
    assert mv.version == 1
    assert mv.stage == "None"


def test_registry_increments_version(tmp_registry) -> None:
    from src.ml.registry import Registry

    reg = Registry(registry_dir=tmp_registry)
    v1 = reg.register(name="x", metrics={"f1": 0.8})
    v2 = reg.register(name="x", metrics={"f1": 0.9})
    assert v2.version == v1.version + 1


def test_registry_promote(tmp_registry) -> None:
    from src.ml.registry import Registry

    reg = Registry(registry_dir=tmp_registry)
    reg.register(name="x", metrics={"f1": 0.7})
    reg.register(name="x", metrics={"f1": 0.9})
    reg.promote("x", 1)
    prod = reg.get_production("x")
    assert prod is not None
    assert prod.stage in {"Production", "Staging", "None"}


def test_registry_promote_archives_previous_prod(tmp_registry) -> None:
    from src.ml.registry import Registry

    reg = Registry(registry_dir=tmp_registry)
    reg.register(name="x", metrics={"f1": 0.7})
    reg.register(name="x", metrics={"f1": 0.9})
    reg.promote("x", 1, stage="Production")
    reg.promote("x", 2, stage="Production")
    versions = {v.version: v.stage for v in reg.list_versions("x")}
    assert versions[2] == "Production"
    assert versions[1] == "Archived"


def test_registry_demote(tmp_registry) -> None:
    from src.ml.registry import Registry

    reg = Registry(registry_dir=tmp_registry)
    reg.register(name="x")
    reg.promote("x", 1, stage="Production")
    reg.demote("x", 1, stage="Archived")
    prod = reg.get_production("x")
    assert prod is None


def test_registry_get_production_fallback(tmp_registry) -> None:
    from src.ml.registry import Registry

    reg = Registry(registry_dir=tmp_registry)
    reg.register(name="x")
    prod = reg.get_production("x")
    assert prod is None
    latest = reg.list_versions("x")[-1]
    assert latest.stage == "None"


def test_registry_persist_across_instances(tmp_registry) -> None:
    from src.ml.registry import Registry

    r1 = Registry(registry_dir=tmp_registry)
    r1.register(name="x", metrics={"f1": 0.9})
    r2 = Registry(registry_dir=tmp_registry)
    vs = r2.list_versions("x")
    assert len(vs) == 1


def test_registry_load_production(tmp_registry) -> None:
    from src.ml.registry import Registry

    artifact = tmp_registry / "model.pkl"
    artifact.write_bytes(b"fake")
    reg = Registry(registry_dir=tmp_registry)
    reg.register(name="x", artifact_path=str(artifact))
    reg.promote("x", 1, stage="Production")
    mv, loaded = reg.load_production("x", lambda b: b)
    assert mv is not None
    assert loaded == b"fake"


def test_registry_load_artifact_specific_version(tmp_registry) -> None:
    from src.ml.registry import Registry

    artifact = tmp_registry / "model.pkl"
    artifact.write_bytes(b"v1")
    reg = Registry(registry_dir=tmp_registry)
    reg.register(name="x", artifact_path=str(artifact))
    loaded = reg.load_artifact("x", 1, lambda b: b)
    assert loaded == b"v1"


def test_registry_get_latest(tmp_registry) -> None:
    from src.ml.registry import Registry

    reg = Registry(registry_dir=tmp_registry)
    reg.register(name="x")
    reg.register(name="x")
    latest = reg.get_latest("x")
    assert latest is not None
    assert latest.version == 2
