"""CLI for OSCAR ML commands.

Usage:
    python -m src.ml.cli build-features
    python -m src.ml.cli train-escalation --horizon 7
    python -m src.ml.cli forecast --regions UKR,RUS --periods 7
    python -m src.ml.cli detect-anomalies --window 14
    python -m src.ml.cli promote --name escalation_h7 --version 1
    python -m src.ml.cli ml-train-all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
except ImportError:
    pass

from src.config import get_settings
from src.observability import configure_logging
from src.persistence.database import session_scope


def cmd_build_features(args: argparse.Namespace) -> int:
    from src.ml.features import build_feature_matrix

    settings = get_settings()
    silver_path = settings.processed_data_dir / "silver" / "events_per_country_day.parquet"
    horizons = tuple(int(h) for h in args.horizons.split(","))
    matrix = build_feature_matrix(silver_path=silver_path, horizons=horizons)
    print(
        f"[features] rows={len(matrix.features)} "
        f"n_features={len(matrix.feature_columns)} "
        f"horizons={horizons} "
        f"positive_rate={matrix.get_target(horizons[0]).mean():.2%}"
    )
    return 0


def cmd_train_escalation(args: argparse.Namespace) -> int:
    from src.ml.escalation import EscalationClassifier, cross_validate
    from src.ml.features import build_feature_matrix
    from src.ml.registry import Registry
    from src.ml.tracking import Tracker

    settings = get_settings()
    silver_path = settings.processed_data_dir / "silver" / "events_per_country_day.parquet"
    matrix = build_feature_matrix(silver_path=silver_path)

    horizon = args.horizon
    force_mode = args.force_mode or "sklearn"
    name = f"escalation_h{horizon}"
    tracker = Tracker(force_mode="json")
    registry = Registry()

    with tracker.start_run(
        experiment_name="escalation",
        run_name=f"{name}_train",
        params={
            "horizon": horizon,
            "force_mode": force_mode,
            "n_features": len(matrix.feature_columns),
        },
    ) as run:
        clf = EscalationClassifier(horizon_days=horizon, force_mode=force_mode)
        clf.fit(matrix, horizon=horizon)

        from sklearn.model_selection import train_test_split

        X = matrix.features[matrix.feature_columns]
        y = matrix.features[f"target_h{horizon}"]
        mask = y.notna()
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True).astype(int)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        clf._model.fit(Xtr.values, ytr.values)
        metrics = clf.score(Xte, yte)
        run.log_metrics(metrics.to_dict())

        cv = cross_validate(matrix, horizon=horizon, n_splits=3, force_mode=force_mode)
        run.log_metrics(cv)

        artifact_path = (
            settings.models_dir
            / "checkpoints"
            / f"{name}_v{registry.list_versions(name).__len__() + 1}.pkl"
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(clf.to_bytes())

        mv = registry.register(
            name=name,
            metrics=metrics.to_dict(),
            params={"horizon": horizon, "force_mode": force_mode},
            artifact_path=str(artifact_path),
            run_id=run.run_id,
            description=f"Trained on {len(matrix.features)} rows, {len(matrix.feature_columns)} features",
        )
        if args.promote:
            registry.promote(name, mv.version)

        print(
            f"[escalation] {name} v{mv.version} | "
            f"acc={metrics.accuracy:.3f} f1={metrics.f1:.3f} pr_auc={metrics.pr_auc:.3f}"
        )
        print(f"[escalation] artifact: {artifact_path}")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    from src.ml.features import build_feature_matrix
    from src.ml.forecast import fit_per_region

    settings = get_settings()
    silver_path = settings.processed_data_dir / "silver" / "events_per_country_day.parquet"
    matrix = build_feature_matrix(silver_path=silver_path)
    regions = args.regions.split(",") if args.regions else None

    forecasts = fit_per_region(
        features_df=matrix.features,
        target_col=args.target,
        regions=regions,
        periods=args.periods,
        force_mode=args.force_mode or "linear",
    )

    out_dir = settings.processed_data_dir / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in forecasts:
        out_path = out_dir / f"forecast_{f.region}.parquet"
        f.forecast.to_parquet(out_path, index=False)
        print(
            f"[forecast] {f.region} ({f.model}) | "
            f"mae={f.metrics.get('mae', 0):.2f} "
            f"rmse={f.metrics.get('rmse', 0):.2f} -> {out_path.name}"
        )
    return 0


def cmd_detect_anomalies(args: argparse.Namespace) -> int:
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    from src.ml.anomaly import AnomalyDetector
    from src.ml.features import build_feature_matrix
    from src.persistence.models import Anomaly

    settings = get_settings()
    silver_path = settings.processed_data_dir / "silver" / "events_per_country_day.parquet"
    matrix = build_feature_matrix(silver_path=silver_path)

    feature_cols = ["event_count", "conflict_count", "avg_tone"]
    feature_cols = [c for c in feature_cols if c in matrix.features.columns]
    if not feature_cols:
        print("[anomaly] No suitable feature columns in matrix")
        return 1

    detector = AnomalyDetector(
        window=args.window, threshold=args.threshold, force_mode=args.force_mode
    )
    anomalies = detector.detect(
        matrix.features,
        feature_cols=feature_cols,
        region_col="actor1_country_code",
        date_col="date",
    )

    with session_scope() as session:
        for a in anomalies:
            row = a.to_db_row()
            row["detected_at"] = a.detected_at or __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            )
            session.execute(sqlite_insert(Anomaly).values(row))

    print(f"[anomaly] detected {len(anomalies)} anomalies ({detector.mode})")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    from src.ml.registry import Registry

    registry = Registry()
    mv = registry.promote(args.name, args.version)
    print(f"[registry] promoted {args.name} v{args.version} -> {mv.stage}")
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    from src.ml.registry import Registry

    registry = Registry()
    for name in ["escalation_h1", "escalation_h3", "escalation_h7"]:
        versions = registry.list_versions(name)
        if not versions:
            continue
        print(f"\n=== {name} ===")
        for v in versions:
            star = " *PROD" if v.stage == "Production" else ""
            print(f"  v{v.version} [{v.stage}]{star} | metrics={v.metrics}")
    return 0


def cmd_ml_train_all(args: argparse.Namespace) -> int:
    for fn in (cmd_build_features, cmd_train_escalation, cmd_forecast, cmd_detect_anomalies):
        rc = fn(args)
        if rc != 0:
            return rc
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oscar-ml", description="OSCAR ML CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    bf = sub.add_parser("build-features", help="Build feature matrix from silver tables.")
    bf.add_argument(
        "--horizons", default="1,3,7", help="Comma-separated horizons (default: 1,3,7)."
    )
    bf.set_defaults(func=cmd_build_features)

    te = sub.add_parser("train-escalation", help="Train escalation classifier.")
    te.add_argument("--horizon", type=int, default=7)
    te.add_argument("--force-mode", choices=["xgboost", "sklearn"], default="sklearn")
    te.add_argument("--promote", action="store_true", help="Promote to Production after training.")
    te.set_defaults(func=cmd_train_escalation)

    fc = sub.add_parser("forecast", help="Forecast per region.")
    fc.add_argument("--regions", default="", help="Comma-separated regions (default: all).")
    fc.add_argument("--target", default="event_count", choices=["event_count", "conflict_count"])
    fc.add_argument("--periods", type=int, default=7)
    fc.add_argument("--force-mode", choices=["prophet", "linear"], default="linear")
    fc.set_defaults(func=cmd_forecast)

    da = sub.add_parser("detect-anomalies", help="Detect anomalies.")
    da.add_argument("--window", type=int, default=14)
    da.add_argument("--threshold", type=float, default=2.5)
    da.add_argument("--force-mode", choices=["iforest", "zscore"], default="iforest")
    da.set_defaults(func=cmd_detect_anomalies)

    pm = sub.add_parser("promote", help="Promote model version.")
    pm.add_argument("--name", required=True)
    pm.add_argument("--version", type=int, required=True)
    pm.set_defaults(func=cmd_promote)

    lm = sub.add_parser("list-models", help="List registered models.")
    lm.set_defaults(func=cmd_list_models)

    all_cmd = sub.add_parser("ml-train-all", help="Build features + train + forecast + detect.")
    all_cmd.set_defaults(func=cmd_ml_train_all)

    return parser


def main() -> int:
    cfg = get_settings()
    configure_logging(level=cfg.log_level, json_format=False)
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
