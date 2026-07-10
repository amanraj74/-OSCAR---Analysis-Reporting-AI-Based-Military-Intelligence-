"""OSCAR unified CLI entry point.

This is the script registered in ``pyproject.toml`` as the ``oscar`` console
script. It simply dispatches to the per-subsystem CLIs so the entire pipeline
can be driven from a single command.

Usage:
    oscar ingest refresh --source gdelt
    oscar ingest transform
    oscar nlp ner-process
    oscar nlp sentiment-score
    oscar nlp topics-discover --n-topics 8
    oscar nlp entities-normalize
    oscar nlp nlp-process
    oscar ml build-features
    oscar ml train-escalation --horizon 7
    oscar ml forecast --regions UKR,RUS --periods 7
    oscar ml detect-anomalies --window 14
    oscar ml promote --name escalation_h7 --version 1
    oscar ml list-models
    oscar ml ml-train-all
    oscar dashboard       # launches the Streamlit app
    oscar --version
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
except ImportError:
    pass

from src import __version__  # noqa: E402


def _build_ingest_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("ingest", help="Run data ingestion commands.")
    inner = parser.add_subparsers(dest="ingest_command", required=True)

    refresh = inner.add_parser("refresh", help="Run an ingestor end-to-end.")
    refresh.add_argument(
        "--source",
        required=True,
        choices=["acled", "gdelt", "newsapi", "reddit"],
        help="Data source to ingest.",
    )
    refresh.add_argument("--query", help="Optional search query (newsapi).")
    refresh.add_argument("--max-files", type=int, help="Limit files (gdelt).")
    refresh.add_argument("--subreddit", action="append", help="Subreddit to ingest (repeatable).")
    refresh.set_defaults(_handler="_ingest_refresh")

    transform = inner.add_parser("transform", help="Build silver Parquet tables.")
    transform.set_defaults(_handler="_ingest_transform")


def _build_nlp_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("nlp", help="Run NLP commands.")
    inner = parser.add_subparsers(dest="nlp_command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--limit", type=int, help="Limit rows processed.")

    ner = inner.add_parser("ner-process", parents=[common], help="Extract entities.")
    ner.set_defaults(_handler="_nlp_ner")

    sentiment = inner.add_parser("sentiment-score", parents=[common], help="Score sentiment.")
    sentiment.set_defaults(_handler="_nlp_sentiment")

    topics = inner.add_parser("topics-discover", parents=[common], help="Discover themes.")
    topics.add_argument("--n-topics", type=int, default=8)
    topics.add_argument("--min-articles", type=int, default=1)
    topics.set_defaults(_handler="_nlp_topics")

    entities = inner.add_parser("entities-normalize", help="Merge entity aliases.")
    entities.add_argument("--threshold", type=float, default=0.8)
    entities.set_defaults(_handler="_nlp_entities")

    all_cmd = inner.add_parser("nlp-process", parents=[common], help="Run all NLP stages.")
    all_cmd.set_defaults(_handler="_nlp_all")


def _build_ml_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("ml", help="Run machine-learning commands.")
    inner = parser.add_subparsers(dest="ml_command", required=True)

    bf = inner.add_parser("build-features", help="Build feature matrix from silver tables.")
    bf.add_argument("--horizons", default="1,3,7")
    bf.set_defaults(_handler="_ml_build_features")

    te = inner.add_parser("train-escalation", help="Train escalation classifier.")
    te.add_argument("--horizon", type=int, default=7)
    te.add_argument("--force-mode", choices=["xgboost", "sklearn"], default="sklearn")
    te.add_argument("--promote", action="store_true")
    te.set_defaults(_handler="_ml_train_escalation")

    fc = inner.add_parser("forecast", help="Forecast per region.")
    fc.add_argument("--regions", default="")
    fc.add_argument("--target", default="event_count", choices=["event_count", "conflict_count"])
    fc.add_argument("--periods", type=int, default=7)
    fc.add_argument("--force-mode", choices=["prophet", "linear"], default="linear")
    fc.set_defaults(_handler="_ml_forecast")

    da = inner.add_parser("detect-anomalies", help="Detect anomalies.")
    da.add_argument("--window", type=int, default=14)
    da.add_argument("--threshold", type=float, default=2.5)
    da.add_argument("--force-mode", choices=["iforest", "zscore"], default="iforest")
    da.set_defaults(_handler="_ml_detect_anomalies")

    pm = inner.add_parser("promote", help="Promote model version.")
    pm.add_argument("--name", required=True)
    pm.add_argument("--version", type=int, required=True)
    pm.set_defaults(_handler="_ml_promote")

    lm = inner.add_parser("list-models", help="List registered models.")
    lm.set_defaults(_handler="_ml_list_models")

    all_cmd = inner.add_parser("ml-train-all", help="Build features + train + forecast + detect.")
    all_cmd.set_defaults(_handler="_ml_train_all")


def _build_dashboard_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("dashboard", help="Launch the Streamlit dashboard.")
    parser.add_argument("--host", default="localhost", help="Streamlit host (default: localhost).")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port (default: 8501).")
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically."
    )
    parser.set_defaults(_handler="_dashboard")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oscar",
        description="OSCAR — Open-Source Conflict Analysis & Reporting CLI.",
    )
    parser.add_argument("--version", action="version", version=f"oscar {__version__}")
    sub = parser.add_subparsers(dest="command")

    _build_ingest_parser(sub)
    _build_nlp_parser(sub)
    _build_ml_parser(sub)
    _build_dashboard_parser(sub)

    return parser


def _run_dispatch(args: argparse.Namespace) -> int:
    handler = getattr(args, "_handler", None)
    if not handler:
        build_parser().print_help()
        return 0

    if handler.startswith("_ingest_"):
        from src.ingestion.cli import main as ingest_main

        sub_action = args.ingest_command
        sub_args = [sub_action]
        for k, v in vars(args).items():
            if k in {
                "command",
                "_handler",
                "ingest_command",
            }:
                continue
            if isinstance(v, list):
                for item in v:
                    sub_args.extend([f"--{k.replace('_', '-')}", str(item)])
            elif isinstance(v, bool):
                if v:
                    sub_args.append(f"--{k.replace('_', '-')}")
            elif v is not None:
                sub_args.extend([f"--{k.replace('_', '-')}", str(v)])
        sys.argv = ["oscar-ingest", *sub_args]
        return ingest_main()

    if handler.startswith("_nlp_"):
        from src.nlp.cli import main as nlp_main

        cmd_map = {
            "_nlp_ner": "ner-process",
            "_nlp_sentiment": "sentiment-score",
            "_nlp_topics": "topics-discover",
            "_nlp_entities": "entities-normalize",
            "_nlp_all": "nlp-process",
        }
        sub_args = [cmd_map[handler]]
        for k, v in vars(args).items():
            if k in {"command", "_handler", "nlp_command"}:
                continue
            if isinstance(v, bool):
                if v:
                    sub_args.append(f"--{k.replace('_', '-')}")
            elif v is not None:
                sub_args.extend([f"--{k.replace('_', '-')}", str(v)])
        sys.argv = ["oscar-nlp", *sub_args]
        return nlp_main()

    if handler.startswith("_ml_"):
        from src.ml.cli import main as ml_main

        cmd_map = {
            "_ml_build_features": "build-features",
            "_ml_train_escalation": "train-escalation",
            "_ml_forecast": "forecast",
            "_ml_detect_anomalies": "detect-anomalies",
            "_ml_promote": "promote",
            "_ml_list_models": "list-models",
            "_ml_train_all": "ml-train-all",
        }
        sub_args = [cmd_map[handler]]
        for k, v in vars(args).items():
            if k in {"command", "_handler", "ml_command"}:
                continue
            if isinstance(v, bool):
                if v:
                    sub_args.append(f"--{k.replace('_', '-')}")
            elif v is not None:
                sub_args.extend([f"--{k.replace('_', '-')}", str(v)])
        sys.argv = ["oscar-ml", *sub_args]
        return ml_main()

    if handler == "_dashboard":
        import subprocess

        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "dashboard" / "app.py"),
            "--server.address",
            args.host,
            "--server.port",
            str(args.port),
        ]
        if args.no_browser:
            cmd.extend(["--server.headless", "true"])
        return subprocess.call(cmd)

    build_parser().print_help()
    return 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return _run_dispatch(args)


if __name__ == "__main__":
    sys.exit(main())
