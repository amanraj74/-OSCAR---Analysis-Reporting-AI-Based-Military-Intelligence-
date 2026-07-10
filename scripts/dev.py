"""Cross-platform developer task runner for OSCAR.

Mirrors every Makefile target so Windows users (no `make`) can run the same
workflows without ceremony. The Makefile simply forwards to this script.

Usage
-----
    python scripts/dev.py <command>

Commands
--------
    install         install runtime deps
    install-dev     install runtime + dev deps + pre-commit hooks + spacy model
    install-hooks   install pre-commit hooks only
    lint            ruff + black --check + isort --check + bandit
    format          black + isort + ruff --fix
    test            pytest with coverage (fails under threshold)
    test-unit       unit tests only (no coverage)
    test-coverage   full coverage HTML report
    refresh         refresh all data feeds
    train           train/retrain all models
    dashboard       launch streamlit
    seed-demo       seed 30-day demo dataset
    clean           remove caches / pyc / build artifacts
    clean-data      DANGEROUS: wipe data/processed + SQLite DB
    docker-build    build docker image
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    """Run a subprocess command, streaming output, returning exit code."""
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, check=check).returncode


def command_install(_: argparse.Namespace) -> int:
    return run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def command_install_dev(_: argparse.Namespace) -> int:
    rc = run([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"])
    if rc != 0:
        return rc
    rc = command_install_hooks(None)
    if rc != 0:
        return rc
    print("\n[*] Installing spaCy English model (en_core_web_sm)...")
    return run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=False)


def command_install_hooks(_: argparse.Namespace) -> int:
    print("[*] Installing pre-commit hooks...")
    return run([sys.executable, "-m", "pre_commit", "install"], check=False)


def command_lint(_: argparse.Namespace) -> int:
    rc = 0
    rc |= run([sys.executable, "-m", "ruff", "check", "src", "tests"], check=False)
    rc |= run([sys.executable, "-m", "black", "--check", "src", "tests"], check=False)
    rc |= run([sys.executable, "-m", "isort", "--check-only", "src", "tests"], check=False)
    rc |= run(
        [
            sys.executable,
            "-m",
            "bandit",
            "-r",
            "src",
            "-c",
            "pyproject.toml",
            "--quiet",
        ],
        check=False,
    )
    return rc


def command_format(_: argparse.Namespace) -> int:
    rc = 0
    rc |= run([sys.executable, "-m", "black", "src", "tests"], check=False)
    rc |= run([sys.executable, "-m", "isort", "src", "tests"], check=False)
    rc |= run([sys.executable, "-m", "ruff", "check", "--fix", "src", "tests"], check=False)
    return rc


def command_test(_: argparse.Namespace) -> int:
    return run([sys.executable, "-m", "pytest"], check=False)


def command_test_unit(_: argparse.Namespace) -> int:
    return run([sys.executable, "-m", "pytest", "tests/unit", "-m", "not slow"], check=False)


def command_test_coverage(_: argparse.Namespace) -> int:
    rc = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing",
        ],
        check=False,
    )
    if rc == 0:
        print("\n[OK] Coverage report: htmlcov/index.html")
    return rc


def command_refresh(args: argparse.Namespace) -> int:
    """Refresh data feeds. Default: GDELT only. Pass --source to specify."""
    sources = args.source or ["gdelt"]
    rc = 0
    for src in sources:
        rc |= run(
            [sys.executable, "-m", "src.ingestion.cli", "refresh", "--source", src],
            check=False,
        )
    return rc


def command_train(_: argparse.Namespace) -> int:
    return run([sys.executable, "-m", "src.ml.cli", "ml-train-all"], check=False)


def command_dashboard(_: argparse.Namespace) -> int:
    return run(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"],
        check=False,
    )


def command_seed_demo(_: argparse.Namespace) -> int:
    return run([sys.executable, "scripts/seed_demo.py"], check=False)


def command_clean(_: argparse.Namespace) -> int:
    patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.egg-info",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".coverage",
        "htmlcov",
        "build",
        "dist",
    ]
    removed = 0
    for pattern in patterns:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            removed += 1
    print(f"[OK] Removed {removed} cache entries.")
    return 0


def command_clean_data(args: argparse.Namespace) -> int:
    if not args.yes:
        print("[!] This will DELETE data/processed/ and the SQLite DB. Pass --yes to confirm.")
        return 1
    targets = [PROJECT_ROOT / "data" / "processed", PROJECT_ROOT / "data" / "oscar.db"]
    for t in targets:
        if t.exists():
            if t.is_dir():
                shutil.rmtree(t)
            else:
                t.unlink()
            print(f"[OK] Removed: {t}")
    return 0


def command_docker_build(_: argparse.Namespace) -> int:
    return run(["docker", "build", "-t", "oscar:latest", "."], check=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oscar-dev",
        description="OSCAR cross-platform developer task runner.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("install", help="Install runtime dependencies.").set_defaults(
        func=command_install
    )
    sub.add_parser(
        "install-dev", help="Install runtime + dev deps + spacy model + hooks."
    ).set_defaults(func=command_install_dev)
    sub.add_parser("install-hooks", help="Install pre-commit hooks.").set_defaults(
        func=command_install_hooks
    )
    sub.add_parser("lint", help="Run all linters.").set_defaults(func=command_lint)
    sub.add_parser("format", help="Auto-format code.").set_defaults(func=command_format)
    sub.add_parser("test", help="Run all tests.").set_defaults(func=command_test)
    sub.add_parser("test-unit", help="Run unit tests only.").set_defaults(func=command_test_unit)
    sub.add_parser("test-coverage", help="Run tests with HTML coverage.").set_defaults(
        func=command_test_coverage
    )

    refresh = sub.add_parser("refresh", help="Refresh data feeds.")
    refresh.add_argument(
        "--source",
        action="append",
        choices=["gdelt", "newsapi", "reddit"],
        help="Source to refresh (repeatable). Defaults to gdelt.",
    )
    refresh.set_defaults(func=command_refresh)

    sub.add_parser("train", help="Train/retrain all models.").set_defaults(func=command_train)
    sub.add_parser("dashboard", help="Launch Streamlit dashboard.").set_defaults(
        func=command_dashboard
    )
    sub.add_parser("seed-demo", help="Seed 30-day demo dataset.").set_defaults(
        func=command_seed_demo
    )
    sub.add_parser("clean", help="Remove caches and build artifacts.").set_defaults(
        func=command_clean
    )

    clean_data = sub.add_parser("clean-data", help="DANGEROUS: wipe data/processed and SQLite DB.")
    clean_data.add_argument("--yes", action="store_true", help="Confirm deletion.")
    clean_data.set_defaults(func=command_clean_data)

    sub.add_parser("docker-build", help="Build Docker image.").set_defaults(
        func=command_docker_build
    )
    return parser


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "install": command_install,
    "install-dev": command_install_dev,
    "install-hooks": command_install_hooks,
    "lint": command_lint,
    "format": command_format,
    "test": command_test,
    "test-unit": command_test_unit,
    "test-coverage": command_test_coverage,
    "refresh": command_refresh,
    "train": command_train,
    "dashboard": command_dashboard,
    "seed-demo": command_seed_demo,
    "clean": command_clean,
    "clean-data": command_clean_data,
    "docker-build": command_docker_build,
}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args) or 0


if __name__ == "__main__":
    sys.exit(main())
