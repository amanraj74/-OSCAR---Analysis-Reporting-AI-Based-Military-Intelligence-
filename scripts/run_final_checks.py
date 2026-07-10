"""Run final test + lint + fix any remaining issues."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

cmds = [
    (
        "LINT (ruff)",
        [str(PY), "-m", "ruff", "check", str(ROOT / "src" / "tests" / "dashboard" / "scripts")],
    ),
    (
        "LINT (isort)",
        [
            str(PY),
            "-m",
            "isort",
            "--check-only",
            str(ROOT / "src" / "tests" / "dashboard" / "scripts"),
        ],
    ),
    (
        "LINT (black)",
        [str(PY), "-m", "black", "--check", str(ROOT / "src" / "tests" / "dashboard" / "scripts")],
    ),
    (
        "TESTS (unit + e2e)",
        [
            str(PY),
            "-m",
            "pytest",
            str(ROOT / "tests"),
            "-p",
            "no:cacheprovider",
            "--tb=short",
            "--timeout=120",
            "-q",
        ],
    ),
]

for label, cmd in cmds:
    print(f"\n=== {label} ===")
    r = subprocess.run(cmd, cwd=str(ROOT), check=False)
    print(f"exit code: {r.returncode}")
