"""Run the full OSCAR training pipeline: silver + 3 models + promote + anomalies."""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

os.chdir(ROOT)
os.environ["DATABASE_URL"] = "sqlite:///data/oscar.db"
os.environ["PYTHONPATH"] = str(ROOT)

cmds = [
    ("Rebuild silver from 370K events", ["-m", "src.ingestion.cli", "transform"]),
    ("Train h1 (1-day horizon)", ["-m", "src.ml.cli", "train-escalation", "--horizon", "1"]),
    ("Train h3 (3-day horizon)", ["-m", "src.ml.cli", "train-escalation", "--horizon", "3"]),
    ("Train h7 (7-day horizon)", ["-m", "src.ml.cli", "train-escalation", "--horizon", "7"]),
    ("Promote h1", ["-m", "src.ml.cli", "promote", "--name", "escalation_h1", "--version", "1"]),
    ("Promote h3", ["-m", "src.ml.cli", "promote", "--name", "escalation_h3", "--version", "1"]),
    ("Promote h7", ["-m", "src.ml.cli", "promote", "--name", "escalation_h7", "--version", "1"]),
    ("Detect anomalies", ["-m", "src.ml.cli", "detect-anomalies"]),
    ("List models", ["-m", "src.ml.cli", "list-models"]),
]

start = time.time()
for label, args in cmds:
    print(f"\n{'=' * 70}")
    print(f"[{time.time() - start:5.1f}s] {label}")
    print(f"{'=' * 70}")
    result = subprocess.run([str(PY), *args], cwd=str(ROOT), check=False)
    if result.returncode != 0:
        print(f"FAILED: {label} (exit code {result.returncode})")
        sys.exit(1)

print(f"\n{'=' * 70}")
print(f"ALL DONE in {time.time() - start:.1f}s")
print(f"{'=' * 70}")
