"""Save Kaggle token and download the GDELT conflict dataset."""

import os
import subprocess
import sys
from pathlib import Path

# Token (replace with your own from https://www.kaggle.com/settings)
TOKEN = os.environ.get("KAGGLE_API_TOKEN", "KGAT_3bd0c53d8767c060aca816c74ffa563e")

# Where to save kaggle.json
# In user's home directory: C:\Users\<USER>\.kaggle\kaggle.json
home = Path.home()
kaggle_dir = home / ".kaggle"
kaggle_dir.mkdir(parents=True, exist_ok=True)
kaggle_json = kaggle_dir / "kaggle.json"

kaggle_config = f'{{"username":"unknown","key":"{TOKEN}"}}'
kaggle_json.write_text(kaggle_config, encoding="utf-8")
print(f"Wrote: {kaggle_json}")

# Set env var
os.environ["KAGGLE_API_TOKEN"] = TOKEN

# Where to download
project_root = Path(__file__).resolve().parents[1]
external_dir = project_root / "data" / "external"
external_dir.mkdir(parents=True, exist_ok=True)

# Check if kaggle CLI is available
kaggle_check = subprocess.run(
    ["where", "kaggle"], capture_output=True, text=True, shell=True, check=False
)
print(f"kaggle location: {kaggle_check.stdout.strip()}")

# Use python -m kaggle as fallback
kaggle_cmd = [
    sys.executable,
    "-m",
    "kaggle",
    "datasets",
    "download",
    "-d",
    "masswheat/global-extremism-database-gdelt-2000-2022",
    "-p",
    str(external_dir),
    "--unzip",
]

print(f"\nDownloading dataset to: {external_dir}")
print(f"Command: {' '.join(kaggle_cmd)}")
result = subprocess.run(kaggle_cmd, capture_output=True, text=True, check=False)

print(f"\nReturn code: {result.returncode}")
print(f"STDOUT: {result.stdout[:500]}")
if result.stderr:
    print(f"STDERR: {result.stderr[:500]}")

# List files in external_dir
print(f"\nFiles in {external_dir}:")
for p in external_dir.iterdir():
    if p.is_file():
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name} ({size_mb:.1f} MB)")
    else:
        print(f"  {p.name}/ (dir)")
