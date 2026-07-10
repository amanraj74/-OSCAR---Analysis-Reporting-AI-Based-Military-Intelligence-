import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import reset_settings_cache

reset_settings_cache()

from dashboard.utils import get_recent_anomalies

df = get_recent_anomalies.__wrapped__(50)
print("shape:", df.shape)
print("columns:", list(df.columns))
print("empty:", df.empty)
