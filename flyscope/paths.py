"""Repository-owned assets, resolved from source rather than the shell CWD."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
LEGACY_MODELS = ROOT / "models" / "legacy"
DATASET_FORMAT = Path(__file__).resolve().parent / "data" / "format.py"
