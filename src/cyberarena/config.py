"""Shared paths. Every phase resolves locations through here so nothing hardcodes cwd."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "artifacts" / "models"
METRICS_DIR = ROOT / "artifacts" / "metrics"
RUNS_DIR = ROOT / "runs"

CLASSIFIERS = ("malware", "phishing", "network")
