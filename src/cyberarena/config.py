"""Shared paths. Every phase resolves locations through here so nothing hardcodes cwd."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("CYBERARENA_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "artifacts" / "models"
METRICS_DIR = ROOT / "artifacts" / "metrics"
# A deploy bundle (python -m cyberarena.deploy) carries this marker at its root, so a hosted copy switches itself to
# the public showcase without any host-side environment settings.
PUBLIC_MARKER = ROOT / "PUBLIC_SHOWCASE"
_bundle = PUBLIC_MARKER.exists()

if os.environ.get("CYBERARENA_RUNS_DIR"):
    RUNS_DIR = Path(os.environ["CYBERARENA_RUNS_DIR"])
else:
    RUNS_DIR = ROOT / ("showcase" if _bundle else "runs")

# Public read-only showcase (hosted dashboard): no Lab, no subprocesses, no writes. See docs/contracts.md v6.
_public_env = os.environ.get("CYBERARENA_PUBLIC", "").strip().lower()
PUBLIC = _public_env in ("1", "true", "yes", "on") if _public_env else _bundle

CLASSIFIERS = ("malware", "phishing", "network")
