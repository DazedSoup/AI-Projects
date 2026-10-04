"""cyberarena dashboard (simulation only).

Launch:  .venv/Scripts/python.exe -m streamlit run src/cyberarena/dashboard/app.py

Pages (top navigation, in this order):

* **Overview** (``overview_page.py``): headline metrics and charts for the selected run.
* **Replay** (``replay_page.py``): step through one game on the 3D network, with the move log and the
  reasoning behind each move; compare the same probe game at two checkpoints.
* **Learning** (``learning_page.py``): how the detectors and the agents learned, from ``learning.jsonl``.
* **Evidence** (``evidence_page.py``): multi-seed experiments: adaptive vs frozen contrasts with intervals and
  paired p-values, per-seed curves, the arms race across seeds, factorial diagnoses.
* **Simulation Lab** (``lab_page.py``): launch, watch and compare training runs (subprocesses only), and choose
  what to publish to the public showcase. Not registered in public mode (``CYBERARENA_PUBLIC=1``).

One run selector in the sidebar (``common.run_selector``) drives every page. The dashboard only reads run
files; it never imports arena, ml or explain code. Runs are found under ``cyberarena.config.RUNS_DIR``
(override with ``CYBERARENA_RUNS_DIR``, or the root with ``CYBERARENA_ROOT``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

try:  # hosted bundles run the app straight from src/ without installing the package
    import cyberarena  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cyberarena.dashboard import common, theme

st.set_page_config(page_title="cyberarena", layout="wide", page_icon=":material/shield:")
theme.inject()

PAGES = [
    st.Page(common.PAGES["overview"], title="Overview", icon=":material/space_dashboard:", default=True),
    st.Page(common.PAGES["replay"], title="Replay", icon=":material/play_circle:", url_path="replay"),
    st.Page(common.PAGES["learning"], title="Learning", icon=":material/neurology:", url_path="learning"),
    st.Page(common.PAGES["evidence"], title="Evidence", icon=":material/fact_check:", url_path="evidence"),
]
if common.admin():  # public showcase (CYBERARENA_PUBLIC=1): the Lab is not registered at all, so /lab doesn't exist
    PAGES.append(st.Page(common.PAGES["lab"], title="Simulation Lab", icon=":material/science:", url_path="lab"))
nav = st.navigation(PAGES, position="top")
common.run_selector()
nav.run()
