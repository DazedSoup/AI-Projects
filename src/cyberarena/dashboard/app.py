"""cyberarena dashboard (simulation only): Replay and Simulation Lab.

Launch:  .venv/Scripts/python.exe -m streamlit run src/cyberarena/dashboard/app.py

* **Replay** (``replay_page.py``, default): reads run logs, never runs agents or calls APIs.
* **Simulation Lab** (``lab_page.py``): builds a form from ``train --describe-params``, launches training plus
  offline enrichment as a detached subprocess, shows live progress and compares runs.

Runs are discovered under ``cyberarena.config.RUNS_DIR`` (override the root with ``CYBERARENA_ROOT``).
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="cyberarena", layout="wide")

REPLAY = st.Page("replay_page.py", title="Replay", icon=":material/replay:", default=True)
LAB = st.Page("lab_page.py", title="Simulation Lab", icon=":material/science:")
st.navigation([REPLAY, LAB]).run()
