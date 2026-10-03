"""Shared page plumbing: the global run selector, cached loaders keyed by file stamps, layout helpers.

The run selector lives in the sidebar and is rendered by ``app.py`` on every page, so every page reads the same
run from ``st.session_state["run"]``. The Lab's "Open in Replay" sets ``st.session_state["_open_run"]`` (a run
id) before switching pages; :func:`run_selector` honours it.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from cyberarena import config
from cyberarena.dashboard import lab
from cyberarena.dashboard import learning as LL
from cyberarena.dashboard import loaders as L
from cyberarena.dashboard import theme as T

ss = st.session_state

PAGES = {
    "overview": "overview_page.py",
    "replay": "replay_page.py",
    "learning": "learning_page.py",
    "evidence": "evidence_page.py",
    "lab": "lab_page.py",
}
FILE_WORDS = {
    L.SUMMARY_FILE: "win rates",
    L.LEARNING_FILE: "learning telemetry",
    L.EPISODES_FILE: "game log",
    L.ENRICHED_FILE: "narration",
    L.GRAPH_FILE: "network",
}


def runs_dir() -> Path:
    return Path(config.RUNS_DIR)


# ============================================================================================== cached loads


def info(run: Path) -> dict:
    return L.cached_run_info(str(run), L.run_stamp(run))


def summary(run: Path):
    return L.cached_summary(str(run), L.file_stamp(Path(run) / L.SUMMARY_FILE))


def enriched(run: Path) -> dict[int, list[dict]]:
    return L.cached_enriched(str(run), L.file_stamp(Path(run) / L.ENRICHED_FILE))


@st.cache_data(show_spinner=False, max_entries=16)
def _learning(run: str, stamp) -> LL.Learning | None:
    return LL.load_learning(Path(run))


def learning(run: Path) -> LL.Learning | None:
    return _learning(str(run), L.file_stamp(Path(run) / L.LEARNING_FILE))


def q_states(run: Path):
    ck = Path(run) / "agents" / "checkpoints"
    stamp = L.file_stamp(ck) if ck.exists() else None
    return L.cached_q_states(str(run), stamp)


def graph(run: Path) -> dict | None:
    return L.cached_graph(str(run)) if (Path(run) / L.GRAPH_FILE).exists() else None


def index_meta(run: Path) -> dict[int, dict] | None:
    """Per-game metadata from the log index (builds the index on first use; cached on disk afterwards)."""
    big = Path(run) / L.EPISODES_FILE
    if not big.exists():
        return None
    s = big.stat()
    idx = L.cached_index(str(run), s.st_size, s.st_mtime_ns)
    return idx.meta if idx else None


def game_turns(run: Path, episode: int) -> list[dict]:
    """Turn records of one game: the narrated copy if there is one, else from the big log."""
    enr = enriched(run)
    if episode in enr:
        return enr[episode]
    big = Path(run) / L.EPISODES_FILE
    if not big.exists():
        return []
    s = big.stat()
    return L.cached_episode(str(run), int(episode), s.st_size, s.st_mtime_ns)


# ============================================================================================== run selector


def _brand() -> None:
    st.sidebar.html(
        '<div class="ca-brand"><div class="mark">ca</div><div><div class="name">cyberarena</div>'
        '<div class="tag">Simulated red vs blue · reinforcement learning</div></div></div>'
    )


def run_selector() -> Path | None:
    """The one global run picker. Returns the selected run directory (``None`` if there are no runs)."""
    _brand()
    runs = L.list_runs(runs_dir())
    runs = ordered_runs(runs) if runs else runs
    if not runs:
        st.sidebar.caption(f"No runs under `{runs_dir()}` yet.")
        return None
    opened = ss.pop("_open_run", None)
    if opened is None and "run" not in ss and st.query_params.get("run"):
        opened = st.query_params.get("run")  # shareable link: ?run=<run id>
    if opened is not None:
        match = next((p for p in runs if p.name == Path(opened).name), None)
        if match is not None:
            ss["run"] = match
            reset_game()
    if ss.get("run") not in runs:
        ss["run"] = default_run(runs)
    labels = {p: L.run_label(info(p)) for p in runs}
    run = st.sidebar.selectbox(
        "Training run", runs, format_func=labels.get, key="run", on_change=reset_game,
        help="Every page except Evidence shows this run. Stand-alone runs come first (newest first), then the "
        "per-seed runs of experiments. The run labelled 'reference' is selected by default.",
    )  # fmt: skip
    i = info(run)
    have = [FILE_WORDS[f] for f in i["files"] if f in FILE_WORDS]
    st.sidebar.caption(
        f"Reads `runs/{run.name}/` — " + (", ".join(have) if have else "no log files yet") + "."
    )
    status = {"done": "Finished", "running": "Training", "training": "Training", "starting": "Starting",
              "enriching": "Narrating", "error": "Stopped early"}.get(i["status"], i["status"] or "unknown")  # fmt: skip
    if i["adaptive"]:
        adaptive = T.badge("Adaptive learning", "learn", T.VIOLET)
    elif i["frozen"]:
        adaptive = T.badge("Frozen detectors (comparison run)", "", T.tok("neutral"))
    else:
        adaptive = T.badge("Predates adaptive learning", "", T.tok("neutral"))
    st.sidebar.html(
        f'<dl class="ca-run"><dt>Run id</dt><dd>{T.esc(run.name)}</dd><dt>Status</dt><dd>{T.esc(status)}</dd>'
        f"<dt>Training games</dt><dd>{i['games']:,}</dd></dl>"
        if i["games"]
        else f'<dl class="ca-run"><dt>Run id</dt><dd>{T.esc(run.name)}</dd><dt>Status</dt><dd>{T.esc(status)}</dd></dl>'
    )
    kind = L.agent_kind(i)
    extra = T.badge(kind) if kind else ""
    st.sidebar.html(f'<div class="ca-badges">{adaptive}{extra}</div>')
    if i["status"] in lab.ACTIVE_STAGES or i["status"] == "running":
        st.sidebar.info(
            "Still running in the Lab: pages show the data written so far.", icon=":material/sync:"
        )
    return run


def default_run(runs: list[Path]) -> Path:
    """The newest finished stand-alone run labelled "reference"; else the newest finished stand-alone run with
    adaptive learning and narration; else the newest finished run. Experiment runs are never the default."""
    solo = [r for r in runs if not L.is_experiment_run(info(r))]
    for pick in (
        lambda i: i["status"] == "done" and (i["label"] or "").strip().lower() == "reference" and i["has_narration"],
        lambda i: i["status"] == "done" and (i["label"] or "").strip().lower() == "reference",
        lambda i: i["status"] == "done" and i["adaptive"] and i["has_narration"],
    ):
        for r in solo:
            if pick(info(r)):
                return r
    return runs[lab.default_replay_run(runs)]


def ordered_runs(runs: list[Path]) -> list[Path]:
    """Stand-alone runs first (newest first), then the per-seed runs of experiments (by name, newest first)."""
    solo = [r for r in runs if not L.is_experiment_run(info(r))]
    return solo + [r for r in runs if r not in solo]


def reset_game() -> None:
    for k in ("episode", "turn", "playing", "rp_kind"):
        ss.pop(k, None)


def current_run() -> Path:
    run = ss.get("run")
    if run is None:
        T.empty_state(
            "No training runs yet", "Launch one from the Simulation Lab, or train from the command line."
        )
        st.stop()
    return Path(run)


# ============================================================================================== layout


def card(key: str):
    """A keyed container styled as a card (see ``theme.py``: ``[class*="st-key-card"]``)."""
    return st.container(key=f"card-{key}")


def toolbar(key: str):
    return st.container(key=f"toolbar-{key}", horizontal=True, vertical_alignment="bottom", gap="medium")


def open_replay(episode: int | None = None) -> None:
    if episode is not None:
        ss["_open_episode"] = int(episode)
    st.switch_page(PAGES["replay"])


def comparison_partners(run: Path) -> list[Path]:
    """Runs to compare against: frozen-detector runs for an adaptive run and vice versa, with the same learner
    and seed (so the only difference is the detector setting), same length first. Experiment runs are left to
    the Evidence page."""
    me = info(run)
    if not (me["adaptive"] or me["frozen"]):
        return []
    out = []
    for r in L.list_runs(runs_dir()):
        if r == Path(run):
            continue
        i = info(r)
        if (i["status"] == "done" and ((me["adaptive"] and i["frozen"]) or (me["frozen"] and i["adaptive"]))
                and i.get("agent_type") == me.get("agent_type") and i.get("seed") == me.get("seed")
                and L.is_experiment_run(i) == L.is_experiment_run(me)):  # fmt: skip
            out.append(r)
    out.sort(key=lambda r: r.name, reverse=True)  # newest first ...
    return sorted(out, key=lambda r: info(r)["games"] != me["games"])  # ... same length first (stable)
