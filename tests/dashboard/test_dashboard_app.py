"""Headless AppTest of every page against the v1 fixture, the synthetic v2 fixture and (if present) the
real reference run."""

import shutil
from pathlib import Path

import pytest

from cyberarena import config

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
import streamlit as st

APP = Path(config.__file__).parent / "dashboard" / "app.py"
FIX = Path(__file__).parent / "fixtures"
V1 = "20260101-000000-1"
V2 = "20261003-101500-7"
PAGES = ("overview_page.py", "replay_page.py", "learning_page.py", "evidence_page.py", "lab_page.py")


@pytest.fixture
def runs(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    shutil.copytree(FIX / "runs", runs)
    shutil.copytree(FIX / "runs_v2" / V2, runs / V2)
    monkeypatch.setattr(config, "RUNS_DIR", runs)
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # keep the index cache out of the real temp dir
    st.cache_data.clear()
    yield runs
    st.cache_data.clear()


def _ok(app):
    assert not app.exception, [e.value for e in app.exception]


def html(app) -> str:
    return " ".join(str(e.value) for e in app.get("html"))


def _open(runs: Path, run_id: str, page: str = "overview_page.py"):
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.session_state["run"] = runs / run_id
    app.run()
    _ok(app)
    if page != "overview_page.py":
        app.switch_page(page).run()
        _ok(app)
    return app


@pytest.mark.parametrize("run_id", [V1, V2])
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(runs, run_id, page):
    app = _open(runs, run_id, page)
    assert "ca-page-head" in html(app)


REAL_RUNS = {
    "20261003-110349-7": "v5 reference (DQN, current rules)",
    "20261003-110348-7": "v5 tabular comparison",
    "20261003-013911-7": "v4 reference (DQN, cheap isolation)",
    "20261003-014109-7": "v4 tabular comparison",
    "20261002-171105-7": "v1 reference",
    "20261002-200243-7": "adaptive reference",
    "20261002-200448-7": "non-adaptive comparison",
}


@pytest.mark.parametrize("run_id", list(REAL_RUNS))
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_real_runs(monkeypatch, run_id, page):
    real = Path(config.ROOT) / "runs"
    if not (real / run_id).exists():
        pytest.skip(f"{run_id} not on disk")
    monkeypatch.setattr(config, "RUNS_DIR", real)  # read-only; the on-disk index cache is reused
    st.cache_data.clear()
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.session_state["run"] = real / run_id
    app.run()
    if page != "overview_page.py":
        app.switch_page(page).run()
    _ok(app)
    h = html(app)
    if page == "learning_page.py":
        if run_id == "20261002-171105-7":
            assert "predates adaptive learning" in h
        elif run_id == "20261002-200448-7":
            assert "Detectors were frozen in this run" in h and "Adaptive vs frozen detectors" in h
        elif run_id in ("20261003-013911-7", "20261003-110349-7"):
            assert "Agent learning · Q-networks" in h and "Adaptive vs frozen detectors" not in h  # no DQN partner
        elif run_id in ("20261003-014109-7", "20261003-110348-7"):
            assert "Agent learning · Q-tables" in h
        else:
            assert "Phishing does not oscillate" in h and "Adaptive vs frozen detectors · one seed" in h
    if page == "evidence_page.py":
        assert "Did it work?" in h or "No experiments yet" in h
    if page == "replay_page.py" and run_id in ("20261003-013911-7", "20261003-110349-7"):
        assert "Chosen vs runner-up" in h
    if page == "replay_page.py" and run_id == "20261002-200243-7":
        app.segmented_control(key="rp_mode").set_value("Same game, earlier agent").run()
        _ok(app)
        assert "Same seed, same opponent" in html(app)


def test_run_selector_labels_and_caption(runs):
    app = _open(runs, V2)
    sel = app.selectbox(key="run")
    assert sel.value == runs / V2
    assert any("synthetic v2 fixture · 1,500 games · finished" in o for o in sel.options)
    assert any(
        "Reads `runs/20261003-101500-7/`" in c.value and "learning telemetry" in c.value for c in app.caption
    )
    assert "Adaptive learning" in html(app)


def test_overview_v1_shows_predates_and_opens_replay(runs):
    app = _open(runs, V1)
    h = html(app)
    assert "This run predates adaptive learning" in h
    assert "Learned red vs scripted blue" in h
    app.button(key="ov_open_replay").click().run()
    _ok(app)
    assert app.session_state["episode"] == 5  # the narrated game


def test_overview_v2_tiles(runs):
    h = html(_open(runs, V2))
    assert "Detector updates" in h and ">32<" in h
    assert "Red evasion now" in h
    assert "recall at its level" in h  # computed arms-race takeaway


def test_replay_steps_through_narrated_game(runs):
    app = _open(runs, V1, "replay_page.py")
    assert app.session_state["episode"] == 5
    assert "Narrated offline (template)" in html(app)
    for _ in range(5):
        app.button(key="rp_next").click().run()
        _ok(app)
    assert app.session_state["turn"] == 5
    assert "Game over" in html(app)
    app.button(key="rp_prev").click().run()
    _ok(app)
    assert app.session_state["turn"] == 4
    app.segmented_control(key="rp_view").set_value("2D").run()
    _ok(app)


def test_replay_other_game_types(runs):
    app = _open(runs, V1, "replay_page.py")
    assert app.selectbox(key="rp_kind").options[0].startswith("Narrated")
    app.selectbox(key="rp_kind").set_value("train").run()
    _ok(app)
    assert app.session_state["episode"] in (0, 6)
    app.slider(key="turn").set_value(1).run()
    _ok(app)


def test_replay_v2_adaptation_line_and_probe_compare(runs):
    app = _open(runs, V2, "replay_page.py")  # narrated probe game, turn 0: red recon read by a detector
    assert "Adaptation this turn" in html(app) and "evasion 0." in html(app) and "detector v" in html(app)
    app.segmented_control(key="rp_mode").set_value("Same game, earlier agent").run()
    _ok(app)
    assert app.selectbox(key="cmp_a").value == 0 and app.selectbox(key="cmp_b").value == 1500
    h = html(app)
    assert "After 0 training games" in h and "After 1,500 training games" in h
    assert "Same seed, same opponent" in h
    app.slider(key="cmp_turn").set_value(10).run()
    _ok(app)


def test_replay_v1_compare_is_empty_state(runs):
    app = _open(runs, V1, "replay_page.py")
    app.segmented_control(key="rp_mode").set_value("Same game, earlier agent").run()
    _ok(app)
    assert "No probe games in this run" in html(app)


def test_learning_v2_controls(runs):
    app = _open(runs, V2, "learning_page.py")
    h = html(app)
    for title in (
        "Arms race",
        "Learning landscape",
        "Detector versions",
        "Agent learning",
        "Probe situations",
    ):
        assert title in h
    app.selectbox(key="lr_model").set_value("malware").run()
    app.segmented_control(key="lr_side").set_value("blue").run()
    app.segmented_control(key="lr_view").set_value("2D").run()
    _ok(app)
    assert "learned blue" in html(app)


def test_learning_v1_empty_state(runs):
    h = html(_open(runs, V1, "learning_page.py"))
    assert "This run predates adaptive learning" in h
