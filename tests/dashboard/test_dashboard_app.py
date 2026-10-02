"""Headless AppTest of the Streamlit app against the fixture run."""

from pathlib import Path

import pytest

from cyberarena import config

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(config.__file__).parent / "dashboard" / "app.py"
FIXTURE_RUNS = Path(__file__).parent / "fixtures" / "runs"


@pytest.fixture
def at(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", FIXTURE_RUNS)
    monkeypatch.setenv("TMP", str(tmp_path))  # keep the index cache out of the real temp dir
    monkeypatch.setenv("TEMP", str(tmp_path))
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()
    return app


def _ok(app):
    assert not app.exception, [e.value for e in app.exception]


def test_app_steps_through_enriched_episode(at):
    _ok(at)
    assert at.title[0].value == "Episode 5"
    assert any(m.value == ":orange-badge[Rationale: offline template]" for m in at.markdown)
    for _ in range(5):
        at.button[2].click().run()  # step forward
        _ok(at)
    assert at.session_state["turn"] == 5
    assert any("red" in s.value and "wins" in s.value for s in at.success)
    at.button[1].click().run()  # step back
    _ok(at)
    assert at.session_state["turn"] == 4


def test_app_replays_non_enriched_episode(at):
    at.selectbox(key="episode").set_value(6).run()
    _ok(at)
    assert at.title[0].value == "Episode 6"
    assert at.session_state["turn"] == 0
    at.slider(key="turn").set_value(1).run()
    _ok(at)


def test_app_episode_without_records(at):
    at.selectbox(key="episode").set_value(7).run()
    _ok(at)
    assert any("no turn records" in w.value for w in at.warning)
