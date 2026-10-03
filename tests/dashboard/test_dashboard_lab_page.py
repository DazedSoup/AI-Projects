"""Headless AppTest of the Simulation Lab page with ``subprocess`` mocked (no training is run)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from cyberarena import config
from cyberarena.dashboard import lab

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
import streamlit as st

APP = Path(config.__file__).parent / "dashboard" / "app.py"
FIX = Path(__file__).parent / "fixtures"
SPEC_TEXT = (FIX / "param_spec.json").read_text(encoding="utf-8")


class FakeSubprocess:
    """Stands in for the ``subprocess`` module inside ``cyberarena.dashboard.lab``."""

    TimeoutExpired = subprocess.TimeoutExpired
    DEVNULL = subprocess.DEVNULL

    def __init__(self, spec_text=SPEC_TEXT, rc=0, stderr=""):
        self.spec_text, self.rc, self.stderr = spec_text, rc, stderr
        self.runs, self.popens = [], []

    def run(self, cmd, **kw):
        self.runs.append(cmd)
        return subprocess.CompletedProcess(cmd, self.rc, self.spec_text if self.rc == 0 else "", self.stderr)

    def Popen(self, cmd, **kw):
        self.popens.append((cmd, kw))
        return type("P", (), {"pid": 424242})()


@pytest.fixture
def runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    shutil.copytree(FIX / "runs", runs)
    monkeypatch.setattr(config, "RUNS_DIR", runs)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    st.cache_data.clear()  # the spec is cached per process
    yield runs
    st.cache_data.clear()


def _lab(fake, monkeypatch):
    monkeypatch.setattr(lab, "subprocess", fake)
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    at.switch_page("lab_page.py").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def html(at) -> str:
    return " ".join(str(e.value) for e in at.get("html"))


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_lab_renders_form_from_spec(runs, monkeypatch):
    fake = FakeSubprocess()
    at = _lab(fake, monkeypatch)
    assert "<h1>Simulation Lab</h1>" in html(at)
    assert fake.runs and fake.runs[0][-1] == "--describe-params"
    assert {"Network", "Red", "Blue", "Opponents", "Training"} <= {t.label for t in at.tabs}
    assert any(e.label.startswith("Advanced") for e in at.expander)
    assert at.slider(key="lab_w:p_phish").value == 0.35
    assert at.checkbox(key="lab_r:n_nodes").value is True  # nullable default null -> random
    assert at.toggle(key="lab_w:no_turn_log").value is False
    assert at.selectbox(key="lab_w:baseline").value == "heuristic"
    assert at.number_input(key="lab_w:seed").value == 7
    assert at.segmented_control(key="lab_preset_sel").options == list(lab.PRESETS)
    # the replay fixture run shows up in the history table
    assert "20260101-000000-1" in at.dataframe[0].value["run_id"].tolist()


def test_lab_change_slider_and_run_builds_command(runs, monkeypatch):
    fake = FakeSubprocess()
    at = _lab(fake, monkeypatch)
    at.slider(key="lab_w:p_phish").set_value(0.5).run()
    _ok(at)
    assert any("●] changed (default 35%)" in c.value for c in at.caption)  # percent-formatted
    assert "<b>1 changed</b>" in html(at) and "<b>Red</b> 1" in html(at)
    at.checkbox(key="lab_r:n_nodes").uncheck().run()
    at.slider(key="lab_w:n_nodes").set_value(12).run()
    at.selectbox(key="lab_w:baseline").set_value("random").run()
    at.segmented_control(key="lab_preset_sel").set_value("Quick test").run()  # keeps the changes above
    at.text_input(key="lab_label").input("phish up").run()
    _ok(at)
    assert at.slider(key="lab_w:p_phish").value == 0.5
    at.button(key="lab_run").click().run()
    _ok(at)

    ((cmd, kw),) = fake.popens
    assert cmd[1:4] == ["-m", "cyberarena.dashboard.lab_runner", "--job"]
    job = json.loads(Path(cmd[4]).read_text(encoding="utf-8"))
    train = job["train_cmd"]
    assert train[1:3] == ["-m", "cyberarena.arena.train"]
    a = train[3:]
    assert a[a.index("--n-nodes") + 1] == "12"
    assert a[a.index("--baseline") + 1] == "random"
    assert a[a.index("--episodes") + 1] == "300" and a[a.index("--eval-every") + 1] == "100"
    assert json.loads(a[a.index("--env-json") + 1]) == {"p_phish": 0.5}
    assert a[a.index("--label") + 1] == "phish up"
    assert a[a.index("--runs-dir") + 1] == str(runs)
    assert "--init-from" not in a and "--init-side" not in a and "--online" not in a
    assert job["changed"] == {"n_nodes": 12, "p_phish": 0.5, "baseline": "random", "episodes": 300,
                              "eval_every": 100}  # fmt: skip
    assert job["pid"] == 424242 and job["stage"] == "starting"
    assert "ANTHROPIC_API_KEY" not in kw["env"]


def test_lab_running_job_shows_progress_and_blocks_second_run(runs, monkeypatch):
    fake = FakeSubprocess()
    monkeypatch.setattr(lab, "pid_alive", lambda pid: pid == 424242)
    run_dir = runs / "20260102-000000-7"
    run_dir.mkdir()
    (run_dir / "progress.json").write_text(json.dumps({"status": "running", "phase": "train", "episode": 150,
                                                       "episodes": 300, "last_eval": {"after_episode": 100,
                                                       "red_win_rate": 0.62, "blue_win_rate": 0.44}}))  # fmt: skip
    job = {"token": "t1", "job_file": str(lab.job_path(runs, "t1")), "created": lab.now_iso(),
           "started": lab.now_iso(), "stage": "training", "pid": 424242, "run_dir": str(run_dir),
           "run_id": run_dir.name, "label": "live"}  # fmt: skip
    lab.save_job(job)
    at = _lab(fake, monkeypatch)
    assert "Running: live" in html(at)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Game"] == "150 / 300"
    assert (
        metrics["Learned red vs scripted blue"] == "62%" and metrics["Learned blue vs scripted red"] == "44%"
    )
    assert metrics["Stage"] == "training · train"
    assert at.button(key="lab_run").disabled
    at.button(key="lab_stop").click().run()
    _ok(at)
    assert lab.stop_requested_at(job) is not None  # graceful: a stop request for the runner
    assert any("Stopping" in w.value for w in at.warning)


def test_lab_finished_job_opens_in_replay(runs, monkeypatch):
    fake = FakeSubprocess()
    run_dir = runs / "20260101-000000-1"
    job = {"token": "t2", "job_file": str(lab.job_path(runs, "t2")), "created": lab.now_iso(),
           "started": lab.now_iso(), "finished": lab.now_iso(), "stage": "done", "pid": 1,
           "run_dir": str(run_dir), "run_id": run_dir.name, "label": "fixture", "changed": {"p_phish": 0.5}}  # fmt: skip
    lab.save_job(job)
    monkeypatch.setattr(lab, "subprocess", fake)
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["lab_token"] = "t2"
    at.run()
    at.switch_page("lab_page.py").run()
    _ok(at)
    assert any("fixture" in s.value and "finished" in s.value for s in at.success)
    at.button(key="lab_open_new").click().run()
    _ok(at)
    assert "<h1>Replay</h1>" in html(at)  # replay page, with the run selected
    assert at.session_state["run"] == run_dir


def test_lab_spec_failure_shows_fix(runs, monkeypatch):
    fake = FakeSubprocess(rc=2, stderr="train: error: unrecognized arguments: --describe-params")
    at = _lab(fake, monkeypatch)
    assert any("unrecognized arguments" in e.value for e in at.error)
    assert any("How to fix" in m.value for m in at.markdown)
    assert at.dataframe  # history still renders
    fake.rc = 0
    at.button(key="lab_retry").click().run()
    _ok(at)
    assert at.slider(key="lab_w:p_phish").value == 0.35
