"""v3/v4 dashboard: DQN move-level decisions, the Evidence page (multi-seed experiments and factorial
diagnoses) and the Lab's experiment mode. Fixtures are cut from real runs by ``fixtures/make_v4_fixture.py``."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from cyberarena import config
from cyberarena.dashboard import evidence as E
from cyberarena.dashboard import lab
from cyberarena.dashboard import loaders as L

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
import streamlit as st

APP = Path(config.__file__).parent / "dashboard" / "app.py"
FIX = Path(__file__).parent / "fixtures"
V4FIX = FIX / "runs_v4"
V4, V3 = "20261003-110349-7", "20261003-110348-7"  # v5 rules: reference (DQN), tabular comparison
PAGES = ("overview_page.py", "replay_page.py", "learning_page.py", "evidence_page.py", "lab_page.py")


SPEC_V4 = (FIX / "param_spec_v4.json").read_text(encoding="utf-8")


class _SpecOnly:
    """``subprocess`` stand-in for the Lab: ``--describe-params`` returns the v4 spec; nothing is launched."""

    TimeoutExpired = subprocess.TimeoutExpired
    DEVNULL = subprocess.DEVNULL

    def run(self, cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, SPEC_V4, "")

    def Popen(self, cmd, **kw):  # pragma: no cover - the page tests never press Run
        raise AssertionError("no process may be started from a page test")


@pytest.fixture
def runs(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    shutil.copytree(V4FIX, runs)
    monkeypatch.setattr(config, "RUNS_DIR", runs)
    monkeypatch.setattr(lab, "subprocess", _SpecOnly())
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    st.cache_data.clear()
    yield runs
    st.cache_data.clear()


def _ok(app):
    assert not app.exception, [e.value for e in app.exception]


def html(app) -> str:
    return " ".join(str(e.value) for e in app.get("html"))


def _open(runs: Path, run_id: str | None, page: str = "overview_page.py", **state):
    app = AppTest.from_file(str(APP), default_timeout=90)
    if run_id:
        app.session_state["run"] = runs / run_id
    for k, v in state.items():
        app.session_state[k] = v
    app.run()
    _ok(app)
    if page != "overview_page.py":
        app.switch_page(page).run()
        _ok(app)
    return app


def _turn(run: Path, episode: int | None = None) -> list[dict]:
    enr = L.load_enriched(run)
    return enr[episode if episode is not None else min(enr)]


# ============================================================================================== pages


@pytest.mark.parametrize("run_id", [V4, V3])
@pytest.mark.parametrize("page", PAGES)
def test_dashboard_v4_every_page_renders(runs, run_id, page):
    app = _open(runs, run_id, page)
    assert "ca-page-head" in html(app)


def test_dashboard_v4_default_run_is_the_reference(runs):
    app = _open(runs, None)
    assert app.session_state["run"] == runs / V4
    assert "DQN agents" in html(app)


def test_dashboard_v4_overview_hero_is_honest(runs):
    h = html(_open(runs, V4))
    assert "Learned blue beats scripted red" in h and ">74%<" in h  # reference run, final checkpoint
    assert "Detector recall on disguised red" in h and "2–8×" in h and "network 9%→69%" in h
    assert "suggestive but not established" in h  # the defender's gain is not claimed
    assert "helps blue" not in h and "wins more" not in h
    assert "How it works" in h and "Gradient steps" in h and "Q-states learned" not in h


def test_dashboard_v4_replay_moves_ghosts_and_runner_up(runs):
    app = _open(runs, V4, "replay_page.py")
    h = html(app)
    assert "Chosen vs runner-up" in h and "Q margin" in h
    assert "integrated gradients" in h and "Detector reading this turn" in h
    assert "__caReplayKeys" in h  # keyboard shortcuts
    net = app.get("plotly_chart")[0]  # the network is the page's first chart
    spec = json.loads(net.proto.spec)
    texts = [t.get("text") for t in spec["data"] if t.get("text")]
    assert any(isinstance(x, list) and x and str(x[0]).startswith("#") for x in texts)  # ghost candidates
    app.button(key="rp_next").click().run()
    _ok(app)
    assert app.session_state["turn"] == 1
    app.segmented_control(key="rp_view").set_value("2D").run()
    _ok(app)


def test_dashboard_v4_replay_evasive_probe_compare(runs):
    app = _open(runs, V4, "replay_page.py")
    app.segmented_control(key="rp_mode").set_value("Same game, earlier agent").run()
    _ok(app)
    opts = app.selectbox(key="cmp_matchup").options
    assert any("disguised at 0.4" in o for o in opts)


def test_dashboard_v4_learning_dqn_stats(runs):
    h = html(_open(runs, V4, "learning_page.py"))
    assert "Agent learning · Q-networks" in h and "gradient steps" in h
    assert "Agent learning · Q-tables" in html(_open(runs, V3, "learning_page.py"))


# ============================================================================================== evidence


def test_dashboard_evidence_main_current_rules(runs):
    h = html(_open(runs, V4, "evidence_page.py"))
    assert "Did it work?" in h and "Suggestive, not established: +7 pts vs red disguised at 0.7" in h
    assert "Learned red wins 6 pts more vs scripted blue with adaptive detectors" in h and "p = 0.004" in h
    assert "No significant difference vs red disguised at 0.7" in h
    assert "Arms race across seeds" in h
    assert "Cross-evaluation: same blue agent" not in h  # that card belongs to the previous-rules runs
    assert "DQN vs tabular agents, same seed" in h
    # the previous-rules finding and its diagnosis, as a documented lesson
    assert "What we learned along the way" in h
    assert "Learned blue wins 19 pts less vs scripted red with adaptive detectors" in h
    assert "Detectors: learned blue wins 10 pts less" in h and "Red disguise: no significant effect" in h


def test_dashboard_evidence_previous_rules_experiment(runs):
    app = _open(runs, V4, "evidence_page.py")
    app.button(key="ev_open_old").click().run()
    _ok(app)
    h = html(app)
    assert app.session_state["ev_exp"] == "main-cheap-isolation"
    assert "previous rules (cheap isolation)" in h and "Cross-evaluation: same blue agent" in h
    assert "Learned blue wins 19 pts less vs scripted red" in h


def test_dashboard_evidence_other_experiments(runs):
    app = _open(runs, V4, "evidence_page.py")
    opts = app.selectbox(key="ev_exp").options  # formatted labels
    assert any(o.startswith("diag-* factorial · 4 cells") for o in opts) and any("running" in o for o in opts)
    app.selectbox(key="ev_exp").set_value("smoke-dqn").run()
    _ok(app)
    assert "With this few seeds only a large effect would show" in html(app)
    app.segmented_control(key="ev_stat").set_value("final").run()
    _ok(app)
    app.selectbox(key="ev_exp").set_value("family:diag").run()
    _ok(app)
    h = html(app)
    assert "Detectors:" in h and "Red disguise:" in h and "interaction" in h.lower()
    app.selectbox(key="ev_exp").set_value("fx-running").run()
    _ok(app)
    assert "Results appear when every run has finished" in html(app)
    app.selectbox(key="ev_exp").set_value("diag-detF-evOff").run()
    _ok(app)
    assert "frozen detectors · red never disguising" in html(app)


def test_dashboard_evidence_empty_state(runs):
    shutil.rmtree(runs / "experiments")
    app = _open(runs, V4, "evidence_page.py")
    assert "No experiments yet" in html(app)
    app.button(key="ev_to_lab").click().run()
    _ok(app)
    assert app.session_state["lab_kind"] == "Experiment"


def test_dashboard_evidence_verdicts():
    neg = {"metric": "blue_learned_vs_red_baseline late_mean", "a": "adaptive", "b": "frozen", "diff_mean": -0.1856,
           "ci95": [-0.3196, -0.0516], "p_value": 0.018, "per_seed_diff": [-0.31, -0.075, -0.209, -0.261, -0.073]}
    v = E.verdict(neg)
    assert v.startswith("Learned blue wins 19 pts less vs scripted red with adaptive detectors")
    assert "95% CI −32 to −5 pts" in v and "p = 0.02" in v and "every seed agrees" in v
    ns = {**neg, "diff_mean": -0.15, "ci95": [-0.34, 0.04], "p_value": 0.095,
          "per_seed_diff": [-0.3, 0.03, -0.2, -0.28, -0.01]}
    v = E.verdict(ns)
    assert v.startswith("No significant difference") and "4 of 5 seeds lower" in v
    pos = {**neg, "metric": "blue_learned_vs_red_evasive@0.7 late_mean", "diff_mean": 0.21, "ci95": [0.15, 0.27],
           "p_value": 0.003, "per_seed_diff": [0.2, 0.25, 0.18]}
    assert E.verdict(pos).startswith("Learned blue wins 21 pts more vs red disguised at 0.7")


def test_dashboard_evidence_did_it_work_and_factorial():
    agg = json.loads((V4FIX / "experiments" / "main" / "aggregate.json").read_text(encoding="utf-8"))
    rows = {r["key"]: r for r in E.did_it_work(agg)}
    assert [rows[k]["answer"] for k in ("detectors", "red", "blue")] == ["yes", "yes", "suggestive"]
    old = json.loads((V4FIX / "experiments" / "main-cheap-isolation" / "aggregate.json").read_text(encoding="utf-8"))
    rows = {r["key"]: r for r in E.did_it_work(old)}
    assert rows["blue"]["answer"] == "no" and rows["red"]["answer"] == "unclear"
    exps = E.list_experiments(V4FIX)
    assert E.default_experiment(exps)["name"] == "main"
    fam = E.factorial_families(exps)
    assert [f["name"] for f in fam] == ["diag"] and len(fam[0]["cells"]) == 4
    eff = E.factorial_effects(fam[0], "blue_learned_vs_red_baseline")
    assert [e["kind"] for e in eff] == ["main effect", "main effect", "interaction"]
    assert all(len(e["per_seed_diff"]) == 5 for e in eff)


def test_dashboard_evidence_crosseval_labels():
    xe = E.load_crosseval(V4FIX)
    assert set(xe["trained_with"]) == {"adaptive", "frozen"}
    assert set(xe["detectors"]) == {"v0", "final", "shuffled"}


# ============================================================================================== loaders


def test_dashboard_v4_candidates_and_headline():
    run = V4FIX / V4
    g = L.load_graph(run)
    t = next(x for x in _turn(run) if x.get("candidates"))
    cands = L.turn_candidates(t)
    assert sum(c["chosen"] for c in cands) == 1 and cands == sorted(cands, key=lambda c: -c["q"])
    head = L.move_headline(t, g)
    assert "→ host" in head and "(" in head
    web = L.decision_web_v4(t, g)
    assert web["mode"] == "moves" and len(web["moves"]) <= 6
    assert sum(m["runner_up"] for m in web["moves"]) == 1
    assert web["attribution"] and all("phrase" in f for f in web["attribution"])
    cmp = L.chosen_vs_runner_up(t, None, g)
    assert cmp is not None and cmp["ranked_by_attribution"] and len(cmp["facts"]) <= 3


def test_dashboard_v4_tabular_turns_have_no_candidates():
    idx = L.build_episode_index(V4FIX / V3)
    for ep in idx.episodes:
        for t in L.read_episode(idx, ep):
            assert L.turn_candidates(t) == []
    assert L.parse_move_key("lateral_move→13") == ("lateral_move", 13) and L.parse_move_key("wait") == ("wait", None)


def test_dashboard_v4_feature_phrases():
    assert L.feature_phrase("t_workstation", {"value": 0.0}) == "target isn't a workstation"
    assert L.feature_phrase("t_workstation", {"value": 1.0}) == "target is a workstation"
    assert L.feature_phrase("a_phish", {"value": 0.0}) == "move is not phish"
    assert L.feature_phrase("t_dist") == "target's hops to the crown jewel"
    assert L.feature_phrase("x_new_thing") == "x new thing"
    assert L.feature_phrase("t_dist", {"phrase": "from explain"}) == "from explain"


def test_dashboard_v4_run_info_and_has_agents():
    i = L.run_info(V4FIX / V4)
    assert i["agent_type"] == "dqn" and i["seed"] == 7 and not L.is_experiment_run(i)
    assert L.run_info(V4FIX / V3)["agent_type"] == "tabular"
    assert lab.has_agents(V4FIX / V4) and lab.has_agents(V4FIX / V3)
    assert lab.agent_type(V4FIX / V4) == "dqn"
    assert L.is_experiment_run({"label": "main · adaptive · seed 1"})


def test_dashboard_v4_catalog_keeps_disguise_levels():
    run = V4FIX / V4
    eps, evals = L.load_summary(run)
    cat = L.episode_catalog(L.load_enriched(run), eps, evals, L.build_episode_index(run).meta)
    pr = L.probe_games(cat)
    assert "blue_learned_vs_red_evasive@0.4" in set(pr["probe_key"])
    row = cat[cat["matchup"] == "blue_learned_vs_red_evasive"].iloc[0]
    assert "disguised at 0.4" in row["label"]


# ============================================================================================== lab: experiments


def test_dashboard_lab_experiment_cmd_and_args():
    extra = lab.experiment_extra_args(["--episodes", "300", "--agent", "tabular", "--no-adaptive", "--label", "x",
                                       "--runs-dir", "r", "--detectors", "frozen", "--eval-n", "50"])
    assert extra == ["--agent", "tabular", "--eval-n", "50"]
    cmd = lab.experiment_cmd("e1", lab.seeds_arg(1, 3), ["adaptive", "frozen"], 600, extra=extra, python="py")
    assert cmd[:3] == ["py", "-m", "cyberarena.arena.experiment"]
    assert cmd[cmd.index("--seeds") + 1] == "1-3" and cmd[cmd.index("--conditions") + 1] == "adaptive,frozen"
    assert cmd[cmd.index("--") + 1:] == extra and "--online" not in cmd
    assert lab.seeds_arg(4, 4) == "4"


def test_dashboard_lab_experiment_name_rules(tmp_path):
    (tmp_path / "experiments" / "taken").mkdir(parents=True)
    (tmp_path / "experiments" / "taken" / "manifest.json").write_text("{}")
    assert lab.valid_experiment_name("taken", tmp_path)
    assert lab.valid_experiment_name("a/b", tmp_path) and lab.valid_experiment_name("", tmp_path)
    assert lab.valid_experiment_name("fresh-1", tmp_path) is None


def test_dashboard_lab_launch_experiment_writes_job(monkeypatch, tmp_path):
    calls = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            calls.append((cmd, kw))
            self.pid = 4242

    monkeypatch.setattr(lab.subprocess, "Popen", FakePopen)
    cmd = lab.experiment_cmd("e2", "1-2", ["adaptive"], 200, python=sys.executable)
    job = lab.launch_experiment(tmp_path, "e2", cmd, python=sys.executable)
    assert job["kind"] == "experiment" and job["exp_name"] == "e2" and job["train_cmd"] == cmd
    assert calls and calls[0][0][2] == "cyberarena.dashboard.lab_runner"
    assert lab.read_json(Path(job["job_file"]))["pid"] == 4242


def test_dashboard_lab_runner_runs_experiment(tmp_path):
    from cyberarena.dashboard import lab_runner

    exp = tmp_path / "experiments" / "e3"
    exp.mkdir(parents=True)
    (exp / "manifest.json").write_text(json.dumps({"runs": [{"status": "done"}, {"status": "error"}]}))
    job_file = tmp_path / ".lab" / "t.json"
    job = {"token": "t", "job_file": str(job_file), "kind": "experiment", "exp_name": "e3", "exp_dir": str(exp),
           "stage": "starting", "train_cmd": [sys.executable, "-c", "print('ok')"], "cwd": None}
    lab.save_job(job)
    assert lab_runner.run(job_file) == 0
    done = lab.read_json(job_file)
    assert done["stage"] == "done" and "1 of 2 runs done, 1 failed" in done["message"]


def test_dashboard_lab_experiment_mode_renders(runs):
    app = _open(runs, V4, "lab_page.py", lab_kind="Experiment")
    assert "training runs of" in html(app)
    assert app.button(key="lab_run_exp") is not None
    agent = app.selectbox(key="lab_w:agent")
    assert agent.value == "dqn" and "Q-network (TensorFlow DQN)" in agent.options
    levels = app.selectbox(key="lab_w:eval_evasion_levels")
    assert "none (skip the disguised-attacker test)" in levels.options
    assert any("only used by the tabular learner" in c.value for c in app.caption)


def test_dashboard_v4_curve_frame_shapes():
    agg = json.loads((V4FIX / "experiments" / "main" / "aggregate.json").read_text(encoding="utf-8"))
    cf = E.curve_frame(agg)
    assert set(cf["condition"]) == {"adaptive", "frozen"}
    assert (cf["lo"] >= 0).all() and (cf["hi"] <= 1).all()
    assert isinstance(cf["per_seed"].iloc[0], list) and isinstance(cf, pd.DataFrame)
