"""Simulation Lab logic: spec -> widgets, presets, command line, status/progress readers, history."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from cyberarena.dashboard import lab
from cyberarena.dashboard import loaders as L

FIX = Path(__file__).parent / "fixtures"
SPEC_TEXT = (FIX / "param_spec.json").read_text(encoding="utf-8")
FIXTURE_RUN = FIX / "runs" / "20260101-000000-1"


@pytest.fixture
def spec():
    return lab.parse_spec(SPEC_TEXT)


@pytest.fixture
def idx(spec):
    return lab.param_index(spec)


# ------------------------------------------------------------------------------------------------ spec parsing


def test_parse_spec_tolerates_leading_log_lines():
    spec = lab.parse_spec("[arena] loading...\nwarning {not json\n" + SPEC_TEXT)
    assert [g["id"] for g in spec["groups"]][:2] == ["network", "red"]


@pytest.mark.parametrize("text", ["", "usage: train [-h]", '{"version": 1}', '{"groups": []}', "{bad json"])
def test_parse_spec_rejects_garbage(text):
    with pytest.raises(lab.SpecError):
        lab.parse_spec(text)


def test_load_param_spec_reports_exit_code(monkeypatch):
    def fake_run(cmd, **kw):
        assert cmd[-2:] == ["cyberarena.arena.train", "--describe-params"]
        return subprocess.CompletedProcess(
            cmd, 2, "", "train: error: unrecognized arguments: --describe-params\n"
        )

    monkeypatch.setattr(lab.subprocess, "run", fake_run)
    with pytest.raises(lab.SpecError, match="exit code 2.*unrecognized arguments"):
        lab.load_param_spec("python")


def test_load_param_spec_ok(monkeypatch):
    monkeypatch.setattr(
        lab.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, SPEC_TEXT, "")
    )
    assert "p_phish" in lab.param_index(lab.load_param_spec("python"))


# ------------------------------------------------------------------------------------------------ widgets


def test_widget_mapping(idx):
    w = lab.widget_for(idx["p_phish"])
    assert w["widget"] == "slider" and (w["min"], w["max"], w["step"]) == (0.0, 1.0, 0.01)
    assert w["fallback"] == 0.35 and not w["nullable"] and w["help"]
    w = lab.widget_for(idx["n_nodes"])
    assert w["widget"] == "slider" and w["nullable"] and w["fallback"] == 15
    assert all(isinstance(w[k], int) for k in ("min", "max", "step", "fallback"))
    assert lab.widget_for(idx["no_turn_log"])["widget"] == "toggle"
    w = lab.widget_for(idx["baseline"])
    assert w["widget"] == "selectbox" and w["options"] == ["heuristic", "random"]
    assert lab.widget_for(idx["seed"])["widget"] == "number"  # no min/max -> number input
    assert lab.widget_for(idx["seed"])["advanced"] is True


def test_widget_huge_range_is_number_input():
    p = {"key": "seed", "type": "int", "default": 7, "min": 0, "max": 999999, "step": 1}
    w = lab.widget_for(p)
    assert w["widget"] == "number" and w["fallback"] == 7 and w["max"] == 999999


def test_widget_mapping_unknown_type():
    assert lab.widget_for({"key": "x", "type": "string"})["widget"] == "unsupported"
    assert lab.widget_for({"key": "x", "type": "choice"})["widget"] == "unsupported"


def test_snap_and_changed(spec, idx):
    assert lab.snap(idx["p_phish"], 1.7) == 1.0
    assert lab.snap(idx["detect_threshold"], 0.53) == 0.55
    assert lab.snap(idx["episodes"], 333) == 300
    vals = lab.defaults(spec)
    assert lab.changed_params(spec, vals) == {}
    vals["p_phish"] = 0.35000000001
    assert lab.changed_params(spec, vals) == {}
    vals.update(p_phish=0.5, n_nodes=12)
    assert lab.changed_params(spec, vals) == {"n_nodes": 12, "p_phish": 0.5}
    red = next(g for g in spec["groups"] if g["id"] == "red")
    assert lab.group_changed(red, vals) == ["p_phish"]


# ------------------------------------------------------------------------------------------------ presets


def test_presets(spec):
    d = lab.defaults(spec)
    assert lab.preset_values("Default", spec) == d
    red = lab.preset_values("Red-favoured", spec)
    blue = lab.preset_values("Blue-favoured", spec)
    assert red["p_phish"] == 0.5 and blue["p_phish"] == 0.2
    assert red["p_exploit"] > d["p_exploit"] > blue["p_exploit"]
    assert (
        red["detect_threshold"] > d["detect_threshold"] > blue["detect_threshold"]
    )  # higher = less detection
    assert red["noise.phish"] < d["noise.phish"] < blue["noise.phish"]
    assert red["p_implant_leak.malware"] < d["p_implant_leak.malware"] < blue["p_implant_leak.malware"]
    assert red["episodes"] == d["episodes"] and red["baseline"] == "heuristic"
    assert set(red) == set(d)  # presets never invent keys
    quick = lab.preset_values("Quick test", spec, {**d, "p_phish": 0.6})
    assert (quick["episodes"], quick["eval_every"], quick["p_phish"]) == (300, 100, 0.6)


def test_presets_only_touch_keys_in_spec():
    tiny = lab.parse_spec('{"groups": [{"id": "red", "params": [{"key": "p_phish", "type": "float", '
                          '"default": 0.9, "min": 0, "max": 1, "step": 0.01, "target": "env"}]}]}')  # fmt: skip
    assert lab.preset_values("Quick test", tiny) == {"p_phish": 0.9}
    assert lab.preset_values("Red-favoured", tiny) == {"p_phish": 1.0}  # clamped to max


# ------------------------------------------------------------------------------------------------ command line


def test_build_args_defaults_is_empty(spec):
    assert lab.build_train_args(spec, lab.defaults(spec)) == []


def test_build_args_all_kinds(spec, tmp_path):
    v = lab.defaults(spec)
    v.update({"p_phish": 0.5, "noise.phish": 0.6, "p_implant_leak.malware": 0.1, "max_rounds": 40,
              "n_nodes": 12, "baseline": "random", "episodes": 300, "no_turn_log": True, "init_side": "red"})  # fmt: skip
    parent = tmp_path / "20260101-000000-1"
    args = lab.build_train_args(spec, v, init_from=parent, label="  phish up ", runs_dir=tmp_path)
    env = json.loads(args[args.index("--env-json") + 1])
    assert env == {"max_rounds": 40, "noise.phish": 0.6, "p_implant_leak.malware": 0.1, "p_phish": 0.5}
    assert args[args.index("--n-nodes") + 1] == "12"
    assert args[args.index("--baseline") + 1] == "random"
    assert args[args.index("--episodes") + 1] == "300"
    assert "--no-turn-log" in args and args[args.index("--no-turn-log") + 1].startswith("--")  # bare flag
    assert args[args.index("--init-from") + 1] == str(parent)
    assert args[args.index("--init-side") + 1] == "red"
    assert args[args.index("--label") + 1] == "phish up"
    assert args[args.index("--runs-dir") + 1] == str(tmp_path)
    assert "--online" not in args


def test_build_args_nullable_bool_and_init_side(spec):
    v = lab.defaults(spec)
    v.update(n_nodes=None, no_turn_log=False, init_side="blue")
    args = lab.build_train_args(spec, v)
    assert args == []  # random n_nodes omitted, false bool omitted, init_side needs --init-from
    args = lab.build_train_args(spec, {**v, "init_side": "both"}, init_from="runs/x")
    assert args == ["--init-side", "both", "--init-from", str(Path("runs/x"))]  # side always explicit


def test_enrich_cmd_is_offline():
    cmd = lab.enrich_cmd("runs/x", "py")
    assert cmd == ["py", "-m", "cyberarena.explain.enrich", "--run", "runs/x", "--offline"]
    assert "--online" not in cmd


def test_child_env_drops_api_key():
    env = lab.child_env({"ANTHROPIC_API_KEY": "sk-x", "PATH": "p"})
    assert "ANTHROPIC_API_KEY" not in env and env["PYTHONUNBUFFERED"] == "1" and env["PATH"] == "p"


# ------------------------------------------------------------------------------------------------ readers


def test_read_json_tolerant(tmp_path):
    p = tmp_path / "progress.json"
    assert lab.read_json(p) is None
    p.write_text('{"status": "running", "epis')  # partial write
    assert lab.read_json(p) is None
    p.write_text("[1, 2]")
    assert lab.read_json(p) is None
    lab.write_json_atomic(p, {"a": 1})
    assert lab.read_json(p) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_read_progress(tmp_path):
    assert lab.read_progress(tmp_path) is None
    (tmp_path / "progress.json").write_text(json.dumps({"status": "running", "episode": "740", "episodes": 2000,
                                                        "last_eval": "bogus"}))  # fmt: skip
    p = lab.read_progress(tmp_path)
    assert (p["episode"], p["episodes"], p["last_eval"], p["error"]) == (740, 2000, None, None)
    assert lab.progress_fraction(p) == pytest.approx(0.37)
    assert lab.progress_fraction(None) == 0.0
    assert lab.progress_fraction({"episode": 5, "episodes": 0}) == 0.0


def test_format_elapsed():
    assert lab.format_elapsed(None) == "-"
    assert lab.format_elapsed(65) == "1m 05s"
    assert lab.format_elapsed(3725) == "1h 02m 05s"


# ------------------------------------------------------------------------------------------------ jobs


def _job(runs, token, stage, pid=None, run_dir=None, **kw):
    job = {"token": token, "job_file": str(lab.job_path(runs, token)), "created": lab.now_iso(),
           "started": lab.now_iso(), "stage": stage, "pid": pid, "run_dir": run_dir, **kw}  # fmt: skip
    lab.save_job(job)
    return job


def test_active_job_from_status_files(tmp_path, monkeypatch):
    runs = tmp_path
    assert lab.active_job(runs) is None
    alive = {111}
    monkeypatch.setattr(lab, "pid_alive", lambda pid: pid in alive)
    _job(runs, "a", "done", pid=5)
    run_dir = runs / "20260102-000000-7"
    run_dir.mkdir()
    _job(runs, "b", "training", pid=111, run_dir=str(run_dir))
    assert (run_dir / "lab_status.json").exists()
    assert lab.active_job(runs)["token"] == "b"
    alive.clear()  # runner died -> not active, reported as error
    assert lab.active_job(runs) is None
    assert lab.effective_stage(lab.find_job(runs, "b")) == "error"
    _job(runs, "c", "starting")  # just launched, no pid yet: active during the grace period
    assert lab.active_job(runs)["token"] == "c"


def test_launch_job_spawns_detached_runner(tmp_path, monkeypatch):
    calls = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            calls.append((cmd, kw))
            self.pid = 4242

    monkeypatch.setattr(lab.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(lab, "pid_alive", lambda pid: pid == 4242)
    job = lab.launch_job(tmp_path, ["--episodes", "300"], label="x", python="py", cwd=tmp_path)
    ((cmd, kw),) = calls
    assert cmd == ["py", "-m", "cyberarena.dashboard.lab_runner", "--job", job["job_file"]]
    if lab.os.name == "nt":
        assert kw["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
        assert kw["creationflags"] & subprocess.CREATE_NO_WINDOW  # own hidden console, not Streamlit's
    else:
        assert kw["start_new_session"]
    assert "ANTHROPIC_API_KEY" not in kw["env"]
    saved = lab.read_json(Path(job["job_file"]))
    assert saved["train_cmd"] == ["py", "-m", "cyberarena.arena.train", "--episodes", "300"]
    assert saved["pid"] == 4242 and saved["stage"] == "starting"
    with pytest.raises(lab.LabBusyError):  # one active Lab run at a time
        lab.launch_job(tmp_path, [], python="py")


def test_stop_job_graceful_then_force(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(lab, "kill_tree", killed.append)
    monkeypatch.setattr(lab, "pid_alive", lambda pid: pid == 1)
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    job = _job(tmp_path, "s", "training", pid=1, train_pid=2, run_dir=str(run_dir))
    assert lab.stop_requested_at(job) is None
    lab.stop_job(job)  # runner alive: only a stop request, the runner forwards CTRL_BREAK to train
    assert killed == [] and lab.stop_requested_at(job) is not None
    assert lab.read_lab_status(run_dir)["stage"] == "training"
    out = lab.stop_job(job, force=True)
    assert killed[:2] == [1, 2]
    assert out["stage"] == "error" and out["stopped"]
    assert lab.read_lab_status(run_dir)["stopped"] is True


def test_stop_job_dead_runner_kills_children(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(lab, "kill_tree", killed.append)
    monkeypatch.setattr(lab, "pid_alive", lambda pid: False)
    job = _job(tmp_path, "d", "training", pid=1, train_pid=2)
    assert lab.stop_job(job)["stopped"] and killed[:2] == [1, 2]


def test_progress_cancelled_and_stale(monkeypatch):
    assert lab.progress_cancelled({"status": "cancelled"})
    assert lab.progress_cancelled({"status": "error", "phase": "cancelled"})
    assert lab.progress_cancelled({"status": "error", "error": "Cancelled by user"})
    assert not lab.progress_cancelled({"status": "error", "error": "boom"})
    monkeypatch.setattr(lab, "pid_alive", lambda pid: pid == 5)
    assert lab.progress_stale({"status": "running", "pid": 6})
    assert not lab.progress_stale({"status": "running", "pid": 5})
    assert lab.progress_stale({"status": "running", "updated": "2020-01-01T00:00:00"})
    assert not lab.progress_stale({"status": "running", "updated": lab.now_iso()})
    assert not lab.progress_stale({"status": "done", "updated": "2020-01-01T00:00:00"})
    assert lab.run_status(Path("x"), None, {"status": "running", "pid": 6}, True) == "cancelled"


def test_pid_alive_self_and_garbage():
    assert lab.pid_alive(lab.os.getpid())
    assert not lab.pid_alive(None) and not lab.pid_alive("x") and not lab.pid_alive(-3)


# ------------------------------------------------------------------------------------------------ history


def test_history_table(tmp_path, spec, monkeypatch):
    monkeypatch.setattr(lab, "pid_alive", lambda pid: False)
    old = tmp_path / FIXTURE_RUN.name
    shutil.copytree(FIXTURE_RUN, old)  # pre-Lab run: no config/label/progress/status files
    new = tmp_path / "20260105-120000-7"
    new.mkdir()
    (new / "config.json").write_text(json.dumps({"label": "phish up", "episodes": 300,
                                                 "init_from": {"run_id": old.name, "side": "red"}}))  # fmt: skip
    (new / "progress.json").write_text(json.dumps({"status": "done", "episode": 300, "episodes": 300,
                                                   "started": "2026-01-05T12:00:00"}))  # fmt: skip
    (new / "summary.jsonl").write_text(
        '{"kind": "eval", "after_episode": 100, "matchup": "red_learned_vs_blue_baseline", "n": 10, '
        '"red_win_rate": 0.3, "blue_win_rate": 0.7}\n'
        '{"kind": "eval", "after_episode": 300, "matchup": "red_learned_vs_blue_baseline", "n": 10, '
        '"red_win_rate": 0.6, "blue_win_rate": 0.4}\n'
        '{"kind": "eval", "after_episode": 300, "matchup": "blue_learned_vs_red_baseline", "n": 10, '
        '"red_win_rate": 0.2, "blue_win_rate": 0.8}\n'
    )
    _job(tmp_path, "t", "done", run_dir=str(new), label="phish up", changed={"p_phish": 0.5})
    running = tmp_path / "20260106-000000-7"
    running.mkdir()
    (running / "progress.json").write_text('{"status": "running", "episode": 10, "episodes": 300, "last_eval": '
                                           '{"after_episode": 0, "red_win_rate": 0.1, "blue_win_rate": 0.9}}')  # fmt: skip
    (tmp_path / ".lab" / "junk").mkdir()

    hist = lab.history_table(tmp_path, spec)
    assert hist["run_id"].tolist() == [running.name, new.name, old.name]
    r = hist.set_index("run_id")
    assert r.loc[new.name, "label"] == "phish up"
    assert r.loc[new.name, "parent"] == f"{old.name} (red)"
    assert r.loc[new.name, "changed"] == "p_phish=0.5"
    assert (r.loc[new.name, "red_eval"], r.loc[new.name, "blue_eval"]) == (0.6, 0.8)
    assert r.loc[new.name, "status"] == "done" and r.loc[new.name, "episodes"] == 300
    assert str(r.loc[new.name, "started"]) == "2026-01-05 12:00:00"
    assert r.loc[running.name, "status"] == "running" and r.loc[running.name, "red_eval"] == 0.1
    assert r.loc[old.name, "label"] == "" and r.loc[old.name, "status"] == "done"
    assert str(r.loc[old.name, "started"]) == "2026-01-01 00:00:00"  # from the run id

    # replay picker helpers
    runs = L.list_runs(tmp_path)
    assert lab.run_display_name(new) == f"phish up · {new.name}"
    assert lab.run_display_name(old) == old.name
    assert runs[lab.default_replay_run(runs)] == new  # newest *finished* run (the running one has no summary)


def test_changed_from_config_prefers_recorded_params(spec):
    cfg = {"params": {"episodes": 300, "n_nodes": 12, "seed": 7, "init_side": "red", "baseline": "heuristic"},
           "env": {"p_phish": 0.5}, "init_from": None}  # fmt: skip
    assert lab.changed_from_config(spec, cfg) == {"n_nodes": 12, "p_phish": 0.5, "episodes": 300}


def test_changed_from_config(spec):
    cfg = {"episodes": 2000, "seed": 9, "n_nodes": 16,
           "env": {"p_phish": 0.35, "noise": {"phish": 0.5}, "p_implant_leak": {"malware": 0.2}}}  # fmt: skip
    assert lab.changed_from_config(spec, cfg) == {"noise.phish": 0.5, "seed": 9}
    assert lab.changed_from_config(spec, None) == {}
    assert lab.format_changed({"a": 0.5, "b": None, "c": True}) == "a=0.5, b=random, c=True"


def test_compare_figure_two_panels():
    from cyberarena.dashboard import charts

    _, evals = L.load_summary(FIXTURE_RUN)
    curve = L.eval_curve(evals)
    fig = charts.compare_figure(
        [{"name": "a", "curve": curve, "slot": 0}, {"name": "b", "curve": curve, "slot": 8}]
    )
    names = [t.name for t in fig.data]
    assert names.count("a") == curve["side"].nunique()
    assert fig.data[0].line.color == charts.RUN_COLORS[0]
    assert fig.data[-1].line.color == charts.RUN_COLORS[8 % len(charts.RUN_COLORS)]


@pytest.mark.parametrize(
    ("cmdline", "should_kill"),
    [
        (r"C:\repo\.venv\Scripts\python.exe -m cyberarena.dashboard.lab_runner --job x.json", True),
        (r"C:\repo\.venv\Scripts\python.exe -m cyberarena.arena.train --episodes 300", True),
        (r"C:\Program Files\SomeApp\app.exe --unrelated", False),  # PID recycled by another program
        (None, False),  # command line unreadable
    ],
)
def test_kill_tree_only_kills_our_processes(monkeypatch, cmdline, should_kill):
    calls = []
    monkeypatch.setattr(lab, "pid_alive", lambda pid: True)
    monkeypatch.setattr(lab, "process_cmdline", lambda pid: cmdline)
    monkeypatch.setattr(lab._REAL_SUBPROCESS, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(lab.os, "killpg", lambda *a: calls.append(a), raising=False)
    monkeypatch.setattr(lab.os, "getpgid", lambda pid: pid, raising=False)
    lab.kill_tree(4242)
    assert bool(calls) is should_kill
