"""Simulation Lab interface (docs/contracts.md, "Simulation Lab"): param spec, warm start, progress, validation."""

import json
import subprocess
import sys
import time

import pytest

from cyberarena.arena.agents import QAgent
from cyberarena.arena.env import ArenaConfig
from cyberarena.arena.params import PARAMS, env_config_fields, spec
from cyberarena.arena.train import build_parser, main

TINY = ["--episodes", "8", "--seed", "3", "--eval-every", "4", "--eval-n", "3", "--eval-log-n", "1",
        "--log-every", "4", "--stub", "--quiet"]  # fmt: skip
GROUP_IDS = {"network", "red", "blue", "rewards", "training", "opponents", "memory"}


def all_params():
    return [p for g in spec()["groups"] for p in g["params"]]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------------------------------ spec


def test_spec_shape():
    s = spec()
    assert s["version"] == 1
    assert {g["id"] for g in s["groups"]} == GROUP_IDS
    keys = [p["key"] for p in all_params()]
    assert len(keys) == len(set(keys))
    for p in all_params():
        assert p["type"] in ("int", "float", "bool", "choice")
        assert p["target"] in ("env", "cli")
        assert p["label"] and p["help"]
        if p["target"] == "cli":
            assert p["flag"].startswith("--")
        if p["type"] in ("int", "float"):
            assert p["min"] < p["max"] and p["step"] > 0
            if p["default"] is None:
                assert p.get("nullable")
            else:
                assert p["min"] <= p["default"] <= p["max"], p["key"]
        if p["type"] == "choice":
            assert p["default"] in p["choices"]
    assert json.loads(json.dumps(s)) == s


def test_spec_env_params_match_arena_config():
    env_keys = {p["key"] for p in all_params() if p["target"] == "env"}
    assert env_keys == env_config_fields()
    d = ArenaConfig().to_dict()
    for p in all_params():
        if p["target"] == "env":
            field, _, sub = p["key"].partition(".")
            assert p["default"] == (d[field][sub] if sub else d[field])
            assert isinstance(p["default"], int if p["type"] == "int" else float)


def test_spec_cli_params_match_parser():
    args = build_parser().parse_args([])
    for p in all_params():
        if p["target"] == "cli":
            assert getattr(args, p["key"]) == p["default"], p["key"]
    assert args.eval_n == 200 and args.eval_log_n == 10


def test_spec_contents():
    by = {p["key"]: p for p in all_params()}
    assert by["n_nodes"]["nullable"] and (by["n_nodes"]["min"], by["n_nodes"]["max"]) == (10, 20)
    assert (by["episodes"]["min"], by["episodes"]["max"]) == (100, 20000)
    assert by["noise.recon"]["advanced"] and by["r_win"]["advanced"]
    assert by["init_side"]["choices"] == ["both", "red", "blue"]
    assert by["baseline"]["flag"] == "--baseline"
    assert all(p.group in GROUP_IDS for p in PARAMS)


def test_describe_params_cli_is_fast_and_tf_free():
    t = time.time()
    code = ("import sys; from cyberarena.arena.train import main; main(['--describe-params']); "
            "assert 'tensorflow' not in sys.modules")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == spec()
    assert time.time() - t < 10


# ------------------------------------------------------------------------------------------ validation


@pytest.mark.parametrize("extra", [
    ["--env-json", '{"bogus": 1}'],
    ["--env-json", '{"p_phish": 1.5}'],
    ["--env-json", '{"noise.nope": 0.1}'],
    ["--env-json", '{"noise": {"recon": -0.1}}'],
    ["--env-json", '{"max_rounds": 3.5}'],
    ["--env-json", '{"p_phish": "high"}'],
    ["--env-json", "[1, 2]"],
    ["--env-json", "{not json"],
    ["--episodes", "50000"],
    ["--n-nodes", "30"],
    ["--alpha", "2"],
    ["--eps-start", "-0.1"],
    ["--baseline", "smart"],
    ["--init-side", "green"],
    ["--max-rounds", "1000"],
    ["--init-from", "does/not/exist"],
])  # fmt: skip
def test_validation_rejects(extra, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*TINY, "--runs-dir", str(tmp_path), *extra])
    assert exc.value.code == 2
    err = capsys.readouterr().err.strip()
    assert err and "\n" not in err
    assert not any(tmp_path.iterdir()), "no run dir on a validation error"


def test_validation_accepts_dotted_and_nested(tmp_path):
    run = main([*TINY, "--runs-dir", str(tmp_path),
                "--env-json", '{"p_phish": 0.5, "noise.recon": 0.1, "p_implant_leak": {"network": 0.3}}'])
    env = json.loads((run / "config.json").read_text())["env"]
    assert env["p_phish"] == 0.5 and env["noise"]["recon"] == 0.1 and env["noise"]["phish"] == 0.85
    assert env["p_implant_leak"] == {"malware": 0.2, "phishing": 0.05, "network": 0.3}


# ------------------------------------------------------------------------------------------ progress + log


@pytest.fixture(scope="module")
def base_run(tmp_path_factory):
    return main([*TINY, "--runs-dir", str(tmp_path_factory.mktemp("runs")), "--label", "base"])


def test_progress_done(base_run):
    p = json.loads((base_run / "progress.json").read_text())
    assert p["status"] == "done" and p["error"] is None
    assert p["episode"] == p["episodes"] == 8
    assert p["last_eval"]["after_episode"] == 8
    assert set(p["last_eval"]) == {"after_episode", "red_win_rate", "blue_win_rate"}
    assert p["started"] and p["updated"]


def test_config_label_and_eval_log_n(base_run):
    cfg = json.loads((base_run / "config.json").read_text())
    assert cfg["label"] == "base" and cfg["init_from"] is None and cfg["eval"]["log_n"] == 1
    evals = [r for r in read_jsonl(base_run / "summary.jsonl") if r["kind"] == "eval"]
    logged_eval_eps = {r["episode"] for r in read_jsonl(base_run / "episodes.jsonl") if r["phase"] == "eval"}
    assert all(r["n"] == 3 for r in evals)
    assert logged_eval_eps == {r["episodes"][0] for r in evals}  # only the first eval episode per block


def test_run_dir_first_stdout_line_and_progress_early(tmp_path):
    proc = subprocess.Popen([sys.executable, "-m", "cyberarena.arena.train", *TINY, "--episodes", "100",
                             "--runs-dir", str(tmp_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # fmt: skip
    t = time.time()
    line = proc.stdout.readline().strip()
    assert line.startswith("RUN_DIR ")
    from pathlib import Path

    run = Path(line.split(" ", 1)[1])
    assert run.is_absolute() and (run / "progress.json").exists()
    assert time.time() - t < 5
    assert proc.wait(timeout=120) == 0
    assert json.loads((run / "progress.json").read_text())["status"] == "done"


def test_crash_writes_error(tmp_path, monkeypatch):
    from cyberarena.arena import train

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(train, "play_episode", boom)
    with pytest.raises(RuntimeError):
        main([*TINY, "--runs-dir", str(tmp_path)])
    (run,) = tmp_path.iterdir()
    p = json.loads((run / "progress.json").read_text())
    assert p["status"] == "error" and "kaboom" in p["error"]


def test_cancel_writes_cancelled(tmp_path, monkeypatch):
    from cyberarena.arena import train

    def stop(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(train, "play_episode", stop)
    with pytest.raises(KeyboardInterrupt):
        main([*TINY, "--runs-dir", str(tmp_path)])
    (run,) = tmp_path.iterdir()
    p = json.loads((run / "progress.json").read_text())
    assert p["status"] == "error" and p["error"] == "cancelled"


# ------------------------------------------------------------------------------------------ warm start


def test_warm_start_loads_q_tables(base_run, tmp_path, monkeypatch):
    from cyberarena.arena import train

    saved = {s: QAgent.load(base_run / "agents" / f"{s}.json").q for s in ("red", "blue")}
    assert saved["red"] and saved["blue"]
    seen = {}
    real_play = train.play_episode

    def spy(env, agents, *a, **k):  # capture the learners' Q-tables at the very first (eval) episode
        for side, ag in agents.items():
            if isinstance(ag, QAgent) and side not in seen:
                seen[side] = {key: v.copy() for key, v in ag.q.items()}
        return real_play(env, agents, *a, **k)

    monkeypatch.setattr(train, "play_episode", spy)
    run = main([*TINY, "--runs-dir", str(tmp_path), "--init-from", str(base_run), "--init-side", "red",
                "--eps-start", "0.3"])  # fmt: skip
    assert set(seen["red"]) == set(saved["red"])
    for key, v in saved["red"].items():
        assert seen["red"][key] == pytest.approx(v)
    assert seen["blue"] == {}  # blue not warm-started -> fresh table
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["init_from"] == {"run_id": base_run.name, "side": "red"}


def test_warm_start_both_across_graph_sizes(base_run, tmp_path):
    run = main([*TINY, "--runs-dir", str(tmp_path), "--init-from", str(base_run), "--n-nodes", "19"])
    for s in ("red", "blue"):
        before = QAgent.load(base_run / "agents" / f"{s}.json").q
        after = QAgent.load(run / "agents" / f"{s}.json").q
        assert set(before) <= set(after)
    assert json.loads((run / "config.json").read_text())["init_from"]["side"] == "both"
