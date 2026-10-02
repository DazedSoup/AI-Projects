"""End-to-end tiny training run with stub classifiers: run-dir layout and episode-log schema."""

import json

import pytest

from cyberarena.arena.actions import ACTION_IDS
from cyberarena.arena.agents import QAgent
from cyberarena.arena.stub import STUB_FEATURES
from cyberarena.arena.train import main

TURN_KEYS = {"run_id", "episode", "turn", "actor", "action_id", "source", "target", "success", "reward",
             "decision_values", "epsilon", "explored", "node_states", "classifier_inputs", "done", "winner",
             "shap", "rationale", "mitre"}  # fmt: skip
NODE_KEYS = {"id", "compromised", "detected", "isolated", "patched", "privilege", "scores"}
ARGS = ["--episodes", "12", "--seed", "5", "--eval-every", "6", "--eval-n", "3", "--log-every", "4",
        "--stub", "--quiet"]  # fmt: skip


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return main([*ARGS, "--runs-dir", str(tmp_path_factory.mktemp("runs"))])


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_dir_layout(run_dir):
    for f in ("config.json", "graph.json", "episodes.jsonl", "summary.jsonl", "agents/red.json",
              "agents/blue.json"):  # fmt: skip
        assert (run_dir / f).exists(), f
    run_id = run_dir.name
    date, clock, seed = run_id.split("-")
    assert len(date) == 8 and len(clock) == 6 and seed == "5"
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["seed"] == 5 and cfg["run_id"] == run_id and "env" in cfg and "agents" in cfg
    assert QAgent.load(run_dir / "agents" / "red.json").side == "red"


def test_graph_json(run_dir):
    g = json.loads((run_dir / "graph.json").read_text())
    assert 10 <= len(g["nodes"]) <= 20
    assert sum(n["crown_jewel"] for n in g["nodes"]) == 1
    for n in g["nodes"]:
        assert set(n) == {"id", "role", "crown_jewel", "x", "y"}
    ids = {n["id"] for n in g["nodes"]}
    assert all(len(e) == 2 and set(e) <= ids for e in g["edges"])


def test_turn_records(run_dir):
    recs = read_jsonl(run_dir / "episodes.jsonl")
    n_nodes = len(json.loads((run_dir / "graph.json").read_text())["nodes"])
    assert recs
    for r in recs:
        assert TURN_KEYS <= set(r)
        assert r["run_id"] == run_dir.name
        assert r["shap"] is None and r["rationale"] is None and r["mitre"] is None
        assert r["action_id"] in ACTION_IDS[r["actor"]]
        assert set(r["decision_values"]) <= set(ACTION_IDS[r["actor"]])
        assert isinstance(r["success"], bool) and isinstance(r["explored"], bool)
        assert len(r["node_states"]) == n_nodes
        for ns in r["node_states"]:
            assert set(ns) == NODE_KEYS and set(ns["scores"]) == {"malware", "phishing", "network"}
        for c in r["classifier_inputs"]:
            assert set(c) == {"node", "model", "row", "score"}
            assert len(c["row"]) == STUB_FEATURES[c["model"]] and 0.0 <= c["score"] <= 1.0
            assert r["node_states"][c["node"]]["scores"][c["model"]] == pytest.approx(c["score"], abs=1e-4)
        if r["done"]:
            assert r["winner"] in ("red", "blue")
        else:
            assert r["winner"] is None
    # turns within an episode are consecutive and alternate red/blue
    by_ep = {}
    for r in recs:
        by_ep.setdefault(r["episode"], []).append(r)
    for ep_recs in by_ep.values():
        assert [r["turn"] for r in ep_recs] == list(range(len(ep_recs)))
        assert all(r["actor"] == ("red" if r["turn"] % 2 == 0 else "blue") for r in ep_recs)
        assert ep_recs[-1]["done"]
    train_eps = {r["episode"] for r in recs if r["phase"] == "train"}
    assert train_eps == {0, 4, 8}


def test_summary_rows(run_dir):
    rows = read_jsonl(run_dir / "summary.jsonl")
    eps = [r for r in rows if r["kind"] == "episode"]
    evals = [r for r in rows if r["kind"] == "eval"]
    assert [r["episode"] for r in eps] == list(range(12))
    for r in eps:
        assert {"winner", "turns", "red_return", "blue_return", "epsilon"} <= set(r)
    assert sorted({r["after_episode"] for r in evals}) == [0, 6, 12]
    for r in evals:
        assert r["matchup"] in ("red_learned_vs_blue_baseline", "blue_learned_vs_red_baseline")
        assert r["n"] == 3 and r["red_win_rate"] + r["blue_win_rate"] == pytest.approx(1.0)


def test_run_is_reproducible(run_dir, tmp_path):
    other = main([*ARGS, "--runs-dir", str(tmp_path)])
    assert read_jsonl(other / "summary.jsonl") == read_jsonl(run_dir / "summary.jsonl")
    strip = lambda rs: [{k: v for k, v in r.items() if k != "run_id"} for r in rs]
    assert strip(read_jsonl(other / "episodes.jsonl")) == strip(read_jsonl(run_dir / "episodes.jsonl"))
