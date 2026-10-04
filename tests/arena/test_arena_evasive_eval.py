"""Disguised-attacker eval matchups (docs/contracts.md, "Experiments & rigor (v3)" -> "Evading-red evaluation").

Stub classifiers / stub adaptive detectors only (no TensorFlow).
"""

import json
import re

import pytest

from cyberarena.arena.params import ParamError, parse_evasion_levels, spec
from cyberarena.arena.train import EVASIVE_MATCHUP, main

TINY = ["--episodes", "24", "--seed", "4", "--eval-every", "12", "--eval-n", "3", "--eval-log-n", "1",
        "--log-every", "6", "--detector-update-every", "8", "--detector-min-samples", "8", "--stats-every", "6",
        "--adaptive-pool-size", "64", "--eval-evasive-n", "7", "--stub", "--quiet"]  # fmt: skip
ROW_KEYS = {"kind", "after_episode", "matchup", "evasion", "n", "red_win_rate", "blue_win_rate", "episodes",
            "logged_n", "probe_episode", "detector_versions"}  # fmt: skip


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("evasive")
    return {
        "adaptive": main([*TINY, "--runs-dir", str(base / "a")]),
        "frozen": main([*TINY, "--runs-dir", str(base / "f"), "--no-adaptive"]),
        "off": main([*TINY, "--runs-dir", str(base / "o"), "--eval-evasion-levels", ""]),
    }


def evasive_rows(run):
    return [r for r in read_jsonl(run / "summary.jsonl") if r["kind"] == "eval" and r["matchup"] == EVASIVE_MATCHUP]


@pytest.mark.parametrize("cond", ["adaptive", "frozen"])
def test_evasive_rows_schema(runs, cond):
    run = runs[cond]
    rows = evasive_rows(run)
    assert [(r["after_episode"], r["evasion"]) for r in rows] == [(a, s) for a in (0, 12, 24) for s in (0.4, 0.7)]
    for r in rows:
        assert set(r) == ROW_KEYS
        assert r["n"] == 7 and r["logged_n"] == 0
        assert r["episodes"][1] - r["episodes"][0] + 1 == 7
        assert r["red_win_rate"] + r["blue_win_rate"] == pytest.approx(1.0)
        assert r["probe_episode"] == r["episodes"][1] + 1
    others = [r for r in read_jsonl(run / "summary.jsonl") if r["kind"] == "eval" and r["matchup"] != EVASIVE_MATCHUP]
    assert all("evasion" not in r and r["n"] == 3 for r in others)
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["eval_evasive"]["levels"] == [0.4, 0.7] and cfg["eval_evasive"]["n"] == 7


@pytest.mark.parametrize("cond", ["adaptive", "frozen"])
def test_only_probe_games_logged_at_fixed_evasion(runs, cond):
    run = runs[cond]
    rows = evasive_rows(run)
    recs = [r for r in read_jsonl(run / "episodes.jsonl") if r["matchup"] == EVASIVE_MATCHUP]
    assert {r["episode"] for r in recs} == {r["probe_episode"] for r in rows}
    for r in recs:
        assert r["probe_game"] and r["phase"] == "eval" and r["evasion"] in (0.4, 0.7)
        assert set(r["red_evasion"].values()) == {r["evasion"]}
        assert {c["evasion"] for c in r["classifier_inputs"]} <= {0.0, r["evasion"]}  # benign rows stay at 0


def test_frozen_detectors_stay_v0_adaptive_follow_updates(runs):
    assert all(set(r["detector_versions"].values()) == {0} for r in evasive_rows(runs["frozen"]))
    assert not (runs["frozen"] / "detectors").exists()
    updates = [r for r in read_jsonl(runs["adaptive"] / "learning.jsonl") if r["kind"] == "detector_update"]
    assert updates, "tiny adaptive run should update its detectors"
    for r in evasive_rows(runs["adaptive"]):
        expect = {m: max([u["version"] for u in updates if u["model"] == m and u["episode"] <= r["after_episode"]],
                         default=0) for m in r["detector_versions"]}  # fmt: skip
        assert r["detector_versions"] == expect


def _strip(rows):
    # eval games (any matchup) add never-trained zero rows to the Q-table dicts on lookup, so the Q-table size
    # telemetry differs; the learning itself (every update, TD error, action mix, detectors, bandit) must not
    return [{k: v for k, v in r.items() if k not in ("seconds", "n_states", "mean_abs_q")} for r in rows]


def test_evasive_eval_does_not_change_training(runs):
    a, o = runs["adaptive"], runs["off"]
    assert not evasive_rows(o)
    eps = lambda run: [r for r in read_jsonl(run / "summary.jsonl") if r["kind"] == "episode"]
    assert eps(a) == eps(o)
    base = lambda run: [{k: v for k, v in r.items() if k not in ("episodes", "probe_episode")}
                        for r in read_jsonl(run / "summary.jsonl") if r["kind"] == "eval" and "evasion" not in r]
    assert base(a) == base(o)
    learn = lambda run: _strip([r for r in read_jsonl(run / "learning.jsonl")
                                if r["kind"] in ("detector_update", "red_evasion", "agent_stats")])
    assert learn(a) == learn(o)
    train = lambda run: [{k: v for k, v in r.items() if k != "run_id"}
                         for r in read_jsonl(run / "episodes.jsonl") if r["phase"] == "train"]
    assert train(a) == train(o)


def test_evasive_eval_is_deterministic(runs, tmp_path):
    again = main([*TINY, "--runs-dir", str(tmp_path)])
    assert evasive_rows(again) == evasive_rows(runs["adaptive"])


def test_param_spec_and_validation(tmp_path, capsys):
    by = {p["key"]: (g["id"], p) for g in spec()["groups"] for p in g["params"]}
    gid, p = by["eval_evasion_levels"]
    assert gid == "training" and p["type"] == "choice" and p["default"] == "0.4,0.7"
    assert "" in p["choices"] and p["flag"] == "--eval-evasion-levels" and p["help"]
    gid, p = by["eval_evasive_n"]
    assert gid == "training" and p["type"] == "int" and p["default"] == 100 and p["flag"] == "--eval-evasive-n"
    assert parse_evasion_levels("0.7, 0.4,0.4") == (0.4, 0.7)
    assert parse_evasion_levels("") == parse_evasion_levels("none") == ()
    for bad in ("0.75", "0.8", "x"):
        with pytest.raises(ParamError):
            parse_evasion_levels(bad)
        with pytest.raises(SystemExit) as e:
            main([*TINY, "--runs-dir", str(tmp_path), "--eval-evasion-levels", bad])
        assert e.value.code == 2
        assert re.search(r"eval_evasion_levels", capsys.readouterr().err)
