"""Diagnostic split of adaptation (--detectors / --red-evasion) and arena/crosseval.py. Stub classifiers only."""

import json

import numpy as np
import pytest

from cyberarena.arena.crosseval import crosseval, input_gradients, shuffle_scores
from cyberarena.arena.dqn import NumpyMLP, glorot_weights
from cyberarena.arena.env import CyberArenaEnv
from cyberarena.arena.params import spec
from cyberarena.arena.stub import stub_detectors
from cyberarena.arena.train import main

TINY = ["--episodes", "24", "--seed", "4", "--eval-every", "12", "--eval-n", "3", "--eval-log-n", "1",
        "--log-every", "6", "--detector-update-every", "8", "--detector-min-samples", "8", "--stats-every", "6",
        "--adaptive-pool-size", "64", "--eval-evasive-n", "3", "--stub", "--quiet"]  # fmt: skip


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def strip(rs):
    return [{k: v for k, v in r.items() if k not in ("run_id", "seconds")} for r in rs]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("diag")
    out = {}
    for name, extra in {"default": [], "explicit": ["--detectors", "adaptive", "--red-evasion", "bandit"],
                        "ev_off": ["--red-evasion", "off"], "det_frozen": ["--detectors", "frozen"]}.items():
        out[name] = main([*TINY, *extra, "--runs-dir", str(base / name)])
    return out


def test_explicit_modes_identical_to_default(runs):
    for f in ("summary.jsonl", "learning.jsonl", "episodes.jsonl"):
        assert strip(rows(runs["default"] / f)) == strip(rows(runs["explicit"] / f)), f


def test_red_evasion_off_keeps_detector_updates(runs):
    s = rows(runs["ev_off"] / "summary.jsonl")
    assert all(set(r["red_evasion"].values()) == {0.0} for r in s if r["kind"] == "episode")
    learn = rows(runs["ev_off"] / "learning.jsonl")
    assert any(r["kind"] == "detector_update" for r in learn)
    assert all(set(r["levels"].values()) == {0.0} for r in learn if r["kind"] == "red_evasion")


def test_frozen_detectors_keep_red_bandit(runs):
    learn = rows(runs["det_frozen"] / "learning.jsonl")
    assert not any(r["kind"] == "detector_update" for r in learn)
    s = rows(runs["det_frozen"] / "summary.jsonl")
    assert any(any(v > 0 for v in r["red_evasion"].values()) for r in s if r["kind"] == "episode")
    assert all(set(r["detector_versions"].values()) == {0} for r in s if "detector_versions" in r)


def test_modes_need_adaptive(tmp_path):
    for extra in (["--detectors", "adaptive"], ["--red-evasion", "bandit"]):
        with pytest.raises(SystemExit) as e:
            main([*TINY, "--no-adaptive", *extra, "--runs-dir", str(tmp_path)])
        assert e.value.code == 2
    # off / frozen with --no-adaptive are no-ops, accepted
    assert main([*TINY, "--no-adaptive", "--red-evasion", "off", "--runs-dir", str(tmp_path)]) is not None


def test_spec_lists_diagnostic_params():
    g = next(g for g in spec()["groups"] if g["id"] == "adaptation")
    keys = {p["key"]: p for p in g["params"]}
    for k, choices in (("detectors", ["auto", "adaptive", "frozen"]), ("red_evasion", ["auto", "bandit", "off"])):
        assert keys[k]["default"] == "auto" and keys[k]["choices"] == choices and keys[k]["advanced"]


def test_input_gradients_match_finite_differences():
    w = glorot_weights([7, 5, 4, 1], seed=3)
    X = np.random.default_rng(0).normal(size=(6, 7))

    def f(X):  # float64 forward (NumpyMLP is float32, too coarse for a tiny step)
        h = X
        for i in range(0, len(w), 2):
            h = h @ w[i].astype(np.float64) + w[i + 1]
            if i < len(w) - 2:
                h = np.maximum(h, 0.0)
        return h[:, 0]

    g = input_gradients(NumpyMLP(w), X)
    eps = 1e-6
    fd = np.array([(f(X + eps * np.eye(7)[j]) - f(X - eps * np.eye(7)[j])) / (2 * eps) for j in range(7)]).T
    assert np.allclose(g, fd, atol=1e-6)


def test_shuffle_scores_keeps_values():
    env = CyberArenaEnv(graph_seed=3, detectors=stub_detectors(), adaptive_pool_size=32)
    before = {m: np.sort(np.concatenate([env.pool.scores[k] for k in env.pool.keys(m)])) for m in ("malware",)}
    shuffle_scores(env, 1)
    after = np.sort(np.concatenate([env.pool.scores[k] for k in env.pool.keys("malware")]))
    assert np.array_equal(before["malware"], after)


def test_crosseval_cells(runs):
    res = crosseval([runs["default"], runs["det_frozen"]], runs["default"], levels=(0.0, 0.7), n=2, stub=True)
    assert {c["detectors"] for c in res["cells"]} == {"v0", "final", "shuffled"}
    assert len(res["cells"]) == 3 * 2 * 2
    final = next(c for c in res["cells"] if c["detectors"] == "final")
    assert max(final["detector_versions"].values()) > 0
    assert set(res["saliency"]) == {runs["default"].name, runs["det_frozen"].name}
    assert 0.0 <= res["saliency"][runs["default"].name]["score_share"] <= 1.0


def test_clean_quantile_view():
    from cyberarena.arena.features import FEATURE_NAMES, candidate_matrix

    env = CyberArenaEnv(graph_seed=3, detectors=stub_detectors(), adaptive_pool_size=64)
    env.reset(seed=5)
    moves = env.all_candidates("blue")
    raw = candidate_matrix(env, "blue", moves)
    assert env.blue_scores() is env.scores
    env.blue_score_view = "clean_quantile"
    q = env.blue_scores()
    assert q.shape == env.scores.shape and q.min() >= 0.0 and q.max() <= 1.0
    for k, m in enumerate(("malware", "phishing", "network")):  # monotone in the raw score
        order = np.argsort(env.scores[:, k], kind="stable")
        assert np.all(np.diff(q[order, k]) >= 0)
        ben = env.pool.scores[(m, 0, 0)]
        assert abs(float(env.pool.clean_quantile(m, np.median(ben))) - 0.5) < 0.05
    quant = candidate_matrix(env, "blue", moves)
    score_cols = [i for i, n in enumerate(FEATURE_NAMES["blue"]) if n.startswith("t_score") or n in
                  ("g_top_undetected", "g_crown_score")]  # fmt: skip
    other = [i for i in range(raw.shape[1]) if i not in score_cols]
    assert np.array_equal(raw[:, other], quant[:, other])
    # a rescore (new detector version) refreshes the clean distribution
    env.pool.scores[("malware", 0, 0)] = env.pool.scores[("malware", 0, 0)] * 0.5
    assert float(env.pool.clean_quantile("malware", np.array([0.49]))[0]) == 1.0


def test_blue_score_view_flag(tmp_path):
    run = main([*TINY, "--blue-score-view", "clean_quantile", "--runs-dir", str(tmp_path)])
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["params"]["blue_score_view"] == "clean_quantile"
    turns = [r for r in rows(run / "episodes.jsonl") if r["actor"] == "blue" and r["agent"] == "learned"]
    assert turns and all(0.0 <= r["chosen_features"]["t_score_max"] <= 1.0 for r in turns)
