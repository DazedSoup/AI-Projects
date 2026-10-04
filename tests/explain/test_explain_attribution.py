"""Integrated gradients on the v4 agents' Q-networks: completeness, checkpoint choice, baseline, nulls. No network."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from explain_fixtures import GRAPH, make_turn, node_states

from cyberarena.arena.features import FEATURE_NAMES
from cyberarena.explain import attribution as A

RED = FEATURE_NAMES["red"]


def mlp(d: int, hidden=(16, 16), act="relu", seed=0) -> A.QNet:
    rng = np.random.default_rng(seed)
    sizes = (d, *hidden, 1)
    layers = [(rng.normal(0, 1.0 / np.sqrt(a), (a, b)), rng.normal(0, 0.3, b), act if i < len(hidden) else "linear")
              for i, (a, b) in enumerate(pairwise(sizes))]  # fmt: skip
    return A.QNet(None, layers)


# --------------------------------------------------------------------------------------------- completeness


@pytest.mark.parametrize("seed", range(5))
def test_exact_ig_is_complete_and_matches_fine_riemann(seed):
    net = mlp(12, seed=seed)
    rng = np.random.default_rng(100 + seed)
    x, b = rng.uniform(0, 1, 12), rng.uniform(0, 1, 12)
    res = A.integrated_gradients(net, x, b)
    assert res["integration"] == "exact_piecewise_linear" and res["segments"] >= 1
    dq = res["q"] - res["q_baseline"]
    assert abs(res["attr"].sum() - dq) < 1e-9
    fine = A.integrated_gradients(net, x, b, schedule=(20000,), exact=False)
    np.testing.assert_allclose(res["attr"], fine["attr"], atol=1e-4)


def test_riemann_fallback_meets_one_percent_and_records_steps():
    net = mlp(10, act="tanh", seed=3)
    rng = np.random.default_rng(7)
    x, b = rng.uniform(0, 1, 10), rng.uniform(0, 1, 10)
    res = A.integrated_gradients(net, x, b)
    assert res["integration"] == "riemann_midpoint" and res["steps"] in A.STEP_SCHEDULE
    assert res["rel"] <= A.COMPLETENESS_TOL or res["abs"] <= A.COMPLETENESS_ABS
    assert res["abs"] == pytest.approx(abs(res["attr"].sum() - (res["q"] - res["q_baseline"])))


def test_numpy_gradient_matches_finite_differences():
    net = mlp(6, act="tanh", seed=1)
    x = np.random.default_rng(2).uniform(0, 1, 6)
    g = net.grad(x)[0]
    eps = 1e-6
    fd = [(net.q(x + eps * e)[0] - net.q(x - eps * e)[0]) / (2 * eps) for e in np.eye(6)]
    np.testing.assert_allclose(g, fd, atol=1e-6)


def test_keras_model_mirror_matches_keras(tmp_path):
    keras = pytest.importorskip("keras")
    model = keras.Sequential([keras.Input((5,)), keras.layers.Dense(8, activation="relu"),
                              keras.layers.Dense(1)])  # fmt: skip
    path = tmp_path / "red_qnet.keras"
    model.save(path)
    net = A.QNet.load(path)
    assert net.layers is not None
    x = np.random.default_rng(0).uniform(0, 1, (4, 5))
    np.testing.assert_allclose(net.q(x), np.asarray(model(x.astype("float32"))).ravel(), atol=1e-5)


# ------------------------------------------------------------------------------------------- checkpoints


def _agents(tmp_path: Path, afters=(0, 250, 500), final=True) -> Path:
    ck = tmp_path / "agents" / "checkpoints"
    ck.mkdir(parents=True)
    for a in afters:
        for side in ("red", "blue"):
            (ck / f"{side}_qnet_{a:05d}.keras").write_bytes(b"")
            (ck / f"{side}_qnet_{a:05d}.json").write_text("{}")
    (ck / "red_evasion_00250.json").write_text("{}")
    if final:
        (tmp_path / "agents" / "red_qnet.keras").write_bytes(b"")
    return tmp_path / "agents"


def test_checkpoint_closest_not_after(tmp_path):
    agents = _agents(tmp_path)
    assert A.select_checkpoint(agents, "red", 250, 500)[1] == 250  # eval turn at its own checkpoint
    assert A.select_checkpoint(agents, "red", 499, 500)[1] == 250  # training game 499: never a later one
    assert A.select_checkpoint(agents, "blue", 0, 500)[1] == 0
    path, after = A.select_checkpoint(agents, "red", 500, 500)
    assert after == 500 and path.name == "red_qnet_00500.keras"


def test_final_model_used_when_no_checkpoint_at_end_of_training(tmp_path):
    agents = _agents(tmp_path, afters=(0, 250))
    path, after = A.select_checkpoint(agents, "red", 600, 600)
    assert path.name == "red_qnet.keras" and after is None
    assert A.select_checkpoint(agents, "blue", 600, 600)[1] == 250  # no final blue file: last checkpoint


def test_training_point_eval_vs_train():
    assert A.training_point({"phase": "eval", "episode": 2600, "after_episode": 2000}) == 2000
    assert A.training_point({"phase": "train", "episode": 412}) == 412


# --------------------------------------------------------------------------------------------- service


def red_features(**kw) -> dict:
    f = dict.fromkeys(RED, 0.0)
    f.update(kw)
    return f


def dqn_turn(prev_states=None, **kw) -> dict:
    """Red lateral_move 1 -> 2 on the 4-node fixture graph (crown 3; max distance 3, max degree 2)."""
    cands = [{"action": "lateral_move", "source": 1, "target": 2, "q": 0.62},
             {"action": "lateral_move", "source": 1, "target": 0, "q": 0.48},
             {"action": "wait", "source": None, "target": None, "q": 0.1}]  # fmt: skip
    feats = red_features(g_footholds=0.2, a_lateral_move=1.0, t_server=1.0, t_dist=1 / 3, t_degree=1.0,
                         t_score_malware=0.05, t_score_phishing=0.1, t_score_network=0.21, t_score_max=0.21,
                         t_closer=1.0, s_has=1.0, s_privilege=0.5, s_score_max=0.1, s_dist=2 / 3, p_success=0.7)  # fmt: skip
    rec = make_turn(episode=2600, turn=4, candidates=cands, chosen_features=feats, phase="eval", after_episode=500)
    rec.update(kw)
    return rec


@pytest.fixture
def v4_run(tmp_path) -> tuple[Path, A.AttributionService, list]:
    run = tmp_path / "run"
    run.mkdir()
    _agents(run, afters=(0, 250, 500))
    (run / "agents" / "red_qnet.json").write_text(json.dumps({"feature_names": list(RED)}))
    loaded: list[Path] = []

    def loader(path):
        loaded.append(path)
        return mlp(len(RED), seed=4)

    cfg = {"episodes": 500, "agents": {"type": "dqn", "feature_names": {k: list(v) for k, v in FEATURE_NAMES.items()}}}
    svc = A.AttributionService(run, GRAPH, cfg, run / "explain_cache" / "agent_ig.jsonl", loader=loader)
    return run, svc, loaded


def test_attribution_record_shape_and_completeness(v4_run):
    _, svc, _ = v4_run
    prev = make_turn(episode=2600, turn=3, actor="blue", node_states=node_states())
    a = svc.explain_turn(dqn_turn(), prev)
    assert a["method"] == "integrated_gradients" and a["baseline"] == "candidate_mean"
    assert a["checkpoint"] == "agents/checkpoints/red_qnet_00500.keras" and a["checkpoint_after"] == 500
    assert a["completeness_error"] < 1e-6
    assert a["attribution_sum"] == pytest.approx(a["q"] - a["q_baseline"], abs=1e-4)
    assert {"name", "value", "baseline", "attribution"} <= set(a["top_features"][0])
    assert a["runner_up"]["target"] == 0 and a["margin"] == pytest.approx(0.14)
    info = a["baseline_info"]
    assert info["pre_state"] == "previous_turn" and info["n_candidates"] == 3 and info["check_error"] == 0.0
    assert "p_success" in info["imputed"]


def test_baseline_is_the_mean_of_the_reconstructed_candidates(v4_run):
    _, svc, _ = v4_run
    names = svc.feature_names("red")
    turn = dqn_turn()
    prev = make_turn(episode=2600, turn=3, node_states=node_states())
    b, kind, _ = svc._baseline(turn, prev, names, turn["chosen_features"])
    col = dict(zip(names, b, strict=True))
    assert kind == "candidate_mean"
    assert col["a_lateral_move"] == pytest.approx(2 / 3) and col["a_wait"] == pytest.approx(1 / 3)
    assert col["t_none"] == pytest.approx(1 / 3)  # the wait has no target
    assert col["t_dmz"] == pytest.approx(1 / 3)  # host 0 is the DMZ host
    assert col["t_dist"] == pytest.approx((1 / 3 + 1) / 3)  # host 2: 1 hop, host 0: 3 hops, wait: 0
    assert col["g_footholds"] == pytest.approx(0.2)  # shared by every candidate


def test_bad_reconstruction_falls_back_to_zeros(v4_run):
    _, svc, _ = v4_run
    turn = dqn_turn()
    turn["chosen_features"]["t_degree"] = 0.1  # inconsistent with graph.json
    a = svc.explain_turn(turn, make_turn(episode=2600, turn=3))
    assert a["baseline"] == "zeros" and "fallback_reason" in a["baseline_info"]


def test_cache_and_models_loaded_once(v4_run):
    run, svc, loaded = v4_run
    prev = make_turn(episode=2600, turn=3)
    first = svc.explain_turn(dqn_turn(), prev)
    again = svc.explain_turn(dqn_turn(turn=6), prev)
    assert again["q"] == first["q"] and svc.stats.cache_hits == 1 and len(loaded) == 1
    other = svc.explain_turn(dqn_turn(after_episode=250), prev)
    assert other["checkpoint_after"] == 250 and len(loaded) == 2
    # a fresh service reads the disk cache: nothing recomputed, nothing loaded
    fresh = A.AttributionService(run, GRAPH, svc.cfg, run / "explain_cache" / "agent_ig.jsonl",
                                 loader=lambda p: pytest.fail("model should not load"))  # fmt: skip
    assert fresh.explain_turn(dqn_turn(), prev)["q"] == first["q"]


@pytest.mark.parametrize("change", [
    {"agent": "heuristic"},  # scripted baseline
    {"candidates": None, "chosen_features": None},  # tabular agent: no candidates logged
    {"chosen_features": None},  # DQN turn without its input vector
])  # fmt: skip
def test_null_for_tabular_scripted_and_missing_input(v4_run, change):
    _, svc, loaded = v4_run
    assert svc.explain_turn(dqn_turn(**change)) is None
    assert not loaded


def test_null_when_no_checkpoint_exists(tmp_path):
    run = tmp_path / "run"
    (run / "agents").mkdir(parents=True)
    cfg = {"agents": {"type": "dqn", "feature_names": {"red": list(RED)}}}
    svc = A.AttributionService(run, GRAPH, cfg, None, loader=lambda p: mlp(len(RED)))
    assert svc.explain_turn(dqn_turn()) is None and svc.stats.nulls == {"no checkpoint": 1}
