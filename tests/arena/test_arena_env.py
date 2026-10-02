"""Env determinism, graph shape and action validity, using stub classifiers (no TensorFlow)."""

import numpy as np
import pytest

from cyberarena.arena.actions import BLUE_ACTION_IDS, RED_ACTION_IDS, BlueAction, RedAction, action_from_id
from cyberarena.arena.agents import HeuristicAgent, QAgent, RandomAgent, blue_features, red_features
from cyberarena.arena.env import MODELS, ArenaConfig, CyberArenaEnv, generate_graph
from cyberarena.arena.stub import stub_classifiers
from cyberarena.arena.train import play_episode


@pytest.fixture(scope="module")
def clfs():
    return stub_classifiers()


def make_env(clfs, seed=7, **kw):
    return CyberArenaEnv(ArenaConfig(pool_size=32, **kw), graph_seed=seed, classifiers=clfs)


def rollout(env, seed, agent_seed=0):
    agents = {"red": RandomAgent("red", agent_seed), "blue": RandomAgent("blue", agent_seed + 1)}
    obs, _ = env.reset(seed=seed)
    trace = [obs.copy()]
    while not env.done:
        d = agents[env.actor].act(env)
        obs, _, _, _, info = env.step((d.action, d.source, d.target))
        trace.append(obs.copy())
        trace.append((info["action_id"], info["source"], info["target"], info["success"],
                      tuple((c["node"], c["model"], c["pool_index"]) for c in info["classifier_inputs"])))
    return trace, env.winner


def test_action_ids_stable():
    assert RED_ACTION_IDS == ("recon", "phish", "exploit", "escalate", "lateral_move", "exfiltrate", "wait")
    assert BLUE_ACTION_IDS == ("monitor", "isolate", "patch", "restore", "reset_credentials", "wait")
    assert action_from_id("red", "lateral_move") is RedAction.LATERAL_MOVE
    assert action_from_id("blue", "reset_credentials") is BlueAction.RESET_CREDENTIALS


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 1234])
def test_graph_shape(seed):
    g = generate_graph(seed)
    assert 10 <= g.n <= 20
    assert g.roles[g.crown] == "server"
    assert {"dmz", "workstation", "server"} <= set(g.roles)
    # connected
    nbrs = g.neighbors()
    seen, stack = {0}, [0]
    while stack:
        for v in nbrs[stack.pop()]:
            if int(v) not in seen:
                seen.add(int(v))
                stack.append(int(v))
    assert len(seen) == g.n
    assert generate_graph(seed).edges == g.edges  # seeded
    assert np.all((g.pos >= 0) & (g.pos <= 1))


def test_reset_step_deterministic(clfs):
    a, wa = rollout(make_env(clfs), seed=3)
    b, wb = rollout(make_env(clfs), seed=3)
    assert wa == wb and len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y)
        else:
            assert x == y
    c, _ = rollout(make_env(clfs), seed=4)
    assert len(c) != len(a) or any(
        not np.array_equal(x, y) if isinstance(x, np.ndarray) else x != y for x, y in zip(a, c, strict=False))


def test_reset_initial_state(clfs):
    env = make_env(clfs)
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.n, 8) and env.observation_space.contains(obs)
    assert env.actor == "red"
    fh = env.footholds()
    assert list(fh) == [info["start_node"]] and env.roles[fh[0]] == "workstation"
    assert not env.detected.any() and not env.isolated.any()


def test_actions_validity_and_turn_order(clfs):
    env = make_env(clfs)
    rng = np.random.default_rng(0)
    for ep in range(30):
        env.reset(seed=ep)
        expected = "red"
        while not env.done:
            side = env.actor
            assert side == expected
            valid = env.valid_actions(side)
            assert valid, "WAIT is always valid"
            for a in valid:
                assert env.candidates(side, a)
            a = int(rng.choice(valid))
            cands = env.candidates(side, a)
            src, tgt = cands[int(rng.integers(len(cands)))]
            _, r, _, _, info = env.step((a, src, tgt))
            assert info["rewards"][side] == r
            for c in info["classifier_inputs"]:
                assert c["model"] in MODELS and 0 <= c["node"] < env.n
                assert len(env.classifier_row(c)) == clfs[c["model"]].n_features
            expected = "blue" if expected == "red" else "red"
        assert env.winner in ("red", "blue")
        assert env.turn <= 2 * env.cfg.max_rounds
    with pytest.raises(RuntimeError):
        env.step(int(RedAction.WAIT))


def test_invalid_action_rejected(clfs):
    env = make_env(clfs)
    env.reset(seed=0)
    assert not env.candidates("red", int(RedAction.EXFILTRATE))  # no crown foothold yet
    with pytest.raises(ValueError):
        env.step((int(RedAction.EXFILTRATE), env.crown, env.crown))
    env.step(int(RedAction.WAIT))
    assert not env.candidates("blue", int(BlueAction.ISOLATE))  # nothing detected yet
    with pytest.raises(ValueError):
        env.step((int(BlueAction.ISOLATE), None, 0))


def test_malicious_actions_emit_positive_rows(clfs):
    env = make_env(clfs, noise={k: 1.0 for k in ("recon", "phish", "exploit", "escalate",
                                                  "lateral_move", "exfiltrate")})  # fmt: skip
    env.reset(seed=0)
    start = int(env.footholds()[0])
    _, _, _, _, info = env.step((int(RedAction.ESCALATE), start, start))
    (c,) = info["classifier_inputs"]
    assert c["node"] == start and c["model"] == "malware" and c["label"] == 1
    assert env.scores[start, MODELS.index("malware")] == pytest.approx(c["score"], abs=1e-4)


def test_red_wins_by_exfiltration_and_blue_by_eviction(clfs):
    env = make_env(clfs, p_exfiltrate=1.0)
    env.reset(seed=0)
    c = env.crown
    env.compromised[c] = True
    env.privilege[c] = 2
    _, r, term, _, info = env.step((int(RedAction.EXFILTRATE), c, c))
    assert term and info["winner"] == "red" and r > 0.9

    env.reset(seed=0)
    start = int(env.footholds()[0])
    env.step(int(RedAction.WAIT))
    env.detected[start] = True
    _, r, term, _, info = env.step((int(BlueAction.ISOLATE), None, start))
    assert term and info["winner"] == "blue" and r > 0.9


def test_agents_choose_valid_actions(clfs, tmp_path):
    env = make_env(clfs)
    agents = {"red": QAgent("red", 0), "blue": QAgent("blue", 1)}
    for ep in range(5):
        res = play_episode(env, agents, ep, learn={"red", "blue"}, explore=True)
        assert res["winner"] in ("red", "blue")
    assert agents["red"].q and agents["blue"].q
    env.reset(seed=0)
    d = agents["red"].act(env, explore=False)
    assert d.action in env.valid_actions("red") and set(d.decision_values) <= set(RED_ACTION_IDS)
    assert not d.explored and d.epsilon == 0.0
    agents["red"].save(tmp_path / "red.json")
    again = QAgent.load(tmp_path / "red.json")
    assert again.q.keys() == agents["red"].q.keys()
    assert len(red_features(env)) == 7 and len(blue_features(env)) == 5
    for side in ("red", "blue"):
        h = HeuristicAgent(side, 0)
        env.reset(seed=1)
        if side == "blue":
            env.step(int(RedAction.WAIT))
        assert h.act(env).action in env.valid_actions(side)
