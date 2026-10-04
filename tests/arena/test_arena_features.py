"""v4 move features (arena/features.py): shapes, names and the information boundary between the sides."""

import numpy as np
import pytest

from cyberarena.arena.actions import BlueAction, RedAction
from cyberarena.arena.agents import RandomAgent
from cyberarena.arena.env import CyberArenaEnv
from cyberarena.arena.features import (
    FEATURE_NAMES,
    HIDDEN_FROM_BLUE,
    N_FEATURES,
    candidate_features,
    candidate_matrix,
)
from cyberarena.arena.stub import stub_classifiers, stub_detectors


def played_env(seed: int, turns: int, adaptive: bool = True) -> CyberArenaEnv:
    """An env some random turns into a game, so flags, scores and footholds are mixed."""
    kw = {"detectors": stub_detectors(), "adaptive_pool_size": 64} if adaptive else {"classifiers": stub_classifiers()}
    env = CyberArenaEnv(graph_seed=seed, **kw)
    env.reset(seed=seed)
    agents = {"red": RandomAgent("red", seed), "blue": RandomAgent("blue", seed + 1)}
    for _ in range(turns):
        if env.done:
            break
        d = agents[env.actor].act(env)
        env.step((d.action, d.source, d.target))
    if env.done:
        env.done = False
        env.winner = None
    return env


def blue_view(env):
    moves = env.all_candidates("blue")
    return moves, candidate_matrix(env, "blue", moves)


def test_names_and_shapes():
    for side in ("red", "blue"):
        names = FEATURE_NAMES[side]
        assert len(names) == len(set(names)) == N_FEATURES[side]
    env = played_env(3, 9)
    for side in ("red", "blue"):
        moves = env.all_candidates(side)
        X = candidate_matrix(env, side, moves)
        assert X.shape == (len(moves), N_FEATURES[side]) and X.dtype == np.float32
        assert np.isfinite(X).all() and X.min() >= 0.0 and X.max() <= 1.0 + 1e-6
        np.testing.assert_array_equal(candidate_features(env, side, moves[-1]), X[-1])
        # action one-hot matches the move
        a_cols = [i for i, n in enumerate(FEATURE_NAMES[side]) if n.startswith("a_")]
        assert (X[:, a_cols].argmax(axis=1) == [m[0] for m in moves]).all()


def test_all_candidates_match_valid_actions():
    env = played_env(5, 7)
    for side in ("red", "blue"):
        moves = env.all_candidates(side)
        assert sorted({m[0] for m in moves}) == env.valid_actions(side)
        for a, s, t in moves:
            assert (s, t) in env.candidates(side, a)


def _mutate_hidden(env, rng):
    """Change everything blue must not see; keep everything blue may see."""
    env.compromised[:] = rng.random(env.n) < 0.5
    env.privilege[:] = rng.integers(0, 3, env.n)
    env.known[:] = rng.random(env.n) < 0.5
    env.recon_done[:] = rng.random(env.n) < 0.5
    env.phished[:] = rng.random(env.n) < 0.5
    env.evasion_idx[:] = rng.integers(0, 8, len(env.evasion_idx))
    for st in env.red_stats.values():
        for k in st:
            st[k] = int(rng.integers(0, 50))
    for tr in env.trail:
        tr.clear()
    for box in env.mailbox:
        box.append(("phishing", 1, 3, 0))


@pytest.mark.parametrize("seed,turns", [(1, 0), (2, 5), (3, 12), (4, 21), (6, 30)])
def test_blue_features_ignore_hidden_state(seed, turns):
    """Blue's features (and its move list) must not move when only red's hidden state changes."""
    env = played_env(seed, turns)
    assert set(HIDDEN_FROM_BLUE) <= set(vars(env))
    moves0, X0 = blue_view(env)
    rng = np.random.default_rng(seed)
    for _ in range(5):
        _mutate_hidden(env, rng)
        moves1, X1 = blue_view(env)
        assert moves1 == moves0
        np.testing.assert_array_equal(X1, X0)


def test_blue_features_do_see_visible_state():
    """Sanity check of the boundary test: what blue may see does change its features."""
    env = played_env(2, 6)
    _, X0 = blue_view(env)
    live = np.flatnonzero(~env.isolated & ~env.detected)
    env.detected[live[0]] = True
    _, X1 = blue_view(env)
    assert X1.shape != X0.shape or not np.array_equal(X1, X0)
    env.detected[live[0]] = False
    env.scores[live[1]] = 0.99
    _, X2 = blue_view(env)
    assert not np.array_equal(X2, X0)


def test_red_features_ignore_blue_flags():
    """Red does not see blue's detected / confirmed flags."""
    env = played_env(4, 10)
    moves = env.all_candidates("red")
    X0 = candidate_matrix(env, "red", moves)
    env.detected[:] = ~env.detected
    env.confirmed[:] = ~env.confirmed
    np.testing.assert_array_equal(candidate_matrix(env, "red", moves), X0)


def test_red_success_feature_follows_rules():
    env = played_env(1, 0, adaptive=False)
    names = FEATURE_NAMES["red"]
    ip = names.index("p_success")
    s = int(env.footholds()[0])
    x_esc = candidate_features(env, "red", (int(RedAction.ESCALATE), s, s))
    assert x_esc[ip] == pytest.approx(env.cfg.p_escalate)
    env.patched[s] = True
    assert candidate_features(env, "red", (int(RedAction.ESCALATE), s, s))[ip] == pytest.approx(
        env.cfg.p_escalate_patched)
    assert candidate_features(env, "red", (int(RedAction.WAIT), None, None))[names.index("t_none")] == 1.0
    assert candidate_features(env, "blue", (int(BlueAction.WAIT), None, None))[
        FEATURE_NAMES["blue"].index("t_none")] == 1.0
