"""Per-move feature vectors for the v4 DQN agents (docs/contracts.md, "TensorFlow agents (v4)").

A *move* is one concrete ``(action, source, target)`` from ``env.all_candidates(side)``. Its feature vector is
global features (the situation) + target-host features + source-host features (red) + derived terms + a one-hot
of the action. ``FEATURE_NAMES[side]`` names every column; ``candidate_matrix`` builds all of a state's moves in
one vectorised pass (what the agents use), ``candidate_features`` a single move.

**Information boundary.** Each side only sees what it would know:

- Red (the intruder) knows its own footholds and privileges, what it has scanned (``known`` / ``recon_done``),
  its own evasion levels, patch levels it ran into, and the sensor readings its own activity produces.
  Red never sees blue's flags (``detected`` / ``confirmed``).
- Blue (the defender) sees only ``detected``, ``isolated``, ``patched``, ``confirmed`` (proven by its own
  actions), the three detector scores, the topology and the clock. Blue never sees ground-truth compromise,
  privilege, red's knowledge or red's evasion level; ``tests/arena/test_arena_features.py`` mutates all of
  those and asserts blue's features do not move.

All values are scaled to roughly [0, 1]. Importing this module never imports TensorFlow.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cyberarena.arena.actions import ACTION_IDS, BlueAction, RedAction
from cyberarena.arena.env import MODELS, CyberArenaEnv

Move = tuple[int, "int | None", "int | None"]

_ROLE = ("t_dmz", "t_workstation", "t_server", "t_crown")
_SCORES = tuple(f"t_score_{m}" for m in MODELS) + ("t_score_max",)

RED_GLOBAL = ("g_turn_frac", "g_footholds", "g_admin_footholds", "g_has_server", "g_has_crown", "g_crown_admin",
              "g_front_dist", "g_hot", "g_evasion_malware", "g_evasion_phishing", "g_evasion_network")  # fmt: skip
RED_TARGET = ("t_none", *_ROLE, "t_compromised", "t_privilege", "t_isolated", "t_patched", "t_recon", *_SCORES,
              "t_dist", "t_degree", "t_closer", "t_open_nbrs")  # fmt: skip
RED_SOURCE = ("s_has", "s_privilege", "s_recon", "s_score_max", "s_dist")
RED_DERIVED = ("p_success",)

BLUE_GLOBAL = ("g_turn_frac", "g_detected", "g_isolated", "g_confirmed", "g_patched_frac", "g_top_undetected",
               "g_crown_detected", "g_crown_isolated", "g_crown_score")  # fmt: skip
BLUE_TARGET = ("t_none", *_ROLE, "t_detected", "t_confirmed", "t_isolated", "t_patched", *_SCORES, "t_dist",
               "t_degree", "t_nbr_detected")  # fmt: skip

FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "red": RED_GLOBAL + RED_TARGET + RED_SOURCE + RED_DERIVED + tuple(f"a_{a}" for a in ACTION_IDS["red"]),
    "blue": BLUE_GLOBAL + BLUE_TARGET + tuple(f"a_{a}" for a in ACTION_IDS["blue"]),
}
N_FEATURES = {s: len(v) for s, v in FEATURE_NAMES.items()}

# what each side's features may read from the env (the boundary test checks blue against the rest)
BLUE_VISIBLE = ("detected", "isolated", "patched", "confirmed", "scores", "turn")
HIDDEN_FROM_BLUE = ("compromised", "privilege", "known", "recon_done", "evasion_idx", "phished")


def _cap(x: np.ndarray | float, k: float) -> np.ndarray:
    return np.minimum(np.asarray(x, dtype=np.float64), k) / k


def _role_onehot(env: CyberArenaEnv) -> np.ndarray:
    r = np.zeros((env.n, 4))
    r[:, 0] = env.is_dmz
    r[:, 1] = env.is_ws
    r[:, 2] = env.is_srv
    r[env.crown, 3] = 1.0
    return r


def _turn_frac(env: CyberArenaEnv) -> float:
    return min(1.0, env.round / max(1, env.cfg.max_rounds))


def _red_tables(env: CyberArenaEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(global (G,), node table (n+1, T) with the last row = "no target", source table (n+1, S), front dist)."""
    fh = env.footholds()
    d = env.dist_to_crown.astype(np.float64)
    md = float(env.max_dist)
    front = float(d[fh].min()) if len(fh) else md
    crown = env.crown
    has_crown = bool(env.compromised[crown] and not env.isolated[crown])
    g = np.array([
        _turn_frac(env), _cap(len(fh), 5), _cap(int((env.privilege[fh] >= 2).sum()) if len(fh) else 0, 3),
        float(env.is_srv[fh].any()) if len(fh) else 0.0, float(has_crown),
        float(has_crown and env.privilege[crown] >= 2), min(1.0, front / md),
        float(env.scores[fh].max()) if len(fh) else 0.0,
        *(env.evasion(m) for m in MODELS),
    ])  # fmt: skip
    n = env.n
    smax = env.scores.max(axis=1)
    open_t = env.known & ~env.compromised & ~env.isolated
    open_nbrs = env.adj[:, :] @ open_t.astype(np.float64)
    T = np.zeros((n + 1, len(RED_TARGET)))
    T[:n, 1:5] = _role_onehot(env)
    T[:n, 5] = env.compromised
    T[:n, 6] = env.privilege / 2.0
    T[:n, 7] = env.isolated
    T[:n, 8] = env.patched
    T[:n, 9] = env.recon_done
    T[:n, 10:13] = env.scores
    T[:n, 13] = smax
    T[:n, 14] = np.minimum(d, md) / md
    T[:n, 15] = env.degree / max(1, env.degree.max())
    T[:n, 16] = d < front
    T[:n, 17] = _cap(open_nbrs, 4)
    T[n, 0] = 1.0
    S = np.zeros((n + 1, len(RED_SOURCE)))
    S[:n, 0] = 1.0
    S[:n, 1] = env.privilege / 2.0
    S[:n, 2] = env.recon_done
    S[:n, 3] = smax
    S[:n, 4] = np.minimum(d, md) / md
    return g, T, S, front


def _red_p_success(env: CyberArenaEnv, a: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Success odds red's tooling would quote for each move (rules + its own evasion; no blue state)."""
    cfg = env.cfg
    tt = np.where(t >= 0, t, 0)
    patched = env.patched[tt]
    p = np.ones(len(a))
    ev = {m: env._evade(m) for m in MODELS}
    p = np.where(a == RedAction.PHISH, cfg.p_phish * ev["phishing"], p)
    recon = np.where(s >= 0, env.recon_done[np.where(s >= 0, s, 0)], False)
    p_ex = np.where(patched, cfg.p_exploit_patched, cfg.p_exploit) + cfg.p_exploit_recon_bonus * recon
    p = np.where(a == RedAction.EXPLOIT, np.minimum(1.0, p_ex) * ev["network"], p)
    p_es = np.where(patched, cfg.p_escalate_patched, cfg.p_escalate)
    p = np.where(a == RedAction.ESCALATE, p_es * ev["malware"], p)
    p = np.where(a == RedAction.LATERAL_MOVE, cfg.p_lateral * ev["network"], p)
    p = np.where(a == RedAction.EXFILTRATE, cfg.p_exfiltrate * ev["network"], p)
    return p


def _blue_tables(env: CyberArenaEnv) -> tuple[np.ndarray, np.ndarray]:
    """Blue's view. Reads only ``BLUE_VISIBLE`` state plus topology and the clock."""
    n = env.n
    live = ~env.isolated
    undet = live & ~env.detected
    crown = env.crown
    d = env.dist_to_crown.astype(np.float64)
    md = float(env.max_dist)
    sc = env.blue_scores()
    smax = sc.max(axis=1)
    g = np.array([
        _turn_frac(env), _cap(int((env.detected & live).sum()), 5), _cap(int(env.isolated.sum()), 5),
        _cap(int(env.confirmed.sum()), 5), float(env.patched.mean()),
        float(smax[undet].max()) if undet.any() else 0.0,
        float(env.detected[crown]), float(env.isolated[crown]), float(smax[crown]),
    ])  # fmt: skip
    T = np.zeros((n + 1, len(BLUE_TARGET)))
    T[:n, 1:5] = _role_onehot(env)
    T[:n, 5] = env.detected
    T[:n, 6] = env.confirmed
    T[:n, 7] = env.isolated
    T[:n, 8] = env.patched
    T[:n, 9:12] = sc
    T[:n, 12] = smax
    T[:n, 13] = np.minimum(d, md) / md
    T[:n, 14] = env.degree / max(1, env.degree.max())
    T[:n, 15] = _cap(env.adj @ (env.detected & live).astype(np.float64), 4)
    T[n, 0] = 1.0
    return g, T


def candidate_matrix(env: CyberArenaEnv, side: str, moves: list[Move]) -> np.ndarray:
    """``(len(moves), N_FEATURES[side])`` float32 feature matrix, one row per move."""
    k = len(moves)
    n = env.n
    a = np.fromiter((m[0] for m in moves), dtype=np.int64, count=k)
    s = np.fromiter((-1 if m[1] is None else m[1] for m in moves), dtype=np.int64, count=k)
    t = np.fromiter((-1 if m[2] is None else m[2] for m in moves), dtype=np.int64, count=k)
    ti = np.where(t >= 0, t, n)
    if side == "red":
        g, T, S, _ = _red_tables(env)
        si = np.where(s >= 0, s, n)
        onehot = np.eye(len(RedAction))[a]
        X = np.concatenate([np.broadcast_to(g, (k, len(g))), T[ti], S[si],
                            _red_p_success(env, a, s, t)[:, None], onehot], axis=1)  # fmt: skip
    else:
        g, T = _blue_tables(env)
        onehot = np.eye(len(BlueAction))[a]
        X = np.concatenate([np.broadcast_to(g, (k, len(g))), T[ti], onehot], axis=1)
    return X.astype(np.float32)


def candidate_features(env: CyberArenaEnv, side: str, move: Move) -> np.ndarray:
    """Feature vector ``(N_FEATURES[side],)`` of one move."""
    return candidate_matrix(env, side, [move])[0]


def named(side: str, x: np.ndarray, nd: int = 4) -> dict[str, Any]:
    """``{feature name: value}`` for one row (turn records' ``chosen_features``)."""
    return {name: round(float(v), nd) for name, v in zip(FEATURE_NAMES[side], x, strict=True)}
