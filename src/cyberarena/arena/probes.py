"""Fixed probe states for ``learning.jsonl`` ``probe`` rows (contract: docs/contracts.md, v2 telemetry).

Each probe is a hand-picked feature tuple of one side's Q-table view (``red_features`` / ``blue_features`` in
``agents.py``), the actions that are legal in such a situation, and a plain-English description. At every eval
checkpoint the learner's Q-values for those actions are logged, so the dashboard can show the same situation
answered differently as training goes on.

Red view:  (stage, front_privilege, n_footholds, progress_move, recon_front, hot, turn_bucket)
Blue view: (n_detected_live, top_undetected_score_bucket, n_isolated, crown_alert, turn_bucket)
"""

from __future__ import annotations

from typing import Any

import numpy as np

PROBES: dict[str, list[tuple[str, str, tuple[int, ...], tuple[str, ...]]]] = {
    "red": [
        ("r1", "One fresh workstation foothold, user rights, not yet scanned, sensors quiet, early game",
         (0, 1, 1, 1, 1, 0, 0), ("recon", "phish", "exploit", "escalate", "wait")),
        ("r2", "Admin on its only foothold, an unowned host one step closer to the crown jewel, quiet",
         (0, 2, 1, 2, 0, 0, 0), ("lateral_move", "exploit", "phish", "wait")),
        ("r3", "Two footholds including a server, user rights at the front, sensors running hot, mid game",
         (1, 1, 2, 1, 0, 1, 1), ("escalate", "exploit", "lateral_move", "wait")),
        ("r4", "Admin on the crown jewel, mid game",
         (2, 2, 3, 0, 0, 0, 1), ("exfiltrate", "lateral_move", "wait")),
        ("r5", "On the crown jewel without admin, sensors hot, late game",
         (2, 1, 2, 0, 1, 1, 2), ("escalate", "recon", "exploit", "wait")),
        ("r6", "Last foothold left, user rights, no way forward, sensors hot, late game",
         (0, 1, 1, 0, 0, 1, 2), ("escalate", "phish", "wait")),
        ("r7", "Three footholds, admin on a server next to an open path to the crown jewel, quiet",
         (1, 2, 3, 2, 0, 0, 1), ("lateral_move", "exploit", "escalate", "wait")),
        ("r8", "Fresh foothold already flagged hot by the sensors, early game",
         (0, 1, 1, 1, 0, 1, 0), ("escalate", "exploit", "phish", "wait")),
    ],
    "blue": [
        ("b1", "All quiet early on: nothing flagged, every reading low",
         (0, 0, 0, 0, 0), ("monitor", "patch", "wait")),
        ("b2", "One host reads very suspicious (above 0.7) but is not confirmed, early game",
         (0, 2, 0, 0, 0), ("monitor", "patch", "wait")),
        ("b3", "One reading is borderline (0.3-0.7), nothing flagged, mid game",
         (0, 1, 0, 0, 1), ("monitor", "patch", "wait")),
        ("b4", "One host flagged and another reading borderline, mid game",
         (1, 1, 0, 0, 1), ("isolate", "reset_credentials", "monitor", "patch", "wait")),
        ("b5", "Two or more hosts flagged and the crown jewel is alerting",
         (2, 2, 0, 1, 1), ("isolate", "reset_credentials", "monitor")),
        ("b6", "Two hosts isolated, nothing else suspicious, mid game",
         (0, 0, 2, 0, 1), ("restore", "monitor", "patch", "wait")),
        ("b7", "Late game: one host flagged, one isolated, crown jewel alerting",
         (1, 0, 1, 1, 2), ("isolate", "reset_credentials", "restore", "monitor")),
        ("b8", "Late game: crown jewel reads hot but unflagged, one host isolated",
         (0, 2, 1, 1, 2), ("monitor", "restore", "patch", "wait")),
    ],
}


def probe_rows(agent: Any, after_episode: int) -> list[dict[str, Any]]:
    """``probe`` rows for one learner. Reads the Q-table without inserting unseen states."""
    rows = []
    idx = {a: i for i, a in enumerate(agent.ids)}
    for pid, desc, state, actions in PROBES[agent.side]:
        qv = agent.q.get(state)
        q = {a: round(float(qv[idx[a]]), 4) if qv is not None else 0.0 for a in actions}
        best = max(q.values())
        chosen = next(a for a in actions if q[a] >= best - 1e-12)
        rows.append({"kind": "probe", "after_episode": after_episode, "side": agent.side, "probe_id": pid,
                     "description": desc, "q": q, "chosen": chosen, "state": list(state),
                     "seen": qv is not None})  # fmt: skip
    return rows


# --------------------------------------------------------------------------------------------- v4 snapshots
#
# DQN agents score concrete moves, so their probes are whole env states (docs/contracts.md, "TensorFlow agents
# (v4)", logging changes). Each builder takes a freshly reset env (fixed seed, so the same start every
# checkpoint) and sets a readable situation on the run's own graph. Sensor scores are set explicitly, so the
# situation does not drift as detectors adapt. ``probe`` rows carry the top moves' Q-values keyed
# ``"<action>→<target>"`` plus the state snapshot.

PROBE_ENV_SEED = 9_090_901
PROBE_TOP_K = 8


def _quiet(env) -> None:
    env.scores[:] = 0.05
    env.detected[:] = False
    env.isolated[:] = False
    env.confirmed[:] = False
    env.patched[:] = False


def _set_red(env, nodes_priv: dict[int, int]) -> None:
    env.compromised[:] = False
    env.privilege[:] = 0
    for v, p in nodes_priv.items():
        env.compromised[v] = True
        env.privilege[v] = p
        env.known[v] = True
        env.known[env.nbrs[v]] = True


def _ws_next_to_server(env) -> int:
    for w in map(int, np.flatnonzero(env.is_ws)):
        if any(env.is_srv[v] for v in env.nbrs[w]):
            return w
    return int(np.flatnonzero(env.is_ws)[0])


def _server_of(env, w: int) -> int:
    return next((int(v) for v in env.nbrs[w] if env.is_srv[v]), int(np.flatnonzero(env.is_srv)[0]))


def _server_next_to_crown(env) -> int:
    for v in map(int, env.nbrs[env.crown]):
        if env.is_srv[v]:
            return v
    return int(env.nbrs[env.crown][0])


def _start(env) -> int:
    return int(np.flatnonzero(env.compromised)[0])


def _red_turn(env, frac: float) -> None:
    env.turn = 2 * int(frac * env.cfg.max_rounds)  # red to move


def _blue_turn(env, frac: float) -> None:
    env.turn = 2 * int(frac * env.cfg.max_rounds) + 1  # blue to move


def _r1(env):
    _quiet(env)
    _red_turn(env, 0.0)


def _r2(env):
    _quiet(env)
    w = _ws_next_to_server(env)
    _set_red(env, {w: 2})
    env.recon_done[w] = True
    _red_turn(env, 0.1)


def _r3(env):
    _quiet(env)
    w = _ws_next_to_server(env)
    srv = _server_of(env, w)
    _set_red(env, {w: 2, srv: 1})
    env.recon_done[w] = True
    env.scores[srv] = (0.8, 0.1, 0.6)
    _red_turn(env, 0.45)


def _r4(env):
    _quiet(env)
    _set_red(env, {_server_next_to_crown(env): 2, env.crown: 2})
    _red_turn(env, 0.45)


def _r5(env):
    _quiet(env)
    _set_red(env, {_server_next_to_crown(env): 2, env.crown: 1})
    env.scores[env.crown] = (0.75, 0.1, 0.7)
    _red_turn(env, 0.8)


def _r6(env):
    _quiet(env)
    w = _start(env)
    _set_red(env, {w: 1})
    env.recon_done[w] = True
    env.scores[w] = (0.7, 0.2, 0.65)
    _red_turn(env, 0.8)


def _r7(env):
    _quiet(env)
    w = _ws_next_to_server(env)
    s = _server_next_to_crown(env)
    _set_red(env, {w: 2, _server_of(env, w): 1, s: 2})
    env.recon_done[s] = True
    _red_turn(env, 0.4)


def _r8(env):
    _quiet(env)
    env.scores[_start(env)] = (0.8, 0.3, 0.75)
    _red_turn(env, 0.05)


def _b1(env):
    _quiet(env)
    _blue_turn(env, 0.0)


def _b2(env):
    _quiet(env)
    env.scores[int(np.flatnonzero(env.is_ws)[-1])] = (0.85, 0.2, 0.4)
    _blue_turn(env, 0.05)


def _b3(env):
    _quiet(env)
    env.scores[_ws_next_to_server(env)] = (0.5, 0.1, 0.45)
    _blue_turn(env, 0.45)


def _b4(env):
    _quiet(env)
    w = _ws_next_to_server(env)
    s = _server_of(env, w)
    _set_red(env, {w: 2, s: 1})
    env.detected[w] = True
    env.scores[w] = (0.8, 0.1, 0.7)
    env.scores[s] = (0.4, 0.05, 0.55)
    _blue_turn(env, 0.45)


def _b5(env):
    _quiet(env)
    s = _server_next_to_crown(env)
    w = _ws_next_to_server(env)
    _set_red(env, {w: 2, s: 2, env.crown: 1})
    env.detected[[w, s]] = True
    env.scores[w] = (0.7, 0.1, 0.8)
    env.scores[s] = (0.8, 0.1, 0.75)
    env.scores[env.crown] = (0.75, 0.05, 0.8)
    _blue_turn(env, 0.5)


def _b6(env):
    _quiet(env)
    ws = np.flatnonzero(env.is_ws)
    env.isolated[ws[:2]] = True
    env.confirmed[ws[0]] = True
    _set_red(env, {int(ws[-1]): 1})
    _blue_turn(env, 0.45)


def _b7(env):
    _quiet(env)
    s = _server_next_to_crown(env)
    w = _ws_next_to_server(env)
    _set_red(env, {s: 2, env.crown: 1})
    env.isolated[w] = True
    env.confirmed[w] = True
    env.detected[s] = True
    env.scores[s] = (0.8, 0.1, 0.7)
    env.scores[env.crown] = (0.7, 0.05, 0.75)
    _blue_turn(env, 0.8)


def _b8(env):
    _quiet(env)
    w = _ws_next_to_server(env)
    _set_red(env, {env.crown: 1, _server_next_to_crown(env): 2})
    env.isolated[w] = True
    env.scores[env.crown] = (0.85, 0.1, 0.8)
    _blue_turn(env, 0.85)


SNAPSHOT_PROBES: dict[str, list[tuple[str, str, Any]]] = {
    "red": [
        ("r1", "Game start: one fresh workstation foothold, user rights, not yet scanned, sensors quiet", _r1),
        ("r2", "Admin on a workstation next to a server, already scanned, quiet, early game", _r2),
        ("r3", "Admin workstation plus a user-level server foothold whose sensors run hot, mid game", _r3),
        ("r4", "Admin on the crown jewel, mid game", _r4),
        ("r5", "On the crown jewel without admin, crown sensors hot, late game", _r5),
        ("r6", "Back to the single starting foothold, user rights, sensors hot, late game", _r6),
        ("r7", "Three footholds, admin on a server next to the crown jewel, quiet", _r7),
        ("r8", "Fresh foothold already reading hot on the sensors, early game", _r8),
    ],
    "blue": [
        ("b1", "All quiet early on: nothing flagged, every reading low", _b1),
        ("b2", "One workstation reads very suspicious (0.85) but is not flagged, early game", _b2),
        ("b3", "One workstation reads borderline (0.5), nothing flagged, mid game", _b3),
        ("b4", "One workstation flagged and its neighbouring server borderline, mid game", _b4),
        ("b5", "Two hosts flagged and the crown jewel reading hot, mid game", _b5),
        ("b6", "Two workstations isolated (one proven compromised), nothing else suspicious, mid game", _b6),
        ("b7", "Late game: server next to the crown jewel flagged, one host isolated, crown jewel hot", _b7),
        ("b8", "Late game: crown jewel reads hot but unflagged, one host isolated", _b8),
    ],
}


def build_probe_state(env, side: str, pid: str) -> None:
    """Reset ``env`` to the fixed probe seed (evasion 0) and apply probe ``pid``'s situation."""
    env.evasion_idx[:] = 0
    env.reset(seed=PROBE_ENV_SEED)
    builder = next(b for p, _, b in SNAPSHOT_PROBES[side] if p == pid)
    builder(env)


def snapshot_probe_rows(agent: Any, env, after_episode: int) -> list[dict[str, Any]]:
    """``probe`` rows for one DQN learner: the top ``PROBE_TOP_K`` moves by Q on each snapshot state.

    Resets ``env``, so call it only between games; the evasion levels in force are restored afterwards."""
    from cyberarena.arena.dqn import move_key

    rows = []
    evasion = env.evasion_idx.copy()
    for pid, desc, _ in SNAPSHOT_PROBES[agent.side]:
        build_probe_state(env, agent.side, pid)
        moves, _, q = agent.q_values(env)
        order = np.argsort(-q, kind="stable")
        qd: dict[str, float] = {}
        for i in order:
            k = move_key(agent.ids[moves[i][0]], moves[i][2])
            if k not in qd:
                qd[k] = round(float(q[i]), 4)
            if len(qd) == PROBE_TOP_K:
                break
        chosen = move_key(agent.ids[moves[order[0]][0]], moves[order[0]][2])
        rows.append({"kind": "probe", "after_episode": after_episode, "side": agent.side, "probe_id": pid,
                     "description": desc, "q": qd, "chosen": chosen, "state": env.snapshot(),
                     "n_candidates": len(moves), "seen": True})  # fmt: skip
    env.evasion_idx[:] = evasion
    return rows
