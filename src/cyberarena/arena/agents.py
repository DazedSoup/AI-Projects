"""Red/blue agents: tabular Q-learning over a compact featurized view, plus random and heuristic baselines.

Each agent picks an *action type*; the concrete (source, target) pair comes from a shared targeting rule so
the learned policy, the heuristic and the random baseline differ only in which action they choose.
Red's view uses what an intruder knows (its footholds, privileges, its own sensor noise). Blue's view uses
only what a defender sees (classifier scores, detected/isolated/patched flags), never ground truth.

The default v4 learner is ``arena/dqn.py::DQNAgent``, which scores every concrete move (host included) with a
Keras Q-network; ``QAgent`` here is the v1-v3 learner kept as ``--agent tabular``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cyberarena.arena.actions import ACTION_IDS, BlueAction, RedAction, n_actions
from cyberarena.arena.env import CyberArenaEnv


@dataclass
class Decision:
    action: int
    source: int | None
    target: int | None
    action_id: str
    decision_values: dict[str, float]
    epsilon: float
    explored: bool
    choice: Any = None  # DQN agents: the scored candidate set (``dqn._Choice``), for logging

    def log_fields(self) -> tuple[dict[str, float], list[dict[str, Any]] | None, dict[str, float] | None]:
        """``decision_values``, ``candidates``, ``chosen_features`` of the turn record (v4).

        DQN agents key decision values by concrete move (``"exploit→7"``); tabular and scripted agents keep
        per-action-type values and have no candidate list or feature vector (``None``)."""
        if self.choice is None:
            return self.decision_values, None, None
        return self.choice.log_fields()


# --------------------------------------------------------------------------------------------- views


def _bucket(x: float, edges: tuple[float, ...]) -> int:
    return int(sum(x >= e for e in edges))


def _turn_bucket(env: CyberArenaEnv) -> int:
    return min(2, 3 * env.round // max(1, env.cfg.max_rounds))


def red_frontier(env: CyberArenaEnv) -> np.ndarray:
    """Footholds closest to the crown jewel."""
    fh = env.footholds()
    if len(fh) == 0:
        return fh
    d = env.dist_to_crown[fh]
    return fh[d == d.min()]


def red_progress(env: CyberArenaEnv) -> int:
    """0: no move gets closer to the crown jewel; 1: only via exploit; 2: via lateral_move (admin creds)."""
    front = red_frontier(env)
    if len(front) == 0:
        return 0
    open_t = env.known & ~env.compromised & ~env.isolated
    best = 0
    for s in front:
        closer = [t for t in env.nbrs[s] if open_t[t] and env.dist_to_crown[t] < env.dist_to_crown[s]]
        if closer:
            best = max(best, 2 if env.privilege[s] >= 2 else 1)
    return best


def red_features(env: CyberArenaEnv) -> tuple[int, ...]:
    fh = env.footholds()
    if len(fh) == 0:
        return (0, 0, 0, 0, 0, 0, _turn_bucket(env))
    front = red_frontier(env)
    crown = env.crown
    stage = 2 if crown in fh else int(env.is_srv[fh].any())
    priv = int(env.privilege[front].max())
    hot = bool(env.scores[fh].max() > 0.5)
    recon_front = bool((~env.recon_done[front]).any())
    return (stage, priv, min(len(fh), 3), red_progress(env), int(recon_front), int(hot), _turn_bucket(env))


def blue_features(env: CyberArenaEnv) -> tuple[int, ...]:
    live = ~env.isolated
    n_det = int((env.detected & live).sum())
    undet = live & ~env.detected
    top = float(env.scores[undet].max()) if undet.any() else 0.0
    n_iso = int(env.isolated.sum())
    crown = env.crown
    crown_alert = bool(env.detected[crown] or env.scores[crown].max() > 0.5)
    return (min(n_det, 2), _bucket(top, (0.3, 0.7)), min(n_iso, 2), int(crown_alert), _turn_bucket(env))


FEATURES = {"red": red_features, "blue": blue_features}


# --------------------------------------------------------------------------------------------- targeting


def choose_target(env: CyberArenaEnv, side: str, action: int, rng: np.random.Generator,
                  randomize: bool = False) -> tuple[int | None, int | None]:  # fmt: skip
    cands = env.candidates(side, action)
    if not cands:
        raise ValueError(f"{side} action {action} has no candidates")
    if randomize or len(cands) == 1:
        return cands[int(rng.integers(len(cands)))]
    if side == "red":
        if action == RedAction.PHISH:
            return cands[int(rng.integers(len(cands)))]
        # head for the crown jewel; prefer unpatched targets
        key = np.array([env.dist_to_crown[t] + 0.5 * env.patched[t] for _, t in cands], dtype=float)
    else:
        if action in (BlueAction.MONITOR, BlueAction.ISOLATE, BlueAction.RESET_CREDENTIALS):
            key = np.array([-env.scores[t].max() for _, t in cands])
        elif action == BlueAction.PATCH:  # harden the path to the crown jewel first
            key = np.array([env.dist_to_crown[t] for _, t in cands], dtype=float)
        else:
            return cands[int(rng.integers(len(cands)))]
    best = np.flatnonzero(key <= key.min() + 1e-9)
    return cands[int(best[rng.integers(len(best))])]


# --------------------------------------------------------------------------------------------- agents


class BaseAgent:
    kind = "base"
    learns = False

    def __init__(self, side: str, seed: int | None = None):
        self.side = side
        self.ids = ACTION_IDS[side]
        self.rng = np.random.default_rng(seed)

    def act(self, env: CyberArenaEnv, explore: bool = True) -> Decision:
        raise NotImplementedError

    def _decide(self, env: CyberArenaEnv, a: int, values: dict[str, float], eps: float, explored: bool,
                randomize_target: bool = False) -> Decision:  # fmt: skip
        src, tgt = choose_target(env, self.side, a, self.rng, randomize_target)
        return Decision(a, src, tgt, self.ids[a], values, eps, explored)


class RandomAgent(BaseAgent):
    kind = "random"

    def act(self, env: CyberArenaEnv, explore: bool = True) -> Decision:
        valid = env.valid_actions(self.side)
        a = int(self.rng.choice(valid))
        return self._decide(env, a, {self.ids[v]: 0.0 for v in valid}, 1.0, True, randomize_target=True)


class HeuristicAgent(BaseAgent):
    """Fixed priority playbook. ``decision_values`` are priority scores (higher = preferred), not Q-values.

    ``noise`` is the probability of playing a uniformly random valid action instead (keeps the baseline
    beatable without being random).
    """

    kind = "heuristic"

    def __init__(self, side: str, seed: int | None = None, noise: float = 0.0):
        super().__init__(side, seed)
        self.noise = noise

    def _priorities(self, env: CyberArenaEnv, valid: list[int]) -> dict[int, float]:
        p: dict[int, float] = {}
        if self.side == "red":
            front = red_frontier(env)
            prog = red_progress(env)
            order = [RedAction.EXFILTRATE]
            if prog == 2:
                order.append(RedAction.LATERAL_MOVE)
            if (env.privilege[front] == 1).any():
                order.append(RedAction.ESCALATE)
            if (~env.recon_done[front]).any():
                order.append(RedAction.RECON)
            if prog == 1:
                order.append(RedAction.EXPLOIT)
            order += [RedAction.ESCALATE, RedAction.LATERAL_MOVE, RedAction.EXPLOIT, RedAction.RECON,
                      RedAction.PHISH, RedAction.WAIT]  # fmt: skip
            seen_r: list[RedAction] = []
            for a in order:
                if a not in seen_r:
                    seen_r.append(a)
            for rank, a in enumerate(seen_r):
                p[int(a)] = float(len(seen_r) - rank)
        else:
            live = ~env.isolated
            undet = live & ~env.detected
            hot = undet.any() and env.scores[undet].max() > 0.5
            order = [BlueAction.ISOLATE]
            order += [BlueAction.MONITOR] if hot else []
            order += [BlueAction.RESTORE] if env.isolated.sum() >= 2 else []
            order += [BlueAction.PATCH] if env.round % 3 == 0 else []
            order += [BlueAction.MONITOR, BlueAction.RESTORE, BlueAction.PATCH, BlueAction.WAIT]
            seen: list[BlueAction] = []
            for a in order:
                if a not in seen:
                    seen.append(a)
            for rank, a in enumerate(seen):
                p[int(a)] = float(len(seen) - rank)
        return {a: p.get(a, 0.0) for a in valid}

    def act(self, env: CyberArenaEnv, explore: bool = True) -> Decision:
        valid = env.valid_actions(self.side)
        pri = self._priorities(env, valid)
        values = {self.ids[a]: v for a, v in pri.items()}
        if self.noise > 0 and self.rng.random() < self.noise:
            return self._decide(env, int(self.rng.choice(valid)), values, self.noise, True)
        a = max(valid, key=lambda x: pri[x])
        return self._decide(env, a, values, self.noise, False)


class QAgent(BaseAgent):
    """Tabular Q-learning, epsilon-greedy over valid actions, Q-table keyed by the side's feature tuple.

    The v1-v3 learner (``--agent tabular``): it picks an action type; ``choose_target`` picks the host."""

    kind = "learned"
    learns = True
    agent_type = "tabular"

    def __init__(self, side: str, seed: int | None = None, alpha: float = 0.1, gamma: float = 0.97,
                 epsilon: float = 1.0, replay_size: int = 20000):  # fmt: skip
        super().__init__(side, seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_size = replay_size
        # (key, action, reward, next_key | None, next_valid) -- next_key None marks a terminal transition
        self.buffer: list[tuple[tuple[int, ...], int, float, tuple[int, ...] | None, tuple[int, ...]]] = []
        self._buf_pos = 0
        self.n = n_actions(side)
        self.q: dict[tuple[int, ...], np.ndarray] = {}
        self.features = FEATURES[side]
        self._pending: tuple[tuple[int, ...], int, float] | None = None
        self.reset_stats()

    def reset_stats(self) -> None:
        """Rolling-window counters behind the ``agent_stats`` rows of learning.jsonl."""
        self.td_abs_sum = 0.0
        self.td_n = 0
        self.action_counts = np.zeros(self.n, dtype=np.int64)

    def count_action(self, a: int) -> None:
        self.action_counts[a] += 1

    def stats(self) -> dict[str, Any]:
        n_act = int(self.action_counts.sum())
        q = np.array([v for v in self.q.values()]) if self.q else np.zeros((0, self.n))
        return {"n_states": len(self.q),
                "mean_abs_q": round(float(np.abs(q).mean()), 4) if q.size else 0.0,
                "td_error": round(self.td_abs_sum / self.td_n, 4) if self.td_n else None,
                "n_updates": self.td_n,
                "epsilon": round(float(self.epsilon), 4),
                "action_mix": {self.ids[i]: round(int(c) / n_act, 4) for i, c in enumerate(self.action_counts)
                               if c} if n_act else {}}  # fmt: skip

    def values(self, key: tuple[int, ...]) -> np.ndarray:
        v = self.q.get(key)
        if v is None:
            v = self.q[key] = np.zeros(self.n)
        return v

    def act(self, env: CyberArenaEnv, explore: bool = True) -> Decision:
        valid = env.valid_actions(self.side)
        key = self.features(env)
        qv = self.values(key)
        values = {self.ids[a]: round(float(qv[a]), 4) for a in valid}
        eps = self.epsilon if explore else 0.0
        if explore and self.rng.random() < eps:
            return self._decide(env, int(self.rng.choice(valid)), values, eps, True)
        best = max(qv[a] for a in valid)
        ties = [a for a in valid if qv[a] >= best - 1e-12]
        a = int(ties[int(self.rng.integers(len(ties)))])
        return self._decide(env, a, values, eps, False)

    # -- learning (turn-based: one transition spans own action + opponent reply) --------------------

    def begin_episode(self) -> None:
        self._pending = None

    # driver hooks shared with ``DQNAgent``: called around ``act`` on the learner's own turns
    def pre_step(self, env: CyberArenaEnv) -> None:
        self.before_act(env)
        self._key = self.features(env)

    def post_step(self, decision: Decision) -> None:
        self.after_act(self._key, decision.action)
        self.count_action(decision.action)

    def before_act(self, env: CyberArenaEnv) -> None:
        """Close the previous transition now that the next own state is known."""
        if self._pending is None:
            return
        key, a, r = self._pending
        nxt = self.features(env)
        valid = tuple(env.valid_actions(self.side))
        self._learn(key, a, r, nxt, valid)
        self._pending = None

    def after_act(self, key: tuple[int, ...], action: int) -> None:
        self._pending = (key, action, 0.0)

    def add_reward(self, r: float) -> None:
        if self._pending is not None:
            k, a, acc = self._pending
            self._pending = (k, a, acc + r)

    def end_episode(self) -> None:
        if self._pending is not None:
            key, a, r = self._pending
            self._learn(key, a, r, None, ())
        self._pending = None

    def _learn(self, key: tuple[int, ...], a: int, r: float, nxt: tuple[int, ...] | None,
               valid: tuple[int, ...]) -> None:  # fmt: skip
        self._update(key, a, r, nxt, valid)
        if self.replay_size > 0:
            item = (key, a, r, nxt, valid)
            if len(self.buffer) < self.replay_size:
                self.buffer.append(item)
            else:
                self.buffer[self._buf_pos] = item
                self._buf_pos = (self._buf_pos + 1) % self.replay_size

    def _update(self, key: tuple[int, ...], a: int, r: float, nxt: tuple[int, ...] | None,
                valid: tuple[int, ...]) -> None:  # fmt: skip
        target = r if nxt is None else r + self.gamma * max(self.values(nxt)[v] for v in valid)
        qv = self.values(key)
        td = target - qv[a]
        self.td_abs_sum += abs(float(td))
        self.td_n += 1
        qv[a] += self.alpha * td

    def replay(self, n: int) -> None:
        """Re-apply ``n`` uniformly sampled past transitions (cheap tabular experience replay)."""
        if not self.buffer or n <= 0:
            return
        for i in self.rng.integers(0, len(self.buffer), size=n):
            self._update(*self.buffer[int(i)])

    # -- persistence ---------------------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {"side": self.side, "kind": "tabular_q", "alpha": self.alpha, "gamma": self.gamma,
                "epsilon": self.epsilon, "actions": list(self.ids), "feature_fn": self.features.__name__,
                "q": {",".join(map(str, k)): [round(float(x), 6) for x in v] for k, v in sorted(self.q.items())}}

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_json()))

    @classmethod
    def load(cls, path: Path | str, seed: int | None = None) -> QAgent:
        d = json.loads(Path(path).read_text())
        agent = cls(d["side"], seed=seed, alpha=d["alpha"], gamma=d["gamma"], epsilon=d["epsilon"])
        agent.q = {tuple(int(x) for x in k.split(",")): np.array(v) for k, v in d["q"].items()}
        return agent


def make_baseline(side: str, kind: str, seed: int | None = None, noise: float = 0.0) -> BaseAgent:
    if kind == "random":
        return RandomAgent(side, seed)
    if kind == "heuristic":
        return HeuristicAgent(side, seed, noise=noise)
    raise ValueError(kind)
