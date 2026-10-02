"""Simulated host-graph arena (Gymnasium env). SIMULATION ONLY.

Hosts are graph nodes with boolean/int attributes; "attacks" are discrete actions that change those attributes
by probability. Each node also carries three sensor readings: feature rows drawn from the held-out test split
of the malware / phishing / network datasets, scored by the trained classifiers. Malicious red activity makes
the acted-on node emit positive-class rows (with an action-specific probability); everything else emits
negative-class rows. Rows are pre-sampled and pre-scored in one batch per (model, class) at construction, so
``step()`` never calls the model.

Turn order: red acts, then blue, alternating. ``step(action)`` applies one actor's action.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cyberarena.arena.actions import BlueAction, RedAction

MODELS: tuple[str, ...] = ("malware", "phishing", "network")
M_IDX = {m: i for i, m in enumerate(MODELS)}
ROLES = ("dmz", "workstation", "server")
NODE_FEATURES = ("compromised", "detected", "isolated", "patched", "privilege",
                 "score_malware", "score_phishing", "score_network")  # fmt: skip


@dataclass
class ArenaConfig:
    """Env hyperparameters. Probabilities are per attempt; rewards are from each side's point of view."""

    max_rounds: int = 36  # one round = one red turn + one blue turn; blue wins at the limit
    pool_size: int = 256  # pre-scored rows per (model, class)
    detect_threshold: float = 0.5  # monitor flags a node when any re-sampled score exceeds this
    # success probability of each red action (patched targets use the *_patched value)
    p_phish: float = 0.35
    p_exploit: float = 0.45
    p_exploit_patched: float = 0.12
    p_exploit_recon_bonus: float = 0.2
    p_escalate: float = 0.65
    p_escalate_patched: float = 0.25
    p_lateral: float = 0.8
    p_exfiltrate: float = 0.75
    # probability that the acted-on node emits a positive-class row for the action's sensor
    noise: dict[str, float] = field(default_factory=lambda: {
        "recon": 0.25, "phish": 0.85, "exploit": 0.75, "escalate": 0.6,
        "lateral_move": 0.4, "exfiltrate": 0.9})  # fmt: skip
    # probability a compromised node leaks a positive row per sensor when blue monitors it
    p_implant_leak: dict[str, float] = field(default_factory=lambda: {
        "malware": 0.2, "phishing": 0.05, "network": 0.1})  # fmt: skip
    phish_max_footholds: int = 1  # phishing is a re-entry tool: only offered while red holds <= this many
    p_reset_evicts: float = 0.5  # reset_credentials on a privilege-1 foothold evicts it with this prob
    # reward shaping (red's view; blue receives the negation of red's progress terms)
    r_win: float = 1.0
    r_new_foothold: float = 0.05
    r_server_foothold: float = 0.1
    r_crown_foothold: float = 0.25
    r_privilege: float = 0.05
    r_foothold_lost: float = 0.1
    r_turn: float = 0.005  # red pays per turn (time pressure); blue earns it
    # blue-only terms
    r_isolate_true: float = 0.1
    r_isolate_false: float = 0.1
    r_detect_true: float = 0.0  # monitor flags a truly compromised node
    r_detect_false: float = 0.0  # monitor flags a clean node (false positive)
    r_isolated_upkeep: float = 0.01  # per isolated node per blue turn (business disruption)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------------- graph


@dataclass
class HostGraph:
    roles: list[str]
    crown: int
    edges: list[tuple[int, int]]
    pos: np.ndarray  # (n, 2) layout in [0, 1]

    @property
    def n(self) -> int:
        return len(self.roles)

    def neighbors(self) -> list[np.ndarray]:
        nb: list[list[int]] = [[] for _ in range(self.n)]
        for u, v in self.edges:
            nb[u].append(v)
            nb[v].append(u)
        return [np.array(sorted(x), dtype=np.int64) for x in nb]

    def to_json(self) -> dict[str, Any]:
        return {
            "nodes": [{"id": i, "role": r, "crown_jewel": i == self.crown,
                       "x": round(float(self.pos[i, 0]), 4), "y": round(float(self.pos[i, 1]), 4)}
                      for i, r in enumerate(self.roles)],
            "edges": [[int(u), int(v)] for u, v in self.edges],
        }


def generate_graph(seed: int, n_nodes: int | None = None) -> HostGraph:
    """Seeded 12-16 node enterprise-ish topology: DMZ -> servers <- workstations, crown jewel behind servers."""
    rng = np.random.default_rng(seed)
    n = int(n_nodes) if n_nodes is not None else int(rng.integers(12, 17))
    if not 10 <= n <= 20:
        raise ValueError("n_nodes must be in [10, 20]")
    n_dmz = int(rng.integers(2, 4))
    n_srv = int(rng.integers(3, 5))
    n_ws = n - n_dmz - n_srv - 1
    dmz = list(range(n_dmz))
    ws = list(range(n_dmz, n_dmz + n_ws))
    srv = list(range(n_dmz + n_ws, n_dmz + n_ws + n_srv))
    crown = n - 1
    roles = ["dmz"] * n_dmz + ["workstation"] * n_ws + ["server"] * n_srv + ["server"]
    edges: set[tuple[int, int]] = set()

    def add(u: int, v: int) -> None:
        if u != v:
            edges.add((min(u, v), max(u, v)))

    for i in range(1, n_dmz):  # DMZ chain
        add(dmz[i - 1], dmz[i])
    front_srv = srv[: max(1, n_srv // 2)]  # servers reachable from the DMZ
    for d in dmz:
        add(d, int(rng.choice(front_srv)))
    for i in range(1, n_ws):  # workstation LAN: random tree + one extra edge
        add(ws[i], ws[int(rng.integers(0, i))])
    if n_ws >= 4:
        a, b = rng.choice(ws, size=2, replace=False)
        add(int(a), int(b))
    ws_uplinks = rng.choice(ws, size=min(n_ws, max(2, n_ws // 2)), replace=False)
    for w in ws_uplinks:  # a few workstations reach file/app servers
        add(int(w), int(rng.choice(srv)))
    for i in range(1, n_srv):  # server segment chain
        add(srv[i - 1], srv[i])
    back_srv = srv[n_srv // 2:] or srv  # crown jewel sits behind the back servers
    for s in rng.choice(back_srv, size=min(len(back_srv), 2), replace=False):
        add(int(s), crown)

    # tiered layout: internet-facing left, crown jewel right
    tier_x = {"dmz": 0.08, "workstation": 0.36, "server": 0.66}
    groups = {"dmz": dmz, "workstation": ws, "server": srv, "crown": [crown]}
    pos = np.zeros((n, 2))
    for key, members in groups.items():
        x = 0.94 if key == "crown" else tier_x[key]
        for k, node in enumerate(members):
            y = (k + 1) / (len(members) + 1)
            jitter = rng.uniform(-0.03, 0.03, size=2) if key != "crown" else np.zeros(2)
            pos[node] = np.clip([x + jitter[0], y + jitter[1]], 0.0, 1.0)
    return HostGraph(roles=roles, crown=crown, edges=sorted(edges), pos=pos)


def _bfs_dist(neighbors: list[np.ndarray], src: int) -> np.ndarray:
    dist = np.full(len(neighbors), 99, dtype=np.int64)
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in neighbors[u]:
            if dist[v] == 99:
                dist[v] = dist[u] + 1
                q.append(int(v))
    return dist


# --------------------------------------------------------------------------------------------- sensors


class SensorPool:
    """Pre-sampled, pre-scored classifier rows per (model, class)."""

    def __init__(self, classifiers: dict[str, Any], pool_size: int, seed: int):
        self.rows: dict[tuple[str, int], np.ndarray] = {}
        self.scores: dict[tuple[str, int], np.ndarray] = {}
        rng = np.random.default_rng(seed)
        for m in MODELS:
            clf = classifiers[m]
            for label in (0, 1):
                X = np.asarray(clf.sample(label, pool_size, rng), dtype=np.float32)
                self.rows[(m, label)] = X
                self.scores[(m, label)] = np.asarray(clf.predict_proba(X), dtype=np.float64).reshape(-1)

    def size(self, model: str, label: int) -> int:
        return len(self.scores[(model, label)])


def load_default_classifiers() -> dict[str, Any]:
    from cyberarena.ml.inference import load_classifier

    return {m: load_classifier(m) for m in MODELS}


# --------------------------------------------------------------------------------------------- env


class CyberArenaEnv(gym.Env):
    """Turn-based red/blue host-graph environment.

    ``step(action)`` takes either an action index for the current actor (target chosen by a default rule) or
    a tuple ``(action_index, source, target)`` with source/target drawn from ``candidates(side, action)``.
    The returned reward is the acting side's reward; ``info["rewards"]`` has both sides' rewards.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self, config: ArenaConfig | None = None, graph_seed: int = 0,
                 classifiers: dict[str, Any] | None = None, n_nodes: int | None = None,
                 pool_seed: int | None = None):  # fmt: skip
        super().__init__()
        self.cfg = config or ArenaConfig()
        self.graph = generate_graph(graph_seed, n_nodes)
        self.n = self.graph.n
        self.roles = np.array(self.graph.roles)
        self.crown = self.graph.crown
        self.nbrs = self.graph.neighbors()
        self.adj = np.zeros((self.n, self.n), dtype=bool)
        for u, v in self.graph.edges:
            self.adj[u, v] = self.adj[v, u] = True
        self.dist_to_crown = _bfs_dist(self.nbrs, self.crown)
        self.is_dmz = self.roles == "dmz"
        self.is_ws = self.roles == "workstation"
        self.is_srv = (self.roles == "server") & (np.arange(self.n) != self.crown)
        self.classifiers = classifiers if classifiers is not None else load_default_classifiers()
        self.pool = SensorPool(self.classifiers, self.cfg.pool_size,
                               graph_seed + 1 if pool_seed is None else pool_seed)  # fmt: skip
        self.action_space = spaces.Discrete(len(RedAction))  # current actor's; blue uses len(BlueAction)
        self.observation_space = spaces.Box(0.0, 2.0, shape=(self.n, len(NODE_FEATURES)), dtype=np.float32)
        self._init_state()

    # -- state -------------------------------------------------------------------------------------

    def _init_state(self) -> None:
        n = self.n
        self.compromised = np.zeros(n, dtype=bool)
        self.detected = np.zeros(n, dtype=bool)
        self.isolated = np.zeros(n, dtype=bool)
        self.patched = np.zeros(n, dtype=bool)
        self.privilege = np.zeros(n, dtype=np.int64)
        self.known = np.zeros(n, dtype=bool)  # red's knowledge of the graph
        self.recon_done = np.zeros(n, dtype=bool)
        self.scores = np.zeros((n, len(MODELS)), dtype=np.float64)
        self.turn = 0
        self.done = False
        self.winner: str | None = None

    @property
    def actor(self) -> str:
        return "red" if self.turn % 2 == 0 else "blue"

    @property
    def round(self) -> int:
        return self.turn // 2

    def footholds(self) -> np.ndarray:
        return np.flatnonzero(self.compromised & ~self.isolated)

    def _emit(self, node: int, model: str, label: int, log: list[dict] | None) -> float:
        idx = int(self.np_random.integers(0, self.pool.size(model, label)))
        score = float(self.pool.scores[(model, label)][idx])
        self.scores[node, M_IDX[model]] = score
        if log is not None:
            log.append({"node": int(node), "model": model, "label": label, "pool_index": idx,
                        "score": round(score, 4)})  # fmt: skip
        return score

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._init_state()
        for i in range(self.n):  # baseline benign sensor readings
            for m in MODELS:
                self._emit(i, m, 0, None)
        # red's initial beachhead: a previously phished workstation (privilege 1, undetected)
        start = int(self.np_random.choice(np.flatnonzero(self.is_ws)))
        self.compromised[start] = True
        self.privilege[start] = 1
        self.known[:] = self.is_dmz | self.is_ws  # DMZ is public; staff directory is phishable
        self.known[self.nbrs[start]] = True
        self.known[start] = True
        return self.observation(), {"start_node": start, "actor": self.actor}

    def observation(self) -> np.ndarray:
        return np.column_stack([self.compromised, self.detected, self.isolated, self.patched,
                                self.privilege, self.scores]).astype(np.float32)  # fmt: skip

    def node_states(self) -> list[dict[str, Any]]:
        out = []
        for i in range(self.n):
            out.append({"id": i, "compromised": bool(self.compromised[i]), "detected": bool(self.detected[i]),
                        "isolated": bool(self.isolated[i]), "patched": bool(self.patched[i]),
                        "privilege": int(self.privilege[i]),
                        "scores": {m: round(float(self.scores[i, k]), 4) for k, m in enumerate(MODELS)}})
        return out

    def classifier_row(self, entry: dict) -> list[float]:
        """Feature row behind a classifier_inputs entry produced by ``step``."""
        return self.pool.rows[(entry["model"], entry["label"])][entry["pool_index"]].tolist()

    # -- candidates --------------------------------------------------------------------------------

    def candidates(self, side: str, action: int) -> list[tuple[int | None, int | None]]:
        """Valid (source, target) pairs for ``action``; empty list means the action is not available."""
        fh = self.footholds()
        if side == "red":
            a = RedAction(action)
            if a == RedAction.WAIT:
                return [(None, None)]
            if a == RedAction.RECON:
                return [(int(s), int(s)) for s in fh if not self.recon_done[s]]
            if a == RedAction.PHISH:
                if len(fh) > self.cfg.phish_max_footholds:
                    return []
                tg = np.flatnonzero(self.is_ws & ~self.compromised & ~self.isolated)
                return [(None, int(t)) for t in tg]
            if a == RedAction.EXPLOIT:
                out: list[tuple[int | None, int | None]] = []
                open_t = self.known & ~self.compromised & ~self.isolated
                out += [(None, int(t)) for t in np.flatnonzero(open_t & self.is_dmz)]
                for s in fh:
                    out += [(int(s), int(t)) for t in self.nbrs[s] if open_t[t]]
                return out
            if a == RedAction.ESCALATE:
                return [(int(s), int(s)) for s in fh if self.privilege[s] == 1]
            if a == RedAction.LATERAL_MOVE:
                out = []
                open_t = self.known & ~self.compromised & ~self.isolated
                for s in fh:
                    if self.privilege[s] >= 2:
                        out += [(int(s), int(t)) for t in self.nbrs[s] if open_t[t]]
                return out
            if a == RedAction.EXFILTRATE:
                c = self.crown
                ok = self.compromised[c] and not self.isolated[c] and self.privilege[c] >= 2
                return [(c, c)] if ok else []
            raise ValueError(a)
        b = BlueAction(action)
        live = ~self.isolated
        if b == BlueAction.WAIT:
            return [(None, None)]
        if b == BlueAction.MONITOR:
            return [(None, int(t)) for t in np.flatnonzero(live & ~self.detected)]
        if b == BlueAction.ISOLATE:
            return [(None, int(t)) for t in np.flatnonzero(live & self.detected)]
        if b == BlueAction.PATCH:
            return [(None, int(t)) for t in np.flatnonzero(live & ~self.patched)]
        if b == BlueAction.RESTORE:
            return [(None, int(t)) for t in np.flatnonzero(self.isolated)]
        if b == BlueAction.RESET_CREDENTIALS:
            return [(None, int(t)) for t in np.flatnonzero(live & self.detected)]
        raise ValueError(b)

    def valid_actions(self, side: str | None = None) -> list[int]:
        side = side or self.actor
        enum = RedAction if side == "red" else BlueAction
        return [int(a) for a in enum if self.candidates(side, int(a))]

    def default_target(self, side: str, action: int) -> tuple[int | None, int | None]:
        """Deterministic-ish target rule used when ``step`` gets a bare action index."""
        cands = self.candidates(side, action)
        if not cands:
            return (None, None)
        if side == "red":
            key = [self.dist_to_crown[t] if t is not None else 0 for _, t in cands]
        else:
            key = [-self.scores[t].max() if t is not None else 0 for _, t in cands]
        return cands[int(np.argmin(key))]

    # -- dynamics ----------------------------------------------------------------------------------

    def step(self, action):
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        if isinstance(action, (tuple, list)):
            a, src, tgt = int(action[0]), action[1], action[2]
        else:
            a = int(action)
            src, tgt = self.default_target(self.actor, a)
        side = self.actor
        valid = (src, tgt) in self.candidates(side, a)
        if not valid:
            raise ValueError(f"invalid {side} action {a} with source={src} target={tgt}")
        emitted: list[dict] = []
        r = {"red": 0.0, "blue": 0.0}
        cfg = self.cfg
        rng = self.np_random
        success = False
        events: list[str] = []

        def progress(amount: float) -> None:  # red progress is blue's loss
            r["red"] += amount
            r["blue"] -= amount

        def gain_foothold(t: int, priv: int) -> None:
            self.compromised[t] = True
            self.privilege[t] = max(self.privilege[t], priv)
            bonus = cfg.r_new_foothold
            if t == self.crown:
                bonus += cfg.r_crown_foothold
            elif self.is_srv[t]:
                bonus += cfg.r_server_foothold
            progress(bonus)
            events.append(f"foothold:{t}")

        def lose_foothold(t: int) -> None:
            progress(-cfg.r_foothold_lost - (cfg.r_crown_foothold if t == self.crown else 0.0))
            events.append(f"evicted:{t}")

        if side == "red":
            act = RedAction(a)
            aid = act.action_id
            if act == RedAction.RECON:
                success = True
                self.recon_done[tgt] = True
                newly = self.nbrs[tgt][~self.known[self.nbrs[tgt]]]
                self.known[self.nbrs[tgt]] = True
                events.append(f"discovered:{len(newly)}")
                self._emit(tgt, "network", int(rng.random() < cfg.noise[aid]), emitted)
            elif act == RedAction.PHISH:
                success = rng.random() < cfg.p_phish
                if success:
                    gain_foothold(tgt, 1)
                self._emit(tgt, "phishing", int(rng.random() < cfg.noise[aid]), emitted)
            elif act == RedAction.EXPLOIT:
                p = cfg.p_exploit_patched if self.patched[tgt] else cfg.p_exploit
                if src is not None and self.recon_done[src]:
                    p += cfg.p_exploit_recon_bonus
                success = rng.random() < p
                if success:
                    gain_foothold(tgt, 1)
                self._emit(tgt, "network", int(rng.random() < cfg.noise[aid]), emitted)
            elif act == RedAction.ESCALATE:
                p = cfg.p_escalate_patched if self.patched[tgt] else cfg.p_escalate
                success = rng.random() < p
                if success:
                    self.privilege[tgt] = 2
                    progress(cfg.r_privilege * (3.0 if tgt == self.crown else 1.0))
                self._emit(tgt, "malware", int(rng.random() < cfg.noise[aid]), emitted)
            elif act == RedAction.LATERAL_MOVE:
                success = rng.random() < cfg.p_lateral
                if success:
                    gain_foothold(tgt, 1)
                self._emit(tgt, "network", int(rng.random() < cfg.noise[aid]), emitted)
            elif act == RedAction.EXFILTRATE:
                success = rng.random() < cfg.p_exfiltrate
                self._emit(tgt, "network", int(rng.random() < cfg.noise[aid]), emitted)
                if success:
                    self.done = True
                    self.winner = "red"
            else:  # wait
                success = True
            progress(-cfg.r_turn)
        else:
            act = BlueAction(a)
            aid = act.action_id
            if act == BlueAction.MONITOR:
                hit = False
                for m in MODELS:
                    label = int(self.compromised[tgt] and rng.random() < cfg.p_implant_leak[m])
                    hit |= self._emit(tgt, m, label, emitted) > cfg.detect_threshold
                if hit:
                    if not self.detected[tgt]:
                        r["blue"] += cfg.r_detect_true if self.compromised[tgt] else -cfg.r_detect_false
                    self.detected[tgt] = True
                    events.append(f"flagged:{tgt}")
                success = hit
            elif act == BlueAction.ISOLATE:
                self.isolated[tgt] = True
                success = bool(self.compromised[tgt])
                if success:
                    lose_foothold(tgt)
                    r["blue"] += cfg.r_isolate_true
                else:
                    r["blue"] -= cfg.r_isolate_false
            elif act == BlueAction.PATCH:
                self.patched[tgt] = True
                success = True
            elif act == BlueAction.RESTORE:
                # reimage: clean node rejoins the network; red's foothold there (if any) is wiped
                self.isolated[tgt] = False
                self.detected[tgt] = False
                self.compromised[tgt] = False
                self.privilege[tgt] = 0
                self.patched[tgt] = True
                for m in MODELS:
                    self._emit(tgt, m, 0, emitted)
                success = True
            elif act == BlueAction.RESET_CREDENTIALS:
                success = False
                if self.compromised[tgt]:
                    if self.privilege[tgt] >= 2:
                        self.privilege[tgt] = 1
                        progress(-cfg.r_privilege)
                        success = True
                    elif rng.random() < cfg.p_reset_evicts:
                        self.compromised[tgt] = False
                        self.privilege[tgt] = 0
                        lose_foothold(tgt)
                        success = True
                self.detected[tgt] = False if not self.compromised[tgt] else self.detected[tgt]
            r["blue"] -= cfg.r_isolated_upkeep * float(self.isolated.sum())

        self.turn += 1
        if not self.done and len(self.footholds()) == 0:
            self.done = True
            self.winner = "blue"
        truncated = False
        if not self.done and self.turn >= 2 * cfg.max_rounds:
            self.done = True
            truncated = True
            self.winner = "blue"
        if self.done:
            r[self.winner] += cfg.r_win
            loser = "blue" if self.winner == "red" else "red"
            r[loser] -= cfg.r_win
        info = {
            "actor": side, "action": a, "action_id": aid, "source": src, "target": tgt,
            "success": bool(success), "rewards": r, "classifier_inputs": emitted, "events": events,
            "winner": self.winner, "turn": self.turn - 1,
        }  # fmt: skip
        return self.observation(), float(r[side]), self.done and not truncated, truncated, info
