"""v4 learners: a Keras Q-network that scores every concrete move (docs/contracts.md, "TensorFlow agents (v4)").

``DQNAgent`` maps each candidate move's feature vector (``arena/features.py``) to a scalar Q-value with a small
MLP and picks a move epsilon-greedily over the candidates. Training is Double DQN:

- experience replay (uniform ring buffer); each stored transition keeps the chosen move's features, the
  (n-step) discounted reward, the bootstrap discount (0 when the game ended) and the next state's full
  candidate matrix;
- target ``y = G + gamma^n * Q_target(s', argmax_a' Q_online(s', a'))``: the online net picks the next move,
  a periodically synced frozen copy (the target network) values it;
- Huber loss, Adam, one ``train_on_batch`` call every ``train_every`` own decisions once the buffer holds
  ``learn_start`` transitions.

Per-step inference uses ``NumpyMLP``, a NumPy mirror of the Keras weights re-synced after every training call
(tested equal to the Keras forward pass to 1e-5), so playing a move never enters TensorFlow. The target network
is a frozen copy of the mirror's weights. TensorFlow is imported lazily, on the first network build.

A turn-based transition spans the agent's own move and the opponent's reply, as for the tabular agent.
"""

from __future__ import annotations

import itertools
import json
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cyberarena.arena.actions import ACTION_IDS
from cyberarena.arena.env import CyberArenaEnv
from cyberarena.arena.features import FEATURE_NAMES, N_FEATURES, candidate_matrix, named

ARROW = "→"  # "<action>→<target>" decision_values keys
TOP_DECISION_VALUES = 8
TOP_CANDIDATES = 12


def move_key(action_id: str, target: int | None) -> str:
    return action_id if target is None else f"{action_id}{ARROW}{target}"


def parse_hidden(text: str | list | tuple) -> tuple[int, ...]:
    """``"64,64"`` -> ``(64, 64)``; 1-3 layers of 4-512 units."""
    if isinstance(text, (list, tuple)):
        parts = [int(x) for x in text]
    else:
        parts = [int(p) for p in str(text).replace(" ", "").split(",") if p]
    if not 1 <= len(parts) <= 3 or any(not 4 <= p <= 512 for p in parts):
        raise ValueError(f"hidden layers must be 1-3 comma-separated sizes in [4, 512], got {text!r}")
    return tuple(parts)


# --------------------------------------------------------------------------------------------- networks


class NumpyMLP:
    """ReLU MLP forward pass over Keras-ordered weights ``[W0, b0, W1, b1, ..., Wout, bout]``."""

    def __init__(self, weights: list[np.ndarray]):
        self.set(weights)

    def set(self, weights: list[np.ndarray]) -> None:
        self.w = [np.array(w, dtype=np.float32) for w in weights]

    def copy(self) -> NumpyMLP:
        return NumpyMLP(self.w)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        h = np.asarray(X, dtype=np.float32)
        last = len(self.w) - 2
        for i in range(0, len(self.w), 2):
            h = h @ self.w[i] + self.w[i + 1]
            if i < last:
                np.maximum(h, 0.0, out=h)
        return h[:, 0]


def glorot_weights(sizes: list[int], seed: int) -> list[np.ndarray]:
    """Glorot-uniform kernels, zero biases (Keras' Dense defaults), from a NumPy seed."""
    rng = np.random.default_rng(seed)
    out = []
    for fan_in, fan_out in itertools.pairwise(sizes):
        lim = np.sqrt(6.0 / (fan_in + fan_out))
        out += [rng.uniform(-lim, lim, size=(fan_in, fan_out)).astype(np.float32), np.zeros(fan_out, np.float32)]
    return out


def build_keras(n_in: int, hidden: tuple[int, ...], lr: float):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import keras

    layers = [keras.Input((n_in,))]
    layers += [keras.layers.Dense(h, activation="relu", name=f"hidden_{i}") for i, h in enumerate(hidden)]
    layers += [keras.layers.Dense(1, name="q")]
    model = keras.Sequential(layers, name="qnet")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=10.0),
                  loss=keras.losses.Huber(delta=1.0))  # fmt: skip
    return model


class QNetwork:
    """Keras model (trained) + NumPy mirror (inference) + target copy."""

    def __init__(self, n_in: int, hidden: tuple[int, ...], lr: float, seed: int,
                 weights: list[np.ndarray] | None = None):  # fmt: skip
        self.n_in, self.hidden, self.lr = n_in, tuple(hidden), float(lr)
        self.model = build_keras(n_in, self.hidden, self.lr)
        self.model.set_weights(weights if weights is not None else glorot_weights([n_in, *self.hidden, 1], seed))
        self.online = NumpyMLP(self.model.get_weights())
        self.target = self.online.copy()

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        loss = self.model.train_on_batch(X, y.reshape(-1, 1).astype(np.float32))
        self.online.set(self.model.get_weights())
        return float(np.asarray(loss).reshape(-1)[0])

    def sync_target(self) -> None:
        self.target = self.online.copy()

    def keras_q(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model(np.asarray(X, dtype=np.float32), training=False)).reshape(-1)


# --------------------------------------------------------------------------------------------- agent


@dataclass
class DQNConfig:
    lr: float = 1e-3
    hidden: tuple[int, ...] = (64, 64)
    gamma: float = 0.95
    n_step: int = 3
    replay_size: int = 10_000
    batch_size: int = 64
    target_sync: int = 250  # gradient steps between target-network syncs
    train_every: int = 4  # own decisions per gradient step
    learn_start: int = 500  # transitions in the buffer before training starts

    def to_dict(self) -> dict[str, Any]:
        return {"lr": self.lr, "hidden": list(self.hidden), "gamma": self.gamma, "n_step": self.n_step,
                "replay_size": self.replay_size, "batch_size": self.batch_size, "target_sync": self.target_sync,
                "train_every": self.train_every, "learn_start": self.learn_start}  # fmt: skip


class Replay:
    """Uniform ring buffer of ``(x, G, discount, next candidate matrix)``."""

    def __init__(self, size: int, n_features: int):
        self.size = int(size)
        self.x = np.zeros((self.size, n_features), dtype=np.float32)
        self.g = np.zeros(self.size, dtype=np.float32)
        self.disc = np.zeros(self.size, dtype=np.float32)
        self.nxt: list[np.ndarray | None] = [None] * self.size
        self.n = 0
        self.pos = 0
        self.added = 0

    def __len__(self) -> int:
        return self.n

    def add(self, x: np.ndarray, g: float, disc: float, nxt: np.ndarray | None) -> None:
        i = self.pos
        self.x[i], self.g[i], self.disc[i], self.nxt[i] = x, g, disc, nxt
        self.pos = (i + 1) % self.size
        self.n = min(self.n + 1, self.size)
        self.added += 1


@dataclass
class _Pending:
    x: np.ndarray
    r: float = 0.0


@dataclass
class _Choice:
    """What a DQN decision needs for logging (built lazily: most turns are not logged)."""

    side: str
    moves: list
    X: np.ndarray
    q: np.ndarray
    idx: int
    extra: dict = field(default_factory=dict)

    def log_fields(self) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, float]]:
        ids = ACTION_IDS[self.side]
        order = np.argsort(-self.q, kind="stable")
        dv: dict[str, float] = {}
        for i in order:
            key = move_key(ids[self.moves[i][0]], self.moves[i][2])
            if key not in dv:
                dv[key] = round(float(self.q[i]), 4)
                if len(dv) == TOP_DECISION_VALUES:
                    break
        cands = [{"action": ids[self.moves[i][0]], "source": self.moves[i][1], "target": self.moves[i][2],
                  "q": round(float(self.q[i]), 4)} for i in order[:TOP_CANDIDATES]]  # fmt: skip
        return dv, cands, named(self.side, self.X[self.idx])


class DQNAgent:
    """Double-DQN learner over concrete moves. Same driver hooks as ``QAgent`` (``pre_step`` / ``post_step`` /
    ``add_reward`` / ``begin_episode`` / ``end_episode``)."""

    kind = "learned"
    learns = True
    agent_type = "dqn"

    def __init__(self, side: str, seed: int | None = None, cfg: DQNConfig | None = None, epsilon: float = 1.0,
                 weights: list[np.ndarray] | None = None):  # fmt: skip
        self.side = side
        self.ids = ACTION_IDS[side]
        self.cfg = cfg or DQNConfig()
        self.seed = 0 if seed is None else int(seed)
        self.rng = np.random.default_rng(seed)
        self.replay_rng = np.random.default_rng([self.seed, 31])
        self.epsilon = float(epsilon)
        self.alpha = None  # tabular-only knob; the trainer sets it on every learner
        self.gamma = self.cfg.gamma
        self.n_features = N_FEATURES[side]
        self.feature_names = FEATURE_NAMES[side]
        self.net = QNetwork(self.n_features, self.cfg.hidden, self.cfg.lr, self.seed % 2**31, weights)
        self.buffer = Replay(self.cfg.replay_size, self.n_features)
        self.grad_steps = 0
        self.decisions = 0
        self._traj: deque[_Pending] = deque()
        self._cache: tuple[Any, int, list, np.ndarray] | None = None
        self.reset_stats()

    # -- candidates and acting -------------------------------------------------------------------------

    def candidates(self, env: CyberArenaEnv) -> tuple[list, np.ndarray]:
        c = self._cache
        if c is not None and c[0] is env and c[1] == env.turn:
            self._cache = None
            return c[2], c[3]
        moves = env.all_candidates(self.side)
        return moves, candidate_matrix(env, self.side, moves)

    def q_values(self, env: CyberArenaEnv) -> tuple[list, np.ndarray, np.ndarray]:
        moves, X = self.candidates(env)
        return moves, X, self.net.online(X)

    def act(self, env: CyberArenaEnv, explore: bool = True):
        from cyberarena.arena.agents import Decision

        moves, X, q = self.q_values(env)
        eps = self.epsilon if explore else 0.0
        explored = bool(explore and self.rng.random() < eps)
        if explored:  # uniform over action types, then over that action's candidates
            acts = sorted({m[0] for m in moves})
            a = acts[int(self.rng.integers(len(acts)))]
            idxs = [i for i, m in enumerate(moves) if m[0] == a]
            idx = idxs[int(self.rng.integers(len(idxs)))]
        else:
            idx = int(np.argmax(q))
        a, src, tgt = moves[idx]
        self._last = (X[idx], float(q[idx]))
        return Decision(a, src, tgt, self.ids[a], {}, eps, explored,
                        choice=_Choice(self.side, moves, X, q, idx))  # fmt: skip

    # -- learning hooks ----------------------------------------------------------------------------------

    def begin_episode(self) -> None:
        self._traj.clear()
        self._cache = None

    def pre_step(self, env: CyberArenaEnv) -> None:
        """Own turn starts: the next-state candidates close the oldest pending n-step transition."""
        if len(self._traj) >= self.cfg.n_step:
            moves = env.all_candidates(self.side)
            X = candidate_matrix(env, self.side, moves)
            self._cache = (env, env.turn, moves, X)
            self._commit(X)

    def post_step(self, decision) -> None:
        x, q = self._last
        self._traj.append(_Pending(x))
        self.action_counts[decision.action] += 1
        self.q_sum += q
        self.q_abs_sum += abs(q)
        self.q_n += 1
        self.decisions += 1
        if len(self.buffer) >= self.cfg.learn_start and self.decisions % self.cfg.train_every == 0:
            self.learn()

    def add_reward(self, r: float) -> None:
        if self._traj:
            self._traj[-1].r += r

    def end_episode(self) -> None:
        while self._traj:
            self._commit(None)

    def replay(self, n: int) -> None:  # tabular-only knob (``--replay``); the DQN trains every few decisions
        return None

    def _commit(self, nxt: np.ndarray | None) -> None:
        g, k = 0.0, 0
        for k, p in enumerate(self._traj):
            g += (self.gamma ** k) * p.r
        first = self._traj.popleft()
        disc = 0.0 if nxt is None else self.gamma ** (k + 1)
        self.buffer.add(first.x, g, disc, nxt)

    def targets(self, idx: np.ndarray) -> np.ndarray:
        """Double-DQN targets for buffer rows ``idx``."""
        b = self.buffer
        y = b.g[idx].astype(np.float64)
        boot = [j for j, i in enumerate(idx) if b.nxt[i] is not None and b.disc[i] > 0]
        if boot:
            mats = [b.nxt[idx[j]] for j in boot]
            lens = np.array([len(m) for m in mats])
            allX = np.concatenate(mats)
            q_on = self.net.online(allX)
            q_tg = self.net.target(allX)
            starts = np.r_[0, np.cumsum(lens)[:-1]]
            for j, s0, ln in zip(boot, starts, lens, strict=True):
                a = s0 + int(np.argmax(q_on[s0:s0 + ln]))
                y[j] += b.disc[idx[j]] * float(q_tg[a])
        return y

    def learn(self) -> float:
        idx = self.replay_rng.integers(0, len(self.buffer), size=self.cfg.batch_size)
        y = self.targets(idx)
        X = self.buffer.x[idx]
        self.td_abs_sum += float(np.abs(y - self.net.online(X)).mean())
        loss = self.net.train(X, y)
        self.loss_sum += loss
        self.td_n += 1
        self.grad_steps += 1
        if self.grad_steps % self.cfg.target_sync == 0:
            self.net.sync_target()
        return loss

    # -- stats ---------------------------------------------------------------------------------------------

    def reset_stats(self) -> None:
        self.td_abs_sum = 0.0
        self.loss_sum = 0.0
        self.td_n = 0
        self.q_sum = 0.0
        self.q_abs_sum = 0.0
        self.q_n = 0
        self.action_counts = np.zeros(len(self.ids), dtype=np.int64)

    def count_action(self, a: int) -> None:  # post_step already counts
        return None

    def stats(self) -> dict[str, Any]:
        n_act = int(self.action_counts.sum())
        r4 = lambda x: round(float(x), 4)
        return {"n_states": None,
                "mean_abs_q": r4(self.q_abs_sum / self.q_n) if self.q_n else 0.0,
                "td_error": r4(self.td_abs_sum / self.td_n) if self.td_n else None,
                "n_updates": self.td_n,
                "epsilon": r4(self.epsilon),
                "action_mix": {self.ids[i]: r4(int(c) / n_act) for i, c in enumerate(self.action_counts) if c}
                if n_act else {},
                "loss": r4(self.loss_sum / self.td_n) if self.td_n else None,
                "q_mean": r4(self.q_sum / self.q_n) if self.q_n else None,
                "replay_size": len(self.buffer),
                "grad_steps": self.grad_steps}  # fmt: skip

    # -- persistence ---------------------------------------------------------------------------------------

    def meta(self) -> dict[str, Any]:
        return {"side": self.side, "kind": "dqn", "actions": list(self.ids), "feature_names": list(self.feature_names),
                "n_features": self.n_features, "hyperparams": self.cfg.to_dict(), "epsilon": round(self.epsilon, 6),
                "grad_steps": self.grad_steps, "decisions": self.decisions, "transitions": self.buffer.added,
                "double_dqn": True, "loss": "huber(delta=1)", "optimizer": "adam(clipnorm=10)",
                "inference": "numpy mirror of the keras weights, synced after every train_on_batch"}  # fmt: skip

    def save(self, base: Path | str) -> Path:
        """``<base>.keras`` + ``<base>.json`` (``base`` like ``agents/red_qnet``)."""
        base = Path(base)
        base.parent.mkdir(parents=True, exist_ok=True)
        self.net.model.save(str(base.with_suffix(".keras")))
        base.with_suffix(".json").write_text(json.dumps(self.meta(), indent=1))
        return base.with_suffix(".keras")

    @staticmethod
    def read_meta(base: Path | str) -> dict[str, Any]:
        return json.loads(Path(base).with_suffix(".json").read_text())

    @staticmethod
    def load_weights(base: Path | str) -> list[np.ndarray]:
        import keras

        model = keras.models.load_model(str(Path(base).with_suffix(".keras")), compile=False)
        return [np.asarray(w) for w in model.get_weights()]

    @classmethod
    def load(cls, base: Path | str, seed: int | None = None, cfg: DQNConfig | None = None) -> DQNAgent:
        meta = cls.read_meta(base)
        if cfg is None:
            h = meta["hyperparams"]
            cfg = DQNConfig(**{**h, "hidden": tuple(h["hidden"])})
        agent = cls(meta["side"], seed=seed, cfg=cfg, epsilon=meta.get("epsilon", 1.0),
                    weights=cls.load_weights(base))  # fmt: skip
        agent.grad_steps = int(meta.get("grad_steps", 0))
        return agent
