"""Arms-race machinery for v2 runs (contract: docs/contracts.md, "Adaptive detectors & learning telemetry (v2)").

SIMULATION ONLY. "Evasion" is interpolation between dataset rows in scaled feature space
(``x = (1-s) * x_mal + s * x_ben``), not a real technique.

- ``EvasionBandit``: red's per-sensor evasion level, an epsilon-greedy bandit over the discrete
  ``EVASION_LEVELS``. Its payoff for a game is what it actually got out of that level: progress made by the
  red actions that model watches, minus a penalty for every malicious reading the detector flagged.
  Nothing here pushes the level directly; it moves only because payoffs change.
- ``DetectorTrainer``: per-model buffer of *revealed* (row, label) pairs (labels come only from blue
  acting on a host; see ``CyberArenaEnv``), periodic ``det.update()``, pool re-scoring, saving, and the
  ``detector_update`` rows of ``learning.jsonl``.

Importing this module never imports TensorFlow.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from cyberarena.arena.env import EVASION_LEVELS, MODELS

N_LEVELS = len(EVASION_LEVELS)
BUFFER_PER_CLASS = 256  # most recent revealed rows kept per (model, class) between updates


def _clean(x: Any, nd: int = 4) -> Any:
    """JSON-safe copy: floats rounded, NaN -> None, float dict keys -> "0.3"-style strings."""
    if isinstance(x, dict):
        return {(f"{k:.1f}" if isinstance(k, float) else k): _clean(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if math.isnan(x) or math.isinf(x) else round(x, nd)
    if isinstance(x, np.integer):
        return int(x)
    return x


# --------------------------------------------------------------------------------------------- red


class EvasionBandit:
    """One epsilon-greedy bandit per sensor model over ``EVASION_LEVELS``.

    Values start at 0 for every level; ties break toward the *lowest* level, so red starts loud (s = 0).
    A level whose observed payoff turns negative (red keeps getting flagged) loses to the untried or
    better-valued levels -- that is the only thing that ever moves red's evasion.
    """

    def __init__(self, seed: int, lr: float = 0.1, explore: float = 0.1, detect_penalty: float = 0.1):
        self.rng = np.random.default_rng(seed)
        self.lr = float(lr)
        self.explore = float(explore)
        self.detect_penalty = float(detect_penalty)
        self.values = {m: np.zeros(N_LEVELS) for m in MODELS}
        self.counts = {m: np.zeros(N_LEVELS, dtype=np.int64) for m in MODELS}

    def greedy_idx(self, m: str) -> int:
        v = self.values[m]
        return int(np.flatnonzero(v >= v.max() - 1e-12)[0])

    def greedy(self) -> dict[str, int]:
        return {m: self.greedy_idx(m) for m in MODELS}

    def choose(self) -> dict[str, int]:
        out = {}
        for m in MODELS:
            if self.rng.random() < self.explore:
                out[m] = int(self.rng.integers(N_LEVELS))
            else:
                out[m] = self.greedy_idx(m)
        return out

    def payoff(self, st: dict[str, float]) -> float | None:
        """Per-action payoff of one game for one model; None when red never touched that sensor."""
        n = st["n_act"] + st["n_leak"]
        if n <= 0:
            return None
        return (st["progress"] - self.detect_penalty * st["caught"]) / max(1.0, st["n_act"])

    def update(self, chosen: dict[str, int], stats: dict[str, dict[str, float]]) -> None:
        for m in MODELS:
            p = self.payoff(stats[m])
            if p is None:
                continue
            i = chosen[m]
            self.counts[m][i] += 1
            self.values[m][i] += self.lr * (p - self.values[m][i])

    def levels(self, idx: dict[str, int] | None = None) -> dict[str, float]:
        idx = self.greedy() if idx is None else idx
        return {m: EVASION_LEVELS[i] for m, i in idx.items()}

    def to_json(self) -> dict[str, Any]:
        return {"kind": "epsilon_greedy_bandit", "levels": list(EVASION_LEVELS), "lr": self.lr,
                "explore": self.explore, "detect_penalty": self.detect_penalty,
                "greedy": self.levels(),
                "values": {m: [round(float(x), 6) for x in v] for m, v in self.values.items()},
                "counts": {m: [int(x) for x in c] for m, c in self.counts.items()}}  # fmt: skip

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json()))

    def load_state(self, path: Path) -> None:
        d = json.loads(Path(path).read_text())
        if list(d["levels"]) != list(EVASION_LEVELS):
            raise ValueError(f"{path}: evasion levels differ")
        for m in MODELS:
            self.values[m] = np.array(d["values"][m], dtype=float)
            self.counts[m] = np.array(d["counts"][m], dtype=np.int64)


# --------------------------------------------------------------------------------------------- blue


def latest_detectors(run_dir: Path) -> dict[str, Path]:
    """Highest-version ``detectors/<model>_v<k>.keras`` per model in ``run_dir`` (empty for v1 runs)."""
    out: dict[str, tuple[int, Path]] = {}
    d = Path(run_dir) / "detectors"
    if not d.is_dir():
        return {}
    for p in d.glob("*_v*.keras"):
        name, _, ver = p.stem.rpartition("_v")
        if name in MODELS and ver.isdigit() and (name not in out or int(ver) > out[name][0]):
            out[name] = (int(ver), p)
    return {m: p for m, (_, p) in out.items()}


def copy_detector(src: Path, dst_dir: Path) -> None:
    """Copy a saved detector (``.keras`` + ``.json`` meta) so this run's turn records resolve its version."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in (src, src.with_suffix(".json")):
        if f.exists():
            shutil.copy2(f, dst_dir / f.name)


class DetectorTrainer:
    """Revealed-label buffers + periodic online updates of the adaptive detectors."""

    def __init__(self, detectors: dict[str, Any], env, out_dir: Path, min_samples: int = 32):
        self.dets = detectors
        self.env = env
        self.out_dir = Path(out_dir)
        self.min_samples = int(min_samples)
        self.buf: dict[tuple[str, int], deque] = {(m, y): deque(maxlen=BUFFER_PER_CLASS)
                                                  for m in MODELS for y in (0, 1)}  # fmt: skip
        self.current_eval: dict[str, dict] = {m: d.evaluate() for m, d in detectors.items()}
        self.initial_eval = {m: _clean(e) for m, e in self.current_eval.items()}

    def add_revealed(self, revealed: list[tuple[str, int, int, int]]) -> None:
        pool = self.env.pool
        for model, label, lvl, idx in revealed:
            self.buf[(model, label)].append(pool.rows[(model, label, lvl)][idx])

    def counts(self, m: str) -> tuple[int, int]:
        return len(self.buf[(m, 0)]), len(self.buf[(m, 1)])

    def ready(self, m: str) -> bool:
        n0, n1 = self.counts(m)
        need_each = max(1, self.min_samples // 8)
        return n0 + n1 >= self.min_samples and n0 >= need_each and n1 >= need_each

    def maybe_update(self, episode: int, red_levels: dict[str, float]) -> list[dict[str, Any]]:
        rows = []
        for m in MODELS:
            if not self.ready(m):
                continue
            n0, n1 = self.counts(m)
            X = np.concatenate([np.asarray(self.buf[(m, 0)]), np.asarray(self.buf[(m, 1)])]).astype(np.float32)
            y = np.r_[np.zeros(n0), np.ones(n1)]
            det = self.dets[m]
            rep = det.update(X, y)
            self.buf[(m, 0)].clear()
            self.buf[(m, 1)].clear()
            self.env.pool.rescore(m, det)
            det.save(self.out_dir / f"{m}_v{det.version}.keras")
            self.current_eval[m] = rep.after
            rows.append(_clean({
                "kind": "detector_update", "episode": episode, "model": m, "version": rep.version,
                "n_new": rep.n_new, "n_replay": rep.n_replay, "n_malicious": n1, "n_benign": n0,
                "loss_before": rep.loss_before, "loss_after": rep.loss_after, "seconds": rep.seconds,
                "red_level": red_levels[m], "before": rep.before, "after": rep.after,
            }))  # fmt: skip
        return rows

    def recall_at(self, m: str, level: float) -> float | None:
        r = self.current_eval[m].get("recall_by_level", {})
        for k, v in r.items():
            if abs(float(k) - level) < 1e-6:
                return _clean(v)
        return None
