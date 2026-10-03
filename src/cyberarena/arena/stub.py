"""Cheap stand-in for the trained classifiers (tests and ``--stub`` smoke runs). No TensorFlow needed."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STUB_FEATURES = {"malware": 196, "phishing": 30, "network": 121}


class StubClassifier:
    def __init__(self, name: str, n_features: int | None = None):
        self.name = name
        self.feature_names = [f"{name}_f{i}" for i in range(n_features or STUB_FEATURES[name])]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        return (1.0 / (1.0 + np.exp(-4.0 * (X.mean(axis=1) - 0.5)))).astype(np.float32)

    def sample(self, label: int, n: int, rng: np.random.Generator | int | None = None) -> np.ndarray:
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        X = rng.normal(loc=float(label), scale=1.0, size=(int(n), self.n_features))
        flip = rng.random(int(n)) < 0.05  # a few hard rows so scores are not perfectly separable
        X[flip] = rng.normal(loc=1.0 - float(label), scale=1.0, size=(int(flip.sum()), self.n_features))
        return X.astype(np.float32)

    def background(self, n: int = 100) -> np.ndarray:
        return self.sample(0, n, 0)


def stub_classifiers(n_features: int | None = None) -> dict[str, StubClassifier]:
    return {m: StubClassifier(m, n_features) for m in STUB_FEATURES}


# --------------------------------------------------------------------------------------------- adaptive


@dataclass
class StubUpdateReport:
    version: int
    n_new: int
    n_replay: int
    loss_before: float
    loss_after: float
    seconds: float
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _auc(neg: np.ndarray, pos: np.ndarray) -> float:
    """Mann-Whitney AUC (ties count half)."""
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


STUB_LEVELS = tuple(round(0.1 * i, 1) for i in range(8))


class StubAdaptiveDetector:
    """Tiny fake of ``cyberarena.ml.adaptive.AdaptiveDetector`` (same public API, numpy only).

    The "model" is a logistic regression on the row mean, initialised to match ``StubClassifier``
    (``sigmoid(4 * (mean - 0.5))``). ``update`` runs a few full-batch gradient steps on the given rows plus an
    equal-weight replay of plain rows, so evasive rows it has seen get scored higher afterwards.
    """

    def __init__(self, name: str, lr: float = 1e-3, replay_frac: float = 0.5, seed: int = 7,
                 n_features: int | None = None, version: int = 0, w: float = 4.0, b: float = -2.0):  # fmt: skip
        self.clf = StubClassifier(name, n_features)
        self.name = name
        self.lr = float(lr)
        self.replay_frac = float(replay_frac)
        self.seed = int(seed)
        self.version = int(version)
        self.w, self.b = float(w), float(b)
        erng = np.random.default_rng(20_241)
        self._eval_ben = self.clf.sample(0, 64, erng)
        self._eval_mal = self.clf.sample(1, 64, erng)
        self._eval_partner = self.clf.sample(0, 64, erng)
        self.n_updates_called = 0
        self.partitions_used: list[str] = []

    @classmethod
    def from_pretrained(cls, name: str, lr: float = 1e-3, replay_frac: float = 0.5, seed: int = 7,
                        **kw) -> StubAdaptiveDetector:  # fmt: skip
        return cls(name, lr=lr, replay_frac=replay_frac, seed=seed, n_features=kw.get("n_features"))

    @property
    def feature_names(self) -> list[str]:
        return self.clf.feature_names

    @property
    def n_features(self) -> int:
        return self.clf.n_features

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        return _sigmoid(self.w * X.mean(axis=1) + self.b).astype(np.float32)

    def sample(self, label: int, n: int, rng=None) -> np.ndarray:
        return self.evasive_rows(label, 0.0, n, rng)

    def evasive_rows(self, label: int = 1, level: float = 0.0, n: int = 1, rng=None) -> np.ndarray:
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        # draw from a stream disjoint from the eval set (eval uses its own fixed generator)
        rows = self.clf.sample(label, n, rng)
        if label == 0:
            return rows
        partners = self.clf.sample(0, n, rng)
        return ((1.0 - level) * rows + level * partners).astype(np.float32)

    def pool_rows(self, label: int = 1, level: float = 0.0, n: int = 1, rng=None,
                  partition: str = "arena_train") -> np.ndarray:  # fmt: skip
        """v4 partitions. ``arena_train`` draws exactly what ``evasive_rows`` drew in v3 (stub runs stay
        reproducible); ``arena_eval`` draws from a separate stream (continuous rows, so disjoint in practice)."""
        if partition not in ("arena_train", "arena_eval"):
            raise ValueError(f"partition must be arena_train or arena_eval, got {partition!r}")
        self.partitions_used.append(partition)
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        if partition == "arena_eval":
            rng = np.random.default_rng([int(rng.integers(2**31)), 0xE7A1])
        return self.evasive_rows(label, level, n, rng)

    def partition_sizes(self) -> dict:
        return {"metric": {"0": 64, "1": 64}, "arena_train": {"0": 10**6, "1": 10**6},
                "arena_eval": {"0": 10**6, "1": 10**6}}  # fmt: skip

    def evaluate(self, level_grid=None) -> dict:
        levels = [round(float(s), 4) for s in (STUB_LEVELS if level_grid is None else level_grid)]
        pb, pm = self.predict_proba(self._eval_ben), self.predict_proba(self._eval_mal)
        out = {"version": self.version, "clean_auc": _auc(pb, pm), "clean_recall": float(np.mean(pm >= 0.5)),
               "fpr": float(np.mean(pb >= 0.5)), "auc_by_level": {}, "recall_by_level": {},
               "n_eval": {"benign": len(pb), "malicious": len(pm)}}  # fmt: skip
        for s in levels:
            ps = self.predict_proba((1 - s) * self._eval_mal + s * self._eval_partner)
            out["auc_by_level"][s] = _auc(pb, ps)
            out["recall_by_level"][s] = float(np.mean(ps >= 0.5))
        return out

    def _loss(self, m: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(_sigmoid(self.w * m + self.b), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def update(self, X: np.ndarray, y: np.ndarray) -> StubUpdateReport:
        t0 = time.perf_counter()
        self.n_updates_called += 1
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        rng = np.random.default_rng([self.seed, self.version])
        n_rep = len(X)
        rep_y = rng.integers(0, 2, n_rep)
        rep_X = np.stack([self.clf.sample(int(c), 1, rng)[0] for c in rep_y]) if n_rep else X[:0]
        m = np.r_[X.mean(axis=1), rep_X.mean(axis=1)]
        yy = np.r_[y, rep_y]
        before, lb = self.evaluate(), self._loss(m, yy)
        step = 200.0 * self.lr
        for _ in range(50):
            g = _sigmoid(self.w * m + self.b) - yy
            self.w -= step * float(np.mean(g * m))
            self.b -= step * float(np.mean(g))
        self.version += 1
        return StubUpdateReport(self.version, len(X), n_rep, lb, self._loss(m, yy), time.perf_counter() - t0,
                                before, self.evaluate())  # fmt: skip

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"name": self.name, "version": self.version, "w": self.w, "b": self.b, "lr": self.lr,
                "replay_frac": self.replay_frac, "seed": self.seed, "n_features": self.n_features, "stub": True}
        path.write_text(json.dumps(meta))  # stand-in for the .keras file
        path.with_suffix(".json").write_text(json.dumps(meta))
        return path

    @classmethod
    def load(cls, path) -> StubAdaptiveDetector:
        d = json.loads(Path(path).with_suffix(".json").read_text())
        return cls(d["name"], lr=d["lr"], replay_frac=d["replay_frac"], seed=d["seed"],
                   n_features=d["n_features"], version=d["version"], w=d["w"], b=d["b"])  # fmt: skip


def stub_detectors(seed: int = 7, lr: float = 1e-3, n_features: int | None = None) -> dict[str, StubAdaptiveDetector]:
    return {m: StubAdaptiveDetector(m, lr=lr, seed=seed + i, n_features=n_features)
            for i, m in enumerate(STUB_FEATURES)}  # fmt: skip
