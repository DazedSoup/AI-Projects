"""Online-adaptive detectors (contract: docs/contracts.md, "Adaptive detectors & learning telemetry (v2)").

    det = AdaptiveDetector.from_pretrained("network", lr=1e-3, replay_frac=0.5, seed=7)
    det.version                                  # 0 = the pretrained Phase 2 model
    det.predict_proba(X)                         # (n,) P(malicious) from the current version
    det.pool_rows(label, level, n, rng, partition="arena_train" | "arena_eval")
    det.partition_sizes()                        # {"metric": {"0": n, "1": n}, "arena_train": ..., "arena_eval": ...}
    det.evasive_rows(label=1, level=s, n=n, rng=rng)   # = pool_rows(..., partition="arena_train") (v2 API)
    det.update(X, y) -> UpdateReport             # bumps version
    det.evaluate(level_grid) -> dict             # clean_auc, auc_by_level, recall_by_level
    det.save(path) / AdaptiveDetector.load(path)

"Evasion" is abstract: a malicious row interpolated toward a benign row in scaled feature space,
``x = (1 - s) * x_mal + s * x_ben``. Nothing here models a real technique.

Data partitions (contract: "Disjoint row partitions (v4)"; all rows already scaled, from
``data/processed/<name>.npz``). The held-out rows (validation + test splits) are split into three disjoint
partitions with a fixed seed that does not depend on the detector seed:

* ``metric`` - only ``evaluate()`` touches it, so versions stay comparable. Drawn from *test* groups only
  (validation rows drove the pretrained model's early stopping), up to ``eval_per_class`` rows per class
  and at most a third of the held-out rows of that class.
* ``arena_train`` - training-env sensor pools; ``pool_rows(..., "arena_train")``, ``evasive_rows`` and
  ``sample`` draw from it. Revealed labels from these rows are what the arena feeds into ``update()``.
* ``arena_eval`` - evading-red eval env pools; ``pool_rows(..., "arena_eval")``. Never seen by ``update()``.

Allocation is by *group* of identical feature vectors, so duplicate rows always share a partition, and is
stratified by class; the non-metric groups are split about 50/50 by row count between the two arena
partitions. Malicious rows are only ever blended toward benign rows of the same partition.

* **replay** - the original training split, mixed into every update to limit forgetting.

The pretrained Keras model is cloned; ``artifacts/models/*.keras`` is never written.

Demo / proof CLI:  python -m cyberarena.ml.adaptive --demo [--model all] [--updates 10] [--level 0.4]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.metrics import roc_auc_score

from cyberarena.ml.datasets import NAMES
from cyberarena.ml.inference import Classifier, load_classifier
from cyberarena.ml.train import METRICS_DIR

EVASION_LEVELS: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(8))  # 0.0 .. 0.7
EVAL_PER_CLASS = 400
EVAL_SEED = 20_241  # fixed and independent of the detector seed: every detector sees the same partitions
PARTITIONS: tuple[str, ...] = ("metric", "arena_train", "arena_eval")
POOL_PARTITIONS: tuple[str, ...] = ("arena_train", "arena_eval")
MIN_MALICIOUS_PER_PARTITION = 150  # contract target; a shortfall only warns
THRESHOLD = 0.5
_PREDICT_CHUNK = 16_384
_EPS = 1e-7


@dataclass
class UpdateReport:
    version: int
    n_new: int
    n_replay: int
    loss_before: float
    loss_after: float
    seconds: float
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    train_seconds: float = 0.0  # gradient steps only (``seconds`` also includes both evaluations)

    def to_dict(self) -> dict:
        return asdict(self)


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


@contextmanager
def _seeded(seed: int, deterministic: bool):
    """Seed keras/TF for model construction (dropout seed generators draw from python ``random``), then
    restore the caller's global ``random`` / ``np.random`` state so the arena's own RNG is unaffected."""
    import keras

    py_state, np_state = random.getstate(), np.random.get_state()
    keras.utils.set_random_seed(int(seed))
    if deterministic:
        import tensorflow as tf

        tf.config.experimental.enable_op_determinism()  # cheap on CPU; process-wide
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _bce(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    p = np.clip(p.astype(np.float64), _EPS, 1 - _EPS)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.sum(w * loss) / np.sum(w))


def _auc(neg: np.ndarray, pos: np.ndarray) -> float:
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return float(roc_auc_score(y, np.r_[neg, pos]))


def _key(s: float) -> float:
    return round(float(s), 4)


def _take(order: np.ndarray, sizes: np.ndarray, target: float) -> np.ndarray:
    """Prefix of ``order`` whose cumulative ``sizes`` first reaches ``target`` (at least one group if target > 0)."""
    if target <= 0 or len(order) == 0:
        return order[:0]
    cum = np.cumsum(sizes[order])
    return order[: int(np.searchsorted(cum, target)) + 1]


def assign_partitions(X: np.ndarray, y: np.ndarray, is_test: np.ndarray, metric_per_class: int,
                      seed: int = EVAL_SEED) -> np.ndarray:  # fmt: skip
    """Partition index (into ``PARTITIONS``) per held-out row; deterministic for fixed inputs and ``seed``.

    Rows with identical feature vectors form one group and share a partition; a group's class is its max
    label. Per class, ``metric`` takes shuffled all-test groups until ``min(metric_per_class, n_rows // 3)``
    rows; the remaining groups, shuffled, fill ``arena_train`` to half their rows and the rest is ``arena_eval``.
    """
    X = np.ascontiguousarray(X, dtype=np.float32) + np.float32(0.0)  # -0.0 -> +0.0: equal vectors, equal bytes
    keys = X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).reshape(-1)
    _, inv = np.unique(keys, return_inverse=True)
    inv = inv.reshape(-1)
    n_groups = int(inv.max()) + 1 if len(inv) else 0
    sizes = np.bincount(inv, minlength=n_groups)
    g_label = np.zeros(n_groups, dtype=int)
    np.maximum.at(g_label, inv, np.asarray(y, dtype=int))
    g_test = np.ones(n_groups, dtype=bool)
    np.logical_and.at(g_test, inv, np.asarray(is_test, dtype=bool))

    rng = np.random.default_rng(seed)
    g_part = np.full(n_groups, -1, dtype=int)
    for label in (0, 1):
        groups = np.flatnonzero(g_label == label)
        n_rows = int(sizes[groups].sum())
        metric = _take(rng.permutation(groups[g_test[groups]]), sizes, min(int(metric_per_class), n_rows // 3))
        g_part[metric] = 0
        rest = rng.permutation(groups[g_part[groups] < 0])
        train = _take(rest, sizes, (n_rows - int(sizes[metric].sum())) / 2.0) if len(rest) > 1 else rest[:0]
        g_part[rest] = 2
        g_part[train] = 1
    return g_part[inv]


class AdaptiveDetector:
    def __init__(self, clf: Classifier, model, *, lr: float = 1e-3, replay_frac: float = 0.5, seed: int = 7,
                 version: int = 0, epochs: int = 3, batch_size: int = 32,
                 eval_levels: tuple[float, ...] = EVASION_LEVELS, eval_per_class: int = EVAL_PER_CLASS,
                 deterministic: bool = True, compile: bool = True,
                 models_dir: str | None = None, processed_dir: str | None = None):  # fmt: skip
        if not 0.0 <= replay_frac < 1.0:
            raise ValueError(f"replay_frac must be in [0, 1), got {replay_frac}")
        if clf._y_train is None:
            raise ValueError(f"{clf.name}: processed data lacks y_train; rebuild with cyberarena.ml.datasets")
        self.clf = clf
        self.name = clf.name
        self.model = model
        self.lr = float(lr)
        self.replay_frac = float(replay_frac)
        self.seed = int(seed)
        self.version = int(version)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.eval_levels = tuple(float(s) for s in eval_levels)
        self.eval_per_class = int(eval_per_class)
        self.deterministic = bool(deterministic)
        self._models_dir, self._processed_dir = models_dir, processed_dir
        if compile:
            import keras

            # compiled once; update() reuses the same train function (no per-update recompile)
            model.compile(optimizer=keras.optimizers.Adam(self.lr), loss="binary_crossentropy")
        self._build_partitions()

    # ------------------------------------------------------------------ construction / persistence
    @classmethod
    def from_pretrained(cls, name: str, lr: float = 1e-3, replay_frac: float = 0.5, seed: int = 7,
                        models_dir: Path | str | None = None, processed_dir: Path | str | None = None,
                        **kwargs) -> AdaptiveDetector:  # fmt: skip
        """Wrap a *copy* of the Phase 2 model ``name``; the shared cached classifier is never modified."""
        import keras

        clf = load_classifier(name, models_dir, processed_dir)
        with _seeded(seed, kwargs.get("deterministic", True)):
            model = keras.models.clone_model(clf.model)
            model.set_weights(clf.model.get_weights())
            return cls(clf, model, lr=lr, replay_frac=replay_frac, seed=seed,
                       models_dir=None if models_dir is None else str(models_dir),
                       processed_dir=None if processed_dir is None else str(processed_dir), **kwargs)  # fmt: skip

    def _meta(self) -> dict:
        return {
            "name": self.name, "version": self.version, "lr": self.lr, "replay_frac": self.replay_frac,
            "seed": self.seed, "epochs": self.epochs, "batch_size": self.batch_size,
            "eval_levels": list(self.eval_levels), "eval_per_class": self.eval_per_class,
            "deterministic": self.deterministic,
            "models_dir": self._models_dir, "processed_dir": self._processed_dir,
        }  # fmt: skip

    def save(self, path: Path | str) -> Path:
        """Write ``path`` (.keras, readable by ``load_classifier(name, model_path=path)``) + ``path.json`` meta."""
        path = Path(path)
        if path.suffix != ".keras":
            raise ValueError(f"detector path must end in .keras, got {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        path.with_suffix(".json").write_text(json.dumps(self._meta(), indent=2))
        return path

    @classmethod
    def load(cls, path: Path | str, models_dir: Path | str | None = None,
             processed_dir: Path | str | None = None) -> AdaptiveDetector:  # fmt: skip
        import keras

        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        models_dir = models_dir if models_dir is not None else meta.get("models_dir")
        processed_dir = processed_dir if processed_dir is not None else meta.get("processed_dir")
        clf = load_classifier(meta["name"], models_dir, processed_dir)
        with _seeded(meta["seed"] + meta["version"], meta.get("deterministic", True)):
            model = keras.models.load_model(path)  # restores the Adam state saved with it
            recompile = getattr(model, "optimizer", None) is None
            return cls(clf, model, lr=meta["lr"], replay_frac=meta["replay_frac"], seed=meta["seed"],
                       version=meta["version"], epochs=meta["epochs"], batch_size=meta["batch_size"],
                       eval_levels=tuple(meta["eval_levels"]), eval_per_class=meta["eval_per_class"],
                       deterministic=meta.get("deterministic", True), compile=recompile,
                       models_dir=None if models_dir is None else str(models_dir),
                       processed_dir=None if processed_dir is None else str(processed_dir))  # fmt: skip

    # ------------------------------------------------------------------ data partitions
    def _build_partitions(self) -> None:
        c = self.clf
        X_parts, y_parts, t_parts = [c._X_test], [c._y_test], [np.ones(len(c._y_test), bool)]
        if c._X_val is not None and c._y_val is not None:
            X_parts.append(c._X_val)
            y_parts.append(c._y_val)
            t_parts.append(np.zeros(len(c._y_val), bool))
        X = np.ascontiguousarray(np.concatenate(X_parts), dtype=np.float32)
        y = np.concatenate(y_parts).astype(int)
        is_test = np.concatenate(t_parts)
        part = assign_partitions(X, y, is_test, self.eval_per_class, EVAL_SEED)
        self._held_X, self._held_y, self._held_part = X, y, part  # part: index into PARTITIONS

        self._pools: dict[str, dict[int, np.ndarray]] = {}
        for pi, pname in enumerate(PARTITIONS):
            for label in (0, 1):
                if not np.any((part == pi) & (y == label)):
                    raise ValueError(f"{self.name}: partition {pname!r} has no held-out rows of class {label}")
            self._pools[pname] = {label: X[(part == pi) & (y == label)] for label in (0, 1)}
        self._pool = self._pools["arena_train"]  # v2 name, kept for callers that peeked at it
        short = {p: len(self._pools[p][1]) for p in PARTITIONS if len(self._pools[p][1]) < MIN_MALICIOUS_PER_PARTITION}
        if short:
            warnings.warn(f"{self.name}: fewer than {MIN_MALICIOUS_PER_PARTITION} malicious rows in {short}",
                          stacklevel=3)  # fmt: skip

        self._eval_ben = self._pools["metric"][0]
        self._eval_mal = self._pools["metric"][1]
        # fixed benign partner for every malicious metric row -> evasive eval rows identical across versions
        rng = np.random.default_rng([EVAL_SEED, 1])
        self._eval_partner = self._eval_ben[rng.integers(0, len(self._eval_ben), len(self._eval_mal))]
        self._eval_cache: dict[float, np.ndarray] = {}
        self._X_train = c._X_train
        self._y_train = c._y_train.astype(np.float32)

    # ------------------------------------------------------------------ inference / sampling
    @property
    def feature_names(self) -> list[str]:
        return self.clf.feature_names

    @property
    def n_features(self) -> int:
        return self.clf.n_features

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"{self.name}: expected (n, {self.n_features}), got {X.shape}")
        if len(X) == 0:
            return np.zeros(0, dtype=np.float32)
        out = [np.asarray(self.model(X[i:i + _PREDICT_CHUNK], training=False)).reshape(-1)
               for i in range(0, len(X), _PREDICT_CHUNK)]  # fmt: skip
        return np.concatenate(out).astype(np.float32)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)

    def partition_sizes(self) -> dict[str, dict[str, int]]:
        """Rows per partition and class: ``{"metric": {"0": n, "1": n}, "arena_train": ..., "arena_eval": ...}``."""
        return {p: {str(label): len(self._pools[p][label]) for label in (0, 1)} for p in PARTITIONS}

    def sample(self, label: int, n: int, rng: np.random.Generator | int | None = None) -> np.ndarray:
        """Plain rows of class ``label`` from ``arena_train`` (v2 API; never the metric rows)."""
        return self.pool_rows(label, 0.0, n, rng, partition="arena_train")

    def evasive_rows(self, label: int = 1, level: float = 0.0, n: int = 1,
                     rng: np.random.Generator | int | None = None) -> np.ndarray:  # fmt: skip
        """v2 API: ``pool_rows(label, level, n, rng, partition="arena_train")``."""
        return self.pool_rows(label, level, n, rng, partition="arena_train")

    def pool_rows(self, label: int = 1, level: float = 0.0, n: int = 1,
                  rng: np.random.Generator | int | None = None, partition: str = "arena_train") -> np.ndarray:
        """Rows from one arena partition. ``label=1``: ``(1-s)*x_mal + s*x_ben``, both drawn from ``partition``;
        ``label=0``: plain benign rows (``level`` ignored). With replacement only if ``n`` exceeds the pool."""
        if partition not in POOL_PARTITIONS:
            raise ValueError(f"partition must be one of {POOL_PARTITIONS}, got {partition!r}")
        if label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {label!r}")
        s = float(level)
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"level must be in [0, 1], got {level}")
        rng, n = _as_rng(rng), int(n)
        pools = self._pools[partition]
        pool = pools[label]
        rows = pool[rng.choice(len(pool), size=n, replace=n > len(pool))]
        if label == 0:
            return rows.copy()
        ben = pools[0]
        partners = ben[rng.choice(len(ben), size=n, replace=n > len(ben))]
        return ((1.0 - s) * rows + s * partners).astype(np.float32)

    # ------------------------------------------------------------------ evaluation
    def _eval_rows(self, s: float) -> np.ndarray:
        k = _key(s)
        if k not in self._eval_cache:
            self._eval_cache[k] = ((1.0 - k) * self._eval_mal + k * self._eval_partner).astype(np.float32)
        return self._eval_cache[k]

    def evaluate(self, level_grid=None) -> dict:
        """On the ``metric`` partition only: AUC (benign rows vs malicious rows blended at level s) and recall at 0.5, per level."""
        levels = [_key(s) for s in (self.eval_levels if level_grid is None else level_grid)]
        blocks = [self._eval_ben, self._eval_mal] + [self._eval_rows(s) for s in levels]
        p = self.predict_proba(np.concatenate(blocks))  # one forward pass
        nb, nm = len(self._eval_ben), len(self._eval_mal)
        p_ben, p_mal = p[:nb], p[nb:nb + nm]
        out = {
            "version": self.version,
            "clean_auc": _auc(p_ben, p_mal),
            "clean_recall": float(np.mean(p_mal >= THRESHOLD)),
            "fpr": float(np.mean(p_ben >= THRESHOLD)),
            "auc_by_level": {},
            "recall_by_level": {},
            "n_eval": {"benign": nb, "malicious": nm},
        }
        for i, s in enumerate(levels):
            ps = p[nb + nm * (i + 1): nb + nm * (i + 2)]
            out["auc_by_level"][s] = _auc(p_ben, ps)
            out["recall_by_level"][s] = float(np.mean(ps >= THRESHOLD))
        return out

    # ------------------------------------------------------------------ online update
    def n_replay_for(self, n_new: int) -> int:
        """Replay rows so they make up ``replay_frac`` of the mixed batch (0.5 -> as many as new rows)."""
        n = round(n_new * self.replay_frac / (1.0 - self.replay_frac))
        return min(n, len(self._X_train))

    def update(self, X: np.ndarray, y: np.ndarray, epochs: int | None = None) -> UpdateReport:
        t0 = time.perf_counter()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y = np.asarray(y).reshape(-1).astype(np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features or len(X) != len(y):
            raise ValueError(f"{self.name}: expected X (n, {self.n_features}) and y (n,), got {X.shape}, {y.shape}")
        if len(X) == 0:
            raise ValueError("update() needs at least one sample")
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("y must be 0/1")

        rng = np.random.default_rng([self.seed, self.version])  # deterministic, and reproducible after load()
        before = self.evaluate()

        # balanced weights over the new rows (mean 1), replay rows weight 1
        counts = np.bincount(y.astype(int), minlength=2).astype(np.float64)
        present = counts > 0
        cw = np.where(present, len(y) / (present.sum() * np.maximum(counts, 1)), 0.0)
        w_new = cw[y.astype(int)]
        n_replay = self.n_replay_for(len(X))
        ridx = rng.choice(len(self._X_train), size=n_replay, replace=False)
        Xm = np.concatenate([X, self._X_train[ridx]])
        ym = np.concatenate([y, self._y_train[ridx]])
        wm = np.concatenate([w_new, np.ones(n_replay)]).astype(np.float32)

        loss_before = _bce(ym, self.predict_proba(Xm), wm)
        t_train = time.perf_counter()
        bs = self.batch_size
        for _ in range(self.epochs if epochs is None else int(epochs)):
            perm = rng.permutation(len(Xm))
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                self.model.train_on_batch(Xm[b], ym[b], sample_weight=wm[b])
        train_seconds = time.perf_counter() - t_train
        loss_after = _bce(ym, self.predict_proba(Xm), wm)

        self.version += 1
        after = self.evaluate()
        return UpdateReport(version=self.version, n_new=len(X), n_replay=n_replay, loss_before=loss_before,
                            loss_after=loss_after, seconds=time.perf_counter() - t0, before=before, after=after,
                            train_seconds=train_seconds)  # fmt: skip


# ---------------------------------------------------------------------- demo / proof
def run_loop(name: str, *, replay_frac: float, updates: int = 10, level: float = 0.4, n_each: int = 64,
             lr: float = 1e-3, seed: int = 7) -> dict:  # fmt: skip
    """Simulated adaptation: each update sees ``n_each`` evasive malicious + ``n_each`` benign rows."""
    det = AdaptiveDetector.from_pretrained(name, lr=lr, replay_frac=replay_frac, seed=seed)
    rng = np.random.default_rng(seed)
    e0 = det.evaluate()
    rows = [{"version": 0, "clean_auc": e0["clean_auc"], "auc_at_level": e0["auc_by_level"][_key(level)],
             "recall_at_level": e0["recall_by_level"][_key(level)], "fpr": e0["fpr"], "seconds": 0.0,
             "train_seconds": 0.0, "loss_before": None, "loss_after": None}]  # fmt: skip
    for _ in range(updates):
        X = np.concatenate([det.evasive_rows(1, level, n_each, rng), det.evasive_rows(0, 0.0, n_each, rng)])
        y = np.r_[np.ones(n_each), np.zeros(n_each)]
        r = det.update(X, y)
        rows.append({"version": r.version, "clean_auc": r.after["clean_auc"],
                     "auc_at_level": r.after["auc_by_level"][_key(level)],
                     "recall_at_level": r.after["recall_by_level"][_key(level)], "fpr": r.after["fpr"],
                     "seconds": r.seconds, "train_seconds": r.train_seconds,
                     "loss_before": r.loss_before, "loss_after": r.loss_after})  # fmt: skip
    return {"model": name, "replay_frac": replay_frac, "lr": lr, "level": level, "rows": rows,
            "final_eval": det.evaluate((0.0, 0.3, 0.5, 0.7))}  # fmt: skip


def demo(names=NAMES, updates: int = 10, level: float = 0.4, lr: float = 1e-3) -> dict:
    out: dict = {"evasion_problem": {}, "loops": []}
    print("== Pretrained detectors under evasion (AUC; recall@0.5 in brackets)")
    print(f"{'model':9s} {'clean':>7s} " + " ".join(f"{'s=' + str(s):>14s}" for s in (0.0, 0.3, 0.5, 0.7)))
    for name in names:
        e = AdaptiveDetector.from_pretrained(name, lr=lr).evaluate((0.0, 0.3, 0.5, 0.7))
        out["evasion_problem"][name] = e
        cells = " ".join(f"{e['auc_by_level'][s]:6.3f} [{e['recall_by_level'][s]:.2f}]" for s in (0.0, 0.3, 0.5, 0.7))
        print(f"{name:9s} {e['clean_auc']:7.3f} {cells}   n_eval={e['n_eval']}")
    for name in names:
        for rf in (0.5, 0.0):
            res = run_loop(name, replay_frac=rf, updates=updates, level=level, lr=lr)
            out["loops"].append(res)
            print(f"\n== {name}: {updates} updates of 64 evasive(s={level}) + 64 benign, lr={lr}, replay_frac={rf}")
            print(f"{'ver':>3s} {'clean_auc':>9s} {'auc@s':>7s} {'recall@s':>8s} {'fpr':>6s} {'loss b->a':>13s} {'sec':>6s}")
            for r in res["rows"]:
                loss = "" if r["loss_before"] is None else f"{r['loss_before']:.3f}->{r['loss_after']:.3f}"
                print(f"{r['version']:3d} {r['clean_auc']:9.4f} {r['auc_at_level']:7.4f} {r['recall_at_level']:8.3f} "
                      f"{r['fpr']:6.3f} {loss:>13s} {r['seconds']:6.3f}")  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--demo", action="store_true", help="evasion problem + adaptation/forgetting tables")
    ap.add_argument("--model", default="all", choices=("all", *NAMES))
    ap.add_argument("--updates", type=int, default=10)
    ap.add_argument("--level", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args(argv)
    if not args.demo:
        ap.print_help()
        return
    names = NAMES if args.model == "all" else (args.model,)
    out = demo(names, args.updates, args.level, args.lr)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "adaptive_demo.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
