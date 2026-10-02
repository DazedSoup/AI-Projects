"""Per-row SHAP attributions for the three classifiers (model-agnostic KernelExplainer).

Rows in the episode log are already scaled, so they go straight into ``Classifier.predict_proba``. The explainer
works in probability space (identity link), so ``base_value + sum(shap) == model(row)`` up to solver tolerance.

Cost control, because KernelExplainer is O(nsamples * |background|) model evaluations per row:
- background = weighted k-means summary (``shap.kmeans``) of ``clf.background(n_pool)`` rather than raw rows;
- ``nsamples`` bounded (default 2000, ~0.1 s/row on the 196-feature malware model), with an L1 ``num_features(k)`` selection on wide models so the
  regression is well posed and the top-k is stable;
- results cached in memory and on disk, keyed by (model, explainer settings, row hash).

DeepExplainer is deliberately not used (unreliable on Keras 3).
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_NSAMPLES = 2000  # measured: top-5 overlap vs nsamples=8000 is ~0.9 here, 0.57-0.72 at 500
DEFAULT_BACKGROUND_K = 50
DEFAULT_BACKGROUND_POOL = 1000
DEFAULT_TOP_K = 5
L1_FEATURES = 20  # feature-selection width for models wider than this


def row_hash(row: np.ndarray | list[float]) -> str:
    """Stable hash of a (rounded, logged) row. Rows are logged at 3 dp, so hash at that precision."""
    arr = np.round(np.asarray(row, dtype=np.float64), 4)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:24]


@dataclass
class ShapSettings:
    nsamples: int = DEFAULT_NSAMPLES
    background_k: int = DEFAULT_BACKGROUND_K
    background_pool: int = DEFAULT_BACKGROUND_POOL
    top_k: int = DEFAULT_TOP_K

    def key(self) -> str:
        return f"ns{self.nsamples}-k{self.background_k}-p{self.background_pool}"


@dataclass
class ShapStats:
    rows: int = 0
    cache_hits: int = 0
    seconds: float = 0.0
    max_additivity_err: float = 0.0
    per_model: dict[str, list[float]] = field(default_factory=dict)  # model -> seconds per computed row

    @property
    def computed(self) -> int:
        return self.rows - self.cache_hits

    def seconds_per_row(self, model: str | None = None) -> float:
        xs = self.per_model.get(model, []) if model else [s for v in self.per_model.values() for s in v]
        return float(np.mean(xs)) if xs else 0.0


class RowExplainer:
    """KernelExplainer for one classifier. ``predict`` is any (n, d) -> (n,) callable (mockable in tests)."""

    def __init__(self, name: str, predict, feature_names: list[str], background: np.ndarray,
                 settings: ShapSettings, scaler_mean=None, scaler_scale=None):  # fmt: skip
        import shap

        self.name = name
        self.predict = predict
        self.feature_names = list(feature_names)
        self.settings = settings
        self.scaler_mean = None if scaler_mean is None else np.asarray(scaler_mean, dtype=np.float64)
        self.scaler_scale = None if scaler_scale is None else np.asarray(scaler_scale, dtype=np.float64)
        background = np.asarray(background, dtype=np.float64)
        k = min(settings.background_k, len(background))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summary = shap.kmeans(background, k) if k < len(background) else background
            self._explainer = shap.KernelExplainer(self._f, summary, link="identity")
        self.base_value = float(np.ravel(self._explainer.expected_value)[0])
        m = len(self.feature_names)
        self._l1 = f"num_features({L1_FEATURES})" if m > L1_FEATURES else False

    @classmethod
    def for_classifier(cls, name: str, settings: ShapSettings) -> RowExplainer:
        from cyberarena.ml.inference import load_classifier

        clf = load_classifier(name)
        return cls(name, clf.predict_proba, clf.feature_names, clf.background(settings.background_pool),
                   settings, clf.scaler_mean, clf.scaler_scale)  # fmt: skip

    def _f(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.predict(np.asarray(X, dtype=np.float32)), dtype=np.float64).reshape(-1)

    def explain(self, row: list[float] | np.ndarray) -> dict:
        """Full attribution for one scaled row: base_value, output, per-feature shap, top-k."""
        x = np.asarray(row, dtype=np.float64).reshape(1, -1)
        seed = int(row_hash(x)[:8], 16)
        rng_state = np.random.get_state()
        np.random.seed(seed)  # KernelExplainer samples coalitions with the global RNG
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sv = self._explainer.shap_values(x, nsamples=self.settings.nsamples, l1_reg=self._l1,
                                                 silent=True)  # fmt: skip
        finally:
            np.random.set_state(rng_state)
        sv = np.asarray(sv, dtype=np.float64).reshape(-1)
        output = float(self._f(x)[0])
        return {
            "model": self.name,
            "base_value": round(self.base_value, 4),
            "output": round(output, 4),
            "additivity_error": round(abs(self.base_value + float(sv.sum()) - output), 6),
            "top_features": self.top_features(x[0], sv, self.settings.top_k),
        }

    def top_features(self, x: np.ndarray, sv: np.ndarray, k: int) -> list[dict]:
        order = np.argsort(-np.abs(sv), kind="stable")[:k]
        out = []
        for i in order:
            if sv[i] == 0.0:
                break
            item = {
                "name": self.feature_names[i],
                "value": round(float(x[i]), 3),
                "shap": round(float(sv[i]), 4),
            }
            if self.scaler_mean is not None and self.scaler_scale is not None:
                item["raw"] = round(float(x[i] * self.scaler_scale[i] + self.scaler_mean[i]), 3)
            out.append(item)
        return out


class ShapCache:
    """Append-only JSONL disk cache: one {"key": ..., "value": ...} per line."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self._mem: dict[str, dict] = {}
        if self.path and self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._mem[rec["key"]] = rec["value"]

    def get(self, key: str) -> dict | None:
        return self._mem.get(key)

    def put(self, key: str, value: dict) -> None:
        self._mem[key] = value
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, separators=(",", ":")) + "\n")

    def __len__(self) -> int:
        return len(self._mem)


class ShapService:
    """Lazily builds one RowExplainer per model and explains logged ``classifier_inputs`` entries."""

    def __init__(self, settings: ShapSettings | None = None, cache_path: Path | None = None,
                 explainers: dict[str, RowExplainer] | None = None):  # fmt: skip
        self.settings = settings or ShapSettings()
        self.cache = ShapCache(cache_path)
        self._explainers: dict[str, RowExplainer] = dict(explainers or {})
        self.stats = ShapStats()

    def explainer(self, model: str) -> RowExplainer:
        if model not in self._explainers:
            self._explainers[model] = RowExplainer.for_classifier(model, self.settings)
        return self._explainers[model]

    def cache_key(self, model: str, row) -> str:
        return f"{model}|{self.settings.key()}|{row_hash(row)}"

    def explain_input(self, ci: dict) -> dict:
        """``ci`` is one ``classifier_inputs`` entry: {"node", "model", "row", "score"}."""
        model, row = ci["model"], ci["row"]
        key = self.cache_key(model, row)
        self.stats.rows += 1
        res = self.cache.get(key)
        if res is None:
            t0 = time.perf_counter()
            res = self.explainer(model).explain(row)
            dt = time.perf_counter() - t0
            self.stats.seconds += dt
            self.stats.per_model.setdefault(model, []).append(dt)
            self.cache.put(key, res)
        else:
            self.stats.cache_hits += 1
        self.stats.max_additivity_err = max(self.stats.max_additivity_err, res.get("additivity_error", 0.0))
        return {"node": ci["node"], **res}

    def explain_turn(self, turn: dict) -> list[dict] | None:
        cis = turn.get("classifier_inputs") or []
        return [self.explain_input(ci) for ci in cis] or None
