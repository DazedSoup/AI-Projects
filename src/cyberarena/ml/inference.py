"""Inference interface for the trained classifiers (contract: docs/contracts.md, "Classifier artifacts").

    clf = load_classifier("network")
    clf.feature_names          # list[str]
    clf.predict_proba(X)       # (n,) P(malicious); X already scaled, shape (n, n_features)
    clf.background(n=100)      # (n, n_features) scaled training sample for SHAP
    clf.sample(label, n, rng)  # (n, n_features) held-out test rows of class label (0/1)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from cyberarena.ml.datasets import NAMES, PROCESSED_DIR, load_processed
from cyberarena.ml.train import MODELS_DIR

BACKGROUND_SEED = 0


class Classifier:
    def __init__(self, name: str, model, preprocess: dict, arrays: dict[str, np.ndarray]):
        self.name = name
        self.model = model
        self.feature_names: list[str] = list(preprocess["feature_names"])
        self.scaler_mean = np.asarray(preprocess["scaler_mean"], dtype=np.float64)
        self.scaler_scale = np.asarray(preprocess["scaler_scale"], dtype=np.float64)
        self.label_map: dict[str, str] = dict(preprocess["label_map"])
        self._X_train = np.asarray(arrays["X_train"], dtype=np.float32)
        self._X_test = np.asarray(arrays["X_test"], dtype=np.float32)
        self._y_test = np.asarray(arrays["y_test"]).astype(int)
        if self._X_train.shape[1] != self.n_features:
            raise ValueError(f"{name}: processed data has {self._X_train.shape[1]} features, "
                             f"preprocess.json has {self.n_features}")  # fmt: skip

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(malicious) for already-scaled rows. Accepts (n_features,) or (n, n_features); returns (n,)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"{self.name}: expected (n, {self.n_features}), got {X.shape}")
        if len(X) == 0:
            return np.zeros(0, dtype=np.float32)
        # direct call is much cheaper than model.predict for the small batches the arena sends
        return np.asarray(self.model(X, training=False)).reshape(-1).astype(np.float32)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)

    def scale(self, X_raw: np.ndarray) -> np.ndarray:
        """Apply the training scaler to rows in the (post one-hot) raw feature space."""
        return ((np.asarray(X_raw, dtype=np.float64) - self.scaler_mean) / self.scaler_scale).astype(
            np.float32
        )

    def background(self, n: int = 100) -> np.ndarray:
        """Deterministic sample of n scaled training rows (all rows if fewer)."""
        rng = np.random.default_rng(BACKGROUND_SEED)
        n = min(int(n), len(self._X_train))
        return self._X_train[rng.choice(len(self._X_train), size=n, replace=False)]

    def sample(self, label: int, n: int, rng: np.random.Generator | int | None = None) -> np.ndarray:
        """n held-out test rows of class ``label`` (0 benign / 1 malicious); with replacement if n > pool."""
        if label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {label!r}")
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        pool = np.flatnonzero(self._y_test == label)
        if len(pool) == 0:
            raise ValueError(f"{self.name}: no test rows with label {label}")
        idx = rng.choice(pool, size=int(n), replace=int(n) > len(pool))
        return self._X_test[idx].copy()


_CACHE: dict[tuple[str, str, str], Classifier] = {}


def load_classifier(name: str, models_dir: Path | str | None = None,
                    processed_dir: Path | str | None = None) -> Classifier:  # fmt: skip
    """Load (and cache) the trained classifier ``name`` in {malware, phishing, network}."""
    if name not in NAMES:
        raise KeyError(f"unknown classifier {name!r}; expected one of {NAMES}")
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    processed_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_DIR
    key = (name, str(models_dir.resolve()), str(processed_dir.resolve()))
    if key not in _CACHE:
        import keras

        model_path = models_dir / f"{name}.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} missing; run `python -m cyberarena.ml.train --model all`")
        model = keras.models.load_model(model_path)
        preprocess = json.loads((models_dir / f"{name}_preprocess.json").read_text())
        _CACHE[key] = Classifier(name, model, preprocess, load_processed(name, processed_dir))
    return _CACHE[key]
