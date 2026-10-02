"""Cheap stand-in for the trained classifiers (tests and ``--stub`` smoke runs). No TensorFlow needed."""

from __future__ import annotations

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
