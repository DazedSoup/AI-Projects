"""Train a tiny model on synthetic data and check the inference contract (no downloads)."""

import json

import numpy as np
import pandas as pd
import pytest

from cyberarena.ml import datasets as ds
from cyberarena.ml import inference, train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("ml")
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n), "c": rng.normal(0, 1, n)})
    y = (X["a"] - X["b"] > 0).astype(np.int64).to_numpy()
    arrays = ds.split_and_scale(ds.Frames(X, y))
    ds.save_processed("network", arrays, "synthetic", root / "processed")
    metrics = train.train_one(
        "network", root / "processed", root / "models", root / "metrics", seed=0
    )
    return root, metrics


def test_artifacts_match_contract(trained):
    root, metrics = trained
    assert (root / "models" / "network.keras").exists()
    pre = json.loads((root / "models" / "network_preprocess.json").read_text())
    assert pre["feature_names"] == ["a", "b", "c"]
    assert len(pre["scaler_mean"]) == len(pre["scaler_scale"]) == 3
    assert pre["label_map"] == {"0": "benign", "1": "malicious"}
    saved = json.loads((root / "metrics" / "network.json").read_text())
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix",
                "n_train", "n_test", "source"):  # fmt: skip
        assert key in saved
    assert saved["source"] == "synthetic"
    assert np.array(saved["confusion_matrix"]).sum() == saved["n_test"]
    assert metrics["roc_auc"] > 0.9  # linearly separable toy problem


def test_inference_interface(trained):
    root, _ = trained
    clf = inference.load_classifier("network", root / "models", root / "processed")
    assert clf is inference.load_classifier("network", root / "models", root / "processed")  # cached
    assert clf.feature_names == ["a", "b", "c"]

    rng = np.random.default_rng(1)
    pos, neg = clf.sample(1, 20, rng), clf.sample(0, 20, rng)
    assert pos.shape == (20, 3) and pos.dtype == np.float32
    p_pos, p_neg = clf.predict_proba(pos), clf.predict_proba(neg)
    assert p_pos.shape == (20,) and ((p_pos >= 0) & (p_pos <= 1)).all()
    assert p_pos.mean() > p_neg.mean() + 0.5

    assert clf.predict_proba(pos[0]).shape == (1,)
    with pytest.raises(ValueError):
        clf.predict_proba(np.zeros((2, 5)))

    bg = clf.background(n=50)
    assert bg.shape == (50, 3)
    np.testing.assert_array_equal(bg, clf.background(n=50))  # deterministic
    assert clf.background(n=10_000).shape[0] < 10_000  # capped at the training-set size

    # sample draws only from the held-out test split, honouring the requested label
    test_rows = {r.tobytes() for r in np.load(root / "processed" / "network.npz")["X_test"]}
    assert all(r.tobytes() in test_rows for r in pos)
    big = clf.sample(0, 1000, np.random.default_rng(2))  # more than the pool -> with replacement
    assert big.shape == (1000, 3)
    np.testing.assert_array_equal(clf.sample(1, 5, 42), clf.sample(1, 5, np.random.default_rng(42)))
    with pytest.raises(ValueError):
        clf.sample(2, 1, rng)


def test_unknown_classifier_name():
    with pytest.raises(KeyError):
        inference.load_classifier("ransomware")
