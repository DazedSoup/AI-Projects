"""AdaptiveDetector on a tiny synthetic problem (no downloads, no real artifacts touched)."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from cyberarena.ml import adaptive, inference, train
from cyberarena.ml import datasets as ds
from cyberarena.ml.adaptive import AdaptiveDetector

# tiny synthetic held-out sets are below the real-data 150-malicious-rows target
pytestmark = pytest.mark.filterwarnings("ignore:.*fewer than 150 malicious rows:UserWarning")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("adaptive")
    rng = np.random.default_rng(0)
    n = 800
    X = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(4)})
    y = (X["f0"] + X["f1"] > 0).astype(np.int64).to_numpy()
    ds.save_processed("network", ds.split_and_scale(ds.Frames(X, y)), "synthetic", root / "processed")
    train.train_one("network", root / "processed", root / "models", root / "metrics", seed=0)
    return root


def _det(root, **kw):
    kw.setdefault("seed", 3)
    return AdaptiveDetector.from_pretrained(
        "network", models_dir=root / "models", processed_dir=root / "processed", eval_per_class=40, **kw
    )


def _batch(det, rng, n=32, level=0.4):
    X = np.concatenate([det.evasive_rows(1, level, n, rng), det.evasive_rows(0, 0.0, n, rng)])
    return X, np.r_[np.ones(n), np.zeros(n)]


def test_update_bumps_version_and_reports(root):
    det = _det(root, replay_frac=0.5)
    assert det.version == 0
    X, y = _batch(det, np.random.default_rng(0))
    r = det.update(X, y)
    assert det.version == r.version == 1
    assert r.n_new == 64 and r.n_replay == 64
    assert r.loss_after < r.loss_before
    assert r.seconds > 0 and r.before["version"] == 0 and r.after["version"] == 1
    d = r.to_dict()
    for k in ("version", "n_new", "n_replay", "loss_before", "loss_after", "seconds", "before", "after"):
        assert k in d
    assert det.update(X, y).version == 2
    assert _det(root, replay_frac=0.0).update(X, y).n_replay == 0
    with pytest.raises(ValueError):
        det.update(np.zeros((2, 9)), np.zeros(2))
    with pytest.raises(ValueError):
        _det(root, replay_frac=1.0)


def test_original_artifact_untouched(root, tmp_path):
    path = root / "models" / "network.keras"
    before = _sha(path)
    base = inference.load_classifier("network", root / "models", root / "processed")
    w_before = [w.copy() for w in base.model.get_weights()]
    det = _det(root)
    X, y = _batch(det, np.random.default_rng(1))
    for _ in range(3):
        det.update(X, y)
    det.save(tmp_path / "network_v3.keras")
    assert _sha(path) == before
    for a, b in zip(w_before, base.model.get_weights(), strict=True):
        np.testing.assert_array_equal(a, b)  # the shared cached classifier is not the fine-tuned copy
    assert not np.allclose(det.predict_proba(X), base.predict_proba(X))


def test_save_load_round_trip_and_model_path(root, tmp_path):
    det = _det(root)
    X, y = _batch(det, np.random.default_rng(2))
    det.update(X, y)
    det.update(X, y)
    path = det.save(tmp_path / "detectors" / "network_v2.keras")
    assert path.exists() and path.with_suffix(".json").exists()

    back = AdaptiveDetector.load(path)
    assert back.version == 2 and back.name == "network" and back.lr == det.lr
    np.testing.assert_allclose(back.predict_proba(X), det.predict_proba(X), atol=1e-6)
    assert back.evaluate()["clean_auc"] == pytest.approx(det.evaluate()["clean_auc"])
    assert back.update(X, y).version == 3

    base = inference.load_classifier("network", root / "models", root / "processed")
    clf = inference.load_classifier("network", root / "models", root / "processed", model_path=path)
    assert clf is not base and clf.model_path == path
    assert clf is inference.load_classifier("network", root / "models", root / "processed", model_path=path)
    np.testing.assert_allclose(clf.predict_proba(X), det.predict_proba(X), atol=1e-6)
    assert clf.feature_names == base.feature_names
    np.testing.assert_array_equal(clf.background(20), base.background(20))
    np.testing.assert_array_equal(clf.sample(1, 5, 0), base.sample(1, 5, 0))
    with pytest.raises(FileNotFoundError):
        inference.load_classifier("network", root / "models", root / "processed", model_path=tmp_path / "nope.keras")


def test_deterministic_for_fixed_seed(root):
    preds = []
    for _ in range(2):
        det = _det(root, seed=11)
        rng = np.random.default_rng(5)
        for _ in range(2):
            det.update(*_batch(det, rng))
        preds.append((det.predict_proba(det.evasive_rows(1, 0.3, 50, 9)), det.evaluate()))
    np.testing.assert_array_equal(preds[0][0], preds[1][0])
    assert preds[0][1] == preds[1][1]


def test_evasive_rows_blend(root):
    det = _det(root)
    mal = det.evasive_rows(1, 0.0, 30, 7)
    ben = det.evasive_rows(1, 1.0, 30, 7)
    for s in (0.25, 0.6):
        np.testing.assert_allclose(det.evasive_rows(1, s, 30, 7), (1 - s) * mal + s * ben, atol=1e-6)
    np.testing.assert_array_equal(det.evasive_rows(1, 0.3, 30, 7), det.evasive_rows(1, 0.3, 30, np.random.default_rng(7)))

    pool_mal = {r.tobytes() for r in det._pools["arena_train"][1]}
    pool_ben = {r.tobytes() for r in det._pools["arena_train"][0]}
    assert all(r.tobytes() in pool_mal for r in mal)
    assert all(r.tobytes() in pool_ben for r in ben)
    plain = det.evasive_rows(0, 0.9, 25, 1)  # label 0: plain benign rows, level ignored
    assert plain.shape == (25, 4) and plain.dtype == np.float32
    assert all(r.tobytes() in pool_ben for r in plain)
    # sampling pool and eval set are disjoint
    eval_rows = {r.tobytes() for r in np.concatenate([det._eval_ben, det._eval_mal])}
    assert not eval_rows & (pool_mal | pool_ben)
    with pytest.raises(ValueError):
        det.evasive_rows(1, 1.5, 3, 0)
    with pytest.raises(ValueError):
        det.evasive_rows(2, 0.1, 3, 0)


def test_evaluate_shape_and_evasion_hurts(root):
    det = _det(root)
    e = det.evaluate((0.0, 0.5, 0.9))
    assert set(e) >= {"clean_auc", "auc_by_level", "recall_by_level"}
    assert list(e["auc_by_level"]) == [0.0, 0.5, 0.9]
    assert e["auc_by_level"][0.0] == pytest.approx(e["clean_auc"])
    assert e["clean_auc"] > 0.9
    assert e["auc_by_level"][0.9] < e["clean_auc"]
    assert e["recall_by_level"][0.9] <= e["recall_by_level"][0.0]
    assert list(det.evaluate()["auc_by_level"]) == list(adaptive.EVASION_LEVELS)
