"""Disjoint held-out partitions (metric / arena_train / arena_eval) and the TUANDROMD loader.

Synthetic data only: no downloads, no real artifacts touched.
"""

import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from cyberarena.ml import adaptive, train
from cyberarena.ml import datasets as ds
from cyberarena.ml.adaptive import PARTITIONS, AdaptiveDetector, assign_partitions

# tiny synthetic held-out sets are below the real-data 150-malicious-rows target
pytestmark = pytest.mark.filterwarnings("ignore:.*fewer than 150 malicious rows:UserWarning")


def _keys(X):
    return [r.tobytes() for r in np.asarray(X, dtype=np.float32)]


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """Tiny model whose held-out rows contain many exact duplicates (incl. conflicting labels)."""
    root = tmp_path_factory.mktemp("partitions")
    rng = np.random.default_rng(1)
    n = 1200
    X = pd.DataFrame({f"f{i}": rng.integers(0, 4, n).astype(float) for i in range(4)})
    y = (X["f0"] + X["f1"] + rng.normal(0, 0.7, n) > 3).astype(np.int64).to_numpy()
    ds.save_processed("phishing", ds.split_and_scale(ds.Frames(X, y)), "synthetic", root / "processed")
    train.train_one("phishing", root / "processed", root / "models", root / "metrics", seed=0)
    return root


def _det(root, **kw):
    kw.setdefault("seed", 3)
    return AdaptiveDetector.from_pretrained(
        "phishing", models_dir=root / "models", processed_dir=root / "processed", eval_per_class=40, **kw
    )


# ---------------------------------------------------------------------- assign_partitions (pure function)
def _held(seed=0, n=900):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(n, 3)).astype(np.float32)  # 27 distinct vectors -> heavy duplication
    X[: n // 3] = rng.normal(size=(n // 3, 3))  # plus unique rows
    y = rng.integers(0, 2, n)
    is_test = rng.random(n) < 0.5
    return X, y, is_test


def test_assign_partitions_deterministic_disjoint_and_duplicates_colocated():
    X, y, is_test = _held()
    part = assign_partitions(X, y, is_test, metric_per_class=60)
    np.testing.assert_array_equal(part, assign_partitions(X, y, is_test, metric_per_class=60))
    assert set(np.unique(part)) == {0, 1, 2}
    by_key: dict[bytes, set] = {}
    for k, p in zip(_keys(X), part, strict=True):
        by_key.setdefault(k, set()).add(int(p))
    assert all(len(v) == 1 for v in by_key.values())  # identical vectors never straddle partitions
    # -0.0 and +0.0 are the same vector
    X2 = X.copy()
    X2[0] = 0.0
    X2[1] = -0.0
    p2 = assign_partitions(X2, y, is_test, metric_per_class=60)
    assert p2[0] == p2[1]


def test_metric_partition_comes_from_test_groups_only():
    X, y, is_test = _held(seed=2)
    part = assign_partitions(X, y, is_test, metric_per_class=60)
    key_all_test: dict[bytes, bool] = {}
    for k, t in zip(_keys(X), is_test, strict=True):
        key_all_test[k] = key_all_test.get(k, True) and bool(t)
    assert all(key_all_test[k] for k, p in zip(_keys(X), part, strict=True) if p == 0)


# ---------------------------------------------------------------------- detector API
def test_partition_sizes_and_rows_disjoint(root):
    det = _det(root)
    sizes = det.partition_sizes()
    assert list(sizes) == list(PARTITIONS)
    assert all(set(v) == {"0", "1"} and min(v.values()) > 0 for v in sizes.values())
    assert sum(sum(v.values()) for v in sizes.values()) == len(det.clf._y_test) + len(det.clf._y_val)
    assert sizes["metric"] == {"0": len(det._eval_ben), "1": len(det._eval_mal)}

    keysets = {p: set(_keys(np.concatenate([det._pools[p][0], det._pools[p][1]]))) for p in PARTITIONS}
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            assert not keysets[a] & keysets[b], (a, b)
    # deterministic and independent of the detector seed
    assert _det(root, seed=99).partition_sizes() == sizes
    for p in PARTITIONS:
        np.testing.assert_array_equal(_det(root, seed=99)._pools[p][1], det._pools[p][1])


@pytest.mark.parametrize("partition", ["arena_train", "arena_eval"])
def test_pool_rows_respect_partition(root, partition):
    det = _det(root)
    mal_keys = set(_keys(det._pools[partition][1]))
    ben_keys = set(_keys(det._pools[partition][0]))
    mal = det.pool_rows(1, 0.0, 200, 7, partition=partition)
    ben = det.pool_rows(1, 1.0, 200, 7, partition=partition)  # s=1: pure benign partner
    assert all(k in mal_keys for k in _keys(mal))
    assert all(k in ben_keys for k in _keys(ben))
    np.testing.assert_allclose(det.pool_rows(1, 0.4, 200, 7, partition=partition), 0.6 * mal + 0.4 * ben, atol=1e-6)
    assert all(k in ben_keys for k in _keys(det.pool_rows(0, 0.5, 100, 3, partition=partition)))
    np.testing.assert_array_equal(
        det.pool_rows(1, 0.3, 50, 5, partition=partition),
        det.pool_rows(1, 0.3, 50, np.random.default_rng(5), partition=partition),
    )


def test_pool_rows_rejects_metric_and_v2_api_uses_arena_train(root):
    det = _det(root)
    for bad in ("metric", "test", None):
        with pytest.raises(ValueError, match="partition"):
            det.pool_rows(1, 0.2, 5, 0, partition=bad)
    np.testing.assert_array_equal(det.evasive_rows(1, 0.3, 40, 11), det.pool_rows(1, 0.3, 40, 11, "arena_train"))
    np.testing.assert_array_equal(det.sample(0, 40, 11), det.pool_rows(0, 0.0, 40, 11, "arena_train"))
    train_keys = set(_keys(np.concatenate(list(det._pools["arena_train"].values()))))
    assert all(k in train_keys for k in _keys(det.sample(1, 40, 2)))
    assert det._pool is det._pools["arena_train"]


def test_evaluate_uses_metric_rows_only(root, monkeypatch):
    det = _det(root)
    before = det.evaluate((0.0, 0.3, 0.7))
    seen = []
    real = det.predict_proba
    monkeypatch.setattr(det, "predict_proba", lambda X: (seen.append(np.array(X)), real(X))[1])
    assert det.evaluate((0.0, 0.3, 0.7)) == before
    allowed = set(_keys(np.concatenate([det._eval_ben, det._eval_mal])))
    for s in (0.0, 0.3, 0.7):
        allowed |= set(_keys(((1.0 - s) * det._eval_mal + s * det._eval_partner).astype(np.float32)))
    assert seen and all(k in allowed for k in _keys(np.concatenate(seen)))
    pool_keys = set(_keys(np.concatenate([det._pools[p][c] for p in ("arena_train", "arena_eval") for c in (0, 1)])))
    assert not pool_keys & set(_keys(np.concatenate(seen)))
    # metric partners are metric benign rows
    assert set(_keys(det._eval_partner)) <= set(_keys(det._eval_ben))

    # corrupting the arena pools cannot change the metric
    for p in ("arena_train", "arena_eval"):
        for c in (0, 1):
            det._pools[p][c] = np.full_like(det._pools[p][c], np.nan)
    monkeypatch.setattr(det, "predict_proba", real)
    assert det.evaluate((0.0, 0.3, 0.7)) == before


def test_shortfall_warns(root, monkeypatch):
    monkeypatch.setattr(adaptive, "MIN_MALICIOUS_PER_PARTITION", 10**6)
    with pytest.warns(UserWarning, match="malicious rows"):
        _det(root)


# ---------------------------------------------------------------------- TUANDROMD still loadable
def _tuandromd_zip(path):
    cols = ["DOWNLOAD_WITHOUT_goodwareTIFICATION", "getLastKgoodwarewnLocation", "SEND_SMS", "Label"]
    rows = [[1, 0, 1, "malware"], [0, 1, 0, "goodware"], [1, 0, 1, "malware"], [0, 0, 1, "goodware"]] * 10
    rows += [[i % 2, (i // 2) % 2, 1, "malware" if i % 3 else "goodware"] for i in range(8)]
    buf = io.StringIO()
    pd.DataFrame(rows, columns=cols).to_csv(buf, index=False)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("TUANDROMD.csv", buf.getvalue())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tuandromd_loadable_with_header_repair(tmp_path, monkeypatch):
    assert "malware_tuandromd" in ds.ALL_NAMES and "malware_tuandromd" not in ds.NAMES
    raw = tmp_path / "raw" / "malware_tuandromd"
    raw.mkdir(parents=True)
    sha = _tuandromd_zip(raw / "TUANDROMD.zip")
    spec = ds.SPECS["malware_tuandromd"]
    rf = spec.files[0]
    monkeypatch.setitem(ds.SPECS, "malware_tuandromd",
                        ds.DatasetSpec(spec.name, spec.source, (ds.RemoteFile(rf.filename, rf.url, sha, 10),)))  # fmt: skip
    monkeypatch.setattr("requests.get", lambda *a, **k: pytest.fail("no downloads in tests"))

    fr = ds.load_frames("malware_tuandromd", raw_dir=tmp_path / "raw")
    assert list(fr.X.columns) == ["DOWNLOAD_WITHOUT_NOTIFICATION", "getLastKnownLocation", "SEND_SMS"]
    assert len(fr.X) == len(fr.y) == len(pd.DataFrame(fr.X).assign(y=fr.y).drop_duplicates())
    assert set(fr.y.tolist()) == {0, 1}

    path = ds.build("malware_tuandromd", raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed")
    back = ds.load_processed("malware_tuandromd", tmp_path / "processed")
    assert path.exists() and "tuandromd" in str(back["source"])
    assert back["X_train"].shape[1] == len(back["feature_names"])
