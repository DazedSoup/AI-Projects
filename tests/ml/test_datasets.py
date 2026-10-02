"""Preprocessing tests on tiny synthetic data (no downloads)."""

import numpy as np
import pandas as pd
import pytest

from cyberarena.ml import datasets as ds


def _synthetic(n=400, seed=0, dup_frac=0.0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": rng.normal(5.0, 2.0, n),
            "b": rng.integers(0, 2, n).astype(float),
            "const": np.ones(n),
        }
    )
    y = (X["a"] + rng.normal(0, 1, n) > 5).astype(np.int64).to_numpy()
    if dup_frac:
        k = int(n * dup_frac)
        X = pd.concat([X, X.iloc[:k]], ignore_index=True)
        y = np.concatenate([y, y[:k]])
    return X, y


def test_split_shapes_and_constant_column_dropped():
    X, y = _synthetic()
    out = ds.split_and_scale(ds.Frames(X, y), seed=1)
    n = sum(len(out[k]) for k in ("y_train", "y_val", "y_test"))
    assert n == len(y)
    assert list(out["feature_names"]) == ["a", "b"]
    for k in ("X_train", "X_val", "X_test"):
        assert out[k].dtype == np.float32 and out[k].shape[1] == 2
    assert 0.1 < len(out["y_test"]) / n < 0.2


def test_scaler_fit_on_train_only():
    X, y = _synthetic()
    out = ds.split_and_scale(ds.Frames(X, y), seed=1)
    np.testing.assert_allclose(out["X_train"].mean(0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out["X_train"].std(0), 1.0, atol=1e-4)
    # test set is transformed with train stats, so it is not exactly standardised
    assert np.abs(out["X_test"].mean(0)).max() > 1e-4
    # scaler params reproduce the transform
    raw_a_mean = out["scaler_mean"][0]
    assert 3.0 < raw_a_mean < 7.0 and out["scaler_scale"][0] > 0


def test_split_is_deterministic_and_stratified():
    X, y = _synthetic()
    a = ds.split_and_scale(ds.Frames(X, y), seed=7)
    b = ds.split_and_scale(ds.Frames(X, y), seed=7)
    for k in ("X_train", "X_test", "y_val"):
        np.testing.assert_array_equal(a[k], b[k])
    assert abs(a["y_train"].mean() - a["y_test"].mean()) < 0.1


def test_duplicate_rows_never_cross_splits():
    X, y = _synthetic(dup_frac=0.5)
    out = ds.split_and_scale(ds.Frames(X, y), seed=3)
    train_rows = {r.tobytes() for r in out["X_train"]}
    assert not any(r.tobytes() in train_rows for r in out["X_test"])
    assert not any(r.tobytes() in train_rows for r in out["X_val"])


def test_one_hot_uses_train_vocabulary_only():
    train = pd.DataFrame({"proto": ["tcp", "udp", "tcp"], "x": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"proto": ["icmp", "tcp"], "x": [4.0, 5.0]})
    enc_train, (enc_test,) = ds.one_hot(train, [test], ("proto",))
    assert list(enc_test.columns) == list(enc_train.columns)
    assert "proto=icmp" not in enc_train.columns
    assert enc_test.iloc[0][["proto=tcp", "proto=udp"]].sum() == 0  # unseen category -> all zeros


def test_clean_phishing_label_mapping_and_dedup():
    df = pd.DataFrame({"f1": [1, -1, 0, 1], "f2": [1, 1, -1, 1], "result": [-1, 1, 1, -1]})
    fr = ds.clean_phishing(df)
    assert len(fr.X) == 3  # last row is an exact duplicate
    assert fr.y.tolist() == [1, 0, 0]  # -1 = phishing = malicious


def test_clean_malware_labels():
    df = pd.DataFrame({"P1": [1, 0, 1], "P2": [0, 0, 1], "Label": ["malware", "goodware", np.nan]})
    fr = ds.clean_malware(df)
    assert fr.y.tolist() == [1, 0] and list(fr.X.columns) == ["P1", "P2"]


def test_clean_network_drops_difficulty_and_binarises():
    def rows(labels):
        out = []
        for i, lab in enumerate(labels):
            r = [0] * len(ds.NSL_KDD_COLUMNS)
            r[1], r[2], r[3] = "tcp", "http", "SF"
            r[4] = i
            r[-2], r[-1] = lab, 21
            out.append(r)
        return pd.DataFrame(out)

    fr = ds.clean_network(rows(["normal", "neptune", "smurf"]), rows(["normal", "satan"]))
    assert "difficulty" not in fr.X.columns and "label" not in fr.X.columns
    assert fr.y.tolist() == [0, 1, 1] and fr.y_test.tolist() == [0, 1]
    assert fr.categorical == ds.NSL_KDD_CATEGORICAL


def test_verify_rejects_bad_checksum(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"x" * 100)
    with pytest.raises(ValueError, match="sha256"):
        ds._verify(p, ds.RemoteFile("f.bin", "http://unused", "0" * 64, 10))
    with pytest.raises(ValueError, match="bytes"):
        ds._verify(p, ds.RemoteFile("f.bin", "http://unused", "0" * 64, 1000))


def test_save_and_load_roundtrip(tmp_path):
    X, y = _synthetic(n=100)
    arrays = ds.split_and_scale(ds.Frames(X, y))
    ds.save_processed("toy", arrays, "synthetic", tmp_path)
    back = ds.load_processed("toy", tmp_path)
    assert str(back["source"]) == "synthetic"
    assert back["feature_names"].tolist() == ["a", "b"]
    np.testing.assert_array_equal(back["X_test"], arrays["X_test"])
