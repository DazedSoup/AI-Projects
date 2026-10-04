"""SHAP output shape and caching, against a small numpy model (no TensorFlow, no network)."""

import numpy as np
import pytest

from cyberarena.explain.shap_values import RowExplainer, ShapService, ShapSettings, row_hash

N_FEATURES = 30
W = np.linspace(-1.5, 1.5, N_FEATURES)


def logistic(X):
    return 1.0 / (1.0 + np.exp(-(np.asarray(X, dtype=np.float64) @ W)))


@pytest.fixture
def explainer():
    rng = np.random.default_rng(0)
    bg = rng.normal(size=(200, N_FEATURES))
    names = [f"f{i}" for i in range(N_FEATURES)]
    settings = ShapSettings(nsamples=300, background_k=10, top_k=5)
    return RowExplainer("toy", logistic, names, bg, settings, np.zeros(N_FEATURES), np.full(N_FEATURES, 2.0))


def test_explain_shape_and_additivity(explainer):
    row = np.round(np.random.default_rng(1).normal(size=N_FEATURES), 3)
    out = explainer.explain(row)
    assert set(out) == {"model", "base_value", "output", "additivity_error", "top_features"}
    assert len(out["top_features"]) == 5
    assert set(out["top_features"][0]) == {"name", "value", "shap", "raw"}
    mags = [abs(x["shap"]) for x in out["top_features"]]
    assert mags == sorted(mags, reverse=True)
    assert out["output"] == pytest.approx(float(logistic(row.reshape(1, -1))[0]), abs=1e-4)
    assert out["additivity_error"] < 1e-6
    top = out["top_features"][0]
    assert top["raw"] == pytest.approx(top["value"] * 2.0, abs=1e-3)


def test_explain_is_deterministic(explainer):
    row = [0.5] * N_FEATURES
    assert explainer.explain(row) == explainer.explain(row)


def test_service_caches_in_memory_and_on_disk(explainer, tmp_path):
    cache = tmp_path / "shap.jsonl"
    svc = ShapService(explainer.settings, cache, explainers={"toy": explainer})
    ci = {"node": 3, "model": "toy", "row": [0.1] * N_FEATURES, "score": 0.5}
    a = svc.explain_input(ci)
    b = svc.explain_input(ci)
    assert a == b and a["node"] == 3
    assert svc.stats.rows == 2 and svc.stats.cache_hits == 1
    svc2 = ShapService(explainer.settings, cache, explainers={})  # no explainer: must come from disk
    assert svc2.explain_input(ci) == a


def test_explain_turn_without_inputs_is_none(explainer):
    svc = ShapService(explainer.settings, None, explainers={"toy": explainer})
    assert svc.explain_turn({"classifier_inputs": []}) is None


def test_row_hash_ignores_float_noise_below_log_precision():
    assert row_hash([0.1, 0.2]) == row_hash([0.10000001, 0.2])
    assert row_hash([0.1, 0.2]) != row_hash([0.1, 0.3])


def test_cache_skips_torn_lines(explainer, tmp_path):
    cache = tmp_path / "shap.jsonl"
    svc = ShapService(explainer.settings, cache, explainers={"toy": explainer})
    ci = {"node": 1, "model": "toy", "row": [0.2] * N_FEATURES, "score": 0.5}
    a = svc.explain_input(ci)
    with cache.open("a", encoding="utf-8") as f:
        f.write('{"key": "x", "val\n\n.0}]}}\n')
    svc2 = ShapService(explainer.settings, cache, explainers={})
    assert svc2.cache.skipped == 2 and svc2.explain_input(ci) == a
