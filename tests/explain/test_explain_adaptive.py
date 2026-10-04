"""Adaptive (v2) runs: version-aware SHAP, adaptation facts/rationales, probe-game selection. No network."""

import json
from pathlib import Path

import numpy as np
import pytest
from explain_fixtures import GRAPH, make_turn

from cyberarena.explain import enrich
from cyberarena.explain.adaptation import (
    AdaptationTracker,
    LearningLog,
    RedMoveHistory,
    adaptation_sentence,
)
from cyberarena.explain.enrich import make_facts
from cyberarena.explain.narrate import build_prompt, template_rationale
from cyberarena.explain.shap_values import ShapService, ShapSettings

N_F = 3
W_PRE = np.array([[1.5], [-1.0], [0.5]], dtype=np.float32)  # pretrained weights
W_V1 = np.array([[-2.0], [0.2], [3.0]], dtype=np.float32)  # "fine-tuned" weights, clearly different
SETTINGS = ShapSettings(nsamples=64, background_k=8, background_pool=40, top_k=3)


def _keras_model(w: np.ndarray):
    import keras

    m = keras.Sequential([keras.Input((N_F,)), keras.layers.Dense(1, activation="sigmoid")])
    m.layers[-1].set_weights([w, np.zeros(1, dtype=np.float32)])
    return m


def _sigmoid(X, w):
    return 1.0 / (1.0 + np.exp(-(np.asarray(X, dtype=np.float64) @ w.astype(np.float64)).ravel()))


@pytest.fixture(scope="module")
def ml_dirs(tmp_path_factory) -> tuple[Path, Path]:
    """Tiny models/ and processed/ dirs for the real ``load_classifier`` (pretrained network.keras)."""
    root = tmp_path_factory.mktemp("ml")
    models, processed = root / "models", root / "processed"
    models.mkdir()
    processed.mkdir()
    _keras_model(W_PRE).save(models / "network.keras")
    names = ["src_bytes", "count", "duration"]
    (models / "network_preprocess.json").write_text(json.dumps({
        "feature_names": names, "scaler_mean": [0.0] * N_F, "scaler_scale": [1.0] * N_F,
        "label_map": {"0": "benign", "1": "malicious"}}))  # fmt: skip
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, N_F)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    np.savez(processed / "network.npz", X_train=X, y_train=y, X_val=X[:10], y_val=y[:10], X_test=X[:20],
             y_test=y[:20], feature_names=np.array(names))  # fmt: skip
    return models, processed


def _v2_turn(episode, turn, actor="red", version=1, evasion=0.4, score=0.71, target=2, **kw):
    ci = {"node": target, "model": "network", "row": [0.4, -0.2, 0.9], "score": score,
          "evasion": evasion, "version": version}  # fmt: skip
    return make_turn(episode=episode, turn=turn, actor=actor, target=target, source=1 if actor == "red" else None,
                     action_id="lateral_move" if actor == "red" else "monitor", classifier_inputs=[ci],
                     detector_versions={"malware": 0, "phishing": 0, "network": version}, **kw)  # fmt: skip


@pytest.fixture
def v2_run(tmp_path: Path) -> Path:
    """Tiny v2 run: two checkpoints (100, 200) of eval games incl. probe games, a detector file, learning.jsonl."""
    run = tmp_path / "20990102-000000-2"
    (run / "detectors").mkdir(parents=True)
    _keras_model(W_V1).save(run / "detectors" / "network_v1.keras")
    (run / "graph.json").write_text(json.dumps(GRAPH))
    (run / "config.json").write_text(json.dumps({"label": "adaptive test", "env": {"max_rounds": 4}}))
    learning = [
        {"kind": "red_evasion", "episode": 50, "levels": {"network": 0.1, "malware": 0.0, "phishing": 0.0}},
        {"kind": "detector_update", "episode": 150, "model": "network", "version": 1, "n_new": 40,
         "n_replay": 40, "loss_before": 0.5, "loss_after": 0.3, "before": {}, "after": {}},
        {"kind": "red_evasion", "episode": 150, "levels": {"network": 0.4, "malware": 0.0, "phishing": 0.0},
         "caught_rate": {"network": 0.62, "malware": None, "phishing": None}},
        {"kind": "agent_stats", "episode": 150, "side": "red", "n_states": 3},
    ]  # fmt: skip
    (run / "learning.jsonl").write_text("\n".join(json.dumps(r) for r in learning) + "\n")
    lines = []
    ep = 300
    for ckpt, version, evasion in ((100, 0, 0.1), (200, 1, 0.4)):
        for mu in enrich.LEARNED_VS_BASELINE:
            for winner, probe in (("red", False), ("blue", False), ("red", False), ("blue", True)):
                n = 4
                for t in range(n):
                    actor = "red" if t % 2 == 0 else "blue"
                    rec = _v2_turn(ep, t, actor, version=version, evasion=evasion if actor == "red" else 0.0,
                                   matchup=mu, after_episode=ckpt, done=t == n - 1,
                                   winner=winner if t == n - 1 else None, probe_game=probe)  # fmt: skip
                    lines.append(json.dumps(rec))
                ep += 1
    (run / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    return run


# ------------------------------------------------------------------------------------------------ SHAP


def test_shap_loads_the_detector_version_that_scored_the_row(ml_dirs, v2_run):
    from cyberarena.ml.inference import load_classifier

    models, processed = ml_dirs
    calls = []

    def loader(name, model_path=None):
        calls.append(model_path)
        return load_classifier(name, models_dir=models, processed_dir=processed, model_path=model_path)

    svc = ShapService(SETTINGS, None, run_dir=v2_run, loader=loader)
    row = [0.4, -0.2, 0.9]
    v1 = svc.explain_input({"node": 2, "model": "network", "row": row, "version": 1})
    v0 = svc.explain_input({"node": 2, "model": "network", "row": row, "version": 0})
    legacy = svc.explain_input({"node": 2, "model": "network", "row": row})  # v1-run entry: no version
    assert v1["version"] == 1 and v0["version"] == 0 and legacy["version"] == 0
    assert v1["output"] == pytest.approx(_sigmoid([row], W_V1)[0], abs=1e-4)
    assert v0["output"] == pytest.approx(_sigmoid([row], W_PRE)[0], abs=1e-4)
    assert legacy == v0
    assert calls == [v2_run / "detectors" / "network_v1.keras", None]  # one load per (model, version)
    svc.explain_input({"node": 3, "model": "network", "row": [0.0, 0.1, 0.2], "version": 1})
    assert len(calls) == 2  # explainer cached per (model, version)


def test_cache_keys_separate_versions_and_keep_v1_keys():
    svc = ShapService(SETTINGS, None)
    row = [0.1, 0.2, 0.3]
    k0, k1, k4 = (svc.cache_key("network", row, v) for v in (0, 1, 4))
    assert len({k0, k1, k4}) == 3
    assert k0 == svc.cache_key("network", row) == f"network|{SETTINGS.key()}|" + k0.rsplit("|", 1)[1]


def test_missing_detector_file_skips_row_not_run(v2_run):
    logs = []
    svc = ShapService(SETTINGS, None, run_dir=v2_run, log=logs.append)
    turn = {"classifier_inputs": [{"node": 2, "model": "network", "row": [0, 0, 0], "version": 7}]}
    assert svc.explain_turn(turn) is None
    assert svc.missing == {"network_v7": 1} and "network_v7.keras missing" in logs[0]


# ------------------------------------------------------------------------------------------ adaptation


def test_v1_turn_has_no_adaptation_and_unchanged_text():
    t = make_turn()
    assert AdaptationTracker().facts(t) is None
    f = make_facts(t, None, None, GRAPH, {})
    assert f.adaptation is None
    assert "evasion" not in template_rationale(f) and "Adaptive" not in build_prompt(f)


def test_adaptation_facts_and_red_rationale(v2_run):
    history = RedMoveHistory()
    enrich.index_episodes(v2_run / "episodes.jsonl", history)
    tracker = AdaptationTracker(LearningLog.load(v2_run / "learning.jsonl"), history)
    turn = _v2_turn(299, 0, "red", version=1, evasion=0.4, score=0.71, after_episode=200,
                    matchup="red_learned_vs_blue_baseline")  # fmt: skip
    a = tracker.facts(turn)
    assert a["evasion"] == {"network": 0.4}
    assert a["detector_versions"]["network"] == 1
    assert a["updated_since_last_move"] == ["network"]  # red last touched host 2 at checkpoint 100 under v0
    assert a["previous_versions"] == {"network": 0} and a["retrained_at"] == {"network": 150}
    assert a["evasion_prev"] == {"network": {"level": 0.1, "episode": 50}}
    text = template_rationale(make_facts(turn, None, None, GRAPH, {}, a))
    assert (
        "Red blends its network activity toward normal traffic (evasion 0.40, up from 0.10 at game 50; recent caught rate 0.62)"
        in text
    )
    assert "network detector v1, retrained at game 150, still flags it at 0.71" in text
    assert "retrained since red last acted on host 2 (v0->v1)" in text
    # same host again in the same episode: no new retrain since that move
    again = tracker.facts(_v2_turn(299, 2, "red", after_episode=200))
    assert again["updated_since_last_move"] == []


def test_adaptation_text_only_when_relevant():
    tracker = AdaptationTracker()
    quiet = _v2_turn(5, 0, "red", version=0, evasion=0.0, score=0.2)
    a = tracker.facts(quiet)
    assert a is not None and a["updated_since_last_move"] == []
    assert adaptation_sentence(quiet, a) == ""
    f = make_facts(quiet, None, None, GRAPH, {}, a)
    assert "evasion" not in template_rationale(f) and "Adaptive" not in build_prompt(f)
    low = _v2_turn(6, 0, "red", version=0, evasion=0.3, score=0.2)
    s = adaptation_sentence(low, tracker.facts(low))
    assert "evasion 0.30" in s and "pretrained network detector scores it only 0.20" in s


def test_blue_rationale_and_online_prompt_mention_version():
    tracker = AdaptationTracker(LearningLog(updates={("network", 3): {"episode": 700}}))
    red = _v2_turn(9, 0, "red", version=3, evasion=0.4)
    tracker.facts(red)
    blue = _v2_turn(9, 1, "blue", version=3, evasion=0.4, score=0.81)
    a = tracker.facts(blue)
    s = adaptation_sentence(blue, a)
    assert s.startswith("Network detector v3, retrained at game 700, reads 0.81 on host 2 (above the 0.50")
    assert "evasion 0.40" in s
    prompt = build_prompt(make_facts(blue, None, None, GRAPH, {}, a))
    assert "red evasion 0.40" in prompt and "detector v3, retrained at game 700" in prompt
    assert "mention red's evasion level" in prompt


def test_learning_log_tolerates_partial_last_line(tmp_path):
    p = tmp_path / "learning.jsonl"
    p.write_text(
        json.dumps({"kind": "detector_update", "episode": 100, "model": "malware", "version": 1}) + '\n{"ki'
    )
    log = LearningLog.load(p)
    assert log.retrained_at("malware", 1) == 100 and log.retrained_at("malware", 2) is None


# ------------------------------------------------------------------------------------------- selection


def test_default_selection_adds_first_and_last_probe_games(v2_run):
    index = enrich.index_episodes(v2_run / "episodes.jsonl")
    picked = enrich.select_default(index)
    probes = [(m, why) for m, why in picked if m.probe_game]
    regular = [m for m, _ in picked if not m.probe_game]
    assert len(regular) == 6 and all(m.after_episode == 200 for m in regular)
    assert sorted((m.after_episode, m.matchup) for m, _ in probes) == sorted(
        (c, mu) for c in (100, 200) for mu in enrich.LEARNED_VS_BASELINE
    )
    assert all("probe game" in why for _, why in probes)
    assert len({m.episode for m, _ in picked}) == len(picked) == 10


def test_v1_selection_has_no_probes(tiny_run):
    picked = enrich.select_default(enrich.index_episodes(tiny_run / "episodes.jsonl"))
    assert len(picked) == 6 and not any(m.probe_game for m, _ in picked)


def test_cli_on_v2_run_offline(v2_run):
    rc = enrich.main(["--run", str(v2_run), "--no-shap"])
    assert rc == 0
    recs = [json.loads(x) for x in (v2_run / "episodes_enriched.jsonl").read_text().splitlines()]
    assert all(r["adaptation"] is not None for r in recs)
    assert any("evasion 0.40" in r["rationale"] for r in recs if r["actor"] == "red")
    summary = json.loads((v2_run / "explain_summary.json").read_text())
    assert summary["adaptation"]["adaptive"] and summary["adaptation"]["detector_updates"] == 1
    assert summary["mode"] == "offline"


def test_fine_tuned_version_alone_is_not_news_but_its_retrain_is():
    tracker = AdaptationTracker()
    plain = _v2_turn(7, 0, "red", version=5, evasion=0.0, score=0.1)
    assert adaptation_sentence(plain, tracker.facts(plain)) == ""
    history = RedMoveHistory()
    history.observe(_v2_turn(6, 0, "red", version=4, after_episode=100, phase="eval"))
    t = _v2_turn(7, 0, "red", version=5, evasion=0.0, score=0.1, after_episode=200, phase="eval")
    t["detector_versions"] = {"malware": 5, "phishing": 5, "network": 5}
    history.observe({**t, "episode": 6, "detector_versions": {"malware": 4, "phishing": 4, "network": 4},
                     "after_episode": 100})  # fmt: skip
    a = AdaptationTracker(history=history).facts(t)
    assert a["updated_since_last_move"] == ["malware", "network", "phishing"]
    s = adaptation_sentence(t, a)
    assert s.endswith("the network detector was retrained since red last acted on host 2 (v4->v5).")


def test_online_narrator_sends_adaptation_facts_to_mocked_client():
    from types import SimpleNamespace

    from cyberarena.explain.narrate import Narrator

    sent = []

    class FakeMessages:
        def create(self, **kw):
            sent.append(kw)
            return SimpleNamespace(
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                content=[SimpleNamespace(type="text", text="Red blends in at evasion 0.40.")],
            )

    narrator = Narrator(offline=False, client=SimpleNamespace(messages=FakeMessages()), rpm=0)
    tracker = AdaptationTracker(LearningLog(updates={("network", 3): {"episode": 700}}))
    t = _v2_turn(9, 0, "red", version=3, evasion=0.4)
    text, meta = narrator.narrate_one(make_facts(t, None, None, GRAPH, {}, tracker.facts(t)))
    assert meta["source"] == "claude" and text.startswith("Red blends")
    prompt = sent[0]["messages"][0]["content"]
    assert "red evasion 0.40" in prompt and "network detector v3, retrained at game 700" in prompt
