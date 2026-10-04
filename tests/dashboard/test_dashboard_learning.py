"""learning.jsonl parsing, arms-race series, takeaways and the v2 loader additions (synthetic v2 fixture)."""

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from cyberarena.dashboard import charts
from cyberarena.dashboard import learning as LL
from cyberarena.dashboard import loaders as L

FIX = Path(__file__).parent / "fixtures"
V2 = FIX / "runs_v2" / "20261003-101500-7"
V1 = FIX / "runs" / "20260101-000000-1"


@pytest.fixture(scope="module")
def lr():
    return LL.load_learning(V2)


def test_v1_has_no_learning():
    assert LL.load_learning(V1) is None


def test_parse_counts(lr):
    assert lr.models == ["network", "malware", "phishing"]
    assert len(lr.updates) == 32 and not lr.evasion.empty and not lr.stats.empty and not lr.probes.empty
    assert LL.detector_metric(lr) == "recall"


def test_bad_line_raises():
    with pytest.raises(LL.LearningFormatError):
        LL.parse_learning(['{"kind": "probe"}', "{nope"])


def test_state_and_arms_race(lr):
    v, e = LL.state_at(lr, "network", 0)
    assert v == 0 and "recall_by_level" in e  # pretrained = the first update's "before"
    ar = LL.arms_race(lr, "network")
    assert set(ar["event"]) == {"evasion", "before", "after"}
    upd = ar[ar["event"] == "after"]
    assert len(upd) == (lr.updates["model"] == "network").sum()
    before = ar[ar["event"] == "before"].set_index("episode")["score"]
    after = upd.set_index("episode")["score"]
    assert (after >= before - 1e-9).mean() > 0.8  # updates mostly help at red's level


def test_landscape_and_versions(lr):
    ls = LL.landscape(lr, "network")
    assert ls["episodes"][0] == 0 and ls["versions"][0] == 0
    assert len(ls["levels"]) == 8 and len(ls["z"][0]) == len(ls["episodes"])
    vt = LL.version_table(lr, "network")
    assert vt["version"].tolist() == sorted(vt["version"]) and vt["clean_vs_pretrained"].notna().all()
    assert "Forgetting check" in LL.versions_takeaway(lr, "network")  # the fixture forgets slightly by v13


def test_probe_change_ignores_untrained_ties():
    pm = {"checkpoints": [0, 1, 2], "chosen": ["a", "b", "b"], "q": [[0, 0.2, 0.3], [0, 0.4, 0.5]]}
    c = LL.probe_change(pm)
    assert c["first"] == "b" and not c["changed"]  # checkpoint 0 is an all-zero tie, not a preference
    pm["chosen"] = ["a", "a", "b"]
    c = LL.probe_change(pm)
    assert c["first"] == "a" and c["last"] == "b" and c["changed"] and c["settled_at"] == 2


def test_sensor_story_stuck_sensor():
    rows = [{"kind": "red_evasion", "episode": e, "levels": {"phishing": 0.7}, "recall_at_level": {"phishing": 0.3}}
            for e in range(50, 1050, 50)]  # fmt: skip
    rows.append({"kind": "detector_update", "episode": 100, "model": "phishing", "version": 1, "n_malicious": 30,
                 "before": {"recall_by_level": {"0.7": 0.3}}, "after": {"recall_by_level": {"0.7": 0.32}}})  # fmt: skip
    lr = LL.parse_learning(json.dumps(r) for r in rows)
    st = LL.sensor_story(lr, "phishing")
    assert st["kind"] == "stuck" and "does not oscillate" in st["text"]


def test_takeaways_are_sentences(lr):
    for t in (LL.arms_race_takeaway(lr), LL.landscape_takeaway(lr, "network"), LL.agent_takeaway(lr),
              LL.mix_takeaway(lr, "red"), LL.probes_takeaway(lr, "blue")):  # fmt: skip
        assert t.endswith(".") and len(t) > 30


def test_figures_build(lr):
    charts.arms_race_figure(lr, lr.models)
    ls = LL.landscape(lr, "malware")
    charts.landscape_figure(ls, [(100, 0.2, 0.9)], three_d=True)
    charts.landscape_figure(ls, None, three_d=False)
    charts.agent_figure(lr.stats)
    a, e, z = LL.mix_matrix(lr, "blue")
    charts.mix_figure(a, e, z, "blue")
    charts.probe_figure(LL.probe_matrix(lr, "red", "r2"), "red")
    g = L.load_graph(V2)
    t = next(iter(L.load_enriched(V2).values()))[3]
    charts.host_network_3d(g, t)
    charts.host_network_2d(g, t, compact=True)


def test_index_meta_finds_probe_games(tmp_path):
    idx = L.build_episode_index(V2, cache_dir=tmp_path)
    probes = [ep for ep, m in idx.meta.items() if m["probe_game"]]
    assert len(probes) == 14
    m = idx.meta[min(probes)]
    assert m["turns"] > 0 and m["winner"] in ("red", "blue") and m["after_episode"] == 0
    eps, evals = L.load_summary(V2)
    cat = L.episode_catalog(L.load_enriched(V2), eps, evals, idx.meta)
    pg = L.probe_games(cat)
    assert len(pg) == 14 and set(pg["matchup"]) == set(L.EVAL_MATCHUPS)
    assert cat["label"].iloc[0].startswith("Game ") and "scripted" in cat["label"].iloc[0]
    assert set(cat["kind"]) == {"narrated", "probe", "train"}  # the two non-probe eval games are narrated


def test_run_label_and_info():
    i = L.run_info(V2)
    assert i["adaptive"] and not i["frozen"] and i["games"] == 1500
    fin = datetime(2026, 10, 3, 10, 21)  # noqa: DTZ001 - run files use naive local time
    label = L.run_label({**i, "finished": fin}, today=date(2026, 10, 3))
    assert label == "synthetic v2 fixture · 1,500 games · finished 10:21"
    assert L.run_label({**i, "finished": fin}, today=date(2026, 10, 4)).endswith("finished Oct 3, 10:21")
    live = {"run_id": "x", "label": "", "games": 300, "status": "running", "episode": 150}
    assert L.run_label(live) == "Run x · 300 games · training 50%"
    assert not L.run_info(V1)["adaptive"]


def test_adaptation_facts_and_sentences():
    turn = {"classifier_inputs": [{"node": 3, "model": "network", "evasion": 0.4, "version": 4, "score": 0.71}],
            "adaptation": {"evasion": {"network": 0.4}, "detector_versions": {"network": 4, "malware": 2},
                           "updated_since_last_move": ["network"], "caught_rate": {"network": 0.5}}}  # fmt: skip
    f = L.adaptation_facts(turn)
    assert (
        f["evasion"] == {"network": 0.4} and f["versions"]["malware"] == 2 and f["retrained"] == ["network"]
    )
    assert L.adaptation_line(turn) == "red evasion 0.4 on network · detector v4 scored 0.71"
    assert L.adaptation_facts({"classifier_inputs": []}) is None
    assert L.sentences("Red waits. Score 0.71 is high. Blue acts!") == [
        "Red waits.",
        "Score 0.71 is high.",
        "Blue acts!",
    ]


def test_checkpoint_q_states():
    q = L.checkpoint_q_states(V2)
    assert set(q["side"]) == {"red", "blue"} and q["after_episode"].min() == 0
    assert L.checkpoint_q_states(V1).empty


def _curve(rate):
    return pd.DataFrame([{"after_episode": e, "side": "blue", "n": 200, "wins": int(rate * 200), "win_rate": rate,
                          "ci_low": rate - 0.07, "ci_high": rate + 0.07} for e in range(0, 2001, 250)])  # fmt: skip


def test_adaptive_vs_frozen_takeaway():
    t = L.adaptive_vs_frozen_takeaway(_curve(0.66), _curve(0.55))
    assert "intervals don't overlap" in t and "final checkpoint alone" in t
    assert "within noise" in L.adaptive_vs_frozen_takeaway(_curve(0.56), _curve(0.55))
