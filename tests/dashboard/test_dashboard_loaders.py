"""Loader tests on the tiny fixture run in tests/dashboard/fixtures/runs/."""

import json
import shutil
from pathlib import Path

import pytest

from cyberarena.dashboard import charts
from cyberarena.dashboard import loaders as L

FIXTURE_RUNS = Path(__file__).parent / "fixtures" / "runs"
RUN = FIXTURE_RUNS / "20260101-000000-1"


def test_list_runs():
    assert L.list_runs(FIXTURE_RUNS) == [RUN]
    assert L.list_runs(FIXTURE_RUNS / "nope") == []


def test_graph():
    g = L.load_graph(RUN)
    assert [n["id"] for n in g["nodes"]] == [0, 1, 2]
    assert g["nodes"][2]["crown_jewel"] is True
    assert g["edges"] == [(0, 1), (1, 2)]


def test_graph_rejects_bad_edge(tmp_path):
    (tmp_path / "graph.json").write_text(json.dumps({"nodes": [{"id": 0}], "edges": [[0, 9]]}))
    with pytest.raises(L.RunFormatError):
        L.load_graph(tmp_path)


def test_summary_split_and_curve():
    eps, evals = L.load_summary(RUN)
    assert len(eps) == 4 and eps["logged"].tolist() == [True, False, False, False]
    assert len(evals) == 3
    curve = L.eval_curve(evals)
    blue = curve[curve.side == "blue"].iloc[0]
    assert blue.win_rate == 1.0 and blue.wins == 1  # learned side = blue for blue_learned_vs_red_baseline
    red = curve[(curve.side == "red")].sort_values("after_episode")
    assert red.win_rate.tolist() == [0.0, 1.0]
    assert (curve.ci_low <= curve.win_rate).all() and (curve.win_rate <= curve.ci_high).all()


def test_summary_missing(tmp_path):
    eps, evals = L.load_summary(tmp_path)
    assert eps.empty and evals.empty
    assert L.eval_curve(evals).empty
    assert L.rolling_head_to_head(eps).empty


def test_wilson():
    lo, hi = L.wilson_interval(3, 50)
    assert lo == pytest.approx(0.0206, abs=1e-3) and hi == pytest.approx(0.1622, abs=1e-3)
    assert L.wilson_interval(0, 10)[0] == 0.0
    assert L.wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert L.wilson_interval(0, 0) == (0.0, 1.0)


def test_rolling_head_to_head_uses_only_h2h():
    eps, _ = L.load_summary(RUN)
    h = L.rolling_head_to_head(eps, window=2)
    assert h["episode"].tolist() == [0, 2, 3]  # episode 1 is red_learned_vs_blue_baseline
    assert h["red_win_rate"].tolist() == [0.0, 0.5, 1.0]


def test_enriched_tolerates_missing_fields():
    enr = L.load_enriched(RUN)
    assert list(enr) == [5]
    turns = enr[5]
    assert [t["turn"] for t in turns] == list(range(6))
    assert all(t["enriched"] for t in turns)
    assert turns[1]["rationale_meta"] is None  # dropped in fixture
    assert turns[2]["mitre"] is None  # red wait
    assert L.rationale_source(turns) == "template"
    assert L.load_enriched(Path("does-not-exist")) == {}


def test_normalize_requires_core_fields():
    with pytest.raises(L.RunFormatError):
        L.normalize_turn({"episode": 1, "turn": 0, "actor": "red"})
    t = L.normalize_turn({"episode": 1, "turn": 0, "actor": "red", "action_id": "wait"})
    assert t["shap"] is None and t["enriched"] is False and t["decision_values"] == {}


def test_index_and_read_episode(tmp_path):
    idx = L.build_episode_index(RUN, cache_dir=tmp_path)
    assert idx.episodes == [0, 5, 6]
    ep5 = L.read_episode(idx, 5)
    assert len(ep5) == 6 and not any(t["enriched"] for t in ep5)
    assert ep5[-1]["winner"] == "red"
    assert L.read_episode(idx, 99) == []
    # second call hits the disk cache
    assert len(list(tmp_path.glob("episodes-index-*.json"))) == 1
    assert L.build_episode_index(RUN, cache_dir=tmp_path).segments == idx.segments


def test_index_cache_invalidates_on_change(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(RUN, run)
    cache = tmp_path / "cache"
    assert L.build_episode_index(run, cache_dir=cache).episodes == [0, 5, 6]
    with open(run / "episodes.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"episode": 42, "turn": 0, "actor": "red", "action_id": "wait"}) + "\n")
    assert L.build_episode_index(run, cache_dir=cache).episodes == [0, 5, 6, 42]


def test_index_handles_noncontiguous_episodes(tmp_path):
    lines = [
        {"episode": e, "turn": t, "actor": "red", "action_id": "wait"} for e, t in [(1, 0), (2, 0), (1, 1)]
    ]
    (tmp_path / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    idx = L.build_episode_index(tmp_path, cache_dir=tmp_path / "c")
    assert [t["turn"] for t in L.read_episode(idx, 1)] == [0, 1]


def test_catalog_enriched_first():
    eps, evals = L.load_summary(RUN)
    cat = L.episode_catalog(L.load_enriched(RUN), eps, evals)
    assert cat["episode"].tolist()[0] == 5 and bool(cat["enriched"].iloc[0])
    assert set(cat["episode"]) == {0, 5, 6, 7, 8}
    assert "red wins" in cat["label"].iloc[0]


def test_node_state_precedence():
    assert L.node_state({}) == "clean"
    assert L.node_state({"patched": True}) == "patched"
    assert L.node_state({"compromised": True, "patched": True}) == "compromised"
    assert L.node_state({"compromised": True, "detected": True}) == "detected"
    assert L.node_state({"compromised": True, "detected": True, "isolated": True}) == "isolated"


def test_move_log_and_decision_web(tmp_path):
    enr = L.load_enriched(RUN)[5]
    log = L.move_log(enr, 4)
    assert len(log) == 5
    assert log.loc[4, "target"] == "1 -> 2"
    assert log.loc[1, "target"] == "-" and log.loc[2, "mitre"] == "-"
    assert log.loc[0, "mitre"].startswith("T1021")

    web = L.decision_web(enr[0])
    assert web["chosen"] == "exploit" and web["value_kind"] == "Q-value" and web["chosen_is_best"]
    assert [a["action"] for a in web["actions"]] == ["recon", "wait"]
    assert web["features"][0]["name"] == "src_bytes" and web["features"][0]["weight"] == 1.0

    blue = L.decision_web(enr[1])
    assert blue["value_kind"] == "heuristic priority" and not blue["chosen_is_best"]
    assert blue["features"] == []

    raw = L.read_episode(L.build_episode_index(RUN, cache_dir=tmp_path), 5)
    assert L.move_log(raw, 0).loc[0, "rationale"] == "(not enriched)"


def test_figures_build():
    g = L.load_graph(RUN)
    enr = L.load_enriched(RUN)[5]
    for t in enr:
        charts.host_graph(g, t)
        charts.decision_web_figure(L.decision_web(t), t["actor"])
    eps, evals = L.load_summary(RUN)
    charts.win_rate_figure(L.eval_curve(evals), L.rolling_head_to_head(eps, 2), 2)
