"""Episode indexing/selection and the CLI on a synthetic run (offline, no SHAP, no network)."""

import json

from cyberarena.explain import enrich


def test_index_and_default_selection(tiny_run):
    index = enrich.index_episodes(tiny_run / "episodes.jsonl")
    assert index[201].turns == 8 and index[201].winner == "blue" and index[40].phase == "train"
    picked = enrich.select_default(index)
    by_mu = {}
    for m, _ in picked:
        by_mu.setdefault(m.matchup, []).append(m)
    assert set(by_mu) == set(enrich.LEARNED_VS_BASELINE)
    for mu, eps in by_mu.items():
        winners = {m.winner for m in eps}
        assert winners == {"red", "blue"}, mu
        assert all(m.phase == "eval" for m in eps)
    assert len(by_mu["red_learned_vs_blue_baseline"]) == 3


def test_episode_spec_parsing(tiny_run):
    index = enrich.index_episodes(tiny_run / "episodes.jsonl")
    assert [m.episode for m, _ in enrich.parse_episodes("200,204-205", index)] == [200, 204, 205]
    assert [m.episode for m, _ in enrich.parse_episodes("last:2", index)] == [205, 206]


def test_read_turns_by_offset(tiny_run):
    index = enrich.index_episodes(tiny_run / "episodes.jsonl")
    turns = enrich.read_turns(tiny_run / "episodes.jsonl", index[203].offsets)
    assert [t["turn"] for t in turns] == list(range(7)) and {t["episode"] for t in turns} == {203}


def test_cli_offline_writes_enriched_episodes(tiny_run):
    rc = enrich.main(["--run", str(tiny_run), "--offline", "--no-shap", "--episodes", "200,205"])
    assert rc == 0
    recs = [json.loads(line) for line in (tiny_run / "episodes_enriched.jsonl").read_text().splitlines()]
    assert [r["episode"] for r in recs] == [200] * 5 + [205] * 8
    for r in recs:
        assert r["rationale"] and r["rationale_meta"] == {"source": "template"}
        if r["actor"] == "red":
            assert r["mitre"]["technique_id"] == "T1021"
        else:
            assert r["mitre"]["framework"] == "D3FEND"
    summary = json.loads((tiny_run / "explain_summary.json").read_text())
    assert summary["turns"] == 13 and summary["online_cost_estimate"]["total"]["calls"] == 13


def test_cli_online_without_key_fails_clearly(tiny_run, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    rc = enrich.main(["--run", str(tiny_run), "--no-shap", "--episodes", "200"])
    assert rc == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
    assert not (tiny_run / "episodes_enriched.jsonl").exists()
