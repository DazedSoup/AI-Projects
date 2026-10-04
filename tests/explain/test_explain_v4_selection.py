"""Default enrich selection for v4 runs: regular picks, first/last probe games, one evading-red probe at 0.7."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from explain_fixtures import GRAPH, make_turn

from cyberarena.explain import enrich

MUS = (*enrich.LEARNED_VS_BASELINE, enrich.EVASIVE_MATCHUP)


@pytest.fixture
def v4_run(tmp_path: Path) -> Path:
    """Checkpoints 0 / 250 / 500: 4 regular games per learned-vs-baseline matchup, one probe game per matchup and
    one evading-red probe game per level (0.4, 0.7)."""
    run = tmp_path / "20990101-000000-4"
    run.mkdir()
    (run / "graph.json").write_text(json.dumps(GRAPH))
    (run / "config.json").write_text(json.dumps({"env": {"max_rounds": 4}, "agents": {"type": "dqn"}}))
    lines, ep = [], 1000

    def game(mu, ckpt, winner, n, probe=False, evasion=None):
        nonlocal ep
        for t in range(n):
            kw = {"evasion": evasion} if evasion is not None else {}
            actor = "red" if t % 2 == 0 else "blue"
            rec = make_turn(episode=ep, turn=t, actor=actor, action_id="exploit" if actor == "red" else "monitor",
                            source=1 if actor == "red" else None, matchup=mu,
                            after_episode=ckpt, done=t == n - 1, winner=winner if t == n - 1 else None,
                            probe_game=probe, classifier_inputs=[], **kw)  # fmt: skip
            lines.append(json.dumps(rec))
        ep += 1

    for ckpt in (0, 250, 500):
        for mu in enrich.LEARNED_VS_BASELINE:
            for winner, n in (("red", 5), ("blue", 8), ("blue", 3), ("red", 7)):
                game(mu, ckpt, winner, n)
            game(mu, ckpt, "blue", 8, probe=True)
        for level in (0.4, 0.7):
            game(enrich.EVASIVE_MATCHUP, ckpt, "blue", 8, probe=True, evasion=level)
    (run / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    return run


def test_v4_default_selection(v4_run):
    index = enrich.index_episodes(v4_run / "episodes.jsonl")
    picked = enrich.select_default(index)
    eps = [m for m, _ in picked]
    assert len(eps) == len({m.episode for m in eps}) <= enrich.MAX_DEFAULT_EPISODES
    regular = [m for m in eps if not m.probe_game]
    assert len(regular) == 6 and all(m.after_episode == 500 for m in regular)
    probes = sorted((m.after_episode, m.matchup) for m in eps if m.probe_game and not enrich.is_evasive(m))
    assert probes == sorted((c, mu) for c in (0, 500) for mu in enrich.LEARNED_VS_BASELINE)
    evasive = [(m, why) for m, why in picked if enrich.is_evasive(m)]
    assert len(evasive) == 1
    m, why = evasive[0]
    assert m.evasion == pytest.approx(0.7) and m.after_episode == 500 and "evasion 0.7" in why
    assert len(eps) == 11


def test_v2_runs_without_evasive_games_are_unchanged(v4_run):
    index = enrich.index_episodes(v4_run / "episodes.jsonl")
    index = {e: m for e, m in index.items() if m.matchup != enrich.EVASIVE_MATCHUP}
    assert not enrich.select_evasive_probe(index)
    assert len(enrich.select_default(index)) == 10


def test_selection_is_capped(v4_run, monkeypatch):
    monkeypatch.setattr(enrich, "MAX_DEFAULT_EPISODES", 8)
    picked = enrich.select_default(enrich.index_episodes(v4_run / "episodes.jsonl"))
    assert len(picked) == 8 and sum(not m.probe_game for m, _ in picked) == 6


def test_cli_writes_agent_attribution_field(v4_run):
    assert enrich.main(["--run", str(v4_run), "--no-shap"]) == 0
    recs = [json.loads(x) for x in (v4_run / "episodes_enriched.jsonl").read_text().splitlines()]
    assert recs and all("agent_attribution" in r and r["agent_attribution"] is None for r in recs)  # no candidates
    summary = json.loads((v4_run / "explain_summary.json").read_text())
    assert summary["agent_attribution"]["null_reasons"] == {"not a DQN decision": len(recs)}
    assert {"shap_per_turn", "ig_per_turn", "lock_wait"} <= set(summary["timing_s"])
