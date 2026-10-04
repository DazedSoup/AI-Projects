from __future__ import annotations

import json
from pathlib import Path

import pytest
from explain_fixtures import GRAPH, make_turn


@pytest.fixture
def tiny_run(tmp_path: Path) -> Path:
    """A synthetic run dir: eval episodes for both learned-vs-baseline matchups with red and blue wins."""
    run = tmp_path / "20990101-000000-1"
    run.mkdir()
    (run / "graph.json").write_text(json.dumps(GRAPH))
    (run / "config.json").write_text(json.dumps({"env": {"max_rounds": 4, "detect_threshold": 0.5}}))
    lines = []
    plan = {  # episode -> (matchup, winner, n_turns)
        200: ("red_learned_vs_blue_baseline", "red", 5), 201: ("red_learned_vs_blue_baseline", "blue", 8),
        202: ("red_learned_vs_blue_baseline", "blue", 3), 203: ("red_learned_vs_blue_baseline", "red", 7),
        204: ("blue_learned_vs_red_baseline", "red", 6), 205: ("blue_learned_vs_red_baseline", "blue", 8),
        206: ("blue_learned_vs_red_baseline", "blue", 8),
    }  # fmt: skip
    for ep, (mu, winner, n) in plan.items():
        for t in range(n):
            actor = "red" if t % 2 == 0 else "blue"
            last = t == n - 1
            rec = make_turn(episode=ep, turn=t, actor=actor, action_id="lateral_move" if actor == "red" else "monitor",
                            source=1 if actor == "red" else None, target=2, matchup=mu, done=last,
                            winner=winner if last else None, classifier_inputs=[])  # fmt: skip
            lines.append(json.dumps(rec, separators=(",", ":")))
    # one training episode, logged, to make sure selection ignores it
    lines.insert(0, json.dumps(make_turn(episode=40, turn=0, phase="train", done=True, winner="red")))
    (run / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    return run
