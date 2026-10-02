"""Shared builders for tests/explain (plain module: tests/ is not a package)."""


def node_states(n: int = 4, compromised=(1,), detected=(), isolated=()) -> list[dict]:
    return [
        {"id": i, "compromised": i in compromised, "detected": i in detected, "isolated": i in isolated,
         "patched": False, "privilege": 1 if i in compromised else 0,
         "scores": {"malware": 0.05, "phishing": 0.1, "network": 0.21 if i == 2 else 0.02}}
        for i in range(n)
    ]  # fmt: skip


def make_turn(episode=10, turn=4, actor="red", action_id="lateral_move", source=1, target=2, **kw) -> dict:
    rec = {
        "run_id": "test-run", "episode": episode, "turn": turn, "actor": actor, "action_id": action_id,
        "source": source, "target": target, "success": True, "reward": 0.5,
        "decision_values": {"lateral_move": 1.42, "escalate": 0.97, "wait": 0.1} if actor == "red"
        else {"monitor": 0.3, "isolate": 0.8, "wait": 0.0},
        "epsilon": 0.0, "explored": False, "node_states": node_states(),
        "classifier_inputs": [{"node": target, "model": "network", "row": [0.1, -1.3, 0.5], "score": 0.21}],
        "done": False, "winner": None, "shap": None, "rationale": None, "mitre": None,
        "phase": "eval", "matchup": "red_learned_vs_blue_baseline", "after_episode": 100, "agent": "learned",
    }  # fmt: skip
    rec.update(kw)
    return rec


GRAPH = {
    "nodes": [{"id": 0, "role": "dmz", "crown_jewel": False, "x": 0, "y": 0},
              {"id": 1, "role": "workstation", "crown_jewel": False, "x": 0, "y": 0},
              {"id": 2, "role": "server", "crown_jewel": False, "x": 0, "y": 0},
              {"id": 3, "role": "server", "crown_jewel": True, "x": 0, "y": 0}],
    "edges": [[0, 1], [1, 2], [2, 3]],
}  # fmt: skip

SHAP_ENTRY = {"node": 2, "model": "network", "base_value": 0.31, "output": 0.21, "additivity_error": 0.0,
              "top_features": [{"name": "src_bytes", "value": 0.82, "shap": -0.12, "raw": 120.0},
                               {"name": "count", "value": -0.4, "shap": 0.03, "raw": 2.0}]}  # fmt: skip
