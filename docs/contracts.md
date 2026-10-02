# cyberarena — interface contracts

The single source of truth for what crosses phase boundaries. Each subagent reads only this file plus its own directory.
Changing a contract means editing this file in the same diff and calling out the change.

**Simulation only.** Nothing here touches real hosts or networks. The only network access in the project is
dataset downloads (Phase 2) and Anthropic API calls for narration (Phase 4).

## Layout

| Path | Owner | Phase |
|---|---|---|
| `src/cyberarena/ml/` | ml-pipeline | 2 |
| `src/cyberarena/arena/` | arena-engineer | 3 |
| `src/cyberarena/explain/` | explainability-writer | 4 |
| `src/cyberarena/dashboard/` | dashboard-builder | 5 |
| `src/cyberarena/config.py`, `requirements.txt`, `Makefile`, `pyproject.toml`, this file | main session | 1 |
| `data/`, `artifacts/`, `runs/` | generated, gitignored | — |

## Classifier artifacts (Phase 2 → 3, 4)

Names: `malware`, `phishing`, `network`.

- `artifacts/models/<name>.keras`: Keras model, input = scaled feature vector, output = sigmoid P(malicious)
- `artifacts/models/<name>_preprocess.json`: `{"feature_names": [...], "scaler_mean": [...], "scaler_scale": [...], "label_map": {"0": "benign", "1": "malicious"}}`
- `artifacts/metrics/<name>.json`: `{"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix", "n_train", "n_test", "source"}`
- `data/processed/<name>.npz`: `X_train, y_train, X_val, y_val, X_test, y_test` (already scaled), `feature_names`

Python interface, in `cyberarena.ml.inference`:

```python
clf = load_classifier("network")
clf.feature_names          # list[str]
clf.predict_proba(X)       # np.ndarray (n,) — P(malicious), X already scaled, shape (n, n_features)
clf.background(n=100)      # np.ndarray (n, n_features) — scaled training sample for SHAP
clf.sample(label, n, rng)  # np.ndarray (n, n_features) — held-out test rows of the given class (0/1)
```

## Episode log (Phase 3 → 4 → 5)

Run directory: `runs/<run_id>/`, where `run_id` = `YYYYmmdd-HHMMSS-<seed>`.

| File | Writer | Content |
|---|---|---|
| `config.json` | arena | env + agent hyperparams, seed |
| `graph.json` | arena | `{"nodes": [{"id", "role", "crown_jewel", "x", "y"}], "edges": [[u, v], ...]}` (layout positions fixed per run) |
| `episodes.jsonl` | arena | one **turn record** per line |
| `summary.jsonl` | arena | one **episode summary** or **eval row** per line |
| `episodes_enriched.jsonl` | explain | same turn records with `shap` / `rationale` / `mitre` filled |
| `agents/red.*`, `agents/blue.*` | arena | saved agent weights |

### Turn record

```json
{
  "run_id": "20261003-101500-7",
  "episode": 412,
  "turn": 9,
  "actor": "red",
  "action_id": "lateral_move",
  "source": 3,
  "target": 7,
  "success": true,
  "reward": 0.5,
  "decision_values": {"lateral_move": 1.42, "escalate": 0.97, "wait": 0.1},
  "epsilon": 0.05,
  "explored": false,
  "node_states": [{"id": 0, "compromised": false, "detected": false, "isolated": false,
                   "patched": false, "privilege": 0,
                   "scores": {"malware": 0.03, "phishing": 0.11, "network": 0.08}}],
  "classifier_inputs": [{"node": 7, "model": "network", "row": [0.12, -1.3, ...], "score": 0.21}],
  "done": false,
  "winner": null,
  "shap": null,
  "rationale": null,
  "mitre": null
}
```

`node_states` is the full post-action state for every node, so the dashboard can replay a turn without re-running the env.

Fields filled in Phase 4 (`episodes_enriched.jsonl`):

```json
"shap": [{"node": 7, "model": "network", "base_value": 0.31,
          "top_features": [{"name": "src_bytes", "value": 0.82, "shap": -0.12}]}],
"rationale": "Red moves laterally to host 7 because its network-anomaly score is low (0.21), so blue is unlikely to notice.",
"mitre": {"framework": "ATT&CK", "version": "v16", "technique_id": "T1021",
          "technique_name": "Remote Services", "tactic": "Lateral Movement"}
```

Blue actions: `mitre` is either a D3FEND tag (`"framework": "D3FEND"`) or `null`. Red `wait` is `null`.

Extra enriched fields (added in Phase 4): `rationale_meta` (`{"source": "claude"|"template"|"template:refusal", "model", "input_tokens", "output_tokens"}`),
`shap[].output` and `shap[].additivity_error`, and `top_features[].raw` (unscaled value).
Phase 4 also writes `runs/<run_id>/explain_summary.json` (selection, timings, cost estimate) and caches in `runs/<run_id>/explain_cache/`.

### Summary rows

```json
{"kind": "episode", "episode": 412, "winner": "blue", "turns": 23, "red_return": -1.0, "blue_return": 1.0, "epsilon": 0.05}
{"kind": "eval", "after_episode": 400, "matchup": "red_learned_vs_blue_baseline", "n": 50, "red_win_rate": 0.62, "blue_win_rate": 0.38}
{"kind": "eval", "after_episode": 400, "matchup": "blue_learned_vs_red_baseline", "n": 50, "red_win_rate": 0.30, "blue_win_rate": 0.70}
```

### Extra fields (added in Phase 3)

- Turn records: `phase` (`"train"`/`"eval"`), `matchup`, `agent` (`"learned"`, `"heuristic"` or `"random"`), and `after_episode` on eval turns.
  For baseline agents `decision_values` are heuristic priorities (or zeros), not Q-values.
- Eval episodes are numbered after training episodes (e.g. 2000+ for a 2000-episode run).
- Not every episode has turn records: full records are logged for every Nth training episode plus all eval
  episodes (see `config.json` → `logging`). Summary rows carry `logged: true|false`; eval rows carry `episodes: [first, last]`.
- `classifier_inputs` holds only the rows emitted this turn (usually 1). Rows are rounded to 3 dp.
- `episodes.jsonl` can be ~200 MB. Readers should stream it or index it by episode, not `json.load` the whole file.
- Checkpoint agents at every eval: `agents/checkpoints/{red,blue}_<after_episode>.json`.

Win conditions: red wins on exfiltrating from the crown jewel. Blue wins when red has no foothold left, or when the turn limit is reached first.
