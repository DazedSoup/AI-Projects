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

## Simulation Lab (arena ↔ dashboard, added after Phase 5)

The dashboard's Lab page launches training as a **subprocess**. It never imports arena code.

### Parameter spec: `python -m cyberarena.arena.train --describe-params`

This prints JSON to stdout and exits without training. The dashboard builds its widgets entirely from this output,
so adding a parameter in the arena makes it appear in the UI automatically.

```json
{"version": 1, "groups": [
  {"id": "network", "label": "Network", "params": [
    {"key": "n_nodes", "label": "Hosts", "type": "int", "default": null, "min": 10, "max": 20, "step": 1,
     "nullable": true, "help": "null = seeded 12-16", "target": "cli", "flag": "--n-nodes"},
    {"key": "p_phish", "label": "Phish success", "type": "float", "default": 0.35, "min": 0.0, "max": 1.0,
     "step": 0.01, "help": "...", "target": "env"},
    {"key": "baseline", "label": "Baseline type", "type": "choice", "default": "heuristic",
     "choices": ["heuristic", "random"], "target": "cli", "flag": "--baseline"}
  ]}
]}
```

- `type` is one of `int`, `float`, `bool` or `choice`.
- `target: "env"` keys go into `--env-json`. Dotted keys like `p_implant_leak.malware` set one entry of a dict field.
- `target: "cli"` keys are passed as `flag value` (a bool passes the flag alone when true).
- `advanced: true` marks params the UI may hide behind an expander.
- Group ids include `network`, `red`, `blue`, `rewards`, `training`, `opponents` and `memory`.

### New train flags

- `--init-from <run_dir>`: warm-start both learners from `<run_dir>/agents/{red,blue}.json`. Combine with
  `--init-side red|blue|both` (default `both`); a side that isn't warm-started trains from scratch.
  `config.json` records `"init_from": {"run_id", "side"}`. Q-table states don't depend on graph size, so this works across graph sizes.
- `--label "<text>"`: a free-text label stored in `config.json` → `"label"`.
- `--eval-log-n K`: write full turn records for only the first K eval episodes per matchup per checkpoint (default 10).
  All `--eval-n` episodes still count toward the win rate. This decouples eval precision from log size.
- Unknown `--env-json` keys, or out-of-range values, exit with code 2 and a one-line error on stderr.

### Progress file: `runs/<run_id>/progress.json`

Rewritten atomically (write to a temp file, then rename) at least every 2 s while training runs:

```json
{"status": "running", "episode": 740, "episodes": 2000, "started": "2026-10-03T10:15:00", "updated": "...",
 "last_eval": {"after_episode": 500, "red_win_rate": 0.62, "blue_win_rate": 0.44}, "error": null}
```

`status` is `running`, `done` or `error`. On a crash, write `error` with the message before exiting non-zero.
The run directory and `progress.json` must exist within ~1 s of launch, before classifiers load. The first line of stdout is
`RUN_DIR <absolute path>`, so the launcher can find the run directory.

Implemented extras:
- `progress.json` also carries `phase` (`setup`/`train`/`eval`/`done`/`error`).
- `config.json` carries `label`, `init_from` and `params` (all CLI values); eval summary rows carry `logged_n`.
- **Cancelling on Windows:** launch train with `CREATE_NEW_PROCESS_GROUP` and send `CTRL_BREAK_EVENT`.
  Train then writes `status: error`, `error: "cancelled"` and exits with code 130. `terminate()` can't be caught.

### Lab runner files (dashboard-owned)

- `runs/.lab/<token>.json`: job record written before launch. `<token>.stop` is the stop request.
  `<token>.train.log` holds train's output.
- `runs/<run_id>/lab_status.json`: `{"stage": "starting"|"training"|"enriching"|"done"|"error", "message", ...}`.
  `lab_enrich.log` holds enrich's output.
- Readers listing runs must skip `runs/.lab/`.

### Enrichment

The Lab runs `python -m cyberarena.explain.enrich --run <run_dir>` afterwards. Narration defaults to **offline**
(template, no API, no cost); online narration needs an explicit `--online` flag.

Win conditions: red wins on exfiltrating from the crown jewel. Blue wins when red has no foothold left, or when the turn limit is reached first.

## Adaptive detectors & learning telemetry (v2)

TensorFlow learns *during* arena training. Blue's three detectors are fine-tuned online, and red adapts how evasive
its activity is. The simulation stays abstract: "evasion" is interpolation between dataset rows in feature space,
not a real technique.

### ml: `cyberarena.ml.adaptive` (owner: ml-pipeline)

```python
det = AdaptiveDetector.from_pretrained("network", lr=1e-3, replay_frac=0.5, seed=7)  # 1e-4 barely moves it
det.version                          # int, 0 = the pretrained Phase 2 model
det.predict_proba(X)                 # (n,) P(malicious) from the current version
det.evasive_rows(label=1, level=s, n, rng)
                                     # arena_train-partition malicious rows blended toward benign: x = (1-s)*x_mal + s*x_ben,
                                     # s in [0, 1]; deterministic for a given rng
det.update(X, y) -> UpdateReport     # few epochs at low lr on X,y mixed with replay of original train data
                                     # (replay_frac) to limit forgetting; bumps version
det.evaluate(level_grid) -> dict     # {"clean_auc", "auc_by_level": {s: auc}, "recall_by_level": {s: recall}}
                                     # on held-out test rows
det.save(path) / AdaptiveDetector.load(path)
```

`UpdateReport` fields: `version`, `n_new`, `n_replay`, `loss_before`, `loss_after`, `seconds`, plus a `before`/`after`
`evaluate()` dict. Each update should take under ~1 s on CPU.

### arena: integration (owner: arena-engineer)

- **Sensor pools must be built from `det.evasive_rows` / `det.sample`**, not `clf.sample`; otherwise rows from the
  fixed eval set leak into detector updates.
- **Evasion levels:** the sensor pool is pre-scored at discrete levels `EVASION_LEVELS = [0.0, 0.1, ..., 0.7]` per model.
  After every detector update, all pools for that model are re-scored in one batched predict. `step()` still never
  calls TF.
- **Red evasion policy:** red keeps an evasion level per sensor model, adapted by a simple bandit on outcomes.
  Being detected pushes it up; reward lost to slow progress pushes it down, because higher evasion lowers action success:
  `p_success *= (1 - evasion_cost * s)`. Each emitted row uses red's current level for that model.
- **Detector training data:** ground-truth labels are revealed only when blue acts on a host (isolate, restore,
  reset_credentials, or a monitor on an already-confirmed host). Those revealed (row, label) pairs go into a per-model buffer.
  Every `--detector-update-every` episodes (default 100), each model with at least `--detector-min-samples` (default 32)
  calls `update()` and the pools are re-scored.
- **Flags:** `--adaptive/--no-adaptive` (default on), `--detector-lr`, `--detector-update-every`, `--evasion-cost`,
  `--red-evasion-lr`. All of them appear in `--describe-params` (new group `adaptation`).
- **Saved detectors:** `runs/<id>/detectors/<model>_v<version>.keras`. Turn records gain
  `"detector_versions": {"malware": 3, ...}`, and each `classifier_inputs[]` entry gains `"evasion": s` and `"version": v`.

### `runs/<id>/learning.jsonl` (arena writes, dashboard reads)

One JSON object per line, each with a `kind`:

```json
{"kind": "detector_update", "episode": 700, "model": "network", "version": 4, "n_new": 61, "n_replay": 61,
 "loss_before": 0.41, "loss_after": 0.29, "before": {"clean_auc": 0.87, "auc_by_level": {"0.3": 0.71}},
 "after": {"clean_auc": 0.88, "auc_by_level": {"0.3": 0.80}}}
{"kind": "red_evasion", "episode": 700, "levels": {"malware": 0.2, "phishing": 0.1, "network": 0.4}}
{"kind": "agent_stats", "episode": 700, "side": "red", "n_states": 280, "mean_abs_q": 0.31, "td_error": 0.042,
 "epsilon": 0.41, "action_mix": {"exploit": 0.3, "lateral_move": 0.25}}
{"kind": "probe", "after_episode": 750, "side": "blue", "probe_id": "p3", "description": "2 footholds, one detected, server hot",
 "q": {"isolate": 0.42, "monitor": 0.10}, "chosen": "isolate"}
```

- `agent_stats` rows cover a rolling window of the last `--stats-every` episodes (default 50).
- `probe` rows are written at every eval checkpoint for a fixed set of 6–10 hand-picked states per side, defined in
  arena code with readable descriptions. They show the same situation answered differently as training goes on.
- **Probe games:** at every eval checkpoint, one fixed-seed game per matchup is fully logged with `"probe_game": true`
  (same env seed every time). The dashboard can replay "the same game at episode 0 vs 2000".

### explain (owner: explainability-writer)

SHAP must use the detector version that actually scored the row: load
`runs/<id>/detectors/<model>_v<version>.keras` when a turn's `classifier_inputs[].version > 0`. `load_classifier` gains an
optional `model_path`. Rationales mention adaptation when it's relevant, e.g. "red raised its network evasion to 0.4
after being caught twice; detector v4 still flags it at 0.71".

Implemented enriched-record additions (v2):
- `shap[].version`: the detector version used (0 = pretrained; every v1 entry gets 0).
- Per-turn `adaptation`, or `null` for v1 turns:
  - `evasion` `{model: s}`, `detector_versions` `{model: v}` and `updated_since_last_move` `[models]`;
  - optional `retrained_at` `{model: episode}`, `previous_versions`, `evasion_prev` `{model: {"level", "episode"}}`
    and `caught_rate` `{model: x}`.
- The default enrich selection for v2 runs adds one probe game per matchup from the first and last checkpoints.
- `explain_cache/*.jsonl` must be written by one enrich process per run at a time. Readers skip corrupt lines.

Implemented arena additions (v2):
- **Param spec:** a bool param may carry `"flag_false"` (e.g. `--no-adaptive`); pass it when the value is false.
  New params: `adaptive_pool_size`, `red_evasion_explore`, `red_detect_penalty`, `stats_every`. Default `evasion_cost` is 0.25.
- **Turn records:** `red_evasion` `{model: s}` and `probe_game`. Eval summary rows: `probe_episode`. Episode summary rows: `red_evasion`.
- **`learning.jsonl` extras:**
  - `detector_update`: `n_malicious`, `n_benign`, `red_level`, `seconds`;
  - `red_evasion`: `detector_versions`, `recall_at_level`, `caught_rate`, `success_rate`, `values`, `games`;
  - `agent_stats`: `n_updates`, `games`, `win_rate`;
  - `probe`: `state`, `seen`.
- **Run files:** `config.json` → `adaptation` (incl. `initial_eval`, `final`); `agents/red_evasion.json`.
  Probe games take one episode id after each matchup's eval block.
- **Warm start** copies the source run's latest detector files and restores red's evasion bandit.

## Dashboard design system (owner: dashboard-builder)

- **Theme:** dark-first "security operations console", with a light theme that also passes contrast.
  `.streamlit/config.toml` (owned by dashboard-builder) sets the base theme and font. All custom CSS lives in one
  `dashboard/theme.py` and is injected once.
- **Tokens:** an 8 px spacing scale; radius 10 px for cards and 8 px for controls; one neutral ramp; team red #E5484D,
  team blue #3E8BFF; one accent for "learning" (violet #8E7CFF) used only for adaptation and learning visuals.
  Node-state colours always pair with marker shape.
- **Type:** Inter for UI and JetBrains Mono for numbers and IDs, loaded from Google Fonts with system fallbacks.
  Tabular numerals for metrics.
- **Copy:** plain language everywhere.
  - "scripted opponent", not "baseline"; "game", not "episode", in UI copy, with episode ids shown in mono.
  - Every select box has a label that says what it chooses, plus a one-line caption saying what data it reads.
- **Components:** consistent control heights, primary versus secondary buttons, grouped toolbars, cards with a title,
  a subtitle and a single takeaway line per chart.
- **3D is used where depth carries meaning:**
  - the host network as stacked zone layers: Internet/DMZ → workstations → servers → crown jewel;
  - the "learning landscape" surface, a detector's AUC over training time × evasion level.

  Every 3D view has a 2D toggle and a sensible default camera.

## Experiments & rigor (v3)

### Evading-red evaluation (arena)

- Each eval checkpoint gains matchups `blue_learned_vs_red_evasive` at each level in `--eval-evasion-levels`
  (default `0.4,0.7`): learned blue against the scripted red playing at fixed evasion `s` on every sensor.
  This is the test that isolates whether adapted detectors beat evasion.
- Summary eval rows add `"evasion": s` for these matchups (absent or 0 for the others). `n` is `--eval-n`, as for the
  other matchups. Turn records for these games are not logged, except one probe game per level.

### Multi-seed experiments: `python -m cyberarena.arena.experiment`

```
python -m cyberarena.arena.experiment --name main --seeds 1-5 --conditions adaptive,frozen --episodes 2000 \
    [--jobs N] [-- <extra train args>]
python -m cyberarena.arena.experiment --aggregate runs/experiments/main
```

- Each (seed, condition) runs as a separate `cyberarena.arena.train` subprocess, in parallel (`--jobs`, default
  min(cpu-2, n_runs)). `frozen` adds `--no-adaptive`. `--label` is set to `"<name> · <condition> · seed <s>"`.
  Run dirs stay under `runs/` as usual.
- `runs/experiments/<name>/manifest.json`:
  `{"name", "created", "episodes", "seeds": [...], "conditions": [...], "extra_args": [...], "runs": [{"seed", "condition", "run_dir", "status"}]}`
- `runs/experiments/<name>/aggregate.json`, written automatically when all runs finish:

```json
{"name": "main", "n_seeds": 5,
 "curves": {"<condition>": {"<matchup>[@<evasion>]": [
    {"after_episode": 500, "mean": 0.61, "sd": 0.04, "ci95": [0.56, 0.66], "per_seed": [0.6, 0.58, ...]}]}},
 "final": {"<condition>": {"<matchup>[@<evasion>]": {"mean", "sd", "ci95", "per_seed"}}},
 "late_mean": {"<condition>": {"<matchup>[@<evasion>]": {...same stats over checkpoints >= episodes/2...}}},
 "contrasts": [{"metric": "blue_learned_vs_red_evasive@0.7 late_mean", "a": "adaptive", "b": "frozen",
                "diff_mean": 0.21, "ci95": [0.15, 0.27], "paired_by_seed": true, "p_value": 0.003}],
 "arms_race": {"<model>": {"cycles_mean", "recall_at_0.7_first", "recall_at_0.7_last"}}}
```

- The 95% CIs are t-intervals across seeds. Contrasts are paired by seed (paired t-test). The win rate for a matchup is
  always the learned side's win rate.
- The dashboard reads `runs/experiments/*/aggregate.json` and `manifest.json`. It never recomputes from scratch if
  `aggregate.json` exists.

### Pipeline: `python -m cyberarena.pipeline` (main session)

A single cross-platform entry point that runs data → classifiers → reference run → enrich → experiment,
skipping any step whose outputs already exist (`--force` redoes them). Offline and free. `--quick` gives a smoke
version that finishes in about 5 minutes.

## TensorFlow agents (v4): learned target selection

Stage 2 replaces "agent picks the action type, a fixed rule picks the host" with a learned Q-network that scores every
concrete move.

### Agent (arena)

- `--agent dqn|tabular` (default `dqn`; `tabular` keeps the v1–v3 behaviour for comparison). Both appear in `--describe-params`.
- **Candidate moves:** every legal `(action, source, target)` from `env.candidates(side, action)` over `valid_actions(side)`.
- **Features:** each candidate is a fixed-length vector built by `arena/features.py::candidate_features(env, side, move)`.
  - Global features: game stage, turn fraction, footholds, crown-jewel status, red evasion (blue sees only detector
    scores, never red's internal evasion level).
  - Target-host features: role one-hot, compromised*/detected/isolated/patched/privilege*, the three detector scores,
    recon'd, hops to the crown jewel, degree.
  - Source-host features where relevant, plus a one-hot of the action.
  - Fields marked * are only visible to the side that would know them: blue never sees ground-truth compromise.
  - `FEATURE_NAMES` is a module constant.
- **Q-network:** a Keras MLP mapping features to a scalar Q, trained with Double DQN, a replay buffer, a target network,
  Huber loss and epsilon-greedy over candidates.
  - Training uses TensorFlow (`train_on_batch`). Per-step inference may use a NumPy mirror of the weights, synced after
    each training call, for speed. It must give the same Q-values as the Keras model to 1e-5 (tested).
- **Saving:** `agents/{red,blue}_qnet.keras` plus `agents/{red,blue}_qnet.json` (feature names, hyperparams, step count),
  and checkpoints `agents/checkpoints/{side}_qnet_<after>.keras`. Warm start (`--init-from`) loads these.
- **Performance:** the default 2000-game run should stay under about 4 minutes on CPU.

### Logging changes

- **Turn record:** `decision_values` keys become `"<action>→<target>"` (or just `"<action>"` when there's no target),
  top 8 by Q.
  - New `candidates`: `[{"action", "source", "target", "q"}]` sorted by Q, top 12.
  - New `chosen_features`: `{name: value}` for the chosen move's feature vector.
  - `agent` stays `"learned"` for DQN agents; config.json `agents.type = "dqn"`.
- **learning.jsonl:** `agent_stats` gains `loss`, `q_mean`, `replay_size` and `grad_steps`.
  `probe` rows become `{"q": {"<action>→<target>": q}}`. Probe states are env snapshots stored in `arena/probes.py`,
  each with a readable description.

### Explanation of agent decisions (explain)

- **Integrated gradients** on the saved Q-network (the checkpoint closest to the turn) for the chosen candidate,
  against an all-zeros or mean-feature baseline. Stored per turn as
  `agent_attribution: {"method": "integrated_gradients", "baseline", "top_features": [{"name", "value", "attribution"}], "q": q, "checkpoint"}`.
  Exact for v4 runs; null for tabular runs.
- **Rationale** explains *why this host*, e.g. "…chooses host 13 over host 9 mainly because it is 1 hop from the crown
  jewel and its network score is low (0.08)", computed from `agent_attribution` and from the runner-up candidate's Q.
- **Cache safety:** `explain_cache/*.jsonl` writes take a lock file (`explain_cache/.lock`, OS-level exclusive create
  with stale-lock detection), so concurrent enrich runs on one run can't corrupt the cache.

### Datasets (ml)

- **Malware moves to UCI NATICUSdroid** (Android permissions; id 722; about 29k rows; no auth), if it's reachable.
  Otherwise use another public, no-auth dataset with at least 5k rows, and say which. TUANDROMD stays loadable as `malware_tuandromd`.
- After the switch, re-run the adaptive demo.

### Disjoint row partitions (v4, fixes eval memorisation)

The v3 evading-red eval drew its rows from the same held-out pool as the training env, so adapted detectors could
partly win by memorising rows. v4 splits each model's held-out rows into three disjoint partitions:

- `metric`: the fixed rows that `det.evaluate()` uses. Never in any pool and never in any update.
- `arena_train`: the training-env sensor pools. Revealed labels from these rows feed `update()`.
- `arena_eval`: the evading-red eval env pools. Never seen by `update()`.

```python
det.pool_rows(label, level, n, rng, partition="arena_train" | "arena_eval")   # blended at `level` for label 1
det.partition_sizes()   # {"metric": {"0": n, "1": n}, "arena_train": {...}, "arena_eval": {...}}
```

The arena builds training pools from `arena_train` and evading-eval pools from `arena_eval`. `config.json` records
`partition_sizes`. Every partition needs at least 150 malicious rows, which the larger malware dataset makes possible.

v3 field additions:
- The evasive eval uses `--eval-evasive-n` (default 100).
- Rows carry `detector_versions`. Probe games carry `evasion`.
- `config.json` has `eval_evasive`; `progress.last_eval` has `evasive_blue_win_rate`.
- Manifest and aggregate entries carry `n`, `seeds`, `runs_used`/`runs_failed`, `definitions`, and per-contrast `t`, `per_seed_diff`.
- A cycle is a ≥0.2 rise in red's evasion from a running low, followed by a ≥0.2 fall from that peak.

Implemented v4 arena details:
- **Per-side features:** `FEATURE_NAMES` is a dict per side (red 43, blue 31), also stored in `config.json` →
  `agents.feature_names` and in each `*_qnet.json`.
- **DQN run files:** `agents/{side}_qnet.keras` + `.json`, and checkpoints `agents/checkpoints/{side}_qnet_<after>.keras` + `.json`.
  There's no `agents/{side}.json` in DQN runs, so has-agents checks must accept either form.
- **Turn records:** `candidates` and `chosen_features` are null for scripted and tabular agents. DQN `decision_values`
  keep the best Q per `"<action>→<target>"`.
- **Probe rows (DQN):** top-8 `q`, `chosen`, `state` (env snapshot), `n_candidates`. `agent_stats.n_states` is null for DQN.
- **New params:** learner hyperparameters (`agent`, `dqn_*`, replay, batch, target sync, train-every, n-step) in the
  `training` group; `phish_forensics` and `phish_campaign_size` in `adaptation`.
- **Pools:** `config.json` has `pools` naming the partition each env used. Malware and phishing train pools sample with replacement.

Implemented v4 explain additions:
- **`agent_attribution`** (null unless a DQN decision has `chosen_features` and a checkpoint):
  - `method`, `baseline` (`candidate_mean`|`zeros`), `baseline_info` {`n_candidates`, `candidate_set`, `pre_state`, `imputed`, `check_error`};
  - `q`, `q_baseline`, `integration` (`exact_piecewise_linear`|`riemann_midpoint`), `steps`|`segments`, `completeness_error`,
    `completeness_rel`, `attribution_sum`;
  - `top_features` [{`name`, `value`, `baseline`, `attribution`}] (top 8 by |attribution|);
  - `q_logged`, `checkpoint`, `checkpoint_after`, `runner_up` {`action`, `source`, `target`, `q`}, `margin`.
- Plain-English feature phrases live in `explain/phrases.py`.
- **Cache:** `explain_cache/agent_ig.jsonl`; lock `explain_cache/.lock`.
- **CLI:** `--no-ig` and `--lock-timeout` (default 300; exit code 3 when locked).
- The v4 default selection adds the final checkpoint's evasive@0.7 probe game (cap 14 episodes).
- Red has 42 features (not 43).

## Rules and diagnostics (v5)

- **Reward defaults:** `r_isolate_false = 0.3` and `r_isolated_upkeep = 0.03`, so isolating a clean host is costly.
  `--rules cheap-isolation` restores the previous 0.1 / 0.01. The preset applies first, then `--env-json` overrides it.
- **Diagnostic flags** (adaptation group, advanced; defaults keep normal behaviour):
  - `--detectors auto|adaptive|frozen`
  - `--red-evasion auto|bandit|off`
  - `--blue-score-view raw|clean_quantile`
  - `--detectors-from RUN` (train only)
- `python -m cyberarena.arena.crosseval` runs fixed-agent cross-evaluation, the shuffled-score ablation, isolation stats and saliency.
- **Experiments on disk:**
  - `main`: the current rules.
  - `main-cheap-isolation`: the previous rules, where adaptive detectors hurt DQN blue.
  - `diag-det{A,F}-ev{B,Off}`: the 2×2 factorial under the previous rules.
  - `diag-fix-quantile`, `diag-costly-isolation`.
- **Reproducibility:** adaptive detector updates aren't bit-reproducible under heavy parallel CPU load (TF float
  nondeterminism). Frozen runs and the DQN are.
