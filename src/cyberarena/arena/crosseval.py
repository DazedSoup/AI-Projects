"""Cross-evaluation of saved blue Q-networks under swapped detectors (diagnostics). SIMULATION ONLY.

    python -m cyberarena.arena.crosseval --runs runs/<id_a> runs/<id_b> --detectors-from runs/<id_a> \\
        [--levels 0,0.4,0.7] [--n 200] [--out results.json]

Holds the agent fixed and changes only the detectors at eval time, which separates "the agent learned worse"
from "the detectors it is scored with differ". For every blue network (``<run>/agents/blue_qnet``) and every
detector set -- ``v0`` (the untouched Phase 2 models), ``final`` (the highest version saved under
``--detectors-from``/detectors) and ``shuffled`` (v0 scores permuted across benign and malicious pools, so the
scores carry no information) -- greedy blue plays the scripted red at each fixed evasion level. All runs passed
must share the graph seed. The env matches the training run's disguised-attacker eval env: ``arena_eval``
partition, pool seed stream ``[seed, 3]``, the run's ``evasion_cost`` and ``adaptive_pool_size``, env seeds
``EVAL_SEED_OFFSET + j``, and the scripted red's RNG reset to the run's baseline stream for every cell.

Per cell: blue win rate, isolations per game, the share of isolations that hit a clean host, monitor flags on
clean hosts per game. ``saliency`` (per network): mean |dQ/dx| over every candidate move of every blue state
visited in any cell (every ``STATE_STRIDE``-th decision; the same pooled state set for every network), scaled by each feature's sd, summed into
detector-score features vs the rest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cyberarena.arena.actions import BlueAction
from cyberarena.arena.adaptation import latest_detectors
from cyberarena.arena.agents import make_baseline
from cyberarena.arena.dqn import DQNAgent, NumpyMLP
from cyberarena.arena.env import EVASION_LEVELS, MODELS, CyberArenaEnv
from cyberarena.arena.features import FEATURE_NAMES, candidate_matrix
from cyberarena.arena.train import EVAL_SEED_OFFSET

STATE_STRIDE = 4  # saliency keeps every 4th blue decision state (memory)
SCORE_FEATURES = tuple(n for n in FEATURE_NAMES["blue"] if "score" in n or n == "g_top_undetected")


def input_gradients(net: NumpyMLP, X: np.ndarray) -> np.ndarray:
    """dQ/dX for a ReLU MLP (``NumpyMLP`` weights), one row per input row."""
    h = np.asarray(X, dtype=np.float64)
    masks = []
    w = [np.asarray(x, dtype=np.float64) for x in net.w]
    last = len(w) - 2
    for i in range(0, last, 2):
        z = h @ w[i] + w[i + 1]
        masks.append(z > 0)
        h = np.maximum(z, 0.0)
    g = np.broadcast_to(w[last][:, 0], (len(X), w[last].shape[0])).copy()
    for i in range(last - 2, -1, -2):
        g = (g * masks[i // 2]) @ w[i].T
    return g


def shuffle_scores(env: CyberArenaEnv, seed: int) -> None:
    """Permute each model's scores across all its pools: readings keep their marginal distribution, lose label
    information."""
    rng = np.random.default_rng(seed)
    for m in MODELS:
        keys = env.pool.keys(m)
        allp = np.concatenate([env.pool.scores[k] for k in keys])
        allp = allp[rng.permutation(len(allp))]
        i = 0
        for k in keys:
            n = len(env.pool.scores[k])
            env.pool.scores[k] = allp[i:i + n]
            i += n


def play_cell(env: CyberArenaEnv, blue: DQNAgent, red, red_seed: int, level: float, n: int,
              states: list[np.ndarray] | None) -> dict[str, Any]:  # fmt: skip
    env.evasion_idx[:] = EVASION_LEVELS.index(level)
    red.rng = np.random.default_rng(red_seed)
    wins = iso = iso_clean = flags_clean = monitors = turns = 0
    for j in range(n):
        env.reset(seed=EVAL_SEED_OFFSET + j)
        while not env.done:
            if env.actor == "red":
                d = red.act(env, explore=False)
                env.step((d.action, d.source, d.target))
                continue
            if states is not None and turns % STATE_STRIDE == 0:
                moves = env.all_candidates("blue")
                states.append(candidate_matrix(env, "blue", moves))
            d = blue.act(env, explore=False)
            tgt = d.target
            clean = tgt is not None and not bool(env.compromised[tgt])
            _, _, _, _, info = env.step((d.action, d.source, d.target))
            turns += 1
            if d.action == BlueAction.ISOLATE:
                iso += 1
                iso_clean += int(clean)
            elif d.action == BlueAction.MONITOR:
                monitors += 1
                flags_clean += int(clean and info["success"])
        wins += int(env.winner == "blue")
    return {"blue_win_rate": round(wins / n, 4), "isolations_per_game": round(iso / n, 3),
            "isolate_clean_share": round(iso_clean / iso, 4) if iso else None,
            "clean_flags_per_game": round(flags_clean / n, 3), "monitors_per_game": round(monitors / n, 3),
            "blue_turns_per_game": round(turns / n, 2)}  # fmt: skip


def build_env(cfg_json: dict[str, Any], detectors: dict[str, Any]) -> CyberArenaEnv:
    from cyberarena.arena.env import ArenaConfig

    seed = int(cfg_json["seed"])
    p = cfg_json["params"]
    s_evpool = int(np.random.SeedSequence([seed, 3]).generate_state(1)[0]) % 2**31
    return CyberArenaEnv(ArenaConfig(**cfg_json["env"]), graph_seed=seed, n_nodes=p["n_nodes"], detectors=detectors,
                         evasion_cost=p["evasion_cost"], adaptive_pool_size=p["adaptive_pool_size"],
                         pool_seed=s_evpool, partition="arena_eval")  # fmt: skip


def load_detector_sets(cfg_json: dict[str, Any], final_from: Path | None, stub: bool) -> dict[str, dict]:
    if stub:
        from cyberarena.arena.stub import StubAdaptiveDetector as Det
    else:
        from cyberarena.ml.adaptive import AdaptiveDetector as Det
    seed = int(cfg_json["seed"])
    s_det = int(np.random.SeedSequence([seed, 2]).spawn(2)[0].generate_state(1)[0]) % 100_000
    kw = {} if stub else {"compile": False}
    v0 = {m: Det.from_pretrained(m, seed=s_det + i, **kw) for i, m in enumerate(MODELS)}
    sets = {"v0": v0, "shuffled": v0}
    if final_from is not None:
        paths = latest_detectors(final_from)
        if set(paths) != set(MODELS):
            raise FileNotFoundError(f"{final_from}/detectors lacks a version for every model")
        sets["final"] = {m: Det.load(paths[m]) for m in MODELS}
    return sets


def crosseval(runs: list[Path], detectors_from: Path | None, levels: tuple[float, ...] = (0.0, 0.4, 0.7),
              n: int = 200, stub: bool = False, shuffled: bool = True) -> dict[str, Any]:  # fmt: skip
    cfgs = [json.loads((Path(r) / "config.json").read_text()) for r in runs]
    seeds = {int(c["seed"]) for c in cfgs}
    if len(seeds) != 1:
        raise ValueError(f"runs must share a seed, got {sorted(seeds)}")
    seed = seeds.pop()
    cfg0 = cfgs[0]
    red_seed = int(np.random.SeedSequence(seed).spawn(5)[2].generate_state(1)[0])
    red = make_baseline("red", cfg0["params"]["baseline"], red_seed, cfg0["params"]["red_baseline_noise"])
    blues = {Path(r).name: DQNAgent.load(Path(r) / "agents" / "blue_qnet", seed=0) for r in runs}
    views = {Path(r).name: c["params"].get("blue_score_view", "raw") for r, c in zip(runs, cfgs, strict=True)}
    sets = load_detector_sets(cfg0, detectors_from, stub)
    if not shuffled:
        sets.pop("shuffled")
    states: list[np.ndarray] = []
    cells = []
    for dname, dets in sets.items():
        env = build_env(cfg0, dets)
        if dname == "shuffled":
            shuffle_scores(env, seed)
        versions = dict(env.pool.version)
        for rid, blue in blues.items():
            env.blue_score_view = views[rid]
            for s in levels:
                res = play_cell(env, blue, red, red_seed, s, n, states if dname != "shuffled" else None)
                cells.append({"blue_run": rid, "detectors": dname, "detector_versions": versions, "evasion": s,
                              "n": n, **res})  # fmt: skip
    X = np.concatenate(states) if states else np.zeros((0, len(FEATURE_NAMES["blue"])))
    sd = X.std(axis=0) if len(X) else np.zeros(X.shape[1])
    score_cols = np.array([nm in SCORE_FEATURES for nm in FEATURE_NAMES["blue"]])
    sal = {}
    for rid, blue in blues.items():  # pooled states use the playing blue's score view
        g = np.abs(input_gradients(blue.net.online, X)).mean(axis=0) if len(X) else np.zeros(X.shape[1])
        gs = g * sd
        tot = float(gs.sum()) or 1.0
        sal[rid] = {"score_share": round(float(gs[score_cols].sum()) / tot, 4),
                    "score_sum": round(float(gs[score_cols].sum()), 4), "other_sum": round(float(gs[~score_cols].sum()), 4),
                    "raw_grad_score_mean": round(float(g[score_cols].mean()), 4),
                    "raw_grad_other_mean": round(float(g[~score_cols].mean()), 4),
                    "top": sorted(((nm, round(float(v), 4)) for nm, v in zip(FEATURE_NAMES["blue"], gs, strict=True)),
                                  key=lambda t: -t[1])[:8]}  # fmt: skip
    return {"seed": seed, "detectors_from": None if detectors_from is None else Path(detectors_from).name,
            "n_states": len(X), "score_features": list(SCORE_FEATURES), "cells": cells, "saliency": sal}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cyberarena.arena.crosseval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="run dirs whose blue_qnet to evaluate")
    p.add_argument("--detectors-from", type=Path, default=None, help="run dir whose latest detectors = 'final'")
    p.add_argument("--levels", default="0,0.4,0.7")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--no-shuffled", action="store_true")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    levels = tuple(float(x) for x in a.levels.split(","))
    res = crosseval(a.runs, a.detectors_from, levels, a.n, a.stub, not a.no_shuffled)
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
