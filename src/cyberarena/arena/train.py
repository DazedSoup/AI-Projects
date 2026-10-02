"""Train red and blue Q-learners simultaneously and log episodes (contract: docs/contracts.md, "Episode log").

    python -m cyberarena.arena.train --episodes 2000 --seed 7

Training episodes cycle through three matchups so each learner keeps meeting both the other learner and a
fixed baseline: learned-vs-learned (both update), learned red vs baseline blue (red updates), baseline red vs
learned blue (blue updates). Every ``--eval-every`` episodes (and before training) an evaluation block plays
greedy learners against the fixed baselines on a fixed set of episode seeds.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from cyberarena.arena.agents import BaseAgent, QAgent, make_baseline
from cyberarena.arena.env import MODELS, ArenaConfig, CyberArenaEnv

TRAIN_MATCHUPS = ("learned_vs_learned", "red_learned_vs_blue_baseline", "blue_learned_vs_red_baseline")
EVAL_MATCHUPS = ("red_learned_vs_blue_baseline", "blue_learned_vs_red_baseline")
EVAL_SEED_OFFSET = 1_000_003


def turn_record(env: CyberArenaEnv, run_id: str, episode: int, decision, info: dict, reward: float,
                extra: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    rec = {
        "run_id": run_id, "episode": episode, "turn": info["turn"], "actor": info["actor"],
        "action_id": info["action_id"], "source": info["source"], "target": info["target"],
        "success": info["success"], "reward": round(reward, 4),
        "decision_values": decision.decision_values, "epsilon": round(float(decision.epsilon), 4),
        "explored": bool(decision.explored), "node_states": env.node_states(),
        "classifier_inputs": [
            {"node": c["node"], "model": c["model"],
             "row": [round(x, 3) for x in env.classifier_row(c)], "score": c["score"]}
            for c in info["classifier_inputs"]
        ],
        "done": env.done, "winner": info["winner"],
        "shap": None, "rationale": None, "mitre": None,
    }  # fmt: skip
    rec.update(extra)
    return rec


def play_episode(env: CyberArenaEnv, agents: dict[str, BaseAgent], env_seed: int, learn: set[str],
                 explore: bool, log: TextIO | None = None, run_id: str = "", episode: int = 0,
                 extra: dict[str, Any] | None = None, replay: int = 0) -> dict[str, Any]:  # fmt: skip
    env.reset(seed=env_seed)
    learners = {s: a for s, a in agents.items() if s in learn and isinstance(a, QAgent)}
    for a in learners.values():
        a.begin_episode()
    returns = {"red": 0.0, "blue": 0.0}
    base_extra = dict(extra or {})
    while not env.done:
        side = env.actor
        agent = agents[side]
        if side in learners:
            learners[side].before_act(env)
            key = learners[side].features(env)
        decision = agent.act(env, explore=explore and side in learners)
        _, _, _, _, info = env.step((decision.action, decision.source, decision.target))
        if side in learners:
            learners[side].after_act(key, decision.action)
        for s in ("red", "blue"):
            returns[s] += info["rewards"][s]
            if s in learners:
                learners[s].add_reward(info["rewards"][s])
        if log is not None:
            rec = turn_record(env, run_id, episode, decision, info, info["rewards"][side],
                              {**base_extra, "agent": agent.kind})  # fmt: skip
            log.write(json.dumps(rec, separators=(",", ":")) + "\n")
    for a in learners.values():
        a.end_episode()
        a.replay(replay)
    return {"winner": env.winner, "turns": env.turn, "red_return": round(returns["red"], 4),
            "blue_return": round(returns["blue"], 4)}  # fmt: skip


def epsilon_at(ep: int, episodes: int, start: float, end: float, decay_frac: float) -> float:
    span = max(1, int(episodes * decay_frac))
    return end + (start - end) * max(0.0, 1.0 - ep / span)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-n", type=int, default=50, help="episodes per eval matchup")
    p.add_argument("--log-every", type=int, default=40,
                   help="write full turn records for every Nth training episode (all eval episodes are logged)")
    p.add_argument("--no-turn-log", action="store_true", help="skip episodes.jsonl turn records (tuning runs)")
    p.add_argument("--baseline", choices=("heuristic", "random"), default="heuristic")
    p.add_argument("--red-baseline-noise", type=float, default=0.3,
                   help="prob. the baseline red plays a random valid action instead of its playbook")
    p.add_argument("--blue-baseline-noise", type=float, default=0.4,
                   help="prob. the baseline blue plays a random valid action instead of its playbook")
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--env-json", type=str, default=None,
                   help="JSON object of ArenaConfig overrides, e.g. '{\"p_phish\": 0.4}'")
    p.add_argument("--alpha", type=float, default=0.1, help="initial learning rate")
    p.add_argument("--alpha-end", type=float, default=0.01, help="final learning rate (linear decay)")
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--replay", type=int, default=0, help="replayed transitions per learner per episode")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--eps-decay-frac", type=float, default=0.6)
    p.add_argument("--n-nodes", type=int, default=None, help="graph size (default: seeded 12-16)")
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--stub", action="store_true", help="use stub classifiers (no TensorFlow) for smoke tests")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    if args.runs_dir is None:
        from cyberarena.config import RUNS_DIR

        runs_dir = Path(RUNS_DIR)
    else:
        runs_dir = Path(args.runs_dir)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{args.seed}"
    run_dir = runs_dir / run_id
    (run_dir / "agents").mkdir(parents=True, exist_ok=True)

    def say(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    cfg = ArenaConfig()
    if args.env_json:
        cfg = ArenaConfig(**{**cfg.to_dict(), **json.loads(args.env_json)})
    if args.max_rounds is not None:
        cfg.max_rounds = args.max_rounds
    if args.stub:
        from cyberarena.arena.stub import stub_classifiers

        classifiers = stub_classifiers()
    else:
        classifiers = None
    env = CyberArenaEnv(cfg, graph_seed=args.seed, classifiers=classifiers, n_nodes=args.n_nodes)
    say(f"[arena] run {run_id}: {env.n} nodes, {len(env.graph.edges)} edges, crown jewel = node {env.crown}"
        f" (setup {time.time() - t0:.1f}s)")  # fmt: skip

    ss = np.random.SeedSequence(args.seed)
    s_red, s_blue, s_bred, s_bblue, s_sched = (int(x.generate_state(1)[0]) for x in ss.spawn(5))
    red = QAgent("red", seed=s_red, alpha=args.alpha, gamma=args.gamma, epsilon=args.eps_start)
    blue = QAgent("blue", seed=s_blue, alpha=args.alpha, gamma=args.gamma, epsilon=args.eps_start)
    base_red = make_baseline("red", args.baseline, s_bred, args.red_baseline_noise)
    base_blue = make_baseline("blue", args.baseline, s_bblue, args.blue_baseline_noise)
    sched = np.random.default_rng(s_sched)

    (run_dir / "graph.json").write_text(json.dumps(env.graph.to_json(), indent=1))
    config = {
        "run_id": run_id, "seed": args.seed, "episodes": args.episodes, "graph_seed": args.seed,
        "n_nodes": env.n, "crown_jewel": env.crown,
        "classifiers": "stub" if args.stub else {m: "artifacts/models/" + m + ".keras" for m in MODELS},
        "env": cfg.to_dict(),
        "agents": {"kind": "tabular_q", "alpha": {"start": args.alpha, "end": args.alpha_end or args.alpha,
                                              "schedule": "linear"}, "gamma": args.gamma,
                   "replay": {"per_episode": args.replay, "buffer": 20000, "sampling": "uniform"},
                   "epsilon": {"start": args.eps_start, "end": args.eps_end, "decay_frac": args.eps_decay_frac,
                               "schedule": "linear"},
                   "red_features": ["stage", "front_privilege", "n_footholds", "progress_move", "recon_front",
                                    "hot", "turn_bucket"],
                   "blue_features": ["n_detected_live", "top_undetected_score_bucket", "n_isolated",
                                     "crown_alert", "turn_bucket"]},
        "baseline": {"kind": args.baseline, "red_noise": args.red_baseline_noise,
                     "blue_noise": args.blue_baseline_noise},
        "train_matchups": {"cycle": list(TRAIN_MATCHUPS), "weights": [0.5, 0.25, 0.25]},
        "eval": {"every": args.eval_every, "n": args.eval_n, "greedy": True,
                 "env_seeds": f"{EVAL_SEED_OFFSET}+j, j<n (same seeds at every checkpoint)",
                 "episode_ids": "eval episodes are numbered after the training episodes (>= episodes)"},
        "logging": {"episodes_jsonl": "full turn records for training episodes with episode % log_every == 0 "
                                      "and for every eval episode",
                    "log_every": args.log_every,
                    "classifier_inputs": "only rows emitted this turn (acted-on node), rounded to 3 dp (scores 4 dp)",
                    "extra_turn_fields": ["phase", "matchup", "agent", "after_episode (eval only)"],
                    "checkpoints": "agents/checkpoints/{red,blue}_<after_episode>.json at every eval"},
    }  # fmt: skip
    (run_dir / "config.json").write_text(json.dumps(config, indent=1))

    alpha_end = args.alpha if args.alpha_end is None else args.alpha_end

    def alpha_at(ep: int) -> float:
        return args.alpha + (alpha_end - args.alpha) * ep / max(1, args.episodes - 1)

    eval_episode_id = args.episodes
    history: list[dict[str, Any]] = []
    logs: dict[str, TextIO] = {}

    def evaluate(after: int) -> None:
        nonlocal eval_episode_id
        # eval must not perturb training: snapshot every agent RNG and restore it afterwards
        rng_states = [a.rng.bit_generator.state for a in (red, blue, base_red, base_blue)]
        for matchup in EVAL_MATCHUPS:
            if matchup.startswith("red"):
                agents = {"red": red, "blue": base_blue}
            else:
                agents = {"red": base_red, "blue": blue}
            first = eval_episode_id
            wins = {"red": 0, "blue": 0}
            for j in range(args.eval_n):
                res = play_episode(env, agents, EVAL_SEED_OFFSET + j, learn=set(), explore=False,
                                   log=None if args.no_turn_log else logs["episodes"], run_id=run_id,
                                   episode=eval_episode_id,
                                   extra={"phase": "eval", "matchup": matchup, "after_episode": after})  # fmt: skip
                wins[res["winner"]] += 1
                eval_episode_id += 1
            row = {"kind": "eval", "after_episode": after, "matchup": matchup, "n": args.eval_n,
                   "red_win_rate": round(wins["red"] / args.eval_n, 4),
                   "blue_win_rate": round(wins["blue"] / args.eval_n, 4),
                   "episodes": [first, eval_episode_id - 1]}  # fmt: skip
            logs["summary"].write(json.dumps(row) + "\n")
            history.append(row)
        rl = history[-2]["red_win_rate"]
        bl = history[-1]["blue_win_rate"]
        say(f"[eval] after {after:5d}: learned red vs baseline blue red-win {rl:5.1%} | "
            f"learned blue vs baseline red blue-win {bl:5.1%} | {time.time() - t0:6.1f}s")  # fmt: skip
        logs["summary"].flush()
        logs["episodes"].flush()
        ckpt = run_dir / "agents" / "checkpoints"
        ckpt.mkdir(exist_ok=True)
        red.save(ckpt / f"red_{after:05d}.json")
        blue.save(ckpt / f"blue_{after:05d}.json")
        for a, st in zip((red, blue, base_red, base_blue), rng_states, strict=True):
            a.rng.bit_generator.state = st

    with (open(run_dir / "episodes.jsonl", "w", encoding="utf-8") as episodes_f,
          open(run_dir / "summary.jsonl", "w", encoding="utf-8") as summary_f):  # fmt: skip
        logs.update(episodes=episodes_f, summary=summary_f)
        evaluate(0)
        for ep in range(args.episodes):
            eps = epsilon_at(ep, args.episodes, args.eps_start, args.eps_end, args.eps_decay_frac)
            red.epsilon = blue.epsilon = eps
            red.alpha = blue.alpha = alpha_at(ep)
            matchup = TRAIN_MATCHUPS[int(sched.choice(3, p=[0.5, 0.25, 0.25]))]
            if matchup == "learned_vs_learned":
                agents, learn = {"red": red, "blue": blue}, {"red", "blue"}
            elif matchup == "red_learned_vs_blue_baseline":
                agents, learn = {"red": red, "blue": base_blue}, {"red"}
            else:
                agents, learn = {"red": base_red, "blue": blue}, {"blue"}
            logged = ep % args.log_every == 0 and not args.no_turn_log
            res = play_episode(env, agents, args.seed * 100_003 + ep, learn=learn, explore=True,
                               log=episodes_f if logged else None, run_id=run_id, episode=ep,
                               extra={"phase": "train", "matchup": matchup}, replay=args.replay)  # fmt: skip
            summary_f.write(json.dumps({"kind": "episode", "episode": ep, **res, "epsilon": round(eps, 4),
                                        "matchup": matchup, "logged": logged}) + "\n")  # fmt: skip
            if (ep + 1) % args.eval_every == 0 or ep + 1 == args.episodes:
                evaluate(ep + 1)

    red.save(run_dir / "agents" / "red.json")
    blue.save(run_dir / "agents" / "blue.json")
    wall = time.time() - t0
    size_mb = (run_dir / "episodes.jsonl").stat().st_size / 1e6
    config["wall_time_s"] = round(wall, 1)
    (run_dir / "config.json").write_text(json.dumps(config, indent=1))
    say(f"[arena] done in {wall:.1f}s -> {run_dir} (episodes.jsonl {size_mb:.1f} MB, "
        f"Q states red={len(red.q)} blue={len(blue.q)})")  # fmt: skip
    return run_dir


if __name__ == "__main__":
    main()
