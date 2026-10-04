"""Train red and blue Q-learners simultaneously and log episodes (contract: docs/contracts.md, "Episode log").

    python -m cyberarena.arena.train --episodes 2000 --seed 7
    python -m cyberarena.arena.train --describe-params          # Simulation Lab parameter spec (JSON)
    python -m cyberarena.arena.train --init-from runs/<id> --eps-start 0.3 --label "continued"

Training episodes cycle through three matchups so each learner keeps meeting both the other learner and a
fixed baseline: learned-vs-learned (both update), learned red vs baseline blue (red updates), baseline red vs
learned blue (blue updates). Every ``--eval-every`` episodes (and before training) an evaluation block plays
greedy learners against the fixed baselines on a fixed set of episode seeds.

The first stdout line is ``RUN_DIR <absolute path>``; ``<run_dir>/progress.json`` tracks status
(contract: docs/contracts.md, "Simulation Lab").

``--adaptive`` (default; contract "Adaptive detectors & learning telemetry (v2)"): blue's detectors are
fine-tuned every ``--detector-update-every`` games on labels revealed by blue's own actions, and the learned
red picks an evasion level per sensor with a bandit. ``learning.jsonl`` logs detector updates, red's levels,
agent stats, probe states; one fixed-seed probe game per matchup is logged at every checkpoint.
``--no-adaptive`` reproduces the v1 behaviour (same summary rows for the same seed).

Disguised-attacker eval (contract "Experiments & rigor (v3)"): each checkpoint also plays learned blue against
the scripted red at fixed evasion ``s`` on every sensor, for each ``s`` in ``--eval-evasion-levels``
(matchup ``blue_learned_vs_red_evasive``, summary rows carry ``evasion``). Those games run in a separate eval
env -- same graph, its own pre-scored pools (fresh benign/malicious pairings, seeded independently of the
training pools), ``--evasion-cost`` applied -- whose scores follow the run's detectors: the adapted versions
in an adaptive run, the untouched v0 models with ``--no-adaptive``. Training never sees that env, and agent
RNGs are restored after every eval block, so adding these matchups does not change training.

Learners (contract "TensorFlow agents (v4)"): ``--agent dqn`` (default) trains a Keras Q-network per side that
scores every concrete move (``arena/dqn.py``, features in ``arena/features.py``); ``--agent tabular`` is the
v1-v3 Q-table learner. With detectors that expose ``pool_rows`` (v4 partitions) the training env's sensor
pools come from ``arena_train`` and the disguised-attacker eval env's from ``arena_eval``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from cyberarena.arena.adaptation import DetectorTrainer, EvasionBandit, copy_detector, latest_detectors
from cyberarena.arena.agents import BaseAgent, QAgent, make_baseline
from cyberarena.arena.dqn import DQNAgent, DQNConfig, parse_hidden
from cyberarena.arena.env import EVASION_LEVELS, MODELS, ArenaConfig, CyberArenaEnv
from cyberarena.arena.features import FEATURE_NAMES
from cyberarena.arena.params import (
    CLI_PARAMS,
    ENV_PARAMS,
    ParamError,
    add_cli_args,
    check_value,
    parse_env_json,
    parse_evasion_levels,
    spec,
    validate_cli,
)
from cyberarena.arena.probes import probe_rows, snapshot_probe_rows

TRAIN_MATCHUPS = ("learned_vs_learned", "red_learned_vs_blue_baseline", "blue_learned_vs_red_baseline")
EVAL_MATCHUPS = ("red_learned_vs_blue_baseline", "blue_learned_vs_red_baseline")
EVASIVE_MATCHUP = "blue_learned_vs_red_evasive"
EVAL_SEED_OFFSET = 1_000_003
PROBE_GAME_SEED = 4_242_421  # env seed of the per-checkpoint probe game (same every checkpoint)
PROG = "cyberarena.arena.train"


def turn_record(env: CyberArenaEnv, run_id: str, episode: int, decision, info: dict, reward: float,
                extra: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    dv, cands, feats = decision.log_fields()
    rec = {
        "run_id": run_id, "episode": episode, "turn": info["turn"], "actor": info["actor"],
        "action_id": info["action_id"], "source": info["source"], "target": info["target"],
        "success": info["success"], "reward": round(reward, 4),
        "decision_values": dv, "candidates": cands, "chosen_features": feats,
        "epsilon": round(float(decision.epsilon), 4),
        "explored": bool(decision.explored), "node_states": env.node_states(),
        "classifier_inputs": [
            {"node": c["node"], "model": c["model"],
             "row": [round(x, 3) for x in env.classifier_row(c)], "score": c["score"],
             "evasion": c["evasion"], "version": c["version"]}
            for c in info["classifier_inputs"]
        ],
        "done": env.done, "winner": info["winner"],
        "detector_versions": dict(env.pool.version),
        "red_evasion": {m: env.evasion(m) for m in MODELS},
        "probe_game": False,
        "shap": None, "rationale": None, "mitre": None,
    }  # fmt: skip
    rec.update(extra)
    return rec


def play_episode(env: CyberArenaEnv, agents: dict[str, BaseAgent], env_seed: int, learn: set[str],
                 explore: bool, log: TextIO | None = None, run_id: str = "", episode: int = 0,
                 extra: dict[str, Any] | None = None, replay: int = 0) -> dict[str, Any]:  # fmt: skip
    env.reset(seed=env_seed)
    learners = {s: a for s, a in agents.items() if s in learn and getattr(a, "learns", False)}
    for a in learners.values():
        a.begin_episode()
    returns = {"red": 0.0, "blue": 0.0}
    base_extra = dict(extra or {})
    while not env.done:
        side = env.actor
        agent = agents[side]
        if side in learners:
            learners[side].pre_step(env)
        decision = agent.act(env, explore=explore and side in learners)
        _, _, _, _, info = env.step((decision.action, decision.source, decision.target))
        if side in learners:
            learners[side].post_step(decision)
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


class Cancelled(Exception):
    """Raised from the SIGTERM / SIGBREAK handler so the run records status ``error`` / ``cancelled``."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # one-line error on stderr, exit code 2
        self.exit(2, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog=PROG, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # seed, n_nodes, episodes, eval_*, alpha*, gamma, eps_*, baseline*, init_side: from arena/params.py
    add_cli_args(p)
    p.add_argument("--describe-params", action="store_true",
                   help="print the Simulation Lab parameter spec as JSON and exit")
    p.add_argument("--init-from", type=Path, default=None,
                   help="warm-start learners from <run_dir>/agents/{red,blue}_qnet.keras (dqn) or "
                        "{red,blue}.json (tabular); see --init-side")
    p.add_argument("--detectors-from", type=Path, default=None,
                   help="diagnostic: start blue's detectors from the latest versions saved in <run_dir>/detectors "
                        "(agents start fresh; combine with --detectors frozen for fixed adapted detectors)")
    p.add_argument("--label", type=str, default=None, help="free-text label stored in config.json")
    p.add_argument("--log-every", type=int, default=40,
                   help="write full turn records for every Nth training episode (eval: see --eval-log-n)")
    p.add_argument("--no-turn-log", action="store_true", help="skip episodes.jsonl turn records (tuning runs)")
    p.add_argument("--max-rounds", type=int, default=None, help="shortcut for --env-json '{\"max_rounds\": N}'")
    p.add_argument("--env-json", type=str, default=None,
                   help="JSON object of ArenaConfig overrides, e.g. '{\"p_phish\": 0.4, \"noise.recon\": 0.1}'")
    p.add_argument("--replay", type=int, default=0, help="replayed transitions per learner per episode")
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--stub", action="store_true", help="use stub classifiers (no TensorFlow) for smoke tests")
    p.add_argument("--quiet", action="store_true")
    return p


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Progress:
    """``progress.json``, rewritten atomically by a 1 s heartbeat thread and on ``finish``."""

    def __init__(self, path: Path, episodes: int):
        self.path = path
        self.state: dict[str, Any] = {"status": "running", "phase": "setup", "episode": 0, "episodes": episodes,
                                      "started": _now(), "updated": None, "last_eval": None,
                                      "error": None}  # fmt: skip
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.write()
        self._thread = threading.Thread(target=self._beat, name="progress", daemon=True)
        self._thread.start()

    def _beat(self) -> None:
        while not self._stop.wait(1.0):
            self.write()

    def update(self, **kw: Any) -> None:
        with self._lock:
            self.state.update(kw)

    def write(self) -> None:
        with self._lock:
            self.state["updated"] = _now()
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.state), encoding="utf-8")
            for i in range(20):  # on Windows, replace fails while a reader holds the target open
                try:
                    os.replace(tmp, self.path)
                    return
                except PermissionError:
                    time.sleep(0.02 * (i + 1))

    def finish(self, status: str, error: str | None = None) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self.update(status=status, phase=status, error=error)
        self.write()


def load_init_dqn(run_dir: Path, side: str) -> list[np.ndarray]:
    """Q-network weights of ``side`` saved under ``run_dir``; ParamError when missing or incompatible."""
    base = Path(run_dir) / "agents" / f"{side}_qnet"
    if not base.with_suffix(".keras").is_file() or not base.with_suffix(".json").is_file():
        hint = " (that run used --agent tabular)" if (Path(run_dir) / "agents" / f"{side}.json").is_file() else ""
        raise ParamError(f"--init-from: {base.with_suffix('.keras')} not found{hint}")
    try:
        meta = DQNAgent.read_meta(base)
    except (ValueError, KeyError, TypeError) as e:
        raise ParamError(f"--init-from: cannot read {base.with_suffix('.json')}: {e}") from None
    if meta.get("side") != side or meta.get("feature_names") != list(FEATURE_NAMES[side]):
        raise ParamError(f"--init-from: {base} is not a compatible {side} Q-network (different features)")
    return meta


def load_init(run_dir: Path, side: str) -> dict[tuple[int, ...], np.ndarray]:
    """Q-table of ``side`` saved under ``run_dir``; ParamError when missing or incompatible."""
    path = Path(run_dir) / "agents" / f"{side}.json"
    if not path.is_file():
        hint = " (that run used --agent dqn)" if (Path(run_dir) / "agents" / f"{side}_qnet.keras").is_file() else ""
        raise ParamError(f"--init-from: {path} not found{hint}")
    try:
        saved = QAgent.load(path)
        actions = json.loads(path.read_text())["actions"]
    except (ValueError, KeyError, TypeError) as e:
        raise ParamError(f"--init-from: cannot read {path}: {e}") from None
    if saved.side != side or actions != list(QAgent(side).ids):
        raise ParamError(f"--init-from: {path} is not a compatible {side} agent")
    return saved.q


def _validate(args: argparse.Namespace) -> tuple[ArenaConfig, dict[str, dict]]:
    validate_cli(args)
    cfg = parse_env_json(args.env_json, args.rules)
    if args.max_rounds is not None:
        cfg.max_rounds = check_value(ENV_PARAMS["max_rounds"], args.max_rounds)
    if args.log_every < 1:
        raise ParamError("--log-every must be >= 1")
    if args.replay < 0:
        raise ParamError("--replay must be >= 0")
    args.evasive_levels = parse_evasion_levels(args.eval_evasion_levels)
    init_q: dict[str, dict] = {}
    if args.init_from is not None:
        if not Path(args.init_from).is_dir():
            raise ParamError(f"--init-from: {args.init_from} is not a run directory")
        sides = ("red", "blue") if args.init_side == "both" else (args.init_side,)
        loader = load_init_dqn if args.agent == "dqn" else load_init
        init_q = {s: loader(args.init_from, s) for s in sides}
    args.dqn_hidden_t = parse_hidden(args.dqn_hidden)
    if not args.adaptive and (args.detectors == "adaptive" or args.red_evasion == "bandit"):
        raise ParamError("--detectors adaptive / --red-evasion bandit need --adaptive")
    # diagnostic split of adaptation (auto = follow --adaptive, the default)
    args.detectors_adapt = args.adaptive and args.detectors != "frozen"
    args.red_evades = args.adaptive and args.red_evasion != "off"
    return cfg, init_q


def init_adaptation(args: argparse.Namespace) -> dict[str, Any]:
    """What ``--init-from`` can restore beyond Q-tables: blue's latest detectors, red's evasion bandit.

    v1 runs have neither, which is fine (they stay valid warm-start sources)."""
    out: dict[str, Any] = {"detectors": {}, "red_evasion": None}
    if args.detectors_from is not None and args.adaptive:
        out["detectors"] = latest_detectors(args.detectors_from)
        if set(out["detectors"]) != set(MODELS):
            raise ParamError(f"--detectors-from: {args.detectors_from}/detectors lacks a version for every model")
    if args.init_from is None or not args.adaptive:
        return out
    src = Path(args.init_from)
    if args.init_side in ("both", "blue") and args.detectors_from is None:
        out["detectors"] = latest_detectors(src)
    if args.init_side in ("both", "red") and (src / "agents" / "red_evasion.json").is_file():
        out["red_evasion"] = src / "agents" / "red_evasion.json"
    return out


def build_detectors(args: argparse.Namespace, seed: int, init_paths: dict[str, Path], det_dir: Path,
                    frozen: bool = False) -> dict:  # fmt: skip
    """One adaptive detector per sensor model: fresh from the Phase 2 model, or the warm-start version.

    ``frozen``: v0 detectors that are never updated (the ``--no-adaptive`` disguised-attacker eval env)."""
    if args.stub:
        from cyberarena.arena.stub import StubAdaptiveDetector as Det
    else:
        from cyberarena.ml.adaptive import EVASION_LEVELS as ML_LEVELS
        from cyberarena.ml.adaptive import AdaptiveDetector as Det

        if tuple(ML_LEVELS) != EVASION_LEVELS:
            raise RuntimeError(f"ml.adaptive EVASION_LEVELS {ML_LEVELS} != arena {EVASION_LEVELS}")
    dets = {}
    for i, m in enumerate(MODELS):
        if m in init_paths:
            det = Det.load(init_paths[m])
            copy_detector(init_paths[m], det_dir)  # this run's turn records reference that version
            if det.lr != args.detector_lr:
                det.lr = args.detector_lr
                opt = getattr(getattr(det, "model", None), "optimizer", None)
                if opt is not None:
                    opt.learning_rate = args.detector_lr
        else:
            kw = {"compile": False} if frozen and not args.stub else {}
            det = Det.from_pretrained(m, lr=args.detector_lr, replay_frac=0.5, seed=seed + i, **kw)
        dets[m] = det
    return dets


def main(argv: list[str] | None = None) -> Path | None:
    t0 = time.time()
    args = build_parser().parse_args(argv)
    if args.describe_params:
        print(json.dumps(spec(), indent=1), flush=True)
        return None
    try:
        cfg, init_q = _validate(args)
    except ParamError as e:
        print(f"{PROG}: error: {e}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None

    if args.runs_dir is None:
        from cyberarena.config import RUNS_DIR

        runs_dir = Path(RUNS_DIR)
    else:
        runs_dir = Path(args.runs_dir)
    while True:  # run_id = YYYYmmdd-HHMMSS-<seed>; on a same-second clash wait for the next second
        run_dir = (runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.seed}").resolve()
        try:
            (run_dir / "agents").mkdir(parents=True)
            break
        except FileExistsError:
            time.sleep(0.2)
    progress = Progress(run_dir / "progress.json", args.episodes)
    print(f"RUN_DIR {run_dir}", flush=True)

    handlers: dict[int, Any] = {}
    cancel_requested = threading.Event()
    if threading.current_thread() is threading.main_thread():

        def on_signal(signum, frame):
            cancel_requested.set()  # the raise may land inside TF/Keras code that wraps it in another error
            raise Cancelled(signal.Signals(signum).name)

        for name in ("SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                handlers[sig] = signal.signal(sig, on_signal)
    try:
        _train(args, cfg, init_q, run_dir, progress, t0)
    except (KeyboardInterrupt, Cancelled):
        progress.finish("error", "cancelled")
        raise
    except BaseException as e:
        if cancel_requested.is_set():
            progress.finish("error", "cancelled")
            raise Cancelled("cancelled") from e
        progress.finish("error", f"{type(e).__name__}: {e}"[:500])
        raise
    else:
        progress.finish("done")
    finally:
        for sig, h in handlers.items():
            signal.signal(sig, h)
    return run_dir


def _train(args: argparse.Namespace, cfg: ArenaConfig, init_q: dict[str, dict], run_dir: Path,
           progress: Progress, t0: float) -> None:  # fmt: skip
    run_id = run_dir.name

    def say(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    adaptive = bool(args.adaptive)
    try:
        init_ad = init_adaptation(args)
    except ParamError as e:
        print(f"{PROG}: error: {e}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None
    # new seed streams for v2 parts; the v1 streams below are untouched so --no-adaptive reproduces v1
    s_det, s_bandit = (int(x.generate_state(1)[0]) for x in np.random.SeedSequence([args.seed, 2]).spawn(2))
    dets = None
    classifiers = None
    det_dir = run_dir / "detectors"
    if adaptive:
        dets = build_detectors(args, s_det % 100_000, init_ad["detectors"], det_dir)
    elif args.stub:
        from cyberarena.arena.stub import stub_classifiers

        classifiers = stub_classifiers()
    env = CyberArenaEnv(cfg, graph_seed=args.seed, classifiers=classifiers, n_nodes=args.n_nodes,
                        detectors=dets, evasion_cost=args.evasion_cost if adaptive else 0.0,
                        adaptive_pool_size=args.adaptive_pool_size)  # fmt: skip
    # disguised-attacker eval env: own pools (seed stream [seed, 3]), the run's detectors (frozen run: v0)
    eval_env = None
    if args.evasive_levels:
        s_evpool = int(np.random.SeedSequence([args.seed, 3]).generate_state(1)[0]) % 2**31
        eval_dets = dets if adaptive else build_detectors(args, s_det % 100_000, {}, det_dir, frozen=True)
        eval_env = CyberArenaEnv(cfg, graph_seed=args.seed, n_nodes=args.n_nodes, detectors=eval_dets,
                                 evasion_cost=args.evasion_cost, adaptive_pool_size=args.adaptive_pool_size,
                                 pool_seed=s_evpool, partition="arena_eval")  # fmt: skip
    for e in (env, eval_env):
        if e is not None:
            e.blue_score_view = args.blue_score_view
    say(f"[arena] run {run_id}: {env.n} nodes, {len(env.graph.edges)} edges, crown jewel = node {env.crown}"
        f"{' (adaptive)' if adaptive else ''} (setup {time.time() - t0:.1f}s)")  # fmt: skip

    ss = np.random.SeedSequence(args.seed)
    s_red, s_blue, s_bred, s_bblue, s_sched = (int(x.generate_state(1)[0]) for x in ss.spawn(5))
    dqn = args.agent == "dqn"
    if dqn:
        dcfg = DQNConfig(lr=args.dqn_lr, hidden=args.dqn_hidden_t, gamma=args.dqn_gamma, n_step=args.n_step,
                         replay_size=args.replay_size, batch_size=args.batch_size, target_sync=args.target_sync,
                         train_every=args.train_every,
                         learn_start=min(500, 10 * args.batch_size, args.replay_size))  # fmt: skip
        init_w = {s: DQNAgent.load_weights(Path(args.init_from) / "agents" / f"{s}_qnet") for s in init_q}
        red = DQNAgent("red", seed=s_red, cfg=dcfg, epsilon=args.eps_start, weights=init_w.get("red"))
        blue = DQNAgent("blue", seed=s_blue, cfg=dcfg, epsilon=args.eps_start, weights=init_w.get("blue"))
    else:
        red = QAgent("red", seed=s_red, alpha=args.alpha, gamma=args.gamma, epsilon=args.eps_start)
        blue = QAgent("blue", seed=s_blue, alpha=args.alpha, gamma=args.gamma, epsilon=args.eps_start)
        for side, agent in (("red", red), ("blue", blue)):
            if side in init_q:
                agent.q = {k: v.copy() for k, v in init_q[side].items()}
    bandit = EvasionBandit(s_bandit, lr=args.red_evasion_lr, explore=args.red_evasion_explore,
                           detect_penalty=args.red_detect_penalty)  # fmt: skip
    if adaptive and init_ad["red_evasion"] is not None:
        bandit.load_state(init_ad["red_evasion"])
    trainer = DetectorTrainer(dets, env, det_dir, args.detector_min_samples) if adaptive else None
    if init_q or init_ad["detectors"] or init_ad["red_evasion"]:
        src_name = Path(args.init_from if args.init_from is not None else args.detectors_from).resolve().name
        say(f"[arena] warm start from {src_name}: "
            + ", ".join(f"{s} Q-network ({q['grad_steps']} steps)" if dqn else f"{s} {len(q)} states"
                        for s, q in init_q.items())
            + "".join(f", {m} detector v{d.version}" for m, d in (dets or {}).items() if m in init_ad["detectors"])
            + (f", red evasion {bandit.levels()}" if init_ad["red_evasion"] else ""))  # fmt: skip
    base_red = make_baseline("red", args.baseline, s_bred, args.red_baseline_noise)
    base_blue = make_baseline("blue", args.baseline, s_bblue, args.blue_baseline_noise)
    sched = np.random.default_rng(s_sched)

    (run_dir / "graph.json").write_text(json.dumps(env.graph.to_json(), indent=1))
    if args.stub:
        clf_info: Any = "stub"
    else:
        clf_info = {m: "artifacts/models/" + m + ".keras" for m in MODELS}
    init_from = None
    if args.init_from is not None:
        init_from = {"run_id": Path(args.init_from).resolve().name, "side": args.init_side,
                     "detectors": {m: dets[m].version for m in init_ad["detectors"]} if dets else {},
                     "red_evasion": bandit.levels() if init_ad["red_evasion"] is not None else None}  # fmt: skip
    if dqn:
        agents_cfg: dict[str, Any] = {
            "type": "dqn", "kind": "dqn", **dcfg.to_dict(), "double_dqn": True, "loss": "huber(delta=1)",
            "optimizer": "adam(clipnorm=10)", "target_network": "frozen copy of the online weights, synced every "
                                                               "target_sync gradient steps",
            "inference": "numpy mirror of the keras weights, synced after every train_on_batch",
            "exploration": "epsilon-greedy: with prob. epsilon a uniform valid action type, then a uniform "
                           "candidate of that action; otherwise argmax Q over all candidate moves",
            "epsilon": {"start": args.eps_start, "end": args.eps_end, "decay_frac": args.eps_decay_frac,
                        "schedule": "linear"},
            "candidates": "every legal (action, source, target) from env.candidates over valid_actions",
            "feature_names": {s: list(v) for s, v in FEATURE_NAMES.items()},
            "files": "agents/{side}_qnet.keras + .json; checkpoints agents/checkpoints/{side}_qnet_<after>.keras",
        }  # fmt: skip
    else:
        agents_cfg = {"type": "tabular", "kind": "tabular_q",
                      "alpha": {"start": args.alpha, "end": args.alpha_end or args.alpha, "schedule": "linear"},
                      "gamma": args.gamma,
                      "replay": {"per_episode": args.replay, "buffer": 20000, "sampling": "uniform"},
                      "epsilon": {"start": args.eps_start, "end": args.eps_end, "decay_frac": args.eps_decay_frac,
                                  "schedule": "linear"},
                      "red_features": ["stage", "front_privilege", "n_footholds", "progress_move", "recon_front",
                                       "hot", "turn_bucket"],
                      "blue_features": ["n_detected_live", "top_undetected_score_bucket", "n_isolated",
                                        "crown_alert", "turn_bucket"]}  # fmt: skip
    partition_sizes = None
    if dets is not None and all(hasattr(d, "partition_sizes") for d in dets.values()):
        partition_sizes = {m: d.partition_sizes() for m, d in dets.items()}
    config = {
        "run_id": run_id, "seed": args.seed, "episodes": args.episodes, "graph_seed": args.seed,
        "n_nodes": env.n, "crown_jewel": env.crown, "label": args.label, "adaptive": adaptive,
        "init_from": init_from,
        "params": {p.key: getattr(args, p.dest) for p in CLI_PARAMS},
        "classifiers": clf_info,
        "env": cfg.to_dict(),
        "agents": agents_cfg,
        "partition_sizes": partition_sizes,
        "pools": {"train_partition": env.pool.partition,
                  "eval_partition": eval_env.pool.partition if eval_env is not None else None,
                  "oversized": "pool_rows samples with replacement when a pool is larger than its partition"},
        "baseline": {"kind": args.baseline, "red_noise": args.red_baseline_noise,
                     "blue_noise": args.blue_baseline_noise},
        "train_matchups": {"cycle": list(TRAIN_MATCHUPS), "weights": [0.5, 0.25, 0.25]},
        "eval": {"every": args.eval_every, "n": args.eval_n, "log_n": args.eval_log_n, "greedy": True,
                 "env_seeds": f"{EVAL_SEED_OFFSET}+j, j<n (same seeds at every checkpoint)",
                 "episode_ids": "eval episodes are numbered after the training episodes (>= episodes)",
                 "probe_game": {"env_seed": PROBE_GAME_SEED, "agent_rng_seed": PROBE_GAME_SEED,
                                "note": "one extra game per matchup per checkpoint, logged in full with "
                                        "probe_game: true, not counted in win rates; eval rows carry "
                                        "probe_episode"}},
        "eval_evasive": {
            "matchup": EVASIVE_MATCHUP, "levels": list(args.evasive_levels), "n": args.eval_evasive_n,
            "opponent": "scripted red at fixed evasion s on every sensor; learned blue greedy",
            "env": "separate eval env: same graph, own sensor pools (fresh benign/malicious pairings, seed "
                   "stream [seed, 3]), evasion_cost applied, scores from this run's current detectors "
                   "(v0 when adaptive is off); never used for training or detector labels",
            "env_seeds": f"{EVAL_SEED_OFFSET}+j, j<n", "turn_log": "probe game only (one per level)",
            "evasion_cost": args.evasion_cost, "pool_size": args.adaptive_pool_size},
        "logging": {"episodes_jsonl": "full turn records for training episodes with episode % log_every == 0 "
                                      "and for the first eval.log_n eval episodes per matchup per checkpoint",
                    "log_every": args.log_every,
                    "classifier_inputs": "only rows emitted this turn (acted-on node), rounded to 3 dp (scores 4 dp)",
                    "extra_turn_fields": ["phase", "matchup", "agent", "after_episode (eval only)",
                                          "detector_versions", "red_evasion", "probe_game"],
                    "checkpoints": "agents/checkpoints/{red,blue}_<after_episode>.json at every eval",
                    "learning_jsonl": "detector_update / red_evasion / agent_stats / probe rows"},
    }  # fmt: skip
    if args.detectors_from is not None and dets:
        config["detectors_from"] = {"run_id": Path(args.detectors_from).resolve().name,
                                    "versions": {m: dets[m].version for m in init_ad["detectors"]}}  # fmt: skip
    if adaptive:
        config["adaptation"] = {
            "evasion_levels": list(EVASION_LEVELS),
            "pool_size": args.adaptive_pool_size,
            "evasion": "malicious row = (1-s)*x_mal + s*x_ben (abstract feature-space blend); red success "
                       "probability x (1 - evasion_cost * s) for actions watched by that sensor",
            "action_sensor": {"recon": "network", "phish": "phishing", "exploit": "network",
                              "escalate": "malware", "lateral_move": "network", "exfiltrate": "network"},
            "red_bandit": {"kind": "epsilon_greedy", "lr": args.red_evasion_lr, "explore": args.red_evasion_explore,
                           "detect_penalty": args.red_detect_penalty, "init_values": 0.0,
                           "payoff": "(progress from actions on that sensor - detect_penalty * flagged malicious "
                                     "readings) / actions, per game; learned red only, scripted red stays at 0"},
            "revealed_labels": "isolate / restore / reset_credentials on a host, or monitor on a host already "
                               "proven compromised, reveal the true labels of that host's last rows "
                               "(training games only)",
            "detector_update": {"every": args.detector_update_every, "min_samples": args.detector_min_samples,
                                "min_each_class": max(1, args.detector_min_samples // 8),
                                "buffer_per_class": 256, "lr": args.detector_lr, "replay_frac": 0.5},
            "detectors_dir": "detectors/<model>_v<version>.keras",
            "initial_eval": trainer.initial_eval if trainer else None,
        }  # fmt: skip
    (run_dir / "config.json").write_text(json.dumps(config, indent=1))

    alpha_end = args.alpha if args.alpha_end is None else args.alpha_end

    def alpha_at(ep: int) -> float:
        return args.alpha + (alpha_end - args.alpha) * ep / max(1, args.episodes - 1)

    eval_episode_id = args.episodes
    history: list[dict[str, Any]] = []
    logs: dict[str, TextIO] = {}
    zero_idx = {m: 0 for m in MODELS}

    def learn_row(row: dict[str, Any]) -> None:
        logs["learning"].write(json.dumps(row, separators=(",", ":")) + "\n")

    def set_evasion(idx: dict[str, int]) -> None:
        env.evasion_idx[:] = [idx[m] for m in MODELS]

    def probe_game(matchup: str, agents: dict[str, BaseAgent], after: int, episode: int,
                   game_env: CyberArenaEnv | None = None, extra: dict[str, Any] | None = None) -> None:  # fmt: skip
        """Same env seed and agent RNG seeds every checkpoint: only the agents' choices differ."""
        states = [a.rng.bit_generator.state for a in agents.values()]
        for k, a in enumerate(agents.values()):
            a.rng.bit_generator.state = np.random.default_rng([PROBE_GAME_SEED, k]).bit_generator.state
        play_episode(game_env or env, agents, PROBE_GAME_SEED, learn=set(), explore=False,
                     log=None if args.no_turn_log else logs["episodes"], run_id=run_id, episode=episode,
                     extra={"phase": "eval", "matchup": matchup, "after_episode": after,
                            "probe_game": True, **(extra or {})})  # fmt: skip
        for a, st in zip(agents.values(), states, strict=True):
            a.rng.bit_generator.state = st

    def evaluate(after: int) -> None:
        nonlocal eval_episode_id
        progress.update(phase="eval")
        # eval must not perturb training: snapshot every agent RNG and restore it afterwards
        rng_states = [a.rng.bit_generator.state for a in (red, blue, base_red, base_blue)]
        for matchup in EVAL_MATCHUPS:
            if matchup.startswith("red"):
                agents = {"red": red, "blue": base_blue}
                set_evasion(bandit.greedy() if args.red_evades else zero_idx)
            else:
                agents = {"red": base_red, "blue": blue}
                set_evasion(zero_idx)
            first = eval_episode_id
            wins = {"red": 0, "blue": 0}
            for j in range(args.eval_n):
                logged = not args.no_turn_log and j < args.eval_log_n
                res = play_episode(env, agents, EVAL_SEED_OFFSET + j, learn=set(), explore=False,
                                   log=logs["episodes"] if logged else None, run_id=run_id,
                                   episode=eval_episode_id,
                                   extra={"phase": "eval", "matchup": matchup, "after_episode": after})  # fmt: skip
                wins[res["winner"]] += 1
                eval_episode_id += 1
            probe_ep = eval_episode_id
            probe_game(matchup, agents, after, probe_ep)
            eval_episode_id += 1
            row = {"kind": "eval", "after_episode": after, "matchup": matchup, "n": args.eval_n,
                   "red_win_rate": round(wins["red"] / args.eval_n, 4),
                   "blue_win_rate": round(wins["blue"] / args.eval_n, 4),
                   "episodes": [first, first + args.eval_n - 1],
                   "logged_n": 0 if args.no_turn_log else min(args.eval_n, args.eval_log_n),
                   "probe_episode": probe_ep}  # fmt: skip
            logs["summary"].write(json.dumps(row) + "\n")
            history.append(row)
        evasive: dict[float, float] = {}
        for s in args.evasive_levels:
            eval_env.evasion_idx[:] = EVASION_LEVELS.index(s)
            agents = {"red": base_red, "blue": blue}
            first = eval_episode_id
            wins = {"red": 0, "blue": 0}
            for j in range(args.eval_evasive_n):  # same env seeds as the other matchups
                res = play_episode(eval_env, agents, EVAL_SEED_OFFSET + j, learn=set(), explore=False)
                wins[res["winner"]] += 1
                eval_episode_id += 1
            probe_ep = eval_episode_id
            probe_game(EVASIVE_MATCHUP, agents, after, probe_ep, game_env=eval_env, extra={"evasion": s})
            eval_episode_id += 1
            n = args.eval_evasive_n
            row = {"kind": "eval", "after_episode": after, "matchup": EVASIVE_MATCHUP, "evasion": s, "n": n,
                   "red_win_rate": round(wins["red"] / n, 4), "blue_win_rate": round(wins["blue"] / n, 4),
                   "episodes": [first, first + n - 1], "logged_n": 0, "probe_episode": probe_ep,
                   "detector_versions": dict(eval_env.pool.version)}  # fmt: skip
            logs["summary"].write(json.dumps(row) + "\n")
            history.append(row)
            evasive[s] = row["blue_win_rate"]
        for agent in (red, blue):
            rows = snapshot_probe_rows(agent, env, after) if dqn else probe_rows(agent, after)
            for prow in rows:
                learn_row(prow)
        latest = {r["matchup"]: r for r in history if r["after_episode"] == after and "evasion" not in r}
        rl = latest["red_learned_vs_blue_baseline"]["red_win_rate"]
        bl = latest["blue_learned_vs_red_baseline"]["blue_win_rate"]
        last_eval: dict[str, Any] = {"after_episode": after, "red_win_rate": rl, "blue_win_rate": bl}
        if evasive:
            last_eval["evasive_blue_win_rate"] = {f"{s:.1f}": v for s, v in evasive.items()}
        progress.update(phase="train", last_eval=last_eval)
        lv = (" | red evasion " + " ".join(f"{m[:3]}={s:.1f}" for m, s in bandit.levels().items())
              + " | det v" + "/".join(str(env.pool.version[m]) for m in MODELS)) if adaptive else ""
        ev = (" | vs evasive red " + " ".join(f"{s:.1f}:{v:.0%}" for s, v in evasive.items())) if evasive else ""
        say(f"[eval] after {after:5d}: learned red vs baseline blue red-win {rl:5.1%} | "
            f"learned blue vs baseline red blue-win {bl:5.1%}{ev}{lv} | {time.time() - t0:6.1f}s")  # fmt: skip
        for f in logs.values():
            f.flush()
        ckpt = run_dir / "agents" / "checkpoints"
        ckpt.mkdir(exist_ok=True)
        if dqn:
            red.save(ckpt / f"red_qnet_{after:05d}")
            blue.save(ckpt / f"blue_qnet_{after:05d}")
        else:
            red.save(ckpt / f"red_{after:05d}.json")
            blue.save(ckpt / f"blue_{after:05d}.json")
        if adaptive:
            bandit.save(ckpt / f"red_evasion_{after:05d}.json")
        for a, st in zip((red, blue, base_red, base_blue), rng_states, strict=True):
            a.rng.bit_generator.state = st

    # rolling windows for agent_stats / red_evasion rows
    def new_window() -> dict[str, Any]:
        return {"games": {"red": 0, "blue": 0}, "wins": {"red": 0, "blue": 0}, "red_games": 0,
                "sensor": {m: {"n_act": 0, "success": 0, "malicious": 0, "caught": 0} for m in MODELS}}  # fmt: skip

    window = new_window()

    def evasion_row(episode: int) -> dict[str, Any]:
        lv = bandit.levels()
        sens = window["sensor"]
        return {"kind": "red_evasion", "episode": episode, "levels": lv,
                "detector_versions": dict(env.pool.version),
                "recall_at_level": {m: trainer.recall_at(m, lv[m]) for m in MODELS},
                "caught_rate": {m: round(v["caught"] / v["malicious"], 4) if v["malicious"] else None
                                for m, v in sens.items()},
                "success_rate": {m: round(v["success"] / v["n_act"], 4) if v["n_act"] else None
                                 for m, v in sens.items()},
                "games": window["red_games"],
                "values": {m: [round(float(x), 4) for x in bandit.values[m]] for m in MODELS}}  # fmt: skip

    def stats_rows(episode: int) -> None:
        for side, agent in (("red", red), ("blue", blue)):
            g = window["games"][side]
            learn_row({"kind": "agent_stats", "episode": episode, "side": side, **agent.stats(), "games": g,
                       "win_rate": round(window["wins"][side] / g, 4) if g else None})  # fmt: skip
            agent.reset_stats()
        if adaptive:
            learn_row(evasion_row(episode))

    with (open(run_dir / "episodes.jsonl", "w", encoding="utf-8") as episodes_f,
          open(run_dir / "summary.jsonl", "w", encoding="utf-8") as summary_f,
          open(run_dir / "learning.jsonl", "w", encoding="utf-8") as learning_f):  # fmt: skip
        logs.update(episodes=episodes_f, summary=summary_f, learning=learning_f)
        if adaptive:
            learn_row(evasion_row(0))
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
            chosen = bandit.choose() if args.red_evades and "red" in learn else zero_idx
            set_evasion(chosen)
            logged = ep % args.log_every == 0 and not args.no_turn_log
            res = play_episode(env, agents, args.seed * 100_003 + ep, learn=learn, explore=True,
                               log=episodes_f if logged else None, run_id=run_id, episode=ep,
                               extra={"phase": "train", "matchup": matchup}, replay=args.replay)  # fmt: skip
            row = {"kind": "episode", "episode": ep, **res, "epsilon": round(eps, 4), "matchup": matchup,
                   "logged": logged}  # fmt: skip
            if adaptive:
                row["red_evasion"] = {m: EVASION_LEVELS[chosen[m]] for m in MODELS}
            summary_f.write(json.dumps(row) + "\n")
            for side in learn:
                window["games"][side] += 1
                window["wins"][side] += int(res["winner"] == side)
            if adaptive:
                trainer.add_revealed(env.revealed)
                if "red" in learn:
                    if args.red_evades:
                        bandit.update(chosen, env.red_stats)
                    window["red_games"] += 1
                    for m in MODELS:
                        for k in ("n_act", "success", "malicious", "caught"):
                            window["sensor"][m][k] += env.red_stats[m][k]
                if args.detectors_adapt and (ep + 1) % args.detector_update_every == 0:
                    for urow in trainer.maybe_update(ep + 1, bandit.levels()):
                        learn_row(urow)
                        if eval_env is not None:
                            eval_env.pool.rescore(urow["model"], dets[urow["model"]])
                        say(f"[adapt] after {ep + 1:5d}: {urow['model']} v{urow['version']} on "
                            f"{urow['n_malicious']}+{urow['n_benign']} revealed rows, loss "
                            f"{urow['loss_before']:.3f}->{urow['loss_after']:.3f} ({urow['seconds']:.2f}s)")
            if (ep + 1) % args.stats_every == 0:
                stats_rows(ep + 1)
                window = new_window()
            progress.update(episode=ep + 1)
            if (ep + 1) % args.eval_every == 0 or ep + 1 == args.episodes:
                evaluate(ep + 1)

    if dqn:
        red.save(run_dir / "agents" / "red_qnet")
        blue.save(run_dir / "agents" / "blue_qnet")
    else:
        red.save(run_dir / "agents" / "red.json")
        blue.save(run_dir / "agents" / "blue.json")
    if adaptive:
        bandit.save(run_dir / "agents" / "red_evasion.json")
        config["adaptation"]["final"] = {"detector_versions": dict(env.pool.version),
                                         "red_evasion": bandit.levels()}  # fmt: skip
    wall = time.time() - t0
    size_mb = (run_dir / "episodes.jsonl").stat().st_size / 1e6
    config["wall_time_s"] = round(wall, 1)
    (run_dir / "config.json").write_text(json.dumps(config, indent=1))
    learned = (f"grad steps red={red.grad_steps} blue={blue.grad_steps}" if dqn
               else f"Q states red={len(red.q)} blue={len(blue.q)}")  # fmt: skip
    say(f"[arena] done in {wall:.1f}s -> {run_dir} (episodes.jsonl {size_mb:.1f} MB, {learned})")


def cli() -> int:
    try:
        main()
    except (KeyboardInterrupt, Cancelled):
        print(f"{PROG}: cancelled", file=sys.stderr, flush=True)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(cli())
