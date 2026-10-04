"""Enrich chosen episodes of a run with SHAP, MITRE tags and move rationale.

    python -m cyberarena.explain.enrich --run runs/<run_id> [--episodes SPEC] [--online] [--model ID]

``--episodes`` SPEC (default ``default``):
  default          3 eval episodes per learned-vs-baseline matchup from the final checkpoint, with at least one
                   red win and one blue win in each (see ``select_default``); in adaptive (v2) runs also the
                   probe games (``probe_game: true``) of the first and last checkpoints, one per
                   learned-vs-baseline matchup; in v3/v4 runs also the evading-red probe game at evasion 0.7 from
                   the final checkpoint (adapted detectors vs a disguised red). At most 14 episodes.
  2803,2850-2852   explicit episode ids / ranges
  last:N           the last N logged episodes
  eval:A           the default pick, but from the eval checkpoint ``after_episode == A``

Writes ``runs/<run_id>/episodes_enriched.jsonl`` (only the selected episodes' turns, in episode/turn order) and
``runs/<run_id>/explain_summary.json`` (selection, timings, SHAP checks, cost estimate). Caches live under
``runs/<run_id>/explain_cache/``. ``episodes.jsonl`` is streamed and indexed by byte offset, never loaded whole.

v4 DQN turns also get ``agent_attribution`` (integrated gradients on the deciding Q-network checkpoint; see
``cyberarena.explain.attribution``) and a "why this host" rationale. One enrich per run at a time: the process holds
``explain_cache/.lock`` (``--lock-timeout`` seconds of waiting, then a clean exit with code 3).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from cyberarena.explain import mitre
from cyberarena.explain.adaptation import AdaptationTracker, LearningLog, RedMoveHistory
from cyberarena.explain.attribution import AttributionService, GraphInfo
from cyberarena.explain.cachelock import CacheLock, LockTimeoutError
from cyberarena.explain.narrate import (
    DEFAULT_MODEL,
    MODEL_ENV,
    NarrationError,
    Narrator,
    TurnFacts,
    build_prompt,
    estimate_cost,
    require_credentials,
    resolve_model,
)
from cyberarena.explain.shap_values import ShapService, ShapSettings

LEARNED_VS_BASELINE = ("red_learned_vs_blue_baseline", "blue_learned_vs_red_baseline")
EVASIVE_MATCHUP = "blue_learned_vs_red_evasive"
EVASIVE_LEVEL = 0.7
PER_MATCHUP = 3
MAX_DEFAULT_EPISODES = 14
LOCK_EXIT = 3


# ------------------------------------------------------------------------------------------------ indexing


@dataclass
class EpisodeMeta:
    episode: int
    phase: str | None = None
    matchup: str | None = None
    after_episode: int | None = None
    winner: str | None = None
    turns: int = 0
    probe_game: bool = False
    evasion: float | None = None  # evading-red eval games (v3): red's fixed evasion level
    offsets: list[int] = field(default_factory=list)


def index_episodes(path: Path, history: RedMoveHistory | None = None) -> dict[int, EpisodeMeta]:
    """One streaming pass: per-episode metadata plus the byte offset of every turn line.

    ``history``, if given, also records red's per-host moves (v2 adaptation facts) during the same pass.
    """
    index: dict[int, EpisodeMeta] = {}
    with path.open("rb") as fh:
        pos = 0
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                ep = rec["episode"]
                m = index.get(ep)
                if m is None:
                    m = index[ep] = EpisodeMeta(
                        ep, rec.get("phase"), rec.get("matchup"), rec.get("after_episode")
                    )
                m.turns += 1
                m.offsets.append(pos)
                if rec.get("probe_game"):
                    m.probe_game = True
                if rec.get("evasion") is not None and m.evasion is None:
                    m.evasion = float(rec["evasion"])
                if history is not None:
                    history.observe(rec)
                if rec.get("done"):
                    m.winner = rec.get("winner")
            pos += len(line)
    return index


def read_turns(path: Path, offsets: list[int]) -> list[dict]:
    out = []
    with path.open("rb") as fh:
        for off in offsets:
            fh.seek(off)
            out.append(json.loads(fh.readline()))
    return out


# ----------------------------------------------------------------------------------------------- selection


def _median(items: list[EpisodeMeta]) -> EpisodeMeta:
    items = sorted(items, key=lambda m: (m.turns, m.episode))
    return items[(len(items) - 1) // 2]


def pick_matchup(eps: list[EpisodeMeta], k: int = PER_MATCHUP) -> list[tuple[EpisodeMeta, str]]:
    """Pick k episodes with at least one red win and one blue win, each with a reason.

    1. the median-length red win; 2. the median-length blue win;
    3. a blue win by eviction (ended before the turn limit) of >= 10 turns, longest first; otherwise the
       shortest red win (red's most efficient attack); then further medians of what is left.
    """
    red = [m for m in eps if m.winner == "red"]
    blue = [m for m in eps if m.winner == "blue"]
    chosen: list[tuple[EpisodeMeta, str]] = []

    def take(m: EpisodeMeta | None, why: str) -> None:
        if m is not None and m not in [c for c, _ in chosen] and len(chosen) < k:
            chosen.append((m, why))

    if red:
        take(_median(red), "median-length red win")
    if blue:
        take(_median(blue), "median-length blue win")
    limit = max((m.turns for m in eps), default=0)
    evictions = sorted(
        [m for m in blue if m.turns < limit and m.turns >= 10], key=lambda m: (-m.turns, m.episode)
    )
    if evictions:
        take(evictions[0], "blue win by eviction before the turn limit")
    elif red:
        take(min(red, key=lambda m: (m.turns, m.episode)), "shortest red win")
    rest = [m for m in eps if m not in [c for c, _ in chosen] and m.winner]
    while len(chosen) < k and rest:
        m = _median(rest)
        take(m, f"median-length remaining episode ({m.winner} win)")
        rest.remove(m)
    return chosen


def select_default(
    index: dict[int, EpisodeMeta], after_episode: int | None = None
) -> list[tuple[EpisodeMeta, str]]:
    evals = [m for m in index.values() if m.phase == "eval" and m.after_episode is not None]
    if not evals:
        raise SystemExit("no eval episodes with turn records in this run; pass --episodes explicitly")
    ckpt = max(m.after_episode for m in evals) if after_episode is None else after_episode
    picked: list[tuple[EpisodeMeta, str]] = []
    for mu in LEARNED_VS_BASELINE:
        eps = [m for m in evals if m.after_episode == ckpt and m.matchup == mu]
        regular = [m for m in eps if not m.probe_game] or eps
        if not eps:
            raise SystemExit(f"no eval episodes for {mu} at after_episode={ckpt}")
        picked += pick_matchup(regular)
    if after_episode is None:
        for m, why in select_probes(index) + select_evasive_probe(index):
            if m.episode not in {p.episode for p, _ in picked} and len(picked) < MAX_DEFAULT_EPISODES:
                picked.append((m, why))
    return picked


def is_evasive(m: EpisodeMeta) -> bool:
    return m.matchup == EVASIVE_MATCHUP or bool(m.evasion)


def select_probes(index: dict[int, EpisodeMeta]) -> list[tuple[EpisodeMeta, str]]:
    """v2 runs: the probe game of each matchup at the first and the last checkpoint (same env seed each time).
    Evading-red probe games are left to ``select_evasive_probe``."""
    probes = [m for m in index.values() if m.probe_game and m.after_episode is not None and not is_evasive(m)]
    if not probes:
        return []
    first, last = min(m.after_episode for m in probes), max(m.after_episode for m in probes)
    out: list[tuple[EpisodeMeta, str]] = []
    for ckpt, label in ((first, "first"), (last, "last")) if first != last else ((last, "only"),):
        by_mu: dict[str, EpisodeMeta] = {}
        for m in sorted(probes, key=lambda m: m.episode):
            if m.after_episode == ckpt:
                by_mu.setdefault(m.matchup or "", m)
        out += [(m, f"probe game, {label} checkpoint (after {ckpt})") for _, m in sorted(by_mu.items())]
    return out


def select_evasive_probe(index: dict[int, EpisodeMeta], level: float = EVASIVE_LEVEL
                         ) -> list[tuple[EpisodeMeta, str]]:  # fmt: skip
    """v3/v4 runs: the evading-red probe game at ``level`` from the final checkpoint: learned blue with its adapted
    detectors against the scripted red disguising every sensor at that evasion."""
    probes = [m for m in index.values() if m.probe_game and m.after_episode is not None and is_evasive(m)
              and m.evasion is not None and abs(m.evasion - level) < 1e-6]  # fmt: skip
    if not probes:
        return []
    last = max(m.after_episode for m in probes)
    m = min((m for m in probes if m.after_episode == last), key=lambda m: m.episode)
    return [(m, f"evading-red probe game at evasion {level:.1f}, final checkpoint (after {last})")]


def parse_episodes(spec: str, index: dict[int, EpisodeMeta]) -> list[tuple[EpisodeMeta, str]]:
    spec = spec.strip()
    if spec == "default":
        return select_default(index)
    if spec.startswith("eval:"):
        return select_default(index, int(spec.split(":", 1)[1]))
    if spec.startswith("last:"):
        n = int(spec.split(":", 1)[1])
        return [(index[e], "last logged") for e in sorted(index)[-n:]]
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            ids += range(int(a), int(b) + 1)
        elif part:
            ids.append(int(part))
    missing = [e for e in ids if e not in index]
    if missing:
        raise SystemExit(f"episodes without turn records in this run: {missing}")
    return [(index[e], "requested") for e in dict.fromkeys(ids)]


# ------------------------------------------------------------------------------------------------ enrichment


def load_run(run_dir: Path) -> tuple[dict, dict]:
    graph = json.loads((run_dir / "graph.json").read_text(encoding="utf-8"))
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    return graph, cfg


def make_facts(turn: dict, shap: list[dict] | None, tag: dict | None, graph: dict, cfg: dict,
               adaptation: dict | None = None, attribution: dict | None = None,
               ginfo: GraphInfo | None = None) -> TurnFacts:  # fmt: skip
    roles = {n["id"]: n["role"] for n in graph["nodes"]}
    cj = next((n["id"] for n in graph["nodes"] if n.get("crown_jewel")), cfg.get("crown_jewel"))
    env = cfg.get("env", {})
    ginfo = ginfo or GraphInfo.from_graph(graph)
    return TurnFacts(turn=turn, roles=roles, crown_jewel=cj, shap=shap, mitre=tag,
                     detect_threshold=env.get("detect_threshold", 0.5), max_rounds=env.get("max_rounds"),
                     adaptation=adaptation, attribution=attribution, hops=ginfo.dist,
                     max_dist=ginfo.max_dist)  # fmt: skip


@dataclass
class Timing:
    shap_s: float = 0.0
    ig_s: float = 0.0
    turns: int = 0


def enrich_turns(turns: list[dict], graph: dict, cfg: dict, shap_svc: ShapService | None, narrator: Narrator,
                 log=print, tracker: AdaptationTracker | None = None, attrib: AttributionService | None = None,
                 timing: Timing | None = None) -> tuple[list[dict], list[TurnFacts]]:  # fmt: skip
    """``turns`` must be one episode's turns in order (the previous turn is the pre-move state for IG)."""
    tracker = tracker or AdaptationTracker()
    timing = timing if timing is not None else Timing()
    roles = {n["id"]: n["role"] for n in graph["nodes"]}
    ginfo = GraphInfo.from_graph(graph)
    facts: list[TurnFacts] = []
    t0 = time.perf_counter()
    prev = None
    for i, t in enumerate(turns):
        ts = time.perf_counter()
        shap = shap_svc.explain_turn(t) if shap_svc else None
        ti = time.perf_counter()
        attribution = attrib.explain_turn(t, prev) if attrib else None
        timing.shap_s += ti - ts
        timing.ig_s += time.perf_counter() - ti
        timing.turns += 1
        prev = t
        tag = mitre.tag(t["actor"], t["action_id"], roles.get(t.get("target")))
        facts.append(make_facts(t, shap, tag, graph, cfg, tracker.facts(t), attribution, ginfo))
        if (i + 1) % 100 == 0:
            log(f"  shap/mitre {i + 1}/{len(turns)} turns ({time.perf_counter() - t0:.1f}s)")
    results = narrator.narrate_many(facts, progress=lambda d, n: d % 100 == 0 and log(f"  narrated {d}/{n}"))
    out = []
    for f, (text, meta) in zip(facts, results, strict=True):
        rec = dict(f.turn)
        rec["shap"] = f.shap
        rec["mitre"] = f.mitre
        rec["adaptation"] = f.adaptation
        rec["agent_attribution"] = f.attribution
        rec["rationale"] = text
        rec["rationale_meta"] = meta
        out.append(rec)
    return out, facts


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m cyberarena.explain.enrich", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("--run", required=True, type=Path, help="runs/<run_id>")
    ap.add_argument("--episodes", default="default", help="default | eval:A | last:N | 2803,2850-2852")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--online", dest="offline", action="store_false",
                      help="Claude-written rationales via the Anthropic API (needs ANTHROPIC_API_KEY; costs money)")
    mode.add_argument("--offline", dest="offline", action="store_true",
                      help="template rationales, no API calls, no key needed (the default)")
    ap.set_defaults(offline=True)
    ap.add_argument(
        "--model", default=None, help=f"narration model (env {MODEL_ENV}; default {DEFAULT_MODEL})"
    )
    ap.add_argument("--concurrency", type=int, default=4, help="parallel API requests (online)")
    ap.add_argument("--rpm", type=float, default=50.0, help="max API requests per minute (online)")
    ap.add_argument(
        "--nsamples", type=int, default=ShapSettings.nsamples, help="KernelExplainer samples per row"
    )
    ap.add_argument(
        "--background", type=int, default=ShapSettings.background_k, help="k-means background size"
    )
    ap.add_argument("--top-k", type=int, default=ShapSettings.top_k, help="SHAP features kept per row")
    ap.add_argument("--no-shap", action="store_true", help="skip SHAP (shap stays null)")
    ap.add_argument("--no-ig", action="store_true", help="skip integrated gradients (agent_attribution stays null)")
    ap.add_argument("--lock-timeout", type=float, default=300.0,
                    help="seconds to wait for another enrich of this run to finish (0 = exit at once)")
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/episodes_enriched.jsonl")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    run_dir: Path = args.run
    episodes_path = run_dir / "episodes.jsonl"
    if not episodes_path.exists():
        print(f"error: {episodes_path} not found", file=sys.stderr)
        return 2
    model = resolve_model(args.model)
    cache_dir = run_dir / "explain_cache"
    if not args.offline:  # fail fast on missing credentials, before waiting for the lock or any slow work
        try:
            require_credentials()
        except NarrationError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    lock = CacheLock(cache_dir, timeout_s=args.lock_timeout)
    try:
        lock.acquire()
    except LockTimeoutError as e:
        print(f"error: {e}", file=sys.stderr)
        return LOCK_EXIT
    try:
        return _run(args, run_dir, episodes_path, model, cache_dir, lock)
    finally:
        lock.release()


def _run(args, run_dir: Path, episodes_path: Path, model: str, cache_dir: Path, lock: CacheLock) -> int:
    # caches are loaded only now, under the lock, so a waiting run sees everything the previous one wrote
    try:
        narrator = Narrator(offline=args.offline, model=model, cache_path=cache_dir / "narration.jsonl",
                            concurrency=args.concurrency, rpm=args.rpm)  # fmt: skip
    except NarrationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    t_start = time.perf_counter()
    print(f"indexing {episodes_path} ({os.path.getsize(episodes_path) / 1e6:.0f} MB) ...")
    history = RedMoveHistory()
    index = index_episodes(episodes_path, history)
    t_index = time.perf_counter() - t_start
    picked = parse_episodes(args.episodes, index)
    print(f"indexed {len(index)} logged episodes in {t_index:.1f}s; enriching {len(picked)}:")
    for m, why in picked:
        print(f"  episode {m.episode}: {m.matchup}, winner={m.winner}, {m.turns} turns ({why})")

    graph, cfg = load_run(run_dir)
    settings = ShapSettings(nsamples=args.nsamples, background_k=args.background, top_k=args.top_k)
    shap_svc = None if args.no_shap else ShapService(settings, cache_dir / "shap.jsonl", run_dir=run_dir)
    attrib = None if args.no_ig else AttributionService(run_dir, graph, cfg, cache_dir / "agent_ig.jsonl")
    timing = Timing()
    learning_path = run_dir / "learning.jsonl"
    learning = LearningLog.load(learning_path)
    adaptive = len(history) > 0
    if adaptive or learning_path.exists():
        print(f"adaptive run: {len(learning.updates)} detector updates, {len(learning.evasion)} red_evasion rows "
              f"in learning.jsonl; red host moves indexed in {len(history)} (host, episode) pairs")  # fmt: skip

    out_path = args.out or run_dir / "episodes_enriched.jsonl"
    enriched: list[dict] = []
    all_facts: list[TurnFacts] = []
    t_enrich = time.perf_counter()
    try:
        for m, _ in picked:
            turns = read_turns(episodes_path, m.offsets)
            recs, facts = enrich_turns(turns, graph, cfg, shap_svc, narrator,
                                       tracker=AdaptationTracker(learning, history), attrib=attrib,
                                       timing=timing)  # fmt: skip
            enriched += recs
            all_facts += facts
            print(
                f"  episode {m.episode}: {len(recs)} turns enriched ({time.perf_counter() - t_enrich:.1f}s)"
            )
    except NarrationError as e:
        print(f"error: {e} (completed narrations are cached; rerun to resume)", file=sys.stderr)
        return 1
    t_enrich = time.perf_counter() - t_enrich

    tmp = out_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in enriched:
            fh.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")
    tmp.replace(out_path)

    prompts = [build_prompt(f) for f in all_facts]
    per_episode = {}
    for m, _ in picked:
        ps = [p for f, p in zip(all_facts, prompts, strict=True) if f.turn["episode"] == m.episode]
        per_episode[m.episode] = estimate_cost(ps, model)
    s = shap_svc.stats if shap_svc else None
    summary = {
        "run": run_dir.name,
        "output": out_path.name,
        "mode": "offline" if args.offline else "online",
        "narration_model": model,
        "episodes": [
            {
                "episode": m.episode,
                "matchup": m.matchup,
                "after_episode": m.after_episode,
                "winner": m.winner,
                "turns": m.turns,
                "reason": why,
            }
            for m, why in picked
        ],
        "turns": len(enriched),
        "timing_s": {
            "index": round(t_index, 2),
            "enrich": round(t_enrich, 2),
            "total": round(time.perf_counter() - t_start, 2),
            "shap": round(timing.shap_s, 2),
            "ig": round(timing.ig_s, 2),
            "shap_per_turn": round(timing.shap_s / max(1, timing.turns), 4),
            "ig_per_turn": round(timing.ig_s / max(1, timing.turns), 5),
            "lock_wait": round(lock.waited_s, 2),
        },
        "agent_attribution": None if attrib is None else _ig_summary(attrib, enriched),
        "shap": None
        if s is None
        else {
            "settings": vars(settings),
            "rows": s.rows,
            "computed": s.computed,
            "cache_hits": s.cache_hits,
            "seconds_computing": round(s.seconds, 2),
            "seconds_per_row": {mo: round(s.seconds_per_row(mo), 4) for mo in sorted(s.per_model)},
            "max_additivity_error": s.max_additivity_err,
            "versions": sorted({e["version"] for r in enriched for e in r["shap"] or []}),
            "skipped_missing_detector": shap_svc.missing,
        },
        "adaptation": {
            "adaptive": adaptive,
            "learning_jsonl": learning_path.exists(),
            "detector_updates": len(learning.updates),
            "turns_with_adaptation": sum(r["adaptation"] is not None for r in enriched),
            "turns_updated_since_last_move": sum(
                bool(r["adaptation"] and r["adaptation"]["updated_since_last_move"]) for r in enriched
            ),
        },
        "narration": vars(narrator.stats) | {"prompt_chars": None},
        "online_cost_estimate": {"total": estimate_cost(prompts, model), "per_episode": per_episode},
        "mitre": {
            "attack_version": mitre.ATTACK_VERSION,
            "d3fend_version": mitre.D3FEND_VERSION,
            "mapping_version": mitre.MAPPING_VERSION,
        },
    }
    (run_dir / "explain_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print(f"wrote {len(enriched)} turns -> {out_path}")
    if s:
        print(f"SHAP: {s.rows} rows ({s.computed} computed, {s.cache_hits} cached) in {s.seconds:.1f}s; "
              f"s/row {summary['shap']['seconds_per_row']}; max |base+sum(shap)-f(x)| = {s.max_additivity_err:.2e}")  # fmt: skip
    if attrib is not None:
        a = summary["agent_attribution"]
        print(f"IG: {a['turns_attributed']} DQN turns ({a['computed']} computed, {a['cache_hits']} cached), "
              f"{a['models_loaded']} Q-networks loaded in {a['load_seconds']}s; max completeness error "
              f"{a['max_completeness_abs']:.1e} (rel {a['max_completeness_rel']:.1e}); null: {a['null_reasons']}")  # fmt: skip
    print(f"per turn: SHAP {summary['timing_s']['shap_per_turn']}s, IG {summary['timing_s']['ig_per_turn']}s")
    est = summary["online_cost_estimate"]["total"]
    print(f"narration: {summary['mode']}; online estimate for this selection with {model}: {est['calls']} calls, "
          f"~{est['input_tokens']} in / ~{est['output_tokens']} out tokens, ~${est['usd']:.4f}")  # fmt: skip
    print(f"timing: index {t_index:.1f}s, enrich {t_enrich:.1f}s, total {summary['timing_s']['total']:.1f}s")
    return 0


def _ig_summary(attrib: AttributionService, enriched: list[dict]) -> dict:
    s = attrib.stats
    return {
        "method": "integrated_gradients",
        "turns_attributed": sum(r.get("agent_attribution") is not None for r in enriched),
        "computed": s.computed,
        "cache_hits": s.cache_hits,
        "null_reasons": s.nulls,
        "models_loaded": s.models_loaded,
        "load_seconds": round(s.load_seconds, 2),
        "seconds_computing": round(s.seconds, 3),
        "integration": s.steps,
        "baselines": s.baselines,
        "max_completeness_abs": s.max_completeness_abs,
        "max_completeness_rel": s.max_completeness_rel,
        "max_q_vs_logged": round(s.max_q_mismatch, 6),
        "max_baseline_check_error": round(s.max_check_error, 5),
        "checkpoints": sorted({r["agent_attribution"]["checkpoint"] for r in enriched if r.get("agent_attribution")}),
    }


if __name__ == "__main__":
    sys.exit(main())
