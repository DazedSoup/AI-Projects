"""Enrich chosen episodes of a run with SHAP, MITRE tags and move rationale.

    python -m cyberarena.explain.enrich --run runs/<run_id> [--episodes SPEC] [--online] [--model ID]

``--episodes`` SPEC (default ``default``):
  default          3 eval episodes per learned-vs-baseline matchup from the final checkpoint, with at least one
                   red win and one blue win in each (see ``select_default``)
  2803,2850-2852   explicit episode ids / ranges
  last:N           the last N logged episodes
  eval:A           the default pick, but from the eval checkpoint ``after_episode == A``

Writes ``runs/<run_id>/episodes_enriched.jsonl`` (only the selected episodes' turns, in episode/turn order) and
``runs/<run_id>/explain_summary.json`` (selection, timings, SHAP checks, cost estimate). Caches live under
``runs/<run_id>/explain_cache/``. ``episodes.jsonl`` is streamed and indexed by byte offset, never loaded whole.
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
from cyberarena.explain.narrate import (
    DEFAULT_MODEL,
    MODEL_ENV,
    NarrationError,
    Narrator,
    TurnFacts,
    build_prompt,
    estimate_cost,
    resolve_model,
)
from cyberarena.explain.shap_values import ShapService, ShapSettings

LEARNED_VS_BASELINE = ("red_learned_vs_blue_baseline", "blue_learned_vs_red_baseline")
PER_MATCHUP = 3


# ------------------------------------------------------------------------------------------------ indexing


@dataclass
class EpisodeMeta:
    episode: int
    phase: str | None = None
    matchup: str | None = None
    after_episode: int | None = None
    winner: str | None = None
    turns: int = 0
    offsets: list[int] = field(default_factory=list)


def index_episodes(path: Path) -> dict[int, EpisodeMeta]:
    """One streaming pass: per-episode metadata plus the byte offset of every turn line."""
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
        if not eps:
            raise SystemExit(f"no eval episodes for {mu} at after_episode={ckpt}")
        picked += pick_matchup(eps)
    return picked


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


def make_facts(turn: dict, shap: list[dict] | None, tag: dict | None, graph: dict, cfg: dict) -> TurnFacts:
    roles = {n["id"]: n["role"] for n in graph["nodes"]}
    cj = next((n["id"] for n in graph["nodes"] if n.get("crown_jewel")), cfg.get("crown_jewel"))
    env = cfg.get("env", {})
    return TurnFacts(turn=turn, roles=roles, crown_jewel=cj, shap=shap, mitre=tag,
                     detect_threshold=env.get("detect_threshold", 0.5), max_rounds=env.get("max_rounds"))  # fmt: skip


def enrich_turns(turns: list[dict], graph: dict, cfg: dict, shap_svc: ShapService | None, narrator: Narrator,
                 log=print) -> tuple[list[dict], list[TurnFacts]]:  # fmt: skip
    roles = {n["id"]: n["role"] for n in graph["nodes"]}
    facts: list[TurnFacts] = []
    t0 = time.perf_counter()
    for i, t in enumerate(turns):
        shap = shap_svc.explain_turn(t) if shap_svc else None
        tag = mitre.tag(t["actor"], t["action_id"], roles.get(t.get("target")))
        facts.append(make_facts(t, shap, tag, graph, cfg))
        if (i + 1) % 100 == 0:
            log(f"  shap/mitre {i + 1}/{len(turns)} turns ({time.perf_counter() - t0:.1f}s)")
    results = narrator.narrate_many(facts, progress=lambda d, n: d % 100 == 0 and log(f"  narrated {d}/{n}"))
    out = []
    for f, (text, meta) in zip(facts, results, strict=True):
        rec = dict(f.turn)
        rec["shap"] = f.shap
        rec["mitre"] = f.mitre
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

    # Fail fast on missing credentials, before any slow work.
    try:
        narrator = Narrator(offline=args.offline, model=model, cache_path=cache_dir / "narration.jsonl",
                            concurrency=args.concurrency, rpm=args.rpm)  # fmt: skip
    except NarrationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    t_start = time.perf_counter()
    print(f"indexing {episodes_path} ({os.path.getsize(episodes_path) / 1e6:.0f} MB) ...")
    index = index_episodes(episodes_path)
    t_index = time.perf_counter() - t_start
    picked = parse_episodes(args.episodes, index)
    print(f"indexed {len(index)} logged episodes in {t_index:.1f}s; enriching {len(picked)}:")
    for m, why in picked:
        print(f"  episode {m.episode}: {m.matchup}, winner={m.winner}, {m.turns} turns ({why})")

    graph, cfg = load_run(run_dir)
    settings = ShapSettings(nsamples=args.nsamples, background_k=args.background, top_k=args.top_k)
    shap_svc = None if args.no_shap else ShapService(settings, cache_dir / "shap.jsonl")

    out_path = args.out or run_dir / "episodes_enriched.jsonl"
    enriched: list[dict] = []
    all_facts: list[TurnFacts] = []
    t_enrich = time.perf_counter()
    try:
        for m, _ in picked:
            turns = read_turns(episodes_path, m.offsets)
            recs, facts = enrich_turns(turns, graph, cfg, shap_svc, narrator)
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
        },
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
    est = summary["online_cost_estimate"]["total"]
    print(f"narration: {summary['mode']}; online estimate for this selection with {model}: {est['calls']} calls, "
          f"~{est['input_tokens']} in / ~{est['output_tokens']} out tokens, ~${est['usd']:.4f}")  # fmt: skip
    print(f"timing: index {t_index:.1f}s, enrich {t_enrich:.1f}s, total {summary['timing_s']['total']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
