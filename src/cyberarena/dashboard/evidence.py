"""Multi-seed experiments (docs/contracts.md, "Experiments & rigor (v3)"). Pure functions, no Streamlit.

Reads ``runs/experiments/<name>/manifest.json`` and ``aggregate.json``. Nothing is recomputed when
``aggregate.json`` exists: every number on the Evidence page is the arena's own aggregate, and the verdict
sentences are built from those numbers (difference, 95% CI, paired p-value, per-seed differences).

Conditions are whatever the manifest names (``adaptive``/``frozen`` for the main experiment; any number of
conditions for factorial ``diag-*`` experiments). Contrasts are exactly the ones listed in ``aggregate.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from cyberarena.dashboard import loaders as L

EXPERIMENTS_DIR = "experiments"
ALPHA = 0.05

CONDITION_LABEL = {
    "adaptive": "adaptive detectors",
    "frozen": "frozen detectors",
}
# matchup order for cards and charts: the headline questions first
METRIC_ORDER = ("blue_learned_vs_red_evasive@0.7", "blue_learned_vs_red_baseline", "blue_learned_vs_red_evasive@0.4",
                "red_learned_vs_blue_baseline")  # fmt: skip


def _read(path: Path) -> dict | None:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


# ============================================================================================== discovery


def list_experiments(runs_dir: Path) -> list[dict]:
    """Every experiment folder with a manifest, newest first: ``{name, dir, manifest, aggregate (bool), status,
    created, n_runs, n_done, n_error, conditions, seeds, episodes}``."""
    root = Path(runs_dir) / EXPERIMENTS_DIR
    if not root.is_dir():
        return []
    out = []
    for d in root.iterdir():
        man = _read(d / "manifest.json") if d.is_dir() else None
        if not man:
            continue
        runs = man.get("runs") or []
        out.append({
            "name": man.get("name") or d.name, "dir": d, "aggregate": (d / "aggregate.json").exists(),
            "status": man.get("status") or "unknown", "created": man.get("created") or "",
            "n_runs": len(runs), "n_done": sum(r.get("status") == "done" for r in runs),
            "n_error": sum(r.get("status") == "error" for r in runs),
            "conditions": list(man.get("conditions") or []), "seeds": list(man.get("seeds") or []),
            "episodes": man.get("episodes"),
        })  # fmt: skip
    return sorted(out, key=lambda e: (e["created"], e["name"]), reverse=True)


def default_experiment(exps: list[dict]) -> dict | None:
    """``main`` when it has an aggregate, else the newest experiment with one, else the newest."""
    done = [e for e in exps if e["aggregate"]]
    for e in done:
        if e["name"] == "main":
            return e
    return (done or exps or [None])[0]


def experiment_label(e: dict) -> str:
    bits = [e["name"]]
    if e.get("seeds"):
        bits.append(f"{len(e['seeds'])} seed{'s' if len(e['seeds']) != 1 else ''}")
    if e.get("conditions"):
        bits.append(" / ".join(e["conditions"]))
    if e.get("episodes"):
        bits.append(f"{int(e['episodes']):,} games")
    if not e.get("aggregate"):
        bits.append("running" if e.get("status") == "running" else e.get("status") or "no results")
    return " · ".join(bits)


def experiment_rules(e: dict, man: dict | None = None) -> str | None:
    """``"current"`` (v5: isolating a clean host is costly), ``"previous"`` (cheap isolation) or ``"custom"``
    (isolation costs set through ``--env-json``); ``None`` when it can't be told. Read from the first run's
    ``config.json`` (``params.rules``; runs older than v5 have no such key), else the name and train flags."""
    man = man if man is not None else (load_manifest(e["dir"]) or {})
    extra = list(man.get("extra_args") or [])
    if "--rules" in extra:
        i = extra.index("--rules")
        return "previous" if i + 1 < len(extra) and extra[i + 1] == "cheap-isolation" else "current"
    if any("r_isolate_false" in a for a in extra):
        return "custom"
    for r in man.get("runs") or []:
        cfg = _read(Path(r["run_dir"]) / L.CONFIG_FILE) if r.get("run_dir") else None
        if cfg is not None:
            rules = (cfg.get("params") or {}).get("rules")
            return "previous" if rules in (None, "cheap-isolation") else "current"
    if "cheap-isolation" in e["name"]:
        return "previous"
    return None


RULES_LABEL = {"current": "current rules (costly isolation)", "previous": "previous rules (cheap isolation)",
               "custom": "isolation costs set by flag"}  # fmt: skip


def load_manifest(exp_dir: Path) -> dict | None:
    return _read(Path(exp_dir) / "manifest.json")


def load_aggregate(exp_dir: Path) -> dict | None:
    return _read(Path(exp_dir) / "aggregate.json")


def run_progress(man: dict) -> pd.DataFrame:
    """One row per (seed, condition) run with its live progress from ``progress.json``."""
    rows = []
    for r in man.get("runs") or []:
        prog = _read(Path(r["run_dir"]) / L.PROGRESS_FILE) if r.get("run_dir") else None
        ep, n = (prog or {}).get("episode"), (prog or {}).get("episodes") or man.get("episodes")
        status = r.get("status") or "pending"
        if status == "running" and prog and prog.get("status") == "done":
            status = "finishing"
        frac = 1.0 if status == "done" else (min(1.0, ep / n) if ep and n else 0.0)
        rows.append({"seed": r.get("seed"), "condition": r.get("condition"), "status": status,
                     "progress": frac, "game": ep, "games": n, "run_id": Path(r["run_dir"]).name if r.get("run_dir") else "",
                     "error": r.get("error") or ""})  # fmt: skip
    return pd.DataFrame(rows, columns=["seed", "condition", "status", "progress", "game", "games", "run_id", "error"])


# ============================================================================================== labels


def condition_label(c: str) -> str:
    """``adaptive`` -> ``adaptive detectors``; a factorial name like ``frozen-bandit`` is spelt out."""
    if c in CONDITION_LABEL:
        return CONDITION_LABEL[c]
    words = c.replace("_", "-").split("-")
    parts = []
    for w in words:
        parts.append({"adaptive": "adaptive detectors", "frozen": "frozen detectors", "bandit": "red disguising",
                      "off": "red not disguising", "noevade": "red not disguising", "evade": "red disguising",
                      "dqn": "DQN", "tabular": "tabular"}.get(w, w))  # fmt: skip
    return ", ".join(parts)


def metric_parts(metric: str) -> tuple[str, str]:
    """``"blue_learned_vs_red_evasive@0.7 late_mean"`` -> ``("blue_learned_vs_red_evasive@0.7", "late_mean")``."""
    m, _, stat = metric.partition(" ")
    return m, stat or "late_mean"


def metric_side(matchup: str) -> str:
    return "red" if matchup.startswith("red_") else "blue"


def metric_label(matchup: str) -> str:
    """``"Learned blue vs scripted red disguised at 0.7"``."""
    s = L.matchup_label(matchup)
    return s[:1].upper() + s[1:]


def metric_short(matchup: str) -> str:
    m, _, lvl = matchup.partition("@")
    if m == L.EVASIVE:
        return f"vs red disguised at {float(lvl):.1f}" if lvl else "vs disguised red"
    return {"blue_learned_vs_red_baseline": "vs scripted red", "red_learned_vs_blue_baseline": "vs scripted blue"}.get(
        m, m.replace("_", " ")
    )


def sort_metrics(keys) -> list[str]:
    keys = list(dict.fromkeys(keys))
    return sorted(keys, key=lambda k: (METRIC_ORDER.index(k) if k in METRIC_ORDER else 99, k))


def pts(x: float | None, signed: bool = True) -> str:
    """A win-rate difference in percentage points: ``+21`` / ``−19`` (true minus sign)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    v = round(100 * x)
    if v == 0:
        return "0"
    s = f"{abs(v)}"
    if not signed:
        return s
    return ("+" if v > 0 else "−") + s


def p_text(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "p n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


# ============================================================================================== contrasts


def contrasts(agg: dict, stat: str | None = None, pair: tuple[str, str] | None = None) -> list[dict]:
    """Contrasts from ``aggregate.json`` (optionally one statistic, ``late_mean``/``final``, and one condition
    pair), with parsed ``matchup`` / ``stat`` / ``side`` and the verdict, in headline order."""
    out = []
    for c in agg.get("contrasts") or []:
        m, s = metric_parts(c.get("metric", ""))
        if stat and s != stat:
            continue
        if pair and (c.get("a"), c.get("b")) != tuple(pair):
            continue
        out.append({**c, "matchup": m, "stat": s, "side": metric_side(m), "verdict": verdict(c)})
    order = sort_metrics([c["matchup"] for c in out])
    return sorted(out, key=lambda c: order.index(c["matchup"]))


def contrast_pairs(agg: dict) -> list[tuple[str, str]]:
    return list(dict.fromkeys((c.get("a"), c.get("b")) for c in agg.get("contrasts") or []))


def stats_available(agg: dict) -> list[str]:
    have = {metric_parts(c.get("metric", ""))[1] for c in agg.get("contrasts") or []}
    return [s for s in ("late_mean", "final") if s in have] + sorted(have - {"late_mean", "final"})


def significant(c: dict) -> bool:
    p, ci = c.get("p_value"), c.get("ci95") or [None, None]
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return False
    excl = ci[0] is not None and ci[1] is not None and (ci[0] > 0 or ci[1] < 0)
    return p < ALPHA and (excl or ci[0] is None)


def seed_agreement(c: dict) -> tuple[int, int, int]:
    """``(same sign as the mean, opposite sign, n)`` over the per-seed differences."""
    d = [x for x in (c.get("per_seed_diff") or []) if x is not None]
    mean = c.get("diff_mean") or 0.0
    if not d:
        return 0, 0, 0
    if abs(mean) < 1e-12:
        return 0, 0, len(d)
    same = sum((x > 0) == (mean > 0) and abs(x) > 1e-12 for x in d)
    opp = sum((x > 0) != (mean > 0) and abs(x) > 1e-12 for x in d)
    return same, opp, len(d)


def agreement_text(c: dict) -> str:
    same, opp, n = seed_agreement(c)
    if n == 0:
        return ""
    if same == n:
        return "every seed agrees" if n > 2 else "both seeds agree" if n == 2 else "one seed"
    mean = c.get("diff_mean") or 0.0
    word = "higher" if mean > 0 else "lower"
    return f"{same} of {n} seeds {word}" + (f", {opp} the other way" if opp else "")


def verdict(c: dict) -> str:
    """Plain-English verdict computed from one contrast. Says "no significant difference" when it isn't."""
    m, stat = metric_parts(c.get("metric", ""))
    side = metric_side(m)
    a, b = condition_label(c.get("a", "a")), condition_label(c.get("b", "b"))
    d = c.get("diff_mean")
    ci = c.get("ci95") or [None, None]
    p = c.get("p_value")
    when = "over the second half of training" if stat == "late_mean" else "at the final checkpoint" if stat == "final" else stat
    vs = metric_short(m)
    ci_txt = f"95% CI {pts(ci[0])} to {pts(ci[1])} pts" if ci[0] is not None else "no CI"
    stats = f"({ci_txt}, {p_text(p)}; {agreement_text(c)})"
    if d is None:
        return "No difference could be computed."
    if significant(c):
        more = "more" if d > 0 else "less"
        return (f"Learned {side} wins {pts(abs(d), False)} pts {more} {vs} with {a} than with {b} {when} {stats}.")
    n = c.get("n") or len(c.get("per_seed_diff") or [])
    few = " With this few seeds only a large effect would show." if n and n < 5 else ""
    return (f"No significant difference {vs}: learned {side}'s win rate is {pts(d)} pts with {a} compared with "
            f"{b} {when} {stats}." + few)


# ============================================================================================== "did it work?"


def arms_race_rows(agg: dict) -> list[dict]:
    """Per detector: recall on red disguised at 0.7 before the first update vs after the last, per seed."""
    out = []
    for model, r in (agg.get("arms_race") or {}).items():
        per = r.get("per_seed") or {}
        first = r.get("recall_at_0.7_first")
        last = r.get("recall_at_0.7_last")
        seeds = sorted(per, key=lambda s: int(s) if str(s).isdigit() else 0)
        gains = [per[s].get("recall_last") - per[s].get("recall_first") for s in seeds
                 if per[s].get("recall_last") is not None and per[s].get("recall_first") is not None]  # fmt: skip
        out.append({"model": model, "first": first, "last": last, "cycles": r.get("cycles_mean"),
                    "updates": r.get("n_updates_mean"), "seeds": seeds,
                    "per_first": [per[s].get("recall_first") for s in seeds],
                    "per_last": [per[s].get("recall_last") for s in seeds],
                    "per_cycles": [per[s].get("cycles") for s in seeds],
                    "gains": gains})  # fmt: skip
    order = ("malware", "network", "phishing")
    out = [r for r in out if r["first"] is not None and r["last"] is not None]  # frozen detectors: no updates
    return sorted(out, key=lambda r: order.index(r["model"]) if r["model"] in order else 9)


def strength(c: dict) -> str:
    """``yes`` (significant and positive), ``no`` (significant and negative), ``suggestive`` (positive, not
    significant, but most seeds agree and p < 0.2), ``unclear`` otherwise."""
    d = c.get("diff_mean") or 0.0
    if significant(c):
        return "yes" if d > 0 else "no"
    same, _, n = seed_agreement(c)
    p = c.get("p_value")
    if d > 0 and n and same >= max(2, math.ceil(0.75 * n)) and p is not None and p < 0.2:
        return "suggestive"
    if d < 0 and n and same >= max(2, math.ceil(0.75 * n)) and p is not None and p < 0.2:
        return "suggestive-negative"
    return "unclear"


def _contrast_row(key: str, question: str, c: dict) -> dict:
    ans = strength(c)
    d = c.get("diff_mean") or 0.0
    a, b = condition_label(c["a"]), condition_label(c["b"])
    side = c["side"]
    vs = metric_short(c["matchup"])
    ci = c.get("ci95") or [None, None]
    stats = f"95% CI {pts(ci[0])} to {pts(ci[1])} pts, {p_text(c.get('p_value'))}, {agreement_text(c)}"
    if ans in ("yes", "no"):
        head = f"Learned {side} wins {pts(abs(d), False)} pts {'more' if d > 0 else 'less'} {vs} with {a}"
    elif ans == "suggestive":
        head = f"Suggestive, not established: {pts(d)} pts {vs} with {a}"
    elif ans == "suggestive-negative":
        head = f"Suggestive, not established: {pts(d)} pts {vs} with {a} (a possible cost)"
    else:
        head = f"No clear difference {vs} ({pts(d)} pts with {a})"
    detail = (f"{pct(c.get('a_mean'))} with {a} vs {pct(c.get('b_mean'))} with {b}, second half of training; {stats}.")
    return {"key": key, "question": question, "answer": ans, "headline": head, "detail": detail}


def did_it_work(agg: dict) -> list[dict]:
    """The levels of the result, each ``{key, question, answer, headline, detail}``; ``answer`` is one of
    ``yes / no / mixed / suggestive / suggestive-negative / unclear``, all computed from ``aggregate.json``.

    * ``detectors``: does retraining raise recall against red's disguise? (``arms_race``: v0 vs last update)
    * ``red``: does learned red win more when detectors adapt? (``red_learned_vs_blue_baseline``)
    * ``blue``: does learned blue win more with adaptive detectors? Judged on the disguised-red test at the highest
      level (the matchup that isolates whether adapted detectors beat disguise); the undisguised test is in the
      detail.
    """
    rows = []
    ar = arms_race_rows(agg)
    if ar:
        all_gains = [g for r in ar for g in r["gains"]]
        up = sum(g > 0 for g in all_gains)
        mean_gain = sum((r["last"] or 0) - (r["first"] or 0) for r in ar) / len(ar)
        per_model = ", ".join(f"{r['model']} {pct(r['first'])} → {pct(r['last'])}" for r in ar)
        ans = "yes" if all_gains and up == len(all_gains) else "mixed" if up else "no"
        rows.append({
            "key": "detectors",
            "question": "Do retrained detectors catch disguised red better?",
            "answer": ans,
            "headline": f"Recall on red disguised at 0.7: {pts(mean_gain)} pts on average",
            "detail": (f"{per_model} (pretrained v0 → after the last retrain, mean of {len(ar[0]['seeds'])} seeds); "
                       f"recall rose in {up} of {len(all_gains)} detector-seed pairs."),
        })  # fmt: skip
    cs = contrasts(agg, "late_mean")
    pair = ("adaptive", "frozen")
    mine = [c for c in cs if (c.get("a"), c.get("b")) == pair] or cs
    red = next((c for c in mine if c["matchup"] == "red_learned_vs_blue_baseline"), None)
    if red is not None:
        rows.append(_contrast_row("red", "Does the attacker gain when detectors adapt?", red))
    ev = sorted((c for c in mine if c["matchup"].startswith(L.EVASIVE + "@")),
                key=lambda c: -float(c["matchup"].split("@")[1]))  # fmt: skip
    plain = next((c for c in mine if c["matchup"] == "blue_learned_vs_red_baseline"), None)
    tests = [c for c in (ev[0] if ev else None, plain) if c is not None]
    # the disguised test is the primary one; a significant result on the other test takes precedence
    blue = next((c for c in tests if significant(c)), tests[0] if tests else None)
    if blue is not None:
        row = _contrast_row("blue", f"Does the defender win more with {condition_label(blue['a'])}?", blue)
        for other in tests:
            if other is blue:
                continue
            word = {"yes": "significantly more", "no": "significantly less"}.get(strength(other), "not significant")
            row["detail"] += (f" {metric_short(other['matchup']).capitalize()}: {pts(other.get('diff_mean'))} pts "
                              f"({word}, {p_text(other.get('p_value'))}).")  # fmt: skip
        rows.append(row)
    return rows


def pct(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100 * float(x):.0f}%"


# ============================================================================================== curves


def curve_frame(agg: dict) -> pd.DataFrame:
    """Long frame of ``curves``: ``condition, matchup, after_episode, mean, lo, hi, per_seed, seeds``."""
    rows = []
    for cond, mets in (agg.get("curves") or {}).items():
        for m, pts_ in mets.items():
            for p in pts_:
                ci = p.get("ci95") or [None, None]
                rows.append({"condition": cond, "matchup": m, "after_episode": int(p["after_episode"]),
                             "mean": p.get("mean"), "lo": None if ci[0] is None else max(0.0, ci[0]),
                             "hi": None if ci[1] is None else min(1.0, ci[1]), "per_seed": p.get("per_seed") or [],
                             "seeds": p.get("seeds") or []})  # fmt: skip
    return pd.DataFrame(rows, columns=["condition", "matchup", "after_episode", "mean", "lo", "hi", "per_seed", "seeds"])


def curve_takeaway(cf: pd.DataFrame, matchup: str, conditions: list[str], labels: dict | None = None) -> str:
    sub = cf[cf["matchup"] == matchup]
    if sub.empty:
        return "No checkpoints for this matchup."
    bits = []
    for c in conditions:
        s = sub[sub["condition"] == c].sort_values("after_episode")
        if s.empty:
            continue
        name = (labels or {}).get(c) or condition_label(c)
        bits.append(f"{pct(s['mean'].iloc[0])} → {pct(s['mean'].iloc[-1])} with {name}")
    side = metric_side(matchup)
    return f"Learned {side} {metric_short(matchup)}, first to last checkpoint (mean of seeds): " + "; ".join(bits) + "."


# ============================================================================================== methods


def definitions(agg: dict) -> dict[str, str]:
    return {k: str(v) for k, v in (agg.get("definitions") or {}).items()}


# ============================================================================================== DQN vs tabular


def agent_comparison_runs(runs: list[Path], infos: dict[Path, dict]) -> tuple[Path, Path] | None:
    """A (DQN, tabular) pair of finished stand-alone runs with the same seed, length and detector setting;
    prefers the run labelled ``reference``."""
    solo = [r for r in runs if infos[r]["status"] == "done" and not L.is_experiment_run(infos[r])]
    dqn = [r for r in solo if infos[r].get("agent_type") == "dqn"]
    dqn.sort(key=lambda r: r.name, reverse=True)  # newest first ...
    dqn.sort(key=lambda r: infos[r].get("label") != "reference")  # ... the reference first (stable)
    for d in dqn:
        i = infos[d]
        for t in solo:
            j = infos[t]
            if (j.get("agent_type") == "tabular" and j.get("seed") == i.get("seed") and j.get("games") == i.get("games")
                    and j.get("adaptive") == i.get("adaptive") and j.get("rules") == i.get("rules")):  # fmt: skip
                return d, t
    return None


def final_rates(curve: pd.DataFrame) -> dict[str, float | None]:
    out: dict[str, Any] = {}
    for side in ("red", "blue"):
        s = curve[curve["side"] == side].sort_values("after_episode") if not curve.empty else curve
        out[side] = float(s["win_rate"].iloc[-1]) if not s.empty else None
        half = s[s["after_episode"] >= (s["after_episode"].max() / 2)] if not s.empty else s
        out[f"{side}_late"] = float(half["wins"].sum() / half["n"].sum()) if not half.empty and half["n"].sum() else None
    return out


# ============================================================================================== factorial families

# level a run has for a diagnostic flag when the experiment didn't pass it
FLAG_DEFAULT = {"--detectors": "adaptive", "--red-evasion": "bandit"}
FLAG_FACTOR = {"--detectors": "detectors", "--red-evasion": "red disguise"}
LEVEL_WORD = {("--detectors", "adaptive"): "adaptive detectors", ("--detectors", "frozen"): "frozen detectors",
              ("--red-evasion", "bandit"): "red adapting its disguise",
              ("--red-evasion", "off"): "red never disguising"}  # fmt: skip
HIGH_LEVELS = ("adaptive", "bandit", "on")


def _flag_pairs(args: list[str]) -> dict[str, str]:
    out, i = {}, 0
    while i < len(args):
        a = args[i]
        if a.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[a] = args[i + 1]
            i += 2
        else:
            out[a] = "on"
            i += 1
    return out


def factorial_families(exps: list[dict], min_cells: int = 3) -> list[dict]:
    """Groups of finished experiments that share a name prefix (``diag-*``) and differ in their extra train
    flags: one cell each of a factorial design. ``{name, cells: [{exp, levels}], factors: [(flag, [levels])]}``;
    ``levels`` maps flag -> level (a flag an experiment didn't pass is at the arena default)."""
    groups: dict[str, list[dict]] = {}
    for e in exps:
        if not e["aggregate"] or "-" not in e["name"] or len(e["conditions"]) != 1:
            continue
        groups.setdefault(e["name"].split("-")[0], []).append(e)
    fams = []
    for prefix, members in groups.items():
        if len(members) < min_cells:
            continue
        mans = {e["name"]: load_manifest(e["dir"]) or {} for e in members}
        flags = sorted({f for m in mans.values() for f in _flag_pairs(list(m.get("extra_args") or []))})
        if not flags:
            continue
        cells = []
        for e in members:
            fp = _flag_pairs(list(mans[e["name"]].get("extra_args") or []))
            cond = e["conditions"][0]
            lv = {}
            for f in flags:
                d = FLAG_DEFAULT.get(f, "default")
                if f == "--detectors" and cond == "frozen":
                    d = "frozen"
                lv[f] = fp.get(f, d)
            cells.append({"exp": e, "levels": lv})
        factors = [(f, sorted({c["levels"][f] for c in cells}, key=lambda x: (x not in HIGH_LEVELS, x)))
                   for f in flags]  # fmt: skip
        factors = [(f, lv) for f, lv in factors if len(lv) > 1]
        if factors:
            fams.append({"name": prefix, "cells": sorted(cells, key=lambda c: c["exp"]["name"]), "factors": factors,
                         "seeds": sorted({s for c in cells for s in c["exp"]["seeds"]}),
                         "episodes": members[0].get("episodes")})  # fmt: skip
    return fams


def experiment_condition_names(man: dict, conds: list[str]) -> dict[str, str]:
    """Plain names for an experiment's conditions; a single-condition experiment with diagnostic flags
    (``--detectors frozen``, ``--red-evasion off``) is named after what the flags set."""
    out = {c: condition_label(c) for c in conds}
    flags = _flag_pairs(list(man.get("extra_args") or []))
    if len(conds) == 1 and any(f in FLAG_DEFAULT for f in flags):
        lv = {f: flags.get(f, "frozen" if f == "--detectors" and conds[0] == "frozen" else d)
              for f, d in FLAG_DEFAULT.items()}  # fmt: skip
        out[conds[0]] = cell_name(lv)
    return out


def level_word(flag: str, level: str) -> str:
    return LEVEL_WORD.get((flag, level), f"{FLAG_FACTOR.get(flag, flag.lstrip('-'))} {level}")


def cell_name(levels: dict[str, str]) -> str:
    return " · ".join(level_word(f, v) for f, v in levels.items())


def _t_stats(diffs: list[float]) -> dict:
    """Mean, 95% t-interval and two-sided one-sample t-test p of paired differences."""
    n = len(diffs)
    if n == 0:
        return {"diff_mean": None, "ci95": [None, None], "p_value": None, "n": 0}
    mean = sum(diffs) / n
    if n < 2:
        return {"diff_mean": mean, "ci95": [None, None], "p_value": None, "n": n}
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1))
    t = mean / (sd / math.sqrt(n)) if sd > 0 else math.inf
    try:
        from scipy import stats as _st

        q = float(_st.t.ppf(0.975, n - 1))
        p = float(2 * _st.t.sf(abs(t), n - 1)) if sd > 0 else 0.0
    except ImportError:  # pragma: no cover - normal approximation
        q = 1.96
        p = math.erfc(abs(t) / math.sqrt(2)) if sd > 0 else 0.0
    half = q * sd / math.sqrt(n)
    return {"diff_mean": mean, "ci95": [mean - half, mean + half], "p_value": p, "n": n, "t": t}


def cell_summary(cell: dict, matchup: str, stat: str = "late_mean") -> dict | None:
    """``{mean, sd, ci95, per_seed, seeds}`` of one cell's single condition, straight from its aggregate."""
    agg = load_aggregate(cell["exp"]["dir"]) or {}
    block = agg.get(stat) or {}
    cond = next(iter(block), None)
    return (block.get(cond) or {}).get(matchup) if cond else None


def cell_values(cell: dict, matchup: str, stat: str = "late_mean") -> dict[int, float]:
    s = cell_summary(cell, matchup, stat)
    if not s:
        return {}
    return {int(k): float(v) for k, v in zip(s.get("seeds") or [], s.get("per_seed") or [], strict=False)
            if v is not None}  # fmt: skip


def factorial_effects(fam: dict, matchup: str, stat: str = "late_mean") -> list[dict]:
    """Main effects (and, for a 2×2, the interaction) paired by seed. Computed here from the cells' own per-seed
    values in their ``aggregate.json`` files (no run is re-read). Each effect is a contrast-like dict with a
    plain-English ``verdict``."""
    vals = {tuple(sorted(c["levels"].items())): cell_values(c, matchup, stat) for c in fam["cells"]}
    out = []
    for f, levels in fam["factors"]:
        if len(levels) != 2:
            continue
        hi, lo = levels
        per: dict[int, list[float]] = {}
        for key, v in vals.items():
            d = dict(key)
            if d.get(f) != hi:
                continue
            twin = tuple(sorted({**d, f: lo}.items()))
            for s, x in v.items():
                if s in vals.get(twin, {}):
                    per.setdefault(s, []).append(x - vals[twin][s])
        seeds = sorted(per)
        diffs = [sum(per[s]) / len(per[s]) for s in seeds]
        out.append({**_t_stats(diffs), "metric": f"{matchup} {stat}", "a": level_word(f, hi), "b": level_word(f, lo),
                    "factor": FLAG_FACTOR.get(f, f), "flag": f, "per_seed_diff": diffs, "seeds": seeds,
                    "kind": "main effect"})  # fmt: skip
    if len(fam["factors"]) == 2 and all(len(lv) == 2 for _, lv in fam["factors"]) and len(vals) == 4:
        (f1, (h1, o1)), (f2, (h2, o2)) = fam["factors"]

        def v(a, b):
            return vals.get(tuple(sorted({f1: a, f2: b}.items())), {})

        seeds = sorted(set(v(h1, h2)) & set(v(o1, h2)) & set(v(h1, o2)) & set(v(o1, o2)))
        diffs = [(v(h1, h2)[s] - v(o1, h2)[s]) - (v(h1, o2)[s] - v(o1, o2)[s]) for s in seeds]
        out.append({**_t_stats(diffs), "metric": f"{matchup} {stat}", "a": level_word(f2, h2), "b": level_word(f2, o2),
                    "factor": "interaction", "per_seed_diff": diffs, "seeds": seeds, "kind": "interaction",
                    "inner": (level_word(f1, h1), level_word(f1, o1))})  # fmt: skip
    for e in out:
        e["verdict"] = effect_verdict(e)
    return out


def effect_verdict(e: dict) -> str:
    m, _ = metric_parts(e["metric"])
    side = metric_side(m)
    vs = metric_short(m)
    ci = e.get("ci95") or [None, None]
    d = e.get("diff_mean")
    if d is None:
        return "No paired seeds to compare."
    stats = (f"(95% CI {pts(ci[0])} to {pts(ci[1])} pts, {p_text(e.get('p_value'))}; {agreement_text(e)})"
             if ci[0] is not None else "(too few seeds for an interval)")  # fmt: skip
    if e["kind"] == "interaction":
        a1, o1 = e["inner"]
        body = (f"the gap between {a1} and {o1} for learned {side} {vs} is {pts(d)} pts with {e['a']} compared "
                f"with {e['b']}")  # fmt: skip
        return (f"Interaction: {body} {stats}." if significant(e) else f"No significant interaction: {body} {stats}.")
    if significant(e):
        return (f"{e['factor'].capitalize()}: learned {side} wins {pts(abs(d), False)} pts "
                f"{'more' if d > 0 else 'less'} {vs} with {e['a']} than with {e['b']}, averaged over the other "
                f"factor {stats}.")  # fmt: skip
    return (f"{e['factor'].capitalize()}: no significant effect {vs} ({e['a']} minus {e['b']}: {pts(d)} pts, "
            f"averaged over the other factor) {stats}.")  # fmt: skip


# ============================================================================================== cross-evaluation


def load_crosseval(runs_dir: Path, name: str = "diag-crosseval") -> pd.DataFrame:
    """``runs/<name>/*.json`` cross-evaluation cells (a trained blue agent replayed with swapped detectors):
    ``seed, blue_run, trained_with, detectors, evasion, n, blue_win_rate, isolate_clean_share``.
    ``trained_with`` comes from the blue run's ``config.json`` (adaptive / frozen). Empty when absent."""
    d = Path(runs_dir) / name
    cols = ["seed", "blue_run", "trained_with", "detectors", "evasion", "n", "blue_win_rate", "isolate_clean_share"]
    rows = []
    if d.is_dir():
        cache: dict[str, str] = {}
        for f in sorted(d.glob("*.json")):
            obj = _read(f)
            if not obj or not isinstance(obj.get("cells"), list):
                continue
            for c in obj["cells"]:
                br = str(c.get("blue_run"))
                if br not in cache:
                    cfg = _read(Path(runs_dir) / br / L.CONFIG_FILE)
                    if cfg is not None:
                        cache[br] = "frozen" if cfg.get("adaptive") is False else "adaptive"
                    else:  # run not on disk: the detectors come from the seed's adaptive run
                        cache[br] = "adaptive" if br == str(obj.get("detectors_from")) else "frozen"
                rows.append({"seed": obj.get("seed"), "blue_run": br, "trained_with": cache[br],
                             "detectors": c.get("detectors"), "evasion": c.get("evasion"), "n": c.get("n"),
                             "blue_win_rate": c.get("blue_win_rate"),
                             "isolate_clean_share": c.get("isolate_clean_share")})  # fmt: skip
    return pd.DataFrame(rows, columns=cols)
