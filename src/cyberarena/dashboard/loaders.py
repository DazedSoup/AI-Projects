"""Read-only loaders for a cyberarena run directory (see docs/contracts.md, "Episode log").

Everything here is a pure function of files on disk. The dashboard never imports arena, ml or explain code
and never calls an API. Missing enriched fields (``shap``, ``rationale``, ``mitre``, ``rationale_meta``)
are tolerated everywhere: they come back as ``None``.

``episodes.jsonl`` can be ~200 MB, so it is never loaded whole. :func:`build_episode_index` scans it once
for per-episode byte ranges (cached on disk next to the system temp dir, keyed by path/size/mtime) and
:func:`read_episode` seeks to just the episode that is asked for.

The ``cached_*`` wrappers at the bottom add ``st.cache_data`` for the app; tests use the plain functions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

try:  # streamlit is optional for the pure loaders (tests, scripts)
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None

ENRICHED_FILE = "episodes_enriched.jsonl"
EPISODES_FILE = "episodes.jsonl"
SUMMARY_FILE = "summary.jsonl"
GRAPH_FILE = "graph.json"

EVAL_MATCHUPS = {
    # matchup -> (learned side, label)
    "red_learned_vs_blue_baseline": ("red", "Learned red vs baseline blue"),
    "blue_learned_vs_red_baseline": ("blue", "Learned blue vs baseline red"),
}
HEAD_TO_HEAD = "learned_vs_learned"

NODE_STATES = ("clean", "patched", "compromised", "detected", "isolated")
ENRICHED_FIELDS = ("shap", "rationale", "mitre", "rationale_meta")
_TURN_DEFAULTS: dict[str, Any] = {
    "source": None,
    "target": None,
    "success": False,
    "reward": 0.0,
    "decision_values": {},
    "epsilon": None,
    "explored": False,
    "node_states": [],
    "classifier_inputs": [],
    "done": False,
    "winner": None,
    "shap": None,
    "rationale": None,
    "mitre": None,
    "rationale_meta": None,
    "phase": None,
    "matchup": None,
    "agent": None,
    "after_episode": None,
}
_EPISODE_RE = re.compile(rb'"episode"\s*:\s*(-?\d+)')
INDEX_VERSION = 1


class RunFormatError(ValueError):
    """A run file exists but does not match docs/contracts.md."""


# ---------------------------------------------------------------------------------------------- run discovery


def list_runs(runs_dir: Path) -> list[Path]:
    """Run directories (those with a summary or episode log), newest name first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    runs = [
        p
        for p in runs_dir.iterdir()
        if p.is_dir() and any((p / f).exists() for f in (SUMMARY_FILE, EPISODES_FILE, ENRICHED_FILE))
    ]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def read_jsonl(path: Path) -> list[dict]:
    """Small JSONL files only (summary, enriched). Blank lines are skipped."""
    out = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RunFormatError(f"{path.name}:{lineno}: invalid JSON ({e.msg})") from e
    return out


# ---------------------------------------------------------------------------------------------- graph


def load_graph(run_dir: Path) -> dict:
    """``{"nodes": [{id, role, crown_jewel, x, y}], "edges": [[u, v]]}``, validated."""
    path = Path(run_dir) / GRAPH_FILE
    g = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(g, dict) or "nodes" not in g:
        raise RunFormatError(f"{path}: expected an object with 'nodes'")
    nodes = []
    for n in g["nodes"]:
        if "id" not in n:
            raise RunFormatError(f"{path}: node without id")
        nodes.append(
            {
                "id": int(n["id"]),
                "role": n.get("role", "host"),
                "crown_jewel": bool(n.get("crown_jewel", False)),
                "x": float(n.get("x", 0.0)),
                "y": float(n.get("y", 0.0)),
            }
        )
    ids = {n["id"] for n in nodes}
    edges = []
    for e in g.get("edges", []):
        u, v = int(e[0]), int(e[1])
        if u not in ids or v not in ids:
            raise RunFormatError(f"{path}: edge {u}-{v} references an unknown node")
        edges.append((u, v))
    return {"nodes": sorted(nodes, key=lambda n: n["id"]), "edges": edges}


# ---------------------------------------------------------------------------------------------- summary


def load_summary(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(episodes, evals)`` dataframes from ``summary.jsonl``. Missing file -> two empty frames."""
    path = Path(run_dir) / SUMMARY_FILE
    ep_cols = ["episode", "winner", "turns", "red_return", "blue_return", "epsilon", "matchup", "logged"]
    ev_cols = ["after_episode", "matchup", "n", "red_win_rate", "blue_win_rate", "first", "last"]
    if not path.exists():
        return pd.DataFrame(columns=ep_cols), pd.DataFrame(columns=ev_cols)
    rows = read_jsonl(path)
    eps, evs = [], []
    for r in rows:
        kind = r.get("kind")
        if kind == "episode":
            eps.append({c: r.get(c) for c in ep_cols})
        elif kind == "eval":
            rng = r.get("episodes") or [None, None]
            evs.append({**{c: r.get(c) for c in ev_cols[:5]}, "first": rng[0], "last": rng[-1]})
    episodes = pd.DataFrame(eps, columns=ep_cols)
    if not episodes.empty:
        episodes["logged"] = episodes["logged"].fillna(False).astype(bool)
        episodes = episodes.sort_values("episode").reset_index(drop=True)
    evals = pd.DataFrame(evs, columns=ev_cols)
    if not evals.empty:
        evals = evals.sort_values(["matchup", "after_episode"]).reset_index(drop=True)
    return episodes, evals


def wilson_interval(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n (well behaved at 0% and 100%)."""
    if not n:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def eval_curve(evals: pd.DataFrame) -> pd.DataFrame:
    """One row per eval point: the learned side's win rate with its Wilson 95% CI.

    For ``red_learned_vs_blue_baseline`` the learned side is red (``red_win_rate``); for
    ``blue_learned_vs_red_baseline`` it is blue (``blue_win_rate``).
    """
    cols = ["after_episode", "matchup", "side", "label", "n", "wins", "win_rate", "ci_low", "ci_high"]
    out = []
    for r in evals.itertuples(index=False):
        if r.matchup not in EVAL_MATCHUPS:
            continue
        side, label = EVAL_MATCHUPS[r.matchup]
        rate = r.red_win_rate if side == "red" else r.blue_win_rate
        if rate is None or (isinstance(rate, float) and math.isnan(rate)):
            continue
        n = int(r.n or 0)
        wins = round(float(rate) * n)
        lo, hi = wilson_interval(wins, n)
        out.append([int(r.after_episode), r.matchup, side, label, n, wins, float(rate), lo, hi])
    return pd.DataFrame(out, columns=cols)


def rolling_head_to_head(episodes: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """Rolling red win rate over training ``learned_vs_learned`` episodes (window counts those episodes)."""
    cols = ["episode", "red_win_rate", "n_in_window"]
    if episodes.empty or "matchup" not in episodes:
        return pd.DataFrame(columns=cols)
    h2h = episodes[episodes["matchup"] == HEAD_TO_HEAD]
    if h2h.empty:
        return pd.DataFrame(columns=cols)
    red = (h2h["winner"] == "red").astype(float)
    minp = max(1, window // 4)
    return pd.DataFrame(
        {
            "episode": h2h["episode"].to_numpy(),
            "red_win_rate": red.rolling(window, min_periods=minp).mean().to_numpy(),
            "n_in_window": red.rolling(window, min_periods=1).count().to_numpy(),
        }
    ).dropna(subset=["red_win_rate"])


# ---------------------------------------------------------------------------------------------- turn records


def normalize_turn(rec: dict) -> dict:
    """Fill missing optional fields so the app never has to guess. Required: episode, turn, actor, action_id."""
    for k in ("episode", "turn", "actor", "action_id"):
        if k not in rec:
            raise RunFormatError(f"turn record missing {k!r}")
    out = {**_TURN_DEFAULTS, **rec}
    out["decision_values"] = out["decision_values"] or {}
    out["node_states"] = out["node_states"] or []
    out["classifier_inputs"] = out["classifier_inputs"] or []
    out["enriched"] = any(rec.get(k) is not None for k in ("shap", "rationale", "mitre", "rationale_meta"))
    return out


def group_turns(records: list[dict]) -> dict[int, list[dict]]:
    eps: dict[int, list[dict]] = {}
    for r in records:
        r = normalize_turn(r)
        eps.setdefault(int(r["episode"]), []).append(r)
    for turns in eps.values():
        turns.sort(key=lambda t: t["turn"])
    return eps


def load_enriched(run_dir: Path) -> dict[int, list[dict]]:
    """Episode -> sorted turn records from ``episodes_enriched.jsonl`` (``{}`` if absent)."""
    path = Path(run_dir) / ENRICHED_FILE
    if not path.exists():
        return {}
    return group_turns(read_jsonl(path))


def rationale_source(turns: list[dict]) -> str:
    """'claude', 'template', 'mixed' or 'none' for an episode's narration."""
    sources = set()
    for t in turns:
        meta = t.get("rationale_meta") or {}
        src = meta.get("source")
        if src:
            sources.add("claude" if src == "claude" else "template")
        elif t.get("rationale"):
            sources.add("unknown")
    if not sources:
        return "none"
    if len(sources) > 1:
        sources.discard("unknown")  # rows without rationale_meta don't override labelled ones
    if len(sources) > 1:
        return "mixed"
    return sources.pop()


# ---------------------------------------------------------------------------------------------- big log index


@dataclass
class EpisodeIndex:
    """Byte ranges per episode in ``episodes.jsonl``. ``segments[ep]`` is a list of ``(start, end)``."""

    path: str
    size: int
    mtime_ns: int
    segments: dict[int, list[tuple[int, int]]] = field(default_factory=dict)

    @property
    def episodes(self) -> list[int]:
        return sorted(self.segments)


def _index_cache_path(path: Path, size: int, mtime_ns: int, cache_dir: Path | None) -> Path:
    key = hashlib.sha1(f"{path.resolve()}|{size}|{mtime_ns}|{INDEX_VERSION}".encode()).hexdigest()[:16]
    base = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "cyberarena-dashboard"
    return base / f"episodes-index-{key}.json"


def scan_episode_index(path: Path) -> dict[int, list[tuple[int, int]]]:
    """One sequential pass: contiguous lines of the same episode merge into one ``(start, end)`` range."""
    segments: dict[int, list[tuple[int, int]]] = {}
    cur_ep, cur_start, off = None, 0, 0
    with open(path, "rb") as f:
        for line in f:
            m = _EPISODE_RE.search(line, 0, 256) or _EPISODE_RE.search(line)
            ep = int(m.group(1)) if m else None
            if ep != cur_ep:
                if cur_ep is not None:
                    segments.setdefault(cur_ep, []).append((cur_start, off))
                cur_ep, cur_start = ep, off
            off += len(line)
    if cur_ep is not None:
        segments.setdefault(cur_ep, []).append((cur_start, off))
    return segments


def build_episode_index(run_dir: Path, cache_dir: Path | None = None) -> EpisodeIndex | None:
    """Index ``episodes.jsonl`` by episode, reusing an on-disk cache when the file is unchanged."""
    path = Path(run_dir) / EPISODES_FILE
    if not path.exists():
        return None
    st_ = path.stat()
    cache = _index_cache_path(path, st_.st_size, st_.st_mtime_ns, cache_dir)
    if cache.exists():
        try:
            raw = json.loads(cache.read_text(encoding="utf-8"))
            segs = {int(k): [tuple(s) for s in v] for k, v in raw["segments"].items()}
            return EpisodeIndex(str(path), st_.st_size, st_.st_mtime_ns, segs)
        except (OSError, ValueError, KeyError):
            pass  # rebuild below
    segs = scan_episode_index(path)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"segments": {str(k): v for k, v in segs.items()}}), encoding="utf-8")
    except OSError:
        pass  # a read-only temp dir only costs a rescan next time
    return EpisodeIndex(str(path), st_.st_size, st_.st_mtime_ns, segs)


def read_episode(index: EpisodeIndex, episode: int) -> list[dict]:
    """Turn records of one episode from the big log, via the byte-offset index."""
    records = []
    with open(index.path, "rb") as f:
        for start, end in index.segments.get(int(episode), []):
            f.seek(start)
            for line in f.read(end - start).splitlines():
                if line.strip():
                    records.append(json.loads(line))
    return group_turns(records).get(int(episode), [])


# ---------------------------------------------------------------------------------------------- catalog


def episode_catalog(
    enriched: dict[int, list[dict]], episodes: pd.DataFrame, evals: pd.DataFrame
) -> pd.DataFrame:
    """Every replayable episode, enriched first. Built from the summary only (no scan of the big log).

    Logged episodes = training episodes with ``logged: true`` plus every eval episode (eval rows carry
    ``episodes: [first, last]``). Whether each really has records is only known once the index is built.
    """
    cols = ["episode", "enriched", "phase", "matchup", "after_episode", "winner", "turns", "label"]
    rows: dict[int, dict] = {}
    for ep, turns in enriched.items():
        last = turns[-1]
        rows[ep] = {
            "episode": ep,
            "enriched": True,
            "phase": last.get("phase"),
            "matchup": last.get("matchup"),
            "after_episode": last.get("after_episode"),
            "winner": last.get("winner"),
            "turns": len(turns),
        }
    if not episodes.empty:
        for r in episodes[episodes["logged"]].itertuples(index=False):
            ep = int(r.episode)
            rows.setdefault(
                ep,
                {
                    "episode": ep,
                    "enriched": False,
                    "phase": "train",
                    "matchup": r.matchup,
                    "after_episode": None,
                    "winner": r.winner,
                    "turns": r.turns,
                },
            )
    if not evals.empty:
        for r in evals.itertuples(index=False):
            if r.first is None or r.last is None or pd.isna(r.first):
                continue
            for ep in range(int(r.first), int(r.last) + 1):
                rows.setdefault(
                    ep,
                    {
                        "episode": ep,
                        "enriched": False,
                        "phase": "eval",
                        "matchup": r.matchup,
                        "after_episode": int(r.after_episode),
                        "winner": None,
                        "turns": None,
                    },
                )
    df = pd.DataFrame(list(rows.values()), columns=cols[:-1])
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["label"] = [_episode_label(r) for r in df.itertuples(index=False)]
    df = df.sort_values(["enriched", "episode"], ascending=[False, True]).reset_index(drop=True)
    return df[cols]


def _episode_label(r) -> str:
    bits = [f"ep {r.episode}"]
    if r.phase == "eval" and r.after_episode is not None and not pd.isna(r.after_episode):
        bits.append(f"eval@{int(r.after_episode)}")
    elif r.phase:
        bits.append(str(r.phase))
    if r.matchup:
        bits.append(str(r.matchup).replace("_", " "))
    if isinstance(r.winner, str):
        bits.append(f"{r.winner} wins")
    if r.turns is not None and not pd.isna(r.turns):
        bits.append(f"{int(r.turns)} turns")
    return " · ".join(bits)


# ---------------------------------------------------------------------------------------------- per-turn views


def node_state(ns: dict) -> str:
    """Display state with precedence isolated > detected > compromised > patched > clean."""
    if ns.get("isolated"):
        return "isolated"
    if ns.get("detected"):
        return "detected"
    if ns.get("compromised"):
        return "compromised"
    if ns.get("patched"):
        return "patched"
    return "clean"


def format_mitre(m: dict | None) -> str:
    if not m:
        return ""
    tid, name = m.get("technique_id", ""), m.get("technique_name", "")
    return f"{tid} {name}".strip()


def move_log(turns: list[dict], upto: int) -> pd.DataFrame:
    """Rows for turns ``<= upto`` (by position), newest last."""
    rows = []
    for t in turns[: upto + 1]:
        src, tgt = t.get("source"), t.get("target")
        if tgt is None:
            target = "-"
        elif src is not None and src != tgt:
            target = f"{src} -> {tgt}"
        else:
            target = str(tgt)
        mitre = t.get("mitre")
        rows.append(
            {
                "turn": t["turn"],
                "actor": t["actor"],
                "action": t["action_id"] + (" (explore)" if t.get("explored") else ""),
                "target": target,
                "ok": "-" if t["action_id"] == "wait" else ("yes" if t.get("success") else "no"),
                "reward": t.get("reward"),
                "mitre": format_mitre(mitre) if mitre else ("not enriched" if not t["enriched"] else "-"),
                "tactic": (mitre or {}).get("tactic", ""),
                "rationale": t.get("rationale") or ("(not enriched)" if not t["enriched"] else ""),
            }
        )
    return pd.DataFrame(
        rows, columns=["turn", "actor", "action", "target", "ok", "reward", "mitre", "tactic", "rationale"]
    )


def decision_web(turn: dict, max_features: int = 8) -> dict:
    """Geometry-free description of the decision web for one turn.

    ``actions``: candidates other than the chosen one, with raw value and a 0..1 weight (min-max over all
    candidates, so the weakest option is 0 and the chosen/best is ~1). ``features``: SHAP features from every
    classifier read this turn, weight = |shap| / max |shap|, sign kept for colour.
    """
    values = {k: float(v) for k, v in (turn.get("decision_values") or {}).items() if v is not None}
    chosen = turn["action_id"]
    lo, hi = (min(values.values()), max(values.values())) if values else (0.0, 0.0)
    span = hi - lo

    def w(v: float) -> float:
        return (v - lo) / span if span > 1e-12 else 1.0

    actions = [
        {"action": a, "value": v, "weight": w(v), "best": v == hi}
        for a, v in sorted(values.items(), key=lambda kv: -kv[1])
        if a != chosen
    ]
    feats = []
    for s in turn.get("shap") or []:
        for f in s.get("top_features") or []:
            if f.get("shap") is None:
                continue
            feats.append(
                {
                    "name": f.get("name", "?"),
                    "model": s.get("model"),
                    "node": s.get("node"),
                    "shap": float(f["shap"]),
                    "value": f.get("value"),
                    "raw": f.get("raw"),
                }
            )
    feats.sort(key=lambda f: -abs(f["shap"]))
    feats = feats[:max_features]
    top = max((abs(f["shap"]) for f in feats), default=0.0)
    for f in feats:
        f["weight"] = abs(f["shap"]) / top if top > 0 else 0.0
    return {
        "chosen": chosen,
        "chosen_value": values.get(chosen),
        "chosen_is_best": bool(values) and values.get(chosen) == hi,
        "value_kind": "Q-value" if turn.get("agent") in (None, "learned") else "heuristic priority",
        "agent": turn.get("agent"),
        "actions": actions,
        "features": feats,
        "has_shap": bool(turn.get("shap")),
        "enriched": bool(turn.get("enriched")),
        "classifiers": [
            {
                "node": s.get("node"),
                "model": s.get("model"),
                "base_value": s.get("base_value"),
                "output": s.get("output"),
            }
            for s in (turn.get("shap") or [])
        ],
    }


# ---------------------------------------------------------------------------------------------- st.cache_data


def _cache(fn):
    if st is None:  # pragma: no cover
        return fn
    return st.cache_data(show_spinner=False, max_entries=16)(fn)


@_cache
def cached_graph(run_dir: str) -> dict:
    return load_graph(Path(run_dir))


@_cache
def cached_summary(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_summary(Path(run_dir))


@_cache
def cached_enriched(run_dir: str) -> dict[int, list[dict]]:
    return load_enriched(Path(run_dir))


@_cache
def cached_index(run_dir: str, size: int, mtime_ns: int) -> EpisodeIndex | None:
    # size/mtime are part of the cache key so a rewritten log is re-indexed
    return build_episode_index(Path(run_dir))


@_cache
def cached_episode(run_dir: str, episode: int, size: int, mtime_ns: int) -> list[dict]:
    index = cached_index(run_dir, size, mtime_ns)
    return read_episode(index, episode) if index else []
