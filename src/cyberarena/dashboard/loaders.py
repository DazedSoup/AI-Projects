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
from datetime import UTC, date, datetime
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

LEARNING_FILE = "learning.jsonl"
CONFIG_FILE = "config.json"
PROGRESS_FILE = "progress.json"

EVAL_MATCHUPS = {
    # matchup -> (learned side, label)
    "red_learned_vs_blue_baseline": ("red", "Learned red vs scripted blue"),
    "blue_learned_vs_red_baseline": ("blue", "Learned blue vs scripted red"),
}
HEAD_TO_HEAD = "learned_vs_learned"
EVASIVE = "blue_learned_vs_red_evasive"  # v3: learned blue vs the scripted red disguised at a fixed level
MATCHUP_LABEL = {
    "red_learned_vs_blue_baseline": "learned red vs scripted blue",
    "blue_learned_vs_red_baseline": "learned blue vs scripted red",
    HEAD_TO_HEAD: "learned red vs learned blue",
    EVASIVE: "learned blue vs disguised scripted red",
}

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
INDEX_VERSION = 3


class RunFormatError(ValueError):
    """A run file exists but does not match docs/contracts.md."""


# never run directories: the showcase selection file and export folder (docs/contracts.md, v6)
NOT_RUNS = ("showcase", "experiments")


def path_name(p) -> str:
    """Last component of a path written on any OS: manifests record Windows paths (``C:\\...\\runs\\<id>``),
    which ``Path(...).name`` doesn't split on Linux (the hosted dashboard)."""
    return re.split(r"[\\/]", str(p).rstrip("\\/"))[-1]


# ---------------------------------------------------------------------------------------------- run discovery


def list_runs(runs_dir: Path) -> list[Path]:
    """Run directories (those with a summary or episode log), newest name first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    runs = [
        p
        for p in runs_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name not in NOT_RUNS
        and any((p / f).exists() for f in (SUMMARY_FILE, EPISODES_FILE, ENRICHED_FILE))
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
    # per episode, read from the log itself: turns, winner, matchup, phase, after_episode, probe_game
    meta: dict[int, dict] = field(default_factory=dict)

    @property
    def episodes(self) -> list[int]:
        return sorted(self.segments)


def _index_cache_path(path: Path, size: int, mtime_ns: int, cache_dir: Path | None) -> Path:
    key = hashlib.sha1(f"{path.resolve()}|{size}|{mtime_ns}|{INDEX_VERSION}".encode()).hexdigest()[:16]
    base = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "cyberarena-dashboard"
    return base / f"episodes-index-{key}.json"


def _line_meta(line: bytes) -> dict:
    try:
        r = json.loads(line)
    except ValueError:
        return {}
    return {k: r.get(k) for k in ("matchup", "phase", "after_episode", "probe_game", "done", "winner", "evasion")}


def scan_episode_index(path: Path) -> tuple[dict[int, list[tuple[int, int]]], dict[int, dict]]:
    """One sequential pass: contiguous lines of the same episode merge into one ``(start, end)`` range.

    Also returns per-episode metadata read from the first and last record of each episode (matchup, phase,
    checkpoint, ``probe_game``, number of turns, winner), so the game picker can label every logged game
    without reading it.
    """
    segments: dict[int, list[tuple[int, int]]] = {}
    meta: dict[int, dict] = {}
    cur_ep, cur_start, off, last = None, 0, 0, b""

    def close(ep: int | None, last_line: bytes) -> None:
        if ep is not None and last_line:
            m = _line_meta(last_line)
            if m.get("done"):
                meta[ep]["winner"] = m.get("winner")

    with open(path, "rb") as f:
        for line in f:
            m = _EPISODE_RE.search(line, 0, 256) or _EPISODE_RE.search(line)
            ep = int(m.group(1)) if m else None
            if ep != cur_ep:
                if cur_ep is not None:
                    segments.setdefault(cur_ep, []).append((cur_start, off))
                    close(cur_ep, last)
                cur_ep, cur_start, last = ep, off, b""
                if ep is not None and ep not in meta:
                    first = _line_meta(line)
                    meta[ep] = {
                        "turns": 0,
                        "winner": None,
                        "matchup": first.get("matchup"),
                        "phase": first.get("phase"),
                        "after_episode": first.get("after_episode"),
                        "probe_game": bool(first.get("probe_game")),
                        "evasion": first.get("evasion"),
                    }
            if ep is not None and line.strip():
                meta[ep]["turns"] += 1
                last = line
            off += len(line)
    if cur_ep is not None:
        segments.setdefault(cur_ep, []).append((cur_start, off))
        close(cur_ep, last)
    return segments, meta


def build_episode_index(run_dir: Path, cache_dir: Path | None = None, disk_cache: bool = True) -> EpisodeIndex | None:
    """Index ``episodes.jsonl`` by episode, reusing an on-disk cache (in the system temp dir, never the run
    directory) when the file is unchanged. ``disk_cache=False`` (the public showcase) neither reads nor writes
    the cache: the index lives in memory only (``st.cache_data``)."""
    path = Path(run_dir) / EPISODES_FILE
    if not path.exists():
        return None
    st_ = path.stat()
    if not disk_cache:
        segs, meta = scan_episode_index(path)
        return EpisodeIndex(str(path), st_.st_size, st_.st_mtime_ns, segs, meta)
    cache = _index_cache_path(path, st_.st_size, st_.st_mtime_ns, cache_dir)
    if cache.exists():
        try:
            raw = json.loads(cache.read_text(encoding="utf-8"))
            segs = {int(k): [tuple(s) for s in v] for k, v in raw["segments"].items()}
            meta = {int(k): v for k, v in raw["meta"].items()}
            return EpisodeIndex(str(path), st_.st_size, st_.st_mtime_ns, segs, meta)
        except (OSError, ValueError, KeyError):
            pass  # rebuild below
    segs, meta = scan_episode_index(path)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "segments": {str(k): v for k, v in segs.items()},
            "meta": {str(k): v for k, v in meta.items()},
        }
        cache.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass  # a read-only temp dir only costs a rescan next time
    return EpisodeIndex(str(path), st_.st_size, st_.st_mtime_ns, segs, meta)


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


CATALOG_COLUMNS = ["episode", "enriched", "phase", "matchup", "after_episode", "winner", "turns", "probe_game",
                   "evasion", "kind", "label"]  # fmt: skip
GAME_KINDS = {  # picker filter -> label, in display order
    "narrated": "Narrated",
    "probe": "Probe games",
    "eval": "Evaluation",
    "train": "Training",
}


def episode_catalog(
    enriched: dict[int, list[dict]],
    episodes: pd.DataFrame,
    evals: pd.DataFrame,
    index_meta: dict[int, dict] | None = None,
) -> pd.DataFrame:
    """Every replayable game, narrated (enriched) first.

    Built from the summary (training games with ``logged: true``, eval rows' ``episodes: [first, last]``) and,
    when given, the log index metadata, which adds probe games plus the winner and length of every logged game
    and drops games the summary promised but the log doesn't contain.
    """
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
            "probe_game": bool(last.get("probe_game")),
            "evasion": last.get("evasion"),
        }
    if not episodes.empty:
        for r in episodes[episodes["logged"]].itertuples(index=False):
            ep = int(r.episode)
            rows.setdefault(ep, {"episode": ep, "enriched": False, "phase": "train", "matchup": r.matchup,
                                 "after_episode": None, "winner": r.winner, "turns": r.turns,
                                 "probe_game": False})  # fmt: skip
    if not evals.empty:
        for r in evals.itertuples(index=False):
            if r.first is None or r.last is None or pd.isna(r.first):
                continue
            for ep in range(int(r.first), int(r.last) + 1):
                rows.setdefault(ep, {"episode": ep, "enriched": False, "phase": "eval", "matchup": r.matchup,
                                     "after_episode": int(r.after_episode), "winner": None, "turns": None,
                                     "probe_game": False})  # fmt: skip
    if index_meta is not None:
        for ep, m in index_meta.items():
            row = rows.setdefault(ep, {"episode": ep, "enriched": False, "phase": m.get("phase"),
                                       "matchup": m.get("matchup"), "after_episode": m.get("after_episode"),
                                       "winner": None, "turns": None, "probe_game": False})  # fmt: skip
            row["probe_game"] = bool(row["probe_game"] or m.get("probe_game"))
            if row.get("evasion") is None and m.get("evasion") is not None:
                row["evasion"] = m["evasion"]
            if not row["enriched"]:
                row["turns"] = m.get("turns") or row["turns"]
                row["winner"] = m.get("winner") or row["winner"]
            if row.get("after_episode") is None and m.get("after_episode") is not None:
                row["after_episode"] = m["after_episode"]
        rows = {ep: r for ep, r in rows.items() if r["enriched"] or ep in index_meta}
    df = pd.DataFrame(list(rows.values()), columns=CATALOG_COLUMNS[:-2])
    if df.empty:
        return pd.DataFrame(columns=CATALOG_COLUMNS)
    df["probe_game"] = df["probe_game"].fillna(False).astype(bool)
    df["kind"] = [
        "narrated" if r.enriched else "probe" if r.probe_game else "eval" if r.phase == "eval" else "train"
        for r in df.itertuples(index=False)
    ]
    df["label"] = [game_label(r) for r in df.itertuples(index=False)]
    df = df.sort_values(["enriched", "episode"], ascending=[False, True]).reset_index(drop=True)
    return df[CATALOG_COLUMNS]


def matchup_label(matchup: str | None, evasion=None) -> str:
    """Plain-English matchup: ``"learned red vs scripted blue"``; ``matchup@0.7`` or ``evasion`` adds the
    disguise level of the scripted red (v3 evasive eval)."""
    m = matchup or ""
    if "@" in m:
        m, _, lvl = m.partition("@")
        evasion = evasion if evasion is not None else lvl
    if m == EVASIVE:
        try:
            return f"learned blue vs scripted red disguised at {float(evasion):.1f}"
        except (TypeError, ValueError):
            return MATCHUP_LABEL[EVASIVE]
    return MATCHUP_LABEL.get(m, (m or "unknown matchup").replace("_", " "))


def _int(v) -> int | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def game_label(r) -> str:
    """``"Game 2836 — red wins in 33 turns · learned red vs scripted blue · checkpoint 2,000"``."""
    turns = _int(r.turns)
    if isinstance(r.winner, str):
        head = f"{r.winner} wins" + (f" in {turns} turns" if turns else "")
    else:
        head = f"{turns} turns" if turns else "result not indexed yet"
    ev = getattr(r, "evasion", None)
    bits = [f"Game {r.episode} — {head}", matchup_label(r.matchup, None if ev is None or pd.isna(ev) else ev)]
    after = _int(r.after_episode)
    if r.phase == "eval" and after is not None:
        bits.append(f"checkpoint {after:,}")
    elif r.phase == "train":
        bits.append("during training")
    return " · ".join(bits)


def probe_games(catalog: pd.DataFrame) -> pd.DataFrame:
    """Probe games (one fixed-seed game per matchup at every checkpoint), by matchup then checkpoint."""
    if catalog.empty:
        return catalog
    p = catalog[catalog["probe_game"].astype(bool) & catalog["after_episode"].notna()].copy()
    p["after_episode"] = p["after_episode"].astype(int)
    # one probe game per matchup (and per disguise level for the evasive eval) at every checkpoint
    p["probe_key"] = [m if ev is None or pd.isna(ev) or m != EVASIVE else f"{m}@{float(ev):.1f}"
                      for m, ev in zip(p["matchup"], p["evasion"], strict=True)]  # fmt: skip
    return (
        p.sort_values(["probe_key", "after_episode", "enriched"], ascending=[True, True, False])
        .drop_duplicates(["probe_key", "after_episode"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------------------------- run info


def _read_json(path: Path) -> dict:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _parse_dt(s) -> datetime | None:
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt.astimezone().replace(tzinfo=None) if dt.tzinfo else dt


def run_info(run_dir: Path) -> dict:
    """Facts for the run picker: label, games, status, finish time, and which files the run has."""
    run_dir = Path(run_dir)
    cfg = _read_json(run_dir / CONFIG_FILE)
    prog = _read_json(run_dir / PROGRESS_FILE)
    lab = _read_json(run_dir / "lab_status.json")
    games = prog.get("episodes") or cfg.get("episodes") or (cfg.get("params") or {}).get("episodes")
    status = prog.get("status") or ("done" if (run_dir / SUMMARY_FILE).exists() else "unknown")
    if lab.get("stage") in ("starting", "training", "enriching"):
        status = lab["stage"]
    elif status == "error" or lab.get("stage") == "error":
        status = "error"
    finished = None
    if status == "done":
        finished = _parse_dt(lab.get("finished")) or _parse_dt(prog.get("updated"))
        if finished is None and (run_dir / SUMMARY_FILE).exists():
            mtime = (run_dir / SUMMARY_FILE).stat().st_mtime
            finished = datetime.fromtimestamp(mtime, tz=UTC).astimezone().replace(tzinfo=None)
    return {
        "run_id": run_dir.name,
        "label": cfg.get("label") or lab.get("label") or "",
        "games": int(games) if games else None,
        "status": status,
        "episode": prog.get("episode"),
        "finished": finished,
        # adaptive: v2 telemetry from a run whose detectors learned; frozen: a --no-adaptive comparison run
        "adaptive": (run_dir / LEARNING_FILE).exists() and cfg.get("adaptive") is not False,
        "frozen": cfg.get("adaptive") is False,
        "has_narration": (run_dir / ENRICHED_FILE).exists(),
        "agent_type": _agent_type(cfg, run_dir),
        "seed": cfg.get("seed"),
        "rules": (cfg.get("params") or {}).get("rules"),  # v5: "default" | "cheap-isolation"; None before v5
        "files": [
            f
            for f in (SUMMARY_FILE, LEARNING_FILE, EPISODES_FILE, ENRICHED_FILE, GRAPH_FILE)
            if (run_dir / f).exists()
        ],
    }


def _agent_type(cfg: dict, run_dir: Path) -> str | None:
    agents = cfg.get("agents") if isinstance(cfg.get("agents"), dict) else {}
    if agents.get("type"):
        return str(agents["type"])
    a = run_dir / "agents"
    if (a / "red_qnet.keras").exists() or (a / "blue_qnet.keras").exists():
        return "dqn"
    if (a / "red.json").exists() or (a / "blue.json").exists():
        return "tabular"
    return None


def run_label(info: dict, today: date | None = None) -> str:
    """``"reference · 2,000 games · finished 17:11"`` (the date is added when it isn't today)."""
    today = today or datetime.now(UTC).astimezone().date()
    bits = [info.get("label") or f"Run {info['run_id']}"]
    if info.get("games"):
        bits.append(f"{info['games']:,} games")
    status, fin = info.get("status"), info.get("finished")
    if status == "done" and fin is not None:
        when = fin.strftime("%H:%M") if fin.date() == today else f"{fin:%b} {fin.day}, {fin:%H:%M}"
        bits.append(f"finished {when}")
    elif status in ("running", "training", "starting", "enriching"):
        ep, n = info.get("episode"), info.get("games")
        bits.append(
            ("training" if status in ("running", "training") else status)
            + (f" {ep / n:.0%}" if ep and n else "")
        )
    elif status == "error":
        bits.append("stopped early")
    return " · ".join(bits)


def checkpoint_q_states(run_dir: Path) -> pd.DataFrame:
    """Q-table size per saved checkpoint, ``after_episode, side, n_states``.

    Reads the plain-JSON agent checkpoints (``agents/checkpoints/{red,blue}_<after_episode>.json``) so Q-table
    growth is available even for runs that predate ``learning.jsonl``.
    """
    rows = []
    ck = Path(run_dir) / "agents" / "checkpoints"
    if ck.is_dir():
        for f in ck.glob("*_*.json"):
            side, _, num = f.stem.partition("_")
            if side in ("red", "blue") and num.isdigit():
                q = _read_json(f).get("q")
                if isinstance(q, dict):
                    rows.append({"after_episode": int(num), "side": side, "n_states": len(q)})
    df = pd.DataFrame(rows, columns=["after_episode", "side", "n_states"])
    return df.sort_values(["side", "after_episode"]).reset_index(drop=True)


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


def file_stamp(path: Path) -> tuple[int, int] | None:
    """``(size, mtime_ns)`` of a file, or ``None``: a cache key that changes when a live run rewrites it."""
    try:
        s = Path(path).stat()
    except OSError:
        return None
    return (s.st_size, s.st_mtime_ns)


@_cache
def cached_summary(run_dir: str, stamp: tuple | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    # stamp: file_stamp(summary.jsonl), so a run that is still training is re-read when it grows
    return load_summary(Path(run_dir))


@_cache
def cached_enriched(run_dir: str, stamp: tuple | None = None) -> dict[int, list[dict]]:
    return load_enriched(Path(run_dir))


@_cache
def cached_index(run_dir: str, size: int, mtime_ns: int, disk_cache: bool = True) -> EpisodeIndex | None:
    # size/mtime are part of the cache key so a rewritten log is re-indexed
    return build_episode_index(Path(run_dir), disk_cache=disk_cache)


@_cache
def cached_episode(run_dir: str, episode: int, size: int, mtime_ns: int, disk_cache: bool = True) -> list[dict]:
    index = cached_index(run_dir, size, mtime_ns, disk_cache)
    return read_episode(index, episode) if index else []


@_cache
def cached_run_info(run_dir: str, stamp: tuple | None = None) -> dict:
    # stamp: progress.json / lab_status.json stamps, so a live run's label updates
    return run_info(Path(run_dir))


@_cache
def cached_q_states(run_dir: str, stamp: tuple | None = None) -> pd.DataFrame:
    return checkpoint_q_states(Path(run_dir))


def run_stamp(run_dir: Path) -> tuple:
    """Cache key for everything small in a run: changes whenever a live run rewrites its status files."""
    run_dir = Path(run_dir)
    return tuple(
        file_stamp(run_dir / f) for f in (PROGRESS_FILE, "lab_status.json", SUMMARY_FILE, LEARNING_FILE)
    )


# ---------------------------------------------------------------------------------------------- adaptation (v2)


def adaptation(turn: dict) -> list[dict]:
    """Detector reads of this turn that carry v2 adaptation fields (``evasion``/``version``)."""
    out = []
    for ci in turn.get("classifier_inputs") or []:
        if ci.get("evasion") is None and ci.get("version") is None:
            continue
        out.append({k: ci.get(k) for k in ("node", "model", "evasion", "version", "score")})
    return out


def adaptation_line(turn: dict) -> str | None:
    """``"red evasion 0.4 on network · detector v4 scored 0.71"`` (``None`` on v1 runs / turns with no read)."""
    parts = []
    for a in adaptation(turn):
        bits = []
        if a["evasion"] is not None:
            bits.append(f"red evasion {float(a['evasion']):.1f} on {a['model']}")
        if a["version"] is not None:
            bits.append(
                f"detector v{a['version']}"
                + (f" scored {float(a['score']):.2f}" if a["score"] is not None else "")
            )
        parts.append(" · ".join(bits))
    return "; ".join(parts) or None


# ---------------------------------------------------------------------------------------------- adaptive vs frozen


def pooled_win_rate(curve: pd.DataFrame, side: str, from_episode: int) -> dict | None:
    """Wins and games pooled over the checkpoints at or after ``from_episode`` for one learned side, with a
    Wilson 95% interval on the pooled rate."""
    sub = (
        curve[(curve["side"] == side) & (curve["after_episode"] >= from_episode)]
        if not curve.empty
        else curve
    )
    if sub.empty:
        return None
    wins, n = int(sub["wins"].sum()), int(sub["n"].sum())
    lo, hi = wilson_interval(wins, n)
    return {"rate": wins / n if n else 0.0, "lo": lo, "hi": hi, "n": n, "wins": wins,
            "first": int(sub["after_episode"].min()), "last": int(sub["after_episode"].max())}  # fmt: skip


def adaptive_vs_frozen_takeaway(adaptive: pd.DataFrame, frozen: pd.DataFrame, side: str = "blue") -> str:
    """Pooled second-half comparison, plus an honest note on the final checkpoint alone."""
    if adaptive.empty or frozen.empty:
        return "One of the two runs has no checkpoint evaluations yet."
    last = int(min(adaptive["after_episode"].max(), frozen["after_episode"].max()))
    start = last // 2
    a, f = pooled_win_rate(adaptive, side, start), pooled_win_rate(frozen, side, start)
    if a is None or f is None:
        return "Not enough shared checkpoints to compare."
    other = "red" if side == "blue" else "blue"
    gap = (
        "a clear gap: the intervals don't overlap"
        if a["lo"] > f["hi"] or f["lo"] > a["hi"]
        else ("but the intervals overlap, so the gap is within noise")
    )
    line = (f"Over checkpoints {a['first']:,}–{a['last']:,}, learned {side} beats the scripted {other} "
            f"{a['rate']:.0%} of the time with adaptive detectors (95% CI {a['lo']:.0%}–{a['hi']:.0%}, "
            f"{a['n']:,} games) vs {f['rate']:.0%} with frozen ones ({f['lo']:.0%}–{f['hi']:.0%}), {gap}.")  # fmt: skip

    def at(c):
        r = c[(c["side"] == side) & (c["after_episode"] == last)]
        return None if r.empty else r.iloc[0]

    ra, rf = at(adaptive), at(frozen)
    if ra is not None and rf is not None:
        overlap = not (ra["ci_low"] > rf["ci_high"] or rf["ci_low"] > ra["ci_high"])
        line += (f" At the final checkpoint alone ({ra['win_rate']:.0%} vs {rf['win_rate']:.0%}) the difference "
                 + ("is within noise." if overlap else "is also outside noise."))  # fmt: skip
    return line


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"(])")


def sentences(text: str | None) -> list[str]:
    """Split a rationale into sentences (decimals like ``0.71`` stay intact)."""
    return [x.strip() for x in _SENTENCE.split(text or "") if x.strip()]


def adaptation_facts(turn: dict) -> dict | None:
    """Structured adaptation facts for one turn, or ``None`` on runs without adaptive learning.

    Prefers the enriched ``adaptation`` field (``evasion``, ``detector_versions``, ``updated_since_last_move``,
    ``caught_rate`` …); falls back to the raw turn (``classifier_inputs[].evasion/version``,
    ``detector_versions``) for games that aren't narrated.
    """
    a = turn.get("adaptation")
    reads = adaptation(turn)
    if not a and not reads and not turn.get("detector_versions"):
        return None
    a = dict(a or {})
    evasion = dict(a.get("evasion") or {})
    versions = dict(a.get("detector_versions") or turn.get("detector_versions") or {})
    for r in reads:
        if r["evasion"] is not None:
            evasion.setdefault(r["model"], float(r["evasion"]))
        if r["version"] is not None:
            versions.setdefault(r["model"], int(r["version"]))
    return {
        "evasion": evasion,
        "versions": versions,
        "read": [r["model"] for r in reads],
        "scores": {r["model"]: r["score"] for r in reads if r["score"] is not None},
        "retrained": list(a.get("updated_since_last_move") or []),
        "caught_rate": {k: v for k, v in (a.get("caught_rate") or {}).items() if v is not None},
    }


# ---------------------------------------------------------------------------------------------- v4: concrete moves

ARROW = "→"


def action_word(action: str | None) -> str:
    return (action or "?").replace("_", " ")


def parse_move_key(key: str) -> tuple[str, int | None]:
    """``"lateral_move→13"`` -> ``("lateral_move", 13)``; ``"wait"`` -> ``("wait", None)``."""
    a, sep, t = str(key).partition(ARROW)
    if not sep:
        return a, None
    try:
        return a, int(t)
    except ValueError:
        return a, None


def move_key(action: str, target) -> str:
    return f"{action}{ARROW}{target}" if target is not None else str(action)


def turn_candidates(turn: dict) -> list[dict]:
    """The concrete moves a v4 (DQN) agent scored this turn, best first: ``[{action, source, target, q, key,
    chosen}]``. Falls back to parsing ``"<action>→<target>"`` decision-value keys; ``[]`` for tabular and
    scripted agents (their decision values are per action type, not per move)."""
    raw = turn.get("candidates")
    out = []
    if raw:
        for c in raw:
            if c.get("q") is None:
                continue
            out.append({"action": c.get("action"), "source": c.get("source"), "target": c.get("target"),
                        "q": float(c["q"])})  # fmt: skip
    else:
        vals = turn.get("decision_values") or {}
        if not any(ARROW in str(k) for k in vals):
            return []
        for k, v in vals.items():
            if v is None:
                continue
            a, t = parse_move_key(k)
            out.append({"action": a, "source": None, "target": t, "q": float(v)})
    out.sort(key=lambda c: -c["q"])
    chosen_a, chosen_t, chosen_s = turn.get("action_id"), turn.get("target"), turn.get("source")
    found = False
    for c in out:
        c["key"] = move_key(c["action"], c["target"])
        hit = (not found and c["action"] == chosen_a and c["target"] == chosen_t
               and (c["source"] is None or chosen_s is None or c["source"] == chosen_s))  # fmt: skip
        c["chosen"] = bool(hit)
        found = found or hit
    return out


def runner_up(cands: list[dict]) -> dict | None:
    """Best-valued candidate that wasn't chosen (the move the agent nearly made)."""
    return next((c for c in cands if not c.get("chosen")), None)


def node_role(graph: dict | None, node: int | None) -> str | None:
    if graph is None or node is None:
        return None
    n = next((x for x in graph.get("nodes", []) if x["id"] == node), None)
    if n is None:
        return None
    if n.get("crown_jewel"):
        return "crown jewel"
    role = str(n.get("role", "")).lower()
    if role in ("dmz", "internet", "edge", "gateway"):
        return "DMZ"
    if role.startswith("work") or role in ("host", "client", "endpoint"):
        return "workstation"
    return role or None


def host_text(graph: dict | None, node: int | None) -> str:
    if node is None:
        return ""
    role = node_role(graph, node)
    return f"host {node}" + (f" ({role})" if role else "")


def move_phrase(action: str | None, target, graph: dict | None = None, source=None) -> str:
    """``"lateral move → host 13 (server)"``; ``"wait"`` for a move without a target."""
    a = action_word(action)
    if target is None:
        return a
    src = f" from host {source}" if source is not None and source != target else ""
    return f"{a}{src} {ARROW} {host_text(graph, target)}"


def short_move(action: str | None, target) -> str:
    """``"lateral move → 13"``: compact label for charts."""
    a = action_word(action)
    return a if target is None else f"{a} {ARROW} {target}"


def move_headline(turn: dict, graph: dict | None = None) -> str:
    return move_phrase(turn.get("action_id"), turn.get("target"), graph, turn.get("source"))


# plain-English names for the v4 agents' inputs (arena/features.py FEATURE_NAMES); unknown names are prettified
_MODELS = ("malware", "phishing", "network")
FEATURE_PHRASES: dict[str, str] = {
    "g_turn_frac": "how late in the game it is",
    "g_footholds": "how many hosts red holds",
    "g_admin_footholds": "hosts red holds with admin rights",
    "g_has_server": "red already holds a server",
    "g_has_crown": "red is on the crown jewel",
    "g_crown_admin": "red is admin on the crown jewel",
    "g_front_dist": "red's closest distance to the crown jewel",
    "g_hot": "how suspicious red's hosts look",
    "g_detected": "hosts blue has flagged",
    "g_isolated": "hosts blue has isolated",
    "g_confirmed": "hosts confirmed compromised",
    "g_patched_frac": "share of hosts patched",
    "g_top_undetected": "highest score among unflagged hosts",
    "g_crown_detected": "crown jewel is flagged",
    "g_crown_isolated": "crown jewel is isolated",
    "g_crown_score": "crown jewel's detector score",
    "t_none": "move has no target host",
    "t_dmz": "target is in the DMZ",
    "t_workstation": "target is a workstation",
    "t_server": "target is a server",
    "t_crown": "target is the crown jewel",
    "t_compromised": "target is already red's",
    "t_privilege": "red's rights on the target",
    "t_isolated": "target is isolated",
    "t_patched": "target is patched",
    "t_recon": "target was already scanned",
    "t_detected": "target is flagged",
    "t_confirmed": "target is a confirmed compromise",
    "t_score_max": "target's highest detector score",
    "t_dist": "target's hops to the crown jewel",
    "t_degree": "how many links the target has",
    "t_closer": "move brings red closer to the crown jewel",
    "t_open_nbrs": "target's neighbours red doesn't hold",
    "t_nbr_detected": "flagged neighbours of the target",
    "s_has": "move starts from a held host",
    "s_privilege": "red's rights on the source host",
    "s_recon": "source host was scanned",
    "s_score_max": "source host's highest detector score",
    "s_dist": "source host's hops to the crown jewel",
    "p_success": "chance the move succeeds",
    **{f"g_evasion_{m}": f"red's {m} disguise level" for m in _MODELS},
    **{f"t_score_{m}": f"target's {m} score" for m in _MODELS},
}


# on/off features: the phrase when the value is 0 (the "on" phrase above would say the opposite)
FEATURE_OFF: dict[str, str] = {
    "g_has_server": "red holds no server yet", "g_has_crown": "red isn't on the crown jewel yet",
    "g_crown_admin": "red isn't admin on the crown jewel", "g_crown_detected": "crown jewel isn't flagged",
    "g_crown_isolated": "crown jewel isn't isolated", "t_none": "move targets a host",
    "t_dmz": "target isn't in the DMZ", "t_workstation": "target isn't a workstation",
    "t_server": "target isn't a server", "t_crown": "target isn't the crown jewel",
    "t_compromised": "target isn't red's yet", "t_isolated": "target isn't isolated",
    "t_patched": "target is unpatched", "t_recon": "target not scanned yet", "t_detected": "target isn't flagged",
    "t_confirmed": "target isn't a confirmed compromise", "t_closer": "move gets red no closer to the crown jewel",
    "s_has": "move needs no foothold", "s_recon": "source host not scanned",
}  # fmt: skip


def feature_phrase(name: str, entry: dict | None = None) -> str:
    """Plain-English name of an agent input feature, worded for its value when it is an on/off feature (so
    ``t_workstation = 0`` reads "target isn't a workstation"). An explain-supplied ``phrase``/``label`` wins."""
    if entry:
        for k in ("phrase", "label", "description"):
            if isinstance(entry.get(k), str) and entry[k].strip():
                return entry[k].strip()
    value = (entry or {}).get("value")
    off = isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0.5
    if name.startswith("a_"):
        return f"move {'is not' if off else 'is'} {action_word(name[2:])}"
    if off and name in FEATURE_OFF:
        return FEATURE_OFF[name]
    if name in FEATURE_PHRASES:
        return FEATURE_PHRASES[name]
    prefix = {"g": "game:", "t": "target:", "s": "source:", "p": ""}
    head, _, rest = name.partition("_")
    if head in prefix and rest:
        return f"{prefix[head]} {rest.replace('_', ' ')}".strip()
    return name.replace("_", " ")


def attribution_features(turn: dict, max_n: int = 6) -> list[dict]:
    """``agent_attribution.top_features`` (integrated gradients on the Q-network) with plain-English names,
    largest |attribution| first, weight = |attribution| / max. ``[]`` when the turn isn't attributed."""
    att = turn.get("agent_attribution") or {}
    out = []
    for f in att.get("top_features") or []:
        a = f.get("attribution")
        if a is None or f.get("name") is None:
            continue
        out.append({"name": f["name"], "phrase": feature_phrase(f["name"], f), "value": f.get("value"),
                    "attribution": float(a)})  # fmt: skip
    out.sort(key=lambda f: -abs(f["attribution"]))
    out = out[:max_n]
    top = max((abs(f["attribution"]) for f in out), default=0.0)
    for f in out:
        f["weight"] = abs(f["attribution"]) / top if top > 0 else 0.0
    return out


def decision_web_v4(turn: dict, graph: dict | None = None, max_moves: int = 6, max_features: int = 6) -> dict:
    """Decision web for an agent that scores concrete moves. Centre: the chosen move. Ring 1: the other top
    candidate moves (weight = min-max Q over the moves shown; the runner-up is flagged). Ring 2: the agent's
    own top attributed input features (``agent_attribution``), signed. ``detector``: the detector SHAP features
    of this turn, for a separate panel."""
    cands = turn_candidates(turn)
    chosen = next((c for c in cands if c["chosen"]), None)
    others = [c for c in cands if not c["chosen"]][:max_moves]
    shown = ([chosen] if chosen else []) + others
    qs = [c["q"] for c in shown]
    lo, hi = (min(qs), max(qs)) if qs else (0.0, 0.0)
    span = hi - lo
    ru = runner_up(cands)
    moves = [{**c, "label": short_move(c["action"], c["target"]), "long": move_phrase(c["action"], c["target"], graph),
              "weight": (c["q"] - lo) / span if span > 1e-12 else 1.0, "runner_up": ru is not None and c is ru,
              "rank": cands.index(c) + 1} for c in others]  # fmt: skip
    base = decision_web(turn)
    att = turn.get("agent_attribution") or {}
    return {
        **base,
        "mode": "moves",
        "chosen_label": short_move(turn.get("action_id"), turn.get("target")),
        "chosen_long": move_headline(turn, graph),
        "chosen_value": chosen["q"] if chosen else None,
        "chosen_rank": cands.index(chosen) + 1 if chosen else None,
        "chosen_is_best": bool(cands) and chosen is cands[0],
        "n_candidates": len(cands),
        "moves": moves,
        "attribution": attribution_features(turn, max_features),
        "attribution_meta": {k: att.get(k) for k in ("method", "baseline", "q", "checkpoint")} if att else None,
        "detector": base["features"],
        "value_kind": "Q-value",
    }


def hops_to_crown(graph: dict | None) -> dict[int, int]:
    """Shortest number of links from every host to the crown jewel (BFS over ``graph.json`` edges)."""
    if not graph:
        return {}
    crown = next((n["id"] for n in graph["nodes"] if n.get("crown_jewel")), None)
    if crown is None:
        return {}
    nbrs: dict[int, list[int]] = {}
    for u, v in graph["edges"]:
        nbrs.setdefault(u, []).append(v)
        nbrs.setdefault(v, []).append(u)
    dist, frontier = {crown: 0}, [crown]
    while frontier:
        nxt = []
        for u in frontier:
            for v in nbrs.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


# fact -> (feature-name prefix whose attribution ranks it, label, number format)
_FACTS = {
    "action": ("a_", "move type", None),
    "role": ("t_crown", "target's role", None),
    "hops": ("t_dist", "hops to the crown jewel", "{:.0f}"),
    "score": ("t_score", "target's highest detector score", "{:.2f}"),
    "owned": ("t_compromised", "target already red's", None),
    "flagged": ("t_detected", "target flagged by blue", None),
    "patched": ("t_patched", "target patched", None),
    "degree": ("t_degree", "links on the target", "{:.0f}"),
}


def _target_facts(c: dict, states: dict[int, dict], graph: dict | None, hops: dict[int, int], actor: str) -> dict:
    t = c.get("target")
    ns = states.get(t, {}) if t is not None else {}
    scores = ns.get("scores") or {}
    deg = sum(1 for u, v in (graph or {}).get("edges", []) if t in (u, v)) if t is not None else None
    facts = {"action": action_word(c.get("action")), "role": node_role(graph, t) or ("none" if t is None else "host"),
             "hops": hops.get(t) if t is not None else None,
             "score": max(scores.values()) if scores else None,
             "flagged": bool(ns.get("detected")) if t is not None else None,
             "patched": bool(ns.get("patched")) if t is not None else None, "degree": deg}  # fmt: skip
    if actor == "red":  # blue never sees ground-truth compromise
        facts["owned"] = bool(ns.get("compromised")) if t is not None else None
    return facts


def chosen_vs_runner_up(turn: dict, before_states: list[dict] | None, graph: dict | None, n: int = 3) -> dict | None:
    """Q margin between the chosen move and the runner-up, plus the ``n`` facts about the two moves that
    differ most. Facts come from the logged state *before* the move (``before_states``: the previous turn's
    ``node_states``) and ``graph.json``; when ``agent_attribution`` is present, facts whose matching input
    feature carried more attribution rank first. ``None`` when the turn has fewer than two candidates."""
    cands = turn_candidates(turn)
    chosen = next((c for c in cands if c["chosen"]), None)
    ru = runner_up(cands)
    if chosen is None or ru is None:
        return None
    states = {ns["id"]: ns for ns in (before_states or turn.get("node_states") or [])}
    hops = hops_to_crown(graph)
    actor = turn.get("actor", "red")
    a, b = _target_facts(chosen, states, graph, hops, actor), _target_facts(ru, states, graph, hops, actor)
    att = {f["name"]: abs(f["attribution"]) for f in attribution_features(turn, 99)}
    top_att = max(att.values(), default=0.0) or 1.0
    diffs = []
    for k, (feat, label, fmt) in _FACTS.items():
        if k not in a or a[k] == b[k]:
            continue
        va, vb = a[k], b[k]
        num = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (va, vb))
        size = abs(va - vb) / {"hops": 4.0, "degree": 4.0}.get(k, 1.0) if num else 0.6
        w = max((v for f, v in att.items() if f.startswith(feat)), default=0.0) / top_att
        diffs.append({"fact": k, "label": label, "chosen": _fmt_fact(va, fmt), "runner_up": _fmt_fact(vb, fmt),
                      "rank": 2 * w + min(1.0, size)})  # fmt: skip
    diffs.sort(key=lambda d: -d["rank"])
    return {"chosen": chosen, "runner_up": ru, "margin": chosen["q"] - ru["q"],
            "chosen_label": move_phrase(chosen["action"], chosen["target"], graph),
            "runner_label": move_phrase(ru["action"], ru["target"], graph),
            "facts": diffs[:n], "ranked_by_attribution": bool(att)}  # fmt: skip


def _fmt_fact(v, fmt: str | None) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if fmt and isinstance(v, (int, float)):
        return fmt.format(v)
    return str(v)


def is_experiment_run(info: dict) -> bool:
    """A run launched by ``arena.experiment`` (its label is ``"<name> · <condition> · seed <s>"``)."""
    return " · seed " in (info.get("label") or "")


def agent_kind(info: dict) -> str:
    """``"DQN agents"``, ``"tabular agents"`` or ``""`` for badges and captions."""
    return {"dqn": "DQN agents", "tabular": "tabular agents"}.get(info.get("agent_type") or "", "")
