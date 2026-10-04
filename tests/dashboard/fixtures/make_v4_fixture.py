"""Cut the **v3/v4** dashboard fixtures out of real runs (deterministic; re-run after the arena changes format).

    .venv/Scripts/python.exe tests/dashboard/fixtures/make_v4_fixture.py

Writes, under ``tests/dashboard/fixtures/runs_v4/``:

* ``20261003-013911-7``: v4 (DQN agents: ``candidates``, ``chosen_features``, ``agent_attribution``) cut from the
  reference run: a few logged games, the narrated games' first turns, every eval row, a thinned learning log;
* ``20261003-014109-7``: v3 (tabular agents, disguised-red eval rows) cut from the tabular comparison run;
* ``experiments/``: ``main`` (current rules), ``main-cheap-isolation`` (previous rules), ``smoke-dqn`` and the
  ``diag-*`` experiments (manifest + aggregate only), a running
  experiment without an aggregate (``fx-running``), and two cross-evaluation files.

Classifier input rows are shortened to 4 values (the dashboard never reads them) to keep the fixture small.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REAL = ROOT / "runs"
OUT = Path(__file__).parent / "runs_v4"
V4, V3 = "20261003-110349-7", "20261003-110348-7"  # v5 rules: reference (DQN) and tabular comparison


def _lines(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _slim(rec: dict) -> dict:
    for ci in rec.get("classifier_inputs") or []:
        ci["row"] = (ci.get("row") or [])[:4]
    return rec


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def cut_run(run_id: str, games_per_kind: int, narrated: int, max_games: int = 5) -> None:
    src, dst = REAL / run_id, OUT / run_id
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    for f in ("config.json", "graph.json", "progress.json"):
        shutil.copy(src / f, dst / f)
    # keep what the dashboard reads from agents/: the saved-agent files (empty stand-ins) and their json
    (dst / "agents").mkdir()
    for f in (src / "agents").iterdir():
        if f.is_file():
            (dst / "agents" / f.name).write_bytes(f.read_bytes() if f.suffix == ".json" and f.stat().st_size < 200_000 else b"")
    # every eval row and every 4th game row
    summ = [r for r in _lines(src / "summary.jsonl") if r.get("kind") == "eval" or r.get("episode", 0) % 4 == 0]
    _write(dst / "summary.jsonl", summ)
    # learning log: every update / evasion / probe row, every other stats row
    if (src / "learning.jsonl").exists():
        lr = [r for r in _lines(src / "learning.jsonl") if r.get("kind") != "agent_stats" or r.get("episode", 0) % 100 == 0]
        for r in lr:
            r.pop("state", None)
        _write(dst / "learning.jsonl", lr)
    # a few logged games of each kind (train / eval per matchup / probe)
    picked: dict[tuple, int] = {}
    keep: set[int] = set()
    first_seen: dict[int, dict] = {}
    for r in _lines(src / "episodes.jsonl"):
        ep = r["episode"]
        if ep not in first_seen:
            first_seen[ep] = r
            k = (r.get("phase"), r.get("matchup"), bool(r.get("probe_game")), r.get("evasion"))
            if picked.get(k, 0) < games_per_kind and len(keep) < max_games:
                picked[k] = picked.get(k, 0) + 1
                keep.add(ep)
    _write(dst / "episodes.jsonl", (_slim(r) for r in _lines(src / "episodes.jsonl") if r["episode"] in keep))
    if narrated and (src / "episodes_enriched.jsonl").exists():
        eps: list[int] = []
        rows = []
        for r in _lines(src / "episodes_enriched.jsonl"):
            if r["episode"] not in eps:
                if len(eps) >= narrated:
                    continue
                eps.append(r["episode"])
            rows.append(_slim(r))
        _write(dst / "episodes_enriched.jsonl", rows)


def cut_experiments() -> None:
    src, dst = REAL / "experiments", OUT / "experiments"
    shutil.rmtree(dst, ignore_errors=True)
    names = ["main", "main-cheap-isolation", "smoke-dqn"] + sorted(p.name for p in src.glob("diag-*") if (p / "aggregate.json").exists())
    for n in names:
        (dst / n).mkdir(parents=True)
        man = json.loads((src / n / "manifest.json").read_text(encoding="utf-8"))
        for r in man["runs"]:  # point at runs that don't exist in the fixture (the page must cope)
            r["run_dir"] = str(Path("runs-not-in-fixture") / Path(r["run_dir"] or "x").name)
        (dst / n / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
        shutil.copy(src / n / "aggregate.json", dst / n / "aggregate.json")
    run = {"name": "fx-running", "created": "2026-10-03T09:00:00", "episodes": 300, "seeds": [1, 2],
           "conditions": ["adaptive", "frozen"], "extra_args": [], "jobs": 4, "status": "running", "finished": None,
           "wall_time_s": None,
           "runs": [{"seed": s, "condition": c, "run_dir": None, "status": "running" if s == 1 else "pending",
                     "log": f"logs/{c}_seed{s}.log", "started": None, "finished": None, "returncode": None,
                     "error": None} for s in (1, 2) for c in ("adaptive", "frozen")]}  # fmt: skip
    (dst / "fx-running").mkdir(parents=True)
    (dst / "fx-running" / "manifest.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    xs = REAL / "diag-crosseval"
    if xs.is_dir():
        (OUT / "diag-crosseval").mkdir(parents=True, exist_ok=True)
        for f in sorted(xs.glob("main_seed[12].json")):
            obj = json.loads(f.read_text(encoding="utf-8"))
            obj.pop("saliency", None)
            (OUT / "diag-crosseval" / f.name).write_text(json.dumps(obj, indent=1), encoding="utf-8")


if __name__ == "__main__":
    cut_run(V4, games_per_kind=1, narrated=3)
    cut_run(V3, games_per_kind=1, narrated=0, max_games=3)
    cut_experiments()
    print("wrote", OUT)
