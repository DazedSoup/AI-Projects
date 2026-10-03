"""One command from a fresh clone to a populated dashboard.

    python -m cyberarena.pipeline              # full build (~10-15 min on a laptop CPU)
    python -m cyberarena.pipeline --quick      # smoke build (~5 min)
    python -m cyberarena.pipeline --dry-run    # show what would run

Steps: data -> classifiers -> reference run -> enrich -> experiment. Each step is skipped when its outputs already
exist (``--force`` redoes them). Everything is offline and free: the only network access is the dataset download,
and narration uses the template narrator, never the paid API.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cyberarena import config

STEPS = ("data", "classifiers", "reference", "enrich", "experiment")


@dataclass(frozen=True)
class Profile:
    episodes: int
    reference_label: str
    experiment_name: str
    seeds: str


FULL = Profile(episodes=2000, reference_label="reference", experiment_name="main", seeds="1-5")
QUICK = Profile(episodes=300, reference_label="quick reference", experiment_name="quick", seeds="1-2")


@dataclass
class Step:
    name: str
    describe: str
    done: Callable[[], bool]
    command: Callable[[], list[str]]
    after: Callable[[], None] = field(default=lambda: None)


def _module(*args: str) -> list[str]:
    return [sys.executable, "-m", *args]


def find_run(label: str, runs_dir: Path | None = None) -> Path | None:
    """Newest finished run whose config.json label matches, or None."""
    runs_dir = runs_dir or config.RUNS_DIR
    if not runs_dir.is_dir():
        return None
    for d in sorted((p for p in runs_dir.iterdir() if p.is_dir() and not p.name.startswith(".")), reverse=True):
        try:
            cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
            prog = json.loads((d / "progress.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if cfg.get("label") == label and prog.get("status") == "done":
            return d
    return None


def build_steps(profile: Profile, runs_dir: Path | None = None) -> list[Step]:
    runs_dir = runs_dir or config.RUNS_DIR
    state: dict[str, Path | None] = {"run": None}

    def reference_run() -> Path | None:
        state["run"] = state["run"] or find_run(profile.reference_label, runs_dir)
        return state["run"]

    def enrich_cmd() -> list[str]:
        run = reference_run()
        if run is None:  # only reachable in --dry-run before the reference step has produced one
            return _module("cyberarena.explain.enrich", "--run", f"<{profile.reference_label} run>")
        return _module("cyberarena.explain.enrich", "--run", str(run))

    experiment_dir = runs_dir / "experiments" / profile.experiment_name

    return [
        Step(
            "data", "download + preprocess the three datasets",
            done=lambda: all((config.DATA_PROCESSED / f"{m}.npz").exists() for m in config.CLASSIFIERS),
            command=lambda: _module("cyberarena.ml.datasets", "--all"),
        ),
        Step(
            "classifiers", "train the three TensorFlow detectors",
            done=lambda: all((config.MODELS_DIR / f"{m}.keras").exists() for m in config.CLASSIFIERS),
            command=lambda: _module("cyberarena.ml.train", "--model", "all"),
        ),
        Step(
            "reference", f"train a {profile.episodes:,}-game reference run (seed 7)",
            done=lambda: reference_run() is not None,
            command=lambda: _module("cyberarena.arena.train", "--episodes", str(profile.episodes), "--seed", "7",
                                    "--label", profile.reference_label, "--runs-dir", str(runs_dir)),
            after=lambda: state.update(run=None),
        ),
        Step(
            "enrich", "explain the showcase games (SHAP, MITRE, offline rationale)",
            done=lambda: (r := reference_run()) is not None and (r / "episodes_enriched.jsonl").exists(),
            command=enrich_cmd,
        ),
        Step(
            "experiment", f"multi-seed experiment, seeds {profile.seeds}, adaptive vs frozen detectors",
            done=lambda: (experiment_dir / "aggregate.json").exists(),
            command=lambda: _module("cyberarena.arena.experiment", "--name", profile.experiment_name,
                                    "--seeds", profile.seeds, "--conditions", "adaptive,frozen",
                                    "--episodes", str(profile.episodes)),
        ),
    ]  # fmt: skip


def run(argv: list[str] | None = None, runner: Callable[[list[str]], int] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cyberarena.pipeline", description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="small smoke build (300 games, 2 seeds)")
    ap.add_argument("--only", default=",".join(STEPS), help=f"comma-separated subset of: {', '.join(STEPS)}")
    ap.add_argument("--force", action="store_true", help="redo steps even if their outputs exist")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and commands without running them")
    ap.add_argument("--dashboard", action="store_true", help="launch the Streamlit dashboard at the end")
    args = ap.parse_args(argv)

    wanted = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = sorted(set(wanted) - set(STEPS))
    if unknown:
        print(f"error: unknown step(s): {', '.join(unknown)} (choose from {', '.join(STEPS)})", file=sys.stderr)
        return 2

    runner = runner or (lambda cmd: subprocess.run(cmd, cwd=config.ROOT, check=False).returncode)
    profile = QUICK if args.quick else FULL
    t0 = time.perf_counter()
    for i, step in enumerate(s for s in build_steps(profile) if s.name in wanted):
        head = f"[{i + 1}/{len(wanted)}] {step.name}: {step.describe}"
        if step.done() and not args.force:
            print(f"{head}  -> already done, skipping")
            continue
        cmd = step.command()
        print(f"{head}\n    $ {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        t = time.perf_counter()
        rc = runner(cmd)
        if rc != 0:
            print(f"error: step '{step.name}' failed with exit code {rc}; fix it and re-run, finished steps are kept",
                  file=sys.stderr)  # fmt: skip
            return rc
        step.after()
        print(f"    done in {time.perf_counter() - t:.0f}s", flush=True)

    if not args.dry_run:
        print(f"\npipeline finished in {time.perf_counter() - t0:.0f}s")
    dash = _module("streamlit", "run", str(Path("src") / "cyberarena" / "dashboard" / "app.py"))
    if args.dashboard and not args.dry_run:
        return runner(dash)
    print(f"open the dashboard with:\n    {' '.join(dash)}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
