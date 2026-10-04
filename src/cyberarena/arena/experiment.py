"""Multi-seed experiments (contract: docs/contracts.md, "Experiments & rigor (v3)"). SIMULATION ONLY.

    python -m cyberarena.arena.experiment --name main --seeds 1-5 --conditions adaptive,frozen --episodes 2000 \\
        [--jobs N] [-- <extra train args>]
    python -m cyberarena.arena.experiment --aggregate runs/experiments/main

Each (seed, condition) is one ``cyberarena.arena.train`` subprocess (``frozen`` adds ``--no-adaptive``), run in
parallel. ``runs/experiments/<name>/manifest.json`` is written before anything starts and rewritten whenever a
run changes state; a run that fails is marked ``error`` and the others carry on. Ctrl+C (or CTRL_BREAK /
SIGTERM to this process) cancels every child the way the Lab does -- CTRL_BREAK_EVENT to its own process
group on Windows, SIGTERM elsewhere -- so each train writes ``status: error, error: "cancelled"``; the
experiment exits 130. ``aggregate.json`` is written when all runs have finished, and ``--aggregate``
recomputes it from the run directories already on disk.

Statistics (numpy + ``scipy.stats``, which scikit-learn already depends on):

- A matchup's value is always the learned side's win rate; ``blue_learned_vs_red_evasive`` is keyed
  ``blue_learned_vs_red_evasive@<s>``.
- ``curves`` / ``final`` / ``late_mean``: mean, sample sd (ddof 1) and a 95% t-interval across seeds.
  ``late_mean`` first averages each seed's checkpoints with ``after_episode >= episodes / 2``.
- ``contrasts``: adaptive minus frozen, paired by seed (paired t-test, two-sided p; t-interval of the mean
  difference). Only seeds where both conditions finished are used.
- ``arms_race`` (adaptive runs): see ``count_cycles`` for what one oscillation cycle is.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

PROG = "cyberarena.arena.experiment"
CONDITIONS = {"adaptive": [], "frozen": ["--no-adaptive"]}
EVASIVE = "blue_learned_vs_red_evasive"
# contrasts listed first (the experiment's headline questions); every other shared metric follows
KEY_CONTRASTS = (f"{EVASIVE}@0.7", f"{EVASIVE}@0.4", "blue_learned_vs_red_baseline")
CYCLE_MIN_SWING = 0.2  # evasion change (two grid steps) that counts as a real rise or fall, not bandit jitter
ARMS_LEVEL = "0.7"
POLL_S = 0.25
WIN = sys.platform == "win32"


class Cancelled(Exception):
    """CTRL_BREAK / SIGTERM delivered to the experiment process."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):
        self.exit(2, f"{self.prog}: error: {message}\n")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_json(path: Path, obj: Any) -> None:
    """Atomic rewrite (temp file + rename, retried while a Windows reader holds the target open)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    for i in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.02 * (i + 1))
    os.replace(tmp, path)


def parse_seeds(text: str) -> list[int]:
    """``"1-5"`` -> [1..5]; ``"1,3,8-9"`` -> [1, 3, 8, 9]."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, sep, b = part.partition("-")
        try:
            lo, hi = int(a), int(b) if sep else int(a)
        except ValueError:
            raise ValueError(f"bad --seeds part {part!r}") from None
        if hi < lo:
            raise ValueError(f"bad --seeds range {part!r}")
        out.extend(range(lo, hi + 1))
    if not out or len(set(out)) != len(out):
        raise ValueError("--seeds must list distinct seeds")
    return out


# --------------------------------------------------------------------------------------------- statistics


def t_quantile(q: float, df: int) -> float:
    try:
        from scipy import stats

        return float(stats.t.ppf(q, df))
    except ImportError:  # normal approximation; scipy ships with scikit-learn, so this is a last resort
        from statistics import NormalDist

        return NormalDist().inv_cdf(q)


def t_sf(t: float, df: int) -> float:
    try:
        from scipy import stats

        return float(stats.t.sf(t, df))
    except ImportError:
        from statistics import NormalDist

        return 1.0 - NormalDist().cdf(t)


def _r(x: float | None, nd: int = 4) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return round(float(x), nd)


def describe(values: list[float]) -> dict[str, Any]:
    """Mean, sample sd and 95% t-interval across seeds (sd / ci95 are null for a single seed)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": None, "sd": None, "ci95": None, "n": 0, "per_seed": []}
    mean = float(v.mean())
    if n < 2:
        return {"mean": _r(mean), "sd": None, "ci95": None, "n": 1, "per_seed": [_r(x) for x in v]}
    sd = float(v.std(ddof=1))
    half = t_quantile(0.975, n - 1) * sd / math.sqrt(n)
    return {"mean": _r(mean), "sd": _r(sd), "ci95": [_r(mean - half), _r(mean + half)], "n": n,
            "per_seed": [_r(x) for x in v]}  # fmt: skip


def paired_contrast(a: list[float], b: list[float]) -> dict[str, Any]:
    """Paired t-test of a - b (one pair per seed): mean difference, its 95% t-interval, two-sided p."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    out: dict[str, Any] = {"n": n, "diff_mean": _r(float(d.mean())) if n else None, "ci95": None,
                           "t": None, "p_value": None, "per_seed_diff": [_r(x) for x in d]}  # fmt: skip
    if n < 2:
        return out
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    half = t_quantile(0.975, n - 1) * se
    out["ci95"] = [_r(mean - half), _r(mean + half)]
    if se == 0.0:
        out["p_value"] = 1.0 if mean == 0.0 else 0.0
        return out
    t = mean / se
    out["t"] = _r(t)
    out["p_value"] = _r(min(1.0, 2.0 * t_sf(abs(t), n - 1)), 6)
    return out


def count_cycles(levels: list[float], min_swing: float = CYCLE_MIN_SWING) -> int:
    """Oscillation cycles in one model's sequence of red evasion levels (``red_evasion`` rows, in order).

    One cycle = a rise of at least ``min_swing`` from a running low, followed by a fall of at least
    ``min_swing`` from the peak it reached (red disguises more, then backs off). Moves smaller than
    ``min_swing`` are bandit jitter and never start or end a leg. A rise that has not yet fallen back does not
    count. E.g. 0, 0.3, 0.1 -> 1; 0, 0.1, 0, 0.1 -> 0; 0, 0.4, 0.1, 0.5, 0.2 -> 2.
    """
    eps = 1e-9
    cycles = 0
    low = high = None
    rising = False
    for x in levels:
        if low is None:
            low = high = x
            continue
        if not rising:
            low = min(low, x)
            if x - low >= min_swing - eps:
                rising, high = True, x
        else:
            high = max(high, x)
            if high - x >= min_swing - eps:
                cycles += 1
                rising, low = False, x
    return cycles


# --------------------------------------------------------------------------------------------- run readers


def metric_key(row: dict[str, Any]) -> str:
    m = row["matchup"]
    s = row.get("evasion")
    return f"{m}@{float(s):.1f}" if s else m


def learned_win_rate(row: dict[str, Any]) -> float:
    side = row["matchup"].split("_", 1)[0]  # red_learned_vs_... / blue_learned_vs_...
    return float(row[f"{side}_win_rate"])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    if not path.is_file():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def run_curves(run_dir: Path) -> dict[str, dict[int, float]]:
    """{metric: {after_episode: learned-side win rate}} from a run's summary.jsonl."""
    out: dict[str, dict[int, float]] = {}
    for r in read_jsonl(Path(run_dir) / "summary.jsonl"):
        if r.get("kind") == "eval":
            out.setdefault(metric_key(r), {})[int(r["after_episode"])] = learned_win_rate(r)
    return out


def run_arms_race(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Per model: oscillation cycles of red's evasion and the detector's recall at s=0.7 before the first
    update (the pretrained v0) vs after the last update (from learning.jsonl detector_update rows)."""
    rows = read_jsonl(Path(run_dir) / "learning.jsonl")
    out: dict[str, dict[str, Any]] = {}
    evas = [r for r in rows if r.get("kind") == "red_evasion"]
    upd = [r for r in rows if r.get("kind") == "detector_update"]
    models = sorted({m for r in evas for m in r.get("levels", {})} | {r["model"] for r in upd})
    for m in models:
        seq = [float(r["levels"][m]) for r in sorted(evas, key=lambda r: r["episode"]) if m in r.get("levels", {})]
        mu = sorted((r for r in upd if r["model"] == m), key=lambda r: r["episode"])

        def recall(r: dict | None, when: str) -> float | None:
            if r is None:
                return None
            return (r.get(when) or {}).get("recall_by_level", {}).get(ARMS_LEVEL)

        out[m] = {"cycles": count_cycles(seq), "n_levels": len(seq), "n_updates": len(mu),
                  "recall_first": recall(mu[0] if mu else None, "before"),
                  "recall_last": recall(mu[-1] if mu else None, "after")}  # fmt: skip
    return out


def aggregate(exp_dir: Path) -> dict[str, Any]:
    """Recompute ``aggregate.json`` from the manifest and the finished run directories (and write it)."""
    exp_dir = Path(exp_dir)
    man = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
    episodes = int(man["episodes"])
    done = [r for r in man["runs"] if r.get("status") == "done" and r.get("run_dir")]
    by_cond: dict[str, dict[int, dict[str, dict[int, float]]]] = {}
    for r in done:
        by_cond.setdefault(r["condition"], {})[int(r["seed"])] = run_curves(Path(r["run_dir"]))

    curves: dict[str, Any] = {}
    final: dict[str, Any] = {}
    late: dict[str, Any] = {}
    late_per_seed: dict[str, dict[str, dict[int, float]]] = {}
    final_per_seed: dict[str, dict[str, dict[int, float]]] = {}
    for cond in man["conditions"]:
        seeds = by_cond.get(cond, {})
        metrics = sorted({k for c in seeds.values() for k in c})
        curves[cond], final[cond], late[cond] = {}, {}, {}
        late_per_seed[cond], final_per_seed[cond] = {}, {}
        for k in metrics:
            have = {s: seeds[s][k] for s in sorted(seeds) if k in seeds[s]}
            ckpts = sorted({a for c in have.values() for a in c})
            curves[cond][k] = []
            for a in ckpts:
                vals = [(s, c[a]) for s, c in have.items() if a in c]
                curves[cond][k].append({"after_episode": a, **describe([v for _, v in vals]),
                                        "seeds": [s for s, _ in vals]})  # fmt: skip
            fin = {s: c[max(c)] for s, c in have.items() if c}
            lt = {}
            for s, c in have.items():
                xs = [v for a, v in c.items() if a >= episodes / 2]
                if xs:
                    lt[s] = float(np.mean(xs))
            final_per_seed[cond][k], late_per_seed[cond][k] = fin, lt
            final[cond][k] = {**describe(list(fin.values())), "seeds": list(fin)}
            late[cond][k] = {**describe(list(lt.values())), "seeds": list(lt)}

    contrasts = []
    if "adaptive" in by_cond and "frozen" in by_cond:
        shared = set(late_per_seed["adaptive"]) & set(late_per_seed["frozen"])
        order = [k for k in KEY_CONTRASTS if k in shared] + sorted(shared - set(KEY_CONTRASTS))
        for stat, table in (("late_mean", late_per_seed), ("final", final_per_seed)):
            for k in order:
                a, b = table["adaptive"][k], table["frozen"][k]
                seeds = sorted(set(a) & set(b))
                if not seeds:
                    continue
                c = paired_contrast([a[s] for s in seeds], [b[s] for s in seeds])
                contrasts.append({"metric": f"{k} {stat}", "a": "adaptive", "b": "frozen",
                                  "diff_mean": c["diff_mean"], "ci95": c["ci95"], "paired_by_seed": True,
                                  "p_value": c["p_value"], "t": c["t"], "n": c["n"], "seeds": seeds,
                                  "a_mean": _r(float(np.mean([a[s] for s in seeds]))),
                                  "b_mean": _r(float(np.mean([b[s] for s in seeds]))),
                                  "per_seed_diff": c["per_seed_diff"]})  # fmt: skip

    arms: dict[str, Any] = {}
    per_run = {int(r["seed"]): run_arms_race(Path(r["run_dir"])) for r in done if r["condition"] == "adaptive"}
    for m in sorted({m for d in per_run.values() for m in d}):
        rows = {s: d[m] for s, d in per_run.items() if m in d}

        def mean_of(key: str, rows=rows) -> float | None:
            xs = [r[key] for r in rows.values() if r[key] is not None]
            return _r(float(np.mean(xs))) if xs else None

        arms[m] = {"cycles_mean": mean_of("cycles"), f"recall_at_{ARMS_LEVEL}_first": mean_of("recall_first"),
                   f"recall_at_{ARMS_LEVEL}_last": mean_of("recall_last"), "n_updates_mean": mean_of("n_updates"),
                   "per_seed": {str(s): r for s, r in sorted(rows.items())}}  # fmt: skip

    agg = {
        "name": man["name"], "generated": _now(), "episodes": episodes,
        "n_seeds": len({int(r["seed"]) for r in done}),
        "conditions": man["conditions"],
        "runs_used": {c: sorted(s) for c, s in by_cond.items()},
        "runs_failed": [{"seed": r["seed"], "condition": r["condition"], "status": r.get("status"),
                         "error": r.get("error")} for r in man["runs"] if r.get("status") != "done"],
        "curves": curves, "final": final, "late_mean": late, "contrasts": contrasts, "arms_race": arms,
        "definitions": {
            "win_rate": "learned side's win rate in that matchup; <matchup>@<s> = scripted red at evasion s",
            "ci95": "t-interval across seeds: mean +/- t(0.975, n-1) * sd / sqrt(n), sd with ddof=1",
            "late_mean": "per seed, mean over checkpoints with after_episode >= episodes/2; then across seeds",
            "final": "last checkpoint",
            "contrasts": "a - b paired by seed: paired t-test (two-sided p), t-interval of the mean difference",
            "cycles": f"per model, in red_evasion rows: a rise >= {CYCLE_MIN_SWING} from a running low followed "
                      f"by a fall >= {CYCLE_MIN_SWING} from the peak = 1 cycle (smaller moves are jitter)",
            f"recall_at_{ARMS_LEVEL}_first": "detector recall on held-out malicious rows blended at s=0.7, "
                                             "before the first update (pretrained v0)",
            f"recall_at_{ARMS_LEVEL}_last": "same, after the last detector update",
        },
    }
    write_json(exp_dir / "aggregate.json", agg)
    return agg


# --------------------------------------------------------------------------------------------- runner


def child_cmd(train_cmd: list[str], seed: int, condition: str, episodes: int, name: str,
              extra: list[str], runs_dir: Path | None) -> list[str]:  # fmt: skip
    cmd = [*train_cmd, "--seed", str(seed), "--episodes", str(episodes), *CONDITIONS[condition],
           "--label", f"{name} · {condition} · seed {seed}"]  # fmt: skip
    if runs_dir is not None:
        cmd += ["--runs-dir", str(runs_dir)]
    return cmd + list(extra)


def _read_run_dir(log: Path) -> str | None:
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("RUN_DIR "):
                    return line[len("RUN_DIR "):].strip()
    except OSError:
        pass
    return None


def _tail_error(log: Path, run_dir: str | None) -> str:
    if run_dir:
        try:
            p = json.loads((Path(run_dir) / "progress.json").read_text(encoding="utf-8"))
            if p.get("error"):
                return str(p["error"])[:500]
        except (OSError, ValueError):
            pass
    try:
        lines = [x for x in log.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
        return lines[-1][:500] if lines else "no output"
    except OSError:
        return "no output"


def _stop_child(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if WIN:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except (OSError, ValueError):
        pass


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog=PROG, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", help="experiment name: runs/experiments/<name>/")
    p.add_argument("--seeds", default="1-5", help="e.g. 1-5 or 1,3,7")
    p.add_argument("--conditions", default="adaptive,frozen", help="comma list of adaptive, frozen")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--jobs", type=int, default=None, help="parallel runs (default min(cpu-2, n_runs))")
    p.add_argument("--runs-dir", type=Path, default=None, help="where train writes run dirs (default runs/)")
    p.add_argument("--experiments-dir", type=Path, default=None,
                   help="where experiment folders go (default <runs-dir>/experiments)")
    p.add_argument("--aggregate", type=Path, default=None, metavar="EXP_DIR",
                   help="recompute aggregate.json for an existing experiment folder and exit")
    p.add_argument("--train-cmd", type=str, default=None, help=argparse.SUPPRESS)  # JSON list; tests only
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    args = build_parser().parse_args(argv)

    def say(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    if args.aggregate is not None:
        if not (args.aggregate / "manifest.json").is_file():
            print(f"{PROG}: error: {args.aggregate / 'manifest.json'} not found", file=sys.stderr)
            return 2
        agg = aggregate(args.aggregate)
        say(f"[experiment] aggregate -> {args.aggregate / 'aggregate.json'}")
        print_contrasts(agg, say)
        return 0
    try:
        if not args.name or any(c in args.name for c in '/\\:*?"<>|'):
            raise ValueError("--name is required and must be a plain folder name")
        seeds = parse_seeds(args.seeds)
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
        bad = [c for c in conditions if c not in CONDITIONS]
        if bad or not conditions or len(set(conditions)) != len(conditions):
            raise ValueError(f"--conditions must be distinct values from {list(CONDITIONS)}")
        if args.episodes < 1:
            raise ValueError("--episodes must be >= 1")
        if args.jobs is not None and args.jobs < 1:
            raise ValueError("--jobs must be >= 1")
        train_cmd = json.loads(args.train_cmd) if args.train_cmd else [sys.executable, "-m",
                                                                        "cyberarena.arena.train"]
    except ValueError as e:
        print(f"{PROG}: error: {e}", file=sys.stderr)
        return 2
    if args.runs_dir is None:
        from cyberarena.config import RUNS_DIR

        runs_root = Path(RUNS_DIR)
    else:
        runs_root = Path(args.runs_dir)
    exp_dir = (Path(args.experiments_dir) if args.experiments_dir else runs_root / "experiments") / args.name
    if (exp_dir / "manifest.json").exists():
        print(f"{PROG}: error: {exp_dir} already exists (pick another --name, or use --aggregate)", file=sys.stderr)
        return 2
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs = [{"seed": s, "condition": c, "run_dir": None, "status": "pending", "log": f"logs/{c}_seed{s}.log",
             "started": None, "finished": None, "returncode": None, "error": None}
            for s in seeds for c in conditions]  # fmt: skip
    jobs = args.jobs or max(1, min((os.cpu_count() or 4) - 2, len(runs)))
    t0 = time.time()
    man = {"name": args.name, "created": _now(), "episodes": args.episodes, "seeds": seeds,
           "conditions": conditions, "extra_args": extra, "jobs": jobs, "status": "running",
           "finished": None, "wall_time_s": None, "runs": runs}  # fmt: skip
    lock = threading.Lock()

    def save() -> None:
        with lock:
            write_json(exp_dir / "manifest.json", man)

    save()
    say(f"[experiment] {args.name}: {len(runs)} runs ({len(seeds)} seeds x {conditions}), {jobs} parallel "
        f"-> {exp_dir}")

    handlers: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():

        def on_signal(signum, frame):
            raise Cancelled(signal.Signals(signum).name)

        for nm in ("SIGTERM", "SIGBREAK"):
            sig = getattr(signal, nm, None)
            if sig is not None:
                handlers[sig] = signal.signal(sig, on_signal)

    procs: dict[int, tuple[subprocess.Popen, Any]] = {}  # run index -> (process, log file)
    pending = list(range(len(runs)))
    status = "done"
    try:
        while pending or procs:
            while pending and len(procs) < jobs:
                i = pending.pop(0)
                r = runs[i]
                cmd = child_cmd(train_cmd, r["seed"], r["condition"], args.episodes, args.name, extra,
                                args.runs_dir)  # fmt: skip
                logf = open(exp_dir / r["log"], "w", encoding="utf-8")  # noqa: SIM115 - closed when it ends
                kw: dict[str, Any] = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WIN
                                      else {"start_new_session": True})  # fmt: skip
                env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
                procs[i] = (subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                             env=env, **kw), logf)  # fmt: skip
                r.update(status="running", started=_now())
                save()
            changed = False
            for i, (proc, logf) in list(procs.items()):
                r = runs[i]
                if r["run_dir"] is None:
                    rd = _read_run_dir(exp_dir / r["log"])
                    if rd:
                        r["run_dir"] = rd
                        changed = True
                rc = proc.poll()
                if rc is None:
                    continue
                logf.close()
                del procs[i]
                r["run_dir"] = r["run_dir"] or _read_run_dir(exp_dir / r["log"])
                r.update(finished=_now(), returncode=rc, status="done" if rc == 0 else "error")
                if rc != 0:
                    r["error"] = _tail_error(exp_dir / r["log"], r["run_dir"])
                    say(f"[experiment] {r['condition']} seed {r['seed']}: error ({r['error']})")
                else:
                    say(f"[experiment] {r['condition']} seed {r['seed']}: done ({time.time() - t0:.0f}s)")
                changed = True
            if changed:
                save()
            time.sleep(POLL_S)
    except (KeyboardInterrupt, Cancelled):
        status = "cancelled"
        say("[experiment] cancelling...")
        for proc, _ in procs.values():
            _stop_child(proc)
        deadline = time.time() + 15
        for i, (proc, logf) in procs.items():
            try:
                proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            logf.close()
            r = runs[i]
            r["run_dir"] = r["run_dir"] or _read_run_dir(exp_dir / r["log"])
            r.update(status="error", error="cancelled", finished=_now(), returncode=proc.returncode)
        for i in pending:
            runs[i].update(status="error", error="cancelled")
    finally:
        for sig, h in handlers.items():
            signal.signal(sig, h)
    if status != "cancelled" and any(r["status"] != "done" for r in runs):
        status = "error" if all(r["status"] != "done" for r in runs) else "done"
    man.update(status=status, finished=_now(), wall_time_s=round(time.time() - t0, 1))
    save()
    if status == "cancelled":
        print(f"{PROG}: cancelled", file=sys.stderr, flush=True)
        return 130
    agg = aggregate(exp_dir)
    n_err = sum(r["status"] != "done" for r in runs)
    say(f"[experiment] finished in {man['wall_time_s']}s ({n_err} failed) -> {exp_dir / 'aggregate.json'}")
    print_contrasts(agg, say)
    return 0 if n_err == 0 else 1


def print_contrasts(agg: dict[str, Any], say) -> None:
    for c in agg.get("contrasts", []):
        if c["metric"].endswith("late_mean"):
            ci = c["ci95"] or [None, None]
            say(f"  {c['metric']:<48} adaptive {c['a_mean']:.3f} frozen {c['b_mean']:.3f} diff {c['diff_mean']:+.3f}"
                f" ci95 [{ci[0]}, {ci[1]}] p={c['p_value']}")


def cli() -> int:
    return main()


if __name__ == "__main__":
    sys.exit(cli())
