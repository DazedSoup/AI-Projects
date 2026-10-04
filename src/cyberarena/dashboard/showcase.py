"""Public showcase (docs/contracts.md, "Public showcase (v6)"): the publish selection and the export.

    python -m cyberarena.showcase [--runs ID|LABEL,...] [--experiments NAME,...] [--out showcase] [--dry-run]

* **Selection** (``runs/.showcase.json``, written atomically by the Lab's "Publish" toggles and the Evidence page,
  admin mode only): ``{"runs": [run_id, ...], "experiments": [name, ...], "default_run": run_id}``.
* **Export** copies only what the read-only dashboard needs into a small folder that ``CYBERARENA_RUNS_DIR`` can
  point at:

  - per published run: ``config.json``, ``graph.json``, ``summary.jsonl``, ``learning.jsonl``, ``progress.json``,
    ``episodes_enriched.jsonl``, ``explain_summary.json`` and a **trimmed** ``episodes.jsonl`` holding only the
    narrated games and the probe games Replay's compare view uses (original line order, byte-for-byte lines, so
    the dashboard's byte-offset index works on it unchanged);
  - per experiment: ``manifest.json``, ``aggregate.json``, each referenced run's ``config.json`` and
    ``summary.jsonl``, and the ``diag-crosseval/*.json`` files about those runs;
  - ``MANIFEST.json``: ``{"created", "runs", "experiments", "default_run", "bytes"}``.

  Agent weights, detectors, checkpoints, caches and logs are never copied. The export is built in a sibling
  temp folder and swapped in only when complete and under ``--max-mb``, so a failed export leaves the previous
  showcase intact.

Exit codes: 0 done, 1 over ``--max-mb``, 2 bad arguments (unknown or ambiguous run label, unknown experiment,
nothing selected, an output folder that isn't a showcase).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import streamlit  # noqa: F401  (imported first so its cache logger exists and can be quietened for the CLI)

logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)  # "No runtime found" noise

from cyberarena import config
from cyberarena.dashboard import lab
from cyberarena.dashboard import loaders as L

SELECTION_FILE = ".showcase.json"
MANIFEST_FILE = "MANIFEST.json"
DEFAULT_OUT = "showcase"
DEFAULT_MAX_MB = 50.0
EXPERIMENTS = "experiments"
RUN_FILES = (L.CONFIG_FILE, L.GRAPH_FILE, L.SUMMARY_FILE, L.LEARNING_FILE, L.PROGRESS_FILE, L.ENRICHED_FILE,
             "explain_summary.json")  # fmt: skip
EXPERIMENT_FILES = ("manifest.json", "aggregate.json")
EXPERIMENT_RUN_FILES = (L.CONFIG_FILE, L.SUMMARY_FILE)  # multi-seed curves and run facts only
DIAG_DIRS = ("diag-crosseval",)
MB = 1024 * 1024


# ============================================================================================== selection


def selection_path(runs_dir: Path) -> Path:
    return Path(runs_dir) / SELECTION_FILE


def _names(v) -> list[str]:
    return list(dict.fromkeys(str(x) for x in v if isinstance(x, (str, int)) and str(x).strip())) if isinstance(v, list) else []


def read_selection(runs_dir: Path) -> dict:
    """The saved selection, normalised (missing or invalid file = nothing selected)."""
    raw = lab.read_json(selection_path(runs_dir)) or {}
    d = raw.get("default_run")
    return {"runs": _names(raw.get("runs")), "experiments": _names(raw.get("experiments")),
            "default_run": str(d) if isinstance(d, str) and d else None}  # fmt: skip


def write_selection(runs_dir: Path, sel: dict) -> dict:
    """Write ``runs/.showcase.json`` atomically. Refused in public mode (``lab.PublicModeError``)."""
    lab.refuse_in_public("Changing the showcase selection")
    runs, exps = _names(sel.get("runs")), _names(sel.get("experiments"))
    d = sel.get("default_run")
    if d not in runs:
        d = runs[0] if runs else None
    out = {"runs": runs, "experiments": exps, "default_run": d}
    lab.write_json_atomic(selection_path(runs_dir), out)
    return out


def set_published(runs_dir: Path, kind: str, name: str, on: bool) -> dict:
    """Add or remove one run id (``kind="runs"``) or experiment name (``kind="experiments"``)."""
    if kind not in ("runs", "experiments"):
        raise ValueError(f"kind must be 'runs' or 'experiments', not {kind!r}")
    sel = read_selection(runs_dir)
    items = [x for x in sel[kind] if x != name]
    if on:
        items.append(name)
    sel[kind] = items
    return write_selection(runs_dir, sel)


def set_default_run(runs_dir: Path, run_id: str) -> dict:
    sel = read_selection(runs_dir)
    if run_id not in sel["runs"]:
        sel["runs"].append(run_id)
    sel["default_run"] = run_id
    return write_selection(runs_dir, sel)


# ============================================================================================== name resolution


class ResolveError(ValueError):
    """A run label or experiment name that matches nothing, or more than one thing."""


def _run_dirs(runs_dir: Path) -> list[Path]:
    return [p for p in lab.list_run_dirs(runs_dir) if p.name not in L.NOT_RUNS]


def resolve_run(runs_dir: Path, name: str) -> str:
    """A run id, or the newest *finished* run whose label is ``name`` (case-insensitive). A name that is no
    label falls back to label prefixes; it must then pick out exactly one label."""
    name = (name or "").strip()
    runs_dir = Path(runs_dir)
    if not name:
        raise ResolveError("empty run name")
    if (runs_dir / name).is_dir() and name in {p.name for p in _run_dirs(runs_dir)}:
        return name
    infos = [(p, L.run_info(p)) for p in _run_dirs(runs_dir)]
    key = name.casefold()
    exact = [(p, i) for p, i in infos if (i["label"] or "").strip().casefold() == key]
    if not exact:
        labels = sorted({(i["label"] or "").strip() for _, i in infos
                         if (i["label"] or "").strip().casefold().startswith(key)})  # fmt: skip
        if len(labels) > 1:
            raise ResolveError(f"run name {name!r} is ambiguous: it starts the labels " + ", ".join(map(repr, labels))
                               + ". Use the full label or a run id.")  # fmt: skip
        if not labels:
            raise ResolveError(f"no run with id or label {name!r} under {runs_dir}")
        exact = [(p, i) for p, i in infos if (i["label"] or "").strip() == labels[0]]
    done = sorted((p for p, i in exact if i["status"] == "done" and lab.run_is_done(p)), key=lambda p: p.name,
                  reverse=True)  # fmt: skip
    if not done:
        raise ResolveError(f"no finished run labelled {exact[0][1]['label']!r} (found "
                           + ", ".join(p.name for p, _ in exact) + ", none finished)")  # fmt: skip
    return done[0].name


def resolve_experiment(runs_dir: Path, name: str) -> str:
    """An experiment folder under ``runs/experiments/`` with a ``manifest.json`` (exact name, then case-insensitive)."""
    name = (name or "").strip()
    root = Path(runs_dir) / EXPERIMENTS
    have = sorted(p.name for p in root.iterdir() if (p / "manifest.json").exists()) if root.is_dir() else []
    if name in have:
        return name
    ci = [h for h in have if h.casefold() == name.casefold()]
    if len(ci) == 1:
        return ci[0]
    raise ResolveError(f"no experiment {name!r} under {root}" + (f" (have: {', '.join(have)})" if have else ""))


def split_names(values: list[str] | None) -> list[str]:
    """``["a,b", "c"]`` -> ``["a", "b", "c"]`` (flags may repeat and take comma lists)."""
    out = []
    for v in values or []:
        out += [x.strip() for x in str(v).split(",") if x.strip()]
    return list(dict.fromkeys(out))


# ============================================================================================== plan


@dataclass
class Item:
    dst: str  # path inside the showcase folder, "/"-separated
    src: Path
    bytes: int
    ranges: list[tuple[int, int]] | None = None  # trimmed episodes.jsonl: byte ranges of src to keep, in order
    note: str = ""


@dataclass
class Plan:
    runs: list[str]
    experiments: list[str]
    default_run: str | None
    runs_dir: Path | None = None  # source folder, for scrubbing absolute paths out of the copies
    items: dict[str, Item] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def bytes(self) -> int:
        return sum(i.bytes for i in self.items.values())

    def add(self, dst: str, src: Path, note: str = "") -> None:
        if dst in self.items or not src.is_file():
            return
        self.items[dst] = Item(dst, src, src.stat().st_size, note=note)


def kept_episodes(run_dir: Path, index: L.EpisodeIndex) -> list[int]:
    """The games a public Replay can show: every narrated (enriched) game plus the probe games of compare mode
    (one per matchup and checkpoint, exactly as ``loaders.probe_games`` picks them)."""
    enr = L.load_enriched(run_dir)
    eps, evals = L.load_summary(run_dir)
    catalog = L.episode_catalog(enr, eps, evals, index.meta)
    probes = L.probe_games(catalog)
    keep = set(enr) | {int(e) for e in probes["episode"]} if not probes.empty else set(enr)
    return sorted(e for e in keep if e in index.segments)


def trimmed_ranges(index: L.EpisodeIndex, keep: list[int]) -> list[tuple[int, int]]:
    """Byte ranges of the kept episodes in file order, adjacent ranges merged."""
    spans = sorted(s for e in keep for s in index.segments.get(int(e), []))
    out: list[tuple[int, int]] = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def run_dir_of(raw: str, runs_dir: Path) -> Path:
    """A manifest's ``run_dir`` resolved against ``runs_dir`` first (manifests store absolute Windows paths)."""
    local = Path(runs_dir) / L.path_name(raw)
    return local if local.exists() else Path(raw)


def make_plan(runs_dir: Path, runs: list[str], experiments: list[str], default_run: str | None = None,
              disk_cache: bool = True) -> Plan:  # fmt: skip
    """Everything the export would write, with sizes (the trimmed log's size is exact, from the log index)."""
    runs_dir = Path(runs_dir)
    if default_run not in runs:
        ref = [r for r in runs if (L.run_info(runs_dir / r)["label"] or "").strip().casefold() == "reference"]
        default_run = ref[0] if ref else (runs[0] if runs else None)
    plan = Plan(list(runs), list(experiments), default_run, runs_dir)
    for rid in runs:
        rd = runs_dir / rid
        for f in RUN_FILES:
            plan.add(f"{rid}/{f}", rd / f)
        idx = L.build_episode_index(rd, disk_cache=disk_cache)
        if idx is None:
            plan.warnings.append(f"{rid}: no {L.EPISODES_FILE}; Replay will show narrated games only")
            continue
        keep = kept_episodes(rd, idx)
        ranges = trimmed_ranges(idx, keep)
        if ranges:
            plan.items[f"{rid}/{L.EPISODES_FILE}"] = Item(
                f"{rid}/{L.EPISODES_FILE}", Path(idx.path), sum(b - a for a, b in ranges), ranges,
                note=f"trimmed: {len(keep)} of {len(idx.segments)} games",
            )  # fmt: skip
    referenced = set(runs)
    for name in experiments:
        ed = runs_dir / EXPERIMENTS / name
        for f in EXPERIMENT_FILES:
            plan.add(f"{EXPERIMENTS}/{name}/{f}", ed / f)
        if not (ed / "aggregate.json").exists():
            plan.warnings.append(f"experiment {name}: no aggregate.json yet (Evidence will show it as unfinished)")
        man = lab.read_json(ed / "manifest.json") or {}
        for r in man.get("runs") or []:
            if not r.get("run_dir"):
                continue
            rid = L.path_name(r["run_dir"])
            referenced.add(rid)
            src = run_dir_of(r["run_dir"], runs_dir)
            if not src.is_dir():
                plan.warnings.append(f"experiment {name}: run {rid} is not on disk")
                continue
            for f in EXPERIMENT_RUN_FILES:
                plan.add(f"{rid}/{f}", src / f)
    for d in DIAG_DIRS:
        dd = runs_dir / d
        for f in sorted(dd.glob("*.json")) if dd.is_dir() else []:
            obj = lab.read_json(f) or {}
            ids = {str(c.get("blue_run")) for c in obj.get("cells") or [] if isinstance(c, dict)}
            if obj.get("detectors_from"):
                ids.add(str(obj["detectors_from"]))
            if ids & referenced:
                plan.add(f"{d}/{f.name}", f)
    return plan


# ============================================================================================== export


def _manifest(plan: Plan, total: int) -> dict:
    # only ids, names, a time and a size: nothing path-like
    return {"created": datetime.now().astimezone().isoformat(timespec="seconds"), "runs": plan.runs,
            "experiments": plan.experiments, "default_run": plan.default_run, "bytes": total}  # fmt: skip


# ---------------------------------------------------------------------------------------------- path scrubbing

# an absolute path token inside a string: a Windows drive path (not a URL scheme), a UNC path, or a POSIX path in
# a home directory; it runs to the next space, quote or separator
_ABS_START = r"""(?<![A-Za-z0-9])[A-Za-z]:[\\/]|(?<!\\)\\\\(?=[^\\\s])|/(?:home|Users|root)/"""
_TOKEN_TAIL = r"""[^\s"'|,;<>]*"""


class Scrubber:
    """Rewrites absolute paths so the public export never names the author's machine: under the runs folder ->
    relative to it (``C:/.../runs/20261003-110349-7`` -> ``20261003-110349-7``); under the repo root -> repo-relative;
    anything else -> its basename. Applied recursively to every string (and key) in JSON / JSONL."""

    def __init__(self, runs_dir: Path | None, root: Path | None):
        self.bases: list[str] = []
        for b in (runs_dir, root):  # runs first: it usually lives under the root
            if b is not None:
                n = self._norm(str(Path(b).resolve()))
                if n and n not in self.bases:
                    self.bases.append(n)
        # the bases themselves are also token starts (they may be POSIX paths outside any home directory)
        starts = [re.escape(b).replace("/", r"[\\/]") for b in self.bases] + [_ABS_START]
        self._pat = re.compile("(?:" + "|".join(starts) + ")" + _TOKEN_TAIL, re.IGNORECASE)

    @staticmethod
    def _norm(s: str) -> str:
        return s.replace("\\", "/").rstrip("/")

    def path(self, s: str) -> str:
        n = self._norm(s)
        low = n.casefold()
        for b in self.bases:
            if low == b.casefold():
                return "."
            if low.startswith(b.casefold() + "/"):
                return n[len(b) + 1 :]
        return L.path_name(n) or "."

    def text(self, s: str) -> str:
        return self._pat.sub(lambda m: self.path(m.group(0)), s)

    def obj(self, o):
        if isinstance(o, str):
            return self.text(o)
        if isinstance(o, list):
            return [self.obj(x) for x in o]
        if isinstance(o, dict):
            return {self.text(k) if isinstance(k, str) else k: self.obj(v) for k, v in o.items()}
        return o

    def needs(self, raw: bytes) -> bool:
        # JSON escapes a backslash as two, which still starts a match ("C:\Users" -> "C:\")
        return bool(self._pat.search(raw.decode("utf-8", "replace")))

    def line(self, raw: bytes) -> bytes:
        """One JSONL line, rewritten only when it contains a path (untouched lines stay byte-identical)."""
        if not raw.strip() or not self.needs(raw):
            return raw
        body = raw.rstrip(b"\r\n")
        end = raw[len(body) :]
        try:
            rec = json.loads(body)
        except ValueError:
            return self.text(body.decode("utf-8", "replace")).encode("utf-8") + end
        return json.dumps(self.obj(rec), ensure_ascii=False).encode("utf-8") + end


def _lines_of(item: Item):
    with open(item.src, "rb") as fi:
        if item.ranges is None:
            yield from fi
            return
        for a, b in item.ranges:
            fi.seek(a)
            yield from fi.read(b - a).splitlines(keepends=True)


def _write_item(item: Item, root: Path, scrub: Scrubber | None = None) -> None:
    """Copy one file. JSON metadata is rewritten through ``scrub`` (absolute paths removed); JSONL is streamed line
    by line and only lines naming a path are rewritten, so a trimmed log's index is rebuilt from what is written."""
    dst = root / item.dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if scrub is not None and item.dst.endswith(".json"):
        raw = item.src.read_bytes()
        if scrub.needs(raw):
            try:
                obj = json.loads(raw)
            except ValueError:
                dst.write_bytes(scrub.text(raw.decode("utf-8", "replace")).encode("utf-8"))
            else:
                dst.write_bytes(json.dumps(scrub.obj(obj), indent=1, ensure_ascii=False).encode("utf-8"))
            return
    if item.ranges is None and (scrub is None or not item.dst.endswith(".jsonl")):
        shutil.copyfile(item.src, dst)
        return
    with open(dst, "wb") as fo:
        fo.writelines(scrub.line(ln) if scrub is not None else ln for ln in _lines_of(item))


def check_out_dir(out: Path, runs_dir: Path) -> str | None:
    """``None`` if ``out`` may be (re)written, else why not."""
    out, runs_dir = Path(out).resolve(), Path(runs_dir).resolve()
    if out == runs_dir or out in runs_dir.parents:
        return f"--out {out} would overwrite the runs folder"
    if (runs_dir / out.name) == out and out.name not in L.NOT_RUNS:
        return f"--out {out} is inside the runs folder; call it '{L.NOT_RUNS[0]}' or put it elsewhere"
    if out.exists() and (not out.is_dir() or (any(out.iterdir()) and not (out / MANIFEST_FILE).exists())):
        return f"--out {out} exists and is not a showcase folder (no {MANIFEST_FILE}); refusing to replace it"
    return None


def export(plan: Plan, out: Path, max_mb: float = DEFAULT_MAX_MB) -> tuple[int, dict | None]:
    """Write the showcase. ``(0, manifest)`` on success; ``(1, None)`` when it would exceed ``max_mb`` (nothing
    is replaced)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".{out.name}.build-{os.getpid()}"
    scrub = Scrubber(plan.runs_dir, Path(config.ROOT))
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        for item in plan.items.values():
            _write_item(item, tmp, scrub)
        data = sum(p.stat().st_size for p in tmp.rglob("*") if p.is_file())
        man = _manifest(plan, data)
        for _ in range(3):  # "bytes" counts MANIFEST.json itself: settle the number's own width
            body = json.dumps(man, indent=1).encode("utf-8")
            man["bytes"] = data + len(body)
        body = json.dumps(man, indent=1).encode("utf-8")
        if man["bytes"] > max_mb * MB:
            shutil.rmtree(tmp, ignore_errors=True)
            return 1, man
        (tmp / MANIFEST_FILE).write_bytes(body)
        if out.exists():
            old = out.parent / f".{out.name}.old-{os.getpid()}"
            try:
                os.replace(out, old)
            except OSError:  # Windows: something holds a file open; delete in place instead
                shutil.rmtree(out)
                old = None
            os.replace(tmp, out)
            if old is not None:
                shutil.rmtree(old, ignore_errors=True)
        else:
            os.replace(tmp, out)
        return 0, man
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


# ============================================================================================== CLI


def _fmt_mb(n: int) -> str:
    return f"{n / MB:.2f} MB"


def print_plan(plan: Plan, file=None) -> None:
    file = file or sys.stdout
    groups: dict[str, list[Item]] = {}
    for it in plan.items.values():
        groups.setdefault(it.dst.split("/")[0] if not it.dst.startswith(EXPERIMENTS + "/")
                          else "/".join(it.dst.split("/")[:2]), []).append(it)  # fmt: skip
    for g in sorted(groups):
        items = groups[g]
        print(f"  {g}/  {_fmt_mb(sum(i.bytes for i in items))}  ({len(items)} files)", file=file)
        for it in sorted(items, key=lambda i: i.dst):
            if it.note:
                print(f"    {it.dst.split('/')[-1]}: {_fmt_mb(it.bytes)} ({it.note})", file=file)
    for w in plan.warnings:
        print(f"  note: {w}", file=file)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cyberarena.showcase", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--runs", action="append", help="run ids or labels, comma-separated (a label picks its newest "
                    "finished run)")  # fmt: skip
    ap.add_argument("--experiments", action="append", help="experiment names under runs/experiments, comma-separated")
    ap.add_argument("--default-run", help="run (id or label) the public dashboard opens on; default: the run "
                    "labelled 'reference', else the first")  # fmt: skip
    ap.add_argument("--out", type=Path, default=None, help=f"output folder (default: <repo>/{DEFAULT_OUT})")
    ap.add_argument("--runs-dir", type=Path, default=None, help="source runs folder (default: config.RUNS_DIR)")
    ap.add_argument("--max-mb", type=float, default=DEFAULT_MAX_MB, help="fail if the export is larger (default 50)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be written, write nothing")
    ap.add_argument("--save-selection", action="store_true", help="also save this selection to runs/.showcase.json")
    args = ap.parse_args(argv)
    runs_dir = Path(args.runs_dir or config.RUNS_DIR)
    out = Path(args.out) if args.out else Path(config.ROOT) / DEFAULT_OUT
    if getattr(config, "PUBLIC", False):
        print("showcase: refusing to export with CYBERARENA_PUBLIC set (the public dashboard is read-only)",
              file=sys.stderr)  # fmt: skip
        return 2
    try:
        if args.runs is None and args.experiments is None:
            sel = read_selection(runs_dir)
            run_names, exp_names, default = sel["runs"], sel["experiments"], sel["default_run"]
            src = f"selection in {selection_path(runs_dir)}"
        else:
            run_names, exp_names, default = split_names(args.runs), split_names(args.experiments), None
            src = "command line"
        if args.default_run:
            default = args.default_run
        runs = list(dict.fromkeys(resolve_run(runs_dir, n) for n in run_names))
        exps = list(dict.fromkeys(resolve_experiment(runs_dir, n) for n in exp_names))
        if default:
            default = resolve_run(runs_dir, default)
            if default not in runs:
                runs.insert(0, default)
    except ResolveError as e:
        print(f"showcase: {e}", file=sys.stderr)
        return 2
    if not runs and not exps:
        print(f"showcase: nothing selected ({src}). Pass --runs/--experiments, or use the Publish toggles in the "
              "Simulation Lab.", file=sys.stderr)  # fmt: skip
        return 2
    problem = None if args.dry_run else check_out_dir(out, runs_dir)
    if problem:
        print(f"showcase: {problem}", file=sys.stderr)
        return 2
    plan = make_plan(runs_dir, runs, exps, default)
    print(f"showcase: {len(runs)} run(s), {len(exps)} experiment(s) from the {src}")
    for r in runs:
        i = L.run_info(runs_dir / r)
        print(f"  run {r}" + (f" ({i['label']})" if i["label"] else "") + (" [default]" if r == plan.default_run else ""))
    for e in exps:
        print(f"  experiment {e}")
    print_plan(plan)
    if args.save_selection:
        write_selection(runs_dir, {"runs": runs, "experiments": exps, "default_run": plan.default_run})
        print(f"saved the selection to {selection_path(runs_dir)}")
    if args.dry_run:
        over = plan.bytes > args.max_mb * MB
        print(f"dry run: would write {len(plan.items) + 1} files, about {_fmt_mb(plan.bytes)}"
              + (f" — over --max-mb {args.max_mb:g}" if over else ""))  # fmt: skip
        return 1 if over else 0
    rc, man = export(plan, out, args.max_mb)
    if rc:
        print(f"showcase: the export would be {_fmt_mb((man or {}).get('bytes', 0))}, over --max-mb {args.max_mb:g}; "
              f"nothing was written. Publish fewer runs or raise --max-mb.", file=sys.stderr)  # fmt: skip
        return rc
    print(f"wrote {out} ({len(plan.items) + 1} files, {_fmt_mb(man['bytes'])})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
