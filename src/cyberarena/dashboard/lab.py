"""Simulation Lab logic (see docs/contracts.md, "Simulation Lab"). Pure functions plus process helpers.

The Lab never imports arena, ml or explain code. It talks to them only through subprocesses:

* ``python -m cyberarena.arena.train --describe-params`` -> the parameter spec that drives the form;
* ``python -m cyberarena.dashboard.lab_runner --job <file>`` -> a detached runner that executes
  ``train`` then ``explain.enrich --offline`` and keeps ``lab_status.json`` up to date.

Job/status files
----------------
``runs/.lab/<token>.json`` is written by the dashboard *before* the runner starts (stage ``starting``), so a
launch is visible before ``train`` has printed its run directory. Once ``RUN_DIR`` is known, the runner mirrors
the same record into ``runs/<run_id>/lab_status.json``. ``.lab`` holds no summary/episode files, so the
replay run picker ignores it.

Nothing here calls Streamlit; the Lab page (``lab_page.py``) is a thin view over these functions.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cyberarena.dashboard import loaders as L

_REAL_SUBPROCESS = subprocess  # creation-flag constants come from here even when tests swap `subprocess`

TRAIN_MODULE = "cyberarena.arena.train"
ENRICH_MODULE = "cyberarena.explain.enrich"
RUNNER_MODULE = "cyberarena.dashboard.lab_runner"

LAB_DIR_NAME = ".lab"
STATUS_FILE = "lab_status.json"
PROGRESS_FILE = "progress.json"
CONFIG_FILE = "config.json"

ACTIVE_STAGES = ("starting", "training", "enriching")
FINAL_STAGES = ("done", "error")
STARTING_GRACE_S = 45.0  # a "starting" job with no live runner pid after this long is stale

PARAM_TYPES = ("int", "float", "bool", "choice")
MAX_SLIDER_STEPS = 5000
STALE_PROGRESS_S = 60.0  # progress.json is rewritten every <= 2 s while training; older than this = dead
STOP_GRACE_S = 25.0  # after a Stop request, offer "Force stop" once this has passed
PRESETS = ("Default", "Red-favoured", "Blue-favoured", "Quick test")
PRESET_HELP = {
    "Default": "Every parameter back to the arena's defaults.",
    "Red-favoured": "From defaults: red's attack success probabilities up, blue's detection weaker "
    "(higher detection threshold, quieter red actions, fewer implant leaks).",
    "Blue-favoured": "From defaults: the reverse — red's success down, detection stronger.",
    "Quick test": "Keeps your other values; sets 300 episodes with an eval every 100, for a run that "
    "finishes in a few minutes.",
}

_RED_SUCCESS = re.compile(r"^p_(phish|exploit|escalate|lateral|exfiltrate)(_|$)")
_RUN_ID_TS = re.compile(r"^(\d{8}-\d{6})")


# ============================================================================================== small file I/O


def local_now() -> datetime:
    """Naive local time, the convention of every timestamp in run files (``progress.json`` etc.)."""
    return datetime.now().astimezone().replace(tzinfo=None)


def now_iso() -> str:
    return local_now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict | None:
    """A JSON object from ``path``, or ``None`` if it is missing, partial, invalid or not an object.

    Files here are rewritten by other processes while we poll, so every failure mode is "not available yet".
    """
    for attempt in range(3):
        try:
            text = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except PermissionError:  # Windows: the writer is mid-rename
            time.sleep(0.05 * (attempt + 1))
            continue
        except OSError:
            return None
        try:
            obj = json.loads(text)
        except ValueError:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def write_json_atomic(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # a reader holds the target open on Windows
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def read_progress(run_dir: Path) -> dict | None:
    """``progress.json`` normalised: numbers coerced, ``last_eval`` always a dict or ``None``."""
    p = read_json(Path(run_dir) / PROGRESS_FILE)
    if p is None:
        return None
    out = dict(p)
    for k in ("episode", "episodes"):
        try:
            out[k] = int(p[k]) if p.get(k) is not None else None
        except (TypeError, ValueError):
            out[k] = None
    le = p.get("last_eval")
    out["last_eval"] = le if isinstance(le, dict) else None
    out.setdefault("status", None)
    out.setdefault("error", None)
    out["cancelled"] = progress_cancelled(out)
    return out


def progress_cancelled(p: dict | None) -> bool:
    if not p:
        return False
    if "cancel" in str(p.get("status", "")).lower() or "cancel" in str(p.get("phase", "")).lower():
        return True
    return p.get("status") == "error" and "cancel" in str(p.get("error") or "").lower()


def progress_stale(p: dict | None, now: datetime | None = None) -> bool:
    """``running`` but its writer is gone: a dead ``pid`` if one is recorded, else no update for a minute."""
    if not p or p.get("status") != "running":
        return False
    if p.get("pid") is not None:
        return not pid_alive(p.get("pid"))
    upd = parse_time(p.get("updated"))
    return upd is not None and ((now or local_now()) - upd).total_seconds() > STALE_PROGRESS_S


def read_lab_status(run_dir: Path) -> dict | None:
    return read_json(Path(run_dir) / STATUS_FILE)


def progress_fraction(progress: dict | None) -> float:
    if not progress or not progress.get("episodes"):
        return 0.0
    return max(0.0, min(1.0, (progress.get("episode") or 0) / progress["episodes"]))


def parse_time(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt.astimezone().replace(tzinfo=None) if dt.tzinfo else dt


def run_started(run_id: str) -> datetime | None:
    m = _RUN_ID_TS.match(run_id or "")
    if not m:
        return None
    d, t = m.group(1).split("-")
    try:
        return datetime.fromisoformat(f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}")
    except ValueError:
        return None


def format_elapsed(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or math.isnan(seconds):
        return "-"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


# ============================================================================================== parameter spec


class SpecError(RuntimeError):
    """``--describe-params`` failed or printed something that isn't a parameter spec."""


def describe_params_cmd(python: str | None = None) -> list[str]:
    return [python or sys.executable, "-m", TRAIN_MODULE, "--describe-params"]


def parse_spec(text: str) -> dict:
    """Parse and validate ``--describe-params`` stdout. Tolerates log lines before the JSON object."""
    dec = json.JSONDecoder()
    spec, err, pos = None, "no JSON object in the output", text.find("{")
    while pos >= 0:  # skip log lines that happen to contain "{"
        try:
            obj, _ = dec.raw_decode(text[pos:])
        except ValueError as e:
            err = f"output is not valid JSON ({e})"
        else:
            if isinstance(obj, dict) and "groups" in obj:
                spec = obj
                break
            spec = spec or obj
        pos = text.find("{", pos + 1)
    if spec is None:
        raise SpecError(err)
    if not isinstance(spec, dict) or not isinstance(spec.get("groups"), list):
        raise SpecError("JSON has no 'groups' list")
    groups = []
    for g in spec["groups"]:
        if not isinstance(g, dict) or not isinstance(g.get("params"), list):
            continue
        params = [p for p in g["params"] if isinstance(p, dict) and p.get("key") and p.get("type")]
        groups.append({**g, "id": str(g.get("id", "")), "label": g.get("label") or g.get("id", ""),
                       "params": params})  # fmt: skip
    if not any(g["params"] for g in groups):
        raise SpecError("the spec lists no parameters")
    return {**spec, "groups": groups}


def load_param_spec(python: str | None = None, cwd: Path | None = None, timeout: float = 120) -> dict:
    """Run ``train --describe-params`` once and return the parsed spec. Raises :class:`SpecError`."""
    cmd = describe_params_cmd(python)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
            cwd=str(cwd) if cwd else None, creationflags=_no_window_flag(), check=False,
        )  # fmt: skip
    except FileNotFoundError as e:
        raise SpecError(f"could not start {cmd[0]!r}: {e}") from e
    except _REAL_SUBPROCESS.TimeoutExpired as e:
        raise SpecError(f"timed out after {timeout:.0f}s") from e
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise SpecError(f"exit code {proc.returncode}: " + (" | ".join(tail) or "no output"))
    return parse_spec(proc.stdout or "")


def iter_params(spec: dict | None):
    """Yield ``(group, param)`` over the whole spec."""
    for g in (spec or {}).get("groups", []):
        for p in g["params"]:
            yield g, p


def param_index(spec: dict | None) -> dict[str, dict]:
    return {p["key"]: p for _, p in iter_params(spec)}


def defaults(spec: dict | None) -> dict[str, Any]:
    return {p["key"]: coerce(p, p.get("default")) for _, p in iter_params(spec)}


def _num_type(p: dict):
    return int if p.get("type") == "int" else float


def coerce(p: dict, v: Any) -> Any:
    """Coerce a form/preset value to the param's type. ``None`` stays ``None`` (nullable = random)."""
    if v is None:
        return None
    t = p.get("type")
    if t == "bool":
        return bool(v)
    if t == "choice":
        return v
    if t in ("int", "float"):
        try:
            return _num_type(p)(v) if t == "float" else round(float(v))
        except (TypeError, ValueError):
            return None
    return v


def widget_for(p: dict) -> dict:
    """Pure mapping from a param spec to the widget the Lab renders (tested without Streamlit).

    Returns ``{"widget": slider|number|toggle|selectbox|unsupported, "nullable": bool, ...}``; sliders carry
    ``min``/``max``/``step``/``fallback`` (the value used when a nullable param leaves "random").
    """
    t = p.get("type")
    nullable = bool(p.get("nullable"))
    base = {"key": p["key"], "label": p.get("label") or p["key"], "help": p.get("help") or None,
            "nullable": nullable, "advanced": bool(p.get("advanced"))}  # fmt: skip
    if t in ("int", "float"):
        cast = _num_type(p)
        step = p.get("step")
        step = cast(step) if step not in (None, 0) else (1 if t == "int" else 0.01)
        lo, hi = p.get("min"), p.get("max")
        if lo is None or hi is None:
            fb = p.get("default")
            fb = cast(fb) if fb is not None else cast(lo if lo is not None else 0)
            return {**base, "widget": "number", "min": None if lo is None else cast(lo),
                    "max": None if hi is None else cast(hi), "step": step, "fallback": fb}  # fmt: skip
        lo, hi = cast(lo), cast(hi)
        if (hi - lo) / step > MAX_SLIDER_STEPS:  # e.g. a seed in 0..999999: a slider would be unusable
            d = p.get("default")
            fb = snap(p, d) if d is not None else lo
            return {**base, "widget": "number", "min": lo, "max": hi, "step": step, "fallback": fb}
        d = p.get("default")
        fb = snap(p, d) if d is not None else snap(p, lo + (hi - lo) / 2)
        return {**base, "widget": "slider", "min": lo, "max": hi, "step": step, "fallback": fb}
    if t == "bool":
        return {**base, "widget": "toggle"}
    if t == "choice":
        choices = list(p.get("choices") or [])
        if not choices:
            return {**base, "widget": "unsupported", "reason": "choice param without choices"}
        return {**base, "widget": "selectbox", "options": choices}
    return {**base, "widget": "unsupported", "reason": f"unknown type {t!r}"}


def snap(p: dict, v: float) -> int | float:
    """Clamp to [min, max] and round to the step grid (anchored at min)."""
    cast = _num_type(p)
    lo, hi, step = p.get("min"), p.get("max"), p.get("step")
    v = float(v)
    if lo is not None:
        v = max(float(lo), v)
    if hi is not None:
        v = min(float(hi), v)
    if step:
        base = float(lo) if lo is not None else 0.0
        v = base + round((v - base) / float(step)) * float(step)
        if hi is not None:
            v = min(float(hi), v)
        decimals = max(0, -math.floor(math.log10(float(step)))) + 2 if float(step) < 1 else 0
        v = round(v, decimals)
    return round(v) if cast is int else float(v)


def same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def changed_params(spec: dict | None, values: dict[str, Any]) -> dict[str, Any]:
    """Params whose value differs from the spec default, in spec order."""
    out = {}
    for _, p in iter_params(spec):
        k = p["key"]
        if k in values and not same(coerce(p, values[k]), coerce(p, p.get("default"))):
            out[k] = coerce(p, values[k])
    return out


def group_changed(group: dict, values: dict[str, Any]) -> list[str]:
    return [p["key"] for p in group["params"]
            if p["key"] in values and not same(coerce(p, values[p["key"]]), coerce(p, p.get("default")))]  # fmt: skip


# ============================================================================================== presets


def preset_values(name: str, spec: dict | None, current: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full value dict for a preset. Only keys present in the spec are ever set."""
    idx = param_index(spec)
    base = defaults(spec)
    if name == "Quick test":
        vals = {**base, **(current or {})}
        for k, v in (("episodes", 300), ("eval_every", 100)):
            if k in idx:
                vals[k] = snap(idx[k], v) if idx[k].get("type") in ("int", "float") else v
        return vals
    if name in ("Default", None):
        return base
    if name not in ("Red-favoured", "Blue-favoured"):
        raise ValueError(f"unknown preset {name!r}")
    sign = 1.0 if name == "Red-favoured" else -1.0
    vals = dict(base)
    for k, p in idx.items():
        if p.get("type") != "float" or p.get("target") != "env" or p.get("default") is None:
            continue
        d = float(p["default"])
        if _RED_SUCCESS.match(k) or k == "detect_threshold":  # red success probabilities
            vals[k] = snap(p, d + sign * 0.15)
        elif k.startswith("noise."):  # how loud red's actions are to the detectors
            vals[k] = snap(p, d - sign * 0.2)
        elif k.startswith("p_implant_leak."):  # implants leaking into detector features
            vals[k] = snap(p, d * (0.5 if sign > 0 else 1.5))
    return vals


# ============================================================================================== command line


def build_train_args(
    spec: dict | None,
    values: dict[str, Any],
    *,
    init_from: str | Path | None = None,
    label: str | None = None,
    runs_dir: str | Path | None = None,
) -> list[str]:
    """Arguments after ``python -m cyberarena.arena.train`` for the given form values.

    * ``target: "cli"`` params that differ from their default become ``flag value``; a bool becomes the bare
      flag when true; a nullable param left on "random" (``None``) is omitted.
    * ``target: "env"`` params that differ go into one ``--env-json`` object with their (possibly dotted) key.
    * ``init_side`` (memory group) is only sent with ``--init-from``, and then always explicitly.
    """
    args: list[str] = []
    env: dict[str, Any] = {}
    for _, p in iter_params(spec):
        k = p["key"]
        if k not in values:
            continue
        v = coerce(p, values[k])
        is_side = k == "init_side" or p.get("flag") == "--init-side"
        if is_side:
            if init_from is None or v is None:
                continue
        elif same(v, coerce(p, p.get("default"))):
            continue
        if p.get("target") == "env":
            if v is not None:
                env[k] = v
            continue
        flag = p.get("flag") or "--" + k.replace("_", "-")
        if p.get("type") == "bool":
            if v:
                args.append(flag)
        elif v is not None:
            args += [flag, _fmt(v)]
    if env:
        args += ["--env-json", json.dumps(env, sort_keys=True, separators=(",", ":"))]
    if init_from is not None:
        args += ["--init-from", str(Path(init_from))]
    if label and label.strip():
        args += ["--label", label.strip()]
    if runs_dir is not None:
        args += ["--runs-dir", str(Path(runs_dir))]
    if "--online" in args:  # pragma: no cover - guard: the Lab never uses the paid API
        raise ValueError("--online is never passed by the Lab")
    return args


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return repr(v) if v != int(v) else f"{v:.1f}"
    return str(v)


def train_cmd(train_args: list[str], python: str | None = None) -> list[str]:
    return [python or sys.executable, "-m", TRAIN_MODULE, *train_args]


def enrich_cmd(run_dir: str | Path, python: str | None = None) -> list[str]:
    """Offline enrichment with the default episode selection. Never ``--online``."""
    return [python or sys.executable, "-m", ENRICH_MODULE, "--run", str(run_dir), "--offline"]


def child_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for train/enrich: unbuffered output, and no API key so nothing can reach a paid API."""
    env = dict(os.environ if base is None else base)
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("ANTHROPIC_API_KEY", None)
    return env


def format_cmd(cmd: list[str]) -> str:
    return _REAL_SUBPROCESS.list2cmdline(cmd) if os.name == "nt" else " ".join(map(_shquote, cmd))


def _shquote(s: str) -> str:
    import shlex

    return shlex.quote(s)


# ============================================================================================== processes


def _no_window_flag() -> int:
    return getattr(_REAL_SUBPROCESS, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def detached_kwargs() -> dict:
    """Popen kwargs so the runner outlives Streamlit reruns, the tab, and the Streamlit process itself."""
    common = {"stdin": _REAL_SUBPROCESS.DEVNULL, "stdout": _REAL_SUBPROCESS.DEVNULL,
              "stderr": _REAL_SUBPROCESS.DEVNULL, "close_fds": True}  # fmt: skip
    if os.name == "nt":
        # CREATE_NO_WINDOW instead of DETACHED_PROCESS: the runner still gets no window and is not attached to
        # Streamlit's console (Ctrl+C there doesn't reach it), but it owns a hidden console that its train
        # child inherits, which is what lets it deliver CTRL_BREAK_EVENT for a graceful cancel.
        flags = _REAL_SUBPROCESS.CREATE_NEW_PROCESS_GROUP | _no_window_flag()
        return {**common, "creationflags": flags}
    return {**common, "start_new_session": True}


def child_popen_flags() -> int:
    """train/enrich get their own process group so CTRL_BREAK_EVENT can target them alone."""
    if os.name == "nt":
        return _REAL_SUBPROCESS.CREATE_NEW_PROCESS_GROUP
    return 0


def pid_alive(pid: Any) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ctypes.get_last_error() == 5  # access denied: exists but not ours
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


KILLABLE_MODULES = (RUNNER_MODULE, "cyberarena.arena.train")


def process_cmdline(pid: int) -> str | None:
    """Command line of a live process, or None if it can't be read."""
    try:
        if os.name == "nt":
            out = _REAL_SUBPROCESS.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"],
                capture_output=True, text=True, timeout=15, check=False, creationflags=_no_window_flag(),
            )  # fmt: skip
            return out.stdout.strip() or None
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace") or None
    except (OSError, ValueError, _REAL_SUBPROCESS.SubprocessError):
        return None


def kill_tree(pid: Any) -> None:
    """Force-kill a Lab runner (or its train child) and its children. Refuses any PID whose command line
    isn't one of ours, so a recycled PID never takes down an unrelated process tree."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return
    if not pid_alive(pid):
        return
    cmd = process_cmdline(pid)
    norm = (cmd or "").replace("\\", "/").replace("/", ".")
    if not any(m in norm for m in KILLABLE_MODULES):
        return
    if os.name == "nt":
        _REAL_SUBPROCESS.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False,
                             creationflags=_no_window_flag())  # fmt: skip
        return
    import signal

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


# ============================================================================================== jobs


def lab_dir(runs_dir: Path) -> Path:
    return Path(runs_dir) / LAB_DIR_NAME


def job_path(runs_dir: Path, token: str) -> Path:
    return lab_dir(runs_dir) / f"{token}.json"


def save_job(job: dict) -> None:
    """Write the job record to its ``.lab`` file and, once the run directory is known, ``lab_status.json``."""
    job["updated"] = now_iso()
    write_json_atomic(Path(job["job_file"]), job)
    if job.get("run_dir"):
        write_json_atomic(Path(job["run_dir"]) / STATUS_FILE, job)


def effective_stage(job: dict | None) -> str | None:
    """Stage, with crashed runners (active stage but no live pid) reported as ``"error"``."""
    if not job:
        return None
    stage = job.get("stage")
    if stage not in ACTIVE_STAGES:
        return stage
    if pid_alive(job.get("pid")):
        return stage
    if stage == "starting" and job.get("pid") is None:
        created = parse_time(job.get("created"))
        if created and (local_now() - created).total_seconds() < STARTING_GRACE_S:
            return stage
    return "error"


def stale_message(job: dict) -> str:
    return "The Lab runner exited unexpectedly (it was killed, or the machine restarted)."


def list_jobs(runs_dir: Path) -> list[dict]:
    """Every job known from status files: ``.lab/*.json`` merged with ``runs/*/lab_status.json``, newest first."""
    runs_dir = Path(runs_dir)
    jobs: dict[str, dict] = {}
    d = lab_dir(runs_dir)
    if d.is_dir():
        for f in d.glob("*.json"):
            j = read_json(f)
            if j and j.get("token"):
                jobs[j["token"]] = j
    if runs_dir.is_dir():
        for f in runs_dir.glob(f"*/{STATUS_FILE}"):
            j = read_json(f)
            if not j:
                continue
            tok = j.get("token") or f.parent.name
            old = jobs.get(tok)
            # the newer of the two copies wins (they should be identical)
            if old is None or str(j.get("updated", "")) >= str(old.get("updated", "")):
                jobs[tok] = j
    return sorted(jobs.values(), key=lambda j: str(j.get("created", "")), reverse=True)


def active_job(runs_dir: Path) -> dict | None:
    """The running Lab job, if any. Decided from status files plus pid liveness, not session state."""
    for j in list_jobs(runs_dir):
        if effective_stage(j) in ACTIVE_STAGES:
            return j
    return None


def find_job(runs_dir: Path, token: str) -> dict | None:
    j = read_json(job_path(runs_dir, token))
    if j and j.get("run_dir"):
        st_ = read_lab_status(Path(j["run_dir"]))
        if st_ and str(st_.get("updated", "")) >= str(j.get("updated", "")):
            return st_
    return j


class LabBusyError(RuntimeError):
    pass


def launch_job(
    runs_dir: Path,
    train_args: list[str],
    *,
    label: str = "",
    changed: dict | None = None,
    init_from: dict | None = None,
    python: str | None = None,
    cwd: Path | None = None,
) -> dict:
    """Write the job file and start the detached runner. Refuses while another Lab job is active."""
    busy = active_job(runs_dir)
    if busy:
        raise LabBusyError(f"a Lab run is already active ({busy.get('run_id') or busy.get('token')})")
    if "--online" in train_args:
        raise ValueError("--online is never passed by the Lab")
    py = python or sys.executable
    token = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    path = job_path(runs_dir, token)
    job = {
        "token": token,
        "job_file": str(path),
        "created": now_iso(),
        "started": now_iso(),
        "stage": "starting",
        "message": "Starting the runner…",
        "pid": None,
        "train_pid": None,
        "run_dir": None,
        "run_id": None,
        "label": label.strip(),
        "changed": changed or {},
        "init_from": init_from,
        "python": py,
        "cwd": str(cwd) if cwd else None,
        "train_cmd": train_cmd(train_args, py),
        "enrich": True,
    }
    save_job(job)
    runner = [py, "-m", RUNNER_MODULE, "--job", str(path)]
    try:
        proc = subprocess.Popen(runner, cwd=str(cwd) if cwd else None, env=child_env(), **detached_kwargs())
    except OSError as e:
        job.update(stage="error", message=f"Could not start the runner: {e}")
        save_job(job)
        return job
    # the runner rewrites the file with its own pid as its first act; record it here too in case it is slow
    cur = read_json(path) or job
    if cur.get("pid") is None:
        cur["pid"] = proc.pid
        save_job(cur)
    return cur


def stop_path(job: dict) -> Path:
    return Path(job["job_file"]).with_suffix(".stop")


def stop_requested_at(job: dict | None) -> datetime | None:
    if not job or not job.get("job_file"):
        return None
    p = stop_path(job)
    if not p.exists():
        return None
    return parse_time(p.read_text(encoding="utf-8").strip()) or local_now()


def request_stop(job: dict) -> None:
    """Ask the runner to cancel: it sends CTRL_BREAK_EVENT (POSIX: SIGINT) to its child, so train can write
    ``progress.json`` status ``error``/``cancelled`` before exiting (code 130)."""
    p = stop_path(job)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(now_iso(), encoding="utf-8")


def stop_job(job: dict, force: bool = False) -> dict:
    """Stop a Lab job. Gracefully through the runner when it is alive; otherwise (or with ``force``) kill the
    runner and child process trees and mark the job stopped ourselves."""
    if not force and pid_alive(job.get("pid")):
        request_stop(job)
        return job
    for pid in (job.get("pid"), job.get("train_pid"), job.get("child_pid")):
        kill_tree(pid)
    cur = read_json(Path(job["job_file"])) or dict(job)
    if cur.get("run_dir"):
        cur = read_lab_status(Path(cur["run_dir"])) or cur
    cur.update(stage="error", stopped=True, finished=now_iso(),
               message="Stopped by user (force-killed)." if force else "Stopped by user.")  # fmt: skip
    save_job(cur)
    return cur


# ============================================================================================== history


def is_run_dir(p: Path) -> bool:
    return (
        p.is_dir()
        and not p.name.startswith(".")
        and any(
            (p / f).exists()
            for f in (CONFIG_FILE, PROGRESS_FILE, STATUS_FILE, L.SUMMARY_FILE, L.EPISODES_FILE)
        )
    )


def list_run_dirs(runs_dir: Path) -> list[Path]:
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    return sorted((p for p in runs_dir.iterdir() if is_run_dir(p)), key=lambda p: p.name, reverse=True)


def _dotted(d: Any, key: str) -> Any:
    for part in key.split("."):
        if not isinstance(d, dict) or part not in d:
            return _MISSING
        d = d[part]
    return d


_MISSING = object()


def changed_from_config(spec: dict | None, config: dict | None) -> dict[str, Any]:
    """Best-effort diff for runs not launched from the Lab: env params from ``config.env`` (dotted lookup),
    CLI params from same-named top-level config keys. Nullable params defaulting to ``None`` are skipped
    (the config records the drawn value, not "random")."""
    if not spec or not config:
        return {}
    out = {}
    params = config.get("params") if isinstance(config.get("params"), dict) else None
    for _, p in iter_params(spec):
        k, d = p["key"], coerce(p, p.get("default"))
        if p.get("target") != "env" and params is not None and k in params:
            v = coerce(p, params[k])  # the CLI values exactly as train recorded them (None = random)
            if k == "init_side" and not config.get("init_from"):
                continue
            if not same(v, d):
                out[k] = v
            continue
        if d is None:
            continue
        v = _dotted(config.get("env") or {}, k) if p.get("target") == "env" else config.get(k, _MISSING)
        if v is _MISSING or isinstance(v, (dict, list)):
            continue
        v = coerce(p, v)
        if not same(v, d):
            out[k] = v
    return out


def format_changed(changed: dict[str, Any]) -> str:
    def f(v):
        if v is None:
            return "random"
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)

    return ", ".join(f"{k}={f(v)}" for k, v in changed.items())


def final_eval(evals: pd.DataFrame) -> dict[str, float | None]:
    """Latest learned-red and learned-blue eval win rates (each vs its baseline)."""
    out: dict[str, float | None] = {"red": None, "blue": None, "after_episode": None}
    curve = L.eval_curve(evals)
    if curve.empty:
        return out
    for side in ("red", "blue"):
        sub = curve[curve["side"] == side]
        if not sub.empty:
            last = sub.sort_values("after_episode").iloc[-1]
            out[side] = float(last["win_rate"])
            out["after_episode"] = max(int(last["after_episode"]), out["after_episode"] or 0)
    return out


def run_status(run_dir: Path, job: dict | None, progress: dict | None, has_evals: bool) -> str:
    if job:
        stage = effective_stage(job)
        if job.get("stopped") or job.get("train_exit") == 130:
            return "stopped"
        if stage in ACTIVE_STAGES:
            return stage
        if stage:
            return stage
    if progress and (progress.get("cancelled") or progress_stale(progress)):
        return "cancelled"
    if progress and progress.get("status") in ("running", "done", "error"):
        return progress["status"]
    return "done" if has_evals else "unknown"


def history_row(run_dir: Path, spec: dict | None, evals: pd.DataFrame | None = None) -> dict:
    run_dir = Path(run_dir)
    config = read_json(run_dir / CONFIG_FILE) or {}
    job = read_lab_status(run_dir)
    progress = read_progress(run_dir)
    if evals is None:
        try:
            _, evals = L.load_summary(run_dir)
        except (L.RunFormatError, OSError):
            evals = pd.DataFrame(columns=["after_episode", "matchup", "n", "red_win_rate", "blue_win_rate"])
    fe = final_eval(evals)
    if fe["red"] is None and fe["blue"] is None and progress and progress.get("last_eval"):
        le = progress["last_eval"]
        fe = {"red": le.get("red_win_rate"), "blue": le.get("blue_win_rate"),
              "after_episode": le.get("after_episode")}  # fmt: skip
    init = config.get("init_from") or (job or {}).get("init_from") or None
    parent = ""
    if isinstance(init, dict) and init.get("run_id"):
        parent = f"{init['run_id']} ({init.get('side', 'both')})"
    elif isinstance(init, str):
        parent = Path(init).name
    if job and isinstance(job.get("changed"), dict):
        changed = job["changed"]
    else:
        changed = changed_from_config(spec, config)
    started = (parse_time((progress or {}).get("started")) or parse_time((job or {}).get("started"))
               or run_started(run_dir.name))  # fmt: skip
    episodes = (progress or {}).get("episodes") or config.get("episodes")
    return {
        "label": config.get("label") or (job or {}).get("label") or "",
        "run_id": run_dir.name,
        "started": started,
        "parent": parent,
        "changed": format_changed(changed),
        "red_eval": fe["red"],
        "blue_eval": fe["blue"],
        "eval_at": fe["after_episode"],
        "episodes": episodes,
        "status": run_status(run_dir, job, progress, not evals.empty),
        "run_dir": str(run_dir),
    }


HISTORY_COLUMNS = ["label", "run_id", "started", "parent", "changed", "red_eval", "blue_eval", "eval_at",
                   "episodes", "status", "run_dir"]  # fmt: skip


def history_table(runs_dir: Path, spec: dict | None, summaries: dict[str, pd.DataFrame] | None = None):
    """One row per run directory, newest first. ``summaries`` optionally maps run_dir -> evals frame."""
    rows = [history_row(p, spec, (summaries or {}).get(str(p))) for p in list_run_dirs(runs_dir)]
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def run_display_name(run_dir: Path) -> str:
    """``"<label> · <run_id>"`` (or just the id) for pickers."""
    run_dir = Path(run_dir)
    label = (read_json(run_dir / CONFIG_FILE) or {}).get("label") or (read_lab_status(run_dir) or {}).get(
        "label"
    )
    return f"{label} · {run_dir.name}" if label else run_dir.name


def run_is_done(run_dir: Path) -> bool:
    """A finished run: Lab status ``done``; else progress ``done``; else (old runs) has summary eval rows."""
    run_dir = Path(run_dir)
    job = read_lab_status(run_dir)
    if job:
        return effective_stage(job) == "done"
    prog = read_progress(run_dir)
    if prog and prog.get("status"):
        return prog["status"] == "done"
    return (run_dir / L.SUMMARY_FILE).exists()


def default_replay_run(runs: list[Path]) -> int:
    """Index of the newest finished run in a newest-first list (0 if none is finished)."""
    for i, r in enumerate(runs):
        if run_is_done(r):
            return i
    return 0


def has_agents(run_dir: Path) -> bool:
    a = Path(run_dir) / "agents"
    return (a / "red.json").exists() or (a / "blue.json").exists()
