"""Detached Simulation Lab runner: ``train`` then offline ``enrich`` (or one ``arena.experiment``), with the
job's status file kept current.

    python -m cyberarena.dashboard.lab_runner --job runs/.lab/<token>.json

The dashboard writes the job file (train command, label, changed params) and starts this module detached, so
the chain finishes even if the browser tab or Streamlit goes away. Stages written: ``training`` ->
``enriching`` -> ``done``, or ``error`` with a message. It only runs subprocesses; it imports no arena, ml or
explain code, and enrichment always runs with ``--offline`` (no API key is passed to the children either).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

from cyberarena.dashboard import lab

_TAIL = 25
CANCEL_EXIT = 130  # train's exit code after a graceful cancel
KILL_AFTER_S = 20.0  # a child that ignores the cancel signal this long is killed


def _cancel(proc: subprocess.Popen) -> None:
    """Graceful cancel: CTRL_BREAK_EVENT to the child's own process group (Windows), SIGINT elsewhere.
    ``terminate()`` would be TerminateProcess on Windows, which train cannot catch to update progress.json."""
    lab.refuse_in_public("Cancelling a process")
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
    except (OSError, ValueError):
        proc.kill()


def _watch_stop(proc: subprocess.Popen, job: dict, state: dict) -> None:
    """Poll for the dashboard's stop-request file while ``proc`` runs."""
    while proc.poll() is None:
        if lab.stop_requested_at(job) is not None:
            state["stopped"] = True
            _cancel(proc)
            deadline = time.monotonic() + KILL_AFTER_S
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.2)
            if proc.poll() is None:
                lab.kill_tree(proc.pid)
            return
        time.sleep(0.5)


def _error_line(tail: deque[str]) -> str:
    lines = [ln.strip() for ln in tail if ln.strip()]
    if not lines:
        return "no output"
    for ln in reversed(lines):  # prefer the exception line of a traceback
        if "Error" in ln or "error" in ln:
            return ln[:500]
    return lines[-1][:500]


def _run_logged(
    cmd: list[str], log_path: Path, job: dict, cwd: str | None, on_line=None, pid_key: str = "child_pid"
) -> tuple[int, deque]:
    lab.refuse_in_public("Running a Lab job")
    tail: deque[str] = deque(maxlen=_TAIL)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("$ " + lab.format_cmd(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
            encoding="utf-8", errors="replace", cwd=cwd, env=lab.child_env(),
            creationflags=lab.child_popen_flags(),
        )  # fmt: skip
        job["child_pid"] = job[pid_key] = proc.pid
        lab.save_job(job)
        state = {"stopped": False}
        threading.Thread(target=_watch_stop, args=(proc, job, state), daemon=True).start()
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            tail.append(line)
            if on_line:
                on_line(line)
        rc = proc.wait()
        log.write(f"[exit code {rc}]\n")
    job["child_pid"] = job[pid_key] = None
    if state["stopped"]:
        job["stopped"] = True
    return rc, tail


def _stopped(job: dict, rc: int, what: str) -> int:
    job.update(stage="error", stopped=True, finished=lab.now_iso(),
               message=f"Stopped by user during {what} (exit code {rc}). The partial run stays on disk.")  # fmt: skip
    lab.save_job(job)
    return rc


def run_experiment(job: dict, job_file: Path) -> int:
    """``arena.experiment`` as the child (it runs its own train processes in parallel). Stop sends it
    CTRL_BREAK_EVENT, which it forwards to every train; exit 130 = cancelled, 1 = finished with failed runs."""
    lab.refuse_in_public("Running an experiment")
    job.update(pid=os.getpid(), stage="training", message="Experiment: starting the runs…")
    lab.save_job(job)
    cmd = list(job["train_cmd"])
    if "--online" in cmd:
        job.update(stage="error", message="Refusing to run: --online is never allowed from the Lab.")
        lab.save_job(job)
        return 2
    log = Path(job_file).parent / f"{job['token']}.experiment.log"
    job["train_log"] = str(log)
    rc, tail = _run_logged(cmd, log, job, job.get("cwd"), pid_key="train_pid")
    job["train_exit"] = rc
    if job.get("stopped") or rc == CANCEL_EXIT:
        return _stopped(job, rc, "the experiment")
    if rc not in (0, 1):
        job.update(stage="error", finished=lab.now_iso(),
                   message=f"Experiment failed (exit code {rc}): {_error_line(tail)}")  # fmt: skip
        lab.save_job(job)
        return rc
    man = lab.read_json(Path(job["exp_dir"]) / "manifest.json") or {}
    runs = man.get("runs") or []
    bad = sum(r.get("status") != "done" for r in runs)
    job.update(stage="done", finished=lab.now_iso(),
               message=f"Finished: {len(runs) - bad} of {len(runs)} runs done" + (f", {bad} failed." if bad else "."))  # fmt: skip
    lab.save_job(job)
    return 0


def run(job_file: Path) -> int:
    lab.refuse_in_public("Running a Lab job")
    job = lab.read_json(job_file)
    if job is None:
        print(f"lab_runner: cannot read job file {job_file}", file=sys.stderr)
        return 2
    job["job_file"] = str(job_file)
    if job.get("kind") == "experiment":
        return run_experiment(job, job_file)
    job.update(pid=os.getpid(), stage="training", message="Training: starting (loading classifiers)…")
    lab.save_job(job)
    cwd = job.get("cwd")
    py = job.get("python") or sys.executable
    train = list(job["train_cmd"])
    if "--online" in train:
        job.update(stage="error", message="Refusing to run: --online is never allowed from the Lab.")
        lab.save_job(job)
        return 2
    log_dir = Path(job_file).parent
    train_log = log_dir / f"{job['token']}.train.log"
    job["train_log"] = str(train_log)

    def on_train_line(line: str) -> None:
        if job.get("run_dir") is None and line.startswith("RUN_DIR "):
            run_dir = line[len("RUN_DIR ") :].strip()
            job.update(run_dir=run_dir, run_id=Path(run_dir).name, message="Training…")
            lab.save_job(job)

    rc, tail = _run_logged(train, train_log, job, cwd, on_train_line, pid_key="train_pid")
    job["train_exit"] = rc
    if job.get("stopped") or rc == CANCEL_EXIT:
        return _stopped(job, rc, "training")
    if rc != 0:
        msg = None
        if job.get("run_dir"):
            prog = lab.read_progress(Path(job["run_dir"]))
            msg = (prog or {}).get("error")
        job.update(stage="error", message=f"Training failed (exit code {rc}): {msg or _error_line(tail)}",
                   finished=lab.now_iso())  # fmt: skip
        lab.save_job(job)
        return rc
    if not job.get("run_dir"):
        job.update(stage="error", message="Training exited without printing 'RUN_DIR <path>'.",
                   finished=lab.now_iso())  # fmt: skip
        lab.save_job(job)
        return 3

    run_dir = Path(job["run_dir"])
    if job.get("enrich", True):
        job.update(stage="enriching", message="Enriching (offline: SHAP, MITRE tags, template rationales)…",
                   enrich_started=lab.now_iso())  # fmt: skip
        lab.save_job(job)
        cmd = lab.enrich_cmd(run_dir, py)
        assert "--online" not in cmd
        enrich_log = run_dir / "lab_enrich.log"
        job["enrich_log"] = str(enrich_log)
        rc, tail = _run_logged(cmd, enrich_log, job, cwd)
        job["enrich_exit"] = rc
        if job.get("stopped"):
            return _stopped(job, rc, "enrichment")
        if rc != 0:
            job.update(stage="error", finished=lab.now_iso(),
                       message=f"Training finished, but enrichment failed (exit code {rc}): {_error_line(tail)}. "
                       "The run can still be replayed; its episodes just aren't enriched.")  # fmt: skip
            lab.save_job(job)
            return rc
    prog = lab.read_progress(run_dir) or {}
    eps = prog.get("episodes")
    job.update(stage="done", finished=lab.now_iso(),
               message=f"Finished{f': {eps} episodes' if eps else ''}, enriched offline.")  # fmt: skip
    lab.save_job(job)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cyberarena.dashboard.lab_runner", description=__doc__)
    ap.add_argument("--job", required=True, type=Path, help="runs/.lab/<token>.json written by the dashboard")
    args = ap.parse_args(argv)
    lab.refuse_in_public("The Lab runner")  # before the try: a public process must not even write the job file
    try:
        return run(args.job)
    except Exception as e:  # noqa: BLE001 - the status file must always say why we stopped
        job = lab.read_json(args.job) or {"token": args.job.stem}
        job["job_file"] = str(args.job)
        job.update(stage="error", message=f"Lab runner crashed: {type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-2000:], finished=lab.now_iso())  # fmt: skip
        try:
            lab.save_job(job)
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
