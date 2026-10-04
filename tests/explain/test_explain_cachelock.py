"""explain_cache/.lock: exclusive, waits or exits cleanly, takes over stale locks, never corrupts the cache."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from cyberarena.explain import enrich
from cyberarena.explain.cachelock import LOCK_NAME, CacheLock, LockTimeoutError, pid_alive, read_lock

WRITER = textwrap.dedent("""
    import json, sys, time
    from cyberarena.explain.cachelock import CacheLock
    cache, tag, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    lock = CacheLock(cache, timeout_s=60, poll_s=0.02, log=lambda *a: None)
    lock.acquire()
    try:
        with open(f"{cache}/shap.jsonl", "a", encoding="utf-8") as fh:
            for i in range(n):  # slow, line-by-line appends: interleaving would show without the lock
                fh.write(json.dumps({"key": f"{tag}-{i}", "value": {"pad": "x" * 2000}}) + "\\n")
                fh.flush()
                time.sleep(0.005)
    finally:
        lock.release()
""")


def test_two_processes_never_interleave_cache_writes(tmp_path):
    cache = tmp_path / "explain_cache"
    cache.mkdir()
    script = tmp_path / "writer.py"
    script.write_text(WRITER)
    procs = [subprocess.Popen([sys.executable, str(script), str(cache), tag, "60"]) for tag in ("A", "B")]
    assert [p.wait(timeout=120) for p in procs] == [0, 0]
    lines = (cache / "shap.jsonl").read_text(encoding="utf-8").splitlines()
    keys = [json.loads(line)["key"] for line in lines]  # every line parses: nothing torn
    assert len(keys) == 120
    tags = [k.split("-")[0] for k in keys]
    assert tags in (["A"] * 60 + ["B"] * 60, ["B"] * 60 + ["A"] * 60)  # one whole block per process
    assert not (cache / LOCK_NAME).exists()


def test_second_process_times_out_with_a_clear_message(tmp_path):
    cache = tmp_path / "explain_cache"
    with CacheLock(cache):
        code = ("import sys; from cyberarena.explain.cachelock import CacheLock, LockTimeoutError\n"
                "try:\n    CacheLock(sys.argv[1], timeout_s=0.3, log=lambda *a: None).acquire()\n"
                "except LockTimeoutError as e:\n    print(e); sys.exit(3)\n")  # fmt: skip
        out = subprocess.run([sys.executable, "-c", code, str(cache)], capture_output=True, text=True, timeout=60,
                             check=False)
    assert out.returncode == 3
    assert f"PID {os.getpid()}" in out.stdout and "--lock-timeout" in out.stdout


def test_waiting_process_gets_the_lock_after_release(tmp_path):
    cache = tmp_path / "explain_cache"
    holder = CacheLock(cache).acquire()
    code = ("import sys, time; from cyberarena.explain.cachelock import CacheLock\n"
            "l = CacheLock(sys.argv[1], timeout_s=30, poll_s=0.05, log=lambda *a: None).acquire()\n"
            "print(round(l.waited_s, 1)); l.release()\n")  # fmt: skip
    p = subprocess.Popen([sys.executable, "-c", code, str(cache)], stdout=subprocess.PIPE, text=True)
    time.sleep(1.5)
    holder.release()
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 0 and float(out) >= 0.5


def _plant(cache: Path, pid: int, age_s: float = 0.0) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / LOCK_NAME
    path.write_text(json.dumps({"pid": pid, "host": socket.gethostname(), "created": time.time() - age_s,
                                "token": "old"}))  # fmt: skip
    if age_s:
        t = time.time() - age_s
        os.utime(path, (t, t))
    return path


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_dead_holder_is_taken_over(tmp_path):
    cache = tmp_path / "explain_cache"
    pid = _dead_pid()
    assert not pid_alive(pid) and pid_alive(os.getpid())
    _plant(cache, pid)
    lock = CacheLock(cache, timeout_s=0, log=lambda *a: None).acquire()
    try:
        assert lock.took_over and "not running" in lock.took_over[0]["reason"]
        assert read_lock(cache / LOCK_NAME)["pid"] == os.getpid()
    finally:
        lock.release()
    assert not (cache / LOCK_NAME).exists() and not list(cache.glob(".lock.stale-*"))


def test_old_lock_is_taken_over_even_if_pid_lives(tmp_path):
    cache = tmp_path / "explain_cache"
    _plant(cache, os.getpid(), age_s=11 * 60)
    with CacheLock(cache, timeout_s=0, log=lambda *a: None) as lock:
        assert "heartbeat" in lock.took_over[0]["reason"]


def test_live_fresh_lock_is_respected(tmp_path):
    cache = tmp_path / "explain_cache"
    _plant(cache, os.getpid())
    with pytest.raises(LockTimeoutError):
        CacheLock(cache, timeout_s=0, log=lambda *a: None).acquire()
    assert read_lock(cache / LOCK_NAME)["token"] == "old"  # untouched


def test_release_never_deletes_someone_elses_lock(tmp_path):
    cache = tmp_path / "explain_cache"
    lock = CacheLock(cache).acquire()
    _plant(cache, os.getpid())  # e.g. taken over by another process meanwhile
    lock.release()
    assert read_lock(cache / LOCK_NAME)["token"] == "old"


def test_enrich_cli_exits_cleanly_when_locked(tiny_run, capsys):
    with CacheLock(tiny_run / "explain_cache"):
        rc = enrich.main(["--run", str(tiny_run), "--no-shap", "--lock-timeout", "0.2"])
    assert rc == enrich.LOCK_EXIT
    assert "another enrich holds" in capsys.readouterr().err
    assert not (tiny_run / "episodes_enriched.jsonl").exists()
    assert enrich.main(["--run", str(tiny_run), "--no-shap", "--lock-timeout", "0"]) == 0  # released afterwards
    assert not (tiny_run / "explain_cache" / LOCK_NAME).exists()
