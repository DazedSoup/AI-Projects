"""One enrich process per run: an OS-level exclusive lock file at ``runs/<id>/explain_cache/.lock``.

- Acquire = ``os.open(O_CREAT | O_EXCL)``: atomic on Windows and POSIX. The file holds ``{"pid", "host",
  "created", "token"}``; the holder touches it (mtime heartbeat) every ``heartbeat_s`` seconds while it runs.
- A lock is **stale** when its PID is dead (same host only) or its last heartbeat is older than ``stale_after_s``
  (10 min). A stale lock is taken over by renaming it aside (only one contender's rename can succeed), checking the
  renamed file is the one judged stale, deleting it, and retrying the exclusive create.
- A live lock makes the second process wait, polling, up to ``timeout_s`` (0 = don't wait); then
  :class:`LockTimeoutError` with a message naming the holder.
- Release (in ``finally`` / ``__exit__``) deletes the file only if it still carries this holder's token.

The lock protects every write under ``explain_cache/`` plus ``episodes_enriched.jsonl`` and ``explain_summary.json``.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

LOCK_NAME = ".lock"
STALE_AFTER_S = 600.0
HEARTBEAT_S = 30.0


class LockTimeoutError(TimeoutError):
    pass


def pid_alive(pid: int) -> bool:
    """True if a process with this PID exists. Never signals the process (``os.kill(pid, 0)`` would terminate it
    on Windows)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: exists, not ours
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock(path: Path) -> dict | None:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return None
    try:
        info = json.loads(text)
        return info if isinstance(info, dict) else {}
    except json.JSONDecodeError:
        return {}  # being written right now, or garbage: judged by age only


class CacheLock:
    def __init__(self, cache_dir: Path, timeout_s: float = 300.0, stale_after_s: float = STALE_AFTER_S,
                 heartbeat_s: float = HEARTBEAT_S, poll_s: float = 0.25, log=print):  # fmt: skip
        self.dir = Path(cache_dir)
        self.path = self.dir / LOCK_NAME
        self.timeout_s = timeout_s
        self.stale_after_s = stale_after_s
        self.heartbeat_s = heartbeat_s
        self.poll_s = poll_s
        self.log = log
        self.token = uuid.uuid4().hex
        self.held = False
        self.took_over: list[dict] = []
        self.waited_s = 0.0
        self._stop = threading.Event()
        self._beat: threading.Thread | None = None

    # ------------------------------------------------------------------------------------------- staleness

    def stale_reason(self, info: dict | None) -> str | None:
        try:
            age = time.time() - self.path.stat().st_mtime
        except FileNotFoundError:
            return None
        if age > self.stale_after_s:
            return f"last heartbeat {age:.0f}s ago (> {self.stale_after_s:.0f}s)"
        same_host = bool(info) and info.get("host") == socket.gethostname()
        if same_host and isinstance(info.get("pid"), int) and not pid_alive(info["pid"]):
            return f"holder PID {info['pid']} is not running"
        return None

    def _try_create(self) -> bool:
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        except PermissionError:  # Windows: the file is being deleted / renamed by another process
            return False
        info = {"pid": os.getpid(), "host": socket.gethostname(), "created": time.time(),
                "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), "token": self.token}  # fmt: skip
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(info))
        return True

    def _take_over(self, info: dict | None, reason: str) -> None:
        aside = self.path.with_name(f"{LOCK_NAME}.stale-{os.getpid()}-{time.time_ns()}")
        try:
            os.rename(self.path, aside)
        except (FileNotFoundError, PermissionError, FileExistsError):
            return  # someone else got there first
        moved = read_lock(aside)
        if info and moved and moved.get("token") != info.get("token"):
            # we renamed a fresh lock created after our staleness check: put it back if the slot is still free
            try:
                os.rename(aside, self.path)
            except OSError:
                pass
            return
        self.took_over.append({**(moved or {}), "reason": reason})
        self.log(f"explain cache: took over a stale lock ({reason})")
        try:
            os.remove(aside)
        except OSError:
            pass

    # ------------------------------------------------------------------------------------------- acquire/release

    def acquire(self) -> CacheLock:
        t0 = time.monotonic()
        announced = False
        while True:
            if self._try_create():
                self.held = True
                self.waited_s = time.monotonic() - t0
                self._start_heartbeat()
                return self
            info = read_lock(self.path)
            reason = self.stale_reason(info)
            if reason:
                self._take_over(info, reason)
                continue
            waited = time.monotonic() - t0
            who = self.describe(info)
            if waited >= self.timeout_s:
                raise LockTimeoutError(
                    f"another enrich holds {self.path} ({who}); gave up after {waited:.0f}s. Wait for it to finish "
                    f"and rerun (cached work is reused), or raise --lock-timeout. If no enrich is running, the lock "
                    f"is taken over automatically once its holder is gone or {self.stale_after_s / 60:.0f} min old."
                )
            if not announced:
                self.log(f"explain cache is locked by another enrich ({who}); waiting up to {self.timeout_s:.0f}s ...")
                announced = True
            time.sleep(min(self.poll_s, max(0.01, self.timeout_s - waited)))

    @staticmethod
    def describe(info: dict | None) -> str:
        if not info:
            return "holder unknown"
        return f"PID {info.get('pid')} on {info.get('host')}, since {info.get('created_iso', '?')}"

    def _start_heartbeat(self) -> None:
        self._stop.clear()

        def beat() -> None:
            while not self._stop.wait(self.heartbeat_s):
                try:
                    os.utime(self.path)
                except OSError:
                    pass

        self._beat = threading.Thread(target=beat, name="explain-cache-lock", daemon=True)
        self._beat.start()

    def release(self) -> None:
        self._stop.set()
        if self._beat is not None:
            self._beat.join(timeout=2)
            self._beat = None
        if not self.held:
            return
        self.held = False
        info = read_lock(self.path)
        if info and info.get("token") == self.token:
            try:
                os.remove(self.path)
            except OSError:
                pass

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
