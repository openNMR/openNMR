"""Autosave and crash recovery (explicit user request, 2026-11-12).

While openNMR runs, the open session is written every few minutes
(Settings ▸ Saving ▸ Autosave) as an ordinary openNMR state -- the same
format File ▸ Save openNMR State writes, so a recovered session reopens
exactly as a saved one does: each experiment re-read from its own data
with its saved processing, everything else from the copy. Nothing is
written while nothing has changed.

**How a crash is recognised.** A running window leaves a small lock file
(``session.lock``, holding its process id) in the recovery folder and
removes it, with the autosave, when it closes normally. A lock from
another process found at start-up therefore means the last session did
not close normally; its autosave is moved aside to ``previous-<time>
.opennmr`` -- so this session's own autosaves cannot overwrite it -- and
the window offers to restore it. Declined, it stays available from File ▸
Recover Autosaved Session… until :data:`KEEP_PREVIOUS` newer ones replace
it. (openNMR runs one window at a time, so a lock from another process is
never a second live window.)

**Written safely.** Each autosave goes to a new folder that replaces the
last only once it is complete, so a crash *during* an autosave leaves the
previous one intact.

The folder is ``~/.opennmr/recovery`` (``OPENNMR_RECOVERY_DIR`` overrides
it; the tests use that). No Qt here.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil

from . import state as io_state

#: Previous sessions kept for File ▸ Recover Autosaved Session….
KEEP_PREVIOUS = 3
#: Default minutes between autosaves; 0 turns autosave off.
DEFAULT_INTERVAL_MIN = 3
LOCK_FILE = "session.lock"
AUTOSAVE_NAME = "autosave" + io_state.STATE_SUFFIX
PREVIOUS_PREFIX = "previous-"


def recovery_root() -> str:
    root = os.environ.get("OPENNMR_RECOVERY_DIR") or os.path.join(
        os.path.expanduser("~"), ".opennmr", "recovery")
    return root


def autosave_folder() -> str:
    return os.path.join(recovery_root(), AUTOSAVE_NAME)


def _lock_path() -> str:
    return os.path.join(recovery_root(), LOCK_FILE)


def read_lock() -> dict | None:
    try:
        with open(_lock_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def write_lock(pid: int | None = None) -> None:
    """Say this process is running (see the module notes)."""
    os.makedirs(recovery_root(), exist_ok=True)
    with open(_lock_path(), "w", encoding="utf-8") as handle:
        json.dump({"pid": int(pid if pid is not None else os.getpid()),
                   "started": _dt.datetime.now().isoformat(timespec="seconds")}, handle)


def clean_exit(pid: int | None = None) -> None:
    """A normal close: remove this process's lock and its autosave. A
    lock written by another process is left alone."""
    lock = read_lock()
    if lock is not None and lock.get("pid") != int(pid if pid is not None else os.getpid()):
        return
    for path in (_lock_path(),):
        try:
            os.remove(path)
        except OSError:
            pass
    shutil.rmtree(autosave_folder(), ignore_errors=True)


def previous_sessions() -> list[str]:
    """Autosaves of sessions that did not close normally, newest first."""
    root = recovery_root()
    try:
        names = [n for n in os.listdir(root) if n.startswith(PREVIOUS_PREFIX)]
    except OSError:
        return []
    folders = [os.path.join(root, n) for n in names]
    return sorted((f for f in folders if io_state.is_state_folder(f)), reverse=True)


def _prune() -> None:
    for folder in previous_sessions()[KEEP_PREVIOUS:]:
        shutil.rmtree(folder, ignore_errors=True)


def take_crashed_session(pid: int | None = None) -> str | None:
    """At start-up: if the last session did not close normally and left an
    autosave, move it aside (``previous-<time>.opennmr``) and return that
    folder; else None. Either way, lock the recovery folder for this
    process. Call once, before the first autosave."""
    pid = int(pid if pid is not None else os.getpid())
    lock = read_lock()
    moved = None
    crashed = lock is not None and lock.get("pid") != pid
    if crashed and io_state.is_state_folder(autosave_folder()):
        stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        moved = os.path.join(recovery_root(), f"{PREVIOUS_PREFIX}{stamp}{io_state.STATE_SUFFIX}")
        suffix = 1
        while os.path.exists(moved):
            moved = os.path.join(recovery_root(),
                                 f"{PREVIOUS_PREFIX}{stamp}-{suffix}{io_state.STATE_SUFFIX}")
            suffix += 1
        os.replace(autosave_folder(), moved)
        _prune()
    elif crashed:
        shutil.rmtree(autosave_folder(), ignore_errors=True)
    write_lock(pid)
    return moved


def write_autosave(save) -> str:
    """Write a new autosave with ``save(folder)`` (which writes an openNMR
    state into ``folder``), replacing the last one only once it is
    complete. Returns the autosave folder."""
    root = recovery_root()
    os.makedirs(root, exist_ok=True)
    final = autosave_folder()
    fresh = final + ".new"
    shutil.rmtree(fresh, ignore_errors=True)
    save(fresh)
    old = final + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(final):
        os.replace(final, old)
    os.replace(fresh, final)
    shutil.rmtree(old, ignore_errors=True)
    return final


def describe(folder: str) -> str:
    """"12 spectra, saved 2026-11-12 16:40" for a recovery folder."""
    try:
        info = io_state.read_state_info(folder)
    except (OSError, ValueError, KeyError):
        return os.path.basename(folder)
    count = len(info.get("entries") or [])
    when = str(info.get("created", "")).replace("T", " ")
    return f"{count} spectr{'um' if count == 1 else 'a'}, autosaved {when}"
