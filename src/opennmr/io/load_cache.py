"""
Saved copies of slow-to-open spectra, so that opening them again is quick.

A large experiment -- an 800 MHz EXSY processed to 512 x 131072 points --
takes many seconds to read and process from its raw ``ser``. The result
depends only on the files and on how they are read, so once it has been
made it can be kept and read back in a second or two next time.

**A copy is only ever used if nothing it depends on has changed.** The key
it is filed under is built from:

* every file of the experiment -- name, size and modification time (the
  raw data, every parameter file, pdata, lists ...), except the files
  openNMR itself writes beside an experiment as you work (history, log,
  metadata and processing sidecars, save points), which no reader reads;
* every setting the read was made with (procno, window, zero-fill, phase,
  pseudo-2D ...): the same experiment opened another way is another copy;
* the openNMR version and a fingerprint of the reading and processing
  code itself, so a changed reader never hands back an old result.

Anything else -- a file that cannot be read, a copy that will not load --
means the experiment is simply read again from its files.

**Where:** the local cache folder of this computer (on Windows
``%LOCALAPPDATA%\\openNMR\\cache\\spectra``), never next to the data -- a
synced folder such as OneDrive would upload gigabytes. ``OPENNMR_CACHE_DIR``
moves it. **How much:** a copy is kept only when the read took at least
``min_seconds`` (2 s by default), and the oldest copies are removed when
the folder passes ``max_bytes`` (5 GB by default) -- or when the disk is
short of room. ``OPENNMR_LOAD_CACHE=0`` turns the whole thing off.

The copies are Python pickles written by openNMR into the user's own cache
folder, and are only read from there.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import time

import numpy as np

from ..core import load_progress

#: Format of a saved copy; a different one is never read.
FORMAT = 1
_MAGIC = "opennmr-load-cache"
ENV_DISABLE = "OPENNMR_LOAD_CACHE"
ENV_DIR = "OPENNMR_CACHE_DIR"

#: openNMR's own files beside an experiment: written while you work, never
#: read by a reader -- so saving your processing does not void the copy.
_OWN_SUFFIXES = ("opennmr_history.json", "opennmr_log.txt", "opennmr_metadata.json",
                 "opennmr_processing.json", "opennmr_history.json.tmp",
                 "opennmr_log.txt.tmp", "opennmr_metadata.json.tmp",
                 "opennmr_processing.json.tmp")
_OWN_DIRS = ("opennmr_savepoints",)
#: An "experiment" with more files than this is not one -- not cached.
_MAX_FILES = 5000

_settings = {"enabled": True, "max_bytes": 5 * 1024 ** 3, "min_seconds": 2.0,
             "directory": None}
_code_fingerprint: str | None = None


def configure(enabled: bool | None = None, max_bytes: int | None = None,
              min_seconds: float | None = None, directory: str | None = None) -> dict:
    """Change the settings (``None`` leaves one as it is). Returns them."""
    if enabled is not None:
        _settings["enabled"] = bool(enabled)
    if max_bytes is not None:
        _settings["max_bytes"] = max(0, int(max_bytes))
    if min_seconds is not None:
        _settings["min_seconds"] = max(0.0, float(min_seconds))
    if directory is not None:
        _settings["directory"] = directory or None
    return dict(_settings)


def enabled() -> bool:
    value = os.environ.get(ENV_DISABLE)
    if value is not None and value.strip().lower() in ("0", "false", "no", "off"):
        return False
    return bool(_settings["enabled"])


def cache_dir() -> str:
    """This computer's folder for the saved copies (not created here)."""
    if os.environ.get(ENV_DIR):
        return os.environ[ENV_DIR]
    if _settings["directory"]:
        return _settings["directory"]
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
        return os.path.join(base, "openNMR", "cache", "spectra")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Caches/openNMR/spectra")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "openNMR", "spectra")


def _code_version() -> str:
    """openNMR's version and a hash of the reading/processing code."""
    global _code_fingerprint
    if _code_fingerprint is None:
        from .. import __version__
        digest = hashlib.sha256(str(__version__).encode())
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for package in ("core", "io", "processing"):
            for name in sorted(glob.glob(os.path.join(root, package, "*.py"))):
                try:
                    with open(name, "rb") as handle:
                        digest.update(os.path.basename(name).encode())
                        digest.update(handle.read())
                except OSError:
                    continue
        _code_fingerprint = digest.hexdigest()
    return _code_fingerprint


def _is_own_file(name: str) -> bool:
    return name.endswith(_OWN_SUFFIXES)


def _file_fingerprint(path: str) -> list | None:
    """``[relative name, size, mtime_ns]`` of everything the read may use;
    None when the experiment is too big a folder to stand for one."""
    path = os.path.abspath(path)
    entries = []
    if os.path.isdir(path):
        root = path
        for folder, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if not d.endswith(_OWN_DIRS))
            for name in sorted(files):
                if _is_own_file(name):
                    continue
                full = os.path.join(folder, name)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                entries.append([os.path.relpath(full, root).replace("\\", "/"),
                                st.st_size, st.st_mtime_ns])
                if len(entries) > _MAX_FILES:
                    return None
    else:
        # One file: it and its neighbours (a reader may read a sibling).
        root = os.path.dirname(path)
        for name in sorted(os.listdir(root)):
            full = os.path.join(root, name)
            if not os.path.isfile(full) or _is_own_file(name) or ".opennmr_" in name:
                continue
            try:
                st = os.stat(full)
            except OSError:
                continue
            entries.append([name, st.st_size, st.st_mtime_ns])
            if len(entries) > _MAX_FILES:
                return None
    return entries


def _json_default(value):
    if isinstance(value, np.ndarray):
        return {"ndarray": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return repr(value)


def cache_key(path: str, vendor: str | None, kwargs: dict | None) -> str | None:
    """The key a copy of this read is filed under, or None if it cannot
    be cached (see the module docstring for what goes into it)."""
    files = _file_fingerprint(path)
    if files is None:
        return None
    blob = json.dumps({"format": FORMAT, "code": _code_version(),
                       "path": os.path.normcase(os.path.abspath(path)),
                       "vendor": vendor, "kwargs": kwargs or {}, "files": files},
                      sort_keys=True, default=_json_default)
    return hashlib.sha256(blob.encode()).hexdigest()


def _copy_path(key: str) -> str:
    return os.path.join(cache_dir(), key[:40] + ".pkl")


def _read_copy(key: str):
    path = _copy_path(key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as handle:
            header = pickle.load(handle)
            if (not isinstance(header, dict) or header.get("magic") != _MAGIC
                    or header.get("format") != FORMAT or header.get("key") != key):
                return None
            spectrum = pickle.load(handle)
    except Exception:  # noqa: BLE001 -- a damaged copy is simply read again
        _remove(path)
        return None
    try:
        os.utime(path, None)                    # most recently used
    except OSError:
        pass
    return spectrum, header


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _array_bytes(obj, seen=None, depth=0) -> int:
    """Rough size of the arrays inside ``obj`` (for the room check)."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 6:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(_array_bytes(v, seen, depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_array_bytes(v, seen, depth + 1) for v in obj)
    if hasattr(obj, "__dict__"):
        return _array_bytes(vars(obj), seen, depth + 1)
    return 0


def _write_copy(key: str, spectrum, header: dict) -> bool:
    folder = cache_dir()
    size = _array_bytes(spectrum)
    if size > _settings["max_bytes"]:
        return False
    try:
        os.makedirs(folder, exist_ok=True)
        free = shutil.disk_usage(folder).free
    except OSError:
        return False
    # Leave the disk at least 1 GB (and twice the copy) to spare.
    if free < 2 * size + 1024 ** 3:
        prune(max(0, _settings["max_bytes"] - (2 * size + 1024 ** 3 - free)))
        try:
            free = shutil.disk_usage(folder).free
        except OSError:
            return False
        if free < 2 * size + 1024 ** 3:
            return False
    fd, temp = tempfile.mkstemp(prefix=".partial-", suffix=".pkl", dir=folder)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(header, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(spectrum, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp, _copy_path(key))
    except Exception:  # noqa: BLE001 -- not being able to keep a copy is not an error
        _remove(temp)
        return False
    prune(_settings["max_bytes"], keep=_copy_path(key))
    return True


def copies() -> list[tuple[str, int, float]]:
    """``(path, bytes, last used)`` of every saved copy, newest first."""
    out = []
    for path in glob.glob(os.path.join(cache_dir(), "*.pkl")):
        try:
            st = os.stat(path)
        except OSError:
            continue
        out.append((path, st.st_size, st.st_mtime))
    return sorted(out, key=lambda item: item[2], reverse=True)


def total_bytes() -> int:
    return sum(size for _path, size, _used in copies())


def prune(max_bytes: int, keep: str | None = None) -> int:
    """Remove the least recently used copies until the folder holds at
    most ``max_bytes`` (``keep`` is spared). Returns the bytes removed."""
    removed = 0
    items = copies()
    total = sum(size for _path, size, _used in items)
    for path, size, _used in reversed(items):          # oldest first
        if total <= max_bytes:
            break
        if keep is not None and os.path.normcase(path) == os.path.normcase(keep):
            continue
        _remove(path)
        total -= size
        removed += size
    for stale in glob.glob(os.path.join(cache_dir(), ".partial-*.pkl")):
        if time.time() - os.path.getmtime(stale) > 3600:
            _remove(stale)
    return removed


def clear() -> int:
    """Remove every saved copy. Returns the bytes freed."""
    return prune(0)


def load(path: str, vendor: str | None = None, kwargs: dict | None = None,
         loader=None) -> tuple[object, dict]:
    """Read ``path`` as :func:`opennmr.io.load` would, using a saved copy
    when one is valid, and keeping one when the read was slow.

    Returns ``(spectrum, info)``: ``info["from_copy"]`` says which, with
    ``"seconds"`` (this time) and, for a copy, ``"first_seconds"`` (what
    the read had taken)."""
    from . import base as io_base
    loader = loader or (lambda p, v, k: io_base.load(p, vendor=v, **k))
    kwargs = dict(kwargs or {})
    started = time.perf_counter()
    key = None
    if enabled():
        try:
            key = cache_key(path, vendor, kwargs)
        except Exception:  # noqa: BLE001 -- no key, no copy: read normally
            key = None
    if key is not None and os.path.isfile(_copy_path(key)):
        load_progress.stage("Reading the saved copy")
        found = _read_copy(key)
        if found is not None:
            spectrum, header = found
            return spectrum, {"from_copy": True,
                              "seconds": time.perf_counter() - started,
                              "first_seconds": header.get("seconds")}
    spectrum = loader(path, vendor, kwargs)
    seconds = time.perf_counter() - started
    info = {"from_copy": False, "seconds": seconds, "kept": False}
    if key is not None and seconds >= _settings["min_seconds"]:
        load_progress.stage("Keeping a copy for quick reopening")
        # The files must still be the ones that were read.
        if cache_key(path, vendor, kwargs) == key:
            info["kept"] = _write_copy(key, spectrum, {
                "magic": _MAGIC, "format": FORMAT, "key": key, "seconds": seconds,
                "path": os.path.abspath(path), "created": time.time()})
    return spectrum, info


def read(path: str, vendor: str | None = None, **kwargs):
    """For the notebook: :func:`opennmr.io.load`, with saved copies --
    ``nmr.load_cached(path, procno=1)``. Returns the spectrum."""
    return load(path, vendor, kwargs)[0]
