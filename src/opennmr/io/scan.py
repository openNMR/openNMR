"""
Finding the experiments under a folder -- fast, cloud-safe, and
remembered between sessions.

Explicit user request (2026-11-13): opening openNMR with a large default
data folder took minutes on a recent machine and up to an hour on older
ones, because the whole folder was scanned before the window could
respond. A data folder is where people pile everything -- thousands to
tens of thousands of experiments, on any disk, often a cloud-synced one.

What made the scan slow, and what this module does instead:

* **Every file was asked "are you an experiment?" by every reader.** One
  of those readers (SIMPSON) recognises its files by their first line,
  so it *opened every file in the folder* -- notes, images, PDFs. On a
  cloud-synced folder, opening an online-only file downloads it. Here a
  file is only examined when its extension is one some reader uses
  (:data:`FILE_EXTENSIONS`), and a cloud-only file is never opened: it
  is recognised from its extension alone (:func:`_detect_file`).
* **Every folder was asked the same, by file-existence checks** -- a
  few per reader per folder, and on Windows, and on network or cloud
  storage, each one is slow. Here a folder is listed once and only
  handed to the readers when one of its names is one an experiment has
  (:data:`DIR_MARKERS`, or an X-Pulse ``arrayN`` subfolder).
* **Nothing was remembered.** :class:`ScanIndex` keeps what each folder
  held, with the folder's modification time. A folder whose time has not
  changed is not listed or examined again; adding or removing anything
  in it changes its time, so new and deleted experiments are found. The
  index lives in this computer's cache folder, never next to the data
  (a synced folder would upload it). ``OPENNMR_SCAN_INDEX_DIR`` moves it.

What is found, and in what order, is unchanged: the same experiments the
readers recognise, folders in natural order ("2" before "10"), an
experiment's own subtree not entered (except its ``opennmr_derived``
folder, and a Spinsolve 2D's pilot scan), X-Pulse per-scan ``_N_raw``
files and redundant ``raw_slices`` left out, stopping at ``max_matches``.

Callable from a notebook (``from opennmr.io import scan``) as well as by
the app, which runs it in the background (``gui/folder_scan.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time

from . import cloud

#: The most experiments one scan lists.
SCAN_LIMIT = 10_000
#: How deep below the chosen folder the scan looks.
MAX_DEPTH = 6

#: A folder is handed to the readers only if it contains one of these
#: names (or an X-Pulse ``arrayN`` subfolder): Bruker ``acqus``, Varian
#: ``procpar``, Spinsolve ``acqu.par``, openNMR's own dataset file. A
#: reader that recognises folders by another file must add it here --
#: ``tests/test_folder_scan.py`` checks every registered folder reader.
DIR_MARKERS = frozenset({"acqus", "procpar", "acqu.par", "opennmr_spectrum.npz"})
_ARRAY_DIR_RE = re.compile(r"^array(\d+)$", re.IGNORECASE)

#: The only files examined: X-Pulse and exported JCAMP-DX, JEOL, tables,
#: and SIMPSON's text output. A SIMPSON file with another name can
#: still be opened with File ▸ Open; it is just not listed here.
FILE_EXTENSIONS = frozenset({".jdx", ".dx", ".jdf", ".csv", ".xy",
                             ".spe", ".fid", ".csdf", ".dat", ".tnt", ".asc", ".fxml"})
#: A cloud-only file is recognised from its extension alone (opening it
#: would download it). Anything else cloud-only is not listed.
CLOUD_EXTENSION_VENDORS = {".jdx": "xpulse", ".dx": "xpulse", ".jdf": "jeol",
                           ".csv": "text", ".xy": "text"}

#: Folders never entered: version control, caches and system folders.
SKIP_DIRS = frozenset({".git", ".svn", ".hg", "__pycache__", "$RECYCLE.BIN",
                       "System Volume Information", ".Trash", ".Trashes",
                       ".DS_Store", "opennmr_savepoints"})

#: X-Pulse writes each scan of an experiment as ``<name>_<N>_raw.jdx``
#: beside the averaged spectrum: those are not offered separately.
XPULSE_RAW_SCAN_RE = re.compile(r"_\d+_raw\.(jdx|dx)$", re.IGNORECASE)

#: Files whose modification time is an experiment's acquisition date,
#: best first (the browser sorts by it).
DATE_FILES = ("fid", "ser", "acqus", "acqu", "data.1d", "data.2d", "acqu.par", "procpar")

_NATURAL_SPLIT = re.compile(r"(\d+)")


def natural_key(name: str) -> list:
    """"2" before "10": digit runs compare as numbers, the rest
    case-insensitively; each chunk tagged so a number never meets a
    string in a comparison."""
    return [(0, int(chunk), "") if chunk.isdigit() else (1, 0, chunk.lower())
            for chunk in _NATURAL_SPLIT.split(str(name)) if chunk != ""]


class ScanCancelled(Exception):
    """Raised inside a scan when ``cancel()`` says to stop."""


class ScanResult(list):
    """The ``(path, vendor)`` pairs found -- a plain list, so it compares
    equal to one -- with what the scan did alongside.

    ``dates``: ``{path: acquisition time}`` where it was cheap to learn.
    ``truncated``: stopped at ``max_matches``. ``cancelled``: stopped
    early on request (the list is what was found by then). ``listed`` /
    ``reused``: folders examined, and folders taken from the index."""

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.dates: dict[str, float] = {}
        self.truncated = False
        self.cancelled = False
        self.listed = 0
        self.reused = 0
        self.seconds = 0.0


# ---------------------------------------------------------------------------
# The remembered scan
# ---------------------------------------------------------------------------

INDEX_FORMAT = 1
ENV_INDEX_DIR = "OPENNMR_SCAN_INDEX_DIR"


def index_dir() -> str:
    """This computer's folder for remembered scans (not created here)."""
    if os.environ.get(ENV_INDEX_DIR):
        return os.environ[ENV_INDEX_DIR]
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
        return os.path.join(base, "openNMR", "cache", "scan_index")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Caches/openNMR/scan_index")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "openNMR", "scan_index")


def _root_key(root: str) -> str:
    return os.path.normcase(os.path.abspath(root))


class ScanIndex:
    """What a scan of one folder found, folder by folder, kept between
    sessions so the next scan only examines what changed.

    Each folder's record holds its modification time (``m``, in ns) and
    either the reader that recognised it (``v``, with ``t`` its
    acquisition time) or its entries in natural order (``e``: subfolders
    to enter, ``["d", name]``, and recognised files, ``["f", name,
    vendor, time]``). Keys are paths relative to the root."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.records: dict[str, dict] = {}
        #: The last complete scan's result, for showing at once.
        self.matches: list[tuple[str, str]] = []
        self.dates: dict[str, float] = {}
        self.truncated = False
        self.saved_at: float | None = None
        self._visited: set[str] = set()
        self._lock = threading.Lock()

    # -- storage -------------------------------------------------------

    @staticmethod
    def path_for(root: str) -> str:
        digest = hashlib.sha1(_root_key(root).encode("utf-8")).hexdigest()[:20]
        return os.path.join(index_dir(), f"{digest}.json")

    @classmethod
    def load(cls, root: str) -> "ScanIndex":
        """The remembered scan of ``root``, or an empty one (never raises:
        a missing or damaged index only means a full scan)."""
        index = cls(root)
        try:
            with open(cls.path_for(root), "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return index
        if not isinstance(data, dict) or data.get("format") != INDEX_FORMAT \
                or data.get("root") != _root_key(root):
            return index
        records = data.get("records")
        if isinstance(records, dict):
            index.records = records
        base = index.root
        for item in data.get("matches") or []:
            try:
                relative, vendor = item[0], item[1]
            except (TypeError, IndexError):
                continue
            path = os.path.normpath(os.path.join(base, relative))
            index.matches.append((path, vendor))
            if len(item) > 2 and isinstance(item[2], (int, float)):
                index.dates[path] = float(item[2])
        index.truncated = bool(data.get("truncated"))
        index.saved_at = data.get("saved_at")
        return index

    def save(self, result: ScanResult | None = None) -> None:
        """Write the index (atomically). With ``result`` from a complete
        scan, its matches are what is shown next time and folders that
        were not visited (deleted, moved) are dropped."""
        if result is not None and not result.cancelled:
            if not result.truncated:
                self.records = {k: v for k, v in self.records.items() if k in self._visited}
            self.matches = list(result)
            self.dates = dict(result.dates)
            self.truncated = result.truncated
        base = self.root
        data = {
            "format": INDEX_FORMAT,
            "root": _root_key(base),
            "saved_at": time.time(),
            "truncated": self.truncated,
            "matches": [[os.path.relpath(path, base), vendor]
                        + ([self.dates[path]] if path in self.dates else [])
                        for path, vendor in self.matches],
            "records": self.records,
        }
        path = self.path_for(base)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temporary = f"{path}.{os.getpid()}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(data, handle, separators=(",", ":"))
            os.replace(temporary, path)
            self.saved_at = data["saved_at"]
        except OSError:
            pass     # an index that cannot be written only costs a full scan next time

    @classmethod
    def forget(cls, root: str) -> None:
        """Delete the remembered scan of ``root``."""
        try:
            os.remove(cls.path_for(root))
        except OSError:
            pass

    # -- records -------------------------------------------------------

    def _key(self, path: str) -> str:
        relative = os.path.relpath(path, self.root)
        return "." if relative == "." else relative.replace(os.sep, "/")

    def get(self, path: str) -> dict | None:
        key = self._key(path)
        with self._lock:
            self._visited.add(key)
            return self.records.get(key)

    def put(self, path: str, record: dict) -> None:
        key = self._key(path)
        with self._lock:
            self._visited.add(key)
            self.records[key] = record


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _stat_of(entry):
    try:
        return entry.stat(follow_symlinks=False)
    except OSError:
        return None


def _detect_file(path: str, info=None) -> str | None:
    """The reader for a file, or None -- without ever opening a
    cloud-only one (:data:`CLOUD_EXTENSION_VENDORS`)."""
    from . import base
    ext = os.path.splitext(path)[1].lower()
    if ext not in FILE_EXTENSIONS:
        return None
    dataless = cloud.is_dataless_stat(info) if info is not None else cloud.is_dataless(path)
    if dataless:
        return CLOUD_EXTENSION_VENDORS.get(ext)
    try:
        return base.detect_vendor(path)
    except (ValueError, OSError):
        return None


def _is_ntuples_2d(entry, info) -> bool:
    ext = os.path.splitext(entry.name)[1].lower()
    if ext not in (".jdx", ".dx"):
        return False
    if info is not None and cloud.is_dataless_stat(info):
        return False     # not opened: the redundant raw_slices stay listed instead
    from . import xpulse
    return xpulse._is_ntuples_2d_file(entry.path)


def _experiment_time(path: str, names) -> float | None:
    for name in DATE_FILES:
        if name in names:
            try:
                return os.stat(os.path.join(path, name)).st_mtime
            except OSError:
                return None
    return None


def _list_dir(path: str, mtime_ns: int) -> dict | None:
    """Examine one folder: an experiment (its reader and date), or the
    entries to go on with. None when it cannot be listed."""
    from . import base
    try:
        with os.scandir(path) as iterator:
            entries = list(iterator)
    except OSError:
        return None
    names = {entry.name for entry in entries}
    record: dict = {"m": mtime_ns}

    def is_dir(entry) -> bool:
        try:
            return entry.is_dir(follow_symlinks=False)
        except OSError:
            return False

    if names & DIR_MARKERS or any(_ARRAY_DIR_RE.match(e.name) and is_dir(e) for e in entries):
        try:
            vendor = base.detect_vendor(path)
        except (ValueError, OSError):
            vendor = None
        if vendor is not None:
            record["v"] = vendor
            when = _experiment_time(path, names)
            if when is not None:
                record["t"] = when
            if "opennmr_derived" in names:
                record["derived"] = True
            if vendor == "spinsolve":
                from . import spinsolve
                record["pilots"] = [
                    entry.name for entry in sorted(entries, key=lambda e: natural_key(e.name))
                    if is_dir(entry) and spinsolve.is_spinsolve_standard(entry.path)]
            return record

    entries.sort(key=lambda e: natural_key(e.name))
    infos = {entry.name: _stat_of(entry) for entry in entries
             if not is_dir(entry) and os.path.splitext(entry.name)[1].lower() in FILE_EXTENSIONS}
    # A genuine X-Pulse 2D (an HSQC's "..._avg.jdx" NTUPLES file) already
    # holds what its sibling raw_slices/ folder has per increment.
    has_ntuples = any(_is_ntuples_2d(entry, infos.get(entry.name)) for entry in entries
                      if entry.name in infos)
    listed = []
    for entry in entries:
        if is_dir(entry):
            if entry.name in SKIP_DIRS or (has_ntuples and entry.name.lower() == "raw_slices"):
                continue
            listed.append(["d", entry.name])
            continue
        info = infos.get(entry.name)
        if info is None or XPULSE_RAW_SCAN_RE.search(entry.name):
            continue
        vendor = _detect_file(entry.path, info)
        if vendor is not None:
            listed.append(["f", entry.name, vendor, float(info.st_mtime)])
    record["e"] = listed
    return record


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------

def scan_for_spectra(directory: str, max_depth: int = MAX_DEPTH,
                     max_matches: int | None = None, *, index: ScanIndex | None = None,
                     cancel=None, progress=None, progress_interval: float = 0.25) -> ScanResult:
    """Every experiment any reader recognises under ``directory`` (or
    ``directory`` itself, if it is one), as ``(path, vendor)`` pairs in
    natural order.

    ``index``: a :class:`ScanIndex` to use and update -- folders it holds
    that have not changed are not examined again (save it afterwards
    with ``index.save(result)``). ``cancel``: a callable; when it returns
    True the scan stops and the result says ``cancelled``.
    ``progress(folders_done, found)``: called at most every
    ``progress_interval`` seconds, ``found`` being the result so far."""
    if max_matches is None:
        max_matches = SCAN_LIMIT
    started = time.monotonic()
    result = ScanResult()
    root = os.path.abspath(directory)
    last_report = [started]

    def report(force: bool = False) -> None:
        if progress is None:
            return
        now = time.monotonic()
        if force or now - last_report[0] >= progress_interval:
            last_report[0] = now
            progress(result.listed + result.reused, result)

    def add(path: str, vendor: str, when: float | None) -> bool:
        if len(result) >= max_matches:
            result.truncated = True
            return False
        result.append((path, vendor))
        if when is not None:
            result.dates[path] = when
        return True

    def walk(path: str, depth: int, mtime_ns: int | None) -> None:
        if cancel is not None and cancel():
            raise ScanCancelled
        if len(result) >= max_matches:
            result.truncated = True
            return
        if mtime_ns is None:
            if index is None:
                mtime_ns = 0          # nothing to compare it with
            else:
                try:
                    mtime_ns = os.stat(path).st_mtime_ns
                except OSError:
                    return
        record = index.get(path) if index is not None else None
        if record is not None and record.get("m") == mtime_ns:
            result.reused += 1
        else:
            record = _list_dir(path, mtime_ns)
            if record is None:
                return
            result.listed += 1
            if index is not None:
                index.put(path, record)
        report()

        vendor = record.get("v")
        if vendor:
            if not add(path, vendor, record.get("t")):
                return
            # Not entered -- except the spectra saved from it (Save
            # Progress on a slice, a CPMG sum ...), which live in its
            # opennmr_derived/ folder, and a Spinsolve 2D's 1D pilot scan.
            if record.get("derived") and depth < max_depth:
                walk(os.path.join(path, "opennmr_derived"), depth + 1, None)
            for name in record.get("pilots") or ():
                if not add(os.path.join(path, name), "spinsolve", None):
                    return
            return
        if depth >= max_depth:
            return
        for item in record.get("e") or ():
            if len(result) >= max_matches:
                result.truncated = True
                return
            if item[0] == "d":
                walk(os.path.join(path, item[1]), depth + 1, None)
            elif not add(os.path.join(path, item[1]), item[2],
                         item[3] if len(item) > 3 else None):
                return

    try:
        if os.path.isfile(root):
            # A file chosen directly: every reader may claim it (any
            # extension), unless it is cloud-only.
            from . import base
            if cloud.is_dataless(root):
                vendor = CLOUD_EXTENSION_VENDORS.get(os.path.splitext(root)[1].lower())
            else:
                try:
                    vendor = base.detect_vendor(root)
                except (ValueError, OSError):
                    vendor = None
            if vendor is not None:
                add(root, vendor, None)
        else:
            walk(root, 0, None)
    except ScanCancelled:
        result.cancelled = True
    result.seconds = time.monotonic() - started
    report(force=True)
    return result
