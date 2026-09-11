"""
Recognising data that is *listed* on this machine but not *on* it.

OneDrive, iCloud Drive and Dropbox all offer "files on demand": the
directory tree is fully present, every name and size is real, but a
file's contents live in the cloud until something opens it. The first
read then blocks -- for as long as the download takes, with no progress
and no way to cancel -- which is why this has to be detected rather
than discovered by stalling.

The specific thing that made this necessary: the Data tab's browser
calls :func:`opennmr.io.peek` on every experiment in a folder to
describe it ("1H, 2D, 1024x4096"). Peeking reads the small parameter
files, which is ~1 ms locally and effectively free -- but on a
cloud-backed folder it *materialises every one of them*, turning
"expand a folder" into hundreds of sequential downloads. Detection is
what lets the browser list such an experiment immediately, say plainly
that it is not downloaded, and leave the decision to the user.

**Detection never reads the file.** ``os.stat`` is safe on a
placeholder -- it reports the real size without triggering a download;
only reading the contents does that. Three signals, in order of
reliability:

* ``UF_DATALESS`` in ``st_flags`` -- macOS's own flag for exactly this
  state, set by the File Provider API that OneDrive, iCloud Drive and
  Dropbox all use on current macOS. Authoritative where it exists.
* **Allocated blocks of zero against a nonzero size.** The portable
  fallback, and what catches Linux/Windows sync clients that use sparse
  placeholders. A genuinely sparse local file would also match, which
  is acceptable: this is used to *defer* work, never to refuse it.
* **An iCloud ``.name.icloud`` sibling** -- older iCloud Drive evicts a
  file and leaves a small plist placeholder under a dotted name, so the
  original name does not exist on disk at all.

Nothing here refuses to open anything. A cloud file opens exactly as it
always did; the caller simply gets to know beforehand that doing so
means a download.
"""

from __future__ import annotations

import os
import stat as stat_module

#: macOS: "this file's contents are not present locally". Read from the
#: stdlib where available so a future Python that defines it wins, with
#: the literal as the fallback -- ``stat.UF_DATALESS`` is not exposed on
#: every version even though the kernel has long set the bit.
UF_DATALESS = getattr(stat_module, "UF_DATALESS", 0x40000000)

#: Parameter files worth checking before describing an experiment --
#: these are what :func:`opennmr.io.peek` reads. Deliberately not the
#: bulk data files: an experiment whose parameters are local can be
#: described instantly even when its `ser` is still in the cloud.
_PARAM_FILES = ("acqus", "acqu2s", "procs", "proc2s", "procpar")

#: The bulk data files -- what an actual load would have to download.
_DATA_FILES = ("ser", "fid", "2rr", "2ri", "1r", "1i", "data")


def is_dataless(path: str) -> bool:
    """Whether ``path`` is a cloud placeholder -- present as a name,
    absent as content.

    Never raises and never reads the file: a path that cannot be
    stat'd is reported as ``False`` (not dataless), because the honest
    answer to "should I avoid reading this" for something already
    unreachable is that the caller's own error handling should see the
    real failure, not a misleading "it's in the cloud"."""
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError:
        # An iCloud-evicted file's own name does not exist; the
        # placeholder sits beside it under a dotted name.
        directory, name = os.path.split(path)
        return os.path.exists(os.path.join(directory, f".{name}.icloud"))
    if getattr(info, "st_flags", 0) & UF_DATALESS:
        return True
    blocks = getattr(info, "st_blocks", None)
    return blocks == 0 and info.st_size > 0


def _matching_files(path: str, names: tuple[str, ...]) -> list[str]:
    """Members of ``names`` that exist under ``path``, or ``path``
    itself when it is a file (a single-file vendor's experiment)."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        return []
    found = [os.path.join(path, name) for name in names]
    return [candidate for candidate in found if os.path.exists(candidate)]


def parameters_are_local(path: str) -> bool:
    """Whether an experiment's *parameter* files are downloaded, i.e.
    whether :func:`opennmr.io.peek` can describe it without triggering
    a download.

    ``True`` when there is nothing to check -- an experiment with no
    recognised parameter file is not a cloud problem, it is a
    different problem, and the caller's normal error path should be
    the one that reports it."""
    return not any(is_dataless(candidate) for candidate in _matching_files(path, _PARAM_FILES))


def pending_download_bytes(path: str) -> int:
    """How many bytes opening ``path`` would have to download -- the
    total size of its not-yet-local data files, or ``0`` when it is
    fully present.

    Sizes are real for a placeholder, so this is exact rather than an
    estimate, and costs only a stat per file."""
    total = 0
    for candidate in _matching_files(path, _PARAM_FILES + _DATA_FILES):
        if not is_dataless(candidate):
            continue
        try:
            total += os.stat(candidate, follow_symlinks=False).st_size
        except OSError:
            continue
    return total


def format_bytes(count: int) -> str:
    """``1536`` -> ``"1.5 KB"``. For telling someone what a click is
    about to cost them."""
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" or size >= 100 else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
