"""Write a file so that it is either the old version or the new one -- never
half of either.

A settings file written in place (``open(path, "w")`` then ``json.dump``)
is empty from the moment it is opened until the last byte is written. A
crash, a power cut or a value ``json`` cannot encode partway through leaves
it truncated, and the next start silently falls back to the defaults
(stability audit, 2026-11-01). Here the new content is

1. serialised *first*, in memory -- a ``TypeError`` from an unencodable value
   is raised before the file is touched;
2. written to a uniquely named temporary file in the same folder (same
   drive, so the rename below is atomic; unique, so two writers never share
   a temporary file), flushed and ``fsync``-ed;
3. renamed over the target with :func:`os.replace` -- atomic on Windows,
   macOS and Linux. On Windows a file that OneDrive, an antivirus scanner
   or a second openNMR has open for a moment makes the rename fail with
   ``PermissionError``; it is retried for up to :data:`RETRY_SECONDS`
   before giving up.

On any failure the temporary file is removed and the original is untouched.
Usable from notebooks as well as the GUI::

    from opennmr.core.atomic_write import write_json_atomic
    write_json_atomic("~/.opennmr/plot_style.json", {"font_size": 9})
"""

from __future__ import annotations

import json
import os
import tempfile
import time

#: How long a rename blocked by another program (Windows) is retried, s.
RETRY_SECONDS = 1.0
_RETRY_STEP = 0.05


def _replace_with_retry(src: str, dst: str) -> None:
    deadline = time.monotonic() + RETRY_SECONDS
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(_RETRY_STEP)


def write_text_atomic(path, text: str, encoding: str = "utf-8", newline: str | None = None) -> str:
    """Replace ``path`` with ``text`` atomically (see the module notes).
    Creates the folder if needed; returns the absolute path written."""
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", suffix=".tmp",
                               dir=folder)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline=newline) as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:          # some network/virtual drives refuse fsync
                pass
        if os.path.exists(path):
            try:                      # keep the target's permissions
                os.chmod(tmp, os.stat(path).st_mode & 0o7777)
            except OSError:
                pass
        _replace_with_retry(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def write_json_atomic(path, obj, *, trailing_newline: bool = False, **dump_kwargs) -> str:
    """``json.dump(obj, ...)`` to ``path`` atomically. ``dump_kwargs`` go to
    :func:`json.dumps` (``indent``, ``sort_keys``, ``default``, ...). The
    object is encoded before the file is touched, so an unencodable value
    raises ``TypeError``/``ValueError`` and leaves the old file as it was."""
    text = json.dumps(obj, **dump_kwargs)
    if trailing_newline:
        text += "\n"
    return write_text_atomic(path, text)
