"""
Stage reports from inside a read -- what a long load is doing right now.

A reader calls :func:`stage` at each step that takes noticeable time
("Reading ser (244 MB)", "Fourier transform (F2)"). Outside a
:func:`reporting` block that is a no-op, so the notebook and the tests
pay nothing. Inside one -- the GUI's background reader -- the callback
receives the text, and may raise :class:`LoadCancelled` to stop the read
*between* stages: interrupting a running transform is not possible, but
not starting the next one is.

The reporter is per thread, so a read on the background thread and
anything on the GUI thread never see each other's callback.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable

_local = threading.local()


class LoadCancelled(Exception):
    """Raised by a stage callback to abandon a read the user cancelled."""


@contextmanager
def reporting(callback: Callable[[str], None]):
    """Send this thread's :func:`stage` reports to ``callback`` while inside."""
    previous = getattr(_local, "callback", None)
    _local.callback = callback
    try:
        yield
    finally:
        _local.callback = previous


def stage(text: str) -> None:
    """Report that the read is now doing ``text``. May raise
    :class:`LoadCancelled` (from the callback) to stop the read."""
    callback = getattr(_local, "callback", None)
    if callback is not None:
        callback(text)


def megabytes(path: str) -> str:
    """``"(244 MB)"`` for a file, or ``""`` if its size cannot be read."""
    import os
    try:
        return f"({os.path.getsize(path) / 1e6:.0f} MB)"
    except OSError:
        return ""
