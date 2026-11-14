"""Quietening printed output and collecting warnings for *one thread*.

``contextlib.redirect_stdout`` and ``warnings.catch_warnings`` swap
process-wide state. A fit running in the background (gui/jobs.py) and the
GUI thread using either at the same time can restore each other's state in
the wrong order -- leaving ``sys.stdout`` pointing at a dead buffer, or
``warnings.showwarning`` appending to a list nobody reads, for the rest of
the session. These two do the same jobs per thread:

* :func:`quiet_stdout` -- what this thread prints goes to a buffer (yielded);
  other threads' output is untouched.
* :func:`collect_warnings` -- the warnings this thread raises are collected
  in a list (yielded) instead of shown; other threads' are shown as usual.
"""

from __future__ import annotations

import io
import sys
import threading
import warnings
from contextlib import contextmanager

_LOCK = threading.RLock()
_LOCAL = threading.local()


class _ThreadStdout:
    """Stands in for ``sys.stdout`` while any thread is quietened: a
    thread with a buffer writes there, every other thread to the stream
    it replaced."""

    def __init__(self, target) -> None:
        self.target = target
        self.users = 0

    def _sink(self):
        stack = getattr(_LOCAL, "stdout", None)
        return stack[-1] if stack else self.target

    def write(self, text):
        return self._sink().write(text)

    def flush(self):
        sink = self._sink()
        flush = getattr(sink, "flush", None)
        if flush is not None:
            flush()

    def __getattr__(self, name):
        return getattr(self.target, name)


@contextmanager
def quiet_stdout():
    """Send what this thread prints to a buffer (yielded) for the block."""
    buffer = io.StringIO()
    with _LOCK:
        proxy = sys.stdout if isinstance(sys.stdout, _ThreadStdout) else None
        if proxy is None:
            proxy = _ThreadStdout(sys.stdout)
            sys.stdout = proxy
        proxy.users += 1
    stack = getattr(_LOCAL, "stdout", None)
    if stack is None:
        stack = _LOCAL.stdout = []
    stack.append(buffer)
    try:
        yield buffer
    finally:
        stack.pop()
        with _LOCK:
            proxy.users -= 1
            if proxy.users <= 0 and sys.stdout is proxy:
                sys.stdout = proxy.target


def _dispatching_showwarning(message, category, filename, lineno, file=None, line=None):
    sink = getattr(_LOCAL, "warnings", None)
    if sink:
        sink[-1].append(warnings.WarningMessage(message, category, filename, lineno, file, line))
        return None
    return _dispatching_showwarning.original(message, category, filename, lineno, file, line)


_dispatching_showwarning.original = warnings.showwarning


def _install_warning_hook() -> None:
    with _LOCK:
        current = warnings.showwarning
        if current is not _dispatching_showwarning:
            _dispatching_showwarning.original = current
            warnings.showwarning = _dispatching_showwarning


@contextmanager
def collect_warnings():
    """Collect the warnings this thread raises (a list of
    ``warnings.WarningMessage``, yielded) instead of showing them. The
    warning filters stay as they are (process-wide), so a warning the
    filters ignore, or one already shown once from the same place, is not
    collected again."""
    _install_warning_hook()
    caught: list = []
    stack = getattr(_LOCAL, "warnings", None)
    if stack is None:
        stack = _LOCAL.warnings = []
    stack.append(caught)
    try:
        yield caught
    finally:
        stack.pop()
