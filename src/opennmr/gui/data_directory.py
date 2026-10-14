"""
The default data folder (Settings ▸ Data; explicit user request,
2026-10-14): where every "open some NMR data" dialog starts, and what
the Open panel's browser shows at start-up. One app-wide value, set by
MainWindow from ``~/.opennmr/ui_settings.json`` (key ``data_folder``);
panels read it here so they need no reference to the window.

A folder that has gone missing (an unplugged drive, a renamed share)
counts as unset: the dialogs then start where they always did, rather
than failing or opening somewhere surprising.
"""

from __future__ import annotations

import os

_default: str = ""


def set_default_data_directory(path: str | None) -> str:
    """Set (``""``/None clears) the default data folder. Returns it,
    expanded; the value is kept even if the folder is missing now."""
    global _default
    _default = os.path.expanduser((path or "").strip())
    return _default


def stored_default_data_directory() -> str:
    """The folder as set, whether or not it exists right now."""
    return _default


def default_data_directory() -> str:
    """The default data folder if it is set and exists, else ``""``."""
    return _default if _default and os.path.isdir(_default) else ""


def start_directory(*preferred: str | None) -> str:
    """Where a data dialog should open: the default data folder when it
    is set and exists, else the first of ``preferred`` that is an
    existing folder (a file's own folder counts), else ``""`` (Qt's own
    choice). The default comes first by design: the user asked for the
    dialogs to *always* point there, and can still navigate away."""
    default = default_data_directory()
    if default:
        return default
    for path in preferred:
        if not path:
            continue
        path = str(path)
        if os.path.isdir(path):
            return path
        parent = os.path.dirname(path)
        if parent and os.path.isdir(parent):
            return parent
    return ""
