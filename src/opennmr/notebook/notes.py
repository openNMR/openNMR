"""
Experiment titles and notes -- one editable piece of free text per
experiment, whatever format it came from.

* **Bruker**: TopSpin's own title, ``<expno>/pdata/<procno>/title``
  (what ``edti`` edits). Read from and written back to that file, so
  TopSpin and openNMR always show the same text. The first overwrite
  keeps the original as ``title.orig`` (see
  :func:`opennmr.io.bruker.write_title`).
* **Everything else**: the ``notes`` entry of the experiment's
  ``opennmr_metadata.json`` sidecar. Until something is saved there,
  the format's own title is shown when it has one -- JEOL's header
  title and comment, Varian/OpenVnmrJ's ``text`` file -- and it is
  only copied to the sidecar when edited; those vendor files are never
  written.
* **A spectrum with nothing on disk** (a simulation, a derived
  spectrum): kept in ``metadata.raw["notes"]`` for the session only.

Usable the same way from a notebook and from the GUI::

    from opennmr import notebook as nb
    nb.experiment_notes(spectrum)          # -> str
    nb.set_experiment_notes(spectrum, "sample 3, after 2 h at 80 C")
    nb.notes_location(spectrum)            # where that went
"""

from __future__ import annotations

import os

from ..io import bruker as io_bruker
from ..io import persistence as io_persistence

__all__ = ["experiment_notes", "set_experiment_notes", "notes_location", "notes_first_line"]


def _on_disk(meta) -> bool:
    return bool(meta.source_path) and meta.vendor != "derived" and os.path.exists(meta.source_path)


def _is_bruker(meta) -> bool:
    return meta.vendor == "bruker" and _on_disk(meta) and os.path.isdir(meta.source_path)


def _procno(meta) -> int:
    try:
        return int(meta.procno or 1)
    except (TypeError, ValueError):
        return 1


def _vendor_title(meta) -> str:
    """The title the format itself records (read-only), or ``""``."""
    raw = meta.raw or {}
    if meta.vendor == "jeol":
        parts = [str(raw.get(key) or "").strip() for key in ("title", "comment")]
        return "\n".join(part for part in parts if part)
    if meta.vendor == "varian" and _on_disk(meta):
        folder = meta.source_path if os.path.isdir(meta.source_path) else os.path.dirname(meta.source_path)
        try:
            with open(os.path.join(folder, "text"), "rb") as f:
                return f.read().decode("utf-8", errors="replace").replace("\r\n", "\n").strip()
        except OSError:
            return ""
    return ""


def experiment_notes(spectrum) -> str:
    """The title/notes of ``spectrum``'s experiment (``""`` if none).
    Read from disk every call, so an edit made in TopSpin shows up."""
    meta = spectrum.metadata
    if _is_bruker(meta):
        return io_bruker.read_title(meta.source_path, _procno(meta)) or ""
    if _on_disk(meta):
        saved = io_persistence.load_notes_sidecar(meta.source_path)
        return saved if saved is not None else _vendor_title(meta)
    return str((meta.raw or {}).get("notes") or "")


def notes_location(spectrum) -> str:
    """Where :func:`set_experiment_notes` stores ``spectrum``'s notes,
    in words -- a path, or "this session only"."""
    meta = spectrum.metadata
    if _is_bruker(meta):
        return io_bruker.title_path(meta.source_path, _procno(meta))
    if _on_disk(meta):
        return io_persistence.metadata_sidecar_path(meta.source_path)
    return "this session only (nothing on disk)"


def set_experiment_notes(spectrum, text: str) -> str:
    """Save ``text`` as ``spectrum``'s title/notes (see the module
    docstring for where) and return :func:`notes_location`. Line endings
    are normalised and trailing whitespace dropped; raises ``OSError``
    if the file cannot be written (e.g. a read-only data folder)."""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").rstrip()
    meta = spectrum.metadata
    if _is_bruker(meta):
        return io_bruker.write_title(meta.source_path, _procno(meta), text)
    if _on_disk(meta):
        return io_persistence.save_notes_sidecar(meta.source_path, text)
    meta.raw["notes"] = text
    return notes_location(spectrum)


def notes_first_line(text: str, limit: int = 80) -> str:
    """The first non-blank line of ``text``, cut to ``limit`` characters
    -- a one-line summary for a tooltip or legend."""
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= limit else line[: limit - 1] + "…"
    return ""
