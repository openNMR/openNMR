"""
Stored integrals -- a table of integrals the user chose to keep, across
spectra and across sessions.

A region integral (Integrate ▸ Integration Regions) is recomputed whenever
the regions change, so it is gone as soon as the next spectrum is selected.
This module keeps the ones you want: tick the regions, press *Store
integral in "Stored integrals"*, and each becomes a row with the region's
minimum / maximum / midpoint, its absolute integral (and error), the
experiment it came from and a note ("500 us contact time"). The table
lives in a file openNMR generates (``~/.opennmr/stored_integrals.json``)
and can be saved as CSV, JSON or plain text for your own use.

The same table is used from a notebook and from the GUI::

    from opennmr import notebook as nb

    store = nb.get_stored_integrals()               # the shared table
    store.add("sample A", low=-5.0, high=5.0, integral=1.23e6,
              error=2e3, notes="500 us contact time")
    store.export("cp_buildup.csv")                # .csv, .json or .txt
    store.rows()                                  # list of dicts

or straight from region-integration results::

    store.add_from_results("sample A", results, regions, which=[0, 2],
                           notes="1 ms contact")

Integrals are absolute: exactly what the integration rule returned for
the region on that spectrum, in the spectrum's intensity units -- not
normalised to the other regions.
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import os

__all__ = ["StoredIntegrals", "get_stored_integrals", "default_path", "COLUMNS", "EXPORT_FORMATS"]

#: ``(key, heading)`` of every column, in table order.
COLUMNS = (
    ("min", "Region min (ppm)"),
    ("max", "Region max (ppm)"),
    ("midpoint", "Midpoint (ppm)"),
    ("integral", "Integral"),
    ("error", "Error"),
    ("experiment", "Experiment"),
    ("notes", "Notes"),
    ("stored", "Stored"),
)

#: File formats :meth:`StoredIntegrals.export` writes.
EXPORT_FORMATS = ("csv", "json", "txt")


def default_path() -> str:
    """Where the table is kept: ``~/.opennmr/stored_integrals.json``, or
    the file named by the ``OPENNMR_STORED_INTEGRALS`` environment variable."""
    return os.environ.get("OPENNMR_STORED_INTEGRALS") or os.path.expanduser(
        "~/.opennmr/stored_integrals.json")


def _number(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value else None        # NaN -> None


class StoredIntegrals:
    """The stored-integrals table. ``path`` is its file (default
    :func:`default_path`); every change is written to it at once, so a
    crash loses nothing. ``path=False`` keeps the table in memory only."""

    def __init__(self, path=None) -> None:
        self.path = None if path is False else (path or default_path())
        self.records: list[dict] = []
        self.load()

    # ---- the internal file --------------------------------------------
    def load(self) -> None:
        """Re-read the internal file (an unreadable one is left alone and
        the table starts empty -- it is never overwritten by a failed read
        until something is stored)."""
        self.records = []
        if not self.path or not os.path.isfile(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as handle:
                data = json.load(handle)
            for entry in data.get("records", []):
                self.records.append(self._clean(entry))
        except (OSError, ValueError, AttributeError, TypeError):
            self.records = []

    def save(self) -> None:
        """Write the internal file."""
        if not self.path:
            return
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        payload = {"columns": [key for key, _ in COLUMNS], "records": self.records}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, self.path)

    # ---- the rows -------------------------------------------------------
    @staticmethod
    def _clean(entry: dict) -> dict:
        low, high = _number(entry.get("min")), _number(entry.get("max"))
        if low is not None and high is not None and low > high:
            low, high = high, low
        mid = _number(entry.get("midpoint"))
        if mid is None and low is not None and high is not None:
            mid = 0.5 * (low + high)
        return {"min": low, "max": high, "midpoint": mid,
                "integral": _number(entry.get("integral")), "error": _number(entry.get("error")),
                "experiment": str(entry.get("experiment") or ""),
                "notes": str(entry.get("notes") or ""),
                "stored": str(entry.get("stored") or "")}

    def __len__(self) -> int:
        return len(self.records)

    def rows(self) -> list[dict]:
        """Copies of the records, oldest first."""
        return [dict(record) for record in self.records]

    def add(self, experiment, low, high, integral, error=None, notes: str = "") -> dict:
        """Store one integral of ``low``..``high`` (ppm) on ``experiment``
        (a name). Returns the record."""
        if _number(low) is None or _number(high) is None:
            raise ValueError("A stored integral needs the region's low and high limits.")
        if _number(integral) is None:
            raise ValueError("A stored integral needs a number for the integral.")
        record = self._clean({
            "min": low, "max": high, "integral": integral, "error": error,
            "experiment": experiment, "notes": notes,
            "stored": _dt.datetime.now().isoformat(timespec="seconds")})
        self.records.append(record)
        self.save()
        return dict(record)

    def add_from_results(self, experiment, results, regions, which=None, notes: str = "") -> list[dict]:
        """Store the integrals of the regions ``which`` (indices into
        ``regions``; default all) from region-integration ``results`` (one
        dict per slice, each with ``"integrals"`` and optionally
        ``"integral_errors"``). A series gives one row per slice, the
        experiment named ``"<experiment> · slice k"``. Returns the records."""
        regions = list(regions)
        which = list(range(len(regions))) if which is None else [int(k) for k in which]
        results = list(results)
        added = []
        for position, entry in enumerate(results):
            integrals = entry.get("integrals")
            if integrals is None:
                continue
            errors = entry.get("integral_errors")
            name = str(experiment) if len(results) == 1 else f"{experiment} · slice {position + 1}"
            for k in which:
                if not 0 <= k < len(regions) or k >= len(integrals):
                    continue
                low, high = regions[k]
                err = errors[k] if errors is not None and k < len(errors) else None
                added.append(self._add_quiet(name, low, high, integrals[k], err, notes))
        if added:
            self.save()
        return [dict(r) for r in added]

    def _add_quiet(self, experiment, low, high, integral, error, notes) -> dict:
        if _number(integral) is None:
            raise ValueError("A stored integral needs a number for the integral.")
        record = self._clean({
            "min": low, "max": high, "integral": integral, "error": error,
            "experiment": experiment, "notes": notes,
            "stored": _dt.datetime.now().isoformat(timespec="seconds")})
        self.records.append(record)
        return record

    def set_notes(self, row: int, text: str) -> None:
        self.records[row]["notes"] = str(text)
        self.save()

    def remove(self, rows) -> None:
        """Delete the rows at these indices."""
        drop = {int(r) for r in rows}
        self.records = [r for i, r in enumerate(self.records) if i not in drop]
        self.save()

    def clear(self) -> None:
        self.records = []
        self.save()

    # ---- leaving the program -------------------------------------------
    @staticmethod
    def _cell(key: str, value) -> str:
        if value is None:
            return ""
        if key in ("min", "max", "midpoint"):
            return f"{value:.6g}"
        if key in ("integral", "error"):
            return f"{value:.8g}"
        return str(value)

    def table(self) -> tuple[list[str], list[list[str]]]:
        """``(headings, rows)`` of text, as shown."""
        headings = [heading for _, heading in COLUMNS]
        rows = [[self._cell(key, record.get(key)) for key, _ in COLUMNS] for record in self.records]
        return headings, rows

    def to_csv(self) -> str:
        headings, rows = self.table()
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(headings)
        writer.writerows(rows)
        return buffer.getvalue()

    def to_json(self) -> str:
        return json.dumps({"columns": [{"key": key, "heading": heading} for key, heading in COLUMNS],
                           "records": self.records}, indent=2)

    def to_txt(self) -> str:
        headings, rows = self.table()
        widths = [max(len(headings[c]), *(len(r[c]) for r in rows)) if rows else len(headings[c])
                  for c in range(len(headings))]
        lines = ["  ".join(h.ljust(w) for h, w in zip(headings, widths)).rstrip()]
        lines.append("  ".join("-" * w for w in widths))
        lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in rows]
        return "\n".join(lines) + "\n"

    def export(self, path, fmt: str | None = None) -> str:
        """Write the table to ``path`` as CSV, JSON or plain text (``fmt``,
        or the file's extension). Returns the format used."""
        path = os.fspath(path)
        fmt = (fmt or os.path.splitext(path)[1].lstrip(".") or "csv").lower()
        if fmt not in EXPORT_FORMATS:
            raise ValueError(f"Unknown format {fmt!r}; use one of {', '.join(EXPORT_FORMATS)}.")
        text = {"csv": self.to_csv, "json": self.to_json, "txt": self.to_txt}[fmt]()
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return fmt


_SHARED: dict[str, StoredIntegrals] = {}


def get_stored_integrals(path=None) -> StoredIntegrals:
    """The shared table for ``path`` (default :func:`default_path`) -- the
    one the GUI shows, so a notebook and the GUI add to the same rows."""
    key = os.fspath(path) if path else default_path()
    store = _SHARED.get(key)
    if store is None:
        store = _SHARED[key] = StoredIntegrals(key)
    return store
