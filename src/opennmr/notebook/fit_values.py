"""
Fitted values, saved in a form anyone can read: a JSON file holding a
table (rows = sites/peaks, columns = parameters) of every fitted value
with its 1σ uncertainty, whether it was held fixed, the fit's quality and
where it came from.

Explicit user request (2026-10-08): "have a button to say 'Save the
fitted values' to store the fit values (and errors) into a
human-readable format (e.g., json) -- ideally some kind of table or
dictionary that is easy to read by anyone."

The file looks like::

    {
      "openNMR fitted values": 1,
      "fit": "Solid-state NMR",
      "spectrum": "sample3/10",
      "saved": "2026-10-08T20:41:07",
      "quality": {"r2": 0.9991, "reduced_chi2": 1.04, ...},
      "columns": ["δiso (ppm)", "C_Q (MHz)", ...],
      "values": {
        "Al1": {"δiso (ppm)": {"value": 12.01, "error": 0.03, "fixed": false}, ...},
        ...
      },
      "as shown": {"Al1": {"δiso (ppm)": "12.01 ± 0.03", ...}, ...},
      "summary": "...", "report": "..."
    }

``values`` holds the numbers -- exact (full precision) where the fit's
own result is available, else read from the table as shown; ``as shown``
is the table exactly as the GUI displayed it.

Notebook::

    from opennmr.notebook import fit_values
    fit = fit_combined(results, sites, ...)
    fit_values.save_fitted_values(fit, "al_fit_values.json", spectrum="27Al MAS")
    fit_values.load_fitted_values("al_fit_values.json")["values"]["Al1"]
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re

import numpy as np
from ..core.atomic_write import write_json_atomic

FORMAT_KEY = "openNMR fitted values"
FORMAT_VERSION = 1

#: Column headings for a fit result's own parameter names.
PARAMETER_HEADINGS = {
    "delta_iso": "δiso (ppm)", "amount": "Amount", "fwhm": "FWHM (ppm)", "eta": "L fraction",
    "aniso": "Δ or span (ppm)", "asym": "η or skew", "alpha": "α (°)", "beta": "β (°)",
    "gamma": "γ (°)", "C_Q": "C_Q (MHz)", "eta_Q": "η_Q", "mas_Hz": "MAS rate (Hz)",
    "b0": "baseline b0", "b1": "baseline b1",
    "share_%": "Share (%)", "P_Q": "P_Q (MHz)", "delta_cg": "CT c.o.g. (ppm)",
}
#: The row for what belongs to the whole spectrum (MAS rate, baseline).
SPECTRUM_ROW = "(spectrum)"

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_VALUE_ERROR = re.compile(rf"^\s*({_NUMBER})\s*(?:(?:±|\+/-|\+-)\s*({_NUMBER}))?\s*(.*?)\s*$",
                          re.DOTALL)


def heading(name: str) -> str:
    """A parameter's column heading (``"J12"`` -> ``"J12 (Hz)"``)."""
    if name in PARAMETER_HEADINGS:
        return PARAMETER_HEADINGS[name]
    return f"{name} (Hz)" if name.startswith("J") else name


def _number(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_value_error(text) -> dict:
    """``"12.01 ± 0.03"`` -> ``{"value": 12.01, "error": 0.03, "fixed": False}``;
    ``"0.7 (fixed)"`` -> fixed, no error; anything not a number keeps its
    text (``{"value": None, ..., "text": ...}``)."""
    text = "" if text is None else str(text).replace("−", "-").replace("\n", " ").strip()
    fixed = "(fixed)" in text
    m = _VALUE_ERROR.match(text.replace("(fixed)", ""))
    if not text or m is None:
        return {"value": None, "error": None, "fixed": fixed, "text": text}
    out = {"value": float(m.group(1)), "error": float(m.group(2)) if m.group(2) else None,
           "fixed": fixed}
    rest = m.group(3).strip()
    if rest:
        out["note"] = rest
    return out


def table_values(columns, rows) -> tuple[dict, dict]:
    """``rows`` = ``[(row heading, [cell text per column]), ...]`` ->
    ``(values, as_shown)``: both ``{row: {column: ...}}``, the first
    parsed into numbers, the second the text as displayed. Blank cells
    are left out."""
    values, shown = {}, {}
    for row, cells in rows:
        row = str(row)
        values[row], shown[row] = {}, {}
        for column, cell in zip(columns, cells):
            if cell is None or str(cell).strip() in ("", "—", "-"):
                continue
            shown[row][str(column)] = str(cell)
            values[row][str(column)] = parse_value_error(cell)
    return values, shown


def parameter_values(result: dict) -> dict:
    """Exact values ``{row: {column: {"value", "error", "fixed"}}}`` from a
    fit result that lists its parameters (``result["parameters"]``: dicts
    with ``label``, ``name``, ``value``, ``error``, ``free`` -- the
    Solid-state NMR, Czjzek and multi-spectrum fits) plus its
    derived quantities (``result["derived"]``: share, P_Q, CT c.o.g.).
    Empty if the result has neither."""
    out = {}
    params = result.get("parameters")
    for p in params if isinstance(params, (list, tuple)) else []:
        if not isinstance(p, dict) or "name" not in p:
            continue
        row = p.get("label") or SPECTRUM_ROW
        out.setdefault(str(row), {})[heading(p["name"])] = {
            "value": _number(p.get("value")),
            "error": None if not p.get("free", True) else _number(p.get("error")),
            "fixed": not bool(p.get("free", True))}
    derived = result.get("derived")
    for site, d in (derived if isinstance(derived, dict) else {}).items():
        if not isinstance(d, dict):
            continue
        for key in ("share_%", "P_Q", "delta_cg"):
            if key in d:
                out.setdefault(str(site), {})[heading(key)] = {
                    "value": _number(d.get(key)), "error": _number(d.get(f"{key}_err")),
                    "fixed": False}
    return out


def _quality(result) -> dict:
    q = (result or {}).get("quality") if isinstance(result, dict) else None
    if not isinstance(q, dict):
        return {}
    return {k: _number(v) for k, v in q.items() if _number(v) is not None}


def fitted_values_record(*, fit: str = "", spectrum: str = "", source: str = "",
                         columns=None, rows=None, result: dict | None = None,
                         summary: str = "", report: str = "", when=None) -> dict:
    """The record :func:`save_fitted_values` writes. Give the table as
    shown (``columns`` + ``rows``, see :func:`table_values`) and/or the
    fit ``result``; the result's own numbers (full precision) take the
    place of the table's rounded ones wherever both exist."""
    values, shown = table_values(columns or [], rows or [])
    exact = parameter_values(result or {}) if isinstance(result, dict) else {}
    for row, cols in exact.items():
        values.setdefault(row, {}).update(cols)
    all_columns = list(columns or [])
    for cols in values.values():
        for c in cols:
            if c not in all_columns:
                all_columns.append(c)
    record = {
        FORMAT_KEY: FORMAT_VERSION,
        "fit": str(fit), "spectrum": str(spectrum),
        "saved": (when or _dt.datetime.now()).isoformat(timespec="seconds"),
        "columns": all_columns,
        "values": values,
        "values are": ("full precision from the fit itself" if exact else
                       "as shown in the table (rounded to the digits the uncertainty supports)"),
        "errors are": "1σ (one standard deviation); null where the value was held fixed",
    }
    if source:
        record["source"] = str(source)
    quality = _quality(result)
    if quality:
        record["quality"] = quality
    if shown and any(shown.values()):
        record["as shown"] = shown
    if summary:
        record["summary"] = str(summary)
    if report:
        record["report"] = str(report)
    return record


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def save_fitted_values(record_or_result, path, **kwargs) -> str:
    """Write the fitted values as indented, UTF-8 JSON and return the path.

    ``record_or_result`` : a record from :func:`fitted_values_record`, or a
    fit result (e.g. from ``fit_combined``) -- then ``kwargs`` go to
    :func:`fitted_values_record` (``fit=``, ``spectrum=``...)."""
    record = record_or_result
    if not (isinstance(record, dict) and FORMAT_KEY in record):
        record = fitted_values_record(result=record_or_result, **kwargs)
    path = os.path.expanduser(os.fspath(path))
    if not path.lower().endswith(".json"):
        path += ".json"
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    write_json_atomic(path, record, trailing_newline=True, indent=2, ensure_ascii=False,
                      default=_json_default)
    return path


def load_fitted_values(path) -> dict:
    with open(os.path.expanduser(os.fspath(path)), encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict) or FORMAT_KEY not in record:
        raise ValueError(f"{path} is not an openNMR fitted-values file.")
    return record


def safe_file_stem(text: str) -> str:
    """A file-name-safe version of ``text``."""
    stem = re.sub(r'[\\/:*?"<>|]+', "_", str(text)).strip().strip(".")
    return re.sub(r"\s+", " ", stem)[:120] or "fitted values"
