"""
One shape for every fit result: a tidy table with one row per parameter.

DOSY/SCORE components, a curve fit (T1, T2, D ...), a spectral fit of
peaks, and the CSA / quadrupolar / Czjzek / combined fits used to export
four different layouts: wide tables with ``X err`` columns, a header block
plus a component table, a long ``Site, Parameter, Value`` table. Results
from different routes could not be concatenated, compared or plotted with
one script, and the kind of uncertainty (item 3a) was stated in one
layout and absent from another.

Every adapter here gives the same rows, with the same columns:

==============  ==========================================================
``fit_type``    what kind of fit made it: ``curve``, ``components``,
                ``spectral``, or the stored fit's own name (``CSA`` ...)
``spectrum``    the dataset it was measured on
``fit``         the model / method (``biexponential``, ``SCORE`` ...)
``group``       the peak, site or component the parameter belongs to
``slice``       the slice (row) of a pseudo-2D, empty otherwise
``parameter``   an ASCII-safe name (``T1``, ``D1``, ``shift``, ``delta_iso``)
``value``       full precision, SI-style raw units
``error``       1σ, empty when held fixed or not estimated
``error_kind``  one of :data:`ERROR_KINDS`: what that ± *is*
``unit``        ``s``, ``m^2/s``, ``ppm``, ``Hz`` ... empty if none
``fixed``       1 if held fixed, else 0
``shared``      1 if one value was shared by several groups (a global fit)
``r_squared``, ``reduced_chi2``, ``rmse``, ``n_points``  the fit's quality
==============  ==========================================================

Notebook::

    from opennmr.notebook import results_table as rt
    rows = rt.curve_fit_rows(fits, spectrum="sample3")
    rows += rt.components_rows(score_result, spectrum="sample3")
    rt.write_results_csv(rows, "results.csv")
    rt.read_results_csv("results.csv")          # round-trips, numbers as floats
    rt.concat_results(rows_a, rows_b)          # one table from several routes

The GUI's "Export Results Table…" writes the same thing.
"""

from __future__ import annotations

import csv
import math
import os
import re

FORMAT_NAME = "openNMR results table"
FORMAT_VERSION = 1

COLUMNS = ("fit_type", "spectrum", "fit", "group", "slice", "parameter", "value", "error",
           "error_kind", "unit", "fixed", "shared", "r_squared", "reduced_chi2", "rmse",
           "n_points")

#: What a ``±`` is. The same dataset can give a 2 % error one way and 8 % the
#: other, so a table that does not say which it holds cannot be compared.
ERROR_KINDS = {
    "measured": "1σ propagated from the per-point measurement errors supplied with the data",
    "measured_inflated": "1σ from the measured errors scaled by sqrt(max(1, reduced χ²))",
    "scatter": "1σ from the scatter of the data about the fit (assumes the model is right)",
    "covariance": "1σ from the covariance matrix of the least-squares fit",
    "none": "not estimated",
}

_NUMERIC = ("value", "error", "r_squared", "reduced_chi2", "rmse")
_INTEGER = ("slice", "n_points", "fixed", "shared")

_TIME_PARAMS = frozenset({"T1", "T2", "T1rho", "T1a", "T1b", "T1c", "T2a", "T2b", "T2c",
                          "tau", "tau_c"})
_DIFFUSION = re.compile(r"^D\d*$|^D[ab]$")


def _number(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parameter_unit(name: str) -> str:
    """The raw unit of a curve-fit parameter (``T1`` -> ``s``, ``D2`` ->
    ``m^2/s``), or ``""``; the same rule the curve-fit CSV uses."""
    name = str(name)
    if name in _TIME_PARAMS or re.fullmatch(r"T[12]\w*", name):
        return "s"
    if _DIFFUSION.match(name):
        return "m^2/s"
    return ""


def error_kind_of(uncertainty) -> str:
    """The :data:`ERROR_KINDS` key of a curve fit's ``uncertainty`` block
    (``"none"`` when it has none)."""
    if isinstance(uncertainty, dict):
        kind = str(uncertainty.get("kind", "") or "")
        if kind in ERROR_KINDS:
            return kind
    return "none"


def make_row(**fields) -> dict:
    """One row of the table with every column present (``None`` where not
    given); unknown keys are refused rather than silently dropped."""
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise KeyError(f"not a results-table column: {sorted(unknown)}")
    row = {c: None for c in COLUMNS}
    row.update(fields)
    for key in _NUMERIC:
        row[key] = _number(row[key])
    for key in _INTEGER:
        if row[key] is not None and row[key] != "":
            row[key] = int(row[key])
    row["fixed"] = int(bool(row["fixed"]))
    row["shared"] = int(bool(row["shared"]))
    if row["fixed"]:
        row["error"] = None
    if row["error"] is None:
        row["error_kind"] = "none"          # a ± that is absent has no kind
    elif row["error_kind"] is None:
        row["error_kind"] = "covariance"
    for key in ("fit_type", "spectrum", "fit", "group", "parameter", "unit"):
        row[key] = "" if row[key] is None else str(row[key])
    return row


def _quality_fields(quality) -> dict:
    q = quality if isinstance(quality, dict) else {}
    return {"r_squared": q.get("r_squared", q.get("r2")), "reduced_chi2": q.get("reduced_chi2"),
            "rmse": q.get("rmse"), "n_points": q.get("n_points", q.get("n"))}


def curve_fit_rows(results, spectrum: str = "", slice_index=None) -> list[dict]:
    """The rows of a curve fit (the list ``NMR2D_fit_relaxation`` returns,
    one entry per peak): every parameter of every peak."""
    rows = []
    for index, result in enumerate(results or [], start=1):
        group = result.get("peak_label") or result.get("label") or f"peak {index}"
        quality = _quality_fields(result.get("fit_quality"))
        kind = error_kind_of(result.get("uncertainty"))
        shared = result.get("shared") or []
        shared = set(shared if not isinstance(shared, dict) else shared.keys())
        fixed = result.get("fixed") or {}
        errors = result.get("errors") or {}
        for name, value in (result.get("params") or {}).items():
            is_fixed = name in fixed
            rows.append(make_row(
                fit_type="curve", spectrum=spectrum, fit=result.get("model", ""), group=group,
                slice=slice_index, parameter=name, value=value, error=errors.get(name),
                error_kind=kind if errors.get(name) is not None else "none",
                unit=parameter_unit(name), fixed=is_fixed, shared=name in shared, **quality))
    return rows


def components_rows(result: dict, spectrum: str = "") -> list[dict]:
    """The rows of a SCORE / DOSY-style component resolution: each
    component's decay constant (``D1``, ``T1`` ... with the same names and
    units a curve fit uses) and its share of the signal. These methods give
    no uncertainty, which is said (``error_kind`` ``"none"``) rather than
    left blank."""
    kind = result.get("kind", "diffusion")
    prefix, unit = ("D", "m^2/s") if kind == "diffusion" else ("T", "s")
    method = str(result.get("method", "score")).upper()
    quality = {"r_squared": result.get("r_squared")}
    rows = []
    for k in range(int(result["n_components"])):
        group = f"component {k + 1}"
        common = dict(fit_type="components", spectrum=spectrum, fit=method, group=group,
                      error=None, error_kind="none", **quality)
        rows.append(make_row(parameter=f"{prefix}{k + 1}", value=result["constants"][k],
                             unit=unit, **common))
        rows.append(make_row(parameter="share", value=result["fractions"][k] * 100.0,
                             unit="%", **common))
    return rows


def spectral_fit_rows(slice_results, spectrum: str = "") -> list[dict]:
    """The rows of a peak fit applied to one spectrum or every slice of a
    series: shift, fwhm, amplitude (and eta, J) of every peak."""
    units = {"shift": "ppm", "fwhm": "ppm"}
    rows = []
    if not any("popt" in r for r in slice_results or []):
        return rows
    from ..processing import fitting as _fitting
    for k, result in enumerate(slice_results or []):
        if "popt" not in result:
            continue
        per_peak = _fitting.extract_params_and_errors(
            result["popt"], result["pcov"], result["peak_defs"], result["is_free"])
        slice_index = result.get("index", k + 1) if len(slice_results) > 1 else None
        quality = _quality_fields({"r_squared": result.get("r_squared"),
                                   "reduced_chi2": result.get("reduced_chi2"),
                                   "rmse": result.get("rmse")})
        for p, peak in enumerate(per_peak, start=1):
            for name, value, error, fixed in zip(peak["names"], peak["params"],
                                                 peak["errors"], peak["fixed"]):
                rows.append(make_row(
                    fit_type="spectral", spectrum=spectrum, fit=peak["type"], group=f"peak {p}",
                    slice=slice_index, parameter=name, value=value, error=error,
                    error_kind="covariance", unit="Hz" if name.startswith("J") else units.get(name, ""),
                    fixed=bool(fixed), **quality))
    return rows


_HEADING = re.compile(r"^\s*(.*?)\s*\(([^()]*)\)\s*$")
_SLICE_ROW = re.compile(r"^\s*Slice\s+(\d+)\s*,\s*(.*)$", re.IGNORECASE)
_ASCII = {"δiso": "delta_iso", "Δ or span": "aniso", "η or skew": "asym", "L fraction": "eta",
          "η_Q": "eta_Q", "α": "alpha", "β": "beta", "γ": "gamma", "Share": "share",
          "CT c.o.g.": "delta_cg", "R²": "r_squared", "baseline b0": "b0", "baseline b1": "b1",
          "MAS rate": "mas_Hz"}


def split_heading(heading: str) -> tuple[str, str]:
    """``"δiso (ppm)"`` -> ``("delta_iso", "ppm")``; a heading with no
    bracketed unit has an empty one. Names are ASCII so a script can use
    them as identifiers."""
    heading = str(heading)
    m = _HEADING.match(heading)
    base, unit = (m.group(1), m.group(2)) if m else (heading.strip(), "")
    base = _ASCII.get(base, base)
    if re.fullmatch(r"J\d+", base) and not unit:
        unit = "Hz"
    base = re.sub(r"[^\w]+", "_", base.replace("²", "2")).strip("_") or "value"
    return base, unit


def record_rows(record: dict, fit_type: str | None = None) -> list[dict]:
    """The rows of any stored fit's values record
    (:func:`opennmr.notebook.fit_values.fitted_values_record` /
    ``values_record``): the CSA, quadrupolar, Czjzek, combined, multi-
    spectrum and kept-model fits, whatever columns they have."""
    from . import fit_values as _fv

    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
    qfields = _quality_fields(quality)
    rows = []
    for cell in _fv.parameter_rows(record):
        group = cell["site"]
        slice_index = None
        m = _SLICE_ROW.match(group)
        if m:
            slice_index, group = int(m.group(1)), m.group(2)
        name, unit = split_heading(cell["parameter"])
        if cell["value"] is None:
            continue                       # a text cell is not a number
        rows.append(make_row(
            fit_type=fit_type or "stored", spectrum=record.get("spectrum", ""),
            fit=record.get("fit", ""), group=group, slice=slice_index, parameter=name,
            value=cell["value"], error=cell["error"], error_kind="covariance",
            unit=unit, fixed=cell["fixed"], **qfields))
    return rows


def concat_results(*tables) -> list[dict]:
    """Several row lists as one table (the same columns throughout, so this
    is a plain concatenation, checked)."""
    out = []
    for table in tables:
        for row in table:
            if set(row) != set(COLUMNS):
                raise ValueError("not a results-table row: columns " + ", ".join(sorted(set(row) ^ set(COLUMNS))))
            out.append(row)
    return out


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def results_to_csv_text(rows, notes: str = "") -> str:
    """The table as CSV text: ``#`` comment lines (format, version, the
    meaning of each error kind that occurs), the header, then the rows."""
    from io import StringIO

    rows = concat_results(rows)
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([f"# {FORMAT_NAME} v{FORMAT_VERSION}",
                     "one row per parameter; error is 1σ, empty if fixed or not estimated"])
    for kind in dict.fromkeys(r["error_kind"] for r in rows if r["error_kind"] != "none"):
        writer.writerow([f"# error_kind {kind}", ERROR_KINDS[kind]])
    if notes:
        writer.writerow([f"# {notes}"])
    writer.writerow(COLUMNS)
    for row in rows:
        writer.writerow([_cell(row[c]) for c in COLUMNS])
    return buffer.getvalue()


def write_results_csv(rows, path, notes: str = "") -> str:
    """Write the table (UTF-8 with a byte-order mark so Excel reads σ) and
    return the path."""
    path = os.path.expanduser(os.fspath(path))
    if not path.lower().endswith(".csv"):
        path += ".csv"
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        handle.write(results_to_csv_text(rows, notes))
    return path


def read_results_csv(path) -> list[dict]:
    """Read a file :func:`write_results_csv` wrote back into rows (numbers as
    numbers; comment lines skipped)."""
    with open(os.path.expanduser(os.fspath(path)), newline="", encoding="utf-8-sig") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    out = []
    for raw in csv.DictReader(lines):
        if tuple(raw) != COLUMNS:
            raise ValueError("not an openNMR results table: columns " + ", ".join(raw))
        row = {}
        for c in COLUMNS:
            text = raw[c]
            if c in _NUMERIC:
                row[c] = float(text) if text != "" else None
            elif c in _INTEGER:
                row[c] = int(float(text)) if text != "" else None
            else:
                row[c] = text
        row["fixed"], row["shared"] = int(row["fixed"] or 0), int(row["shared"] or 0)
        out.append(row)
    return out


def table_for_spectrum(spectrum, label: str = "", score_result=None) -> list[dict]:
    """Every fit result held on a spectrum as one table: its peak fit(s),
    its curve fit, and its SCORE / DOSY components result (the one stored
    on the spectrum, unless ``score_result`` is given).

    Stored panel fits (CSA, quadrupolar ...) are added by the GUI, which
    holds their records; in a notebook build them with :func:`record_rows`."""
    raw = spectrum.metadata.raw or {}
    label = label or str(getattr(spectrum.metadata, "title", "") or "")
    rows = spectral_fit_rows([r for r in raw.get("slice_fit_results") or []
                              if isinstance(r, dict) and "popt" in r], label)
    rows += curve_fit_rows(raw.get("relaxation_fit_results"), label)
    score_result = score_result or raw.get("score_result")
    if isinstance(score_result, dict) and "constants" in score_result:
        rows += components_rows(score_result, label)
    return rows
