"""
DMFit importer: the spectrum-and-model text export (``.asc``) and the fit
file (``.fxml``).

DMFit (Massiot et al.) saves a fit in two files, and this module reads both
with the standard library and numpy only:

``.asc`` -- "spectrum and model all lines"
    ::

        ti:7Li NMR
        ##freq 194.38
        ##col_ Hz   Spectrum   Model   Line#1   Line#2
        1.23984e+06  -806477  -253469  29.0349  23.8384
        ...

    One row per point: the axis (``Hz`` or ``ppm``, named in the ``##col_``
    line), the data, the sum of the fitted model, and each line on its own.
    Opened as a spectrum (the ``Spectrum`` column); the model and every line
    are kept in ``metadata.raw["dmfit"]`` (:func:`fit_curves` returns them).
    The file prints six significant figures, so the axis is rebuilt as the
    evenly spaced one it came from rather than read back rounded.

``.fxml`` -- the fit
    The lines (model name, amplitude, position, width, Gauss/Lorentz
    fraction, CSA/quadrupolar parameters ...) with, for each parameter, its
    value, whether it was fitted, its unit and its error when DMFit
    computed one, plus the nucleus, observe frequency, spin rate, baseline
    and the fit's standard deviation. :func:`read_fxml` returns it as a
    dict. Opening a ``.fxml`` opens the ``.asc`` beside it (the one whose
    name starts with the ``.fxml``'s name, or the only ``.asc`` there).

The observe frequency is taken from the ``.fxml`` when there is one (the
``.asc`` rounds it to two decimals), else from the ``##freq`` line, else
from ``sfo1=``; ppm = Hz / MHz. Without any, the axis stays in Hz.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import numpy as np

from . import base
from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum

RAW_KEY = "dmfit"


# ---------------------------------------------------------------------------
# Recognising the files
# ---------------------------------------------------------------------------

def _head_text(path: str, n: int = 600) -> str:
    try:
        with open(path, "rb") as handle:
            return handle.read(n).decode("latin1")
    except OSError:
        return ""


def is_asc(path: str) -> bool:
    """A DMFit ``.asc`` export: a ``ti:`` title line and a ``##col_`` line."""
    if not (os.path.isfile(path) and path.lower().endswith(".asc")):
        return False
    lines = _head_text(path).splitlines()[:6]
    return bool(lines) and lines[0].lower().startswith("ti:") and any(
        line.startswith("##col_") for line in lines)


def is_fxml(path: str) -> bool:
    """A DMFit fit file: an XML document whose root is ``NMRFit``."""
    return (os.path.isfile(path) and path.lower().endswith(".fxml")
            and "<NMRFit" in _head_text(path))


def _detect(path: str) -> bool:
    return is_asc(path) or is_fxml(path)


# ---------------------------------------------------------------------------
# The fit file
# ---------------------------------------------------------------------------

def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return text.strip() if isinstance(text, str) else text


def _parameter(element) -> dict:
    record = {"value": _number((element.text or "").strip())}
    if "Fit" in element.attrib:
        record["fitted"] = bool(element.attrib["Fit"].strip())
        record["fit_flag"] = element.attrib["Fit"]
    else:
        record["fitted"] = False
    if "Unit" in element.attrib:
        record["unit"] = element.attrib["Unit"]
    if "Error" in element.attrib:
        record["error"] = _number(element.attrib["Error"])
    other = {k: v for k, v in element.attrib.items() if k not in ("Fit", "Unit", "Error")}
    if other:
        record["attributes"] = other
    return record


def read_fxml(path: str) -> dict:
    """The fit in a DMFit ``.fxml`` as a dict::

        {"version", "timestamp", "mode", "sdev", "compute": {...},
         "dimensions": [{"name": "F2", "nucleus", "frequency_MHz", "spinrate_Hz",
                         "baseline": {...},
                         "lines": [{"model": "CSA mas",
                                    "parameters": {"amp": {"value", "fitted", "unit",
                                                           "error"}, "pos": {...}, ...}}]}],
         "graph": {...}, "file": {"name", "title"}}

    ``fitted`` is True for a parameter DMFit varied (its ``Fit`` flag is set);
    ``error`` is there only for those it gave an uncertainty for."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{path} is not a readable DMFit .fxml file: {exc}") from None
    if root.tag != "NMRFit":
        raise ValueError(f"{path} is not a DMFit .fxml file (root element {root.tag!r}).")

    def text(node, tag):
        child = node.find(tag) if node is not None else None
        return None if child is None or child.text is None else child.text.strip()

    params = root.find("FitParameters")
    out = {"version": text(params, "DMFitVersion"), "timestamp": text(params, "timestamp"),
           "mode": text(params, "FitModeAsc"), "sdev": _number(text(params, "Sdev")),
           "compute": {child.tag: _number((child.text or "").strip())
                       for child in (params.find("ComputeParam") if params is not None else None) or []},
           "dimensions": []}
    for dim in (params.findall("Dimension") if params is not None else []):
        name = (dim.text or "").strip()
        baseline = {}
        node = dim.find("BaseLine")
        if node is not None:
            baseline = {child.tag: _number((child.text or "").strip()) for child in node}
        lines = []
        for line in dim.findall("line"):
            model = text(line, "ModelName")
            lines.append({"model": model,
                          "parameters": {child.tag: _parameter(child) for child in line
                                         if child.tag != "ModelName"}})
        out["dimensions"].append({
            "name": name, "nucleus": text(dim, "nucleus"),
            "frequency_MHz": _number(text(dim, "frequency")),
            "spinrate_Hz": _number(text(dim, "spinrate")), "baseline": baseline, "lines": lines})
    graph = root.find("GraphParameters/Dimensions")
    out["graph"] = ({child.tag: {leaf.tag: _number((leaf.text or "").strip()) for leaf in child}
                     for child in graph} if graph is not None else {})
    out["file"] = {"name": text(root.find("DataFile"), "FileName"),
                   "title": text(root.find("DataFile"), "FileTitle")}
    return out


def fit_table(fit: dict) -> list[dict]:
    """One row per parameter of every line of ``fit`` (from :func:`read_fxml`):
    ``{"line", "model", "parameter", "value", "error", "unit", "fitted"}`` --
    ready for a table or a CSV."""
    rows = []
    for dim in fit.get("dimensions", []):
        for number, line in enumerate(dim["lines"], 1):
            for name, record in line["parameters"].items():
                rows.append({"line": number, "model": line["model"], "parameter": name,
                             "value": record["value"], "error": record.get("error"),
                             "unit": record.get("unit"), "fitted": record["fitted"]})
    return rows


# ---------------------------------------------------------------------------
# The spectrum-and-model file
# ---------------------------------------------------------------------------

def read_asc_table(path: str) -> dict:
    """``{"title", "freq_MHz" (or None), "axis_unit" ("hz"|"ppm"), "axis", "columns":
    {name: array}}`` from a DMFit ``.asc``. ``axis`` is the first column,
    rebuilt as an evenly spaced axis when it is (it is printed to six
    significant figures)."""
    title, freq, names = "", None, None
    with open(path, "r", encoding="latin1") as handle:
        header = []
        for line in handle:
            if line.startswith(("ti:", "##")):
                header.append(line.rstrip("\r\n"))
                continue
            break
    for line in header:
        if line.lower().startswith("ti:"):
            title = line[3:].strip()
        elif line.startswith("##freq"):
            try:
                freq = float(line.split()[1])
            except (IndexError, ValueError):
                freq = None
        elif line.startswith("##col_"):
            names = line[len("##col_"):].split()
    if not names:
        raise ValueError(f"{path}: no '##col_' line naming the columns.")
    table = np.loadtxt(path, comments=None, skiprows=len(header), ndmin=2)
    if table.shape[1] != len(names):
        raise ValueError(f"{path}: {table.shape[1]} columns of numbers, but the header names "
                         f"{len(names)} ({', '.join(names)}).")
    first = names[0].lower()
    unit = "ppm" if "ppm" in first else "hz" if first.startswith("hz") else None
    if unit is None:
        raise ValueError(f"{path}: the first column is {names[0]!r}; expected Hz or ppm.")
    axis = table[:, 0].copy()
    n = len(axis)
    if n > 2:
        # A straight line through every printed value: the rounding of the many points
        # averages out, where the two end points alone would carry theirs in full.
        slope, intercept = np.polyfit(np.arange(n, dtype=float), axis, 1)
        uniform = intercept + slope * np.arange(n)
        step = abs(slope)
        # Six significant figures put each printed value within ~5e-6 of its size of the truth.
        printed_error = float(np.max(np.abs(axis))) * 5e-6 + step * 1e-3
        if float(np.max(np.abs(uniform - axis))) <= printed_error:
            axis = uniform
    return {"title": title, "freq_MHz": freq, "axis_unit": unit, "axis": axis,
            "columns": {name: table[:, i + 1] for i, name in enumerate(names[1:])}}


def _find_sibling_fxml(asc_path: str) -> str | None:
    """The ``.fxml`` that belongs to a ``.asc``: the one whose stem the ``.asc``'s
    name starts with, else the only one in the folder."""
    folder = os.path.dirname(os.path.abspath(asc_path)) or "."
    try:
        found = sorted(n for n in os.listdir(folder) if n.lower().endswith(".fxml"))
    except OSError:
        return None
    name = os.path.basename(asc_path).lower()
    best = [n for n in found if name.startswith(os.path.splitext(n)[0].lower() + " ")
            or name.startswith(os.path.splitext(n)[0].lower() + "_")]
    if best:
        return os.path.join(folder, max(best, key=len))
    return os.path.join(folder, found[0]) if len(found) == 1 else None


def _find_sibling_asc(fxml_path: str) -> str | None:
    folder = os.path.dirname(os.path.abspath(fxml_path)) or "."
    stem = os.path.splitext(os.path.basename(fxml_path))[0].lower()
    try:
        found = sorted(n for n in os.listdir(folder) if n.lower().endswith(".asc")
                       and is_asc(os.path.join(folder, n)))
    except OSError:
        return None
    best = [n for n in found if n.lower().startswith(stem + " ") or n.lower().startswith(stem + "_")]
    if best:
        return os.path.join(folder, best[0])
    return os.path.join(folder, found[0]) if len(found) == 1 else None


def read(path: str, *, fxml: str | None = None, nucleus: str | None = None,
         sfo1: float | None = None, reference_ppm: float = 0.0, **_ignored) -> NMRSpectrum:
    """A DMFit ``.asc`` (or the ``.asc`` beside a ``.fxml``) as an
    :class:`NMRSpectrum`, with the fitted model and lines in
    ``metadata.raw["dmfit"]`` (see :func:`fit_curves`).

    ``fxml``: the fit file to take the frequency, nucleus and line parameters
    from (default: the one beside the ``.asc``). ``nucleus`` / ``sfo1``
    (MHz) override or supply them."""
    path = os.fspath(path)
    if is_fxml(path):
        asc = _find_sibling_asc(path)
        if asc is None:
            raise ValueError(
                f"{path} holds the fit (use dmfit.read_fxml) but no spectrum. In DMFit, export "
                "'spectrum and model all lines' (.asc) next to it, then open that file.")
        return read(asc, fxml=fxml or path, nucleus=nucleus, sfo1=sfo1,
                    reference_ppm=reference_ppm)
    if not is_asc(path):
        raise ValueError(f"{path} is not a DMFit .asc or .fxml file.")
    table = read_asc_table(path)
    fxml_path = fxml or _find_sibling_fxml(path)
    fit = None
    if fxml_path and os.path.isfile(fxml_path):
        try:
            fit = read_fxml(fxml_path)
        except ValueError:
            fit = None
    dimension = (fit["dimensions"][0] if fit and fit["dimensions"] else {})
    observe = sfo1 or dimension.get("frequency_MHz") or table["freq_MHz"]
    nucleus = nucleus or dimension.get("nucleus")
    frequency_source = ("sfo1 argument" if sfo1 else "fit file (.fxml)" if dimension.get("frequency_MHz")
                        else "##freq line" if table["freq_MHz"] else None)

    axis, unit = table["axis"], table["axis_unit"]
    names = list(table["columns"])
    spectrum_name = next((n for n in names if n.lower() == "spectrum"), names[0])
    if unit == "hz":
        hz = axis
        ppm = hz / observe + reference_ppm if observe else None
    else:
        ppm = axis
        hz = ppm * observe if observe else None
    order = np.argsort(-(ppm if ppm is not None else hz), kind="stable")
    columns = {name: values[order] for name, values in table["columns"].items()}
    hz_axis = hz[order]
    ppm_axis = ppm[order] if ppm is not None else None
    lines = [name for name in names if name.lower().startswith("line")]
    dmfit_raw = {"columns": names, "model": columns.get("Model"),
                 "lines": [columns[name] for name in lines], "line_labels": lines,
                 "fxml": fit, "fxml_path": fxml_path if fit else None,
                 "frequency_MHz": observe, "frequency_source": frequency_source,
                 "axis_unit_in_file": unit}
    metadata = NMRMetadata(
        vendor="dmfit", nucleus=nucleus, sfo1=observe or 1.0, bf1=observe or 1.0,
        o1=reference_ppm * (observe or 0.0), source_path=path,
        raw={RAW_KEY: dmfit_raw, "axis_is_hz": observe is None, "title": table["title"]})
    if ppm_axis is None:
        x_axis = NMRAxis(kind="frequency", ppm=hz_axis, hz=hz_axis, label="Frequency (Hz)")
    else:
        x_axis = NMRAxis(kind="frequency", ppm=ppm_axis, hz=hz_axis)
    return NMRSpectrum(data=np.asarray(columns[spectrum_name], dtype=float), axes=[x_axis],
                       metadata=metadata)


def fit_curves(spectrum: NMRSpectrum) -> dict:
    """What a DMFit spectrum carries of the fit: ``{"x" (ppm), "model" (sum of
    the lines, with the baseline), "lines": {label: array}, "fit": the .fxml
    dict or None}`` -- on the spectrum's own axis."""
    record = (spectrum.metadata.raw or {}).get(RAW_KEY)
    if not record:
        raise ValueError("This spectrum was not read from a DMFit file.")
    return {"x": spectrum.axes[-1].ppm, "model": record["model"],
            "lines": dict(zip(record["line_labels"], record["lines"])), "fit": record["fxml"]}


def peek(path: str, **_ignored) -> dict:
    """Length and nucleus from the files' headers."""
    if is_fxml(path):
        asc = _find_sibling_asc(path)
        fit = read_fxml(path)
        dimension = fit["dimensions"][0] if fit["dimensions"] else {}
        n = read_asc_table(asc)["axis"].shape[0] if asc else 0
        return {"vendor": "dmfit", "ndim": 1, "shape": (n,), "nucleus": dimension.get("nucleus"),
                "nucleus_indirect": None, "pseudo_2d": False}
    table = read_asc_table(path)
    fxml = _find_sibling_fxml(path)
    nucleus = None
    if fxml:
        try:
            dims = read_fxml(fxml)["dimensions"]
            nucleus = dims[0]["nucleus"] if dims else None
        except ValueError:
            nucleus = None
    return {"vendor": "dmfit", "ndim": 1, "shape": (len(table["axis"]),), "nucleus": nucleus,
            "nucleus_indirect": None, "pseudo_2d": False}


base.register("dmfit", _detect, read, peek=peek)
