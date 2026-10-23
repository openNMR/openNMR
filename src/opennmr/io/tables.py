"""
Open spectra and FIDs saved as plain text: ``.csv``, ``.xy`` and JCAMP-DX
(``.jdx`` / ``.dx``) -- the files :func:`opennmr.io.export.export_spectrum`
writes, and most files of the same kind from other programs.

What is read
------------
``.csv`` / ``.xy`` / ``.txt`` / ``.dat``
    Two or three columns: x, real[, imaginary]. A header row (``ppm,real,imag``)
    and ``#`` comment lines are used when present: openNMR's own comment lines
    say whether the table is a spectrum or a FID, the nucleus and the observe
    frequency. A 2D table (header ``F1\\F2 (ppm)``) is read as a 2D spectrum.
``.jdx`` / ``.dx``
    A JCAMP-DX file with plain-number tables (``XYDATA`` or an ``NTUPLES``
    real/imaginary pair), as openNMR writes. Compressed tables are not read.

The x axis is ppm, Hz or time (seconds) -- from the header, the comment lines
or ``x_unit=``. Without an observe frequency a Hz axis stays in Hz (as for
SIMPSON); give ``sfo1=`` (MHz) or ``nucleus=`` with ``field=`` to get ppm.
A FID goes through the same processing as every other FID (window, zero-fill,
FFT, phase), so it can be re-processed in the GUI.
"""

from __future__ import annotations

import os
import re

import numpy as np

from . import base, simpson
from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum

#: ``metadata.raw`` key recording how the table was read.
RAW_KEY = "text_table"

#: Extensions recognised automatically. ``.txt`` and ``.dat`` are too generic
#: (SIMPSON uses ``.dat``): open those by naming ``vendor="text"``.
EXTENSIONS = (".csv", ".xy", ".jdx", ".dx")
_JCAMP = (".jdx", ".dx", ".jcm", ".jcamp")


def _is_jcamp(path: str) -> bool:
    return os.path.splitext(str(path))[1].lower() in _JCAMP


def _unit_of(name: str | None) -> str | None:
    """``"ppm"``, ``"hz"`` or ``"s"`` from a column / unit name, else ``None``."""
    if not name:
        return None
    text = name.strip().lower()
    if "ppm" in text:
        return "ppm"
    if text.startswith("hz") or "(hz" in text or text in ("hertz",):
        return "hz"
    if text.startswith(("time", "t_", "seconds")) or text in ("s", "sec") or "(s)" in text \
            or text.endswith("_s"):
        return "s"
    return None


def _split(line: str) -> list[str]:
    return [t for t in re.split(r"[,;\s]+", line.strip()) if t != ""]


def _floats(tokens: list[str]) -> list[float] | None:
    try:
        return [float(t) for t in tokens]
    except ValueError:
        return None


def _parse_comments(lines: list[str]) -> dict:
    """openNMR's ``# key: value`` lines (and the title, its first line)."""
    out: dict = {"title": None}
    first = True
    for line in lines:
        text = line.lstrip("#").strip()
        if not text:
            continue
        if first and ":" not in text:
            out["title"] = text
            first = False
            continue
        first = False
        low = text.lower()
        if low.startswith("data:"):
            for part in text.split(";"):
                key, _, value = part.partition(":")
                key, value = key.strip().lower(), value.strip()
                if key == "data":
                    out["domain"] = "fid" if value.lower() == "fid" else "spectrum"
                elif key == "x":
                    out["x_name"] = value
        elif low.startswith("nucleus:"):
            out["nucleus"] = text.split(":", 1)[1].strip() or None
        elif low.startswith("observe frequency"):
            try:
                out["observe_frequency"] = float(text.split(":", 1)[1])
            except ValueError:
                pass
        elif low.startswith("columns:"):
            out["columns"] = text.split(":", 1)[1].split()
    return out


def parse_text_table(path: str) -> dict:
    """A CSV / XY table as ``{"x", "real", "imag" (or None), "matrix" (2D or
    None), "f1", "title", "domain", "x_unit", "nucleus", "observe_frequency"}``."""
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as handle:
        raw_lines = handle.read().splitlines()
    comments = [ln for ln in raw_lines if ln.lstrip().startswith("#")]
    meta = _parse_comments(comments)
    body = [ln for ln in raw_lines if ln.strip() and not ln.lstrip().startswith("#")]
    header: list[str] | None = None
    rows: list[list[float]] = []
    for ln in body:
        tokens = _split(ln)
        values = _floats(tokens)
        if values is None:
            if rows:
                raise ValueError(f"{path}: a non-numeric line ({ln[:40]!r}) inside the table.")
            if header is None:
                header = [t.strip() for t in re.split(r"[,;\t]", ln.strip())]
            continue
        rows.append(values)
    if not rows:
        raise ValueError(f"{path}: no numeric rows found.")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{path}: rows have different numbers of columns {sorted(widths)}.")
    table = np.asarray(rows, dtype=float)

    out = {"title": meta.get("title"), "nucleus": meta.get("nucleus"),
           "observe_frequency": meta.get("observe_frequency"), "imag": None, "matrix": None,
           "f1": None}
    head0 = (header[0] if header else "") or ""
    if header and head0.replace(" ", "").lower().startswith("f1\\f2") and table.shape[1] > 2:
        # A 2D table: the header row holds the F2 axis, the first column F1.
        out["domain"] = "spectrum"
        match = re.search(r"\(([^)]*)\)", head0)
        out["x_unit"] = _unit_of(match.group(1)) if match else None
        axis = _floats([h for h in header[1:] if h != ""])
        if axis is None or len(axis) != table.shape[1] - 1:
            raise ValueError(f"{path}: the 2D header row does not match its columns.")
        out["x"] = np.asarray(axis)
        out["f1"] = table[:, 0]
        out["matrix"] = table[:, 1:]
        return out
    if table.shape[1] not in (2, 3):
        raise ValueError(f"{path}: expected 2 or 3 columns (x, real[, imaginary]); found "
                         f"{table.shape[1]}.")
    out["x"], out["real"] = table[:, 0], table[:, 1]
    if table.shape[1] == 3:
        out["imag"] = table[:, 2]
    names = header or meta.get("columns") or []
    unit = _unit_of(names[0]) if names else None
    if unit is None:
        unit = _unit_of(meta.get("x_name"))
    out["x_unit"] = unit
    domain = meta.get("domain")
    if domain is None:
        domain = "fid" if unit == "s" else "spectrum"
    out["domain"] = domain
    return out


def parse_jcamp(path: str) -> dict:
    from .export import read_jcamp_spectrum

    data = read_jcamp_spectrum(path)
    unit = (data.get("x_unit") or "").strip().lower()
    unit = {"hz": "hz", "ppm": "ppm", "seconds": "s", "sec": "s", "s": "s"}.get(unit)
    kind = (data.get("data_type") or "").upper()
    domain = "fid" if ("FID" in kind or unit == "s") else "spectrum"
    return {"title": data.get("title"), "x": np.asarray(data["x"], float),
            "real": np.asarray(data["real"], float),
            "imag": None if data.get("imag") is None else np.asarray(data["imag"], float),
            "matrix": None, "f1": None, "x_unit": unit, "domain": domain,
            "nucleus": data.get("nucleus"), "observe_frequency": data.get("observe_frequency")}


def parse(path: str) -> dict:
    return parse_jcamp(path) if _is_jcamp(path) else parse_text_table(path)


def _frequency_axis_descending(x: np.ndarray, *arrays):
    order = np.argsort(-x, kind="stable")
    return (x[order],) + tuple(None if a is None else a[order] for a in arrays)


def read(path: str, *, nucleus: str | None = None, field=None, sfo1: float | None = None,
         x_unit: str | None = None, reference_ppm: float = 0.0, kind: str | None = None,
         time_unit: str = "s", sw: float | None = None, **options) -> NMRSpectrum:
    """Read a CSV / XY / JCAMP-DX file as an :class:`NMRSpectrum`.

    Parameters
    ----------
    nucleus, field, sfo1
        Override or supply what the file does not say (``sfo1`` in MHz wins;
        see :func:`opennmr.io.simpson.resolve_frequency`).
    x_unit : {"ppm", "hz", "s"}, optional
        What the x column is, if the file does not say.
    kind : {"fid", "spe"}, optional
        Force FID or spectrum, if the file does not say.
    reference_ppm
        Shift of the zero of a Hz axis, as for SIMPSON.
    Other keywords are the FID-processing options of
    :func:`opennmr.io.simpson.read_1d` (``window``, ``zero_fill`` ...).
    """
    data = parse(path)
    title = data.get("title") or os.path.basename(path)
    nucleus = nucleus or data.get("nucleus")
    unit = (x_unit or data.get("x_unit") or "").lower() or None
    domain = data["domain"]
    if kind:
        domain = "fid" if kind.lower() in ("fid",) else "spectrum"
    file_freq = data.get("observe_frequency")
    observe, _b0 = simpson.resolve_frequency(nucleus, field, sfo1)
    observe = observe or file_freq

    if domain == "fid":
        if data["matrix"] is not None:
            raise ValueError(f"{path}: a 2D table cannot be read as a FID.")
        x = data["x"]
        imag = data["imag"] if data["imag"] is not None else np.zeros_like(data["real"])
        scale = simpson._TIME_UNITS.get(time_unit, 1.0)
        if unit == "ms":
            scale = 1e-3
        raw = {"NP": x.size, "real": data["real"], "imag": imag, "NELEM": 1, "element": 0,
               "kind": "FID", "format": "table", "path": path}
        if sw is None:
            if x.size < 2:
                raise ValueError(f"{path}: one point has no dwell time; give sw= (Hz).")
            dt = abs(simpson.uniform_axis(x)[1]) * scale
            if dt <= 0:
                raise ValueError(f"{path}: the time column does not increase.")
            sw = 1.0 / dt
        raw["SW"] = float(sw)
        spec = simpson.read_1d(path, nucleus=nucleus, field=field, sfo1=observe,
                               reference_ppm=reference_ppm, raw=raw, title=title,
                               conjugate=options.pop("conjugate", False), **options)
        spec.metadata.raw[RAW_KEY] = {"domain": "fid", "x_unit": "s"}
        return spec

    x = np.asarray(data["x"], float)
    if unit is None:
        unit = "ppm"
    if unit == "s":
        raise ValueError(f"{path}: the x column is time; pass kind='fid' (or read it as a FID).")
    if data["matrix"] is not None:
        return _spectrum_2d(path, data, unit, observe, nucleus, reference_ppm, title)

    real = np.asarray(data["real"], float)
    imag = data["imag"]
    x, real, imag = _frequency_axis_descending(x, real, None if imag is None else np.asarray(
        imag, float))
    ppm, hz, used, label, axis_is_hz = _axes_for(x, unit, observe, reference_ppm)
    metadata = NMRMetadata(
        vendor="text", nucleus=nucleus, sfo1=used, bf1=used, o1=reference_ppm * (observe or 0.0),
        source_path=path, ns=None, pulse_program=None,
        raw={RAW_KEY: {"domain": "spectrum", "x_unit": unit, "observe_frequency": observe},
             "axis_is_hz": axis_is_hz, "title": title})
    return NMRSpectrum(data=real, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=hz, label=label)],
                       metadata=metadata, data_imag=imag)


def _axes_for(x: np.ndarray, unit: str, observe: float | None, reference_ppm: float):
    """``(ppm array, hz array, sfo1 used, label, axis_is_hz)``."""
    if unit == "ppm":
        sfo1 = observe or 1.0
        return x, x * sfo1, sfo1, None, False
    if observe:
        ppm = x / observe + reference_ppm
        return ppm, ppm * observe, observe, None, False
    return x.astype(float), x.astype(float), 1.0, "Frequency (Hz)", True


def _spectrum_2d(path, data, unit, observe, nucleus, reference_ppm, title) -> NMRSpectrum:
    x = np.asarray(data["x"], float)
    matrix = np.asarray(data["matrix"], float)
    order = np.argsort(-x, kind="stable")
    x, matrix = x[order], matrix[:, order]
    ppm, hz, used, label, axis_is_hz = _axes_for(x, unit, observe, reference_ppm)
    f1 = np.asarray(data["f1"], float)
    if np.allclose(f1, np.arange(1, f1.size + 1)):
        y_axis = NMRAxis(kind="index", index=f1)
    else:
        y_axis = NMRAxis(kind="index", index=f1, label="F1")
    metadata = NMRMetadata(
        vendor="text", nucleus=nucleus, sfo1=used, bf1=used, o1=reference_ppm * (observe or 0.0),
        source_path=path, ns=None, pulse_program=None,
        raw={RAW_KEY: {"domain": "spectrum", "x_unit": unit, "observe_frequency": observe},
             "axis_is_hz": axis_is_hz, "title": title})
    return NMRSpectrum(data=matrix, axes=[y_axis, NMRAxis(kind="frequency", ppm=ppm, hz=hz,
                                                           label=label)], metadata=metadata)


def peek(path: str, **_ignored) -> dict:
    """Shape and nucleus without building the spectrum."""
    data = parse(path)
    shape = data["matrix"].shape if data["matrix"] is not None else (len(data["x"]),)
    return {"vendor": "text", "ndim": len(shape), "shape": tuple(shape),
            "nucleus": data.get("nucleus"), "nucleus_indirect": None,
            "pseudo_2d": len(shape) == 2, "domain": data["domain"]}


def _detect(path: str) -> bool:
    """A file with a ``.csv`` / ``.xy`` / ``.jdx`` / ``.dx`` name that is
    numbers (a JCAMP-DX that X-Pulse owns is claimed earlier)."""
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(str(path))[1].lower()
    if ext not in EXTENSIONS:
        return False
    if ext in (".jdx", ".dx"):
        try:
            with open(path, "r", errors="ignore") as handle:
                head = handle.read(8192)
        except OSError:
            return False
        return "##" in head and ("XYDATA" in head.upper() or "DATA TABLE" in head.upper()
                                 or "NTUPLES" in head.upper() or "PEAK TABLE" not in head.upper())
    return True


base.register("text", _detect, read, peek=peek)
