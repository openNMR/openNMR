"""
Export a spectrum (from any vendor) as .csv, .xy or JCAMP-DX (.jdx/.dx).

Whatever the data were read from -- Bruker, Varian, JEOL, X-Pulse, Spinsolve,
SIMPSON, a saved openNMR dataset, a spectrum derived or simulated here -- they
are an :class:`~opennmr.core.spectrum.NMRSpectrum` by the time they are shown,
and that is what is written::

    from opennmr.io.export import export_spectrum
    export_spectrum(spectrum, "sample.csv")                    # ppm, real, imaginary
    export_spectrum(spectrum, "sample.xy", x_unit="hz")        # two columns
    export_spectrum(spectrum, "sample.jdx", include_imag=True) # JCAMP-DX, complex
    export_spectrum(spectrum, "fid.csv", domain="fid")         # the acquired FID

Formats
-------
``csv``
    A table with a header row: ``ppm`` (or ``Hz``), ``real`` and, if the
    spectrum has one, ``imag``. A 2D or pseudo-2D spectrum is written whole
    (a matrix: the first row is the direct-dimension axis, the first column
    the indirect one) unless a ``row`` is chosen.
``xy``
    Plain text, one point per line: x, y (and the imaginary part as a third
    column with ``include_imag``). Comment lines start with ``#`` and say what
    the file is (switch off with ``metadata_comments=False``). Reads in
    anything that reads columns.
``jdx`` / ``dx``
    JCAMP-DX 5.01, ``NMR SPECTRUM`` (or ``NMR FID``): the real part as
    ``XYDATA`` -- the form every NMR program reads -- or, with
    ``include_imag``, real and imaginary as an ``NTUPLES`` pair of pages. The
    x axis is in Hz with ``##.OBSERVE FREQUENCY`` (the JCAMP convention, from
    which a program recovers ppm), or ppm with ``x_unit="ppm"``.

A 2D spectrum is written to ``xy``/``jdx`` one row at a time (``row=``): those
formats hold a single trace.

Points are written in the order the spectrum holds them (high shift first)
unless ``ascending=True``. Numbers carry 12 significant figures.
:func:`read_jcamp_spectrum` reads these JCAMP-DX files back (plain-number
tables only), which is how they are tested.
"""

from __future__ import annotations

import csv
import os
import re

import numpy as np

from ..core.spectrum import NMRSpectrum

FORMATS = {"csv": ".csv", "xy": ".xy", "jdx": ".jdx"}
_EXTENSIONS = {".csv": "csv", ".xy": "xy", ".txt": "xy", ".dat": "xy", ".jdx": "jdx",
               ".dx": "jdx", ".jcm": "jdx", ".jcamp": "jdx"}
_PRECISION = 12


def format_of(path: str) -> str:
    """The format a file name asks for, from its extension."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext not in _EXTENSIONS:
        raise ValueError(f"Cannot tell the format from {ext or 'a name with no extension'!r}: use "
                         ".csv, .xy, .jdx or .dx (or pass fmt=).")
    return _EXTENSIONS[ext]


def _number(value: float) -> str:
    return f"{float(value):.{_PRECISION}g}"


def _title(spectrum: NMRSpectrum, title: str | None) -> str:
    if title:
        return " ".join(str(title).split())
    raw = spectrum.metadata.raw or {}
    name = raw.get("display_label") or raw.get("title")
    if not name and spectrum.metadata.source_path:
        name = os.path.basename(str(spectrum.metadata.source_path).rstrip("/\\"))
    return " ".join(str(name or "openNMR spectrum").split())


def _hz_axis(spectrum: NMRSpectrum, axis_index: int = -1) -> np.ndarray:
    axis = spectrum.axes[axis_index]
    if axis.hz is not None:
        return np.asarray(axis.hz, dtype=float)
    if axis.ppm is not None:
        return np.asarray(axis.ppm, dtype=float) * float(spectrum.metadata.sfo1 or 1.0)
    raise ValueError("This axis has no frequency values to export.")


def _trace(spectrum: NMRSpectrum, domain: str, row: int | None, x_unit: str):
    """``(x, real, imag or None, x_name, info)`` for one trace."""
    info: dict = {}
    if domain == "fid":
        raw = spectrum.metadata.raw or {}
        fid = raw.get("fid_raw")
        dt = raw.get("fid_dt")
        if fid is None or dt is None:
            raise ValueError("This spectrum has no stored FID to export (it was not read from a "
                             "raw FID, or it is a derived spectrum).")
        fid = np.asarray(fid)
        if fid.ndim == 2:
            if row is None:
                raise ValueError("This holds a FID per row: choose which with row=.")
            fid = fid[row]
        x = np.arange(fid.shape[-1]) * float(dt)
        return x, np.real(fid).astype(float), np.imag(fid).astype(float), "time_s", \
            {"x_unit": "SECONDS", "dwell": float(dt)}
    if domain != "spectrum":
        raise ValueError("domain must be 'spectrum' or 'fid'.")
    data = np.asarray(spectrum.data, dtype=float)
    imag = None if spectrum.data_imag is None else np.asarray(spectrum.data_imag, dtype=float)
    if data.ndim == 2:
        if row is None:
            raise ValueError("A 2D spectrum holds many traces: choose one with row= (0-based).")
        if not 0 <= row < data.shape[0]:
            raise IndexError(f"row {row} is outside the {data.shape[0]} rows.")
        data = data[row]
        imag = None if imag is None else imag[row]
    if (spectrum.metadata.raw or {}).get("axis_is_hz"):
        # No observe frequency was given when it was read, so its axis is Hz already.
        x, name, unit = np.asarray(spectrum.axes[-1].ppm, dtype=float), "Hz", "HZ"
    elif x_unit == "ppm":
        if spectrum.axes[-1].ppm is None:
            raise ValueError("This spectrum has no ppm axis; use x_unit='hz'.")
        x, name, unit = np.asarray(spectrum.axes[-1].ppm, dtype=float), "ppm", "PPM"
    elif x_unit == "hz":
        x, name, unit = _hz_axis(spectrum), "Hz", "HZ"
    else:
        raise ValueError("x_unit must be 'ppm' or 'hz'.")
    return x, data, imag, name, {"x_unit": unit}


def _header_lines(spectrum: NMRSpectrum, title: str, domain: str, x_name: str) -> list[str]:
    from .._version import __version__

    meta = spectrum.metadata
    lines = [f"{title}", f"exported by openNMR {__version__}",
             f"data: {'FID' if domain == 'fid' else 'spectrum'}; x: {x_name}"]
    if meta.nucleus:
        lines.append(f"nucleus: {meta.nucleus}")
    if meta.sfo1:
        lines.append(f"observe frequency (MHz): {_number(meta.sfo1)}")
    if meta.vendor:
        lines.append(f"read from: {meta.vendor}")
    return lines


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_csv(path, spectrum, domain, x_unit, include_imag, row, ascending, metadata_comments,
               title):
    data = np.asarray(spectrum.data)
    if domain == "spectrum" and data.ndim == 2 and row is None:
        x = np.asarray(spectrum.axes[-1].ppm if x_unit == "ppm" else _hz_axis(spectrum), float)
        f1_axis = spectrum.axes[0]
        f1 = (f1_axis.ppm if f1_axis.ppm is not None
              else f1_axis.index if f1_axis.index is not None else np.arange(data.shape[0]))
        order = np.argsort(x) if ascending else np.arange(x.size)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if metadata_comments:
                for line in _header_lines(spectrum, title, domain, x_unit):
                    handle.write(f"# {line}\n")
            writer.writerow([f"F1\\F2 ({x_unit})"] + [_number(v) for v in x[order]])
            for i in range(data.shape[0]):
                writer.writerow([_number(f1[i])] + [_number(v) for v in data[i][order]])
        return
    x, real, imag, name, _info = _trace(spectrum, domain, row, x_unit)
    order = np.argsort(x) if ascending else np.arange(x.size)
    columns = [x[order], real[order]]
    heads = [name, "real"]
    if include_imag and imag is not None:
        columns.append(imag[order])
        heads.append("imag")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        if metadata_comments:
            for line in _header_lines(spectrum, title, domain, name):
                handle.write(f"# {line}\n")
        writer = csv.writer(handle)
        writer.writerow(heads)
        for values in zip(*columns):
            writer.writerow([_number(v) for v in values])


def _write_xy(path, spectrum, domain, x_unit, include_imag, row, ascending, metadata_comments,
              title):
    x, real, imag, name, _info = _trace(spectrum, domain, row, x_unit)
    order = np.argsort(x) if ascending else np.arange(x.size)
    columns = [x[order], real[order]]
    heads = [name, "real"]
    if include_imag and imag is not None:
        columns.append(imag[order])
        heads.append("imag")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        if metadata_comments:
            for line in _header_lines(spectrum, title, domain, name):
                handle.write(f"# {line}\n")
            handle.write(f"# columns: {'  '.join(heads)}\n")
        for values in zip(*columns):
            handle.write("  ".join(_number(v) for v in values) + "\n")


def _nucleus_jcamp(nucleus: str | None) -> str | None:
    if not nucleus:
        return None
    match = re.fullmatch(r"(\d+)([A-Za-z]+)", str(nucleus).strip())
    return f"^{match.group(1)}{match.group(2)}" if match else str(nucleus).strip()


def _table_lines(x0: float, dx: float, values: np.ndarray, per_line: int = 5) -> list[str]:
    """The (X++(Y..Y)) table: each line starts with the x of its first point."""
    out = []
    for start in range(0, values.size, per_line):
        chunk = values[start:start + per_line]
        out.append(" ".join([_number(x0 + dx * start)] + [_number(v) for v in chunk]))
    return out


def _write_jdx(path, spectrum, domain, x_unit, include_imag, row, ascending, metadata_comments,
               title):
    from .._version import __version__

    x, real, imag, _name, info = _trace(spectrum, domain, row, x_unit)
    order = np.argsort(x) if ascending else np.arange(x.size)
    x, real = x[order], real[order]
    imag = None if imag is None else imag[order]
    n = x.size
    if n < 2:
        raise ValueError("A JCAMP-DX table needs at least two points.")
    steps = np.diff(x)
    uniform = np.allclose(steps, steps.mean(), rtol=1e-6, atol=0)
    if not uniform:
        raise ValueError("The x axis is not evenly spaced, which JCAMP-DX (X++(Y..Y)) requires.")
    dx = float(steps.mean())
    complex_out = bool(include_imag and imag is not None)
    meta = spectrum.metadata
    kind = "NMR FID" if domain == "fid" else "NMR SPECTRUM"
    lines = [f"##TITLE={_title(spectrum, title)}", "##JCAMP-DX=5.01", f"##DATA TYPE={kind}",
             f"##DATA CLASS={'NTUPLES' if complex_out else 'XYDATA'}",
             f"##ORIGIN=openNMR {__version__}", "##OWNER=", ]
    if meta.sfo1:
        lines.append(f"##.OBSERVE FREQUENCY={_number(meta.sfo1)}")
    nucleus = _nucleus_jcamp(meta.nucleus)
    if nucleus:
        lines.append(f"##.OBSERVE NUCLEUS={nucleus}")
    x_units = info["x_unit"]
    if domain == "fid":
        lines.append(f"##$DWELL TIME={_number(info['dwell'])}")
    first_x, last_x = float(x[0]), float(x[-1])
    if complex_out:
        lines += [
            f"##NTUPLES={kind}",
            f"##VAR_NAME={'TIME' if domain == 'fid' else 'FREQUENCY'}, "
            f"{'FID' if domain == 'fid' else 'SPECTRUM'}/REAL, "
            f"{'FID' if domain == 'fid' else 'SPECTRUM'}/IMAG, PAGE NUMBER",
            "##SYMBOL=X, R, I, N", "##VAR_TYPE=INDEPENDENT, DEPENDENT, DEPENDENT, PAGE",
            "##VAR_FORM=AFFN, AFFN, AFFN, AFFN", f"##VAR_DIM={n}, {n}, {n}, 2",
            f"##UNITS={x_units}, ARBITRARY UNITS, ARBITRARY UNITS, ",
            "##FACTOR=1, 1, 1, 1",
            f"##FIRST={_number(first_x)}, {_number(real[0])}, {_number(imag[0])}, 1",
            f"##LAST={_number(last_x)}, {_number(real[-1])}, {_number(imag[-1])}, 2",
        ]
        for page, (symbol, values) in enumerate((("R", real), ("I", imag)), start=1):
            lines.append(f"##PAGE=N={page}")
            lines.append(f"##NPOINTS={n}")
            lines.append(f"##FIRST={_number(first_x)}, {_number(values[0])}")
            lines.append(f"##DATA TABLE=(X++({symbol}..{symbol})), XYDATA")
            lines += _table_lines(first_x, dx, values)
        lines.append(f"##END NTUPLES={kind}")
    else:
        lines += [f"##XUNITS={x_units}", "##YUNITS=ARBITRARY UNITS", "##XFACTOR=1", "##YFACTOR=1",
                  f"##FIRSTX={_number(first_x)}", f"##LASTX={_number(last_x)}",
                  f"##DELTAX={_number(dx)}", f"##NPOINTS={n}", f"##FIRSTY={_number(real[0])}",
                  "##XYDATA=(X++(Y..Y))"]
        lines += _table_lines(first_x, dx, real)
    lines.append("##END=")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


_WRITERS = {"csv": _write_csv, "xy": _write_xy, "jdx": _write_jdx}


def export_spectrum(spectrum: NMRSpectrum, path: str, fmt: str | None = None, *,
                    x_unit: str | None = None, include_imag: bool | None = None,
                    domain: str = "spectrum", row: int | None = None,
                    ascending: bool = False, metadata_comments: bool | None = None,
                    title: str | None = None) -> str:
    """Write ``spectrum`` to ``path`` and return the path.

    Parameters
    ----------
    fmt : {"csv", "xy", "jdx"}, optional
        Default: from the file extension (``.csv``, ``.xy``, ``.jdx``/``.dx``).
    x_unit : {"ppm", "hz"}, optional
        The x axis. Default: ppm for csv and xy; Hz for JCAMP-DX, which is the
        convention there (the observe frequency in the file gives ppm back).
    include_imag : bool, optional
        Write the imaginary part too. Default: yes for csv, no for xy and jdx.
        Ignored when the spectrum has none.
    domain : {"spectrum", "fid"}
        ``"fid"`` writes the acquired FID (as read, before any window or
        zero-fill) against time in seconds, for spectra that were read from one.
    row : int, optional
        0-based row of a 2D / pseudo-2D spectrum. Required for xy and jdx;
        csv writes the whole matrix without it.
    ascending : bool
        Lowest x first (default: the spectrum's own order, high shift first).
    metadata_comments : bool, optional
        ``#`` header lines saying what the file is. Default: yes for xy, no for
        csv (so the file is a clean table).
    title : str, optional
        The name written into the file (default: the experiment's).
    """
    if not isinstance(spectrum, NMRSpectrum):
        raise TypeError("export_spectrum needs an NMRSpectrum (what Open experiments holds, or "
                        "opennmr.io.load / SIMPSONload returns).")
    fmt = (fmt or format_of(path)).lower()
    fmt = {"dx": "jdx", "jcamp": "jdx", "jcamp-dx": "jdx", "txt": "xy"}.get(fmt, fmt)
    if fmt not in _WRITERS:
        raise ValueError(f"fmt must be one of {sorted(_WRITERS)}.")
    if include_imag is None:
        include_imag = fmt == "csv"
    if metadata_comments is None:
        metadata_comments = fmt == "xy"
    if x_unit is None:
        x_unit = "hz" if fmt == "jdx" else "ppm"
    if x_unit not in ("ppm", "hz"):
        raise ValueError("x_unit must be 'ppm' or 'hz'.")
    if fmt in ("xy", "jdx") and domain == "spectrum" and np.ndim(spectrum.data) == 2 and row is None:
        raise ValueError(f"A 2D spectrum holds many traces and a .{fmt} file one: choose a row= "
                         "(0-based), or use csv for the whole matrix.")
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    _WRITERS[fmt](str(path), spectrum, domain, x_unit, include_imag, row, ascending,
                  metadata_comments, _title(spectrum, title))
    return str(path)


# ---------------------------------------------------------------------------
# Reading back (plain-number JCAMP-DX tables)
# ---------------------------------------------------------------------------

def _ldr_key(text: str) -> str:
    return re.sub(r"[\s_\-/.$]", "", text).upper()


def read_jcamp_spectrum(path: str) -> dict:
    """Read a JCAMP-DX spectrum or FID whose tables are plain numbers (what
    :func:`export_spectrum` writes).

    Returns ``{"title", "data_type", "x", "real", "imag" (or None), "x_unit",
    "observe_frequency" (MHz or None), "nucleus" (or None)}`` with ``x`` and
    the values multiplied by the file's ``XFACTOR``/``YFACTOR``. Compressed
    (ASDF) tables are not read."""
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read()
    header: dict[str, str] = {}
    pages: list[list[tuple[float, list[float]]]] = []
    current: list | None = None
    in_table = False
    page_symbols: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            key, _, value = stripped[2:].partition("=")
            k = _ldr_key(key)
            value = value.strip()
            in_table = False
            if k == "XYDATA" or k == "DATATABLE":
                in_table = True
                current = []
                pages.append(current)
                match = re.search(r"\(X\+\+\((\w)\.\.\w\)\)", value)
                page_symbols.append(match.group(1) if match else "Y")
            elif k not in ("END",):
                header.setdefault(k, value)
            continue
        if in_table and stripped and not stripped.startswith("$$"):
            tokens = stripped.replace(",", " ").split()
            try:
                numbers = [float(t) for t in tokens]
            except ValueError as exc:
                raise ValueError(f"{path}: compressed (non-numeric) JCAMP-DX tables are not "
                                 "supported here.") from exc
            current.append((numbers[0], numbers[1:]))
    if not pages:
        raise ValueError(f"{path}: no XYDATA or DATA TABLE found.")

    def number(key, default=1.0):
        try:
            return float(header.get(key, default))
        except ValueError:
            return default

    xfactor = number("XFACTOR", 1.0)
    yfactor = number("YFACTOR", 1.0)
    factors = header.get("FACTOR")
    if factors:
        parts = [p.strip() for p in factors.split(",")]
        try:
            xfactor, yfactor = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            pass

    def flatten(page):
        ys = [y for _x, line in page for y in line]
        n = len(ys)
        x0 = page[0][0] * xfactor
        last = header.get("LASTX") or (header.get("LAST", "").split(",")[0].strip() or None)
        if "DELTAX" in header:
            step = number("DELTAX") * xfactor
        elif last and n > 1:
            step = (float(last) * xfactor - x0) / (n - 1)
        else:
            step = (page[-1][0] - page[0][0]) * xfactor / max(len(page) - 1, 1) / max(
                len(page[0][1]), 1)
        return x0 + step * np.arange(n), np.asarray(ys) * yfactor

    x, real = flatten(pages[0])
    imag = flatten(pages[1])[1] if len(pages) > 1 else None
    units = header.get("XUNITS") or (header.get("UNITS", "").split(",")[0].strip() or None)
    nucleus = header.get("OBSERVENUCLEUS")
    freq = header.get("OBSERVEFREQUENCY")
    return {"title": header.get("TITLE"), "data_type": header.get("DATATYPE"), "x": x,
            "real": real, "imag": imag, "x_unit": units,
            "observe_frequency": float(freq) if freq else None,
            "nucleus": nucleus.lstrip("^") if nucleus else None}
