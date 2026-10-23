"""
SIMPSON importer: spectra and FIDs written by SIMPSON's ``fsave``.

SIMPSON (Bak, Rasmussen & Nielsen) simulates solid-state NMR experiments
and saves its results as plain text. This module reads those files into
the same :class:`~opennmr.core.spectrum.NMRSpectrum` every other vendor
produces, so a simulated spectrum can be plotted, compared, referenced,
phased, integrated and fitted exactly like a measured one.

What is read
------------
``SIMP`` text files (``.spe`` for a spectrum, ``.fid`` for a time-domain
signal, or whatever extension the ``.tcl`` script chose -- the file's
*content* decides, never its name)::

    SIMP
    NP=1024
    SW=50000
    TYPE=SPE            (or FID)
    DATA
    <real> <imag>       one line per point
    END

and ``fsave -xreim`` ``.dat`` files (three columns: x, real, imaginary).
Those have no header, so the caller says whether they are a spectrum or a
FID (``kind=``; a FID's x column is time, in ``time_unit``).

Binary files (``fsave -binary``) are not supported: re-save with a plain
``fsave``. The error says so rather than reading garbage.

A spectrum
----------
Read as it was saved: ``SW`` and ``NP`` give a frequency axis running from
``-SW/2`` upwards in ``SW/NP`` steps (SIMPSON's convention), shown
high-frequency-first like every other spectrum here.

A FID
-----
Read as a *raw FID*: the same window, zero-fill, truncation, phase and
(through the GUI) linear-prediction controls as any measured FID, applied
by :mod:`opennmr.processing.ft` -- and the same ones rebuild it when
changed. SIMPSON's FIDs rotate the opposite way to the convention that
pipeline expects, so the FID is **conjugated on reading** (``conjugate=
True``, the default): that puts each line where SIMPSON's own ``fft`` puts
it, which the tests pin. Pass ``conjugate=False`` for data that already
follow the other convention.

Frequency axis
--------------
SIMPSON files carry **no observe frequency**, so ppm needs it from you:
``nucleus`` with ``field`` or ``sfo1``. ``field`` is a string such as
``"500"`` for the conversational magnet name (the frequency in MHz at
which 1H resonates), or a number for the field in Tesla (the convention
of :func:`opennmr.notebook.simpson.SIMPSONimport`). ``reference_ppm`` is
the shift the carrier (0 Hz in SIMPSON) stands for: simulations are
usually run on resonance, so adding the shift of the line you put at the
carrier makes the axis comparable with a measured spectrum. With neither,
the axis is in **Hz** and says so.
"""

from __future__ import annotations

import os
import re

import numpy as np

from . import base
from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from ..processing import ft as processing_ft

#: ``metadata.raw`` key holding what was asked of the reader, so a panel,
#: a state file or a log can say exactly how the spectrum was made.
RAW_KEY = "simpson"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def sniff_format(path: str) -> str:
    """``"simp"``, ``"dat"`` (three columns), ``"shape"`` (two columns) or
    ``"tcl"`` (a SIMPSON input script), by content. Raises ValueError for JSON
    (genuine CSDM) and anything else."""
    first = ""
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                first = line.strip()
                break
    if first.startswith("{"):
        raise ValueError(
            f"{path} looks like JSON (genuine CSDM format), not SIMPSON text. Use "
            "opennmr.notebook.CSDMimport() for that.")
    if first == "SIMP":
        return "simp"
    if str(path).lower().endswith(".tcl") or first.startswith("#") or first.split()[:1] in (
            ["spinsys"], ["par"], ["proc"]):
        return "tcl"
    columns = len(first.split())
    if columns == 3:
        return "dat"
    if columns == 2:
        return "shape"
    raise ValueError(f"Could not identify {path} as a SIMPSON file (first line has "
                     f"{columns} columns; expected the SIMP header, or 3 columns for -xreim).")


def parse_simp_text(path: str) -> dict:
    """A ``SIMP`` file: ``NP``, ``SW``, ``NELEM``, ``TYPE``, ``real``, ``imag``,
    ``path`` (plus any other ``KEY=value`` header lines under ``header``).

    With ``NELEM=n`` the file holds ``n`` FIDs (or spectra) of ``NP`` points
    one after the other; ``real``/``imag`` then have ``n*NP`` entries and
    :func:`load_raw` picks one with ``element=``."""
    header: dict[str, str] = {}
    real: list[float] = []
    imag: list[float] = []
    in_data = False
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if in_data:
                if line == "END":
                    break
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"{path}: a data line should hold two numbers (real, "
                                     f"imaginary); got {line!r}.")
                real.append(float(parts[0]))
                imag.append(float(parts[1]))
            elif line == "DATA":
                in_data = True
            elif "=" in line:
                key, _, value = line.partition("=")
                header[key.strip().upper()] = value.strip()
    if str(header.get("FORMAT", "")).upper().startswith("BIN"):
        raise ValueError(f"{path} is a binary SIMPSON file (fsave -binary), which is not "
                         "supported. Save it again without -binary.")
    if "NP" not in header or "SW" not in header:
        raise ValueError(f"{path}: missing NP/SW header – not a valid SIMPSON SIMP file.")
    np_header = int(float(header["NP"]))
    nelem = max(int(float(header.get("NELEM", 1) or 1)), 1)
    if len(real) != np_header * nelem:
        raise ValueError(f"{path}: header declares NP={np_header}"
                         + (f" and NELEM={nelem}" if nelem > 1 else "")
                         + f" ({np_header * nelem} data lines) but {len(real)} were read.")
    return {"NP": np_header, "SW": float(header["SW"]), "NELEM": nelem,
            "TYPE": header.get("TYPE", "").upper() or None,
            "real": np.asarray(real, dtype=float), "imag": np.asarray(imag, dtype=float),
            "path": path, "header": header}


def parse_xreim(path: str) -> dict:
    """An ``fsave -xreim`` file: x, real, imaginary. One data row (an NP=1
    efficiency test) is handled."""
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 3:
        raise ValueError(f"{path}: expected 3 columns (x, real, imaginary).")
    return {"NP": data.shape[0], "x": data[:, 0], "real": data[:, 1], "imag": data[:, 2],
            "path": path}


def classify_columns(x) -> str:
    """``"xreim"`` when the first column is one evenly spaced axis (strictly
    monotonic, and no point further than a quarter of a step from a straight
    line -- SIMPSON writes x to about six figures, so it is not exact), else
    ``"table"``: x repeating while another column steps, a parameter scan."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return "xreim"
    steps = np.diff(x)
    if not (np.all(steps > 0) or np.all(steps < 0)):
        return "table"
    index = np.arange(x.size)
    slope, intercept = np.polyfit(index, x, 1)
    if slope == 0:
        return "table"
    return "xreim" if np.max(np.abs(x - (intercept + slope * index))) <= 0.25 * abs(slope) else "table"


def uniform_axis(x):
    """``(axis, step)``: the straight-line fit through ``x`` -- the axis the
    rounded values in the file stand for."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return x.copy(), 0.0
    index = np.arange(x.size)
    slope, intercept = np.polyfit(index, x, 1)
    return intercept + slope * index, float(slope)


_TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6}


def n_elements(path: str) -> int:
    """``NELEM`` of a SIMP file (1 when absent or for any other file)."""
    try:
        with open(path, "r", errors="ignore") as handle:
            if handle.readline().strip() != "SIMP":
                return 1
            for line in handle:
                line = line.strip()
                if line == "DATA":
                    break
                if line.upper().startswith("NELEM="):
                    return max(int(float(line.split("=", 1)[1])), 1)
    except (OSError, ValueError):
        pass
    return 1


def describe(path: str) -> dict:
    """What a file is, without reading its data: ``{"role", "detail"}``.

    ``role`` is ``"data"`` (a spectrum or FID: SIMP text, or x/re/im columns),
    ``"table"`` (x, y, z columns of a parameter scan), ``"shape"`` (a pulse
    shape), ``"script"`` (a .tcl input) or ``"unknown"``."""
    try:
        fmt = sniff_format(path)
    except (ValueError, OSError) as exc:
        return {"role": "unknown", "detail": str(exc)}
    if fmt == "simp":
        return {"role": "data", "detail": "SIMP text", "elements": n_elements(path)}
    if fmt == "shape":
        return {"role": "shape", "detail": "pulse shape (amplitude, phase)"}
    if fmt == "tcl":
        return {"role": "script", "detail": "SIMPSON input script"}
    try:
        data = np.loadtxt(path, ndmin=2)
    except ValueError as exc:
        return {"role": "unknown", "detail": str(exc)}
    if data.shape[1] != 3:
        return {"role": "unknown", "detail": "expected 3 columns"}
    if classify_columns(data[:, 0]) == "table":
        return {"role": "table", "detail": "x, y, z table"}
    return {"role": "data", "detail": "x, real, imaginary columns"}


def load_raw(path: str, kind: str | None = None, time_unit: str = "s",
             sw: float | None = None, element: int = 0) -> dict:
    """Read ``path`` into ``{"kind": "FID"|"SPE", "NP", "SW", "real", "imag",
    "x_Hz" (SPE), "format", "NELEM", "element"}``.

    ``kind`` : ``None`` takes the file's own ``TYPE`` (SIMP text). A headerless
    x/re/im ``.dat`` has none: it is called a spectrum when its x column runs
    through zero (frequencies about the carrier) and a FID when it starts at or
    after zero (times); give ``"fid"`` or ``"spe"`` to say otherwise. ``sw`` : a
    ``.dat`` FID's spectral width, if its time column cannot give it.
    ``element`` : which of an ``NELEM`` file's FIDs/spectra (0-based)."""
    fmt = sniff_format(path)
    if fmt == "shape":
        raise ValueError(f"{path} looks like a SIMPSON pulse-shape file (2 columns, no header), "
                         "not a spectrum or FID. Use SIMPSONread_shape().")
    if fmt == "tcl":
        raise ValueError(f"{path} is a SIMPSON input script (.tcl), not data. Its nucleus and "
                         "field can be read with read_tcl_settings().")
    wanted = kind.upper() if kind else None
    if wanted not in (None, "FID", "SPE"):
        raise ValueError("kind must be 'fid' or 'spe' (or None to read it from the file).")
    if fmt == "simp":
        raw = parse_simp_text(path)
        found = raw["TYPE"]
        if found not in ("FID", "SPE"):
            raise ValueError(f"{path}: TYPE is {found!r}; expected FID or SPE.")
        if wanted and wanted != found:
            raise ValueError(f"{path} holds a {found}, not a {wanted}.")
        nelem = raw["NELEM"]
        element = int(element)
        if not 0 <= element < nelem:
            raise ValueError(f"{path} holds {nelem} element(s); element={element} is out of "
                             f"range (0 to {nelem - 1}).")
        if nelem > 1:
            block = slice(element * raw["NP"], (element + 1) * raw["NP"])
            raw["real"], raw["imag"] = raw["real"][block], raw["imag"][block]
        raw["element"] = element
        raw["kind"] = found
        raw["format"] = "SIMP"
        if found == "SPE":
            raw["x_Hz"] = -raw["SW"] / 2 + np.arange(raw["NP"]) * (raw["SW"] / raw["NP"])
        return raw
    raw = parse_xreim(path)
    if classify_columns(raw["x"]) == "table":
        raise ValueError(
            f"{path} has three columns, but the first is not one evenly spaced axis (it repeats "
            "or jumps): it is a table of x, y, z values from a parameter scan, not x, real, "
            "imaginary. Read it with read_table() / SIMPSONread_table().")
    raw.update({"NELEM": 1, "element": 0})
    guessed = wanted is None
    if guessed:
        wanted = "SPE" if (raw["x"].min() < 0 < raw["x"].max()) else "FID"
    raw["kind"] = wanted
    raw["kind_guessed"] = guessed
    raw["format"] = "xreim"
    axis, step = uniform_axis(raw["x"])
    if wanted == "SPE":
        raw["x_Hz"] = axis
        raw["SW"] = float(abs(step) * raw["NP"]) if raw["NP"] > 1 else float(sw or 0.0)
    else:
        scale = _TIME_UNITS.get(time_unit)
        if scale is None:
            raise ValueError(f"time_unit must be one of {sorted(_TIME_UNITS)}.")
        if sw is None:
            if raw["NP"] < 2:
                raise ValueError(f"{path}: one point has no dwell time; give sw= (Hz).")
            dt = abs(step) * scale
            if dt <= 0:
                raise ValueError(f"{path}: the time column does not increase.")
            sw = 1.0 / dt
        raw["SW"] = float(sw)
    return raw


def read_table(path: str) -> dict:
    """A headerless three-column parameter-scan table (``fsave`` of a scan, or
    ``-xyz``-style output): ``{"x", "y", "z"}`` columns, and when x and y form a
    complete grid, ``"xs"``, ``"ys"`` and ``"Z"`` (``Z[j, i]`` at ``ys[j]``,
    ``xs[i]``)."""
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 3:
        raise ValueError(f"{path}: expected 3 columns (x, y, z).")
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    out = {"path": path, "x": x, "y": y, "z": z, "grid": False}
    xs, ys = np.unique(x), np.unique(y)
    if xs.size * ys.size == x.size and xs.size > 1 and ys.size > 1:
        grid = np.full((ys.size, xs.size), np.nan)
        grid[np.searchsorted(ys, y), np.searchsorted(xs, x)] = z
        if not np.isnan(grid).any():
            out.update({"grid": True, "xs": xs, "ys": ys, "Z": grid})
    return out


_NUMBER_NAMES = ("spin_rate", "proton_frequency", "gamma_angles", "sw", "np")


def _eval_number(expression: str, names: dict):
    """A number from ``8192*6`` / ``spin_rate*gamma_angles``, or ``None``."""
    import ast
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return None

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return float(names[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult,
                                                               ast.Div, ast.Pow)):
            left, right = walk(node.left), walk(node.right)
            return {ast.Add: left + right, ast.Sub: left - right, ast.Mult: left * right,
                    ast.Div: left / right, ast.Pow: left ** right}[type(node.op)]
        raise ValueError("not a plain number")

    try:
        return walk(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def read_tcl_settings(path: str) -> dict:
    """What a SIMPSON input script says about the experiment, for pre-filling
    the nucleus and field boxes: ``{"nucleus", "channels", "proton_mhz",
    "spin_rate", "sw", "np"}`` (each ``None`` if absent or not a plain number).

    Only the ``spinsys`` and ``par`` blocks are looked at, and only arithmetic
    on numbers and earlier ``par`` entries is evaluated -- the script is never
    run."""
    text = open(path, "r", errors="ignore").read()
    text = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    out: dict = {"nucleus": None, "channels": [], "proton_mhz": None, "spin_rate": None,
                 "sw": None, "np": None}
    match = re.search(r"\bchannels\s+([^\n]+)", text)
    if match:
        out["channels"] = match.group(1).split()
        out["nucleus"] = out["channels"][0] if out["channels"] else None
    block = re.search(r"\bpar\s*\{(.*?)\n\s*\}", text, flags=re.S)
    values: dict = {}
    if block:
        for line in block.group(1).splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2 or parts[0] == "variable":
                continue
            number = _eval_number(parts[1], values)
            if number is not None:
                values[parts[0]] = number
    if "proton_frequency" in values:
        out["proton_mhz"] = values["proton_frequency"] / 1e6
    for key in ("spin_rate", "sw", "np"):
        out[key] = values.get(key)
    return out


# ---------------------------------------------------------------------------
# Observe frequency
# ---------------------------------------------------------------------------

def resolve_frequency(nucleus: str | None, field=None, sfo1: float | None = None):
    """``(sfo1_MHz or None, B0_tesla or None)``.

    ``sfo1`` wins. Otherwise ``nucleus`` with ``field`` (a string = the 1H
    frequency in MHz, a number = Tesla). ``None`` for the frequency when
    nothing was given -- the axis is then Hz."""
    if sfo1:
        return float(sfo1), None
    if nucleus is None or field in (None, ""):
        return None, None
    from ..core.registry import find_gamma
    from ..notebook.utils import compute_BF1_SFO1

    if isinstance(field, str):
        try:
            proton_mhz = float(field)
        except ValueError as exc:
            raise ValueError(f"field={field!r} is a string but not a number – give e.g. "
                             "field='500' (a '500 MHz' magnet) or field=11.7 (Tesla).") from exc
        gamma_1h = find_gamma("1H")
        if gamma_1h is None:
            raise ValueError("Could not look up 1H in the gyromagnetic-ratio table.")
        b0 = proton_mhz / (gamma_1h["gamma"] / 1.0e6)
    else:
        b0 = float(field)
    _bf1, observe, _canonical = compute_BF1_SFO1(nucleus, b0)
    return float(observe), b0


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def _frequency_axis(freq_hz: np.ndarray, sfo1: float | None, reference_ppm: float):
    """(ppm-or-Hz array, hz array, sfo1 used, label)."""
    if sfo1:
        ppm = freq_hz / sfo1 + reference_ppm
        return ppm, ppm * sfo1, sfo1, None
    return freq_hz.astype(float), freq_hz.astype(float), 1.0, "Frequency (Hz)"


def read_1d(path: str, *, nucleus: str | None = None, field=None, sfo1: float | None = None,
            reference_ppm: float = 0.0, kind: str | None = None, time_unit: str = "s",
            sw: float | None = None, conjugate: bool = True,
            window: str = "exponential", window_kwargs: dict | None = None,
            zero_fill: int | bool | None = None, truncate: int | bool | None = None,
            fid_shift: float = 0.0, align_echo: bool = False, echo_offset: float = 0.0,
            linear_predict_points: int = 0, linear_predict_order: int = 20,
            linear_predict_direction: str = "forward",
            autophase: bool = False, phase_mode: str = "apc0",
            phase0: float = 0.0, phase1: float = 0.0, pivot_frac: float = 0.5,
            phase0_inc: float = 0.1, phase1_max: float = 20.0, phase1_improvement: float = 0.15,
            phase_penalty: str = "magnitude", mass: float = 1.0, element: int = 0,
            raw: dict | None = None, title: str | None = None,
            **_ignored) -> NMRSpectrum:
    """A SIMPSON spectrum or FID as an :class:`NMRSpectrum`. See the module
    docstring for what is read and the frequency-axis arguments.

    A FID is processed here -- ``window`` (default exponential with no
    broadening, i.e. none), ``zero_fill`` (``None``: as acquired; ``True``:
    the next power of two; an int: that many points), ``truncate``,
    ``fid_shift``/``align_echo``/linear prediction, then FFT and phase -- and
    its raw points are kept in ``metadata.raw`` for the live Processing
    controls, as for every other FID. ``autophase=False`` (the default) leaves
    a simulated FID as it is: its phase is whatever the .tcl script made it.

    ``raw`` / ``title`` : an already-parsed ``load_raw``-style dict and a name
    for it, used by the text-table importer (:mod:`opennmr.io.tables`) so a
    FID read from CSV/XY/JCAMP-DX goes through this same pipeline.

    Extra keyword arguments (the other vendors' reader options) are ignored,
    so a caller with one set of options can use any reader."""
    if raw is None:
        raw = load_raw(path, kind=kind, time_unit=time_unit, sw=sw, element=element)
    title = title or os.path.basename(path)
    if raw["NELEM"] > 1:
        title += f" [{raw['element'] + 1}/{raw['NELEM']}]"
    observe, b0 = resolve_frequency(nucleus, field, sfo1)
    asked = {"nucleus": nucleus, "field": field, "sfo1": sfo1, "reference_ppm": reference_ppm,
             "kind": raw["kind"], "format": raw["format"], "conjugate": bool(conjugate),
             "element": raw["element"], "n_elements": raw["NELEM"],
             "kind_guessed": bool(raw.get("kind_guessed", False))}
    metadata_common = dict(
        vendor="simpson", nucleus=nucleus, sfo1=observe or 1.0, bf1=observe or 1.0,
        o1=reference_ppm * (observe or 0.0), source_path=path, ns=None,
        pulse_program=None)

    if raw["kind"] == "SPE":
        # Ascending Hz in the file; held descending, like every spectrum here.
        order = np.argsort(-raw["x_Hz"], kind="stable")
        freq_hz = raw["x_Hz"][order]
        complex_spectrum = (raw["real"] + 1j * raw["imag"])[order] / (mass or 1.0)
        ppm, hz, used, label = _frequency_axis(freq_hz, observe, reference_ppm)
        asked["spectral_width_Hz"] = raw["SW"]
        metadata = NMRMetadata(raw={RAW_KEY: asked, "axis_is_hz": observe is None,
                                    "title": title}, **metadata_common)
        metadata.sfo1 = used
        metadata.bf1 = used
        return NMRSpectrum(data=np.real(complex_spectrum),
                           axes=[NMRAxis(kind="frequency", ppm=ppm, hz=hz, label=label)],
                           metadata=metadata, data_imag=np.imag(complex_spectrum))

    # ---- a FID: the shared raw-FID pipeline --------------------------------
    sw_hz = float(raw["SW"])
    if sw_hz <= 0:
        raise ValueError(f"{path}: spectral width must be > 0 to compute the dwell time.")
    dt = 1.0 / sw_hz
    fid = (raw["real"] + 1j * raw["imag"]) / (mass or 1.0)
    if conjugate:
        fid = np.conj(fid)
    fid = np.asarray(fid, dtype=np.complex128)
    fid_raw = fid
    original_length = len(fid)

    if truncate is True:
        truncate_points = None
    elif isinstance(truncate, int) and not isinstance(truncate, bool):
        truncate_points = truncate
    else:
        truncate_points = None
    fid = processing_ft.prepare_fid(
        fid, fid_shift=fid_shift, align_echo_enabled=align_echo, echo_offset=echo_offset,
        truncate=truncate_points, linear_predict_points=linear_predict_points,
        linear_predict_order=linear_predict_order,
        linear_predict_direction=linear_predict_direction)
    n_acquired = len(fid)
    fid = processing_ft.apodize(fid, dt, window=window, **(window_kwargs or {}))
    if zero_fill is True:
        target = 1 << max(int(np.ceil(np.log2(max(len(fid), 1)))), 0)
    elif isinstance(zero_fill, int) and not isinstance(zero_fill, bool):
        target = zero_fill
    else:
        target = None
    if target is not None and target > len(fid):
        fid = processing_ft.zero_fill(fid, target)
    elif target is None:
        fid = fid[:max(n_acquired, 1)]
    freq_hz, spectrum = processing_ft.fft_spectrum(fid, dt)
    ppm, hz, used, label = _frequency_axis(freq_hz, observe, reference_ppm)

    autophased = bool(autophase)
    if autophase:
        if phase_mode == "joint":
            phase0, phase1 = processing_ft.autophase01(
                spectrum, ppm, phase0_range=(-180, 180), phase0_inc=phase0_inc,
                phase1_max=phase1_max, phase1_improvement=phase1_improvement,
                penalty=phase_penalty, pivot_frac=0.5)
        elif phase_mode == "apc0":
            phase0 = processing_ft.apc0(spectrum, ppm, phase0_inc)
        else:
            raise ValueError("phase_mode must be 'apc0' or 'joint'")
    spectrum = processing_ft.apply_phase(spectrum, ppm, phase0, phase1, pivot_frac=pivot_frac)

    from .xpulse import _phase_metadata
    asked.update({"spectral_width_Hz": sw_hz, "dwell_time_s": dt})
    metadata = NMRMetadata(raw={
        RAW_KEY: asked, "axis_is_hz": observe is None, "title": title,
        **_phase_metadata(phase0, phase1, autophased, pivot_frac),
        "fid": fid, "fid_raw": fid_raw, "fid_dt": dt, "fid_original_length": original_length,
    }, **metadata_common)
    metadata.sfo1 = used
    metadata.bf1 = used
    return NMRSpectrum(data=np.real(spectrum),
                       axes=[NMRAxis(kind="frequency", ppm=ppm, hz=hz, label=label)],
                       metadata=metadata, data_imag=np.imag(spectrum))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _detect(path: str) -> bool:
    """A file whose first line is ``SIMP``. (Headerless ``.dat`` files cannot be
    told from other text, so they are opened by naming the vendor.)"""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as handle:
            return handle.read(8).split(b"\n")[0].strip() == b"SIMP"
    except OSError:
        return False


def peek(path: str, **_ignored) -> dict:
    """Shape and kind from the header alone."""
    header: dict[str, str] = {}
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line == "DATA":
                break
            if "=" in line:
                key, _, value = line.partition("=")
                header[key.strip().upper()] = value.strip()
    return {"vendor": "simpson", "ndim": 1, "shape": (int(float(header.get("NP", 0))),),
            "nucleus": None, "nucleus_indirect": None, "pseudo_2d": False,
            "simpson_type": header.get("TYPE"),
            "n_elements": int(float(header.get("NELEM", 1) or 1))}


base.register("simpson", _detect, read_1d, peek=peek)
