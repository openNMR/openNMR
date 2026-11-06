"""
SIMPSON single-point (and few-point) series: pulse-sequence efficiencies.

A common SIMPSON job is an *efficiency test*: simulate the same experiment
many times while one quantity changes (the pulse length, the RF amplitude,
the offset, the MAS rate...) and record just the first point -- or the first
few points -- of the FID each time. The plot of that point against the swept
quantity is the nutation curve, power profile or offset profile.

This module reads such a series and combines the points:

    series = SIMPSONpoint_series(paths, x=[...], xname="B1", xlabel="B1 (kHz)")
    SIMPSONpoint_table(series, ["A", "A + B - 0.3*C"])      # a DataFrame
    SIMPSONplot_point_series(series, ["A", "A + B - 0.3*C"], part="real")

**Points are lettered.** The first point of each file is ``A``, the second
``B``, the third ``C`` and so on (up to ``Z``). Each is a complex number, as
stored: ``A`` is ``real + i*imag`` of the first point. An *expression* is
ordinary arithmetic on them -- ``+ - * / **``, brackets, numbers, and the
functions ``abs re im angle sqrt conj`` -- evaluated file by file, so
``A + B - 0.3*C`` combines the first three points of every file, and
``abs(A)`` is the magnitude of the first. ``x`` stands for the swept value
itself (``A / x`` divides by it). The result is then shown as its ``part``:
``"real"`` (default), ``"imag"``, ``"magnitude"`` or ``"phase"`` (degrees).

Expressions are parsed, never ``eval``'d: only the names and operations above
are accepted, so a typo (or a pasted line) cannot run anything.

The points are the file's own numbers: unlike a FID that is Fourier
transformed (see :mod:`opennmr.io.simpson`), they are **not** conjugated.
"""

from __future__ import annotations

import ast
import os
import re
import warnings

import numpy as np

from ..io import simpson as _io

#: A..Z: the points an expression can name.
POINT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PARTS = ("real", "imag", "magnitude", "phase")

_FUNCTIONS = {
    "abs": np.abs, "re": np.real, "im": np.imag, "real": np.real, "imag": np.imag,
    "angle": lambda z: np.degrees(np.angle(z)), "phase": lambda z: np.degrees(np.angle(z)),
    "sqrt": lambda z: np.sqrt(np.asarray(z, dtype=complex)), "conj": np.conj,
}
_BINARY = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
           ast.Div: np.divide, ast.Pow: np.power}
_UNARY = {ast.UAdd: lambda a: a, ast.USub: np.negative}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def SIMPSONread_points(path: str, n_points: int | None = None, kind: str | None = None,
                       time_unit: str = "s") -> np.ndarray:
    """The first ``n_points`` points of a SIMPSON FID (all, if ``None``) as a
    complex array, exactly as stored. At most 26 (A..Z)."""
    if _io.sniff_format(path) == "dat":
        # A headerless -xreim file: the time column is irrelevant to the points.
        raw = _io.load_raw(path, kind=kind or "fid", time_unit=time_unit, sw=1.0)
    else:
        raw = _io.load_raw(path)
    values = raw["real"] + 1j * raw["imag"]
    limit = len(POINT_LETTERS) if n_points is None else min(int(n_points), len(POINT_LETTERS))
    if limit < 1:
        raise ValueError("n_points must be at least 1.")
    return np.asarray(values[:limit], dtype=complex)


def x_from_filenames(paths) -> np.ndarray:
    """The number in each file's name (its last one): ``nut_12.5.fid`` -> 12.5,
    ``B1_0030.spe`` -> 30. Raises ValueError naming a file without one."""
    out = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(str(path)))[0]
        found = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", stem)
        if not found:
            raise ValueError(f"No number in the file name {os.path.basename(str(path))!r}.")
        out.append(float(found[-1]))
    return np.asarray(out, dtype=float)


def SIMPSONpoint_series(paths, x=None, xname: str = "x", xlabel: str | None = None,
                        n_points: int | None = None, kind: str | None = None) -> dict:
    """Read the leading points of each file in ``paths`` into one series.

    Parameters
    ----------
    paths : list of str
        SIMPSON FID files (``fsave`` output), one per value of the swept
        quantity. Each holds at least one point; files with fewer points than
        the others limit what can be named (A, B, C ... only as far as every
        file goes).
    x : array-like, "filenames" or None
        The swept values, one per path. ``"filenames"`` takes the number from
        each file name; ``None`` numbers them 1, 2, 3...
    xname : str
        A short name for the quantity (a table column heading: ``"B1"``,
        ``"tp"``).
    xlabel : str, optional
        The axis text (``"RF amplitude (kHz)"``); defaults to ``xname``.
    n_points : int, optional
        How many leading points to keep per file (default: as many as the
        shortest file has, up to 26).

    Returns
    -------
    dict
        ``x`` (float array), ``xname``, ``xlabel``, ``points`` (complex array,
        files x points), ``letters`` (the names available), ``paths``.
    """
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("Give at least one file.")
    if isinstance(x, str) and x == "filenames":
        x_values = x_from_filenames(paths)
    elif x is None:
        x_values = np.arange(1, len(paths) + 1, dtype=float)
    else:
        x_values = np.asarray(x, dtype=float)
        if x_values.shape != (len(paths),):
            raise ValueError(f"x has {x_values.size} values but {len(paths)} files were given – "
                             "they must correspond one to one.")
    rows = [SIMPSONread_points(p, n_points=n_points, kind=kind) for p in paths]
    shortest = min(len(r) for r in rows)
    if n_points is not None:
        shortest = min(shortest, int(n_points))
    points = np.vstack([r[:shortest] for r in rows])
    return {"x": x_values, "xname": str(xname or "x"), "xlabel": xlabel or str(xname or "x"),
            "points": points, "letters": POINT_LETTERS[:shortest], "paths": paths}


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

def _evaluate(node, names: dict):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)) \
            and not isinstance(node.value, bool):
        return node.value if isinstance(node.value, complex) else float(node.value)
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if len(node.id) == 1 and node.id in POINT_LETTERS:
            raise ValueError(f"Point {node.id} is not available: the files hold only "
                             f"{len([n for n in names if n in POINT_LETTERS])} point(s) "
                             f"({', '.join(n for n in POINT_LETTERS if n in names)}).")
        raise ValueError(f"Unknown name {node.id!r}: use the point letters A-Z, x, numbers and "
                         f"the functions {', '.join(sorted(_FUNCTIONS))}.")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_evaluate(node.left, names), _evaluate(node.right, names))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand, names))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCTIONS and len(node.args) == 1 and not node.keywords:
        return _FUNCTIONS[node.func.id](_evaluate(node.args[0], names))
    raise ValueError("Only + - * / ** brackets, numbers (1j is the imaginary unit), the letters "
                     "A-Z, x and the functions "
                     f"{', '.join(sorted(_FUNCTIONS))} are allowed in an expression.")


def check_expression(series: dict, expression: str) -> str | None:
    """``None`` when ``expression`` can be evaluated on ``series``, else
    a sentence saying what is wrong (for a dialog or a status line)."""
    try:
        SIMPSONcombine_points(series, expression)
    except (ValueError, SyntaxError, TypeError) as exc:
        return str(exc)
    return None


def SIMPSONcombine_points(series: dict, expression: str, part: str = "real") -> np.ndarray:
    """``expression`` evaluated on every file of ``series``, as ``part``.

    ``"A + B - 0.3*C"`` is the first point plus the second less 0.3 of the
    third, for each file in turn. Returns one value per file."""
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}.")
    text = (expression or "").strip().replace("−", "-").replace("×", "*")
    if not text:
        raise ValueError("The expression is empty: type e.g. A, or A + B - 0.3*C.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Cannot read {expression!r} as an expression "
                         f"({exc.msg}).") from exc
    points = np.asarray(series["points"], dtype=complex)
    names = {letter: points[:, i] for i, letter in enumerate(series.get("letters",
                                                                         POINT_LETTERS[:points.shape[1]]))}
    names["x"] = np.asarray(series["x"], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = _evaluate(tree, names)
    value = np.broadcast_to(np.asarray(value), (points.shape[0],)).astype(complex)
    if part == "real":
        return np.real(value)
    if part == "imag":
        return np.imag(value)
    if part == "magnitude":
        return np.abs(value)
    return np.degrees(np.angle(value))


def SIMPSONpoint_table(series: dict, expressions, part: str = "real"):
    """A DataFrame: the swept quantity (column named ``series["xname"]``)
    and one column per expression."""
    import pandas as pd

    expressions = [expressions] if isinstance(expressions, str) else list(expressions)
    frame = {series["xname"]: np.asarray(series["x"], dtype=float)}
    for expression in expressions:
        frame[expression] = SIMPSONcombine_points(series, expression, part)
    return pd.DataFrame(frame)


def SIMPSONsave_point_table(series: dict, expressions, path: str, part: str = "real") -> str:
    """Write :func:`SIMPSONpoint_table` as a CSV; returns ``path``."""
    SIMPSONpoint_table(series, expressions, part).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def SIMPSONplot_point_series(series: dict, expressions="A", part: str = "real", ax=None,
                             ylabel: str | None = None, xlabel: str | None = None,
                             labels=None, normalise: bool = False, marker: str = "o",
                             linestyle: str = "-", colours=None, legend: bool | None = None,
                             pl_width: float = 3.375, pl_height: float = 2.5,
                             font: str = "Arial", fontsize: float = 12,
                             save_fig: bool = False, fig_name: str = "point_series",
                             save_format: str = "png"):
    """Plot one or more expressions of ``series`` against its swept quantity.

    Parameters
    ----------
    expressions : str or list of str
        What to plot, e.g. ``"A"`` or ``["A", "A + B - 0.3*C"]``.
    part : {"real", "imag", "magnitude", "phase"}
    ax : matplotlib Axes, optional
        Draw into this (the GUI does); otherwise a new figure is made.
    ylabel, xlabel : str, optional
        Axis text. ``xlabel`` defaults to the series' own ``xlabel``; ``ylabel``
        to the expression (one curve) or "Intensity" (several, with a legend).
    labels : list of str, optional
        Legend text per expression (default: the expressions).
    normalise : bool
        Divide every curve by the largest absolute value among them all.

    Returns ``(fig, ax)``. Quantity symbols in the labels are italicised."""
    import matplotlib.pyplot as plt
    from ..plotting.accessibility import default_series_colours
    from ..plotting.labels import italicise_symbols

    expressions = [expressions] if isinstance(expressions, str) else list(expressions)
    curves = [SIMPSONcombine_points(series, e, part) for e in expressions]
    if normalise:
        peak = max((np.nanmax(np.abs(c)) for c in curves if c.size), default=1.0) or 1.0
        curves = [c / peak for c in curves]

    if ax is None:
        from .plotting import setup_plotting
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                               pl_height=pl_height)
        fig, ax = plt.subplots()
        fig.set_figwidth(style["pl_width"])
        fig.set_figheight(style["pl_height"])
    else:
        fig = ax.figure
    colours = list(colours) if colours else default_series_colours(len(curves))
    names = list(labels) if labels else expressions
    for i, (curve, name) in enumerate(zip(curves, names)):
        ax.plot(series["x"], curve, color=colours[i % len(colours)], marker=marker,
                linestyle=linestyle, linewidth=1.2, label=name)
    ax.set_xlabel(italicise_symbols(xlabel if xlabel is not None else series["xlabel"]))
    default_y = expressions[0] if len(expressions) == 1 else "Intensity"
    if ylabel is None:
        ylabel = default_y if part == "real" or len(expressions) > 1 else f"{default_y} ({part})"
    ax.set_ylabel(italicise_symbols(ylabel))
    if legend or (legend is None and len(curves) > 1):
        ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not getattr(fig, "_opennmr_managed", False):
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 -- layout is cosmetic
            pass
    if save_fig:
        from .plotting import save_figure
        save_figure(fig, fig_name=fig_name, save_format=save_format)
    return fig, ax


# ---------------------------------------------------------------------------
# One file as a curve (REDOR-like): every point of one file, against an x
# the user supplies
# ---------------------------------------------------------------------------
#
# A REDOR (or other dephasing / build-up) simulation stores its curve as the
# points of one "FID": one point per recoupling increment, the real part the
# signal, the imaginary part zero, and the spacing written as the rate --
# SIMPSON's SW, which for a rotor-synchronised curve is the MAS rate (or a
# fraction of it). Here those points are the y values, and the x values are
# whatever the user says they are: the file's own k/rate, a typed list, a list
# file, or start/step (the same choices as the vdlist in Dynamics).

CURVE_PARTS = ("real", "imag", "magnitude")
CURVE_X_SOURCES = ("file", "index", "typed", "list", "steps")
_COMMENT = ("#", "!", "%", ";", "//")
_RATE_KEYS = re.compile(
    r"^\s*(spin[_ ]?rate|mas(?:[_ ]?rate)?|sw|rate|spinning(?:[_ ]?rate)?|nu_?r)\s*[=:]\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(hz|khz|mhz)?\s*$", re.IGNORECASE)
_TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6}


def SIMPSONread_curve(path: str, element: int = 0) -> dict:
    """Read every point of one file as a curve.

    Understood:

    * a SIMPSON ``SIMP`` file (``NP``, ``SW``, ``DATA`` of real, imaginary):
      the points, with ``SW`` as the rate (one point every ``1/SW`` s -- a
      rotor period when SW is the MAS rate). ``element`` picks one of an
      ``NELEM`` file's curves.
    * plain columns, no header: one column (real), two (real, imaginary) or
      three. Three columns are (real, imaginary, rate) when the third is the
      same number on every line -- the MAS rate written beside each point --
      and otherwise (x, real, imaginary), as ``fsave -xreim`` writes.
      Commas or spaces separate; lines starting # ! % ; or // are comments;
      a ``mas = 10000`` / ``spin_rate: 10 kHz`` / ``sw = ...`` line gives the
      rate.

    Returns ``{"values" (complex), "n", "rate_Hz" (or None), "rate_source",
    "x_file" (or None), "x_file_kind" ("time" in s, "column", or None),
    "path", "format"}``. The points are the file's own numbers (not
    conjugated)."""
    path = str(path)
    with open(path, "r", errors="ignore") as handle:
        first = handle.readline().strip()
    if first == "SIMP":
        raw = _io.load_raw(path, element=element)
        values = np.asarray(raw["real"], dtype=float) + 1j * np.asarray(raw["imag"], dtype=float)
        rate = float(raw["SW"]) if raw.get("SW") else None
        x_file = np.arange(values.size) / rate if rate else None
        return {"values": values, "n": int(values.size), "rate_Hz": rate,
                "rate_source": "SW in the header" if rate else None, "x_file": x_file,
                "x_file_kind": "time" if rate else None, "path": path, "format": "SIMP"}

    rows: list[list[float]] = []
    rate = None
    rate_source = None
    with open(path, "r", errors="ignore") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith(_COMMENT):
                continue
            match = _RATE_KEYS.match(text)
            if match:
                scale = {"khz": 1e3, "mhz": 1e6}.get((match.group(3) or "").lower(), 1.0)
                rate = float(match.group(2)) * scale
                rate_source = f"line {number} ({match.group(1)})"
                continue
            try:
                rows.append([float(t) for t in text.replace(",", " ").split()])
            except ValueError:
                raise ValueError(f"{os.path.basename(path)} line {number}: {text!r} is not a "
                                 "row of numbers.") from None
    if not rows:
        raise ValueError(f"{os.path.basename(path)} holds no numbers.")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{os.path.basename(path)}: the rows do not all have the same number "
                         f"of columns ({sorted(widths)}).")
    data = np.asarray(rows, dtype=float)
    columns = data.shape[1]
    x_file = None
    kind = None
    if columns == 1:
        values = data[:, 0].astype(complex)
    elif columns == 2:
        values = data[:, 0] + 1j * data[:, 1]
    elif columns == 3:
        third = data[:, 2]
        if np.all(third == third[0]) and third[0] != 0 and data.shape[0] > 1:
            values = data[:, 0] + 1j * data[:, 1]
            if rate is None:
                rate, rate_source = float(third[0]), "column 3"
        else:
            values = data[:, 1] + 1j * data[:, 2]
            x_file, kind = data[:, 0], "column"
    else:
        raise ValueError(f"{os.path.basename(path)} has {columns} columns; expected 1 (real), "
                         "2 (real, imaginary) or 3 (real, imaginary, rate - or x, real, "
                         "imaginary).")
    if x_file is None and rate:
        x_file, kind = np.arange(values.size) / rate, "time"
    return {"values": values, "n": int(values.size), "rate_Hz": rate,
            "rate_source": rate_source, "x_file": x_file, "x_file_kind": kind,
            "path": path, "format": f"{columns} column(s)"}


def stepped_values(start: float, step: float | None = None, multiplier: float = 1.0,
                   stop: float | None = None, points: int | None = None,
                   n_default: int | None = None) -> np.ndarray:
    """Evenly (or geometrically) spaced values, as in Dynamics' Start + step.

    The k-th value is ``start + step*k`` (``multiplier`` 1), or
    ``start + step*(m**k - 1)/(m - 1)`` -- each step ``m`` times the last.
    How many:

    * ``points`` alone: that many.
    * ``stop`` alone: every value from ``start`` up to and including ``stop``
      (within rounding); the step must head towards it.
    * ``stop`` and ``points``: ``points`` values from ``start`` to exactly
      ``stop`` -- the step is worked out from them (``step`` is not used).
    * neither: ``n_default`` (the number of data points).
    """
    from ..processing.relaxation import start_step_values
    m = float(multiplier)
    if m <= 0:
        raise ValueError("The step multiplier must be positive.")
    start = float(start)
    if points is not None and int(points) < 1:
        raise ValueError("The number of points must be at least 1.")
    if stop is not None and points is not None:
        n = int(points)
        if n == 1:
            return np.array([start])
        span = float(stop) - start
        derived = span / (n - 1) if abs(m - 1.0) < 1e-12 else span * (m - 1.0) / (m ** (n - 1) - 1.0)
        values = start_step_values(start, derived, m, n)
        values[-1] = float(stop)                      # exactly, not to rounding
        return values
    if step is None or float(step) == 0:
        raise ValueError("Give a non-zero step (or Stop and Points, to work it out).")
    step = float(step)
    if points is not None:
        return start_step_values(start, step, m, int(points))
    if stop is not None:
        stop = float(stop)
        if (stop - start) * step < 0:
            raise ValueError(f"A step of {step:g} from {start:g} never reaches {stop:g}.")
        tolerance = 1e-9 * max(abs(stop), abs(start), abs(step))
        # Longer and longer runs until one passes Stop: each value is
        # computed once per run, not the whole series again for every value.
        n = 16
        while True:
            values = start_step_values(start, step, m, n)
            beyond = (values - stop) * np.sign(step) > tolerance
            if beyond.any():
                return values[:int(np.argmax(beyond))]
            if n >= 1_000_000:
                raise ValueError(f"A million values from {start:g} in steps of {step:g} "
                                 f"(multiplier {m:g}) do not reach {stop:g}.")
            n = min(2 * n, 1_000_000)
    if n_default is None:
        raise ValueError("Give Stop or a number of points.")
    return start_step_values(start, step, m, int(n_default))


def parse_typed_values(text: str) -> np.ndarray:
    """Numbers typed into a box: comma- or space-separated; a Bruker time
    suffix (``10m`` = 0.01) is honoured, as in a vdlist."""
    from ..io.lists import parse_value
    tokens = [t for t in str(text).replace(",", " ").replace(";", " ").split() if t]
    if not tokens:
        raise ValueError("Type at least one value.")
    out = []
    for token in tokens:
        try:
            out.append(parse_value(token))
        except ValueError:
            raise ValueError(f"{token!r} is not a number.") from None
    return np.asarray(out, dtype=float)


def SIMPSONcurve(path: str, x="file", xname: str = "x", xlabel: str | None = None,
                 element: int = 0, time_unit: str = "ms", factor: float = 1.0,
                 **steps) -> dict:
    """Every point of one file (:func:`SIMPSONread_curve`) against chosen x values.

    ``x`` :
        ``"file"`` -- the file's own: point k at ``k / rate`` (the rate is SW,
        or the MAS rate the file gives), shown in ``time_unit`` ("s", "ms",
        "us"); or its x column.
        ``"index"`` -- 0, 1, 2, ...
        ``"steps"`` -- :func:`stepped_values` with ``start``, ``step``,
        ``multiplier``, ``stop``, ``points`` passed as keywords.
        a path to a list file (one value per line, as a vdlist), times
        ``factor``; a string of typed numbers; or any sequence of numbers.

    There must be at least as many x values as points; extra ones are not
    used (as TopSpin uses the first TD entries of a longer vdlist). Fewer is
    an error, never a silently shortened curve.

    Returns ``{"x", "values" (complex), "n", "xname", "xlabel", "x_source",
    "rate_Hz", "path"}``."""
    curve = SIMPSONread_curve(path, element=element)
    n = curve["n"]
    source = x
    if isinstance(x, str) and x == "file":
        if curve["x_file"] is None:
            raise ValueError(f"{os.path.basename(str(path))} gives no rate (SW or MAS rate) or "
                             "x column to place its points: choose the x values another way.")
        if curve["x_file_kind"] == "time":
            if time_unit not in _TIME_UNITS:
                raise ValueError(f"time_unit must be one of {sorted(_TIME_UNITS)}.")
            values = curve["x_file"] / _TIME_UNITS[time_unit]
            source = f"k / {curve['rate_Hz']:g} Hz ({curve['rate_source']}), in {time_unit}"
            if xlabel is None:
                xlabel = f"t ({'µs' if time_unit == 'us' else time_unit})"
        else:
            values = curve["x_file"]
            source = "x column of the file"
    elif isinstance(x, str) and x == "index":
        values = np.arange(n, dtype=float)
        source = "point number"
    elif isinstance(x, str) and x == "steps":
        values = stepped_values(steps.get("start", 0.0), steps.get("step"),
                                steps.get("multiplier", 1.0), steps.get("stop"),
                                steps.get("points"), n_default=n)
        source = "start + step"
    elif isinstance(x, str) and os.path.isfile(x):
        from ..io.lists import read_value_list
        values = read_value_list(x)["values"] * float(factor)
        source = f"list file {os.path.basename(x)}" + (f" × {factor:g}" if factor != 1 else "")
    elif isinstance(x, str):
        values = parse_typed_values(x)
        source = "typed"
    else:
        values = np.asarray(x, dtype=float).ravel()
        source = "given"
    values = np.asarray(values, dtype=float)
    if values.size < n:
        raise ValueError(f"{values.size} x value(s) for {n} points in "
                         f"{os.path.basename(str(path))}: give one per point.")
    return {"x": values[:n], "values": curve["values"], "n": n, "xname": str(xname or "x"),
            "xlabel": xlabel or str(xname or "x"), "x_source": source,
            "x_unused": int(values.size - n), "rate_Hz": curve["rate_Hz"], "path": str(path)}


def curve_y(curve: dict, part: str = "real", normalise: str | None = None) -> np.ndarray:
    """The plotted y of a :func:`SIMPSONcurve`: its ``part`` ("real", "imag",
    "magnitude"), optionally divided by the first point (``"first"``, S/S0)
    or by the largest absolute value (``"max"``)."""
    values = curve["values"]
    if part == "real":
        y = np.real(values)
    elif part == "imag":
        y = np.imag(values)
    elif part == "magnitude":
        y = np.abs(values)
    else:
        raise ValueError(f"part must be one of {CURVE_PARTS}.")
    y = np.asarray(y, dtype=float)
    if normalise == "first":
        if y[0] == 0:
            raise ValueError("The first point is zero, so the curve cannot be divided by it.")
        y = y / y[0]
    elif normalise == "max":
        peak = np.max(np.abs(y))
        y = y / peak if peak else y
    elif normalise not in (None, "", "none"):
        raise ValueError("normalise must be None, 'first' or 'max'.")
    return y


def SIMPSONcurve_table(curve: dict, part: str = "real", normalise: str | None = None):
    """The curve as a DataFrame: the x column (named ``xname``) and y."""
    import pandas as pd
    return pd.DataFrame({curve["xname"]: curve["x"], "y": curve_y(curve, part, normalise)})


def SIMPSONsave_curve(curve: dict, path: str, part: str = "real",
                      normalise: str | None = None) -> str:
    """Write the curve (x, y as plotted) to ``path`` as CSV."""
    SIMPSONcurve_table(curve, part, normalise).to_csv(path, index=False)
    return path


def SIMPSONplot_curve(curve: dict, part: str = "real", normalise: str | None = None, ax=None,
                      xlabel: str | None = None, ylabel: str | None = None, marker: str = "o",
                      linestyle: str = "-", colour: str = "black",
                      pl_width: float = 3.375, pl_height: float = 2.5, font: str = "Arial",
                      fontsize: float = 12, save_fig: bool = False,
                      fig_name: str = "simpson_curve", save_format: str = "png"):
    """Plot a :func:`SIMPSONcurve`. ``ylabel`` defaults to S/S0 when divided
    by the first point, else Intensity. Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt
    from ..plotting.labels import italicise_symbols

    y = curve_y(curve, part, normalise)
    if ax is None:
        from .plotting import setup_plotting
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                               pl_height=pl_height)
        fig, ax = plt.subplots()
        fig.set_figwidth(style["pl_width"])
        fig.set_figheight(style["pl_height"])
    else:
        fig = ax.figure
    ax.plot(curve["x"], y, color=colour, marker=marker, linestyle=linestyle, linewidth=1.2,
            markersize=4, label=os.path.basename(curve["path"]))
    ax.set_xlabel(italicise_symbols(xlabel if xlabel is not None else curve["xlabel"]))
    if ylabel is None:
        ylabel = (r"$\mathit{S}/\mathit{S}_0$" if normalise == "first" else
                  "Intensity" + ("" if part == "real" else f" ({part})"))
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(italicise_symbols(ylabel))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not getattr(fig, "_opennmr_managed", False):
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 -- layout is cosmetic
            pass
    if save_fig:
        from .plotting import save_figure
        save_figure(fig, fig_name=fig_name, save_format=save_format)
    return fig, ax


# ---------------------------------------------------------------------------
# Two swept quantities: a heatmap
# ---------------------------------------------------------------------------
#
# A scan over two quantities at once (two offsets, rf amplitude and offset,
# ...) gives one value per combination. It arrives either as a table -- one
# row per combination, columns such as "O1 O2 Re Im", with or without a
# header line -- or as one FID per combination, the two values written in
# each file name (``scan_O1_10_O2_-5.fid``). Either way the values are
# placed on the grid of the distinct x and y values (a pivot): a combination
# that was not simulated is left empty (not invented), and one simulated
# twice is refused rather than one of them silently winning.

def _numbers(text: str):
    """The row's numbers, or None if any token is not one."""
    try:
        return [float(t) for t in text.replace(",", " ").split()]
    except ValueError:
        return None


def _strip_comment(text: str) -> str:
    for prefix in sorted(_COMMENT, key=len, reverse=True):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _split_names(text: str, width: int):
    """Column names from a header line, if it names exactly ``width``
    columns: split on tabs, then commas, then any space -- so a tab- or
    comma-separated header may hold names with spaces ("Offset 1 (kHz)").
    A leading comment mark (``# O1 O2 Re``) is allowed. None otherwise."""
    text = _strip_comment(text.strip())
    for sep in ("\t", ",", None):
        if sep is not None and sep not in text:
            continue
        tokens = [t.strip() for t in text.split(sep)]
        tokens = [t for t in tokens if t]
        if len(tokens) == width:
            return tokens
    return None


def SIMPSONread_scan(path: str, skip_rows: int | None = None, header="auto") -> dict:
    """A table of numbers, one row per simulated combination.

    ``skip_rows`` : lines at the top of the file to pass over (a title, a
    note, the simulation's settings). ``None`` (default) finds them: the
    data are the rows of numbers, all of the same width, running to the
    end of the file, and everything above them is passed over.
    ``header`` : whether the line just above the data (or, with
    ``skip_rows``, the first line after the skipped ones) names the
    columns -- ``"auto"`` (default: yes when it holds one name per column,
    e.g. ``O1 O2 Re Im``, ``# O1 O2 Re Im`` or a tab-separated
    ``Offset 1 (kHz)<tab>Offset 2 (kHz)<tab>Re``), ``True`` (it must) or
    ``False`` (the columns are called ``c1, c2, ...``).

    Spaces, tabs or commas separate numbers; blank lines, and lines
    starting # ! % ; or // among the data, are passed over.

    Returns ``{"names", "columns" (name -> array), "path", "skipped" (lines
    passed over above the data, header not counted), "header_line" (its
    1-based line number, or None), "first_data_line"}``."""
    path = str(path)
    name = os.path.basename(path)
    if header not in ("auto", True, False):
        raise ValueError("header must be 'auto', True or False.")
    if skip_rows is not None and int(skip_rows) < 0:
        raise ValueError("skip_rows must be 0 or more (or None to find them).")
    with open(path, "r", errors="ignore") as handle:
        lines = [line.rstrip("\r\n") for line in handle]

    def is_content(text):
        text = text.strip()
        return bool(text) and not text.startswith(_COMMENT)

    if skip_rows is None:
        # Walk up from the end over rows of numbers of one width.
        start, width = None, None
        for k in range(len(lines) - 1, -1, -1):
            text = lines[k].strip()
            if not is_content(text):
                continue
            row = _numbers(text)
            if row is None or (width is not None and len(row) != width):
                break
            width, start = len(row), k
        if start is None:
            last = next((k for k in range(len(lines) - 1, -1, -1) if is_content(lines[k])), None)
            if last is None:
                raise ValueError(f"{name} holds no numbers.")
            raise ValueError(f"{name} line {last + 1}: {lines[last].strip()!r} is not a row of "
                             "numbers (the data must run to the end of the file).")
        above = next((k for k in range(start - 1, -1, -1) if lines[k].strip()), None)
        data_above = next((k for k in range(start - 1, -1, -1) if is_content(lines[k])), None)
        if data_above is not None:
            other = _numbers(lines[data_above].strip())
            if other is not None:
                # A row of numbers is data, never a title to pass over.
                raise ValueError(f"{name}: the rows do not all have the same number of columns "
                                 f"(line {data_above + 1} has {len(other)}, line {start + 1} has "
                                 f"{width}).")
        header_line = None
        names = None
        if header is not False and above is not None:
            names = _split_names(lines[above], width)
            if names is not None and _numbers(_strip_comment(lines[above].strip())) is not None:
                names = None                     # a row of numbers of another width
            if names is not None:
                header_line = above
        if header is True and names is None:
            raise ValueError(f"{name}: no line with one name per column ({width}) just above "
                             f"the data (line {start + 1}).")
        top = header_line if header_line is not None else start
        skipped = sum(1 for k in range(top) if lines[k].strip())
    else:
        skip = int(skip_rows)
        if skip >= len(lines):
            raise ValueError(f"{name} has only {len(lines)} line(s); cannot skip {skip}.")
        start = skip
        names = None
        header_line = None
        first = next((k for k in range(skip, len(lines)) if lines[k].strip()), None)
        if first is None:
            raise ValueError(f"{name} holds no numbers after line {skip}.")
        first_numbers = _numbers(_strip_comment(lines[first].strip()))
        if header is True or (header == "auto" and first_numbers is None):
            header_line = first
            start = first + 1
        skipped = skip
        width = None
    rows: list[list[float]] = []
    first_data = None
    for k in range(start, len(lines)):
        text = lines[k].strip()
        if not is_content(text):
            continue
        row = _numbers(text)
        if row is None:
            raise ValueError(f"{name} line {k + 1}: {text!r} is not a row of numbers"
                             + (" (set the lines to skip, or whether the first row names the "
                                "columns)" if skip_rows is not None else "") + ".")
        if first_data is None:
            first_data = k + 1
        rows.append(row)
    if not rows:
        raise ValueError(f"{name} holds no numbers.")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{name}: the rows do not all have the same number of columns "
                         f"({sorted(widths)}).")
    width = widths.pop()
    if header_line is not None and names is None:
        names = _split_names(lines[header_line], width)
        if names is None:
            raise ValueError(f"{name} line {header_line + 1}: {lines[header_line].strip()!r} does "
                             f"not name {width} columns.")
    if names is None:
        names = [f"c{k + 1}" for k in range(width)]
    if len(set(names)) != len(names):
        raise ValueError(f"{name}: two columns have the same name.")
    data = np.asarray(rows, dtype=float)
    return {"names": list(names), "columns": {n: data[:, k] for k, n in enumerate(names)},
            "path": path, "skipped": int(skipped),
            "header_line": None if header_line is None else header_line + 1,
            "first_data_line": first_data}


def _distinct(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted distinct values and each value's index into them. Values equal
    to 9 significant figures count as one (scan values written by a script
    carry rounding)."""
    values = np.asarray(values, dtype=float)
    keys = np.array([float(f"{v:.9g}") for v in values])
    distinct, index = np.unique(keys, return_inverse=True)
    return distinct, index


def scan_grid(x, y, z, xname: str = "x", yname: str = "y", zname: str = "z") -> dict:
    """Place ``z`` on the grid of distinct ``x`` and ``y`` values.

    Returns ``{"xs", "ys", "Z" (Z[j, i] at ys[j], xs[i]; NaN where no value),
    "missing", "xname", "yname", "zname"}``. Raises if a combination appears
    twice."""
    x, y, z = (np.asarray(a, dtype=float).ravel() for a in (x, y, z))
    if not (x.size == y.size == z.size):
        raise ValueError("x, y and z must have one value each per row.")
    xs, ix = _distinct(x)
    ys, iy = _distinct(y)
    seen = {}
    grid = np.full((ys.size, xs.size), np.nan)
    for row, (i, j) in enumerate(zip(ix, iy)):
        if (i, j) in seen:
            raise ValueError(f"{xname} = {xs[i]:g}, {yname} = {ys[j]:g} appears twice (rows "
                             f"{seen[(i, j)] + 1} and {row + 1}): which one to show is not "
                             "clear.")
        seen[(i, j)] = row
        grid[j, i] = z[row]
    return {"xs": xs, "ys": ys, "Z": grid, "missing": int(np.isnan(grid).sum()),
            "xname": xname, "yname": yname, "zname": zname}


def SIMPSONscan_from_table(path: str, x: str | int = 0, y: str | int = 1,
                           z: str | int = 2, skip_rows: int | None = None,
                           header="auto") -> dict:
    """A grid (:func:`scan_grid`) from a table file (:func:`SIMPSONread_scan`).
    ``x``, ``y``, ``z`` : column names, or 0-based column numbers.
    ``skip_rows``, ``header`` : as for :func:`SIMPSONread_scan`."""
    table = SIMPSONread_scan(path, skip_rows=skip_rows, header=header)
    names = table["names"]

    def pick(key):
        if isinstance(key, (int, np.integer)):
            if not 0 <= int(key) < len(names):
                raise ValueError(f"Column {int(key) + 1} does not exist ({len(names)} columns).")
            return names[int(key)]
        if key not in table["columns"]:
            raise ValueError(f"No column called {key!r}; the columns are {', '.join(names)}.")
        return key

    xn, yn, zn = pick(x), pick(y), pick(z)
    if len({xn, yn}) < 2:
        raise ValueError("x and y must be different columns.")
    grid = scan_grid(table["columns"][xn], table["columns"][yn], table["columns"][zn],
                     xn, yn, zn)
    grid["path"] = path
    return grid


def numbers_in_name(path) -> list[float]:
    """Every number in a file's name (without its extension), in order:
    ``O1_10_O2_-5.fid`` -> [1, 10, 2, -5]; ``rf50_off-20`` -> [50, -20]. A
    number starts at the beginning, after a separator (_ space = ( ,) or
    after a letter; a sign in front of it is kept."""
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    found = re.findall(r"(?:(?<=^)|(?<=[_\s=(,]))[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
                       r"|(?<=[A-Za-z])[-+]?\d+(?:\.\d+)?", stem)
    return [float(v) for v in found]


def SIMPSONscan_from_files(paths, x_number: int = -2, y_number: int = -1,
                           expression: str = "A", part: str = "real",
                           xname: str = "x", yname: str = "y", kind: str | None = None) -> dict:
    """A grid from one FID per combination, the two values in each name.

    ``x_number``, ``y_number`` : which of the numbers in the name
    (:func:`numbers_in_name`) is x and which y, counted from 0 (or from the
    end with negative numbers: the default is the last two). ``expression``
    and ``part`` as for :func:`SIMPSONcombine_points` (``"A"`` = first point)."""
    paths = [str(p) for p in paths]
    if len(paths) < 2:
        raise ValueError("A heatmap needs more than one file.")
    xv, yv = [], []
    for path in paths:
        numbers = numbers_in_name(path)
        try:
            xv.append(numbers[x_number])
            yv.append(numbers[y_number])
        except IndexError:
            raise ValueError(f"{os.path.basename(path)} has {len(numbers)} number(s) in its "
                             "name – not enough for the two chosen.") from None
    series = SIMPSONpoint_series(paths, x=np.arange(len(paths), dtype=float), kind=kind)
    problem = check_expression(series, expression)
    if problem:
        raise ValueError(f"{expression!r}: {problem}")
    z = SIMPSONcombine_points(series, expression, part)
    grid = scan_grid(xv, yv, z, xname, yname, expression)
    grid["paths"] = paths
    return grid


def SIMPSONscan_table(grid: dict):
    """The grid as a long table (one row per cell that has a value)."""
    import pandas as pd
    jj, ii = np.nonzero(~np.isnan(grid["Z"]))
    return pd.DataFrame({grid["xname"]: grid["xs"][ii], grid["yname"]: grid["ys"][jj],
                         grid["zname"]: grid["Z"][jj, ii]})


def SIMPSONplot_heatmap(grid: dict, ax=None, xlabel: str | None = None,
                        ylabel: str | None = None, zlabel: str | None = None,
                        cmap: str = "viridis", vmin: float | None = None,
                        vmax: float | None = None, pl_width: float = 3.375,
                        pl_height: float = 2.5, font: str = "Arial", fontsize: float = 12,
                        save_fig: bool = False, fig_name: str = "simpson_heatmap",
                        save_format: str = "png"):
    """A :func:`scan_grid` as a heatmap: one cell per simulated combination,
    centred on its values (uneven steps are drawn as they are), a colour bar
    for z. Empty cells (no simulation) are left blank. ``vmin``/``vmax`` fix
    the colour range (e.g. to compare maps). Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from ..plotting.labels import italicise_symbols

    if ax is None:
        from .plotting import setup_plotting
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                               pl_height=pl_height)
        fig, ax = plt.subplots()
        fig.set_figwidth(style["pl_width"])
        fig.set_figheight(style["pl_height"])
    else:
        fig = ax.figure
    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError("The colour-scale minimum must be below the maximum.")
    xs, ys, Z = grid["xs"], grid["ys"], np.ma.masked_invalid(grid["Z"])
    if xs.size == 1 or ys.size == 1:
        raise ValueError("Both swept quantities need at least two values for a heatmap.")
    mesh = ax.pcolormesh(xs, ys, Z, cmap=cmap, shading="nearest",
                         norm=Normalize(vmin=vmin, vmax=vmax), rasterized=True)
    bar = fig.colorbar(mesh, ax=ax)
    bar.set_label(italicise_symbols(zlabel if zlabel is not None else grid["zname"]))
    ax.set_xlabel(italicise_symbols(xlabel if xlabel is not None else grid["xname"]))
    ax.set_ylabel(italicise_symbols(ylabel if ylabel is not None else grid["yname"]))
    if not getattr(fig, "_opennmr_managed", False):
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 -- layout is cosmetic
            pass
    if save_fig:
        from .plotting import save_figure
        save_figure(fig, fig_name=fig_name, save_format=save_format)
    return fig, ax
