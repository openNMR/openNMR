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
            raise ValueError(f"x has {x_values.size} values but {len(paths)} files were given -- "
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
