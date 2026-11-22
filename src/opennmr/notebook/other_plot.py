"""
Plot other data: any table of numbers as a 2-D or 3-D scatter, line or bar chart.

For the numbers that do not come from a spectrum (a fitted shift against temperature, a
capacity against cycle number, a column pasted from a spreadsheet). The figure is the same one
the GUI (Apply ▸ Plot other data) draws, so a notebook call reproduces it.

Notebook::

    from opennmr.notebook import other_plot as op
    table = op.read_table("shifts.csv")
    fig = op.plot_other_data(table, x="T_K", y=["delta_1", "delta_2"], kind="line_markers",
                             xlabel="$T$ (K)", ylabel="Shift (ppm)")
    fig3 = op.plot_other_data(table, x="T_K", y="delta_1", z="width_Hz", kind="scatter")

``kind`` is ``"scatter"``, ``"line"``, ``"line_markers"`` or ``"bar"``. Giving ``z`` makes the
figure three-dimensional (``y`` is then a single column). Columns are named from the header
row, or given as sequences of numbers.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd

KINDS = ("scatter", "line", "line_markers", "bar")
KIND_NAMES = {"scatter": "Scatter", "line": "Line", "line_markers": "Line with markers", "bar": "Bar chart"}


def read_table(source) -> pd.DataFrame:
    """A table of numbers from a csv/tsv/txt file (path) or from pasted text (a string with
    line breaks). The separator is found for you (comma, tab, semicolon or spaces); a first row
    that is not numbers is the header, otherwise the columns are called ``col1``, ``col2`` …"""
    text = os.fspath(source) if isinstance(source, os.PathLike) else str(source)
    if "\n" in text or not os.path.exists(text):
        if "\n" not in text and not text.strip():
            raise ValueError("There is no data to read.")
        if "\n" not in text:
            raise ValueError(f"No such file: {text}")
        handle = io.StringIO(text)
    else:
        handle = text
    def read(sep):
        src = io.StringIO(text) if isinstance(handle, io.StringIO) else handle
        return pd.read_csv(src, sep=sep, engine="python", skip_blank_lines=True, comment="#")
    try:
        try:
            df = read(None)
        except pd.errors.ParserError:       # the separator could not be guessed (one column)
            df = read(",")
    except Exception as exc:        # pandas raises several types for a malformed table
        raise ValueError(f"Could not read the table: {exc}") from None
    # A header row that is itself numbers means there was no header.
    names_numeric = all(_is_number(c) for c in df.columns)
    if names_numeric:
        first = [float(c) for c in df.columns]
        df.columns = [f"col{i + 1}" for i in range(df.shape[1])]
        df = pd.concat([pd.DataFrame([first], columns=df.columns), df], ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if df.shape[1] < 2:
        raise ValueError("The table needs at least two numeric columns.")
    return df.reset_index(drop=True)


def _is_number(value) -> bool:
    try:
        float(str(value).replace(",", "."))
        return True
    except ValueError:
        return False


def _column(table, ref, name: str):
    """``(values, label)`` for a column name, an integer position or a sequence of numbers."""
    if isinstance(ref, str):
        if table is None or ref not in table.columns:
            raise ValueError(f"No column called {ref!r} for {name}.")
        return np.asarray(table[ref], dtype=float), ref
    if isinstance(ref, (int, np.integer)) and table is not None:
        if not -table.shape[1] <= ref < table.shape[1]:
            raise ValueError(f"Column number {ref} for {name} is outside the table.")
        return np.asarray(table.iloc[:, ref], dtype=float), str(table.columns[ref])
    values = np.asarray(ref, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one column of numbers.")
    return values, ""


def plot_other_data(data=None, x=None, y=None, z=None, *, kind: str = "scatter", figure=None,
                    labels=None, colours=None, colour_by=None, colormap: str = "viridis",
                    xlabel: str | None = None, ylabel: str | None = None, zlabel: str | None = None,
                    title: str | None = None, fontsize: float = 12.0, marker: str = "o",
                    marker_size: float = 6.0, linewidth: float = 1.5, grid: bool = False,
                    legend: bool = True, bar_width: float = 0.8, elevation: float = 25.0,
                    azimuth: float = -60.0, invert_x: bool = False):
    """Draw ``y`` (one column, or several for a 2-D figure) against ``x``, and against ``z`` too
    for a 3-D figure. ``data`` is a DataFrame (or ``None`` when ``x``/``y``/``z`` are sequences).

    ``colour_by``: a column (or numbers) that colours the points of a scatter, with a colour bar.
    ``labels``: a name for each ``y`` series. ``colours``: one colour for each ``y`` series.
    Bars of a 3-D figure stand on the x, y grid with height ``z``. Returns the figure."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}.")
    if x is None or y is None:
        raise ValueError("Give the columns for x and y.")
    table = data if isinstance(data, pd.DataFrame) else (pd.DataFrame(data) if data is not None else None)
    xs, xname = _column(table, x, "x")
    multi = isinstance(y, (list, tuple)) and (len(y) == 0 or isinstance(y[0], (str, np.integer)) or
                                              np.ndim(y[0]) == 1)
    ys_ref = list(y) if multi else [y]
    if not ys_ref:
        raise ValueError("Choose at least one column for y.")
    three_d = z is not None
    if three_d and len(ys_ref) != 1:
        raise ValueError("A 3-D figure has one y column and one z column.")
    series = [_column(table, r, "y") for r in ys_ref]
    for values, _ in series:
        if len(values) != len(xs):
            raise ValueError("x and y must be the same length.")
    zs, zname = (_column(table, z, "z") if three_d else (None, ""))
    if three_d and len(zs) != len(xs):
        raise ValueError("x, y and z must be the same length.")
    cvals = None
    if colour_by is not None:
        cvals, cname = _column(table, colour_by, "colour")
        if len(cvals) != len(xs):
            raise ValueError("The colour column must be as long as x.")
    names = list(labels) if labels is not None else [n or f"y{i + 1}" for i, (_, n) in enumerate(series)]
    if len(names) != len(series):
        raise ValueError("Give one label for each y series.")
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colours = list(colours) if colours is not None else [cycle[i % len(cycle)] for i in range(len(series))]

    fig = figure if figure is not None else Figure(figsize=(6.0, 4.5))
    fig.clear()
    ax = fig.add_subplot(projection="3d") if three_d else fig.add_subplot()
    ok = np.isfinite(xs)
    if three_d:
        ys = series[0][0]
        ok &= np.isfinite(ys) & np.isfinite(zs)
        if kind == "bar":
            span = float(np.ptp(xs[ok])) / max(len(xs[ok]), 1) or 1.0
            ax.bar3d(xs[ok], ys[ok], np.zeros(ok.sum()), span * bar_width, span * bar_width, zs[ok],
                     color=colours[0], shade=True)
        else:
            if kind in ("line", "line_markers"):
                ax.plot(xs[ok], ys[ok], zs[ok], color=colours[0], linewidth=linewidth,
                        marker=marker if kind == "line_markers" else None, markersize=marker_size)
            else:
                colour_args = (dict(c=cvals[ok], cmap=colormap) if cvals is not None
                               else dict(color=colours[0]))
                sc = ax.scatter(xs[ok], ys[ok], zs[ok], s=marker_size ** 2, marker=marker,
                                **colour_args)
                if cvals is not None:
                    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=cname)
        ax.set_xlabel(xlabel if xlabel is not None else xname, fontsize=fontsize)
        ax.set_ylabel(ylabel if ylabel is not None else series[0][1], fontsize=fontsize)
        ax.set_zlabel(zlabel if zlabel is not None else zname, fontsize=fontsize)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.tick_params(labelsize=fontsize * 0.85)
    else:
        n = len(series)
        for i, (ys, _) in enumerate(series):
            good = ok & np.isfinite(ys)
            colour, name = colours[i], names[i]
            if kind == "scatter":
                if cvals is not None and i == 0:
                    sc = ax.scatter(xs[good], ys[good], s=marker_size ** 2, c=cvals[good],
                                    cmap=colormap, marker=marker, label=name)
                    fig.colorbar(sc, ax=ax, label=cname)
                else:
                    ax.scatter(xs[good], ys[good], s=marker_size ** 2, color=colour, marker=marker,
                               label=name)
            elif kind in ("line", "line_markers"):
                ax.plot(xs[good], ys[good], color=colour, linewidth=linewidth,
                        marker=marker if kind == "line_markers" else None, markersize=marker_size,
                        label=name)
            else:
                steps = np.unique(xs[good])
                gap = float(np.min(np.diff(steps))) if len(steps) > 1 else 1.0
                width = gap * bar_width / n
                ax.bar(xs[good] + (i - (n - 1) / 2) * width, ys[good], width=width, color=colour,
                       label=name)
        ax.set_xlabel(xlabel if xlabel is not None else xname, fontsize=fontsize)
        ylab = ylabel if ylabel is not None else (series[0][1] if n == 1 else "")
        ax.set_ylabel(ylab, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize * 0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if legend and n > 1:
            ax.legend(frameon=False, fontsize=fontsize * 0.85)
    if invert_x:
        ax.invert_xaxis()
    if grid:
        ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=fontsize * 1.1)
    return fig
