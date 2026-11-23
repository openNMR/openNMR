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

Several datasets on one figure (a profile and its fit, files from different samples)::

    sets = [op.dataset(t1, y1, label="sample A", kind="scatter"),
            op.dataset(t2, y2, label="sample B", kind="line_markers")]
    sets += op.curve_fit_datasets(spectrum)          # a Dynamics curve fit: points + fitted curve
    fig = op.plot_datasets(sets, xlabel="Delay (s)", ylabel="Intensity")

Each dataset has its own ``kind``, colour, marker and error bars; all are 2-D or all 3-D.

``skip_lines`` (in :func:`read_table`) leaves out that many lines at the top of the file or
pasted text before the header row is looked for -- instrument or software headers, titles,
comments. ``top_spine`` / ``right_spine`` (in :func:`plot_other_data`) add the closing top and
right lines of a 2-D plot, with their ticks; a 3-D figure has no spines, so they draw the
remaining edges of its box instead.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd

KINDS = ("scatter", "line", "line_markers", "bar")
KIND_NAMES = {"scatter": "Scatter", "line": "Line", "line_markers": "Line with markers", "bar": "Bar chart"}


def read_table(source, skip_lines: int = 0) -> pd.DataFrame:
    """A table of numbers from a csv/tsv/txt file (path) or from pasted text (a string with
    line breaks). The separator is found for you (comma, tab, semicolon or spaces); a first row
    that is not numbers is the header, otherwise the columns are called ``col1``, ``col2`` …

    ``skip_lines``: how many lines at the top of the file (or text) to leave out first -- a
    title, instrument settings, anything before the header row or the numbers. Blank lines
    count as lines."""
    skip_lines = int(skip_lines or 0)
    if skip_lines < 0:
        raise ValueError("The number of header lines to skip cannot be negative.")
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
        return pd.read_csv(src, sep=sep, engine="python", skip_blank_lines=True, comment="#",
                           skiprows=skip_lines or None, header=None, dtype=str)
    try:
        try:
            df = read(None)
        except pd.errors.ParserError:       # the separator could not be guessed (one column)
            df = read(",")
    except Exception as exc:        # pandas raises several types for a malformed table
        raise ValueError(f"Could not read the table: {exc}") from None
    # Read without a header so the first row is never altered (pandas renames repeated
    # header values -- a first data row "0  0" came back as 0 and 0.1, a point that is
    # not in the file). A first row that is not all numbers is the header.
    df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)
    first_row = list(df.iloc[0]) if len(df) else []
    if first_row and all(_is_number(v) for v in first_row if isinstance(v, str)):
        df.columns = [f"col{i + 1}" for i in range(df.shape[1])]
    else:
        df.columns = [str(v).strip() if isinstance(v, str) and v.strip() else f"col{i + 1}"
                      for i, v in enumerate(first_row)]
        df = df.iloc[1:]
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


def _draw_box_edges(ax) -> None:
    """The twelve edges of a 3-D axes' box, drawn as lines (a 3-D plot has no spines): the
    closing lines a 2-D plot gets from its top and right spines."""
    import itertools
    (x0, x1), (y0, y1), (z0, z1) = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    corners = list(itertools.product((x0, x1), (y0, y1), (z0, z1)))
    colour = ax.xaxis.line.get_color() if hasattr(ax.xaxis, "line") else "black"
    for a, b in itertools.combinations(corners, 2):
        if sum(p != q for p, q in zip(a, b)) == 1:           # corners sharing an edge
            ax.plot(*zip(a, b), color=colour, linewidth=0.8, zorder=0)
    ax.set_xlim3d(x0, x1)
    ax.set_ylim3d(y0, y1)
    ax.set_zlim3d(z0, z1)


def number_list(text) -> list[float]:
    """Numbers typed or pasted as a list: separated by commas, spaces, semicolons or new lines.
    ``start:stop:step`` expands to a range that includes ``stop`` (``0:10:2`` is 0, 2, … 10), and
    ``start:stop`` steps by 1. A sequence of numbers is returned as a list of floats.

    For data that is a single column, or that has no file: plot it against a list of your own."""
    import re
    if not isinstance(text, str):
        return [float(v) for v in text]
    out: list[float] = []
    for token in re.split(r"[\s,;]+", text.strip()):
        if not token:
            continue
        try:
            if ":" in token:
                parts = [float(p) for p in token.split(":")]
                if len(parts) not in (2, 3):
                    raise ValueError
                start, stop = parts[0], parts[1]
                step = parts[2] if len(parts) == 3 else 1.0
                if step == 0 or (stop - start) * step < 0:
                    raise ValueError
                count = int(np.floor((stop - start) / step + 1e-9)) + 1
                out.extend(float(v) for v in start + step * np.arange(count))
            else:
                out.append(float(token))
        except ValueError:
            raise ValueError(f"“{token}” is not a number or a start:stop:step range.") from None
    return out


def dataset(x, y, z=None, *, label: str | None = None, kind: str | None = None,
            colour: str | None = None, yerr=None, marker: str | None = None,
            linestyle: str | None = None, xname: str = "", yname: str = "", zname: str = "",
            colour_by=None, colour_name: str = "") -> dict:
    """One series for :func:`plot_datasets`: ``x`` and ``y`` (and ``z`` for a 3-D figure) as
    sequences of numbers.

    ``kind``: draw this one as ``scatter``/``line``/``line_markers``/``bar`` (the figure's
    ``kind`` when ``None``) -- a measured profile as points and its fit as a line, say.
    ``yerr``: error bars on a 2-D scatter or line. ``colour``, ``marker``, ``linestyle``:
    this series' own. ``xname``/``yname``/``zname``: what an axis is called when the figure
    is given no label."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    zs = None if z is None else np.asarray(z, dtype=float)
    if xs.ndim != 1 or ys.ndim != 1 or len(xs) != len(ys) or (zs is not None and len(zs) != len(xs)):
        raise ValueError("x, y (and z) of a dataset must be one column each, of the same length.")
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}.")
    err = None if yerr is None else np.asarray(yerr, dtype=float)
    if err is not None and err.shape != ys.shape:
        raise ValueError("yerr must be as long as y.")
    cvals = None if colour_by is None else np.asarray(colour_by, dtype=float)
    return {"x": xs, "y": ys, "z": zs, "label": label, "kind": kind, "colour": colour, "yerr": err,
            "marker": marker, "linestyle": linestyle, "xname": xname, "yname": yname, "zname": zname,
            "colour_by": cvals, "colour_name": colour_name}


def curve_fit_datasets(source, peak=None, *, points: bool = True, curve: bool = True,
                       label: str | None = None, colours=None, n_points: int = 400) -> list[dict]:
    """The datasets of a Dynamics curve fit (or of any list of fit results), ready for
    :func:`plot_datasets`: for each peak, its measured points (with error bars when the fit used
    them) and, if ``curve``, the fitted model as a smooth line in the same colour.

    ``source``: an open ``NMRSpectrum`` (its ``metadata.raw["relaxation_fit_results"]``), that
    ``raw`` dict, or the list of fit results itself. ``peak``: a peak number (from 0), a list of
    them, or ``None`` for all. ``label``: names the dataset (``"<label> <peak>"``)."""
    from ..processing import relaxation
    raw = getattr(getattr(source, "metadata", None), "raw", None)
    results = (raw if raw is not None else source)
    if isinstance(results, dict):
        results = results.get("relaxation_fit_results")
    if not isinstance(results, (list, tuple)) or not results:
        raise ValueError("There is no curve fit to take: fit the peak areas against the delays in "
                         "Dynamics first.")
    wanted = (range(len(results)) if peak is None
              else [peak] if isinstance(peak, (int, np.integer)) else list(peak))
    cycle = _cycle()
    out = []
    for n, k in enumerate(wanted):
        if not -len(results) <= k < len(results):
            raise ValueError(f"The curve fit has {len(results)} peak(s); there is no peak {k}.")
        result = results[k]
        if not isinstance(result, dict) or result.get("t") is None:
            raise ValueError(f"Peak {k} has no stored points to plot.")
        colour = (colours[n % len(colours)] if colours else cycle[n % len(cycle)])
        name = result.get("peak_label") or f"Peak {k + 1}"
        name = f"{label} {name}" if label else name
        t, y = np.asarray(result["t"], dtype=float), np.asarray(result["y"], dtype=float)
        order = np.argsort(t)
        err = result.get("y_err")
        if points:
            out.append(dataset(t[order], y[order], label=f"{name}" if not curve else f"{name} data",
                               kind="scatter", colour=colour, xname="Delay", yname="Intensity",
                               yerr=None if err is None else np.asarray(err, dtype=float)[order]))
        if curve:
            try:
                tf, yf = relaxation.fit_curve(result, n_points=n_points)
            except Exception:  # noqa: BLE001 -- no model left to evaluate: join the fitted points
                tf, yf = t[order], np.asarray(result.get("y_fit", y), dtype=float)[order]
            out.append(dataset(tf, yf, label=f"{name} fit" if points else name, kind="line",
                               colour=colour, xname="Delay", yname="Intensity"))
    return out


def _cycle() -> list:
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    # The style's cycle may hold RGB tuples (openNMR's own does): hex strings are what a
    # dataset's colour, a colour button and generated notebook code can all carry.
    return [to_hex(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]


def plot_datasets(datasets, *, kind: str = "scatter", figure=None, colormap: str = "viridis",
                  xlabel: str | None = None, ylabel: str | None = None, zlabel: str | None = None,
                  title: str | None = None, fontsize: float = 12.0, marker: str = "o",
                  marker_size: float = 6.0, linewidth: float = 1.5, grid: bool = False,
                  legend: bool = True, bar_width: float = 0.8, elevation: float = 25.0,
                  azimuth: float = -60.0, invert_x: bool = False,
                  top_spine: bool = False, right_spine: bool = False):
    """Several datasets (from :func:`dataset` / :func:`curve_fit_datasets`) on one 2-D or 3-D
    figure. A dataset with ``z`` makes the figure 3-D, and then every dataset needs one. Each
    is drawn as its own ``kind`` (else ``kind``), colour, marker and error bars; the legend
    names them. Bars are grouped side by side at each x. Returns the figure.

    ``kind``, ``marker`` and ``marker_size`` etc. are the figure's defaults; the axis labels
    default to those of the first dataset."""
    from matplotlib.figure import Figure

    sets = [d for d in (datasets or [])]
    if not sets:
        raise ValueError("There is nothing to plot: give at least one dataset.")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}.")
    three_d = sets[0].get("z") is not None
    if any((d.get("z") is not None) != three_d for d in sets):
        raise ValueError("Every dataset must be 2-D (x, y) or every one 3-D (x, y, z); they cannot "
                         "be mixed on one figure.")
    cycle = _cycle()
    fig = figure if figure is not None else Figure(figsize=(6.0, 4.5))
    fig.clear()
    ax = fig.add_subplot(projection="3d") if three_d else fig.add_subplot()
    n_bars = sum(1 for d in sets if (d.get("kind") or kind) == "bar")
    bar_index = 0
    named = 0
    for i, d in enumerate(sets):
        colour = d.get("colour") or cycle[i % len(cycle)]
        k = d.get("kind") or kind
        mk = d.get("marker") or marker
        name = d.get("label") or (d.get("yname") or f"y{i + 1}")
        named += 1
        xs, ys, zs = d["x"], d["y"], d.get("z")
        ok = np.isfinite(xs) & np.isfinite(ys) & (np.isfinite(zs) if zs is not None else True)
        cvals = d.get("colour_by")
        if three_d:
            if k == "bar":
                span = float(np.ptp(xs[ok])) / max(len(xs[ok]), 1) or 1.0
                ax.bar3d(xs[ok], ys[ok], np.zeros(ok.sum()), span * bar_width, span * bar_width,
                         zs[ok], color=colour, shade=True)
            elif k in ("line", "line_markers"):
                ax.plot(xs[ok], ys[ok], zs[ok], color=colour, linewidth=linewidth,
                        marker=mk if k == "line_markers" else None, markersize=marker_size,
                        linestyle=d.get("linestyle") or "-", label=name)
            else:
                colour_args = (dict(c=cvals[ok], cmap=colormap) if cvals is not None
                               else dict(color=colour))
                sc = ax.scatter(xs[ok], ys[ok], zs[ok], s=marker_size ** 2, marker=mk,
                                label=name, **colour_args)
                if cvals is not None:
                    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=d.get("colour_name") or "")
            continue
        err = d.get("yerr")
        if k == "scatter":
            if cvals is not None:
                sc = ax.scatter(xs[ok], ys[ok], s=marker_size ** 2, c=cvals[ok], cmap=colormap,
                                marker=mk, label=name)
                fig.colorbar(sc, ax=ax, label=d.get("colour_name") or "")
            elif err is not None:
                ax.errorbar(xs[ok], ys[ok], yerr=err[ok], fmt=mk, color=colour, markersize=marker_size,
                            capsize=2, elinewidth=linewidth * 0.7, label=name)
            else:
                ax.scatter(xs[ok], ys[ok], s=marker_size ** 2, color=colour, marker=mk, label=name)
        elif k in ("line", "line_markers"):
            if err is not None:
                ax.errorbar(xs[ok], ys[ok], yerr=err[ok], color=colour, linewidth=linewidth,
                            marker=mk if k == "line_markers" else None, markersize=marker_size,
                            capsize=2, linestyle=d.get("linestyle") or "-", label=name)
            else:
                ax.plot(xs[ok], ys[ok], color=colour, linewidth=linewidth,
                        marker=mk if k == "line_markers" else None, markersize=marker_size,
                        linestyle=d.get("linestyle") or "-", label=name)
        else:
            steps = np.unique(xs[ok])
            gap = float(np.min(np.diff(steps))) if len(steps) > 1 else 1.0
            width = gap * bar_width / max(n_bars, 1)
            ax.bar(xs[ok] + (bar_index - (n_bars - 1) / 2) * width, ys[ok], width=width,
                   color=colour, label=name,
                   **({"yerr": err[ok], "capsize": 2} if err is not None else {}))
            bar_index += 1
    first = sets[0]
    ax.set_xlabel(xlabel if xlabel is not None else first.get("xname", ""), fontsize=fontsize)
    one = len(sets) == 1
    ax.set_ylabel(ylabel if ylabel is not None else
                  (first.get("yname", "") if one or len({d.get("yname", "") for d in sets}) == 1 else ""),
                  fontsize=fontsize)
    ax.tick_params(labelsize=fontsize * 0.85)
    if three_d:
        ax.set_zlabel(zlabel if zlabel is not None else first.get("zname", ""), fontsize=fontsize)
        ax.view_init(elev=elevation, azim=azimuth)
    else:
        ax.spines["top"].set_visible(bool(top_spine))
        ax.spines["right"].set_visible(bool(right_spine))
        ax.tick_params(which="both", top=bool(top_spine), right=bool(right_spine),
                       labeltop=False, labelright=False)
    if legend and len(sets) > 1:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, fontsize=fontsize * 0.85)
    if invert_x:
        ax.invert_xaxis()
    if three_d and (top_spine or right_spine):
        _draw_box_edges(ax)
    if grid:
        ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=fontsize * 1.1)
    return fig


def plot_other_data(data=None, x=None, y=None, z=None, *, kind: str = "scatter", figure=None,
                    labels=None, colours=None, colour_by=None, colormap: str = "viridis",
                    xlabel: str | None = None, ylabel: str | None = None, zlabel: str | None = None,
                    title: str | None = None, fontsize: float = 12.0, marker: str = "o",
                    marker_size: float = 6.0, linewidth: float = 1.5, grid: bool = False,
                    legend: bool = True, bar_width: float = 0.8, elevation: float = 25.0,
                    azimuth: float = -60.0, invert_x: bool = False,
                    top_spine: bool = False, right_spine: bool = False):
    """Draw ``y`` (one column, or several for a 2-D figure) against ``x``, and against ``z`` too
    for a 3-D figure. ``data`` is a DataFrame (or ``None`` when ``x``/``y``/``z`` are sequences).

    ``colour_by``: a column (or numbers) that colours the points of a scatter, with a colour bar.
    ``labels``: a name for each ``y`` series. ``colours``: one colour for each ``y`` series.
    Bars of a 3-D figure stand on the x, y grid with height ``z``.
    ``top_spine`` / ``right_spine``: add the top and right lines of a 2-D plot (with ticks); in a
    3-D figure they draw the remaining edges of the box. Returns the figure."""
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
        ax.spines["top"].set_visible(bool(top_spine))
        ax.spines["right"].set_visible(bool(right_spine))
        ax.tick_params(which="both", top=bool(top_spine), right=bool(right_spine),
                       labeltop=False, labelright=False)
        if legend and n > 1:
            ax.legend(frameon=False, fontsize=fontsize * 0.85)
    if invert_x:
        ax.invert_xaxis()
    if three_d and (top_spine or right_spine):
        _draw_box_edges(ax)
    if grid:
        ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=fontsize * 1.1)
    return fig
