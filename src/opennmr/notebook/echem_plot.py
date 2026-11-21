"""
Electrochemistry beside NMR spectra: operando series, ex situ samples, or fitted values.

One figure, three ways of relating the cell to the spectra (``mode`` of :func:`link_table`):

* **operando**: a series of spectra recorded while the cell cycled. The user says when the first
  began on the cycler's clock and how long each took (or gives each start time); the order of
  the experiments is theirs to get right, nothing is synchronised automatically.
* **ex situ**: one spectrum per sample, each placed on the curve from one value (usually its
  voltage) plus the cycle and the direction (charge or discharge) it was stopped in.
* **values**: numbers taken from the spectra (fitted shifts, areas, linewidths …) drawn against
  the same cell quantity.

The cell can sit **left, right, above or below** the spectra and can be drawn against **any
column of the file** (voltage, time, current, capacity, composition x, a temperature column …).
When the cell's axis and the spectra's position quantity are the same (time vertical on both, say)
the two share that axis, so a row of the heat map lines up with the point on the curve.
The crosshair is linked: hovering over the cell highlights the spectrum recorded there, and
hovering over a spectrum marks its point on the cell.

Notebook::

    from opennmr.notebook import echem, echem_plot as ep
    cell = echem.load_echem("cell.mpr", mass_mg=5.2, formula="NaMnO2", x0=1.0)
    starts, ends = echem.operando_times(len(spectra), start_s=600, per_spectrum_s=1800)
    link = ep.link_table(cell["data"], "operando", starts_s=starts, ends_s=ends)
    ep.plot_echem_nmr(cell["data"], ppm, matrix, link, x="voltage_V", y="time_h",
                      side="right", display="heatmap", position="time_h")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import echem

SIDES = ("left", "right", "top", "bottom")
DISPLAYS = ("stacked", "superimposed", "heatmap", "values")
POSITIONS_NOTE = "index"


# ============================================================
# Which experiments
# ============================================================

def experiment_numbers(start: int, stop: int, step: int = 1) -> list[int]:
    """``start, start + step, … ≤ stop`` (``stop`` included), so a long series is a range rather
    than a typed list. A negative step counts down."""
    start, stop, step = int(start), int(stop), int(step)
    if step == 0:
        raise ValueError("The step must not be zero.")
    if (stop - start) * step < 0:
        raise ValueError(f"Cannot go from {start} to {stop} in steps of {step}.")
    return list(range(start, stop + (1 if step > 0 else -1), step))


def experiment_number(path) -> int | None:
    """The number of a Bruker experiment from its folder path: ``…/sample/12`` or
    ``…/sample/12/pdata/1`` give 12. ``None`` when the path has no such number."""
    if not path:
        return None
    parts = [p for p in str(path).replace("\\", "/").split("/") if p]
    if "pdata" in parts:
        parts = parts[:parts.index("pdata")]
    return int(parts[-1]) if parts and parts[-1].isdigit() else None


# ============================================================
# Relating spectra to the cell
# ============================================================

def _interpolated_row(df: pd.DataFrame, t: float) -> dict:
    """Every numeric column of ``df`` at time ``t`` (s): linear, with the integer state columns
    taken from the nearer sample."""
    numeric = df.select_dtypes(include="number")
    times = df["time_s"].to_numpy(float)
    order = np.argsort(times, kind="stable")
    times = times[order]
    if t < times[0] or t > times[-1]:
        return {c: np.nan for c in numeric.columns}
    row = {}
    for c in numeric.columns:
        values = numeric[c].to_numpy(float)[order]
        if c in ("cycle", "half_cycle"):
            row[c] = float(values[int(np.argmin(np.abs(times - t)))])
        else:
            row[c] = float(np.interp(t, times, values))
    return row


def link_table(df: pd.DataFrame, mode: str, starts_s=None, ends_s=None, points=None,
               quantity: str = "voltage_V", n: int | None = None) -> pd.DataFrame:
    """One row per spectrum: where it sits on the cell data, with every numeric column.

    ``mode="operando"``: ``starts_s`` and ``ends_s`` (cycler clock, see
    :func:`opennmr.notebook.echem.operando_times`); the row is the cell at the **middle of the
    acquisition**. ``mode="ex_situ"``: ``points``, a list of dicts as for
    :func:`opennmr.notebook.echem.ex_situ_points` (``value``, ``cycle``, ``half`` …);
    a sample that cannot be placed gets NaN and a note in ``attrs["notes"]``.
    Always has ``spectrum`` (0, 1, 2 …) and ``index`` (1, 2, 3 …)."""
    if mode == "operando":
        if starts_s is None or ends_s is None:
            raise ValueError("Operando needs the start and end time of every spectrum.")
        if "time_s" not in df:
            raise ValueError("The cell data have no time column.")
        starts, ends = np.asarray(starts_s, float), np.asarray(ends_s, float)
        rows = [_interpolated_row(df, 0.5 * (a + b)) for a, b in zip(starts, ends)]
        table = pd.DataFrame(rows)
        table["start_s"], table["end_s"] = starts, ends
        table["mid_s"] = 0.5 * (starts + ends)
        outside = int(table["time_s"].isna().sum())
        table.attrs["notes"] = ([f"{outside} spectrum(s) lie outside the time recorded by the "
                                 "cell: check the start time."] if outside else [])
    elif mode == "ex_situ":
        if not points:
            raise ValueError("Ex situ needs at least one sample.")
        placed = echem.ex_situ_points(df, points, quantity=quantity)
        columns = list(df.select_dtypes(include="number").columns)
        rows, notes = [], []
        for item in placed:
            if "error" in item:
                rows.append({c: np.nan for c in columns})
                notes.append(f"{item['label']}: {item['error']}")
            else:
                rows.append({c: item["row"].get(c, np.nan) for c in columns})
                notes.extend(f"{item['label']}: {m}" for m in item["notes"])
        table = pd.DataFrame(rows)
        table["label"] = [p["label"] for p in placed]
        table.attrs["notes"] = notes
    else:
        raise ValueError("mode must be 'operando' or 'ex_situ'.")
    table.insert(0, "spectrum", np.arange(len(table)))
    table.insert(1, "index", np.arange(1, len(table) + 1))
    if n is not None and len(table) != n:
        raise ValueError(f"{len(table)} positions on the cell for {n} spectra.")
    return table


def common_axis(spectra) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The shift axis (ppm) of the first spectrum and a matrix with one row per spectrum, the
    others interpolated onto that axis. Returns ``(ppm, matrix, notes)``."""
    spectra = list(spectra)
    if not spectra:
        raise ValueError("No spectra given.")
    for s in spectra:
        if s.ndim != 1:
            raise ValueError("Only 1D spectra can be shown beside the cell (keep a slice first).")
    ppm = np.asarray(spectra[0].axes[0].values("ppm"), float)
    rows, notes = [], []
    for k, s in enumerate(spectra):
        x = np.asarray(s.axes[0].values("ppm"), float)
        y = np.asarray(s.data, float)
        if x.size == ppm.size and np.allclose(x, ppm):
            rows.append(y)
        else:
            order = np.argsort(x)
            rows.append(np.interp(ppm, x[order], y[order], left=np.nan, right=np.nan))
            if k:
                notes.append(f"Spectrum {k + 1} was interpolated onto the axis of the first.")
    return ppm, np.vstack(rows), notes[:1] if notes else []


# ============================================================
# The figure
# ============================================================

def _layout(figure, side: str, fraction: float, gap: float):
    """``(ax_spectra, ax_cell)`` on the chosen sides. The spectra Axes is always made first, so it
    is ``figure.axes[0]`` whichever side the cell is on: the canvas treats the first Axes as the
    plot's main one (zoom, axis limits, Measure), and it is the spectra, not the cell."""
    from matplotlib.gridspec import GridSpec
    fraction = min(max(float(fraction), 0.1), 0.7)
    if side in ("left", "right"):
        cell_first = side == "left"
        ratios = [fraction, 1 - fraction] if cell_first else [1 - fraction, fraction]
        gs = GridSpec(1, 2, figure=figure, width_ratios=ratios, wspace=gap)
        spectra_slot, cell_slot = (gs[0, 1], gs[0, 0]) if cell_first else (gs[0, 0], gs[0, 1])
    else:
        cell_first = side == "top"
        ratios = [fraction, 1 - fraction] if cell_first else [1 - fraction, fraction]
        gs = GridSpec(2, 1, figure=figure, height_ratios=ratios, hspace=gap)
        spectra_slot, cell_slot = (gs[1, 0], gs[0, 0]) if cell_first else (gs[0, 0], gs[1, 0])
    ax_spectra = figure.add_subplot(spectra_slot)
    ax_cell = figure.add_subplot(cell_slot)
    return ax_spectra, ax_cell


def _positions(link: pd.DataFrame, position: str) -> np.ndarray:
    if position == "index":
        return link["index"].to_numpy(float)
    if position not in link.columns:
        raise ValueError(f"The cell data have no column {position!r} to place the spectra by.")
    return link[position].to_numpy(float)


def _edges(values: np.ndarray) -> np.ndarray:
    """Cell edges around sorted centres (for a heat map with uneven spacing)."""
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    mid = 0.5 * (values[1:] + values[:-1])
    return np.concatenate(([values[0] - (mid[0] - values[0])], mid,
                           [values[-1] + (values[-1] - mid[-1])]))


class LinkedCrosshair:
    """The crosshair shared by the cell and the spectra (see the module notes).

    ``update_from_cell(x, y)`` and ``update_from_spectra(position)`` are what the mouse handlers
    call, and can be called directly (a notebook without an interactive backend, a test)."""

    def __init__(self, figure, ax_cell, ax_spectra, cell_x, cell_y, curve_time, link_time,
                 link_x, link_y, positions, transposed: bool, colour: str = "0.35"):
        self.figure, self.ax_cell, self.ax_spectra = figure, ax_cell, ax_spectra
        self.cell_x, self.cell_y = np.asarray(cell_x, float), np.asarray(cell_y, float)
        self.curve_time = None if curve_time is None else np.asarray(curve_time, float)
        self.link_time = None if link_time is None else np.asarray(link_time, float)
        self.link_x, self.link_y = np.asarray(link_x, float), np.asarray(link_y, float)
        self.positions = np.asarray(positions, float)
        self.transposed = transposed
        self.index: int | None = None
        style = dict(color=colour, linewidth=0.8, linestyle="--", zorder=5, visible=False)
        self._cell_v = ax_cell.axvline(0, **style)
        self._cell_h = ax_cell.axhline(0, **style)
        self._cell_dot, = ax_cell.plot([], [], "o", color=colour, markersize=6, zorder=6,
                                       markerfacecolor="none", visible=False)
        self._spec_line = (ax_spectra.axvline(0, **style) if transposed
                           else ax_spectra.axhline(0, **style)) if ax_spectra is not None else None
        self._cids: list = []

    # -- state -----------------------------------------------------------
    def show(self, index: int | None) -> None:
        self.index = index
        visible = index is not None and np.isfinite(self.link_x[index]) if index is not None else False
        if visible:
            x, y = self.link_x[index], self.link_y[index]
            self._cell_v.set_xdata([x, x])
            self._cell_h.set_ydata([y, y])
            self._cell_dot.set_data([x], [y])
            if self._spec_line is not None:
                p = self.positions[index]
                (self._spec_line.set_xdata if self.transposed else self._spec_line.set_ydata)([p, p])
        for artist in (self._cell_v, self._cell_h, self._cell_dot, self._spec_line):
            if artist is not None:
                artist.set_visible(bool(visible))

    def update_from_cell(self, x: float, y: float) -> int | None:
        """The spectrum nearest the point (x, y) on the cell's axes, shown; its index."""
        index = None
        if self.curve_time is not None and self.link_time is not None and self.cell_x.size:
            ax = self.ax_cell
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            dx = (self.cell_x - x) / (abs(xlim[1] - xlim[0]) or 1.0)
            dy = (self.cell_y - y) / (abs(ylim[1] - ylim[0]) or 1.0)
            distance = np.where(np.isfinite(dx + dy), dx * dx + dy * dy, np.inf)
            if np.isfinite(distance.min()):
                t = self.curve_time[int(np.argmin(distance))]
                gaps = np.abs(self.link_time - t)
                if np.isfinite(gaps).any():
                    index = int(np.nanargmin(gaps))
        elif self.link_x.size:
            ax = self.ax_cell
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            dx = (self.link_x - x) / (abs(xlim[1] - xlim[0]) or 1.0)
            dy = (self.link_y - y) / (abs(ylim[1] - ylim[0]) or 1.0)
            distance = dx * dx + dy * dy
            if np.isfinite(distance).any():
                index = int(np.nanargmin(distance))
        self.show(index)
        return index

    def update_from_spectra(self, position: float) -> int | None:
        """The spectrum nearest ``position`` along the spectra's position axis, shown."""
        if not np.isfinite(self.positions).any():
            return None
        index = int(np.nanargmin(np.abs(self.positions - position)))
        self.show(index)
        return index

    # -- mouse ------------------------------------------------------------
    def connect(self) -> None:
        canvas = self.figure.canvas
        self._cids = [canvas.mpl_connect("motion_notify_event", self._on_move),
                      canvas.mpl_connect("axes_leave_event", self._on_leave)]

    def disconnect(self) -> None:
        for cid in self._cids:
            self.figure.canvas.mpl_disconnect(cid)
        self._cids = []

    def _on_move(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        if event.inaxes is self.ax_cell:
            self.update_from_cell(event.xdata, event.ydata)
        elif event.inaxes is self.ax_spectra:
            self.update_from_spectra(event.xdata if self.transposed else event.ydata)
        else:
            return
        self.figure.canvas.draw_idle()

    def _on_leave(self, _event) -> None:
        if self.index is not None:
            self.show(None)
            self.figure.canvas.draw_idle()


def _drop_previous_crosshair(figure) -> None:
    previous = getattr(figure, "_echem_crosshair", None)
    if previous is not None:
        previous.disconnect()
        figure._echem_crosshair = None


DISTRIBUTE = ("position", "even", "intensity")


def scaling_factors(spectra, per_scan: bool = False, divide_rg: bool = False, masses_mg=None,
                    factors=None) -> tuple[np.ndarray, list[str]]:
    """One multiplier per spectrum for putting a series on a common footing.

    ``per_scan``: divide by the number of scans (data already per scan, as Bruker's are on
    loading, are left alone); ``divide_rg``: divide by the receiver gain; ``masses_mg``: a
    sample mass (mg) per spectrum, or one number for all, to divide by (a spectrum with no mass
    raises ``ValueError``); ``factors``: a number per spectrum (or one for all) to multiply
    by afterwards, to bring one spectrum up or down. Returns ``(multipliers, notes)``."""
    from . import intensity_scaling as isc
    spectra = list(spectra)
    n = len(spectra)
    out, notes = np.ones(n), []
    if per_scan or divide_rg:
        for k, s in enumerate(spectra):
            info = isc.comparison_factors(s, "per_scan" if per_scan else "max", divide_rg)
            if not per_scan:       # "max" only makes data per scan: undo that, keep the RG part
                divisor, _ = isc.scan_divisor(s)
                info["factor"] *= divisor
            out[k] *= info["factor"]
            notes.extend(f"Spectrum {k + 1}: {m}" for m in info["notes"])
    if masses_mg is not None:
        masses = np.broadcast_to(np.asarray(masses_mg, dtype=float), (n,)) if np.ndim(masses_mg) == 0 \
            else np.asarray(masses_mg, dtype=float)
        if masses.size != n:
            raise ValueError(f"{masses.size} masses for {n} spectra.")
        bad = [k + 1 for k, m in enumerate(masses) if not np.isfinite(m) or m <= 0]
        if bad:
            raise ValueError("Every spectrum needs a sample mass (mg) to be divided by it: missing "
                             f"for spectrum {bad[0]}.")
        out /= masses
    if factors is not None:
        f = np.broadcast_to(np.asarray(factors, dtype=float), (n,)) if np.ndim(factors) == 0 \
            else np.asarray(factors, dtype=float)
        if f.size != n:
            raise ValueError(f"{f.size} factors for {n} spectra.")
        out *= np.where(np.isfinite(f), f, 1.0)
    return out, notes


def _auto_colours(n: int, colormap: str) -> list:
    import matplotlib
    cmap = matplotlib.colormaps[colormap]
    return [cmap(0.05 + 0.85 * (k / max(n - 1, 1))) for k in range(n)]


def _shift_label(nucleus) -> str:
    """The x label of a spectrum, as everywhere else in the package: ``δ ²³Na (ppm)`` with
    upright superscripts (mathtext's default would italicise the mass number)."""
    if not nucleus:
        return r"$\delta\ \mathregular{(ppm)}$"
    from ..core.registry import NUC_label
    return NUC_label(str(nucleus), x_unit="ppm")


def _tick_mode(axis, which: str, mode: str) -> None:
    """``"both"``: leave as drawn; ``"ticks"``: marks without numbers; ``"none"``: neither."""
    if mode not in ("both", "ticks", "none"):
        raise ValueError("The tick setting must be 'both', 'ticks' or 'none'.")
    if mode == "both":
        return
    if which == "x":
        axis.tick_params(axis="x", which="both", labelbottom=False, labeltop=False)
        if mode == "none":
            axis.tick_params(axis="x", which="both", bottom=False, top=False)
    else:
        axis.tick_params(axis="y", which="both", labelleft=False, labelright=False)
        if mode == "none":
            axis.tick_params(axis="y", which="both", left=False, right=False)


def _point_label(name, linked: bool) -> str:
    """A point and its spectrum share a name (and so a colour in Plot Style's series list) when
    linked; otherwise the point is its own series."""
    return str(name) if linked else f"{name} (point)"


def _broken(part: pd.DataFrame, x: str, y: str):
    """The curve as arrays with a gap (NaN) wherever the half cycle changes, so the end of one
    half cycle is never joined to the start of the next (capacity restarts at zero each half
    cycle, which would draw a line straight across the plot)."""
    xs, ys = part[x].to_numpy(float), part[y].to_numpy(float)
    if "half_cycle" not in part or len(part) < 2:
        return xs, ys
    change = np.flatnonzero(np.diff(part["half_cycle"].to_numpy(float)) != 0) + 1
    if change.size == 0:
        return xs, ys
    return np.insert(xs, change, np.nan), np.insert(ys, change, np.nan)


def _hide_inner(axis, which: str) -> None:
    """The axis of a pair that shares an axis hides that axis's ticks, tick labels and name."""
    if which == "y":
        axis.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
        axis.set_ylabel("")
    else:
        axis.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=False)
        axis.set_xlabel("")


def plot_echem_nmr(df: pd.DataFrame, ppm, matrix, link: pd.DataFrame, labels=None, *,
                   x: str | None = None, y: str | None = None, side: str = "right",
                   display: str = "stacked", position: str = "index", differences: str | None = None,
                   values: pd.DataFrame | None = None, value_columns=None,
                   cycles=None, half: str = "both", colour_by_cycle: bool = False,
                   figure=None, fontsize: float = 12.0, cell_fraction: float = 0.3, gap: float = 0.05,
                   cell_colour: str = "tab:blue", cell_linewidth: float = 1.0,
                   show_cell_ticklabels: bool = True, show_spectra_ticklabels: bool = True,
                   cell_xlabel: str | None = None, cell_ylabel: str | None = None,
                   spectra_xlabel: str | None = None, spectra_ylabel: str | None = None,
                   show_points: bool | None = None, marker: str = "o", marker_size: float = 6.0,
                   marker_colour: str = "tab:red", label_points: bool = False,
                   colormap: str = "viridis", stack_height: float = 1.0, normalise: bool = False,
                   ppm_limits=None, crosshair: bool = True, connect_values: bool = True,
                   colourbar: bool = True, point_colours=None, spectrum_colours=None,
                   link_colours: bool = True, distribute: str = "position",
                   intensity_label: str = "Intensity (a.u.)", nucleus: str | None = None,
                   cell_xticks: str | None = None, cell_yticks: str | None = None,
                   spectra_xticks: str | None = None, spectra_yticks: str | None = None,
                   cell_top_spine: bool = False, cell_right_spine: bool = False,
                   spectra_top_spine: bool = False, spectra_right_spine: bool = False,
                   spectra_left_spine: bool = True):
    """The cell beside the spectra (or beside fitted ``values``). Returns the figure; the linked
    crosshair is ``figure._echem_crosshair``.

    ``x``, ``y``: the cell's axes (any column; defaults: voltage against time for ``left``/``right``,
    time against voltage for ``top``/``bottom``). ``position``: which cell quantity places each
    spectrum (``"index"`` or a column such as ``"time_h"``, ``"x"``, ``"voltage_V"``). ``display``:
    ``"stacked"``, ``"superimposed"``, ``"heatmap"``, or ``"values"`` (``values``, a table with one
    row per spectrum, drawn against ``position``). ``differences``: ``"successive"``, ``"from_first"``
    or ``"from_last"`` plots differences between spectra. ``cycles`` / ``half`` / ``colour_by_cycle``
    choose what part of the cell curve is drawn.

    ``distribute``: ``"position"`` places each spectrum where its ``position`` value says (the
    spectra then share that axis with the cell); ``"even"`` gives every spectrum the same space, in
    the order given (the natural choice for ex situ samples), and ``"intensity"`` gives each a space
    as tall as its own maximum; both label the spectra by name. ``point_colours`` /
    ``spectrum_colours``: one colour per spectrum (``None`` for automatic); ``link_colours`` makes
    each spectrum take its point's colour. Several axis styles are options so the figure needs no
    editing: tick labels off, axis names, colours, markers.

    ``spectrum_colours`` that are given always win; ``link_colours`` only fills the blanks with
    the point's colour. ``nucleus`` (``"23Na"``) names the shift axis the way every spectrum is labelled, ``δ ²³Na (ppm)``. ``cell_xticks`` … ``spectra_yticks``: ``"both"`` (ticks and labels), ``"ticks"``
    (no labels) or ``"none"``, for each axis separately (the older ``show_*_ticklabels`` apply to
    both axes of a plot when these are ``None``). ``*_top_spine`` / ``*_right_spine`` add the
    closing lines of each plot (with their ticks). ``spectra_left_spine=False`` removes the spectra's
    left line, with its ticks and numbers. ``gap`` is the space between the two plots (a fraction of
    the plot size)."""
    import matplotlib
    import matplotlib.pyplot as plt
    from ..plotting.style import apply_spine_style, apply_tick_style

    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}.")
    if display not in DISPLAYS:
        raise ValueError(f"display must be one of {DISPLAYS}.")
    if distribute not in DISTRIBUTE:
        raise ValueError(f"distribute must be one of {DISTRIBUTE}.")
    vertical = side in ("left", "right")
    x = x or ("voltage_V" if vertical else "time_h")
    y = y or ("time_h" if vertical else "voltage_V")
    for column in (x, y):
        if column not in df.columns:
            raise ValueError(f"The cell data have no column {column!r}. Available: "
                             f"{', '.join(echem.plottable_columns(df))}.")
    if display != "values":
        matrix = np.asarray(matrix, dtype=float)
        ppm = np.asarray(ppm, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(link):
            raise ValueError(f"{matrix.shape[0] if matrix.ndim == 2 else '?'} spectra for "
                             f"{len(link)} positions on the cell.")
    elif values is None or len(values) != len(link):
        raise ValueError("'values' needs a table with one row per spectrum.")
    labels = list(labels) if labels is not None else [f"{i}" for i in link["index"]]
    link = link.reset_index(drop=True)
    n_all = len(link)
    auto = _auto_colours(n_all, colormap)
    pcols = [(point_colours[k] if point_colours is not None and k < len(point_colours)
              and point_colours[k] else None) for k in range(n_all)]
    scols = [(spectrum_colours[k] if spectrum_colours is not None and k < len(spectrum_colours)
              and spectrum_colours[k] else None) for k in range(n_all)]
    if differences is not None and display != "values":
        matrix = echem.spectrum_differences(matrix, differences)
        if differences == "successive":
            matrix = matrix[1:]
            link, labels = link.iloc[1:].reset_index(drop=True), labels[1:]
            pcols, scols, auto = pcols[1:], scols[1:], auto[1:]
    n = len(link)
    point_cols = [c or marker_colour for c in pcols]
    spec_cols = [scols[k] or (point_cols[k] if link_colours else None) for k in range(n)]
    shift_label = _shift_label(nucleus)
    positions = _positions(link, position)
    if display != "values" and not (distribute != "position" and display in ("stacked", "heatmap")):
        if not np.isfinite(positions).any():
            notes = list(link.attrs.get("notes", []))
            raise ValueError("No spectrum could be placed on the cell"
                             + (": " + "; ".join(notes[:3]) if notes else "."))
    if show_points is None:
        show_points = "label" in link.columns
    if figure is None:
        figure = plt.figure(figsize=(3.375 * 2.0, 3.0 * 1.6) if vertical else (3.375 * 1.4, 3.0 * 2.0))
    else:
        figure.clear()
    _drop_previous_crosshair(figure)
    ax_s, ax_c = _layout(figure, side, cell_fraction, gap)

    # -- the cell --------------------------------------------------------------
    curve = echem.select_cycles(df, cycles, half) if (cycles is not None or half != "both") else df
    ax_c.set_axisbelow(False)
    if colour_by_cycle and "cycle" in curve:
        numbers = sorted(curve["cycle"].unique())
        palette = matplotlib.colormaps["viridis"](np.linspace(0.05, 0.9, max(len(numbers), 1)))
        for colour, number in zip(palette, numbers):
            part = curve[curve["cycle"] == number]
            xs, ys = _broken(part, x, y)
            ax_c.plot(xs, ys, color=colour, linewidth=cell_linewidth, label=f"cycle {int(number)}")
    else:
        xs, ys = _broken(curve, x, y)
        ax_c.plot(xs, ys, color=cell_colour, linewidth=cell_linewidth, label="cell")
    if show_points:
        for k in range(n):
            if np.isfinite(link[x].iloc[k]) and np.isfinite(link[y].iloc[k]):
                ax_c.plot([link[x].iloc[k]], [link[y].iloc[k]], linestyle="none", marker=marker,
                          markersize=marker_size, color=point_cols[k], zorder=4,
                          label=_point_label(labels[k], link_colours))
                if label_points:
                    ax_c.annotate(str(labels[k]), (link[x].iloc[k], link[y].iloc[k]),
                                  textcoords="offset points", xytext=(4, 4), fontsize=fontsize * 0.7)
    ax_c.set_xlabel(cell_xlabel if cell_xlabel is not None else echem.axis_label(x), fontsize=fontsize)
    ax_c.set_ylabel(cell_ylabel if cell_ylabel is not None else echem.axis_label(y), fontsize=fontsize)

    # -- the spectra -----------------------------------------------------------
    spaced = distribute != "position" and display in ("stacked", "heatmap")
    transposed = display == "heatmap" and not vertical
    pos_label = "Spectrum" if position == "index" else echem.axis_label(position)
    spec_pos = positions.copy()           # where each spectrum is drawn (also for the crosshair)
    colourbar_handle = None
    if display == "values":
        names = list(value_columns) if value_columns is not None else list(values.columns[:1])
        for name in names:
            vals = pd.to_numeric(values[name], errors="coerce").to_numpy(float)
            xs, ys = (vals, positions) if vertical else (positions, vals)
            ax_s.plot(xs, ys, marker="o", markersize=4, linestyle="-" if connect_values else "none",
                      label=str(name))
        if len(names) > 1:
            ax_s.legend(frameon=False, fontsize=fontsize * 0.75)
        value_label = ", ".join(map(str, names))
        ax_s.set_xlabel(spectra_xlabel if spectra_xlabel is not None
                        else (value_label if vertical else pos_label), fontsize=fontsize)
        ax_s.set_ylabel(spectra_ylabel if spectra_ylabel is not None
                        else (pos_label if vertical else value_label), fontsize=fontsize)
    else:
        diverging = differences is not None
        peaks = np.nanmax(np.abs(matrix), axis=1)
        peaks = np.where(np.isfinite(peaks) & (peaks > 0), peaks, 1.0)
        overall = float(np.nanmax(peaks)) or 1.0
        if spaced:
            band = peaks / overall if distribute == "intensity" else np.ones(n)
            heights = stack_height * band
            pad = 0.15 * stack_height if distribute == "intensity" else 0.0
            offsets = np.concatenate(([0.0], np.cumsum(heights + pad)[:-1]))
            if distribute == "even":
                offsets = np.arange(n) * stack_height * 1.0
            spec_pos = offsets + (0.0 if display == "stacked" else 0.5 * heights)
            order = np.arange(n)
            pos_sorted, mat = spec_pos, matrix
        else:
            order = np.argsort(np.where(np.isfinite(positions), positions, np.inf), kind="stable")
            finite = np.isfinite(positions[order])
            order = order[finite]
            pos_sorted, mat = positions[order], matrix[order]
        if display == "heatmap":
            limit = np.nanmax(np.abs(mat)) if diverging else None
            kwargs = dict(cmap="RdBu_r", vmin=-limit, vmax=limit) if diverging else dict(cmap=colormap)
            if transposed:
                mesh = ax_s.pcolormesh(pos_sorted, ppm, mat.T, shading="nearest", **kwargs)
            else:
                mesh = ax_s.pcolormesh(ppm, pos_sorted, mat, shading="nearest", **kwargs)
            colourbar_handle = mesh
        elif display == "stacked":
            if spaced:
                scale = stack_height / overall
            else:
                spacing = np.median(np.diff(pos_sorted)) if pos_sorted.size > 1 else 1.0
                scale = stack_height * spacing / overall
            for k, p in zip(order, pos_sorted):
                ax_s.plot(ppm, p + scale * matrix[k], color=spec_cols[k] or "black", linewidth=0.8,
                          label=str(labels[k]))
        else:
            cmap = matplotlib.colormaps[colormap]
            span = (pos_sorted.max() - pos_sorted.min()) or 1.0 if pos_sorted.size else 1.0
            for k, p in zip(order, pos_sorted):
                colour = spec_cols[k] or cmap((p - pos_sorted.min()) / span)
                ax_s.plot(ppm, matrix[k], color=colour, linewidth=0.8, label=str(labels[k]))
            if not any(spec_cols):
                colourbar_handle = matplotlib.cm.ScalarMappable(
                    norm=matplotlib.colors.Normalize(pos_sorted.min(), pos_sorted.max()), cmap=cmap)
        shown_ppm = ppm_limits if ppm_limits is not None else (float(np.nanmax(ppm)), float(np.nanmin(ppm)))
        axis_name = "Spectrum" if spaced else pos_label
        if transposed:
            ax_s.set_ylim(*shown_ppm)
            ax_s.set_xlabel(spectra_xlabel if spectra_xlabel is not None else axis_name, fontsize=fontsize)
            ax_s.set_ylabel(spectra_ylabel if spectra_ylabel is not None else shift_label,
                            fontsize=fontsize)
        else:
            ax_s.set_xlim(*shown_ppm)
            ax_s.set_xlabel(spectra_xlabel if spectra_xlabel is not None else shift_label,
                            fontsize=fontsize)
            ax_s.set_ylabel(spectra_ylabel if spectra_ylabel is not None else
                            (intensity_label if display == "superimposed" else axis_name),
                            fontsize=fontsize)
        if spaced:
            ticks, names = list(spec_pos), [str(t) for t in labels]
            if transposed:
                ax_s.set_xticks(ticks)
                ax_s.set_xticklabels(names, rotation=45, ha="right")
            else:
                ax_s.set_yticks(ticks)
                ax_s.set_yticklabels(names)
        if colourbar_handle is not None and colourbar:
            bar = figure.colorbar(colourbar_handle, ax=ax_s, pad=0.02, fraction=0.05)
            bar.set_label(pos_label if display == "superimposed" else
                          ("Difference" if diverging else intensity_label), fontsize=fontsize)
            bar.ax.tick_params(labelsize=fontsize * 0.8)

    # -- share the axis the spectra are placed by (never when they are evenly spaced) ----
    shared = False
    if display in ("stacked", "heatmap", "values") and position != "index" and not spaced:
        if vertical and y == position:
            ax_c.sharey(ax_s)
            _hide_inner(ax_c if side == "right" else ax_s, "y")
            shared = True
        elif not vertical and x == position:
            ax_c.sharex(ax_s)
            _hide_inner(ax_c if side == "top" else ax_s, "x")
            shared = True

    for axis, top, right, left in ((ax_c, cell_top_spine, cell_right_spine, True),
                                   (ax_s, spectra_top_spine, spectra_right_spine, spectra_left_spine)):
        apply_spine_style(axis, top=top, right=right, bottom=True, left=left)
        apply_tick_style(axis, labelsize=fontsize * 0.9)
        axis.tick_params(which="both", top=top, right=right, labeltop=False, labelright=False)
        if not left:               # the ticks and numbers go with the line
            axis.tick_params(axis="y", which="both", left=False, labelleft=False)
    if shared:           # apply_tick_style puts the inner ticks back: hide them again
        inner = (ax_c if side in ("right", "top") else ax_s)
        _hide_inner(inner, "y" if vertical else "x")
    for axis, xmode, ymode, legacy in ((ax_c, cell_xticks, cell_yticks, show_cell_ticklabels),
                                       (ax_s, spectra_xticks, spectra_yticks, show_spectra_ticklabels)):
        _tick_mode(axis, "x", xmode or ("both" if legacy else "ticks"))
        _tick_mode(axis, "y", ymode or ("both" if legacy else "ticks"))
    figure.link_notes = list(link.attrs.get("notes", [])) if hasattr(link, "attrs") else []

    # -- the linked crosshair ---------------------------------------------------
    figure._echem_crosshair = None
    if crosshair and n:
        curve_time = curve["time_s"].to_numpy(float) if "time_s" in curve else None
        link_time = None
        if "start_s" in link and "end_s" in link:
            link_time = 0.5 * (link["start_s"].to_numpy(float) + link["end_s"].to_numpy(float))
        cross = LinkedCrosshair(
            figure, ax_c, ax_s, curve[x].to_numpy(float), curve[y].to_numpy(float), curve_time,
            link_time if curve_time is not None else None, link[x].to_numpy(float),
            link[y].to_numpy(float), spec_pos, transposed)
        cross.connect()
        figure._echem_crosshair = cross
    figure.axes_cell, figure.axes_spectra = ax_c, ax_s
    return figure
