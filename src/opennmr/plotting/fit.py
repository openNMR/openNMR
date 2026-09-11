"""
Fit-result plotting: spectrum+fit overlay, per-slice integrals across a
series, and fitted relaxation/diffusion curves.

Ported from ``nmrtools2``'s ``fitting.py``/``two_d.py``/``relaxation.py``
(``NMR1dplot_with_fit``, ``NMR2D_plot_integrals``, ``NMR2D_plot_relaxation``)
-- the survey's own category A, flagged as the biggest concrete gap:
``opennmr.processing.fitting``/``relaxation`` produce numbers but had no
matching "look at my fit" plot helpers, so the GUI's own Analysis tab
built its plots by hand instead of calling a shared function.

``NMR2D_plot_slices`` (stack-plot extracted 2D rows/columns, with nQ-
crossing markers and text/legend label styles) and ``NMR2D_summary_plot``
(one figure combining a contour, a slice stack, and an integrals panel)
are deliberately **not** ported here -- both are substantially larger
than the three below, and their core capability (stacking extracted
slices with an optional fit overlay) already substantially overlaps
with the GUI's own working Compare-tab/pseudo-2D-stack rendering. Left
as a separate, explicit follow-up rather than silently dropped.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..core.registry import find_gamma, NUC_label
from ..processing.relaxation import RELAXATION_MODELS, format_time_value
from .style import (
    setup_plotting, apply_tick_style, apply_spine_style, save_figure, qualitative_colors,
)
# The 2D contour panel's own level spacing, reused rather than
# reimplemented: a DOSY map is a contour plot, and having its levels
# bunch differently from every other contour in the app is exactly the
# kind of quiet divergence that makes two plots of the same data look
# like different measurements.
from .twod import _contour_levels


def plot_1d_with_fit(fit_result: dict, x_unit: str = "ppm", show_components: bool = True,
                      color_data: str = "black", color_fit: str = "red", color_components: str = "gray",
                      xlims=None, xlabel: str | None = None, field: str = "",
                      tick_dir: str = "in", tick_length: float = 4.0, tick_width: float = 1.0,
                      tick_pad: float = 4.0, tick_number: int | None = None,
                      ax=None, fig=None, pl_width: float = 3.375, pl_height: float = 3.375,
                      font: str = "Arial", fontsize: float = 12, output_format: str = "",
                      save_fig: bool = False, fig_name: str = "nmr_fit_plot", save_format: str = "png"):
    """Plot a 1D spectrum together with its fitted model (and,
    optionally, individual peak components).

    ``fit_result`` : one entry from
    :func:`opennmr.processing.fitting.NMR2D_fit_slices`'s own
    ``fitted["rows"]``/``["columns"]`` (a plain 1D spectrum is fitted
    the same way -- as a single-slice pseudo-2D -- by the GUI's own
    "Fit Spectrum", so this covers both). Ported from ``nmrtools2``'s
    ``NMR1dplot_with_fit``, simplified for opennmr's own data model:
    ``fit_result`` already carries its own ``x``/``y``/``y_fit``/
    ``components`` (computed once, inside ``NMR2D_fit_slices`` itself),
    so this only plots them rather than re-deriving from ``popt``/
    ``peak_defs`` the way the original had to. ``x_unit`` labels the
    axis (``fit_result["x"]`` is already fixed in whatever unit the
    original fit call used -- this does not re-select/convert it).

    ``ax`` : if given, plot into this axis instead of creating a new
    figure -- figure-level options (``pl_height``, ``pl_width``,
    ``font``, ``fontsize``, ``output_format``, ``save_fig``) are then
    ignored, since they'd affect the whole (shared) figure.

    Returns ``fig, ax``.
    """
    standalone = ax is None
    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style["pl_width"], style["pl_height"]
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    x = np.asarray(fit_result["x"])
    y = np.asarray(fit_result["y"])
    y_fit = np.asarray(fit_result["y_fit"])

    ax.plot(x, y, color=color_data, linewidth=1.5, label="Data")
    ax.plot(x, y_fit, color=color_fit, linewidth=2.0, label="Fit")

    if show_components and fit_result.get("components") is not None:
        components = fit_result["components"]
        baseline = np.asarray(components[-1])
        for comp in components[:-1]:
            ax.plot(x, np.asarray(comp) + baseline, color=color_components, linewidth=1.0, alpha=0.6)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xlims is not None:
        ax.set_xlim(xlims[0], xlims[1])
    else:
        ax.set_xlim(xmin, xmax)
    ax.invert_xaxis()
    ax.set_yticks([])
    apply_spine_style(ax, top=False, right=False, bottom=True, left=False)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis="x")

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        spectrum = fit_result.get("slice")
        if spectrum is not None:
            nucgamma = find_gamma(spectrum.metadata.nucleus, field=field, SFO1=spectrum.metadata.sfo1)
            ax.set_xlabel(NUC_label(nucgamma["Nuclide"], x_unit=x_unit))

    ax.legend()

    if standalone:
        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.tight_layout()
        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax


def plot_2d_integrals(fits: list[dict], mode: str = "rows", x_axis: str = "shift",
                       normalised: bool = False, show_errors: bool = True,
                       peak_labels: list[str] | None = None, dataset_label: str | None = None,
                       colors: list | None = None, ax=None, fig=None,
                       xlabel: str | None = None, ylabel: str | None = None, title: str | None = None,
                       xlims=None, ylims=None, show_top_spine: bool = False, show_right_spine: bool = False,
                       pl_width: float = 3.375, pl_height: float = 3.375, font: str = "Arial",
                       fontsize: float = 12, output_format: str = "",
                       tick_dir: str = "in", tick_length: float = 4.0, tick_width: float = 1.0,
                       tick_pad: float = 4.0, tick_number: int | None = None,
                       save_fig: bool = False, fig_name: str = "nmr_integrals_plot", save_format: str = "png"):
    """Plot fitted (or region-integrated) peak integrals across a
    series of 2D slices -- one point per slice per peak, e.g. as a
    quick visual check before committing to a relaxation-model fit.

    ``fits`` : ``fitted["rows"]``/``["columns"]`` from
    :func:`opennmr.processing.fitting.NMR2D_fit_slices`, or
    :func:`opennmr.processing.relaxation.NMR2D_integrate_slices`'s
    equivalent output.

    ``mode`` : ``'rows'``/``'columns'``, only used to build a default
    x-axis label.

    ``x_axis`` : ``'shift'`` (default) or ``'index'``. ``'shift'`` is
    automatically treated as ``'index'`` for a pseudo-2D series, where
    "shift" has no chemical-shift meaning (each slice's own
    ``metadata.raw["slice"]["shift_unit"]`` -- see
    :func:`opennmr.processing.slicing.get_slices` -- records which).
    For an actual delay/gradient axis, use
    :func:`opennmr.processing.relaxation.NMR2D_fit_relaxation` +
    :func:`plot_relaxation` instead -- unlike ``nmrtools2``'s own
    ``NMR2D_plot_integrals``, this does not itself read a vdlist.

    ``normalised`` : plot each slice's integrals as a percentage of
    that slice's own total, rather than the raw integral.

    ``peak_labels`` : legend label per peak (default "Peak 1", "Peak
    2", ...). ``dataset_label`` : prefixed onto every peak's label
    (e.g. "Sample A: Peak 1") -- lets multiple datasets share one
    ``ax`` via repeated calls. ``colors`` : explicit colour per peak;
    ``None`` uses matplotlib's own colour cycle.

    ``ax`` : if given, plot into this axis instead of creating a new
    figure -- figure-level options are then ignored, as in
    :func:`plot_1d_with_fit`.

    Returns ``fig, ax``.
    """
    if not fits:
        raise ValueError("No fitted slices available.")
    if x_axis not in ("shift", "index"):
        raise ValueError("x_axis must be 'shift' or 'index'")

    standalone = ax is None
    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style["pl_width"], style["pl_height"]
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    shifts, indices, integrals, errors, integrals_norm = [], [], [], [], []
    shift_units = set()

    for fit in fits:
        shifts.append(fit.get("shift", np.nan))
        indices.append(fit.get("index", np.nan))
        integrals.append(fit.get("integrals", []))
        errors.append(fit.get("integral_errors", None))
        integrals_norm.append(fit.get("integrals_norm", None))
        slice_spectrum = fit.get("slice")
        slice_info = getattr(slice_spectrum, "metadata", None)
        shift_units.add(slice_info.raw.get("slice", {}).get("shift_unit") if slice_info else None)

    is_index_axis = shift_units == {"index"}

    if x_axis == "index" or is_index_axis:
        x = np.asarray(indices, dtype=float)
        xlabel_default = "Slice number"
    else:
        x = np.asarray(shifts, dtype=float)
        xlabel_default = f"{mode.capitalize()} slice position (ppm)"

    order = np.argsort(x)
    x = x[order]
    integrals = [integrals[i] for i in order]
    errors = [errors[i] for i in order]
    integrals_norm = [integrals_norm[i] for i in order]

    n_peaks = max(len(i) for i in integrals)
    label_prefix = f"{dataset_label}: " if dataset_label else ""

    for peak_i in range(n_peaks):

        source = integrals_norm if normalised else integrals
        y = np.array([vals[peak_i] if len(vals) > peak_i else np.nan for vals in source])

        label = label_prefix + (
            peak_labels[peak_i] if peak_labels is not None and peak_i < len(peak_labels)
            else f"Peak {peak_i + 1}"
        )
        colour = colors[peak_i] if colors is not None and peak_i < len(colors) else None

        have_errors = show_errors and all(
            err is not None and len(err) > peak_i and np.isfinite(err[peak_i]) for err in errors
        )

        if have_errors:
            yerr = np.array([err[peak_i] for err in errors])
            ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", capsize=3, label=label, color=colour)
        else:
            ax.plot(x, y, marker="o", linestyle="-", label=label, color=colour)

    ax.set_xlabel(xlabel if xlabel is not None else xlabel_default)
    ax.set_ylabel(ylabel if ylabel is not None else ("Peak integral (%)" if normalised else "Peak integral"))

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    if title is not None:
        ax.set_title(title)

    apply_spine_style(ax, top=show_top_spine, right=show_right_spine)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis="both")
    ax.legend()

    if standalone:
        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.tight_layout()
        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax


def _lighten_color(color, amount: float = 0.45):
    """Blend ``color`` toward white by ``amount`` (0 = unchanged, 1 =
    white) -- used to derive a relaxation fit curve's colour from its
    data points' colour, so the two are visibly related."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    amount = min(max(amount, 0.0), 1.0)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _broadcast(value, n: int, name: str):
    """Resolve a per-series plot-styling argument -- ``None``/a scalar
    applies to every series; a ``list`` must have length ``n`` (one
    entry per series). A tuple is always one scalar value (e.g. one RGB
    colour, or one annotation offset), never "one value per series" --
    pass a list of tuples for that."""
    if value is None:
        return [None] * n
    if isinstance(value, list):
        if len(value) != n:
            raise ValueError(
                f"{name} has {len(value)} entries but {n} series are being plotted -- "
                f"give one entry per series, or a single value to apply to all."
            )
        return value
    return [value] * n


def _resolve_annotate_anchor(position, t_curve, y_curve):
    """Resolve an ``annotate_position`` entry to an (x, y) point on the
    plotted fit curve: ``'start'``/``'middle'``/``'end'``, or a float
    fraction in [0, 1] along the curve."""
    if isinstance(position, str):
        mapping = {"start": 0.0, "middle": 0.5, "end": 1.0}
        if position not in mapping:
            raise ValueError(
                f"annotate_position must be 'start', 'middle', 'end', or a float in "
                f"[0, 1], got {position!r}."
            )
        frac = mapping[position]
    else:
        frac = float(position)
        if not (0.0 <= frac <= 1.0):
            raise ValueError("annotate_position as a float must be in [0, 1].")
    idx = int(round(frac * (len(t_curve) - 1)))
    return t_curve[idx], y_curve[idx]


def _select_series(fit_results, show_series):
    """Filter a list of :func:`opennmr.processing.relaxation.fit_relaxation`
    -style results (each with a ``"peak_label"`` key) down to a
    requested subset, given either 0-based indices or label strings.
    Returns every result unchanged if ``show_series`` is None."""
    if show_series is None:
        return list(fit_results)
    selected = []
    for item in show_series:
        if isinstance(item, str):
            match = next((f for f in fit_results if f.get("peak_label") == item), None)
            if match is None:
                available = [f.get("peak_label") for f in fit_results]
                raise ValueError(f"No series labelled '{item}'. Available: {available}")
            selected.append(match)
        else:
            selected.append(fit_results[item])
    return selected


def plot_relaxation(fit_results: list[dict], show_series=None, annotate: bool = False,
                     xlabel: str = "Delay (s)", ylabel: str = "Integral", logplot=False,
                     colors=None, fit_colors=None, fit_shade: float = 0.45,
                     annotate_position="end", annotate_offset=(5, 0),
                     annotate_ha="left", annotate_va="center",
                     ax=None, fig=None, pl_width: float = 3.375, pl_height: float = 3.375,
                     font: str = "Arial", fontsize: float = 12, output_format: str = "",
                     tick_dir: str = "in", tick_length: float = 4.0, tick_width: float = 1.0,
                     tick_pad: float = 4.0, tick_number: int | None = None,
                     show_top_spine: bool = False, show_right_spine: bool = False,
                     n_curve_points: int = 300,
                     save_fig: bool = False, fig_name: str = "nmr_relaxation_plot", save_format: str = "png"):
    """Plot one or more :func:`opennmr.processing.relaxation.fit_relaxation`
    results (data + fitted curve) on a single axis -- typically the
    output of :func:`opennmr.processing.relaxation.NMR2D_fit_relaxation`.

    Ported from ``nmrtools2``'s ``NMR2D_plot_relaxation`` -- opennmr's
    own ``fit_relaxation``/``NMR2D_fit_relaxation`` already return this
    exact dict shape (``t``, ``y``, ``y_err``, ``params``, ``errors``,
    ``model``, ``param_names``, ``rate_param``, ``peak_label``), so
    this ports over unchanged beyond the module it imports
    ``RELAXATION_MODELS`` from.

    ``show_series`` : 0-based indices or ``peak_label`` strings to
    plot a subset; ``None`` (default) plots every series.

    ``annotate`` : label each curve on the plot with its fitted rate
    constant (T1/T2/.../shift-formatted, via
    :func:`opennmr.processing.relaxation.format_time_value`) -- off by
    default, since the legend already names each series.

    ``colors``/``fit_colors`` : per-series colour for data points /
    fit curve (a single value applies to every series, a list gives
    one per series). If ``fit_colors`` is not given, each fit curve is
    a paler tint of its own series' data colour (``fit_shade``).

    ``logplot`` : ``False``, ``'x'``, ``'y'``, or ``'both'``.

    ``ax`` : if given, plot into this axis instead of creating a new
    figure -- figure-level options are then ignored, as in
    :func:`plot_1d_with_fit`.

    Returns ``fig, ax``.
    """
    fit_results = _select_series(fit_results, show_series)
    n = len(fit_results)
    if n == 0:
        raise ValueError("No relaxation fit results to plot.")

    data_colors = _broadcast(colors, n, "colors")
    fit_colors = _broadcast(fit_colors, n, "fit_colors")
    positions = _broadcast(annotate_position, n, "annotate_position")
    offsets = _broadcast(annotate_offset, n, "annotate_offset")
    ha_list = _broadcast(annotate_ha, n, "annotate_ha")
    va_list = _broadcast(annotate_va, n, "annotate_va")

    standalone = ax is None
    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style["pl_width"], style["pl_height"]
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    prop_cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])

    for i, fit in enumerate(fit_results):

        data_colour = data_colors[i] if data_colors[i] is not None \
            else prop_cycle_colors[i % len(prop_cycle_colors)]
        curve_colour = fit_colors[i] if fit_colors[i] is not None \
            else _lighten_color(data_colour, fit_shade)

        label = fit.get("peak_label", f"Fit {i + 1}")

        if fit.get("y_err") is not None:
            ax.errorbar(fit["t"], fit["y"], yerr=fit["y_err"], marker="o",
                        linestyle="none", capsize=3, color=data_colour, label=None)
        else:
            ax.plot(fit["t"], fit["y"], marker="o", linestyle="none", color=data_colour, label=None)

        spec = RELAXATION_MODELS[fit["model"]]
        t_curve = np.linspace(fit["t"].min(), fit["t"].max(), n_curve_points)
        y_curve = spec["func"](t_curve, *[fit["params"][p] for p in fit["param_names"]])

        ax.plot(t_curve, y_curve, linestyle="-", color=curve_colour, label=label)

        rate_param = fit["rate_param"]
        rate_param_ = {"T1": r"$T_{1}$", "T2": r"$T_{2}$"}.get(rate_param, rate_param)

        if annotate and rate_param:
            time_str = format_time_value(fit["params"][rate_param], fit["errors"][rate_param])
            anchor_x, anchor_y = _resolve_annotate_anchor(positions[i], t_curve, y_curve)
            ax.annotate(
                rf"{rate_param_} = {time_str}",
                xy=(anchor_x, anchor_y), xytext=offsets[i],
                textcoords="offset points", ha=ha_list[i], va=va_list[i],
                fontsize=0.8 * fontsize, color=data_colour,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if logplot:
        if logplot == "x":
            ax.set_xscale("log")
            ax.set_yscale("linear")
        elif logplot == "y":
            ax.set_yscale("log")
            ax.set_xscale("linear")
        elif logplot == "both":
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            raise ValueError(f"logplot must be False, 'x', 'y', or 'both', got {logplot!r}.")

    apply_spine_style(ax, top=show_top_spine, right=show_right_spine)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis="both")
    ax.legend()

    if standalone:
        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.tight_layout()
        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax


def _dosy_d_limits(z, d_grid, factor: float, pad_decades: float = 0.5):
    """The D range worth showing: whatever carries signal above the
    lowest contour level, padded by ``pad_decades`` either side.

    The ILT grid deliberately spans several decades more than any real
    sample occupies -- it has to, since where the populations lie is
    what is being solved for. Plotting all of it squashed every peak
    into a corner of an otherwise blank panel. The 2D contour panel
    fits itself to its data for the same reason; scrolling the D axis
    still reaches the rest of the grid.

    Falls back to the whole grid when nothing clears the threshold,
    rather than collapsing to a degenerate range.
    """
    d_grid = np.asarray(d_grid, dtype=float)
    full = (float(np.min(d_grid)), float(np.max(d_grid)))
    z = np.asarray(z, dtype=float)
    peak = float(np.nanmax(z)) if z.size else 0.0
    if peak <= 0:
        return full
    rows = np.flatnonzero(np.nanmax(z, axis=1) > peak * max(float(factor), 1e-6))
    if rows.size == 0:
        return full
    low = float(d_grid[rows.min()]) * 10.0 ** -pad_decades
    high = float(d_grid[rows.max()]) * 10.0 ** pad_decades
    return max(low, full[0]), min(high, full[1])


def plot_dosy(dosy: dict, ax=None, fig=None, contour_ax=None, projection_ax=None,
              clevels: int = 8, factor: float = 0.02, cmap: str | None = None,
              level_mode: str = "power", power: float = 1.25,
              d_lims=None, ppm_lims=None,
              # mathtext, not the Unicode superscripts: Arial (this
              # package's default face) has no U+207B SUPERSCRIPT MINUS,
              # so "m²s⁻¹" renders with a missing-glyph box and emits a
              # warning on every draw.
              xlabel: str = "δ / ppm", ylabel: str = r"D / $\mathregular{m^{2}\,s^{-1}}$",
              show_projection: bool = True, projection_height: float = 0.22,
              projection_color: str = "black", linewidths: float = 0.8,
              pl_width: float = 5.0, pl_height: float = 5.0,
              font: str = "Arial", fontsize: float = 12,
              tick_dir: str = "in", tick_length: float = 4.0, tick_width: float = 1.0,
              save_fig: bool = False, fig_name: str = "nmr_dosy_plot", save_format: str = "png"):
    """Draw a DOSY map -- chemical shift across, diffusion coefficient
    down a log axis -- from :func:`opennmr.processing.ilt.dosy_map`'s
    own output dict.

    Conventions this follows because they are what a DOSY *is*, not
    because they are configurable preferences:

    * **The shift axis runs high-to-low**, like every other NMR
      spectrum in this package.
    * **The D axis is logarithmic and runs small-at-the-bottom**, so
      slowly-diffusing (large, aggregated) species sit low and fast
      small molecules sit high. That is the orientation of every
      published DOSY, and flipping it would make the plot unreadable
      to anyone who has seen one before.
    * **The ordinary 1D spectrum is drawn along the top**, from the
      map's own lowest-gradient row -- a DOSY with no reference
      spectrum above it is very hard to assign.

    Drawn as contour lines rather than an image: the map is smooth
    along D by construction (the ILT reconstructs onto a grid finer
    than the data can resolve), so contours state the resolution
    honestly where a pixel image would imply detail that isn't there.

    Styled to match :func:`opennmr.plotting.twod.plot_2d_contour` --
    unfilled black lines on a full frame, at the same font size and
    line width, with the levels spaced the same power-law way. It is
    the same kind of plot, and an earlier filled-contour version read
    as a different (and worse-looking) thing sitting in the same app.
    ``cmap`` colours the lines by level instead, for the cases where a
    2D would take ``colour_spectra``.

    ``ax`` : plot the contour into an existing axis; the projection
    strip is then only drawn if ``projection_ax`` is also supplied.
    Returns ``fig, (contour_ax, projection_ax)`` -- the second is
    ``None`` when no strip was drawn.
    """
    z = np.asarray(dosy["Z"], dtype=float)
    d_grid = np.asarray(dosy["D"], dtype=float)
    ppm = np.asarray(dosy["ppm"], dtype=float)

    external_axis = ax is not None
    if external_axis:
        contour_ax = ax
        fig = fig if fig is not None else contour_ax.get_figure()
    else:
        setup_plotting(font=font, fontsize=fontsize)
        fig = plt.figure(figsize=(pl_width, pl_height))
        if show_projection:
            # A shared x so panning/zooming the map keeps the spectrum
            # above it lined up -- the whole point of the strip.
            grid = fig.add_gridspec(2, 1, height_ratios=[projection_height, 1.0], hspace=0.04)
            projection_ax = fig.add_subplot(grid[0])
            contour_ax = fig.add_subplot(grid[1], sharex=projection_ax)
        else:
            contour_ax = fig.add_subplot(111)
            projection_ax = None

    peak = float(np.nanmax(z)) if z.size else 0.0
    if peak > 0:
        # Levels start above zero: the map is exactly zero everywhere
        # the ILT was not run (below the intensity threshold), and a
        # level at zero would draw a contour right around that entire
        # background region.
        #
        # ``factor`` is the lowest level as a fraction of the map's own
        # peak -- the same "start" a 2D contour's own scroll adjusts,
        # so a DOSY's levels scale the same way and by the same
        # gesture -- and they are spaced by the same power law, so the
        # two kinds of plot read alike.
        levels = _contour_levels(peak * max(float(factor), 1e-6), peak, clevels, level_mode, power)
        if cmap:
            contour_ax.contour(ppm, d_grid, z, levels=levels, cmap=cmap, linewidths=linewidths)
        else:
            contour_ax.contour(ppm, d_grid, z, levels=levels, colors="black", linewidths=linewidths)

    contour_ax.set_yscale("log")
    contour_ax.set_xlabel(xlabel, fontsize=fontsize)
    contour_ax.set_ylabel(ylabel, fontsize=fontsize)
    contour_ax.set_xlim(*(ppm_lims if ppm_lims else (float(np.max(ppm)), float(np.min(ppm)))))
    contour_ax.set_ylim(*(d_lims if d_lims else _dosy_d_limits(z, d_grid, factor)))
    contour_ax.tick_params(
        axis="both", which="both", direction=tick_dir, labelsize=fontsize,
        length=tick_length, width=tick_width, bottom=True, left=True, top=True, right=True,
    )
    # A full frame, like the 2D contour panel -- and unlike the rest of
    # this module's open-sided line plots. A contour is read as a box:
    # with the top and right spines hidden, a peak near either edge
    # trails off into nothing and the D axis has no closing edge to
    # measure against.
    apply_spine_style(contour_ax, top=True, right=True, bottom=True, left=True,
                      linewidth=tick_width)

    if projection_ax is not None:
        projection = np.asarray(dosy["projection"], dtype=float)
        projection_ax.plot(ppm, projection, color=projection_color, linewidth=linewidths)
        projection_ax.set_xlim(contour_ax.get_xlim())
        # No ticks at all, on either axis, and no frame -- exactly what
        # plot_2d_contour does with its own projection strips. Leaving
        # the x ticks on drew a row of marks floating above the contour
        # with no spine to belong to, and the strip's own intensity
        # scale is arbitrary, so its y ticks say nothing either.
        projection_ax.tick_params(axis="x", which="both", bottom=False, top=False,
                                   labelbottom=False, labeltop=False)
        projection_ax.tick_params(axis="y", which="both", left=False, right=False,
                                   labelleft=False, labelright=False)
        apply_spine_style(projection_ax, top=False, right=False, left=False, bottom=False)

    if not external_axis and save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, (contour_ax, projection_ax)


def plot_score_components(result: dict, ppm, ax=None, fig=None, reference=None,
                          offset: float = 1.15, normalise: bool = True,
                          colors=None, annotate: bool = True, xlims=None,
                          xlabel: str = "δ / ppm",
                          pl_width: float = 5.0, pl_height: float = 5.0,
                          font: str = "Arial", fontsize: float = 12,
                          tick_dir: str = "in", tick_length: float = 4.0, tick_width: float = 1.0,
                          save_fig: bool = False, fig_name: str = "nmr_score_plot",
                          save_format: str = "png"):
    """Stack the component spectra resolved by
    :func:`opennmr.processing.score.score`, one per row, each labelled
    with its own decay constant and share of the signal.

    A stack rather than an overlay: the point of SCORE is that these
    are *separate species*, and overlaying them re-creates exactly the
    superposition the method just undid. The measured spectrum can be
    drawn above them (``reference``, normally the lowest-gradient row)
    so it is visible which peaks went to which component -- the useful
    check being whether an overlapped peak has been split sensibly
    rather than assigned wholesale to one component.

    Normalised per component by default: the components' absolute
    scales are set by the arbitrary intensity units of the data, and a
    weak component drawn to scale is simply invisible. The *share* of
    each is in its label instead, which is the honest place for it.
    """
    spectra = np.asarray(result["spectra"], dtype=float)
    ppm = np.asarray(ppm, dtype=float)
    n_components = spectra.shape[0]

    external_axis = ax is not None
    if external_axis:
        fig = fig if fig is not None else ax.get_figure()
    else:
        setup_plotting(font=font, fontsize=fontsize)
        fig, ax = plt.subplots(figsize=(pl_width, pl_height))

    if colors is None:
        colors = qualitative_colors(n_components)
    elif isinstance(colors, str):
        colors = [colors] * n_components

    rows = []
    if reference is not None:
        rows.append((np.asarray(reference, dtype=float), "black", "measured"))
    for index in range(n_components):
        label = _score_component_label(result, index)
        rows.append((spectra[index], colors[index], label))

    for position, (trace, color, label) in enumerate(rows):
        peak = float(np.max(np.abs(trace))) if trace.size else 0.0
        scaled = trace / peak if (normalise and peak > 0) else trace
        baseline = position * offset
        ax.plot(ppm, scaled + baseline, color=color, linewidth=1.0)
        if annotate:
            # Just above its own trace's baseline, with an opaque
            # backing box: a label placed mid-height lands on top of
            # whichever peak happens to be at that end of the spectrum
            # (it did, on the first real component plot), and there is
            # no edge that is reliably empty for every dataset.
            ax.text(0.995, baseline + offset * 0.06, label, color=color, fontsize=fontsize * 0.8,
                    ha="right", va="bottom", transform=ax.get_yaxis_transform(), zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="none", alpha=0.85))

    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_xlim(*(xlims if xlims else (float(np.max(ppm)), float(np.min(ppm)))))
    apply_tick_style(ax, axis="x", direction=tick_dir, length=tick_length, width=tick_width)
    apply_spine_style(ax, top=False, right=False, left=False, bottom=True)

    if not external_axis and save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)
    return fig, ax


def _score_component_label(result: dict, index: int) -> str:
    """One component's legend text: its decay constant in its own unit,
    plus its share -- the two numbers a resolved component is quoted
    by."""
    from ..processing.relaxation import format_diffusion_value, format_time_value

    share = result["fractions"][index]
    if result.get("kind") == "diffusion":
        value = format_diffusion_value(result["constants"][index])
    else:
        value = format_time_value(result["T"][index])
    return f"{index + 1}: {value}  ({share * 100:.0f}%)"
