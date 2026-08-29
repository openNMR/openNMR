"""
1D spectrum plotting.

``plot_1d`` is ``ax``-first (factored out of ``nmrtools2``'s
``NMR1dplot`` per-spectrum plotting loop) and never touches global
matplotlib state, so it's safe to call from a GUI canvas as well as
from the standalone convenience wrappers below (``plot_1d_stack``,
``plot_1d_heatmap``), which create their own figure and *do* call
:func:`opennmr.plotting.style.setup_plotting` -- harmless for a
one-off figure, not for a GUI (see that module's docstring).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ..core.registry import find_gamma, NUC_label
from ..core.spectrum import NMRSpectrum
from ..processing.arithmetic import combine_spectra
from .style import setup_plotting, apply_tick_style, apply_spine_style, get_contour_cmap, save_figure


def plot_1d(ax, spectrum: NMRSpectrum, x_unit: str = "ppm", normalize: bool = True,
            color: str = "black", y_offset: float = 0.0, linewidth: float = 1.5,
            field: str = "", full_resolution: bool = False) -> tuple[float, float, str]:
    """Plot one 1D :class:`NMRSpectrum` onto an existing ``ax``.

    Returns ``(xmin, xmax, labeltext)`` -- the spectrum's x-range
    (sorted ascending) and its nucleus-based axis label, both handy
    for a caller plotting a stack of these (see :func:`plot_1d_stack`).

    Rendered at no more than :data:`opennmr.plotting.style.DOWNSAMPLE_1D_MAX_POINTS`
    points (evenly, via :func:`opennmr.plotting.style.downsample_indices`)
    unless ``full_resolution=True`` -- display only, ``xmin``/``xmax``
    are always the true full-resolution range regardless, and nothing
    about ``spectrum`` itself is ever touched either way. Explicit user
    feedback: with many stacked traces or a dynamically-redrawn view
    (phase-dragging), rendering every point of a high-digital-
    resolution or diffusion/imaging acquisition (routinely 10^5-10^6
    points) cost real, noticeable time for no visible benefit at any
    normal screen resolution -- but silently downsampling with no way
    back is a real problem for a tool whose whole point is showing the
    actual data, so this is opt-out (``full_resolution=True``), not
    forced, and the returned line always carries the true full-
    resolution arrays too (as ``line._opennmr_full_res``, a plain
    ``(x, y)`` pair) whenever downsampling would apply at all --
    whether or not it's actually in effect right now -- so a caller
    (:class:`~opennmr.gui.widgets.canvas.MplCanvas`) can toggle between
    the two, or guarantee a saved/exported figure is always full
    resolution even while the live view isn't, without re-plotting
    from ``spectrum`` again.
    """
    from .style import downsample_indices

    x = spectrum.axis_values(0, unit=x_unit.lower())
    y = spectrum.normalized() if normalize else spectrum.data

    xmin, xmax = float(x[0]), float(x[-1])
    if xmin > xmax:
        xmin, xmax = xmax, xmin

    nucgamma = find_gamma(spectrum.metadata.nucleus, field=field, SFO1=spectrum.metadata.sfo1)
    labeltext = NUC_label(nucgamma["Nuclide"], x_unit=x_unit)

    idx = None if full_resolution else downsample_indices(len(y))
    x_plot, y_plot = (x, y) if idx is None else (x[idx], y[idx])
    line, = ax.plot(x_plot, y_plot + y_offset, color=color, linewidth=linewidth)
    if downsample_indices(len(y)) is not None:  # would need it at the default cap, regardless of full_resolution
        line._opennmr_full_res = (x, y + y_offset)

    return xmin, xmax, labeltext


def plot_1d_stack(spectra: list[NMRSpectrum], ax=None, fig=None, color: str = "black",
                   x_unit: str = "ppm", normalize: bool = True, y_offset: float = 0.2,
                   field: str = "", xlims: str | tuple = "default", xlabel: str | None = None,
                   tick_dir: str = "in", tick_length: float = 4.0, tick_width: float = 1.0,
                   tick_pad: float = 4.0, tick_number: int | None = None,
                   pl_width: float = 3.375, pl_height: float = 3.375, font: str = "Arial",
                   fontsize: float = 12, output_format: str = "", full_resolution: bool = False,
                   save_fig: bool = False, fig_name: str = "nmr_1d_plot", save_format: str = "png"):
    """Plot a stack of 1D spectra with a vertical offset between them,
    coloured along :func:`opennmr.plotting.style.get_stack_colormap`.

    Ported from ``nmrtools2``'s ``NMR1dplot`` (its non-VOCS,
    non-heatmap path). If ``ax`` is None (default), creates a
    standalone figure and applies
    :func:`opennmr.plotting.style.setup_plotting`; otherwise plots
    into the given ``ax`` without touching global style (for
    embedding, e.g. in a GUI canvas).
    """
    from .style import stack_colors

    standalone = ax is None
    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style["pl_width"], style["pl_height"]
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    colors = stack_colors(len(spectra), color)

    # One shared step size for every gap in the stack, from the
    # tallest spectrum in it -- not each spectrum's own max, which
    # made the gaps themselves uneven (a spectrum with a much smaller
    # peak got a much smaller gap above it) whenever normalize=False
    # (the Compare tab's "None"/"By sample mass" modes, where seeing
    # each spectrum's true relative height is the whole point, so
    # per-spectrum renormalizing the gap on top of that isn't wanted).
    y_scale = max((float(np.max(s.normalized() if normalize else s.data)) for s in spectra), default=0.0)

    y_base = 0.0
    xmin = xmax = labeltext = None

    for cnt, spectrum in enumerate(spectra):
        this_xmin, this_xmax, labeltext = plot_1d(
            ax, spectrum, x_unit=x_unit, normalize=normalize,
            color=colors[cnt], y_offset=y_base, field=field, full_resolution=full_resolution,
        )
        xmin = this_xmin if xmin is None else min(xmin, this_xmin)
        xmax = this_xmax if xmax is None else max(xmax, this_xmax)
        y_base += y_scale * (1 + y_offset)

    if xlims == "default":
        ax.set_xlim(xmin, xmax)
    else:
        ax.set_xlim(xlims[0], xlims[1])

    ax.set_xlabel(xlabel if xlabel is not None else labeltext)
    apply_spine_style(ax, top=False, right=False, bottom=True, left=False)
    ax.set_yticks([])
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis="x")
    ax.invert_xaxis()

    if standalone:
        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.tight_layout()
        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax


def plot_1d_heatmap(spectra: list[NMRSpectrum], ax=None, fig=None, x_unit: str = "ppm",
                     xlims: str | tuple = "default", xlabel: str | None = None,
                     ylabel: str | None = None, y_values: list | None = None,
                     y_ticks: str = "none", y_tick_number: int = 5,
                     common_range: str = "intersection", npoints: int | None = None,
                     interp_kind: str = "linear", normalize: bool = True,
                     cmap: str = "viridis", vmin: float | None = None, vmax: float | None = None,
                     colorbar: bool = False, colorbar_label: str | None = None,
                     field: str = "", tick_dir: str = "in", tick_length: float = 4.0,
                     tick_width: float = 1.0, tick_pad: float = 4.0, tick_number: int | None = None,
                     pl_width: float = 3.375, pl_height: float = 3.375, font: str = "Arial",
                     fontsize: float = 12, output_format: str = "",
                     save_fig: bool = False, fig_name: str = "nmr_1d_heatmap", save_format: str = "png"):
    """Plot multiple 1D spectra as rows of a 2D heatmap (cf.
    :func:`plot_1d_stack`'s offset stack).

    Every spectrum is interpolated onto one common x-axis via
    :func:`opennmr.processing.arithmetic.combine_spectra` (reusing its
    interpolation machinery -- see that function's ``metadata.raw
    ["interpolated_spectra"]``), stacked into a ``(n_spectra,
    npoints)`` matrix, and displayed with ``pcolormesh``. Rows are
    evenly spaced (one per input spectrum, in order given) -- the
    y-axis is categorical, not itself an interpolated physical axis.

    Ported from ``nmrtools2``'s ``NMR1dheatmap``.
    """
    if len(spectra) < 2:
        raise ValueError("Need at least 2 spectra to build a heatmap.")

    if y_ticks not in ("none", "all", "auto"):
        raise ValueError("y_ticks must be one of 'none', 'all', 'auto'.")
    if y_ticks != "none":
        if y_values is None:
            raise ValueError(f"y_ticks='{y_ticks}' requires y_values (one entry per spectrum).")
        if len(y_values) != len(spectra):
            raise ValueError("y_values must have one entry per spectrum.")

    standalone = ax is None
    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style["pl_width"], style["pl_height"]
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    prepared = []
    for s in spectra:
        y = s.normalized() if normalize else s.data
        prepared.append(NMRSpectrum(data=y, axes=[s.axes[0]], metadata=s.metadata))

    combined = combine_spectra(prepared, x_unit=x_unit, common_range=common_range,
                                npoints=npoints, interp_kind=interp_kind)
    x_common = combined.axis_values(0, x_unit)
    data_matrix = combined.metadata.raw["interpolated_spectra"]

    nucgamma = find_gamma(spectra[0].metadata.nucleus, field=field, SFO1=spectra[0].metadata.sfo1)
    labeltext = NUC_label(nucgamma["Nuclide"], x_unit=x_unit)

    try:
        cmap_obj = get_contour_cmap(cmap)
    except ValueError:
        cmap_obj = plt.get_cmap(cmap)

    row_positions = np.arange(len(spectra))
    im = ax.pcolormesh(x_common, row_positions, data_matrix, cmap=cmap_obj,
                        vmin=vmin, vmax=vmax, shading="auto")

    if xlims == "default":
        ax.set_xlim(x_common[0], x_common[-1])
    else:
        ax.set_xlim(xlims[0], xlims[1])

    ax.set_xlabel(xlabel if xlabel is not None else labeltext)
    ax.invert_xaxis()

    apply_spine_style(ax, top=False, right=False, bottom=True, left=True)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis="x", minor=True)

    if y_ticks == "none":
        ax.set_yticks([])
    else:
        if y_ticks == "all":
            tick_idx = list(range(len(spectra)))
        else:
            locator = MaxNLocator(nbins=y_tick_number, integer=True)
            raw_idx = locator.tick_values(0, len(spectra) - 1)
            tick_idx = sorted({int(round(i)) for i in raw_idx if 0 <= i <= len(spectra) - 1})

        ax.set_yticks(tick_idx)
        ax.set_yticklabels([y_values[i] for i in tick_idx])
        ax.tick_params(axis="y", which="major", direction=tick_dir,
                        length=tick_length, width=tick_width, pad=tick_pad)
        ax.tick_params(axis="y", which="minor", length=0)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if colorbar:
        cbar = fig.colorbar(im, ax=ax)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)

    if standalone:
        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.tight_layout()
        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax, im
