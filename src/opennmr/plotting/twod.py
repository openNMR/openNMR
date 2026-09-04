"""
2D spectrum plotting: contour + projections, multi-panel, and overlay.

``plot_2d_contour`` is ``ax``-first (``nmrtools2``'s ``NMR2Dplotter``
already accepted an external ``ax=``, so this ports over directly) and
never touches global matplotlib state when embedded that way -- see
:mod:`opennmr.plotting.style`'s docstring for why that matters for a
GUI canvas. Unlike the original, this only ever takes an already-
loaded :class:`NMRSpectrum` (not a raw path) -- keeping the plotting
layer decoupled from :mod:`opennmr.io`; load first, plot second.

Ported from ``nmrtools2``'s ``two_d.py``.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, MaxNLocator
from mpl_toolkits.axes_grid1 import axes_size
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider

from ..core.registry import find_gamma, NUC_label
from ..core.spectrum import NMRSpectrum
from .style import setup_plotting, get_stack_colormap, get_contour_cmap, downsample_2d, save_figure


def _prepare_contour(spectrum: NMRSpectrum, contour_fraction_x=0.25, contour_fraction_y=0.25,
                      ncontourpts_x=16384, ncontourpts_y=1024, factor=0.07, clevels=7,
                      level_mode="power", power=1.25):
    """Shared contour-preparation logic: resolve the F1 axis (ppm, or
    slice index for pseudo-2D), downsample for fast rendering, and
    compute the contour levels. Internal helper used by both
    :func:`plot_2d_contour` and :func:`overlay_plot`.

    ``level_mode`` : ``"power"`` (default) spaces ``clevels`` levels
    between ``factor * max(data)`` and ``max(data)`` on a power-law
    curve controlled by ``power`` (steeper/more levels bunched near
    the noise floor as ``power`` increases past 1) -- good for picking
    out weak peaks near a strong one without saturating the display.
    ``"linear"`` spaces them evenly instead, which better reflects
    relative peak heights at a glance.

    ``factor``/``clevels`` defaults (0.07, 7) are deliberately modest --
    a much lower ``factor`` (e.g. 1e-3) combined with many levels
    (e.g. 20) reaches so far into the noise floor, and bunches so many
    of those levels near it, that real but weaker signal ends up
    buried under a dense mass of near-noise-level rings rather than a
    few clean, readable contours (reported directly as "the contours
    seem to start out scaled up way too high").
    """
    data = spectrum.data
    x_ppm = spectrum.axes[1].ppm
    pseudo2d = spectrum.is_pseudo_2d
    y_axis = spectrum.axes[0].index.astype(float) if pseudo2d else spectrum.axes[0].ppm

    if contour_fraction_x is not None or contour_fraction_y is not None:
        max_x = min(
            int(len(x_ppm) * contour_fraction_x) if contour_fraction_x is not None else len(x_ppm),
            len(x_ppm), ncontourpts_x,
        )
        max_y = min(
            int(len(y_axis) * contour_fraction_y) if contour_fraction_y is not None else len(y_axis),
            len(y_axis), ncontourpts_y,
        )
        contour_data, contour_x, contour_y = downsample_2d(
            data, x_ppm, y_axis, max_points_x=max_x, max_points_y=max_y,
        )
    else:
        contour_x, contour_y, contour_data = x_ppm, y_axis, data

    if level_mode not in ("power", "linear"):
        raise ValueError("level_mode must be 'power' or 'linear'")

    thresvec = _contour_levels(factor * np.amax(data), np.amax(data), clevels, level_mode, power)

    neg_max = np.amin(data)  # most-negative value, i.e. the "positive" end for the negative side
    neg_thresvec = (
        _contour_levels(factor * neg_max, neg_max, clevels, level_mode, power) if neg_max < 0 else None
    )

    return contour_x, contour_y, contour_data, thresvec, y_axis, neg_thresvec


def _contour_levels(thresh_min, thresh_max, clevels, level_mode, power):
    """``clevels`` threshold levels between ``thresh_min`` and
    ``thresh_max`` (same sign as each other; either both positive, for
    the usual case, or both negative, for the mirrored negative-side
    levels ``_prepare_contour`` also computes) -- factored out of
    :func:`_prepare_contour` so the same spacing logic (and its own
    bug-fix history, see below) applies to both sides identically."""
    if level_mode == "linear":
        return list(np.linspace(thresh_min, thresh_max, clevels))
    # Real bug, found and fixed: the exponent used to be
    # `power * i / clevels`, which only reaches exactly `power`
    # (not 1) at the top level (i == clevels) -- for power > 1
    # (the default is 1.25), that overshoots thresh_max, so the
    # top several "contour levels" sit above every real data
    # point and are never actually drawn. The visible levels were
    # then squeezed into the remaining lower fraction of the
    # range, several coarser than intended -- moderate, real
    # signal often fell entirely between two of them, looking
    # like a 2D/pseudo-2D dataset "starts out scaled way too
    # high" (barely anything visible) until the threshold was
    # manually scrolled down a lot. `(i / clevels) ** power`
    # always reaches exactly 1 at i == clevels (so the top level
    # is exactly thresh_max, regardless of power), and for
    # power > 1 still bunches more levels toward thresh_min first
    # -- matching this function's own docstring ("steeper/more
    # levels bunched near the noise floor as power increases").
    cc2 = np.linspace(1, clevels, clevels)
    return [thresh_min * (thresh_max / thresh_min) ** ((i / clevels) ** power) for i in cc2]


def plot_2d_contour(spectrum: NMRSpectrum, contour_fraction_x=0.25, contour_fraction_y=0.25,
                     ncontourpts_x=16384, ncontourpts_y=1024,
                     pl_height=7.0, pl_width=7.0, factor=0.07, clevels=7,
                     level_mode="power", power=1.25,
                     proj_scale=1.0, projection_mode="auto", show_projections=True,
                     projection_size=2.0, projection_pad=0.1, square_aspect=True,
                     field='', xlabel=None, ylabel=None, xlims=None, ylims=None,
                     font='Arial', fontsize=12, output_format='',
                     colour_spectra=None, equal_tickspacing=False, tick_number=None,
                     tick_dir='in', tick_length=4, tick_width=1.0, tick_pad=4,
                     show_nQ_line=False, nQ=1, nQ_colors=None,
                     ax=None, fig=None, save_fig=False, fig_name='nmr_2d_spectrum',
                     save_format='png'):
    """Plot a 2D NMR spectrum as a contour plot with F1/F2 projections.

    ``projection_mode`` : ``'sum'`` for conventional projections;
    ``'max'`` (of ``abs()``) for pseudo-2D relaxation/diffusion series,
    where a sum projection would wash out slice-to-slice changes.
    ``'auto'`` (default) picks ``'max'`` for pseudo-2D, ``'sum'``
    otherwise.

    ``show_projections`` : ``True`` (both), ``False`` (neither, e.g.
    embedding a compact panel), or ``'f1'``/``'f2'`` (one only).

    ``square_aspect`` : ``True`` (default) makes the main contour panel
    itself square, via the same ``AxesDivider`` that lays out the F1/F2
    projections (``divider.set_aspect(True)``) rather than
    ``ax.set_box_aspect`` -- the divider coordinates *all* of its cells
    (main panel + any appended projections) together when resizing for
    the aspect constraint, so the projections stay correctly sized/
    positioned against the now-square panel; ``ax.set_box_aspect``
    shrinks only the main axes in isolation, leaving sibling axes
    created via ``divider.append_axes`` stranded at their pre-shrink
    position/size (confirmed directly -- a real, visible misalignment,
    not a theoretical concern). Set ``False`` to leave the panel's
    aspect fully auto (e.g. a caller that will apply its own
    ``ax.set_box_aspect``/``set_aspect`` afterward -- doing so together
    with ``square_aspect=True`` is redundant, not additive, and doing
    so with ``square_aspect=True`` when projections are also shown
    reintroduces the exact misalignment this parameter exists to
    avoid).

    ``clevels``/``level_mode``/``power`` : see :func:`_prepare_contour`.

    ``ax`` : if given, plot into this axis instead of creating a new
    figure (for embedding, e.g. in a GUI canvas, or in
    :func:`multi_plot`) -- figure-level options (``pl_height``,
    ``pl_width``, ``font``, ``fontsize``, ``output_format``,
    ``save_fig``) are then ignored, since they'd affect the whole
    (shared) figure.

    Returns ``fig, ax, proj_axes`` -- ``proj_axes`` is
    ``{"f1": Axes | None, "f2": Axes | None}``, the F1 (left)/F2 (top)
    projection panels' own axes when drawn (``None`` for whichever
    ``show_projections`` excluded), so a caller can address their
    intensity scale independently of the main contour (e.g. its own
    scroll-to-zoom).
    """
    data = spectrum.data
    x_ppm = spectrum.axes[1].ppm
    pseudo2d = spectrum.is_pseudo_2d
    homonuclear = spectrum.metadata.homonuclear

    contour_x, contour_y, contour_data, thresvec, y_axis, neg_thresvec = _prepare_contour(
        spectrum, contour_fraction_x=contour_fraction_x, contour_fraction_y=contour_fraction_y,
        ncontourpts_x=ncontourpts_x, ncontourpts_y=ncontourpts_y, factor=factor, clevels=clevels,
        level_mode=level_mode, power=power,
    )

    nuc1 = find_gamma(spectrum.metadata.nucleus, field=field, SFO1=spectrum.metadata.sfo1)['Nuclide']
    labeltextx = NUC_label(nuc1, x_unit='ppm')

    if homonuclear:
        labeltexty = NUC_label(nuc1, x_unit='ppm')
    elif not pseudo2d:
        labeltexty = NUC_label(spectrum.metadata.nucleus_indirect, x_unit='ppm')
    else:
        labeltexty = 'Slice number'

    if ylabel is not None:
        labeltexty = ylabel
    if xlabel is not None:
        labeltextx = xlabel

    standalone = ax is None
    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style['pl_width'], style['pl_height']
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    if colour_spectra:
        masked = np.ma.masked_where(contour_data <= 0, contour_data)
        ax.contour(contour_x, contour_y, masked, levels=thresvec,
                   cmap=get_contour_cmap(colour_spectra), norm=LogNorm())
    else:
        ax.contour(contour_x, contour_y, contour_data, colors='black', levels=thresvec)

    if neg_thresvec is not None:
        # Negative-going signal (dispersive phase error, an artifact,
        # or genuinely negative-intensity data) drawn in a contrasting
        # colour rather than invisible or lost among the positive
        # levels -- real user request, and a real practical need: it's
        # what makes an incoherent (no single correct phase) artifact
        # visually distinguishable from the genuine, phaseable peak
        # next to it.
        ax.contour(contour_x, contour_y, contour_data, colors='red', levels=sorted(neg_thresvec))

    # A plain make_axes_locatable(ax) divider's default reference sizes
    # (Size.AxesX/AxesY) are the axes' own *data-coordinate range*, not
    # its physical size -- fine for divider.set_aspect(False) (this
    # function's own non-square path, where the reference size choice
    # doesn't affect the result either way), but wrong for
    # square_aspect=True: equalising those would make 1 ppm on F2 equal
    # 1 ppm on F1 *physically* (data-aspect-locked), not make the panel
    # itself a square box -- and F1/F2 ppm spans are essentially never
    # numerically equal for real NMR data (confirmed directly: a real
    # HSQC's F2/F1 spans differed by ~10x, producing a panel roughly
    # 1 inch by 10 inches under the plain make_axes_locatable default,
    # not square at all, despite divider.set_aspect(True) being set).
    # Size.Scaled(1) is a plain, data-independent relative unit -- using
    # it for both directions makes divider.set_aspect(True) equalise
    # genuine physical box dimensions instead.
    divider = AxesDivider(ax, xref=axes_size.Scaled(1), yref=axes_size.Scaled(1))
    ax.set_axes_locator(divider.new_locator(nx=0, ny=0))
    divider.set_aspect(square_aspect)

    ax.set_xlabel(labeltextx, fontsize=fontsize)
    if pseudo2d:
        ax.set_yticks([])
    else:
        ax.set_ylabel(labeltexty, fontsize=fontsize)

    ax.tick_params(
        axis='both', which='both', direction=tick_dir, labelsize=fontsize,
        labelbottom=True, labelleft=not pseudo2d, bottom=True, left=True,
        length=tick_length, width=tick_width, pad=tick_pad,
    )
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    if xlims is not None:
        ax.set_xlim(min(xlims), max(xlims))
    if ylims is not None:
        ax.set_ylim(min(ylims), max(ylims))

    ax.invert_xaxis()

    if pseudo2d:
        if ylims is None:
            ax.set_ylim(y_axis.min() - 0.5, y_axis.max() + 0.5)
    else:
        ax.invert_yaxis()

    if equal_tickspacing:
        x_range = abs(np.diff(ax.get_xlim())[0])
        y_range = abs(np.diff(ax.get_ylim())[0])
        max_range = max(x_range, y_range)
        locator = MaxNLocator(nbins=tick_number if tick_number is not None else 6)
        tick_values = locator.tick_values(0, max_range)
        tick_spacing = np.min(np.diff(tick_values))
        ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(MultipleLocator(tick_spacing))
    elif tick_number is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=tick_number))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=tick_number))

    ax.tick_params(direction=tick_dir, labelsize=fontsize, length=tick_length,
                    width=tick_width, pad=tick_pad)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    if show_nQ_line:
        nQ_list = nQ if isinstance(nQ, (list, tuple, np.ndarray)) else [nQ]
        nQ_list = [n for n in nQ_list if n != 0]
        if nQ_colors is None:
            nQ_colors = ['gray'] * len(nQ_list)
        elif len(nQ_colors) != len(nQ_list):
            raise ValueError("nQ_colors must have the same length as nQ.")

        xmin, xmax = ax.get_xlim()
        x_line = np.linspace(xmin, xmax, 500)
        for n, colour in zip(nQ_list, nQ_colors):
            ax.plot(x_line, n * x_line, linestyle='--', linewidth=1.5,
                    color=colour, alpha=0.8, zorder=10)

    show_f2_projection = show_projections in (True, 'f2')
    show_f1_projection = show_projections in (True, 'f1')
    if show_projections not in (True, False, 'f1', 'f2'):
        raise ValueError("show_projections must be True, False, 'f1', or 'f2'")

    proj_axes = {"f1": None, "f2": None}

    if show_f2_projection or show_f1_projection:
        spec_float = np.asarray(data, dtype=np.float64)
        mode = projection_mode
        if mode == "auto":
            mode = "max" if pseudo2d else "sum"
        elif mode not in ("sum", "max"):
            raise ValueError("projection_mode must be 'auto', 'sum', or 'max'")

        if show_f2_projection:
            ax_f2 = divider.append_axes("top", projection_size, pad=projection_pad, sharex=ax)
            ax_f2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
            ax_f2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
            f2_projection = np.sum(spec_float, axis=0) if mode == "sum" else np.max(np.abs(spec_float), axis=0)
            ax_f2.plot(x_ppm, f2_projection + (0.02 * np.min(f2_projection)), 'k')
            ax_f2.set_ylim([np.min(f2_projection) * 0.9, np.max(f2_projection) * proj_scale])
            proj_axes["f2"] = ax_f2

        if show_f1_projection:
            ax_f1 = divider.append_axes("left", projection_size, pad=projection_pad, sharey=ax)
            ax_f1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
            ax_f1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
            f1_projection = np.sum(spec_float, axis=1) if mode == "sum" else np.max(np.abs(spec_float), axis=1)
            ax_f1.plot(-f1_projection + (0.02 * np.min(-f1_projection)), y_axis, 'k')
            ax_f1.set_xlim([np.min(-f1_projection) * proj_scale, np.max(-f1_projection) * 0.99])
            proj_axes["f1"] = ax_f1

        for proj_ax in proj_axes.values():
            if proj_ax is None:
                continue
            for spine in ('top', 'right', 'bottom', 'left'):
                proj_ax.spines[spine].set_visible(False)

    if standalone:
        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.subplots_adjust(left=0.15, right=0.88, bottom=0.12, top=0.88)
        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax, proj_axes


def multi_plot(spectra: list[NMRSpectrum], layout="row", panel_kwargs=None, titles=None,
               panel_width=5.0, panel_height=5.0, pl_width=None, pl_height=None,
               font='Arial', fontsize=12, output_format='',
               save_fig=False, fig_name='nmr_2d_multi', save_format='png',
               **shared_kwargs):
    """Plot multiple 2D contour panels (:func:`plot_2d_contour`) on one
    figure, arranged in a row, column, or grid -- one panel per
    spectrum. Ported from ``nmrtools2``'s ``NMR2D_multi_plot``.

    ``panel_kwargs`` : per-panel keyword overrides forwarded to
    :func:`plot_2d_contour`, one dict per panel (``{}`` for none),
    taking precedence over the same key in ``**shared_kwargs``.

    Figure-level :func:`plot_2d_contour` arguments (``ax``, ``fig``,
    ``save_fig``, ``fig_name``, ``save_format``, ``pl_width``,
    ``pl_height``, ``font``, ``fontsize``, ``output_format``) are
    owned by this function at the figure level -- passing them in
    ``**shared_kwargs`` or ``panel_kwargs`` raises ``ValueError``.

    Returns ``fig, axes`` (one ``Axes`` per panel, in ``spectra`` order).
    """
    if layout not in ("row", "column", "grid"):
        raise ValueError("layout must be 'row', 'column', or 'grid'")

    n = len(spectra)
    if n == 0:
        raise ValueError("spectra must contain at least one spectrum.")

    if panel_kwargs is not None and len(panel_kwargs) != n:
        raise ValueError("panel_kwargs must have the same length as spectra.")
    panel_kwargs = panel_kwargs if panel_kwargs is not None else [{}] * n

    if titles is not None and len(titles) != n:
        raise ValueError("titles must have the same length as spectra.")

    figure_level_keys = {
        "ax", "fig", "save_fig", "fig_name", "save_format",
        "pl_width", "pl_height", "font", "fontsize", "output_format",
    }
    bad = figure_level_keys & set(shared_kwargs)
    if bad:
        raise ValueError(
            f"Figure-level keys are not valid as shared per-panel kwargs: {sorted(bad)}; "
            f"use multi_plot's own same-named arguments instead."
        )
    for i, kw in enumerate(panel_kwargs):
        bad = figure_level_keys & set(kw)
        if bad:
            raise ValueError(f"panel_kwargs[{i}] contains figure-level keys not valid per panel: {sorted(bad)}")

    setup_plotting(font=font, fontsize=fontsize, output_format=output_format)

    if layout == "row":
        nrows, ncols = 1, n
    elif layout == "column":
        nrows, ncols = n, 1
    else:
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))

    if pl_width is None:
        pl_width = panel_width * ncols
    if pl_height is None:
        pl_height = panel_height * nrows

    fig = plt.figure(figsize=(pl_width, pl_height))
    gs = fig.add_gridspec(nrows, ncols)

    axes = []
    for i, spectrum in enumerate(spectra):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        kw = dict(shared_kwargs)
        kw.update(panel_kwargs[i])
        plot_2d_contour(spectrum, ax=ax, fig=fig, **kw)
        if titles is not None:
            ax.set_title(titles[i])
        axes.append(ax)

    fig.tight_layout()
    if save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, axes


def _check_overlay_compatible(spectra: list[NMRSpectrum]):
    """Raise ``ValueError`` unless every spectrum shares the same
    F1/F2 nuclides and 2D experiment type (conventional vs pseudo-2D,
    homo- vs heteronuclear) -- i.e. their axes actually mean the same
    thing and can be overlaid directly. Deliberately does not compare
    SFO1 (transmitter frequency): ppm is field-independent, so the
    same experiment at two fields is still a valid overlay."""
    ref = spectra[0]
    for i, s in enumerate(spectra[1:], start=1):
        if (s.metadata.nucleus, s.metadata.nucleus_indirect) != (ref.metadata.nucleus, ref.metadata.nucleus_indirect):
            raise ValueError(
                f"Cannot overlay spectra with different F1/F2 nuclides: "
                f"spectrum 0 is {ref.metadata.nucleus}/{ref.metadata.nucleus_indirect}, "
                f"spectrum {i} is {s.metadata.nucleus}/{s.metadata.nucleus_indirect}."
            )
        if s.is_pseudo_2d != ref.is_pseudo_2d:
            raise ValueError(f"Cannot overlay a pseudo-2D spectrum with a conventional 2D spectrum (spectrum {i} vs 0).")
        if s.metadata.homonuclear != ref.metadata.homonuclear:
            raise ValueError(f"Cannot overlay a homonuclear spectrum with a heteronuclear spectrum (spectrum {i} vs 0).")


def overlay_plot(spectra: list[NMRSpectrum], contour_fraction_x=0.25, contour_fraction_y=0.25,
                  ncontourpts_x=16384, ncontourpts_y=1024,
                  pl_height=7.0, pl_width=7.0, factor=0.07, clevels=7,
                  colours='diverging', labels=None, legend_kwargs=None,
                  field='', xlabel=None, ylabel=None, xlims=None, ylims=None,
                  font='Arial', fontsize=12, output_format='',
                  equal_tickspacing=False, tick_number=None,
                  tick_dir='in', tick_length=4, tick_width=1.0, tick_pad=4,
                  show_nQ_line=False, nQ=1, nQ_colors=None,
                  ax=None, fig=None, save_fig=False, fig_name='nmr_2d_overlay',
                  save_format='png'):
    """Overlay 2-3 2D NMR spectra as line contours on one set of axes,
    each in its own flat colour (cf. :func:`multi_plot`'s side-by-side
    panels). Ported from ``nmrtools2``'s ``NMR2D_overlay_plot``.

    Checks every spectrum shares the same F1/F2 nuclides and
    experiment type first (:func:`_check_overlay_compatible`) and
    raises ``ValueError`` otherwise.

    ``colours`` : ``'diverging'`` (default, from
    :func:`opennmr.plotting.style.get_stack_colormap`) or an explicit
    list of matplotlib colours, one per spectrum.

    Returns ``fig, ax``.
    """
    if not (2 <= len(spectra) <= 3):
        raise ValueError(f"overlay_plot accepts 2 or 3 spectra, got {len(spectra)}.")

    _check_overlay_compatible(spectra)
    n = len(spectra)

    if colours == 'diverging':
        cmap = get_stack_colormap('diverging')
        resolved_colours = [cmap(i / max(n - 1, 1)) for i in range(n)]
    elif isinstance(colours, (list, tuple)):
        if len(colours) != n:
            raise ValueError("colours must have the same length as spectra.")
        resolved_colours = list(colours)
    else:
        raise ValueError("colours must be 'diverging' or a list of colours, one per spectrum.")

    if labels is not None and len(labels) != n:
        raise ValueError("labels must have the same length as spectra.")

    ref = spectra[0]
    pseudo2d = ref.is_pseudo_2d
    homonuclear = ref.metadata.homonuclear

    nuc1 = find_gamma(ref.metadata.nucleus, field=field, SFO1=ref.metadata.sfo1)['Nuclide']
    labeltextx = NUC_label(nuc1, x_unit='ppm')
    if homonuclear:
        labeltexty = NUC_label(nuc1, x_unit='ppm')
    elif not pseudo2d:
        labeltexty = NUC_label(ref.metadata.nucleus_indirect, x_unit='ppm')
    else:
        labeltexty = 'Slice number'
    if ylabel is not None:
        labeltexty = ylabel
    if xlabel is not None:
        labeltextx = xlabel

    standalone = ax is None
    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style['pl_width'], style['pl_height']
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    for spectrum, colour, label in zip(spectra, resolved_colours, labels if labels is not None else [None] * n):
        contour_x, contour_y, contour_data, thresvec, _, _ = _prepare_contour(
            spectrum, contour_fraction_x=contour_fraction_x, contour_fraction_y=contour_fraction_y,
            ncontourpts_x=ncontourpts_x, ncontourpts_y=ncontourpts_y, factor=factor, clevels=clevels,
        )
        ax.contour(contour_x, contour_y, contour_data, colors=[colour], levels=thresvec)
        if label is not None:
            ax.plot([], [], color=colour, label=label)

    ax.set_xlabel(labeltextx, fontsize=fontsize)
    if pseudo2d:
        ax.set_yticks([])
    else:
        ax.set_ylabel(labeltexty, fontsize=fontsize)

    ax.tick_params(
        axis='both', which='both', direction=tick_dir, labelsize=fontsize,
        labelbottom=True, labelleft=not pseudo2d, bottom=True, left=True,
        length=tick_length, width=tick_width, pad=tick_pad,
    )
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    if xlims is not None:
        ax.set_xlim(min(xlims), max(xlims))
    if ylims is not None:
        ax.set_ylim(min(ylims), max(ylims))

    ax.invert_xaxis()

    if pseudo2d:
        if ylims is None:
            y_axis = ref.axes[0].index
            ax.set_ylim(y_axis.min() - 0.5, y_axis.max() + 0.5)
    else:
        ax.invert_yaxis()

    if equal_tickspacing:
        x_range = abs(np.diff(ax.get_xlim())[0])
        y_range = abs(np.diff(ax.get_ylim())[0])
        max_range = max(x_range, y_range)
        locator = MaxNLocator(nbins=tick_number if tick_number is not None else 6)
        tick_values = locator.tick_values(0, max_range)
        tick_spacing = np.min(np.diff(tick_values))
        ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(MultipleLocator(tick_spacing))
    elif tick_number is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=tick_number))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=tick_number))

    ax.tick_params(direction=tick_dir, labelsize=fontsize, length=tick_length, width=tick_width, pad=tick_pad)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    if show_nQ_line:
        nQ_list = nQ if isinstance(nQ, (list, tuple, np.ndarray)) else [nQ]
        nQ_list = [n_ for n_ in nQ_list if n_ != 0]
        if nQ_colors is None:
            nQ_colors = ['gray'] * len(nQ_list)
        elif len(nQ_colors) != len(nQ_list):
            raise ValueError("nQ_colors must have the same length as nQ.")

        xmin, xmax = ax.get_xlim()
        x_line = np.linspace(xmin, xmax, 500)
        for nq, colour in zip(nQ_list, nQ_colors):
            ax.plot(x_line, nq * x_line, linestyle='--', linewidth=1.5, color=colour, alpha=0.8, zorder=10)

    if labels is not None:
        ax.legend(**(legend_kwargs or {}))

    if standalone:
        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.subplots_adjust(left=0.15, right=0.88, bottom=0.12, top=0.88)
        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax
