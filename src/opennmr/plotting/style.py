"""
Plotting style helpers shared by ``oned``/``twod``.

``apply_tick_style``/``apply_spine_style``/``get_stack_colormap``/
``get_contour_cmap``/``downsample_2d``/``save_figure`` are ``ax``-
scoped (they only ever touch the ``Axes``/data passed to them), so
they're safe to call from GUI code embedding a canvas -- ported from
``nmrtools2``'s ``plotting.py``. ``downsample_2d`` has since diverged
from that original (still evenly-spaced decimation there) -- see its
own docstring for why.

``setup_plotting`` is the one exception: it calls
``plt.rcParams.update(...)``, a *global* matplotlib style change. That
is harmless for a one-off standalone figure (a script or notebook cell
calling one of ``plotting.oned``/``plotting.twod``'s convenience
wrappers), which is the only place this package still calls it -- but
it must never be called from GUI code, since restyling one embedded
canvas would silently restyle every other open canvas/tab sharing the
same matplotlib process.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

VALID_SAVE_FORMATS = ("png", "pdf", "svg")

CONTOUR_CMAP_NAMES = [
    'magma', 'coolwarm', 'cividis', 'viridis', 'plasma', 'inferno',
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn',
    'Spectral', 'bwr', 'seismic', 'Greys', 'Purples', 'Blues', 'Greens',
    'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
    'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
]


def apply_tick_style(ax, direction='in', length=4.0, width=1.0, pad=4.0,
                      num_ticks=None, axis='both', minor=True, labelsize=None):
    """Apply consistent tick styling to an ``ax``: direction, size,
    and (optionally) an explicit target tick count.

    ``length``/``width`` : major tick length/width. Minor ticks (when
    ``minor=True``) are drawn at 3/4 the width and half the length.
    ``num_ticks`` : if given, an approximate target major-tick count
    via ``MaxNLocator`` on the requested axis/axes.
    ``labelsize`` : if given, sets the tick label font size directly
    (equivalent to ``setup_plotting``'s ``xtick.labelsize``/
    ``ytick.labelsize`` rcParams, but ax-scoped -- needed anywhere
    ``setup_plotting`` itself can't be called, e.g. a GUI canvas).
    """
    tick_kwargs = {} if labelsize is None else {'labelsize': labelsize}
    ax.tick_params(axis=axis, which='major', direction=direction,
                    length=length, width=width, pad=pad, **tick_kwargs)

    if minor:
        ax.minorticks_on()
        ax.tick_params(axis=axis, which='minor', direction=direction,
                        length=length / 2, width=0.75 * width)
    else:
        ax.minorticks_off()

    if num_ticks is not None:
        if axis in ('x', 'both'):
            ax.xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
        if axis in ('y', 'both'):
            ax.yaxis.set_major_locator(MaxNLocator(nbins=num_ticks))

    return ax


def apply_spine_style(ax, top=False, right=False, bottom=True, left=True, linewidth=None):
    """Show/hide individual axis spines. Defaults match the package's
    usual "open" plot style (top and right hidden).

    ``linewidth`` : if given, also sets every spine's line width
    (equivalent to ``setup_plotting``'s ``axes.linewidth`` rcParam,
    but ax-scoped) -- needed anywhere ``setup_plotting`` itself can't
    be called, e.g. a GUI canvas, since a spine's width otherwise just
    falls back to matplotlib's own default regardless of what
    ``apply_tick_style`` was given."""
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)
    if linewidth is not None:
        for spine in ax.spines.values():
            spine.set_linewidth(linewidth)
    return ax


def setup_plotting(font='Arial', fontsize=12, pl_width=3.375, pl_height=3.375,
                    output_format=''):
    """Apply the shared matplotlib style and resolve figure dimensions
    for a standalone (non-GUI) figure -- see the module docstring for
    why this must not be called from GUI code.

    ``output_format`` : ``''``, ``'manuscript'`` (3.375x3.375in,
    fontsize 12), or ``'presentation'`` (10x10in, fontsize 24).

    Returns a dict with the resolved ``fontsize``, ``pl_width``,
    ``pl_height``, ``lw`` (line width, scaled to figure width), and
    ``ticklen``.
    """
    if output_format == 'manuscript':
        fontsize = 12
        pl_height = 3.375
        pl_width = 3.375
    elif output_format == 'presentation':
        fontsize = 24
        pl_height = 10
        pl_width = 10

    lw = pl_width / 3.375
    ticklen = lw * 5

    plt.rcParams['font.sans-serif'] = [
        'DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif',
        'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Arial', 'Helvetica',
        'Avant Garde', 'sans-serif', 'Tahoma',
    ]

    params = {
        'legend.fontsize': 'large',
        'font.family': 'sans-serif',
        'font.sans-serif': font,
        'font.weight': 'regular',
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'xtick.major.width': lw,
        'xtick.major.size': ticklen,
        'xtick.minor.visible': True,
        'xtick.minor.width': 3 / 4 * lw,
        'xtick.minor.size': ticklen / 2,
        'ytick.labelsize': fontsize,
        'ytick.major.width': lw,
        'ytick.major.size': ticklen,
        'ytick.minor.visible': True,
        'ytick.minor.width': 3 / 4 * lw,
        'ytick.minor.size': ticklen / 2,
        'axes.labelweight': 'regular',
        'axes.linewidth': lw,
        'axes.titlepad': 25,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    }

    plt.rcParams.update(params)

    return {
        'fontsize': fontsize,
        'pl_width': pl_width,
        'pl_height': pl_height,
        'lw': lw,
        'ticklen': ticklen,
    }


def get_stack_colormap(color='black'):
    """Colour map used to distinguish a stack of traces (multiple 1D
    spectra, or extracted 2D row/column slices).

    ``color`` : ``'black'`` (every trace black), ``'diverging'`` (a
    four-stop brown/teal diverging map), or any matplotlib colormap
    name (falls back to black if not recognised).
    """
    if color == 'diverging':
        return cl.LinearSegmentedColormap.from_list(
            "", ["#a6611a", "#dfc27d", "#80cdc1", "#018571"]
        )

    if color == 'black':
        return cl.LinearSegmentedColormap.from_list(
            "", ["#000000", "#000000", "#000000", "#000000"]
        )

    try:
        return plt.get_cmap(color)
    except (ValueError, KeyError):
        print(f"Colour scheme '{color}' not recognised; defaulting to black.")
        return cl.LinearSegmentedColormap.from_list(
            "", ["#000000", "#000000", "#000000", "#000000"]
        )


def stack_colors(n, color='black'):
    """``n`` RGBA colours evenly spaced along :func:`get_stack_colormap`
    -- the exact colour-resolving logic ``plot_1d_stack`` uses per
    trace, factored out so other stacked-trace views (e.g. a GUI
    canvas plotting each trace on its own subplot rather than
    ``plot_1d_stack``'s own single shared axes) can match it without
    duplicating the ``Normalize``/``ScalarMappable`` setup."""
    values = range(n)
    norm = plt.Normalize(vmin=values[0], vmax=values[-1]) if n > 1 else plt.Normalize(vmin=0, vmax=1)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=get_stack_colormap(color))
    return [scalar_map.to_rgba(v) for v in values]


def get_contour_cmap(name):
    """Look up a named colormap for 2D contour plots, restricted to
    :data:`CONTOUR_CMAP_NAMES`."""
    if name not in CONTOUR_CMAP_NAMES:
        raise ValueError(
            f"Unknown contour colour scheme '{name}'. "
            f"Available options: {sorted(CONTOUR_CMAP_NAMES)}"
        )
    try:
        return plt.get_cmap(name)
    except AttributeError:  # pragma: no cover - very old matplotlib
        return cm.get_cmap(name)


def save_figure(fig, fig_name='nmr_plot', save_format='png', dpi=300):
    """Save a figure using the package's shared saving convention.

    ``fig_name`` : output path/filename, with or without an
    extension. If it already ends in ``.png``/``.pdf``/``.svg``, that
    extension is used as-is and ``save_format`` is ignored; otherwise
    ``.{save_format}`` is appended.

    Returns the path the figure was actually saved to.
    """
    if save_format not in VALID_SAVE_FORMATS:
        raise ValueError(
            f"save_format must be one of {VALID_SAVE_FORMATS}, got '{save_format}'."
        )

    root, ext = os.path.splitext(fig_name)
    path = fig_name if ext.lower().lstrip('.') in VALID_SAVE_FORMATS else f"{fig_name}.{save_format}"

    fig.savefig(path, dpi=dpi)
    return path


def downsample_2d(data, x, y, max_points_x=2000, max_points_y=500):
    """Downsample a 2D array (and its axes) for fast contour
    rendering, keeping each output cell's own largest-magnitude
    (signed) value rather than a plain evenly-spaced index pick.

    ``nmrtools2``'s original (and this function's own, until this
    fix) picked evenly-spaced *indices* (``np.linspace``) and threw
    everything else away -- fine for a smooth, broad feature, but a
    thin/sparse one (an EXSY/NOESY diagonal ridge, often only 1-2
    points wide at real digital resolution) can fall entirely between
    the sampled indices and vanish, or survive at scattered points
    that no longer look like a continuous ridge -- even though the
    underlying data never changed. Real user report, confirmed by
    direct reproduction against a real 2H EXSY dataset: the Processing
    tab's own preview (this function, ~16x fewer total points at the
    default 0.25 contour fraction) and the Spectrum tab's own full-
    resolution view of the *exact same spectrum object* looked like
    "two different 2Ds" -- the diagonal visible in one, not the other.
    Max-magnitude-per-bin instead of index-skipping means a bin
    containing any part of a real feature always keeps *a* point from
    it, not nothing.

    Reduces one axis at a time (X then Y) via :func:`_max_abs_bin` --
    a separable approximation of a true joint-2D block-max (which
    would need reducing both axes of each block simultaneously), but
    a large improvement over index-skipping, and correct for the
    common case (contour fraction applied independently per axis,
    matching every existing caller's own ``contour_fraction_x``/
    ``contour_fraction_y`` convention).

    **Not free, unlike the old index-skip -- has to touch every input
    element at least once to find each bin's own max, where index-
    skipping only ever touched the output-sized subset.** Measured
    directly on a real, large conventional-2D array (1024x65536, a
    real EXSY dataset): ~0.35s here vs. a few ms for the old method.
    Fine for a one-off settled redraw (spectrum selection, a drag
    settling), but a real, known cost during a *live 2D phase-drag*
    on data this large specifically, since that redraws on every
    mouse-move tick with no in-place update path for a contour (see
    ``MplCanvas.set_full_resolution``'s own docstring on why). Not
    optimized further here -- the actual fix for 2D drag
    responsiveness at any data size is phasing only the relevant row/
    column during the drag itself, not re-rendering (let alone
    downsampling) the whole contour on every tick; see this project's
    own TODO.md "2D roadmap" for that separate, queued work.

    ``data`` : shape ``(len(y), len(x))``. Returns ``data_ds, x_ds,
    y_ds`` unchanged (well, each axis fed back through unchanged) if
    already within the point-count limits.
    """
    n_y, n_x = data.shape

    max_points_x = max(2, min(max_points_x, n_x))
    max_points_y = max(2, min(max_points_y, n_y))

    # X (columns, the last/contiguous axis) first -- reducing along
    # the last axis keeps every argmax/take_along_axis's own memory
    # access contiguous; reducing Y (a middle axis once reshaped)
    # first was measured directly to be *slower* on a real-sized
    # array despite operating on fewer total elements, since each of
    # its per-bin comparisons strides across memory instead of
    # scanning it in order.
    data_ds, x_ds = _max_abs_bin(data, x, max_points_x, axis=1)
    data_ds, y_ds = _max_abs_bin(data_ds, y, max_points_y, axis=0)

    return data_ds, x_ds, y_ds


def _max_abs_bin(data, coords, n_bins, axis):
    """Reduce ``data``/``coords`` along ``axis`` to ``n_bins`` bins --
    each bin keeps the *signed* value with the largest magnitude
    across its own span (not just an evenly-spaced index pick, see
    :func:`downsample_2d`'s own docstring for why), with the bin's own
    middle-index coordinate standing in for the whole span. A no-op
    (returns ``data``/``coords`` unchanged) if ``n_bins`` already
    covers every index along ``axis``.

    Fully vectorized (reshape + one ``argmax``/``take_along_axis`` --
    no Python-level loop over bins) -- an earlier version used
    ``numpy.array_split`` plus a per-bin loop instead, simpler but
    ~0.7s on a real 1024x65536 contour, unusably slow for a live 2D
    phase-drag tick (a fresh downsample on every mouse-move, since a
    ``QuadContourSet`` has no in-place data swap the way a ``Line2D``
    does). Achieves the uniform bin size a reshape needs by trimming
    at most ``n_bins - 1`` trailing indices (never more than one bin's
    own width) rather than requiring ``n_bins`` to evenly divide the
    axis length -- a negligible edge effect for any real acquisition
    size, and the same "drop a few, don't fail" spirit as this
    project's own raw-``ser``-shape floor-division fallback."""
    n = data.shape[axis]
    if n_bins >= n:
        return data, coords
    bin_size = n // n_bins
    trimmed_n = bin_size * n_bins
    if trimmed_n != n:
        data = np.take(data, np.arange(trimmed_n), axis=axis)
    new_shape = data.shape[:axis] + (n_bins, bin_size) + data.shape[axis + 1:]
    reshaped = data.reshape(new_shape)
    peak_index = np.argmax(np.abs(reshaped), axis=axis + 1, keepdims=True)
    reduced = np.take_along_axis(reshaped, peak_index, axis=axis + 1)
    reduced = np.squeeze(reduced, axis=axis + 1)
    reduced_coords = coords[:trimmed_n].reshape(n_bins, bin_size)[:, bin_size // 2]
    return reduced, reduced_coords


#: Default cap for :func:`downsample_indices` -- rendering (not
#: fitting/processing) more points than this per 1D line rarely adds
#: visible detail at any normal screen resolution, but does cost real
#: time redrawing, especially for a *dynamic* view (phase-dragging
#: redraws on every mouse-move tick) or *many* lines at once (a
#: Compare-tab stack of dozens of traces) -- both explicitly reported
#: as slow for high-point-count data (diffusion/imaging acquisitions,
#: or just a high digital resolution, routinely land in the
#: 10^5-10^6 point range).
DOWNSAMPLE_1D_MAX_POINTS = 8000


def downsample_indices(n, max_points=DOWNSAMPLE_1D_MAX_POINTS):
    """Evenly-spaced indices for downsampling a length-``n`` 1D array
    to at most ``max_points``, for fast *display* rendering only --
    the same idea as :func:`downsample_2d`, for a 1D line instead of a
    2D contour. Returns ``None`` (not an index array) when ``n`` is
    already within ``max_points``, so a caller can cheaply skip
    indexing altogether in the (usual) case no downsampling is
    actually needed.

    Deterministic in ``n``/``max_points`` alone (no data-dependent
    peak-preservation logic) so that re-deriving it later for a
    same-length array -- e.g. :meth:`~opennmr.gui.widgets.canvas.MplCanvas.update_data`
    updating a line's y-values in place after :func:`~opennmr.plotting.oned.plot_1d`
    already downsampled its x -- reproduces the exact same indices,
    without either function needing to remember or pass them to the
    other.
    """
    if n <= max_points:
        return None
    return np.linspace(0, n - 1, max_points, dtype=int)
