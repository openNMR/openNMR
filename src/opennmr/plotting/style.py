"""
Plotting style helpers shared by ``oned``/``twod``.

``apply_tick_style``/``apply_spine_style``/``get_stack_colormap``/
``get_contour_cmap``/``downsample_2d``/``save_figure`` are ``ax``-
scoped (they only ever touch the ``Axes``/data passed to them), so
they're safe to call from GUI code embedding a canvas -- ported from
``nmrtools2``'s ``plotting.py`` unchanged.

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

    plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Bitstream Vera Sans', 'Arial', 'Helvetica', 'sans-serif']

    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = font      
    plt.rcParams['mathtext.it'] = f'{font}:italic'
    plt.rcParams['mathtext.bf'] = f'{font}:bold'
    plt.rcParams['mathtext.default'] = 'regular'  

    params = {
        'legend.fontsize': 'large',
        'font.family': 'sans-serif',
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
    """Evenly downsample a 2D array (and its axes) for fast contour
    rendering.

    ``data`` : shape ``(len(y), len(x))``. Returns ``data_ds, x_ds,
    y_ds`` unchanged if already within the point-count limits.
    """
    n_y, n_x = data.shape

    max_points_x = max(2, min(max_points_x, n_x))
    max_points_y = max(2, min(max_points_y, n_y))

    x_idx = np.linspace(0, n_x - 1, max_points_x, dtype=int)
    y_idx = np.linspace(0, n_y - 1, max_points_y, dtype=int)

    data_ds = data[np.ix_(y_idx, x_idx)]
    x_ds = x[x_idx]
    y_ds = y[y_idx]

    return data_ds, x_ds, y_ds


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
