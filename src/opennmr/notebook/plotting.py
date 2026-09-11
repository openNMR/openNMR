"""
Plotting utilities shared by ``one_d``, ``two_d`` and ``fitting``.

Previously the font/rcParams setup and colour-mapping logic were
duplicated (with small, silent inconsistencies) between the 1D and 2D
plotting functions. This module is the single source of truth for:

- ``setup_plotting``   : matplotlib rcParams (fonts, tick styling, figure
                          size) shared by every plot in the package.
- ``get_stack_colormap``: colours used to distinguish a stack of traces
                          (multiple 1D spectra, or extracted 2D slices).
- ``get_contour_cmap`` : named colormaps offered for 2D contour plots.
- ``downsample_2d``    : evenly-spaced decimation of a 2D array for fast
                          contour rendering.
- ``save_figure``      : shared ``fig_name``/``save_format`` saving
                          convention used by every plotting function
                          that generates its own figure.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

# One implementation of the too-light-colour rule, in the live
# plotting package; this legacy copy delegates rather than repeating it.
from ..plotting.style import readable_cmap

# Output formats every save-capable plotting function in nmrtools
# accepts for its ``save_format`` argument.
VALID_SAVE_FORMATS = ('png', 'pdf', 'svg')


# ============================================================
# Tick / spine styling
# ============================================================

def apply_tick_style(ax, direction='in', length=4.0, width=1.0, pad=4.0,
                      num_ticks=None, axis='both', minor=True):
    """
    Apply consistent tick styling to an axis -- the single place every
    plotting function in nmrtools goes for tick direction, size, and
    (optionally) an explicit target tick count.

    Parameters
    ----------
    ax : matplotlib.axes.Axes

    direction : {'in', 'out', 'inout'}, optional

    length, width : float, optional
        Major tick length/width. Minor ticks (when ``minor=True``) are
        drawn at 3/4 the width and half the length.

    pad : float, optional
        Padding between ticks and tick labels.

    num_ticks : int, optional
        If given, set an approximate target number of major ticks via
        ``MaxNLocator`` on the requested axis/axes. If None, the
        existing locator is left alone.

    axis : {'both', 'x', 'y'}, optional
        Which axis/axes to style.

    minor : bool, optional
        If True (default), also draw and style minor ticks.

    Returns
    -------
    ax
    """

    ax.tick_params(axis=axis, which='major', direction=direction,
                    length=length, width=width, pad=pad)

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


def apply_spine_style(ax, top=False, right=False, bottom=True, left=True):
    """Show/hide individual axis spines. Defaults match nmrtools' usual
    "open" plot style (top and right hidden)."""
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)
    return ax


# ============================================================
# rcParams / figure setup
# ============================================================

def setup_plotting(font='Arial', fontsize=12, pl_width=3.375, pl_height=3.375,
                    output_format=''):
    """
    Apply the shared nmrtools matplotlib style and resolve figure
    dimensions.

    Parameters
    ----------
    font : str
        Font family.

    fontsize : float
        Base font size for labels/ticks.

    pl_width, pl_height : float
        Figure size in inches.

    output_format : {'', 'manuscript', 'presentation'}
        Predefined formatting presets.

        ``'manuscript'``   : 3.375 x 3.375 in, fontsize 12.
        ``'presentation'`` : 10 x 10 in, fontsize 24.

    Returns
    -------
    dict with the resolved 'fontsize', 'pl_width', 'pl_height', 'lw'
    (line width, scaled to figure width) and 'ticklen'.
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
    plt.rcParams['mathtext.rm'] = font        # regular
    plt.rcParams['mathtext.it'] = f'{font}:italic'
    plt.rcParams['mathtext.bf'] = f'{font}:bold'
    plt.rcParams['mathtext.default'] = 'regular'  # avoids italicising bare mathtext text like 'mix'

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


# ============================================================
# Colour mapping
# ============================================================

def get_stack_colormap(color='black'):
    """
    Return a matplotlib colormap used to colour a stack of traces
    (multiple 1D spectra, or extracted 2D row/column slices).

    Parameters
    ----------
    color : str
        ``'black'``     : every trace plotted in black.
        ``'diverging'``  : a four-stop brown/teal diverging colormap.
        any other string : looked up as a standard matplotlib colormap
            name (e.g. ``'coolwarm'``, ``'viridis'``), falling back to
            black if not recognised.

    Returns
    -------
    matplotlib.colors.Colormap
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
        # Through the shared luminance filter, so this legacy notebook
        # API draws with the same readable colours the live plotting
        # package does rather than quietly keeping the near-white ends
        # -- see opennmr.plotting.style.MAX_LINE_LUMINANCE. Delegated
        # rather than reimplemented: two copies of a rule like this
        # drift.
        return readable_cmap(plt.get_cmap(color))
    except (ValueError, KeyError):
        print(f"Colour scheme '{color}' not recognised; defaulting to black.")
        return cl.LinearSegmentedColormap.from_list(
            "", ["#000000", "#000000", "#000000", "#000000"]
        )


# Named colormaps offered for 2D contour plots (kept as an explicit
# allow-list rather than exposing every matplotlib colormap, since not
# all of them are useful/legible for NMR contour plots).
CONTOUR_CMAP_NAMES = [
    'magma', 'coolwarm', 'cividis', 'viridis', 'plasma', 'inferno',
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn',
    'Spectral', 'bwr', 'seismic', 'Greys', 'Purples', 'Blues', 'Greens',
    'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
    'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
]


def get_contour_cmap(name):
    """
    Look up a named colormap for 2D contour plots, restricted to
    ``CONTOUR_CMAP_NAMES``.
    """
    if name not in CONTOUR_CMAP_NAMES:
        raise ValueError(
            f"Unknown contour colour scheme '{name}'. "
            f"Available options: {sorted(CONTOUR_CMAP_NAMES)}"
        )
    try:
        base = plt.get_cmap(name)  # modern matplotlib
    except AttributeError:  # pragma: no cover - very old matplotlib
        base = cm.get_cmap(name)
    return readable_cmap(base)


# ============================================================
# Saving
# ============================================================

def save_figure(fig, fig_name='nmr_plot', save_format='png', dpi=300):
    """
    Save a figure using nmrtools' shared save convention -- the single
    place every plotting function in the package goes to write out a
    ``fig_name`` with a chosen (or default) format.

    Parameters
    ----------
    fig : matplotlib.figure.Figure

    fig_name : str, optional
        Output path/filename. May be given with or without an
        extension:

        - If ``fig_name`` already ends in one of
          ``.png``/``.pdf``/``.svg``, that extension is used as-is and
          ``save_format`` is ignored.
        - Otherwise, ``.{save_format}`` is appended.

        Default ``'nmr_plot'``.

    save_format : {'png', 'pdf', 'svg'}, optional
        Output format used when ``fig_name`` has no recognised
        extension. Default ``'png'``.

    dpi : float, optional
        Resolution used when saving raster (``png``) output. Ignored
        for vector formats (``pdf``, ``svg``). Default 300.

    Returns
    -------
    str
        The path the figure was actually saved to.
    """

    if save_format not in VALID_SAVE_FORMATS:
        raise ValueError(
            f"save_format must be one of {VALID_SAVE_FORMATS}, got '{save_format}'."
        )

    root, ext = os.path.splitext(fig_name)

    if ext.lower().lstrip('.') in VALID_SAVE_FORMATS:
        path = fig_name
    else:
        path = f"{fig_name}.{save_format}"

    fig.savefig(path, dpi=dpi)

    return path


# ============================================================
# Downsampling
# ============================================================

def downsample_2d(data, x, y, max_points_x=2000, max_points_y=500):
    """
    Evenly downsample a 2D array (and its axes) for fast contour
    rendering.

    Parameters
    ----------
    data : ndarray, shape (len(y), len(x))
        2D intensity matrix.

    x, y : ndarray
        Axes corresponding to the columns/rows of ``data``.

    max_points_x, max_points_y : int
        Maximum number of points retained along each axis. If the
        input already has fewer points than this, it is returned
        unchanged.

    Returns
    -------
    data_ds, x_ds, y_ds : ndarray
        Downsampled data and axes.
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
