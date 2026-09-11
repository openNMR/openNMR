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
- ``plot_spectrum_dict``, ``plot_comparison``, ``plot_stacked_spectra``
                        : plot any dict of the shared 10-key spectrum
                          shape (x_ppm, x_Hz, real_spectrum,
                          normalised_real_spectrum, imag_spectrum,
                          normalised_imag_spectrum, nuclide, BF1, SFO1,
                          path) -- the shape returned by NMR1dimport
                          (one_d.py, experimental), generate_spectrum_dict
                          (pnmr_simulator.py, CIF-predicted) and
                          generate_bms_spectrum_dict (bms_spectrum.py,
                          BMS-predicted). Moved here (previously defined
                          only inside pnmr_simulator.py) since none of
                          the three functions below have anything
                          CIF-specific in them -- keeping them in a
                          module about CIF parsing was just an accident
                          of which pipeline was written first, and meant
                          bms_spectrum.py would otherwise have had to
                          import plotting code from a CIF module to plot
                          a result that has nothing to do with CIFs.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

<<<<<<< HEAD
from .utils import NUC_label
=======
# One implementation of the too-light-colour rule, in the live
# plotting package; this legacy copy delegates rather than repeating it.
from ..plotting.style import readable_cmap
>>>>>>> upstream/main

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


# ============================================================
# Generic spectrum_dict plotting
# ============================================================
#
# All three functions below share the same "usual plot modifiers":
#
#   tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
#   tick_number=None, xlims='default', pl_width=3.375, pl_height=3.375,
#   font='Arial', fontsize=12, fname=True
#
# xlims : 'default' auto-ranges from the plotted data (inverted per NMR
#         convention when invert_x=True); pass an explicit (xmin, xmax) to
#         override (used exactly as given, via ax.set_xlim(*xlims)).
# pl_width / pl_height : figure size in inches, only used when `ax=None`
#         (i.e. this call creates its own figure).
# fname : True  -> auto-name and save next to the spectrum's own 'path'
#                   (falls back to the current directory if unavailable);
#         str   -> save to that exact path;
#         False/None -> don't save.

def _apply_axis_style(ax, tick_dir, tick_length, tick_width, tick_pad,
                       tick_number, fontsize, show_yaxis):
    ax.tick_params(axis='x', direction=tick_dir, length=tick_length,
                    width=tick_width, pad=tick_pad, labelsize=fontsize)
    if tick_number is not None:
        ax.xaxis.set_major_locator(MaxNLocator(tick_number))

    if show_yaxis:
        ax.tick_params(axis='y', direction=tick_dir, length=tick_length,
                        width=tick_width, pad=tick_pad, labelsize=fontsize)
    else:
        ax.yaxis.set_visible(False)
        for spine in ("left", "right", "top"):
            ax.spines[spine].set_visible(False)


def _save_spectrum_figure(fig, fname, default_dir, default_name):
    """Save-convention for the spectrum_dict plotters below. Distinct from
    save_figure() above (which is fig_name/save_format-driven and used by
    one_d.py/two_d.py's own figures): this one derives its default
    filename/location from the spectrum_dict itself ('path' + nuclide),
    since a simulated spectrum_dict generally has no user-chosen fig_name
    the way a NMR1dplot() call does."""
    if not fname:
        return
    out_path = os.path.join(default_dir or ".", default_name) if fname is True else fname
    fig.savefig(out_path, dpi=300, bbox_inches="tight")


def plot_spectrum_dict(spectrum_dict, ax=None, axis="ppm", channel="real",
                        normalised=True, invert_x=True, offset=0.0, label=None,
                        show_yaxis=False,
                        tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                        tick_number=None, xlims='default', pl_width=3.375, pl_height=3.375,
                        font='Arial', fontsize=12, fname=True, **plot_kwargs):
    """
    Plot a spectrum dict (experimental, CIF-simulated, or BMS-simulated -
    same shape) on a Matplotlib Axes.

    axis     : "ppm" or "Hz"
    channel  : "real" or "imag"
    normalised : use the normalised_* arrays
    offset   : constant added to y after selecting the channel/normalisation
               (e.g. to stack several plot_spectrum_dict calls on one Axes)
    """
    plt.rcParams['font.family'] = font

    x_key = "x_ppm" if axis == "ppm" else "x_Hz"
    prefix = "normalised_" if normalised else ""
    y_key = f"{prefix}{channel}_spectrum"

    x = spectrum_dict[x_key]
    y = spectrum_dict[y_key] + offset

    if ax is None:
        fig, ax = plt.subplots(figsize=(pl_width, pl_height))
    else:
        fig = ax.figure

    ax.plot(x, y, label=label, **plot_kwargs)

    _apply_axis_style(ax, tick_dir, tick_length, tick_width, tick_pad,
                       tick_number, fontsize, show_yaxis)

    if xlims == 'default':
        if invert_x:
            ax.set_xlim(max(x), min(x))
    else:
        ax.set_xlim(xlims[0], xlims[1])

    nuclide = spectrum_dict.get("nuclide", "")
    ax.set_xlabel(NUC_label(nuclide, x_unit=axis), fontsize=fontsize)

    if label is not None:
        ax.legend(fontsize=fontsize * 0.85)

    safe_nuclide = re.sub(r"[^\w\-]", "", nuclide)
    _save_spectrum_figure(fig, fname, spectrum_dict.get("path"),
                           f"{safe_nuclide}_{channel}_{axis}_spectrum.png")

    return ax


def plot_comparison(simulated_dict, experimental_dict, ax=None, axis="ppm",
                     channel="real", normalised=True, invert_x=True,
                     sim_label="Simulated", exp_label="Experimental", offset=0.0,
                     show_yaxis=False,
                     tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                     tick_number=None, xlims='default', pl_width=3.375, pl_height=3.375,
                     font='Arial', fontsize=12, fname=True):
    """
    Overlay a simulated spectrum dict (CIF-predicted or BMS-predicted) and
    an experimental spectrum dict.

    offset : added to the simulated spectrum's y-values only, so the two
             traces can be separated vertically (e.g. offset=0.2 with
             normalised=True plots the simulated trace 0.2 above the
             experimental one).
    """
    plt.rcParams['font.family'] = font

    if ax is None:
        fig, ax = plt.subplots(figsize=(pl_width, pl_height))
    else:
        fig = ax.figure

    common = dict(axis=axis, channel=channel, normalised=normalised, invert_x=False,
                   show_yaxis=show_yaxis, tick_dir=tick_dir, tick_length=tick_length,
                   tick_width=tick_width, tick_pad=tick_pad, tick_number=tick_number,
                   xlims='default', pl_width=pl_width, pl_height=pl_height,
                   font=font, fontsize=fontsize, fname=False)

    plot_spectrum_dict(experimental_dict, ax=ax, offset=0.0, label=exp_label,
                        color="tab:gray", **common)
    plot_spectrum_dict(simulated_dict, ax=ax, offset=offset, label=sim_label,
                        color="tab:red", **common)

    x_key = "x_ppm" if axis == "ppm" else "x_Hz"
    if xlims == 'default':
        if invert_x:
            x_all = np.concatenate([simulated_dict[x_key], experimental_dict[x_key]])
            ax.set_xlim(x_all.max(), x_all.min())
    else:
        ax.set_xlim(xlims[0], xlims[1])

    nuclide = simulated_dict.get("nuclide") or experimental_dict.get("nuclide") or ""
    safe_nuclide = re.sub(r"[^\w\-]", "", nuclide)
    path = simulated_dict.get("path") or experimental_dict.get("path")
    _save_spectrum_figure(fig, fname, path, f"{safe_nuclide}_comparison_{channel}_{axis}.png")

    return ax


def plot_stacked_spectra(spectra, labels=None, offset_step=0.2, colors=None,
                          axis="ppm", channel="real", normalised=True, invert_x=True,
                          ax=None, show_yaxis=False,
                          tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                          tick_number=None, xlims='default', pl_width=3.375, pl_height=3.375,
                          font='Arial', fontsize=12, fname=True):
    """
    Stack any number of spectrum dicts (experimental, CIF-simulated,
    and/or BMS-simulated, freely mixed) vertically, offsetting spectrum i
    by i * offset_step. Handy for comparing a simulated spectrum against
    several experimental ones (or several simulated candidate models) at
    once.
    """
    plt.rcParams['font.family'] = font

    if ax is None:
        fig, ax = plt.subplots(figsize=(pl_width, pl_height))
    else:
        fig = ax.figure

    labels = labels or [None] * len(spectra)
    colors = colors or [None] * len(spectra)

    common = dict(axis=axis, channel=channel, normalised=normalised, invert_x=False,
                   show_yaxis=show_yaxis, tick_dir=tick_dir, tick_length=tick_length,
                   tick_width=tick_width, tick_pad=tick_pad, tick_number=tick_number,
                   xlims='default', pl_width=pl_width, pl_height=pl_height,
                   font=font, fontsize=fontsize, fname=False)

    for i, (spec, lab, col) in enumerate(zip(spectra, labels, colors)):
        plot_spectrum_dict(spec, ax=ax, offset=i * offset_step, label=lab,
                            color=col, **common)

    x_key = "x_ppm" if axis == "ppm" else "x_Hz"
    if xlims == 'default':
        if invert_x and spectra:
            x_all = np.concatenate([s[x_key] for s in spectra])
            ax.set_xlim(x_all.max(), x_all.min())
    else:
        ax.set_xlim(xlims[0], xlims[1])

    nuclide = spectra[0].get("nuclide", "") if spectra else ""
    safe_nuclide = re.sub(r"[^\w\-]", "", nuclide)
    path = spectra[0].get("path") if spectra else None
    _save_spectrum_figure(fig, fname, path, f"{safe_nuclide}_stack_{channel}_{axis}.png")

    return ax
