"""
Tools for importing, plotting, and extracting slices from 2D Bruker
NMR spectra (conventional homonuclear/heteronuclear 2D, and pseudo-2D).

Workflow
--------

    NMR2dimport        -> plot manually with matplotlib, or
                       -> NMR2Dplotter (contour + projections)
                       -> NMR2D_multi_plot (multiple contour panels,
                          side by side) / NMR2D_overlay_plot (2-3
                          spectra overlaid on one set of axes)
                       -> NMR2d_get_trace / NMR2D_get_slices (extract
                          rows and/or columns by index or shift)
                       -> NMR2D_plot_slices (plot extracted traces)
                       -> hand off to nmrtools.fitting for peak fitting
                       -> NMR2D_summary_plot (combined contour +
                          slices [+ integrals] figure)

Row/column convention
----------------------
``dataset["spectrum"]`` has shape ``(n_F1, n_F2)``: axis 0 is F1 (the
indirect dimension -- a chemical shift for conventional 2D, or the
experiment/slice number for pseudo-2D), axis 1 is F2 (the direct
dimension, always a chemical shift).

    - A **row** of the matrix fixes an F1 position and spans F2.
      This is what :func:`NMR2d_get_trace` returns for ``axis="f1"``.
    - A **column** of the matrix fixes an F2 position and spans F1.
      This is what :func:`NMR2d_get_trace` returns for ``axis="f2"``.

This matches Bruker/TopSpin's own "row"/"column" trace terminology,
where a row trace is selected by its F1 experiment number.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.transforms import blended_transform_factory
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .utils import find_gamma, NUC_label
from .plotting import (
    setup_plotting, get_stack_colormap, get_contour_cmap, downsample_2d,
    apply_tick_style, apply_spine_style, save_figure,
)


# ============================================================
# Import
# ============================================================

def NMR2dimport(datapath, procno=1, f1l=0, f1r=0, f2l=0, f2r=0, homonuclear=False):
    """
    Import a processed Bruker 2D NMR dataset.

    Loads processed Bruker TopSpin 2D spectra directly from the pdata
    directory and reconstructs the spectrum together with all relevant
    acquisition and processing metadata.

    Supports conventional 2D experiments, pseudo-2D datasets,
    homonuclear and heteronuclear spectra, and automatic Bruker XDIM
    block reconstruction.

    Parameters
    ----------
    datapath : str
        Path to the Bruker experiment directory.

    procno : int, optional
        Processed dataset number inside ``pdata/``. Default 1.

    f1l, f1r : float, optional
        F1 display limits, in ppm for conventional 2D data, or as
        1-based slice numbers for pseudo-2D data. Order does not
        matter. Both zero imports the full range.

    f2l, f2r : float, optional
        F2 display limits, in ppm. Both zero imports the full range.

    homonuclear : bool, optional
        If True, treat both dimensions as the same nucleus (F1 read
        from ``acqus`` rather than ``acqu2s``/``proc2s``).

    Returns
    -------
    dict
        ``spectrum``           : 2D intensity matrix, shape (n_F1, n_F2).
        ``x_ppm``, ``x_Hz``     : F2 (direct) axis.
        ``y_ppm``, ``y_Hz``     : F1 (indirect) axis. Only physically
            meaningful for conventional 2D; for pseudo-2D use ``y_index``.
        ``y_index``             : 1-based F1 slice/experiment number.
        ``NUC1``, ``NUC2``      : nucleus labels.
        ``SF1``, ``SF2``, ``OBS1``, ``OBS2``, ``O1``, ``O1_2``,
        ``SW1``, ``SW2``, ``SI1``, ``SI2`` : acquisition/processing
            parameters.
        ``pseudo_2d``           : True if the indirect dimension is a
            pseudo (non-frequency) dimension.
        ``procno``, ``datapath``, ``homonuclear``.

    Notes
    -----
    Data suitable for further analysis using :func:`NMR2d_get_trace`,
    :func:`NMR2D_get_slices`, :func:`NMR2Dplotter`, or custom
    processing.
    """

    real_spectrum_path = os.path.join(datapath, "pdata", str(procno), "2rr")
    procs = os.path.join(datapath, "pdata", str(procno), "procs")
    acqus = os.path.join(datapath, "acqus")
    proc2s = os.path.join(datapath, "pdata", str(procno), "proc2s")
    acqu2s = os.path.join(datapath, "acqu2s")

    # ----- acqus (F2 / direct dimension) -----

    O1str = '##$O1= '
    OBSstr = '##$BF1= '
    NUCstr = '##$NUC1= <'
    SWstr = "##$SW_h= "

    O1 = float("NaN")
    OBS = float("NaN")
    NUC = ""
    SW = float("NaN")

    with open(acqus, "rb") as inp:
        for line in inp:
            if SWstr in line.decode():
                linestr = line.decode()
                SW = float(linestr[len(SWstr):len(linestr) - 1])
            if O1str in line.decode():
                linestr = line.decode()
                O1 = float(linestr[len(O1str):len(linestr) - 1])
            if OBSstr in line.decode():
                linestr = line.decode()
                OBS = float(linestr[len(OBSstr):len(linestr) - 1])
            if NUCstr in line.decode():
                linestr = line.decode()
                NUC = linestr.split('= <')[1].split('>')[0]
            if ~np.isnan(SW) and ~np.isnan(O1) and ~np.isnan(OBS) and not len(NUC) == 0:
                break

    # ----- procs (F2 / direct dimension) -----

    SIstr = '##$SI= '
    SFstr = '##$SF= '
    NCstr = '##$NC_proc= '
    XDIM_F2str = '##$XDIM= '
    GRPDLYstr = '##$GRPDLY= '
    GRPDLY = 0.0

    SI = float("NaN")
    SF = float("NaN")
    NC_proc = float("NaN")
    XDIM_F2 = int(0)

    with open(procs, "rb") as inp:
        for line in inp:
            if SIstr in line.decode():
                linestr = line.decode()
                SI = float(linestr[len(SIstr):len(linestr) - 1])
            if SFstr in line.decode():
                linestr = line.decode()
                SF = float(linestr[len(SFstr):len(linestr) - 1])
            if NCstr in line.decode():
                linestr = line.decode()
                NC_proc = float(linestr[len(NCstr):len(linestr) - 1])
            if XDIM_F2str in line.decode():
                linestr = line.decode()
                XDIM_F2 = int(linestr[len(XDIM_F2str):len(linestr) - 1])
            if GRPDLYstr in line.decode():
                GRPDLY = float(line.decode().split('=')[1])
            if ~np.isnan(SI) and ~np.isnan(NC_proc) and ~np.isnan(SF) and XDIM_F2 != int(0):
                break

    # ----- acqu2s (F1 / indirect dimension) -----

    O1str_2 = '##$O1= '
    OBSstr_2 = '##$BF1= '
    NUCstr_2 = '##$NUC1= <'
    SWstr_2 = '##$SW_h= '

    O1_2 = float("NaN")
    OBS_2 = float("NaN")
    NUC_2 = ""
    SW_2 = float("NaN")

    with open(acqu2s, "rb") as inp:
        for line in inp:
            if SWstr_2 in line.decode():
                linestr = line.decode()
                SW_2 = float(linestr[len(SWstr_2):len(linestr) - 1])
            if O1str_2 in line.decode():
                linestr = line.decode()
                O1_2 = float(linestr[len(O1str_2):len(linestr) - 1])
            if OBSstr_2 in line.decode():
                linestr = line.decode()
                OBS_2 = float(linestr[len(OBSstr_2):len(linestr) - 1])
            if NUCstr_2 in line.decode():
                linestr = line.decode()
                NUC_2 = linestr.split('= <')[1].split('>')[0]
            if ~np.isnan(SW_2) and ~np.isnan(O1_2) and ~np.isnan(OBS_2) and not len(NUC_2) == 0:
                break

    # ----- proc2s (F1 / indirect dimension) -----

    SIstr_2 = '##$SI= '
    SFstr_2 = '##$SF= '
    NCstr_2 = '##$NC_proc= '
    XDIM_F1str = '##$XDIM= '

    XDIM_F1 = int(0)
    SI_2 = float("NaN")
    SF_2 = float("NaN")
    NC_proc_2 = float("NaN")

    with open(proc2s, "rb") as inp:
        for line in inp:
            if SIstr_2 in line.decode():
                linestr = line.decode()
                SI_2 = float(linestr[len(SIstr_2):len(linestr) - 1])
            if SFstr_2 in line.decode():
                linestr = line.decode()
                SF_2 = float(linestr[len(SFstr_2):len(linestr) - 1])
            if NCstr_2 in line.decode():
                linestr = line.decode()
                NC_proc_2 = float(linestr[len(NCstr_2):len(linestr) - 1])
            if XDIM_F1str in line.decode():
                linestr = line.decode()
                XDIM_F1 = int(linestr[len(XDIM_F1str):len(linestr) - 1])
            if ~np.isnan(SI_2) and ~np.isnan(NC_proc_2) and ~np.isnan(SF_2) and XDIM_F1 != int(0):
                break

    # ----- F2 (direct) axis -----

    SR = (SF - OBS) * 1000000
    true_centre = O1 - SR
    xmin = true_centre - SW / 2
    xmax = true_centre + SW / 2
    xAxHz = np.linspace(xmax, xmin, num=int(SI))
    xAxppm = xAxHz / SF

    # ----- F1 (indirect) axis -----

    if homonuclear:
        NUC_2 = NUC
        SR_2 = SR
        O1_2 = O1
    else:
        SR_2 = (SF_2 - OBS_2) * 1000000

    true_centre_2 = O1_2 - SR_2
    ymin = true_centre_2 - SW_2 / 2
    ymax = true_centre_2 + SW_2 / 2

    yAxHz = np.linspace(ymax, ymin, num=int(SI_2))
    yAxppm = yAxHz / SF_2

    # ----- experiment type -----

    pseudo_2d = (NUC_2 == 'off')

    # For pseudo-2D data the indirect dimension is not a frequency
    # dimension -- it is simply the experiment/slice number.
    # 1-based, since that is the natural way to refer to an increment.
    yAx_index = np.arange(1, int(SI_2) + 1)

    # ----- resolve display limits to array indices -----

    if f2l == 0 and f2r == 0:
        f2l = max(xAxppm)
        f2r = min(xAxppm)

    if pseudo_2d:

        # F1 limits are interpreted as 1-based slice numbers
        if f1l == 0 and f1r == 0:
            f1l = 1
            f1r = int(SI_2)

        slice_low = int(min(f1l, f1r))
        slice_high = int(max(f1l, f1r))

        ylow = max(slice_low - 1, 0)
        yhigh = min(slice_high, int(SI_2))

    else:

        if f1l == 0 and f1r == 0:
            f1l = max(yAxppm)
            f1r = min(yAxppm)

        ylow = np.where(yAxppm <= f1l)[0][0]
        yhigh = np.where(yAxppm <= f1r)[0][-1] + 1

        if ylow > yhigh:
            ylow, yhigh = yhigh, ylow

    xlow = np.where(xAxppm <= f2l)[0][0]
    xhigh = np.where(xAxppm <= f2r)[0][-1] + 1

    if xlow > xhigh:
        xlow, xhigh = xhigh, xlow

    xAxppm = xAxppm[xlow:xhigh]
    yAxppm = yAxppm[ylow:yhigh]
    yAx_index = yAx_index[ylow:yhigh]

    # ----- read and reshape the binary spectrum -----

    real_spectrum = np.fromfile(real_spectrum_path, dtype='<i4')

    if XDIM_F1 == 1:
        real_spectrum_2d = real_spectrum.reshape([int(SI_2), int(SI)])
    else:
        # Bruker stores 2D matrices as (XDIM_F1, XDIM_F2) submatrix
        # blocks, so reshaping when XDIM_F1 != 1 needs this procedure.
        column_matrix = real_spectrum
        submatrix_rows = int(SI_2 // XDIM_F1)
        submatrix_cols = int(SI // XDIM_F2)
        submatrix_number = submatrix_cols * submatrix_rows

        blocks = np.array(np.array_split(column_matrix, submatrix_number))
        blocks = np.reshape(blocks, (submatrix_rows, submatrix_cols, -1))
        real_spectrum_2d = np.vstack([
            np.hstack([np.reshape(c, (XDIM_F1, XDIM_F2)) for c in b])
            for b in blocks
        ])

    real_spectrum_2d = real_spectrum_2d[ylow:yhigh, xlow:xhigh]

    if GRPDLY != 0:
        nshift = int(np.floor(GRPDLY))
        real_spectrum_2d = np.roll(real_spectrum_2d, -nshift, axis=1)

    results = {
        "spectrum": real_spectrum_2d,

        "x_ppm": xAxppm,
        "y_ppm": yAxppm,
        "y_index": yAx_index,

        "x_Hz": xAxppm * SF,
        "y_Hz": yAxppm * SF_2,

        "NUC1": NUC,
        "NUC2": NUC_2,

        "SF1": SF,
        "SF2": SF_2,

        "OBS1": OBS,
        "OBS2": OBS_2,

        "O1": O1,
        "O1_2": O1_2,

        "SW1": SW,
        "SW2": SW_2,

        "SI1": int(SI),
        "SI2": int(SI_2),

        "pseudo_2d": pseudo_2d,

        "procno": procno,
        "datapath": datapath,
        "homonuclear": homonuclear,
    }

    return results


# ============================================================
# Contour plotting
# ============================================================

# ============================================================
# Shared contour preparation
# ============================================================

def _prepare_2d_contour(dataset, contour_fraction_x=0.25, contour_fraction_y=0.25,
                         ncontourpts_x=16384, ncontourpts_y=1024,
                         factor=1e-03, clevels=20):
    """
    Shared contour-preparation logic: resolve the F1 axis (ppm, or
    slice index for pseudo-2D), downsample for fast rendering, and
    compute log-spaced contour levels. Internal helper used by both
    :func:`NMR2Dplotter` and :func:`NMR2D_overlay_plot` -- kept as one
    source of truth rather than duplicated inline in each (see
    :mod:`nmrtools.plotting`'s own docstring for why that matters:
    exactly this kind of duplication is how the 1D/2D rcParams and
    colour-mapping logic silently diverged before).

    Returns
    -------
    contour_x, contour_y, contour_spectrum : ndarray
        Downsampled axes/data, ready for ``ax.contour()``.
    thresvec : list of float
        Log-spaced contour levels between ``factor * max`` and ``max``.
    y_axis : ndarray
        Full-resolution F1 axis (ppm, or slice index for pseudo-2D) --
        e.g. for axis limit/inversion logic, which should use the true
        range rather than the downsampled one.
    """

    real_spectrum_2d = dataset["spectrum"]
    xAxppm = dataset["x_ppm"]
    yAxppm = dataset["y_ppm"]
    yAx_index = dataset["y_index"]
    pseudo2d = dataset["pseudo_2d"]

    y_axis = yAx_index if pseudo2d else yAxppm

    if contour_fraction_x is not None or contour_fraction_y is not None:
        max_x = min(
            int(dataset["SI1"] * contour_fraction_x) if contour_fraction_x is not None else len(xAxppm),
            len(xAxppm), ncontourpts_x,
        )
        max_y = min(
            int(dataset["SI2"] * contour_fraction_y) if contour_fraction_y is not None else len(y_axis),
            len(y_axis), ncontourpts_y,
        )
        contour_spectrum, contour_x, contour_y = downsample_2d(
            real_spectrum_2d, xAxppm, y_axis, max_points_x=max_x, max_points_y=max_y,
        )
    else:
        contour_x, contour_y, contour_spectrum = xAxppm, y_axis, real_spectrum_2d

    Threshmin = factor * np.amax(real_spectrum_2d)
    Threshmax = np.amax(real_spectrum_2d)
    cc2 = np.linspace(1, clevels, clevels)
    thresvec = [Threshmin * ((Threshmax / Threshmin) ** (1 / clevels)) ** (1.25 * i) for i in cc2]

    return contour_x, contour_y, contour_spectrum, thresvec, y_axis


def NMR2Dplotter(dataset, procno=1, f1l=0.0, f1r=0.0, f2l=0.0, f2r=0.0, homonuclear=False,
                  contour_fraction_x=0.25, contour_fraction_y=0.25,
                  ncontourpts_x=16384, ncontourpts_y=1024,
                  pl_height=7.0, pl_width=7.0, factor=1e-03, clevels=20,
                  proj_scale=1.0, projection_mode="auto", show_projections=True,
                  projection_size=2.0, projection_pad=0.1,
                  field='400', xlabel=None, ylabel=None, xlims=None, ylims=None,
                  font='Arial', fontsize=None, output_format='',
                  colour_spectra=None, equal_tickspacing=False, tick_number=None,
                  tick_dir='in', tick_length=4, tick_width=1.0, tick_pad=4,
                  show_nQ_line=False, nQ=1, nQ_colours=None,
                  ax=None, fig=None, save_fig=False, fig_name='nmr_2d_spectrum',
                  save_format='png'):
    """
    Plot a 2D NMR spectrum as a contour plot with F1/F2 projections.

    Parameters
    ----------
    dataset : dict or str
        Output of :func:`NMR2dimport`, so the raw file is read once
        and can be reused for slice extraction and plotting. For
        convenience, a Bruker experiment path is also accepted, in
        which case it is imported internally using ``procno``, ``f1l``,
        ``f1r``, ``f2l``, ``f2r``, ``homonuclear`` (any ``dataset``
        already provided as a dict ignores these import-only
        arguments -- crop before calling, or re-import).

    contour_fraction_x, contour_fraction_y : float or None, optional
        Fraction of the *processed* number of points (``SI1``/``SI2``)
        used for contour rendering, for speed. Set to None to disable
        fraction-based capping (only ``ncontourpts_*`` then applies).

    ncontourpts_x, ncontourpts_y : int, optional
        Hard cap on the number of contour points along each axis.

    factor : float, optional
        Lowest contour level, as a fraction of the maximum intensity.

    clevels : int, optional
        Number of contour levels (log-spaced between ``factor`` and 1).

    projection_mode : {'auto', 'sum', 'max'}, optional
        ``'sum'`` for conventional projections; ``'max'`` (of
        ``abs()``) for pseudo-2D relaxation/diffusion series, where a
        sum projection would wash out slice-to-slice changes.
        ``'auto'`` picks ``'max'`` for pseudo-2D and ``'sum'``
        otherwise.

    show_projections : bool or {'f1', 'f2'}, optional
        ``True`` (default) shows both the F1 and F2 projection panels.
        ``False`` shows neither (useful when embedding a compact
        contour panel in a combined figure). ``'f1'`` or ``'f2'`` shows
        only that one projection -- e.g. when the other dimension's
        projection adds nothing (a pseudo-2D F1 "projection" is just
        the per-slice max/sum and rarely useful) or would crowd a
        multi-panel figure.

    projection_size : float, optional
        Size (in inches) of each projection panel, i.e. how far it
        extends above/left of the contour axes. Shrink this (and/or
        ``projection_pad``) when embedding a contour panel in a
        crowded multi-panel figure. Only used when ``show_projections``.

    projection_pad : float, optional
        Padding (in inches) between the contour axes and each
        projection panel.

    show_nQ_line, nQ : bool, float or list of float, optional
        If True, overlay one or more diagonal lines through the
        origin at gradient(s) ``nQ`` -- e.g. ``nQ=1`` for a single-
        quantum diagonal, ``nQ=[1, 2]`` to also show a double-quantum
        diagonal on a DQ-SQ correlation spectrum.

    nQ_colours : list of str, optional
        Colours for each diagonal in ``nQ``. Defaults to grey for all.

    tick_dir : {'in', 'out', 'inout'}, optional

    tick_length, tick_width, tick_pad : float, optional
        Major tick length/width/label-padding.

    tick_number : int, optional
        Approximate target number of ticks on each axis. If None
        (default), matplotlib's default locator is used.

    equal_tickspacing : bool, optional
        If True, force identical tick spacing on both axes (useful for
        square, e.g. homonuclear, spectra), chosen so there are
        approximately ``tick_number`` ticks (default 6 if
        ``tick_number`` is None) across the larger of the two axis
        ranges.

    xlabel, ylabel : str, optional
        Override the automatically-generated (nuclide-based) F2/F1
        axis labels.

    xlims, ylims : [left, right], optional
        Explicit F2 (x)/F1 (y) display limits. Takes precedence over
        any range implied by ``f2l``/``f2r`` and ``f1l``/``f1r``.

    font, fontsize : optional
        Typeface settings, see :func:`nmrtools.plotting.setup_plotting`.
        Only applied when ``ax`` is not given (see below).

    output_format : {'', 'manuscript', 'presentation'}, optional
        Predefined figure formatting, see
        :func:`nmrtools.plotting.setup_plotting`. Only applied when
        ``ax`` is not given.

    ax : matplotlib.axes.Axes, optional
        If given, plot the contour into this axis instead of creating
        a new figure -- e.g. for embedding in :func:`NMR2D_summary_plot`
        or a custom subplot grid. When ``ax`` is given, figure-level
        options (``pl_height``, ``pl_width``, ``font``, ``fontsize``,
        ``output_format``, ``save_fig``) are ignored, since they would
        affect the whole (shared) figure.

    save_fig : bool, optional
        If True, save the figure (see :func:`nmrtools.plotting.save_figure`).
        Default False. Only applies when ``ax`` is not given.

    fig_name : str, optional
        Output path/filename, with or without an extension. Default
        ``'nmr_2d_spectrum'``.

    save_format : {'png', 'pdf', 'svg'}, optional
        Output format used when ``fig_name`` has no extension already.
        Default ``'png'``.

    Returns
    -------
    fig, ax
    """

    if isinstance(dataset, str):
        dataset = NMR2dimport(dataset, procno=procno, f1l=f1l, f1r=f1r,
                               f2l=f2l, f2r=f2r, homonuclear=homonuclear)

    real_spectrum_2d = dataset["spectrum"]
    xAxppm = dataset["x_ppm"]
    yAxppm = dataset["y_ppm"]
    pseudo2d = dataset["pseudo_2d"]
    homonuclear = dataset.get("homonuclear", homonuclear)

    contour_x, contour_y, contour_spectrum, thresvec, y_axis = _prepare_2d_contour(
        dataset, contour_fraction_x=contour_fraction_x, contour_fraction_y=contour_fraction_y,
        ncontourpts_x=ncontourpts_x, ncontourpts_y=ncontourpts_y, factor=factor, clevels=clevels,
    )

    # ----- axis labels -----

    NUC = dataset["NUC1"]
    NUC_2 = dataset["NUC2"]
    nucgamma = find_gamma(NUC, field=field, SFO1=dataset["SF1"])
    NUC = nucgamma['Nuclide']

    labeltextx = NUC_label(NUC, x_unit='ppm')

    if homonuclear:
        labeltexty = NUC_label(NUC, x_unit='ppm')
    elif not pseudo2d:
        labeltexty = NUC_label(NUC_2, x_unit='ppm')
    else:
        labeltexty = 'Slice number'

    if ylabel is not None:
        labeltexty = ylabel
    if xlabel is not None:
        labeltextx = xlabel

    # ----- figure / axis -----

    standalone = ax is None

    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style['pl_width'], style['pl_height']
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    # ----- contour -----

    if colour_spectra:
        masked_spectrum = np.ma.masked_where(contour_spectrum <= 0, contour_spectrum)
        ax.contour(
            contour_x, contour_y, masked_spectrum,
            levels=thresvec, cmap=get_contour_cmap(colour_spectra), norm=LogNorm(),
        )
    else:
        ax.contour(contour_x, contour_y, contour_spectrum, colors='black', levels=thresvec)

    divider = make_axes_locatable(ax)

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
    elif f2l + f2r != 0:
        ax.set_xlim(min(f2l, f2r), max(f2l, f2r))

    if ylims is not None:
        ax.set_ylim(min(ylims), max(ylims))
    elif f1l + f1r != 0:
        ax.set_ylim(min(f1l, f1r), max(f1l, f1r))

    # F2 is conventionally displayed in decreasing ppm
    ax.invert_xaxis()

    if pseudo2d:
        # Show the actual cropped slice-index range, not a hardcoded
        # 0..N-1 (which is wrong whenever f1l/f1r crop away slice 1).
        if ylims is None:
            ax.set_ylim(y_axis.min() - 0.5, y_axis.max() + 0.5)
    else:
        # Conventional 2D NMR: decreasing ppm upward
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

    # ----- diagonal line(s), e.g. for multiple-quantum correlation spectra -----

    if show_nQ_line:

        nQ_list = nQ if isinstance(nQ, (list, tuple, np.ndarray)) else [nQ]
        nQ_list = [n for n in nQ_list if n != 0]

        if nQ_colours is None:
            nQ_colours = ['gray'] * len(nQ_list)
        elif len(nQ_colours) != len(nQ_list):
            raise ValueError("nQ_colours must have the same length as nQ.")

        xmin, xmax = ax.get_xlim()
        x_line = np.linspace(xmin, xmax, 500)

        for n, colour in zip(nQ_list, nQ_colours):
            ax.plot(x_line, n * x_line, linestyle='--', linewidth=1.5,
                     color=colour, alpha=0.8, zorder=10)

    # ----- projections -----

    show_f2_projection = show_projections in (True, 'f2')
    show_f1_projection = show_projections in (True, 'f1')

    if show_projections not in (True, False, 'f1', 'f2'):
        raise ValueError("show_projections must be True, False, 'f1', or 'f2'")

    if show_f2_projection or show_f1_projection:

        spec_float = np.asarray(real_spectrum_2d, dtype=np.float64)

        mode = projection_mode
        if mode == "auto":
            mode = "max" if pseudo2d else "sum"
        elif mode not in ("sum", "max"):
            raise ValueError("projection_mode must be 'auto', 'sum', or 'max'")

        proj_axes = []

        if show_f2_projection:

            ax_f2 = divider.append_axes("top", projection_size, pad=projection_pad, sharex=ax)
            ax_f2.tick_params(axis='x', which='both', bottom=False, top=False,
                               labelbottom=False, labeltop=False)
            ax_f2.tick_params(axis='y', which='both', left=False, right=False,
                               labelleft=False, labelright=False)

            # Maximum-magnitude projection ('max'): useful for pseudo-2D
            # series (saturation/inversion recovery, T1, T2, diffusion),
            # where a sum projection would wash out slice-to-slice changes.
            f2_projection = (np.sum(spec_float, axis=0) if mode == "sum"
                              else np.max(np.abs(spec_float), axis=0))

            ax_f2.plot(xAxppm, f2_projection + (0.02 * np.min(f2_projection)), 'k')
            ax_f2.set_ylim([np.min(f2_projection) * 0.9, np.max(f2_projection) * proj_scale])
            proj_axes.append(ax_f2)

        if show_f1_projection:

            ax_f1 = divider.append_axes("left", projection_size, pad=projection_pad, sharey=ax)
            ax_f1.tick_params(axis='x', which='both', bottom=False, top=False,
                               labelbottom=False, labeltop=False)
            ax_f1.tick_params(axis='y', which='both', left=False, right=False,
                               labelleft=False, labelright=False)

            f1_projection = (np.sum(spec_float, axis=1) if mode == "sum"
                              else np.max(np.abs(spec_float), axis=1))

            ax_f1.plot(-f1_projection + (0.02 * np.min(-f1_projection)), y_axis, 'k')
            ax_f1.set_xlim([np.min(-f1_projection) * proj_scale, np.max(-f1_projection) * 0.99])
            proj_axes.append(ax_f1)

        for proj_ax in proj_axes:
            for spine in ('top', 'right', 'bottom', 'left'):
                proj_ax.spines[spine].set_visible(False)

    if standalone:

        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.subplots_adjust(left=0.15, right=0.88, bottom=0.12, top=0.88)

        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax


# ============================================================
# Multi-panel / overlay contour plotting
# ============================================================

def NMR2D_multi_plot(datasets, layout="row", panel_kwargs=None, titles=None,
                      panel_width=5.0, panel_height=5.0, pl_width=None, pl_height=None,
                      font='Arial', fontsize=12, output_format='',
                      save_fig=False, fig_name='nmr_2d_multi', save_format='png',
                      **shared_kwargs):
    """
    Plot multiple 2D contour panels (:func:`NMR2Dplotter`) on one
    figure, arranged in a row, column, or grid -- one panel per
    dataset.

    Each panel is an independent :func:`NMR2Dplotter` call, so it
    accepts every argument that function does (contour levels,
    projections, nQ lines, tick styling, axis limits, ...); this
    function only handles the shared figure/gridspec and per-panel
    dataset/kwargs bookkeeping.

    To show different *regions* of the same spectrum side by side
    (rather than different datasets), pass the same dataset (or path)
    more than once in ``datasets`` and set a different ``xlims``/
    ``ylims`` per panel via ``panel_kwargs`` -- the dataset is only
    read/imported once per list entry, but since a dict is passed by
    reference, repeating an already-imported dataset dict costs
    nothing extra.

    Parameters
    ----------
    datasets : list of dict or str
        One dataset per panel, in the order panels are filled (row-
        major). Each entry is either the output of :func:`NMR2dimport`
        or a raw Bruker path (imported internally, exactly as
        :func:`NMR2Dplotter` accepts).

    layout : {'row', 'column', 'grid'}, optional
        Panel arrangement. ``'grid'`` chooses a near-square layout.

    panel_kwargs : list of dict, optional
        Per-panel keyword overrides forwarded to :func:`NMR2Dplotter`,
        one dict per panel (in the same order as ``datasets``; use
        ``{}`` for a panel with no overrides). Takes precedence over
        the same key in ``**shared_kwargs`` for that panel only.

    titles : list of str, optional
        Optional per-panel title (``ax.set_title``), one per panel.

    panel_width, panel_height : float, optional
        Size (in inches) of each panel, used to auto-size the overall
        figure when ``pl_width``/``pl_height`` are not given.

    pl_width, pl_height : float, optional
        Overall figure size in inches. If either is None (default),
        it is computed from ``panel_width``/``panel_height`` and the
        resolved grid shape.

    font, fontsize : optional
        Typeface settings, see :func:`nmrtools.plotting.setup_plotting`.
        Applied to the whole (shared) figure.

    output_format : {'', 'manuscript', 'presentation'}, optional
        Predefined figure formatting, see
        :func:`nmrtools.plotting.setup_plotting`.

    save_fig : bool, optional
        If True, save the figure (see :func:`nmrtools.plotting.save_figure`).
        Default False.

    fig_name : str, optional
        Output path/filename, with or without an extension. Default
        ``'nmr_2d_multi'``.

    save_format : {'png', 'pdf', 'svg'}, optional
        Output format used when ``fig_name`` has no extension already.
        Default ``'png'``.

    **shared_kwargs :
        Any other :func:`NMR2Dplotter` keyword argument (``factor``,
        ``clevels``, ``colour_spectra``, ``show_projections``,
        ``tick_number``, ``show_nQ_line``, ...), applied to every
        panel unless overridden for that panel in ``panel_kwargs``.
        Figure-level :func:`NMR2Dplotter` arguments that only make
        sense for a standalone figure (``ax``, ``fig``, ``save_fig``,
        ``fig_name``, ``save_format``, ``pl_width``, ``pl_height``,
        ``font``, ``fontsize``, ``output_format``) are not valid here
        or in ``panel_kwargs`` -- this function already owns those at
        the figure level; passing them raises ``ValueError``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        One per panel, in the same order as ``datasets``.
    """

    if layout not in ("row", "column", "grid"):
        raise ValueError("layout must be 'row', 'column', or 'grid'")

    n = len(datasets)
    if n == 0:
        raise ValueError("datasets must contain at least one dataset.")

    if panel_kwargs is not None and len(panel_kwargs) != n:
        raise ValueError("panel_kwargs must have the same length as datasets.")
    panel_kwargs = panel_kwargs if panel_kwargs is not None else [{}] * n

    if titles is not None and len(titles) != n:
        raise ValueError("titles must have the same length as datasets.")

    figure_level_keys = {
        "ax", "fig", "save_fig", "fig_name", "save_format",
        "pl_width", "pl_height", "font", "fontsize", "output_format",
    }
    bad = figure_level_keys & set(shared_kwargs)
    if bad:
        raise ValueError(
            f"Figure-level keys are not valid as shared per-panel kwargs: {sorted(bad)}; "
            f"use NMR2D_multi_plot's own same-named arguments instead."
        )
    for i, kw in enumerate(panel_kwargs):
        bad = figure_level_keys & set(kw)
        if bad:
            raise ValueError(
                f"panel_kwargs[{i}] contains figure-level keys not valid per panel: {sorted(bad)}"
            )

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
    for i, dataset in enumerate(datasets):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        kw = dict(shared_kwargs)
        kw.update(panel_kwargs[i])
        NMR2Dplotter(dataset, ax=ax, fig=fig, **kw)
        if titles is not None:
            ax.set_title(titles[i])
        axes.append(ax)

    fig.tight_layout()

    if save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, axes


def _check_overlay_compatible(datasets):
    """
    Raise ``ValueError`` unless every dataset in ``datasets`` shares
    the same F1/F2 nuclides and 2D experiment type (conventional vs
    pseudo-2D, homo- vs heteronuclear) -- i.e. their axes actually
    mean the same thing and can be overlaid directly. Internal helper
    for :func:`NMR2D_overlay_plot`.

    Deliberately does not compare ``SF1``/``SF2`` (transmitter
    frequency): ppm is field-independent by construction, so the same
    experiment recorded at two different fields is still a valid
    overlay -- it's the nuclide/experiment-type identity that would
    make an overlay meaningless (e.g. a heteronuclear HMQC against a
    homonuclear DQ/SQ spectrum), not the field it was recorded at.
    """

    ref = datasets[0]
    for i, ds in enumerate(datasets[1:], start=1):
        if ds.get("NUC1") != ref.get("NUC1") or ds.get("NUC2") != ref.get("NUC2"):
            raise ValueError(
                f"Cannot overlay datasets with different F1/F2 nuclides: "
                f"dataset 0 is {ref.get('NUC1')}/{ref.get('NUC2')}, "
                f"dataset {i} is {ds.get('NUC1')}/{ds.get('NUC2')}."
            )
        if bool(ds.get("pseudo_2d")) != bool(ref.get("pseudo_2d")):
            raise ValueError(
                f"Cannot overlay a pseudo-2D dataset with a conventional 2D "
                f"dataset (dataset {i} vs dataset 0)."
            )
        if bool(ds.get("homonuclear")) != bool(ref.get("homonuclear")):
            raise ValueError(
                f"Cannot overlay a homonuclear dataset with a heteronuclear "
                f"dataset (dataset {i} vs dataset 0)."
            )


def NMR2D_overlay_plot(datasets, procno=1, f1l=0.0, f1r=0.0, f2l=0.0, f2r=0.0, homonuclear=False,
                        contour_fraction_x=0.25, contour_fraction_y=0.25,
                        ncontourpts_x=16384, ncontourpts_y=1024,
                        pl_height=7.0, pl_width=7.0, factor=1e-03, clevels=20,
                        colours='diverging', labels=None, legend_kwargs=None,
                        x_offsets=None, y_offsets=None,
                        field='400', xlabel=None, ylabel=None, xlims=None, ylims=None,
                        font='Arial', fontsize=12, output_format='',
                        equal_tickspacing=False, tick_number=None,
                        tick_dir='in', tick_length=4, tick_width=1.0, tick_pad=4,
                        show_nQ_line=False, nQ=1, nQ_colours=None,
                        ax=None, fig=None, save_fig=False, fig_name='nmr_2d_overlay',
                        save_format='png'):
    """
    Overlay 2-3 2D NMR spectra as line contours on one set of axes,
    each in its own flat colour -- for directly comparing spectra
    (e.g. before/after a reaction, different conditions), rather than
    :func:`NMR2D_multi_plot`'s side-by-side panels.

    Deliberately limited to plain single-colour line contours (no
    colour-mapped/filled contours, no projections): with more than a
    couple of overlaid spectra a colour-mapped contour is no longer
    easy to read at a glance, which is the entire point of an overlay
    plot. Use :func:`NMR2Dplotter`/:func:`NMR2D_multi_plot` if you
    want any of that for a single spectrum or side-by-side panels.

    Before plotting, checks that every dataset shares the same F1/F2
    nuclides and experiment type (conventional vs pseudo-2D, homo- vs
    heteronuclear) -- see :func:`_check_overlay_compatible` -- and
    raises ``ValueError`` otherwise (e.g. refuses to overlay a
    heteronuclear HMQC on a homonuclear DQ/SQ spectrum, since F1 would
    not mean the same thing in both).

    Parameters
    ----------
    datasets : list of dict or str
        2 or 3 datasets to overlay (output of :func:`NMR2dimport`, or
        raw Bruker paths -- imported internally exactly as
        :func:`NMR2Dplotter` accepts). Each is imported/cropped using
        the *same* ``procno``/``f1l``/``f1r``/``f2l``/``f2r``/
        ``homonuclear`` -- import and crop beforehand if datasets need
        different import settings, and pass the resulting dicts here.

    colours : {'diverging'} or list of str, optional
        ``'diverging'`` (default): colours drawn from
        :func:`nmrtools.plotting.get_stack_colormap`'s four-stop
        brown/teal diverging map, spaced evenly across ``datasets``.
        Otherwise, an explicit list of matplotlib colours (e.g.
        ``['blue', 'red']``), one per dataset, in the same order.

    labels : list of str, optional
        Legend label per dataset, in the same order. If given, a
        legend is drawn. If None (default), no legend.

    legend_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ax.legend()`` (only
        used when ``labels`` is given).

    x_offsets, y_offsets : list of float, optional
        Per-dataset shift applied to that dataset's F2 (x)/F1 (y) axis
        before plotting, in the same units as the axis itself (ppm,
        or slice index for a pseudo-2D F1 axis) -- one value per
        dataset, same order as ``datasets``. Use this to nudge
        spectra apart when peaks from different datasets fall on top
        of each other and are hard to distinguish by colour alone.
        This shifts *display position only*, not the underlying data
        or its reported shifts -- ``xlims``/``ylims`` and any
        downstream slice/fit extraction are unaffected. Default None
        (no offset, all datasets plotted at their true shifts).

    factor : float, optional
        Lowest contour level, as a fraction of each dataset's own
        maximum intensity (applied independently per dataset, so
        differently-scaled spectra overlay sensibly).

    clevels : int, optional
        Number of contour levels per dataset (log-spaced between
        ``factor`` and 1).

    field, xlabel, ylabel, xlims, ylims, equal_tickspacing, tick_number,
    tick_dir, tick_length, tick_width, tick_pad, show_nQ_line, nQ,
    nQ_colours : optional
        Same meaning as the corresponding :func:`NMR2Dplotter`
        arguments -- axis labels are generated from the (shared)
        nuclides of ``datasets[0]``.

    ax : matplotlib.axes.Axes, optional
        If given, plot into this axis instead of creating a new
        figure. When given, figure-level options (``pl_height``,
        ``pl_width``, ``font``, ``fontsize``, ``output_format``,
        ``save_fig``) are ignored.

    save_fig : bool, optional
        If True, save the figure. Default False. Only applies when
        ``ax`` is not given.

    fig_name : str, optional
        Output path/filename, with or without an extension. Default
        ``'nmr_2d_overlay'``.

    save_format : {'png', 'pdf', 'svg'}, optional
        Output format used when ``fig_name`` has no extension already.
        Default ``'png'``.

    Returns
    -------
    fig, ax
    """

    if not (2 <= len(datasets) <= 3):
        raise ValueError(
            f"NMR2D_overlay_plot accepts 2 or 3 datasets, got {len(datasets)}."
        )

    datasets = [
        NMR2dimport(ds, procno=procno, f1l=f1l, f1r=f1r, f2l=f2l, f2r=f2r,
                    homonuclear=homonuclear)
        if isinstance(ds, str) else ds
        for ds in datasets
    ]

    _check_overlay_compatible(datasets)

    n = len(datasets)

    if colours == 'diverging':
        cmap = get_stack_colormap('diverging')
        resolved_colours = [cmap(i / max(n - 1, 1)) for i in range(n)]
    elif isinstance(colours, (list, tuple)):
        if len(colours) != n:
            raise ValueError("colours must have the same length as datasets.")
        resolved_colours = list(colours)
    else:
        raise ValueError("colours must be 'diverging' or a list of colours, one per dataset.")

    if labels is not None and len(labels) != n:
        raise ValueError("labels must have the same length as datasets.")

    if x_offsets is None:
        x_offsets = [0.0] * n
    elif len(x_offsets) != n:
        raise ValueError("x_offsets must have the same length as datasets.")

    if y_offsets is None:
        y_offsets = [0.0] * n
    elif len(y_offsets) != n:
        raise ValueError("y_offsets must have the same length as datasets.")

    ref = datasets[0]
    pseudo2d = ref["pseudo_2d"]
    homonuclear = ref.get("homonuclear", homonuclear)

    # ----- axis labels (shared nuclides, checked above) -----

    NUC = ref["NUC1"]
    NUC_2 = ref["NUC2"]
    nucgamma = find_gamma(NUC, field=field, SFO1=ref["SF1"])
    NUC = nucgamma['Nuclide']

    labeltextx = NUC_label(NUC, x_unit='ppm')

    if homonuclear:
        labeltexty = NUC_label(NUC, x_unit='ppm')
    elif not pseudo2d:
        labeltexty = NUC_label(NUC_2, x_unit='ppm')
    else:
        labeltexty = 'Slice number'

    if ylabel is not None:
        labeltexty = ylabel
    if xlabel is not None:
        labeltextx = xlabel

    # ----- figure / axis -----

    standalone = ax is None

    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style['pl_width'], style['pl_height']
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    # ----- contours, one flat colour per dataset -----

    for ds, colour, label, dx, dy in zip(
        datasets, resolved_colours, labels if labels is not None else [None] * n,
        x_offsets, y_offsets,
    ):
        contour_x, contour_y, contour_spectrum, thresvec, _ = _prepare_2d_contour(
            ds, contour_fraction_x=contour_fraction_x, contour_fraction_y=contour_fraction_y,
            ncontourpts_x=ncontourpts_x, ncontourpts_y=ncontourpts_y, factor=factor, clevels=clevels,
        )
        ax.contour(contour_x + dx, contour_y + dy, contour_spectrum, colors=[colour], levels=thresvec)
        if label is not None:
            # A contour set has no direct legend handle; use an
            # invisible proxy line in the same colour, matplotlib's
            # documented approach for legending a QuadContourSet.
            ax.plot([], [], color=colour, label=label)

    divider = make_axes_locatable(ax)

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
    elif f2l + f2r != 0:
        ax.set_xlim(min(f2l, f2r), max(f2l, f2r))

    if ylims is not None:
        ax.set_ylim(min(ylims), max(ylims))
    elif f1l + f1r != 0:
        ax.set_ylim(min(f1l, f1r), max(f1l, f1r))

    ax.invert_xaxis()

    if pseudo2d:
        if ylims is None:
            y_axis = ref["y_index"]
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
        nQ_list = [n_ for n_ in nQ_list if n_ != 0]

        if nQ_colours is None:
            nQ_colours = ['gray'] * len(nQ_list)
        elif len(nQ_colours) != len(nQ_list):
            raise ValueError("nQ_colours must have the same length as nQ.")

        xmin, xmax = ax.get_xlim()
        x_line = np.linspace(xmin, xmax, 500)

        for nq, colour in zip(nQ_list, nQ_colours):
            ax.plot(x_line, nq * x_line, linestyle='--', linewidth=1.5,
                     color=colour, alpha=0.8, zorder=10)

    if labels is not None:
        ax.legend(**(legend_kwargs or {}))

    if standalone:

        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.subplots_adjust(left=0.15, right=0.88, bottom=0.12, top=0.88)

        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax


# ============================================================
# Slice extraction
# ============================================================

def NMR2d_get_trace(dataset, axis="f1", index=None, shift=None, width=1,
                     combine="sum", window=None):
    """
    Extract a single 1D trace (row or column) from an imported 2D
    dataset.

    Parameters
    ----------
    dataset : dict
        Output of :func:`NMR2dimport`.

    axis : {'f1', 'f2'}, optional
        ``'f1'``: fix a position along F1 (a *row* of ``dataset["spectrum"]``).
            For conventional 2D, F1 is a chemical shift
            (``dataset["y_ppm"]``); for pseudo-2D it is the
            experiment/slice number (``dataset["y_index"]``). The
            returned trace runs along F2 (``dataset["x_ppm"]``).
        ``'f2'``: fix a position along F2 (``dataset["x_ppm"]``, a
            *column* of the matrix). The returned trace runs along F1.

    index : int, optional
        1-based position along the fixed axis (TopSpin convention),
        counted within the currently imported/cropped range.

    shift : float, optional
        Chemical shift (ppm) used to locate the nearest position along
        the fixed axis. Not valid for the pseudo-2D F1 axis (use
        ``index`` instead), since slice number has no chemical-shift
        meaning.

    width : int, optional
        Number of adjacent points, centred on the selected position,
        to combine for extra S/N (e.g. summing 3 adjacent rows).
        Default 1 (no averaging).

    combine : {'sum', 'mean'}, optional
        How to combine points across ``width``.

    window : tuple(float, float), optional
        Restrict the returned trace to this ppm range (only applied
        when the trace's own axis is a chemical shift).

    Returns
    -------
    dict
        ``x``, ``x_Hz`` : trace axis (``x_Hz`` is None where not
            meaningful, e.g. pseudo-2D column traces).
        ``x_is_ppm``    : whether ``x`` is a chemical shift (False for
            a pseudo-2D column trace, where ``x`` is the slice index).
        ``y``           : intensity.
        ``axis``, ``slice_type`` ('row'/'column').
        ``index``       : 1-based position along the fixed axis.
        ``shift``       : chemical shift (ppm) at that position, or
            (for the pseudo-2D F1 axis) the slice index as a float.
        ``shift_unit``  : 'ppm' or 'index', clarifying the above.
        ``width``, ``nuclide``, ``SFO1``, ``pseudo_2d``.
    """

    if axis not in ("f1", "f2"):
        raise ValueError("axis must be 'f1' or 'f2'")

    if index is None and shift is None:
        raise ValueError("Either index or shift must be provided.")

    spectrum = dataset["spectrum"]
    pseudo2d = dataset["pseudo_2d"]

    if axis == "f1":
        fixed_axis_ppm = dataset["y_ppm"]
        n_fixed = spectrum.shape[0]
    else:
        fixed_axis_ppm = dataset["x_ppm"]
        n_fixed = spectrum.shape[1]

    # ----- resolve position -> zero-based array index -----

    if index is not None:
        pos = int(index) - 1
        if pos < 0 or pos >= n_fixed:
            raise ValueError(f"index {index} out of range (1..{n_fixed})")
    else:
        if axis == "f1" and pseudo2d:
            raise ValueError(
                "shift cannot be used to select an F1 position for "
                "pseudo-2D data; use index instead."
            )
        pos = int(np.argmin(np.abs(fixed_axis_ppm - shift)))

    if axis == "f1" and pseudo2d:
        actual_shift = float(dataset["y_index"][pos])
        shift_unit = "index"
    else:
        actual_shift = float(fixed_axis_ppm[pos])
        shift_unit = "ppm"

    actual_index = pos + 1

    # ----- width averaging -----

    half = width // 2
    lo = max(0, pos - half)
    hi = min(n_fixed, lo + width)
    lo = max(0, hi - width)  # re-clamp if hi was capped near the edge

    if axis == "f1":

        block = spectrum[lo:hi, :]
        y = block.sum(axis=0) if combine == "sum" else block.mean(axis=0)

        x = dataset["x_ppm"]
        x_Hz = dataset.get("x_Hz")
        x_is_ppm = True
        slice_type = "row"
        nuclide = dataset["NUC1"]
        SFO1 = dataset["SF1"]

    else:

        block = spectrum[:, lo:hi]
        y = block.sum(axis=1) if combine == "sum" else block.mean(axis=1)

        if pseudo2d:
            x = dataset["y_index"].astype(float)
            x_Hz = None
            x_is_ppm = False
        else:
            x = dataset["y_ppm"]
            x_Hz = dataset.get("y_Hz")
            x_is_ppm = True

        slice_type = "column"
        nuclide = dataset["NUC1"] if dataset.get("homonuclear") else dataset["NUC2"]
        SFO1 = dataset["SF1"] if dataset.get("homonuclear") else dataset["SF2"]

    x = np.asarray(x)
    y = np.asarray(y)

    if window is not None and x_is_ppm:
        lo_w, hi_w = sorted(window)
        mask = (x >= lo_w) & (x <= hi_w)
        x = x[mask]
        y = y[mask]
        if x_Hz is not None:
            x_Hz = np.asarray(x_Hz)[mask]

    return {
        "x": x,
        "x_Hz": x_Hz,
        "x_is_ppm": x_is_ppm,
        "y": y,
        "axis": axis,
        "slice_type": slice_type,
        "index": actual_index,
        "shift": actual_shift,
        "shift_unit": shift_unit,
        "width": width,
        "nuclide": nuclide,
        "SFO1": SFO1,
        "pseudo_2d": pseudo2d,
        "datapath": dataset.get("datapath"),
    }


def NMR2D_get_slices(dataset, rows=None, row_shifts=None, row_width=1,
                      columns=None, column_shifts=None, column_width=1,
                      combine="sum", window=None, normalise=False):
    """
    Extract multiple row and/or column traces from a 2D dataset.

    A thin batch wrapper around :func:`NMR2d_get_trace` -- kept as one
    source of truth for the underlying array slicing, rather than
    re-implementing it (this replaces two separate, divergent,
    unfinished extraction paths from the previous version of this
    module).

    Parameters
    ----------
    dataset : dict
        Output of :func:`NMR2dimport`.

    rows, row_shifts : list, optional
        1-based F1 indices / F1 chemical shifts (ppm) of rows to
        extract. May be combined (duplicates are extracted once).

    row_width : int, optional
        Points to combine around each selected row (see
        :func:`NMR2d_get_trace`).

    columns, column_shifts : list, optional
        1-based F2 indices / F2 chemical shifts (ppm) of columns to
        extract.

    column_width : int, optional
        Points to combine around each selected column.

    combine : {'sum', 'mean'}, optional
        How to combine points within a given width.

    window : tuple(float, float), optional
        Restrict every extracted trace to this ppm range.

    normalise : bool, optional
        If True, scale every extracted trace to its own maximum
        absolute intensity.

    Returns
    -------
    dict with keys ``"rows"`` and ``"columns"``, each a list of trace
    dicts as returned by :func:`NMR2d_get_trace`, sorted by position
    along the fixed axis.
    """

    slices = {"rows": [], "columns": []}

    def _collect(axis, indices, shifts, width):

        out = []
        seen_index = set()

        for idx in (indices or []):
            if idx in seen_index:
                continue
            seen_index.add(idx)
            out.append(NMR2d_get_trace(dataset, axis=axis, index=idx, width=width,
                                        combine=combine, window=window))

        for sh in (shifts or []):
            trace = NMR2d_get_trace(dataset, axis=axis, shift=sh, width=width,
                                     combine=combine, window=window)
            if trace["index"] in seen_index:
                continue
            seen_index.add(trace["index"])
            out.append(trace)

        out.sort(key=lambda t: t["index"])
        return out

    if rows is not None or row_shifts is not None:
        slices["rows"] = _collect("f1", rows, row_shifts, row_width)

    if columns is not None or column_shifts is not None:
        slices["columns"] = _collect("f2", columns, column_shifts, column_width)

    if normalise:
        for group in slices.values():
            for tr in group:
                m = np.max(np.abs(tr["y"]))
                if m > 0:
                    tr["y"] = tr["y"] / m

    return slices


def prepare_trace_for_fit(trace, normalise=True):
    """
    Convert an extracted 2D trace (see :func:`NMR2d_get_trace`) into
    the dictionary format expected by ``nmrtools.fitting``'s 1D
    fitting routines.

    Parameters
    ----------
    trace : dict
        Output of :func:`NMR2d_get_trace`.

    normalise : bool, optional
        If True, normalise the spectrum to its maximum absolute
        intensity.

    Returns
    -------
    dict with keys ``x_ppm``, ``real_spectrum``,
    ``normalised_real_spectrum``, ``nuclide``, ``SFO1`` -- directly
    compatible with :func:`nmrtools.fitting.fit_spectrum` and related
    functions.
    """

    results = {
        "x_ppm": np.asarray(trace["x"], dtype=float),
        "real_spectrum": np.asarray(trace["y"], dtype=float),
        "nuclide": trace.get("nuclide", "unknown"),
        "SFO1": trace.get("SFO1", 0),
    }

    if normalise:
        scale = np.max(np.abs(trace["y"]))
        results["normalised_real_spectrum"] = trace["y"] / scale if scale != 0 else trace["y"]
    else:
        results["normalised_real_spectrum"] = trace["y"]

    return results


# ============================================================
# Slice plotting
# ============================================================

def _hyphen_to_minus(text):
    """
    Replace a leading ASCII hyphen-minus with a proper Unicode minus
    sign (U+2212), so negative shift values in labels render with a
    typographic minus rather than a hyphen.
    """
    if text.startswith("-"):
        return "\u2212" + text[1:]
    return text


def _default_slice_label(tr):
    """
    Default per-slice label: ``"F1 = <shift> ppm"`` / ``"F2 = <shift>
    ppm"`` for a conventional trace, or ``"F1 = <index>"`` (unitless)
    for a pseudo-2D F1 slice, where ``<shift>``/``<index>`` is the
    fixed-axis position the trace was extracted at (see
    :func:`NMR2d_get_trace`'s ``shift``/``shift_unit``).
    """
    axis_name = tr.get("axis", "f1").upper()
    shift = tr.get("shift", 0.0)
    if tr.get("shift_unit") == "index":
        value = f"{shift:.0f}"
        return f"{axis_name} = {_hyphen_to_minus(value)}"
    value = f"{shift:.2f}"
    return f"{axis_name} = {_hyphen_to_minus(value)} ppm"


def NMR2D_plot_slices(slices, mode="rows", fits=None, show_fits=False,
                       show_fit_components=False, colour_spectra='black',
                       offset_scale=0.1, normalise=False, xlims=None, xlabel=None,
                       fit_colour='red', component_colour='gray',
                       show_nQ_markers=False, nQ=1, nQ_colours=None,
                       nQ_marker='*', nQ_markersize=10, nQ_marker_offset=0.05,
                       show_labels=False, labels=None, label_style='legend',
                       label_side='right', label_pad=0.05, label_fontsize=None,
                       legend_kwargs=None,
                       tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                       tick_number=None,
                       ax=None, fig=None, pl_width=7.0, pl_height=5.0,
                       font='Arial', fontsize=12, output_format='', field='',
                       save_fig=False, fig_name='nmr_2d_slices', save_format='png'):
    """
    Stack-plot a set of extracted 2D slices.

    Parameters
    ----------
    slices : dict or list
        Either the ``{"rows": [...], "columns": [...]}`` dict returned
        by :func:`NMR2D_get_slices`, or a flat list of trace dicts
        (only valid for ``mode='rows'`` or ``mode='columns'``, not
        ``mode='both'``).

    mode : {'rows', 'columns', 'both'}, optional

        ``'rows'``:
            Plot rows on one axis.

        ``'columns'``:
            Plot columns on one axis.

        ``'both'``:
            Plot rows and columns on separate axes. Requires ``slices``
            to be the dict form.

            When ``ax`` is None, two independent standalone figures are
            created and returned (see Returns below).

            When ``ax`` is a dict of the form
            ``{"rows": rows_axis, "columns": columns_axis}``, the two
            groups are plotted into those axes on a shared figure --
            this is what :func:`NMR2D_summary_plot` uses internally.

    fits : list, optional
        Optional list of fit-result dicts, one per trace (same order
        as the selected slices), used to overlay a fitted curve and/or
        components. Each entry is duck-typed: it only needs ``"x"``
        and ``"y_fit"`` (and optionally ``"components"``, a list of
        per-peak arrays) -- this module does not import
        ``nmrtools.fitting``, so any dict with that shape works,
        including the output of :func:`nmrtools.fitting.NMR2D_fit_slices`.
        For ``mode='both'``, the same ``fits`` list is forwarded to
        both the rows and columns calls.

    show_fits, show_fit_components : bool, optional
        Overlay the fitted curve / individual peak components.

    colour_spectra : optional
        See :func:`nmrtools.plotting.get_stack_colormap`.

    offset_scale : float, optional
        Vertical spacing between traces, as a fraction of each trace's
        own peak-to-peak range.

    normalise : bool, optional
        If True, normalise each trace to its own maximum absolute
        intensity before plotting (independent of any normalisation
        already applied in ``slices``).

    show_nQ_markers, nQ : bool, float or list of float, optional
        If True, mark where each trace's fixed-shift line crosses the
        nQ diagonal(s) ``F1 = n * F2`` (see :func:`NMR2Dplotter`'s
        ``show_nQ_line``) -- e.g. for a row trace (fixed F1), the
        crossing sits at ``F2 = F1 / n``; for a column trace (fixed
        F2), at ``F1 = n * F2``. The trace's own y-value is
        interpolated at that position and the marker drawn just above
        it. Only meaningful for traces with ``shift_unit == 'ppm'``
        (conventional 2D); silently skipped for pseudo-2D/index-based
        traces, and skipped per-trace if the crossing falls outside
        the trace's own data range. ``nQ`` accepts the same single
        value or list of values as :func:`NMR2Dplotter`. For
        ``mode='both'``, applied independently to both the rows and
        columns plots (each against its own fixed shift).

    nQ_colours : list of str, optional
        Colours for each diagonal in ``nQ``, in the same order.
        Defaults to black for all.

    nQ_marker, nQ_markersize : optional
        Marker style/size for the crossing points. Default a black
        star.

    nQ_marker_offset : float, optional
        Vertical offset of the marker above the interpolated trace
        height, as a fraction of that trace's own peak-to-peak span
        (so the marker sits clear of the line rather than on top of
        it). Default 0.05.

    show_labels : bool, optional
        If True, label each slice. Default False.

    labels : list of str, optional
        Custom label text, one per trace, in the same order as the
        selected slices (``slices[mode]``, or ``slices`` itself if a
        flat list was given). Must match the number of traces if
        given. If None (default) and ``show_labels`` is True, each
        label is generated automatically as ``"F1 = <shift> ppm"`` /
        ``"F2 = <shift> ppm"`` (or ``"F1 = <index>"``, unitless, for a
        pseudo-2D F1 slice) from that trace's own fixed-axis position
        -- see :func:`NMR2d_get_trace`. For ``mode='both'``, pass a
        dict ``{"rows": [...], "columns": [...]}`` instead, so each
        group gets its own labels; a flat list is only valid for
        ``mode='rows'``/``'columns'``.

    label_style : {'legend', 'text'}, optional
        ``'legend'`` (default) adds a standard matplotlib legend,
        colour-keyed to each trace. ``'text'`` instead writes each
        label directly onto the axes, positioned just above that
        trace's own baseline (near its noise floor) at the left- or
        right-hand edge of the plot (see ``label_side``) -- avoids a
        legend box competing for space in a busy stack plot, at the
        cost of only working well when traces don't overlap much
        near that edge.

    label_side : {'left', 'right'}, optional
        For ``label_style='text'`` only: which edge of the axes each
        label sits at (in axes-fraction terms, so this is unaffected
        by chemical-shift axis inversion). Default ``'right'``.

    label_pad : float, optional
        For ``label_style='text'`` only: vertical offset of the label
        above its trace's own baseline, as a fraction of that trace's
        peak-to-peak span. Default 0.05.

    label_fontsize : float, optional
        For ``label_style='text'`` only: font size for the labels.
        Defaults to the current ``axes.labelsize`` rcParam (i.e. the
        shared ``fontsize`` from :func:`nmrtools.plotting.setup_plotting`).

    legend_kwargs : dict, optional
        For ``label_style='legend'`` only: extra keyword arguments
        forwarded to ``ax.legend()`` (e.g. ``{"loc": "upper left",
        "fontsize": 8, "frameon": False}``).

    xlims : [left, right], optional
        Explicit x-axis limits; otherwise set from the data. For
        ``mode='both'``, applied to both the rows and columns axes
        (rarely useful there, since rows and columns axes differ in
        meaning -- F2 vs F1 -- so this is mainly for ``mode='rows'``
        or ``mode='columns'``; see :func:`NMR2D_summary_plot` for
        axis-appropriate limit matching against a contour panel).

    xlabel : str, optional
        Override the automatically-generated (nuclide-based) x-axis
        label.

    tick_dir : {'in', 'out', 'inout'}, optional

    tick_length, tick_width, tick_pad : float, optional
        Major tick length/width/label-padding (x-axis only -- the
        y-axis carries no ticks, since traces are offset-stacked).

    tick_number : int, optional
        Approximate target number of x-axis ticks. If None, matplotlib's
        default locator is used.

    ax : matplotlib.axes.Axes or dict, optional
        Plot into an existing axis instead of creating a new figure --
        e.g. for :func:`NMR2D_summary_plot`. For ``mode='rows'`` or
        ``mode='columns'``, an ordinary Axes. For ``mode='both'``, a
        dict ``{"rows": rows_axis, "columns": columns_axis}`` (see
        ``mode`` above). When ``ax`` is given, figure-level options
        (``pl_height``, ``pl_width``, ``font``, ``fontsize``,
        ``output_format``, ``save_fig``) are ignored.

    fig : optional
        Shared figure to plot into, paired with ``ax``. Only relevant
        when ``ax`` is given.

    pl_width, pl_height : float, optional
        Figure size in inches. Only applied when ``ax`` is not given.

    font, fontsize : optional
        Typeface settings, see :func:`nmrtools.plotting.setup_plotting`.
        Only applied when ``ax`` is not given.

    output_format : {'', 'manuscript', 'presentation'}, optional
        Predefined figure formatting, see
        :func:`nmrtools.plotting.setup_plotting`. Only applied when
        ``ax`` is not given.

    save_fig : bool, optional
        If True, save the figure (see :func:`nmrtools.plotting.save_figure`).
        Default False. Only applies when ``ax`` is not given. For
        ``mode='both'`` with no ``ax``, both figures are saved, as
        ``f"{fig_name}_rows"`` and ``f"{fig_name}_columns"``.

    fig_name : str, optional
        Output path/filename, with or without an extension. Default
        ``'nmr_2d_slices'``.

    save_format : {'png', 'pdf', 'svg'}, optional
        Output format used when ``fig_name`` has no extension already.
        Default ``'png'``.

    Returns
    -------
    fig, ax
        For ``mode='rows'`` or ``mode='columns'``, the usual ``fig, ax``.

        For ``mode='both'`` with no ``ax`` supplied::

            figs = {"rows": fig_rows, "columns": fig_columns}
            axes = {"rows": ax_rows, "columns": ax_columns}

        For ``mode='both'`` with ``ax`` supplied, the shared figure
        and the same ``axes`` dict passed in.
    """

    if mode not in ("rows", "columns", "both"):
        raise ValueError("mode must be 'rows', 'columns' or 'both'")

    if mode == "both":

        if not isinstance(slices, dict):
            raise ValueError(
                "mode='both' requires slices={'rows': [...], 'columns': [...]}"
            )

        if len(slices.get("rows", [])) == 0:
            raise ValueError("No row slices to plot.")

        if len(slices.get("columns", [])) == 0:
            raise ValueError("No column slices to plot.")

        if labels is not None and not isinstance(labels, dict):
            raise ValueError(
                "For mode='both', labels must be a dict {'rows': [...], "
                "'columns': [...]} (a flat list is only valid for "
                "mode='rows'/'columns')."
            )
        labels = labels or {}

        common_kwargs = dict(
            fits=fits, show_fits=show_fits, show_fit_components=show_fit_components,
            colour_spectra=colour_spectra, offset_scale=offset_scale, normalise=normalise,
            xlims=xlims, xlabel=xlabel, fit_colour=fit_colour, component_colour=component_colour,
            show_nQ_markers=show_nQ_markers, nQ=nQ, nQ_colours=nQ_colours,
            nQ_marker=nQ_marker, nQ_markersize=nQ_markersize, nQ_marker_offset=nQ_marker_offset,
            show_labels=show_labels, label_style=label_style, label_side=label_side,
            label_pad=label_pad, label_fontsize=label_fontsize, legend_kwargs=legend_kwargs,
            tick_dir=tick_dir, tick_length=tick_length, tick_width=tick_width,
            tick_pad=tick_pad, tick_number=tick_number, field=field,
        )

        if ax is None:

            fig_rows, ax_rows = NMR2D_plot_slices(
                slices, mode="rows", pl_width=pl_width, pl_height=pl_height,
                font=font, fontsize=fontsize, output_format=output_format,
                save_fig=save_fig, fig_name=f"{fig_name}_rows", save_format=save_format,
                labels=labels.get("rows"), **common_kwargs,
            )
            fig_columns, ax_columns = NMR2D_plot_slices(
                slices, mode="columns", pl_width=pl_width, pl_height=pl_height,
                font=font, fontsize=fontsize, output_format=output_format,
                save_fig=save_fig, fig_name=f"{fig_name}_columns", save_format=save_format,
                labels=labels.get("columns"), **common_kwargs,
            )

            return (
                {"rows": fig_rows, "columns": fig_columns},
                {"rows": ax_rows, "columns": ax_columns},
            )

        if not isinstance(ax, dict) or "rows" not in ax or "columns" not in ax:
            raise ValueError(
                "For mode='both', ax must be a dict with 'rows' and 'columns' axes."
            )

        shared_fig = fig if fig is not None else ax["rows"].figure

        NMR2D_plot_slices(slices, mode="rows", ax=ax["rows"], fig=shared_fig,
                           labels=labels.get("rows"), **common_kwargs)
        NMR2D_plot_slices(slices, mode="columns", ax=ax["columns"], fig=shared_fig,
                           labels=labels.get("columns"), **common_kwargs)

        return shared_fig, ax

    if isinstance(slices, dict):
        traces = slices[mode]
    else:
        traces = slices

    if len(traces) == 0:
        raise ValueError(f"No slices to plot (mode='{mode}').")

    if labels is not None and len(labels) != len(traces):
        raise ValueError("labels must have the same length as the number of traces.")

    if label_style not in ("legend", "text"):
        raise ValueError("label_style must be 'legend' or 'text'")

    if label_side not in ("left", "right"):
        raise ValueError("label_side must be 'left' or 'right'")

    def _nQ_crossing_x(tr, n):
        """
        x-position (on ``tr``'s own axis) where its fixed-shift line
        crosses the nQ diagonal F1 = n * F2, or None if not
        applicable (pseudo-2D trace, or n == 0).
        """
        if n == 0 or tr.get("shift_unit") != "ppm":
            return None
        fixed_shift = tr["shift"]
        if tr.get("slice_type") == "row":
            # fixed F1, trace x-axis is F2 -> F2 = F1 / n
            return fixed_shift / n
        else:
            # fixed F2, trace x-axis is F1 -> F1 = n * F2
            return fixed_shift * n

    standalone = ax is None

    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style['pl_width'], style['pl_height']
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    cmap = get_stack_colormap(colour_spectra)
    n = len(traces)
    norm_ = plt.Normalize(vmin=0, vmax=max(n, 1))
    scalarMap = cmx.ScalarMappable(norm=norm_, cmap=cmap)

    if show_nQ_markers:
        nQ_list = nQ if isinstance(nQ, (list, tuple, np.ndarray)) else [nQ]
        nQ_list = [nq for nq in nQ_list if nq != 0]
        if nQ_colours is None:
            nQ_marker_colours = ['black'] * len(nQ_list)
        elif len(nQ_colours) != len(nQ_list):
            raise ValueError("nQ_colours must have the same length as nQ.")
        else:
            nQ_marker_colours = nQ_colours

    yBase = 0.0
    xmin = xmax = None
    x_is_ppm = True

    for i, tr in enumerate(traces):

        x = np.asarray(tr["x"])
        y = np.asarray(tr["y"])
        x_is_ppm = tr.get("x_is_ppm", True)

        if normalise:
            m = np.max(np.abs(y))
            y = y / m if m != 0 else y

        colour = scalarMap.to_rgba(i)
        text = (labels[i] if labels is not None else _default_slice_label(tr)) if show_labels else None

        ax.plot(x, y + yBase, color=colour, linewidth=1.2,
                label=text if (show_labels and label_style == 'legend') else None)

        if (show_fits or show_fit_components) and fits is not None and i < len(fits):

            fit = fits[i]

            if show_fits and "y_fit" in fit:
                fx = np.asarray(fit.get("x", x))
                ax.plot(fx, np.asarray(fit["y_fit"]) + yBase, color=fit_colour,
                        linewidth=1.0, linestyle='--')

            if show_fit_components and fit.get("components") is not None:
                fx = np.asarray(fit.get("x", x))
                for comp in fit["components"]:
                    ax.plot(fx, np.asarray(comp) + yBase, color=component_colour,
                            linewidth=0.7, alpha=0.7)

        span = (np.max(y) - np.min(y)) if len(y) else 1.0
        if span == 0:
            span = 1.0

        if show_labels and label_style == 'text' and text is not None:
            trans = blended_transform_factory(ax.transAxes, ax.transData)
            x_frac = 0.02 if label_side == 'left' else 0.98
            ax.text(x_frac, yBase + label_pad * span, text, transform=trans,
                    ha=label_side, va='bottom', color=colour,
                    fontsize=label_fontsize, clip_on=False)

        if show_nQ_markers and len(x):
            order = np.argsort(x)
            x_sorted, y_sorted = x[order], y[order]
            for nq, mcolour in zip(nQ_list, nQ_marker_colours):
                x_cross = _nQ_crossing_x(tr, nq)
                if x_cross is None:
                    continue
                if not (x_sorted[0] <= x_cross <= x_sorted[-1]):
                    continue
                y_cross = np.interp(x_cross, x_sorted, y_sorted)
                ax.plot(x_cross, y_cross + yBase + span * nQ_marker_offset,
                        marker=nQ_marker, markersize=nQ_markersize,
                        color=mcolour, linestyle='None', zorder=12, clip_on=False)

        yBase += span * (1 + offset_scale)

        if len(x):
            lo, hi = float(np.min(x)), float(np.max(x))
            xmin = lo if xmin is None else min(xmin, lo)
            xmax = hi if xmax is None else max(xmax, hi)

    if xlims is not None:
        ax.set_xlim(min(xlims), max(xlims))
        if x_is_ppm:
            ax.invert_xaxis()
    elif xmin is not None:
        ax.set_xlim(xmin, xmax)
        if x_is_ppm:
            ax.invert_xaxis()

    ax.set_yticks([])
    apply_spine_style(ax, top=False, right=False, bottom=True, left=False)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis='x')
    ax.tick_params(axis='x', labelsize=fontsize)

    if show_labels and label_style == 'legend':
        ax.legend(**(legend_kwargs or {}))

    label_nuclide = traces[0].get("nuclide")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    elif x_is_ppm and label_nuclide:
        nucgamma = find_gamma(label_nuclide, field=field, SFO1=traces[0].get("SFO1", 0))
        ax.set_xlabel(NUC_label(nucgamma['Nuclide'], x_unit='ppm'), fontsize=fontsize)
    elif not x_is_ppm:
        ax.set_xlabel("Slice number", fontsize=fontsize)

    ax.set_title(f"Extracted {mode}")

    if standalone:
        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.tight_layout()

        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax


# ============================================================
# Combined layouts
# ============================================================

def NMR2D_summary_plot(dataset, slices=None, fits=None, integrals=None,
                        slices_mode="rows", layout="row", pl_width=None, pl_height=None,
                        show_contour=True, show_slices=None, show_integrals=None,
                        show_projections=None, projection_mode="auto", colour_spectra=None,
                        factor=None, clevels=None,
                        projection_size=None, projection_pad=None,
                        xlims=None, ylims=None, f1l=0.0, f1r=0.0, f2l=0.0, f2r=0.0,
                        show_nQ_line=False, nQ=1, nQ_colours=None, show_nQ_markers=False,
                        nQ_marker='*', nQ_markersize=10, nQ_marker_offset=0.05,
                        slices_on_contour_mode=None, slices_on_contour_linestyle='--',
                        slices_on_contour_alpha=0.6,
                        show_labels=False, labels=None, label_style='legend',
                        label_side='right', label_pad=0.05, label_fontsize=None,
                        legend_kwargs=None,
                        font='Arial', fontsize=12, output_format='',
                        save_fig=False, fig_name='nmr_2d_summary', save_format='png',
                        contour_kwargs=None, slices_kwargs=None, integrals_kwargs=None):
    """
    Combined figure: 2D contour, extracted slices, and (optionally)
    their integrals, arranged in a row, column, or grid.

    This keeps ``two_d`` self-contained (no dependency on
    ``nmrtools.fitting``): the integrals panel is generic, plotting
    whatever curves you pass it. To combine a contour with *fitted*
    slice integrals, run :func:`nmrtools.fitting.NMR2D_fit_slices`
    first and pass its integral output through ``integrals`` (see
    that function's docstring for the expected format), or use
    :func:`nmrtools.fitting.NMR2D_workflow`, which does this for you
    end-to-end.

    Parameters
    ----------
    dataset : dict or str
        Output of :func:`NMR2dimport`, or a Bruker path (see
        :func:`NMR2Dplotter`).

    slices : dict or list, optional
        Output of :func:`NMR2D_get_slices`, or a flat list of traces
        (only valid for ``slices_mode='rows'`` or ``'columns'``, not
        ``'both'``). Required for the slices panel(s).

    fits : list, optional
        Fit-result dicts to overlay on the slices panel(s) (see
        :func:`NMR2D_plot_slices`). For ``slices_mode='both'``, the
        same list is forwarded to both the rows and columns panels.

    integrals : list of dict, optional
        Curves for the integrals panel. Each dict:
        ``{"x": ..., "y": ..., "yerr": <optional>, "label": <optional>}``.
        Required for the integrals panel.

    slices_mode : {'rows', 'columns', 'both'}, optional
        Which group of ``slices`` to plot, when ``slices`` is the dict
        form. ``'both'`` adds two separate slices panels (one for
        rows, one for columns) instead of one -- see ``layout``.

    layout : {'row', 'column', 'grid'}, optional
        Arrangement of the requested panels.

    pl_width, pl_height : float, optional
        Overall figure size in inches. If either is None (default),
        it is chosen automatically based on the layout and number of
        panels (5.5 in wide x 5.0 in tall per panel).

    show_contour, show_slices, show_integrals : bool, optional
        Explicitly include/exclude a panel. ``show_slices`` and
        ``show_integrals`` default to whether ``slices``/``integrals``
        were provided.

    show_projections : bool or {'f1', 'f2'}, optional
        Show the F1/F2 projections on the contour panel (see
        :func:`NMR2Dplotter`'s ``show_projections`` -- ``'f1'``/``'f2'``
        shows only that one). Defaults to True only when the contour
        is the sole panel (projections can crowd a multi-panel
        figure); set explicitly to override either way. A value here
        is only used if ``contour_kwargs`` doesn't already set
        ``"show_projections"``.

    projection_mode : {'auto', 'sum', 'max'}, optional
        Forwarded to :func:`NMR2Dplotter` (only relevant when
        projections are shown).

    colour_spectra : optional
        Colour scheme shared by the contour and the slice traces --
        forwarded to :func:`NMR2Dplotter`'s ``colour_spectra`` (None
        for black contours, or a named colormap) *and* to
        :func:`NMR2D_plot_slices`'s ``colour_spectra`` (None leaves
        that function's own 'black' default), so the two panels use
        matching colours. Also used to colour the
        ``slices_on_contour_mode`` position lines. Only used if
        ``contour_kwargs``/``slices_kwargs`` doesn't already set
        ``"colour_spectra"``.

    factor : float, optional
        Lowest contour level, as a fraction of the maximum intensity.
        Forwarded to :func:`NMR2Dplotter` (see that function's
        ``factor``). Only used if ``contour_kwargs`` doesn't already
        set ``"factor"``; if omitted entirely, :func:`NMR2Dplotter`'s
        own default is used.

    clevels : int, optional
        Number of contour levels. Forwarded to :func:`NMR2Dplotter`
        (see that function's ``clevels``). Only used if
        ``contour_kwargs`` doesn't already set ``"clevels"``.

    projection_size : float, optional
        Size (in inches) of the F1/F2 projection panels on the contour
        axes. Forwarded to :func:`NMR2Dplotter`. Shrinking this is
        often useful in a multi-panel summary figure, where the full
        standalone default can crowd the other panels. Only used if
        ``contour_kwargs`` doesn't already set ``"projection_size"``.

    projection_pad : float, optional
        Padding (in inches) between the contour axes and each
        projection panel. Forwarded to :func:`NMR2Dplotter`. Only used
        if ``contour_kwargs`` doesn't already set ``"projection_pad"``.

    xlims, ylims : [left, right], optional
        F2 (x)/F1 (y) display limits for the contour panel -- same
        convention as :func:`NMR2Dplotter`, and take precedence over
        ``f2l``/``f2r``/``f1l``/``f1r`` below. These are also used to
        set the *slices* panel's own x-axis limits, matched to the
        contour axis a row/column trace actually runs along: rows run
        along F2, so a rows slices panel gets ``xlims`` (F2); columns
        run along F1, so a columns slices panel gets ``ylims`` (F1).
        For ``slices_mode='both'``, both slices panels are matched
        this way simultaneously. Only applied to a panel if that
        panel's own kwargs dict doesn't already set its ``xlims``.

    f1l, f1r, f2l, f2r : float, optional
        Fallback F1/F2 display limits, in the same convention as
        :func:`NMR2dimport`/:func:`NMR2Dplotter` (both zero means "no
        fallback limit"). Only used where the corresponding
        ``xlims``/``ylims`` above is None.

    show_nQ_line, nQ : bool, float or list of float, optional
        Single toggle that draws the nQ diagonal(s) ``F1 = n * F2`` on
        the contour panel (:func:`NMR2Dplotter`'s ``show_nQ_line``)
        *and* marks, on the slices panel(s), where each extracted row/
        column crosses that same diagonal
        (:func:`NMR2D_plot_slices`'s ``show_nQ_markers``) -- so one
        setting keeps every panel showing the same diagonal(s).
        Only applied to a panel if that panel's own kwargs dict
        (``contour_kwargs``/``slices_kwargs``) doesn't already set it.

    nQ_colours : list of str, optional
        Colours for each diagonal in ``nQ``, forwarded to every panel
        the same way. Defaults to grey on the contour and black for
        the slice-panel crossing markers (each function's own
        default), if not given.

    slices_on_contour_mode : {None, 'rows', 'columns', 'both'}, optional
        If given, draw a horizontal line at each extracted row's F1
        position and/or a vertical line at each extracted column's F2
        position directly on the contour panel, coloured to match that
        trace's colour in the slices panel(s) (via ``colour_spectra``).
        ``'rows'``/``'columns'`` draws only that group;
        ``'both'`` draws both groups regardless of ``slices_mode``
        (independent settings -- e.g. you can stack only rows in the
        side panel while still marking both rows and columns on the
        contour). Requires ``slices`` to be the dict form and a
        ``"contour"`` panel to be shown; silently does nothing
        otherwise. Default None (off).

    slices_on_contour_linestyle, slices_on_contour_alpha : optional
        Line style / transparency for the ``slices_on_contour_mode``
        position lines.

    contour_kwargs, slices_kwargs, integrals_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`NMR2Dplotter`,
        :func:`NMR2D_plot_slices`, and the integrals panel
        respectively. Take precedence over ``show_projections``,
        ``projection_mode``, ``colour_spectra``, ``factor``,
        ``clevels``, ``projection_size``, ``projection_pad``,
        ``xlims``/``ylims``/``f1l``/``f1r``/``f2l``/``f2r``,
        ``show_nQ_line``, ``nQ``, and ``nQ_colours`` above if both are
        given. For ``slices_mode='both'``, the same ``slices_kwargs``
        dict is used as the base for both the rows and columns calls
        (their ``xlims`` are then set independently as described
        above). Use these for any other :func:`NMR2Dplotter`/
        :func:`NMR2D_plot_slices` control -- overall figure size is
        set once via ``pl_width``/``pl_height`` above, since the
        figure is shared across panels -- but ``proj_scale``,
        ``tick_number``, etc. all work the same way.

    font, fontsize : optional
        Typeface settings, see :func:`nmrtools.plotting.setup_plotting`.
        Applied to the whole (shared) figure via ``setup_plotting``,
        and also forwarded as the default ``font``/``fontsize`` for
        the contour and slices panels (an explicit ``font``/
        ``fontsize`` already present in ``contour_kwargs``/
        ``slices_kwargs`` takes precedence). The integrals panel has
        no ``font``/``fontsize`` of its own and always follows the
        shared rcParams.

    output_format : {'', 'manuscript', 'presentation'}, optional
        Predefined figure formatting, see
        :func:`nmrtools.plotting.setup_plotting`.

    save_fig : bool, optional
        If True, save the figure (see :func:`nmrtools.plotting.save_figure`).
        Default False.

    fig_name : str, optional
        Output path/filename, with or without an extension. Default
        ``'nmr_2d_summary'``.

    save_format : {'png', 'pdf', 'svg'}, optional
        Output format used when ``fig_name`` has no extension already.
        Default ``'png'``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict
        Mapping of panel name to its Axes, for any further
        customisation: ``'contour'``, ``'integrals'``, and either
        ``'slices'`` (``slices_mode='rows'``/``'columns'``) or both
        ``'slices_rows'`` and ``'slices_columns'``
        (``slices_mode='both'``).
    """

    setup_plotting(font=font, fontsize=fontsize, output_format=output_format)

    contour_kwargs = dict(contour_kwargs or {})
    slices_kwargs = dict(slices_kwargs or {})
    integrals_kwargs = dict(integrals_kwargs or {})

    # Explicitly forward the resolved typeface settings to each panel
    # rather than relying solely on the (figure-wide) rcParams that
    # setup_plotting() configures above -- individual panel kwargs
    # can still override these via contour_kwargs/slices_kwargs.
    # (_plot_integral_curves has no font/fontsize parameters of its
    # own -- it always follows the shared rcParams.)
    if fontsize is not None:
        contour_kwargs.setdefault("fontsize", fontsize)
        slices_kwargs.setdefault("fontsize", fontsize)
    if font is not None:
        contour_kwargs.setdefault("font", font)
        slices_kwargs.setdefault("font", font)

    if show_slices is None:
        show_slices = slices is not None
    if show_integrals is None:
        show_integrals = integrals is not None

    panels = []
    if show_contour:
        panels.append("contour")
    if show_slices:
        if slices is None:
            raise ValueError("show_slices=True requires 'slices' to be provided.")
        if slices_mode == "both":
            panels.extend(["slices_rows", "slices_columns"])
        else:
            panels.append("slices")
    if show_integrals:
        if integrals is None:
            raise ValueError("show_integrals=True requires 'integrals' to be provided.")
        panels.append("integrals")

    if not panels:
        raise ValueError("At least one of show_contour/show_slices/show_integrals must be True.")

    if layout == "row":
        nrows, ncols = 1, len(panels)
    elif layout == "column":
        nrows, ncols = len(panels), 1
    elif layout == "grid":
        ncols = int(np.ceil(np.sqrt(len(panels))))
        nrows = int(np.ceil(len(panels) / ncols))
    else:
        raise ValueError("layout must be 'row', 'column', or 'grid'")

    if pl_width is None:
        pl_width = 5.5 * ncols
    if pl_height is None:
        pl_height = 5.0 * nrows

    fig = plt.figure(figsize=(pl_width, pl_height))
    gs = fig.add_gridspec(nrows, ncols)

    axes = {}
    for i, name in enumerate(panels):
        r, c = divmod(i, ncols)
        axes[name] = fig.add_subplot(gs[r, c])

    if "contour" in axes:
        if show_projections is None:
            show_projections = (len(panels) == 1)
        contour_kwargs.setdefault("show_projections", show_projections)
        contour_kwargs.setdefault("projection_mode", projection_mode)
        if colour_spectra is not None:
            contour_kwargs.setdefault("colour_spectra", colour_spectra)
        if factor is not None:
            contour_kwargs.setdefault("factor", factor)
        if clevels is not None:
            contour_kwargs.setdefault("clevels", clevels)
        if projection_size is not None:
            contour_kwargs.setdefault("projection_size", projection_size)
        if projection_pad is not None:
            contour_kwargs.setdefault("projection_pad", projection_pad)
        contour_kwargs.setdefault("show_nQ_line", show_nQ_line)
        contour_kwargs.setdefault("nQ", nQ)
        if nQ_colours is not None:
            contour_kwargs.setdefault("nQ_colours", nQ_colours)
        if xlims is not None:
            contour_kwargs.setdefault("xlims", [min(xlims), max(xlims)])
        elif f2l + f2r != 0:
            contour_kwargs.setdefault("xlims", [min(f2l, f2r), max(f2l, f2r)])
        if ylims is not None:
            contour_kwargs.setdefault("ylims", [min(ylims), max(ylims)])
        elif f1l + f1r != 0:
            contour_kwargs.setdefault("ylims", [min(f1l, f1r), max(f1l, f1r)])
        NMR2Dplotter(dataset, ax=axes["contour"], fig=fig, **contour_kwargs)

    def _slices_common_kwargs(base_kwargs):
        kw = dict(base_kwargs)
        if colour_spectra is not None:
            kw.setdefault("colour_spectra", colour_spectra)
        kw.setdefault("show_nQ_markers", show_nQ_markers)
        kw.setdefault("nQ", nQ)
        if nQ_colours is not None:
            kw.setdefault("nQ_colours", nQ_colours)
        #show_nQ_markers=False, nQ_marker='*', nQ_markersize=10, nQ_marker_offset=0.05,
        kw.setdefault("nQ_marker", nQ_marker)
        kw.setdefault("nQ_markersize", nQ_markersize)
        kw.setdefault("nQ_marker_offset", nQ_marker_offset)
        #show_labels=False, labels=None, label_style='legend',
        #label_side='right', label_pad=0.05, label_fontsize=None,
        #legend_kwargs=None,
        kw.setdefault("show_labels", show_labels)
        kw.setdefault("labels", labels)
        kw.setdefault("label_style", label_style)
        kw.setdefault("label_side", label_side)
        kw.setdefault("label_pad", label_pad)
        kw.setdefault("label_fontsize", label_fontsize)
        kw.setdefault("legend_kwargs", legend_kwargs)
        
        return kw

    if slices_mode == "both":

        if "slices_rows" in axes:
            rows_kwargs = _slices_common_kwargs(slices_kwargs)
            if xlims is not None:
                rows_kwargs.setdefault("xlims", [min(xlims), max(xlims)])
            elif f2l + f2r != 0:
                rows_kwargs.setdefault("xlims", [min(f2l, f2r), max(f2l, f2r)])
            NMR2D_plot_slices(slices, mode="rows", fits=fits,
                               ax=axes["slices_rows"], fig=fig, **rows_kwargs)

        if "slices_columns" in axes:
            columns_kwargs = _slices_common_kwargs(slices_kwargs)
            if ylims is not None:
                columns_kwargs.setdefault("xlims", [min(ylims), max(ylims)])
            elif f1l + f1r != 0:
                columns_kwargs.setdefault("xlims", [min(f1l, f1r), max(f1l, f1r)])
            NMR2D_plot_slices(slices, mode="columns", fits=fits,
                               ax=axes["slices_columns"], fig=fig, **columns_kwargs)

    elif "slices" in axes:

        panel_kwargs = _slices_common_kwargs(slices_kwargs)
        if slices_mode == "rows":
            if xlims is not None:
                panel_kwargs.setdefault("xlims", [min(xlims), max(xlims)])
            elif f2l + f2r != 0:
                panel_kwargs.setdefault("xlims", [min(f2l, f2r), max(f2l, f2r)])
        elif slices_mode == "columns":
            if ylims is not None:
                panel_kwargs.setdefault("xlims", [min(ylims), max(ylims)])
            elif f1l + f1r != 0:
                panel_kwargs.setdefault("xlims", [min(f1l, f1r), max(f1l, f1r)])
        NMR2D_plot_slices(slices, mode=slices_mode, fits=fits,
                           ax=axes["slices"], fig=fig, **panel_kwargs)

    if "contour" in axes and slices_on_contour_mode is not None:
        _draw_slice_position_lines(axes["contour"], slices, slices_on_contour_mode,
                                    colour_spectra, slices_on_contour_linestyle,
                                    slices_on_contour_alpha)

    if "integrals" in axes:
        _plot_integral_curves(integrals, ax=axes["integrals"], **integrals_kwargs)

    fig.tight_layout()

    if save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, axes


def _draw_slice_position_lines(contour_ax, slices, mode, colour_spectra,
                                linestyle='--', alpha=0.6):
    """
    Draw horizontal/vertical lines on ``contour_ax`` marking the F1/F2
    positions of extracted row/column slices, coloured to match
    :func:`NMR2D_plot_slices`'s own trace colouring. Internal helper
    for :func:`NMR2D_summary_plot`'s ``slices_on_contour_mode``.

    Silently does nothing if ``slices`` isn't the ``{"rows": [...],
    "columns": [...]}`` dict form (e.g. a flat list, which carries no
    row/column grouping to draw from).
    """

    if not isinstance(slices, dict) or mode not in ("rows", "columns", "both"):
        return

    # Match NMR2D_plot_slices' own default (colour_spectra=None passed
    # through to get_stack_colormap would fall back to matplotlib's
    # rcParams colormap, not black) so the lines match the traces.
    resolved_colour_spectra = colour_spectra if colour_spectra is not None else 'black'

    def _draw(group, line_fn):
        traces = slices.get(group, [])
        n = len(traces)
        if n == 0:
            return
        cmap = get_stack_colormap(resolved_colour_spectra)
        norm_ = plt.Normalize(vmin=0, vmax=max(n, 1))
        scalarMap = cmx.ScalarMappable(norm=norm_, cmap=cmap)
        for i, tr in enumerate(traces):
            line_fn(tr['shift'], color=scalarMap.to_rgba(i),
                    linestyle=linestyle, alpha=alpha)

    if mode in ("rows", "both"):
        _draw("rows", contour_ax.axhline)
    if mode in ("columns", "both"):
        _draw("columns", contour_ax.axvline)


def _plot_integral_curves(curves, ax, xlabel="Slice position", ylabel="Integral",
                           title="Integrals", xlims=None, ylims=None,
                           show_top_spine=False, show_right_spine=False,
                           tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                           tick_number=None):
    """
    Plot a list of generic ``{"x", "y", "yerr", "label"}`` curves onto
    ``ax``. Internal helper for :func:`NMR2D_summary_plot`.
    """

    for curve in curves:

        x = np.asarray(curve["x"])
        y = np.asarray(curve["y"])
        label = curve.get("label")
        yerr = curve.get("yerr")

        if yerr is not None:
            ax.errorbar(x, y, yerr=np.asarray(yerr), marker='o', linestyle='-',
                        capsize=3, label=label)
        else:
            ax.plot(x, y, marker='o', linestyle='-', label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    apply_spine_style(ax, top=show_top_spine, right=show_right_spine)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis='both')

    if any(c.get("label") for c in curves):
        ax.legend()

    return ax
