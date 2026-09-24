"""
Tools for importing and plotting 1D Bruker NMR spectra, including
Variable Offset Cumulative Spectroscopy (VOCS) processing.

Workflow
--------

    NMR1dimport   -> plot manually with matplotlib, or
                  -> NMR1dplot (optionally with VOCS processing)
                  -> hand off to nmrtools.fitting for peak fitting
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d

from .utils import find_gamma, NUC_label, read_bruker_metadata, read_bruker_fid
from .plotting import (
    setup_plotting, get_stack_colormap, get_contour_cmap, apply_tick_style,
    apply_spine_style, save_figure,
)
from .arithmetic import combine_spectra


# ============================================================
# Import
# ============================================================

def NMR1dimport(data_path, procno=1, mass=1, f1p=0, f2p=0, VOCS=False,
                 pulse="P1", heteronuclear_2d_slice=False, slice_dimension=2):
    """
    Import a processed 1D Bruker TopSpin NMR spectrum.

    Reads processed spectral data directly from a Bruker experiment
    directory and returns the frequency axes, real and imaginary
    spectrum, normalised spectrum, and key acquisition parameters.

    The function can optionally apply a Variable Offset Cumulative
    Spectroscopy (VOCS) correction using the specified pulse length.

    Parameters
    ----------
    data_path : str
        Path to the top-level Bruker experiment directory. This should
        contain the ``acqus`` and ``pdata`` directories.

    procno : int, optional
        Processing number inside the ``pdata`` directory. Default is 1.

    mass : float, optional
        Sample mass used to scale the spectrum. The processed spectrum
        is divided by this value to allow comparison between samples
        with different masses. Default is 1.

    f1p, f2p : float, optional
        Limits of the spectral region to import in ppm. The order does
        not matter. If both are zero, the complete processed spectrum
        is imported.

    VOCS : bool, optional
        If True, apply the VOCS correction to the real and imaginary
        spectra using the pulse specified by ``pulse``. Default False.

    pulse : str, optional
        Acquisition parameter containing the pulse length used to
        calculate the VOCS excitation profile, e.g. ``"P1"`` or
        ``"P2"``. Case-insensitive. Default is ``"P1"``.

    heteronuclear_2d_slice : bool, optional
        If True, use ``acqu2s`` or ``acqu3s`` rather than ``acqus`` to
        determine the spectral axis. Useful when importing individual
        slices from heteronuclear 2D experiments. Default False.

    slice_dimension : {2, 3}, optional
        Indirect dimension containing the spectral information when
        ``heteronuclear_2d_slice`` is True. Default is 2.

    Returns
    -------
    results : dict
        ``x_ppm``, ``x_Hz``                        : frequency axes.
        ``real_spectrum``, ``imag_spectrum``        : intensity, scaled
            by sample mass and number of scans.
        ``normalised_real_spectrum``,
        ``normalised_imag_spectrum``                : intensity scaled
            to [0, 1].
        ``nuclide``                                 : e.g. "1H", "27Al".
        ``BF1``, ``SFO1``                           : spectrometer
            frequencies in MHz.
        ``path``                                    : original path.

    Raises
    ------
    ValueError
        If ``slice_dimension`` is not 2 or 3 when importing a
        heteronuclear 2D slice, or if the requested VOCS pulse
        parameter is not found in the Bruker ``acqus`` parameters.

    Examples
    --------
    >>> spec = NMR1dimport(r"C:\\data\\exp1")
    >>> spec = NMR1dimport(r"C:\\data\\exp1", f1p=10, f2p=-2)
    >>> spec = NMR1dimport(r"C:\\data\\exp1", VOCS=True, pulse="P2")
    """

    real_spectrum_path = os.path.join(data_path, "pdata", str(procno), "1r")
    imag_spectrum_path = os.path.join(data_path, "pdata", str(procno), "1i")
    procs = os.path.join(data_path, "pdata", str(procno), "procs")
    acqus = os.path.join(data_path, "acqus")

    if heteronuclear_2d_slice:
        if slice_dimension == 2:
            acqus_axis = os.path.join(data_path, "acqu2s")
        elif slice_dimension == 3:
            acqus_axis = os.path.join(data_path, "acqu3s")
        else:
            raise ValueError("slice_dimension must be 2 or 3")
    else:
        acqus_axis = acqus

    real_spectrum = np.fromfile(real_spectrum_path, dtype='int32', count=-1)
    imag_spectrum = np.fromfile(imag_spectrum_path, dtype='int32', count=-1)

    meta = read_bruker_metadata(acqus_axis)

    NUC = meta["NUC"]
    OBS = meta["OBS"]
    SFO1 = meta["SFO1"]
    O1 = meta["O1"]

    acq_meta = read_bruker_metadata(acqus)

    NS = acq_meta["NS"]

    # ----- processing parameters (procs) -----

    SWstr = '##$SW_p= '
    SIstr = '##$SI= '
    SFstr = '##$SF= '
    NCstr = '##$NC_proc= '

    SW = float("NaN")
    SI = float("NaN")
    SF = float("NaN")
    NC_proc = float("NaN")

    with open(procs, "rb") as inp:
        for line in inp:
            if SWstr in line.decode():
                linestr = line.decode()
                SW = float(linestr[len(SWstr):len(linestr) - 1])
            if SIstr in line.decode():
                linestr = line.decode()
                SI = float(linestr[len(SIstr):len(linestr) - 1])
            if SFstr in line.decode():
                linestr = line.decode()
                SF = float(linestr[len(SFstr):len(linestr) - 1])
            if NCstr in line.decode():
                linestr = line.decode()
                NC_proc = float(linestr[len(NCstr):len(linestr) - 1])
            if ~np.isnan(SW) and ~np.isnan(SI) and ~np.isnan(NC_proc) and ~np.isnan(SF):
                break

    # ----- x axis -----
    SR = (SF - OBS) * 1000000  # convert to MHz
    true_centre = O1 - SR
    xmin = true_centre - SW / 2
    xmax = true_centre + SW / 2
    xAxHz = np.linspace(xmax, xmin, num=int(SI))
    xAxppm = xAxHz / SF
    real_spectrum = real_spectrum * 2 ** NC_proc
    imag_spectrum = imag_spectrum * 2 ** NC_proc

    if f1p != 0 or f2p != 0:
        xlowi = np.argmax(xAxppm < f1p)
        xhighi = np.argmax(xAxppm < f2p)
        if xlowi > xhighi:
            xlowi, xhighi = xhighi, xlowi
        xAxppm = xAxppm[xlowi:xhighi]
        xAxHz = xAxHz[xlowi:xhighi]
        real_spectrum = real_spectrum[xlowi:xhighi]
        imag_spectrum = imag_spectrum[xlowi:xhighi]

    real_spectrum_norm = (
        (real_spectrum - min(real_spectrum)) / (max(real_spectrum) - min(real_spectrum))
    )
    real_spectrum = real_spectrum / mass / NS

    imag_spectrum_norm = (
        (imag_spectrum - min(imag_spectrum)) / (max(imag_spectrum) - min(imag_spectrum))
    )
    imag_spectrum = imag_spectrum / mass / NS

    if VOCS:

        pulse = pulse.upper()

        try:
            pulse_length = acq_meta[pulse]
        except KeyError:
            raise ValueError(f"{pulse} not found in acqus.")

        w1 = pulse_to_w1(pulse_length)

        scaling = vocs_power(xAxHz, w1, window_filter=True)

        real_spectrum *= scaling
        imag_spectrum *= scaling

    results = {
        'x_ppm': xAxppm, 'x_Hz': xAxHz,
        'real_spectrum': real_spectrum, 'normalised_real_spectrum': real_spectrum_norm,
        'imag_spectrum': imag_spectrum, 'normalised_imag_spectrum': imag_spectrum_norm,
        'nuclide': NUC, 'BF1': OBS, 'SFO1': SF, "path": data_path,
    }

    return results


# ============================================================
# Plotting
# ============================================================

# ============================================================
# Insets
# ============================================================

#: Keys that only make sense for a standalone figure (typeface,
#: figure size, saving, ...) and are therefore not valid inside an
#: ``inset`` spec dict -- an inset always shares the parent figure
#: and is never saved independently.
_INSET_FIGURE_LEVEL_KEYS = {
    "ax", "fig", "save_fig", "fig_name", "save_format",
    "pl_width", "pl_height", "font", "output_format",
}


def _draw_1d_insets(parent_ax, fig, inset, default_data_paths):
    """
    Draw one or more insets on ``parent_ax`` by recursively calling
    :func:`NMR1dplot` into a new inset axes -- so each inset accepts
    *exactly* the same keyword arguments as :func:`NMR1dplot` itself
    (including ``heatmap=True``, which dispatches to
    :func:`NMR1dheatmap` exactly as an ordinary top-level call would),
    plus its own placement/size and (optionally) its own data.

    Parameters
    ----------
    parent_ax : matplotlib.axes.Axes
        The main 1D axes to place the inset(s) on.

    fig : matplotlib.figure.Figure
        The shared figure (forwarded to :func:`NMR1dplot` so it does
        not try to create a new one).

    inset : dict or list of dict
        One spec per inset. Each dict must contain ``bounds`` --
        ``(x0, y0, width, height)`` in ``parent_ax`` axes-fraction
        coordinates (0-1), passed to ``parent_ax.inset_axes`` to set
        the inset's placement and size. Two optional keys are
        reserved: ``mark_inset`` (bool, default False) draws a
        connector/box between the inset and the region of the parent
        axes it corresponds to (via ``indicate_inset_zoom``), and
        ``mark_inset_kwargs`` forwards extra keyword arguments to that
        call (e.g. ``edgecolor``, ``alpha``). Every other key is
        forwarded to :func:`NMR1dplot` for that inset -- including
        ``data_paths`` to show different spectra in the inset than in
        the parent plot; if omitted, the inset defaults to the same
        ``data_paths`` as the parent. Inset settings are otherwise
        entirely independent of the parent's -- e.g. ``xlims``,
        ``colour``, ``normalise``, or ``fontsize`` are not inherited
        and must be repeated in the inset spec if wanted there too.

    default_data_paths : str or list of str
        Data to fall back on for an inset spec that doesn't supply its
        own ``data_paths``.

    Returns
    -------
    list of matplotlib.axes.Axes
        The created inset axes, in the order given in ``inset``.
    """

    specs = inset if isinstance(inset, (list, tuple)) else [inset]

    inset_axes = []
    for i, spec in enumerate(specs):

        if "bounds" not in spec:
            raise ValueError(
                f"inset[{i}] is missing required key 'bounds' "
                f"(x0, y0, width, height in axes-fraction coordinates)."
            )

        spec = dict(spec)
        bounds = spec.pop("bounds")
        if len(bounds) != 4:
            raise ValueError(
                f"inset[{i}]['bounds'] must be (x0, y0, width, height); got {bounds!r}."
            )

        mark_inset = spec.pop("mark_inset", False)
        mark_inset_kwargs = spec.pop("mark_inset_kwargs", None) or {}

        bad = _INSET_FIGURE_LEVEL_KEYS & set(spec)
        if bad:
            raise ValueError(
                f"inset[{i}] contains figure-level keys not valid inside an inset "
                f"spec: {sorted(bad)}; an inset always shares the parent figure."
            )

        data_paths = spec.pop("data_paths", default_data_paths)

        inset_ax = parent_ax.inset_axes(bounds)
        NMR1dplot(data_paths, ax=inset_ax, fig=fig, **spec)

        if mark_inset:
            parent_ax.indicate_inset_zoom(inset_ax, **mark_inset_kwargs)

        inset_axes.append(inset_ax)

    return inset_axes


def NMR1dplot(data_paths, mass=[1], procno=[1], colour='black', f1p=0, f2p=0,
              xlims='default', xlabel=None, pl_width=3.375, pl_height=3.375,
              normalise=True, yoffset=0.2, font='Arial', fontsize=12, fname=True,
              x_unit='ppm', output_format='',
              vocs_sum=False, plot_vocs_parts=False, vocs_average=False, heteronuclear_2d_slice=False,
              vocs_pulse="P1", vocs_npoints=8192, vocs_colours=None, vocs_yoffset=0.2,
              tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0, tick_number=None,
              inset=None,
              field='', save_fig=False, fig_name='nmr_1d_plot', save_format='png',
              heatmap=False, heatmap_cmap='viridis', heatmap_vmin=None, heatmap_vmax=None,
              heatmap_colorbar=False, heatmap_colorbar_label=None,
              ylabel=None, y_values=None, y_ticks='none', y_tick_number=5,
              common_range='intersection', heatmap_npoints=None, interp_kind='linear',
              ax=None, fig=None):
    """
    Plot one or more processed 1D Bruker TopSpin NMR spectra.

    Spectra can be plotted individually or, when ``vocs_sum=True``,
    multiple VOCS spectra can be VOCS-corrected, interpolated onto a
    common chemical-shift axis, and summed or averaged (see
    :func:`VOCSSum`).

    Individual VOCS components can optionally be plotted alongside the
    summed spectrum. When VOCS components are displayed, the summed
    spectrum is treated as an additional spectrum in the stack, giving
    equal baseline spacing between all components and the final sum.

    Parameters
    ----------
    data_paths : str or list of str
        Path or list of paths to Bruker experiment directories.

    mass, procno : list, optional
        Sample masses / processing numbers corresponding to
        ``data_paths``. A single-element list is broadcast to all
        spectra.

    colour : {'black', 'diverging', <matplotlib colormap name>}, optional
        Colour scheme used for ordinary (non-VOCS) spectra.

    f1p, f2p : float, optional
        Limits of the spectral region to import, in ppm.

    xlims : 'default' or [left, right], optional
        X-axis display limits. ``'default'`` (the default) uses the
        union of all plotted spectra's ranges, so spectra covering
        different ppm windows are all shown in full rather than being
        cropped to whichever spectrum happens to be plotted last. Not
        used when ``vocs_sum=True``, where all spectra are already
        interpolated onto one common axis beforehand.

    xlabel : str, optional
        Override the automatically-generated (nuclide-based) x-axis
        label.

    pl_width, pl_height : float, optional
        Figure size in inches.

    normalise : bool, optional
        If True, normalise each plotted spectrum to its maximum
        absolute intensity.

    yoffset : float, optional
        Vertical offset between ordinary 1D spectra when
        ``vocs_sum`` is False.

    font, fontsize : optional
        Typeface settings, see :func:`nmrtools.plotting.setup_plotting`.

    x_unit : {'ppm', 'Hz'}, optional
        Unit used for the x-axis.

    output_format : {'', 'manuscript', 'presentation'}, optional
        Predefined figure formatting, see
        :func:`nmrtools.plotting.setup_plotting`.

    vocs_sum, plot_vocs_parts, vocs_average, vocs_pulse, vocs_npoints,
    vocs_colours, vocs_yoffset, heteronuclear_2d_slice : optional
        See :func:`VOCSSum`. When ``vocs_sum=True``, spectra are
        VOCS-corrected, interpolated onto a common axis, and summed or
        averaged instead of being plotted as independent spectra.

    tick_dir : {'in', 'out', 'inout'}, optional
        Tick direction.

    tick_length, tick_width : float, optional
        Major tick length/width (minor ticks scale automatically).

    tick_pad : float, optional
        Padding between ticks and tick labels.

    tick_number : int, optional
        Approximate target number of x-axis ticks. If None, matplotlib's
        default locator is used.

    inset : dict or list of dict, optional
        Add one or more insets to the plot. Each dict must contain
        ``bounds`` -- ``(x0, y0, width, height)`` in axes-fraction
        coordinates of *this* plot's axes, defining the inset's
        placement and size (via ``ax.inset_axes``). Every other key
        accepts exactly the same keyword arguments as
        :func:`NMR1dplot` itself (``xlims``, ``colour``, ``normalise``,
        ``fontsize``, ``heatmap``, ...) -- so an inset can, for
        example, zoom into a crowded region of the spectrum, use a
        different colour scheme, or even switch to a heatmap, all
        independent of the parent plot's own settings (nothing is
        inherited automatically). ``data_paths`` may also be set per
        inset to show different spectra than the parent plot; if
        omitted, the inset shows the same ``data_paths``. Two more
        reserved keys: ``mark_inset`` (bool, default False) draws a
        connector box linking the inset to the region of the parent
        plot it corresponds to; ``mark_inset_kwargs`` forwards extra
        arguments to that connector (e.g. ``edgecolor``). Figure-level
        keys (``font``, ``output_format``, ``pl_width``/``pl_height``,
        ``save_fig``, ...) are not valid inside an inset spec -- it
        always shares the parent figure. Default None (no inset). The
        created inset axes are accessible afterwards via
        ``ax.child_axes``.

    field : optional

    save_fig : bool, optional
        If True, save the figure (see :func:`nmrtools.plotting.save_figure`).
        Default False.

    fig_name : str, optional
        Output path/filename, with or without an extension. Default
        ``'nmr_1d_plot'``.

    save_format : {'png', 'pdf', 'svg'}, optional
        Output format used when ``fig_name`` has no extension already.
        Default ``'png'``.

    ax : matplotlib.axes.Axes, optional
        If given, plot into this axis instead of creating a new
        figure -- e.g. for embedding in a custom subplot grid, or (see
        ``inset`` above) as the target of an inset. When ``ax`` is
        given, figure-level options (``pl_height``, ``pl_width``,
        ``font``, ``output_format``, ``save_fig``) are ignored, since
        they would affect the whole (shared) figure.

    fig : matplotlib.figure.Figure, optional
        The figure ``ax`` belongs to. If ``ax`` is given and ``fig``
        is not, it is inferred from ``ax.figure``.

    heatmap : bool, optional
        If True, ignore the stacked-line plotting below entirely and
        dispatch to :func:`NMR1dheatmap`, which interpolates every
        spectrum in ``data_paths`` onto a common axis and displays
        them as rows of a 2D heatmap instead of an offset stack. When
        True, ``colour``/``yoffset``/the VOCS options above are not
        used; use the ``heatmap_*``, ``ylabel``, ``y_values``,
        ``y_ticks``, ``y_tick_number``, ``common_range``,
        ``heatmap_npoints`` and ``interp_kind`` parameters below
        instead. See :func:`NMR1dheatmap` for full parameter
        documentation.

    heatmap_cmap : str, optional
        Colour map for the heatmap (see
        :func:`nmrtools.plotting.get_contour_cmap` for the allow-list,
        or any matplotlib colormap name). Only used if ``heatmap=True``.

    heatmap_vmin, heatmap_vmax : float, optional
        Colour-scale limits for the heatmap. Only used if
        ``heatmap=True``.

    heatmap_colorbar : bool, optional
        If True, draw a colourbar next to the heatmap. Only used if
        ``heatmap=True``.

    heatmap_colorbar_label : str, optional
        Label for the colourbar. Only used if ``heatmap=True``.

    ylabel : str, optional
        Label for the heatmap's y-axis (e.g. ``"Temperature (\u00b0C)"``).
        Only used if ``heatmap=True``.

    y_values : list, optional
        One value per spectrum in ``data_paths`` (e.g. a list of
        temperatures), used to label the y-axis when ``y_ticks`` is
        ``'all'`` or ``'auto'``. Only used if ``heatmap=True``.

    y_ticks : {'none', 'all', 'auto'}, optional
        Controls y-axis ticks on the heatmap. ``'none'`` (default): no
        y-ticks. ``'all'``: one tick per spectrum, aligned with each
        row, labelled from ``y_values`` (which must then be supplied).
        ``'auto'``: an automatically chosen, smaller number of ticks
        (see ``y_tick_number``), still labelled from ``y_values``.
        Only used if ``heatmap=True``.

    y_tick_number : int, optional
        Approximate target number of y-ticks when ``y_ticks='auto'``.
        Default 5. Only used if ``heatmap=True``.

    common_range : {'intersection', 'union'}, optional
        How the common x-axis used to interpolate all spectra onto the
        heatmap grid is chosen -- see :func:`nmrtools.arithmetic.combine_spectra`.
        Default ``'intersection'``. Only used if ``heatmap=True``.

    heatmap_npoints : int, optional
        Number of points in the common x-axis used for the heatmap. If
        None, uses the largest number of points among the input
        spectra. Only used if ``heatmap=True``.

    interp_kind : str, optional
        Passed to ``scipy.interpolate.interp1d`` when building the
        heatmap's common axis. Default ``'linear'``. Only used if
        ``heatmap=True``.

    Returns
    -------
    fig, ax
        Or, if ``heatmap=True``, ``fig, ax, im`` -- see
        :func:`NMR1dheatmap`.
    """

    if isinstance(data_paths, str):
        data_paths = [data_paths]

    if heatmap:
        return NMR1dheatmap(
            data_paths, mass=mass, procno=procno, f1p=f1p, f2p=f2p,
            xlims=xlims, xlabel=xlabel, x_unit=x_unit,
            ylabel=ylabel, y_values=y_values, y_ticks=y_ticks,
            y_tick_number=y_tick_number, common_range=common_range,
            npoints=heatmap_npoints, interp_kind=interp_kind,
            normalise=normalise, cmap=heatmap_cmap, vmin=heatmap_vmin,
            vmax=heatmap_vmax, colorbar=heatmap_colorbar,
            colorbar_label=heatmap_colorbar_label,
            pl_width=pl_width, pl_height=pl_height, font=font, fontsize=fontsize,
            output_format=output_format, heteronuclear_2d_slice=heteronuclear_2d_slice,
            tick_dir=tick_dir, tick_length=tick_length, tick_width=tick_width,
            tick_pad=tick_pad, tick_number=tick_number, field=field,
            save_fig=save_fig, fig_name=fig_name, save_format=save_format,
            inset=inset, ax=ax, fig=fig,
        )

    if len(mass) != len(data_paths):
        if mass != [1]:
            print("Mass list length doesn't match number of data files, "
                  "continuing with assumption that all masses are equal.")
        mass = [1] * len(data_paths)

    if len(procno) != len(data_paths):
        if procno != [1]:
            print("procno list length doesn't match number of data files, "
                  "continuing with assumption that all are 1.")
        procno = [1] * len(data_paths)

    standalone = ax is None

    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style['pl_width'], style['pl_height']
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    values = range(len(data_paths))
    norm = plt.Normalize(vmin=values[0], vmax=values[-1])
    cmap = get_stack_colormap(colour)
    scalarMap = cmx.ScalarMappable(norm=norm, cmap=cmap)

    # ============================================================
    # VOCS summed spectrum
    # ============================================================

    if vocs_sum:

        vocs_results = VOCSSum(
            data_paths, procno=procno[0], heteronuclear_2d_slice=heteronuclear_2d_slice,
            pulse=vocs_pulse, npoints=vocs_npoints, average=vocs_average,
            f1p=f1p, f2p=f2p,
        )

        x = vocs_results["ppm"]

        if vocs_colours is None:
            vocs_colours = ['black'] * len(data_paths)
        elif len(vocs_colours) != len(data_paths):
            raise ValueError(
                "vocs_colours must contain one colour for each VOCS spectrum."
            )

        interpolated = vocs_results["interpolated_slices"]

        plotted_slices = []

        for y in interpolated:
            if normalise:
                max_signal = np.max(np.abs(y))
                y_plot = y / max_signal if max_signal != 0 else y
            else:
                y_plot = y
            plotted_slices.append(y_plot)

        y_sum = vocs_results["VOCS_sum"]

        if normalise:
            max_signal = np.max(np.abs(y_sum))
            if max_signal != 0:
                y_sum = y_sum / max_signal

        # yBase = cnt * vocs_yoffset:
        #   vocs_yoffset = 0   -> all spectra overlap exactly
        #   vocs_yoffset = 0.5 -> 0.5 y-unit between baselines
        #   vocs_yoffset = 1   -> 1 y-unit between baselines

        if plot_vocs_parts:
            for cnt, y_plot in enumerate(plotted_slices):
                yBase = cnt * vocs_yoffset
                ax.plot(x, y_plot + yBase, color=vocs_colours[cnt],
                        linewidth=1.0, alpha=0.8)

        sum_index = len(plotted_slices) if plot_vocs_parts else 0
        sum_offset = sum_index * vocs_yoffset

        ax.plot(x, y_sum + sum_offset, color='black', linewidth=1.5, label='VOCS sum')

        xmin = x[0]
        xmax = x[-1]

        NUC = vocs_results["slices"][0]["nuclide"]
        SFO1 = vocs_results["slices"][0]["SFO1"]

        nucgamma = find_gamma(NUC, field=field, SFO1=SFO1)
        NUC = nucgamma['Nuclide']

        labeltext = NUC_label(NUC, x_unit=x_unit)

    # ============================================================
    # Normal individual spectra
    # ============================================================

    else:

        yBase = 0
        cnt = 0
        xmin = xmax = labeltext = None

        for expt in data_paths:

            results = NMR1dimport(expt, procno=procno[cnt], mass=mass[cnt], f1p=f1p, f2p=f2p)

            x = results['x_ppm'] if x_unit == 'ppm' else results['x_Hz']
            y = results['normalised_real_spectrum'] if normalise else results['real_spectrum']

            this_xmin, this_xmax = x[0], x[-1]
            if this_xmin > this_xmax:
                this_xmin, this_xmax = this_xmax, this_xmin

            # Union across spectra, not just the last one plotted: with
            # xlims='default' and spectra that span different ranges,
            # the display should cover all of them.
            xmin = this_xmin if xmin is None else min(xmin, this_xmin)
            xmax = this_xmax if xmax is None else max(xmax, this_xmax)

            NUC = results['nuclide']
            SFO1 = results['SFO1']

            nucgamma = find_gamma(NUC, field=field, SFO1=SFO1)
            NUC = nucgamma['Nuclide']

            labeltext = NUC_label(NUC, x_unit=x_unit)

            ax.plot(x, y + yBase, color=scalarMap.to_rgba(values[cnt]), linewidth=1.5)

            yBase += max(y) * (1 + yoffset)
            cnt += 1

    if xlims == 'default':
        ax.set_xlim(xmin, xmax)
    else:
        xmin, xmax = min(xlims), max(xlims)
        ax.set_xlim(xmin, xmax)

    ax.set_xlabel(xlabel if xlabel is not None else labeltext, fontsize=fontsize)
    apply_spine_style(ax, top=False, right=False, bottom=True, left=False)
    ax.set_yticks([])
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis='x')
    ax.tick_params(axis='x', which='major', labelsize=fontsize)

    ax.invert_xaxis()

    if standalone:
        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.tight_layout()

    # ----- inset(s) -----
    if inset is not None:
        fig.canvas.draw()
        _draw_1d_insets(ax, fig, inset, data_paths)

    if standalone and save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax


# ============================================================
# Heatmap (multiple 1D spectra as a 2D image)
# ============================================================

def NMR1dheatmap(data_paths, mass=[1], procno=[1], f1p=0, f2p=0,
                  xlims='default', xlabel=None, x_unit='ppm',
                  ylabel=None, y_values=None, y_ticks='none', y_tick_number=5,
                  common_range='intersection', npoints=None, interp_kind='linear',
                  normalise=True, cmap='viridis', vmin=None, vmax=None,
                  colorbar=False, colorbar_label=None,
                  pl_width=3.375, pl_height=3.375, font='Arial', fontsize=12,
                  output_format='', heteronuclear_2d_slice=False,
                  tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0, tick_number=None,
                  field='', save_fig=False, fig_name='nmr_1d_heatmap', save_format='png',
                  inset=None, ax=None, fig=None):
    """
    Plot multiple processed 1D Bruker TopSpin NMR spectra as rows of a
    2D heatmap, instead of as an offset stack (cf. :func:`NMR1dplot`).

    Every spectrum is imported with :func:`NMR1dimport`, then all
    spectra are interpolated onto one common x-axis with
    :func:`nmrtools.arithmetic.combine_spectra` (the same interpolation
    machinery used for signed spectrum combinations), stacked into a
    ``(n_spectra, npoints)`` matrix, and displayed with
    ``pcolormesh``. Rows are evenly spaced (one per input spectrum,
    in the order given) -- the y-axis is categorical, not itself an
    interpolated physical axis.

    Parameters
    ----------
    data_paths : str or list of str
        Path or list of paths to Bruker experiment directories. At
        least 2 required.

    mass, procno : list, optional
        Sample masses / processing numbers corresponding to
        ``data_paths``. A single-element list is broadcast to all
        spectra.

    f1p, f2p : float, optional
        Limits of the spectral region to import, in ppm.

    xlims : 'default' or [left, right], optional
        X-axis display limits.

    xlabel : str, optional
        Override the automatically-generated (nuclide-based) x-axis
        label.

    x_unit : {'ppm', 'Hz'}, optional
        Unit used for the x-axis, and the axis interpolated onto for
        the heatmap grid.

    ylabel : str, optional
        Label for the y-axis (e.g. ``"Temperature (\u00b0C)"``). None
        (default) leaves the y-axis unlabelled.

    y_values : list, optional
        One value per spectrum in ``data_paths`` (e.g. a list of
        temperatures, one per spectrum, in the same order), used to
        label the y-axis when ``y_ticks`` is ``'all'`` or ``'auto'``.
        Required in that case; ignored (and not required) when
        ``y_ticks='none'``.

    y_ticks : {'none', 'all', 'auto'}, optional
        ``'none'`` (default): no y-axis ticks at all, matching
        :func:`NMR1dplot`'s stacked spectra.
        ``'all'``: one tick per spectrum, aligned with that
        spectrum's row, labelled from ``y_values``.
        ``'auto'``: an automatically-chosen, smaller number of ticks
        (see ``y_tick_number``) at rows spread evenly across the
        stack, still labelled with the corresponding entries of
        ``y_values`` -- use this when there are too many spectra to
        label every row legibly.

    y_tick_number : int, optional
        Approximate target number of y-ticks when ``y_ticks='auto'``.
        Default 5.

    common_range : {'intersection', 'union'}, optional
        Passed to :func:`nmrtools.arithmetic.combine_spectra` to build
        the common x-axis. ``'intersection'`` (default) only uses the
        x-range covered by every input spectrum -- recommended here,
        since ``'union'`` would zero-fill each row outside that
        spectrum's own range, which reads as real (missing) signal in
        a heatmap.

    npoints : int, optional
        Number of points in the common x-axis. If None, uses the
        largest number of points among the input spectra.

    interp_kind : str, optional
        Passed to ``scipy.interpolate.interp1d`` for building the
        common x-axis. Default ``'linear'``.

    normalise : bool, optional
        If True (default), normalise each spectrum to [0, 1] before
        interpolation (matches :func:`NMR1dimport`'s
        ``normalised_real_spectrum``), so no single spectrum's
        intensity dominates the colour scale.

    cmap : str, optional
        Colour map for the heatmap. Checked first against
        :func:`nmrtools.plotting.get_contour_cmap`'s allow-list, then
        falls back to any matplotlib colormap name.

    vmin, vmax : float, optional
        Colour-scale limits. If None, matplotlib chooses them from the
        data.

    colorbar : bool, optional
        If True, draw a colourbar next to the heatmap. Default False.

    colorbar_label : str, optional
        Label for the colourbar.

    pl_width, pl_height : float, optional
        Figure size in inches.

    font, fontsize : optional
        Typeface settings, see :func:`nmrtools.plotting.setup_plotting`.

    output_format : {'', 'manuscript', 'presentation'}, optional
        Predefined figure formatting, see
        :func:`nmrtools.plotting.setup_plotting`.

    heteronuclear_2d_slice : optional
        See :func:`NMR1dimport`.

    tick_dir : {'in', 'out', 'inout'}, optional
        Tick direction (applied to the x-axis, and to the y-axis when
        ``y_ticks`` is not ``'none'``).

    tick_length, tick_width : float, optional
        Major tick length/width (x-axis minor ticks scale
        automatically; the categorical y-axis never gets minor ticks).

    tick_pad : float, optional
        Padding between ticks and tick labels.

    tick_number : int, optional
        Approximate target number of x-axis ticks. If None, matplotlib's
        default locator is used.

    inset : dict or list of dict, optional
        Add one or more insets to the heatmap. Same mechanism as
        :func:`NMR1dplot`'s ``inset`` (which this dispatches from,
        since :func:`NMR1dplot` is itself what every inset recurses
        through) -- each dict needs ``bounds`` (``(x0, y0, width,
        height)`` in axes-fraction coordinates); every other key
        accepts exactly the same keyword arguments as
        :func:`NMR1dplot` itself. Default None (no inset).

    ax : matplotlib.axes.Axes, optional
        If given, plot into this axis instead of creating a new
        figure. When given, figure-level options (``pl_height``,
        ``pl_width``, ``font``, ``output_format``, ``save_fig``) are
        ignored.

    fig : matplotlib.figure.Figure, optional
        The figure ``ax`` belongs to. If ``ax`` is given and ``fig``
        is not, it is inferred from ``ax.figure``.

    field : optional
        See :func:`nmrtools.utils.find_gamma`.

    save_fig : bool, optional
        If True, save the figure (see :func:`nmrtools.plotting.save_figure`).
        Default False.

    fig_name : str, optional
        Output path/filename, with or without an extension. Default
        ``'nmr_1d_heatmap'``.

    save_format : {'png', 'pdf', 'svg'}, optional
        Output format used when ``fig_name`` has no extension already.
        Default ``'png'``.

    Returns
    -------
    fig, ax, im
        ``im`` is the ``QuadMesh`` returned by ``pcolormesh``, handy
        for further colour-scale tweaks after the call.

    Raises
    ------
    ValueError
        If fewer than 2 spectra are given, ``y_ticks`` is ``'all'`` or
        ``'auto'`` without a matching ``y_values``, or ``y_ticks`` is
        not one of ``'none'``/``'all'``/``'auto'``.
    """

    if isinstance(data_paths, str):
        data_paths = [data_paths]

    if len(data_paths) < 2:
        raise ValueError("Need at least 2 spectra to build a heatmap.")

    if len(mass) != len(data_paths):
        if mass != [1]:
            print("Mass list length doesn't match number of data files, "
                  "continuing with assumption that all masses are equal.")
        mass = [1] * len(data_paths)

    if len(procno) != len(data_paths):
        if procno != [1]:
            print("procno list length doesn't match number of data files, "
                  "continuing with assumption that all are 1.")
        procno = [1] * len(data_paths)

    if y_ticks not in ('none', 'all', 'auto'):
        raise ValueError("y_ticks must be one of 'none', 'all', 'auto'.")

    if y_ticks != 'none':
        if y_values is None:
            raise ValueError(
                f"y_ticks='{y_ticks}' requires y_values (one entry per spectrum)."
            )
        if len(y_values) != len(data_paths):
            raise ValueError("y_values must have one entry per spectrum.")

    standalone = ax is None

    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style['pl_width'], style['pl_height']

    # ----- import every spectrum -----

    imported = [
        NMR1dimport(expt, procno=procno[cnt], mass=mass[cnt], f1p=f1p, f2p=f2p,
                    heteronuclear_2d_slice=heteronuclear_2d_slice)
        for cnt, expt in enumerate(data_paths)
    ]

    NUC = imported[0]['nuclide']
    SFO1 = imported[0]['SFO1']
    nucgamma = find_gamma(NUC, field=field, SFO1=SFO1)
    labeltext = NUC_label(nucgamma['Nuclide'], x_unit=x_unit)

    # ----- interpolate every spectrum onto a common x-axis -----
    #
    # combine_spectra() only ever looks at the x_ppm/x_Hz and
    # real_spectrum keys, and returns each input spectrum interpolated
    # onto the common axis (before any signed combination) in
    # 'interpolated_spectra', shape (n_spectra, npoints) -- exactly
    # the matrix a heatmap needs. Feed it the normalised or raw
    # spectrum depending on `normalise`, and ignore the combined sum.

    key = 'normalised_real_spectrum' if normalise else 'real_spectrum'
    stack_input = [
        {'x_ppm': r['x_ppm'], 'x_Hz': r['x_Hz'], 'real_spectrum': r[key]}
        for r in imported
    ]

    combined = combine_spectra(
        stack_input, x_unit=x_unit, common_range=common_range,
        npoints=npoints, interp_kind=interp_kind,
    )

    x_common = combined['x_ppm'] if x_unit == 'ppm' else combined['x_Hz']
    data_matrix = combined['interpolated_spectra']

    # ----- plot -----

    if standalone:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    try:
        cmap_obj = get_contour_cmap(cmap)
    except ValueError:
        cmap_obj = plt.get_cmap(cmap)

    row_positions = np.arange(len(data_paths))

    im = ax.pcolormesh(x_common, row_positions, data_matrix, cmap=cmap_obj,
                        vmin=vmin, vmax=vmax, shading='auto')

    if xlims == 'default':
        ax.set_xlim(x_common[0], x_common[-1])
    else:
        ax.set_xlim(min(xlims), max(xlims))

    ax.set_xlabel(xlabel if xlabel is not None else labeltext, fontsize=fontsize)
    ax.invert_xaxis()

    apply_spine_style(ax, top=False, right=False, bottom=True, left=True)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis='x', minor=True)
    ax.tick_params(axis='x', which='major', labelsize=fontsize)

    # The y-axis is categorical (one row per spectrum), so it is
    # styled directly rather than via apply_tick_style, which would
    # otherwise toggle minor ticks package-wide for both axes.
    if y_ticks == 'none':
        ax.set_yticks([])
    else:
        if y_ticks == 'all':
            tick_idx = list(range(len(data_paths)))
        else:  # 'auto'
            locator = MaxNLocator(nbins=y_tick_number, integer=True)
            raw_idx = locator.tick_values(0, len(data_paths) - 1)
            tick_idx = sorted({
                int(round(i)) for i in raw_idx if 0 <= i <= len(data_paths) - 1
            })

        ax.set_yticks(tick_idx)
        ax.set_yticklabels([y_values[i] for i in tick_idx])
        ax.tick_params(axis='y', which='major', direction=tick_dir,
                        length=tick_length, width=tick_width, pad=tick_pad,
                        labelsize=fontsize)
        ax.tick_params(axis='y', which='minor', length=0)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)

    if colorbar:
        if ax.get_axes_locator() is not None:
            # ax's position is dictated by a fixed locator (e.g. it is
            # itself an inset axes created via `Axes.inset_axes`, as
            # happens for every inset drawn through `inset=`, or for
            # any ax the caller built the same way) rather than being
            # free/subplot-managed. fig.colorbar()'s default approach
            # shrinks such an ax in place via set_position(), but that
            # shrink is silently discarded on every redraw by the
            # ax's own locator -- leaving the colorbar overlapping the
            # right-hand edge of the plot instead of sitting beside
            # it. Carve the colorbar's own axes directly out of ax's
            # own axes-fraction space instead (via a second, small
            # inset), which composes correctly with any locator.
            cax = ax.inset_axes((1.03, 0.0, 0.04, 1.0))
            cbar = fig.colorbar(im, cax=cax)
        else:
            cbar = fig.colorbar(im, ax=ax)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)

    if standalone:
        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.tight_layout()

    # ----- inset(s) -----
    if inset is not None:
        fig.canvas.draw()
        _draw_1d_insets(ax, fig, inset, data_paths)

    if standalone and save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax, im


# ============================================================
# VOCS (Variable Offset Cumulative Spectroscopy)
# ============================================================

def vocs_f1(x, w1t1=np.pi / 2, w1t2=np.pi):
    """
    Calculate the first finite-pulse-width excitation-profile term
    used in the VOCS correction.

    Parameters
    ----------
    x : array-like
        Dimensionless frequency offset (offset from the VOCS centre
        divided by the RF nutation frequency).

    w1t1, w1t2 : float, optional
        Flip angles of the first/second pulse in radians.
        Defaults are ``pi/2`` and ``pi``.
    """
    s = np.sqrt(1.0 + x ** 2)
    return (
        np.sin(w1t1 * s) * (1.0 - np.cos(w1t2 * s)) / (1.0 + x ** 2) ** 1.5
    )


def vocs_f2(x, w1t1=np.pi / 2, w1t2=np.pi):
    """
    Calculate the second finite-pulse-width excitation-profile term
    used in the VOCS correction. See :func:`vocs_f1` for parameters.
    """
    s = np.sqrt(1.0 + x ** 2)
    return (
        -x * (1.0 - np.cos(w1t1 * s)) * (1.0 - np.cos(w1t2 * s)) / (1.0 + x ** 2) ** 2.0
    )


def vocs_power(freq_Hz, w1, window_filter=True, w1t1=np.pi / 2, w1t2=np.pi):
    """
    Calculate the VOCS frequency-dependent scaling function.

    Constructed from the magnitude of the two finite-pulse-width
    excitation-profile terms, :func:`vocs_f1` and :func:`vocs_f2`:

        ``power = sqrt(f1**2 + f2**2)``

    where ``x = (freq_Hz - freq_Hz[centre]) / w1``.

    Parameters
    ----------
    freq_Hz : array-like
        Frequency axis in Hz.

    w1 : float
        RF nutation frequency in Hz.

    window_filter : bool, optional
        If True, zero the scaling function outside ``|x| < 1.73``, to
        limit the correction to the useful VOCS excitation bandwidth.
        Default True.

    w1t1, w1t2 : float, optional
        Flip angles of the first/second pulse in radians.

    Returns
    -------
    power : ndarray
    """

    centre = len(freq_Hz) // 2
    x = (freq_Hz - freq_Hz[centre]) / w1

    f1 = vocs_f1(x, w1t1=w1t1, w1t2=w1t2)
    f2 = vocs_f2(x, w1t1=w1t1, w1t2=w1t2)

    power = np.sqrt(f1 ** 2 + f2 ** 2)

    if window_filter:
        power *= (np.abs(x) < 1.73)

    return power


def pulse_to_w1(pulse_length_us):
    """
    Convert a 90-degree pulse length (microseconds) to RF nutation
    frequency in Hz: ``w1 = 1e6 / (4 * pulse_length_us)`` (equivalent
    to the TopSpin expression).

    Examples
    --------
    >>> pulse_to_w1(5)
    50000.0
    """
    return 1e6 / (4.0 * pulse_length_us)


def apply_VOCS(results, pulse="P1", window_filter=True):
    """
    Apply a VOCS correction to an already-imported NMR spectrum.

    The pulse length is read from the Bruker ``acqus`` file associated
    with the spectrum (``results["path"]``). The scaling function is
    applied to both the real and imaginary components.

    Parameters
    ----------
    results : dict
        Output of :func:`NMR1dimport`. Must contain ``"path"`` and
        ``"x_Hz"``.

    pulse : str, optional
        Bruker acquisition parameter containing the pulse length, e.g.
        ``"P1"`` or ``"P2"``. Case-insensitive. Default ``"P1"``.

    window_filter : bool, optional
        If True, apply the ``|x| < 1.73`` VOCS bandwidth filter.

    Returns
    -------
    results : dict
        The input dictionary, with ``"real_spectrum"`` and
        ``"imag_spectrum"`` VOCS-corrected in place, and a new
        ``"VOCS_scaling"`` entry.

    Raises
    ------
    ValueError
        If the requested pulse parameter is not found.
    """

    acqus = os.path.join(results["path"], "acqus")
    meta = read_bruker_metadata(acqus)

    pulse = pulse.upper()

    try:
        pulse_length = meta[pulse]
    except KeyError:
        raise ValueError(f"{pulse} not found in acqus.")

    w1 = pulse_to_w1(pulse_length)

    scaling = vocs_power(results["x_Hz"], w1, window_filter=window_filter)

    results["real_spectrum"] *= scaling
    results["imag_spectrum"] *= scaling
    results["VOCS_scaling"] = scaling

    return results


def VOCSSum(data_paths, procno=1, heteronuclear_2d_slice=False, slice_dimension=2,
            pulse="P1", npoints=16384, average=False, f1p=0, f2p=0):
    """
    Import, VOCS-correct, interpolate, and combine multiple NMR spectra.

    Each spectrum is independently imported and VOCS-corrected.
    Because individual VOCS spectra can have different spectral
    offsets, all spectra are interpolated onto a common ppm axis
    covering the union of their individual spectral ranges, and then
    summed or averaged.

    Parameters
    ----------
    data_paths : list of str
        Paths to top-level Bruker experiment directories.

    procno : int, optional
        Processing number used for each experiment. Default 1.

    heteronuclear_2d_slice, slice_dimension : optional
        See :func:`NMR1dimport`.

    pulse : str, optional
        Bruker acquisition parameter containing the pulse length used
        for the VOCS correction. Default ``"P1"``.

    npoints : int, optional
        Number of points in the common ppm axis used for
        interpolation. Default 16384.

    average : bool, optional
        If True, average the interpolated spectra; if False (default),
        sum them.

    f1p, f2p : float, optional
        Limits of the spectral region to import, in ppm.

    Returns
    -------
    dict
        ``"slices"``              : list of :func:`NMR1dimport` outputs.
        ``"ppm"``                 : common ppm axis.
        ``"VOCS_sum"``            : summed/averaged VOCS spectrum.
        ``"interpolated_slices"`` : each VOCS spectrum interpolated
            onto the common axis, shape (n_spectra, npoints).

    Notes
    -----
    Spectra are sorted by their ppm axis before interpolation, since
    ``scipy.interpolate.interp1d`` requires increasing x. Values
    outside the original range of each individual spectrum are
    assigned zero.
    """

    spectra = []

    for path in data_paths:
        spec = NMR1dimport(
            path, procno=procno, f1p=f1p, f2p=f2p,
            heteronuclear_2d_slice=heteronuclear_2d_slice,
            slice_dimension=slice_dimension, VOCS=True, pulse=pulse,
        )
        spectra.append(spec)

    ppm_low = min(np.min(s["x_ppm"]) for s in spectra)
    ppm_high = max(np.max(s["x_ppm"]) for s in spectra)
    ppm_common = np.linspace(ppm_low, ppm_high, npoints)

    interpolated = []

    for spec in spectra:
        x = np.asarray(spec["x_ppm"])
        y = np.asarray(spec["real_spectrum"])

        sort_idx = np.argsort(x)
        x_sorted, y_sorted = x[sort_idx], y[sort_idx]

        interp_function = interp1d(x_sorted, y_sorted, bounds_error=False, fill_value=0.0)
        interpolated.append(interp_function(ppm_common))

    interpolated = np.asarray(interpolated)

    summed = np.mean(interpolated, axis=0) if average else np.sum(interpolated, axis=0)

    return {
        "slices": spectra,
        "ppm": ppm_common,
        "VOCS_sum": summed,
        "interpolated_slices": interpolated,
    }


# ============================================================
# CPMG (Carr-Purcell-Meiboom-Gill) processing
# ============================================================
#
# Splits a raw echo-train FID into individual echoes, co-adds them,
# optionally apodizes, zero-fills, and Fourier transforms -- the same
# workflow as manually stepping through a CPMG-processing notebook,
# but as a single parameterized call with auto-detected defaults for
# the steps that are normally tuned by eye (locating the first echo,
# choosing how many echoes to zero-fill to).
#
# The Bruker delay/constant/loop-counter indices used to find the
# echo spacing, blanking, and echo count (``D[6]``, ``D[3]``,
# ``CNST[20]``, ``L[22]`` by default) are a property of the pulse
# program, not a fixed nmrtools convention -- they are exposed as
# keyword arguments so a different CPMG pulse program can be
# supported by overriding them, rather than requiring a code change.

def _cpmg_first_echo_point(fid, offset=20):
    """
    Guess the index of the first non-zero point of a raw CPMG FID,
    i.e. the start of the blanking-delay-free region just before the
    first echo, plus a small safety ``offset``.

    Mirrors "make a guess for the first data point that is non-zero"
    in the manual notebook workflow, so :func:`CPMGprocess` can locate
    the echo train automatically instead of requiring it to be found
    by eye on a plot.

    Parameters
    ----------
    fid : ndarray of complex
        Raw FID, as returned by :func:`nmrtools.utils.read_bruker_fid`.

    offset : int, optional
        Number of extra points to move past the detected first
        non-zero point, to safely clear any residual blanking
        artefact right at the edge. Default 20.

    Returns
    -------
    int
    """

    mask = fid.real != 0

    if not mask.any():
        raise ValueError(
            "CPMG FID is entirely zero; cannot locate the first echo. "
            "Pass first_point explicitly instead of 'auto'."
        )

    return int(np.argmax(mask)) + offset


def _cpmg_apodization_window(npoints, offset_zero=20, shape=2, alpha=0.54):
    """
    Build a single-echo apodization window: a Hamming-type window of
    ``npoints - 2 * offset_zero`` points, flanked by ``offset_zero``
    exact zeros on each side.

    The zero flanks remove the (already very small) contribution of
    the blanking-adjacent edges of each co-added echo without
    softening the whole echo the way a plain Hamming window would;
    the shaped centre then tapers the remaining echo smoothly to
    (near) zero at both ends before the FFT.

    Parameters
    ----------
    npoints : int
        Length of the co-added echo to be windowed.

    offset_zero : int, optional
        Number of exact-zero points at each end. Default 20.

    shape : float, optional
        Exponent applied to the ``|sin|`` term; 2 gives a
        conventional Hamming-like taper. Default 2.

    alpha : float, optional
        Hamming window coefficient. Default 0.54.

    Returns
    -------
    ndarray, length ``npoints``
    """

    core_n = npoints - 2 * offset_zero

    if core_n <= 0:
        raise ValueError(
            f"offset_zero ({offset_zero}) leaves no points to window "
            f"in an echo of only {npoints} points."
        )

    idx = np.arange(core_n)
    core = alpha + (1 - alpha) * (
        1 - np.abs(np.sin((np.pi / core_n) * (idx - core_n / 2))) ** shape
    )

    zeros = np.zeros(offset_zero)

    return np.concatenate([zeros, core, zeros])


def _cpmg_phase_correction(spectrum, phase0=0.0, phase1=0.0):
    """
    Apply a conventional 0th/1st-order phase correction to a
    frequency-domain spectrum.

    Parameters
    ----------
    spectrum : ndarray of complex

    phase0, phase1 : float, optional
        Zero- and first-order phase, in degrees. Default 0 (no
        correction).

    Returns
    -------
    ndarray of complex
    """

    n = len(spectrum)
    phase = np.deg2rad(phase0) + np.deg2rad(phase1) * np.arange(n) / n

    return spectrum * np.exp(1j * phase)


def _cpmg_reference(data_path, procno, meta):
    """``(SF, SR)`` for the ppm axis of a CPMG spectrum, in MHz / Hz.

    The same referencing :mod:`opennmr.io.bruker` uses for every other
    Bruker spectrum: ``SF`` from ``pdata/<procno>/procs`` when there is
    one (so a spectrum referenced in TopSpin keeps its ``SR``), else
    ``BF1`` with no ``SR``.
    """
    bf1 = float(meta["OBS"])
    procs_path = os.path.join(data_path, "pdata", str(procno), "procs")
    sf = None
    if os.path.isfile(procs_path):
        from ..io.jcampdx import parse_jcampdx_params
        try:
            sf = float(parse_jcampdx_params(procs_path).get("SF"))
        except (TypeError, ValueError):
            sf = None
    if not sf:
        sf = bf1
    return sf, (sf - bf1) * 1e6


def cpmg_echo_layout(data_path, fid=None, meta=None, fid_name="fid",
                     echo_delay_index=6, blanking_delay_index=3,
                     blanking_cnst_index=20, num_echoes_index=22,
                     first_point='auto', first_point_offset=20, safety=10):
    """
    Where :func:`CPMGprocess` will cut a CPMG echo train, without
    processing anything -- for checking the split before running it.

    Reads ``acqus`` (and the raw FID, unless ``fid`` is given) and
    resolves the echo spacing, blanking, echo count and the first echo
    exactly as :func:`CPMGprocess` does.

    Returns
    -------
    dict
        ``dwell_us`` (per complex point), ``echo_length_us`` (``D[echo]``),
        ``blanking_us`` (``2 D[blanking] + CNST[blanking]``),
        ``num_echoes`` (``L[count]``), ``period_exact`` (echo period in
        points -- not necessarily a whole number), ``period`` (whole
        points kept per echo window), ``first_point``, ``starts`` (the
        index each echo window starts at), ``trim`` (points dropped
        from each end of each window: blanking plus ``safety``),
        ``echo_points`` (points per co-added echo), ``fid``, ``meta``,
        and the indices used, for the record.

    Raises
    ------
    ValueError
        If a parameter array or index is missing, the spacing is not
        positive, or the FID is too short for the echo train.
    """

    if meta is None:
        meta = read_bruker_metadata(os.path.join(data_path, "acqus"))

    for name, index in (("D", echo_delay_index), ("D", blanking_delay_index),
                        ("CNST", blanking_cnst_index), ("L", num_echoes_index)):
        values = meta.get(name)
        if values is None:
            raise ValueError(f"The {name} array is not in acqus at {data_path!r}.")
        if not 0 <= int(index) < len(values):
            raise ValueError(f"{name}{index} is out of range: acqus has {name}0..{name}{len(values) - 1}.")

    if not meta.get("SW_h"):
        raise ValueError("SW_h is missing from acqus, so the dwell time is unknown.")

    if fid is None:
        fid = read_bruker_fid(data_path, meta=meta, fid_name=fid_name)

    dw = 0.5e6 / meta["SW_h"]  # half the dwell per complex point, microseconds
    echo_length = 1e6 * meta["D"][echo_delay_index]  # microseconds
    blanking = 2e6 * meta["D"][blanking_delay_index] + meta["CNST"][blanking_cnst_index]
    num_echoes = int(meta["L"][num_echoes_index])

    if num_echoes < 1:
        raise ValueError(f"L{num_echoes_index} = {num_echoes}: no echoes to split.")

    period_exact = (echo_length + blanking) / (2 * dw)
    if period_exact <= 0:
        raise ValueError(
            f"The echo period is not positive (D{echo_delay_index} = {echo_length:g} us, "
            f"blanking = {blanking:g} us). Check the acqus indices.")

    if first_point == 'auto':
        first = _cpmg_first_echo_point(fid, offset=first_point_offset)
    else:
        first = int(first_point)

    period = int(period_exact)
    blanking_p = int(0.5 * blanking / (2 * dw)) + 2
    # Points dropped from each end of a window. Clamped at zero: a
    # safety larger than half the blanking would otherwise turn into a
    # negative slice index and silently take the wrong points.
    trim = max(0, int(0.5 * blanking / (2 * dw) - safety))
    begin = int(first - blanking / (2 * dw) + blanking_p)

    # Each echo starts at its own exact position, rounded, rather than
    # at a multiple of a truncated whole-point period. The two are
    # identical whenever the period is a whole number of points; when
    # it is not, the truncated version drifts by the fraction every
    # echo -- 0.4 point/echo over 200 echoes puts the last echo 80
    # points (most of an echo) away from where it was cut.
    starts = begin + np.rint(np.arange(num_echoes) * period_exact).astype(int)

    needed = int(starts[-1]) + period
    if begin < 0 or needed > len(fid):
        raise ValueError(
            "Raw FID is too short for the requested echo train "
            f"(need points {begin}..{needed}, the FID has {len(fid)}). "
            "Check first_point and the echo/blanking acqus indices.")

    echo_points = period - 2 * trim
    if echo_points <= 0:
        raise ValueError(
            f"Nothing is left of each echo after trimming {trim} points from each end "
            f"of a {period}-point window. Reduce safety or check the blanking indices.")

    return {
        "dwell_us": 2 * dw, "echo_length_us": echo_length, "blanking_us": blanking,
        "num_echoes": num_echoes, "period_exact": period_exact, "period": period,
        "first_point": first, "begin": begin, "starts": starts, "trim": trim,
        "echo_points": echo_points, "fid": fid, "meta": meta,
        "indices": {"echo_delay": echo_delay_index, "blanking_delay": blanking_delay_index,
                    "blanking_cnst": blanking_cnst_index, "num_echoes": num_echoes_index},
    }


def CPMGprocess(data_path, procno=1, mass=1, fid_name="fid",
                 echo_delay_index=6, blanking_delay_index=3,
                 blanking_cnst_index=20, num_echoes_index=22,
                 cutout=0, num_echoes_added=None, safety=10,
                 first_point='auto', first_point_offset=20,
                 apodize=True, apod_offset_zero=20, apod_shape=2, apod_alpha=0.54,
                 zero_fill='auto', magnitude_mode=True, phase0=0.0, phase1=0.0,
                 f1p=0, f2p=0):
    """
    Process a raw Bruker CPMG echo-train FID into a spectrum.

    Reads the raw time-domain signal directly (not the processed
    ``pdata``), splits it into its individual echoes using the echo
    spacing/blanking/echo-count read from ``acqus``, co-adds the
    echoes, optionally apodizes the co-added echo, zero-fills, and
    Fourier transforms. Equivalent to working through a CPMG
    echo-splitting notebook by hand, but with the steps normally
    tuned by eye (finding the first echo, choosing the zero-fill
    size) auto-detected by default.

    Parameters
    ----------
    data_path : str
        Path to the top-level Bruker experiment directory (containing
        ``acqus`` and the raw ``fid_name`` file directly inside it --
        not ``pdata``).

    procno : int, optional
        Only used for referencing: ``SF`` (and so ``SR``) is read from
        ``pdata/<procno>/procs`` when it exists, as for any other
        Bruker spectrum; otherwise ``BF1`` is used. Default 1.

    mass : float, optional
        Sample mass used to scale the spectrum, as in
        :func:`NMR1dimport`. Default 1.

    fid_name : str, optional
        Name of the raw data file inside ``data_path``. Default
        ``"fid"``.

    echo_delay_index, blanking_delay_index, blanking_cnst_index,
    num_echoes_index : int, optional
        Indices, within the ``D``, ``D``, ``CNST``, and ``L`` Bruker
        parameter arrays respectively, that this pulse program stores
        the echo length, blanking delay, blanking constant, and
        number of acquired echoes in. Defaults (6, 3, 20, 22) match a
        commonly-used CPMG pulse program; override them if a
        different one stores these elsewhere.

    cutout : int, optional
        Number of echoes to omit from the *start* of the echo train
        before co-adding. Default 0.

    num_echoes_added : int, optional
        Number of echoes (counting from ``cutout``) to co-add. Default
        None, meaning every acquired echo is used.

    safety : int, optional
        Number of points trimmed from each end of every echo, inside
        the blanking-free window, before co-adding -- guards against
        residual blanking artefacts right at the echo edges. Default
        10.

    first_point : 'auto' or int, optional
        Index of the first non-zero point of the raw FID, i.e. the
        start of the echo train. ``'auto'`` (default) locates it with
        :func:`_cpmg_first_echo_point`; pass an explicit integer to
        override (e.g. if the automatic guess splits the echoes at
        the wrong point).

    first_point_offset : int, optional
        Only used when ``first_point='auto'``: extra points added past
        the detected first non-zero point. Default 20.

    apodize : bool, optional
        If True (default), apply a Hamming-type window with zeroed
        edges (:func:`_cpmg_apodization_window`) to the co-added echo
        before the FFT.

    apod_offset_zero, apod_shape, apod_alpha : optional
        See :func:`_cpmg_apodization_window`.

    zero_fill : 'auto' or int, optional
        Number of points to zero-fill the co-added echo to before the
        FFT. ``'auto'`` (default) uses
        ``2 ** (ceil(log2(n)) + 2)``, matching the manual workflow's
        default. Pass an explicit integer to override.

    magnitude_mode : bool, optional
        If True (default), take the magnitude of the FFT (matching
        the manual workflow's default absolute-value display) --
        ``imag_spectrum`` is then all-zero. If False, apply the
        ``phase0``/``phase1`` correction instead and return the
        (still generally imperfectly phased) real/imaginary parts
        directly.

    phase0, phase1 : float, optional
        Zero- and first-order phase correction in degrees, only
        applied when ``magnitude_mode=False``. Default 0.

    f1p, f2p : float, optional
        Limits of the spectral region to keep, in ppm, applied after
        the FFT exactly as in :func:`NMR1dimport`. If both are zero
        (default), the complete spectrum is kept.

    Returns
    -------
    results : dict
        Same core keys as :func:`NMR1dimport` (``x_ppm``, ``x_Hz``,
        ``real_spectrum``, ``imag_spectrum``, their normalised
        counterparts, ``nuclide``, ``BF1``, ``SFO1``, ``path``), so
        the output can be handed to the same downstream tools (e.g.
        :func:`nmrtools.fitting.fit_spectrum`). Plus CPMG-specific
        diagnostics:

        ``num_echoes``, ``num_echoes_used``  : echoes acquired vs. co-added.
        ``first_point``                      : first-echo index actually used.
        ``echoes_real``, ``echoes_imag``     : each trimmed echo, individually
            (list of arrays, one per *used* echo) -- for inspecting the
            echo train, e.g. ``plt.plot(results["echoes_real"][0])``.
        ``fid_coadded``                      : co-added echo before apodization.
        ``fid_windowed``                     : co-added echo after apodization
            (identical to ``fid_coadded`` if ``apodize=False``).
        ``window``                           : the apodization window applied
            (all-ones if ``apodize=False``).

    Raises
    ------
    ValueError
        If the raw FID is entirely zero (``first_point='auto'`` could
        not locate an echo), or if the requested acqus array indices
        are out of range for this experiment.

    Notes
    -----
    No digital-filter (group-delay) correction is applied to the raw
    FID -- see :func:`nmrtools.utils.read_bruker_fid`. The ppm axis is
    ``(offset + O1 - SR) / SF``, the same referencing as
    :mod:`opennmr.io.bruker`. Each echo window starts at its own exact
    position (see :func:`cpmg_echo_layout`), so a period that is not a
    whole number of points does not drift along the train.

    Examples
    --------
    >>> results = CPMGprocess(r"C:\\data\\exp11")
    >>> results = CPMGprocess(r"C:\\data\\exp11", cutout=2, num_echoes_added=200)
    >>> results = CPMGprocess(r"C:\\data\\exp11", first_point=1050)  # override auto-detect
    """

    layout = cpmg_echo_layout(
        data_path, fid_name=fid_name, echo_delay_index=echo_delay_index,
        blanking_delay_index=blanking_delay_index, blanking_cnst_index=blanking_cnst_index,
        num_echoes_index=num_echoes_index, first_point=first_point,
        first_point_offset=first_point_offset, safety=safety)
    meta, fid = layout["meta"], layout["fid"]
    num_echoes, period, trim = layout["num_echoes"], layout["period"], layout["trim"]
    first = layout["first_point"]
    dw = layout["dwell_us"] / 2

    echoes = [fid[start + trim: start + period - trim] for start in layout["starts"]]
    echoes_real = [e.real for e in echoes]
    echoes_imag = [e.imag for e in echoes]

    cutout = int(cutout)
    if not 0 <= cutout < num_echoes:
        raise ValueError(f"cutout = {cutout}, but only {num_echoes} echoes were acquired.")
    # num_echoes_added counts from cutout, as documented above (it used
    # to be taken as the index to stop at, so cutout=2,
    # num_echoes_added=200 co-added 198 echoes, not 200).
    stop = num_echoes if num_echoes_added is None else min(num_echoes, cutout + int(num_echoes_added))
    if stop <= cutout:
        raise ValueError(f"num_echoes_added = {num_echoes_added}: no echoes to co-add.")
    sl = slice(cutout, stop)
    real_added = np.sum(echoes_real[sl], axis=0)
    imag_added = np.sum(echoes_imag[sl], axis=0)

    fid_coadded = real_added + 1j * imag_added

    if apodize:
        window = _cpmg_apodization_window(
            len(real_added), offset_zero=apod_offset_zero,
            shape=apod_shape, alpha=apod_alpha,
        )
    else:
        window = np.ones(len(real_added))

    fid_windowed = real_added * window + 1j * imag_added * window

    si = (
        int(2 ** (np.ceil(np.log2(len(fid_windowed))) + 2))
        if zero_fill == 'auto' else int(zero_fill)
    )

    spectrum = np.fft.fftshift(np.fft.fft(fid_windowed, n=si))

    # np.fft's own bin ordering (via fftfreq) is ascending in frequency;
    # build the axis directly from it -- rather than hand-rolling a
    # linspace, which is easy to get silently backwards -- then flip
    # spectrum and axis together to match nmrtools' descending
    # ppm/Hz convention (as in NMR1dimport).
    dt = 2 * dw * 1e-6  # seconds between successive points of fid_windowed
    # Referenced like every other Bruker spectrum (io.bruker's
    # _ppm_hz_axis): offset from the carrier, minus SR, over SF (BF1
    # when there is no procs). Dividing by SFO1 instead, as this did
    # before, is wrong by the fraction O1/BF1 -- 1 ppm at 1000 ppm for
    # a 100 kHz offset at 100 MHz, and it ignored a TopSpin SR.
    sf, sr = _cpmg_reference(data_path, procno, meta)
    hz_axis = np.fft.fftshift(np.fft.fftfreq(si, d=dt)) + meta["O1"] - sr
    ppm_axis = hz_axis / sf

    spectrum = spectrum[::-1]
    hz_axis = hz_axis[::-1]
    ppm_axis = ppm_axis[::-1]

    if magnitude_mode:
        real_spectrum = np.abs(spectrum)
        imag_spectrum = np.zeros_like(real_spectrum)
    else:
        spectrum = _cpmg_phase_correction(spectrum, phase0=phase0, phase1=phase1)
        real_spectrum = spectrum.real
        imag_spectrum = spectrum.imag

    if f1p != 0 or f2p != 0:
        xlowi = np.argmax(ppm_axis < f1p)
        xhighi = np.argmax(ppm_axis < f2p)
        if xlowi > xhighi:
            xlowi, xhighi = xhighi, xlowi
        ppm_axis = ppm_axis[xlowi:xhighi]
        hz_axis = hz_axis[xlowi:xhighi]
        real_spectrum = real_spectrum[xlowi:xhighi]
        imag_spectrum = imag_spectrum[xlowi:xhighi]

    real_spectrum_norm = (
        (real_spectrum - real_spectrum.min())
        / (real_spectrum.max() - real_spectrum.min())
    )
    real_spectrum = real_spectrum / mass

    if imag_spectrum.max() != imag_spectrum.min():
        imag_spectrum_norm = (
            (imag_spectrum - imag_spectrum.min())
            / (imag_spectrum.max() - imag_spectrum.min())
        )
    else:
        imag_spectrum_norm = np.zeros_like(imag_spectrum)
    imag_spectrum = imag_spectrum / mass

    return {
        'x_ppm': ppm_axis, 'x_Hz': hz_axis,
        'real_spectrum': real_spectrum, 'normalised_real_spectrum': real_spectrum_norm,
        'imag_spectrum': imag_spectrum, 'normalised_imag_spectrum': imag_spectrum_norm,
        'nuclide': meta["NUC"], 'BF1': meta["OBS"], 'SFO1': meta["SFO1"], 'path': data_path,
        'num_echoes': num_echoes, 'num_echoes_used': len(echoes_real[sl]),
        'first_point': first, 'SF': sf, 'SR': sr, 'O1': meta["O1"], 'dwell_us': layout["dwell_us"],
        'period_exact': layout["period_exact"], 'period': period, 'echo_starts': layout["starts"],
        'cutout': cutout,
        'echoes_real': echoes_real[sl], 'echoes_imag': echoes_imag[sl],
        'fid_coadded': fid_coadded, 'fid_windowed': fid_windowed, 'window': window,
    }


def cpmg_to_spectrum(results, procno=1):
    """
    A :func:`CPMGprocess` result as an :class:`opennmr.core.spectrum.NMRSpectrum`
    -- what the GUI adds to Open experiments, and what the rest of
    openNMR (phasing, fitting, VOCS, Compare) works on.

    The experiment's ``acqus`` travels with it (``metadata.raw["acqus"]``),
    so pulse lengths can still be looked up by name, e.g. for VOCS on a
    set of CPMG pieces. A magnitude-mode result has no imaginary part.
    ``metadata.raw["cpmg"]`` records the echo split that produced it.
    """
    from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    from ..io.jcampdx import parse_jcampdx_params

    path = results["path"]
    acqus_path = os.path.join(path, "acqus")
    acqus = parse_jcampdx_params(acqus_path) if os.path.isfile(acqus_path) else {}
    magnitude = not np.any(results["imag_spectrum"])
    record = {key: results[key] for key in (
        "num_echoes", "num_echoes_used", "cutout", "first_point", "period", "period_exact",
        "dwell_us", "SF", "SR")}
    record["magnitude"] = bool(magnitude)
    record["co_added_points"] = int(len(results["fid_coadded"]))
    metadata = NMRMetadata(
        vendor="bruker", nucleus=results["nuclide"], sfo1=float(results["SF"]),
        bf1=float(results["BF1"]), o1=float(results["O1"]), source_path=path, procno=procno,
        pulse_program=acqus.get("PULPROG"), probe=acqus.get("PROBHD"),
        raw={"acqus": acqus, "cpmg": record},
    )
    axis = NMRAxis(kind="frequency", ppm=np.asarray(results["x_ppm"], dtype=float),
                   hz=np.asarray(results["x_Hz"], dtype=float))
    return NMRSpectrum(
        data=np.asarray(results["real_spectrum"], dtype=float), axes=[axis], metadata=metadata,
        data_imag=None if magnitude else np.asarray(results["imag_spectrum"], dtype=float))


def CPMGplot(results, colour='black', xlims='default', xlabel=None,
             pl_width=3.375, pl_height=3.375, normalise=True,
             font='Arial', fontsize=12, x_unit='ppm', output_format='',
             tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
             tick_number=None, save_fig=False, fig_name='cpmg_plot', save_format='png',
             ax=None, fig=None):
    """
    Plot a spectrum already processed by :func:`CPMGprocess`.

    A thin, single-spectrum counterpart to :func:`NMR1dplot` that
    plots an existing results dict directly rather than re-importing
    from a Bruker directory, using the same figure/axis styling
    conventions (:func:`nmrtools.plotting.setup_plotting`,
    :func:`nmrtools.plotting.apply_tick_style`,
    :func:`nmrtools.plotting.apply_spine_style`).

    Parameters
    ----------
    results : dict
        Output of :func:`CPMGprocess`.

    colour : optional
        Line colour. Default ``'black'``.

    xlims : 'default' or [left, right], optional
        X-axis display limits; ``'default'`` uses the full processed
        range.

    xlabel : str, optional
        Override the automatically-generated (nuclide-based) x-axis
        label.

    pl_width, pl_height : float, optional
        Figure size in inches. Ignored if ``ax`` is given.

    normalise : bool, optional
        If True (default), plot the normalised spectrum.

    font, fontsize : optional
        Typeface settings, see :func:`nmrtools.plotting.setup_plotting`.

    x_unit : {'ppm', 'Hz'}, optional
        Unit used for the x-axis.

    output_format : {'', 'manuscript', 'presentation'}, optional
        Predefined figure formatting, see
        :func:`nmrtools.plotting.setup_plotting`.

    tick_dir, tick_length, tick_width, tick_pad, tick_number : optional
        See :func:`NMR1dplot`.

    save_fig, fig_name, save_format : optional
        See :func:`NMR1dplot`. Ignored if ``ax`` is given.

    ax : matplotlib.axes.Axes, optional
        If given, plot into this axis instead of creating a new
        figure.

    fig : matplotlib.figure.Figure, optional
        The figure ``ax`` belongs to. If ``ax`` is given and ``fig``
        is not, it is inferred from ``ax.figure``.

    Returns
    -------
    fig, ax
    """

    standalone = ax is None

    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style['pl_width'], style['pl_height']
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    x = results['x_ppm'] if x_unit == 'ppm' else results['x_Hz']
    y = results['normalised_real_spectrum'] if normalise else results['real_spectrum']

    ax.plot(x, y, color=colour, lw=1.0)

    nucgamma = find_gamma(results['nuclide'], SFO1=results['SFO1'])
    labeltext = NUC_label(nucgamma['Nuclide'], x_unit=x_unit)
    ax.set_xlabel(xlabel if xlabel is not None else labeltext, fontsize=fontsize)

    if xlims == 'default':
        ax.set_xlim(x[0], x[-1])
    else:
        ax.set_xlim(min(xlims), max(xlims))

    ax.set_yticks([])

    apply_spine_style(ax, top=False, right=False, bottom=True, left=False)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis='x', minor=True)
    ax.tick_params(axis='x', which='major', labelsize=fontsize)

    if standalone:
        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.tight_layout()

        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax
