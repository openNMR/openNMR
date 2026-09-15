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

from .utils import find_gamma, NUC_label, read_bruker_metadata
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

def NMR1dplot(data_paths, mass=[1], procno=[1], colour='black', f1p=0, f2p=0,
              xlims='default', xlabel=None, pl_width=3.375, pl_height=3.375,
              normalise=True, yoffset=0.2, font='Arial', fontsize=12, fname=True,
              x_unit='ppm', output_format='',
              vocs_sum=False, plot_vocs_parts=False, vocs_average=False, heteronuclear_2d_slice=False,
              vocs_pulse="P1", vocs_npoints=8192, vocs_colours=None, vocs_yoffset=0.2,
              tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0, tick_number=None,
              field='', save_fig=False, fig_name='nmr_1d_plot', save_format='png',
              heatmap=False, heatmap_cmap='viridis', heatmap_vmin=None, heatmap_vmax=None,
              heatmap_colorbar=False, heatmap_colorbar_label=None,
              ylabel=None, y_values=None, y_ticks='none', y_tick_number=5,
              common_range='intersection', heatmap_npoints=None, interp_kind='linear'):
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

    style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                            pl_height=pl_height, output_format=output_format)
    pl_width, pl_height = style['pl_width'], style['pl_height']

    values = range(len(data_paths))
    norm = plt.Normalize(vmin=values[0], vmax=values[-1])
    cmap = get_stack_colormap(colour)
    scalarMap = cmx.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots()

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
        xmin, xmax = xlims[0], xlims[1]
        ax.set_xlim(xmin, xmax)

    ax.set_xlabel(xlabel if xlabel is not None else labeltext)
    apply_spine_style(ax, top=False, right=False, bottom=True, left=False)
    ax.set_yticks([])
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis='x')

    ax.invert_xaxis()
    fig.set_figheight(pl_height)
    fig.set_figwidth(pl_width)
    fig.tight_layout()

    if save_fig:
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
                  field='', save_fig=False, fig_name='nmr_1d_heatmap', save_format='png'):
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

    fig, ax = plt.subplots()

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
        ax.set_xlim(xlims[0], xlims[1])

    ax.set_xlabel(xlabel if xlabel is not None else labeltext)
    ax.invert_xaxis()

    apply_spine_style(ax, top=False, right=False, bottom=True, left=True)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis='x', minor=True)

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
                        length=tick_length, width=tick_width, pad=tick_pad)
        ax.tick_params(axis='y', which='minor', length=0)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if colorbar:
        cbar = fig.colorbar(im, ax=ax)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)

    fig.set_figheight(pl_height)
    fig.set_figwidth(pl_width)
    fig.tight_layout()

    if save_fig:
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
