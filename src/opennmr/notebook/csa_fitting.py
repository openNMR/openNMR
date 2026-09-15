"""
Fit static or finite-speed-MAS CSA powder lineshapes to experimental
1D NMR spectra, using the lineshape simulator in
:mod:`nmrtools.csa_quad_simulator` as the fit model and the shared
baseline/region-masking machinery from :mod:`nmrtools.fitting`.

Workflow
--------

    NMR1dimport -> fit_csa_spectrum -> NMR1dplot_with_csa_fit
                                     -> print_csa_fit_report

or, end-to-end, :func:`NMRCSAFitWorkflow`.

Unlike the Gaussian/Lorentzian/pseudo-Voigt models in
:mod:`nmrtools.fitting`, which are closed-form functions that can be
evaluated pointwise at any x, a CSA powder lineshape is only defined
here via a powder average that gets histogrammed onto a uniform ppm
axis (see :func:`nmrtools.csa_quad_simulator.static_csa`). So instead
of evaluating the model only at the fitted x-points, every fit
iteration below simulates the full spectrum on the (ascending-sorted)
experimental ppm axis and then selects the requested ``fit_regions``
subset -- see :func:`csa_model_full` and :func:`fit_csa_spectrum`. The
powder-average loop over crystallite orientations, not the number of
ppm points, dominates the simulation cost, so this is cheap relative
to a finer ppm grid but NOT cheap relative to ``n_theta * n_phi`` (or,
for the finite-MAS model, ``n_theta * n_phi * n_time``): keep those
coarse while fitting and only refine them for the final plot/report.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .utils import find_gamma, NUC_label
from .plotting import setup_plotting, apply_tick_style, apply_spine_style, save_figure
from .one_d import NMR1dimport
from .fitting import build_region_mask, estimate_baseline_robust, fit_quality, expand_pcov
from .simulations import csa_definition, make_static_csa_spectrum, finite_mas_csa_spectrum

_CSA_PARAM_NAMES = ["d_iso", "span_or_Delta", "skew_or_eta", "fwhm", "amplitude"]


# ============================================================
# Model
# ============================================================

def csa_model_full(x_ppm_sorted, d_iso, span_or_Delta, skew_or_eta, fwhm,
                    convention="Haeberlen", static=True,
                    larmor_Hz=None, mas_Hz=None, rotor_angle="magic",
                    n_theta=60, n_phi=120, use_octant_symmetry=True,
                    n_time=1024, n_sidebands=15, subtract_residual_mean=True,
                    powder_scheme='rectangular', n_zcw=None):
    """
    Simulate a unit-amplitude, normalised CSA lineshape on
    ``x_ppm_sorted`` (must be ascending and evenly spaced -- the axis
    convention used by
    :func:`nmrtools.csa_quad_simulator.make_static_csa_spectrum` and
    :func:`nmrtools.csa_quad_simulator.finite_mas_csa_spectrum`).

    Parameters
    ----------
    static : bool, optional
        If True (default), simulate a static powder pattern via
        :func:`nmrtools.csa_quad_simulator.make_static_csa_spectrum`.
        If False, simulate a finite-speed-MAS sideband pattern via
        :func:`nmrtools.csa_quad_simulator.finite_mas_csa_spectrum`,
        which requires ``larmor_Hz`` and ``mas_Hz``.
    n_theta, n_phi, use_octant_symmetry : optional
        Powder-grid size for the static case. Only used when
        ``powder_scheme='rectangular'``.
    n_theta, n_phi, n_time, n_sidebands, rotor_angle,
    subtract_residual_mean : optional
        Powder-grid/FFT size and gamma-averaging options for the
        finite-MAS case. See
        :func:`nmrtools.csa_quad_simulator.finite_mas_csa_spectrum`.
    powder_scheme : {'rectangular', 'zcw'}, optional
        Powder-averaging scheme, forwarded to the simulator (see
        :func:`nmrtools.csa_quad_simulator.get_powder_grid`). 'zcw'
        typically converges with far fewer orientations for the
        finite-MAS case (each orientation costs an FFT, and this
        function is called ~100x per fit via curve_fit's Jacobian) --
        it's a smaller win for the static case, where
        ``use_octant_symmetry`` already gives an exact 8x reduction.
    n_zcw : int, optional
        Minimum ZCW orientation count; only used when
        ``powder_scheme='zcw'``. Defaults to ``n_theta * n_phi``.

    Returns
    -------
    ndarray
        Normalised lineshape (max absolute value 1), same shape as
        ``x_ppm_sorted``.
    """
    csa = csa_definition(d_iso, span_or_Delta, skew_or_eta, convention=convention)

    if static:
        spec = make_static_csa_spectrum(
            csa, x_ppm_sorted, fwhm=fwhm, n_theta=n_theta, n_phi=n_phi,
            use_octant_symmetry=use_octant_symmetry,
            powder_scheme=powder_scheme, n_zcw=n_zcw,
        )
    else:
        if larmor_Hz is None or mas_Hz is None:
            raise ValueError("larmor_Hz and mas_Hz are required when static=False.")
        spec = finite_mas_csa_spectrum(
            csa, x_ppm_sorted, larmor_Hz=larmor_Hz, mas_Hz=mas_Hz,
            rotor_angle=rotor_angle, n_theta=n_theta, n_phi=n_phi,
            n_time=n_time, n_sidebands=n_sidebands, fwhm=fwhm,
            subtract_residual_mean=subtract_residual_mean,
            powder_scheme=powder_scheme, n_zcw=n_zcw,
        )

    return spec["normalised_real_spec"]


def build_csa_model(x_ppm_sorted, params, x_ref=None, **sim_kwargs):
    """
    Full model: ``amplitude * csa_model_full(...) + b0 + b1*(x - x_ref)``.

    Parameters
    ----------
    params : ndarray
        Ordered ``(d_iso, span_or_Delta, skew_or_eta, fwhm, amplitude,
        b0, b1)``.
    sim_kwargs :
        Forwarded to :func:`csa_model_full` (``convention``,
        ``static``, ``larmor_Hz``, ``mas_Hz``, grid sizes, etc).
    """
    d_iso, span_or_Delta, skew_or_eta, fwhm, amplitude, b0, b1 = params

    shape = csa_model_full(x_ppm_sorted, d_iso, span_or_Delta, skew_or_eta,
                            fwhm, **sim_kwargs)

    x_ref = np.mean(x_ppm_sorted) if x_ref is None else x_ref
    return amplitude * shape + b0 + b1 * (np.asarray(x_ppm_sorted) - x_ref)


# ============================================================
# Model construction (initial parameters, bounds)
# ============================================================

def initialise_csa_params(p0, b0_init, b1_init, x=None, y=None, fixed=None):
    """
    Build the initial parameter vector and free-parameter mask for a
    CSA fit.

    Parameters
    ----------
    p0 : dict
        ``{"d_iso": ..., "span_or_Delta": ..., "skew_or_eta": ...,
        "fwhm": ..., ["amplitude": ...]}``. If ``"amplitude"`` is
        omitted, it is estimated as ``max(abs(y))`` over the fit
        region (``x``/``y`` must then be supplied).
    fixed : list or dict, optional
        Names from
        ``["d_iso","span_or_Delta","skew_or_eta","fwhm","amplitude"]``
        to hold fixed at their ``p0`` value.

    Returns
    -------
    params0 : ndarray
    is_free : ndarray(bool)
    """
    if "amplitude" in p0:
        amplitude = p0["amplitude"]
    else:
        if x is None or y is None:
            raise ValueError(
                'p0["amplitude"] must be given, or x and y supplied to estimate it.'
            )
        amplitude = np.max(np.abs(y))

    values = [p0["d_iso"], p0["span_or_Delta"], p0["skew_or_eta"], p0["fwhm"], amplitude]

    if fixed is None:
        fixed = []
    if isinstance(fixed, dict):
        fixed = [name for name, is_fixed in fixed.items() if is_fixed]

    is_free = [name not in fixed for name in _CSA_PARAM_NAMES]

    values.extend([b0_init, b1_init])
    is_free.extend([True, True])  # flipped to fixed by fit_csa_spectrum, cf. fit_spectrum

    return np.array(values, dtype=float), np.array(is_free, dtype=bool)


def build_csa_bounds(convention="Haeberlen", bounds=None,
                      positive_intensity=True, allow_negative_intensity=False):
    """
    Construct lower/upper bounds for
    ``(d_iso, span_or_Delta, skew_or_eta, fwhm, amplitude)``.

    Defaults: ``d_iso`` and ``span_or_Delta`` unbounded (Haeberlen's
    ``Delta`` can be signed; pass ``bounds={"span_or_Delta": (0,
    np.inf)}`` for a Herzfeld-Berger ``span``, which is non-negative
    by definition). ``skew_or_eta`` defaults to (0, 1) for Haeberlen
    ``eta`` or (-1, 1) for Herzfeld-Berger ``skew``. ``fwhm`` defaults
    to (1e-6, inf). ``amplitude`` defaults to (0, inf) unless negative
    intensities are allowed (mirrors
    :func:`nmrtools.fitting.build_bounds`).

    Parameters
    ----------
    bounds : dict, optional
        Per-parameter ``(lower, upper)`` overrides, keyed by name from
        ``["d_iso","span_or_Delta","skew_or_eta","fwhm","amplitude"]``.
    """
    bounds = bounds or {}

    default_skew_or_eta = (0.0, 1.0) if convention == "Haeberlen" else (-1.0, 1.0)
    amp_lower = 0.0 if (positive_intensity and not allow_negative_intensity) else -np.inf

    defaults = {
        "d_iso": (-np.inf, np.inf),
        "span_or_Delta": (-np.inf, np.inf),
        "skew_or_eta": default_skew_or_eta,
        "fwhm": (1e-6, np.inf),
        "amplitude": (amp_lower, np.inf),
    }

    lower, upper = [], []
    for name in _CSA_PARAM_NAMES:
        lo, hi = bounds.get(name, defaults[name])
        if lo > hi:
            print(f"Warning: swapped bounds for {name}: {(lo, hi)} -> fixing order")
            lo, hi = hi, lo
        lower.append(lo)
        upper.append(hi)

    return np.array(lower), np.array(upper)


# ============================================================
# Fitting
# ============================================================

def fit_csa_spectrum(results, p0, convention="Haeberlen", static=True,
                      larmor_Hz=None, mas_Hz=None, rotor_angle="magic",
                      n_theta=60, n_phi=120, use_octant_symmetry=True,
                      n_time=1024, n_sidebands=15, subtract_residual_mean=True,
                      powder_scheme='rectangular', n_zcw=None,
                      fit_regions=None, use_normalised=True,
                      baseline_regions=None, baseline_sigma=2.5, baseline_max_iter=10,
                      bounds=None, fixed=None,
                      positive_intensity=True, allow_negative_intensity=False):
    """
    Fit a static or finite-MAS CSA powder lineshape to an experimental
    1D spectrum by nonlinear least squares.

    Mirrors :func:`nmrtools.fitting.fit_spectrum`'s workflow (a robust
    linear baseline is estimated and held fixed, then ``curve_fit``
    optimises the remaining parameters), but the model is the CSA
    powder simulation in :mod:`nmrtools.csa_quad_simulator` rather
    than a closed-form peak shape -- see the module docstring for why
    that means the model is always evaluated on the full experimental
    ppm axis and then subset, rather than pointwise at arbitrary x.

    Parameters
    ----------
    results : dict
        Output of :func:`nmrtools.one_d.NMR1dimport`.
    p0 : dict
        Initial guess, see :func:`initialise_csa_params`.
    convention : {'Haeberlen', 'Herzfeld-Berger'}, optional
    static : bool, optional
        If False, fit a finite-speed-MAS sideband pattern instead of a
        static powder pattern (see :func:`csa_model_full`).
    larmor_Hz : float, optional
        Only used if ``static=False``. If None, taken from
        ``results['SFO1']`` (MHz) * 1e6.
    mas_Hz : float, optional
        Required if ``static=False`` -- the spinning speed is not
        recorded in ``results`` and must be supplied explicitly.
    rotor_angle, n_time, n_sidebands, subtract_residual_mean : optional
        Only used if ``static=False``. See
        :func:`nmrtools.csa_quad_simulator.finite_mas_csa_spectrum`.
    n_theta, n_phi, use_octant_symmetry : optional
        Powder-grid size. Keep coarse for fitting (defaults 60/120)
        and refine for the final plot -- the powder-average loop, not
        the ppm-axis length, dominates the cost of each ``curve_fit``
        iteration.
    powder_scheme, n_zcw : optional
        Powder-averaging scheme -- see :func:`csa_model_full`. Worth
        trying ``powder_scheme='zcw'`` for finite-MAS fits (``static=
        False``) since that's where each ``curve_fit`` iteration is
        most expensive; check the fit result is unchanged when you
        raise ``n_zcw`` before trusting a low value.
    fit_regions : list of tuple, optional
        ``[(xmin, xmax), ...]`` in ppm. If None, the complete spectrum
        is fitted.
    use_normalised : bool, optional
        Fit the spectrum normalised to its maximum absolute intensity
        (recommended -- keeps parameter scales close to 1 for the
        optimiser). Amplitude is rescaled back afterwards.
    baseline_regions, baseline_sigma, baseline_max_iter :
        See :func:`nmrtools.fitting.estimate_baseline_robust`.
    bounds : dict, optional
        See :func:`build_csa_bounds`.
    fixed : list or dict, optional
        CSA/lineshape parameter names to hold fixed at their ``p0``
        value (the baseline is always held fixed, matching
        :func:`nmrtools.fitting.fit_spectrum`).
    positive_intensity, allow_negative_intensity : bool, optional
        See :func:`nmrtools.fitting.fit_spectrum`.

    Returns
    -------
    popt : ndarray
        ``(d_iso, span_or_Delta, skew_or_eta, fwhm, amplitude, b0, b1)``.
    pcov_free : ndarray
        Covariance of the free parameters only -- expand with
        :func:`nmrtools.fitting.expand_pcov`.
        CAVEAT: unlike ``popt``, this is NOT rescaled back to raw
        units when ``use_normalised=True`` (same caveat as
        :func:`nmrtools.fitting.fit_spectrum`'s ``pcov_free`` --
        flagged there in more detail). Uncertainties derived from it
        (``check_csa_bound_hits``, ``print_csa_fit_report``) are
        systematically too small by roughly a factor of ``scale``
        whenever ``use_normalised=True``.
    is_free : ndarray(bool)
    scale : float
    x_ref : float
    sim_kwargs : dict
        Simulation settings used (``convention``, ``static``,
        ``larmor_Hz``, ``mas_Hz``, grid sizes, ...) -- pass this on to
        :func:`evaluate_csa_fit` / :func:`NMR1dplot_with_csa_fit` so
        the fitted model is re-evaluated consistently.
    """

    x_full = results["x_ppm"]

    if use_normalised:
        y_raw = results["real_spectrum"]
        scale = np.max(np.abs(y_raw))
        scale = scale if scale != 0 else 1.0
        y_full = y_raw / scale
    else:
        y_full = results["real_spectrum"]
        scale = 1.0

    # The CSA simulator bins its powder average onto a uniform,
    # ascending ppm axis (see the module docstring), so sort once here
    # and mask/fit on the sorted axis throughout.
    order = np.argsort(x_full)
    x_sorted = x_full[order]
    y_sorted = y_full[order]

    mask = build_region_mask(x_sorted, fit_regions)
    x_fit = x_sorted[mask]
    y_fit = y_sorted[mask]

    x_ref = np.mean(x_fit)

    b0_init, b1_init, _ = estimate_baseline_robust(
        x_sorted, y_sorted, baseline_regions=baseline_regions,
        sigma=baseline_sigma, max_iter=baseline_max_iter, x_ref=x_ref,
    )

    p0_arr, is_free = initialise_csa_params(
        p0, b0_init, b1_init, x=x_fit, y=y_fit, fixed=fixed,
    )
    is_free[-2:] = False  # baseline held fixed, as in fit_spectrum

    if not static and larmor_Hz is None:
        larmor_Hz = results["SFO1"] * 1.0e6

    sim_kwargs = dict(
        convention=convention, static=static, larmor_Hz=larmor_Hz, mas_Hz=mas_Hz,
        rotor_angle=rotor_angle, n_theta=n_theta, n_phi=n_phi,
        use_octant_symmetry=use_octant_symmetry, n_time=n_time,
        n_sidebands=n_sidebands, subtract_residual_mean=subtract_residual_mean,
        powder_scheme=powder_scheme, n_zcw=n_zcw,
    )

    lower, upper = build_csa_bounds(
        convention=convention, bounds=bounds,
        positive_intensity=positive_intensity,
        allow_negative_intensity=allow_negative_intensity,
    )
    lower = np.append(lower, [-np.inf, -np.inf])
    upper = np.append(upper, [np.inf, np.inf])

    assert len(p0_arr) == len(is_free) == len(lower) == len(upper), "parameter length mismatch"

    for i, (val, lo, hi) in enumerate(zip(p0_arr[is_free], lower[is_free], upper[is_free])):
        if val < lo or val > hi:
            print("BAD INITIAL PARAMETER", i, "value=", val, "bounds=", lo, hi)

    def model_wrapper(x, *free_params):
        full = p0_arr.copy()
        full[is_free] = free_params
        # `x` here is always x_fit (== x_sorted[mask]), as passed by
        # curve_fit -- the model is simulated on the full sorted axis
        # and then masked, so the argument itself is not reused below.
        y_model_full = build_csa_model(x_sorted, full, x_ref=x_ref, **sim_kwargs)
        return y_model_full[mask]

    popt_free, pcov_free = curve_fit(
        model_wrapper, x_fit, y_fit, p0=p0_arr[is_free],
        bounds=(lower[is_free], upper[is_free]), maxfev=20000,
    )

    popt = p0_arr.copy()
    popt[is_free] = popt_free

    if use_normalised:
        popt = popt.copy()
        # Bugfix: previously only the amplitude (index 4) was rescaled
        # back to raw intensity units here; the baseline (b0, b1,
        # indices 5-6 -- also intensity-axis quantities) was left in
        # curve_fit's internal normalised scale, silently mixing units
        # within a single popt vector whenever use_normalised=True
        # (raw-scale amplitude, normalised-scale baseline). Same bug,
        # same fix, as nmrtools.fitting.fit_spectrum's rescale_params.
        popt[4] *= scale        # amplitude
        popt[5:7] *= scale      # baseline b0, b1

    return popt, pcov_free, is_free, scale, x_ref, sim_kwargs


def evaluate_csa_fit(x_ppm, popt, x_ref=None, return_components=False, **sim_kwargs):
    """
    Evaluate the fitted CSA model at ``x_ppm`` (any order/spacing -- it
    is sorted internally and un-sorted back to match the input order).

    Parameters
    ----------
    popt : ndarray
        As returned by :func:`fit_csa_spectrum`.
    sim_kwargs : dict
        As returned by :func:`fit_csa_spectrum` (its ``sim_kwargs``).
    return_components : bool, optional
        If True, also return ``[csa_component, baseline]``.
    """
    x_ppm = np.asarray(x_ppm)
    order = np.argsort(x_ppm)
    x_sorted = x_ppm[order]

    x_ref = np.mean(x_sorted) if x_ref is None else x_ref

    shape = csa_model_full(x_sorted, *popt[:4], **sim_kwargs)
    csa_component_sorted = popt[4] * shape
    baseline_sorted = popt[5] + popt[6] * (x_sorted - x_ref)
    y_sorted = csa_component_sorted + baseline_sorted

    y = np.empty_like(y_sorted)
    y[order] = y_sorted

    if not return_components:
        return y

    csa_component = np.empty_like(y_sorted)
    csa_component[order] = csa_component_sorted
    baseline = np.empty_like(y_sorted)
    baseline[order] = baseline_sorted

    return y, [csa_component, baseline]


def check_csa_bound_hits(popt, pcov, is_free, lower, upper):
    """Flag any free CSA/lineshape parameter whose fitted value (plus
    or minus its uncertainty) reaches a bound -- a sign the bound may
    be too tight."""
    warnings = []
    for i, name in enumerate(_CSA_PARAM_NAMES):
        if not is_free[i]:
            continue
        val, lo, hi = popt[i], lower[i], upper[i]
        err = np.sqrt(max(pcov[i, i], 0.0))
        if np.isfinite(hi) and val + err >= hi:
            warnings.append(f"{name} near upper bound {hi}")
        if np.isfinite(lo) and val - err <= lo:
            warnings.append(f"{name} near lower bound {lo}")
    return warnings


def print_csa_fit_report(x, y, y_fit, popt, pcov, is_free, convention="Haeberlen",
                          warnings=None, save_txt=False, txt_filename="csa_fit_report.txt"):
    """
    Print (and optionally save) a human-readable report of the fitted
    CSA parameters, uncertainties, and fit-quality metrics.

    Parameters
    ----------
    y_fit : ndarray
        Model evaluated at ``x`` -- see :func:`evaluate_csa_fit`.
    """
    errors = np.sqrt(np.diag(pcov))
    n_params = len(popt)
    metrics = fit_quality(y, y_fit, n_params)

    span_or_delta_label = "Delta" if convention == "Haeberlen" else "span"
    skew_or_eta_label = "eta" if convention == "Haeberlen" else "skew"
    names = ["d_iso", span_or_delta_label, skew_or_eta_label, "fwhm", "amplitude", "b0", "b1"]

    report = []

    def write(line=""):
        print(line)
        report.append(str(line))

    write("\n" + "=" * 60)
    write(f"CSA FIT REPORT ({convention} convention)")
    write("=" * 60)
    write("\nParameters:")
    write("-" * 60)

    for name, val, err, is_f in zip(names, popt, errors, is_free):
        if is_f:
            write(f"  {name:<12}= {val:.6f} \u00b1 {err:.6f}")
        else:
            write(f"  {name:<12}= {val:.6f} (fixed)")

    write("\nFit quality:")
    write("-" * 60)
    write(f"RMSE          : {metrics['rmse']:.6e}")
    write(f"Chi^2         : {metrics['chi2']:.6e}")
    write(f"Reduced Chi^2 : {metrics['reduced_chi2']:.6e}")
    write("=" * 60 + "\n")

    if warnings:
        write("=" * 60 + "\n")
        write("Warnings:")
        write("-" * 60)
        for warning in warnings:
            write(warning)
            write("----")
        write("=" * 60 + "\n")

    if save_txt:
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(report))


# ============================================================
# 1D plotting with fit overlay
# ============================================================

def NMR1dplot_with_csa_fit(results, popt, sim_kwargs, x_ref=None, fit_regions=None,
                            xlims='default', xlabel=None, pl_width=3.375, pl_height=3.375,
                            font='Arial', fontsize=12, output_format='',
                            show_components=False, colour_data='black', colour_fit='red',
                            colour_baseline='gray',
                            tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                            tick_number=None, field='', use_normalised=True,
                            save_fig=False, fig_name='nmr_csa_fit_plot', save_format='png'):
    """
    Plot a 1D spectrum together with its fitted CSA model.

    Parameters
    ----------
    results : dict
        Output of :func:`nmrtools.one_d.NMR1dimport`.
    popt, sim_kwargs, x_ref :
        As returned by :func:`fit_csa_spectrum`.
    fit_regions : list of tuple, optional
        Region(s) actually shown/compared, ``[(xmin, xmax), ...]``. If
        None, the complete spectrum is shown.
    xlims : 'default' or [left, right], optional
    xlabel : str, optional
        Override the automatically-generated (nuclide-based) x-axis
        label.
    pl_width, pl_height : float, optional
        Figure size in inches.
    font, fontsize : optional
        Typeface settings, see :func:`nmrtools.plotting.setup_plotting`.
    show_components : bool, optional
        If True, also draw the fixed linear baseline.
    use_normalised : bool, optional
        Must match the ``use_normalised`` value used when fitting, so
        data and model are plotted on the same intensity scale.
    tick_dir : {'in', 'out', 'inout'}, optional
    tick_length, tick_width, tick_pad : float, optional
    tick_number : int, optional
    save_fig : bool, optional
        If True, save the figure (see :func:`nmrtools.plotting.save_figure`).
    fig_name, save_format : optional

    Returns
    -------
    fig, ax
    """
    style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                            pl_height=pl_height, output_format=output_format)
    pl_width, pl_height = style['pl_width'], style['pl_height']

    x_full = results['x_ppm']
    order = np.argsort(x_full)
    x_sorted = x_full[order]

    if use_normalised:
        scale = np.max(np.abs(results['real_spectrum']))
        scale = scale if scale != 0 else 1.0
        y_sorted = (results['real_spectrum'] / scale)[order]
    else:
        scale = 1.0
        y_sorted = results['real_spectrum'][order]

    mask = build_region_mask(x_sorted, fit_regions) if fit_regions is not None \
        else np.ones_like(x_sorted, dtype=bool)
    x_fit = x_sorted[mask]
    y_data = y_sorted[mask]

    x_ref = np.mean(x_fit) if x_ref is None else x_ref

    # popt (from fit_csa_spectrum) is always in raw intensity units,
    # regardless of use_normalised -- amplitude and baseline are both
    # rescaled back after fitting (see fit_csa_spectrum). So the
    # evaluated model is in raw units too, and must be divided by the
    # same `scale` as the data above to land on the same axis.
    # Previously this division was missing here, so the fit line (and
    # the baseline component) were plotted at the wrong height/scale
    # relative to the data whenever use_normalised=True.
    if show_components:
        y_model_full, components = evaluate_csa_fit(
            x_sorted, popt, x_ref=x_ref, return_components=True, **sim_kwargs,
        )
        y_model = y_model_full[mask] / scale
        components = [c[mask] / scale for c in components]
    else:
        y_model = evaluate_csa_fit(x_sorted, popt, x_ref=x_ref, **sim_kwargs)[mask] / scale
        components = None

    xmin, xmax = x_fit[0], x_fit[-1]

    fig, ax = plt.subplots()

    ax.plot(x_fit, y_data, color=colour_data, lw=1.5, label='Data')
    ax.plot(x_fit, y_model, color=colour_fit, lw=2, label='Fit')

    if show_components:
        baseline = components[1]
        ax.plot(x_fit, baseline, color=colour_baseline, lw=1, alpha=0.6, label='Baseline')

    if xlims == 'default':
        ax.set_xlim(xmin, xmax)
    else:
        ax.set_xlim(xlims[0], xlims[1])

    ax.invert_xaxis()
    ax.set_yticks([])
    apply_spine_style(ax, top=False, right=False, bottom=True, left=False)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis='x')

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        NUC = results.get('nuclide', 'unknown')
        SFO1 = results.get('SFO1', 0)
        nucgamma = find_gamma(NUC, field=field, SFO1=SFO1)
        ax.set_xlabel(NUC_label(nucgamma['Nuclide'], x_unit='ppm'))

    ax.legend()
    fig.set_figheight(pl_height)
    fig.set_figwidth(pl_width)
    fig.tight_layout()

    if save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax


# ============================================================
# 1D workflow
# ============================================================

def NMRCSAFitWorkflow(datapath, p0, procno=1, convention="Haeberlen", static=True,
                       larmor_Hz=None, mas_Hz=None, rotor_angle="magic",
                       n_theta=60, n_phi=120, use_octant_symmetry=True,
                       n_time=1024, n_sidebands=15, subtract_residual_mean=True,
                       powder_scheme='rectangular', n_zcw=None,
                       fit_regions=None, baseline_regions=None,
                       baseline_sigma=2.5, baseline_max_iter=10, normalise=True,
                       bounds=None, fixed=None,
                       positive_intensity=True, allow_negative_intensity=False,
                       show_plot=True, xlims='default', xlabel=None,
                       pl_width=3.375, pl_height=3.375, font='Arial', fontsize=12,
                       output_format='', tick_dir='in', tick_length=4.0,
                       tick_width=1.0, tick_pad=4.0, tick_number=None,
                       show_components=False, show_report=True,
                       save_csv=False, csv_filename="csa_fit_results.csv",
                       save_report=False, report_filename="csa_fit_report.txt",
                       save_fig=False, fig_name='nmr_csa_fit_plot', save_format='png'):
    """
    Fit a static or finite-MAS CSA powder lineshape to a processed 1D
    Bruker spectrum.

    The CSA analogue of :func:`nmrtools.fitting.NMRFitWorkflow`:
    imports the spectrum, fits the CSA tensor parameters (plus a
    linewidth, amplitude, and fixed linear baseline) by nonlinear
    least squares against the powder simulation in
    :mod:`nmrtools.csa_quad_simulator`, and optionally plots and
    reports the result.

    Parameters
    ----------
    datapath : str
        Path to the Bruker experiment directory.
    p0 : dict
        Initial guess -- see :func:`initialise_csa_params`.
    procno : int, optional
    convention, static, larmor_Hz, mas_Hz, rotor_angle, n_theta, n_phi,
    use_octant_symmetry, n_time, n_sidebands, subtract_residual_mean,
    powder_scheme, n_zcw,
    fit_regions, baseline_regions, baseline_sigma, baseline_max_iter,
    bounds, fixed, positive_intensity, allow_negative_intensity :
        See :func:`fit_csa_spectrum`.
    normalise : bool, optional
        Fit (and plot) the spectrum normalised to its maximum absolute
        intensity.
    show_plot, show_components, show_report : bool, optional
    xlims, xlabel, pl_width, pl_height, font, fontsize, output_format,
    tick_dir, tick_length, tick_width, tick_pad, tick_number :
        Forwarded to :func:`NMR1dplot_with_csa_fit`.
    save_csv, csv_filename, save_report, report_filename : optional
    save_fig, fig_name, save_format : optional

    Returns
    -------
    dict with keys ``popt``, ``pcov`` (full, expanded), ``sim_kwargs``,
    ``x``, ``y``, ``y_fit``, ``components``, ``figure``, ``axes``.

    Notes
    -----
    Parameters are ordered ``d_iso, span_or_Delta, skew_or_eta, fwhm,
    amplitude``, followed by baseline ``b0, b1``. The fitted model is
    ``amplitude * csa_lineshape + b0 + b1 * (x - x_ref)``.
    """

    setup_plotting()

    results = NMR1dimport(datapath, procno=procno)

    popt, pcov_free, is_free, scale, x_ref, sim_kwargs = fit_csa_spectrum(
        results, p0, convention=convention, static=static,
        larmor_Hz=larmor_Hz, mas_Hz=mas_Hz, rotor_angle=rotor_angle,
        n_theta=n_theta, n_phi=n_phi, use_octant_symmetry=use_octant_symmetry,
        n_time=n_time, n_sidebands=n_sidebands,
        subtract_residual_mean=subtract_residual_mean,
        powder_scheme=powder_scheme, n_zcw=n_zcw,
        fit_regions=fit_regions, use_normalised=normalise,
        baseline_regions=baseline_regions, baseline_sigma=baseline_sigma,
        baseline_max_iter=baseline_max_iter, bounds=bounds, fixed=fixed,
        positive_intensity=positive_intensity,
        allow_negative_intensity=allow_negative_intensity,
    )

    pcov = expand_pcov(pcov_free, is_free, len(popt))

    lower, upper = build_csa_bounds(
        convention=convention, bounds=bounds,
        positive_intensity=positive_intensity,
        allow_negative_intensity=allow_negative_intensity,
    )
    lower = np.append(lower, [-np.inf, -np.inf])
    upper = np.append(upper, [np.inf, np.inf])
    warnings = check_csa_bound_hits(popt, pcov, is_free, lower, upper)

    x_full = results['x_ppm']
    order = np.argsort(x_full)
    x_sorted = x_full[order]

    # popt/y_model (below) are always raw intensity units regardless
    # of `normalise` -- fit_csa_spectrum only uses use_normalised as
    # an internal numerical-conditioning trick and rescales amplitude
    # and baseline back afterwards (see fit_csa_spectrum). So the data
    # used for the report and CSV export must also be raw units to
    # match popt/y_model, not divided by `scale` (that division was
    # previously applied here but never undone before comparing
    # against y_model, which stayed in raw/mixed units -- the same
    # bug as NMRFitWorkflow's, see nmrtools.fitting). `normalise`
    # still controls the *plotted* display scale, via use_normalised
    # in NMR1dplot_with_csa_fit below.
    y_sorted = results['real_spectrum'][order]

    mask = build_region_mask(x_sorted, fit_regions) if fit_regions is not None \
        else np.ones_like(x_sorted, dtype=bool)
    x_fit = x_sorted[mask]
    y_fit_data = y_sorted[mask]

    if show_components:
        y_model_full, components_full = evaluate_csa_fit(
            x_sorted, popt, x_ref=x_ref, return_components=True, **sim_kwargs,
        )
        y_model = y_model_full[mask]
        components = [c[mask] for c in components_full]
    else:
        y_model = evaluate_csa_fit(x_sorted, popt, x_ref=x_ref, **sim_kwargs)[mask]
        components = None

    fig, ax = None, None
    if show_plot:
        fig, ax = NMR1dplot_with_csa_fit(
            results=results, popt=popt, sim_kwargs=sim_kwargs, x_ref=x_ref,
            fit_regions=fit_regions, show_components=show_components,
            use_normalised=normalise, xlims=xlims, xlabel=xlabel,
            pl_width=pl_width, pl_height=pl_height, font=font, fontsize=fontsize,
            output_format=output_format, tick_dir=tick_dir, tick_length=tick_length,
            tick_width=tick_width, tick_pad=tick_pad, tick_number=tick_number,
        )
        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    if show_report:
        print_csa_fit_report(x_fit, y_fit_data, y_model, popt, pcov, is_free,
                              convention=convention, warnings=warnings,
                              save_txt=save_report, txt_filename=report_filename)

    if save_csv:
        import csv

        columns = {"x": x_fit, "y": y_fit_data, "y_fit": y_model}
        if components is not None:
            columns["csa_component"] = components[0]
            columns["baseline"] = components[1]

        with open(csv_filename, "w", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(columns.keys())
            for i in range(len(x_fit)):
                writer.writerow([columns[col][i] for col in columns])

    return {
        "popt": popt, "pcov": pcov, "sim_kwargs": sim_kwargs,
        "x": x_fit, "y": y_fit_data, "y_fit": y_model,
        "components": components, "figure": fig, "axes": ax,
    }
