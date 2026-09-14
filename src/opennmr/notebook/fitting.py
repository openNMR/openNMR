"""
Peak fitting tools for 1D NMR spectra, and 2D slice-fit workflows built
on top of them.

Provides:

- Gaussian, Lorentzian and pseudo-Voigt peak models
- Robust linear baseline estimation
- Automatic construction of model functions and parameter bounds
- Non-linear least-squares fitting (``fit_spectrum``)
- Fit evaluation, integration, and reporting
- A single-spectrum 1D workflow (``NMRFitWorkflow``)
- A 2D slice-fitting workflow (``NMR2D_fit_slices``,
  ``NMR2D_plot_integrals``, ``NMR2D_workflow``)

Workflow (2D)
-------------

    NMR2dimport -> NMR2D_get_slices -> NMR2D_fit_slices -> NMR2D_plot_integrals

or, end-to-end, ``NMR2D_workflow``.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .utils import find_gamma, NUC_label, read_vdlist
from .plotting import (
    setup_plotting, apply_tick_style, apply_spine_style, get_stack_colormap,
    save_figure,
)
from .one_d import NMR1dimport
from .two_d import (
    NMR2dimport, NMR2d_get_trace, NMR2D_get_slices, prepare_trace_for_fit,
    NMR2D_plot_slices,
)

# numpy >= 2.0 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ============================================================
# Peak models
# ============================================================

def gaussian(x, x0, fwhm, A):
    """
    Evaluate a Gaussian peak.

    Parameters
    ----------
    x : ndarray
        x-axis values (ppm or Hz).
    x0 : float
        Peak centre.
    fwhm : float
        Full width at half maximum.
    A : float
        Peak height.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def lorentzian(x, x0, fwhm, A):
    """Evaluate a Lorentzian peak. See :func:`gaussian` for parameters."""
    gamma = fwhm / 2
    return A * (gamma ** 2 / ((x - x0) ** 2 + gamma ** 2))


def pseudo_voigt(x, x0, fwhm, A, eta):
    """
    Evaluate a pseudo-Voigt peak: ``eta * Lorentzian + (1 - eta) * Gaussian``.

    eta = 0 -> pure Gaussian, eta = 1 -> pure Lorentzian.
    """
    return eta * lorentzian(x, x0, fwhm, A) + (1 - eta) * gaussian(x, x0, fwhm, A)


# ============================================================
# Baseline estimation
# ============================================================

def estimate_baseline_from_edges(x, y, frac=0.05):
    """
    Simple linear baseline from the first/last ``frac`` of the
    spectrum. A lighter-weight alternative to
    :func:`estimate_baseline_robust` (which is what ``fit_spectrum``
    actually uses); kept as a standalone utility.
    """
    n = len(x)
    k = int(n * frac)

    x_noise = np.concatenate([x[:k], x[-k:]])
    y_noise = np.concatenate([y[:k], y[-k:]])

    A = np.vstack([np.ones_like(x_noise), x_noise]).T
    coeff, *_ = np.linalg.lstsq(A, y_noise, rcond=None)

    return coeff[0], coeff[1]


def estimate_baseline_robust(x, y, baseline_regions=None, frac=0.05, sigma=2.5,
                              max_iter=10, x_ref=None):
    """
    Robustly estimate a linear baseline, ``baseline = b0 + b1 * (x - x_ref)``.

    Parameters
    ----------
    x, y : ndarray
        Spectrum.

    baseline_regions : tuple or list of tuples, optional
        Explicit baseline-only regions, e.g. ``(-1, 0)`` or
        ``[(-1, 0), (9, 10)]``. If None, the first and last ``frac`` of
        the spectrum are used as candidate baseline regions.

    frac : float, optional
        Fraction of the spectrum used at each edge when
        ``baseline_regions`` is None.

    sigma : float, optional
        Sigma-clipping threshold.

    max_iter : int, optional
        Maximum number of clipping iterations.

    x_ref : float, optional
        Reference x-value. Defaults to the mean of the candidate
        baseline points.

    Returns
    -------
    b0, b1, x_ref : float
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if baseline_regions is not None:

        if isinstance(baseline_regions[0], (int, float, np.integer, np.floating)):
            baseline_regions = [baseline_regions]

        mask = np.zeros_like(x, dtype=bool)
        for lo, hi in baseline_regions:
            lo, hi = sorted([lo, hi])
            mask |= (x >= lo) & (x <= hi)

        x_base = x[mask]
        y_base = y[mask]

    else:

        n = len(x)
        k = max(1, int(n * frac))
        x_base = np.concatenate([x[:k], x[-k:]])
        y_base = np.concatenate([y[:k], y[-k:]])

    if len(x_base) < 3:
        raise ValueError("Not enough points available for baseline estimation.")

    if x_ref is None:
        x_ref = np.mean(x_base)

    xc = x_base - x_ref

    A = np.column_stack([np.ones_like(xc), xc])
    coeff, *_ = np.linalg.lstsq(A, y_base, rcond=None)

    keep = np.ones(len(y_base), dtype=bool)

    for _ in range(max_iter):

        coeff_old = coeff.copy()
        coeff, *_ = np.linalg.lstsq(A[keep], y_base[keep], rcond=None)

        residual = y_base - A @ coeff
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        noise = 1.4826 * mad

        if noise <= 0:
            break

        new_keep = np.abs(residual - median) <= sigma * noise

        if np.sum(new_keep) < 3:
            break

        keep = new_keep

        if np.allclose(coeff, coeff_old, rtol=1e-6, atol=1e-10):
            break

    return coeff[0], coeff[1], x_ref


# ============================================================
# Model construction
# ============================================================

def build_region_mask(x, fit_regions):
    """Build a boolean mask from fit regions ``[(xmin, xmax), ...]``."""
    if fit_regions is None:
        return np.ones_like(x, dtype=bool)

    mask = np.zeros_like(x, dtype=bool)
    for xmin, xmax in fit_regions:
        lo, hi = min(xmin, xmax), max(xmin, xmax)
        mask |= (x >= lo) & (x <= hi)

    return mask


def _peak_param_names(peak):
    return ["shift", "fwhm", "A"] if peak["type"] in ("gaussian", "lorentzian") \
        else ["shift", "fwhm", "A", "eta"]


def initialise_params_with_map(x, y, peak_defs, b0_init, b1_init,
                                positive_intensity=True, allow_negative_intensity=False):
    """
    Build the initial parameter vector ``p0`` and a boolean ``is_free``
    mask (peak parameters, in order, followed by baseline ``b0``, ``b1``).

    ``positive_intensity``/``allow_negative_intensity`` mirror
    :func:`build_bounds`'s own unlock condition (amplitude allowed
    negative whenever ``not positive_intensity or
    allow_negative_intensity``): when a peak's true starting amplitude
    is negative but the amplitude bound is locked to >= 0 anyway, the
    guess is floored to 0 (it has to be, to sit inside its own bound).
    But when negative amplitudes are actually allowed, flooring a
    genuinely negative starting value to 0 is just a worse initial
    guess with no purpose -- and for data that is legitimately
    negative-going at the fitted point (e.g. an inversion-recovery
    slice before the null point, or a difference spectrum), it can
    leave the optimiser stranded at a bad starting point.
    """

    amplitude_locked_positive = positive_intensity and not allow_negative_intensity

    p0 = []
    is_free = []

    for peak in peak_defs:

        x0 = peak['shift']
        fwhm = peak['fwhm']

        if "A_init" in peak:
            A = peak["A_init"]
        else:
            idx = np.argmin(np.abs(x - x0))
            A = y[idx]
            if A < 0 and amplitude_locked_positive:
                A = 0.0

        names = _peak_param_names(peak)

        for name in names:
            if name == "eta":
                val = peak.get("eta", 0.5)
            elif name == "A" and "A_init" in peak:
                val = peak["A_init"]
            else:
                val = {"shift": x0, "fwhm": fwhm, "A": A}[name]

            p0.append(val)

            fixed = peak.get("fixed", {})
            if isinstance(fixed, dict):
                is_free.append(not fixed.get(name, False))
            else:
                is_free.append(name not in fixed)

    p0.extend([b0_init, b1_init])
    is_free.extend([True, True])

    return np.array(p0, dtype=float), np.array(is_free, dtype=bool)


def reconstruct_full(p0, free_params, is_free):
    """Substitute fitted free parameters back into the full parameter vector."""
    full = p0.copy()
    full[is_free] = free_params
    return full


def build_model(x, params, peak_defs, x_ref=None):
    """
    Construct the complete spectral model: sum of peaks + linear baseline.

    Parameters
    ----------
    params : ndarray
        Ordered as, for each peak, (shift, FWHM, amplitude[, eta]),
        followed by (baseline offset, baseline slope).
    """

    y = np.zeros_like(x)
    idx = 0

    for peak in peak_defs:
        if peak['type'] in ['gaussian', 'lorentzian']:
            x0, fwhm, A = params[idx:idx + 3]
            idx += 3
            y += gaussian(x, x0, fwhm, A) if peak['type'] == 'gaussian' \
                else lorentzian(x, x0, fwhm, A)
        else:
            x0, fwhm, A, eta = params[idx:idx + 4]
            idx += 4
            y += pseudo_voigt(x, x0, fwhm, A, eta)

    b0, b1 = params[idx:idx + 2]
    x_ref = np.mean(x) if x_ref is None else x_ref
    y += b0 + b1 * (x - x_ref)

    return y


def normalise_bound(b, param_name, idx):
    """
    Ensure a user-supplied ``(lower, upper)`` bound is correctly
    ordered, silently swapping it (with a warning) if reversed.
    """
    if b is None:
        return (-np.inf, np.inf)

    low, high = b
    if low > high:
        print(f"Warning: swapped bounds for {param_name} @ site {idx}: {(low, high)} -> fixing order")
        low, high = high, low

    return (low, high)


def build_bounds(peak_defs, positive_intensity=True, allow_negative_intensity=False):
    """
    Construct lower/upper parameter bounds for ``scipy.optimize.curve_fit``.

    Amplitude bounds may optionally enforce positive peak intensities,
    the physically meaningful default for most NMR spectra.
    """

    lower, upper = [], []

    for i, peak in enumerate(peak_defs):

        b = peak.get("bounds", {})

        amp_bounds = normalise_bound(
            b.get("amplitude", b.get("A", (-np.inf, np.inf))), "amplitude", i,
        )

        amp_lower = 0.0 if (positive_intensity and not allow_negative_intensity) else amp_bounds[0]
        amp_upper = amp_bounds[1]

        shift_b = normalise_bound(b.get("shift", (-np.inf, np.inf)), "shift", i)
        fwhm_b = normalise_bound(b.get("fwhm", (1e-6, np.inf)), "fwhm", i)

        if peak["type"] in ["gaussian", "lorentzian"]:
            lower.extend([shift_b[0], fwhm_b[0], amp_lower])
            upper.extend([shift_b[1], fwhm_b[1], amp_upper])
        else:
            eta_b = normalise_bound(b.get("eta", (0.0, 1.0)), "eta", i)
            lower.extend([shift_b[0], fwhm_b[0], amp_lower, eta_b[0]])
            upper.extend([shift_b[1], fwhm_b[1], amp_upper, eta_b[1]])

    return np.array(lower), np.array(upper)


def enforce_positive_intensity(popt, peak_defs):
    """Clip fitted amplitudes to zero if negative."""
    corrected = popt.copy()
    idx = 0

    for peak in peak_defs:
        step = 3 if peak["type"] in ["gaussian", "lorentzian"] else 4
        if corrected[idx + 2] < 0:
            corrected[idx + 2] = 0.0
        idx += step

    return corrected


# ============================================================
# Fitting
# ============================================================

def fit_spectrum(results, peak_defs, fit_regions=None, use_normalised=True, x_unit="ppm",
                  baseline_regions=None, baseline_sigma=2.5, baseline_max_iter=10,
                  positive_intensity=True, allow_negative_intensity=False):
    """
    Fit an NMR spectrum using nonlinear least squares.

    Workflow: extract spectrum -> optionally normalise -> estimate a
    robust linear baseline (held fixed during peak fitting) -> build
    parameter vector and bounds -> optimise with
    ``scipy.optimize.curve_fit``.

    Parameters
    ----------
    results : dict
        Output of :func:`nmrtools.one_d.NMR1dimport` or
        :func:`nmrtools.two_d.prepare_trace_for_fit`.

    peak_defs : list of dict
        Each dict: ``{"type": "gaussian"|"lorentzian"|"pseudo_voigt",
        "shift": ..., "fwhm": ..., ["A_init": ..., "eta": ...,
        "fixed": {...}, "bounds": {...}]}``.

    fit_regions : list of tuple, optional
        Regions of the spectrum included in the fit, ``[(xmin, xmax), ...]``.
        If None, the complete spectrum is fitted.

    use_normalised : bool, optional
        If True, fit the spectrum normalised to its maximum absolute
        intensity (recommended -- keeps parameter scales close to 1
        for the optimiser), rescaling amplitudes back afterwards.

    x_unit : {'ppm', 'Hz'}, optional

    baseline_regions, baseline_sigma, baseline_max_iter :
        See :func:`estimate_baseline_robust`.

    positive_intensity : bool, optional
        Constrain fitted peak amplitudes to be non-negative (default).
        Also used to floor each peak's *initial* amplitude guess to 0
        when its true starting value is negative but the fit is locked
        positive anyway (see :func:`initialise_params_with_map`).

    allow_negative_intensity : bool, optional
        Overrides ``positive_intensity`` to unlock negative amplitudes
        even when ``positive_intensity=True``. Negative amplitudes are
        allowed whenever ``not positive_intensity or
        allow_negative_intensity`` -- set this (or
        ``positive_intensity=False``) for genuinely negative-going data,
        e.g. an inversion-recovery slice before the null point, or a
        difference spectrum (see :func:`nmrtools.arithmetic.combine_spectra`).
        A per-peak ``bounds`` entry in ``peak_defs`` still overrides this.

    Returns
    -------
    popt : ndarray
        Optimised parameter vector (free parameters only expanded back
        to the full vector; fixed baseline included).
    pcov_free : ndarray
        Covariance matrix for the free parameters only -- expand with
        :func:`expand_pcov` before indexing by full parameter position.
        CAVEAT: unlike ``popt``, this is NOT rescaled back to raw
        units when ``use_normalised=True`` -- it still reflects
        curve_fit's fit in the internally-normalised intensity scale.
        Since ``popt``'s amplitude and baseline entries ARE rescaled,
        any uncertainty reported from ``pcov`` (error bars, bound-hit
        warnings, integral errors) is systematically too small by
        roughly a factor of ``scale`` whenever ``use_normalised=True``.
        This is a separate, pre-existing issue from the popt/baseline
        rescale bug fixed above; it's flagged here rather than fixed,
        since correcting it touches every downstream consumer of pcov
        (:func:`compute_integral_errors`, :func:`check_bound_hits`,
        :func:`print_fit_report`, and the 2D equivalents) and deserves
        its own review.
    is_free : ndarray(bool)
    scale : float
        Intensity normalisation factor applied (1.0 if
        ``use_normalised=False``).
    x_ref : float
        Reference x-value used for the baseline.
    """

    if x_unit == "ppm":
        x_full = results["x_ppm"]
    elif x_unit == "Hz":
        x_full = results["x_Hz"]
    else:
        raise ValueError("x_unit must be 'ppm' or 'Hz'")

    if use_normalised:
        y_raw = results['real_spectrum']
        scale = np.max(np.abs(y_raw))
        scale = scale if scale != 0 else 1.0
        y_full = y_raw / scale
    else:
        y_full = results['real_spectrum']
        scale = 1.0

    mask = build_region_mask(x_full, fit_regions)
    x = x_full[mask]
    y = y_full[mask]

    x_ref = np.mean(x)

    b0_init, b1_init, _ = estimate_baseline_robust(
        x_full, y_full, baseline_regions=baseline_regions,
        sigma=baseline_sigma, max_iter=baseline_max_iter, x_ref=x_ref,
    )

    p0, is_free = initialise_params_with_map(
        x, y, peak_defs, b0_init, b1_init,
        positive_intensity=positive_intensity,
        allow_negative_intensity=allow_negative_intensity,
    )

    # Baseline is estimated from baseline-only regions and then held
    # fixed during peak fitting.
    is_free[-2:] = False

    lower, upper = build_bounds(peak_defs, positive_intensity=positive_intensity,
                                 allow_negative_intensity=allow_negative_intensity)
    lower = np.append(lower, [-np.inf, -np.inf])
    upper = np.append(upper, [np.inf, np.inf])

    assert len(p0) == len(is_free) == len(lower) == len(upper), "parameter length mismatch"

    def model_wrapper(x, *free_params):
        full = reconstruct_full(p0, free_params, is_free)
        return build_model(x, full, peak_defs, x_ref)

    for i, (val, lo, hi) in enumerate(zip(p0[is_free], lower[is_free], upper[is_free])):
        if val < lo or val > hi:
            print("BAD INITIAL PARAMETER", i, "value=", val, "bounds=", lo, hi)

    popt_free, pcov_free = curve_fit(
        model_wrapper, x, y, p0=p0[is_free], bounds=(lower[is_free], upper[is_free]),
        maxfev=20000,
    )

    popt = reconstruct_full(p0, popt_free, is_free)

    def rescale_params(popt, scale):
        # Rescales every intensity-axis quantity in popt back from the
        # curve_fit-internal normalised scale to raw units -- peak
        # amplitudes AND the linear baseline (b0, b1), which are also
        # intensity-axis quantities. Bugfix: this previously rescaled
        # only the amplitudes and left the baseline in the normalised
        # scale, so final_popt silently mixed units whenever
        # use_normalised=True (raw-scale peaks, normalised-scale
        # baseline) -- correct amplitudes, wrong baseline, and wrong
        # anywhere the two were combined (e.g. the fitted curve).
        popt = popt.copy()
        idx = 0
        for peak in peak_defs:
            step = 3 if peak["type"] in ["gaussian", "lorentzian"] else 4
            popt[idx + 2] *= scale
            idx += step
        popt[idx:idx + 2] *= scale  # baseline b0, b1
        return popt

    final_popt = rescale_params(popt, scale) if use_normalised else popt

    return final_popt, pcov_free, is_free, scale, x_ref


def evaluate_fit(x, popt, peak_defs, x_ref=None, return_components=False):
    """
    Evaluate the fitted model at ``x``. If ``return_components=True``,
    also return each individual peak (plus the baseline, last) as a
    list -- useful for plotting and integration.
    """

    y_fit = np.zeros_like(x)
    components = []
    idx = 0

    for peak in peak_defs:
        if peak['type'] in ['gaussian', 'lorentzian']:
            x0, fwhm, A = popt[idx:idx + 3]
            idx += 3
            y = gaussian(x, x0, fwhm, A) if peak['type'] == 'gaussian' else lorentzian(x, x0, fwhm, A)
        else:
            x0, fwhm, A, eta = popt[idx:idx + 4]
            idx += 4
            y = pseudo_voigt(x, x0, fwhm, A, eta)

        components.append(y)
        y_fit += y

    b0, b1 = popt[idx:idx + 2]
    x_ref = np.mean(x) if x_ref is None else x_ref
    baseline = b0 + b1 * (x - x_ref)

    components.append(baseline)
    y_fit += baseline

    return (y_fit, components) if return_components else y_fit


def compute_integrals(x, components):
    """
    Integrate each peak component (all but the last entry, which is
    the baseline) by trapezoidal rule. Returns ``(integrals,
    integrals_norm)`` where the latter is each integral as a
    percentage of their sum.

    Integrals are signed, not ``abs()``'d: for an ordinary positive-
    going absorption peak this makes no difference (the integral is
    already positive), but a peak fitted with a genuinely negative
    amplitude -- allowed via ``positive_intensity=False`` or
    ``allow_negative_intensity=True`` in :func:`fit_spectrum`, e.g. an
    inversion-recovery slice before the null point, or a difference
    spectrum -- has a genuinely negative integral, and discarding that
    sign would corrupt anything computed from it (a relaxation curve
    fitted to these integrals, for instance, needs the sign to see the
    magnetisation cross zero). ``integrals_norm`` is only a meaningful
    "percentage of total" when every integral shares the same sign;
    with mixed-sign integrals the percentages can exceed 100% or be
    negative, since the total can be small relative to the individual
    (partially cancelling) terms -- this is expected, not a bug.

    ``x`` is not assumed to be ascending -- nmrtools' convention (see
    :func:`nmrtools.one_d.NMR1dimport`) stores ppm/Hz axes in
    descending order, and ``np.trapz`` silently flips sign under a
    descending axis. Each component is integrated against ``x``
    sorted ascending, so the returned sign reflects the data (peak
    vs. dip), never the axis direction.
    """

    order = np.argsort(x)
    x_sorted = x[order]
    integrals = [_trapz(np.asarray(comp)[order], x_sorted) for comp in components[:-1]]
    total = np.sum(integrals)

    if total != 0:
        integrals_norm = [100 * a / total for a in integrals]
    else:
        integrals_norm = [np.nan for _ in integrals]

    return integrals, integrals_norm


def compute_integral_errors(popt, pcov, peak_defs, integrals):
    """
    Estimate the uncertainty in each fitted integral from the
    amplitude uncertainty only (i.e. treating shift/fwhm/eta as exact
    -- a reasonable approximation when they are well-determined).
    """

    errors = []
    idx = 0

    for peak, I in zip(peak_defs, integrals):

        step = 3 if peak["type"] in ["gaussian", "lorentzian"] else 4
        amp_index = idx + 2
        idx += step

        A = popt[amp_index]
        sigma_A = np.sqrt(max(pcov[amp_index, amp_index], 0.0))

        errors.append(np.nan if A == 0 else abs(I) * sigma_A / abs(A))

    return errors


def fit_quality(y, y_fit, n_params):
    """Compute RMSE, chi-squared, and reduced chi-squared."""
    residual = y - y_fit
    rmse = np.sqrt(np.mean(residual ** 2))
    chi2 = np.sum(residual ** 2)
    red_chi2 = chi2 / max(len(y) - n_params, 1)
    return {"rmse": rmse, "chi2": chi2, "reduced_chi2": red_chi2}


def expand_pcov(pcov_free, mask, n_full):
    """Expand the free-parameter covariance matrix to the full parameter count."""
    pcov_full = np.zeros((n_full, n_full), dtype=float)
    free_idx = np.where(mask)[0]
    for i, fi in enumerate(free_idx):
        for j, fj in enumerate(free_idx):
            pcov_full[fi, fj] = pcov_free[i, j]
    return pcov_full


def check_bound_hits(popt, pcov, is_free, lower, upper, peak_defs):
    """Flag any free parameter whose fitted value (plus/minus its
    uncertainty) reaches a bound -- a sign the bound may be too tight."""

    warnings = []
    idx = 0

    for i, peak in enumerate(peak_defs):

        names = ["shift", "fwhm", "amplitude"] if peak["type"] in ["gaussian", "lorentzian"] \
            else ["shift", "fwhm", "amplitude", "eta"]

        for name in names:

            val, lo, hi = popt[idx], lower[idx], upper[idx]

            if not is_free[idx]:
                idx += 1
                continue

            err = np.sqrt(max(pcov[idx, idx], 0.0))

            if np.isfinite(hi) and val + err >= hi:
                warnings.append(f"Peak {i + 1} {name} near upper bound {hi}")
            if np.isfinite(lo) and val - err <= lo:
                warnings.append(f"Peak {i + 1} {name} near lower bound {lo}")

            idx += 1

    return warnings


def extract_params_and_errors(popt, pcov, peak_defs, is_free):
    """Split the flat parameter/covariance arrays back out per-peak."""

    per_peak = []
    idx = 0

    for peak in peak_defs:

        names = ["shift", "fwhm", "amplitude"] if peak["type"] in ["gaussian", "lorentzian"] \
            else ["shift", "fwhm", "amplitude", "eta"]
        n = len(names)

        params = popt[idx:idx + n]
        cov = pcov[idx:idx + n, idx:idx + n]
        errors = np.sqrt(np.diag(cov))
        fixed_flags = ~is_free[idx:idx + n]

        per_peak.append({
            "type": peak["type"], "params": params, "errors": errors,
            "fixed": fixed_flags, "names": names,
        })

        idx += n

    return per_peak


def print_fit_report(x, y, popt, pcov, peak_defs, warnings, is_free, components=None,
                      save_txt=False, txt_filename="fit_report.txt"):
    """Print (and optionally save) a human-readable report of fitted
    parameters, uncertainties, integrals, and fit-quality metrics."""

    per_peak = extract_params_and_errors(popt, pcov, peak_defs, is_free)
    n_params = len(popt)

    if components is None:
        _, components = evaluate_fit(x, popt, peak_defs, x_ref=np.mean(x), return_components=True)

    y_fit = np.sum(components, axis=0)
    metrics = fit_quality(y, y_fit, n_params)
    integrals, integrals_norm = compute_integrals(x, components)

    report = []

    def write(line=""):
        print(line)
        report.append(str(line))

    write("\n" + "=" * 60)
    write("NMR FIT REPORT")
    write("=" * 60)
    write("\nPeak parameters:")
    write("-" * 60)

    for i, peak in enumerate(per_peak):
        write(f"\nPeak {i + 1} ({peak['type']}):")
        for name, val, err, is_fixed in zip(peak["names"], peak["params"], peak["errors"], peak["fixed"]):
            if is_fixed:
                write(f"  {name:<10}= {val:.6f} (fixed)")
            else:
                write(f"  {name:<10}= {val:.6f} \u00b1 {err:.6f}")
        write(f"  integral (raw) = {integrals[i]:.4f}")
        write(f"  integral (%)    = {integrals_norm[i]:.2f}")

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

def NMR1dplot_with_fit(results, popt, peak_defs, fit_regions=None, x_unit='ppm',
                        xlims='default', xlabel=None, pl_width=3.375, pl_height=3.375,
                        font='Arial', fontsize=12, output_format='',
                        show_components=False, color_data='black', color_fit='red',
                        color_components=None, peak_labels=None, show_component_labels=False,
                        component_fill=True, component_fill_alpha=0.25,
                        lw_data=2.2, lw_fit=1.2, lw_components=1.2,
                        tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                        tick_number=None,
                        field='', use_normalised=True, 
                        save_fig=False, fig_name='nmr_fit_plot', save_format='png'):
    """
    Plot a 1D spectrum together with its fitted model (and optionally
    individual peak components).

    Parameters
    ----------
    results : dict
        Output of :func:`nmrtools.one_d.NMR1dimport`.
    popt, peak_defs :
        As returned by / passed to :func:`fit_spectrum`. If an entry
        in ``peak_defs`` has a ``"label"`` key, it is used as that
        peak's default label (overridden by ``peak_labels`` if given).
    xlims : 'default' or [left, right], optional
        X-axis display limits.
    xlabel : str, optional
        Override the automatically-generated (nuclide-based) x-axis
        label.
    pl_width, pl_height : float, optional
        Figure size in inches.
    font, fontsize : optional
        Typeface settings, see :func:`nmrtools.plotting.setup_plotting`.
    color_data, color_fit : optional
        Colours for the data and overall fit lines.
    color_components : str, list of str, or None, optional
        Colours for the individual peak components (only used when
        ``show_components=True``). A single colour is applied to all
        components; a list gives one colour per peak, in ``peak_defs``
        order. If None (default), colours are drawn from matplotlib's
        current colour cycle, one per peak.
    peak_labels : list of str, optional
        One label per peak, in ``peak_defs`` order. Falls back to
        ``peak_defs[i]["label"]`` if present, else ``"Peak i+1"``.
        Only shown in the legend when ``show_component_labels=True``.
    show_component_labels : bool, optional
        If True, each peak component gets its own legend entry using
        ``peak_labels`` (or the fallback above). Default False, since
        with many peaks this can crowd the legend -- consider
        annotating peaks directly on the plot instead in that case.
    component_fill : bool or list of bool, optional
        Whether to shade each component with ``fill_between`` (between
        the baseline and the component + baseline). A single bool
        applies to all peaks; a list gives one flag per peak. Default
        True. The fill colour always matches that peak's line colour,
        distinguished only by ``component_fill_alpha``.
    component_fill_alpha : float, optional
        Opacity of the component fill. Default 0.25.
    lw_data, lw_fit, lw_components : float, optional
        Line widths for the data, overall fit, and individual
        components. Defaults make the data line the thickest
        (``lw_data=2.2``) and the fit line thinner (``lw_fit=1.2``),
        so deviations between data and fit are easier to see.
    use_normalised : bool, optional
        Must match the ``use_normalised``/``normalise`` value used
        when fitting, so the data and model are plotted on the same
        intensity scale.
    tick_dir : {'in', 'out', 'inout'}, optional
    tick_length, tick_width, tick_pad : float, optional
        Major tick length/width/label-padding (x-axis only).
    tick_number : int, optional
        Approximate target number of x-axis ticks. If None, matplotlib's
        default locator is used.
    save_fig : bool, optional
        If True, save the figure (see :func:`nmrtools.plotting.save_figure`).
        Default False.
    fig_name : str, optional
        Output path/filename, with or without an extension. Default
        ``'nmr_fit_plot'``.
    save_format : {'png', 'pdf', 'svg'}, optional
        Output format used when ``fig_name`` has no extension already.
        Default ``'png'``.

    Returns
    -------
    fig, ax
    """
    style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                            pl_height=pl_height, output_format=output_format)
    pl_width, pl_height = style['pl_width'], style['pl_height']

    x = results['x_ppm'] if x_unit == 'ppm' else results['x_Hz']

    mask = build_region_mask(x, fit_regions) if fit_regions is not None else np.ones_like(x, dtype=bool)
    x_fit_for_ref = x[mask]
    x_ref = np.mean(x_fit_for_ref)

    if use_normalised:
        scale = np.max(np.abs(results['real_spectrum']))
        scale = scale if scale != 0 else 1.0
        y = results['real_spectrum'] / scale
    else:
        scale = 1.0
        y = results['real_spectrum']

    x_fit = x[mask]
    y_fit_data = y[mask]

    xmin, xmax = x_fit[0], x_fit[-1]
    if xmin > xmax:
        xmin, xmax = xmax, xmin

    # popt (from fit_spectrum) is always in raw intensity units,
    # regardless of use_normalised -- it rescales amplitudes and
    # baseline back after fitting (see fit_spectrum/rescale_params).
    # So the evaluated model is in raw units too, and must be divided
    # by the same `scale` as the data above to land on the same axis.
    # Previously this division was missing: when use_normalised=True,
    # the data line was plotted on the normalised scale but the fit
    # (and its components) were left in raw units -- the two would
    # only coincide by chance (scale == 1), and would otherwise show
    # a fit line at the wrong height/scale relative to the data.
    if show_components:
        y_fit, components = evaluate_fit(x_fit, popt, peak_defs, x_ref=x_ref, return_components=True)
        components = [comp / scale for comp in components]
    else:
        y_fit = evaluate_fit(x_fit, popt, peak_defs, x_ref=x_ref)
        components = None

    y_fit = y_fit / scale

    fig, ax = plt.subplots()

    ax.plot(x_fit, y_fit_data, color=color_data, lw=lw_data, label='Data', zorder=2)
    ax.plot(x_fit, y_fit, color=color_fit, lw=lw_fit, label='Fit', zorder=3)

    if show_components:
        baseline = components[-1]
        n_peaks = len(components) - 1

        # Per-peak colours: explicit single colour, explicit list, or
        # fall back to matplotlib's own colour cycle (public API, not
        # the current axes' internal cycle state).
        if color_components is None:
            cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            comp_colors = [cycle_colors[i % len(cycle_colors)] for i in range(n_peaks)]
        elif isinstance(color_components, str):
            comp_colors = [color_components] * n_peaks
        else:
            comp_colors = list(color_components)
            if len(comp_colors) < n_peaks:
                raise ValueError(
                    f"color_components has {len(comp_colors)} entries but "
                    f"there are {n_peaks} peaks."
                )

        # Per-peak fill toggle.
        if isinstance(component_fill, bool):
            comp_fill = [component_fill] * n_peaks
        else:
            comp_fill = list(component_fill)
            if len(comp_fill) < n_peaks:
                raise ValueError(
                    f"component_fill has {len(comp_fill)} entries but "
                    f"there are {n_peaks} peaks."
                )

        # Per-peak labels: explicit peak_labels > peak_defs[i]["label"] > "Peak i+1".
        comp_labels = []
        for i in range(n_peaks):
            if peak_labels is not None and i < len(peak_labels):
                comp_labels.append(peak_labels[i])
            else:
                comp_labels.append(peak_defs[i].get('label', f'Peak {i + 1}'))

        for i, comp in enumerate(components[:-1]):
            y_comp = comp + baseline
            ax.plot(x_fit, y_comp, color=comp_colors[i], lw=lw_components,
                    label=comp_labels[i] if show_component_labels else None,
                    zorder=1)
            if comp_fill[i]:
                ax.fill_between(x_fit, baseline, y_comp, color=comp_colors[i],
                                 alpha=component_fill_alpha, lw=0, zorder=0)

    if xlims == 'default':
        ax.set_xlim(xmin, xmax)
    else:
        xmin, xmax = xlims[0], xlims[1]
        ax.set_xlim(xmin, xmax)

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
        ax.set_xlabel(NUC_label(nucgamma['Nuclide'], x_unit=x_unit))

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

def estimate_noise_level(x, y, noise_region=None):
    """
    Estimate the spectral noise standard deviation. If
    ``noise_region`` is None, use the first and last 5% of the
    spectrum; otherwise use the given ppm region(s), e.g. ``(-40,
    -30)`` or ``[(-40, -30), (20, 30)]``.
    """

    if noise_region is None:
        n = len(y)
        edge = int(0.05 * n)
        noise_points = np.concatenate([y[:edge], y[-edge:]])
    else:
        if isinstance(noise_region[0], (int, float)):
            noise_region = [noise_region]

        mask = np.zeros_like(x, dtype=bool)
        for lo, hi in noise_region:
            lo, hi = sorted([lo, hi])
            mask |= (x >= lo) & (x <= hi)

        noise_points = y[mask]
        if len(noise_points) == 0:
            raise ValueError("Noise region contains no points")

    return np.std(noise_points)


def NMRFitWorkflow(datapath, peak_defs, procno=1, fit_regions=None, baseline_regions=None,
                    baseline_sigma=2.5, baseline_max_iter=10, x_unit='ppm', normalise=True,
                    show_plot=True, xlims='default', xlabel=None, pl_width=3.375, pl_height=3.375,
                    font='Arial', fontsize=12, output_format='',
                    tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                    tick_number=None,
                    show_components=False, show_report=True,
                    color_data='black', color_fit='red', color_components=None,
                    peak_labels=None, show_component_labels=False,
                    component_fill=True, component_fill_alpha=0.25,
                    lw_data=2.2, lw_fit=1.2, lw_components=1.2,
                    positive_intensity=True, allow_negative_intensity=False,
                    save_csv=False, csv_filename="fit_results.csv",
                    save_report=False, report_filename="fit_report.txt",
                    save_fig=False, fig_name='nmr_fit_plot', save_format='png'):
    """
    Fit a 1D Bruker NMR spectrum using predefined peak models.

    The primary user-facing workflow for fitting a processed Bruker 1D
    spectrum: imports it, estimates the baseline, fits the specified
    peak models by nonlinear least squares, and optionally plots and
    reports the result.

    Parameters
    ----------
    datapath : str
        Path to the Bruker experiment directory.
    peak_defs : list of dict
        See :func:`fit_spectrum`.
    procno : int, optional
    fit_regions : list of tuple, optional
        ``[(xmin, xmax), ...]``. If None, the complete spectrum is fitted.
    x_unit : {'ppm', 'Hz'}, optional
    normalise : bool, optional
        Fit (and plot) the spectrum normalised to its maximum absolute
        intensity.
    show_plot, show_components, show_report : bool, optional
    xlims : 'default' or [left, right], optional
        X-axis display limits, forwarded to :func:`NMR1dplot_with_fit`.
    xlabel : str, optional
        Override the automatically-generated x-axis label, forwarded
        to :func:`NMR1dplot_with_fit`.
    pl_width, pl_height : float, optional
        Figure size in inches.
    font, fontsize : optional
        Typeface settings, see :func:`nmrtools.plotting.setup_plotting`.
    tick_dir : {'in', 'out', 'inout'}, optional
    tick_length, tick_width, tick_pad : float, optional
    tick_number : int, optional
        Forwarded to :func:`NMR1dplot_with_fit`.
    color_data, color_fit, color_components, peak_labels,
    show_component_labels, component_fill, component_fill_alpha,
    lw_data, lw_fit, lw_components :
        Forwarded to :func:`NMR1dplot_with_fit`; see there for details.
    positive_intensity : bool, optional
        Constrain peak amplitudes to be non-negative (default).
    allow_negative_intensity : bool, optional
        Overrides ``positive_intensity`` to unlock negative amplitudes
        even when ``positive_intensity=True``. Amplitudes are allowed
        negative whenever ``not positive_intensity or
        allow_negative_intensity`` -- for genuinely negative-going
        data (e.g. an inversion-recovery series before the null point,
        or a difference spectrum).
    save_csv, csv_filename, save_report, report_filename : optional
    save_fig : bool, optional
        If True, save the figure (see :func:`nmrtools.plotting.save_figure`).
        Default False.
    fig_name : str, optional
        Output path/filename, with or without an extension. Default
        ``'nmr_fit_plot'``.
    save_format : {'png', 'pdf', 'svg'}, optional
        Output format used when ``fig_name`` has no extension already.
        Default ``'png'``.

    Returns
    -------
    dict with keys ``popt``, ``pcov`` (full, expanded), ``x``, ``y``,
    ``y_fit``, ``components``, ``figure``, ``axes``.

    Notes
    -----
    Peak parameters are ordered ``shift, fwhm, amplitude[, eta]``,
    followed by baseline ``b0, b1``. The fitted model is
    ``sum(peaks) + b0 + b1 * (x - x_ref)``.
    """

    setup_plotting()

    results = NMR1dimport(datapath, procno=procno)

    x = results['x_ppm'] if x_unit == 'ppm' else results['x_Hz']
    # popt (below) is always returned in raw intensity units regardless
    # of `normalise` -- fit_spectrum only uses use_normalised as an
    # internal numerical-conditioning trick and rescales the fitted
    # parameters back afterwards. So the data used for the report and
    # CSV export must also be raw units to match popt for residuals
    # etc. `results['normalised_real_spectrum']` is a *different*
    # normalisation (min-max to [0, 1], for heatmap/stacked-plot
    # display -- see NMR1dimport/NMR1dheatmap) and was never on the
    # same scale as popt; using it here silently compared the model
    # against the wrong version of the data whenever normalise=True.
    # `normalise` still controls the *plotted* display scale, via
    # use_normalised below and in NMR1dplot_with_fit.
    y = results['real_spectrum']

    mask = build_region_mask(x, fit_regions) if fit_regions is not None else np.ones_like(x, dtype=bool)
    x_fit, y_fit = x[mask], y[mask]

    popt, pcov_free, is_free, scale, x_ref = fit_spectrum(
        results, peak_defs, fit_regions=fit_regions, use_normalised=normalise,
        baseline_regions=baseline_regions, baseline_sigma=baseline_sigma,
        baseline_max_iter=baseline_max_iter, x_unit=x_unit,
        positive_intensity=positive_intensity, allow_negative_intensity=allow_negative_intensity,
    )

    pcov = expand_pcov(pcov_free, is_free, len(popt))

    # Amplitudes are actually unlocked (in the curve_fit bounds used
    # above) whenever "not positive_intensity or
    # allow_negative_intensity" -- matching build_bounds' own
    # condition. Only the locked-positive case needs the post-hoc
    # clip (a harmless safety net there, since the bounds already
    # enforce it); any unlocked case gets a one-line heads-up instead.
    amplitude_locked_positive = positive_intensity and not allow_negative_intensity

    if amplitude_locked_positive:
        popt = enforce_positive_intensity(popt, peak_defs)
    else:
        print("\n[WARNING] Negative peak intensities are enabled "
              "(positive_intensity=False or allow_negative_intensity=True). "
              "This may produce non-physical fits in NMR spectra.\n")

    lower, upper = build_bounds(peak_defs, positive_intensity=positive_intensity,
                                 allow_negative_intensity=allow_negative_intensity)
    warnings = check_bound_hits(popt, pcov, is_free, lower, upper, peak_defs)

    if show_components:
        y_model, components = evaluate_fit(x_fit, popt, peak_defs, x_ref=x_ref, return_components=True)
    else:
        y_model = evaluate_fit(x_fit, popt, peak_defs, x_ref=x_ref)
        components = None

    fig, ax = None, None
    if show_plot:
        fig, ax = NMR1dplot_with_fit(
            results=results, popt=popt, peak_defs=peak_defs, fit_regions=fit_regions,
            x_unit=x_unit, show_components=show_components, use_normalised=normalise, 
            xlims=xlims, xlabel=xlabel, pl_width=pl_width, pl_height=pl_height,
            font=font, fontsize=fontsize, output_format=output_format,
            color_data=color_data, color_fit=color_fit, color_components=color_components,
            peak_labels=peak_labels, show_component_labels=show_component_labels,
            component_fill=component_fill, component_fill_alpha=component_fill_alpha,
            lw_data=lw_data, lw_fit=lw_fit, lw_components=lw_components,
            tick_dir=tick_dir, tick_length=tick_length, tick_width=tick_width,
            tick_pad=tick_pad, tick_number=tick_number,
        )
        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    if show_report:
        print_fit_report(x_fit, y_fit, popt, pcov, peak_defs, warnings, is_free,
                          components=components, save_txt=save_report, txt_filename=report_filename)

    if save_csv:
        import csv

        columns = {"x": x_fit, "y": y_fit, "y_fit": y_model}
        if components is not None:
            for i, comp in enumerate(components):
                columns[f"component_{i + 1}"] = comp

        with open(csv_filename, "w", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(columns.keys())
            for i in range(len(x_fit)):
                writer.writerow([columns[col][i] for col in columns])

    return {
        "popt": popt, "pcov": pcov, "x": x_fit, "y": y_fit, "y_fit": y_model,
        "components": components, "figure": fig, "axes": ax,
    }


# ============================================================
# 2D slice selection
# ============================================================

def select_2d_traces(dataset, axis="f1", indices=None, shifts=None,
                      index_range=None, shift_range=None, fit_all=False):
    """
    Build a sorted, de-duplicated list of 1-based trace indices from a
    mix of selection criteria -- a convenience for building the
    ``rows``/``columns`` (or ``row_shifts``/``column_shifts``) argument
    to :func:`nmrtools.two_d.NMR2D_get_slices`.

    Parameters
    ----------
    dataset : dict
        Output of :func:`nmrtools.two_d.NMR2dimport`.
    axis : {'f1', 'f2'}, optional
        Matches :func:`nmrtools.two_d.NMR2d_get_trace`'s convention.
    indices : list of int, optional
        Explicit 1-based indices.
    shifts : list of float, optional
        Chemical shifts (ppm), converted to the nearest index.
    index_range : tuple(int, int), optional
        Inclusive 1-based index range.
    shift_range : tuple(float, float), optional
        ppm range, converted to the enclosed indices.
    fit_all : bool, optional
        If True, ignore other arguments and return every index.

    Returns
    -------
    list of int
    """

    pseudo2d = dataset.get("pseudo_2d", False)

    if axis.lower() == "f1" and pseudo2d:
        ppm_axis = dataset["y_index"].astype(float)
    elif axis.lower() == "f1":
        ppm_axis = dataset["y_ppm"]
    else:
        ppm_axis = dataset["x_ppm"]

    if fit_all:
        return list(range(1, len(ppm_axis) + 1))

    selected = []

    if indices is not None:
        selected.extend(indices)

    if shifts is not None:
        for s in shifts:
            selected.append(int(np.argmin(np.abs(ppm_axis - s))) + 1)

    if index_range is not None:
        a, b = index_range
        selected.extend(range(a, b + 1))

    if shift_range is not None:
        low, high = sorted(shift_range)
        mask = (ppm_axis >= low) & (ppm_axis <= high)
        selected.extend(np.where(mask)[0] + 1)

    return sorted(set(selected))


# ============================================================
# 2D slice fitting
# ============================================================

def NMR2D_fit_slices(slices, peak_defs, mode="rows", fit_regions=None,
                      baseline_regions=None, baseline_sigma=2.5, baseline_max_iter=10,
                      x_unit='ppm', noise_region=None, normalise=False,
                      show_components=True, positive_intensity=True,
                      allow_negative_intensity=False, sn_threshold=3,
                      show_report=False, show_warnings=True,
                      plot_integrals=False, integral_x_axis="shift",
                      integral_normalised=False, integral_show_errors=True):
    """
    Fit one or more extracted 2D NMR slices (rows and/or columns).

    Parameters
    ----------
    slices : dict
        Output of :func:`nmrtools.two_d.NMR2D_get_slices`.
    peak_defs : list of dict
        See :func:`fit_spectrum`.
    mode : {'rows', 'columns', 'both'}, optional
        Which extracted slices to fit.
    fit_regions, baseline_regions, baseline_sigma, baseline_max_iter, x_unit :
        See :func:`fit_spectrum`.
    noise_region : tuple or list of tuples, optional
        If given, peaks with signal-to-noise below ``sn_threshold`` in
        a given slice are fixed to zero amplitude for that slice's fit
        (useful for a pseudo-2D series where a peak decays into the
        noise).
    normalise : bool, optional
        Normalise each trace before fitting.
    show_components : bool, optional
        Compute individual fitted peak components (needed for
        integrals and for ``show_fit_components`` on
        :func:`nmrtools.two_d.NMR2D_plot_slices`); components are
        always computed for integration internally regardless of this
        flag, but this controls whether they are also stored/returned.
    positive_intensity, allow_negative_intensity : bool, optional
        Amplitudes are allowed negative whenever ``not
        positive_intensity or allow_negative_intensity`` (both the
        curve_fit bounds and initial guesses respect this) -- for
        genuinely negative-going data, e.g. an inversion-recovery
        series before the null point.
    sn_threshold : float, optional
    show_report, show_warnings : bool, optional
        Print a per-slice fit report / bound-hit warnings as each
        slice is fitted.
    plot_integrals : bool, optional
        If True, call :func:`NMR2D_plot_integrals` once fitting is
        complete (for each of rows/columns present in the result).
    integral_x_axis, integral_normalised, integral_show_errors : optional
        Forwarded to :func:`NMR2D_plot_integrals`.

    Returns
    -------
    dict with keys ``"rows"`` and ``"columns"``, each a list of
    fit-result dicts with keys: ``slice`` (the original trace dict),
    ``slice_type``, ``index``, ``shift``, ``popt``, ``pcov``,
    ``is_free``, ``scale``, ``x_ref``, ``x``, ``y`` (data, in the
    fitting intensity scale), ``y_raw``, ``y_fit``, ``components``,
    ``integrals``, ``integral_errors``, ``integrals_norm``,
    ``fit_quality``, ``warnings``, ``peak_defs`` (the per-slice copy,
    including any S/N-based amplitude fixing).
    """

    if mode not in ["rows", "columns", "both"]:
        raise ValueError("mode must be 'rows', 'columns', or 'both'")

    selected = []
    if mode in ["rows", "both"]:
        selected.extend(("row", s) for s in slices.get("rows", []))
    if mode in ["columns", "both"]:
        selected.extend(("column", s) for s in slices.get("columns", []))

    if not selected:
        raise ValueError("No slices available for fitting.")

    fitted = {"rows": [], "columns": []}

    for slice_type, s in selected:

        # NOTE: prepare_trace_for_fit is defined in two_d.py (not
        # touched by this rename), so its own keyword is still spelt
        # the American way -- only the value passed to it, and every
        # name on this side of the boundary, use the British spelling.
        results = prepare_trace_for_fit(s, normalize=normalise)
        x = results["x_ppm"]
        y = results["real_spectrum"]

        local_peak_defs = [p.copy() for p in peak_defs]

        # ----- S/N-based amplitude gating -----

        if noise_region is not None:

            noise = estimate_noise_level(x, y, noise_region=noise_region)
            if noise <= 0:
                noise = np.finfo(float).eps

            for peak in local_peak_defs:

                shift = peak["shift"]
                idx = np.argmin(np.abs(x - shift))

                dx = np.abs(np.mean(np.diff(x))) if len(x) > 1 else 1.0
                fwhm = peak.get("fwhm", 0.01)
                half_window = max(3, int(abs(fwhm / dx)))

                lo = max(0, idx - half_window)
                hi = min(len(y), idx + half_window + 1)

                signal = np.max(np.abs(y[lo:hi]))
                sn = signal / noise
                peak["_sn"] = sn

                if sn < sn_threshold:
                    peak["fixed"] = {**peak.get("fixed", {}), "A": True}
                    peak["A_init"] = 0.0

        # ----- fit -----
        # NOTE: fits with local_peak_defs (which may have S/N-gated
        # amplitudes fixed to zero for this slice) -- not the original
        # peak_defs, which would silently ignore that gating.

        popt, pcov_free, is_free, scale, x_ref = fit_spectrum(
            results, local_peak_defs, fit_regions=fit_regions,
            use_normalised=normalise, baseline_regions=baseline_regions,
            baseline_sigma=baseline_sigma, baseline_max_iter=baseline_max_iter,
            x_unit=x_unit,
            positive_intensity=positive_intensity, allow_negative_intensity=allow_negative_intensity,
        )

        pcov = expand_pcov(pcov_free, is_free, len(popt))

        # See NMRFitWorkflow for why this checks both flags together:
        # amplitudes are only actually locked >= 0 (in the bounds used
        # above) when positive_intensity=True AND
        # allow_negative_intensity=False -- checking positive_intensity
        # alone would re-clip amplitudes back to zero even when the
        # caller explicitly unlocked them via allow_negative_intensity.
        if positive_intensity and not allow_negative_intensity:
            popt = enforce_positive_intensity(popt, local_peak_defs)

        y_fit, components = evaluate_fit(x, popt, local_peak_defs, x_ref=x_ref, return_components=True)

        integrals, integrals_norm = compute_integrals(x, components)
        errors = compute_integral_errors(popt, pcov, local_peak_defs, integrals)

        # Same fix as NMRFitWorkflow: popt/y_fit are always raw-unit,
        # so the comparison data must be too, not the unrelated
        # min-max-normalised field (see the note there for why).
        y_fit_data = results["real_spectrum"]
        metrics = fit_quality(y_fit_data, y_fit, len(popt))

        lower, upper = build_bounds(local_peak_defs, positive_intensity=positive_intensity,
                                     allow_negative_intensity=allow_negative_intensity)
        warnings = check_bound_hits(popt, pcov, is_free, lower, upper, local_peak_defs)

        fit_result = {
            "slice": s,
            "slice_type": slice_type,
            "index": s["index"],
            "shift": s["shift"],

            "popt": popt, "pcov": pcov, "is_free": is_free,
            "scale": scale, "x_ref": x_ref,

            "x": x, "y": y_fit_data, "y_raw": y,
            "y_fit": y_fit, "components": components if show_components else None,

            "integrals": integrals, "integral_errors": errors, "integrals_norm": integrals_norm,

            "fit_quality": metrics,
            "warnings": warnings,
            "peak_defs": local_peak_defs,
        }

        if show_report or show_warnings:
            NMR2D_print_fit_report(
                fit_result, peak_defs=local_peak_defs,
                slice_index=fit_result["index"], slice_shift=fit_result["shift"],
                show_warnings=show_warnings,
            )

        fitted["rows" if slice_type == "row" else "columns"].append(fit_result)

    if plot_integrals:
        for key in ("rows", "columns"):
            if fitted[key]:
                NMR2D_plot_integrals(
                    fitted[key], mode=key, x_axis=integral_x_axis,
                    normalised=integral_normalised, show_errors=integral_show_errors,
                )
                plt.show()

    return fitted


# ============================================================
# Reporting
# ============================================================

def NMR2D_print_fit_report(fit, peak_defs=None, slice_index=None, slice_shift=None,
                            show_warnings=True):
    """
    Print a fit report for one 2D slice (wraps :func:`print_fit_report`).

    ``fit`` should contain at least ``x``, ``y``, ``popt``, ``pcov``,
    ``is_free`` (as produced by :func:`NMR2D_fit_slices`).
    """

    if slice_index is not None:
        print("\n" + "=" * 70)
        print(f"2D NMR SLICE REPORT | slice {slice_index}")

    if slice_shift is not None:
        print(f"Slice position: {slice_shift:.5f} ppm")

    print("=" * 70)

    if all(key in fit for key in ["x", "y", "popt", "pcov", "is_free"]):
        print_fit_report(
            x=fit["x"], y=fit["y"], popt=fit["popt"], pcov=fit["pcov"],
            peak_defs=peak_defs if peak_defs is not None else fit.get("peak_defs", []),
            warnings=fit.get("warnings", []), is_free=fit["is_free"],
            components=fit.get("components"),
        )
    else:
        print("Fit report unavailable: fit dictionary does not contain the required fitting information.")

    if show_warnings:
        warnings = fit.get("warnings", [])
        if warnings:
            print("\n" + "!" * 70)
            print("FIT WARNINGS")
            print("!" * 70)
            for warning in warnings:
                print(f" - {warning}")
            print("!" * 70)
        else:
            print("\nNo fit warnings.")


def NMR2D_plot_integrals(fits, mode="rows", x_axis="shift", normalised=False,
                          show_errors=True, peak_labels=None, dataset_label=None,
                          colors=None, ax=None, xlabel=None, ylabel=None, title=None,
                          xlims=None, ylims=None,
                          show_top_spine=False, show_right_spine=False,
                          pl_width=3.375, pl_height=3.375, font='Arial', fontsize=12,
                          output_format='',
                          tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                          tick_number=None, vdlist=None, 
                          save_fig=False, fig_name='nmr_integrals_plot', save_format='png'):
    """
    Plot fitted peak integrals across a series of 2D slices.

    Parameters
    ----------
    fits : list
        Output of :func:`NMR2D_fit_slices` (``fitted["rows"]`` or
        ``fitted["columns"]``).

    mode : {'rows', 'columns'}, optional
        Only used to build a default x-axis label.

    x_axis : {'shift', 'index', 'vdlist'}, optional
        Quantity used for the x-axis. ``'shift'`` is automatically
        treated as ``'index'`` (no ppm units) for pseudo-2D slices,
        where "shift" has no chemical-shift meaning -- see ``vdlist``
        below for reading actual delay values for a T1/T2/diffusion
        series.

    normalised : bool, optional
        If True, plot each slice's integrals normalised to 100%.

    show_errors : bool, optional
        Show integral uncertainties when available for every slice.

    peak_labels : list of str, optional
        Legend label for each peak. Defaults to "Peak 1", "Peak 2", ...

    dataset_label : str, optional
        Prefixed onto every peak's legend label (e.g. ``"Sample A"`` ->
        ``"Sample A: Peak 1"``), useful for distinguishing multiple
        datasets plotted onto the same ``ax`` by calling this function
        more than once.

    colors : list, optional
        Explicit colour per peak (matched to ``peak_labels`` order).
        If None, matplotlib's automatic colour cycle is used -- so
        repeated calls onto the same ``ax`` for different datasets get
        different colours automatically even without this.

    ax : matplotlib.axes.Axes, optional

    xlabel, ylabel, title : str, optional
        Override the default axis labels / title.

    xlims, ylims : [left, right], optional
        Explicit axis display limits; otherwise set automatically
        from the data.

    show_top_spine, show_right_spine : bool, optional
        Default False (open-style axes, matching the rest of nmrtools).

    tick_dir : {'in', 'out', 'inout'}, optional
    tick_length, tick_width, tick_pad : float, optional
    tick_number : int, optional
        Approximate target number of ticks on both axes. If None,
        matplotlib's default locator is used.

    vdlist : optional
        Only used when ``x_axis='vdlist'``. One of:

        - a Bruker experiment path, in which case a file named
          ``"vdlist"`` is read from inside it (see
          :func:`nmrtools.utils.read_vdlist`);
        - a direct path to a delay-list file anywhere on disk, e.g.
          one kept outside the experiment directory;
        - the dict returned by :func:`nmrtools.utils.read_vdlist`;
        - an array-like of delays, in seconds, indexed the same way as
          the fitted slices (1-based F1 index).

        If not given, the experiment path is recovered automatically
        from the fitted slices when available (i.e. when they were
        extracted with :func:`nmrtools.two_d.NMR2D_get_slices` /
        :func:`nmrtools.two_d.NMR2d_get_trace`).

    save_fig : bool, optional
        If True, save the figure (see :func:`nmrtools.plotting.save_figure`).
        Default False. Only applies when a new standalone figure is
        created (i.e. ``ax`` was not passed in).

    fig_name : str, optional
        Output path/filename, with or without an extension. Default
        ``'nmr_integrals_plot'``.

    save_format : {'png', 'pdf', 'svg'}, optional
        Output format used when ``fig_name`` has no extension already.
        Default ``'png'``.

    Returns
    -------
    fig, ax
    """

    if fits is None or len(fits) == 0:
        raise ValueError("No fitted slices available.")

    if x_axis not in ("shift", "index", "vdlist"):
        raise ValueError("x_axis must be 'shift', 'index', or 'vdlist'")

    standalone = ax is None

    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style['pl_width'], style['pl_height']
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    shifts, indices, integrals, errors, integrals_norm = [], [], [], [], []
    shift_units = set()

    for fit in fits:
        shifts.append(fit.get("shift", np.nan))
        indices.append(fit.get("index", np.nan))
        integrals.append(fit.get("integrals", []))
        errors.append(fit.get("integral_errors", None))
        integrals_norm.append(fit.get("integrals_norm", None))
        shift_units.add(fit.get("slice", {}).get("shift_unit"))

    # Pseudo-2D slices (T1, T2, diffusion, ...) have no real chemical
    # shift along the extracted axis -- "shift" there is just the
    # slice index as a float. Fall back to a unitless slice-number
    # axis rather than mislabelling it "ppm".
    is_index_axis = shift_units == {"index"}

    if x_axis == "vdlist":

        if vdlist is None:
            datapath = next(
                (fit.get("slice", {}).get("datapath") for fit in fits
                 if fit.get("slice", {}).get("datapath")),
                None,
            )
            if datapath is None:
                raise ValueError(
                    "x_axis='vdlist' needs a vdlist to read delays from, but "
                    "no experiment path could be recovered from the fitted "
                    "slices. Pass vdlist explicitly: a Bruker experiment "
                    "path, the dict from nmrtools.utils.read_vdlist(), or an "
                    "array of delays in seconds."
                )
            vdlist = datapath

        if isinstance(vdlist, str):
            vdlist_seconds = read_vdlist(vdlist)["seconds"]
        elif isinstance(vdlist, dict):
            vdlist_seconds = np.asarray(vdlist["seconds"], dtype=float)
        else:
            vdlist_seconds = np.asarray(vdlist, dtype=float)

        try:
            x = np.array([vdlist_seconds[int(idx) - 1] for idx in indices], dtype=float)
        except IndexError:
            raise ValueError(
                f"vdlist has {len(vdlist_seconds)} entries, but a fitted "
                f"slice index falls outside that range -- check the vdlist "
                f"matches this dataset."
            )

        xlabel_default = "Delay (s)"

    elif x_axis == "index" or (x_axis == "shift" and is_index_axis):
        x = np.asarray(indices, dtype=float)
        xlabel_default = "Slice number"

    else:  # x_axis == "shift", real chemical shift
        x = np.asarray(shifts, dtype=float)
        xlabel_default = f"{mode.capitalize()} slice position (ppm)"

    order = np.argsort(x)
    x = x[order]
    integrals = [integrals[i] for i in order]
    errors = [errors[i] for i in order]
    integrals_norm = [integrals_norm[i] for i in order]

    n_peaks = max(len(i) for i in integrals)
    label_prefix = f"{dataset_label}: " if dataset_label else ""

    for peak_i in range(n_peaks):

        source = integrals_norm if normalised else integrals
        y = np.array([vals[peak_i] if len(vals) > peak_i else np.nan for vals in source])

        label = label_prefix + (
            peak_labels[peak_i] if peak_labels is not None and peak_i < len(peak_labels)
            else f"Peak {peak_i + 1}"
        )
        colour = colors[peak_i] if colors is not None and peak_i < len(colors) else None

        have_errors = show_errors and all(
            err is not None and len(err) > peak_i and np.isfinite(err[peak_i]) for err in errors
        )

        if have_errors:
            yerr = np.array([err[peak_i] for err in errors])
            ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", capsize=3,
                        label=label, color=colour)
        else:
            ax.plot(x, y, marker="o", linestyle="-", label=label, color=colour)

    ax.set_xlabel(xlabel if xlabel is not None else xlabel_default)
    ax.set_ylabel(ylabel if ylabel is not None else ("Peak integral (%)" if normalised else "Peak integral"))

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    if title is not None:
        ax.set_title(title)

    apply_spine_style(ax, top=show_top_spine, right=show_right_spine)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis='both')

    ax.legend()

    if standalone:
        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.tight_layout()

        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax


# ============================================================
# End-to-end 2D workflow
# ============================================================

def NMR2D_workflow(datapath, peak_defs=None, procno=1,
                    rows=None, columns=None, row_width=1, column_width=1,
                    row_shifts=None, column_shifts=None,
                    fit=True, integrate=True,
                    fit_regions=None, baseline_regions=None, baseline_sigma=2.5,
                    baseline_max_iter=10, normalise=True,
                    positive_intensity=True, allow_negative_intensity=False,
                    show_slices=True, show_fits=True, show_components=False,
                    show_integrals=True, show_report=False, show_warnings=True,
                    colour_spectra="coolwarm", offset_scale=1.05, xlims=None,
                    font='Arial', fontsize=12,
                    integral_x_axis="shift", integral_normalised=False,
                    integral_show_errors=True, peak_labels=None,
                    normalise_slices=False,
                    save_fig=False, fig_name='nmr_2d_slices',
                    integral_fig_name='nmr_2d_integrals', save_format='png'):
    """
    High-level, end-to-end workflow for analysing selected rows and/or
    columns from a Bruker 2D NMR spectrum: import once, extract, fit,
    integrate, report, and plot.

    Parameters
    ----------
    datapath : str
        Bruker experiment path.
    peak_defs : list of dict, optional
        Required if ``fit=True``. See :func:`fit_spectrum`.
    rows, columns : list of int, optional
        1-based indices to extract.
    row_shifts, column_shifts : list of float, optional
        Chemical shifts (ppm) to extract.
    row_width, column_width : int, optional
        Points to combine around each selected row/column (see
        :func:`nmrtools.two_d.NMR2d_get_trace`).
    fit, integrate : bool, optional
    show_report, show_warnings : bool, optional
        Print a fit report / bound-hit warnings for every fitted slice.
    colour_spectra, offset_scale, xlims :
        Forwarded to :func:`nmrtools.two_d.NMR2D_plot_slices`.
    font, fontsize : optional
        Forwarded to both :func:`nmrtools.two_d.NMR2D_plot_slices` and
        :func:`NMR2D_plot_integrals`.
    integral_x_axis, integral_normalised, integral_show_errors, peak_labels :
        Forwarded to :func:`NMR2D_plot_integrals`.
    save_fig : bool, optional
        If True, save both the slices figure and the integrals figure
        (see :func:`nmrtools.plotting.save_figure`). Default False.
    fig_name : str, optional
        Output path/filename for the slices figure. Default
        ``'nmr_2d_slices'``.
    integral_fig_name : str, optional
        Output path/filename for the integrals figure. Default
        ``'nmr_2d_integrals'``.
    save_format : {'png', 'pdf', 'svg'}, optional
        Output format used when a fig_name has no extension already.
        Default ``'png'``.

    Returns
    -------
    dict
        ``{"dataset": ..., "rows": {...} or None, "columns": {...} or None}``,
        where each present group is
        ``{"slices", "fits", "figure", "axes", "integral_figure", "integral_axes"}``.
    """

    if rows is None and columns is None and row_shifts is None and column_shifts is None:
        raise ValueError("No rows or columns were requested.")

    dataset = NMR2dimport(datapath, procno=procno)

    output = {"dataset": dataset, "rows": None, "columns": None}

    def process_direction(mode, indices, shifts, width):

        if indices is None and shifts is None:
            return None

        if mode == "rows":
            slices = NMR2D_get_slices(dataset, rows=indices, row_shifts=shifts,
                                       row_width=width, normalise=normalise_slices)
        else:
            slices = NMR2D_get_slices(dataset, columns=indices, column_shifts=shifts,
                                       column_width=width, normalise=normalise_slices)

        fits = None

        if fit:
            if peak_defs is None:
                raise ValueError("peak_defs must be supplied when fit=True.")

            fits = NMR2D_fit_slices(
                slices, peak_defs=peak_defs, mode=mode,
                fit_regions=fit_regions, baseline_regions=baseline_regions,
                baseline_sigma=baseline_sigma, baseline_max_iter=baseline_max_iter,
                normalise=normalise, positive_intensity=positive_intensity,
                allow_negative_intensity=allow_negative_intensity,
                # Reports/warnings are printed here (per-slice, right
                # after fitting) so the workflow-level switches control
                # them in one place.
                show_report=show_report, show_warnings=show_warnings,
            )[mode]

        fig_slices = ax_slices = None
        if show_slices:
            fig_slices, ax_slices = NMR2D_plot_slices(
                slices, mode=mode, fits=fits,
                show_fits=show_fits and fit, show_fit_components=show_components and fit,
                colour_spectra=colour_spectra, offset_scale=offset_scale, xlims=xlims,
                font=font, fontsize=fontsize,
            )
            if save_fig:
                save_figure(fig_slices, fig_name=f"{fig_name}_{mode}", save_format=save_format)

        fig_integrals = ax_integrals = None
        if integrate and fit and show_integrals and fits:
            fig_integrals, ax_integrals = NMR2D_plot_integrals(
                fits, mode=mode, x_axis=integral_x_axis, normalised=integral_normalised,
                show_errors=integral_show_errors, peak_labels=peak_labels,
                font=font, fontsize=fontsize,
            )
            if save_fig:
                save_figure(fig_integrals, fig_name=f"{integral_fig_name}_{mode}",
                            save_format=save_format)

        return {
            "slices": slices, "fits": fits,
            "figure": fig_slices, "axes": ax_slices,
            "integral_figure": fig_integrals, "integral_axes": ax_integrals,
        }

    if rows is not None or row_shifts is not None:
        output["rows"] = process_direction("rows", rows, row_shifts, row_width)

    if columns is not None or column_shifts is not None:
        output["columns"] = process_direction("columns", columns, column_shifts, column_width)

    return output
