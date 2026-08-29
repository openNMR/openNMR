"""
Peak fitting tools for 1D NMR spectra, and 2D slice-fit workflows built
on top of them.

Ported from ``nmrtools2``'s ``fitting.py``. The plotting-oriented,
one-shot workflow functions there (``NMR1dplot_with_fit``,
``NMRFitWorkflow``, ``NMR2D_plot_integrals``, ``NMR2D_workflow``) are
deliberately left out here -- they're deferred to a future GUI fitting
panel, which needs the fine-grained pieces below (fit once, let a user
tweak regions/peaks, refit) rather than one-shot import+fit+plot+report
convenience wrappers.

Provides:

- Gaussian, Lorentzian and pseudo-Voigt peak models
- Robust linear baseline estimation
- Automatic construction of model functions and parameter bounds
- Non-linear least-squares fitting (``fit_spectrum``)
- Fit evaluation, integration, and reporting
- A 2D slice-fitting workflow (``NMR2D_fit_slices``)

Workflow (2D)
-------------

    opennmr.processing.slicing.get_slices -> NMR2D_fit_slices

Each extracted trace from :func:`opennmr.processing.slicing.get_slices`
is already a 1D :class:`opennmr.core.spectrum.NMRSpectrum`, so it goes
straight into :func:`fit_spectrum` -- ``nmrtools2``'s
``prepare_trace_for_fit`` adapter is unnecessary here (see
``opennmr.processing.slicing``'s module docstring for why).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from ..core.spectrum import NMRSpectrum

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

    # Each pass: score every candidate point's residual against the
    # *current* fit, then only refit once membership (`keep`) actually
    # changes -- so a refit always incorporates the outlier exclusion
    # that motivated it, and the loop naturally converges once no
    # further points are added or dropped.
    for _ in range(max_iter):

        residual = y_base - A @ coeff
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        noise = 1.4826 * mad

        if noise <= 0:
            break

        new_keep = np.abs(residual - median) <= sigma * noise

        if np.sum(new_keep) < 3:
            break

        if np.array_equal(new_keep, keep):
            break

        keep = new_keep
        coeff, *_ = np.linalg.lstsq(A[keep], y_base[keep], rcond=None)

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


def initialize_params_with_map(x, y, peak_defs, b0_init, b1_init,
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


def normalize_bound(b, param_name, idx):
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

        amp_bounds = normalize_bound(
            b.get("amplitude", b.get("A", (-np.inf, np.inf))), "amplitude", i,
        )

        amp_lower = 0.0 if (positive_intensity and not allow_negative_intensity) else amp_bounds[0]
        amp_upper = amp_bounds[1]

        shift_b = normalize_bound(b.get("shift", (-np.inf, np.inf)), "shift", i)
        fwhm_b = normalize_bound(b.get("fwhm", (1e-6, np.inf)), "fwhm", i)

        if peak["type"] in ["gaussian", "lorentzian"]:
            lower.extend([shift_b[0], fwhm_b[0], amp_lower])
            upper.extend([shift_b[1], fwhm_b[1], amp_upper])
        else:
            eta_b = normalize_bound(b.get("eta", (0.0, 1.0)), "eta", i)
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

def fit_spectrum(spectrum: NMRSpectrum, peak_defs, fit_regions=None, use_normalised=True, x_unit="ppm",
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
    spectrum : NMRSpectrum
        A 1D spectrum -- e.g. from an ``opennmr.io`` importer, or a
        trace extracted with
        :func:`opennmr.processing.slicing.get_trace`/``get_slices``.

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

    x_unit : {'ppm', 'hz'}, optional
        Case-insensitive; passed to ``spectrum.axis_values(0, unit=...)``.

    baseline_regions, baseline_sigma, baseline_max_iter :
        See :func:`estimate_baseline_robust`.

    positive_intensity : bool, optional
        Constrain fitted peak amplitudes to be non-negative (default).
        Also used to floor each peak's *initial* amplitude guess to 0
        when its true starting value is negative but the fit is locked
        positive anyway (see :func:`initialize_params_with_map`).

    allow_negative_intensity : bool, optional
        Overrides ``positive_intensity`` to unlock negative amplitudes
        even when ``positive_intensity=True``. Negative amplitudes are
        allowed whenever ``not positive_intensity or
        allow_negative_intensity`` -- set this (or
        ``positive_intensity=False``) for genuinely negative-going data,
        e.g. an inversion-recovery slice before the null point, or a
        difference spectrum (see
        :func:`opennmr.processing.arithmetic.combine_spectra`). A
        per-peak ``bounds`` entry in ``peak_defs`` still overrides this.

    Returns
    -------
    popt : ndarray
        Optimised parameter vector (free parameters only expanded back
        to the full vector; fixed baseline included).
    pcov_free : ndarray
        Covariance matrix for the free parameters only -- expand with
        :func:`expand_pcov` before indexing by full parameter position.
    is_free : ndarray(bool)
    scale : float
        Intensity normalisation factor applied (1.0 if
        ``use_normalised=False``).
    x_ref : float
        Reference x-value used for the baseline.
    """

    x_full = spectrum.axis_values(0, unit=x_unit.lower())

    y_raw = spectrum.data
    if use_normalised:
        scale = np.max(np.abs(y_raw))
        scale = scale if scale != 0 else 1.0
        y_full = y_raw / scale
    else:
        y_full = y_raw
        scale = 1.0

    mask = build_region_mask(x_full, fit_regions)
    x = x_full[mask]
    y = y_full[mask]

    x_ref = np.mean(x)

    b0_init, b1_init, _ = estimate_baseline_robust(
        x_full, y_full, baseline_regions=baseline_regions,
        sigma=baseline_sigma, max_iter=baseline_max_iter, x_ref=x_ref,
    )

    p0, is_free = initialize_params_with_map(
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
        popt = popt.copy()
        idx = 0
        for peak in peak_defs:
            step = 3 if peak["type"] in ["gaussian", "lorentzian"] else 4
            popt[idx + 2] *= scale
            idx += step
        return popt

    final_popt = popt if use_normalised else rescale_params(popt, scale)

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

    ``x`` is not assumed to be ascending -- opennmr's Bruker
    importers store ppm/Hz axes in descending order (see
    ``opennmr.io.bruker``'s ``_ppm_hz_axis``), and ``np.trapz``
    silently flips sign under a descending axis. Each component is
    integrated against ``x`` sorted ascending, so the returned sign
    reflects the data (peak vs. dip), never the axis direction.
    """

    order = np.argsort(x)
    x_sorted = x[order]
    integrals = [_trapz(np.asarray(comp)[order], x_sorted) for comp in components[:-1]]
    total = np.sum(integrals)

    if total > 0:
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
                write(f"  {name:<10}= {val:.6f} ± {err:.6f}")
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
# Noise estimation
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


# ============================================================
# 2D slice selection
# ============================================================

def select_2d_traces(dataset: NMRSpectrum, axis: str = "f1", indices=None, shifts=None,
                      index_range=None, shift_range=None, fit_all=False) -> list[int]:
    """
    Build a sorted, de-duplicated list of 1-based trace indices from a
    mix of selection criteria -- a convenience for building the
    ``rows``/``columns`` (or ``row_shifts``/``column_shifts``) argument
    to :func:`opennmr.processing.slicing.get_slices`.

    Parameters
    ----------
    dataset : NMRSpectrum
        A 2D spectrum.
    axis : {'f1', 'f2'}, optional
        Matches :func:`opennmr.processing.slicing.get_trace`'s convention.
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

    pseudo2d = dataset.is_pseudo_2d

    if axis.lower() == "f1" and pseudo2d:
        ppm_axis = dataset.axes[0].index.astype(float)
    elif axis.lower() == "f1":
        ppm_axis = dataset.axes[0].ppm
    else:
        ppm_axis = dataset.axes[1].ppm

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

def NMR2D_fit_slices(slices: dict[str, list[NMRSpectrum]], peak_defs, mode: str = "rows", fit_regions=None,
                      baseline_regions=None, baseline_sigma=2.5, baseline_max_iter=10,
                      x_unit='ppm', noise_region=None, normalize=False,
                      show_components=True, positive_intensity=True,
                      allow_negative_intensity=False, sn_threshold=3,
                      show_report=False, show_warnings=True) -> dict:
    """
    Fit one or more extracted 2D NMR slices (rows and/or columns).

    Parameters
    ----------
    slices : dict
        Output of :func:`opennmr.processing.slicing.get_slices`:
        ``{"rows": [NMRSpectrum, ...], "columns": [NMRSpectrum, ...]}``.
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
    normalize : bool, optional
        Normalise each trace before fitting.
    show_components : bool, optional
        Compute individual fitted peak components (needed for
        integrals, and useful for a future plotting layer to show
        them); components are always computed for integration
        internally regardless of this flag, but this controls whether
        they are also stored/returned.
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

    Returns
    -------
    dict with keys ``"rows"`` and ``"columns"``, each a list of
    fit-result dicts with keys: ``slice`` (the trace ``NMRSpectrum``
    itself), ``slice_type``, ``index``, ``shift``, ``popt``, ``pcov``,
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

    for slice_type, trace in selected:

        x = trace.axis_values(0, unit=x_unit.lower())
        y = trace.normalized() if normalize else trace.data

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
        # Fits with local_peak_defs (which may have S/N-gated
        # amplitudes fixed to zero for this slice) -- not the original
        # peak_defs, which would silently ignore that gating.

        popt, pcov_free, is_free, scale, x_ref = fit_spectrum(
            trace, local_peak_defs, fit_regions=fit_regions,
            use_normalised=normalize, baseline_regions=baseline_regions,
            baseline_sigma=baseline_sigma, baseline_max_iter=baseline_max_iter,
            x_unit=x_unit,
            positive_intensity=positive_intensity, allow_negative_intensity=allow_negative_intensity,
        )

        pcov = expand_pcov(pcov_free, is_free, len(popt))

        # Amplitudes are only actually locked >= 0 (in the bounds used
        # above) when positive_intensity=True AND
        # allow_negative_intensity=False -- checking positive_intensity
        # alone would re-clip amplitudes back to zero even when the
        # caller explicitly unlocked them via allow_negative_intensity.
        if positive_intensity and not allow_negative_intensity:
            popt = enforce_positive_intensity(popt, local_peak_defs)

        y_fit, components = evaluate_fit(x, popt, local_peak_defs, x_ref=x_ref, return_components=True)

        integrals, integrals_norm = compute_integrals(x, components)
        errors = compute_integral_errors(popt, pcov, local_peak_defs, integrals)

        metrics = fit_quality(y, y_fit, len(popt))

        lower, upper = build_bounds(local_peak_defs, positive_intensity=positive_intensity,
                                     allow_negative_intensity=allow_negative_intensity)
        warnings = check_bound_hits(popt, pcov, is_free, lower, upper, local_peak_defs)

        slice_info = trace.metadata.raw["slice"]

        fit_result = {
            "slice": trace,
            "slice_type": slice_type,
            "index": slice_info["index"],
            "shift": slice_info["shift"],

            "popt": popt, "pcov": pcov, "is_free": is_free,
            "scale": scale, "x_ref": x_ref,

            "x": x, "y": y, "y_raw": trace.data,
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
