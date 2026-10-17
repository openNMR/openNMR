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

from functools import lru_cache
from math import comb

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

from ..core.spectrum import NMRSpectrum
from ..core import fit_progress as _fit_progress

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


def _select_baseline_candidates(x, y, baseline_regions, frac):
    """Shared first step of every baseline estimator below (robust
    polynomial, sinusoidal, spline): pick the candidate "probably just
    baseline, not signal" points -- either explicit ``baseline_regions``
    (lo, hi) window(s), or, absent those, the first/last ``frac`` of the
    spectrum by point count. Pulled out once three near-identical
    copies of this exact logic (region masking across possibly several
    ranges, the frac-based edge fallback) started drifting apart across
    the different baseline methods -- one implementation, not three to
    keep in sync."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if baseline_regions is not None:
        if isinstance(baseline_regions[0], (int, float, np.integer, np.floating)):
            baseline_regions = [baseline_regions]
        mask = np.zeros_like(x, dtype=bool)
        for lo, hi in baseline_regions:
            lo, hi = sorted([lo, hi])
            mask |= (x >= lo) & (x <= hi)
        return x[mask], y[mask]

    n = len(x)
    k = max(1, int(n * frac))
    return np.concatenate([x[:k], x[-k:]]), np.concatenate([y[:k], y[-k:]])


def estimate_baseline_polynomial(x, y, baseline_regions=None, frac=0.05, sigma=2.5,
                                  max_iter=10, degree=1, x_ref=None):
    """
    Robustly estimate a degree-``degree`` polynomial baseline,
    ``baseline = polyval(coeffs, x - x_ref)`` -- ``degree=1`` (the
    default) is a straight line, the same model
    :func:`estimate_baseline_robust` fits; higher degrees follow a
    real curved/rolling baseline (common on wideline/solid-state
    spectra, and on spectra with a large recorded first-order phase --
    see :func:`opennmr.processing.ft.baseline_correct`'s own
    ``method="polynomial"`` docs) that a straight line structurally
    cannot, without needing a full spline.

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

    degree : int, optional
        Polynomial degree (1 = linear, 2 = quadratic, ...). Must be
        less than the number of surviving candidate points.

    x_ref : float, optional
        Reference x-value. Defaults to the mean of the candidate
        baseline points -- centring the fit here (rather than at
        x=0, often far outside the spectrum's own ppm range) keeps
        the design matrix well-conditioned at higher degrees.

    Returns
    -------
    coeffs, x_ref : ndarray, float
        ``coeffs`` in ``numpy.polyval`` order (highest power first),
        length ``degree + 1``.
    """
    x_base, y_base = _select_baseline_candidates(x, y, baseline_regions, frac)

    if len(x_base) < degree + 2:
        raise ValueError("Not enough points available for baseline estimation.")

    if x_ref is None:
        x_ref = np.mean(x_base)

    xc = x_base - x_ref
    # Vandermonde, highest power first -- matches np.polyval's own
    # coefficient order, so callers can evaluate with polyval directly.
    A = np.vander(xc, degree + 1)
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

        if np.sum(new_keep) < degree + 2:
            break

        if np.array_equal(new_keep, keep):
            break

        keep = new_keep
        coeff, *_ = np.linalg.lstsq(A[keep], y_base[keep], rcond=None)

    return coeff, x_ref


def estimate_baseline_robust(x, y, baseline_regions=None, frac=0.05, sigma=2.5,
                              max_iter=10, x_ref=None):
    """
    Robustly estimate a linear baseline, ``baseline = b0 + b1 * (x - x_ref)``.

    A thin, signature-preserving wrapper around
    :func:`estimate_baseline_polynomial` (``degree=1``) -- kept as its
    own function since ``fit_spectrum``/``NMR2D_fit_relaxation`` (and
    any external code) already depend on this exact ``(b0, b1, x_ref)``
    return shape, not the general ``(coeffs, x_ref)`` one degree-N
    fitting needs.

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
    coeffs, x_ref = estimate_baseline_polynomial(
        x, y, baseline_regions=baseline_regions, frac=frac, sigma=sigma,
        max_iter=max_iter, degree=1, x_ref=x_ref,
    )
    b1, b0 = coeffs  # np.vander/polyval order is [highest power, ..., constant]
    return b0, b1, x_ref


def estimate_baseline_sinusoidal(x, y, baseline_regions=None, frac=0.05, sigma=2.5,
                                  max_iter=10, n_harmonics=1, period=None):
    """
    Robustly estimate a sinusoidal baseline,
    ``baseline = a0 + sum_k a_k*cos(2*pi*k*x/period) + b_k*sin(2*pi*k*x/period)``
    for ``k = 1..n_harmonics`` -- a slowly rolling, wave-like baseline
    that neither a straight line nor a low-degree polynomial
    represents well, without spline's own "follows anything, including
    noise" flexibility.

    Linear in its own parameters (the basis functions are sinusoidal,
    the *coefficients* aren't), so this fits by the same weighted
    least-squares + sigma-clip loop as :func:`estimate_baseline_polynomial`,
    just with a different design matrix.

    Parameters
    ----------
    x, y : ndarray
        Spectrum.

    baseline_regions, frac, sigma, max_iter : see
        :func:`estimate_baseline_polynomial`.

    n_harmonics : int, optional
        Number of sine/cosine harmonics (1 = a single sinusoid).

    period : float, optional
        The fundamental period, in ``x``'s own units. Defaults to
        twice the candidate points' own x-range -- with no other
        periodicity known a priori (unlike, say, a fixed instrumental
        artifact frequency), this treats the observed range as half a
        cycle, i.e. one smooth rise-and-fall by default, the most
        common real "rolling baseline" shape -- rather than guessing
        at a shorter, oscillation-prone period that would just
        overfit noise.

    Returns
    -------
    coeffs, period : ndarray, float
        ``coeffs = [a0, a1, b1, a2, b2, ...]``, length
        ``2*n_harmonics + 1``. Evaluate with
        :func:`evaluate_baseline_sinusoidal`.
    """
    x_base, y_base = _select_baseline_candidates(x, y, baseline_regions, frac)

    min_points = 2 * n_harmonics + 2
    if len(x_base) < min_points:
        raise ValueError("Not enough points available for baseline estimation.")

    if period is None:
        span = x_base.max() - x_base.min()
        period = 2 * span if span > 0 else 1.0

    def _design_matrix(xb):
        cols = [np.ones_like(xb)]
        for k in range(1, n_harmonics + 1):
            w = 2 * np.pi * k / period
            cols.append(np.cos(w * xb))
            cols.append(np.sin(w * xb))
        return np.column_stack(cols)

    A = _design_matrix(x_base)
    coeff, *_ = np.linalg.lstsq(A, y_base, rcond=None)

    keep = np.ones(len(y_base), dtype=bool)
    for _ in range(max_iter):
        residual = y_base - A @ coeff
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        noise = 1.4826 * mad

        if noise <= 0:
            break

        new_keep = np.abs(residual - median) <= sigma * noise

        if np.sum(new_keep) < min_points:
            break

        if np.array_equal(new_keep, keep):
            break

        keep = new_keep
        coeff, *_ = np.linalg.lstsq(A[keep], y_base[keep], rcond=None)

    return coeff, period


def evaluate_baseline_sinusoidal(coeffs, period, x):
    """Evaluate an :func:`estimate_baseline_sinusoidal` fit at ``x``."""
    x = np.asarray(x, dtype=float)
    n_harmonics = (len(coeffs) - 1) // 2
    total = np.full_like(x, coeffs[0])
    for k in range(1, n_harmonics + 1):
        w = 2 * np.pi * k / period
        a_k, b_k = coeffs[2 * k - 1], coeffs[2 * k]
        total = total + a_k * np.cos(w * x) + b_k * np.sin(w * x)
    return total


#: Default Whittaker penalty, for a spectrum of
#: :data:`_WHITTAKER_REFERENCE_POINTS` points. The penalty acts on
#: differences between *neighbouring points*, so the same physical
#: stiffness needs a larger lambda as the point spacing gets finer --
#: hence the rescaling in :func:`estimate_baseline_whittaker` rather
#: than one fixed number.
#:
#: Measured, not guessed. Swept over 1e4..1e13 against a synthetic
#: rolling baseline (a 0.35-amplitude sine plus a slope, span 0.67)
#: carrying a solvent peak 100x the analyte peaks, at 4k/8k/16k/32k
#: points. Best recovery was at 1e9, 1e10, 1e11-1e12 and 1e12-1e13
#: respectively -- i.e. the optimum moves by about a factor of ten per
#: doubling, which is what the ``2 * d - 1`` exponent below predicts
#: (the penalty sums n-d terms each scaling as h**-2d, and h goes as
#: 1/n). At this value every one of those four sizes recovers the
#: baseline to under 3% of its own span while keeping at least 99% of
#: the weakest analyte peak.
_WHITTAKER_DEFAULT_LAM = 1.0e11
_WHITTAKER_REFERENCE_POINTS = 16384


def whittaker_smooth(y, w, lam: float, d: int = 2):
    """One Whittaker smoother solve: the ``z`` minimising
    ``sum(w * (y - z)**2) + lam * sum(diff(z, d)**2)``.

    That is, the weighted least-squares fit to ``y`` that is also
    penalised for its own ``d``-th differences -- ``lam`` sets how
    much smoother-than-the-data the answer has to be. The solution
    satisfies ``(W + lam * D'D) z = W y``, which is symmetric,
    positive-definite and banded with ``d`` superdiagonals.

    Solved with :func:`scipy.linalg.solveh_banded` rather than a
    general sparse solve: banded Cholesky is O(n) here, which matters
    because a baseline fit re-solves this once per reweighting
    iteration on a spectrum that is routinely tens of thousands of
    points.
    """
    from scipy.linalg import solveh_banded

    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = y.size
    if n <= d + 1:
        raise ValueError(f"Need more than {d + 1} points for a {d}-order Whittaker smoother.")

    # Copied, because the cached bands are shared with every other call
    # at this (n, lam, d) -- including, on a pseudo-2D, every other row.
    ab = _whittaker_penalty_bands(n, float(lam), d).copy()
    ab[d] += w
    return solveh_banded(ab, w * y, lower=False)


@lru_cache(maxsize=8)
def _whittaker_penalty_bands(n: int, lam: float, d: int):
    """``lam * D'D`` in ``solveh_banded``'s upper-banded form, where
    ``D`` is the ``d``-th difference operator.

    Cached: it depends only on ``(n, lam, d)``, all of which are fixed
    for an entire baseline fit -- and for every row of a pseudo-2D,
    since they share a point count. Rebuilding it inside the solve
    meant forming a sparse matrix product once per reweighting
    iteration, i.e. ~30 times per row and ~15,000 times for a 512-row
    series, to arrive at the same array each time.

    The cache holds a ``(d+1, n)`` float array per entry -- 1.5 MB at
    64k points -- so the size is deliberately small. Callers must copy
    before adding their weights to the diagonal.
    """
    from scipy.sparse import diags

    # The d-th difference operator, as the (n-d, n) banded matrix whose
    # rows are the binomial coefficients with alternating signs -- the
    # same thing np.diff(z, d) computes.
    coefficients = np.array([(-1) ** k * comb(d, k) for k in range(d + 1)], dtype=float)
    D = diags([np.full(n - d, c) for c in coefficients], offsets=list(range(d + 1)),
              shape=(n - d, n))
    penalty = (lam * (D.T @ D)).tocsr()

    # Upper-triangular banded form: ab[u + i - j, j] == A[i, j], so the
    # main diagonal is ab[u] and superdiagonal k sits in ab[u - k].
    ab = np.zeros((d + 1, n))
    for k in range(d + 1):
        ab[d - k, k:] = penalty.diagonal(k)
    return ab


def estimate_baseline_whittaker(x, y, lam: float | None = None, p: float = 0.001,
                                 d: int = 2, max_iter: int = 30, tol: float = 1e-4,
                                 method: str = "arpls"):
    """Estimate a baseline with an asymmetrically reweighted Whittaker
    smoother -- the "Whittaker filter" baseline, explicitly requested.

    Unlike every other estimator in this module, **this one is not
    given a set of candidate baseline points**. It fits a smooth curve
    to the *whole* spectrum and then iteratively down-weights whatever
    sits above it, so the fit falls to the bottom of the data on its
    own. That is the reason to want it: the sigma-clipped estimators
    all start from "the first and last few percent of the spectrum are
    baseline", which is wrong for a crowded spectrum, one with signal
    at the edges, or one whose baseline rolls in the middle where
    there are no candidate points to see it.

    ``method`` :

    - ``"arpls"`` (default) -- asymmetrically reweighted penalised
      least squares (Baek et al., *Analyst* 2015). Weights come from
      a logistic function of each residual measured against the
      *noise* of the points currently below the fit, so how hard a
      point is pushed down scales with the baseline's own scatter and
      there is no asymmetry parameter to guess. This is the one to use
      on a spectrum with a dominant solvent peak: a fixed asymmetry
      lets an enormous peak drag the baseline up under its own feet,
      while this adapts.
    - ``"asls"`` -- the original asymmetric least squares (Eilers &
      Boelens 2005): points above the fit get weight ``p``, points
      below get ``1 - p``. Simpler and faster to converge, and worth
      having when a specific ``p`` is already known to suit a
      particular kind of spectrum.

    ``lam`` : the smoothness penalty. Higher is stiffer -- follows less
    structure, and eventually becomes a straight line; lower follows
    more, and eventually starts following the peaks it is supposed to
    run underneath. ``None`` (default) uses
    :data:`_WHITTAKER_DEFAULT_LAM`, **rescaled for this spectrum's own
    point count** (as ``lam * (n / 16384) ** (2 * d - 1)``): the
    penalty is on differences between neighbouring points, so a fixed
    number means a very different stiffness at 4k points than at 64k,
    and a default that had to be re-tuned per dataset would not be a
    usable default. See that constant for the measurements the
    exponent and the value come from.

    ``p`` : ``method="asls"`` only -- the weight given to points above
    the current fit. Small (0.001-0.05).

    ``d`` : order of the difference penalty. 2 (default) penalises
    curvature, which is what "smooth baseline" usually means; 1
    penalises slope and gives a flatter, more piecewise-constant
    result.

    ``tol`` : stop once the weights change by less than this, relative
    to their own magnitude.

    ``x`` is accepted for signature compatibility with the other
    estimators and is **not used** -- the penalty is on differences
    between successive points, which assumes they are evenly spaced.
    An NMR frequency axis always is.

    Returns the baseline itself, evaluated at every point of ``y``
    (not a callable, unlike :func:`estimate_baseline_spline`: there is
    no closed form to evaluate anywhere else).
    """
    if method not in ("arpls", "asls"):
        raise ValueError(f"method must be 'arpls' or 'asls', got {method!r}")
    y = np.asarray(y, dtype=float)
    n = y.size
    if n <= d + 1:
        raise ValueError(f"Need more than {d + 1} points for a {d}-order Whittaker baseline.")

    if lam is None:
        lam = _WHITTAKER_DEFAULT_LAM * (n / _WHITTAKER_REFERENCE_POINTS) ** (2 * d - 1)
    if lam <= 0:
        raise ValueError("lam must be positive.")

    # Solved on a unit-scaled copy purely for conditioning: the system
    # is linear in y for fixed weights, and both weighting rules are
    # scale-free, so this changes nothing about the answer.
    scale = float(np.max(np.abs(y)))
    if scale == 0:
        return np.zeros_like(y)
    y_scaled = y / scale

    w = np.ones(n)
    z = y_scaled
    for _ in range(max_iter):
        z = whittaker_smooth(y_scaled, w, lam, d=d)
        residual = y_scaled - z
        negative = residual[residual < 0]
        if negative.size == 0:
            # Nothing below the fit at all -- there is no scatter to
            # measure an asymmetry against, and another iteration
            # cannot move anywhere useful.
            break
        if method == "asls":
            w_new = np.where(residual > 0, p, 1.0 - p)
        else:
            mean, std = float(negative.mean()), float(negative.std())
            if std == 0:
                break
            # Baek et al.'s own logistic: a point well below the fit
            # keeps full weight, one well above it loses essentially
            # all of it, and the transition is measured in units of
            # the below-fit scatter. Clipped before exp() because a
            # strong peak makes the exponent enormous and numpy warns
            # about the overflow on its way to the 0 this wants.
            exponent = np.clip(2.0 * (residual - (2.0 * std - mean)) / std, -500.0, 500.0)
            w_new = 1.0 / (1.0 + np.exp(exponent))
        if np.linalg.norm(w_new - w) < tol * max(np.linalg.norm(w), 1.0):
            w = w_new
            break
        w = w_new
    return z * scale


def find_baseline_points(x, y, half_window=None, threshold=3.0, min_run=None,
                         margin=None):
    """Which points of a spectrum are baseline (no signal), anywhere in it --
    not just at its edges. Golotvin & Williams' test (J. Magn. Reson. 146,
    122, 2000): a point is baseline when the spread (max - min) of the
    spectrum in a window around it stays within ``threshold`` × the
    noise -- signal makes the spread grow, a slowly varying baseline does
    not. The noise is measured from point-to-point differences (robust
    MAD), so a baseline offset or roll does not change it.

    ``half_window`` : points either side (default 0.4% of the spectrum,
    at least 5). ``min_run`` : runs shorter than this are dropped as
    accidents of the noise (default 2 × half_window). ``margin`` : points
    removed beside every signal region, so a peak's wings are never taken
    for baseline (default half_window).

    Returns a boolean array, True for baseline."""
    y = np.real(np.asarray(y, dtype=float))
    n = y.size
    if n < 16:
        return np.ones(n, dtype=bool)
    h = int(half_window) if half_window else max(5, int(round(0.004 * n)))
    d = np.diff(y)
    noise = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2.0)
    if not noise > 0:
        return np.ones(n, dtype=bool)
    # Spread (max - min) over a sliding window, by a strided view.
    pad = np.pad(y, h, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, 2 * h + 1)
    spread = windows.max(axis=1) - windows.min(axis=1)
    # The spread of pure noise over 2h+1 points is about 2 sqrt(2 ln N) σ;
    # ``threshold`` is on top of that.
    expected = 2.0 * np.sqrt(2.0 * np.log(2 * h + 1)) * noise
    base = spread <= expected + threshold * noise
    # Drop short runs, then keep away from signal by ``margin``.
    min_run = int(min_run) if min_run else 2 * h
    edges = np.flatnonzero(np.diff(np.concatenate([[0], base.astype(int), [0]])))
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a < min_run:
            base[a:b] = False
    m = int(margin) if margin is not None else h
    if m:
        signal = ~base
        grown = np.convolve(signal.astype(float), np.ones(2 * m + 1), mode="same") > 0
        base &= ~grown
    return base


def _natural_spline_curve(xs, nx, ny):
    """A natural cubic spline through nodes ``(nx, ny)`` (ascending),
    evaluated on ascending ``xs``, continued past the outer nodes as
    straight lines with the spline's end slopes. Two nodes: a line."""
    from scipy.interpolate import CubicSpline
    nx = np.asarray(nx, dtype=float)
    ny = np.asarray(ny, dtype=float)
    if len(nx) == 2:
        slope = (ny[1] - ny[0]) / (nx[1] - nx[0])
        return ny[0] + slope * (xs - nx[0])
    spline = CubicSpline(nx, ny, bc_type="natural")
    deriv = spline.derivative()
    left = ny[0] + float(deriv(nx[0])) * (xs - nx[0])
    right = ny[-1] + float(deriv(nx[-1])) * (xs - nx[-1])
    return np.where(xs < nx[0], left, np.where(xs > nx[-1], right, spline(xs)))


def spline_through_points(x, y, node_x, half_width=None):
    """A cubic-spline baseline through points the user chose
    (Process ▸ Baseline ▸ Spline ▸ points chosen on the plot; user
    request, 2026-10-17).

    Only the *positions* ``node_x`` (ppm) are the user's: each node's
    height is the median of the trace within ``half_width`` points of it
    (default: 0.25 % of the spectrum, at least 3 points), read from
    whatever trace is being corrected -- so the same points serve every
    slice of a series and survive re-phasing. The curve is that of
    :func:`auto_spline_baseline` (natural cubic, straight past the end
    nodes). Nodes outside the spectrum are dropped; nodes closer than a
    point apart are merged.

    Returns ``{"baseline", "nodes_x", "nodes_y"}``. Raises ``ValueError``
    with fewer than two usable points."""
    x = np.asarray(x, dtype=float)
    y = np.real(np.asarray(y, dtype=float))
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n = xs.size
    hw = int(half_width) if half_width else max(3, int(round(0.0025 * n)))
    picks = sorted(float(v) for v in (node_x or []) if xs[0] <= float(v) <= xs[-1])
    nx, ny, last = [], [], None
    for value in picks:
        i = int(np.clip(np.searchsorted(xs, value), 0, n - 1))
        if last is not None and i == last:
            continue
        last = i
        lo, hi = max(0, i - hw), min(n, i + hw + 1)
        nx.append(float(xs[i]))
        ny.append(float(np.median(ys[lo:hi])))
    if len(nx) < 2:
        raise ValueError("Choose at least two baseline points on the plot.")
    curve = _natural_spline_curve(xs, nx, ny)
    out = np.empty_like(curve)
    out[order] = curve
    return {"baseline": out, "nodes_x": np.asarray(nx), "nodes_y": np.asarray(ny)}


def auto_spline_baseline(x, y, n_points=None, half_window=None, threshold=3.0):
    """A cubic-spline baseline through points placed automatically on the
    signal-free parts of the spectrum (explicit user request, 2026-10-06).

    1. :func:`find_baseline_points` finds the baseline anywhere in the
       spectrum (not only at its edges).
    2. The axis is cut into ``n_points`` equal stretches (default 20, one
       per 5% of the spectrum); in each stretch with
       baseline in it, one node is placed at the median position of its
       baseline points, at their median height (robust to noise and to a
       stray spike) -- plus one at each end of the spectrum, from its
       outermost baseline points.
    3. A natural cubic spline goes through the nodes; past the outermost
       node it continues as a straight line with the spline's end slope
       (the natural spline's own extension: its curvature is zero there).
       A cubic extrapolated past its data swings wildly, and holding the
       end value flat was measured 18 σ off on a sloping baseline whose
       edge held a peak.

    Returns a dict: ``baseline`` (on ``x``), ``nodes_x``, ``nodes_y`` (the
    automatic points, ascending x), ``baseline_points`` (the mask from
    step 1)."""
    from scipy.interpolate import CubicSpline
    x = np.asarray(x, dtype=float)
    y = np.real(np.asarray(y, dtype=float))
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    base = find_baseline_points(xs, ys, half_window=half_window, threshold=threshold)
    if base.sum() < 4:
        raise ValueError("Too little of the spectrum is baseline to place spline points on: "
                         "use Whittaker, or draw baseline regions.")
    n = int(n_points) if n_points else 20
    cuts = np.linspace(0, xs.size, n + 1).astype(int)
    nx, ny = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        sel = base[a:b]
        if sel.sum() < 3:
            continue
        px, py = xs[a:b][sel], ys[a:b][sel]
        nx.append(float(np.median(px)))
        ny.append(float(np.median(py)))
    # A node at each end of the spectrum too, from its outermost baseline
    # points: otherwise the curve is held flat from the first stretch's
    # middle outwards (measured: 10 σ off at the edge of a sloping baseline).
    idx = np.flatnonzero(base)
    k = max(3, min(idx.size // 20, int(round(0.01 * xs.size))))
    stretch = (xs[-1] - xs[0]) / n if n else 0.0
    for ends in (idx[:k], idx[-k:]):
        ex, ey = float(np.median(xs[ends])), float(np.median(ys[ends]))
        if all(abs(ex - v) > 0.3 * stretch for v in nx):
            nx.append(ex)
            ny.append(ey)
    if len(nx) < 2:
        raise ValueError("Fewer than two stretches of the spectrum hold baseline: use fewer "
                         "points, Whittaker, or draw baseline regions.")
    order_n = np.argsort(nx)
    nx, ny = np.asarray(nx)[order_n], np.asarray(ny)[order_n]
    if len(nx) == 2:
        curve = np.interp(xs, nx, ny)
    else:
        spline = CubicSpline(nx, ny, bc_type="natural")
        slope = spline.derivative()
        left = ny[0] + float(slope(nx[0])) * (xs - nx[0])
        right = ny[-1] + float(slope(nx[-1])) * (xs - nx[-1])
        curve = np.where(xs < nx[0], left, np.where(xs > nx[-1], right, spline(xs)))
    out = np.empty_like(curve)
    out[order] = curve
    mask = np.empty_like(base)
    mask[order] = base
    return {"baseline": out, "nodes_x": nx, "nodes_y": ny, "baseline_points": mask}


#: Most candidate points a smoothing-spline baseline is fitted through
#: (see :func:`estimate_baseline_spline`).
_SPLINE_MAX_NODES = 400


def _bin_medians(x, y, max_points):
    """``(x, y)`` (ascending ``x``) cut to at most ``max_points``: each
    output point is the median of a run of consecutive inputs. Returned
    unchanged when already short enough."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size <= max_points:
        return x, y
    cuts = np.linspace(0, x.size, max_points + 1).astype(int)
    xs = np.array([np.median(x[a:b]) for a, b in zip(cuts[:-1], cuts[1:]) if b > a])
    ys = np.array([np.median(y[a:b]) for a, b in zip(cuts[:-1], cuts[1:]) if b > a])
    xs, idx = np.unique(xs, return_index=True)
    return xs, ys[idx]


def _robust_noise(y):
    """Noise σ of ``y`` from its point-to-point differences (MAD), which a
    slow baseline trend barely affects."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return float(np.std(y))
    d = np.diff(y)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def estimate_baseline_spline(x, y, baseline_regions=None, frac=0.05, sigma=2.5,
                              max_iter=10, smoothing=None) -> UnivariateSpline:
    """Robustly estimate a smooth (not necessarily linear) baseline as
    a cubic smoothing spline through the candidate baseline points.

    Same candidate-region selection as :func:`estimate_baseline_robust`
    (``baseline_regions`` if given, else the first/last ``frac`` of the
    spectrum), but the outlier-rejection loop scores each iteration's
    residual against the *current spline fit* rather than a straight
    line -- a real, curved (rolling/sloping) baseline's own points
    would otherwise show large residuals against a forced-linear fit
    and get wrongly sigma-clipped out as if they were peaks, which is
    exactly the failure mode a straight line can't help with: it
    doesn't just fit a curved baseline badly, it actively throws away
    the points needed to fit it well at all.

    ``smoothing`` : ``UnivariateSpline``'s own ``s`` parameter --
    higher means smoother/less sensitive to point-to-point noise,
    ``0`` interpolates exactly through every surviving point (almost
    never wanted for a baseline: it would fit noise, not just its
    trend). ``None`` (default) picks a data-driven value from the
    candidate points' own residual noise (the same robust estimate the
    rejection loop itself converges on), following the scaling
    ``scipy`` itself recommends (``s`` on the order of the point count
    when residuals have unit variance) -- reasonable without the user
    needing to guess a number, but still a real, overridable knob for
    a baseline that needs to follow more or less local curvature than
    the default finds.

    Returns the fitted ``UnivariateSpline`` itself (call it with any
    x array, including ones outside the candidate points' own range --
    ``ext=3`` clips to the boundary value rather than extrapolating
    the spline's own polynomial wildly past its fitted domain)."""
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

    # UnivariateSpline requires strictly increasing x.
    order = np.argsort(x_base)
    x_base, y_base = x_base[order], y_base[order]
    x_base, unique_idx = np.unique(x_base, return_index=True)
    y_base = y_base[unique_idx]

    if len(x_base) < 4:
        raise ValueError("Not enough points available for baseline estimation.")

    # At most _SPLINE_MAX_NODES candidates, each the median of a run of
    # neighbours (user report, 2026-10-17: Spline froze openNMR on an
    # ordinary 1D). The 5 % edges of a 64k spectrum are ~6,500 points,
    # and FITPACK's knot search is far worse than linear in them; a
    # median of each run also keeps a stray spike out of the fit.
    x_base, y_base = _bin_medians(x_base, y_base, _SPLINE_MAX_NODES)

    def _fit(xb, yb, s):
        k = 3 if len(xb) > 3 else 1  # cubic once there are enough points, else linear
        return UnivariateSpline(xb, yb, k=k, s=s, ext=3)

    keep = np.ones(len(y_base), dtype=bool)
    # The first fit's s from the points' own noise, as in the loop below.
    # It used to be len(x) alone -- right only for unit-variance noise, so
    # on real intensities (~1e4-1e7) it asked FITPACK to all but
    # interpolate every point: a very slow knot search, then a curve that
    # swung wildly between the two edges.
    noise0 = _robust_noise(y_base)
    spline = _fit(x_base, y_base, smoothing if smoothing is not None
                  else len(x_base) * max(noise0, np.finfo(float).tiny) ** 2)

    for _ in range(max_iter):
        residual = y_base - spline(x_base)
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        noise = 1.4826 * mad

        if noise <= 0:
            break

        new_keep = np.abs(residual - median) <= sigma * noise

        if np.sum(new_keep) < 4:
            break
        if np.array_equal(new_keep, keep):
            break

        keep = new_keep
        s = smoothing if smoothing is not None else len(x_base[keep]) * noise ** 2
        spline = _fit(x_base[keep], y_base[keep], s)

    return spline


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
# Auto peak-picking
# ============================================================

def auto_pick_peaks(spectrum: NMRSpectrum, x_unit: str = "ppm", prominence: float = 0.05,
                     min_distance_fwhm: float = 1.0, max_peaks: int | None = None) -> list[dict]:
    """Detect peaks in a 1D spectrum automatically, as a fast starting
    point for :func:`fit_spectrum`/:func:`NMR2D_fit_slices` -- a much
    lighter-weight alternative to manually entering (or clicking) every
    peak's approximate position first. A cleaned-up, decoupled version
    of the peak-finding core of ``leonmr``'s ``xf2_peak_pick`` (which
    also plotted and tracked intensities across a whole pseudo-2D
    stack -- neither of which belongs here; this only finds peaks in
    one spectrum).

    ``prominence`` : minimum peak prominence, as a fraction of the
    spectrum's own peak-to-peak range (0-1) -- filters out noise
    bumps. Lower finds more (weaker) peaks; higher finds fewer.

    ``min_distance_fwhm`` : minimum spacing between two detected peaks,
    as a multiple of *each* peak's own estimated FWHM -- merges/drops
    a peak that's really just a shoulder or noise spike right next to
    a stronger one, rather than double-counting it. Uses each
    detected peak's own half-prominence width as a first-pass FWHM
    estimate for this filtering.

    ``max_peaks`` : if given, keep only this many, strongest-first.

    Returns a list of peak-def dicts (``{"type": "pseudo_voigt",
    "shift": ..., "fwhm": ..., "eta": 0.5}``), sorted by position --
    directly usable as :func:`fit_spectrum`'s ``peak_defs`` (or, in the
    GUI, to populate the Analysis tab's peak table) as an initial
    guess; every value here still only seeds the fit, same as a
    manually-entered peak.
    """
    from scipy.signal import find_peaks, peak_widths

    x = spectrum.axis_values(0, unit=x_unit.lower())
    y = np.asarray(spectrum.normalized(), dtype=np.float64)

    y_range = np.ptp(y)
    if y_range <= 0:
        return []
    y_norm = (y - np.min(y)) / y_range

    dx = np.abs(np.mean(np.diff(x))) if len(x) > 1 else 1.0

    indices, properties = find_peaks(y_norm, prominence=prominence)
    if len(indices) == 0:
        return []

    widths_pts, *_ = peak_widths(y_norm, indices, rel_height=0.5)
    order = np.argsort(-properties["prominences"])  # strongest first, for max_peaks/distance filtering

    kept_positions: list[float] = []
    kept: list[dict] = []
    for i in order:
        shift = float(x[indices[i]])
        fwhm = max(float(widths_pts[i]) * dx, dx)  # at least one sample wide
        if any(abs(shift - other) < min_distance_fwhm * fwhm for other in kept_positions):
            continue
        kept_positions.append(shift)
        kept.append({"type": "pseudo_voigt", "shift": shift, "fwhm": fwhm, "eta": 0.5})
        if max_peaks is not None and len(kept) >= max_peaks:
            break

    kept.sort(key=lambda p: p["shift"])
    return kept


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
    # A width with no upper limit of its own is capped at twice the whole
    # spectrum's span: no lineshape is broader than that, and without a
    # cap a fit whose amplitude is pinned at a bound (amplitude_bound_
    # regions, a manual limit) widens the peak forever towards a flat
    # line and never converges -- seen as "did not converge within 2,000
    # iterations" where the capped amplitude was the right answer
    # (2026-10-17).
    span = float(np.ptp(x_full)) if np.size(x_full) > 1 else 0.0
    if span > 0:
        index = 0
        for peak in peak_defs:
            step = 3 if peak["type"] in ["gaussian", "lorentzian"] else 4
            if not np.isfinite(upper[index + 1]):
                upper[index + 1] = 2.0 * span
            index += step
    lower = np.append(lower, [-np.inf, -np.inf])
    upper = np.append(upper, [np.inf, np.inf])

    assert len(p0) == len(is_free) == len(lower) == len(upper), "parameter length mismatch"

    # p0's own data-driven guess (initialize_params_with_map -- e.g. an
    # amplitude read straight off the spectrum at each peak's shift) is
    # built with no knowledge of bounds, and a tight bound -- a manual
    # "Set Bounds..." entry, or NMR2D_fit_slices's own
    # amplitude_bound_regions -- can leave it outside its own box.
    # scipy's bounded curve_fit raises immediately in that case ("Initial
    # guess is outside of provided bounds") rather than clipping it
    # itself, so it has to happen here. Only clips the *free* subset --
    # a fixed parameter's value must never be silently altered, and
    # nothing outside is_free is ever passed to curve_fit anyway.
    p0[is_free] = np.clip(p0[is_free], lower[is_free], upper[is_free])

    def model_wrapper(x, *free_params):
        full = reconstruct_full(p0, free_params, is_free)
        return build_model(x, full, peak_defs, x_ref)

    try:
        popt_free, pcov_free = _fit_progress.limited_curve_fit(
            _fit_progress.tracked(model_wrapper, y, n_free=int(is_free.sum())), x, y, p0=p0[is_free], bounds=(lower[is_free], upper[is_free]),
            label="The spectral fit",
        )
    except _fit_progress.FitNotConverged as exc:
        # One retry from the best point reached, with each parameter
        # scaled by its own sensitivity (x_scale="jac"). A parameter
        # pinned at a bound (an amplitude capped by amplitude_bound_
        # regions) leaves the others creeping along a nearly flat
        # direction, which unscaled TRF crawls along until the limit
        # (2026-10-17). Only a fit that failed is retried, so every
        # converging fit is exactly as before; if the retry fails too,
        # the original report stands.
        start = p0[is_free] if exc.best_params is None or len(exc.best_params) != int(is_free.sum()) \
            else np.clip(exc.best_params, lower[is_free], upper[is_free])
        try:
            popt_free, pcov_free = _fit_progress.limited_curve_fit(
                _fit_progress.tracked(model_wrapper, y, n_free=int(is_free.sum())), x, y, p0=start,
                bounds=(lower[is_free], upper[is_free]), label="The spectral fit", x_scale="jac",
            )
        except _fit_progress.FitNotConverged:
            raise exc from None

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

#: Area under a unit-height (A=1) peak of unit FWHM, for each
#: lineshape -- i.e. ``area = A * fwhm * <this factor>`` (Gaussian:
#: sigma = fwhm / (2*sqrt(2*ln2)), area = A*sigma*sqrt(2*pi), which
#: simplifies to A*fwhm*sqrt(pi/(4*ln2)); Lorentzian: gamma = fwhm/2,
#: area = A*gamma*pi = A*fwhm*(pi/2)). Already independently verified
#: in this project against a hand-computed value (see the fit-report
#: CSV export's own writeup in TODO.md) -- reused here, not re-derived.
_GAUSSIAN_AREA_FACTOR = float(np.sqrt(np.pi / (4 * np.log(2))))
_LORENTZIAN_AREA_FACTOR = np.pi / 2


def _peak_area_per_unit_amplitude(peak: dict) -> float:
    """Area contributed per unit amplitude (peak height), for one
    peak's own lineshape/FWHM -- i.e. ``area = A * this``. Used by
    :func:`NMR2D_fit_slices`'s ``amplitude_bound_regions`` to convert a
    raw, measured region integral into an amplitude upper bound."""
    fwhm = peak["fwhm"]
    if peak["type"] == "gaussian":
        return fwhm * _GAUSSIAN_AREA_FACTOR
    if peak["type"] == "lorentzian":
        return fwhm * _LORENTZIAN_AREA_FACTOR
    eta = peak.get("eta", 0.5)
    return fwhm * (eta * _LORENTZIAN_AREA_FACTOR + (1 - eta) * _GAUSSIAN_AREA_FACTOR)


def _reseed_peak_position(peak: dict, seed: dict) -> None:
    """Update ``peak`` in place to start from ``seed``'s own converged
    shift/FWHM(/eta) -- :func:`NMR2D_fit_slices`'s own ``seed_mode``
    helper. Also *recentres* any existing ``bounds["shift"]`` window
    onto the new shift, keeping its width unchanged, rather than
    leaving it where it was. That window is normally set once by a
    caller (e.g. the GUI's own position-tolerance control) relative to
    the *original* table guess -- left untouched, it would silently
    cap how far ``seed_mode="propagate"``/``"apply_first"`` can
    actually follow a drifting peak once the drift exceeds that
    original window, defeating the entire point of seeding a new
    starting guess in the first place (found by real end-to-end
    testing through the GUI's own call path, not just this function in
    isolation -- a bare backend call with no bounds at all never
    exercised this interaction)."""
    bounds = peak.get("bounds")
    if bounds and "shift" in bounds:
        lo, hi = bounds["shift"]
        half_width = (hi - lo) / 2
        peak["bounds"] = {**bounds, "shift": (seed["shift"] - half_width, seed["shift"] + half_width)}
    peak["shift"], peak["fwhm"] = seed["shift"], seed["fwhm"]
    if "eta" in seed:
        peak["eta"] = seed["eta"]


def NMR2D_fit_slices(slices: dict[str, list[NMRSpectrum]], peak_defs, mode: str = "rows", fit_regions=None,
                      baseline_regions=None, baseline_sigma=2.5, baseline_max_iter=10,
                      x_unit='ppm', noise_region=None, normalize=False,
                      show_components=True, positive_intensity=True,
                      allow_negative_intensity=False, sn_threshold=3,
                      show_report=False, show_warnings=True, seed_mode: str = "independent",
                      anchor_slice: int | None = None,
                      amplitude_bound_regions=None) -> dict:
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
    anchor_slice : int or {"first", "last"}, optional
        Which slice ``seed_mode="apply_first"`` takes its fixed
        lineshape from, as a 1-based slice number. Default (``None``)
        is the first, which is this function's own long-standing
        behaviour.

        Worth choosing rather than accepting: the anchor should be the
        slice with the **best signal**, since every other slice is
        pinned to its lineshape. That is the first slice of a PFG or T2
        series (least attenuated) but the *last* of an inversion- or
        saturation-recovery series (fully recovered) -- anchoring an
        inversion recovery on its first slice pins every peak to a
        lineshape measured near the null, where there is almost no
        signal to measure it from.

        Ignored for the other two seed modes, which have no fixed
        lineshape to take from anywhere.
    seed_mode : {"independent", "propagate", "apply_first"}, optional
        How each slice after the first gets its own starting
        shift/FWHM/eta -- never amplitude, which is expected to vary
        across a relaxation/diffusion series (that's the point of one);
        S/N-based amplitude gating above still applies exactly the same
        regardless of this setting.

        ``"independent"`` (default): every slice starts from the same
        user-supplied ``peak_defs``, unchanged -- this function's own
        original, only behaviour before this parameter existed.

        ``"propagate"``: each slice's own *converged* shift/FWHM/eta
        becomes the *next* slice's starting guess (never fixed, just a
        better initial guess than the original one) -- useful when a
        peak might slowly drift position/width across a series (e.g.
        a temperature-dependent shift), or a later, weaker slice needs
        a closer starting point than the first slice's own guess to
        converge reliably at all.

        ``"apply_first"``: the anchor slice (the first by default --
        see ``anchor_slice``) fits normally (from
        ``peak_defs``, shift/FWHM/eta free); every other slice
        has shift/FWHM/eta *fixed* at the anchor slice's own converged
        values (added into that slice's own ``fixed`` dict, without
        overwriting whatever the caller's own ``peak_defs`` already
        marked fixed there) -- only amplitude (and the baseline) is
        actually re-optimised per slice. Useful when the peak's shape
        genuinely shouldn't change across the series and only
        intensity should, and is both the fastest option and often the
        most robust one for a slice too weak/noisy to reliably
        converge its own shift/width from scratch.
    amplitude_bound_regions : list of (float, float), optional
        Region-derived amplitude upper bounds -- for each peak whose
        *starting* shift falls inside one of these ``(lo, hi)``
        regions (typically the same list as ``fit_regions``, e.g. the
        GUI's own Integration Regions table), this slice's own raw
        (unfitted) data is trapezoidally integrated over that region,
        then converted to an amplitude bound via the peak's own
        lineshape/FWHM (:func:`_peak_area_per_unit_amplitude`) and
        used as an upper bound on that peak's amplitude for *this*
        slice -- recomputed fresh per slice, so it tracks a
        genuinely-varying true amplitude across a relaxation/diffusion
        series rather than freezing in whatever the first slice
        happened to measure. Deliberately loose, not a tight per-peak
        estimate: an entire region's own raw integral is used as-is
        (no extra margin) as the cap for *each* peak inside it, since
        that's already a true physical upper bound for any single
        peak sharing the region (it can't individually contain more
        signal than the whole region measures) even when several
        peaks/baseline share it. When a peak's shift falls inside more
        than one given region, the tightest (smallest) resulting bound
        is used. A region whose raw integral comes out <= 0 (e.g. a
        peak decayed into the noise) is skipped for that peak/slice
        rather than forcing a near-zero cap -- S/N-based gating
        (``noise_region``/``sn_threshold``) already exists as the
        deliberate, explicit tool for "fix amplitude to zero once a
        peak is gone"; this shouldn't silently do that as a side
        effect of an unrelated, mildly-negative trapz artifact. Only
        ever *adds* an upper bound where none already exists -- a
        peak's own explicit ``bounds["amplitude"]``/``bounds["A"]``
        (e.g. from the GUI's "Set Bounds..." dialog) always wins
        outright and is never overridden. ``None`` (default): off,
        identical to this parameter never having existed.

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
    if seed_mode not in ("independent", "propagate", "apply_first"):
        raise ValueError("seed_mode must be 'independent', 'propagate', or 'apply_first'")

    selected = []
    if mode in ["rows", "both"]:
        selected.extend(("row", s) for s in slices.get("rows", []))
    if mode in ["columns", "both"]:
        selected.extend(("column", s) for s in slices.get("columns", []))

    if not selected:
        raise ValueError("No slices available for fitting.")

    # The anchor is fitted *first* whatever its position, so every
    # other slice can be fixed to its converged lineshape -- including
    # the slices before it, which a plain in-order pass could not have
    # done. Results are put back in the original order afterwards, so
    # nothing downstream (the delay/b-value pairing above all) can tell
    # the difference; getting that wrong would silently pair each
    # intensity with the wrong delay.
    # NOT called `order`: the per-slice loop below already binds that
    # name to a numpy argsort of the x-axis, so a plain `order` here is
    # silently rebound on the first slice and the restore at the end
    # then compares an array against a list -- ValueError: the truth
    # value of an array with more than one element is ambiguous, raised
    # out of a fit that had otherwise completed.
    slice_order = list(range(len(selected)))
    if seed_mode == "apply_first" and anchor_slice is not None:
        position = _anchor_position(anchor_slice, len(selected))
        slice_order = [position] + [i for i in slice_order if i != position]
        selected = [selected[i] for i in slice_order]

    fitted = {"rows": [], "columns": []}

    # Only ever read/written when seed_mode != "independent" -- see
    # that parameter's own docstring above for what each holds and why.
    previous_converged = None
    first_converged = None

    for k_slice, (slice_type, trace) in enumerate(selected):
        if len(selected) > 1:
            _fit_progress.set_stage(f"{slice_type} {k_slice + 1} of {len(selected)}")

        x = trace.axis_values(0, unit=x_unit.lower())
        y = trace.normalized() if normalize else trace.data

        local_peak_defs = [p.copy() for p in peak_defs]

        if seed_mode == "propagate" and previous_converged is not None:
            for peak, seed in zip(local_peak_defs, previous_converged):
                _reseed_peak_position(peak, seed)
        elif seed_mode == "apply_first" and first_converged is not None:
            for peak, seed in zip(local_peak_defs, first_converged):
                _reseed_peak_position(peak, seed)
                fixed = dict(peak.get("fixed", {}))
                fixed["shift"] = True
                fixed["fwhm"] = True
                if "eta" in seed:
                    fixed["eta"] = True
                peak["fixed"] = fixed

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

        # ----- region-integral-derived amplitude bounds -----

        if amplitude_bound_regions:

            order = np.argsort(x)
            x_sorted, y_sorted = x[order], y[order]

            for peak in local_peak_defs:

                existing = peak.get("bounds", {})
                if "amplitude" in existing or "A" in existing:
                    continue  # an explicit user bound always wins outright

                best_bound = None
                for lo, hi in amplitude_bound_regions:
                    if not (lo <= peak["shift"] <= hi):
                        continue
                    mask = (x_sorted >= lo) & (x_sorted <= hi)
                    if not np.any(mask):
                        continue
                    integral = _trapz(y_sorted[mask], x_sorted[mask])
                    if integral <= 0:
                        continue
                    candidate = integral / _peak_area_per_unit_amplitude(peak)
                    if best_bound is None or candidate < best_bound:
                        best_bound = candidate

                if best_bound is not None:
                    peak["bounds"] = {**existing, "amplitude": (-np.inf, best_bound)}

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

        if seed_mode != "independent":
            converged = [
                {name: value for name, value in zip(peak["names"], peak["params"]) if name in ("shift", "fwhm", "eta")}
                for peak in extract_params_and_errors(popt, pcov, local_peak_defs, is_free)
            ]
            previous_converged = converged
            if first_converged is None:
                first_converged = converged

        if show_report or show_warnings:
            NMR2D_print_fit_report(
                fit_result, peak_defs=local_peak_defs,
                slice_index=fit_result["index"], slice_shift=fit_result["shift"],
                show_warnings=show_warnings,
            )

        fitted["rows" if slice_type == "row" else "columns"].append(fit_result)

    if slice_order != list(range(len(slice_order))):
        for key in ("rows", "columns"):
            if len(fitted[key]) == len(slice_order):
                restored = [None] * len(slice_order)
                for fitted_position, original_position in enumerate(slice_order):
                    restored[original_position] = fitted[key][fitted_position]
                fitted[key] = restored

    return fitted


def _anchor_position(anchor_slice, n_slices: int) -> int:
    """Resolve an anchor to a 0-based position among the slices being
    fitted. Accepts ``"first"``, ``"last"``, or a 1-based slice number
    (clamped, so a stale anchor left over from a longer series does not
    raise -- it just falls on the nearest real slice)."""
    if anchor_slice == "first":
        return 0
    if anchor_slice == "last":
        return n_slices - 1
    return int(np.clip(int(anchor_slice) - 1, 0, n_slices - 1))


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
