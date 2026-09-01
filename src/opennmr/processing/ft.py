"""
Vendor-agnostic FID -> spectrum processing: small, independent,
composable steps (roll/align, apodize, zero-fill, FFT, phase, baseline,
linear prediction) rather than one fixed pipeline function, so
different acquisition styles (a plain decaying FID vs. a full echo that
needs aligning first, for example) can be built from the same parts.

Used directly by ``opennmr.io.xpulse`` (X-Pulse's raw FID needs the
whole pipeline) and for the shared, non-Bruker-specific finishing
touches of ``opennmr.io.bruker``'s raw-FID path (``autophase01`` as a
fallback when no interactive phase was recorded, ``baseline_correct``).
Kept as its own module -- not folded into ``io.xpulse`` -- so future
JEOL/Varian importers (also raw-FID formats) can reuse it too.

The phase-scoring/autophasing logic here (``phase_score``,
``apply_phase``, ``autophase01``) is ported from ``leonmr``'s
``xpulse_processing.py``, which was already written to be format-
agnostic. ``linear_predict`` is ported from the same module.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal.windows import gaussian as _scipy_gaussian


# ============================================================
# Roll / echo alignment
# ============================================================

def roll_fid(fid: np.ndarray, shift: float) -> np.ndarray:
    """Circularly shift a complex FID by ``shift`` samples. ``shift``
    may be fractional (a Fourier-domain phase ramp is used in that
    case); an integer ``shift`` uses a plain ``np.roll``."""
    if float(shift).is_integer():
        return np.roll(fid, int(shift))
    n = len(fid)
    freqs = np.fft.fftfreq(n)
    return np.fft.ifft(np.fft.fft(fid) * np.exp(-2j * np.pi * freqs * shift))


def find_echo_top(complex_fid: np.ndarray) -> float:
    """Fractional sample index of a full echo's peak, via parabolic
    interpolation of ``|complex_fid|`` around its maximum -- for
    signals that rise then fall (a full echo) rather than a one-sided
    decaying FID."""
    magnitude = np.abs(complex_fid)
    i = int(np.argmax(magnitude))
    if 0 < i < len(magnitude) - 1:
        y0, y1, y2 = magnitude[i - 1], magnitude[i], magnitude[i + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    return i + delta


def align_echo(fid: np.ndarray, target_index: int = 0) -> np.ndarray:
    """Roll a full-echo FID so its peak lands at ``target_index``
    (default: 0, i.e. t=0) -- :func:`find_echo_top` followed by a
    (possibly fractional) :func:`roll_fid`.

    Default 0, not the array centre: these acquisitions record a
    gradient/spin echo starting some fixed dead time before the true
    echo top, so after this roll the signal from the echo top onward
    looks like an ordinary one-sided decaying FID -- exactly what the
    rest of the pipeline (window/zero-fill/FFT) already assumes.
    Ported from (and validated against) ``leonmr``'s
    ``xpulse_imaging.py``: ``xpulse_image_import`` rolls by
    ``-(echo_shift_points + echo_shift_subsample)``, i.e. to index 0 --
    rolling to the centre instead (this function's behaviour before
    this was corrected) leaves the pre-echo portion wrapped into the
    middle of the array rather than discarded past the end, which is
    not what "aligned like an FID" means for the window/zero-fill/FFT
    steps that follow.
    """
    top = find_echo_top(fid)
    return roll_fid(fid, target_index - top)


def prepare_fid(fid: np.ndarray, *, fid_shift: float = 0.0, align_echo_enabled: bool = False,
                 truncate: int | None = None, linear_predict_points: int = 0,
                 linear_predict_order: int = 20, linear_predict_direction: str = "forward") -> np.ndarray:
    """Shift -> echo-align -> truncate -> linear-predict, in that
    order, on a raw time-domain FID -- everything that happens to the
    FID *before* windowing.

    Extracted from (and now shared by) ``io.xpulse.read_1d``'s own
    step sequence -- this exact ordering was hard-won there: rolling
    the echo top to index 0 wraps the pre-echo portion (dead time,
    then the rise to the top -- roughly a mirror image of the
    post-echo decay for a symmetric echo) onto the array's tail, so
    ``truncate`` has to run *after* that roll, capped at the wrap
    point (``len(fid) - ceil(echo_top)``), not at whatever value it
    would otherwise be -- see :func:`align_echo`'s own docstring for
    the full story. Getting this order (or the cap) wrong doesn't
    crash, it silently produces an FID with a mirror-image artifact
    spliced onto its tail once FFT'd. Having exactly one implementation
    of this sequence (rather than each caller re-deriving it) is the
    whole point of pulling it out here -- used by every vendor reader
    that supports these steps *and* by the GUI's own live Processing-
    preview (:meth:`opennmr.gui.main_window.MainWindow._compute_
    processing_preview_spectrum`), so a fast in-memory preview can't
    silently drift from what a real reload would actually produce.

    ``truncate`` : ``None`` (no truncation) or an explicit point
    count -- unlike the vendor readers' own ``truncate`` parameter,
    there's no ``True`` "use the file's own recorded default" here;
    resolve that to a concrete int before calling.

    ``align_echo_enabled`` -- named this way, not ``align_echo``, so
    it doesn't shadow the :func:`align_echo` function this same
    function calls."""
    if fid_shift:
        fid = roll_fid(fid, fid_shift)

    echo_wrap_limit = None
    if align_echo_enabled:
        top = find_echo_top(fid)
        fid = align_echo(fid)
        echo_wrap_limit = len(fid) - int(np.ceil(top))

    tdeff_points = truncate
    if echo_wrap_limit is not None:
        tdeff_points = echo_wrap_limit if tdeff_points is None else min(tdeff_points, echo_wrap_limit)
    if tdeff_points is not None and tdeff_points < len(fid):
        fid = fid[:tdeff_points]

    if linear_predict_points:
        # order must be < len(fid)-1 (see linear_predict's docstring) --
        # clamp rather than let a fixed UI/caller default (e.g. 20) raise
        # once truncate/align_echo has shrunk fid below that, silently
        # skipping only if even order=1 isn't valid for what's left.
        max_valid_order = len(fid) - 2
        if max_valid_order >= 1:
            fid = linear_predict(
                fid, order=min(linear_predict_order, max_valid_order),
                n_predict=linear_predict_points, direction=linear_predict_direction,
            )

    return fid


# ============================================================
# Apodization -- pluggable window registry
# ============================================================

def _window_exponential(n: int, dt: float, lb: float = 0.0) -> np.ndarray:
    """Exponential multiplication (line broadening ``lb``, in Hz)."""
    t = np.arange(n) * dt
    return np.exp(-t * np.pi * lb)


def _window_hann(n: int, dt: float) -> np.ndarray:
    return np.hanning(n)


def _window_gaussian(n: int, dt: float, std: float | None = None) -> np.ndarray:
    if std is None:
        std = n / 6.0
    return _scipy_gaussian(n, std)


def _window_sine(n: int, dt: float) -> np.ndarray:
    return np.sin(np.pi * np.arange(n) / n)


def _window_qsine(n: int, dt: float) -> np.ndarray:
    return np.sin(np.pi * np.arange(n) / n) ** 2


_WINDOWS = {
    "exponential": _window_exponential,
    "hann": _window_hann,
    "gaussian": _window_gaussian,
    "sine": _window_sine,
    "qsine": _window_qsine,
}


def apodize(fid: np.ndarray, dt: float, window: str = "exponential", **params) -> np.ndarray:
    """Multiply ``fid`` by a named window function.

    ``window`` : one of ``_WINDOWS`` (``"exponential"`` (default, needs
    ``lb`` in Hz), ``"hann"``, ``"gaussian"`` (optional ``std`` in
    samples), ``"sine"``, ``"qsine"``). Adding a new window later is
    adding one function to ``_WINDOWS``, not touching call sites.

    These are generic textbook window formulas -- for *exactly*
    reproducing a specific vendor's own apodization (e.g. Bruker's
    WDW-coded windows), see ``opennmr.io.bruker``, which uses
    ``nmrglue`` for that instead of this registry.
    """
    if window not in _WINDOWS:
        raise ValueError(f"Unknown window {window!r}; available: {sorted(_WINDOWS)}")
    weights = _WINDOWS[window](len(fid), dt, **params)
    return fid * weights


# ============================================================
# Zero-fill / FFT
# ============================================================

def zero_fill(fid: np.ndarray, n: int) -> np.ndarray:
    """Zero-fill (or truncate) ``fid`` to length ``n``."""
    if n <= len(fid):
        return fid[:n]
    return np.pad(fid, (0, n - len(fid)), mode="constant")


def fft_spectrum(fid: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """FFT a time-domain FID into a frequency-domain spectrum.

    Returns ``(freq_hz, spectrum)``, ``freq_hz`` length ``fid.shape[-1]``
    (decreasing, matching NMR's high-frequency-on-the-left display
    convention) and ``spectrum`` the same shape as ``fid`` -- ``fid``
    is meant to be a single 1D FID (the common case, where
    ``fid.shape[-1] == len(fid)`` so this is a no-op distinction), but
    ``np.fft.fft``'s own default ``axis=-1`` already does the right
    thing for a batch of FIDs stacked along a leading axis too;
    ``fid.shape[-1]`` (not ``len(fid)``, which reads the *first*
    dimension for anything with more than one) is what makes
    ``freq_hz`` come out the right length in that case as well, rather
    than silently matching the wrong axis. Found via a real bug: a
    caller passed a 2D array directly here expecting per-row batch
    FFT, and the previous ``len(fid)`` produced a frequency axis
    matching the *row* count instead of the row length -- every other
    caller doing multi-row FFT works around this today by calling
    ``fft_spectrum`` once per row via ``np.apply_along_axis`` instead
    (still fine, and unaffected by this fix), but there's no reason
    this function itself has to be wrong about its own output shape
    for 2D input.
    """
    n = fid.shape[-1]
    spectrum = np.fft.fftshift(np.fft.fft(fid, axis=-1), axes=-1)
    freq_hz = -np.fft.fftshift(np.fft.fftfreq(n, dt))
    return freq_hz, spectrum


# ============================================================
# Phasing
# ============================================================

def apply_phase(spectrum: np.ndarray, freq: np.ndarray, phase0_deg: float,
                 phase1_deg: float, pivot_frac: float = 0.5) -> np.ndarray:
    """Apply a zero- and first-order phase correction, first-order
    pivoted at ``pivot_frac`` (0..1) across ``freq``'s range (0.5 =
    centre)."""
    freq = np.asarray(freq, dtype=float)
    span = (freq.max() - freq.min()) or 1.0
    freq_norm = (freq - (freq.min() + pivot_frac * span)) / span
    phi = np.deg2rad(phase0_deg) + np.deg2rad(phase1_deg) * freq_norm
    return spectrum * np.exp(-1j * phi)


def phase_score(spectrum: np.ndarray, penalty: str = "negarea") -> float:
    """Lower is better.

    ``"negarea"`` : penalize negative-going real signal (Chen et al.
    2002 ACME-style, simplified).

    ``"magnitude"`` : penalize ``(real - |spectrum|)**2`` -- how far
    the phased real part falls short of the phase-independent
    magnitude. More robust than ``"negarea"`` once apodization/line
    broadening is in use, where a small residual phase error's
    dispersive tail can stay positive (so ``"negarea"`` barely notices
    it) while still tilting the whole baseline.

    ``"entropy"`` : Shannon entropy of the real part's derivative plus
    a small negarea penalty.
    """
    real = np.real(spectrum)
    if penalty == "negarea":
        return np.sum(np.abs(real[real < 0]))
    elif penalty == "magnitude":
        return np.sum((real - np.abs(spectrum)) ** 2)
    elif penalty == "entropy":
        deriv = np.abs(np.diff(real))
        p = deriv / (np.sum(deriv) + 1e-30)
        h = -np.sum(p * np.log(p + 1e-30))
        neg_penalty = np.sum(np.abs(real[real < 0]))
        return h + 1e-3 * neg_penalty
    else:
        raise ValueError("penalty must be 'negarea', 'magnitude', or 'entropy'")


def apc0(spectrum: np.ndarray, freq: np.ndarray, phase0_inc: float = 0.1, penalty: str = "magnitude") -> float:
    """Zero-order-only autophase: grid search minimizing
    :func:`phase_score` (same criterion/default ``penalty`` as
    :func:`autophase01`'s own stage-1 search -- see :func:`autophase01`
    for a joint zero-/first-order fit).

    Used to score by raw ``sum(real(corrected))`` (maximize) instead --
    changed because it's unreliable on anything but a strong, clean
    peak: that sum is dominated by whatever's biggest across the
    *whole* array, so on a spectrum that's mostly noise/baseline with
    only a small real peak (a late PFG gradient-step slice, a weak
    signal generally) it easily locks onto a phase that's actually
    fitting the noise, not the peak -- confirmed on a synthetic
    decaying-FID test: at moderate noise, the old scoring was off by
    11 deg from the true phase where ``phase_score``'s magnitude
    penalty (``sum((real - |spectrum|)**2)``, phase-independent by
    construction) was off by only 2 deg, and the gap widens further at
    higher noise. Real bug, not just theoretical: this is what
    "Autophase zeroes everything" on real, lower-SNR data actually
    was."""
    phase0_values = np.arange(-180, 180, phase0_inc)
    best_score = np.inf
    best_phase0 = 0.0

    for phase0 in phase0_values:
        corrected = spectrum * np.exp(1j * -np.deg2rad(phase0))
        score = phase_score(corrected, penalty)
        if score < best_score:
            best_score = score
            best_phase0 = phase0

    return float(best_phase0)


def autophase01(spectrum: np.ndarray, freq: np.ndarray, phase0_range: tuple[float, float] = (-180, 180),
                 phase0_inc: float = 1.0, phase1_max: float = 20.0, phase1_improvement: float = 0.15,
                 penalty: str = "magnitude", pivot_frac: float = 0.5, verbose: bool = False) -> tuple[float, float]:
    """``phase0, phase1 = autophase01(spectrum, freq, ...)``

    Joint zero- and first-order autophasing, but phase1 is only ever
    used when it demonstrably improves the lineshape over phase0 alone
    -- a significance test, not a soft joint fit, so noise-level
    "improvements" on low-SNR data don't buy phase1 a spurious nonzero
    value.

    1. Stage 1: fine phase0-only grid search (phase1=0), scored by
       :func:`phase_score`.
    2. Stage 2: joint (phase0, phase1) Nelder-Mead refinement from the
       stage-1 starting point, bounded to ``phase1 in [-phase1_max,
       phase1_max]`` and ``phase0`` within +/-90 deg of the stage-1
       result.
    3. Stage 2's result is only kept if it improves :func:`phase_score`
       by more than ``phase1_improvement`` (relative to the stage-1
       score); otherwise ``phase1=0`` (stage-1 ``phase0``) is returned.

    Returns ``(phase0_deg, phase1_deg)``. ``phase1`` is exactly ``0.0``
    whenever stage 2 didn't earn its keep.
    """
    phase0_grid = np.arange(phase0_range[0], phase0_range[1], phase0_inc)
    scores = [
        phase_score(apply_phase(spectrum, freq, p0, 0.0, pivot_frac), penalty)
        for p0 in phase0_grid
    ]
    p0_start = float(phase0_grid[int(np.argmin(scores))])
    stage1_score = min(scores)

    def objective(params):
        p0, p1 = params
        return phase_score(apply_phase(spectrum, freq, p0, p1, pivot_frac), penalty)

    bounds = [(p0_start - 90, p0_start + 90), (-phase1_max, phase1_max)]
    result = minimize(objective, x0=[p0_start, 0.0], method="Nelder-Mead", bounds=bounds,
                       options={"xatol": 1e-2, "fatol": 1e-6, "maxiter": 2000})
    stage2_score = result.fun
    rel_improvement = (stage1_score - stage2_score) / stage1_score if stage1_score else 0.0

    if rel_improvement > phase1_improvement:
        phase0, phase1 = result.x
    else:
        phase0, phase1 = p0_start, 0.0

    phase0 = ((phase0 + 180) % 360) - 180

    if verbose:
        print(f"autophase01: stage-1 phase0={p0_start:.1f} deg (score {stage1_score:.4g}) -> "
              f"stage-2 phase0={result.x[0]:.2f}, phase1={result.x[1]:.2f} deg "
              f"(score {stage2_score:.4g}, {rel_improvement:.1%} improvement) -> "
              f"phase0={phase0:.2f} deg, phase1={phase1:.2f} deg")

    return float(phase0), float(phase1)


# ============================================================
# Baseline
# ============================================================

def baseline_correct(freq: np.ndarray, spectrum: np.ndarray,
                      baseline_regions: list[tuple[float, float]] | tuple[float, float] | None = None,
                      frac: float = 0.05, sigma: float = 2.5, max_iter: int = 10,
                      method: str = "linear", smoothing: float | None = None,
                      degree: int = 2, n_harmonics: int = 1, period: float | None = None) -> np.ndarray:
    """Subtract a robust baseline fit to the real part, via
    :mod:`opennmr.processing.fitting`'s baseline estimators -- the same
    ones ``fit_spectrum`` already uses to hold a baseline fixed under a
    peak fit, so this and the peak-fitting workflow both fit baselines
    the same way.

    A single unweighted fit (this function's own original
    implementation, plain ``np.polyfit`` over every point) gets pulled
    toward any real peaks included in it, which -- for anything but a
    dead-flat spectrum -- systematically under-corrects: barely moving
    the line, since the peaks it should run underneath instead drag it
    upward. Every estimator below instead sigma-clips its way to the
    points that are actually baseline (starting from
    ``baseline_regions`` if given, otherwise the first/last ``frac`` of
    the spectrum) before fitting, so real peaks don't bias the result.

    ``method`` :

    - ``"linear"`` (default, :func:`opennmr.processing.fitting.
      estimate_baseline_robust`) fits a single straight line -- fast
      and usually enough, but systematically under-corrects a real
      curved/rolling baseline (common on wideline/solid-state spectra,
      and on any spectrum with a large recorded first-order phase --
      a real first-order phase correction's own effect grows with
      distance from the pivot, so a residual phase error shows up as
      *curvature* concentrated toward the spectrum's edges, exactly
      where the default edge-only candidate region draws its baseline
      points from), since a line simply cannot follow that shape
      regardless of which points it's fit to.
    - ``"polynomial"`` (:func:`opennmr.processing.fitting.
      estimate_baseline_polynomial`, degree ``degree``) generalizes
      ``"linear"`` (``degree=1`` is identical to it) to follow real
      curvature without needing a full spline.
    - ``"sinusoidal"`` (:func:`opennmr.processing.fitting.
      estimate_baseline_sinusoidal`, ``n_harmonics``/``period``) fits a
      slowly rolling, wave-like baseline -- a shape neither a
      polynomial nor a spline (which can chase noise between distant
      candidate points) represents as naturally.
    - ``"spline"`` (:func:`opennmr.processing.fitting.
      estimate_baseline_spline`) fits a smooth cubic spline -- the
      most flexible option, and the only one that can follow a shape
      not well-described by a low-order polynomial or a single
      sinusoid, at the cost of being the easiest to overfit to noise
      between sparse candidate points.

    ``smoothing`` : ``method="spline"`` only -- see
    :func:`estimate_baseline_spline`'s own ``smoothing`` parameter
    (``None`` picks a data-driven default).

    ``degree`` : ``method="polynomial"`` only -- polynomial degree
    (1 = linear, 2 = quadratic, ...).

    ``n_harmonics``/``period`` : ``method="sinusoidal"`` only -- see
    :func:`estimate_baseline_sinusoidal`'s own parameters of the same
    names (``period=None`` picks a data-driven default: twice the
    candidate points' own x-range).

    ``baseline_regions`` : explicit (lo, hi) ppm window(s) known to be
    peak-free -- e.g. from the Analysis tab's own peak table, expanded
    a little past each peak's FWHM -- for a spectrum where the default
    edge-only assumption doesn't hold (a peak near either end, or a
    baseline that isn't well represented by just the two edges)."""
    from .fitting import (
        estimate_baseline_polynomial, estimate_baseline_sinusoidal,
        estimate_baseline_spline, evaluate_baseline_sinusoidal,
    )
    freq = np.asarray(freq, dtype=float)
    if baseline_regions is None:
        # Guarantee a handful of candidate baseline points per edge even
        # for a short spectrum -- frac=0.05 alone rounds down to under 3
        # points (the estimators' own minimum) well before a real
        # (thousands-of-points) spectrum's usual size would ever need
        # this.
        min_edge_points = min(5, max(1, len(freq) // 2))
        frac = max(frac, min_edge_points / len(freq))
    if method == "spline":
        spline = estimate_baseline_spline(
            freq, np.real(spectrum), baseline_regions=baseline_regions,
            frac=frac, sigma=sigma, max_iter=max_iter, smoothing=smoothing,
        )
        baseline = spline(freq)
    elif method in ("linear", "polynomial"):
        coeffs, x_ref = estimate_baseline_polynomial(
            freq, np.real(spectrum), baseline_regions=baseline_regions,
            frac=frac, sigma=sigma, max_iter=max_iter, degree=(1 if method == "linear" else degree),
        )
        baseline = np.polyval(coeffs, freq - x_ref)
    elif method == "sinusoidal":
        coeffs, fitted_period = estimate_baseline_sinusoidal(
            freq, np.real(spectrum), baseline_regions=baseline_regions,
            frac=frac, sigma=sigma, max_iter=max_iter, n_harmonics=n_harmonics, period=period,
        )
        baseline = evaluate_baseline_sinusoidal(coeffs, fitted_period, freq)
    else:
        raise ValueError(f"method must be 'linear', 'polynomial', 'sinusoidal' or 'spline', got {method!r}")
    return spectrum - baseline


# ============================================================
# Linear prediction
# ============================================================

def linear_predict(fid: np.ndarray, order: int = 20, n_predict: int = 8,
                    direction: str = "forward") -> np.ndarray:
    """``extended_fid = linear_predict(fid, order=20, n_predict=8, direction='forward')``

    Extend (or backward-reconstruct) a complex FID using linear
    prediction: fits an order-``p`` autoregressive model
    ``x[n] = sum_k a_k x[n-k]`` by least-squares (covariance method) on
    the existing points, then uses it to predict ``n_predict``
    additional points.

    ``direction='forward'`` : appends points predicting the
        continuation of the FID past its measured end -- reduces
        truncation (Gibbs) artifacts from a short acquisition.
    ``direction='backward'`` : prepends points predicting *before* the
        measured start -- reconstructs points lost to a fixed dead-time
        window before acquisition began.

    ``order`` : AR model order. Must be less than ``len(fid) - 1``.
    Higher order fits more spectral components but is more prone to
    overfitting noise; ``order=20-30`` is a reasonable default for a
    single dominant echo/peak.

    Returns the extended complex FID (length ``len(fid) + n_predict``).
    """
    fid = np.asarray(fid, dtype=np.complex128)
    n = len(fid)
    if order >= n - 1:
        raise ValueError(f"linear_predict: order ({order}) must be < len(fid)-1 ({n - 1})")
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be 'forward' or 'backward'")

    x = fid[::-1].copy() if direction == "backward" else fid.copy()

    n_eq = n - order
    A = np.empty((n_eq, order), dtype=np.complex128)
    for j in range(order):
        A[:, j] = x[order - 1 - j: n - 1 - j]
    b = x[order:n]
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)

    predicted = list(x)
    for _ in range(n_predict):
        next_val = np.dot(coeffs, predicted[-1:-order - 1:-1])
        predicted.append(next_val)
    predicted = np.array(predicted)

    return predicted[::-1] if direction == "backward" else predicted


# ============================================================
# Hypercomplex indirect-dimension (F1) decoding
# ============================================================

def decode_hypercomplex_f1(data: np.ndarray, fn_mode: int) -> np.ndarray:
    """Combine adjacent raw row-pairs along axis 0 into genuine complex
    F1 (indirect-dimension) points, per Bruker's ``FnMODE`` acquisition
    convention -- must run *before* the F1 FFT, on data that's already
    been fully processed along F2 (each row already a proper complex
    F2 spectrum/FID; this function only touches axis 0).

    ``fn_mode`` : the integer from ``acqus``/``acqu2s``'s own
    ``FnMODE`` field. Mapping confirmed directly from
    ``nmrglue.fileio.bruker.guess_udic``, the authoritative source used
    here (Bruker's own manual documents each mode's *acquisition*
    behaviour, not this exact integer table) -- 0 undefined, 1 QF,
    2 QSEQ, 3 TPPI, 4 States, 5 States-TPPI, 6 Echo-Antiecho.

    Modes 0-3 need no row-pairing here and are returned unchanged: QF
    (1) is already a genuine complex point per row (ordinary quadrature
    detection, one row per t1 increment -- this is what every real
    dataset in this project's own test_data actually uses); undefined
    (0), QSEQ (2), and plain TPPI (3, a *single* real interferogram per
    t1 point via receiver-phase incrementation, not a row-pair scheme
    at all) aren't implemented -- not requested, and no real data using
    them exists in this project to validate against.

    **States (4)**: row ``2k`` is the "cosine"-modulated (real) row,
    row ``2k+1`` the "sine"-modulated (imaginary) row, for the *same*
    t1 increment ``k`` -- ``decoded[k] = row[2k] + 1j*row[2k+1]``. Each
    row is itself already complex (F2's own quadrature detection); the
    extra ``1j`` here is a *second*, independent multiplication for the
    F1 axis specifically -- this is what "hypercomplex" means.

    **States-TPPI (5)**: identical row-pairing, plus every other
    resulting hypercomplex point gets multiplied by ``-1``
    (``(-1)**k``) -- displaces the axial (unmodulated-artifact) peak to
    the edge of the F1 spectral window instead of its centre, which is
    the entire point of the TPPI-style receiver-phase increment.
    Confirmed via NMR Wiki's States-TPPI reference. **Validated against
    5 real States-TPPI (FnMODE=5) Bruker datasets in this project's own
    test_data** -- cross-checked against ``read_2d_processed``
    (TopSpin's own numbers) the same way the raw-`ser` shape work was.

    **Echo-Antiecho (6, a.k.a. Rance-Kay)**: row ``2k`` ("echo"/P-type)
    and row ``2k+1`` ("antiecho"/N-type) combine as
    ``decoded[k] = (echo+antiecho) + 1j*(echo-antiecho)`` -- the
    standard sensitivity-enhanced (PEP) recombination. **Not verified
    against real data**: no Echo-Antiecho (FnMODE=6) dataset exists
    anywhere in this project's test_data to check against, and this
    project's own from-nmrglue-source investigation of the closest
    available reference (``nmrglue.process.nmrtxt.rance_kay.
    bruk_ranceY``, and nmrglue's own S3E worked example) could not
    fully pin down Bruker's specific row *order* (echo-then-antiecho vs
    the reverse) with certainty from documentation alone -- getting
    that specific detail wrong would mirror F1 peaks across the
    spectrum's centre without otherwise looking obviously broken,
    exactly the "plausible but wrong" risk this project's own raw-FID
    work has flagged before. Treat this mode as a best-effort
    implementation until a real Echo-Antiecho dataset is available to
    check it against, same standing as the Varian ppm-axis and JEOL
    ``FILTER_FACTOR`` gaps already documented elsewhere in this
    project.

    An odd row count (shouldn't happen for genuinely-paired data, but
    the raw-`ser` row-*count* resolution this project's own
    :func:`opennmr.io.bruker._guess_raw_2d_shape` uses already has a
    small, documented, unexplained residual on some real files) drops
    the final unpaired row rather than raising.
    """
    if fn_mode not in (4, 5, 6):
        return data
    n_pairs = data.shape[0] // 2
    paired = data[: 2 * n_pairs]
    first, second = paired[0::2], paired[1::2]
    if fn_mode in (4, 5):
        decoded = first + 1j * second
        if fn_mode == 5:
            sign = np.where(np.arange(n_pairs) % 2 == 0, 1.0, -1.0)
            decoded = decoded * sign[:, np.newaxis]
    else:
        decoded = (first + second) + 1j * (first - second)
    return decoded
