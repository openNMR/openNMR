"""
Vendor-agnostic FID -> spectrum processing: small, independent,
composable steps (roll/align, apodize, zero-fill, FFT, phase, baseline,
linear prediction) rather than one fixed pipeline function, so
different acquisition styles (a plain decaying FID vs. a full echo that
needs aligning first, for example) can be built from the same parts.

Used directly by ``nmrsuite.io.xpulse`` (X-Pulse's raw FID needs the
whole pipeline) and for the shared, non-Bruker-specific finishing
touches of ``nmrsuite.io.bruker``'s raw-FID path (``autophase01`` as a
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
    WDW-coded windows), see ``nmrsuite.io.bruker``, which uses
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

    Returns ``(freq_hz, spectrum)``, both length ``len(fid)``, with
    ``freq_hz`` decreasing (matching NMR's high-frequency-on-the-left
    display convention).
    """
    n = len(fid)
    spectrum = np.fft.fftshift(np.fft.fft(fid))
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


def apc0(spectrum: np.ndarray, freq: np.ndarray, phase0_inc: float = 0.1) -> float:
    """Zero-order-only autophase: grid search maximizing the real
    part's integral. Simple and fast, but blind to first-order (phase1)
    errors -- see :func:`autophase01` for a joint fit."""
    phase0_values = np.arange(-180, 180, phase0_inc)
    best_score = -np.inf
    best_phase0 = 0.0

    for phase0 in phase0_values:
        corrected = spectrum * np.exp(1j * -np.deg2rad(phase0))
        score = np.sum(np.real(corrected))
        if score > best_score:
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

def baseline_correct(freq: np.ndarray, spectrum: np.ndarray, order: int = 1) -> np.ndarray:
    """Subtract a low-order polynomial baseline fit to the real part
    (``order=1``, linear, by default)."""
    coeffs = np.polyfit(freq, np.real(spectrum), order)
    baseline = np.polyval(coeffs, freq)
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
