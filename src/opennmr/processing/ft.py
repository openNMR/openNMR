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

import os

import numpy as np
from . import windows as _windows
import scipy.fft as _scipy_fft
from scipy.optimize import minimize


# ============================================================
# FFT worker policy
# ============================================================

# Measured plateau, not a guess: on the project's own EXSY benchmark
# shapes, F2 (few very long transforms, `(1002, 65536)`) stops
# improving past 4 workers -- 0.442s numpy, 0.140s at 4, 0.145s at 8 --
# because it is memory-bandwidth-bound, while F1 (many short
# transforms, `(65536, 2048)`) keeps scaling to 8. A single cap has to
# serve both, and over-threading is the worse failure: scipy/numpy's
# own BLAS/OpenMP pools are already running in the same process, and
# oversubscribing them on a many-core machine costs more than the
# remaining headroom is worth. 4 is the conservative choice; revisit
# with measurements on a many-core Linux box, which nobody has run
# this on yet.
_FFT_WORKER_CAP = 4

_FFT_WORKERS_ENV = "OPENNMR_FFT_WORKERS"


def _available_cpus() -> int:
    """CPUs this *process* may actually use.

    Deliberately not just ``os.cpu_count()``, which reports the
    machine's cores rather than this process's own allowance -- wrong
    under container/cgroup CPU limits and under CPU affinity masks,
    both routine on Linux and in CI. ``os.process_cpu_count()`` is the
    correct API but is Python 3.13+, and this project supports 3.10+,
    hence the fallback chain (``sched_getaffinity`` is Linux-only and
    absent on macOS/Windows, so it is probed, not assumed).
    """
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count:
            return count
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            return len(sched_getaffinity(0))
        except OSError:
            pass
    return os.cpu_count() or 1


def fft_workers() -> int:
    """How many worker threads :func:`fft_spectrum` should ask
    ``scipy.fft`` for.

    ``OPENNMR_FFT_WORKERS`` overrides the automatic choice entirely: a
    positive integer pins that many workers, and ``1`` pins the fully
    serial path -- which is measurably identical to ``numpy.fft`` (not
    merely close: ``workers=1`` timed the same to within noise, and the
    output is bit-identical), so "threads off" is a genuinely supported
    configuration rather than a degraded one. That matters for CI
    determinism, for constrained machines, and for bisecting a
    suspected threading problem. An unparseable or non-positive value
    is ignored in favour of the automatic choice rather than raising --
    a bad environment variable should not stop a spectrum loading.
    """
    override = os.environ.get(_FFT_WORKERS_ENV)
    if override:
        try:
            requested = int(override)
        except ValueError:
            requested = 0
        if requested > 0:
            return requested
    return max(1, min(_FFT_WORKER_CAP, _available_cpus()))


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


def find_echo_top(complex_fid: np.ndarray, offset: float = 0.0) -> float:
    """Fractional sample index of a full echo's peak, via parabolic
    interpolation of ``|complex_fid|`` around its maximum -- for
    signals that rise then fall (a full echo) rather than a one-sided
    decaying FID.

    ``offset`` : points added to the detected top, to correct it by
    hand. The detector takes the largest magnitude sample, which is
    the right answer for a clean symmetric echo and can be a few
    points out for a noisy or asymmetric one -- and being out by even
    one point puts a first-order phase ramp across the whole spectrum,
    since the echo top is what defines t=0. Fractional values are
    allowed: :func:`roll_fid` interpolates, so the correction does not
    have to land on a sample boundary.

    Clamped to the array, so a large offset cannot roll the echo top
    outside the data it is meant to index."""
    magnitude = np.abs(complex_fid)
    i = int(np.argmax(magnitude))
    if 0 < i < len(magnitude) - 1:
        y0, y1, y2 = magnitude[i - 1], magnitude[i], magnitude[i + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    return float(np.clip(i + delta + offset, 0.0, max(len(magnitude) - 1, 0)))


def align_echo(fid: np.ndarray, target_index: int = 0, echo_offset: float = 0.0) -> np.ndarray:
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
    top = find_echo_top(fid, offset=echo_offset)
    return roll_fid(fid, target_index - top)


def prepare_fid(fid: np.ndarray, *, fid_shift: float = 0.0, align_echo_enabled: bool = False,
                 echo_offset: float = 0.0, truncate: int | None = None,
                 linear_predict_points: int = 0,
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
    function calls.

    ``echo_offset`` : points added to the detected echo top before
    rolling -- see :func:`find_echo_top`. The wrap limit that caps
    ``truncate`` is recomputed from the offset top, so nudging the
    alignment cannot silently leave the mirror-image tail in."""
    if fid_shift:
        fid = roll_fid(fid, fid_shift)

    echo_wrap_limit = None
    if align_echo_enabled:
        # One detection, reused: the wrap limit below has to be
        # computed from the *same* top the roll used, offset included,
        # or the truncation cap no longer matches where the pre-echo
        # portion actually wrapped to.
        top = find_echo_top(fid, offset=echo_offset)
        fid = align_echo(fid, echo_offset=echo_offset)
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
    """Exponential multiplication (line broadening ``lb``, in Hz).
    ``exp(-t*pi*lb) == exp(-pi*i*(lb/sw_hz))`` (``t = i*dt = i/sw_hz``)
    -- exactly :func:`nmrglue.process.proc_base.em`'s own formula with
    ``lb_nmrglue = lb/sw_hz`` (confirmed algebraically, not just by
    inspection, before switching this from a hand-rolled ``np.exp``
    call to the shared primitive below, for the same reason
    :func:`_window_gaussian`/:func:`_window_sine` now use it too: one
    consistent, ``nmrglue``-backed implementation per window shape,
    shared with ``io.bruker``'s own "Recorded" ``WDW`` path rather than
    a second hand-rolled formula that could quietly drift from it."""
    sw_hz = 1.0 / dt
    weights = _windows.em(np.ones(n, dtype=complex), lb=lb / sw_hz)
    return np.real(weights)


def _window_hann(n: int, dt: float) -> np.ndarray:
    return np.hanning(n)


def _window_gaussian(n: int, dt: float, lb: float = 0.0, gb: float = 0.0) -> np.ndarray:
    """Lorentz-to-Gauss (Bruker/NMRPipe "GM") apodization: cancels the
    natural exponential decay with ``lb`` (Hz, typically *negative* --
    real user report: "Gaussian I think need[s] a negative lb"), then
    tapers with a Gaussian envelope of width ``gb`` (Hz, same units and
    scale as ``lb`` -- confirmed via NMRPipe's own published GM formula
    (``e = pi*i*g1/sw``, ``g = 0.6*pi*g2*(g3*(size-1)-i)/sw``, both
    ``g1``/``g2`` "in Hz"), not a bare 0-1 fraction as an earlier draft
    of this function assumed -- that gave a window crushed to ~0
    everywhere except one edge point for any realistic array size, an
    easy, non-obvious mistake this project's own web search caught
    before it shipped. ``gb=0.0`` (default) degenerates to a plain
    exponential (no Gaussian term at all), matching ``lb`` alone being
    equivalent to :func:`_window_exponential` with the opposite sign.

    Uses :func:`nmrglue.process.proc_base.gm` -- the same
    NMRPipe-compatible primitive already used for the Bruker "Recorded"
    ``WDW`` path elsewhere in this package -- with ``g3`` (location of
    the Gaussian maximum, 0..1) fixed at 0.5 rather than exposed:
    TopSpin's own GM UI likewise only ever exposes LB/GB, not a
    separate centre control, and a location fixed at the array's very
    end (``g3=1.0``, tried first) degenerately puts the window's peak
    at the last point unconditionally, for any ``gb`` -- not a useful
    apodization shape.

    ``nmrglue.gm``'s own ``g1`` sign is opposite ``em``'s: positive
    ``g1`` sharpens/grows, negative decays -- so ``lb``'s own sign is
    negated below (keeping this function's own ``lb`` in the same
    "positive shortens T2*, negative sharpens/cancels decay" sense
    :func:`_window_exponential`'s ``lb`` already uses, for a consistent
    feel across every window in this registry). ``g2`` only ever
    appears squared in NMRPipe's own formula, so ``gb``'s sign never
    matters, only its magnitude.

    **Not verified against real Bruker GM-processed data** (unlike
    ``_apply_bruker_window``'s own WDW 0/1/3/4 cases, or
    ``decode_hypercomplex_f1``'s Echo-Antiecho mode, both cross-checked
    against real TopSpin output) -- flagged, not silently presented as
    certain, same standing as this project's other documented-but-
    unverified gaps (Varian's ppm-axis convention, JEOL's
    ``FILTER_FACTOR``). Bruker's own TopSpin may derive its two-box
    LB/GB from a different internal formula than NMRPipe's published
    one; this reproduces the latter, which is at minimum a real,
    working, correctly-shaped Gaussian-tapering window, just not
    guaranteed bit-for-bit identical to what TopSpin itself would show
    for the same nominal numbers."""
    sw_hz = 1.0 / dt
    g1 = -lb / sw_hz
    g2 = gb / sw_hz
    weights = _windows.gm(np.ones(n, dtype=complex), g1=g1, g2=g2, g3=0.5)
    return np.real(weights)


def _window_sine(n: int, dt: float, ssb: float = 0.0) -> np.ndarray:
    """Shifted sine-bell, Bruker's ``SSB`` convention: ``ssb >= 2``
    starts the bell ``1/ssb`` of the way in (``ssb=2`` is a pure cosine
    bell, matching a "phase-shifted sine" description many vendors
    use); ``ssb < 2`` (default 0.0) is a plain full sine bell from 0,
    same shape :func:`_window_qsine`'s own former hardcoded formula
    used before this parameter existed. Same ``off`` formula
    :func:`opennmr.io.bruker._apply_bruker_window` already uses (and
    has real-data-validated) for Bruker's own recorded ``WDW``
    3/SINE."""
    off = 1.0 / ssb if ssb >= 2 else 0.0
    weights = _windows.sp(np.ones(n, dtype=complex), off=off, end=1.0, pow=1.0)
    return np.real(weights)


def _window_qsine(n: int, dt: float, ssb: float = 0.0) -> np.ndarray:
    """Shifted sine-bell squared -- see :func:`_window_sine`, same
    ``ssb`` convention, matching Bruker's own recorded ``WDW`` 4/QSINE
    (:func:`opennmr.io.bruker._apply_bruker_window`)."""
    off = 1.0 / ssb if ssb >= 2 else 0.0
    weights = _windows.sp(np.ones(n, dtype=complex), off=off, end=1.0, pow=2.0)
    return np.real(weights)


_WINDOWS = {
    "exponential": _window_exponential,
    "hann": _window_hann,
    "gaussian": _window_gaussian,
    "sine": _window_sine,
    "qsine": _window_qsine,
}


def apodize(fid: np.ndarray, dt: float, window: str = "exponential", **params) -> np.ndarray:
    """Multiply ``fid`` by a named window function.

    ``window`` : one of ``_WINDOWS`` -- ``"exponential"`` (default,
    ``lb`` in Hz), ``"hann"`` (no parameters), ``"gaussian"`` (``lb``/
    ``gb``, both in Hz -- see :func:`_window_gaussian`'s own docstring
    for the sign convention and what's/isn't verified), ``"sine"``/
    ``"qsine"`` (``ssb``, Bruker's own convention -- see
    :func:`_window_sine`). Adding a new window later is adding one
    function to ``_WINDOWS``, not touching call sites.

    Every window here (bar ``"hann"``, parameter-free even in the
    Bruker/NMRPipe convention) is built from ``nmrglue``'s own
    NMRPipe-compatible primitives (``proc_base.em``/``gm``/``sp``) --
    the same functions ``opennmr.io.bruker._apply_bruker_window``
    already uses for exactly reproducing a Bruker file's own recorded
    ``WDW`` -- so a user-chosen window here (not "Recorded") uses the
    identical math, just with independently dialed-in parameters
    instead of whatever TopSpin itself recorded.

    ``fid`` may be a batch of FIDs stacked along a leading axis (the
    window only ever depends on the row length, identical for every
    row) -- ``fid.shape[-1]``, not ``len(fid)``, is what makes this
    correct for 2D input instead of silently sizing ``weights`` to the
    row *count*; same fix, same reasoning as :func:`fft_spectrum`'s
    own ``len(fid)`` bug, found while vectorizing ``io.bruker.read_2d``
    away from a 66,000-call ``np.apply_along_axis`` loop.
    """
    if window not in _WINDOWS:
        raise ValueError(f"Unknown window {window!r}; available: {sorted(_WINDOWS)}")
    weights = _WINDOWS[window](fid.shape[-1], dt, **params)
    return fid * weights


# ============================================================
# Zero-fill / FFT
# ============================================================

def zero_fill(fid: np.ndarray, n: int) -> np.ndarray:
    """Zero-fill (or truncate) ``fid`` to length ``n`` along its last
    axis -- a single 1D FID (the common case) or a batch stacked along
    a leading axis, matching :func:`apodize`/:func:`fft_spectrum`'s own
    ``shape[-1]`` convention rather than assuming 1D input."""
    length = fid.shape[-1]
    if n <= length:
        return fid[..., :n]
    pad_width = [(0, 0)] * (fid.ndim - 1) + [(0, n - length)]
    return np.pad(fid, pad_width, mode="constant")


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
    # scipy.fft rather than numpy.fft purely for its `workers`
    # parameter -- the transform itself is the same pocketfft, and the
    # result is bit-identical (verified directly against numpy on the
    # real EXSY benchmark, not assumed: threading splits the batch
    # across independent 1D transforms, so no individual transform's
    # own summation order changes). At workers=1 this is numpy's own
    # performance to within noise; the win is on batched 2D input,
    # where it is ~3-4x (see fft_workers() for the measured numbers and
    # why the worker count is capped rather than -1).
    spectrum = _scipy_fft.fftshift(
        _scipy_fft.fft(fid, axis=-1, workers=fft_workers()), axes=-1,
    )
    freq_hz = -np.fft.fftshift(np.fft.fftfreq(n, dt))
    return freq_hz, spectrum


def ifft_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Inverse of :func:`fft_spectrum`: a frequency-domain spectrum
    back to the time-domain FID it came from.

    Exactly undoes that function's own ``fftshift`` + ``fft`` pair, in
    the reverse order, so ``ifft_spectrum(fft_spectrum(fid)[1])``
    returns ``fid`` to numerical precision. There is no ``dt``
    parameter and no frequency axis returned, because neither is
    needed to invert: the shift and the transform are both purely
    positional.

    Added for reference deconvolution (:mod:`opennmr.processing.refdeconv`),
    which isolates a reference resonance *in the frequency domain* and
    then needs its FID back. Written here rather than inline there so
    the forward and inverse conventions live side by side -- getting
    the ``fftshift`` half wrong is silent, and shows up only as a
    correction function that is a mirrored or rotated version of the
    right one.
    """
    return _scipy_fft.ifft(
        _scipy_fft.ifftshift(spectrum, axes=-1), axis=-1, workers=fft_workers(),
    )


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

def compute_baseline(freq: np.ndarray, spectrum: np.ndarray,
                     baseline_regions: list[tuple[float, float]] | tuple[float, float] | None = None,
                     frac: float = 0.05, sigma: float = 2.5, max_iter: int = 10,
                     method: str = "linear", smoothing: float | None = None,
                     degree: int = 2, n_harmonics: int = 1, period: float | None = None,
                     lam: float | None = None, asymmetry: float | None = None,
                     diff_order: int = 2, manual_params=None, manual_fixed=None,
                     manual_refit: bool = False, n_points: int | None = None) -> np.ndarray:
    """The baseline :func:`baseline_correct` would subtract, returned
    instead of subtracted.

    Same parameters, same estimators, same result -- ``baseline_
    correct`` is literally ``spectrum - compute_baseline(...)``. Split
    out so the GUI can *draw* the baseline over the spectrum before
    committing to it (explicit user request: "add a processing overlay
    to the plot showing the baseline corrections before they are
    committed"), without either recomputing it a second time to
    recover it by subtraction or risking the drawn curve and the
    subtracted one drifting apart.

    Returns a real array, whatever ``spectrum``'s own dtype: every
    estimator here fits the real part, which is the only part a
    baseline is defined on.
    """
    from .fitting import (
        estimate_baseline_polynomial, estimate_baseline_sinusoidal,
        estimate_baseline_spline, estimate_baseline_whittaker, auto_spline_baseline,
        evaluate_baseline_sinusoidal,
    )
    freq = np.asarray(freq, dtype=float)
    if method in ("manual_polynomial", "manual_sine"):
        # The user's own model (see notebook.manual_baseline):
        # evaluated from the recorded coefficients as they stand, or --
        # with manual_refit, or before any coefficients exist -- fitted
        # afresh to baseline_regions on this very trace (so every slice
        # of a series, and a re-phased trace, gets its own fit).
        from ..notebook import manual_baseline as nb_manual
        model = method.split("_", 1)[1]
        names = nb_manual.MANUAL_BASELINE_PARAMS[model]
        if manual_params is not None and not manual_refit:
            return nb_manual.evaluate_manual_baseline(freq, model, manual_params)
        fixed = None
        if manual_fixed and manual_params is not None:
            values = nb_manual._as_param_list(model, manual_params)
            fixed = {n: v for n, v, f in zip(names, values, manual_fixed) if f}
        return nb_manual.fit_manual_baseline(
            freq, np.real(spectrum), model, regions=baseline_regions, fixed=fixed)["baseline"]
    if method == "spline_auto":
        # Like Whittaker, it finds the baseline itself, anywhere in the
        # spectrum -- no edge regions.
        return auto_spline_baseline(freq, np.real(spectrum), n_points=n_points)["baseline"]
    if method == "whittaker":
        # Before the candidate-region setup below, which this method
        # has no use for -- see its own docstring.
        return estimate_baseline_whittaker(
            freq, np.real(spectrum), lam=lam, d=diff_order,
            method="arpls" if asymmetry is None else "asls",
            p=0.001 if asymmetry is None else asymmetry,
        )
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
        return spline(freq)
    if method in ("linear", "polynomial"):
        coeffs, x_ref = estimate_baseline_polynomial(
            freq, np.real(spectrum), baseline_regions=baseline_regions,
            frac=frac, sigma=sigma, max_iter=max_iter,
            degree=(1 if method == "linear" else degree),
        )
        return np.polyval(coeffs, freq - x_ref)
    if method == "sinusoidal":
        coeffs, fitted_period = estimate_baseline_sinusoidal(
            freq, np.real(spectrum), baseline_regions=baseline_regions,
            frac=frac, sigma=sigma, max_iter=max_iter, n_harmonics=n_harmonics, period=period,
        )
        return evaluate_baseline_sinusoidal(coeffs, fitted_period, freq)
    raise ValueError(
        "method must be 'linear', 'polynomial', 'sinusoidal', 'spline', 'spline_auto', "
        f"'whittaker', 'manual_polynomial' or 'manual_sine', got {method!r}"
    )


def baseline_correct(freq: np.ndarray, spectrum: np.ndarray,
                      baseline_regions: list[tuple[float, float]] | tuple[float, float] | None = None,
                      frac: float = 0.05, sigma: float = 2.5, max_iter: int = 10,
                      method: str = "linear", smoothing: float | None = None,
                      degree: int = 2, n_harmonics: int = 1, period: float | None = None,
                      lam: float | None = None, asymmetry: float | None = None,
                      diff_order: int = 2, manual_params=None, manual_fixed=None,
                      manual_refit: bool = False, n_points: int | None = None) -> np.ndarray:
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
    - ``"spline_auto"`` (:func:`opennmr.processing.fitting.
      auto_spline_baseline`) -- a cubic spline through points placed
      automatically wherever the spectrum is baseline (Golotvin-Williams
      test), anywhere in it, ``n_points`` of them at most (None: 20).
      Follows a roll in the middle of the spectrum that edge-based
      methods cannot see.
    - ``"whittaker"`` (:func:`opennmr.processing.fitting.
      estimate_baseline_whittaker`) -- **the one that does not need
      candidate regions at all.** Every method above starts from "the
      first and last ``frac`` of the spectrum are baseline" and
      sigma-clips from there, which is wrong for a crowded spectrum,
      one with signal at the edges, or one whose baseline rolls in
      the middle where there are no candidate points to see it. This
      fits a penalised smooth curve to the *whole* spectrum and
      iteratively down-weights whatever sits above it, so it falls to
      the bottom of the data on its own. ``baseline_regions``/``frac``/
      ``sigma`` do not apply to it.

    ``smoothing`` : ``method="spline"`` only -- see
    :func:`estimate_baseline_spline`'s own ``smoothing`` parameter
    (``None`` picks a data-driven default).

    ``degree`` : ``method="polynomial"`` only -- polynomial degree
    (1 = linear, 2 = quadratic, ...).

    ``n_harmonics``/``period`` : ``method="sinusoidal"`` only -- see
    :func:`estimate_baseline_sinusoidal`'s own parameters of the same
    names (``period=None`` picks a data-driven default: twice the
    candidate points' own x-range).

    ``lam``/``asymmetry``/``diff_order`` : ``method="whittaker"`` only.
    ``lam`` is the smoothness penalty (``None`` picks a point-count-
    scaled default); ``asymmetry`` is ``None`` for adaptive arPLS
    weighting (the default, and the right choice with a dominant
    solvent peak) or a small number for fixed-``p`` AsLS;
    ``diff_order`` is which order of difference is penalised (2 =
    curvature). See :func:`estimate_baseline_whittaker`.

    ``baseline_regions`` : explicit (lo, hi) ppm window(s) known to be
    peak-free -- e.g. from the Analysis tab's own peak table, expanded
    a little past each peak's FWHM -- for a spectrum where the default
    edge-only assumption doesn't hold (a peak near either end, or a
    baseline that isn't well represented by just the two edges).

    ``method="manual_polynomial"`` / ``"manual_sine"`` : the user's own
    model, ``A x^4 + B x^3 + C x^2 + D x + E`` or ``A sin(B x + C) + D``
    (x in ppm; see :mod:`opennmr.notebook.manual_baseline`).
    ``manual_params`` (ordered A, B, ...) are evaluated as given; with
    ``manual_refit`` (or no ``manual_params``) the model is instead
    fitted to ``baseline_regions`` on this trace, holding the
    parameters flagged in ``manual_fixed`` (booleans, same order) at
    their ``manual_params`` values. No sigma-clipping: the regions are
    the user's statement of where the baseline is."""
    return spectrum - compute_baseline(
        freq, spectrum, baseline_regions=baseline_regions, frac=frac, sigma=sigma,
        max_iter=max_iter, method=method, smoothing=smoothing, degree=degree,
        n_harmonics=n_harmonics, period=period, lam=lam, asymmetry=asymmetry,
        diff_order=diff_order, manual_params=manual_params, manual_fixed=manual_fixed,
        manual_refit=manual_refit, n_points=n_points,
    )


# ============================================================
# Linear prediction
# ============================================================

def _reflect_unstable_roots(coeffs: np.ndarray) -> np.ndarray:
    """Force an AR model to be stable, by reflecting any of its
    poles that lie outside the unit circle back inside.

    Without this, linear prediction can diverge *exponentially*
    instead of extrapolating a decaying signal -- not a subtle
    inaccuracy: on real Bruker data whose own ``proc2s`` asks for F1
    linear prediction, three of five datasets came back with peak
    magnitudes of 1e85 to 1e104 against an expected ~1e11, which
    reduced their correlation with TopSpin to exactly 0.0000. The
    fit is least-squares and unconstrained, so nothing otherwise
    stops a pole landing outside the unit circle; TopSpin and
    NMRPipe both apply the same root reflection.

    A pole at ``r`` outside the circle is replaced by
    ``1/conj(r)`` -- the same frequency, mirrored to a decaying
    magnitude -- and the coefficients rebuilt from the corrected
    pole set. Returns the coefficients unchanged when every pole is
    already inside, which is the common case and costs one
    root-finding call.
    """
    # x[n] = sum_k coeffs[k] * x[n-1-k]  ->  characteristic
    # polynomial z^p - c0 z^(p-1) - ... - c_(p-1).
    roots = np.roots(np.concatenate(([1.0], -np.asarray(coeffs))))
    outside = np.abs(roots) > 1.0
    if not outside.any():
        return coeffs
    roots = roots.copy()
    # Guard the degenerate r == 0 case, which has no reflection.
    safe = outside & (np.abs(roots) > 0)
    roots[safe] = 1.0 / np.conj(roots[safe])
    return -np.poly(roots)[1:]


def linear_predict(fid: np.ndarray, order: int = 20, n_predict: int = 8,
                    direction: str = "forward", stabilise: bool = True) -> np.ndarray:
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

    ``stabilise`` : reflect any AR poles that land outside the unit
    circle back inside before predicting (see
    :func:`_reflect_unstable_roots`). On by default because an
    unconstrained least-squares fit can otherwise diverge
    exponentially rather than extrapolate a decaying signal --
    observed on real data, not hypothetical.

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

    # Forward only. Backward prediction reverses the FID first, so a
    # normally-decaying signal becomes a *growing* series whose poles
    # legitimately sit outside the unit circle -- reflecting them
    # forces decay in the reversed direction, which is exactly wrong
    # (it predicts a *smaller* signal before t=0, when the true one is
    # larger). Caught by the existing backward-reconstruction test,
    # which reproduces a known signal and started failing by 75%.
    if stabilise and direction == "forward":
        coeffs = _reflect_unstable_roots(coeffs)

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

    **Vendor conventions are normalised before this point, not here.**
    This function had a ``conjugate_f1`` flag for one release, because
    Bruker's F1 quadrature sense is the opposite of X-Pulse's. That was
    a vendor flag inside a deliberately vendor-agnostic module, so it
    has been moved into the Bruker reader, which now negates the
    sine-modulated rows itself -- exactly equivalent
    (``a - 1j*b == a + 1j*(-b)``) and leaves this module free of vendor
    knowledge.

    The convention was identified against ground truth, not guessed:
    solving ``interf[k] ~ c1*a + c2*b + c3*conj(a) + c4*conj(b)``
    against TopSpin's own F1 interferogram, recovered by inverse-
    transforming its finished 2D along F1 (see below).

    **Each row's F2 imaginary part must not leak into F1's quadrature.**
    This is the single mistake both hypercomplex branches used to make,
    and it has a characteristic look: an *hourglass* (bowtie) contour
    shape, reported by a user as "the phases of f2 and f1 are somehow
    offset from each other's phases", which is exactly what it is.

    After the F2 FFT each row ``a`` is complex: ``Re(a)`` is F2
    absorption, ``Im(a)`` F2 dispersion. Combining whole complex rows
    as ``a + 1j*b`` looks like the textbook hypercomplex step but is
    not it, because ``a + 1j*b == (Re(a) + 1j*Re(b)) + 1j*(Im(a) +
    1j*Im(b))`` -- the F2-real hypercomplex series *plus 1j times* the
    F2-imaginary one. F1-transforming that and taking the real part
    yields ``RR - II``, i.e. ``A1*A2 - D1*D2``: the phase-twist
    lineshape, whose contours are the hourglass. The pure double-
    absorption spectrum wanted is ``RR`` alone, so the F2 imaginary
    has to be dropped (or cancelled) *before* the F1 transform.

    **States (4) / States-TPPI (5)**: row ``2k`` is the "cosine"-
    modulated row and row ``2k+1`` the "sine"-modulated row for the
    *same* t1 increment ``k``, and the F1 point is built from their F2
    **real parts only** -- ``decoded[k] = Re(row[2k]) + 1j*Re(row[2k+1])``.
    The extra ``1j`` is a *second*, independent multiplication for the
    F1 axis specifically; that is what "hypercomplex" means, and it
    only works if what it multiplies is real.

    States-TPPI (5) adds ``(-1)**k`` on top -- displaces the axial
    (unmodulated-artifact) peak to the edge of the F1 spectral window
    instead of its centre, which is the entire point of the TPPI-style
    receiver-phase increment. Confirmed via NMR Wiki's States-TPPI
    reference.

    **Echo-Antiecho (6, a.k.a. Rance-Kay)**: row ``2k`` is the "echo"
    (P-type) and row ``2k+1`` the "antiecho" (N-type), and they combine
    as ``decoded[k] = 1j * (conj(echo) - antiecho)``. Here the F2
    imaginary parts cancel rather than being discarded: writing it out,
    ``Re(decoded) == Im(echo) + Im(antiecho)`` and ``Im(decoded) ==
    Re(echo) - Re(antiecho)``, so the cosine- and sine-modulated
    components are each a single real F2 plane, as the hypercomplex
    step requires -- just with the F2 real/imaginary roles distributed
    across the pair differently than the naive textbook form suggests
    (the gradient-selected pair is 90 degrees apart in F2 as well as
    opposite in F1). No raw-row transform is applied before the F2 FFT;
    see ``io.bruker.read_2d``'s own ``fn_mode == 6`` note for the two
    earlier attempts at one and why both were wrong.

    **How both forms were measured** (2026-09-06, superseding two
    earlier rounds of pattern-guessing that are recorded in
    ``io.bruker.read_2d`` because their failure modes are instructive):
    TopSpin's own recombined F1 interferogram is recoverable from a
    finished 2D -- undo ``proc2s``'s ``PHC0``/``PHC1``, then inverse-
    FFT along F1. The pair to inverse-transform is ``2rr + 1j*2ri``;
    that pairing is not a guess either, see this module's caller note
    and ``io.bruker``'s ``_PROCESSED_BLOCK_NAMING_NOTE``. Fitting
    ``interf[k] ~ c1*a + c2*b + c3*conj(a) + c4*conj(b)`` per t1 point
    against real Echo-Antiecho data (``test_data/Bruker Example
    Data/Cyclosporine/203``) then gives, unambiguously, coefficients
    of 1.0000 on ``conj(a)`` and ``b`` and 0.0003/0.0004 on the other
    two, at a median relative residual of **0.0172**, with a per-point
    ``|corr|`` of **0.997-1.0000** and a scale factor whose phase is a
    constant **+90 degrees** across every t1 point (its magnitude being
    just the F1 window, halved at k=0 by ``FCOR``). That constant 90
    degrees is the ``1j``; dropping it costs everything, since it turns
    F1 absorption into F1 dispersion -- end-to-end correlation against
    TopSpin's own ``2rr`` is +0.9948 with it and +0.0010 without.

    Overall, against TopSpin's ``2rr``: Echo-Antiecho +0.4448 ->
    **+0.9948**, States-TPPI +0.6883 -> **+0.9986**.

    **What the caller gets back, and why it matters.** For modes 4/5/6
    the returned array's imaginary part is the **F1** quadrature
    partner, not F2's -- the F2 imaginary is spent here and does not
    survive the F1 transform. After that transform the real part is
    TopSpin's ``2rr`` and the imaginary part its ``2ri``, which is
    exactly the pair F1 phasing needs (and the same pair
    ``io.bruker.read_2d_processed`` returns, so the raw and processed
    readers now agree on what ``data_imag`` means). Re-phasing **F2**
    on such a spectrum therefore cannot be done from the returned
    array and requires a rebuild from the raw FID. Modes 0-3 are
    unaffected: their ``data_imag`` is still F2's own.

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
        # Real parts only -- see this function's own "Each row's F2
        # imaginary part must not leak into F1's quadrature" note.
        decoded = np.real(first) + 1j * np.real(second)
        if fn_mode == 5:
            sign = np.where(np.arange(n_pairs) % 2 == 0, 1.0, -1.0)
            decoded = decoded * sign[:, np.newaxis]
    else:
        decoded = 1j * (np.conj(first) - second)
    return decoded
