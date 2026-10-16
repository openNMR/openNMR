"""
DNP: enhancement factors and field-sweep profiles.

Two measurements, each available from a notebook. (The GUI's Integrate ▸
DNP sections use the step-by-step :mod:`opennmr.notebook.dnp_workflow`
since 2026-09-30 -- align, confirm, then measure -- which builds on the
measuring functions here.)

**Enhancement factor** (:func:`enhancement`). Spectrum 1 over spectrum 2
-- normally microwaves on over microwaves off -- measured two
independent ways: the height of a picked peak, and the integral of a
chosen region. Doing both is encouraged but not required. The two
estimate the same number only if the line shape is unchanged between
the spectra, so disagreement is itself informative: typically
microwave-induced broadening or heating, a baseline problem, or a
region that does not hold the whole line. When both are given, the
result says whether they agree within their errors.

**Field sweep** (:func:`field_sweep`). In a DNP field profile the
magnet's field B0 is swept (sweep coil / BSMS field value) while the
microwave frequency is fixed, and the observe frequency SFO1 is moved
by hand to follow the NMR line. On a fixed-reference ppm axis the line
therefore appears to move by thousands of ppm from one spectrum to the
next, although nothing chemical has changed: only B0 and the carrier.

Everything here therefore works on the **offset from each spectrum's
own carrier** (Hz), where the line sits near zero in every spectrum,
and the field is computed from the nucleus's gyromagnetic ratio:
``B0 = nu / (gamma (1 + delta))``.

**The line does not sit exactly on the carrier, and this matters.**
SFO1 is set by hand, and in the first dataset this was built against
(23Na, Na2CdMnFe(CN)6, Lyon, 2026-05) the line sat anywhere from -7.6
to +13.8 kHz off its carrier -- for 23Na, up to 1.2 mT -- while the
BSMS steps being resolved were ~0.4 mT, and the line itself is ~1.1
kHz wide. A field taken from SFO1 alone put some points in the wrong
order along the sweep. So :func:`field_sweep` **finds the line in every
spectrum** (the largest signal within ``window`` of the pick) and takes
the field from the line's own absolute frequency, which is the direct
measurement of B0 at the sample. The SFO1-based field is reported
alongside it, with their difference, and is used only as a flagged
fallback for a spectrum where no line was found (at a zero-crossing of
the profile there may be nothing to find; a noise maximum is then not
passed off as a peak).

**Chemical shift and the absolute field.** ``nu = gamma B0 (1 + delta)``,
so an absolute field needs the line's shift delta. By default delta is
read off the reference spectrum's own ppm axis at the tracked line --
equivalent to assuming the reference spectrum was taken at the field
its ppm axis implies (B0 = SF / gamma). Pass ``shift_ppm`` to override.
delta changes every field by the same factor, 1 ppm -> 1e-6 B0 (9.4 uT
at 9.4 T): it moves the whole profile, never its shape, and field
*differences* are unaffected.

Intensities are compared on one scale: every openNMR importer divides
by NS already (data are per scan), and ``correct_rg`` (default True)
also removes differences in receiver gain, which openNMR does not
normalise. TopSpin's own processed ``1r`` files are *not* on one scale
(``NC_proc`` varies between experiments) and are not used.
"""

from __future__ import annotations

import csv
import math

import numpy as np

from ..core.spectrum import NMRSpectrum
from ..processing.fitting import estimate_baseline_robust, estimate_noise_level
from ..processing.relaxation import _trapz
from .utils import find_gamma

__all__ = [
    "carrier_frequency_mhz", "reference_frequency_mhz", "absolute_frequency_hz",
    "carrier_offset_hz", "field_from_frequency", "field_from_sfo1",
    "peak_height", "region_integral", "enhancement", "enhancement_report",
    "print_enhancement_report", "field_sweep", "field_sweep_table", "field_sweep_to_csv",
    "field_sweep_report", "print_field_sweep_report", "plot_field_profile",
    "plot_sweep_spectra", "plot_enhancement_spectra", "profile_values",
]

#: Search half-width (Hz on the carrier-offset axis) for finding the
#: line in each spectrum of a sweep. Large enough for the 14 kHz carrier
#: mismatches seen in the first real sweep, small enough (for most
#: nuclei at these fields) to exclude a different resonance.
DEFAULT_WINDOW_HZ = 20_000.0

#: A line is "found" only if it stands at least this many noise
#: standard deviations above the baseline.
DEFAULT_MIN_SNR = 5.0


# ============================================================
# Frequencies and fields
# ============================================================

def _one_d(spectrum: NMRSpectrum, what: str = "spectrum") -> None:
    if spectrum.ndim != 1:
        raise ValueError(
            f"DNP analysis needs 1D spectra; {what} is {spectrum.ndim}D. "
            "Extract a slice first.")


def _acqus(spectrum: NMRSpectrum) -> dict:
    raw = spectrum.metadata.raw or {}
    acqus = raw.get("acqus")
    return acqus if isinstance(acqus, dict) else {}


def reference_frequency_mhz(spectrum: NMRSpectrum) -> float:
    """The frequency the ppm axis is referenced to (Bruker ``SF``), MHz:
    the one ``ppm = 0`` corresponds to.

    This is what openNMR stores as ``metadata.sfo1`` -- the name is
    historical; it is *not* the observe/carrier frequency (see
    :func:`carrier_frequency_mhz`)."""
    sf = spectrum.metadata.sfo1
    if sf:
        return float(sf)
    axis = spectrum.axes[-1]
    if axis.ppm is not None and axis.hz is not None:
        nonzero = np.abs(axis.ppm) > 1e-9
        if np.any(nonzero):
            return float(np.median(axis.hz[nonzero] / axis.ppm[nonzero]))
    raise ValueError("This spectrum records no reference frequency (SF).")


def carrier_frequency_mhz(spectrum: NMRSpectrum) -> tuple[float, str]:
    """The observe (carrier) frequency the spectrum was acquired at, MHz,
    and where it came from.

    Bruker ``SFO1`` from ``acqus`` when available; otherwise ``BF1 + O1``;
    otherwise the reference frequency, which is only right if the
    carrier sat at 0 ppm -- the source says so, so a caller can warn."""
    sfo1 = _acqus(spectrum).get("SFO1")
    try:
        if sfo1 is not None and float(sfo1) > 0:
            return float(sfo1), "SFO1"
    except (TypeError, ValueError):
        pass
    md = spectrum.metadata
    if md.bf1 and md.o1 is not None:
        return float(md.bf1) + float(md.o1) * 1e-6, "BF1 + O1"
    return reference_frequency_mhz(spectrum), "reference frequency (no carrier recorded)"


def absolute_frequency_hz(spectrum: NMRSpectrum) -> np.ndarray:
    """Absolute frequency of every point of a 1D spectrum, Hz."""
    _one_d(spectrum)
    axis = spectrum.axes[-1]
    sf = reference_frequency_mhz(spectrum)
    if axis.hz is not None:
        return sf * 1e6 + np.asarray(axis.hz, dtype=float)
    return sf * 1e6 * (1.0 + np.asarray(axis.ppm, dtype=float) * 1e-6)


def carrier_offset_hz(spectrum: NMRSpectrum) -> np.ndarray:
    """Every point's offset from the spectrum's own carrier, Hz -- the
    axis on which a field-swept line stays put when SFO1 follows it."""
    carrier, _source = carrier_frequency_mhz(spectrum)
    return absolute_frequency_hz(spectrum) - carrier * 1e6


def _nucleus(spectrum: NMRSpectrum) -> str:
    from .two_d import canonical_nucleus
    nucleus = canonical_nucleus(spectrum.metadata.nucleus)
    if not nucleus:
        raise ValueError("This spectrum records no nucleus, so no gyromagnetic ratio is known.")
    return nucleus


def _gamma_hz_per_t(nucleus: str) -> float:
    return float(find_gamma(nucleus)["gamma"])


def field_from_frequency(frequency_hz, nucleus: str, shift_ppm: float = 0.0):
    """B0 (T) at which ``nucleus`` resonates at ``frequency_hz`` with
    chemical shift ``shift_ppm``: ``nu / (gamma (1 + delta))``, gamma
    from openNMR's gyromagnetic-ratio table."""
    gamma = _gamma_hz_per_t(nucleus)
    return np.asarray(frequency_hz, dtype=float) / (gamma * (1.0 + shift_ppm * 1e-6))


def field_from_sfo1(spectrum: NMRSpectrum, shift_ppm: float = 0.0) -> float:
    """B0 (T) implied by the spectrum's carrier frequency, **assuming the
    line (of shift ``shift_ppm``) sat exactly on the carrier**. Usually
    close; see this module's docstring for why :func:`field_sweep`
    prefers the line's own frequency."""
    carrier, _source = carrier_frequency_mhz(spectrum)
    return float(field_from_frequency(carrier * 1e6, _nucleus(spectrum), shift_ppm))


# ============================================================
# Measuring one spectrum
# ============================================================

def _axis(spectrum: NMRSpectrum, unit: str) -> np.ndarray:
    unit = unit.lower()
    if unit == "ppm":
        return np.asarray(spectrum.axes[-1].ppm, dtype=float)
    if unit == "hz":
        axis = spectrum.axes[-1]
        if axis.hz is not None:
            return np.asarray(axis.hz, dtype=float)
        return np.asarray(axis.ppm, dtype=float) * reference_frequency_mhz(spectrum)
    if unit == "offset":
        return carrier_offset_hz(spectrum)
    raise ValueError(f"unit must be 'ppm', 'hz' or 'offset', not {unit!r}")


BASELINES = ("local", "global", "none")


def _check_baseline(baseline: str) -> str:
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {BASELINES}, not {baseline!r}")
    return baseline


def _noise_sigma(y) -> float:
    """Noise standard deviation, robust to a rolling baseline and to the
    lines themselves: the residual of ``y`` about a wide (101-point,
    quadratic) Savitzky-Golay smooth, measured by its MAD.

    Not the standard deviation of the spectrum's edges -- a rolled
    baseline inflated that 20-fold on the first real DNP data (a badly
    distorted microwave-on/off pair). Not point-to-point differences
    either: zero-filling and apodisation correlate adjacent points (lag-1
    correlation 0.4-0.6 on that data), which made the difference estimate
    ~18% low and every error bar with it. On spectra with flat baselines
    this estimator agrees with the edges' standard deviation to ~5%."""
    from scipy.signal import savgol_filter
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 7:
        return float(np.std(y))
    window = min(101, n - 1 if (n - 1) % 2 else n - 2)
    window = max(window, 5)
    residual = y - savgol_filter(y, window, 2)
    return float(1.4826 * np.median(np.abs(residual - np.median(residual))))


def _prepared(spectrum: NMRSpectrum, baseline: str = "local", noise_region=None):
    """(Hz axis, real data, noise sigma). With ``baseline="global"`` the
    data have the Quantify tab's robust linear baseline (fitted to the
    spectrum's edges) removed; ``"local"`` baselines are removed per
    measurement instead (:func:`_local_line`). Noise from point-to-point
    differences (:func:`_noise_sigma`), or from ``noise_region`` (Hz) if
    given."""
    _one_d(spectrum)
    x_hz = _axis(spectrum, "hz")
    y = np.asarray(spectrum.data, dtype=float)
    if _check_baseline(baseline) == "global":
        b0, b1, x_ref = estimate_baseline_robust(x_hz, y)
        y = y - (b0 + b1 * (x_hz - x_ref))
    if noise_region is not None:
        noise = float(estimate_noise_level(x_hz, y, noise_region=noise_region))
    else:
        noise = _noise_sigma(y)
    return x_hz, y, noise


def _local_line(x, y, lo, hi):
    """A smooth baseline under ``lo..hi``, fitted to the data just
    *outside* it: a quadratic through margins of half the width (at least
    5 points) on each side. Returns ``f(x)``.

    This is what follows a rolling baseline -- a badly phased or
    distorted spectrum can sit a large fraction of the line's height off
    zero under the line, which a baseline fitted to the spectrum's far
    edges cannot follow. A *straight* local line is not enough either:
    under a strongly curved baseline it cut an integral by a third in
    testing, where the quadratic stays within a few percent. With data on
    one side only (a region at the spectrum's edge) it falls back to a
    constant."""
    lo, hi = sorted((float(lo), float(hi)))
    step = float(np.median(np.abs(np.diff(x)))) if len(x) > 1 else 0.0
    margin = max(0.5 * (hi - lo), 5 * step)
    left = (x >= lo - margin) & (x < lo)
    right = (x > hi) & (x <= hi + margin)
    n_left, n_right = int(np.count_nonzero(left)), int(np.count_nonzero(right))
    if n_left + n_right < 2:
        return lambda xx: np.zeros_like(np.asarray(xx, dtype=float))
    side = left | right
    xs, ys = x[side], y[side]
    if n_left < 3 or n_right < 3 or np.ptp(xs) == 0:
        c = float(np.mean(ys))
        return lambda xx: np.full_like(np.asarray(xx, dtype=float), c)
    centre = 0.5 * (lo + hi)
    coeffs = np.polyfit(xs - centre, ys, 2)
    return lambda xx: np.polyval(coeffs, np.asarray(xx, dtype=float) - centre)


#: Where a peak's local baseline is fitted, in FWHM either side. A
#: Lorentzian's tail is still 1/(1 + 4k^2) of its height at k FWHM --
#: 2.7% at 3, 1.0% at 5 -- and a baseline fitted on the tail makes the
#: height read low by about that much.
BASELINE_FWHMS = 5.0


def _line_window(x, y, j, fwhms: float = BASELINE_FWHMS):
    """``(lo, hi)`` around the line at index ``j``: +-``fwhms`` FWHM, the FWHM
    measured at half height above the median of the surroundings. Used
    to place a local baseline under a peak whose height is wanted."""
    n = len(y)
    span = max(20, n // 20)
    a, b = max(0, j - span), min(n, j + span + 1)
    floor = float(np.median(y[a:b]))
    top = y[j] - floor
    half = 0.5 * abs(top)
    left = j
    while left > a and abs(y[left] - floor) > half:
        left -= 1
    right = j
    while right < b - 1 and abs(y[right] - floor) > half:
        right += 1
    fwhm = max(abs(x[right] - x[left]), 2 * float(np.median(np.abs(np.diff(x)))))
    return x[j] - fwhms * fwhm, x[j] + fwhms * fwhm


def _height_at(x, y, j, baseline, fwhms: float = BASELINE_FWHMS):
    """Signed height of the point ``j``, above a local baseline (fitted
    just outside +-``fwhms`` FWHM) if asked."""
    if baseline != "local":
        return float(y[j])
    lo, hi = _line_window(x, y, j, fwhms)
    return float(y[j] - _local_line(x, y, lo, hi)(x[j]))


def _area_between(x, y, lo, hi, noise, baseline):
    """Integral over ``lo..hi`` (x is the Hz axis), above a local baseline
    if asked."""
    if baseline == "local":
        y = y - _local_line(x, y, lo, hi)(x)
    return _integrate(x, y, lo, hi, noise)


def _rg(spectrum: NMRSpectrum) -> float | None:
    try:
        rg = float(_acqus(spectrum).get("RG"))
    except (TypeError, ValueError):
        return None
    return rg if rg > 0 else None


def _find_extremum(x, y, centre, halfwidth, detrend: bool = True):
    """Index of the largest |y| with ``|x - centre| <= halfwidth``, and
    whether the true maximum probably lies *outside* the window.

    That second part is not "is the maximum on the last point": a line
    just beyond the window puts its rising tail inside it, and with noise
    the largest point of that tail can sit a point or two in from the
    edge, pass an S/N test, and be taken for the line. So the search is
    repeated over a window 25% wider; if the maximum there lies outside
    the original window, the original one is flagged. A window holding
    no point falls back to the nearest one (flagged).

    The search runs on the data less a straight line through the
    window's outer parts (``detrend``): under a rolling baseline larger
    than the line, the largest raw |y| is baseline, not the line."""
    distance = np.abs(x - centre)
    mask = distance <= halfwidth
    if not np.any(mask):
        return int(np.argmin(distance)), True
    wider = np.flatnonzero(distance <= 1.25 * halfwidth)
    yw = y[wider].astype(float)
    if detrend and len(wider) >= 10:
        xw = x[wider]
        edge = np.abs(xw - centre) >= 0.85 * halfwidth
        if np.count_nonzero(edge) >= 4 and np.ptp(xw[edge]) > 0:
            b1, b0 = np.polyfit(xw[edge], yw[edge], 1)
            yw = yw - (b0 + b1 * xw)
    inner = distance[wider] <= halfwidth
    j_local = int(np.flatnonzero(inner)[np.argmax(np.abs(yw[inner]))])
    k_local = int(np.argmax(np.abs(yw)))
    at_edge = bool(distance[wider[k_local]] > halfwidth and abs(yw[k_local]) > abs(yw[j_local]))
    return int(wider[j_local]), at_edge


def _integrate(x, y, lo, hi, noise):
    """Signed trapezoid integral of ``y`` over ``lo <= x <= hi`` and its
    noise-propagated error -- the same rule, sign convention and error
    propagation as :func:`opennmr.processing.relaxation.NMR2D_integrate_slices`."""
    lo, hi = sorted((float(lo), float(hi)))
    mask = (x >= lo) & (x <= hi)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"Region ({lo:g}, {hi:g}) contains fewer than 2 points.")
    xr, yr = x[mask], y[mask]
    order = np.argsort(xr)
    xr, yr = xr[order], yr[order]
    area = float(_trapz(yr, xr))
    w = np.empty_like(xr)
    w[1:-1] = 0.5 * np.abs(xr[2:] - xr[:-2])
    w[0] = 0.5 * np.abs(xr[1] - xr[0])
    w[-1] = 0.5 * np.abs(xr[-1] - xr[-2])
    return area, float(noise * np.sqrt(np.sum(w ** 2)))


def peak_height(spectrum: NMRSpectrum, position: float, unit: str = "ppm",
                snap: float | None = None, baseline: str = "local",
                noise_region=None, baseline_fwhms: float = BASELINE_FWHMS) -> dict:
    """Signed height of the peak nearest ``position``.

    ``snap`` (in ``unit``) : the largest |signal| within ``position ±
    snap`` is taken, so a click need not land on the maximum. Default
    1% of the spectral width. ``0`` takes the point nearest ``position``.

    ``baseline`` : 'local' (default) -- height above a quadratic fitted
    just outside +-5 FWHM of the peak; 'global' -- above the
    Quantify tab's baseline fitted to the spectrum's edges; 'none'.

    Returns ``{"value", "error", "position", "unit", "at_edge", "snr"}``;
    ``error`` is the noise standard deviation, ``at_edge`` True when the
    maximum sits on the edge of the snap window (widen it)."""
    x_hz, y, noise = _prepared(spectrum, baseline, noise_region)
    x = _axis(spectrum, unit)
    if snap is None:
        snap = 0.01 * float(np.ptp(x))
    j, at_edge = _find_extremum(x, y, float(position), float(snap))
    value = _height_at(x_hz, y, j, baseline, baseline_fwhms)
    return {"value": value, "error": noise, "position": float(x[j]), "unit": unit,
            "at_edge": at_edge, "snr": abs(value) / noise if noise > 0 else math.inf}


def region_integral(spectrum: NMRSpectrum, lo: float, hi: float, unit: str = "ppm",
                    baseline: str = "local", noise_region=None) -> dict:
    """Signed integral over ``lo..hi`` (given in ``unit``), **always
    taken against a Hz axis** so it does not depend on the reference
    frequency (which differs across a field sweep). ``baseline``:
    'local' (default; a quadratic fitted just outside the region),
    'global' (the Quantify tab's) or 'none'. Returns
    ``{"value", "error", "lo", "hi", "unit"}``."""
    x_hz, y, noise = _prepared(spectrum, baseline, noise_region)
    x = _axis(spectrum, unit)
    lo, hi = sorted((float(lo), float(hi)))
    mask = (x >= lo) & (x <= hi)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"Region ({lo:g}, {hi:g}) {unit} contains fewer than 2 points.")
    sel = np.flatnonzero(mask)
    area, err = _area_between(x_hz, y, x_hz[sel].min(), x_hz[sel].max(), noise, baseline)
    return {"value": area, "error": err, "lo": lo, "hi": hi, "unit": unit}


def _ratio(a, ea, b, eb):
    if b == 0:
        return math.nan, math.nan
    r = a / b
    rel = 0.0
    if a != 0:
        rel += (ea / a) ** 2
    rel += (eb / b) ** 2
    return r, abs(r) * math.sqrt(rel)


# ============================================================
# Enhancement factor
# ============================================================

def enhancement(spectrum_1: NMRSpectrum, spectrum_2: NMRSpectrum, peak=None, region=None,
                unit: str = "ppm", snap: float | None = None, peak_2=None, region_2=None,
                baseline: str = "local", correct_rg: bool = True, noise_region=None,
                labels=("spectrum 1", "spectrum 2")) -> dict:
    """Enhancement factor spectrum_1 / spectrum_2 (normally microwaves on /
    off), by peak height and/or by region integral.

    ``peak`` : position of the peak to compare (in ``unit``: 'ppm' on
    each spectrum's own ppm axis, 'hz', or 'offset' from the carrier).
    Heights are taken at the largest |signal| within ``± snap`` in each
    spectrum. ``peak_2`` : a different position for spectrum 2 (default:
    the same one).

    ``region`` : ``(lo, hi)`` to integrate, in ``unit``; ``region_2`` for
    a different region in spectrum 2 (default: the same one).

    ``baseline`` : 'local' (default), 'global' or 'none' -- see
    :func:`peak_height` and :func:`region_integral`.

    Give at least one of ``peak``/``region``; both is recommended -- they
    agree only if the line shape is unchanged, and the result says
    whether they do.

    Returns ``{"intensity": {...} | None, "area": {...} | None, "rg":
    {...}, "notes": [str], "agreement": {...} | None, ...}``. Each of
    ``intensity``/``area`` holds both spectra's values (after RG
    correction), their raw values, errors, positions, ``ratio`` and
    ``ratio_error``."""
    if peak is None and region is None:
        raise ValueError("Give a peak position, a region, or (better) both.")
    _check_baseline(baseline)
    for s, name in ((spectrum_1, labels[0]), (spectrum_2, labels[1])):
        _one_d(s, name)
    notes: list[str] = []

    n1, n2 = _nucleus(spectrum_1), _nucleus(spectrum_2)
    if n1 != n2:
        notes.append(f"The spectra observe different nuclei ({n1}, {n2}); a ratio of them is "
                     "not an enhancement factor.")

    rg1, rg2 = _rg(spectrum_1), _rg(spectrum_2)
    scale_1 = 1.0
    if correct_rg and rg1 and rg2 and rg1 != rg2:
        scale_1 = rg2 / rg1
        notes.append(f"Receiver gains differ ({labels[0]}: {rg1:g}, {labels[1]}: {rg2:g}); "
                     f"{labels[0]} scaled by {scale_1:.6g} to {labels[1]}'s gain.")
    elif correct_rg and (rg1 is None or rg2 is None):
        notes.append("Receiver gain not recorded for both spectra, so not corrected -- check "
                     "they were acquired with the same RG.")
    elif not correct_rg and rg1 and rg2 and rg1 != rg2:
        notes.append(f"Receiver gains differ ({rg1:g}, {rg2:g}) and were NOT corrected.")

    c1, _ = carrier_frequency_mhz(spectrum_1)
    c2, _ = carrier_frequency_mhz(spectrum_2)
    if abs(c1 - c2) / c2 * 1e6 > 1.0 and unit.lower() != "offset":
        notes.append(f"The carriers differ by {(c1 - c2) * 1e6:,.0f} Hz ({abs(c1 - c2) / c2 * 1e6:.1f} "
                     "ppm): if the field differs too, the peak is not at the same ppm in both "
                     "-- compare on the carrier-offset axis (unit='offset').")

    result = {"labels": list(labels), "unit": unit, "intensity": None, "area": None,
              "rg": {"1": rg1, "2": rg2, "scale_1": scale_1, "corrected": bool(correct_rg)},
              "carrier_MHz": {"1": c1, "2": c2}, "notes": notes, "agreement": None,
              "settings": {"peak": peak, "peak_2": peak_2, "region": region, "region_2": region_2,
                           "snap": snap, "baseline": baseline, "correct_rg": correct_rg}}

    if peak is not None:
        p1 = peak_height(spectrum_1, peak, unit, snap, baseline, noise_region)
        p2 = peak_height(spectrum_2, peak if peak_2 is None else peak_2, unit, snap,
                         baseline, noise_region)
        v1, e1 = p1["value"] * scale_1, p1["error"] * scale_1
        r, er = _ratio(v1, e1, p2["value"], p2["error"])
        # Systematic: how much the ratio moves if the local baseline
        # under the peak is fitted at +-4 or +-6 FWHM instead of +-5.
        spread = 0.0
        if baseline == "local":
            for k in (BASELINE_FWHMS - 1.0, BASELINE_FWHMS + 1.0):
                q1 = peak_height(spectrum_1, peak, unit, snap, baseline, noise_region, k)
                q2 = peak_height(spectrum_2, peak if peak_2 is None else peak_2, unit, snap,
                                 baseline, noise_region, k)
                if q2["value"]:
                    spread = max(spread, abs(q1["value"] * scale_1 / q2["value"] - r))
        result["intensity"] = {"1": v1, "2": p2["value"], "error_1": e1, "error_2": p2["error"],
                               "raw_1": p1["value"], "raw_2": p2["value"],
                               "position_1": p1["position"], "position_2": p2["position"],
                               "snr_1": p1["snr"], "snr_2": p2["snr"],
                               "ratio": r, "ratio_error": er, "ratio_systematic": spread}
        for p, name in ((p1, labels[0]), (p2, labels[1])):
            if p["at_edge"]:
                notes.append(f"In {name} the largest signal is at the edge of the snap window "
                             f"({p['position']:g} {unit}): the peak may lie outside it -- widen "
                             "the snap or pick closer.")
        spread = abs(p1["position"] - p2["position"])
        if peak_2 is None and snap and spread > 0.5 * snap:
            notes.append(f"The two peak maxima were found {spread:g} {unit} apart: check both "
                         "are the same peak.")

    if region is not None:
        a1 = region_integral(spectrum_1, *region, unit=unit, baseline=baseline,
                             noise_region=noise_region)
        a2 = region_integral(spectrum_2, *(region if region_2 is None else region_2), unit=unit,
                             baseline=baseline, noise_region=noise_region)
        v1, e1 = a1["value"] * scale_1, a1["error"] * scale_1
        r, er = _ratio(v1, e1, a2["value"], a2["error"])
        # Systematic: how much the ratio moves if both regions are 20%
        # narrower or wider about their centres. On the first real data
        # this was ~10x the noise error -- the region, not the noise,
        # limits an area ratio.
        spread = 0.0
        reg_2 = region if region_2 is None else region_2
        for f in (0.8, 1.2):
            def scaled(reg, f=f):
                c, h = 0.5 * (reg[0] + reg[1]), 0.5 * abs(reg[1] - reg[0]) * f
                return c - h, c + h
            try:
                b1 = region_integral(spectrum_1, *scaled(region), unit=unit, baseline=baseline,
                                     noise_region=noise_region)["value"] * scale_1
                b2 = region_integral(spectrum_2, *scaled(reg_2), unit=unit, baseline=baseline,
                                     noise_region=noise_region)["value"]
            except ValueError:
                continue
            if b2:
                spread = max(spread, abs(b1 / b2 - r))
        result["area"] = {"1": v1, "2": a2["value"], "error_1": e1, "error_2": a2["error"],
                          "raw_1": a1["value"], "raw_2": a2["value"],
                          "region_1": (a1["lo"], a1["hi"]), "region_2": (a2["lo"], a2["hi"]),
                          "ratio": r, "ratio_error": er, "ratio_systematic": spread}

    if result["intensity"] and result["area"]:
        i, a = result["intensity"], result["area"]
        ri, ra = i["ratio"], a["ratio"]
        diff = ri - ra
        # Noise and the choice of baseline window / region both count.
        sigma = math.sqrt(i["ratio_error"] ** 2 + i["ratio_systematic"] ** 2
                          + a["ratio_error"] ** 2 + a["ratio_systematic"] ** 2)
        rel = abs(diff) / abs(ra) * 100 if ra else math.inf
        consistent = abs(diff) <= 2 * sigma if sigma > 0 else diff == 0
        result["agreement"] = {"difference": diff, "percent": rel, "sigma": sigma,
                               "consistent": bool(consistent)}
        if not consistent:
            notes.append(
                f"By height and by area the enhancement differs by {rel:.1f}% (more than twice "
                "the combined noise and region/baseline uncertainty): the line shape differs between the spectra (e.g. "
                "microwave-induced broadening or heating), the region misses part of the line, "
                "or the baseline differs. Height and area answer different questions -- report "
                "which one you used.")
    elif result["intensity"] and not result["area"]:
        notes.append("Measured by peak height only: integrate a region too, to check the line "
                     "shape is unchanged.")
    else:
        notes.append("Measured by area only: pick the peak too, to check against the height "
                     "ratio.")
    return result


def enhancement_report(result: dict) -> str:
    """Readable summary of :func:`enhancement`'s result."""
    l1, l2 = result["labels"]
    unit = result["unit"]
    lines = [f"DNP enhancement  ε = {l1} / {l2}"]
    i = result["intensity"]
    if i:
        lines.append(f"  By peak height: ε = {_fmt_sys(i['ratio'], i['ratio_error'], i['ratio_systematic'])}")
        lines.append(f"    {l1}: {_fmt(i['1'], i['error_1'])} at {i['position_1']:.6g} {unit}"
                     f" (S/N {i['snr_1']:.0f})")
        lines.append(f"    {l2}: {_fmt(i['2'], i['error_2'])} at {i['position_2']:.6g} {unit}"
                     f" (S/N {i['snr_2']:.0f})")
    a = result["area"]
    if a:
        lines.append(f"  By area:        ε = {_fmt_sys(a['ratio'], a['ratio_error'], a['ratio_systematic'])}")
        lines.append(f"    {l1}: {_fmt(a['1'], a['error_1'])} over {a['region_1'][0]:.6g}"
                     f" to {a['region_1'][1]:.6g} {unit}")
        lines.append(f"    {l2}: {_fmt(a['2'], a['error_2'])} over {a['region_2'][0]:.6g}"
                     f" to {a['region_2'][1]:.6g} {unit}")
    ag = result["agreement"]
    if ag:
        verdict = "agree" if ag["consistent"] else "DISAGREE"
        lines.append(f"  Height vs area: {verdict} ({ag['percent']:.1f}% apart, combined 1σ "
                     f"{ag['sigma']:.3g})")
    if i or a:
        lines.append("  (± noise; 'sys' is how far the ratio moves when the peak's baseline "
                     "window is ±4 or ±6 FWHM instead of ±5, or the region 20% narrower or wider)")
    s = result["settings"]
    lines.append(f"  Baseline: {_BASELINE_TEXT[s['baseline']]}; "
                 f"RG corrected: {'yes' if s['correct_rg'] else 'no'}; intensities per scan.")
    for note in result["notes"]:
        lines.append(f"  Note: {note}")
    return "\n".join(lines)


_BASELINE_TEXT = {"local": "local (quadratic fitted just outside each peak/region)",
                  "global": "global (robust line fitted to the spectrum's edges)",
                  "none": "not subtracted"}


def print_enhancement_report(result: dict) -> None:
    print(enhancement_report(result))


def _fmt_sys(value, error, systematic) -> str:
    """"1.078 ± 0.002 (± 0.010 sys)"."""
    text = _fmt(value, error)
    if systematic and np.isfinite(systematic) and systematic > 0:
        text += f" (± {systematic:.2g} sys)"
    return text


def _fmt(value, error) -> str:
    if error is None or not np.isfinite(error) or error <= 0:
        return f"{value:.4g}"
    digits = max(0, -int(math.floor(math.log10(error))) + 1)
    return f"{value:.{digits}f} ± {error:.{digits}f}" if digits < 8 else f"{value:.4g} ± {error:.2g}"


# ============================================================
# Field sweep
# ============================================================

def field_sweep(spectra, peak: float | None = None, region=None, reference: int = 0,
                window: float = DEFAULT_WINDOW_HZ, track: bool = True,
                min_snr: float = DEFAULT_MIN_SNR, field_from: str = "peak",
                shift_ppm: float | None = None, baseline: str = "local",
                correct_rg: bool = True, labels=None, noise_region=None) -> dict:
    """Signal against B0 across a DNP field sweep.

    ``spectra`` : 1D spectra of one nucleus, any order.

    ``peak`` : the line's offset from the carrier (Hz), as picked on the
    carrier-offset axis (:func:`plot_sweep_spectra`). ``region`` :
    ``(lo, hi)`` offsets (Hz) to integrate. At least one; both gives
    height and area together.

    ``track`` (default True) : find the line in each spectrum -- the
    largest |signal| within ``± window`` Hz of ``peak`` (or of the
    region's centre) -- and move the region with it. A spectrum where
    that signal is below ``min_snr`` noise standard deviations, or sits
    on the window's edge, is marked ``found=False``: its value is read at
    the pick position unmoved, and its field comes from SFO1.

    ``field_from`` : ``"peak"`` (default; the tracked line's absolute
    frequency) or ``"sfo1"`` (the carrier, assuming the line sits where
    it sat in the reference spectrum relative to its carrier). Both are
    always reported.

    ``baseline`` : 'local' (default), 'global' or 'none' -- see
    :func:`peak_height` and :func:`region_integral`.

    ``reference`` : index of the spectrum defining the chemical shift
    (read off its own ppm axis at the line, unless ``shift_ppm`` is
    given), the RG everything is scaled to, and ΔB0 = 0.

    Returns a dict with ``rows`` (one per spectrum, in the order given),
    ``nucleus``, ``gamma_Hz_per_T``, ``shift_ppm``, ``reference``,
    ``settings`` and ``notes``. Each row: ``label``, ``carrier_MHz``,
    ``rg``, ``ns``, ``found``, ``snr``, ``peak_offset_Hz``, ``shift_Hz``
    (how far the line was found from the pick), ``field_peak_T``,
    ``field_sfo1_T``, ``field_T``, ``field_source``, ``delta_B_mT``,
    ``sfo1_error_mT`` (SFO1 field minus peak field), ``intensity``,
    ``intensity_error``, ``area``, ``area_error``, ``region_Hz``.
    """
    spectra = list(spectra)
    if not spectra:
        raise ValueError("No spectra given.")
    if peak is None and region is None:
        raise ValueError("Give the line's position (peak), a region, or both.")
    if field_from not in ("peak", "sfo1"):
        raise ValueError("field_from must be 'peak' or 'sfo1'.")
    _check_baseline(baseline)
    if not 0 <= reference < len(spectra):
        raise ValueError(f"reference must index one of the {len(spectra)} spectra.")
    labels = list(labels) if labels is not None else [
        str(getattr(s.metadata, "source_path", None) or f"spectrum {k + 1}") for k, s in
        enumerate(spectra)]
    for k, s in enumerate(spectra):
        _one_d(s, labels[k])
    nuclei = {_nucleus(s) for s in spectra}
    if len(nuclei) != 1:
        raise ValueError(f"A field sweep must observe one nucleus; these observe {sorted(nuclei)}.")
    nucleus = nuclei.pop()
    gamma = _gamma_hz_per_t(nucleus)
    region = None if region is None else tuple(sorted(map(float, region)))
    anchor = float(peak) if peak is not None else 0.5 * (region[0] + region[1])
    notes: list[str] = []

    measured = []
    for s in spectra:
        x_hz, y, noise = _prepared(s, baseline, noise_region)
        off = carrier_offset_hz(s)
        carrier, source = carrier_frequency_mhz(s)
        if track:
            j, at_edge = _find_extremum(off, y, anchor, window)
        else:
            j, at_edge = int(np.argmin(np.abs(off - anchor))), False
        height = _height_at(x_hz, y, j, baseline)
        snr = abs(height) / noise if noise > 0 else math.inf
        found = bool(track and snr >= min_snr and not at_edge)
        position = float(off[j]) if found else anchor
        if not found:
            j = int(np.argmin(np.abs(off - anchor)))
            height = _height_at(x_hz, y, j, baseline)
        intensity = height
        measured.append({"x_hz": x_hz, "y": y, "off": off, "noise": noise, "carrier": carrier,
                         "carrier_source": source, "found": bool(found), "at_edge": bool(at_edge),
                         "snr": float(snr), "position": position, "intensity": intensity})

    ref = measured[reference]
    ref_spectrum = spectra[reference]
    if not ref["found"] and track:
        notes.append(f"The line was not found in the reference spectrum ({labels[reference]}); "
                     "its shift and field come from the pick position.")
    ref_abs = ref["carrier"] * 1e6 + ref["position"]
    if shift_ppm is None:
        sf = reference_frequency_mhz(ref_spectrum) * 1e6
        shift_ppm = (ref_abs - sf) / sf * 1e6
        shift_source = f"read off {labels[reference]}'s ppm axis at the line"
    else:
        shift_source = "given"
    factor = gamma * (1.0 + shift_ppm * 1e-6)
    ref_offset = ref["position"]

    rg_ref = _rg(ref_spectrum)
    rgs = [_rg(s) for s in spectra]
    if correct_rg and any(r is None for r in rgs):
        notes.append("Receiver gain not recorded for every spectrum; those were not corrected.")
    if len({r for r in rgs if r}) > 1:
        notes.append("Receiver gains differ across the sweep; " + (
            f"every spectrum scaled to the reference's RG ({rg_ref:g})." if correct_rg and rg_ref
            else "NOT corrected."))
    programs = {(s.metadata.pulse_program or "").strip("<> ") for s in spectra}
    programs.discard("")
    if len(programs) > 1:
        notes.append(f"The sweep mixes pulse programmes ({', '.join(sorted(programs))}): their "
                     "intensities are not directly comparable (echo vs. single pulse, "
                     "different delays).")
    sources = {m["carrier_source"] for m in measured}
    if any(src != "SFO1" and src != "BF1 + O1" for src in sources):
        notes.append("Some spectra record no carrier frequency: their offsets and fields are "
                     "relative to the ppm reference instead and are unreliable.")

    rows = []
    for k, (s, m) in enumerate(zip(spectra, measured)):
        scale = (rg_ref / rgs[k]) if (correct_rg and rg_ref and rgs[k]) else 1.0
        shift = (m["position"] - anchor) if m["found"] else 0.0
        field_sfo1 = (m["carrier"] * 1e6 + ref_offset) / factor
        field_peak = (m["carrier"] * 1e6 + m["position"]) / factor if m["found"] else math.nan
        if field_from == "peak" and m["found"]:
            field, field_source = field_peak, "peak"
        elif field_from == "sfo1":
            field, field_source = field_sfo1, "SFO1"
        else:
            field, field_source = field_sfo1, ("SFO1 (line not found)" if track
                                               else "SFO1 (tracking off)")
        row = {"label": labels[k], "carrier_MHz": m["carrier"], "carrier_source": m["carrier_source"],
               "rg": rgs[k], "rg_scale": scale, "ns": s.metadata.ns, "found": m["found"],
               "at_edge": m["at_edge"], "snr": m["snr"], "peak_offset_Hz": m["position"],
               "shift_Hz": shift, "field_peak_T": field_peak, "field_sfo1_T": field_sfo1,
               "field_T": field, "field_source": field_source,
               "sfo1_error_mT": (field_sfo1 - field_peak) * 1e3 if m["found"] else math.nan,
               "intensity": None, "intensity_error": None, "area": None, "area_error": None,
               "region_Hz": None}
        if peak is not None:
            row["intensity"] = m["intensity"] * scale
            row["intensity_error"] = m["noise"] * scale
        if region is not None:
            lo, hi = region[0] + shift, region[1] + shift
            sel = (m["off"] >= lo) & (m["off"] <= hi)
            if np.count_nonzero(sel) < 2:
                raise ValueError(f"The region {lo:g} to {hi:g} Hz holds fewer than 2 points in "
                                 f"{labels[k]}.")
            idx = np.flatnonzero(sel)
            area, err = _area_between(m["x_hz"], m["y"], m["x_hz"][idx].min(),
                                      m["x_hz"][idx].max(), m["noise"], baseline)
            row["area"], row["area_error"] = area * scale, err * scale
            row["region_Hz"] = (lo, hi)
        rows.append(row)

    ref_field = rows[reference]["field_T"]
    for row in rows:
        row["delta_B_mT"] = (row["field_T"] - ref_field) * 1e3

    lost = [r["label"] for r in rows if not r["found"]] if track else []
    if lost:
        notes.append(f"No line found (S/N < {min_snr:g} within ±{window:g} Hz, or at the window "
                     f"edge) in {len(lost)} spectrum/spectra: {', '.join(lost)}. Their values are "
                     "read at the pick position and their fields come from SFO1.")
    errs = [abs(r["sfo1_error_mT"]) for r in rows if r["found"]]
    if errs and max(errs) > 0:
        notes.append(f"SFO1 did not follow the line exactly: a field taken from SFO1 would be off "
                     f"by up to {max(errs):.3g} mT ({'used' if field_from == 'sfo1' else 'not used'}).")

    return {"rows": rows, "nucleus": nucleus, "gamma_Hz_per_T": gamma, "shift_ppm": shift_ppm,
            "shift_source": shift_source, "reference": reference, "reference_label": labels[reference],
            "notes": notes,
            "settings": {"peak_Hz": peak, "region_Hz": region, "window_Hz": window, "track": track,
                         "min_snr": min_snr, "field_from": field_from,
                         "baseline": baseline, "correct_rg": correct_rg}}


_TABLE_COLUMNS = (
    ("label", "Spectrum"), ("field_T", "B0 (T)"), ("delta_B_mT", "ΔB0 (mT)"),
    ("field_source", "Field from"), ("carrier_MHz", "SFO1 (MHz)"),
    ("peak_offset_Hz", "Line offset (Hz)"), ("found", "Line found"), ("snr", "S/N"),
    ("field_peak_T", "B0 from line (T)"), ("field_sfo1_T", "B0 from SFO1 (T)"),
    ("sfo1_error_mT", "SFO1 − line (mT)"), ("intensity", "Intensity"),
    ("intensity_error", "Intensity error"), ("area", "Area"), ("area_error", "Area error"),
    ("rg", "RG"), ("ns", "NS"))


def field_sweep_table(result: dict):
    """The sweep as a pandas DataFrame, one row per spectrum, in the
    order given."""
    import pandas as pd
    return pd.DataFrame([{title: row.get(key) for key, title in _TABLE_COLUMNS}
                         for row in result["rows"]])


def field_sweep_to_csv(result: dict, path) -> None:
    """Write :func:`field_sweep_table` plus the settings (as ``#``
    comment lines) to ``path``."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        for line in field_sweep_report(result).splitlines():
            handle.write(f"# {line}\n")
        writer = csv.writer(handle)
        writer.writerow([title for _key, title in _TABLE_COLUMNS])
        for row in result["rows"]:
            writer.writerow(["" if row.get(key) is None else row.get(key)
                             for key, _title in _TABLE_COLUMNS])


def field_sweep_report(result: dict) -> str:
    """Readable summary of a :func:`field_sweep` result: settings, how the
    field was obtained, and anything the user should know."""
    s = result["settings"]
    rows = result["rows"]
    fields = np.array([r["field_T"] for r in rows])
    lines = [
        f"DNP field sweep: {len(rows)} spectra of {result['nucleus']} "
        f"(γ/2π = {result['gamma_Hz_per_T'] / 1e6:.6f} MHz/T)",
        f"  B0 from {s['field_from']} (line's absolute frequency / carrier), range "
        f"{fields.min():.6f} to {fields.max():.6f} T "
        f"({(fields.max() - fields.min()) * 1e3:.3f} mT)",
        f"  Line shift δ = {result['shift_ppm']:.3f} ppm ({result['shift_source']}); "
        f"ΔB0 relative to {result['reference_label']}",
    ]
    if s["peak_Hz"] is not None:
        lines.append(f"  Peak picked at {s['peak_Hz']:.1f} Hz from the carrier")
    if s["region_Hz"] is not None:
        lines.append(f"  Region {s['region_Hz'][0]:.1f} to {s['region_Hz'][1]:.1f} Hz from the "
                     "carrier" + (", moved with the line" if s["track"] else ""))
    lines.append(f"  Tracking: {'on' if s['track'] else 'off'} (window ±{s['window_Hz']:g} Hz, "
                 f"min S/N {s['min_snr']:g}); baseline: {_BASELINE_TEXT[s['baseline']]}; RG corrected: "
                 f"{'yes' if s['correct_rg'] else 'no'}; intensities per scan")
    for note in result["notes"]:
        lines.append(f"  Note: {note}")
    return "\n".join(lines)


def print_field_sweep_report(result: dict) -> None:
    print(field_sweep_report(result))
    print(field_sweep_table(result).to_string(index=False))


def profile_values(result: dict, quantity: str = "intensity", normalise=None,
                   field_unit: str = "mT", relative: bool = False):
    """``(field, value, error, labels, found)`` sorted by field, ready to
    plot. ``quantity`` 'intensity' or 'area'. ``normalise``: None, 'max'
    (divide by the largest |value|) or an int (divide by that spectrum's
    value -- e.g. a microwaves-off spectrum, giving an enhancement
    profile). ``relative``: field as ΔB0 from the reference spectrum."""
    rows = result["rows"]
    key = {"intensity": "intensity", "area": "area"}[quantity]
    if rows[0][key] is None:
        raise ValueError(f"This sweep has no {quantity} (it was measured by "
                         f"{'area' if quantity == 'intensity' else 'peak height'} only).")
    value = np.array([r[key] for r in rows], dtype=float)
    error = np.array([r[f"{key}_error"] for r in rows], dtype=float)
    if normalise == "max":
        d = np.max(np.abs(value))
        value, error = value / d, error / d
    elif isinstance(normalise, (int, np.integer)) and not isinstance(normalise, bool):
        d, ed = value[normalise], error[normalise]
        ratio = value / d
        error = np.abs(ratio) * np.sqrt((error / np.where(value == 0, np.inf, value)) ** 2
                                        + (ed / d) ** 2)
        value = ratio
    elif normalise is not None:
        raise ValueError("normalise must be None, 'max' or a spectrum index.")
    if relative:
        field = np.array([r["delta_B_mT"] for r in rows]) / (1e3 if field_unit == "T" else 1.0)
    else:
        field = np.array([r["field_T"] for r in rows]) * (1e3 if field_unit == "mT" else 1.0)
    order = np.argsort(field)
    labels = [rows[i]["label"] for i in order]
    found = np.array([rows[i]["found"] for i in order])
    return field[order], value[order], error[order], labels, found


def _nucleus_label(nucleus) -> str:
    """"23Na" -> "$^{23}$Na" (mathtext), for axis labels."""
    import re
    m = re.fullmatch(r"(\d+)([A-Za-z]+)", str(nucleus or ""))
    return f"$^{{{m.group(1)}}}${m.group(2)}" if m else str(nucleus or "")


def _field_label(field_unit: str, relative: bool) -> str:
    return ("ΔB$_0$" if relative else "B$_0$") + f" ({field_unit})"


def _value_label(result, quantity, normalise) -> str:
    nuc = _nucleus_label(result.get("nucleus"))
    base = "Intensity" if quantity == "intensity" else "Area"
    if normalise == "max":
        return f"{nuc} {base.lower()} (normalised)".strip()
    if normalise is not None:
        return f"{nuc} {base.lower()} / {result['rows'][normalise]['label']}".strip()
    return f"{nuc} {base.lower()} (a.u.)".strip()


def plot_field_profile(result: dict, quantity: str = "intensity", normalise=None,
                       field_unit: str = "mT", relative: bool = False, figure=None,
                       color: str = "black", fontsize: float = 12.0, zero_line: bool = True,
                       show_top_spine: bool = False, show_right_spine: bool = False):
    """The DNP field profile: signal against B0, 1σ error bars, points
    where no line was found drawn open (their field is from SFO1).
    ``quantity`` 'intensity', 'area' or 'both' (two panels sharing the
    field axis). ``show_top_spine``/``show_right_spine`` close the frame
    (off by default, as in the other notebook curve plots). Returns the
    figure."""
    import matplotlib.pyplot as plt
    from ..plotting.style import apply_spine_style, apply_tick_style

    quantities = ["intensity", "area"] if quantity == "both" else [quantity]
    if figure is None:
        figure = plt.figure(figsize=(3.375 * 1.4, 3.0 * len(quantities)))
    else:
        figure.clear()
    axes = figure.subplots(len(quantities), 1, sharex=True, squeeze=False)[:, 0]
    for ax, q in zip(axes, quantities):
        field, value, error, _labels, found = profile_values(result, q, normalise, field_unit,
                                                             relative)
        # A power of ten in the axis label, not matplotlib's floating
        # "1e8" offset text above the axis.
        exponent = 0
        biggest = np.nanmax(np.abs(value)) if value.size else 0.0
        if normalise is None and biggest > 0:
            exponent = int(math.floor(math.log10(biggest)))
            exponent = exponent if abs(exponent) >= 4 else 0
        if exponent:
            value, error = value / 10.0 ** exponent, error / 10.0 ** exponent
        ax.plot(field, value, "-", color=color, linewidth=1.0, zorder=1)
        ax.errorbar(field[found], value[found], yerr=error[found], fmt="o", color=color,
                    markersize=5, capsize=2.5, linewidth=1.0, zorder=2)
        if np.any(~found):
            ax.errorbar(field[~found], value[~found], yerr=error[~found], fmt="o", color=color,
                        markerfacecolor="white", markersize=5, capsize=2.5, linewidth=1.0,
                        zorder=2, label="open: no line found (field from SFO1)")
            ax.legend(frameon=False, fontsize=fontsize * 0.75)
        if zero_line and np.nanmin(value) < 0 < np.nanmax(value):
            ax.axhline(0, color="0.5", linewidth=0.8, zorder=0)
        label = _value_label(result, q, normalise)
        if exponent:
            label = label.replace("(a.u.)", f"(10$^{{{exponent}}}$ a.u.)")
        ax.set_ylabel(label, fontsize=fontsize)
        apply_spine_style(ax, top=show_top_spine, right=show_right_spine, bottom=True, left=True)
        apply_tick_style(ax, labelsize=fontsize * 0.9)
        ax.tick_params(which="both", top=show_top_spine, right=show_right_spine)
    axes[-1].set_xlabel(_field_label(field_unit, relative), fontsize=fontsize)
    figure.tight_layout()
    return figure


def plot_sweep_spectra(spectra, result: dict | None = None, aligned: bool = False, labels=None,
                       xlim_khz=None, figure=None, cmap: str = "viridis",
                       fontsize: float = 12.0):
    """Every spectrum of a sweep against its offset from its own carrier
    (kHz) -- the axis to pick the line or drag a region on. With a
    ``result``, the tracked line positions (and regions) are marked;
    ``aligned=True`` also shifts each spectrum so the tracked lines
    coincide (display only). Colours run along the field, from palette
    ``cmap`` -- any name the Compare stack accepts (a matplotlib map,
    'black', 'diverging', or a qualitative palette such as 'okabe_ito'),
    through :func:`opennmr.plotting.style.stack_colors`, so the
    too-light end of a map is never used."""
    import matplotlib.pyplot as plt
    from ..plotting.style import apply_spine_style, apply_tick_style, stack_colors

    spectra = list(spectra)
    if figure is None:
        figure = plt.figure(figsize=(3.375 * 1.6, 3.375))
    else:
        figure.clear()
    ax = figure.add_subplot(111)
    if result is not None:
        fields = np.array([r["field_T"] for r in result["rows"]])
    else:
        fields = np.array([field_from_sfo1(s) for s in spectra])
    palette = stack_colors(len(spectra), cmap)
    rank = np.empty(len(spectra), dtype=int)
    rank[np.argsort(fields, kind="stable")] = np.arange(len(spectra))
    for k, s in enumerate(spectra):
        off = carrier_offset_hz(s) / 1e3
        shift = 0.0
        if aligned and result is not None:
            shift = result["rows"][k]["shift_Hz"] / 1e3
        colour = palette[rank[k]]
        ax.plot(off - shift, s.data, color=colour, linewidth=0.8,
                label=(labels[k] if labels else None))
        if result is not None and not aligned:
            row = result["rows"][k]
            if row["found"]:
                ax.plot(row["peak_offset_Hz"] / 1e3, np.interp(
                    row["peak_offset_Hz"] / 1e3, np.sort(off), s.data[np.argsort(off)]),
                    "v", color=colour, markersize=4)
    if result is not None:
        st = result["settings"]
        if st["region_Hz"] is not None and aligned:
            ax.axvspan(st["region_Hz"][0] / 1e3, st["region_Hz"][1] / 1e3, color="0.85", zorder=0)
        if st["peak_Hz"] is not None:
            ax.axvline(st["peak_Hz"] / 1e3, color="0.4", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel("Offset from carrier (kHz)" + (" , aligned on the line" if aligned else ""),
                  fontsize=fontsize)
    ax.invert_xaxis()
    if xlim_khz is not None:
        ax.set_xlim(max(xlim_khz), min(xlim_khz))
        _fit_y_to_visible(ax)
    _spectrum_axes(ax, fontsize)
    figure.tight_layout()
    return figure


def _spectrum_axes(ax, fontsize: float) -> None:
    """Style ``ax`` as every 1D spectrum in openNMR is drawn: the
    frequency axis along the bottom and nothing else -- no top, left or
    right border, no intensity ticks or label (explicit user request,
    2026-09-28: DNP spectra "are normal spectra -- format them the same
    way as any 1-d plot"). The tick settings persist through a later
    ``minorticks_on`` (the GUI's style pass), so no y ticks come back."""
    from ..plotting.style import apply_spine_style, apply_tick_style

    apply_spine_style(ax, top=False, right=False, bottom=True, left=False)
    apply_tick_style(ax, axis="x", labelsize=fontsize * 0.9)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.tick_params(axis="y", which="both", left=False, right=False,
                   labelleft=False, labelright=False)
    ax.tick_params(axis="x", which="both", top=False, labeltop=False)


def _fit_y_to_visible(ax, margin: float = 0.06) -> None:
    """After an x-limit is set, fit the y-range to the lines inside it
    (matplotlib keeps the whole spectrum's range, which a rolling
    baseline can make several times the line's height)."""
    lo, hi = sorted(ax.get_xlim())
    ys = []
    for line in ax.get_lines():
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.shape != y.shape or x.size < 2:
            continue
        inside = (x >= lo) & (x <= hi) & np.isfinite(y)
        ys.extend(y[inside])
    if ys:
        a, b = float(np.min(ys)), float(np.max(ys))
        pad = margin * ((b - a) or abs(b) or 1.0)
        ax.set_ylim(a - pad, b + pad)


def plot_enhancement_spectra(spectrum_1: NMRSpectrum, spectrum_2: NMRSpectrum,
                             result: dict | None = None, unit: str = "ppm",
                             labels=("spectrum 1", "spectrum 2"), xlim=None, figure=None,
                             colors=("#c0392b", "black"), fontsize: float = 12.0):
    """Both spectra of an enhancement measurement on one axis (``unit``
    'ppm', 'hz' or 'offset' -- kHz from the carrier for the last), the
    picked peak maxima and the integrated region(s) marked when a
    :func:`enhancement` ``result`` is given. Spectrum 1 is RG-scaled as
    in the result, so the picture shows the ratio that was computed."""
    import matplotlib.pyplot as plt
    from ..plotting.style import apply_spine_style, apply_tick_style

    if figure is None:
        figure = plt.figure(figsize=(3.375 * 1.6, 3.375))
    else:
        figure.clear()
    ax = figure.add_subplot(111)
    scale_1 = result["rg"]["scale_1"] if result else 1.0
    to_plot = 1e-3 if unit == "offset" else 1.0
    for s, lab, colour, scale in ((spectrum_2, labels[1], colors[1], 1.0),
                                  (spectrum_1, labels[0], colors[0], scale_1)):
        ax.plot(_axis(s, unit) * to_plot, np.asarray(s.data) * scale, color=colour,
                linewidth=0.9, label=lab)
    if result:
        baseline = result["settings"].get("baseline", "local")
        pairs = ((spectrum_1, "1", colors[0], scale_1), (spectrum_2, "2", colors[1], 1.0))
        i = result.get("intensity")
        if i:
            # Marker on the drawn peak top; with a local baseline, the
            # baseline it was measured from is drawn dashed beneath it.
            for s, key, colour, scale in pairs:
                x_unit, x_hz = _axis(s, unit), _axis(s, "hz")
                y = np.asarray(s.data, dtype=float)
                j = int(np.argmin(np.abs(x_unit - i[f"position_{key}"])))
                ax.plot(x_unit[j] * to_plot, y[j] * scale, "v", color=colour, markersize=6)
                if baseline == "local":
                    lo, hi = _line_window(x_hz, y, j)
                    sel = (x_hz >= lo) & (x_hz <= hi)
                    ax.plot(x_unit[sel] * to_plot, _local_line(x_hz, y, lo, hi)(x_hz[sel]) * scale,
                            "--", color=colour, linewidth=0.8)
        a = result.get("area")
        if a:
            ax.axvspan(a["region_1"][0] * to_plot, a["region_1"][1] * to_plot, color="0.88",
                       zorder=0)
            if tuple(a["region_2"]) != tuple(a["region_1"]):
                ax.axvspan(a["region_2"][0] * to_plot, a["region_2"][1] * to_plot, color="0.8",
                           zorder=0, hatch="//", fill=False)
            if baseline == "local":
                for s, key, colour, scale in pairs:
                    x_unit, x_hz = _axis(s, unit), _axis(s, "hz")
                    y = np.asarray(s.data, dtype=float)
                    lo_u, hi_u = a[f"region_{key}"]
                    sel = (x_unit >= lo_u) & (x_unit <= hi_u)
                    if np.count_nonzero(sel) >= 2:
                        f = _local_line(x_hz, y, x_hz[sel].min(), x_hz[sel].max())
                        ax.plot(x_unit[sel] * to_plot, f(x_hz[sel]) * scale, ":", color=colour,
                                linewidth=1.0)
    handles, names = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], names[::-1], frameon=False, fontsize=fontsize * 0.8)
    ax.set_xlabel({"ppm": "δ (ppm)", "hz": "Frequency (Hz)",
                   "offset": "Offset from carrier (kHz)"}[unit], fontsize=fontsize)
    ax.invert_xaxis()
    if xlim is not None:
        ax.set_xlim(max(xlim), min(xlim))
        _fit_y_to_visible(ax)
    _spectrum_axes(ax, fontsize)
    figure.tight_layout()
    return figure
