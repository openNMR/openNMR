"""
Variable Offset Cumulative Spectroscopy (VOCS) processing.

Ported from ``nmrtools2``'s ``one_d.py`` (the VOCS half). The
finite-pulse-width excitation-profile math (``vocs_f1``/``vocs_f2``/
``vocs_power``/``pulse_to_w1``) is unchanged; ``apply_vocs``/
``vocs_sum`` are adapted to operate on :class:`NMRSpectrum` instead of
a raw dict.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.interpolate import interp1d

from ..core.acquisition import acquisition_constants
from ..core.spectrum import NMRSpectrum


def vocs_f1(x, w1t1: float = np.pi / 2, w1t2: float = np.pi):
    """First finite-pulse-width excitation-profile term used in the
    VOCS correction.

    ``x`` : dimensionless frequency offset (offset from the VOCS
    centre divided by the RF nutation frequency). ``w1t1``/``w1t2`` :
    flip angles of the first/second pulse in radians.
    """
    s = np.sqrt(1.0 + x ** 2)
    return np.sin(w1t1 * s) * (1.0 - np.cos(w1t2 * s)) / (1.0 + x ** 2) ** 1.5


def vocs_f2(x, w1t1: float = np.pi / 2, w1t2: float = np.pi):
    """Second finite-pulse-width excitation-profile term. See
    :func:`vocs_f1` for parameters."""
    s = np.sqrt(1.0 + x ** 2)
    return -x * (1.0 - np.cos(w1t1 * s)) * (1.0 - np.cos(w1t2 * s)) / (1.0 + x ** 2) ** 2.0


def vocs_power(freq_hz, w1: float, window_filter: bool = True,
                w1t1: float = np.pi / 2, w1t2: float = np.pi):
    """VOCS frequency-dependent scaling function:
    ``power = sqrt(f1**2 + f2**2)``, where ``x = (freq_hz -
    freq_hz[centre]) / w1``.

    ``w1`` : RF nutation frequency in Hz. ``window_filter`` : if True
    (default), zero the scaling function outside ``|x| < 1.73``, to
    limit the correction to the useful VOCS excitation bandwidth.
    """
    centre = len(freq_hz) // 2
    x = (freq_hz - freq_hz[centre]) / w1

    f1 = vocs_f1(x, w1t1=w1t1, w1t2=w1t2)
    f2 = vocs_f2(x, w1t1=w1t1, w1t2=w1t2)
    power = np.sqrt(f1 ** 2 + f2 ** 2)

    if window_filter:
        power *= (np.abs(x) < 1.73)

    return power


def pulse_to_w1(pulse_length_us: float) -> float:
    """Convert a 90-degree pulse length (microseconds) to RF nutation
    frequency in Hz: ``w1 = 1e6 / (4 * pulse_length_us)``.

    >>> pulse_to_w1(5)
    50000.0
    """
    return 1e6 / (4.0 * pulse_length_us)


def _pulse_length_us(spectrum: NMRSpectrum, pulse: str) -> float:
    """Look up a pulse length (microseconds) from a spectrum's recorded
    acquisition parameters, by the name a pulse programme uses for it
    (``P1``, ``P2``, ...).

    Goes through :func:`opennmr.core.acquisition.acquisition_constants`
    rather than indexing ``metadata.raw["acqus"]`` directly. **A direct
    lookup of ``"P1"`` never works** -- the vendor file records ``P``
    as a 64-entry *array*, so that key does not exist and this used to
    fail on every real dataset with "P1 not found in this spectrum's
    acquisition parameters" (real user report: VOCS "can't seem to
    find P1"). ``acquisition_constants`` is what expands those arrays
    into ``p1``/``d1``/``cnst2``/``l4``/``gpz1``-style names, and was
    already doing so for fitting equations.

    A pulse that *exists but is zero* is rejected the same way a
    missing one is. A real ``acqus`` always carries all 64 ``P`` slots
    and a sequence typically uses two or three, so most resolve to
    ``0.0`` -- and ``pulse_to_w1(0)`` is a division by zero, which
    would surface as a bare ``ZeroDivisionError`` rather than
    something a user can act on."""
    key = pulse.strip().lower()
    constants = acquisition_constants(spectrum)
    value = constants.get(key)
    if value is None or value <= 0:
        if not constants:
            raise ValueError(
                f"This spectrum has no recorded acquisition parameters, so {pulse} "
                f"cannot be looked up (only Bruker data retains an acqus table; a "
                f"derived spectrum has none)."
            )
        # Only the pulses actually set, in index order -- listing all
        # 64 slots would bury the two or three that are really in use.
        used = sorted(
            int(k[1:]) for k, v in constants.items()
            if k.startswith("p") and k[1:].isdigit() and v > 0
        )
        available = ", ".join(f"P{i} ({constants[f'p{i}']:g} us)" for i in used[:12])
        if len(used) > 12:
            available += ", ..."
        was_zero = "" if value is None else " (it is recorded as 0, i.e. unused)"
        raise ValueError(
            f"{pulse} is not a usable pulse length on this spectrum{was_zero}. "
            f"Pulse lengths set on this dataset: {available or 'none'}."
        )
    return value


def apply_vocs(spectrum: NMRSpectrum, pulse: str = "P1", window_filter: bool = True) -> NMRSpectrum:
    """Apply a VOCS correction to an already-imported 1D
    :class:`NMRSpectrum`, using the pulse length recorded in
    ``spectrum.metadata.raw["acqus"]``.

    Returns a new spectrum with ``real``/``imag`` VOCS-corrected and
    ``metadata.raw["vocs_scaling"]`` recording the applied scaling.
    """
    if spectrum.ndim != 1:
        raise ValueError("apply_vocs expects a 1D NMRSpectrum.")

    pulse_length_us = _pulse_length_us(spectrum, pulse)
    w1 = pulse_to_w1(pulse_length_us)
    freq_hz = spectrum.axis_values(0, unit="hz")
    scaling = vocs_power(freq_hz, w1, window_filter=window_filter)

    data = spectrum.data * scaling
    data_imag = spectrum.data_imag * scaling if spectrum.data_imag is not None else None

    metadata = dataclasses.replace(
        spectrum.metadata, raw={**spectrum.metadata.raw, "vocs_scaling": scaling},
    )
    return NMRSpectrum(data=data, axes=spectrum.axes, metadata=metadata, data_imag=data_imag)


def vocs_sum(spectra: list[NMRSpectrum], pulse: str = "P1", npoints: int = 16384,
             average: bool = False, window_filter: bool = True) -> dict:
    """Apply :func:`apply_vocs` to each of ``spectra``, interpolate
    them onto a common ppm axis (covering the union of their
    individual ranges), and sum or average.

    Returns a dict with ``"slices"`` (the VOCS-corrected inputs),
    ``"ppm"`` (common axis), ``"vocs_sum"`` (summed/averaged
    spectrum), ``"interpolated_slices"`` (each corrected spectrum
    interpolated onto the common axis, shape ``(n_spectra, npoints)``).

    Values outside an individual spectrum's original range are zero
    after interpolation. Spectra are sorted by their ppm axis first,
    since ``scipy.interpolate.interp1d`` requires increasing x.
    """
    corrected = [apply_vocs(s, pulse=pulse, window_filter=window_filter) for s in spectra]

    ppm_low = min(np.min(s.axis_values(0, "ppm")) for s in corrected)
    ppm_high = max(np.max(s.axis_values(0, "ppm")) for s in corrected)
    ppm_common = np.linspace(ppm_low, ppm_high, npoints)

    interpolated = []
    for s in corrected:
        x = np.asarray(s.axis_values(0, "ppm"))
        y = np.asarray(s.data)
        order = np.argsort(x)
        f = interp1d(x[order], y[order], bounds_error=False, fill_value=0.0)
        interpolated.append(f(ppm_common))
    interpolated = np.asarray(interpolated)

    summed = np.mean(interpolated, axis=0) if average else np.sum(interpolated, axis=0)

    return {
        "slices": corrected,
        "ppm": ppm_common,
        "vocs_sum": summed,
        "interpolated_slices": interpolated,
    }
