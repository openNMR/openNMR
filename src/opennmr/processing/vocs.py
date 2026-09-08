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


def excitation_profile(freq_hz, w1: float, window_filter: bool = True,
                       w1t1: float = np.pi / 2, w1t2: float = np.pi):
    """The finite-pulse excitation profile as a genuine *fraction of
    full excitation*: 1.0 on resonance, falling off with offset.

    This is :func:`vocs_power` divided by its own on-resonance value,
    and that division is the whole scale fix. ``vocs_power`` peaks at
    **2.0** for the default flip angles, so weighting a spectrum by it
    directly multiplied every intensity by up to 2 -- measured on real
    17O data, 6971 -> 11908. A weight above 1.0 is not physically
    meaningful either: a pulse cannot excite more than fully.

    Weighting by *this* leaves an on-resonance peak exactly as it was
    and only ever attenuates off resonance, which is what makes the
    scale survive both the per-spectrum step and the combination.

    ``vocs_power`` itself is left exactly as ported from
    ``nmrtools2`` -- see this module's own docstring -- so anything
    relying on its published form still gets it.
    """
    power = vocs_power(freq_hz, w1, window_filter=window_filter, w1t1=w1t1, w1t2=w1t2)
    # f2 is proportional to x, so it vanishes on resonance and the
    # on-resonance value is |f1(0)| alone -- computed rather than
    # hardcoded to 2.0, since it depends on the flip angles.
    on_resonance = abs(vocs_f1(0.0, w1t1=w1t1, w1t2=w1t2))
    if not on_resonance:
        return power
    return power / on_resonance


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


def _profile_for(spectrum: NMRSpectrum, pulse: str, window_filter: bool):
    """This spectrum's own excitation profile, on its own Hz axis."""
    w1 = pulse_to_w1(_pulse_length_us(spectrum, pulse))
    return excitation_profile(spectrum.axis_values(0, unit="hz"), w1, window_filter=window_filter)


def apply_vocs(spectrum: NMRSpectrum, pulse: str = "P1", window_filter: bool = True) -> NMRSpectrum:
    """Weight a 1D :class:`NMRSpectrum` by its own finite-pulse
    excitation profile -- ``S * w``, the original VOCS methodology.

    Each offset acquisition is only trustworthy near its own carrier,
    and multiplying by the profile is what expresses that: full
    confidence on resonance, tailing off with offset. It is a
    *weighting*, not a deconvolution, and it only ever attenuates.

    **The scale fix is that ``w`` is now normalised** (see
    :func:`excitation_profile`): it is 1.0 on resonance instead of
    2.0, so a spectrum keeps its own intensity where the pulse
    excited it fully, rather than being silently doubled. That was the
    real defect -- measured on real 17O data, 6971 -> 11908. Real user
    report: "the intensities are not quite right ... The spectra
    should maintain their original intensities."

    **Deliberately not a divide-by-the-profile deconvolution.** That
    was tried and reverted: dividing recovers true amplitude in
    principle, but the profile has genuine nulls *inside* its own
    bandwidth window (``sqrt(f1**2 + f2**2)`` has both terms vanish
    together, reaching 2e-6 on real data), so it amplified whatever
    sat at the sweep edges until the corrected maximum landed there
    instead of on the real peak -- "signal is large at the edges and
    suddenly drops to zero and has low intensity at the centres".
    Multiplying has no such failure mode: the weight never exceeds
    1.0, so nothing is ever amplified and no threshold or
    regularisation is needed anywhere.

    ``metadata.raw["vocs_profile"]`` records the weight used, which is
    what :func:`vocs_sum` needs to combine spectra without inflating
    their scale.
    """
    if spectrum.ndim != 1:
        raise ValueError("apply_vocs expects a 1D NMRSpectrum.")

    profile = _profile_for(spectrum, pulse, window_filter)

    def _corrected(values):
        return np.asarray(values, dtype=float) * profile

    data = _corrected(spectrum.data)
    data_imag = _corrected(spectrum.data_imag) if spectrum.data_imag is not None else None

    metadata = dataclasses.replace(
        spectrum.metadata, raw={**spectrum.metadata.raw, "vocs_profile": profile},
    )
    return NMRSpectrum(data=data, axes=spectrum.axes, metadata=metadata, data_imag=data_imag)


def vocs_sum(spectra: list[NMRSpectrum], pulse: str = "P1", npoints: int = 16384,
             average: bool = False, window_filter: bool = True) -> dict:
    """Combine offset VOCS acquisitions onto one common ppm axis::

        sum(S_i * w_i) / plateau,   plateau = max of sum(w_i ** 2)

    ``nmrtools2``'s original ``sum(S_i * w_i)``, divided by **one
    scalar** rather than by a per-point denominator.

    **Why a scalar and not a per-point divide.** Feed a genuinely flat
    spectrum in and the bare sum reports ``sum(w_i ** 2)``. Where the
    offsets tile the range that quantity is *already flat* -- measured
    on this project's own 17O series with the coverage gap filled, it
    varies by **1.3% rms (a 1.06x ripple)** -- so the bare sum is the
    true lineshape times a constant, and dividing by that constant
    gives back the amplitude with the lineshape untouched. A per-point
    denominator instead imposes its own structure: the weighted mean
    ``sum(S*w)/sum(w)`` this replaced measured **3.5% rms (1.17x)** on
    the same well-covered data, i.e. it made a flat response *less*
    flat.

    **The plateau is where coverage is best**, so a fully-covered shift
    reads its true amplitude; regions the offsets cover less well read
    proportionally lower, which is honest -- less signal was excited
    there. Nothing is ever scaled *up*.

    **Poor coverage is an acquisition problem, not an arithmetic one.**
    On the real 17O series as acquired, offsets sit at -6010, -2010,
    5900 and 9900 ppm: spacings of 4000, **7910** and 4000 ppm against
    an excitation half-width of ~3687 ppm. That middle gap is wider
    than the pulse can reach, so there is a genuine dead zone, and the
    bare sum's response ripples **8.72x** across it. One more offset
    near 1950 ppm collapses that to 1.06x. No choice of denominator
    can invent the signal that was never excited -- earlier attempts to
    flatten it arithmetically (a matched filter, a per-spectrum
    deconvolution, Wiener regularisation) each traded the ripple for a
    worse artifact. See :func:`apply_vocs`.

    ``average`` is accepted but **ignored**: the scaling is fixed by
    the coverage plateau, so "sum" versus "average" does not name a
    real choice.

    Returns ``"slices"`` (each input weighted, for plotting the
    parts), ``"ppm"`` (common axis), ``"vocs_sum"`` (the composite),
    ``"interpolated_slices"`` (the weighted inputs on the common
    axis), ``"weight"`` (``sum(w_i ** 2)``, the coverage -- its own
    flatness is the diagnostic for whether the offsets tile the range),
    and ``"plateau"`` (the scalar divided out).
    """
    if not spectra:
        raise ValueError("vocs_sum needs at least one spectrum.")

    ppm_low = min(np.min(s.axis_values(0, "ppm")) for s in spectra)
    ppm_high = max(np.max(s.axis_values(0, "ppm")) for s in spectra)
    ppm_common = np.linspace(ppm_low, ppm_high, npoints)

    def _onto_common(x, y):
        """``y`` sampled on the common axis, zero outside its own
        range. Sorted first, since interp1d requires increasing x."""
        order = np.argsort(np.asarray(x))
        f = interp1d(np.asarray(x)[order], np.asarray(y)[order],
                     bounds_error=False, fill_value=0.0)
        return f(ppm_common)

    numerator = np.zeros(npoints)
    coverage = np.zeros(npoints)
    for spectrum in spectra:
        ppm = spectrum.axis_values(0, "ppm")
        weight = _profile_for(spectrum, pulse, window_filter)
        numerator += _onto_common(ppm, np.asarray(spectrum.data, dtype=float) * weight)
        # sum(w**2), which is what the bare sum's own response to a
        # flat spectrum is -- so its peak is the scale factor to remove.
        coverage += _onto_common(ppm, weight ** 2)

    # One scalar for the whole spectrum: no per-point division, so the
    # lineshape the sum produces is carried through untouched, and
    # there is no small number anywhere to divide by. Falls back to 1.0
    # only if nothing was excited at all, which leaves an all-zero sum
    # alone rather than raising.
    plateau = float(coverage.max())
    combined = numerator / plateau if plateau > 0 else numerator

    # The individually weighted inputs, for plotting the parts
    # alongside the composite -- same S*w the numerator is built from,
    # so they sit on the composite's own scale.
    corrected = [
        apply_vocs(s, pulse=pulse, window_filter=window_filter)
        for s in spectra
    ]
    interpolated = np.asarray([
        _onto_common(s.axis_values(0, "ppm"), s.data) for s in corrected
    ])

    return {
        "slices": corrected,
        "ppm": ppm_common,
        "vocs_sum": combined,
        "interpolated_slices": interpolated,
        "weight": coverage,
        "plateau": plateau,
    }
