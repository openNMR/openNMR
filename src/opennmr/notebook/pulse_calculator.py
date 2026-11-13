"""
Pulse calculator: pulse lengths, RF powers, nutation frequencies and
excitation bandwidths of rectangular (hard) pulses.

Explicit user request (2026-11-13): "a function that allows someone to
calculate and convert pulses and their bandwidths and nutation
frequencies -- this should accept pulse lengths and powers (the powers
can be in W, or in 'scaled' units, where the scaling multiplicative
factor is something you can enter manually or leave blank; the
underlying unit for the scaling will be Watts, or in dB)". The same
functions drive Simulate ▸ Pulse calculator in the app.

The physics
-----------
* A rectangular pulse of length ``τ`` at nutation frequency ``ν₁`` turns
  the magnetisation by ``θ = 360° · ν₁ · τ``; so ``ν₁ = 1 / (4 τ₉₀)``.
* The RF amplitude goes as the square root of the power, so
  ``ν₁(P) = ν₁(P_ref) · √(P / P_ref)`` and ``τ_θ(P) = τ_θ(P_ref) · √(P_ref / P)``.
  Converting between powers therefore needs one calibrated pulse (its
  length, flip angle and power) -- a *reference*.
* Powers can be given in W, in dB, or in "scaled" units.

  - **dB** is TopSpin's: ``P / W = 10^(−dB / 10)`` -- a *higher* dB is
    *less* power, and 0 dB is 1 W (``db_ref_w`` changes what 0 dB is).
  - **Scaled** units are watts times a factor, ``P / W = factor × value``
    (e.g. an amplifier setting in %, or a console's own scale). With the
    factor left blank (None) the scaled numbers are only *relative* to
    each other: they convert among themselves exactly (only the ratio
    matters), but cannot be converted to W or dB.
* The excitation profile of a rectangular pulse is computed exactly (a
  rotation about the effective field, no small-angle approximation).
  The **bandwidth** reported is the full width at half maximum of that
  profile: of the transverse magnetisation ``|M_xy|`` for a flip angle up
  to 135°, and of the inversion ``(1 − M_z) / 2`` above it (where ``|M_xy|``
  has a hole on resonance). ``1 / τ`` -- the first zero of the low-flip
  (sinc) profile, the usual rule of thumb -- is given alongside.
* For a half-integer quadrupolar nucleus with ``ν_Q ≫ ν₁``, the central
  transition nutates ``(I + ½)`` times faster than a liquid: its selective
  90° is ``τ₉₀ / (I + ½)`` (give ``spin=``).

Example::

    from opennmr.notebook import pulse_calculator as pc
    out = pc.pulse_calculator(ref_pulse_us=2.5, ref_power=100, ref_unit="W",
                              power=25, unit="W")
    out["nu1_hz"]        # 50 000 Hz: a quarter of the power, half the field
    out["pulse_us"]      # {90: 5.0, 180: 10.0}
    print(out["report"])
"""

from __future__ import annotations

import math

import numpy as np

#: Units a power can be given in.
POWER_UNITS = ("W", "dB", "scaled")


# ---------------------------------------------------------------------------
# Powers
# ---------------------------------------------------------------------------

def _check_unit(unit: str) -> str:
    unit = {"w": "W", "db": "dB", "scaled": "scaled"}.get(str(unit).strip().lower())
    if unit is None:
        raise ValueError(f"A power unit must be one of {POWER_UNITS}.")
    return unit


def power_to_watts(value: float, unit: str = "W", *, scale: float | None = None,
                   db_ref_w: float = 1.0) -> float | None:
    """``value`` (in ``unit``: "W", "dB" or "scaled") in watts.

    dB: ``db_ref_w · 10^(−value / 10)`` (TopSpin: 0 dB = 1 W, higher dB =
    less power). Scaled: ``scale · value``; ``None`` when ``scale`` is
    None, as a scaled value without its factor is only relative."""
    unit = _check_unit(unit)
    value = float(value)
    if unit == "W":
        if value <= 0:
            raise ValueError("A power in W must be positive.")
        return value
    if unit == "dB":
        if not db_ref_w or db_ref_w <= 0:
            raise ValueError("What 0 dB is, in W, must be positive.")
        return float(db_ref_w) * 10.0 ** (-value / 10.0)
    if value <= 0:
        raise ValueError("A scaled power must be positive.")
    if scale is None:
        return None
    if scale <= 0:
        raise ValueError("The scaling factor (W per scaled unit) must be positive.")
    return float(scale) * value


def watts_to_power(watts: float, unit: str = "W", *, scale: float | None = None,
                   db_ref_w: float = 1.0) -> float | None:
    """The inverse of :func:`power_to_watts` (None: scaled with no factor)."""
    unit = _check_unit(unit)
    watts = float(watts)
    if watts <= 0:
        raise ValueError("A power must be positive.")
    if unit == "W":
        return watts
    if unit == "dB":
        return -10.0 * math.log10(watts / float(db_ref_w))
    return None if scale is None else watts / float(scale)


def power_ratio(value: float, unit: str, ref_value: float, ref_unit: str, *,
                scale: float | None = None, db_ref_w: float = 1.0) -> float:
    """``P / P_ref`` for two powers, each in its own unit.

    Two scaled powers compare without their factor (it cancels), as do two
    in dB; a scaled power against one in W or dB needs the factor."""
    unit, ref_unit = _check_unit(unit), _check_unit(ref_unit)
    if unit == ref_unit == "scaled":
        if float(value) <= 0 or float(ref_value) <= 0:
            raise ValueError("A scaled power must be positive.")
        return float(value) / float(ref_value)
    if unit == ref_unit == "dB":
        return 10.0 ** (-(float(value) - float(ref_value)) / 10.0)
    watts = power_to_watts(value, unit, scale=scale, db_ref_w=db_ref_w)
    ref_watts = power_to_watts(ref_value, ref_unit, scale=scale, db_ref_w=db_ref_w)
    if watts is None or ref_watts is None:
        raise ValueError("To compare a scaled power with one in W or dB, enter the scaling "
                         "factor (W per scaled unit).")
    return watts / ref_watts


# ---------------------------------------------------------------------------
# Nutation
# ---------------------------------------------------------------------------

def nutation_frequency(pulse_us: float, flip_deg: float = 90.0) -> float:
    """``ν₁`` in Hz of a pulse of ``pulse_us`` µs that turns by ``flip_deg``."""
    pulse_us, flip_deg = float(pulse_us), float(flip_deg)
    if pulse_us <= 0 or flip_deg <= 0:
        raise ValueError("A pulse length and flip angle must be positive.")
    return flip_deg / 360.0 / (pulse_us * 1e-6)


def pulse_length(nu1_hz: float, flip_deg: float = 90.0) -> float:
    """The length in µs of a ``flip_deg`` pulse at ``ν₁ = nu1_hz``."""
    nu1_hz, flip_deg = float(nu1_hz), float(flip_deg)
    if nu1_hz <= 0 or flip_deg <= 0:
        raise ValueError("A nutation frequency and flip angle must be positive.")
    return flip_deg / 360.0 / nu1_hz * 1e6


def flip_angle(nu1_hz: float, pulse_us: float) -> float:
    """The flip angle in degrees of a ``pulse_us`` µs pulse at ``nu1_hz``."""
    return 360.0 * float(nu1_hz) * float(pulse_us) * 1e-6


def scale_nutation(nu1_ref_hz: float, ratio: float) -> float:
    """``ν₁`` at ``ratio`` times the reference power: ``ν₁_ref · √ratio``."""
    if ratio <= 0:
        raise ValueError("A power ratio must be positive.")
    return float(nu1_ref_hz) * math.sqrt(float(ratio))


def power_ratio_for_nutation(nu1_ref_hz: float, nu1_hz: float) -> float:
    """``P / P_ref`` that gives ``nu1_hz``: ``(ν₁ / ν₁_ref)²``."""
    if nu1_ref_hz <= 0 or nu1_hz <= 0:
        raise ValueError("Nutation frequencies must be positive.")
    return (float(nu1_hz) / float(nu1_ref_hz)) ** 2


def b1_field(nu1_hz: float, nucleus: str) -> float:
    """The RF field ``B₁`` in tesla behind ``ν₁`` for ``nucleus`` ('1H',
    '27Al', ...): ``ν₁ / |γ / 2π|``."""
    from ..core.registry import find_gamma
    found = find_gamma(str(nucleus), quiet=True)
    if not found:
        raise ValueError(f"Unknown nucleus {nucleus!r} (use e.g. '1H' or '27Al').")
    return float(nu1_hz) / abs(float(found["gamma"]))


# ---------------------------------------------------------------------------
# Excitation profile and bandwidth
# ---------------------------------------------------------------------------

def excitation_profile(offsets_hz, nu1_hz: float, pulse_us: float) -> dict:
    """The magnetisation after a rectangular pulse along x, from +z, at
    each resonance offset in ``offsets_hz`` -- exact: a rotation by
    ``2π ν_eff τ`` about the effective field ``(ν₁, 0, Δ) / ν_eff``.
    Returns ``{"offset_hz", "mx", "my", "mz", "mxy"}``."""
    offsets = np.asarray(offsets_hz, dtype=float)
    nu1 = float(nu1_hz)
    tau = float(pulse_us) * 1e-6
    nu_eff = np.sqrt(nu1 ** 2 + offsets ** 2)
    nx = np.where(nu_eff > 0, nu1 / np.where(nu_eff > 0, nu_eff, 1.0), 0.0)
    nz = np.where(nu_eff > 0, offsets / np.where(nu_eff > 0, nu_eff, 1.0), 1.0)
    beta = 2.0 * np.pi * nu_eff * tau
    c, s = np.cos(beta), np.sin(beta)
    mx = nx * nz * (1.0 - c)
    my = -nx * s
    mz = c + nz ** 2 * (1.0 - c)
    return {"offset_hz": offsets, "mx": mx, "my": my, "mz": mz, "mxy": np.hypot(mx, my)}


def _fwhm_about_zero(offsets, values) -> float:
    """Full width at half maximum of the lobe containing the maximum
    nearest zero offset, from a symmetric grid; NaN if it does not drop
    to half within the grid."""
    peak = float(np.max(values))
    if peak <= 0:
        return float("nan")
    half = 0.5 * peak
    centre = int(np.argmin(np.abs(offsets)))
    # Walk out from the centre to the highest point of the central lobe.
    above = values >= half
    if not above[centre]:
        return float("nan")
    hi = centre
    while hi + 1 < values.size and above[hi + 1]:
        hi += 1
    lo = centre
    while lo - 1 >= 0 and above[lo - 1]:
        lo -= 1
    if hi + 1 >= values.size or lo - 1 < 0:
        return float("nan")

    def crossing(i, j):            # linear between the last point above and the first below
        vi, vj = values[i], values[j]
        return offsets[i] + (half - vi) * (offsets[j] - offsets[i]) / (vj - vi)

    return float(crossing(hi, hi + 1) - crossing(lo, lo - 1))


def pulse_bandwidth(nu1_hz: float, pulse_us: float, n_points: int = 8001) -> dict:
    """The bandwidth of a rectangular pulse of ``pulse_us`` µs at ``nu1_hz``.

    ``fwhm_hz``: full width at half maximum of the exact excitation
    profile -- ``|M_xy|`` for a flip of 135° or less (``"excitation"``),
    the inversion ``(1 − M_z) / 2`` above it (``"inversion"``).
    ``first_null_hz``: ``1 / τ``, the low-flip rule of thumb.
    ``flip_deg``: the pulse's flip angle on resonance."""
    flip = flip_angle(nu1_hz, pulse_us)
    tau = float(pulse_us) * 1e-6
    span = 6.0 / tau + 4.0 * float(nu1_hz)
    offsets = np.linspace(-span, span, int(n_points))
    profile = excitation_profile(offsets, nu1_hz, pulse_us)
    if flip <= 135.0:
        kind, values = "excitation", profile["mxy"]
    else:
        kind, values = "inversion", (1.0 - profile["mz"]) / 2.0
    return {"fwhm_hz": _fwhm_about_zero(offsets, values), "kind": kind,
            "first_null_hz": 1.0 / tau, "flip_deg": flip}


# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------

def _fmt(value, digits=4) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}g}"


def pulse_calculator(ref_pulse_us: float, ref_power: float | None = None, ref_unit: str = "W",
                     ref_flip_deg: float = 90.0, *, power: float | None = None,
                     unit: str | None = None, nu1_hz: float | None = None,
                     target_pulse_us: float | None = None, target_flip_deg: float = 90.0,
                     flips_deg=(90.0, 180.0), scale: float | None = None,
                     db_ref_w: float = 1.0, nucleus: str | None = None,
                     spin: float | None = None) -> dict:
    """Convert a calibrated pulse to another power, nutation frequency or
    pulse length.

    The reference: a ``ref_flip_deg`` pulse of ``ref_pulse_us`` µs at
    ``ref_power`` (in ``ref_unit``: "W", "dB" or "scaled"; may be None
    when no other power is asked about).

    The target -- one of:

    * ``power`` (in ``unit``, default ``ref_unit``): the ν₁ and pulse
      lengths there;
    * ``nu1_hz``: the power that gives it;
    * ``target_pulse_us``: the power at which a ``target_flip_deg`` pulse
      is that long;
    * none of them: the reference itself.

    ``scale``: W per scaled unit (None: scaled powers are relative only).
    ``db_ref_w``: what 0 dB is in W (TopSpin: 1). ``nucleus``: adds B₁.
    ``spin``: a half-integer quadrupolar nucleus's I, adding the central
    transition's selective pulse lengths, ``(I + ½)`` times shorter.

    Returns a dict: ``nu1_hz``, ``pulse_us`` ({flip: µs} for ``flips_deg``
    and the target flip), ``power`` ({"W", "dB", "scaled"} -- None where it
    cannot be expressed), ``power_ratio`` (P / P_ref), ``bandwidth`` (of the
    90° pulse, :func:`pulse_bandwidth`), ``b1_mT``, ``ct`` (for ``spin``),
    ``reference`` (the same for the reference) and ``report`` -- the
    working, one line per step."""
    targets = [t is not None for t in (power, nu1_hz, target_pulse_us)]
    if sum(targets) > 1:
        raise ValueError("Give at most one of power, nu1_hz and target_pulse_us.")
    ref_unit = _check_unit(ref_unit)
    unit = _check_unit(unit) if unit is not None else ref_unit
    nu1_ref = nutation_frequency(ref_pulse_us, ref_flip_deg)
    lines = [f"Reference: a {_fmt(ref_flip_deg)}° pulse of {_fmt(ref_pulse_us)} µs → "
             f"ν₁ = {_fmt(ref_flip_deg)}° / (360° × {_fmt(ref_pulse_us)} µs) = "
             f"{_fmt(nu1_ref / 1e3, 5)} kHz"]
    ref_w = None
    if ref_power is not None:
        ref_w = power_to_watts(ref_power, ref_unit, scale=scale, db_ref_w=db_ref_w)
        lines.append(f"  at {_fmt(ref_power)} {ref_unit}"
                     + (f" = {_fmt(ref_w)} W" if ref_w is not None and ref_unit != "W" else ""))

    if power is not None:
        if ref_power is None:
            raise ValueError("Give the reference pulse's power to convert to another power.")
        ratio = power_ratio(power, unit, ref_power, ref_unit, scale=scale, db_ref_w=db_ref_w)
        nu1 = scale_nutation(nu1_ref, ratio)
        lines.append(f"Target power {_fmt(power)} {unit}: P / P_ref = {_fmt(ratio, 5)}; "
                     f"ν₁ = ν₁_ref × √(P / P_ref) = {_fmt(nu1 / 1e3, 5)} kHz")
    elif nu1_hz is not None:
        nu1 = float(nu1_hz)
        ratio = power_ratio_for_nutation(nu1_ref, nu1)
        lines.append(f"Target ν₁ = {_fmt(nu1 / 1e3, 5)} kHz: P / P_ref = (ν₁ / ν₁_ref)² = "
                     f"{_fmt(ratio, 5)}")
    elif target_pulse_us is not None:
        nu1 = nutation_frequency(target_pulse_us, target_flip_deg)
        ratio = power_ratio_for_nutation(nu1_ref, nu1)
        lines.append(f"Target: a {_fmt(target_flip_deg)}° pulse of {_fmt(target_pulse_us)} µs → "
                     f"ν₁ = {_fmt(nu1 / 1e3, 5)} kHz; P / P_ref = (ν₁ / ν₁_ref)² = "
                     f"{_fmt(ratio, 5)}")
    else:
        nu1, ratio = nu1_ref, 1.0

    # The power, in every unit it can be expressed in.
    powers = {"W": None, "dB": None, "scaled": None}
    if ref_power is not None:
        target_w = ref_w * ratio if ref_w is not None else None
        if target_w is not None:
            powers["W"] = target_w
            powers["dB"] = watts_to_power(target_w, "dB", db_ref_w=db_ref_w)
            powers["scaled"] = watts_to_power(target_w, "scaled", scale=scale)
        if ref_unit == "scaled":
            powers["scaled"] = float(ref_power) * ratio
        if ref_unit == "dB":
            powers["dB"] = float(ref_power) - 10.0 * math.log10(ratio)
        shown = [f"{_fmt(v, 5)} {k if k != 'scaled' else 'scaled units'}"
                 for k, v in powers.items() if v is not None]
        if shown:
            lines.append("Power: " + " = ".join(shown))

    flips = [float(f) for f in flips_deg]
    if float(target_flip_deg) not in flips:
        flips.append(float(target_flip_deg))
    pulses = {f: pulse_length(nu1, f) for f in flips}
    lines.append("Pulse lengths: " + ", ".join(f"{_fmt(f)}° = {_fmt(t, 5)} µs"
                                               for f, t in pulses.items()))
    bandwidth = pulse_bandwidth(nu1, pulses.get(90.0, pulse_length(nu1, 90.0)))
    lines.append(f"Bandwidth of the 90° pulse: {_fmt(bandwidth['fwhm_hz'] / 1e3, 4)} kHz FWHM "
                 f"of its exact excitation profile; 1/τ₉₀ = "
                 f"{_fmt(bandwidth['first_null_hz'] / 1e3, 4)} kHz (first-null rule of thumb)")
    b1_mt = None
    if nucleus:
        b1_mt = b1_field(nu1, nucleus) * 1e3
        lines.append(f"B₁ for {nucleus}: ν₁ / |γ/2π| = {_fmt(b1_mt, 4)} mT")
    ct = None
    if spin is not None:
        spin = float(spin)
        if spin < 1.5 or abs(spin * 2 - round(spin * 2)) > 1e-9 or round(spin * 2) % 2 != 1:
            raise ValueError("The central-transition factor applies to a half-integer spin "
                             "of 3/2 or more.")
        factor = spin + 0.5
        ct = {"factor": factor, "nu1_hz": factor * nu1,
              "pulse_us": {f: t / factor for f, t in pulses.items()}}
        lines.append(f"Central transition (I = {_fmt(spin)}, ν_Q ≫ ν₁): nutates (I + ½) = "
                     f"{_fmt(factor)} times faster; selective "
                     + ", ".join(f"{_fmt(f)}° = {_fmt(t, 5)} µs" for f, t in ct["pulse_us"].items()))
    return {
        "nu1_hz": nu1, "pulse_us": pulses, "power": powers, "power_ratio": ratio,
        "bandwidth": bandwidth, "b1_mT": b1_mt, "ct": ct,
        "reference": {"nu1_hz": nu1_ref, "pulse_us": float(ref_pulse_us),
                      "flip_deg": float(ref_flip_deg), "power": ref_power, "unit": ref_unit,
                      "watts": ref_w},
        "report": "\n".join(lines),
    }
