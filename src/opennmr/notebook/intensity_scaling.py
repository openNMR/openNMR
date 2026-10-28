"""
Intensities that can be compared between spectra: per scan, per scan and
per gram of sample, and (optionally) per unit receiver gain.

Explicit user request (2026-10-16): "for 1-d spectra, signals can be
'compared' with either normalised with respect to only number of scans,
or number of scans*sample mass".

**What the readers already do.** Bruker, Varian, Spinsolve and X-Pulse
readers divide the data by NS when they load it (``data = spectrum / ns``),
so their spectra are *already* per scan and dividing again would be wrong.
The JEOL reader does **not** (it keeps the summed signal); its number of
scans is read from the file's ``total_scans`` / ``scans`` parameter and
divided here. So "per scan" is a statement about the result, not an
instruction to divide blindly -- :func:`scan_divisor` decides, per
spectrum, what (if anything) still has to be divided.

**Receiver gain** is not divided by any reader. Two spectra recorded with
different RG are not comparable in absolute intensity, per scan or not;
:func:`comparison_factors` can divide by RG (Bruker's ``RG``; linear gain
only -- Varian's ``gain`` is in dB and is never used) and otherwise warns
when the gains differ.

**Unscaled** (2026-10-28) shows the intensities as recorded -- as TopSpin
shows them: a spectrum whose reader divided by NS is multiplied back by it,
nothing is divided by gain or mass, and nothing is normalised. Compare it
with the same intensity scale for every spectrum (the stacked view's
default), so a real difference in height is a real difference in signal.

Notebook use::

    from opennmr.notebook import intensity_scaling as isc
    scaled = isc.scale_for_comparison([s1, s2], mode="per_scan_mass", divide_rg=True)
    scaled["spectra"], scaled["notes"], scaled["skipped"]
"""

from __future__ import annotations

import dataclasses
import math

from ..core.spectrum import NMRSpectrum

#: The comparison modes, as the GUI's Compare ▸ Normalize offers them.
MODES = ("per_scan", "max", "per_scan_mass", "unscaled")

#: Readers that divide by NS at load time (data are already per scan).
PER_SCAN_VENDORS = frozenset({"bruker", "varian", "spinsolve", "xpulse"})


def _jeol_scans(raw: dict) -> float | None:
    params = raw.get("params")
    if not isinstance(params, list):
        return None
    by_name = {str(p.get("name", "")).lower(): p for p in params if isinstance(p, dict)}
    for name in ("total_scans", "scans"):
        p = by_name.get(name)
        if p is None:
            continue
        try:
            value = float(p.get("value"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    return None


def number_of_scans(spectrum: NMRSpectrum) -> float | None:
    """The number of scans the spectrum was recorded with, or None when
    the file does not say (``metadata.ns``, or a JEOL file's own
    ``total_scans`` / ``scans``)."""
    ns = spectrum.metadata.ns
    try:
        ns = float(ns) if ns is not None else None
    except (TypeError, ValueError):
        ns = None
    if ns and ns > 0:
        return ns
    return _jeol_scans(spectrum.metadata.raw or {})


def data_is_per_scan(spectrum: NMRSpectrum) -> bool | None:
    """True when the data were divided by NS on loading, False when they
    were not (JEOL), None when that is not known (a simulation, a
    spectrum from another program)."""
    raw = spectrum.metadata.raw or {}
    flag = raw.get("per_scan")
    if isinstance(flag, bool):
        return flag
    vendor = (spectrum.metadata.vendor or "").lower()
    if vendor in PER_SCAN_VENDORS:
        return True
    if vendor == "jeol":
        return False
    if vendor == "derived":
        # Arithmetic, kept slices and measured traces start from loaded
        # spectra, which are per scan unless they came from JEOL.
        return not ("params" in raw and "header" in raw)
    return None


def scan_divisor(spectrum: NMRSpectrum) -> tuple[float, str | None]:
    """What the data must still be divided by to be per scan, and a note
    when that is not certain. ``(1.0, None)`` for a spectrum already per
    scan."""
    per_scan = data_is_per_scan(spectrum)
    if per_scan:
        return 1.0, None
    if per_scan is None:
        return 1.0, "not known whether it is per scan (taken as it is)"
    ns = number_of_scans(spectrum)
    if ns is None:
        return 1.0, "summed over its scans, but the number of scans is not recorded (not divided)"
    return ns, None


def receiver_gain(spectrum: NMRSpectrum) -> float | None:
    """The linear receiver gain (Bruker ``RG``), or None."""
    acqus = (spectrum.metadata.raw or {}).get("acqus")
    if not isinstance(acqus, dict):
        return None
    try:
        rg = float(acqus.get("RG"))
    except (TypeError, ValueError):
        return None
    return rg if math.isfinite(rg) and rg > 0 else None


def _unscaled_factors(spectrum: NMRSpectrum) -> dict:
    """Back to the intensity as recorded: a reader that divided by NS on
    loading is multiplied by it again; data that were never divided (JEOL)
    are left alone."""
    per_scan = data_is_per_scan(spectrum)
    notes, factor = [], 1.0
    if per_scan:
        ns = number_of_scans(spectrum)
        if ns:
            factor = float(ns)
        else:
            notes.append("number of scans not recorded (shown as loaded, per scan)")
    elif per_scan is None:
        notes.append("not known whether it is per scan (shown as it is)")
    return {"factor": factor, "ns_divisor": 1.0 / factor, "mass_g": spectrum.metadata.mass,
            "rg": receiver_gain(spectrum), "notes": notes, "skip": None}


def comparison_factors(spectrum: NMRSpectrum, mode: str = "per_scan",
                       divide_rg: bool = False) -> dict:
    """The factor the spectrum's data are multiplied by for ``mode``
    (``"per_scan"``, ``"per_scan_mass"``, ``"unscaled"`` -- as recorded --
    or ``"max"``, which only makes the data per scan; the plot scales each
    to its own maximum),
    with what went into it. ``{"factor", "ns_divisor", "mass_g", "rg",
    "notes", "skip"}``; ``skip`` is a reason (no sample mass) or None."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if mode == "unscaled":
        return _unscaled_factors(spectrum)
    divisor, note = scan_divisor(spectrum)
    notes = [note] if note else []
    factor = 1.0 / divisor
    mass = spectrum.metadata.mass
    skip = None
    if mode == "per_scan_mass":
        if not mass or mass <= 0:
            skip = "no sample mass set"
        else:
            factor /= float(mass)
    rg = receiver_gain(spectrum)
    if divide_rg and mode != "max":
        if rg is None:
            notes.append("receiver gain not recorded (not divided)")
        else:
            factor /= rg
    return {"factor": factor, "ns_divisor": divisor, "mass_g": mass, "rg": rg,
            "notes": notes, "skip": skip}


def scale_for_comparison(spectra: list[NMRSpectrum], mode: str = "per_scan",
                         divide_rg: bool = False, labels=None) -> dict:
    """Copies of ``spectra`` scaled for ``mode`` (the originals are never
    changed), the positions kept and skipped, and plain-language notes.

    Returns ``{"spectra", "kept", "skipped": [(position, reason)],
    "factors", "notes"}``. A note is added when the receiver gains differ
    and ``divide_rg`` is off, since absolute intensities are then not
    comparable."""
    labels = list(labels) if labels is not None else [f"spectrum {i + 1}" for i in range(len(spectra))]
    out, kept, skipped, factors, notes = [], [], [], [], []
    gains = []
    for i, spectrum in enumerate(spectra):
        info = comparison_factors(spectrum, mode, divide_rg)
        for n in info["notes"]:
            notes.append(f"{labels[i]}: {n}.")
        if info["skip"]:
            skipped.append((i, info["skip"]))
            continue
        gains.append(info["rg"])
        f = info["factor"]
        if f != 1.0:
            imag = spectrum.data_imag * f if spectrum.data_imag is not None else None
            spectrum = dataclasses.replace(spectrum, data=spectrum.data * f, data_imag=imag)
        out.append(spectrum)
        kept.append(i)
        factors.append(f)
    known = {g for g in gains if g is not None}
    if mode != "max" and not divide_rg and len(known) > 1:
        notes.append("Receiver gains differ (RG " + ", ".join(f"{g:g}" for g in sorted(known))
                     + "): absolute intensities are not comparable unless divided by RG.")
    return {"spectra": out, "kept": kept, "skipped": skipped, "factors": factors, "notes": notes}


#: What the y axis / colour bar means, per mode (and with RG divided).
def intensity_label(mode: str, divide_rg: bool = False) -> str:
    if mode == "unscaled":
        return "Intensity as recorded (a.u.)"
    base = {"per_scan": "Intensity per scan", "max": "Intensity / own maximum",
            "per_scan_mass": "Intensity per scan per gram"}[mode]
    if divide_rg and mode != "max":
        base += " per unit RG"
    unit = " (a.u. g$^{-1}$)" if mode == "per_scan_mass" else ("" if mode == "max" else " (a.u.)")
    return base + unit
