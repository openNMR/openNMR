"""
Acquisition parameters, named the way a pulse programme names them.

Lives in ``core`` rather than beside the reader that populates it
(:mod:`opennmr.io.bruker`) because both layers need it and
``processing`` must not import ``io`` -- ``io.bruker`` already imports
``processing.ft``, so that direction would be circular. ``core``
imports neither, which is what makes it the shared home.
"""

from __future__ import annotations

#: Vendor parameters recorded as *indexed arrays* rather than scalars,
#: so ``P`` holds every pulse length and ``P1`` is not a key at all.
#: These are expanded to ``p0``, ``p1``, ... below. Covers the ones a
#: pulse programme actually refers to by index -- delays, pulses,
#: constants, loop counters, and the gradient tables.
_CONSTANT_ARRAYS = ("D", "P", "CNST", "L", "IN", "INF", "PL", "PLW",
                    "SP", "SPW", "GPZ", "GPX", "GPY")


def acquisition_constants(spectrum) -> dict[str, float]:
    """Every numeric acquisition parameter of ``spectrum``, named the
    way a pulse programme names it -- ``d1``, ``p1``, ``cnst2``,
    ``l4``, ``gpz1`` from the indexed arrays, plus every scalar
    (``ns``, ``td``, ``te``, ``sw``, ...) under its own lowercase name.

    Used as known values in a fitting equation (see
    :func:`opennmr.processing.relaxation.build_custom_model`) and to
    resolve a named pulse length for VOCS (see
    :func:`opennmr.processing.vocs.apply_vocs`).

    **This is the only correct way to ask for one of these by name.**
    A raw ``metadata.raw["acqus"]`` lookup of ``"P1"`` fails: the file
    records ``P`` as a 64-entry array, so the key simply is not there.

    **Values are exactly as the vendor recorded them, not converted**
    -- ``d1`` in seconds, ``p1`` in microseconds, because that is what
    those numbers mean in TopSpin and on the spectrometer, and quietly
    rescaling them would make an equation copied off a pulse programme
    silently wrong. Only ``acqus`` is read: these are acquisition
    facts, not processing choices.

    Returns an empty dict for a spectrum with no retained ``acqus``
    (a non-Bruker dataset, or a derived trace).
    """
    acqus = (getattr(spectrum.metadata, "raw", None) or {}).get("acqus")
    if not isinstance(acqus, dict):
        return {}
    constants: dict[str, float] = {}
    for key, value in acqus.items():
        if key in _CONSTANT_ARRAYS and isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    constants[f"{key.lower()}{index}"] = float(item)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            constants[key.lower()] = float(value)
    return constants
