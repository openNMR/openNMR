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
    way its own pulse programme names it.

    * **Bruker** (``acqus``): lowercase, with the indexed arrays
      expanded -- ``d1``, ``p1``, ``cnst2``, ``l4``, ``gpz1`` -- plus
      every scalar (``ns``, ``td``, ``te``, ``sw``, ...).
    * **X-Pulse** (``exp_params``, and the ``$``-prefixed JCAMP
      ``params``): **exactly as recorded**, e.g. ``TAU``, ``P90``,
      ``D70``, ``DELTA``, ``NS``. A lowercase alias is added as well
      where it is unambiguous (see :func:`_add_alias`) -- X-Pulse is
      the one vendor here whose parameter names are case-*significant*
      (a real PFG file records both ``delta`` and ``DELTA``, which are
      different numbers), so lowercasing everything the way the Bruker
      branch does would silently merge them.

    Used as known values in a fitting equation (see
    :func:`opennmr.processing.relaxation.build_custom_model`), as the
    Values-source factor for a list file (see
    ``MainWindow._resolve_list_factor``), and to resolve a named pulse
    length for VOCS (see :func:`opennmr.processing.vocs.apply_vocs`).

    **This is the only correct way to ask for one of these by name.**
    A raw ``metadata.raw["acqus"]`` lookup of ``"P1"`` fails: the file
    records ``P`` as a 64-entry array, so the key simply is not there.

    **Values are exactly as the vendor recorded them, not converted**
    -- ``d1`` in seconds, ``p1`` in microseconds, ``TAU`` in whatever
    unit the X-Pulse sequence writes it in, because that is what those
    numbers mean on the spectrometer, and quietly rescaling them would
    make an equation copied off a pulse programme silently wrong. Only
    acquisition parameters are read: these are acquisition facts, not
    processing choices.

    Returns an empty dict for a spectrum with no retained parameters
    (a derived trace, or a vendor whose reader keeps none).
    """
    raw = getattr(spectrum.metadata, "raw", None) or {}
    constants: dict[str, float] = {}

    acqus = raw.get("acqus")
    if isinstance(acqus, dict):
        for key, value in acqus.items():
            if key in _CONSTANT_ARRAYS and isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    if _is_number(item):
                        constants[f"{key.lower()}{index}"] = float(item)
            elif _is_number(value):
                constants[key.lower()] = float(value)

    # X-Pulse records its sequence parameters under their own names,
    # case and all. Added after the Bruker branch and without
    # overwriting it, so a dataset that somehow carried both keeps the
    # established lowercase Bruker spelling as the authority.
    exp_params = raw.get("exp_params")
    if isinstance(exp_params, dict):
        for key, value in exp_params.items():
            if _is_number(value):
                constants.setdefault(str(key), float(value))
        for key, value in list(exp_params.items()):
            if _is_number(value):
                _add_alias(constants, str(key), exp_params)

    # The JCAMP header's own "$NS", "$P1", "$RELAXATIONDELAY" and the
    # like, stripped of the "$". Lower precedence than exp_params,
    # which is the sequence's own parameter block.
    params = raw.get("params")
    if isinstance(params, dict):
        for key, value in params.items():
            name = str(key).lstrip("$")
            if name and not name.startswith(".") and _is_number(value):
                constants.setdefault(name, float(value))
    return constants


def _is_number(value) -> bool:
    """Numeric, and not a bool -- ``True`` is an ``int`` in Python and
    a parameter that is really a flag is not a quantity."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _add_alias(constants: dict, key: str, source: dict) -> None:
    """Add a lowercase alias for ``key``, unless another key in
    ``source`` differs from it only by case.

    ``TAU`` is easier to type as ``tau`` in a factor expression, and
    offering both costs nothing. But a real X-Pulse PFG file records
    ``delta`` (the gradient duration) *and* ``DELTA`` (the diffusion
    delay) as separate parameters with a 12x difference between them,
    so a blanket lowercase would resolve one of them to the other's
    value with nothing to show for it. Where that collision exists,
    only the recorded spelling is offered.
    """
    lowered = key.lower()
    if lowered == key or lowered in constants:
        return
    if sum(1 for other in source if str(other).lower() == lowered) > 1:
        return
    constants[lowered] = constants[key]
