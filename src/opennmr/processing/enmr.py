"""
Electrophoretic NMR -- ion mobility from the phase a resonance picks up
under an applied electric field.

A PFG sequence encodes position as phase. In a diffusion experiment the
ions move randomly, so the phase *spreads* and the signal attenuates.
Apply a field as well and the charged species also **drifts**, coherently
-- so on top of the attenuation every ion of a given species acquires
the *same* extra phase, proportional to how far it drifted::

    dphi = gamma * g * delta * DELTA * mu * E        (E = V / d)

Everything on the right except ``mu`` and ``V`` is fixed by the
experiment, so the phase is linear in the applied voltage. A neutral
species drifts nowhere and its phase does not move, which is the
control that tells you the ramp is electrophoresis and not something
the cell did to the spectrum.

**The slope is the result.** It is what the data determines, in rad/V,
and it needs nothing but the data. Converting it to a mobility needs
the cell's electrode gap and the sequence's gradient and timings --
numbers that live in a lab notebook rather than in any vendor file --
so :func:`fit_enmr` reports the slope always and the mobility only when
all of those are supplied.

Gradient directions
-------------------
These experiments are run with the gradient applied in **both
directions**, interleaved, so each voltage appears twice. Reversing the
gradient reverses the sign of the electrophoretic phase but leaves
anything not driven by the field alone. Writing the two fitted slopes
as ``s_a = -s + spurious`` and ``s_b = +s + spurious``:

* the **half-difference** ``(s_b - s_a)/2`` is the measurement;
* the plain mean ``(s_a + s_b)/2`` is the artefact that did not cancel.

:func:`unfold_polarity` puts the two polarities on one signed voltage
axis, so the series reads as a single ramp through the origin instead
of a folded one where every point sits on its partner, and
:func:`branch_slopes` fits each arm separately and returns both -- so a
run where one direction misbehaved is visible rather than averaged into
silence. On that signed axis the roles swap round: both arms carry the
same slope, so their **mean** is the measurement and their
**half-difference** is the part that did not reverse.

Validated against a real measurement, not only in principle: on
``test_data/eNMR/Diff50_1H_eNMR_Calibration_2024-05-13/21`` -- a TMABr
standard -- TMA+'s two directions give -7.64e-3 and +8.16e-3 rad/V, a
half-difference of 7.90e-3 (R^2 = 0.993) and a common residual of
+2.6e-4, about 3% of the signal. The water in the same spectrum gives
3.3e-5 rad/V, zero to within its own error. With the cell's 38.5 mm
electrode gap that slope is mu = 3.79e-8 m^2/Vs, the accepted
literature value for TMA+.
"""

from __future__ import annotations

import numpy as np

#: Electrode separations (metres) are a property of the cell, not of
#: the data, and no vendor file records one. There is deliberately no
#: default: a mobility scales directly with whatever is assumed here,
#: so a plausible-looking number silently rescales every result.
#: :func:`mobility_from_slope` requires it.
_NO_DEFAULT_GAP = object()


def pfg_params(spectrum, nucleus: str | None = None) -> dict:
    """The gradient strength, δ, Δ and γ for a Bruker PFG/eNMR
    experiment, read from the experiment itself.

    Saves typing four constants that are already recorded, and removes
    the transcription errors that come with them.

    ``spectrum`` : a loaded :class:`~opennmr.core.spectrum.NMRSpectrum`
    (its ``metadata.source_path`` is used) or a path.

    Returns ``{"gradient", "delta", "big_delta", "gamma", "sources"}``
    -- in T/m, s, s and **rad/s/T** (already multiplied by 2*pi, unlike
    the Hz/T the file records) -- with ``sources`` naming where each
    came from, because "where did this number come from" is the first
    question anyone asks of an automatic value.

    **What comes from where, and why it is not all the pulse
    programme.** The pulse programme documents its own parameters
    (``;d18: gradient duration``, ``;d20: DELTA``, ``;cnst3:
    z-gradient maximum amplitude``), so it is tempting to read all
    four from ``acqus``. Measured across five real PFG/eNMR datasets:

    * ``Δ = d20`` reproduces ``diff.xml``'s own ``DELTA`` **exactly**
      on all five, so it is read from ``acqus`` and used to cross-check.
    * **δ does not.** ``d17 + d18`` matches on one dataset (the
      calibration) and is 67-83% too large on the other four, whose
      ``diff.xml`` all report δ = 1 ms against ``d18`` values from
      0.896 to 1.568 ms. The sequence composes its gradient pulse from
      shape and ramp in a way that is not recoverable from the
      parameter list alone, so δ is taken from ``diff.xml``, which is
      TopSpin's own computed value.
    * **The gradient cannot be**, at all: ``acqus`` records a
      *percentage* of the probe's maximum, and converting it to T/m
      needs the probe's gradient calibration, which lives in
      ``diff.xml``/``diffConf.xml`` and nowhere else.

    So ``diff.xml`` is the source of record and ``acqus`` is the
    cross-check. Raises when there is no ``diff.xml`` rather than
    guessing a gradient from a percentage.
    """
    from ..io import bruker as io_bruker

    path = spectrum if isinstance(spectrum, str) else getattr(
        getattr(spectrum, "metadata", None), "source_path", None)
    if not path:
        raise ValueError(
            "Need a Bruker experiment path (or a spectrum loaded from one) to read "
            "the PFG parameters."
        )
    if nucleus is None and not isinstance(spectrum, str):
        nucleus = getattr(getattr(spectrum, "metadata", None), "nucleus", None)

    try:
        diff = io_bruker.read_diff_xml(path, nucleus=nucleus)
    except FileNotFoundError as exc:
        raise ValueError(
            f"No diff.xml in {path!r}. The gradient strength in T/m is not "
            f"recoverable without it – acqus records only a percentage of the "
            f"probe's maximum, and the calibration that turns that into T/m is "
            f"not in the parameter list. Supply gradient, delta and big_delta "
            f"explicitly."
        ) from exc

    if diff.get("gamma") is None:
        raise ValueError(
            f"diff.xml in {path!r} records no gyromagnetic ratio for nucleus "
            f"{nucleus!r}. Pass nucleus= explicitly (a bare path carries no "
            f"nucleus of its own, unlike a loaded spectrum)."
        )
    gradient = np.asarray(diff["gradient"], dtype=float)
    sources = {"gradient": "diff.xml", "delta": "diff.xml",
               "big_delta": "diff.xml", "gamma": "diff.xml"}

    big_delta = float(diff["DELTA"])
    # Cross-check against the pulse programme's own DELTA parameter.
    # Agreement is the normal case; disagreement is worth knowing about
    # rather than silently preferring one.
    from ..core.acquisition import acquisition_constants
    d20 = acquisition_constants(spectrum).get("d20") if not isinstance(spectrum, str) else None
    if d20 is not None and not np.isclose(d20, big_delta, rtol=1e-6):
        sources["big_delta"] = f"diff.xml (acqus d20 disagrees: {d20:g} s)"
    elif d20 is not None:
        sources["big_delta"] = "diff.xml (= acqus d20)"

    return {
        # A PFG series varies the gradient; an eNMR series holds it
        # fixed and varies the voltage, so one value is the normal
        # case here -- but the whole array is kept for a caller that
        # needs it.
        "gradient": float(gradient[0]) if gradient.size else float("nan"),
        "gradients": gradient,
        "delta": float(diff["delta"]),
        "big_delta": big_delta,
        # diff.xml records gamma in Hz/T; the phase formula needs
        # rad/s/T. Converting here rather than at every call site is
        # what stops a stray factor of 2*pi.
        "gamma": float(2.0 * np.pi * diff["gamma"]),
        "sources": sources,
    }


def phase_series(rows, freq_ppm, region: tuple[float, float], unwrap: bool = True,
                  stride: int = 1) -> np.ndarray:
    """The phase of one resonance in every slice, in radians.

    Taken as the argument of the **complex integral** over ``region``,
    not of the single tallest point: integrating uses every point of
    the line, so it is far less sensitive to noise and to a line whose
    maximum wanders by a point between slices. Measured on the TMABr
    standard, this alone gives R^2 = 0.995 against voltage with no
    lineshape fit involved at all -- worth knowing before reaching for
    a full complex fit, which is needed for *overlapped* peaks rather
    than for phase accuracy as such.

    ``rows`` : complex spectra, one row per slice. The imaginary part
    is essential -- a real-only spectrum has no phase to read.

    ``unwrap`` : follow the ramp through +/-pi rather than folding it
    (the default). Turn it off only to inspect the raw wrapped values;
    a fit against wrapped phase is meaningless as soon as the ramp
    passes pi.

    ``stride`` : unwrap every ``stride``-th point as its own series.
    **Pass 2 for a +V/-V interleaved acquisition**, which is how these
    experiments are normally run.

    That is not a refinement, it is a correctness requirement, and
    getting it wrong is silent. Consecutive points of an interleaved
    series are *opposite polarity*, so they legitimately sit far apart
    -- unwrapping across them makes ``np.unwrap`` read a genuine
    polarity jump as a wrap and insert a 2*pi step that was never
    there. Caught on real data
    (``Diff50_eNMR_Li_Ac_6p25M_2024-09-03/21``): the last point of one
    polarity went from -1.588 to +4.695 rad, which turned a clean
    monotonic ramp (slope -1.40e-2, R^2 = 0.88) into no correlation at
    all (+1.58e-3, R^2 = 0.009). The fit did not look broken; it looked
    like a sample with no mobility.
    """
    rows = np.asarray(rows)
    if rows.ndim != 2:
        raise ValueError(f"Expected one row per slice, got shape {rows.shape}.")
    if not np.iscomplexobj(rows):
        raise ValueError(
            "eNMR phase needs the complex spectrum – this one has no imaginary "
            "part, so there is no phase to measure."
        )
    freq_ppm = np.asarray(freq_ppm, dtype=float)
    if rows.shape[1] != freq_ppm.size:
        raise ValueError(f"{rows.shape[1]} spectral points against {freq_ppm.size} shifts.")

    low, high = (min(region), max(region))
    mask = (freq_ppm >= low) & (freq_ppm <= high)
    if not mask.any():
        raise ValueError(
            f"No spectral points between {low:g} and {high:g} ppm (axis runs "
            f"{freq_ppm.min():g} to {freq_ppm.max():g})."
        )
    phases = np.angle(rows[:, mask].sum(axis=1))
    if not unwrap:
        return phases
    if stride <= 1:
        return np.unwrap(phases)
    unwrapped = phases.copy()
    for offset in range(stride):
        unwrapped[offset::stride] = np.unwrap(phases[offset::stride])
    return unwrapped


def phase_ramp(rows, freq_ppm, region: tuple[float, float], voltages) -> np.ndarray:
    """:func:`phase_series` with the unwrapping done correctly for the
    acquisition it is given.

    The one-call path, and the one to use by default. It has the
    voltages, so it can see whether the series is +V/-V interleaved and
    unwrap each polarity as its own series -- which
    :func:`phase_series` cannot decide on its own, and which is silently
    wrong if left to the default ``stride=1`` (see there).
    """
    voltages = np.asarray(voltages, dtype=float)
    stride = 2 if polarity_pairs(voltages, np.zeros(voltages.size))["paired"] else 1  # noqa: E501
    return phase_series(rows, freq_ppm, region, unwrap=True, stride=stride)


def polarity_pairs(voltages, phases, positive_first: bool = True) -> dict:
    """Collapse the **+V/-V pairs within one experiment** into one
    phase difference per voltage magnitude.

    Each voltage is acquired at both polarities. Reversing the field
    reverses the drift, so the two differ by ``2*dphi`` while every
    phase error that is *not* field-driven -- receiver phase, shim
    drift, the electrodes' own effect on the lineshape -- is common to
    both and cancels in the half-difference of the pair::

        dphi = (phi_plus - phi_minus) / 2

    **The acquisition takes the positive polarity first** (confirmed by
    the user for this instrument), so the first of each pair is
    ``phi_plus``. ``positive_first=False`` for an acquisition written
    the other way round.

    This is the literal reading, and it makes the returned slope agree
    with :func:`branch_slopes`'s arm mean rather than being its
    negative. What it does **not** fix is the sign of a *mobility*:
    that also needs to know which way the cell's positive electrode
    points relative to the gradient, which no file records. See
    ``cell_polarity`` on :func:`mobility_from_slope`.

    This is the *within-experiment* pairing. Reversing the **gradient**
    is a separate thing done by re-running the whole experiment; see
    :func:`combine_experiments`.

    Returns ``{"voltages", "dphi", "paired"}``. ``paired`` is False
    when the series is not in pairs, in which case the inputs come
    back untouched and the caller fits the signed series directly.
    Pairing is **detected, not assumed**: consecutive entries must
    share a magnitude. Assuming it on an unpaired series differences
    unrelated points, which produces a clean straight line through
    nonsense.
    """
    voltages = np.asarray(voltages, dtype=float)
    phases = np.asarray(phases, dtype=float)
    if voltages.size != phases.size:
        raise ValueError(f"{voltages.size} voltages against {phases.size} phases.")
    if voltages.size < 4:
        return {"voltages": voltages, "dphi": phases, "paired": False}

    even, odd = voltages[0::2], voltages[1::2]
    if even.size != odd.size or not np.allclose(np.abs(even), np.abs(odd)):
        return {"voltages": voltages, "dphi": phases, "paired": False}

    first, second = phases[0::2], phases[1::2]
    plus, minus = (first, second) if positive_first else (second, first)
    return {
        "voltages": np.abs(odd),
        "dphi": (plus - minus) / 2.0,
        "paired": True,
    }


def unfold_polarity(voltages, phases=None, positive_first: bool = True) -> dict:
    """Unfold the +V/-V pairs onto a **signed** voltage axis.

    A vdlist records the voltage *magnitude*, and each magnitude is
    acquired twice at opposite polarity -- so the recorded axis is
    ``0, 0, 8, 8, 16, 16, ...`` and every pair of points lands on top
    of itself. Unfolding assigns the first of each pair ``+|V|`` and
    the second ``-|V|``, which spreads the same data across
    ``-Vmax .. +Vmax`` as one ramp through the origin.

    Two things become visible that the folded axis hides:

    * **the zero crossing** -- the two arms are one line, and where
      they meet is where the field is off;
    * **the two arms separately**, which is the whole point of running
      both polarities. They should be mirror images. When bubble
      formation drives convection in a single direction, one arm is
      distorted and the other is not, and averaging them buries that.
      :func:`branch_slopes` puts numbers on it.

    ``positive_first`` : the acquisition's own ordering. True (the
    default) is **positive first, negative second within each pair**.

    Returns ``{"voltages", "phases", "paired", "positive", "negative",
    "order", "positive_first"}``. ``positive``/``negative`` are boolean
    masks and ``order`` sorts the points along the signed axis for
    plotting. When the series is not in pairs nothing is assumed:
    ``paired`` is False and the inputs come back untouched.

    The masks are built from the acquisition order, **not** from
    ``signed >= 0``: the V = 0 pair unfolds to ``+0.0`` and ``-0.0``,
    and ``-0.0 >= 0`` is True, so a comparison would put both zero
    points in the positive arm and leave the negative one a point
    short.
    """
    voltages = np.asarray(voltages, dtype=float)
    phases = None if phases is None else np.asarray(phases, dtype=float)
    probe = np.zeros(voltages.size) if phases is None else phases
    if not polarity_pairs(voltages, probe)["paired"]:
        return {
            "voltages": voltages, "phases": phases, "paired": False,
            "positive": np.ones(voltages.size, dtype=bool),
            "negative": np.zeros(voltages.size, dtype=bool),
            "order": np.argsort(voltages, kind="stable"),
            "positive_first": positive_first,
        }

    positive = np.zeros(voltages.size, dtype=bool)
    positive[(0 if positive_first else 1)::2] = True
    signed = np.where(positive, np.abs(voltages), -np.abs(voltages))
    return {
        "voltages": signed, "phases": phases, "paired": True,
        "positive": positive, "negative": ~positive,
        "order": np.argsort(signed, kind="stable"),
        "positive_first": positive_first,
    }


def _fit_mask(signed, max_voltage, exclude, paired: bool = True,
               balance: bool = False) -> np.ndarray:
    """Which points a fit should use, as a boolean mask over the
    unfolded series.

    ``max_voltage`` is either a magnitude -- ``|V| <= max_voltage``,
    symmetric, which drops whole pairs -- or a ``(low, high)`` pair of
    limits **on the signed axis**, which need not be symmetric.
    One-sided is the useful case: a single polarised electrode drives
    the distortion one way only, so the range worth keeping is
    typically not centred on zero.

    ``exclude`` drops named points (a boolean mask or an index array),
    and does so **point by point**. Bubble formation from one
    electrode polarising adds a phase in one direction only, so the
    bad points come one at a time rather than in pairs, and widening
    an exclusion to the partner would throw away good data to keep a
    symmetry the artefact has already broken.

    ``balance`` widens each dropped point to its +-V partner, for a
    caller that does want the arms covering identical ranges -- their
    slopes are then comparable point for point, at the cost of the
    surviving half of every broken pair. Off by default.
    """
    signed = np.asarray(signed, dtype=float)
    used = np.ones(signed.size, dtype=bool)
    if max_voltage is not None:
        limits = np.atleast_1d(np.asarray(max_voltage, dtype=float))
        # Tolerant at the boundary: a voltage computed from a vdlist
        # lands on 56.000000000000007 as easily as on 56.
        slack = 1e-9 * max(1.0, float(np.abs(limits).max()))
        if limits.size == 1:
            used &= np.abs(signed) <= limits[0] + slack
        elif limits.size == 2:
            low, high = float(limits.min()), float(limits.max())
            used &= (signed >= low - slack) & (signed <= high + slack)
        else:
            raise ValueError(
                f"max_voltage is a magnitude or a (low, high) pair, got "
                f"{limits.size} values."
            )
    if exclude is not None:
        exclude = np.asarray(exclude)
        if exclude.dtype == bool:
            used &= ~exclude
        elif exclude.size:
            used[exclude.astype(int)] = False
    if not balance or not paired or used.size % 2:
        return used
    both = used.reshape(-1, 2).all(axis=1)
    return np.repeat(both, 2)


def zero_phase(voltages, phases) -> float | None:
    """The phase measured with the field **off**, or None when the
    series has no ``V = 0`` point.

    Every eNMR series in the available data starts with a ``0 V`` pair,
    which is the acquisition's own record of "no shift": whatever
    receiver phase, shim and lineshape contribute, they contribute it
    here too, with no electrophoresis on top. Both members of the pair
    are averaged -- at zero volts the polarity means nothing, so they
    are two measurements of the same thing.

    This is the reference a phase shift should be quoted against. The
    alternative is the *fitted* common offset (``centre`` on
    :func:`branch_slopes`), which is an extrapolation; they agree when
    the ramp is clean and differ when it is not, and the measured one
    is the one that does not move when the fit changes.
    """
    voltages = np.asarray(voltages, dtype=float)
    phases = np.asarray(phases, dtype=float)
    at_zero = voltages == 0
    return float(phases[at_zero].mean()) if at_zero.any() else None


def branch_slopes(voltages, phases, *, positive_first: bool = True,
                   tolerance: float = 2.0, max_voltage=None,
                   exclude=None, balance_pairs: bool = False,
                   reference: str | None = "zero") -> dict:
    """Fit each polarity arm of an unfolded series **separately**, so
    the two can be compared before anything is averaged.

    On the signed axis of :func:`unfold_polarity` the electrophoretic
    phase is *odd* in the voltage, so both arms have the **same**
    slope: writing a non-reversing contribution as ``a(V)``,

        phi(+V) = s*V + c + a(V)      slope  s + a'
        phi(-V) = -s*V + c + a(V)     slope  s - a'

    which makes the two arms' **mean** the measurement and their
    **half-difference** exactly the part that did not reverse::

        slope     = (s_pos + s_neg) / 2      <- the measurement
        asymmetry = (s_pos - s_neg) / 2      <- what did not reverse

    That second number is the reason the experiment is run at both
    polarities at all. Convection from bubble formation runs one way
    only: it adds a phase that does *not* follow the field, so it
    cancels in the mean and lands entirely in ``asymmetry``. It is
    identically the slope of the pairwise half-sum against ``|V|``,
    which is the same statement read off the folded axis.

    **Both arms can be excellent straight lines and still disagree**,
    which is why R^2 on the averaged fit does not catch this. Measured
    on real data, worst first:

    ===========================  =========  =========  =========
    dataset                        s_pos      s_neg     asymmetry
    ===========================  =========  =========  =========
    Li/acetate 560                +6.9e-3    -1.1e-2    465%
    Li/acetate 20                 -9.1e-4    -2.4e-2     93%
    Li/acetate 572                +9.6e-3    +5.4e-3     28%
    TMABr calibration             -7.6e-3    -8.2e-3      3%
    ===========================  =========  =========  =========

    On 560 the positive arm is a clean line (R^2 = 0.990) and the
    negative one is wreckage (R^2 = 0.559) -- one direction convecting
    and the other not, exactly as described. Their average, -1.9e-3
    rad/V, is not a measurement of anything, and nothing in the
    averaged fit says so.

    ``reference`` : ``"zero"`` (the default) quotes every phase
    against the **field-off measurement** -- see :func:`zero_phase` --
    so ``V = 0`` reads zero and the numbers are shifts rather than
    phases carrying the receiver's along with them. ``None`` leaves the
    raw phase. It is a change of origin: slope, asymmetry and R^2 are
    identical either way.

    ``tolerance`` : how many combined standard errors the two arms may
    differ by and still count as agreeing (``agree`` in the result).

    ``max_voltage`` : fit only ``|V| <= max_voltage``. The phase ramp
    goes nonlinear at the top of the range -- the field heats the
    sample, and what that starts is not proportional to the voltage --
    so the highest points can pull a slope that the rest of the series
    determines perfectly well. Excluded points are still returned and
    still plotted; the fitted line is extrapolated across them, which
    is what makes the deviation legible instead of hidden.

    ``exclude`` : drop named points as well (boolean mask or indices),
    **one at a time**. A single polarised electrode bubbles and shifts
    one polarity's phase, so the bad points do not come in pairs;
    widening an exclusion to the partner would discard good data to
    preserve a symmetry the artefact has already broken.
    ``balance_pairs=True`` widens them anyway, for a caller who wants
    the arms covering identical ranges.

    ``max_voltage`` also accepts a ``(low, high)`` pair on the signed
    axis rather than a magnitude, for trimming one side only.

    When the arms end up covering different voltages the result says
    so (``balanced`` False), because ``asymmetry`` is then partly the
    range difference and not only the sample.

    Returns the two fits, ``slope``/``slope_error``, ``asymmetry``/
    ``asymmetry_error``/``asymmetry_fraction``, the ``combined`` fit
    through every unfolded point, ``zero_crossing`` (its intercept),
    ``branch_gap`` (how far apart the two arms extrapolate at V = 0 --
    they should meet), ``agree``, and ``fit``/``fit_combined``: the
    per-arm and single-line predictions, in the input's own order, for
    plotting. A per-arm line that kinks at the origin *is* the
    asymmetry, drawn.

    **Sign.** ``slope`` is the negative of :func:`fit_enmr`'s paired
    slope, exactly -- the two differ only in which polarity is called
    positive, and no vendor file records which electrode the first of
    each pair drove. :func:`fit_enmr`'s convention is the one pinned
    down by an external value: it returns a *positive* mobility for
    TMA+, a cation, against the accepted literature figure.
    """
    voltages = np.asarray(voltages, dtype=float)
    phases = np.asarray(phases, dtype=float)
    unfolded = unfold_polarity(voltages, phases, positive_first=positive_first)
    if not unfolded["paired"]:
        raise ValueError(
            "This series is not in +V/-V pairs, so there are no two arms to "
            "compare – consecutive points must share a voltage magnitude. Fit "
            "it with fit_enmr directly."
        )

    signed = unfolded["voltages"]
    # Quote the shift against the field-off measurement, so V = 0 reads
    # zero and every number below is a *shift* rather than a phase that
    # happens to carry the receiver's along with it. Subtracting a
    # constant moves intercepts and nothing else -- slope, asymmetry
    # and R^2 are untouched, and there is a test that says so.
    reference_phase, referenced_to = 0.0, "nothing (raw phase)"
    if reference == "zero":
        measured = zero_phase(voltages, phases)
        if measured is not None:
            reference_phase, referenced_to = float(measured), "V = 0"
        else:
            referenced_to = "nothing (no V = 0 point in this series)"
    elif reference is not None:
        raise ValueError(
            f"reference is 'zero' (the field-off measurement) or None (raw "
            f"phase), got {reference!r}."
        )
    phases = phases - reference_phase
    positive, negative = unfolded["positive"], unfolded["negative"]
    used = _fit_mask(signed, max_voltage, exclude, balance=balance_pairs)
    fit_positive, fit_negative = positive & used, negative & used
    if min(int(fit_positive.sum()), int(fit_negative.sum())) < 3:
        kept = "" if used.all() else (
            f" ({int(used.sum())} of {used.size} points kept"
            + (f", |V| <= {max_voltage:g}" if max_voltage is not None else "") + ")")
        raise ValueError(
            f"Each polarity needs at least 3 points to fit a slope and judge it; "
            f"this series gives {int(fit_positive.sum())} and "
            f"{int(fit_negative.sum())}{kept}. Raise the voltage cutoff, or fit "
            f"it with fit_enmr, which uses the pairs together."
        )

    pos = _line(signed[fit_positive], phases[fit_positive])
    neg = _line(signed[fit_negative], phases[fit_negative])
    combined = _line(signed[used], phases[used])

    slope = (pos["slope"] + neg["slope"]) / 2.0
    asymmetry = (pos["slope"] - neg["slope"]) / 2.0
    # The arms are independent fits, so their errors add in quadrature
    # and the halving carries through to both the mean and the
    # difference alike.
    error = float(np.hypot(pos["slope_error"], neg["slope_error"]) / 2.0)

    prediction = np.where(
        positive, pos["slope"] * signed + pos["intercept"],
        neg["slope"] * signed + neg["intercept"])
    centre = float((pos["intercept"] + neg["intercept"]) / 2.0)
    # The same two slopes drawn through the *shared* zero crossing.
    # Each arm keeps its own slope, so a disagreement still shows as a
    # kink at the origin -- but the line is continuous there, instead
    # of jumping by ``branch_gap`` and reading as a spike on the plot.
    # The gap is reported as a number; it does not need drawing twice.
    joined = np.where(positive, pos["slope"] * signed + centre,
                       neg["slope"] * signed + centre)

    return {
        "voltages": signed, "phases": phases, "order": unfolded["order"],
        "positive": positive, "negative": negative,
        # Judged against every point it fitted -- all of them unless a
        # cutoff was set -- which is what a fit made of two separate
        # lines has to be judged against. The predictions below still
        # span the whole axis, so an excluded point's deviation from
        # the line it did not influence stays visible.
        "r_squared": _r_squared(phases[used], prediction[used]),
        "used": used, "excluded": ~used, "max_voltage": max_voltage,
        "reference_phase": reference_phase, "referenced_to": referenced_to,
        # The arms are compared by slope, so they should be covering
        # the same voltages. Removing single points is often right --
        # one polarised electrode distorts one polarity -- but it
        # leaves the two arms measuring different ranges, and then part
        # of `asymmetry` is that difference rather than the sample.
        "balanced": bool(np.array_equal(
            np.sort(np.abs(signed[positive & used])),
            np.sort(np.abs(signed[negative & used])))),
        # The ramp's common phase offset: the average of what the two
        # arms extrapolate to where the field is off. Subtracting it
        # leaves only the field-driven part, which is the only part
        # worth averaging between two acquisitions.
        "centre": centre,
        "positive_branch": pos, "negative_branch": neg, "combined": combined,
        "slope": float(slope), "slope_error": error,
        "asymmetry": float(asymmetry), "asymmetry_error": error,
        "asymmetry_fraction": (abs(asymmetry) / abs(slope)
                                if slope else float("inf")),
        "zero_crossing": combined["intercept"],
        "branch_gap": float(pos["intercept"] - neg["intercept"]),
        "agree": bool(abs(asymmetry) <= tolerance * error),
        "fit": prediction, "fit_joined": joined,
        "fit_combined": combined["slope"] * signed + combined["intercept"],
        "positive_first": positive_first, "tolerance": float(tolerance),
    }


def cutoff_sweep(voltages, phases, cutoffs=None, *, positive_first: bool = True) -> list:
    """How the fitted slope moves as the top of the voltage range is
    trimmed away.

    Removing the highest voltages is often necessary -- the ramp goes
    nonlinear where the field starts heating the sample -- but the
    cutoff is **not** a free parameter to tune until the fit looks
    good. On this data the slope moves a long way with it, and R^2
    moves the *opposite* way, so a cutoff chosen by fit quality alone
    picks the answer:

    ==========  ==========  ======  ============
    |V| <=      slope       R^2     arms disagree
    ==========  ==========  ======  ============
    all (120)   -1.32e-2    0.882   8%
    100         -1.04e-2    0.849   10%
    80          -7.04e-3    0.818   14%
    60          -3.64e-3    0.959   44%
    40          -2.66e-3    0.982   113%
    ==========  ==========  ======  ============

    (the 20/21 gradient pair, averaged). A seven-fold change in the
    answer with R^2 improving throughout is the thing to see before
    choosing, which is what this returns: one entry per cutoff, with
    ``cutoff``, ``slope``, ``slope_error``, ``r_squared``,
    ``asymmetry_fraction`` and ``n_points``. A slope that is stable
    across several cutoffs is one the range is not deciding.

    ``cutoffs`` : the magnitudes to try. Defaults to every distinct
    voltage magnitude in the series, largest first, which is every
    cutoff that changes anything.
    """
    voltages = np.asarray(voltages, dtype=float)
    phases = np.asarray(phases, dtype=float)
    if cutoffs is None:
        cutoffs = sorted({abs(float(v)) for v in voltages if v}, reverse=True)
    swept = []
    for cutoff in cutoffs:
        try:
            branches = branch_slopes(voltages, phases, positive_first=positive_first,
                                      max_voltage=cutoff)
        except ValueError:
            # Too few points left to judge a slope: not an error, just
            # the end of the useful range.
            continue
        swept.append({
            "cutoff": float(cutoff), "slope": branches["slope"],
            "slope_error": branches["slope_error"],
            "r_squared": branches["r_squared"],
            "asymmetry_fraction": branches["asymmetry_fraction"],
            "n_points": int(np.asarray(branches["used"]).sum()),
        })
    return swept


def format_branch_report(branches: dict, label: str | None = None,
                          width: int = 66) -> str:
    """A readable comparison of the two polarity arms.

    Puts the arms side by side and the asymmetry underneath, because
    the question this answers is whether averaging them is justified --
    not what the average is.
    """
    rule = "=" * width
    lines = [rule, (label or "eNMR — ± POLARITY ARMS").center(width).rstrip(), rule, ""]
    lines.append("  Arm        slope (rad/V)        R²      φ at V = 0")
    lines.append("  " + "-" * (width - 4))
    for name, fit in (("+V", branches["positive_branch"]),
                       ("−V", branches["negative_branch"])):
        lines.append(f"   {name:>4}  {fit['slope']:+11.4g} ± {fit['slope_error']:<9.2g}"
                      f"{fit['r_squared']:6.4f}   {fit['intercept']:+8.4f}")
    lines.append("  " + "-" * (width - 4))
    lines.append(f"   mean  {branches['slope']:+11.4g} ± {branches['slope_error']:<9.2g}"
                  f"← the measurement")
    share = branches["asymmetry_fraction"] * 100
    lines.append(f"   asym. {branches['asymmetry']:+11.4g} ± "
                  f"{branches['asymmetry_error']:<9.2g}← did not reverse "
                  f"({share:.0f}% of it)")
    lines.append("")
    lines.append(f"  Shift quoted against {branches.get('referenced_to', 'nothing')}"
                  f"  (φ at 0 V = 0 by construction)")
    lines.append(f"  Zero crossing: φ = {np.degrees(branches['zero_crossing']):+.3f}°;"
                  f"  the arms meet to {abs(np.degrees(branches['branch_gap'])):.3f}°")
    lines.append(f"  mean slope = {np.degrees(branches['slope']):+.4g} °/V")
    if branches.get("max_voltage") is not None:
        used = np.asarray(branches["used"], dtype=bool)
        lines.append(f"  Fitted over {_range_text(branches['max_voltage'])}: "
                      f"{int(used.sum())} of {used.size} points "
                      f"({int((~used).sum())} excluded, still plotted)")
    if not branches.get("balanced", True):
        lines.append("")
        for chunk in _wrap(
            "The arms no longer cover the same voltages, so part of the "
            "asymmetry above is that difference and not the sample. That is "
            "the right trade when one polarity is the damaged one; read the "
            "arm you kept, not the difference between them.", width - 4):
            lines.append(f"  {chunk}")
    lines.append("")
    if branches["agree"]:
        message = (
            "The two arms agree to within their own errors, so the field-driven "
            "phase reversed as it should and averaging them is justified.")
    else:
        message = (
            "The two arms DISAGREE by more than their errors allow. Something "
            "in this measurement did not reverse with the field – convection "
            "from bubble formation runs one direction only and does exactly "
            "this. Look at the arms before trusting their average: a clean "
            "line on one side and a poor one on the other is one direction "
            "convecting, and two clean lines of different slope is a drift "
            "common to both.")
    for chunk in _wrap(message, width - 4):
        lines.append(f"  {chunk}")
    lines.append(rule)
    return "\n".join(lines)


def combine_experiments(results, align_sign: bool = True) -> dict:
    """Average the slopes of the **repeat experiments** run with the
    gradient in each direction.

    The experiment is generally run twice with the gradients switched,
    and averaging the two removes effects that do not follow the
    gradient. This is a separate operation from
    :func:`polarity_pairs`, which handles the +V/-V pairs *inside* one
    experiment.

    ``results`` : :func:`fit_enmr` results, or plain slopes.

    ``align_sign`` : whether to flip any slope that opposes the first
    before averaging (the default).

    **Prefer :func:`average_gradient_pair` where the ramps themselves
    are available.** Aligning signs by comparing slopes is guesswork,
    and it is no longer necessary: Bruker's ``diff.xml`` records a
    *signed* ``gmax``, and every reversed repeat in the available data
    carries it (+4.05/-4.05 T/m on experiments 20/21; +-0.9 on 550/551,
    560/561, 570/571, 572/573). An earlier version of this docstring
    said no dataset settled the question -- it was reading the
    unsigned magnitude, and the sign was in the file all along. The
    two agreeing repeats it cited (-7.29e-3 and -7.55e-3 rad/V) are
    570 and 572, which were both run at **+0.9 T/m**: a same-direction
    repeat, not a reversed pair.

    This function still exists for slopes with no gradient recorded
    alongside them. Aligning is the safe default there, because a sign
    mismatch then shows up as a large ``spread`` rather than silently
    averaging the measurement down toward zero. Set it False to
    average exactly as given.

    Returns ``{"slope", "slope_error", "spread", "n", "slopes",
    "sign_flipped", "aligned"}``. **Check ``spread``**: it is the
    standard deviation of the combined slopes, and two runs of the same
    sample should agree to well within their own errors. A spread
    comparable to the slope means the repeats disagree about more than
    noise, and the average of them is not a measurement.
    """
    slopes, errors = [], []
    for entry in results:
        if isinstance(entry, dict):
            slopes.append(float(entry["slope"]))
            errors.append(float(entry.get("slope_error", 0.0)))
        else:
            slopes.append(float(entry))
            errors.append(0.0)
    if len(slopes) < 2:
        raise ValueError(
            f"Combining repeats needs at least 2 experiments, got {len(slopes)}."
        )

    slopes = np.asarray(slopes, dtype=float)
    errors = np.asarray(errors, dtype=float)
    flipped = np.zeros(slopes.size, dtype=bool)
    if align_sign and slopes[0] != 0:
        flipped = np.sign(slopes) != np.sign(slopes[0])
        slopes = np.where(flipped, -slopes, slopes)

    mean = float(np.mean(slopes))
    # Errors of independent repeats combine in quadrature and the
    # averaging divides by n.
    combined_error = float(np.sqrt(np.sum(errors ** 2)) / slopes.size)
    return {
        "slope": mean, "slope_error": combined_error,
        "spread": float(np.std(slopes, ddof=1)) if slopes.size > 1 else 0.0,
        "n": int(slopes.size), "slopes": slopes,
        "sign_flipped": flipped, "aligned": bool(align_sign),
    }


def _measure_ramp(voltages, phases, positive_first: bool,
                   max_voltage=None, exclude=None, balance_pairs: bool = False,
                   reference: str | None = "zero") -> dict:
    """One phase ramp's slope, its common offset and a prediction,
    measured **through the +-V arms** where the series has them.

    The one estimator used at both levels -- a single experiment and
    an average of several -- so the numbers are comparable. A single
    least-squares line through every unfolded point is not the same
    thing: the arms have separate intercepts whenever something did
    not reverse, and forcing one shared intercept then drags the
    slope. Falls back to that single line only when there are no pairs
    to fit arms to.
    """
    unfolded = unfold_polarity(voltages, phases, positive_first=positive_first)
    # Decided by whether the series *has* arms, not by catching
    # whatever branch_slopes threw. A bare `except ValueError` here
    # silently swapped in the single-line estimator the moment anything
    # else went wrong -- which is exactly how a name collision in
    # average_gradient_pair went unnoticed, moving the 20/21 answer
    # from -1.32e-2 to -1.05e-2 with nothing to say so.
    if unfolded["paired"]:
        return _measure_through_arms(voltages, phases, positive_first,
                                      max_voltage, exclude, balance_pairs, reference)

    used = _fit_mask(unfolded["voltages"], max_voltage, exclude,
                      paired=False, balance=balance_pairs)
    fit = _line(unfolded["voltages"][used], phases[used])
    return {
        "slope": fit["slope"], "slope_error": fit["slope_error"],
        "centre": fit["intercept"], "r_squared": fit["r_squared"],
        "prediction": fit["slope"] * unfolded["voltages"] + fit["intercept"],
        "branches": None, "used": used, "reference": 0.0,
    }


def _measure_through_arms(voltages, phases, positive_first, max_voltage, exclude,
                           balance_pairs, reference) -> dict:
    """The paired half of :func:`_measure_ramp`."""
    branches = branch_slopes(voltages, phases, positive_first=positive_first,
                              max_voltage=max_voltage, exclude=exclude,
                              balance_pairs=balance_pairs, reference=reference)
    return {
        "slope": branches["slope"], "slope_error": branches["slope_error"],
        "centre": branches["centre"], "r_squared": branches["r_squared"],
        "prediction": branches["fit_joined"], "branches": branches,
        "used": branches["used"], "reference": branches["reference_phase"],
    }


def average_gradient_pair(experiments, *, positive_first: bool = True,
                           tolerance: float = 2.0, max_voltage=None,
                           exclude=None, balance_pairs: bool = False,
                           reference: str | None = "zero") -> dict:
    """Average the **gradient-reversed repeats** at the level of the
    phase ramps themselves, and keep both inputs so they can be
    compared before the average is trusted.

    This is the *second* pairing in an eNMR measurement, and it is a
    different acquisition rather than a different point in one:

    * the **voltage pairs** inside one experiment come from the vdlist,
      which records each magnitude twice at opposite polarity --
      :func:`unfold_polarity` and :func:`branch_slopes`;
    * the **gradient pair** is two whole experiments, one run with
      ``+g`` and one with ``-g``. It does not always exist, and it is
      often what the measurement needs.

    ``experiments`` : one dict per experiment, each with ``voltages``,
    ``phases`` and ``gradient`` (signed, in T/m -- what
    :func:`pfg_params` returns), plus an optional ``label``.

    **The gradient sign is read, not guessed.** Bruker's ``diff.xml``
    records a signed ``gmax``, and every reversed repeat in the
    available data carries it: +4.05 / -4.05 T/m on experiments 20/21,
    +0.9 / -0.9 on 550/551, 560/561, 570/571 and 572/573. Reversing the
    gradient reverses the encoded phase, so each ramp is multiplied by
    the sign of its own gradient before averaging -- which is what
    :func:`combine_experiments` had to infer from the slopes.

    Each ramp is also **centred on its own fitted zero crossing**
    first. The constant part of a phase is receiver phase and shim, it
    differs between two acquisitions for no physical reason, and
    averaging it in would tilt the result.

    The voltage axes must match: these are repeats of one experiment,
    the vdlists are identical in every available pair, and averaging
    ramps measured at different voltages would silently interpolate
    something nobody asked for.

    ``max_voltage`` / ``exclude`` : restrict the fit, applied
    identically to every experiment and to their average -- see
    :func:`branch_slopes`. The same cutoff for all of them is the
    point: fitting two repeats over different ranges would turn a range
    difference into a disagreement between them.

    Returns the shared signed ``voltages``, ``corrected`` (one
    sign-corrected, centred ramp per experiment), their pointwise
    ``average`` and its ``fit``, the per-experiment ``slopes`` and
    ``spread``, and ``agree``. Everything needed to draw the two inputs
    and the average on one axes.

    **Check ``agree`` before using the average.** Two runs of one
    sample differing by more than their own errors do not average into
    a measurement -- and on this data they often do differ: sign for
    sign, 560/561 disagree by 133% and 572/573 by more than their own
    size, while the same-gradient repeats 570/572 agree to 4%.
    """
    prepared = []
    for index, entry in enumerate(experiments):
        try:
            voltages = np.asarray(entry["voltages"], dtype=float)
            phases = np.asarray(entry["phases"], dtype=float)
            gradient = float(entry["gradient"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Experiment {index} needs 'voltages', 'phases' and a signed "
                f"'gradient' (pfg_params reads it from diff.xml)."
            ) from exc
        if gradient == 0:
            raise ValueError(
                f"Experiment {index} reports a zero gradient, so there is no "
                f"direction to correct for."
            )
        unfolded = unfold_polarity(voltages, phases, positive_first=positive_first)
        measured = _measure_ramp(voltages, phases, positive_first,
                                  max_voltage=max_voltage, exclude=exclude,
                                  balance_pairs=balance_pairs, reference=reference)
        prepared.append({
            "label": entry.get("label") or f"experiment {index + 1}",
            "gradient": gradient, "signed_voltages": unfolded["voltages"],
            # Sign-corrected onto the +g frame and centred on its own
            # common phase offset, which is the only part of the
            # offset that is not physics.
            # Against each run's own field-off measurement, not its
            # fitted offset: the receiver phase differs between two
            # acquisitions for no physical reason, and the measured
            # zero is the reference that does not move when the fit
            # does.
            "corrected": np.sign(gradient) * (phases - measured["reference"]),
            "slope": np.sign(gradient) * measured["slope"],
            "slope_error": measured["slope_error"],
            "r_squared": measured["r_squared"],
            "zero_crossing": measured["centre"], "reference": measured["reference"],
            "branches": measured["branches"], "voltages_in": voltages,
        })

    if len(prepared) < 2:
        raise ValueError(
            f"Averaging the gradient directions needs at least 2 experiments, "
            f"got {len(prepared)}."
        )

    shared_axis = prepared[0]["signed_voltages"]
    for entry in prepared[1:]:
        if (entry["signed_voltages"].shape != shared_axis.shape
                or not np.allclose(entry["signed_voltages"], shared_axis)):
            raise ValueError(
                f"{prepared[0]['label']} and {entry['label']} were measured at "
                f"different voltages, so their ramps cannot be averaged point "
                f"by point. Fit each and combine the slopes instead "
                f"(combine_experiments)."
            )

    signs = {float(np.sign(entry["gradient"])) for entry in prepared}
    # Two experiments run the same way round are a repeat, not the
    # reversed pair, and nothing that follows the gradient cancels
    # between them. Worth saying rather than quietly averaging.
    reversed_note = len(signs) >= 2

    corrected = np.vstack([entry["corrected"] for entry in prepared])
    average = corrected.mean(axis=0)
    # Measured the same way the inputs were. A single line through
    # every unfolded point is *not* that: when the two +-V arms
    # disagree they also have different intercepts, and one shared
    # intercept then drags the slope. On experiment 20, whose arms
    # differ by 93%, the single line gives -9.37e-3 rad/V against the
    # arms' -1.24e-2 -- a 25% error, from the estimator alone.
    measured = _measure_ramp(prepared[0]["voltages_in"], average, positive_first,
                              max_voltage=max_voltage, exclude=exclude,
                              balance_pairs=balance_pairs, reference=reference)
    fit = {"slope": measured["slope"], "intercept": measured["centre"],
           "r_squared": measured["r_squared"], "prediction": measured["prediction"]}

    slopes = np.array([entry["slope"] for entry in prepared])
    errors = np.array([entry["slope_error"] for entry in prepared])
    # Independent repeats: errors add in quadrature, the averaging
    # divides by n.
    combined_error = float(np.sqrt(np.sum(errors ** 2)) / slopes.size)
    spread = float(np.std(slopes, ddof=1)) if slopes.size > 1 else 0.0
    # Do the repeats differ by more than their own errors say they
    # should? Compared pairwise against the combined error of the two.
    pair_error = float(np.hypot(errors[0], errors[1])) if slopes.size == 2 else combined_error
    agree = bool(abs(slopes.max() - slopes.min()) <= tolerance * pair_error)

    return {
        "voltages": shared_axis, "experiments": prepared,
        "corrected": corrected, "average": average,
        "branches": measured["branches"],
        "fit": fit["prediction"],
        "slope": fit["slope"], "slope_error": combined_error,
        "intercept": fit["intercept"], "r_squared": fit["r_squared"],
        "slopes": slopes, "slope_errors": errors, "spread": spread,
        "spread_fraction": (spread / abs(float(np.mean(slopes)))
                             if np.mean(slopes) else float("inf")),
        "agree": agree, "gradients_reversed": reversed_note,
        "n": int(slopes.size), "tolerance": float(tolerance),
        # The same cutoff for every experiment: they share a voltage
        # axis, and fitting them over different ranges would make the
        # spread between them a range difference.
        "used": measured["used"], "excluded": ~measured["used"],
        "max_voltage": max_voltage,
    }


def format_gradient_pair_report(combined: dict, label: str | None = None,
                                 width: int = 66) -> str:
    """A readable summary of an :func:`average_gradient_pair` result.

    Leads with the two repeats rather than with their average, because
    whether averaging them was justified is the question.
    """
    rule = "=" * width
    lines = [rule, (label or "eNMR — GRADIENT-REVERSED PAIR").center(width).rstrip(),
             rule, ""]
    lines.append("  Experiment                g (T/m)   slope (rad/V)      R²")
    lines.append("  " + "-" * (width - 4))
    for entry in combined["experiments"]:
        lines.append(f"   {entry['label'][:22]:<22} {entry['gradient']:+8.3g}"
                      f"   {entry['slope']:+11.4g}  {entry['r_squared']:6.4f}")
    lines.append("  " + "-" * (width - 4))
    lines.append(f"   {'average':<22} {'':>8}   {combined['slope']:+11.4g}"
                  f"  {combined['r_squared']:6.4f}")
    lines.append("")
    lines.append(f"  slope = {combined['slope']:.5g} ± {combined['slope_error']:.2g} rad/V")
    lines.append(f"  spread across {combined['n']} repeats = {combined['spread']:.3g} rad/V"
                  f"  ({combined['spread_fraction'] * 100:.0f}%)")
    if combined.get("max_voltage") is not None:
        used = np.asarray(combined["used"], dtype=bool)
        lines.append(f"  fitted over {_range_text(combined['max_voltage'])} — "
                      f"{int(used.sum())} of {used.size} points, the same range "
                      f"for every repeat")
    lines.append("")
    if not combined["gradients_reversed"]:
        for chunk in _wrap(
            "NOTE: every experiment here was run with the gradient the same way "
            "round, so this is a repeat of one direction and not the reversed "
            "pair. Nothing that follows the gradient cancels.", width - 4):
            lines.append(f"  {chunk}")
        lines.append("")
    if combined["agree"]:
        message = (
            "The two directions agree to within their own errors, so what the "
            "gradient encoded reversed as it should and their average is a "
            "measurement.")
    else:
        message = (
            "The two directions DISAGREE by more than their errors allow, so "
            "their average is not a measurement of anything. Each ramp is kept "
            "on the plot beside the average for this reason: look at which one "
            "moved, and at its ±V arms (branch_slopes), before using either.")
    for chunk in _wrap(message, width - 4):
        lines.append(f"  {chunk}")
    lines.append(rule)
    return "\n".join(lines)


def _r_squared(y, prediction) -> float:
    """Fraction of the variance a prediction accounts for. Local so a
    fit built from two separate lines can be judged against all of its
    own points, which ``_line`` cannot do."""
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    total = float(np.sum((y - y.mean()) ** 2))
    residual = float(np.sum((y - prediction) ** 2))
    return 1.0 - residual / total if total > 0 else float("nan")


def _line(x, y) -> dict:
    """Least-squares straight line, with the slope's standard error and
    R^2."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3:
        raise ValueError(
            f"Need at least 3 points to fit a slope and judge it, got {x.size}."
        )
    design = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - (slope * x + intercept)
    dof = max(x.size - 2, 1)
    covariance = np.linalg.inv(design.T @ design) * float(np.sum(residual ** 2)) / dof
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope": float(slope), "intercept": float(intercept),
        "slope_error": float(np.sqrt(covariance[0, 0])),
        "r_squared": 1.0 - float(np.sum(residual ** 2)) / total if total > 0 else float("nan"),
        "voltages": x, "phases": y,
    }


def mobility_from_slope(slope: float, electrode_gap: float, gamma: float,
                         gradient: float, delta: float, big_delta: float,
                         cell_polarity: int = 1) -> float:
    """``mu`` in m^2/Vs from a phase-versus-voltage slope in rad/V.

    ``mu = slope * d / (gamma * g * delta * DELTA)``.

    Units, all SI and all easy to get wrong:

    * ``gamma`` -- **rad/s/T**. A vendor file typically records the
      gyromagnetic ratio as Hz/T (Bruker's ``diff.xml`` does, 4.2577e7
      for 1H), so multiply by ``2*pi`` first. Getting this wrong is a
      factor of 6.28 in the answer with nothing to show for it.
    * ``gradient`` -- T/m.
    * ``delta``, ``big_delta`` -- seconds.
    * ``cell_polarity`` -- ``+1`` when the cell's positive electrode
      drives ions along the gradient's **+z**, ``-1`` when it opposes
      it. Dimensionless, and the last thing in this calculation that
      no file records: ``acqus`` and ``diff.xml`` between them fix the
      gradient's sign and the vdlist's polarity order, but nothing
      records which way round the cell was wired.

      **Measured for the cell this module was validated against: -1.**
      With the vdlist's first-is-positive order and that cell's
      ``gmax = +0.3 T/m``, the TMABr standard gives a slope of
      -7.90e-3 rad/V -- the right magnitude, and a *negative* mobility
      for TMA+, which is a cation. The literal ``+1`` is the default
      because a compensating minus does not belong hidden in a
      formula; pass ``-1`` for this cell and TMA+ comes back at the
      accepted +3.79e-8.

    * ``electrode_gap`` -- metres. A property of the cell, which no
      vendor file records; it has no default here for that reason.
      ``big_delta`` is the drift time, which for the sequence this was
      validated against (``ENMR_DSTE``) is the recorded ``DELTA``
      itself -- confirmed by recovering the accepted TMA+ mobility
      with the cell's independently-known 38.5 mm gap, which rules out
      the factor of 2 a double stimulated echo might have introduced.
    """
    if electrode_gap is _NO_DEFAULT_GAP or electrode_gap is None:
        raise ValueError(
            "electrode_gap is required: a mobility scales directly with it, and "
            "no vendor file records the cell's electrode separation."
        )
    if electrode_gap <= 0:
        raise ValueError(f"electrode_gap must be positive, got {electrode_gap:g} m.")
    denominator = gamma * gradient * delta * big_delta
    if denominator == 0:
        raise ValueError(
            "gamma * gradient * delta * DELTA is zero – check the gradient "
            "strength and the diffusion timings."
        )
    if int(cell_polarity) not in (1, -1):
        raise ValueError(
            f"cell_polarity is +1 or -1 (which way the positive electrode faces "
            f"relative to the gradient), got {cell_polarity!r}."
        )
    return (int(cell_polarity) * float(slope) * float(electrode_gap)
            / float(denominator))


def fit_enmr(voltages, phases, *, electrode_gap=None, gamma: float | None = None,
              gradient: float | None = None, delta: float | None = None,
              big_delta: float | None = None, spectrum=None,
              pair_polarity: bool = True, positive_first: bool = True,
              max_voltage=None, exclude=None, cell_polarity: int = 1,
              balance_pairs: bool = False) -> dict:
    """Fit one experiment's phase ramp and report **the slope**.

    The slope in rad/V is the measurement: it is what the data
    determines, and it needs nothing beyond the data itself. Turning it
    into a mobility needs the cell's electrode gap and the sequence's
    gradient and timings -- numbers that live in a lab notebook, not in
    the file -- so that conversion is **optional** and is only done
    when all of them are supplied.

    Deliberately this way round. An earlier version made the mobility
    the headline and required a gap, which forces a number into the
    calculation whether or not the user has a trustworthy one, and
    quietly rescales everything if they do not.

    ``spectrum`` : read ``gamma``, ``gradient``, ``delta`` and
    ``big_delta`` from the experiment itself (see :func:`pfg_params`)
    instead of typing them. Only ``electrode_gap`` is then needed,
    since it is the one number no file records. Anything passed
    explicitly still wins, so a single value can be overridden without
    losing the other three.

    ``pair_polarity`` : collapse the +V/-V pairs first (see
    :func:`polarity_pairs`); ignored when the series is not paired.
    For the **gradient**-reversed repeat of the whole experiment, fit
    each separately and pass both to :func:`combine_experiments`.

    ``positive_first`` : the acquisition's polarity ordering, passed
    through to :func:`branch_slopes`.

    The result carries ``branches``: the same data with each polarity
    fitted separately (None when the series is not paired, or is too
    short to give three points per arm). **Check its ``agree``
    before using this slope.** A pair-averaged fit cancels whatever
    reversed with the field and says nothing at all about what did
    not, so its own R^2 cannot see a run where one polarity convected
    and the other did not -- which is what bubble formation does.
    """
    voltages = np.asarray(voltages, dtype=float)
    phases = np.asarray(phases, dtype=float)

    if spectrum is not None:
        read = pfg_params(spectrum)
        gamma = read["gamma"] if gamma is None else gamma
        gradient = read["gradient"] if gradient is None else gradient
        delta = read["delta"] if delta is None else delta
        big_delta = read["big_delta"] if big_delta is None else big_delta
    pairing = (polarity_pairs(voltages, phases, positive_first=positive_first)
               if pair_polarity
               else {"voltages": voltages, "dphi": phases, "paired": False})
    # One cutoff for the whole result: the headline slope and the arm
    # comparison beside it must be measured over the same points, or
    # the report contradicts itself.
    point_mask = _fit_mask(np.asarray(voltages, dtype=float), max_voltage, exclude,
                            paired=bool(pairing["paired"]), balance=balance_pairs)
    # A half-difference needs *both* members of its pair, so a pair
    # with one point removed cannot contribute one -- unlike the arm
    # fit, which uses each point on its own and keeps the survivor.
    # Removing single points is therefore a reason to read `branches`
    # rather than this slope, and the count says how much was lost.
    pair_mask = (point_mask.reshape(-1, 2).all(axis=1) if pairing["paired"]
                 else point_mask)
    broken = (int((point_mask.reshape(-1, 2).any(axis=1) & ~pair_mask).sum())
              if pairing["paired"] else 0)
    # Only speak up about the *range* when the range is what removed
    # them; a series too short to begin with gets _line's own message,
    # which says so.
    if pair_mask.sum() < 3 and not point_mask.all():
        raise ValueError(
            f"Only {int(pair_mask.sum())} point(s) survive the fit range, and a "
            f"slope needs at least 3. Raise the voltage cutoff."
        )
    fit = _line(pairing["voltages"][pair_mask], pairing["dphi"][pair_mask])

    # The same data read the other way round: each polarity fitted on
    # its own, which is what says whether averaging them was
    # justified. Attached rather than left to a second call so that
    # every caller of fit_enmr gets the check, including the GUI.
    # Only the pairs make two arms, and too short a series makes none
    # -- neither is an error in a fit that did succeed.
    branches = None
    if pairing["paired"]:
        try:
            branches = branch_slopes(voltages, phases, positive_first=positive_first,
                                      max_voltage=max_voltage, exclude=exclude,
                                      balance_pairs=balance_pairs)
        except ValueError:
            branches = None

    constants = {"electrode_gap": electrode_gap, "gamma": gamma, "gradient": gradient,
                 "delta": delta, "big_delta": big_delta}
    # Not in `constants`: it has a sensible literal default, so
    # requiring it alongside the others would block the slope-only path
    # for no reason.
    supplied = [name for name, value in constants.items() if value is not None]
    if supplied and len(supplied) < len(constants):
        missing = sorted(set(constants) - set(supplied))
        raise ValueError(
            f"A mobility needs all of electrode_gap, gamma, gradient, delta and "
            f"big_delta; missing {', '.join(missing)}. Leave every one out to get "
            f"the slope alone."
        )

    mobility = mobility_error = None
    if supplied:
        mobility = mobility_from_slope(
            fit["slope"], electrode_gap, gamma, gradient, delta, big_delta,
            cell_polarity=cell_polarity)
        # The mobility is the slope times a constant, so it carries the
        # slope's own relative error. The gap and the timings have
        # systematic uncertainties of their own, which this is not and
        # which the fit cannot see.
        mobility_error = (abs(mobility) * (fit["slope_error"] / abs(fit["slope"]))
                          if fit["slope"] else float("nan"))

    return {
        "slope": fit["slope"], "slope_error": fit["slope_error"],
        "intercept": fit["intercept"], "r_squared": fit["r_squared"],
        "voltages": fit["voltages"], "dphi": fit["phases"],
        "fit": fit["slope"] * fit["voltages"] + fit["intercept"],
        "all_voltages": pairing["voltages"], "all_dphi": pairing["dphi"],
        "used": pair_mask, "max_voltage": max_voltage, "broken_pairs": broken,
        "paired": bool(pairing["paired"]), "branches": branches,
        "mobility": mobility, "mobility_error": mobility_error,
        "cell_polarity": int(cell_polarity), **constants,
    }


#: Kept so existing callers keep working; :func:`fit_enmr` is the name
#: that says what it returns.
fit_mobility = fit_enmr


def format_enmr_report(result: dict, label: str | None = None, width: int = 66) -> str:
    """A readable summary of a :func:`fit_enmr` result.

    Leads with the slope, because that is what the data measured. The
    mobility appears only when the constants to compute it were given,
    and the electrode gap is named beside it -- it is the assumption the
    whole number rests on and is in no vendor file.
    """
    rule = "=" * width
    lines = [rule, (label or "eNMR PHASE RAMP").center(width).rstrip(), rule, ""]
    lines.append(f"  slope = {result['slope']:.5g} ± {result['slope_error']:.2g} rad/V"
                  f"   (R² = {result['r_squared']:.4f})")
    # Degrees as well: phase is read in degrees at the spectrometer,
    # and the conversion is the kind of thing that gets done wrong in a
    # notebook. Radians stay the primary unit -- the mobility formula
    # needs them, and a stray factor of 180/pi there is invisible.
    lines.append(f"          = {np.degrees(result['slope']):.5g}"
                  f" ± {np.degrees(result['slope_error']):.2g} °/V")
    lines.append("")
    lines.append("  Phase ramp")
    lines.append("  " + "-" * (width - 4))
    voltages = np.asarray(result["voltages"], dtype=float)
    lines.append(f"    points  = {voltages.size}"
                  f"{'  (+V/−V pairs, differenced)' if result['paired'] else '  (unpaired)'}")
    lines.append(f"    V range = {voltages.min():g} to {voltages.max():g} V")
    if result.get("max_voltage") is not None:
        used = np.asarray(result["used"], dtype=bool)
        lines.append(f"    fitted  = {_range_text(result['max_voltage'])}"
                      f"   ({int(used.sum())} of {used.size} pairs;"
                      f" {int((~used).sum())} excluded)")
    if not result["paired"]:
        lines.append("")
        for chunk in _wrap(
            "No +V/−V pairs found, so nothing cancels: receiver phase, drift and "
            "eddy currents are all still in this slope.", width - 6):
            lines.append(f"    {chunk}")
    lines.append("")

    if result.get("broken_pairs"):
        for chunk in _wrap(
            f"{result['broken_pairs']} pair(s) lost one point and so could not "
            f"be differenced – a half-difference needs both polarities. The "
            f"arm fit below still uses the survivors; prefer it when points "
            f"were removed singly.", width - 6):
            lines.append(f"    {chunk}")
        lines.append("")

    # Whether averaging the two polarities was justified belongs next
    # to the number that averaged them, not in a report nobody ran.
    branches = result.get("branches")
    if branches is not None:
        share = branches["asymmetry_fraction"] * 100
        lines.append("  ± polarity arms")
        lines.append("  " + "-" * (width - 4))
        lines.append(f"    +V: {branches['positive_branch']['slope']:+.4g} rad/V "
                      f"(R² = {branches['positive_branch']['r_squared']:.4f})")
        lines.append(f"    −V: {branches['negative_branch']['slope']:+.4g} rad/V "
                      f"(R² = {branches['negative_branch']['r_squared']:.4f})")
        lines.append(f"    did not reverse: {branches['asymmetry']:+.3g} rad/V "
                      f"({share:.0f}% of the signal)")
        lines.append("")
        for chunk in _wrap(
            "The arms agree within their errors, so averaging them is justified."
            if branches["agree"] else
            "WARNING: the arms disagree by more than their errors. Something here "
            "did not reverse with the field — convection from bubble formation "
            "runs one direction only. See format_branch_report.", width - 6):
            lines.append(f"    {chunk}")
        lines.append("")

    if result.get("mobility") is not None:
        lines.append("  Mobility")
        lines.append("  " + "-" * (width - 4))
        lines.append(f"    µ = {result['mobility']:.4g} ± {result['mobility_error']:.2g} m²/Vs")
        lines.append(f"    electrode gap = {result['electrode_gap'] * 1e3:g} mm"
                      f"   (µ scales directly with this)")
        lines.append(f"    cell polarity = {result.get('cell_polarity', 1):+d}"
                      f"   (the wiring; sets µ's sign)")
        lines.append(f"    gradient = {result['gradient']:g} T/m,  "
                      f"δ/Δ = {result['delta'] * 1e3:g}/{result['big_delta'] * 1e3:g} ms")
    else:
        lines.append("  Mobility not computed — supply electrode_gap, gamma,")
        lines.append("  gradient, delta and big_delta to convert the slope.")
    lines.append("")
    for chunk in _wrap(
        "This is one experiment. The gradient-reversed repeat is a separate "
        "acquisition; read both ramps and pass them to average_gradient_pair.",
            width - 4):
        lines.append(f"  {chunk}")
    lines.append(rule)
    return "\n".join(lines)


def format_combined_report(combined: dict, label: str | None = None, width: int = 66) -> str:
    """A readable summary of a :func:`combine_experiments` result.

    Puts ``spread`` next to the slope, because two runs of the same
    sample agreeing is the evidence that either of them measured
    anything.
    """
    rule = "=" * width
    lines = [rule, (label or "eNMR — COMBINED REPEATS").center(width).rstrip(), rule, ""]
    lines.append(f"  slope = {combined['slope']:.5g} ± {combined['slope_error']:.2g} rad/V")
    share = (combined["spread"] / abs(combined["slope"]) * 100
             if combined["slope"] else float("nan"))
    lines.append(f"  spread across {combined['n']} repeats = "
                  f"{combined['spread']:.3g} rad/V  ({share:.1f}%)")
    lines.append("")
    lines.append("  Repeats")
    lines.append("  " + "-" * (width - 4))
    for index, (slope, flipped) in enumerate(
            zip(combined["slopes"], combined["sign_flipped"]), start=1):
        note = "  (sign flipped to match the first)" if flipped else ""
        lines.append(f"    {index}: {slope:+.5g} rad/V{note}")
    lines.append("")
    for chunk in _wrap(
        "Reversing the gradient flips the sign of the fitted slope, so an "
        "opposing repeat is aligned before averaging. Two runs of one sample "
        "should agree to well within their own errors: a spread comparable to "
        "the slope means they disagree about more than noise, and their average "
        "is not a measurement.", width - 4):
        lines.append(f"  {chunk}")
    lines.append(rule)
    return "\n".join(lines)


def _range_text(max_voltage) -> str:
    """How a fit range reads in a report: a magnitude or two limits."""
    limits = np.atleast_1d(np.asarray(max_voltage, dtype=float))
    if limits.size == 1:
        return f"|V| ≤ {limits[0]:g} V"
    return f"{limits.min():g} ≤ V ≤ {limits.max():g} V"


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap, local so this report controls its own layout."""
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


#: Kept so existing callers keep working.
format_mobility_report = format_enmr_report
