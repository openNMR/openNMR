"""Phase every slice of a pseudo-2D on its own.

A pseudo-2D (a relaxation, diffusion or variable-temperature series;
one spectrum per row) is normally phased with one phase for all its
rows. That is right when the rows differ only in intensity, and wrong
when they do not -- a series whose phase drifts, or one whose rows were
each acquired with a different receiver phase. Explicit user request,
2026-10-20: "per-slice independent autophasing ... do not use the phase
of slice 1 to then apply to slice 2. This should be a separate option."

:func:`autophase_slices` finds each row's own phase and applies it to
that row alone. It is the function behind the Process tab's "Autophase
each slice independently" box, and behaves identically in a notebook::

    from opennmr import io, notebook
    spectrum = io.load("data/T1_series/3")
    found = notebook.autophase_slices(spectrum)            # zero-order, per slice
    found = notebook.autophase_slices(spectrum, joint=True)  # + first order where it helps
    found["phase0"]    # one angle (degrees) per slice
    spectrum.metadata.raw["slice_phase0"]   # recorded, so a reprocess keeps it

A row that is negative on purpose (an inversion-recovery slice) keeps its
sign when ``window=`` is given (see :func:`autophase_slices`); rows too noisy
to phase take their neighbours' phase.
"""

from __future__ import annotations

import numpy as np

from ..processing import ft as processing_ft

#: Where each row's own phase is recorded, on top of the dataset-wide one.
SLICE_PHASE_KEYS = ("slice_phase0", "slice_phase1")


def add_slice_phase(spectrum, row: int, delta0: float, delta1: float) -> None:
    """Fold a delta into row ``row``'s own recorded phase, growing the
    table to the dataset's row count on first use. Zero-order wrapped
    into [0, 360); first-order left unwrapped."""
    n_rows = spectrum.data.shape[0]
    for key, delta, wrap in zip(SLICE_PHASE_KEYS, (delta0, delta1), (True, False)):
        values = list(spectrum.metadata.raw.get(key) or ())
        values = (values + [0.0] * n_rows)[:n_rows]
        if 0 <= row < n_rows:
            total = float(values[row]) + float(delta)
            values[row] = total % 360.0 if wrap else total
        spectrum.metadata.raw[key] = values


def autophase_slices(spectrum, joint: bool = False, phase0_inc: float = 0.5,
                     pivot_ppm: float | None = None, rows=None, min_snr: float = 10.0,
                     window: float | None = None) -> dict:
    """Autophase each row of a pseudo-2D ``spectrum`` on its own, in place.

    Every row is phased from its own data, but not blindly:

    * A row with a signal-to-noise of at least ``min_snr`` is phased on its own,
      over the full circle, so a series whose phase drifts or jumps from row to
      row comes out absorptive throughout.
    * A row below ``min_snr`` has no reliable phase of its own (a late
      diffusion or relaxation slice is mostly noise): it takes the phase
      interpolated between its nearest good rows instead of one fitted to
      noise.
    * ``window`` : for a series whose rows are negative on purpose (inversion
      recovery), give a number of degrees. The strongest row then sets a
      reference phase, every other good row is searched within ``window`` of it
      ignoring the sign of its signal, so negative rows stay negative, and the
      series is given the sign that makes its summed signal positive. ``None``
      (the default) phases every row to be positive.

    ``joint`` : also fit a first-order phase per good row
    (:func:`opennmr.processing.ft.autophase01`, which keeps it only where it
    demonstrably improves the lineshape); default zero-order only.
    ``phase0_inc`` : zero-order grid step in degrees.
    ``pivot_ppm`` : the first-order pivot (default: the centre of the axis).
    ``rows`` : indices of the rows to phase (default: all).
    ``min_snr`` : the signal-to-noise a row needs to be phased from its own
    data.

    Each row's correction is measured against the row as it stands, so it
    adds to any phase the row already carries; it is also recorded in
    ``spectrum.metadata.raw["slice_phase0"/"slice_phase1"]`` so a reprocess or a
    saved state reproduces it.

    Returns ``{"phase0": [...], "phase1": [...], "snr": [...], "inherited":
    [...], "reference": row}`` -- for every row in ``rows`` (degrees), in
    order; ``reference`` is the index of the row that set the reference phase."""
    if spectrum.ndim != 2 or not spectrum.is_pseudo_2d:
        raise ValueError("autophase_slices needs a pseudo-2D (one spectrum per row); a "
                         "genuine 2D is phased along F2 and F1, not slice by slice.")
    if spectrum.data_imag is None:
        raise ValueError("autophase_slices needs the imaginary part of the spectrum.")
    freq_ppm = np.asarray(spectrum.axis_values(-1, "ppm"), dtype=float)
    span = float(freq_ppm.max() - freq_ppm.min())
    pivot_frac = (float(pivot_ppm) - float(freq_ppm.min())) / span if (pivot_ppm is not None and span) else 0.5
    complex_data = np.array(spectrum.complex_data, copy=True)
    selected = list(range(complex_data.shape[0])) if rows is None else [int(r) for r in rows]
    block = complex_data[selected]

    snr = np.array([processing_ft.row_snr(row) for row in block])
    good = snr >= float(min_snr)
    if not good.any():
        good[int(np.argmax(snr))] = True  # the best row has to anchor the rest
    ref_local = int(np.argmax(np.where(good, np.sum(np.abs(block), axis=1), -1.0)))
    ref0 = processing_ft.apc0(block[ref_local], freq_ppm, phase0_inc=phase0_inc)
    keep_sign = window is not None

    phase0 = np.zeros(len(selected))
    phase1 = np.zeros(len(selected))
    for k in np.flatnonzero(good):
        if window is None:
            phase0[k] = processing_ft.apc0(block[k], freq_ppm, phase0_inc=phase0_inc)
        elif k == ref_local:
            phase0[k] = ref0
        else:
            found = processing_ft.apc0(block[k], freq_ppm, phase0_inc=phase0_inc,
                                       penalty="signless")
            # Phase and phase + 180 score alike: take the one nearest the reference.
            options = np.array([found, found + 180.0, found - 180.0])
            phase0[k] = options[int(np.argmin(np.abs(options - ref0)))]
            if abs(phase0[k] - ref0) > window:
                phase0[k] = ref0 + np.sign(phase0[k] - ref0) * window
        if joint:
            rotated = processing_ft.apply_phase(block[k], freq_ppm, phase0[k], 0.0,
                                                pivot_frac=pivot_frac)
            _p0, p1 = processing_ft.autophase01(rotated, freq_ppm, phase0_range=(-30, 30),
                                                phase0_inc=max(phase0_inc, 1.0),
                                                pivot_frac=pivot_frac)
            phase0[k] += _p0
            phase1[k] = p1
    index = np.arange(len(selected), dtype=float)
    weak = ~good
    if weak.any():
        order = np.flatnonzero(good)
        unwrapped = np.unwrap(np.deg2rad(phase0[order]))
        phase0[weak] = np.rad2deg(np.interp(index[weak], index[order], unwrapped))
        phase1[weak] = np.interp(index[weak], index[order], phase1[order])

    # The sign of the whole series: positive summed signal.
    trial = np.array([processing_ft.apply_phase(block[k], freq_ppm, phase0[k], phase1[k],
                                                pivot_frac=pivot_frac)
                      for k in range(len(selected))])
    if keep_sign and np.sum(np.real(trial)) < 0:
        phase0 = phase0 + 180.0
    phase0 = ((phase0 + 180.0) % 360.0) - 180.0

    for k, row in enumerate(selected):
        complex_data[row] = processing_ft.apply_phase(
            complex_data[row], freq_ppm, phase0[k], phase1[k], pivot_frac=pivot_frac)
        add_slice_phase(spectrum, row, float(phase0[k]), float(phase1[k]))

    spectrum.data = np.real(complex_data)
    spectrum.data_imag = np.imag(complex_data)
    return {"phase0": [float(v) for v in phase0], "phase1": [float(v) for v in phase1],
            "snr": [float(v) for v in snr], "inherited": [bool(v) for v in weak],
            "reference": int(selected[ref_local])}
