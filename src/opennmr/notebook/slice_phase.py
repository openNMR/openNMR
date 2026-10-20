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

**Do not use it on an inversion-recovery series**: its rows change sign
on purpose, and an independent phase per row would turn the negative
ones positive and destroy the relaxation curve. Phase one slice and
apply it to all of them instead.
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
                     pivot_ppm: float | None = None, rows=None) -> dict:
    """Autophase each row of a pseudo-2D ``spectrum`` independently, in
    place.

    ``joint`` : also fit a first-order phase per row
    (:func:`opennmr.processing.ft.autophase01`, which keeps it only where
    it demonstrably improves the lineshape); default zero-order only
    (:func:`~opennmr.processing.ft.apc0`).
    ``phase0_inc`` : the zero-order grid step in degrees (0.5 is quick
    across a long series).
    ``pivot_ppm`` : the first-order pivot (default: the centre of the
    axis).
    ``rows`` : indices of the rows to phase (default: all).

    Each row's correction is measured against the row as it stands, so it
    adds to any phase the row already carries; it is also recorded in
    ``spectrum.metadata.raw["slice_phase0"/"slice_phase1"]`` so a
    reprocess or a saved state reproduces it. No row's phase is ever
    taken from another.

    Returns ``{"phase0": [...], "phase1": [...]}`` -- the correction found
    for every row in ``rows`` (degrees), in order."""
    if spectrum.ndim != 2 or not spectrum.is_pseudo_2d:
        raise ValueError("autophase_slices needs a pseudo-2D (one spectrum per row); a "
                         "genuine 2D is phased along F2 and F1, not slice by slice.")
    if spectrum.data_imag is None:
        raise ValueError("autophase_slices needs the imaginary part of the spectrum.")
    freq_ppm = np.asarray(spectrum.axis_values(-1, "ppm"), dtype=float)
    span = float(freq_ppm.max() - freq_ppm.min())
    pivot_frac = (float(pivot_ppm) - float(freq_ppm.min())) / span if (pivot_ppm is not None and span) else 0.5
    complex_data = np.array(spectrum.complex_data, copy=True)
    selected = range(complex_data.shape[0]) if rows is None else [int(r) for r in rows]

    found0, found1 = [], []
    for row in selected:
        if joint:
            phase0, phase1 = processing_ft.autophase01(
                complex_data[row], freq_ppm, phase0_inc=max(phase0_inc, 1.0), pivot_frac=pivot_frac)
        else:
            phase0 = processing_ft.apc0(complex_data[row], freq_ppm, phase0_inc=phase0_inc)
            phase1 = 0.0
        found0.append(float(phase0))
        found1.append(float(phase1))
        complex_data[row] = processing_ft.apply_phase(
            complex_data[row], freq_ppm, phase0, phase1, pivot_frac=pivot_frac)
        add_slice_phase(spectrum, row, phase0, phase1)

    spectrum.data = np.real(complex_data)
    spectrum.data_imag = np.imag(complex_data)
    return {"phase0": found0, "phase1": found1}
