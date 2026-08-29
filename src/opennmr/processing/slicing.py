"""
Extract 1D traces (rows/columns) from a 2D :class:`NMRSpectrum`.

An extracted trace is just a 1D :class:`NMRSpectrum` -- the same class
used everywhere else -- carrying provenance (which row/column, at what
shift) in ``metadata.raw["slice"]`` rather than in a separate ad-hoc
dict shape. That unification is also what removes the need for
``nmrtools2``'s ``prepare_trace_for_fit`` adapter: a trace can be
handed straight to :func:`opennmr.processing.fitting.fit_spectrum`.

Ported from ``nmrtools2``'s ``NMR2d_get_trace``/``NMR2D_get_slices``
(``two_d.py``).
"""

from __future__ import annotations

import numpy as np

from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum


def get_trace(spectrum: NMRSpectrum, axis: str = "f1", index: int | None = None,
              shift: float | None = None, width: int = 1, combine: str = "sum",
              window: tuple[float, float] | None = None) -> NMRSpectrum:
    """Extract a single 1D trace (row or column) from a 2D
    :class:`NMRSpectrum`.

    ``axis`` : ``"f1"`` fixes a position along F1 (``spectrum.axes[0]``,
    a *row* of ``spectrum.data``) and returns a trace running along F2.
    ``"f2"`` fixes a position along F2 (a *column*) and returns a trace
    running along F1.

    ``index`` : 1-based position along the fixed axis (TopSpin
    convention). ``shift`` : chemical shift (ppm), converted to the
    nearest index -- not valid for the pseudo-2D F1 axis (use ``index``
    instead, since slice number has no chemical-shift meaning).

    ``width``/``combine`` : sum or average this many adjacent points,
    centred on the selected position, for extra S/N.

    ``window`` : restrict the returned trace to this ppm range (only
    applied when the trace's own axis is a chemical shift).

    The returned spectrum's ``metadata.raw["slice"]`` records
    ``slice_type`` (``"row"``/``"column"``), ``index``, ``shift``,
    ``shift_unit`` (``"ppm"`` or ``"index"``), and ``width``.
    """
    if spectrum.ndim != 2:
        raise ValueError("get_trace expects a 2D NMRSpectrum.")
    if axis not in ("f1", "f2"):
        raise ValueError("axis must be 'f1' or 'f2'")
    if index is None and shift is None:
        raise ValueError("Either index or shift must be provided.")

    data = spectrum.data
    pseudo2d = spectrum.is_pseudo_2d
    fixed_axis = spectrum.axes[0] if axis == "f1" else spectrum.axes[1]
    n_fixed = data.shape[0] if axis == "f1" else data.shape[1]

    if index is not None:
        pos = int(index) - 1
        if pos < 0 or pos >= n_fixed:
            raise ValueError(f"index {index} out of range (1..{n_fixed})")
    else:
        if axis == "f1" and pseudo2d:
            raise ValueError(
                "shift cannot be used to select an F1 position for pseudo-2D "
                "data; use index instead."
            )
        pos = int(np.argmin(np.abs(fixed_axis.values("ppm") - shift)))

    if axis == "f1" and pseudo2d:
        actual_shift = float(fixed_axis.index[pos])
        shift_unit = "index"
    else:
        actual_shift = float(fixed_axis.values("ppm")[pos])
        shift_unit = "ppm"
    actual_index = pos + 1

    half = width // 2
    lo = max(0, pos - half)
    hi = min(n_fixed, lo + width)
    lo = max(0, hi - width)

    data_imag = spectrum.data_imag

    if axis == "f1":
        block = data[lo:hi, :]
        y = block.sum(axis=0) if combine == "sum" else block.mean(axis=0)
        y_imag = None
        if data_imag is not None:
            block_imag = data_imag[lo:hi, :]
            y_imag = block_imag.sum(axis=0) if combine == "sum" else block_imag.mean(axis=0)
        trace_axis = spectrum.axes[1]
        slice_type = "row"
        nucleus = spectrum.metadata.nucleus
        sfo1 = spectrum.metadata.sfo1
    else:
        block = data[:, lo:hi]
        y = block.sum(axis=1) if combine == "sum" else block.mean(axis=1)
        y_imag = None
        if data_imag is not None:
            block_imag = data_imag[:, lo:hi]
            y_imag = block_imag.sum(axis=1) if combine == "sum" else block_imag.mean(axis=1)
        trace_axis = spectrum.axes[0]
        slice_type = "column"
        if spectrum.metadata.homonuclear:
            nucleus, sfo1 = spectrum.metadata.nucleus, spectrum.metadata.sfo1
        else:
            nucleus, sfo1 = spectrum.metadata.nucleus_indirect, spectrum.metadata.sfo1_indirect

    x_is_ppm = trace_axis.kind == "frequency"
    y = np.asarray(y)
    if y_imag is not None:
        y_imag = np.asarray(y_imag)

    if window is not None and x_is_ppm:
        lo_w, hi_w = sorted(window)
        ppm = trace_axis.values("ppm")
        mask = (ppm >= lo_w) & (ppm <= hi_w)
        new_axis = NMRAxis(
            kind="frequency", ppm=ppm[mask],
            hz=trace_axis.hz[mask] if trace_axis.hz is not None else None,
        )
        y = y[mask]
        if y_imag is not None:
            y_imag = y_imag[mask]
    else:
        new_axis = trace_axis

    metadata = NMRMetadata(
        vendor=spectrum.metadata.vendor, nucleus=nucleus, sfo1=sfo1,
        source_path=spectrum.metadata.source_path, procno=spectrum.metadata.procno,
        pseudo_2d=pseudo2d,
        raw={"slice": {
            "slice_type": slice_type, "index": actual_index, "shift": actual_shift,
            "shift_unit": shift_unit, "width": width,
        }},
    )
    return NMRSpectrum(data=y, axes=[new_axis], metadata=metadata, data_imag=y_imag)


def get_slices(spectrum: NMRSpectrum, rows: list[int] | None = None, row_shifts: list[float] | None = None,
                row_width: int = 1, columns: list[int] | None = None, column_shifts: list[float] | None = None,
                column_width: int = 1, combine: str = "sum", window: tuple[float, float] | None = None,
                normalise: bool = False) -> dict:
    """Extract multiple row and/or column traces from a 2D
    :class:`NMRSpectrum`. A thin batch wrapper around :func:`get_trace`.

    ``rows``/``row_shifts`` : 1-based F1 indices / F1 chemical shifts
    (ppm) of rows to extract (may be combined; duplicates extracted
    once). ``columns``/``column_shifts`` : likewise for F2.

    ``normalise`` : if True, scale every extracted trace to its own
    maximum absolute intensity.

    Returns a dict with keys ``"rows"`` and ``"columns"``, each a list
    of :class:`NMRSpectrum` traces sorted by position along the fixed
    axis.
    """
    slices = {"rows": [], "columns": []}

    def _collect(axis, indices, shifts, width):
        out = []
        seen_index = set()

        for idx in (indices or []):
            if idx in seen_index:
                continue
            seen_index.add(idx)
            out.append(get_trace(spectrum, axis=axis, index=idx, width=width, combine=combine, window=window))

        for sh in (shifts or []):
            trace = get_trace(spectrum, axis=axis, shift=sh, width=width, combine=combine, window=window)
            trace_index = trace.metadata.raw["slice"]["index"]
            if trace_index in seen_index:
                continue
            seen_index.add(trace_index)
            out.append(trace)

        out.sort(key=lambda t: t.metadata.raw["slice"]["index"])
        return out

    if rows is not None or row_shifts is not None:
        slices["rows"] = _collect("f1", rows, row_shifts, row_width)
    if columns is not None or column_shifts is not None:
        slices["columns"] = _collect("f2", columns, column_shifts, column_width)

    if normalise:
        for group in slices.values():
            for tr in group:
                m = np.max(np.abs(tr.data))
                if m > 0:
                    tr.data = tr.data / m

    return slices
