"""
Arithmetic combination (addition/subtraction) of 1D NMR spectra.

Different experiments essentially never share an identical ppm axis
(different SW/SI, slightly different referencing), so spectra can't
just be subtracted element-by-element. This module interpolates every
input spectrum onto one common axis and then combines them -- the same
approach :func:`opennmr.processing.vocs.vocs_sum` uses for summing
VOCS components, extended here to signed combinations of an arbitrary
number of spectra.

Ported from ``nmrtools2``'s ``arithmetic.py``, adapted to operate on
:class:`NMRSpectrum` and to return one directly (folding in what was
previously a separate ``combine_spectra_as_results`` step).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.interpolate import interp1d

from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum


def combine_spectra(spectra: list[NMRSpectrum], ops: list[str] | None = None,
                     scaling_factors: list[float] | None = None, x_unit: str = "ppm",
                     common_range: str = "intersection", npoints: int | None = None,
                     interp_kind: str = "linear") -> NMRSpectrum:
    """Add and/or subtract an arbitrary number of 1D spectra on a
    common x-axis.

    ``ops`` : one sign per spectrum, e.g. ``['+', '-', '-']`` for
    ``spectra[0] - spectra[1] - spectra[2]``. Defaults to all ``'+'``
    (a sum); the first entry's sign is honoured, not forced positive.

    ``scaling_factors`` : one multiplier per spectrum, e.g. ``[1, 2,
    0.6]`` for ``1.0*spectra[0] +/- 2.0*spectra[1] +/- 0.6*spectra[2]``.

    ``x_unit`` : axis to interpolate on. ``'ppm'`` (default) is the
    physically sensible choice when spectra come from different
    spectrometer frequencies, since ppm is field-independent; ``'hz'``
    assumes all input spectra share (approximately) the same SF.

    ``common_range`` : ``'intersection'`` (default) only uses the
    x-range covered by every input spectrum -- recommended for
    difference spectra, since ``'union'`` would zero-fill each
    spectrum outside its own original range, manufacturing spurious
    "signal" exactly where a difference spectrum is meant to reveal
    real differences, not create them. ``'union'`` matches
    :func:`opennmr.processing.vocs.vocs_sum`'s convention instead.

    ``npoints`` : points in the common axis; if None, uses the largest
    number of points among the inputs. ``interp_kind`` : passed to
    ``scipy.interpolate.interp1d`` (``'linear'`` is safest for noisy
    data; higher orders can ring near sharp peaks).

    Returns a new :class:`NMRSpectrum` on the common axis, with the
    combination in ``data`` and, in ``metadata.raw``: ``"ops"`` (signs
    used) and ``"interpolated_spectra"`` (each input interpolated onto
    the common axis, shape ``(n_spectra, npoints)``, before signs were
    applied).

    Raises ``ValueError`` if fewer than 2 spectra are given, ``ops``/
    ``scaling_factors`` don't match the spectrum count, or (with
    ``common_range='intersection'``) the inputs' ranges don't overlap.
    """
    if len(spectra) < 2:
        raise ValueError("Need at least 2 spectra to combine.")

    if ops is None:
        ops = ["+"] * len(spectra)
    if len(ops) != len(spectra):
        raise ValueError("ops must have one entry per spectrum.")
    if any(op not in ("+", "-") for op in ops):
        raise ValueError("ops entries must be '+' or '-'.")

    if scaling_factors is None:
        scaling_factors = [1] * len(spectra)
    if len(scaling_factors) != len(spectra):
        raise ValueError("scaling_factors must have one entry per spectrum if applied")

    axes = [np.asarray(s.axis_values(0, x_unit), dtype=float) for s in spectra]
    values = [np.asarray(s.data, dtype=float) for s in spectra]

    sorted_axes, sorted_values = [], []
    for x, y in zip(axes, values):
        order = np.argsort(x)
        sorted_axes.append(x[order])
        sorted_values.append(y[order])

    if common_range == "intersection":
        lo = max(x[0] for x in sorted_axes)
        hi = min(x[-1] for x in sorted_axes)
        if lo >= hi:
            raise ValueError(
                "Input spectra's x-ranges do not overlap -- no common region "
                "to combine. Use common_range='union' if a zero-filled "
                "combination is genuinely what you want."
            )
    elif common_range == "union":
        lo = min(x[0] for x in sorted_axes)
        hi = max(x[-1] for x in sorted_axes)
    else:
        raise ValueError("common_range must be 'intersection' or 'union'.")

    if npoints is None:
        npoints = max(len(x) for x in sorted_axes)

    x_common = np.linspace(lo, hi, npoints)

    interpolated = []
    for x, y in zip(sorted_axes, sorted_values):
        f = interp1d(x, y, bounds_error=False, fill_value=0.0, kind=interp_kind)
        interpolated.append(f(x_common))
    interpolated = np.asarray(interpolated)

    signs = np.array([1.0 if op == "+" else -1.0 for op in ops])
    scales = np.asarray(scaling_factors, dtype=float)
    combined = np.sum(signs[:, None] * scales[:, None] * interpolated, axis=0)

    ref = spectra[0]
    if x_unit == "ppm":
        x_ppm = x_common
        x_hz = x_common * ref.metadata.sfo1 if ref.metadata.sfo1 else None
    else:
        x_hz = x_common
        x_ppm = x_common / ref.metadata.sfo1 if ref.metadata.sfo1 else None

    axis = NMRAxis(kind="frequency", ppm=x_ppm, hz=x_hz)
    metadata = dataclasses.replace(
        ref.metadata, source_path=None,
        raw={"ops": ops, "interpolated_spectra": interpolated},
    )
    return NMRSpectrum(data=combined, axes=[axis], metadata=metadata)
