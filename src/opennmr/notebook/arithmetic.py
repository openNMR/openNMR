"""
Arithmetic combination (addition/subtraction) of 1D NMR spectra for
nmrtools.

Different experiments essentially never share an identical ppm axis
(different SW/SI, slightly different referencing), so spectra can't
just be subtracted element-by-element. This module interpolates every
input spectrum onto one common axis and then combines them --
the same approach already used by :func:`nmrtools.one_d.VOCSSum` for
summing VOCS components, extended here to signed combinations of an
arbitrary number of spectra.
"""

import numpy as np
from scipy.interpolate import interp1d


def combine_spectra(spectra, ops=None, scaling_factors=None, x_unit='ppm', common_range='intersection',
                     npoints=None, interp_kind='linear'):
    """
    Add and/or subtract an arbitrary number of 1D spectra on a common
    x-axis.

    Parameters
    ----------
    spectra : list of dict
        Each a :func:`nmrtools.one_d.NMR1dimport` result (or anything
        with the same ``x_ppm``/``x_Hz``/``real_spectrum`` keys).
        At least 2 required.

    ops : list of {'+', '-'}, optional
        One sign per spectrum, e.g. ``['+', '-', '-']`` for
        ``spectra[0] - spectra[1] - spectra[2]``. If None, defaults to
        all ``'+'`` (i.e. a sum). The first entry's sign is honoured
        (so ``['-', '+']`` gives ``-spectra[0] + spectra[1]``) --
        it is not forced positive.
    
    scaling_factors : list of scaling factors, optional
        One scaling factor to multiply spectra by, e.g., ``[1,2,0.6]`` for
        ``1.0*spectra[0] +/- 2.0*spectra[1] +/- 0.6*spectra[2]``.

    x_unit : {'ppm', 'Hz'}, optional
        Axis to interpolate on. 'ppm' (default) is the physically
        sensible choice when spectra come from different spectrometer
        frequencies, since ppm is field-independent; 'Hz' assumes all
        input spectra share (approximately) the same SF.

    common_range : {'intersection', 'union'}, optional
        ``'intersection'`` (default): only the x-range covered by
        *every* input spectrum is used. Recommended for
        subtraction/difference spectra -- using the union and
        zero-filling outside each spectrum's original range (as
        :func:`nmrtools.one_d.VOCSSum` does for summing) would
        otherwise manufacture a spurious "signal" wherever only one
        spectrum has real data, which is exactly the kind of artefact
        a difference spectrum is meant to reveal, not create.
        ``'union'`` : the union of all ranges, matching
        :func:`nmrtools.one_d.VOCSSum`'s convention (zero-filled
        outside each original range).

    npoints : int, optional
        Number of points in the common axis. If None, uses the
        largest number of points among the input spectra (i.e. no
        coarser than the finest-resolution input).

    interp_kind : str, optional
        Passed to ``scipy.interpolate.interp1d`` (e.g. ``'linear'``,
        ``'cubic'``). Default ``'linear'`` -- safest for noisy
        spectral data; higher-order interpolation can ring near sharp
        peaks.

    Returns
    -------
    dict
        ``x_ppm`` or ``x_Hz``    : the common axis (key matches
            ``x_unit``).
        ``combined_spectrum``    : the signed combination.
        ``interpolated_spectra`` : each input spectrum interpolated
            onto the common axis, shape (n_spectra, npoints), *before*
            signs are applied.
        ``ops``                  : the signs actually used.

    Raises
    ------
    ValueError
        If fewer than 2 spectra are given, ``ops`` doesn't match the
        number of spectra, or (with ``common_range='intersection'``)
        the input spectra's ranges don't overlap at all.
    """

    if len(spectra) < 2:
        raise ValueError("Need at least 2 spectra to combine.")

    if ops is None:
        ops = ['+'] * len(spectra)
    if len(ops) != len(spectra):
        raise ValueError("ops must have one entry per spectrum.")
    if any(op not in ('+', '-') for op in ops):
        raise ValueError("ops entries must be '+' or '-'.")

    if scaling_factors is None:
        scaling_factors = [1] * len(spectra)
    if len(scaling_factors) != len(spectra):
        raise ValueError("scaling_factors must have one entry per spectrum if applied")

    key = 'x_ppm' if x_unit == 'ppm' else 'x_Hz'

    axes = [np.asarray(s[key], dtype=float) for s in spectra]
    values = [np.asarray(s['real_spectrum'], dtype=float) for s in spectra]

    sorted_axes, sorted_values = [], []
    for x, y in zip(axes, values):
        order = np.argsort(x)
        sorted_axes.append(x[order])
        sorted_values.append(y[order])

    if common_range == 'intersection':
        lo = max(x[0] for x in sorted_axes)
        hi = min(x[-1] for x in sorted_axes)
        if lo >= hi:
            raise ValueError(
                "Input spectra's x-ranges do not overlap – no common "
                "region to combine. Use common_range='union' if a "
                "zero-filled combination is genuinely what you want."
            )
    elif common_range == 'union':
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

    signs = np.array([1.0 if op == '+' else -1.0 for op in ops])
    scales = np.asarray(scaling_factors, dtype=float)
    combined = np.sum(signs[:, None] * scales[:, None] * interpolated, axis=0)

    return {
        key: x_common,
        'combined_spectrum': combined,
        'interpolated_spectra': interpolated,
        'ops': ops,
    }


def combine_spectra_as_results(spectra, ops=None, scaling_factors=None, x_unit='ppm',
                                common_range='intersection', npoints=None,
                                interp_kind='linear'):
    """
    Same as :func:`combine_spectra`, but returns a dict shaped like a
    :func:`nmrtools.one_d.NMR1dimport` result (``x_ppm``, ``x_Hz``,
    ``real_spectrum``, ``normalised_real_spectrum``), so the combined
    spectrum can be dropped directly into
    :func:`nmrtools.fitting.fit_spectrum`,
    :func:`nmrtools.fitting.NMR1dplot_with_fit`, etc.

    Note ``x_Hz`` here is derived from the common ppm axis using the
    *first* input spectrum's SFO1 -- an approximation only, since
    combined spectra from different experiments may have slightly
    different SFO1 values. If your spectra were acquired at
    meaningfully different fields, treat the returned ``x_Hz`` with
    caution (``x_ppm`` is unaffected and remains exact).
    """

    out = combine_spectra(spectra, ops=ops, scaling_factors=scaling_factors, 
                           x_unit=x_unit,
                           common_range=common_range, npoints=npoints,
                           interp_kind=interp_kind)

    if x_unit == 'ppm':
        x_ppm = out['x_ppm']
        sfo1 = spectra[0].get('SFO1', None)
        x_hz = x_ppm * sfo1 if sfo1 else None
    else:
        x_hz = out['x_Hz']
        sfo1 = spectra[0].get('SFO1', None)
        x_ppm = x_hz / sfo1 if sfo1 else None

    y = out['combined_spectrum']
    peak = np.max(np.abs(y))
    y_norm = y / peak if peak != 0 else y

    return {
        'x_ppm': x_ppm, 'x_Hz': x_hz,
        'real_spectrum': y, 'normalised_real_spectrum': y_norm,
        'nuclide': spectra[0].get('nuclide'),
        'BF1': spectra[0].get('BF1'), 'SFO1': sfo1,
        'ops': out['ops'], 'path': None,
    }
