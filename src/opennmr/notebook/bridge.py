"""
The seam between the live application's :class:`~opennmr.core.spectrum.NMRSpectrum`
and the notebook API's results dict.

Every entry point in this package -- :func:`..csa_fitting.fit_csa_spectrum`,
the pNMR fit, the BMS and SIMPSON importers -- takes the ten-key
dictionary :func:`..one_d.NMR1dimport` produces, because they were all
written for a notebook workflow that starts by importing from disk.
Nothing in ``src/opennmr`` outside this package produced one, so a
spectrum already loaded in the GUI could not be handed to any of them
without re-reading it from disk in a different format.

That is the whole purpose here: convert, rather than re-read. The
returned dict is the same shape ``NMR1dimport`` returns and is accepted
anywhere one of those is::

    from opennmr.notebook.bridge import results_from_spectrum
    from opennmr.notebook.csa_fitting import fit_csa_spectrum

    popt, pcov, is_free, scale, info = fit_csa_spectrum(
        results_from_spectrum(spectrum), p0=[{...}],
    )

Deliberately one function in its own module rather than a method on
``NMRSpectrum``: the core object should not know about a legacy dict
format, and the notebook package should not import the GUI.
"""

from __future__ import annotations

import numpy as np

__all__ = ["results_from_spectrum"]

#: The keys ``NMR1dimport`` returns -- see :func:`..one_d.NMR1dimport`.
#: Named here so a caller can assert the contract rather than discover
#: it by ``KeyError`` three frames deep inside a fit.
RESULTS_KEYS = (
    "x_ppm", "x_Hz",
    "real_spectrum", "normalised_real_spectrum",
    "imag_spectrum", "normalised_imag_spectrum",
    "nuclide", "BF1", "SFO1", "path",
)


def _normalise(values):
    """``values`` scaled to a maximum absolute value of 1, or left
    alone when it is all zeros (a derived or not-yet-processed
    spectrum) rather than dividing by zero."""
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    return values / peak if peak else values


def results_from_spectrum(spectrum, row: int | None = None) -> dict:
    """``spectrum`` as the notebook API's results dict.

    Parameters
    ----------
    spectrum : NMRSpectrum
        A 1D spectrum, or a 2D/pseudo-2D one together with ``row``.
    row : int, optional
        Which row of a 2D to convert, zero-based. Required for a 2D:
        there is no single "the" spectrum in a 2D dataset, and picking
        one silently is how a fit ends up describing a slice nobody
        chose. A 1D spectrum ignores it.

    Returns
    -------
    dict
        The keys in :data:`RESULTS_KEYS`. ``imag_spectrum`` is zeros
        when the spectrum carries no imaginary part, which keeps the
        dict's shape fixed -- every consumer of it reads
        ``real_spectrum``, and the ones that touch the imaginary part
        check it for content first.

    Notes
    -----
    The frequency axis is taken as-is, **not** re-sorted. Both this
    application and ``NMR1dimport`` hold ppm descending (high shift
    first), and the notebook's fitting code sorts ascending itself
    where it needs to -- see ``csa_fitting.fit_csa_spectrum``'s own
    ``x_ppm_sorted``. Sorting here as well would be a second, silent
    convention for the same data.
    """
    data = np.asarray(spectrum.data)
    if data.ndim == 2:
        if row is None:
            raise ValueError(
                "results_from_spectrum needs a row for a 2D spectrum -- "
                "pass row=<zero-based index> to choose the slice to convert."
            )
        if not 0 <= row < data.shape[0]:
            raise IndexError(f"row {row} is outside the spectrum's {data.shape[0]} rows.")
        real = np.asarray(data[row], dtype=float)
        imaginary = spectrum.data_imag
        imaginary = (np.asarray(imaginary[row], dtype=float)
                     if imaginary is not None else np.zeros_like(real))
        axis = spectrum.axes[-1]
    elif data.ndim == 1:
        real = np.asarray(data, dtype=float)
        imaginary = (np.asarray(spectrum.data_imag, dtype=float)
                     if spectrum.data_imag is not None else np.zeros_like(real))
        axis = spectrum.axes[0]
    else:
        raise ValueError(f"results_from_spectrum needs a 1D or 2D spectrum, got {data.ndim}D.")

    if axis.kind != "frequency" or axis.ppm is None:
        raise ValueError(
            "results_from_spectrum needs a frequency axis in ppm -- this "
            f"spectrum's is {axis.kind!r}. A pseudo-2D's F1 is a slice "
            "index, so convert along F2 instead."
        )

    metadata = spectrum.metadata
    sfo1 = float(metadata.sfo1) if metadata.sfo1 else 1.0
    x_ppm = np.asarray(axis.ppm, dtype=float)
    # Hz from ppm and the transmitter frequency, rather than from
    # axis.hz: an axis carries hz only when its reader filled it in,
    # and every consumer of this dict expects both to be present.
    x_hz = np.asarray(axis.hz, dtype=float) if axis.hz is not None else x_ppm * sfo1

    return {
        "x_ppm": x_ppm, "x_Hz": x_hz,
        "real_spectrum": real, "normalised_real_spectrum": _normalise(real),
        "imag_spectrum": imaginary, "normalised_imag_spectrum": _normalise(imaginary),
        "nuclide": metadata.nucleus, "BF1": sfo1, "SFO1": sfo1,
        "path": metadata.source_path,
    }
