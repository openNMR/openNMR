"""
Apodization windows, without ``nmrglue``.

Replaces the three ``nmrglue.process.proc_base`` window functions this
project uses -- ``em`` (exponential), ``gm`` (Gaussian) and ``sp``
(sine-bell). They are short, closed-form expressions, so these are
faithful transcriptions rather than reinterpretations: the parameter
names, the ``inv``/``rev`` flags and the ``.astype(data.dtype)``
placement are all kept, because changing where the cast happens is
enough to shift a result in its last bits.

Pinned by tests asserting **byte-identical** output against
``nmrglue`` over the parameter ranges this project actually passes.
"""

from __future__ import annotations

import numpy as np


def em(data: np.ndarray, lb: float = 0.0, inv: bool = False,
       rev: bool = False) -> np.ndarray:
    """Exponential apodization -- Bruker's ``WDW=1`` (EM), where ``lb``
    is the line broadening expressed as a fraction of the sweep width
    (``LB / SW``), not in Hz."""
    apod = np.exp(-np.pi * np.arange(data.shape[-1]) * lb).astype(data.dtype)
    if rev:
        apod = apod[::-1]
    if inv:
        apod = 1 / apod
    return apod * data


def gm(data: np.ndarray, g1: float = 0.0, g2: float = 0.0, g3: float = 0.0,
       inv: bool = False, rev: bool = False) -> np.ndarray:
    """Gaussian apodization -- ``g1`` inverse-exponential width, ``g2``
    Gaussian width, ``g3`` the position of the maximum as a fraction of
    the acquisition (0 = start)."""
    size = data.shape[-1]
    e = np.pi * np.arange(size) * g1
    g = 0.6 * np.pi * g2 * (g3 * (size - 1) - np.arange(size))
    apod = np.exp(e - g * g).astype(data.dtype)
    if rev:
        apod = apod[::-1]
    if inv:
        apod = 1 / apod
    return apod * data


def sp(data: np.ndarray, off: float = 0.0, end: float = 1.0, pow: float = 1.0,  # noqa: A002
       inv: bool = False, rev: bool = False) -> np.ndarray:
    """Sine-bell apodization -- Bruker's ``WDW=3`` (SINE, ``pow=1``)
    and ``WDW=4`` (QSINE, ``pow=2``). ``off`` is the starting point of
    the sine in half-cycles, ``SSB`` being ``1/off``.

    ``pow`` shadows the builtin, deliberately: it is ``nmrglue``'s own
    parameter name and this is a drop-in replacement."""
    size = data.shape[-1]
    apod = np.power(
        np.sin(np.pi * off + np.pi * (end - off) * np.arange(size) / (size - 1)).astype(data.dtype),
        pow,
    ).astype(data.dtype)
    if rev:
        apod = apod[::-1]
    if inv:
        apod = 1 / apod
    return apod * data
