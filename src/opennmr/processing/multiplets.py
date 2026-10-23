"""First-order multiplets for the isotropic peak fit.

A peak coupled to ``n`` equivalent spin-1/2 neighbours by ``J`` (Hz) is
``n + 1`` lines, ``J`` apart, with binomial intensities (a doublet 1:1, a
triplet 1:2:1, a quartet 1:3:3:1 ...). Several different couplings combine
(a doublet of triplets is ``dt``): every line of one splitting is split by the
next. The total area of the multiplet is that of one line -- the fitted
amplitude and width describe the multiplet, not a single line.

A peak definition asks for one with ``"couplings": [1, 2]`` (a ``dt``: one
neighbour, then two) and, optionally, ``"J_init": [7.5, 1.8]`` in Hz. Each
``J`` is then a fitted parameter named ``J1``, ``J2`` ..., free unless
``"fixed": {"J1": True}``, bounded by ``"bounds": {"J1": (lo, hi)}`` (Hz).
``"j_scale"`` is the size of 1 Hz in the axis units (``1 / spectrometer MHz``
on a ppm axis, 1 on a Hz axis); the fit fills it in when it is missing.

This is the first-order picture: it is exact when the shift difference is
much larger than J. Strongly coupled spins need the exact spin-system fit
(Solid-state NMR / Solid-state NMR, "Strongly coupled peaks").
"""

from __future__ import annotations

import re
from math import comb

import numpy as np

#: A starting J when none is given, Hz.
DEFAULT_J_HZ = 7.0
#: Upper bound on a fitted J, Hz.
MAX_J_HZ = 1000.0

#: Letters of a multiplet name -> number of equivalent spin-1/2 neighbours.
NAMES = {"s": 0, "d": 1, "t": 2, "q": 3, "p": 4, "x": 5, "h": 6}
_WORDS = (("sept", 6), ("hept", 6), ("sext", 5), ("quint", 4), ("sing", 0), ("doub", 1),
          ("trip", 2), ("quar", 3), ("pent", 4), ("hex", 5))
_LETTER_NAMES = {0: "s", 1: "d", 2: "t", 3: "q", 4: "p", 5: "x", 6: "h"}


def parse_multiplet(text) -> list[int]:
    """``"dt"`` -> ``[1, 2]``; ``""``, ``"s"`` or ``"singlet"`` -> ``[]``.

    Letters: s d t q p (quintet) x (sextet) h (septet). Words such as
    "doublet", "triplet", "quartet", "quintet", "sextet" and "septet" also work,
    and ``n=4`` means four equivalent neighbours. Raises ``ValueError`` for
    anything else."""
    if text is None:
        return []
    raw = str(text).strip().lower()
    if raw in ("", "s", "singlet", "-", "none", "m"):
        return []
    counts: list[int] = []
    rest = raw.replace(" ", "").replace(",", "")
    while rest:
        match = re.match(r"n=(\d+)", rest)
        if match:
            counts.append(int(match.group(1)))
            rest = rest[match.end():]
            continue
        for word, n in _WORDS:
            if rest.startswith(word):
                rest = re.sub(r"^(let|tet|et)", "", rest[len(word):])
                if n:
                    counts.append(n)
                break
        else:
            if rest[0] in NAMES:
                n = NAMES[rest[0]]
                if n:
                    counts.append(n)
                rest = rest[1:]
            else:
                raise ValueError(f"Cannot read the multiplet {text!r}: use letters s, d, t, q, p "
                                 "(quintet), x (sextet), h (septet) – e.g. 'dt' for a doublet of "
                                 "triplets – or n=4 for four equivalent neighbours.")
    return [c for c in counts if c > 0]


def multiplet_name(counts) -> str:
    """The inverse of :func:`parse_multiplet`: ``[1, 2]`` -> ``"dt"``; counts
    with no letter (more than six neighbours) are written ``n=7``."""
    if not counts:
        return "s"
    return "".join(_LETTER_NAMES.get(int(n), f"n={int(n)}") for n in counts)


def parse_couplings(text, n_couplings: int) -> list[float]:
    """The J values (Hz) typed for a multiplet: numbers separated by commas or
    spaces. Fewer than ``n_couplings`` are padded with the default; extras are
    refused."""
    parts = [p for p in re.split(r"[,\s;]+", str(text or "").strip()) if p]
    try:
        values = [abs(float(p)) for p in parts]
    except ValueError as exc:
        raise ValueError(f"J values must be numbers in Hz, separated by commas; got {text!r}.") from exc
    if len(values) > n_couplings:
        raise ValueError(f"{len(values)} J values for a multiplet with {n_couplings} coupling(s).")
    return values + [DEFAULT_J_HZ] * (n_couplings - len(values))


def lines(counts, js_hz):
    """``(offsets_hz, weights)`` of every line of the multiplet, weights
    summing to 1. Offsets are from the multiplet centre."""
    offsets = np.zeros(1)
    weights = np.ones(1)
    for n, j in zip(counts, js_hz):
        step = np.arange(n + 1) - n / 2.0
        w = np.array([comb(n, k) for k in range(n + 1)], dtype=float)
        w /= w.sum()
        offsets = (offsets[:, None] + float(j) * step[None, :]).ravel()
        weights = (weights[:, None] * w[None, :]).ravel()
    return offsets, weights


def apply(single, x0, counts, js_hz, j_scale: float = 1.0):
    """The multiplet curve. ``single(centre)`` is the peak's lineshape (with its
    full amplitude) centred at ``centre``; the result is the weighted sum over
    the multiplet's lines. With no couplings it is just ``single(x0)``."""
    if not counts:
        return single(x0)
    offsets, weights = lines(counts, js_hz)
    total = None
    for offset, weight in zip(offsets, weights):
        part = weight * single(x0 + offset * j_scale)
        total = part if total is None else total + part
    return total


def j_scale_for(sfo1_mhz, x_unit: str) -> float:
    """Size of 1 Hz in the axis units."""
    if str(x_unit).lower() == "ppm" and sfo1_mhz:
        return 1.0 / float(sfo1_mhz)
    if str(x_unit).lower() == "khz":
        return 1e-3
    return 1.0


def ensure_j_scale(peak_defs, sfo1_mhz, x_unit: str) -> None:
    """Fill ``j_scale`` in, in place, on every peak that has couplings but no scale."""
    for peak in peak_defs:
        if peak.get("couplings") and not peak.get("j_scale"):
            peak["j_scale"] = j_scale_for(sfo1_mhz, x_unit)
