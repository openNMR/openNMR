"""
Bruker digital-filter removal, without ``nmrglue``.

**NOT WIRED IN. Do not use this yet -- it does not meet the bar.**

Intended to replace ``nmrglue.bruker.remove_digital_filter``/
``rm_dig_filter``, and it is close, but two things are unresolved and
both must be settled before anything imports it:

1. **It is not byte-identical on real data**, only equal to within
   1 ULP (1.37e-16 relative, deterministic -- 0/20 runs identical).
   On *synthetic* data it matched exactly, across all 37 real
   ``(DECIM, DSPFVS, GRPDLY)`` combinations in ``test_data`` and all
   three modes (102 comparisons, 0 mismatches). Every intermediate
   step -- ``_ifft_positive``, the phase ramp, their product,
   ``_fft_positive`` -- is byte-identical to ``nmrglue``'s in
   isolation, and both implementations are self-stable and do not
   mutate their input, yet the composition differs by one bit. That
   was not isolated.
2. **Wiring it in broke 24 tests**, several with "acqus does not
   contain a DECIM parameter" -- even though ``nmrglue`` raises for a
   missing ``DECIM`` too, and the same tests pass against it. So the
   two are reached differently somehow, and that difference is not
   understood either.

A 1-ULP difference could not change any scientific conclusion. The
problem is that it was *unexplained*, and an unexplained discrepancy in
a filter is exactly the shape of the bug this module exists to avoid --
silent, plausible, and wrong. Kept here with its findings rather than
deleted, because the algorithm and the DSP table are the slow part and
they are done.
The algorithm is not obvious and the consequences of getting it subtly
wrong are silent -- a plausible spectrum with a wrong lineshape -- so
this is a deliberate, faithful reimplementation pinned by a test
asserting **byte-identical** output against ``nmrglue`` on every real
Bruker dataset in ``test_data``, not a fresh derivation.

What it does, and why. An oversampled Bruker acquisition applies a
digital filter whose group delay puts the FID's true start some
fractional number of points into the file. Removing it is a
*fractional* left shift, done in the frequency domain: FFT, apply a
first-order phase equal to that delay, inverse FFT. The leftover
integer part is then trimmed from the end, and the wrapped-around tail
is folded back onto the beginning.

The delay comes from ``GRPDLY`` when the file records one; otherwise it
is looked up by ``(DSPFVS, DECIM)`` in :data:`BRUKER_DSP_TABLE`.
``DSPFVS >= 14`` needs no correction at all.
"""

from __future__ import annotations

import numpy as np

#: Group delay by ``(DSPFVS, DECIM)``, for files predating ``GRPDLY``.
#: Reproduced from ``nmrglue``'s own table -- it is measured firmware
#: data, not something derivable.
BRUKER_DSP_TABLE = {
    10: {
        2: 44.75, 3: 33.5, 4: 66.625,
        6: 59.083333333333336, 8: 68.5625, 12: 60.375,
        16: 69.53125, 24: 61.020833333333336, 32: 70.015625,
        48: 61.34375, 64: 70.2578125, 96: 61.505208333333336,
        128: 70.37890625, 192: 61.5859375, 256: 70.439453125,
        384: 61.626302083333336, 512: 70.4697265625, 768: 61.646484375,
        1024: 70.48486328125, 1536: 61.656575520833336, 2048: 70.492431640625,
    },
    11: {
        2: 46.0, 3: 36.5, 4: 48.0,
        6: 50.166666666666664, 8: 53.25, 12: 69.5,
        16: 72.25, 24: 70.16666666666667, 32: 72.75,
        48: 70.5, 64: 73.0, 96: 70.66666666666667,
        128: 72.5, 192: 71.33333333333333, 256: 72.25,
        384: 71.66666666666667, 512: 72.125, 768: 71.83333333333333,
        1024: 72.0625, 1536: 71.91666666666667, 2048: 72.03125,
    },
    12: {
        2: 46.0, 3: 36.5, 4: 48.0,
        6: 50.166666666666664, 8: 53.25, 12: 69.5,
        16: 71.625, 24: 70.16666666666667, 32: 72.125,
        48: 70.5, 64: 72.375, 96: 70.66666666666667,
        128: 72.5, 192: 71.33333333333333, 256: 72.25,
        384: 71.66666666666667, 512: 72.125, 768: 71.83333333333333,
        1024: 72.0625, 1536: 71.91666666666667, 2048: 72.03125,
    },
    13: {
        2: 2.75, 3: 2.8333333333333335, 4: 2.875,
        6: 2.9166666666666665, 8: 2.9375, 12: 2.9583333333333335,
        16: 2.96875, 24: 2.9791666666666665, 32: 2.984375,
        48: 2.9895833333333335, 64: 2.9921875, 96: 2.9947916666666665,
    },
}


def _fft_positive(data: np.ndarray) -> np.ndarray:
    """FFT with a positive exponential, in NMR ordering.

    A positive exponential is an inverse FFT, so this undoes numpy's
    own 1/N scaling to keep amplitudes unchanged."""
    n = float(data.shape[-1])
    return np.fft.fftshift(np.fft.ifft(data, axis=-1).astype(data.dtype), -1) * n


def _ifft_positive(data: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_fft_positive`."""
    n = 1.0 / float(data.shape[-1])
    return np.fft.fft(np.fft.ifftshift(data, -1), axis=-1).astype(data.dtype) * n


def _frequency_shift(data: np.ndarray, points: float) -> np.ndarray:
    """Shift ``data`` by a *fractional* number of ``points`` along its
    last axis, via a first-order phase in the frequency domain -- the
    only way to shift by a non-integer amount without interpolating."""
    n = float(data.shape[-1])
    ramp = np.exp(2.0j * np.pi * points * np.arange(n) / n).astype(data.dtype)
    return _fft_positive(ramp * _ifft_positive(data))


def group_delay(decim: float, dspfvs: float, grpdly: float = 0.0) -> float:
    """The digital filter's group delay, in points.

    ``GRPDLY`` wins whenever the file records a usable one (> 0);
    otherwise the ``(DSPFVS, DECIM)`` table applies, and ``DSPFVS >=
    14`` means no correction is needed."""
    if grpdly > 0:
        return float(grpdly)
    if dspfvs >= 14:
        return 0.0
    if dspfvs not in BRUKER_DSP_TABLE:
        raise ValueError(f"DSPFVS {dspfvs} is not in the group-delay lookup table.")
    if decim not in BRUKER_DSP_TABLE[dspfvs]:
        raise ValueError(
            f"DECIM {decim} is not in the group-delay lookup table for DSPFVS {dspfvs}."
        )
    return float(BRUKER_DSP_TABLE[dspfvs][decim])


def rm_dig_filter(data: np.ndarray, decim: float, dspfvs: float, grpdly: float = 0.0,
                  truncate_grpdly: bool = True, post_proc: bool = False) -> np.ndarray:
    """Remove the digital filter from ``data``.

    ``truncate_grpdly`` floors the delay before applying it, which is
    what makes the result match TopSpin's own; the untruncated value
    keeps more of the FID but looks worse.

    ``post_proc=True`` treats ``data`` as *already* transformed and
    applies the equivalent first-order phase in the frequency domain
    instead, returning a corrected spectrum rather than a corrected
    FID."""
    phase = group_delay(decim, dspfvs, grpdly)
    if truncate_grpdly:
        phase = np.floor(phase)

    # Points to trim from the end (the integer part of the shift,
    # rounded up by two), and how many of the wrapped tail to fold back
    # onto the start. Below six there is nothing worth folding.
    skip = int(np.floor(phase + 2.0))
    add = int(max(skip - 6, 0))

    if post_proc:
        n = data.shape[-1]
        return (data * np.exp(2.0j * np.pi * phase * np.arange(n) / n)).astype(data.dtype)

    shifted = _frequency_shift(data, phase)
    # The shift wraps the FID's own end around to its start; fold that
    # back rather than discarding it.
    shifted[..., :add] = shifted[..., :add] + shifted[..., :-(add + 1):-1]
    return shifted[..., :-skip]


def remove_digital_filter(dic: dict, data: np.ndarray, truncate: bool = True,
                          post_proc: bool = False) -> np.ndarray:
    """:func:`rm_dig_filter`, taking its parameters from an ``acqus``
    dictionary -- ``nmrglue.bruker.remove_digital_filter``'s shape."""
    acqus = dic.get("acqus") if isinstance(dic, dict) else None
    if not acqus:
        raise ValueError("dictionary does not contain acqus parameters")
    if "DECIM" not in acqus:
        raise ValueError("acqus does not contain a DECIM parameter")
    if "DSPFVS" not in acqus:
        raise ValueError("acqus does not contain a DSPFVS parameter")
    return rm_dig_filter(
        data, acqus["DECIM"], acqus["DSPFVS"], acqus.get("GRPDLY", 0),
        truncate_grpdly=truncate, post_proc=post_proc,
    )
