"""Multiply a spectrum by a constant -- and say what the constant was.

Used by the plot toolbar's × button (a new spectrum, the original untouched),
by Compare's per-spectrum factor (the drawing only; nothing stored) and by a
notebook::

    import opennmr.notebook as nmr
    big = nmr.scale_spectrum(spectrum, "1/3")       # a copy, a third as tall
    nmr.format_factor(0.3333333)                    # '1/3'
"""
from __future__ import annotations

import dataclasses
import math
import re
from fractions import Fraction

import numpy as np

__all__ = ["parse_factor", "format_factor", "scale_spectrum"]

_PREFIX = re.compile(r"^\s*[x×*]\s*", re.IGNORECASE)


def parse_factor(text) -> float:
    """A factor from what a person types: ``3``, ``-1.5``, ``2e3``, ``1/4``,
    ``x2`` or ``×2`` (a leading multiplication sign is ignored) and ``/4`` or
    ``÷4`` (divide by 4). Raises ``ValueError`` for anything else, for zero (a
    spectrum multiplied by zero is not a spectrum) and for inf/nan."""
    if isinstance(text, (int, float, np.integer, np.floating)):
        value = float(text)
    else:
        raw = str(text).strip().replace(",", ".")
        divide = raw[:1] in ("/", "÷")
        if divide:
            raw = raw[1:]
        raw = _PREFIX.sub("", raw)
        try:
            value = float(Fraction(raw.replace(" ", ""))) if "/" in raw else float(raw)
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"“{text}” is not a number. Type a factor such as 2, 0.5 or 1/3.") from None
        if divide:
            if value == 0:
                raise ValueError("Cannot divide by zero.")
            value = 1.0 / value
    if not math.isfinite(value):
        raise ValueError("The factor must be a finite number.")
    if value == 0:
        raise ValueError("The factor cannot be zero.")
    return value


def format_factor(value: float, max_denominator: int = 20) -> str:
    """The factor as a person writes it: an integer (``3``), a simple fraction
    (``1/3``, ``-3/2``) when it is one to within rounding, else a decimal
    (``0.125``, ``1.5e-05``). Plain ASCII, so it is safe in a legend."""
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    if abs(value - round(value)) <= 1e-9 * max(1.0, abs(value)):
        return str(int(round(value)))
    frac = Fraction(value).limit_denominator(max_denominator)
    if frac.denominator > 1 and abs(float(frac) - value) <= 1e-9 * abs(value):
        return f"{frac.numerator}/{frac.denominator}"
    text = f"{value:.4g}"
    return text


def scale_spectrum(spectrum, factor):
    """A copy of ``spectrum`` with ``data`` (and ``data_imag``) multiplied by
    ``factor`` (a number or what :func:`parse_factor` reads). The metadata is
    copied, with the running product recorded as ``raw["scale_factor"]``; the
    original is not touched."""
    value = parse_factor(factor)
    data = np.asarray(spectrum.data) * value
    data_imag = None if spectrum.data_imag is None else np.asarray(spectrum.data_imag) * value
    raw = dict(spectrum.metadata.raw or {})
    raw["scale_factor"] = float(raw.get("scale_factor", 1.0)) * value
    metadata = dataclasses.replace(spectrum.metadata, raw=raw)
    return dataclasses.replace(spectrum, data=data, data_imag=data_imag, metadata=metadata)
