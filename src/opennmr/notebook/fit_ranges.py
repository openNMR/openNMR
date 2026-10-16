"""
Several ppm ranges for a fit: "5 to 10, 12 to 30" -> [(5, 10), (12, 30)].

Explicit user request (2026-10-16): "enable users to only fit a model to
(multiple) selected regions of spectra - e.g., between 5 and 10 ppm, and
between 12 and 30 ppm (but not 10 to 12)". Every fitter here already
takes ``fit_regions=[(lo, hi), ...]`` (``combined_fitting``,
``quadrupolar_fitting``, ``czjzek``, ``jcoupling_fitting``,
``multi_fitting``); this is the one parser the GUI's "Several ranges"
boxes and a notebook share.

Accepted: pairs separated by ``,`` ``;`` or new lines; a pair written
``5 to 10``, ``5-10``, ``5 – 10``, ``5:10``, ``(5, 10)`` or ``5 10``;
either order; negative shifts (``-20 to -5``, ``-20--5``). Overlapping or
touching ranges are merged, and the result is sorted.
"""

from __future__ import annotations

import re

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _numbers(text: str) -> list[float]:
    t = str(text).replace("−", "-").replace("–", " to ").replace("—", " to ")
    t = t.replace("..", " to ").replace(":", " to ")
    # a hyphen written between two numbers is a range ("5-10", "5 - 10",
    # "-20--5"); one written against a number after a space is a sign
    t = re.sub(r"(?<=[\d.])-(?=[-+]?[\d.])", " to ", t)            # 5-10, -20--5
    t = re.sub(r"(?<=[\d.])\s+-\s+(?=[-+]?[\d.])", " to ", t)      # 5 - 10
    t = re.sub(r"ppm", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bto\b", " ", t)
    t = re.sub(r"[(),;\[\]\n\t]", " ", t)
    leftovers = re.sub(_NUMBER, " ", t).strip()
    if leftovers:
        raise ValueError(f"Could not read {leftovers!r} in '{text}': give ranges such as "
                         "'5 to 10, 12 to 30'.")
    return [float(v) for v in re.findall(_NUMBER, t)]


def merge_ranges(ranges) -> list[tuple[float, float]]:
    """Sorted (lo, hi) pairs with overlapping or touching ones joined."""
    pairs = sorted((min(a, b), max(a, b)) for a, b in ranges)
    merged: list[list[float]] = []
    for lo, hi in pairs:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(float(lo), float(hi)) for lo, hi in merged]


def parse_ppm_ranges(text: str) -> list[tuple[float, float]]:
    """Ranges typed as text -> sorted, merged ``[(lo, hi), ...]`` (ppm).
    Raises ValueError, saying what is wrong, for anything unreadable."""
    values = _numbers(text)
    if not values:
        raise ValueError("No ranges given: type them as e.g. '5 to 10, 12 to 30'.")
    if len(values) % 2:
        raise ValueError(f"'{text}' has an odd number of limits ({len(values)}): each range "
                         "needs a low and a high limit.")
    pairs = list(zip(values[0::2], values[1::2]))
    for a, b in pairs:
        if a == b:
            raise ValueError(f"The range {a:g} to {b:g} has no width.")
    return merge_ranges(pairs)


def format_ppm_ranges(ranges) -> str:
    """[(5, 10), (12, 30)] -> '5 to 10, 12 to 30'."""
    return ", ".join(f"{lo:g} to {hi:g}" for lo, hi in ranges)


def excluded_gaps(ranges) -> list[tuple[float, float]]:
    """The gaps between consecutive ranges (what the fit leaves out
    inside its overall span) -- shaded on the fit plot."""
    merged = merge_ranges(ranges)
    return [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]
