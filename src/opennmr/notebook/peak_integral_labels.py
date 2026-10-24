"""
Integral labels under the peaks of a 1D spectrum.

The integral of each region is written under it as "∫ = 1.00", scaled so the
smallest one is 1 -- what a reader of a proton spectrum expects
("∫ = 1.00, ∫ = 2.03, ∫ = 3.05"). ``symbol=False`` writes the bare number.

In a notebook::

    from opennmr.notebook import integral_labels, add_integral_labels
    items = integral_labels(spectrum, [(7.1, 7.4), (3.6, 3.8)])
    fig, ax = plt.subplots(); ax.plot(ppm, spectrum.data); ax.invert_xaxis()
    add_integral_labels(ax, items)

    # The text is ordinary matplotlib text: size, colour, typeface, weight,
    # italics, height above the axis and rotation are keywords.
    add_integral_labels(ax, items, fontsize=9, color="tab:red", family="Times New Roman",
                        bold=True, italic=False, offset_axes=0.03, rotation=90)

In the GUI the same labels are switched on with Integrate ▸ Integration
Regions ▸ "Label integrals under the peaks", and stay on the plot.
"""

from __future__ import annotations

import numpy as np

#: Tag on the text artists, so they can be found and replaced.
GID = "opennmr:integral-label"

#: Written before each number: "∫ = 1.00".
INTEGRAL_SYMBOL = "\u222b"


def normalised_integrals(values) -> list[float]:
    """``values`` scaled so the smallest magnitude is 1.

    The sign is dropped (a label is a relative area), and zero or non-finite
    integrals stay 0 / NaN rather than dividing everything by nothing."""
    arr = np.abs(np.asarray(values, dtype=float))
    positive = arr[np.isfinite(arr) & (arr > 0)]
    if positive.size == 0:
        return [float("nan")] * arr.size
    return [float(v) / float(positive.min()) if np.isfinite(v) else float("nan") for v in arr]


def format_value(value: float, decimals: int = 2) -> str:
    if not np.isfinite(value):
        return "–"
    return f"{value:.{max(int(decimals), 0)}f}"


def label_text(value: float, decimals: int = 2, symbol=True) -> str:
    """One label: "∫ = 1.00". ``symbol``: True for the integral sign, a string for
    another symbol, False or "" for the bare number."""
    number = format_value(value, decimals)
    if symbol is True:
        symbol = INTEGRAL_SYMBOL
    return f"{symbol} = {number}" if symbol else number


def label_items(regions, integrals, decimals: int = 2, symbol=True) -> list[tuple[float, str]]:
    """``[(x of the region's centre, text), ...]`` from region limits and their integrals."""
    values = normalised_integrals(integrals)
    return [(0.5 * (float(lo) + float(hi)), label_text(v, decimals, symbol))
            for (lo, hi), v in zip(regions, values)]


def integral_labels(spectrum, regions, decimals: int = 2, method: str = "trapezoid",
                    subtract_baseline: bool = True, symbol=True) -> list[tuple[float, str]]:
    """Integrate ``regions`` (ppm ``(low, high)`` pairs) of a 1D ``spectrum`` and
    return the label items (see :func:`label_items`)."""
    from ..processing import relaxation as _relax

    if spectrum.ndim != 1:
        raise ValueError("Integral labels are for a 1D spectrum; take a row first.")
    regions = [(min(a, b), max(a, b)) for a, b in regions]
    if not regions:
        return []
    import dataclasses
    tagged = dataclasses.replace(spectrum, metadata=dataclasses.replace(
        spectrum.metadata, raw={**(spectrum.metadata.raw or {}), "slice": {
            "slice_type": "row", "index": 1, "shift": None, "shift_unit": None, "width": None}}))
    out = _relax.NMR2D_integrate_slices({"rows": [tagged]}, regions, mode="rows", method=method,
                                        subtract_baseline=subtract_baseline)
    return label_items(regions, out["rows"][0]["integrals"], decimals, symbol)


def _family_with_fallback(family):
    """A named typeface followed by DejaVu Sans, which has the integral sign
    (and the rest of the maths symbols) when the named one does not."""
    if not family:
        return None
    families = [family] if isinstance(family, str) else list(family)
    if "DejaVu Sans" not in families:
        families.append("DejaVu Sans")
    return families


def add_integral_labels(ax, items, fontsize: float | None = None, color: str = "black",
                        offset_axes: float = 0.015, family: str | None = None,
                        bold: bool = False, italic: bool = False, rotation: float = 0.0,
                        **text_kwargs) -> list:
    """Write ``items`` under their peaks on ``ax`` (replacing earlier labels).

    The text sits just above the bottom edge of the axes (``offset_axes`` is
    the height as a fraction of the axes), centred on each region, whatever
    the y range. ``fontsize``, ``color``, ``family`` (typeface), ``bold``,
    ``italic`` and ``rotation`` style it; any other matplotlib text keyword
    (``alpha``, ``bbox`` ...) passes through. Returns the text artists."""
    import matplotlib.transforms as mtransforms

    for artist in list(ax.texts):
        if artist.get_gid() == GID:
            artist.remove()
    transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    kwargs = dict(text_kwargs)
    fallback = _family_with_fallback(family)
    if fallback is not None:
        kwargs["family"] = fallback
    if bold:
        kwargs["fontweight"] = "bold"
    if italic:
        kwargs["fontstyle"] = "italic"
    if rotation:
        kwargs["rotation"] = float(rotation)
        kwargs.setdefault("rotation_mode", "anchor")
    artists = []
    for x, text in items:
        artist = ax.text(x, offset_axes, text, transform=transform, ha="center", va="bottom",
                         fontsize=fontsize, color=color, clip_on=True, **kwargs)
        artist.set_gid(GID)
        artists.append(artist)
    return artists


def remove_integral_labels(ax) -> None:
    for artist in list(ax.texts):
        if artist.get_gid() == GID:
            artist.remove()
