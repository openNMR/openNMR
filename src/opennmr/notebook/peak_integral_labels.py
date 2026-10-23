"""
Integral labels under the peaks of a 1D spectrum.

The integral of each region is written under it, scaled so the smallest one
is 1 -- what a reader of a proton spectrum expects ("1.00, 2.03, 3.05").

In a notebook::

    from opennmr.notebook import integral_labels, add_integral_labels
    items = integral_labels(spectrum, [(7.1, 7.4), (3.6, 3.8)])
    fig, ax = plt.subplots(); ax.plot(ppm, spectrum.data); ax.invert_xaxis()
    add_integral_labels(ax, items)

In the GUI the same labels are switched on with Integrate ▸ Integration
Regions ▸ "Label integrals under the peaks", and stay on the plot.
"""

from __future__ import annotations

import numpy as np

#: Tag on the text artists, so they can be found and replaced.
GID = "opennmr:integral-label"


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


def label_items(regions, integrals, decimals: int = 2) -> list[tuple[float, str]]:
    """``[(x of the region's centre, text), ...]`` from region limits and their integrals."""
    values = normalised_integrals(integrals)
    return [(0.5 * (float(lo) + float(hi)), format_value(v, decimals))
            for (lo, hi), v in zip(regions, values)]


def integral_labels(spectrum, regions, decimals: int = 2, method: str = "trapezoid",
                    subtract_baseline: bool = True) -> list[tuple[float, str]]:
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
    return label_items(regions, out["rows"][0]["integrals"], decimals)


def add_integral_labels(ax, items, fontsize: float | None = None, color: str = "black",
                        offset_axes: float = 0.015, family: str | None = None) -> list:
    """Write ``items`` under their peaks on ``ax`` (replacing earlier labels).

    The text sits just above the bottom edge of the axes, centred on each
    region, whatever the y range. Returns the text artists."""
    import matplotlib.transforms as mtransforms

    for artist in list(ax.texts):
        if artist.get_gid() == GID:
            artist.remove()
    transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    kwargs = {} if family is None else {"family": family}
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
