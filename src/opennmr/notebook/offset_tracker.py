"""Offset tracking: integrals of a series of 1D spectra against the
offset of a second channel (O2 or O3), e.g. 1D sideband-selective
HMQC or J-resolved data -- a 1D spectrum of X recorded while pulsing on
Y, repeated with Y's offset stepped across Y's lineshape (explicit user
request, 2026-10-17).

Each spectrum is integrated over the same region of the observed (X)
spectrum, and the integrals are plotted against the Y offset in ppm, as
bars, stems or triangles. The bars' skyline can be kept as a spectrum
of Y (:func:`skyline_spectrum`), which can then be fitted like any other
1D spectrum (CSA, quadrupolar, ...).

The offset in ppm is ``O2 / BF2`` (Hz / MHz) -- the offset from the
channel's basic frequency, which is 0 ppm of that nucleus when the
spectrometer's referencing for it is the standard one. Where the Y
reference differs, give ``ppm_shift`` (the ppm to add).

Notebook use::

    from opennmr.notebook import offset_tracker as ot
    series = ot.offset_series(spectra, region=(10, -10), channel="O2")
    ot.plot_offset_series(series, style="bars")
    y_spectrum = ot.skyline_spectrum(series)
"""
from __future__ import annotations

import os

import numpy as np

from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from . import intensity_scaling
from .dnp import region_integral

#: The channels a pulsed offset can be read from, and the acqus keys
#: that go with them: (offset, basic frequency, nucleus).
CHANNELS = {"O2": ("O2", "BF2", "NUC2"), "O3": ("O3", "BF3", "NUC3"),
            "O4": ("O4", "BF4", "NUC4")}

#: How the integrals are drawn.
STYLES = ("bars", "stems", "triangles")


def _acqus(spectrum: NMRSpectrum) -> dict:
    raw = getattr(spectrum.metadata, "raw", None) or {}
    acqus = raw.get("acqus")
    if isinstance(acqus, dict) and acqus:
        return acqus
    path = spectrum.metadata.source_path
    if path and os.path.isfile(os.path.join(path, "acqus")):
        from ..io.bruker import parse_jcampdx_params
        return parse_jcampdx_params(os.path.join(path, "acqus"))
    return {}


def channel_offset(spectrum: NMRSpectrum, channel: str = "O2") -> dict:
    """The offset of ``channel`` ('O2', 'O3' or 'O4') on ``spectrum``:
    ``{"hz", "ppm", "bf_MHz", "nucleus", "channel"}``. Raises ValueError
    when the spectrum does not record it."""
    channel = str(channel).upper()
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {sorted(CHANNELS)}, got {channel!r}")
    key_o, key_bf, key_nuc = CHANNELS[channel]
    acqus = _acqus(spectrum)
    try:
        hz = float(acqus[key_o])
        bf = float(acqus[key_bf])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"no {key_o}/{key_bf} recorded for this spectrum (Bruker acqus "
                         "needed; type the offsets instead)") from None
    if not bf > 0:
        raise ValueError(f"{key_bf} is {bf:g}: channel {channel} is not in use")
    nucleus = str(acqus.get(key_nuc, "") or "").strip("<> ") or None
    if nucleus and nucleus.lower() == "off":
        nucleus = None
    return {"hz": hz, "ppm": hz / bf, "bf_MHz": bf, "nucleus": nucleus, "channel": channel}


def offset_series(spectra, region, channel: str = "O2", offsets_ppm=None, labels=None,
                  baseline: str = "local", divide_rg: bool = True, ppm_shift: float = 0.0,
                  normalise: bool = False) -> dict:
    """Integral of ``region`` (lo, hi ppm of the observed nucleus) in each
    spectrum against the ``channel`` offset in ppm.

    ``offsets_ppm``: one per spectrum to use instead of reading
    ``channel`` (a ``None`` entry is still read). ``baseline``: as
    :func:`opennmr.notebook.dnp.region_integral`. ``divide_rg``: per scan
    and per unit receiver gain (where recorded), so spectra recorded
    with different gains compare. ``ppm_shift`` is added to every offset.
    ``normalise``: divide by the largest |integral|.

    Returns ``{"offset_ppm", "offset_hz", "y", "y_err", "labels",
    "nucleus", "bf_MHz", "channel", "region", "notes", "unit"}``, sorted
    by offset."""
    spectra = list(spectra)
    if not spectra:
        raise ValueError("No spectra given.")
    labels = list(labels) if labels is not None else [f"spectrum {i + 1}"
                                                      for i in range(len(spectra))]
    if offsets_ppm is None:
        offsets_ppm = [None] * len(spectra)
    offsets_ppm = list(offsets_ppm)
    if len(offsets_ppm) != len(spectra):
        raise ValueError(f"{len(spectra)} spectra but {len(offsets_ppm)} offsets.")
    lo, hi = sorted(float(v) for v in region)
    rows, notes = [], []
    nuclei, bfs = set(), set()
    for spectrum, label, given in zip(spectra, labels, offsets_ppm):
        if spectrum.ndim != 1:
            raise ValueError(f"{label} is not a 1D spectrum (keep a slice first).")
        bf = None
        if given is None:
            try:
                info = channel_offset(spectrum, channel)
            except ValueError as exc:
                raise ValueError(f"{label}: {exc}") from None
            ppm, hz, bf = info["ppm"], info["hz"], info["bf_MHz"]
            if info["nucleus"]:
                nuclei.add(info["nucleus"])
        else:
            ppm = float(given)
            try:
                info = channel_offset(spectrum, channel)
                bf = info["bf_MHz"]
                if info["nucleus"]:
                    nuclei.add(info["nucleus"])
            except ValueError:
                bf = None
            hz = ppm * bf if bf else float("nan")
        if bf:
            bfs.add(round(bf, 6))
        r = region_integral(spectrum, lo, hi, baseline=baseline)
        scale = intensity_scaling.comparison_factors(spectrum, "per_scan", divide_rg)
        notes.extend(f"{label}: {n}." for n in scale["notes"])
        rows.append({"label": label, "ppm": ppm + float(ppm_shift), "hz": hz,
                     "y": r["value"] * scale["factor"],
                     "err": abs(r["error"] * scale["factor"])})
    rows.sort(key=lambda r: r["ppm"])
    if len(nuclei) > 1:
        notes.append(f"The pulsed channel's nucleus differs across the series: "
                     f"{', '.join(sorted(nuclei))}.")
    if len(bfs) > 1:
        notes.append("The pulsed channel's basic frequency differs across the series.")
    y = np.array([r["y"] for r in rows], float)
    e = np.array([r["err"] for r in rows], float)
    unit = "integral per scan" + (" per unit RG" if divide_rg else "") + " (a.u.)"
    if normalise and y.size and np.max(np.abs(y)) > 0:
        m = float(np.max(np.abs(y)))
        y, e = y / m, e / m
        unit = "integral / largest integral"
    ppm = np.array([r["ppm"] for r in rows], float)
    if np.unique(ppm).size != ppm.size:
        notes.append("Some offsets repeat.")
    return {"offset_ppm": ppm, "offset_hz": np.array([r["hz"] for r in rows], float),
            "y": y, "y_err": e, "labels": [r["label"] for r in rows],
            "nucleus": next(iter(nuclei)) if len(nuclei) == 1 else None,
            "bf_MHz": next(iter(bfs)) if len(bfs) == 1 else None, "channel": str(channel).upper(),
            "region": (lo, hi), "notes": notes, "unit": unit, "baseline": baseline,
            "divide_rg": divide_rg}


def skyline_spectrum(series: dict, n_points: int = 2048, shape: str = "steps",
                     pad: float = 0.5, label: str | None = None) -> NMRSpectrum:
    """The bars' skyline as a 1D spectrum of the pulsed nucleus, on an
    even ppm grid of ``n_points`` from half a spacing below the lowest
    offset to half above the highest (``pad`` spacings either side).

    ``shape``: ``"steps"`` -- each offset's integral held across its bar
    (the skyline itself; the grid is the bars' outline), or ``"linear"``
    -- straight lines between the offsets. Outside the outermost bars
    the spectrum is 0. ``metadata``: the pulsed nucleus and its basic
    frequency (so Hz axes and fits in Hz work), vendor ``"derived"``.

    Fitting it is fitting a lineshape *sampled* at the offsets: the
    selectivity of the pulse broadens it, and points between the offsets
    carry no new information (``raw["skyline_offsets_ppm"]`` keeps
    them)."""
    x = np.asarray(series["offset_ppm"], float)
    y = np.asarray(series["y"], float)
    if x.size < 2:
        raise ValueError("A skyline needs at least two offsets.")
    order = np.argsort(x)
    x, y = x[order], y[order]
    spacing = float(np.min(np.diff(x))) if np.unique(x).size > 1 else 1.0
    if spacing <= 0:
        spacing = float(np.ptp(x)) / max(x.size - 1, 1) or 1.0
    lo = x[0] - pad * spacing
    hi = x[-1] + pad * spacing
    grid = np.linspace(hi, lo, int(n_points))           # descending, as NMR axes are
    if shape == "steps":
        edges = np.concatenate(([lo], 0.5 * (x[1:] + x[:-1]), [hi]))
        idx = np.clip(np.searchsorted(edges, grid, side="right") - 1, 0, x.size - 1)
        data = y[idx]
    elif shape == "linear":
        data = np.interp(grid, x, y, left=0.0, right=0.0)
    else:
        raise ValueError("shape must be 'steps' or 'linear'")
    bf = series.get("bf_MHz")
    axis = NMRAxis(kind="frequency", ppm=grid, hz=grid * bf if bf else None)
    meta = NMRMetadata(
        vendor="derived", nucleus=series.get("nucleus"), sfo1=bf, bf1=bf,
        source_path=label or "offset skyline",
        raw={"skyline_offsets_ppm": [float(v) for v in x],
             "skyline_integrals": [float(v) for v in y],
             "skyline_region": list(series.get("region") or ()),
             "skyline_shape": shape})
    return NMRSpectrum(data=data, axes=[axis], metadata=meta,
                       data_imag=np.zeros_like(data))


def plot_offset_series(series: dict, style: str = "bars", ax=None, colour=None,
                       width: float | None = None, show_skyline: bool = False,
                       markersize: float = 7.0, linewidth: float = 1.2, text_size="small"):
    """The integrals against the offset (ppm of the pulsed nucleus):
    ``style`` 'bars', 'stems' (vertical lines with a dot) or 'triangles'
    (vertical lines with a triangle). The x axis runs high to low ppm,
    as a spectrum of that nucleus does. ``show_skyline`` draws the bars'
    outline. Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    x = np.asarray(series["offset_ppm"], float)
    y = np.asarray(series["y"], float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
    else:
        fig = ax.figure
    colour = colour or "black"
    if width is None:
        width = 0.8 * (float(np.min(np.diff(np.sort(x)))) if np.unique(x).size > 1 else 1.0)
    if style == "bars":
        ax.bar(x, y, width=width, color=colour, edgecolor=colour, linewidth=0.5, zorder=2)
    else:
        ax.vlines(x, 0.0, y, color=colour, lw=linewidth, zorder=2)
        marker = "o" if style == "stems" else "^"
        ax.plot(x, y, marker, color=colour, ms=markersize if style == "triangles"
                else 0.7 * markersize, ls="none", zorder=3)
    if show_skyline and x.size > 1:
        sky = skyline_spectrum(series, n_points=4096)
        ax.plot(sky.axes[0].ppm, sky.data, "-", color="0.4", lw=0.8, zorder=1)
    ax.axhline(0.0, color="0.6", lw=0.6, zorder=0)
    nucleus = series.get("nucleus")
    ax.set_xlabel(f"{series.get('channel', 'O2')} offset"
                  + (f", {_pretty(nucleus)} shift" if nucleus else "") + " (ppm)")
    ax.set_ylabel(series.get("unit", "integral"))
    if x.size:
        span = float(np.ptp(x)) or 1.0
        ax.set_xlim(float(x.max()) + 0.05 * span + width, float(x.min()) - 0.05 * span - width)
    fig.tight_layout()
    return fig, ax


def _pretty(nucleus: str) -> str:
    """'13C' -> '$^{13}$C' for an axis label."""
    digits = "".join(ch for ch in nucleus if ch.isdigit())
    letters = "".join(ch for ch in nucleus if not ch.isdigit())
    return f"$^{{{digits}}}${letters}" if digits else nucleus


def offset_report(series: dict) -> str:
    """A plain-text table of the series, with its notes."""
    lines = [f"Region {series['region'][0]:g} to {series['region'][1]:g} ppm, "
             f"{series['channel']} offsets"
             + (f" ({series['nucleus']}, BF {series['bf_MHz']:g} MHz)" if series.get("nucleus")
                and series.get("bf_MHz") else "") + ":"]
    for lab, p, v, e in zip(series["labels"], series["offset_ppm"], series["y"], series["y_err"]):
        lines.append(f"  {p:10.4g} ppm   {v:.6g} ± {e:.2g}   {lab}")
    lines += series.get("notes") or []
    return "\n".join(lines)
