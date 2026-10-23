"""Offset tracking: integrals of a series of 1D spectra against the
offset of a second channel (O2 or O3), e.g. 1D sideband-selective
HMQC or J-resolved data -- a 1D spectrum of X recorded while pulsing on
Y, repeated with Y's offset stepped across Y's lineshape (explicit user
request, 2026-10-17).

Each spectrum is integrated over the same region of the observed (X)
spectrum, and the integrals are plotted against the Y offset in ppm, as
bars, vertical lines ("stems") or a lineshape -- a Gaussian, Lorentzian
or pseudo-Voigt of one chosen width centred on every offset, its height
the integral there (2026-10-18; it replaced the triangles). With the
offsets placed on Y's MAS sidebands, that lineshape is a picture of Y's
MAS spectrum: one peak per sideband (or the centreband). Whatever is
drawn can be kept as a spectrum of Y (:func:`pattern_spectrum`:
:func:`skyline_spectrum` for bars, :func:`stick_spectrum` for vertical
lines, :func:`sideband_spectrum` for the lineshape), which can then be
fitted like any other 1D spectrum (CSA, quadrupolar, ...).

The offset in ppm is ``O2 / BF2`` (Hz / MHz) -- the offset from the
channel's basic frequency, which is 0 ppm of that nucleus when the
spectrometer's referencing for it is the standard one. Where the Y
reference differs, give ``ppm_shift`` (the ppm to add).

Notebook use::

    from opennmr.notebook import offset_tracker as ot
    series = ot.offset_series(spectra, region=(10, -10), channel="O2")
    ot.plot_offset_series(series, style="bars")
    y_spectrum = ot.skyline_spectrum(series)
    # one Gaussian of 2 kHz FWHM per offset, drawn and kept as a spectrum
    ot.plot_offset_series(series, style="lineshape", lineshape="gaussian",
                          fwhm=2000, width_unit="Hz")
    y_mas = ot.pattern_spectrum(series, style="lineshape", lineshape="gaussian",
                                fwhm=2000, width_unit="Hz")
"""
from __future__ import annotations

import os

import numpy as np

from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from . import intensity_scaling
from .dnp import region_integral
from .fitting import gaussian, lorentzian, pseudo_voigt

#: The channels a pulsed offset can be read from, and the acqus keys
#: that go with them: (offset, basic frequency, nucleus).
CHANNELS = {"O2": ("O2", "BF2", "NUC2"), "O3": ("O3", "BF3", "NUC3"),
            "O4": ("O4", "BF4", "NUC4")}

#: How the integrals are drawn (and kept): bars (the skyline), vertical
#: lines, or a lineshape centred on every offset. "triangles" was
#: replaced by "lineshape" on 2026-10-18 (user request).
STYLES = ("bars", "stems", "lineshape")

#: The lineshapes a "lineshape" pattern can use, with their display names.
#: Definitions are notebook.fitting's (``eta`` = Lorentzian fraction).
LINESHAPES = {"gaussian": "Gaussian", "lorentzian": "Lorentzian",
              "pseudo_voigt": "Pseudo-Voigt"}

#: Units a lineshape width can be given in.
WIDTH_UNITS = ("ppm", "Hz", "kHz")

#: Fewest grid points across one FWHM of a kept lineshape: fewer and the
#: peaks' heights and widths are misrepresented, so the grid is enlarged.
MIN_POINTS_PER_FWHM = 10

#: Largest grid a kept spectrum is given.
MAX_POINTS = 262144


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
    rg_divided = []
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
        rg_divided.append(bool(divide_rg) and scale.get("rg") is not None)
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
    # "per unit RG" only when every spectrum's RG was known and divided out
    # (2026-10-18: the label claimed it even when no RG was recorded).
    if divide_rg and rg_divided and not all(rg_divided):
        notes.append("Receiver gain divided out of only some spectra: their integrals are "
                     "not on one scale.")
    unit = "integral per scan" + (" per unit RG" if rg_divided and all(rg_divided) else "") \
        + " (a.u.)"
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
    return _derived_spectrum(series, grid, data, label or "offset skyline",
                             {"style": "bars", "shape": shape},
                             {"skyline_offsets_ppm": [float(v) for v in x],
                              "skyline_integrals": [float(v) for v in y],
                              "skyline_region": list(series.get("region") or ()),
                              "skyline_shape": shape})


# ---- lineshapes and sticks at the offsets (2026-10-18) -------------------------

def _sorted_offsets(series: dict):
    x = np.asarray(series["offset_ppm"], float)
    y = np.asarray(series["y"], float)
    if x.size < 1:
        raise ValueError("The series has no offsets.")
    order = np.argsort(x)
    return x[order], y[order]


def _spacing(x: np.ndarray) -> float:
    """Smallest non-zero spacing of the offsets (1 ppm for a single one)."""
    d = np.diff(np.unique(x))
    return float(d.min()) if d.size else 1.0


def offset_spacing(series: dict) -> dict:
    """The offsets' spacing: ``{"ppm", "hz", "uniform"}`` -- the smallest
    spacing in ppm and Hz (Hz needs the channel's BF, else None), and
    whether every step is that spacing (to 0.1 %). When the offsets were
    placed on the pulsed nucleus's spinning sidebands, a uniform spacing
    in Hz is the MAS rate."""
    x, _y = _sorted_offsets(series)
    ppm = _spacing(x)
    d = np.diff(np.unique(x))
    uniform = bool(d.size) and bool(np.allclose(d, ppm, rtol=1e-3, atol=0.0))
    bf = series.get("bf_MHz")
    return {"ppm": ppm, "hz": ppm * bf if bf else None, "uniform": uniform}


def fwhm_in_ppm(series: dict, fwhm: float, width_unit: str = "ppm") -> float:
    """A width given in ``width_unit`` ('ppm', 'Hz' or 'kHz') in ppm of
    the pulsed nucleus. Hz/kHz need the channel's basic frequency
    (``series["bf_MHz"]``, read from BF2/BF3/BF4)."""
    unit = str(width_unit).strip().lower()
    try:
        value = float(fwhm)
    except (TypeError, ValueError):
        raise ValueError(f"The width must be a number, got {fwhm!r}.") from None
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"The width (FWHM) must be above zero, got {fwhm!r}.")
    if unit == "ppm":
        return value
    if unit in ("hz", "khz"):
        bf = series.get("bf_MHz")
        if not bf:
            raise ValueError("A width in Hz needs the pulsed channel's basic frequency (BF), "
                             "which this series does not record: give the width in ppm.")
        return value * (1000.0 if unit == "khz" else 1.0) / float(bf)
    raise ValueError(f"width_unit must be one of {WIDTH_UNITS}, got {width_unit!r}")


def default_fwhm_ppm(series: dict) -> float:
    """A starting width: a quarter of the smallest spacing of the offsets,
    so neighbouring peaks barely overlap (a Gaussian falls to 10^-5 of its
    height at the next offset)."""
    x, _y = _sorted_offsets(series)
    return 0.25 * _spacing(x)


def peak_profile(x, centre: float, fwhm: float, lineshape: str = "gaussian",
                 eta: float = 0.5, height: float = 1.0) -> np.ndarray:
    """One peak of ``lineshape`` ('gaussian', 'lorentzian' or
    'pseudo_voigt') at ``centre``, of full width at half maximum ``fwhm``
    and of maximum ``height`` (x, centre and fwhm in the same unit).
    ``eta`` (pseudo-Voigt only) is the Lorentzian fraction, 0 to 1. The
    shapes are :mod:`opennmr.notebook.fitting`'s, so a fit with the same
    lineshape reproduces them."""
    x = np.asarray(x, float)
    key = str(lineshape).lower().replace("-", "_").replace(" ", "_")
    if key == "gaussian":
        return gaussian(x, centre, fwhm, height)
    if key == "lorentzian":
        return lorentzian(x, centre, fwhm, height)
    if key == "pseudo_voigt":
        eta = float(eta)
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta (the Lorentzian fraction) must be 0 to 1, got {eta:g}.")
        return pseudo_voigt(x, centre, fwhm, height, eta)
    raise ValueError(f"lineshape must be one of {tuple(LINESHAPES)}, got {lineshape!r}")


def _tail_widths(lineshape: str, eta: float) -> float:
    """FWHMs beyond the outermost peaks for the grid to reach: a Gaussian
    is below 10^-10 of its height at 3 FWHM; a Lorentzian only falls to
    0.25 % at 10 FWHM, so it is given 10."""
    key = str(lineshape).lower().replace("-", "_")
    if key == "gaussian" or (key == "pseudo_voigt" and float(eta) == 0.0):
        return 3.0
    return 10.0


def sideband_pattern(series: dict, lineshape: str = "gaussian", fwhm: float | None = None,
                     width_unit: str = "ppm", eta: float = 0.5, x=None,
                     n_points: int | None = None, pad: float | None = None,
                     components: bool = False) -> dict:
    """A peak of ``lineshape`` centred on every offset, its height the
    integral there, all of one width (``fwhm`` in ``width_unit``; None ->
    :func:`default_fwhm_ppm`), summed -- with the offsets on the pulsed
    nucleus's sidebands, its MAS spectrum: a peak per sideband. Only the
    type and the width are chosen; the positions are the offsets.

    ``x``: ppm values to evaluate on (else a descending grid reaching
    ``pad`` ppm beyond the outermost offsets -- by default half a spacing
    plus 3 FWHM for a Gaussian, 10 for a Lorentzian or pseudo-Voigt).
    ``n_points``: points of that grid (default 4096); it is enlarged so a
    FWHM spans at least :data:`MIN_POINTS_PER_FWHM` points (at most
    :data:`MAX_POINTS`), and the notes say so.

    Returns ``{"ppm", "y", "fwhm_ppm", "fwhm_hz", "lineshape", "eta",
    "offsets_ppm", "heights", "components" (n_offsets x n_points, if
    asked), "notes"}``. A note warns when the peaks overlap enough that
    the summed curve at an offset exceeds its integral by over 1 % of the
    largest."""
    offsets, heights = _sorted_offsets(series)
    key = str(lineshape).lower().replace("-", "_").replace(" ", "_")
    if key not in LINESHAPES:
        raise ValueError(f"lineshape must be one of {tuple(LINESHAPES)}, got {lineshape!r}")
    w = default_fwhm_ppm(series) if fwhm is None else fwhm_in_ppm(series, fwhm, width_unit)
    notes = []
    if x is None:
        spacing = _spacing(offsets)
        if pad is None:
            pad = 0.5 * spacing + _tail_widths(key, eta) * w
        lo, hi = float(offsets[0] - pad), float(offsets[-1] + pad)
        n = int(n_points or 4096)
        needed = int(np.ceil((hi - lo) / w * MIN_POINTS_PER_FWHM)) + 1
        if needed > n:
            if needed > MAX_POINTS:
                notes.append(f"The width ({w:.4g} ppm) is narrow for the {hi - lo:.4g} ppm range: "
                             f"{MAX_POINTS} points give {(MAX_POINTS - 1) * w / (hi - lo):.1f} "
                             "points per FWHM, so the peaks are under-sampled.")
                n = MAX_POINTS
            else:
                notes.append(f"Grid enlarged from {n} to {needed} points so that one FWHM "
                             f"spans at least {MIN_POINTS_PER_FWHM} points.")
                n = needed
        grid = np.linspace(hi, lo, n)
    else:
        grid = np.asarray(x, float)
    comps = np.array([peak_profile(grid, c, w, key, eta, h) for c, h in zip(offsets, heights)])
    total = comps.sum(axis=0) if comps.size else np.zeros_like(grid)
    if offsets.size > 1:
        at = np.array([peak_profile(offsets, c, w, key, eta, h)
                       for c, h in zip(offsets, heights)]).sum(axis=0)
        scale = float(np.max(np.abs(heights))) or 1.0
        excess = float(np.max(np.abs(at - heights))) / scale
        if excess > 0.01:
            notes.append(f"The peaks overlap: at an offset the summed curve differs from that "
                         f"offset's integral by up to {100 * excess:.1f} % of the largest "
                         f"(FWHM {w:.4g} ppm, smallest spacing {_spacing(offsets):.4g} ppm). "
                         "Narrow the width for heights equal to the integrals.")
    bf = series.get("bf_MHz")
    out = {"ppm": grid, "y": total, "fwhm_ppm": w, "fwhm_hz": w * bf if bf else None,
           "lineshape": key, "eta": float(eta) if key == "pseudo_voigt" else None,
           "offsets_ppm": offsets, "heights": heights, "notes": notes}
    if components:
        out["components"] = comps
    return out


def sideband_spectrum(series: dict, lineshape: str = "gaussian", fwhm: float | None = None,
                      width_unit: str = "ppm", eta: float = 0.5, n_points: int = 4096,
                      pad: float | None = None, label: str | None = None) -> NMRSpectrum:
    """:func:`sideband_pattern` as a 1D spectrum of the pulsed nucleus
    (vendor "derived"; its nucleus and BF, so Hz axes and Hz fits work),
    to keep in Open experiments and fit. ``metadata.raw["offset_pattern"]``
    records the offsets, integrals, lineshape, width and notes.

    The width is the one chosen, not a measured linewidth: fit it fixed
    (or read a fitted width as this choice). The relative heights are the
    filtered integrals, which also carry the selective pulse's excitation
    profile and the transfer efficiency at each offset."""
    pat = sideband_pattern(series, lineshape=lineshape, fwhm=fwhm, width_unit=width_unit,
                           eta=eta, n_points=n_points, pad=pad)
    return _derived_spectrum(
        series, pat["ppm"], pat["y"], label or f"offset {LINESHAPES[pat['lineshape']]} pattern",
        {"style": "lineshape", "lineshape": pat["lineshape"], "fwhm_ppm": pat["fwhm_ppm"],
         "fwhm_hz": pat["fwhm_hz"], "eta": pat["eta"], "notes": list(pat["notes"])})


def stick_spectrum(series: dict, n_points: int = 4096, pad: float = 0.5,
                   label: str | None = None) -> NMRSpectrum:
    """The vertical lines as a 1D spectrum: each offset's integral at one
    grid point, zero elsewhere (``pad`` spacings beyond the outermost
    offsets). With evenly spaced offsets the grid step divides the
    spacing exactly, so every line sits on its offset; otherwise each
    goes to the nearest point and ``raw["offset_pattern"]
    ["max_displacement_ppm"]`` says how far that moved it. Offsets that
    share a point are added."""
    x, y = _sorted_offsets(series)
    spacing = _spacing(x)
    span = float(x[-1] - x[0]) + 2.0 * pad * spacing
    m = max(1, int(np.ceil((int(n_points) - 1) * spacing / span)))
    step = spacing / m
    edge = int(np.ceil(pad * spacing / step))           # whole steps, so offsets land on points
    hi = float(x[-1]) + edge * step
    n = int(round((float(x[-1] - x[0])) / step)) + 2 * edge + 1
    grid = hi - step * np.arange(n)                     # descending
    idx = np.clip(np.rint((hi - x) / step).astype(int), 0, n - 1)
    data = np.zeros(n)
    np.add.at(data, idx, y)
    moved = float(np.max(np.abs(grid[idx] - x))) if x.size else 0.0
    notes = []
    if moved > 1e-9 * max(abs(spacing), 1.0):
        notes.append(f"Uneven offsets: lines moved to the nearest grid point, by at most "
                     f"{moved:.3g} ppm.")
    return _derived_spectrum(series, grid, data, label or "offset sticks",
                             {"style": "stems", "max_displacement_ppm": moved, "notes": notes})


def pattern_spectrum(series: dict, style: str = "lineshape", n_points: int = 4096,
                     shape: str = "steps", lineshape: str = "gaussian",
                     fwhm: float | None = None, width_unit: str = "ppm", eta: float = 0.5,
                     label: str | None = None) -> NMRSpectrum:
    """What :func:`plot_offset_series` draws with ``style``, as a 1D
    spectrum to keep and fit: 'bars' -> :func:`skyline_spectrum`
    (``shape``), 'stems' -> :func:`stick_spectrum`, 'lineshape' ->
    :func:`sideband_spectrum` (``lineshape``, ``fwhm``, ``width_unit``,
    ``eta``)."""
    if style == "bars":
        return skyline_spectrum(series, n_points=n_points, shape=shape, label=label)
    if style == "stems":
        return stick_spectrum(series, n_points=n_points, label=label)
    if style == "lineshape":
        return sideband_spectrum(series, lineshape=lineshape, fwhm=fwhm, width_unit=width_unit,
                                 eta=eta, n_points=n_points, label=label)
    raise ValueError(_style_error(style))


def _style_error(style) -> str:
    text = f"style must be one of {STYLES}, got {style!r}"
    if style == "triangles":
        text += " ('triangles' is now called 'lineshape')"
    return text


def _derived_spectrum(series: dict, grid, data, label: str, pattern: dict,
                      extra_raw: dict | None = None) -> NMRSpectrum:
    """A kept 1D spectrum of the pulsed nucleus on ``grid`` (ppm,
    descending), recording how it was made in ``raw["offset_pattern"]``."""
    grid = np.asarray(grid, float)
    data = np.asarray(data, float)
    x, y = _sorted_offsets(series)
    bf = series.get("bf_MHz")
    axis = NMRAxis(kind="frequency", ppm=grid, hz=grid * bf if bf else None)
    record = {"offsets_ppm": [float(v) for v in x], "integrals": [float(v) for v in y],
              "region": list(series.get("region") or ()), "channel": series.get("channel")}
    record.update(pattern)
    raw = {"offset_pattern": record}
    raw.update(extra_raw or {})
    meta = NMRMetadata(vendor="derived", nucleus=series.get("nucleus"), sfo1=bf, bf1=bf,
                       source_path=label, raw=raw)
    return NMRSpectrum(data=data, axes=[axis], metadata=meta, data_imag=np.zeros_like(data))


def _draw_series(ax, series: dict, style: str, colour, y0: float = 0.0, width=None,
                 markersize: float = 7.0, linewidth: float = 1.2, lineshape: str = "gaussian",
                 fwhm=None, width_unit: str = "ppm", eta: float = 0.5,
                 show_components: bool = False, show_skyline: bool = False):
    """Draw one series on ``ax`` with its zero line at ``y0``. Returns
    ``(lo_x, hi_x, margin)``: the x range drawn and the margin to leave."""
    x = np.asarray(series["offset_ppm"], float)
    y = np.asarray(series["y"], float)
    spacing = _spacing(np.sort(x)) if x.size else 1.0
    if width is None:
        width = 0.8 * spacing
    lo_x, hi_x = (float(x.min()), float(x.max())) if x.size else (0.0, 1.0)
    margin = 0.05 * (float(np.ptp(x)) or 1.0) + width
    if style == "bars":
        ax.bar(x, y, width=width, bottom=y0, color=colour, edgecolor=colour, linewidth=0.5,
               zorder=2)
    elif style == "stems":
        ax.vlines(x, y0, y + y0, color=colour, lw=linewidth, zorder=2)
        ax.plot(x, y + y0, "o", color=colour, ms=0.7 * markersize, ls="none", zorder=3)
    else:
        pat = sideband_pattern(series, lineshape=lineshape, fwhm=fwhm, width_unit=width_unit,
                               eta=eta, components=show_components)
        if show_components:
            for comp in pat["components"]:
                ax.plot(pat["ppm"], np.asarray(comp) + y0, "--", color=colour,
                        lw=0.6 * linewidth, alpha=0.6, zorder=1)
        ax.plot(pat["ppm"], pat["y"] + y0, "-", color=colour, lw=linewidth, zorder=2)
        lo_x, hi_x = float(pat["ppm"].min()), float(pat["ppm"].max())
        margin = 0.0
    if show_skyline and x.size > 1:
        sky = skyline_spectrum(series, n_points=4096)
        ax.plot(sky.axes[0].ppm, np.asarray(sky.data) + y0, "-", color="0.4", lw=0.8, zorder=1)
    ax.axhline(y0, color="0.6", lw=0.6, zorder=0)
    return lo_x, hi_x, margin


def _axis_labels(ax, series: dict, ylabel: str | None = None):
    nucleus = series.get("nucleus")
    ax.set_xlabel(_italic_label(f"{series.get('channel', 'O2')} offset"
                  + (f", {_pretty(nucleus)} shift" if nucleus else "") + " (ppm)"))
    ax.set_ylabel(_italic_label(ylabel or series.get("unit", "integral")))


def plot_offset_series(series: dict, style: str = "bars", ax=None, colour=None,
                       width: float | None = None, show_skyline: bool = False,
                       markersize: float = 7.0, linewidth: float = 1.2, text_size="small",
                       lineshape: str = "gaussian", fwhm: float | None = None,
                       width_unit: str = "ppm", eta: float = 0.5,
                       show_components: bool = False):
    """The integrals against the offset (ppm of the pulsed nucleus):
    ``style`` 'bars', 'stems' (vertical lines with a dot) or 'lineshape'
    -- a ``lineshape`` ('gaussian', 'lorentzian', 'pseudo_voigt' with
    Lorentzian fraction ``eta``) of FWHM ``fwhm`` (in ``width_unit``;
    None -> a quarter of the smallest spacing) centred on every offset,
    its height the integral, summed (:func:`sideband_pattern`);
    ``show_components`` adds each peak, dashed. ``width`` is the bars'
    width (ppm). The x axis runs high to low ppm, as a spectrum of that
    nucleus does. ``show_skyline`` draws the bars' outline. Returns
    ``(fig, ax)``. For several regions on one plot see
    :func:`plot_offset_series_regions`."""
    import matplotlib.pyplot as plt
    if style not in STYLES:
        raise ValueError(_style_error(style))
    x = np.asarray(series["offset_ppm"], float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
    else:
        fig = ax.figure
    colour = colour or "black"
    lo_x, hi_x, margin = _draw_series(
        ax, series, style, colour, 0.0, width=width, markersize=markersize, linewidth=linewidth,
        lineshape=lineshape, fwhm=fwhm, width_unit=width_unit, eta=eta,
        show_components=show_components, show_skyline=show_skyline)
    _axis_labels(ax, series)
    if x.size:
        ax.set_xlim(hi_x + margin, lo_x - margin)
    fig.tight_layout()
    return fig, ax


#: How several regions share one plot.
ARRANGEMENTS = ("stacked", "superimposed")


def region_name(series: dict) -> str:
    """'5 to -5 ppm' for a series' region (its own ``name`` if it has one)."""
    if series.get("name"):
        return str(series["name"])
    lo, hi = series["region"]
    return f"{hi:g} to {lo:g} ppm"


def offset_series_regions(spectra, regions, names=None, **kwargs) -> list[dict]:
    """:func:`offset_series` for each of several ``regions`` (a list of
    ``(lo, hi)`` ppm pairs) of the same spectra, so the integrals of
    every region are against the same offsets. ``names``: one per region
    (default '<hi> to <lo> ppm'). ``kwargs`` are :func:`offset_series`'s
    (channel, offsets_ppm, labels, baseline, divide_rg, ppm_shift,
    normalise). Returns one series per region, each with a ``name``."""
    spectra = list(spectra)
    regions = [tuple(float(v) for v in r) for r in regions]
    if not regions:
        raise ValueError("No regions given.")
    if names is not None and len(names) != len(regions):
        raise ValueError(f"{len(regions)} regions but {len(names)} names.")
    out = []
    for i, region in enumerate(regions):
        series = offset_series(spectra, region, **kwargs)
        series["name"] = str(names[i]) if names is not None else region_name(series)
        out.append(series)
    return out


def plot_offset_series_regions(series_list, style: str = "bars", arrangement: str = "stacked",
                               ax=None, colours=None, scale_each: bool = False,
                               gap: float = 0.15, show_legend: bool = True,
                               width: float | None = None, show_skyline: bool = False,
                               markersize: float = 7.0, linewidth: float = 1.2,
                               lineshape: str = "gaussian", fwhm: float | None = None,
                               width_unit: str = "ppm", eta: float = 0.5,
                               show_components: bool = False):
    """The integrals of several regions against the offsets on one plot.

    ``arrangement``: 'superimposed' (all on one baseline, one colour per
    region, with a legend) or 'stacked' (each region on its own baseline,
    the first at the top, named at its left; ``gap`` is the free space
    between neighbours as a fraction of the tallest one; the y axis has
    no numbers, as the baselines are shifted for clarity).

    ``scale_each`` divides every region by its own largest |integral|, so
    regions of very different size can be compared in shape (the y axis
    then reads 'relative to each region's largest'); without it all share
    one scale. ``colours``: one per region (default a colour-blind-friendly
    palette). ``style`` and the lineshape arguments are
    :func:`plot_offset_series`'s. Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt
    if style not in STYLES:
        raise ValueError(_style_error(style))
    if arrangement not in ARRANGEMENTS:
        raise ValueError(f"arrangement must be one of {ARRANGEMENTS}, got {arrangement!r}")
    series_list = list(series_list)
    if not series_list:
        raise ValueError("No series to plot.")
    n = len(series_list)
    if colours is None:
        # The palette Compare shades the regions in, so a region has the
        # same colour on both views.
        from opennmr.plotting.accessibility import ordered_qualitative_colours
        palette = ordered_qualitative_colours("okabe_ito")
        colours = [palette[i % len(palette)] for i in range(n)]
    if len(colours) < n:
        raise ValueError(f"{n} regions but {len(colours)} colours.")
    drawn = []
    for series in series_list:
        y = np.asarray(series["y"], float)
        if scale_each and y.size and np.max(np.abs(y)) > 0:
            m = float(np.max(np.abs(y)))
            series = dict(series, y=y / m, y_err=np.asarray(series["y_err"], float) / m)
        drawn.append(series)
    kw = dict(width=width, markersize=markersize, linewidth=linewidth, lineshape=lineshape,
              fwhm=fwhm, width_unit=width_unit, eta=eta, show_components=show_components,
              show_skyline=show_skyline)

    def extent(series):
        y = np.asarray(series["y"], float)
        if style == "lineshape":
            pat = sideband_pattern(series, lineshape=lineshape, fwhm=fwhm, width_unit=width_unit,
                                   eta=eta)
            y = np.concatenate([y, np.asarray(pat["y"], float)])
        return float(min(y.min(), 0.0)), float(max(y.max(), 0.0))

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.0 if arrangement == "superimposed"
                                        else 1.2 + 0.9 * n))
    else:
        fig = ax.figure
    bases = [0.0] * n
    if arrangement == "stacked" and n > 1:
        ext = [extent(s) for s in drawn]
        step = (max(e[1] for e in ext) - min(e[0] for e in ext)) * (1.0 + float(gap))
        if step <= 0:
            step = 1.0
        bases = [(n - 1 - i) * step for i in range(n)]
    lo_x, hi_x, margin = np.inf, -np.inf, 0.0
    for i, series in enumerate(drawn):
        a, b, m = _draw_series(ax, series, style, colours[i], bases[i], **kw)
        lo_x, hi_x, margin = min(lo_x, a), max(hi_x, b), max(margin, m)
        if arrangement == "stacked":
            ax.annotate(region_name(series_list[i]), xy=(0.01, bases[i]),
                        xycoords=ax.get_yaxis_transform(), xytext=(0, 3),
                        textcoords="offset points", ha="left", va="bottom",
                        color=colours[i], fontsize="small")
        else:
            ax.plot([], [], "-", color=colours[i], lw=max(linewidth, 2.0),
                    label=region_name(series_list[i]))
    ref = series_list[0]
    ylabel = ("relative to each region's largest" if scale_each
              else ref.get("unit", "integral"))
    if arrangement == "stacked" and n > 1:
        ylabel += " (regions shifted)"
        ax.set_yticks([])
    _axis_labels(ax, ref, ylabel)
    if lo_x <= hi_x:
        ax.set_xlim(hi_x + margin, lo_x - margin)
    if arrangement == "superimposed" and show_legend and n > 1:
        ax.legend(frameon=False, fontsize="small")
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
    if len(series["offset_ppm"]) > 1:
        sp = offset_spacing(series)
        if sp["uniform"]:
            line = f"Offsets evenly spaced by {sp['ppm']:.6g} ppm"
            if sp["hz"]:
                line += (f" ({sp['hz']:.6g} Hz): if they were placed on the sidebands, "
                         "that is the MAS rate.")
            lines.append(line if line.endswith(".") else line + ".")
        else:
            lines.append(f"Offsets unevenly spaced (smallest step {sp['ppm']:.6g} ppm).")
    lines += series.get("notes") or []
    return "\n".join(lines)


def _italic_label(text):
    """Quantity symbols italic (plotting.labels.italicise_symbols)."""
    from opennmr.plotting.labels import italicise_symbols
    return italicise_symbols(text)
