"""
Colour-accessibility checks and palettes: colour-blind-friendly and
print-friendly series colours and trace colours.

Everything here is *measured*, not asserted. A palette is called
colour-blind-friendly or print-friendly because the colours it actually
produces -- after :data:`opennmr.plotting.style.MAX_LINE_LUMINANCE`
has darkened anything too pale to see on white -- pass the checks
below for a given number of series, not because a palette's designer
said so. That matters more than it sounds: the line-contrast cap keeps
only the darker part of every map, which is exactly the part where
colours crowd together, so a map that is fine for three traces can
fail for six.

The checks
----------
*Colour-blind-friendly* (:func:`is_colourblind_safe`): every pair of
colours in the set differs by at least :data:`MIN_DELTA_E` (CIE76
ΔE*ab, CIELAB, D65) under normal vision **and** under simulated
protanopia, deuteranopia and tritanopia. The simulation is Machado,
Oliveira & Fernandes (2009, IEEE TVCG 15, 1291), severity 1.0,
applied in linear sRGB -- the same model and matrices as the
``colorspacious`` package, against which the test suite checks it.

*Print-friendly* (:func:`is_print_safe`): every pair differs by at
least :data:`MIN_DELTA_L` in CIELAB lightness L* alone -- i.e. the
traces stay distinguishable when the figure is printed or photocopied
in greyscale. (This is stricter than, and different from, ColorBrewer's
"print friendly" flag, which is a judgement about CMYK colour
reproduction and cannot be computed from the colours.)

The thresholds are a design choice, stated here so they can be argued
with: a just-noticeable difference for large patches is ΔE ≈ 2.3, and
thin plotted lines need several times that, so 10 is used for both.
Every function takes the threshold as an argument.

Palettes
--------
Besides matplotlib's own maps (the ColorBrewer ones among them), four
qualitative palettes designed for colour-vision deficiency are
registered with matplotlib on import, so any code that takes a
colormap name accepts them:

* ``okabe_ito`` -- Okabe & Ito (2008); Wong, *Nat. Methods* 8, 441 (2011).
* ``tol_bright``, ``tol_muted``, ``tol_high_contrast`` -- Paul Tol's
  qualitative schemes (SRON technical note, 2021).

A qualitative palette is used as a *list* -- series take its entries in
turn -- not sampled as a continuous ramp, which would blend neighbouring
categories into colours that belong to neither (see
:func:`ordered_qualitative_colours` for the order).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors

#: Minimum CIE76 ΔE*ab between any two colours, under normal and each
#: simulated dichromatic vision, for a set to count as colour-blind-
#: friendly. See the module docstring for why 10.
MIN_DELTA_E = 10.0

#: Minimum difference in CIELAB lightness L* between any two colours
#: for a set to count as print-(greyscale-)friendly.
MIN_DELTA_L = 10.0

#: The largest series count the "safe up to N" figures are worked out
#: to. Past a dozen overlaid traces no palette separates them, and the
#: answer to "is this safe for 30 series?" is no.
MAX_SERIES_CHECKED = 12

#: Qualitative palettes designed for colour-vision deficiency, as
#: published (designer's order). Registered with matplotlib on import.
ACCESSIBLE_QUALITATIVE_PALETTES = {
    "okabe_ito": ("#000000", "#E69F00", "#56B4E9", "#009E73",
                  "#F0E442", "#0072B2", "#D55E00", "#CC79A7"),
    "tol_bright": ("#4477AA", "#EE6677", "#228833", "#CCBB44",
                   "#66CCEE", "#AA3377", "#BBBBBB"),
    "tol_muted": ("#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
                  "#DDCC77", "#CC6677", "#882255", "#AA4499"),
    "tol_high_contrast": ("#004488", "#DDAA33", "#BB5566"),
}

#: matplotlib's own qualitative (listed, few-entry) maps that are worth
#: offering as series palettes alongside the ones above. ColorBrewer's
#: Dark2/Set1/Set2/Paired and matplotlib's tab10.
BUILTIN_QUALITATIVE_PALETTES = ("Dark2", "Set1", "Set2", "Paired", "tab10")

#: Human-readable names for the palettes that have one, for menus.
PALETTE_DISPLAY_NAMES = {
    "okabe_ito": "Okabe–Ito",
    "tol_bright": "Tol bright",
    "tol_muted": "Tol muted",
    "tol_high_contrast": "Tol high-contrast",
}

#: A ListedColormap with at most this many entries is treated as a
#: set of categories rather than a ramp. Continuous maps are listed
#: with 256 entries; the largest qualitative one matplotlib ships
#: (tab20) has 20.
_QUALITATIVE_MAX_ENTRIES = 32

# Machado et al. (2009), severity 1.0, linear-sRGB. Rows map input
# (R, G, B) to simulated (R, G, B).
_CVD_MATRICES = {
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779],
                        [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}

#: The colour-vision deficiencies simulated, in report order.
CVD_TYPES = tuple(_CVD_MATRICES)

_SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])
_D65_WHITE = np.array([0.95047, 1.0, 1.08883])


# ------------------------------------------------------------------
# Colour science
# ------------------------------------------------------------------

def _as_rgb_array(colors) -> np.ndarray:
    """Any sequence of matplotlib colour specs as an (n, 3) sRGB array
    in [0, 1] (alpha dropped)."""
    return np.array([mcolors.to_rgb(c) for c in colors], dtype=float).reshape(-1, 3)


def _to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _from_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def simulate_cvd(colors, deficiency: str) -> np.ndarray:
    """``colors`` as seen with ``deficiency`` (``"protan"``,
    ``"deutan"`` or ``"tritan"``): an (n, 3) sRGB array.

    Machado et al. (2009) at full severity, i.e. dichromacy -- the
    worst case of each type, which is the right one to design for."""
    if deficiency not in _CVD_MATRICES:
        raise ValueError(f"Unknown colour-vision deficiency {deficiency!r}; "
                         f"expected one of {CVD_TYPES}.")
    linear = _to_linear(_as_rgb_array(colors))
    return _from_linear(linear @ _CVD_MATRICES[deficiency].T)


def to_lab(colors) -> np.ndarray:
    """``colors`` in CIELAB (D65): an (n, 3) array of (L*, a*, b*)."""
    xyz = _to_linear(_as_rgb_array(colors)) @ _SRGB_TO_XYZ.T
    t = xyz / _D65_WHITE
    delta = 6.0 / 29.0
    f = np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)
    return np.stack([116.0 * f[:, 1] - 16.0,
                     500.0 * (f[:, 0] - f[:, 1]),
                     200.0 * (f[:, 1] - f[:, 2])], axis=-1)


def _min_pairwise(points: np.ndarray) -> float:
    if len(points) < 2:
        return float("inf")
    diffs = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return float(diffs[np.triu_indices(len(points), 1)].min())


def palette_distinctness(colors) -> dict:
    """How far apart the closest two of ``colors`` are, per viewer.

    Returns ``{"normal", "protan", "deutan", "tritan", "greyscale"}``:
    the first four are the smallest pairwise ΔE*ab under that vision,
    ``"greyscale"`` the smallest pairwise ΔL*. ``inf`` for fewer than
    two colours (nothing to confuse)."""
    colors = list(colors)
    lab = to_lab(colors) if colors else np.zeros((0, 3))
    report = {"normal": _min_pairwise(lab)}
    for kind in CVD_TYPES:
        report[kind] = _min_pairwise(to_lab(simulate_cvd(colors, kind))) if colors else float("inf")
    report["greyscale"] = _min_pairwise(lab[:, :1])
    return report


def is_colourblind_safe(colors, min_delta_e: float = MIN_DELTA_E) -> bool:
    """Whether every pair of ``colors`` stays at least ``min_delta_e``
    apart under normal vision and each simulated dichromacy."""
    report = palette_distinctness(colors)
    return all(report[k] >= min_delta_e for k in ("normal",) + CVD_TYPES)


def is_print_safe(colors, min_delta_l: float = MIN_DELTA_L) -> bool:
    """Whether every pair of ``colors`` stays at least ``min_delta_l``
    apart in lightness -- i.e. distinguishable printed in greyscale."""
    return palette_distinctness(colors)["greyscale"] >= min_delta_l


def describe_accessibility(colors, min_delta_e: float = MIN_DELTA_E,
                           min_delta_l: float = MIN_DELTA_L) -> str:
    """One plain-English line on whether ``colors`` pass both checks,
    naming the vision that fails -- for a GUI status line or a log."""
    colors = list(colors)
    if len(colors) < 2:
        return "One series: nothing to tell apart."
    report = palette_distinctness(colors)
    failing = [name for name, key in (("normal vision", "normal"), ("protanopia", "protan"),
                                      ("deuteranopia", "deutan"), ("tritanopia", "tritan"))
               if report[key] < min_delta_e]
    greyscale_ok = report["greyscale"] >= min_delta_l
    parts = []
    if failing:
        parts.append("hard to tell apart with " + ", ".join(failing))
    else:
        parts.append("colour-blind-friendly")
    parts.append("print-friendly" if greyscale_ok else "not distinguishable in greyscale")
    return f"{len(colors)} series: " + "; ".join(parts) + "."


# ------------------------------------------------------------------
# Palettes
# ------------------------------------------------------------------

def register_palettes() -> None:
    """Register :data:`ACCESSIBLE_QUALITATIVE_PALETTES` with matplotlib
    (idempotent), so ``plt.get_cmap("okabe_ito")`` and every colormap
    argument in this package accept them."""
    for name, colors in ACCESSIBLE_QUALITATIVE_PALETTES.items():
        if name in mpl.colormaps:
            continue
        mpl.colormaps.register(mcolors.ListedColormap(list(colors), name=name))


def is_qualitative(cmap) -> bool:
    """Whether ``cmap`` (name or colormap) is a set of categories --
    a short :class:`~matplotlib.colors.ListedColormap` -- rather than a
    continuous ramp."""
    if isinstance(cmap, str):
        if cmap in ACCESSIBLE_QUALITATIVE_PALETTES:
            return True
        try:
            cmap = mpl.colormaps[cmap]
        except KeyError:
            return False
    return isinstance(cmap, mcolors.ListedColormap) and cmap.N <= _QUALITATIVE_MAX_ENTRIES


def _worst_case_distance(a, b) -> float:
    """ΔE between two colours for the viewer who finds them most alike."""
    pair = [a, b]
    distances = [_min_pairwise(to_lab(pair))]
    distances += [_min_pairwise(to_lab(simulate_cvd(pair, kind))) for kind in CVD_TYPES]
    return min(distances)


@lru_cache(maxsize=None)
def _ordered_qualitative(name: str, max_luminance) -> tuple:
    from . import style  # local: style imports this module at load time

    cmap = mpl.colormaps[name]
    darkened = [tuple(style._darkened(c, max_luminance)) for c in cmap.colors]
    order = [0]
    remaining = list(range(1, len(darkened)))
    while remaining:
        best = max(remaining, key=lambda j: min(
            _worst_case_distance(darkened[j][:3], darkened[i][:3]) for i in order))
        order.append(best)
        remaining.remove(best)
    return tuple(darkened[i] for i in order)


def ordered_qualitative_colours(name: str, max_luminance: float = None) -> list:
    """The entries of qualitative palette ``name``, made dark enough to
    read on white and put in the order series should take them.

    **The order is chosen, not the designer's.** Each entry is first
    darkened to the line-contrast cap (see
    :func:`opennmr.plotting.style._darkened`), which moves some of them
    -- Okabe–Ito's yellow and sky blue, for instance -- towards their
    neighbours. The first colour is kept; each next one is the entry
    farthest (worst case over normal and dichromatic vision) from all
    already chosen. So the first *n* entries are as far apart as this
    palette allows for any *n*, which is what a plot with *n* series
    uses."""
    return list(_ordered_qualitative(name, max_luminance))


def series_colours(n: int, name: str, max_luminance: float = None) -> list:
    """The ``n`` colours a plot with ``n`` series draws from palette
    ``name`` -- :func:`opennmr.plotting.style.stack_colors`, which this
    is a named shortcut for."""
    from . import style

    return style.stack_colors(n, name, max_luminance)


@lru_cache(maxsize=None)
def _series_report(name: str, n: int) -> tuple:
    """:func:`palette_distinctness` of palette ``name``'s first ``n``
    series colours, cached: the palette lists ask the same question
    for every palette and several criteria."""
    report = palette_distinctness(series_colours(n, name))
    return tuple(sorted(report.items()))


@lru_cache(maxsize=None)
def max_safe_series(name: str, criterion: str = "colourblind",
                    min_delta: float = None) -> int:
    """The largest number of series (up to :data:`MAX_SERIES_CHECKED`)
    for which palette ``name`` passes ``criterion`` -- ``"colourblind"``,
    ``"print"``, or ``"both"`` -- *for every count from two up to it*.

    Measured on the colours actually drawn (:func:`series_colours`).
    Returns 1 when even two series fail."""
    if criterion not in ("colourblind", "print", "both"):
        raise ValueError("criterion must be 'colourblind', 'print' or 'both'")
    safe = 1
    for n in range(2, MAX_SERIES_CHECKED + 1):
        report = dict(_series_report(name, n))
        if criterion in ("colourblind", "both"):
            threshold = MIN_DELTA_E if min_delta is None else min_delta
            if any(report[k] < threshold for k in ("normal",) + CVD_TYPES):
                break
        if criterion in ("print", "both"):
            threshold = MIN_DELTA_L if min_delta is None else min_delta
            if report["greyscale"] < threshold:
                break
        safe = n
    return safe


def series_palette_names() -> list:
    """Every palette offered for series colouring: the contour/ramp maps
    plus the qualitative ones, accessible ones first."""
    from .style import CONTOUR_CMAP_NAMES

    return (list(CONTOUR_CMAP_NAMES) + list(ACCESSIBLE_QUALITATIVE_PALETTES)
            + list(BUILTIN_QUALITATIVE_PALETTES))


def friendly_palettes(criterion: str = "colourblind", min_series: int = 3) -> list:
    """``(name, safe_up_to)`` for every offered palette that passes
    ``criterion`` for at least ``min_series`` series, best first (then
    by name). What the GUI's "Colour blind-friendly" and
    "Print-friendly" palette lists show."""
    rated = [(name, max_safe_series(name, criterion)) for name in series_palette_names()]
    kept = [(name, n) for name, n in rated if n >= min_series]
    return sorted(kept, key=lambda item: (-item[1], item[0].lower()))


#: Default palettes for *n* series, tried in order until one is
#: colour-blind friendly (measured, see :func:`max_safe_series`) for *n*:
#: categories (sites, species, fit components) and ordered series (a
#: sweep of D, of power) differently. Past what any of them separates,
#: the last one (the most separating) is used.
DEFAULT_CATEGORY_PALETTES = ("tol_muted", "Paired", "PRGn")
DEFAULT_ORDERED_PALETTES = ("cividis", "PRGn")


def default_palette(n: int, ordered: bool = False) -> str:
    """The palette a plot of ``n`` series uses unless told otherwise:
    the first of :data:`DEFAULT_CATEGORY_PALETTES` (or, for an ordered
    sweep, :data:`DEFAULT_ORDERED_PALETTES`) that is colour-blind
    friendly for ``n``. Explicit user request (2026-10-01): every default
    plot colour colour-blind friendly."""
    names = DEFAULT_ORDERED_PALETTES if ordered else DEFAULT_CATEGORY_PALETTES
    register_palettes()
    for name in names:
        if max_safe_series(name, "colourblind") >= int(n):
            return name
    return names[-1]


def default_series_colours(n: int, ordered: bool = False, max_luminance: float = None) -> list:
    """The ``n`` colours of :func:`default_palette` -- as
    :func:`series_colours` draws them (dark enough to read on white)."""
    return series_colours(int(n), default_palette(n, ordered), max_luminance)


# ------------------------------------------------------------------
# Single-trace colours
# ------------------------------------------------------------------

#: Candidates for a colour-blind-friendly single trace colour: the
#: Okabe–Ito and Tol bright entries. Filtered by
#: :func:`trace_colour_presets` to those that meet the 3:1 line
#: contrast against white.
_COLOURBLIND_TRACE_CANDIDATES = (
    ("Black", "#000000"),
    ("Blue (Okabe–Ito)", "#0072B2"),
    ("Vermilion (Okabe–Ito)", "#D55E00"),
    ("Bluish green (Okabe–Ito)", "#009E73"),
    ("Reddish purple (Okabe–Ito)", "#CC79A7"),
    ("Orange (Okabe–Ito)", "#E69F00"),
    ("Blue (Tol)", "#4477AA"),
    ("Purple (Tol)", "#AA3377"),
    ("Green (Tol)", "#228833"),
)

#: Candidates for a print-friendly trace colour -- one that still prints
#: as a strong dark line in greyscale.
_PRINT_TRACE_CANDIDATES = (
    ("Black", "#000000"),
    ("Dark grey", "#404040"),
    ("Blue (Tol high-contrast)", "#004488"),
    ("Wine (Tol muted)", "#882255"),
    ("Indigo (Tol muted)", "#332288"),
    ("Green (Tol muted)", "#117733"),
    ("Red (Tol high-contrast)", "#BB5566"),
)

#: A print-friendly trace must print at least this dark: L* at or
#: below 50 is the darker half of the grey scale.
PRINT_TRACE_MAX_LIGHTNESS = 50.0


def trace_colour_presets(kind: str) -> list:
    """``(label, hex)`` single-trace colours of ``kind``
    (``"colourblind"`` or ``"print"``).

    Both lists are filtered here rather than trusted: every entry meets
    the 3:1 contrast against white that every plotted line in this
    package does (:data:`opennmr.plotting.style.MAX_LINE_LUMINANCE`),
    and a print entry must also print dark
    (L* <= :data:`PRINT_TRACE_MAX_LIGHTNESS`)."""
    from . import style

    if kind == "colourblind":
        candidates = _COLOURBLIND_TRACE_CANDIDATES
    elif kind == "print":
        candidates = _PRINT_TRACE_CANDIDATES
    else:
        raise ValueError("kind must be 'colourblind' or 'print'")
    kept = []
    for label, colour in candidates:
        if style.is_too_light(colour):
            continue
        if kind == "print" and to_lab([colour])[0, 0] > PRINT_TRACE_MAX_LIGHTNESS:
            continue
        kept.append((label, colour))
    return kept


register_palettes()
