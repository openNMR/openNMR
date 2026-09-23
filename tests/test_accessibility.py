"""Colour-blind-friendly and print-friendly palettes
(:mod:`opennmr.plotting.accessibility`), and the qualitative-palette
path through :func:`opennmr.plotting.style.stack_colors`."""

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pytest

from opennmr import plotting
from opennmr.plotting import accessibility as acc
from opennmr.plotting import style


# ------------------------------------------------------------------
# Colour science: checked against an independent implementation
# ------------------------------------------------------------------

_RANDOM_RGB = np.random.default_rng(0).random((40, 3))


@pytest.mark.parametrize("ours, theirs", [
    ("protan", "protanomaly"), ("deutan", "deuteranomaly"), ("tritan", "tritanomaly"),
])
def test_cvd_simulation_matches_colorspacious(ours, theirs):
    colorspacious = pytest.importorskip("colorspacious")
    expected = colorspacious.cspace_convert(
        _RANDOM_RGB, {"name": "sRGB1+CVD", "cvd_type": theirs, "severity": 100}, "sRGB1")
    np.testing.assert_allclose(acc.simulate_cvd(_RANDOM_RGB, ours),
                               np.clip(expected, 0, 1), atol=1e-6)


def test_lab_matches_colorspacious():
    colorspacious = pytest.importorskip("colorspacious")
    expected = colorspacious.cspace_convert(_RANDOM_RGB, "sRGB1", "CIELab")
    np.testing.assert_allclose(acc.to_lab(_RANDOM_RGB), expected, atol=0.05)


def test_lab_reference_values():
    # Textbook sRGB -> CIELAB (D65) values, independent of any package.
    lab = acc.to_lab(["#FFFFFF", "#000000", "#FF0000"])
    np.testing.assert_allclose(lab[0], [100.0, 0.0, 0.0], atol=0.01)
    np.testing.assert_allclose(lab[1], [0.0, 0.0, 0.0], atol=0.01)
    np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.05)


@pytest.mark.parametrize("kind", acc.CVD_TYPES)
def test_cvd_simulation_leaves_greys_grey(kind):
    # Every row of each Machado matrix sums to 1, so an achromatic
    # colour is seen unchanged -- a check on the matrices as typed.
    greys = [(v, v, v) for v in (0.0, 0.2, 0.5, 0.8, 1.0)]
    np.testing.assert_allclose(acc.simulate_cvd(greys, kind), np.array(greys), atol=2e-3)


def test_unknown_deficiency_is_an_error():
    with pytest.raises(ValueError):
        acc.simulate_cvd(["#000000"], "achromat")


def test_red_green_pair_collapses_under_deuteranopia():
    report = acc.palette_distinctness(["#E41A1C", "#4DAF4A"])  # ColorBrewer Set1 red/green
    assert report["deutan"] < 0.25 * report["normal"]


def test_colourblind_safe_is_not_print_safe():
    # Okabe-Ito vermilion and bluish green: designed to survive
    # colour-vision deficiency, and do, but nearly the same grey.
    pair = ["#D55E00", "#009E73"]
    assert acc.is_colourblind_safe(pair)
    assert not acc.is_print_safe(pair)


def test_single_colour_has_nothing_to_confuse():
    report = acc.palette_distinctness(["#123456"])
    assert all(value == float("inf") for value in report.values())
    assert acc.is_colourblind_safe(["#123456"]) and acc.is_print_safe(["#123456"])


def test_describe_accessibility_names_the_failing_vision():
    text = acc.describe_accessibility(["#E41A1C", "#4DAF4A", "#377EB8"], min_delta_e=40)
    assert text.startswith("3 series:")
    assert "deuteranopia" in text
    assert acc.describe_accessibility(["#000000"]).startswith("One series")


# ------------------------------------------------------------------
# Palettes
# ------------------------------------------------------------------

@pytest.mark.parametrize("name", list(acc.ACCESSIBLE_QUALITATIVE_PALETTES))
def test_accessible_palettes_are_registered_with_matplotlib(name):
    cmap = mpl.colormaps[name]
    assert isinstance(cmap, mcolors.ListedColormap)
    assert [mcolors.to_hex(c).upper() for c in cmap.colors] == \
        [c.upper() for c in acc.ACCESSIBLE_QUALITATIVE_PALETTES[name]]


def test_qualitative_detection():
    for name in list(acc.ACCESSIBLE_QUALITATIVE_PALETTES) + list(acc.BUILTIN_QUALITATIVE_PALETTES):
        assert acc.is_qualitative(name), name
    for name in ("viridis", "RdBu", "Greys", "not-a-map"):
        assert not acc.is_qualitative(name), name


@pytest.mark.parametrize("name", ["okabe_ito", "tol_muted", "Dark2", "tab10"])
@pytest.mark.parametrize("n", [1, 3, 5, 12])
def test_qualitative_stack_colours_are_palette_entries_not_blends(name, n):
    colours = style.stack_colors(n, name)
    assert len(colours) == n
    darkened_entries = {tuple(np.round(style._darkened(c)[:3], 6)) for c in mpl.colormaps[name].colors}
    for colour in colours:
        assert tuple(np.round(colour[:3], 6)) in darkened_entries
        assert style.relative_luminance(colour[:3]) <= style.MAX_LINE_LUMINANCE + 1e-9


def test_qualitative_stack_colours_are_distinct_until_the_palette_runs_out():
    size = len(acc.ACCESSIBLE_QUALITATIVE_PALETTES["tol_muted"])
    colours = style.stack_colors(size, "tol_muted")
    assert len({tuple(np.round(c, 6)) for c in colours}) == size
    # ... and then cycle.
    assert style.stack_colors(size + 1, "tol_muted")[size] == colours[0]


def test_qualitative_order_keeps_the_first_entry_and_prefix_is_stable():
    ordered = acc.ordered_qualitative_colours("okabe_ito")
    assert mcolors.to_hex(ordered[0]) == "#000000"
    # The first n of a longer stack are the n-series colours.
    assert style.stack_colors(3, "okabe_ito") == style.stack_colors(6, "okabe_ito")[:3]


def test_ramp_maps_are_unchanged_by_the_qualitative_path():
    # viridis etc. still go through the usable-range sweep.
    colours = style.stack_colors(4, "viridis")
    cmap = mpl.colormaps["viridis"]
    usable = style.usable_cmap_positions(cmap)
    expected = [style._darkened(cmap(p)) for p in np.interp(
        np.linspace(0, 1, 4), np.linspace(0, 1, usable.size), usable)]
    np.testing.assert_allclose(np.array(colours), np.array(expected))


@pytest.mark.parametrize("criterion", ["colourblind", "print", "both"])
def test_friendly_palettes_pass_their_own_check_and_fail_one_past_it(criterion):
    listed = acc.friendly_palettes(criterion)
    assert listed, "no palette qualifies at all"
    for name, safe_n in listed:
        assert safe_n >= 3
        for n in range(2, safe_n + 1):
            colours = acc.series_colours(n, name)
            if criterion in ("colourblind", "both"):
                assert acc.is_colourblind_safe(colours), (name, n)
            if criterion in ("print", "both"):
                assert acc.is_print_safe(colours), (name, n)
        if safe_n < acc.MAX_SERIES_CHECKED:
            colours = acc.series_colours(safe_n + 1, name)
            ok = ((criterion == "print" or acc.is_colourblind_safe(colours))
                  and (criterion == "colourblind" or acc.is_print_safe(colours)))
            assert not ok, (name, safe_n + 1)
    # Best first.
    assert [n for _name, n in listed] == sorted((n for _name, n in listed), reverse=True)


def test_both_is_the_intersection():
    blind = dict(acc.friendly_palettes("colourblind"))
    grey = dict(acc.friendly_palettes("print"))
    for name, n in acc.friendly_palettes("both"):
        assert n <= blind[name] and n <= grey[name]


def test_diverging_maps_are_never_print_friendly_for_many_series():
    # Symmetric in lightness: the two halves print as the same greys.
    assert acc.max_safe_series("RdBu", "print") < 3
    assert acc.max_safe_series("coolwarm", "print") < 3


def test_known_greyscale_safe_map_is_listed():
    assert "Greys" in dict(acc.friendly_palettes("print"))


def test_every_offered_palette_resolves():
    for name in acc.series_palette_names():
        assert len(acc.series_colours(3, name)) == 3


def test_max_safe_series_rejects_unknown_criterion():
    with pytest.raises(ValueError):
        acc.max_safe_series("viridis", "sepia")


# ------------------------------------------------------------------
# Single-trace presets
# ------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["colourblind", "print"])
def test_trace_presets_are_readable_lines(kind):
    presets = acc.trace_colour_presets(kind)
    assert presets
    assert ("Black", "#000000") in presets
    for _label, colour in presets:
        assert style.contrast_against_white(colour) >= 3.0


def test_print_trace_presets_print_dark():
    for _label, colour in acc.trace_colour_presets("print"):
        assert acc.to_lab([colour])[0, 0] <= acc.PRINT_TRACE_MAX_LIGHTNESS


def test_light_candidates_are_filtered_out():
    # Okabe-Ito orange is a candidate but misses 3:1 on white.
    assert "#E69F00" not in [c for _l, c in acc.trace_colour_presets("colourblind")]


def test_trace_presets_reject_unknown_kind():
    with pytest.raises(ValueError):
        acc.trace_colour_presets("neon")


# ------------------------------------------------------------------
# Notebook API
# ------------------------------------------------------------------

def test_notebook_stack_colormap_keeps_qualitative_palettes_categorical():
    from opennmr.notebook import plotting as nb_plotting

    cmap = nb_plotting.get_stack_colormap("okabe_ito")
    assert isinstance(cmap, mcolors.ListedColormap)
    np.testing.assert_allclose(np.array(cmap.colors),
                               np.array(acc.ordered_qualitative_colours("okabe_ito")))


def test_package_exports():
    for name in ("friendly_palettes", "describe_accessibility", "trace_colour_presets",
                 "simulate_cvd", "max_safe_series"):
        assert hasattr(plotting, name)


def test_notebook_package_exports_the_same_functions():
    import opennmr.notebook as nb

    assert nb.friendly_palettes is acc.friendly_palettes
    assert nb.describe_accessibility is acc.describe_accessibility
    for name in ("friendly_palettes", "trace_colour_presets", "is_print_safe"):
        assert name in nb.__all__


def test_notebook_stack_plot_accepts_an_accessible_palette():
    # NMR1dplot-style callers resolve colours through get_stack_colormap
    # + a ScalarMappable; the qualitative map must give real palette
    # entries there too.
    import matplotlib.cm as cmx
    import matplotlib.pyplot as plt
    from opennmr.notebook import plotting as nb_plotting

    cmap = nb_plotting.get_stack_colormap("tol_muted")
    mappable = cmx.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=3), cmap=cmap)
    entries = {tuple(np.round(c[:3], 6)) for c in cmap.colors}
    for i in range(3):
        assert tuple(np.round(mappable.to_rgba(i)[:3], 6)) in entries
