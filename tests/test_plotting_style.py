import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opennmr.plotting import style
from opennmr.plotting.style import downsample_2d, downsample_1d


# ============================================================
# downsample_2d -- max-magnitude-preserving, not evenly-spaced
# index-skipping (see the function's own docstring for the real
# EXSY-diagonal bug this fixed)
# ============================================================

def test_downsample_2d_preserves_a_thin_diagonal_ridge():
    # The exact real-world failure mode: an evenly-spaced index pick
    # can step clean over a 1-point-wide ridge at an irrational slope
    # and lose most of it. Deliberately asymmetric shape (matching a
    # real 2D's own F1<<F2), irrational-ish slope so no bin/ridge
    # alignment happens to save the old naive method by coincidence.
    n_y, n_x = 256, 4096
    data = np.zeros((n_y, n_x))
    slope = (n_x / n_y) * 0.9137
    ridge_cols = set()
    for i in range(n_y):
        j = int(round(i * slope)) % n_x
        data[i, j] = 1.0
        ridge_cols.add(j)
    x = np.arange(n_x, dtype=float)
    y = np.arange(n_y, dtype=float)

    data_ds, x_ds, y_ds = downsample_2d(data, x, y, max_points_x=1024, max_points_y=64)

    # Every original row had exactly one ridge point -- almost all of
    # them should still be individually visible (some collisions are
    # expected/acceptable once multiple original rows collapse into
    # the same output row).
    assert np.count_nonzero(data_ds) >= n_y * 0.8


def test_downsample_2d_naive_index_skip_would_have_lost_the_same_ridge():
    # Confirms the test above is actually discriminating -- the OLD
    # evenly-spaced-index approach loses the vast majority of the same
    # ridge, so "preserves most of it" above is a real improvement,
    # not a tautology that would pass either way.
    n_y, n_x = 256, 4096
    data = np.zeros((n_y, n_x))
    slope = (n_x / n_y) * 0.9137
    for i in range(n_y):
        j = int(round(i * slope)) % n_x
        data[i, j] = 1.0

    x_idx = np.linspace(0, n_x - 1, 1024, dtype=int)
    y_idx = np.linspace(0, n_y - 1, 64, dtype=int)
    naive_ds = data[np.ix_(y_idx, x_idx)]

    assert np.count_nonzero(naive_ds) < n_y * 0.8 * (64 / n_y)  # far fewer survive than the fixed version


def test_downsample_2d_preserves_sign():
    data = np.zeros((64, 64))
    data[10, 10] = -5.0
    data[10, 11] = 3.0  # smaller magnitude, same bin-ish neighborhood
    x = np.arange(64, dtype=float)
    y = np.arange(64, dtype=float)

    data_ds, _, _ = downsample_2d(data, x, y, max_points_x=16, max_points_y=16)

    # The largest-magnitude value in the bin containing column 10/11
    # is -5.0 -- must survive as negative, not get sign-dropped.
    assert data_ds.min() == pytest.approx(-5.0)


def test_downsample_2d_returns_unchanged_when_already_within_limits():
    data = np.random.default_rng(0).normal(size=(10, 12))
    x = np.arange(12, dtype=float)
    y = np.arange(10, dtype=float)

    data_ds, x_ds, y_ds = downsample_2d(data, x, y, max_points_x=2000, max_points_y=500)

    np.testing.assert_array_equal(data_ds, data)
    np.testing.assert_array_equal(x_ds, x)
    np.testing.assert_array_equal(y_ds, y)


def test_downsample_2d_handles_axis_length_not_evenly_divisible_by_target():
    # Real acquisition sizes rarely divide evenly into a target point
    # count -- must not crash, and must still land on the requested
    # output shape (trimming a few trailing points rather than failing).
    data = np.random.default_rng(1).normal(size=(1002, 39418))
    x = np.arange(39418, dtype=float)
    y = np.arange(1002, dtype=float)

    data_ds, x_ds, y_ds = downsample_2d(data, x, y, max_points_x=16384, max_points_y=256)

    assert data_ds.shape == (256, 16384)
    assert len(x_ds) == 16384
    assert len(y_ds) == 256


def test_downsample_2d_output_shape_matches_requested_caps():
    data = np.random.default_rng(2).normal(size=(1024, 65536))
    x = np.arange(65536, dtype=float)
    y = np.arange(1024, dtype=float)

    data_ds, x_ds, y_ds = downsample_2d(data, x, y, max_points_x=16384, max_points_y=256)

    assert data_ds.shape == (256, 16384)
    assert len(x_ds) == 16384
    assert len(y_ds) == 256


# ============================================================
# downsample_1d -- max-magnitude-preserving, not evenly-spaced
# index-skipping (the 1D counterpart of downsample_2d above; see the
# function's own docstring for the real "a well-defined peak becomes
# 3-4 points" bug this fixed in the Processing/Compare tabs)
# ============================================================

def _spike_index_missed_by_naive_stride(n, max_points):
    """An index guaranteed to fall *between* two of
    ``downsample_indices``'s own evenly-spaced picks -- the middle of
    the widest gap in ``np.linspace(0, n-1, max_points, dtype=int)`` --
    so a test placing a spike there is actually exercising the "falls
    between two strides" failure mode, not relying on a hand-picked
    constant that happens to dodge it for one particular n/max_points."""
    naive_idx = np.linspace(0, n - 1, max_points, dtype=int)
    gap_i = int(np.argmax(np.diff(naive_idx)))
    mid = (int(naive_idx[gap_i]) + int(naive_idx[gap_i + 1])) // 2
    assert mid not in set(naive_idx.tolist())  # actually missed, not a coincidence
    return mid


def test_downsample_1d_preserves_a_narrow_spike():
    # The exact real-world failure mode: an evenly-spaced index pick
    # can step clean over a narrow, well-digitized peak and lose it
    # almost entirely.
    n = 50000
    spike_idx = _spike_index_missed_by_naive_stride(n, 8000)
    y = np.zeros(n)
    y[spike_idx] = 100.0
    x = np.arange(n, dtype=float)

    x_ds, y_ds = downsample_1d(x, y, max_points=8000)

    assert len(y_ds) == 8000
    assert y_ds.max() == pytest.approx(100.0)


def test_downsample_1d_naive_index_skip_would_have_lost_the_same_spike():
    # Confirms the test above is actually discriminating -- the OLD
    # evenly-spaced-index approach (downsample_indices) misses the
    # same spike, so "preserves the spike" above is a real
    # improvement, not a tautology that would pass either way.
    n = 50000
    spike_idx = _spike_index_missed_by_naive_stride(n, 8000)
    y = np.zeros(n)
    y[spike_idx] = 100.0

    naive_idx = np.linspace(0, n - 1, 8000, dtype=int)
    naive_ds = y[naive_idx]

    assert naive_ds.max() < 100.0


def test_downsample_1d_preserves_sign():
    y = np.zeros(64)
    y[10] = -5.0
    y[11] = 3.0  # smaller magnitude, same bin-ish neighborhood
    x = np.arange(64, dtype=float)

    _, y_ds = downsample_1d(x, y, max_points=16)

    # The largest-magnitude value in the bin containing indices 10/11
    # is -5.0 -- must survive as negative, not get sign-dropped.
    assert y_ds.min() == pytest.approx(-5.0)


def test_downsample_1d_returns_unchanged_when_already_within_limits():
    y = np.random.default_rng(0).normal(size=64)
    x = np.arange(64, dtype=float)

    x_ds, y_ds = downsample_1d(x, y, max_points=8000)

    np.testing.assert_array_equal(x_ds, x)
    np.testing.assert_array_equal(y_ds, y)


def test_downsample_1d_handles_length_not_evenly_divisible_by_target():
    # Real acquisition sizes rarely divide evenly into a target point
    # count -- must not crash, and must still land on the requested
    # output length (trimming a few trailing points rather than
    # failing).
    y = np.random.default_rng(1).normal(size=39418)
    x = np.arange(39418, dtype=float)

    x_ds, y_ds = downsample_1d(x, y, max_points=16384)

    assert len(y_ds) == 16384
    assert len(x_ds) == 16384


def test_downsample_1d_x_none_returns_none_x_and_correct_length_y():
    # x=None is the fast in-place-refresh path (re-plotting a phased
    # spectrum's y-values without re-deriving the unchanged x already
    # on screen) -- see the function's own docstring.
    y = np.random.default_rng(2).normal(size=50000)

    x_ds, y_ds = downsample_1d(None, y, max_points=8000)

    assert x_ds is None
    assert len(y_ds) == 8000


def test_downsample_1d_x_none_still_finds_a_relocated_peak():
    # Bin *boundaries* depend only on len(y)/max_points, never on
    # data values -- so re-calling with x=None after the underlying
    # peak has moved (e.g. a phase change) must still find the new
    # peak, not silently reuse a stale bin choice from before.
    n = 50000
    y1 = np.zeros(n)
    y1[5000] = 10.0
    _, y1_ds = downsample_1d(None, y1, max_points=8000)
    assert y1_ds.max() == pytest.approx(10.0)

    y2 = np.zeros(n)
    y2[45000] = 10.0
    _, y2_ds = downsample_1d(None, y2, max_points=8000)
    assert y2_ds.max() == pytest.approx(10.0)
    assert np.argmax(y1_ds) != np.argmax(y2_ds)


# ============================================================
# fit_labels_within_figure -- axis labels must not fall off the
# figure when the canvas is small (see its own docstring)
# ============================================================

def _labelled_figure(width, height):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(np.arange(10), np.arange(10))
    ax.set_xlabel("Chemical shift / ppm")
    ax.set_ylabel("Intensity")
    return fig, ax


def _font_sizes(fig):
    """Every text size the label fitter controls: the persistent axis
    labels and titles, and the tick labels it now resizes through the
    axis rather than per artist (see style._shrink_text_once)."""
    sizes = {round(a.get_fontsize(), 2) for a in style._axis_label_artists(fig)}
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            sizes.add(round(style._tick_labelsize(axis), 2))
    return sizes


def test_fit_labels_leaves_a_roomy_figure_untouched():
    # The safeguard must cost nothing when nothing is wrong -- an
    # earlier version tested tick labels too and drove a perfectly good
    # 6x4in plot to the 5pt floor, because the label at either end of
    # an axis routinely overhangs the figure edge by a pixel or two.
    import matplotlib.pyplot as plt

    fig, _ax = _labelled_figure(6, 4)
    before = _font_sizes(fig)

    result = style.fit_labels_within_figure(fig)

    assert result["fitted"]
    assert result["steps"] == 0
    assert _font_sizes(fig) == before
    plt.close(fig)


def test_fit_labels_shrinks_only_as_far_as_it_has_to():
    import matplotlib.pyplot as plt

    fig, _ax = _labelled_figure(1.4, 0.9)
    before = max(_font_sizes(fig))
    assert style.clipped_labels(fig)  # genuinely clipped to start with

    result = style.fit_labels_within_figure(fig)

    assert result["fitted"]
    assert style.clipped_labels(fig) == []
    after = max(_font_sizes(fig))
    assert after < before                               # did shrink...
    assert after > style.LABEL_FIT_MIN_FONTSIZE         # ...but not to the floor
    plt.close(fig)


def test_fit_labels_reports_what_it_still_cannot_fit():
    # Below a certain size no font helps, and that is a real "make this
    # bigger" situation the caller has to be able to surface rather
    # than silently render labels off the edge.
    import matplotlib.pyplot as plt

    fig, _ax = _labelled_figure(0.7, 0.45)

    result = style.fit_labels_within_figure(fig)

    assert not result["fitted"]
    assert result["clipped"]
    assert min(_font_sizes(fig)) == pytest.approx(style.LABEL_FIT_MIN_FONTSIZE)
    plt.close(fig)


def test_fit_labels_does_not_let_tight_layouts_warning_escape():
    # matplotlib's "Tight layout not applied" UserWarning names no
    # spectrum and offers no remedy; it is the *condition* this
    # function exists to answer, so it is captured and acted on rather
    # than printed at the user.
    import warnings

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(12, 1, figsize=(6, 2.5))
    for ax in axes:
        ax.set_xlabel("Chemical shift / ppm")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        style.fit_labels_within_figure(fig)

    assert not [w for w in caught if "ight layout" in str(w.message)]
    plt.close(fig)


def test_clipped_labels_ignores_tick_labels():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(10), np.arange(10))
    ax.set_xlabel("")   # no axis labels at all...
    ax.set_ylabel("")
    fig.tight_layout()

    # ...so whatever the tick labels are doing at the axis ends, there
    # is nothing here that counts as clipped.
    assert style.clipped_labels(fig) == []
    plt.close(fig)


# ============================================================
# Too-light colours removed from every colormap
# ============================================================

_SWEEP = np.linspace(0.0, 1.0, 256)


def _luminances(cmap):
    return np.array([style.relative_luminance(cmap(x)[:3]) for x in _SWEEP])


def test_relative_luminance_is_perceptual_not_a_channel_average():
    # The whole reason for the weighted-linear form: yellow and blue
    # have the same mean channel value and wildly different apparent
    # brightness. A naive average would keep yellow and discard blue --
    # exactly backwards.
    assert style.relative_luminance("#FFFF00") > 0.9
    assert style.relative_luminance("#0000FF") < 0.1
    assert style.relative_luminance("#000000") == pytest.approx(0.0)
    assert style.relative_luminance("#FFFFFF") == pytest.approx(1.0)


def test_contrast_against_white_matches_the_threshold_derivation():
    # MAX_LINE_LUMINANCE is defined as the luminance giving 3:1.
    at_cap = style.contrast_against_white("#000000")
    assert at_cap == pytest.approx(21.0)
    assert 1.05 / (style.MAX_LINE_LUMINANCE + 0.05) == pytest.approx(3.0)


def test_is_too_light_uses_the_shared_cap():
    assert style.is_too_light("#FDE725") is True      # viridis' yellow end
    assert style.is_too_light("#440154") is False     # viridis' purple end
    # And is overridable per call.
    assert style.is_too_light("#440154", max_luminance=0.0) is True


@pytest.mark.parametrize("name", style.CONTOUR_CMAP_NAMES)
def test_every_contour_colormap_clears_three_to_one(name):
    # The headline guarantee: no colour this package draws a contour
    # with is invisible on white. Before this, the lightest entries of
    # these maps sat at 1.0-1.4:1.
    worst = _luminances(style.get_contour_cmap(name)).max()
    assert worst <= style.MAX_LINE_LUMINANCE + 1e-9


@pytest.mark.parametrize("name", ["viridis", "coolwarm", "Greys", "YlGn", "Spectral"])
def test_contour_colormaps_were_actually_too_light_before(name):
    # Guards the test above from becoming vacuous: these maps really do
    # contain colours the filter has to remove.
    assert _luminances(plt.get_cmap(name)).max() > style.MAX_LINE_LUMINANCE


def test_filtered_contour_colormap_keeps_its_dark_end():
    # Trimming must not shift the map wholesale -- the dark end is the
    # part that was fine, and it stays put.
    base, filtered = plt.get_cmap("viridis"), style.get_contour_cmap("viridis")
    np.testing.assert_allclose(filtered(0.0)[:3], base(0.0)[:3], atol=0.02)


def test_filtering_can_be_disabled():
    full = style.get_contour_cmap("viridis", max_luminance=1.0)
    np.testing.assert_allclose(full(1.0)[:3], plt.get_cmap("viridis")(1.0)[:3], atol=0.02)


@pytest.mark.parametrize("name", ["viridis", "coolwarm", "Greys", "YlGn", "plasma"])
@pytest.mark.parametrize("n", [1, 2, 5, 12])
def test_stack_colors_are_all_dark_enough(name, n):
    colors = style.stack_colors(n, name)
    assert len(colors) == n
    for color in colors:
        assert style.relative_luminance(color[:3]) <= style.MAX_LINE_LUMINANCE + 1e-9


def test_stack_colors_remain_distinguishable():
    # Filtering must not collapse a stack onto one colour -- the point
    # of a stack is telling the traces apart.
    colors = style.stack_colors(6, "viridis")
    assert len({tuple(np.round(c, 4)) for c in colors}) == 6


def test_a_single_trace_gets_the_darkest_usable_colour():
    # One trace has nothing to be distinguished from, so it should not
    # be handed a map's own lightest usable end.
    only = style.stack_colors(1, "Greys")[0]
    darkest = min(style.relative_luminance(c[:3]) for c in style.stack_colors(8, "Greys"))
    assert style.relative_luminance(only[:3]) == pytest.approx(darkest, abs=0.1)


def test_black_stack_is_untouched():
    assert all(style.relative_luminance(c[:3]) == pytest.approx(0.0)
               for c in style.stack_colors(4, "black"))


def test_stack_colors_of_zero_is_empty():
    assert style.stack_colors(0, "viridis") == []


def test_an_all_white_colormap_is_darkened_rather_than_returned_unusable():
    # The last-resort path: nothing in the map clears the cap, so
    # scaling toward black is the only way to return something usable.
    import matplotlib.colors as mcolors

    white = mcolors.LinearSegmentedColormap.from_list("white", ["#ffffff", "#fefefe"])
    colors = [style._darkened(white(x)) for x in (0.0, 1.0)]
    for color in colors:
        assert style.relative_luminance(color[:3]) <= style.MAX_LINE_LUMINANCE + 1e-9


def test_usable_positions_never_come_back_empty():
    import matplotlib.colors as mcolors

    white = mcolors.LinearSegmentedColormap.from_list("white", ["#ffffff", "#ffffff"])
    assert style.usable_cmap_positions(white).size >= 1


def test_usable_positions_find_the_light_part_wherever_it_sits():
    # Light at the top (viridis), at the bottom (Greys) and in the
    # middle (coolwarm) -- a fixed "trim the last 20%" rule would fix
    # only the first.
    assert style.usable_cmap_positions(plt.get_cmap("viridis")).max() < 0.95
    assert style.usable_cmap_positions(plt.get_cmap("Greys")).min() > 0.05
    middle = style.usable_cmap_positions(plt.get_cmap("coolwarm"))
    assert np.max(np.diff(middle)) > 0.05   # a gap, i.e. the pale middle is gone


def test_qualitative_colors_are_distinct_and_dark_enough():
    colors = style.qualitative_colors(10)
    assert len({tuple(np.round(c, 4)) for c in colors}) == 10
    for color in colors:
        assert style.relative_luminance(color[:3]) <= style.MAX_LINE_LUMINANCE + 1e-9
    # tab10's own olive and cyan only reach 2.0:1 unaided, so at least
    # one entry must actually have been darkened.
    raw = [plt.get_cmap("tab10")(i) for i in range(10)]
    assert any(style.relative_luminance(c[:3]) > style.MAX_LINE_LUMINANCE for c in raw)


def test_qualitative_colors_cycle_past_the_palette_length():
    # A named palette cycles; the default (None) picks a palette that is
    # colour-blind friendly for n instead (2026-10-01).
    colors = style.qualitative_colors(12, "tab10")
    assert len(colors) == 12
    np.testing.assert_allclose(colors[10], colors[0])


def test_readable_cmap_accepts_a_name_or_an_object():
    from_name = style.readable_cmap("viridis")
    from_object = style.readable_cmap(plt.get_cmap("viridis"))
    np.testing.assert_allclose(from_name(0.5)[:3], from_object(0.5)[:3])


def test_the_legacy_notebook_api_uses_the_same_filter():
    # opennmr.notebook is publicly exported and had its own duplicated
    # copies of these functions. It now delegates, so the rule lives in
    # one place and the two cannot drift.
    from opennmr.notebook.plotting import get_contour_cmap as legacy_contour
    from opennmr.notebook.plotting import get_stack_colormap as legacy_stack

    assert _luminances(legacy_contour("viridis")).max() <= style.MAX_LINE_LUMINANCE + 1e-9
    assert _luminances(legacy_stack("Greys")).max() <= style.MAX_LINE_LUMINANCE + 1e-9
    # And its two special cases are untouched.
    assert style.relative_luminance(legacy_stack("black")(0.5)[:3]) == pytest.approx(0.0)
    assert legacy_stack("diverging") is not None


def test_shrinking_ticks_goes_through_the_axis_not_the_artists():
    """A Tick's label is not a persistent artist: matplotlib keeps a
    pool and creates more from the axis's *stored* tick parameters when
    a view needs more ticks than the pool holds. Writing a size onto
    the artists left that stored size at full size, so a later pan or
    zoom brought some ticks back big while the axis labels stayed
    small -- reported as a DOSY with tiny labels but correct ticks."""
    fig, ax = _labelled_figure(1.4, 1.0)
    ax.set_xlabel("delta / ppm", fontsize=16)
    ax.set_ylabel("D", fontsize=16)
    ax.tick_params(axis="both", which="both", labelsize=16)

    result = style.fit_labels_within_figure(fig)
    assert result["steps"] > 0, "this figure should have been too small to fit"

    shrunk = ax.xaxis.label.get_fontsize()
    assert shrunk < 16
    # The axis itself now carries the smaller size, so any tick created
    # later comes out at it too -- labels and ticks stay in step.
    assert style._tick_labelsize(ax.xaxis) == pytest.approx(shrunk)
    assert style._tick_labelsize(ax.yaxis) == pytest.approx(shrunk)


def test_tick_and_label_sizes_stay_together_across_a_view_change():
    fig, ax = _labelled_figure(1.4, 1.0)
    ax.set_xlabel("delta / ppm", fontsize=16)
    ax.set_ylabel("D", fontsize=16)
    ax.tick_params(axis="both", which="both", labelsize=16)
    style.fit_labels_within_figure(fig)

    label_size = ax.xaxis.label.get_fontsize()
    ax.set_xlim(0.2, 0.4)          # regenerates ticks
    fig.canvas.draw()

    drawn = {round(t.get_fontsize(), 2) for t in ax.get_xticklabels() if t.get_text()}
    assert drawn == {round(label_size, 2)}


# ============================================================
# apply_default_font -- the GUI's one-time global typeface default
# (see the module docstring for why setup_plotting itself never runs
# from GUI code, and apply_default_font's own docstring for why
# typeface is the one piece of that safe to set globally anyway)
# ============================================================

def test_apply_default_font_sets_family_and_mathtext():
    plt.rcdefaults()
    try:
        style.apply_default_font("Arial")
        assert plt.rcParams["font.family"] == ["sans-serif"]
        assert plt.rcParams["font.sans-serif"][0] == "Arial"
        assert plt.rcParams["mathtext.fontset"] == "custom"
        assert plt.rcParams["mathtext.rm"] == "Arial"
        assert plt.rcParams["mathtext.it"] == "Arial:italic"
        assert plt.rcParams["mathtext.bf"] == "Arial:bold"
        assert plt.rcParams["mathtext.default"] == "regular"
    finally:
        plt.rcdefaults()


def test_apply_default_font_falls_back_gracefully_when_the_font_is_missing():
    # A machine without "Arial" installed still gets *a* sans-serif,
    # not a warning and a silent substitution to whatever matplotlib
    # feels like -- explicit user request: "add Arial as the default
    # font, not the unpleasant matplotlib default".
    plt.rcdefaults()
    try:
        style.apply_default_font("Arial")
        fallbacks = plt.rcParams["font.sans-serif"]
        assert "DejaVu Sans" in fallbacks
        assert "sans-serif" in fallbacks
    finally:
        plt.rcdefaults()


def test_apply_default_font_does_not_touch_axis_or_tick_sizing():
    # The one thing that must stay false: this is not setup_plotting.
    # A GUI canvas keeps its own linewidth/tick sizing per instance
    # (see MplCanvas), and a global rcParam change here would silently
    # restyle every open canvas the same way calling setup_plotting
    # from GUI code would.
    plt.rcdefaults()
    default_linewidth = plt.rcParams["axes.linewidth"]
    try:
        style.apply_default_font("Arial")
        assert plt.rcParams["axes.linewidth"] == default_linewidth
    finally:
        plt.rcdefaults()


def test_setup_plotting_still_sets_its_own_tick_and_axis_rcparams():
    # Regression guard for factoring the font/mathtext rcParams out of
    # setup_plotting into apply_default_font -- everything else it did
    # before must still happen.
    plt.rcdefaults()
    try:
        result = style.setup_plotting(font="Arial", fontsize=16)
        assert plt.rcParams["axes.labelsize"] == 16
        assert plt.rcParams["xtick.direction"] == "in"
        assert plt.rcParams["axes.linewidth"] == pytest.approx(result["lw"])
        # And it still gets the same typeface apply_default_font would.
        assert plt.rcParams["mathtext.rm"] == "Arial"
        assert plt.rcParams["font.sans-serif"][0] == "Arial"
    finally:
        plt.rcdefaults()


def test_default_qualitative_colors_are_colourblind_friendly():
    from opennmr.plotting import accessibility
    for n in range(2, 10):
        colors = style.qualitative_colors(n)
        assert len(colors) == n
        assert accessibility.is_colourblind_safe([c[:3] for c in colors]), n
        for color in colors:
            assert style.relative_luminance(color[:3]) <= style.MAX_LINE_LUMINANCE + 1e-9
