import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import ConnectionPatch
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.plotting import twod


def _make_2d(n_f1=20, n_f2=30, pseudo=False, homonuclear=False, nucleus="1H", nucleus_indirect="13C"):
    x_ppm = np.linspace(10, 0, n_f2)
    xx, yy = np.meshgrid(np.arange(n_f2), np.arange(n_f1))
    data = np.exp(-(((xx - n_f2 / 2) ** 2 + (yy - n_f1 / 2) ** 2)) / 20.0) * 1000

    if pseudo:
        f1_axis = NMRAxis(kind="index", index=np.arange(1, n_f1 + 1))
    else:
        y_ppm = np.linspace(150, 100, n_f1)
        f1_axis = NMRAxis(kind="frequency", ppm=y_ppm, hz=y_ppm * 100.0)

    f2_axis = NMRAxis(kind="frequency", ppm=x_ppm, hz=x_ppm * 400.0)
    metadata = NMRMetadata(
        vendor="bruker", nucleus=nucleus, nucleus_indirect=nucleus_indirect,
        sfo1=400.0, sfo1_indirect=100.0, pseudo_2d=pseudo, homonuclear=homonuclear,
    )
    return NMRSpectrum(data=data, axes=[f1_axis, f2_axis], metadata=metadata)


def test_prepare_contour_power_mode_top_level_never_exceeds_data_max():
    # Regression test: the exponent used to be `power * i / clevels`,
    # which only reaches exactly `power` (not 1) at the top level --
    # for power > 1 (the default is 1.25) that overshoots the real
    # data max, so the top several "contour levels" sat above every
    # real data point and were never drawn, squeezing the visible
    # ones into a smaller range than intended. This looked like a 2D
    # dataset "starting out scaled way too high" (barely anything
    # visible) until the threshold was manually scrolled down a lot.
    spec = _make_2d()
    _, _, _, thresvec, _, _ = twod._prepare_contour(spec, factor=1e-3, clevels=20, level_mode="power", power=1.25)
    assert thresvec[-1] == pytest.approx(np.amax(spec.data))
    assert all(level <= np.amax(spec.data) + 1e-6 for level in thresvec)


def test_prepare_contour_power_mode_bunches_toward_noise_floor():
    # power > 1 should push more levels toward thresh_min (the noise
    # floor), not toward thresh_max -- matching this function's own
    # docstring ("steeper/more levels bunched near the noise floor as
    # power increases past 1").
    spec = _make_2d()
    _, _, _, linear_like, _, _ = twod._prepare_contour(spec, factor=1e-3, clevels=20, level_mode="power", power=1.0)
    _, _, _, bunched, _, _ = twod._prepare_contour(spec, factor=1e-3, clevels=20, level_mode="power", power=1.25)
    # Every level below the top one should sit lower (closer to the
    # noise floor) with the steeper power than with power=1.
    assert all(b <= lin for b, lin in zip(bunched[:-1], linear_like[:-1]))


def test_prepare_contour_neg_thresvec_none_when_all_positive():
    spec = _make_2d()  # a pure positive Gaussian, no negative content at all
    _, _, _, _, _, neg_thresvec = twod._prepare_contour(spec)
    assert neg_thresvec is None


def test_prepare_contour_neg_thresvec_mirrors_positive_side():
    # Real user request: colour contrast for positive and negative
    # contours -- needs its own, correctly-signed set of levels for
    # the negative side first. Built from a spectrum with a real
    # negative-going lobe (mimicking a dispersive/incoherent artifact
    # sitting next to a well-phased positive peak), not just an
    # arbitrary negative constant.
    spec = _make_2d()
    spec.data = spec.data - 1.5 * np.amax(spec.data) * np.exp(
        -(((np.arange(spec.data.shape[1]) - 5) ** 2)) / 10.0
    )
    assert np.amin(spec.data) < 0
    _, _, _, thresvec, _, neg_thresvec = twod._prepare_contour(spec)
    assert neg_thresvec is not None
    assert len(neg_thresvec) == len(thresvec)
    assert all(level < 0 for level in neg_thresvec)
    # Magnitude increasing away from zero, same convention as the
    # positive side's own thresvec (small-to-large magnitude, not
    # numeric order -- plot_2d_contour itself sorts numerically right
    # before handing levels to matplotlib's own contour(), which
    # test_plot_2d_contour_draws_negative_contours_in_contrasting_colour
    # already exercises without matplotlib raising on level ordering).
    assert neg_thresvec == sorted(neg_thresvec, key=abs)
    assert abs(neg_thresvec[0]) < abs(neg_thresvec[-1])


def test_plot_2d_contour_draws_negative_contours_in_contrasting_colour(monkeypatch):
    spec = _make_2d()
    spec.data = spec.data - 1.5 * np.amax(spec.data) * np.exp(
        -(((np.arange(spec.data.shape[1]) - 5) ** 2)) / 10.0
    )
    calls = []
    real_contour = plt.Axes.contour

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("colors"))
        return real_contour(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "contour", spy)
    fig, ax, _ = twod.plot_2d_contour(spec, show_projections=False)
    assert "black" in calls
    # Teal, not red: red read as an alarm colour and dominated the black
    # positive contours visually. Asserted via the module constant so
    # the colour can be retuned in one place.
    assert twod.NEGATIVE_CONTOUR_COLOUR in calls
    assert twod.NEGATIVE_CONTOUR_COLOUR != "red"
    plt.close(fig)


def test_plot_2d_contour_conventional_standalone():
    spec = _make_2d()
    fig, ax, proj_axes = twod.plot_2d_contour(spec)
    assert fig is not None
    assert proj_axes["f1"] is not None and proj_axes["f2"] is not None
    plt.close(fig)


def test_plot_2d_contour_pseudo2d_no_yticks():
    spec = _make_2d(pseudo=True)
    fig, ax, proj_axes = twod.plot_2d_contour(spec, show_projections=False)
    assert list(ax.get_yticks()) == []
    assert proj_axes == {"f1": None, "f2": None}
    plt.close(fig)


def test_plot_2d_contour_onto_existing_ax():
    fig, ax = plt.subplots()
    spec = _make_2d()
    twod.plot_2d_contour(spec, ax=ax, show_projections=False)
    plt.close(fig)


def test_multi_plot_creates_one_panel_per_spectrum():
    specs = [_make_2d(), _make_2d()]
    fig, axes = twod.multi_plot(specs, layout="row", show_projections=False)
    assert len(axes) == 2
    plt.close(fig)


def test_multi_plot_rejects_figure_level_kwargs():
    specs = [_make_2d(), _make_2d()]
    with pytest.raises(ValueError):
        twod.multi_plot(specs, ax=None)


def test_overlay_plot_two_compatible_spectra():
    specs = [_make_2d(), _make_2d()]
    fig, ax = twod.overlay_plot(specs, labels=["a", "b"])
    plt.close(fig)


def test_overlay_plot_rejects_mismatched_nuclides():
    a = _make_2d(nucleus="1H", nucleus_indirect="13C")
    b = _make_2d(nucleus="1H", nucleus_indirect="15N")
    with pytest.raises(ValueError):
        twod.overlay_plot([a, b])


def test_overlay_plot_rejects_pseudo_vs_conventional():
    a = _make_2d(pseudo=False)
    b = _make_2d(pseudo=True)
    with pytest.raises(ValueError):
        twod.overlay_plot([a, b])


def test_overlay_plot_wrong_count_raises():
    specs = [_make_2d()]
    with pytest.raises(ValueError):
        twod.overlay_plot(specs)


def test_cap_contour_points_leaves_normal_sized_requests_alone():
    # The ceiling is a safety valve for outliers, not a downsampling
    # policy -- full-resolution Spectrum-tab rendering is an explicit
    # standing user instruction, so every ordinary dataset must come
    # back untouched. These are the real shapes in this project's own
    # test data (COSY, bruker_T2, bruker_PFG), all far below the cap.
    for max_x, max_y in [(2048, 2048), (32768, 64), (262144, 16), (1, 1)]:
        assert twod._cap_contour_points(max_x, max_y) == (max_x, max_y)

    # Exactly at the ceiling is still "within" it, not over. Derived
    # from the constant rather than hardcoded, so raising the ceiling
    # (as happened once already, on user request) doesn't leave this
    # asserting a stale number.
    at_cap = (twod._MAX_CONTOUR_POINTS // 1024, 1024)
    assert at_cap[0] * at_cap[1] == twod._MAX_CONTOUR_POINTS
    assert twod._cap_contour_points(*at_cap) == at_cap


def test_max_contour_points_is_a_quarter_of_the_exsy_benchmark():
    # Explicit user instruction: "restrict any dataset to 0.25x the
    # EXSY's number of points". Pinned against the benchmark's own point
    # count rather than a literal, so the relationship survives any
    # later edit -- this ceiling has already been moved once (briefly to
    # half, on an earlier reading) and corrected back.
    assert twod._EXSY_BENCHMARK_POINTS == 1002 * 65536
    assert twod._MAX_CONTOUR_POINTS == twod._EXSY_BENCHMARK_POINTS // 4


def test_cap_contour_points_never_returns_more_than_the_ceiling():
    # Regression test for a real bug in this function's own first
    # draft, caught here rather than in review: reducing F2 alone is
    # not always enough, because F2 has a floor (F1 is far smaller and
    # far more sensitive to losing rows, so the cap comes out of F2
    # first) -- a request whose F1 is itself oversized used to come
    # back 16x over the ceiling.
    oversized = [
        (65536, 1002),    # the real EXSY benchmark
        (4096, 8192),     # F1 oversized once F2 is at its floor
        (65536, 65536),   # both oversized -- the case that was broken
        (100, 500_000),   # F2 already below the floor, F1 enormous
    ]
    for max_x, max_y in oversized:
        capped_x, capped_y = twod._cap_contour_points(max_x, max_y)
        assert capped_x * capped_y <= twod._MAX_CONTOUR_POINTS, (max_x, max_y)
        assert capped_x <= max_x and capped_y <= max_y  # only ever reduces


def test_cap_contour_points_reduces_the_direct_dimension_first():
    # F1 (max_y) is usually far smaller and carries far fewer points to
    # spare, so a dataset that is over only because F2 is huge must
    # keep every F1 row.
    capped_x, capped_y = twod._cap_contour_points(65536, 1002)
    assert capped_y == 1002
    assert capped_x < 65536


def test_prepare_contour_caps_a_very_large_spectrum_but_keeps_its_peak():
    # Even when capped, downsample_2d keeps each bin's largest-magnitude
    # value (see its own docstring for the EXSY-diagonal bug that
    # motivated that), so the contour levels -- which are computed from
    # the *full* data -- still have real data reaching them.
    spectrum = _make_2d(n_f1=8, n_f2=64)
    monkey_cap = 128  # far below this fixture, to exercise the path cheaply
    original = twod._MAX_CONTOUR_POINTS
    original_floor = twod._MIN_CAPPED_CONTOUR_X
    try:
        twod._MAX_CONTOUR_POINTS = monkey_cap
        twod._MIN_CAPPED_CONTOUR_X = 4
        _, _, contour_data, _, _, _ = twod._prepare_contour(
            spectrum, contour_fraction_x=1.0, contour_fraction_y=1.0,
            ncontourpts_x=10**9, ncontourpts_y=10**9,
        )
    finally:
        twod._MAX_CONTOUR_POINTS = original
        twod._MIN_CAPPED_CONTOUR_X = original_floor

    assert contour_data.size <= monkey_cap
    assert np.abs(contour_data).max() == pytest.approx(np.abs(spectrum.data).max())


def test_prepare_contour_below_the_cap_is_the_untouched_full_resolution_array():
    # Not merely "the same shape" -- the identical array, so a
    # full-resolution request really is bit-for-bit what gets contoured.
    spectrum = _make_2d(n_f1=20, n_f2=30)
    _, _, contour_data, _, _, _ = twod._prepare_contour(
        spectrum, contour_fraction_x=1.0, contour_fraction_y=1.0,
        ncontourpts_x=10**9, ncontourpts_y=10**9,
    )
    assert np.array_equal(contour_data, spectrum.data)


def _make_signed_2d(n_f1=64, n_f2=128):
    """A 2D with one known-*positive* peak and one known-*negative*
    peak, at different F1 and F2 positions so each projection's own
    polarity can be read off independently."""
    yy, xx = np.mgrid[0:n_f1, 0:n_f2]
    data = (
        1000 * np.exp(-(((xx - n_f2 // 4) ** 2) / 8 + ((yy - n_f1 // 4) ** 2) / 8))
        - 1000 * np.exp(-(((xx - 3 * n_f2 // 4) ** 2) / 8 + ((yy - 3 * n_f1 // 4) ** 2) / 8))
    )
    return NMRSpectrum(
        data=data,
        axes=[
            NMRAxis(kind="frequency", ppm=np.linspace(150, 100, n_f1), hz=np.linspace(150, 100, n_f1) * 100),
            NMRAxis(kind="frequency", ppm=np.linspace(10, 0, n_f2), hz=np.linspace(10, 0, n_f2) * 400),
        ],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", nucleus_indirect="13C", sfo1=400.13),
    )


def test_projections_preserve_sign_and_agree_with_each_other():
    # Real user report: "the phase of the contours and the projections
    # are definitely inverted... it's more about the matching of the
    # contour phase to the projection phase". The F1 projection used to
    # be plotted negated (`-f1_projection`) while F2 was not, so a
    # positive (black-contour) peak showed as a negative-going trace in
    # the F1 strip and vice versa -- the two projections disagreed with
    # each other, and F1 disagreed with the contour beside it.
    spectrum = _make_signed_2d()
    n_f1, n_f2 = spectrum.data.shape
    fig, ax = plt.subplots()
    try:
        _, _, proj_axes = twod.plot_2d_contour(
            spectrum, ax=ax, contour_fraction_x=1.0, contour_fraction_y=1.0,
            ncontourpts_x=10**9, ncontourpts_y=10**9, projection_mode="sum",
        )
        f2_plotted = proj_axes["f2"].lines[0].get_ydata()
        f1_plotted = proj_axes["f1"].lines[0].get_xdata()
    finally:
        plt.close(fig)

    # F2: positive peak reads positive, negative peak reads negative.
    assert f2_plotted[n_f2 // 4] > 0
    assert f2_plotted[3 * n_f2 // 4] < 0
    # F1: the same way round -- this is the assertion that fails on the
    # pre-fix negated version.
    assert f1_plotted[n_f1 // 4] > 0
    assert f1_plotted[3 * n_f1 // 4] < 0


def test_f1_projection_peaks_still_point_away_from_the_contour():
    # The negation the fix removed was doing real layout work: this
    # panel sits left of the contour with intensity along x, and
    # negating pointed positive peaks leftward, away from it. The fix
    # reverses the *axis* instead of the data, so this has to stay
    # true -- otherwise the polarity fix would have silently flipped
    # the panel's orientation as a side effect.
    spectrum = _make_signed_2d()
    fig, ax = plt.subplots()
    try:
        _, _, proj_axes = twod.plot_2d_contour(
            spectrum, ax=ax, contour_fraction_x=1.0, contour_fraction_y=1.0,
            ncontourpts_x=10**9, ncontourpts_y=10**9, projection_mode="sum",
        )
        xlim = proj_axes["f1"].get_xlim()
    finally:
        plt.close(fig)

    assert xlim[0] > xlim[1], "F1 projection axis must descend, so peaks point away from the contour"
    # ...and the baseline must remain the xmax (second) end, which is
    # what MplCanvas._zoom_projection_intensity pins when zooming.
    assert abs(xlim[1]) < abs(xlim[0])


def test_projection_max_mode_is_unaffected_by_the_sign_fix():
    # 'max' mode (the pseudo-2D default) takes max(abs(...)), so it is
    # positive by construction -- it must still render, and still on a
    # descending F1 axis.
    spectrum = _make_signed_2d()
    fig, ax = plt.subplots()
    try:
        _, _, proj_axes = twod.plot_2d_contour(
            spectrum, ax=ax, contour_fraction_x=1.0, contour_fraction_y=1.0,
            ncontourpts_x=10**9, ncontourpts_y=10**9, projection_mode="max",
        )
        f1_plotted = proj_axes["f1"].lines[0].get_xdata()
        xlim = proj_axes["f1"].get_xlim()
    finally:
        plt.close(fig)

    assert np.max(f1_plotted) > 0
    assert xlim[0] > xlim[1]


def test_max_mode_projections_respond_to_a_phase_inversion():
    # End-to-end through plot_2d_contour, on the pseudo-2D default
    # ('auto' -> 'max'), since that is the path the report came from.
    spectrum = _make_signed_2d()
    spectrum.metadata.pseudo_2d = True
    spectrum.axes[0] = NMRAxis(kind="index", index=np.arange(1, spectrum.data.shape[0] + 1))

    def _projections(data):
        spec = NMRSpectrum(data=data, axes=spectrum.axes, metadata=spectrum.metadata)
        fig, ax = plt.subplots()
        try:
            _, _, proj = twod.plot_2d_contour(
                spec, ax=ax, contour_fraction_x=1.0, contour_fraction_y=1.0,
                ncontourpts_x=10**9, ncontourpts_y=10**9,
            )
            return proj["f2"].lines[0].get_ydata().copy(), proj["f1"].lines[0].get_xdata().copy()
        finally:
            plt.close(fig)

    f2, f1 = _projections(spectrum.data)
    f2_flipped, f1_flipped = _projections(-spectrum.data)

    # Must actually move -- this is the assertion that fails outright
    # on the pre-fix max(abs(...)) version, which was bit-identical.
    assert not np.allclose(f2, f2_flipped)
    assert not np.allclose(f1, f1_flipped)
    # ...and at a column whose sign is known from the fixture, it must
    # change sign rather than merely differ. Checked at that specific
    # column, not at the projection's global argmax: this fixture's two
    # peaks are equal and opposite, so the global peak simply swaps
    # between them under a flip and its sign is uninformative.
    positive_peak_col = spectrum.data.shape[1] // 4
    assert f2[positive_peak_col] > 0
    assert f2_flipped[positive_peak_col] < 0


def test_projections_default_to_the_positive_maximum_not_the_sum():
    # Real user request: "the 2D projections should probably just be the
    # positive projections (not the sum)". A sum projection of a
    # phase-sensitive spectrum cancels its own positive and negative
    # lobes against each other and washes the peaks out.
    spectrum = _make_signed_2d()
    n_f1, n_f2 = spectrum.data.shape
    fig, ax = plt.subplots()
    try:
        _, _, proj = twod.plot_2d_contour(
            spectrum, ax=ax, contour_fraction_x=1.0, contour_fraction_y=1.0,
            ncontourpts_x=10**9, ncontourpts_y=10**9,   # 'auto'
        )
        auto_f2 = proj["f2"].lines[0].get_ydata().copy()
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        _, _, proj = twod.plot_2d_contour(
            spectrum, ax=ax, contour_fraction_x=1.0, contour_fraction_y=1.0,
            ncontourpts_x=10**9, ncontourpts_y=10**9, projection_mode="max",
        )
        explicit_f2 = proj["f2"].lines[0].get_ydata().copy()
    finally:
        plt.close(fig)

    # 'auto' now resolves to 'max' for a conventional 2D as well, not
    # just pseudo-2D.
    np.testing.assert_allclose(auto_f2, explicit_f2)
    # ...and 'max' is the positive maximum, so the positive peak
    # dominates its column rather than being cancelled by the negative
    # one a sum would fold in.
    assert explicit_f2[n_f2 // 4] > 0


def test_sum_projection_is_still_available_explicitly():
    spectrum = _make_signed_2d()
    fig, ax = plt.subplots()
    try:
        _, _, proj = twod.plot_2d_contour(
            spectrum, ax=ax, contour_fraction_x=1.0, contour_fraction_y=1.0,
            ncontourpts_x=10**9, ncontourpts_y=10**9, projection_mode="sum",
        )
        summed = proj["f2"].lines[0].get_ydata().copy()
    finally:
        plt.close(fig)
    expected = np.sum(spectrum.data, axis=0)
    np.testing.assert_allclose(summed - summed.min(), (expected - expected.min()), rtol=1e-6, atol=1e-6)


def test_contour_levels_are_scaled_to_the_greatest_absolute_value():
    # Real user report: "the negative contours look really prominent vs
    # the positive. can the size of the contours be based on the
    # greatest absolute value, not the greatest negative value?"
    #
    # The negative side used to be scaled to amin(data) independently,
    # so on a spectrum with a big positive peak and a small negative one
    # its lowest level sat far below the positive side's -- drawing many
    # more rings and reading as "prominent" when the signal was weaker.
    spectrum = _make_2d()
    n_f1, n_f2 = spectrum.data.shape
    yy, xx = np.mgrid[0:n_f1, 0:n_f2]
    spectrum.data = (
        1000 * np.exp(-(((xx - n_f2 // 4) ** 2) / 8 + ((yy - n_f1 // 4) ** 2) / 8))
        - 60 * np.exp(-(((xx - 3 * n_f2 // 4) ** 2) / 8 + ((yy - 3 * n_f1 // 4) ** 2) / 8))
    )
    _, _, _, thresvec, _, neg_thresvec = twod._prepare_contour(
        spectrum, contour_fraction_x=1.0, contour_fraction_y=1.0,
        ncontourpts_x=10**9, ncontourpts_y=10**9,
    )
    # A -60 excursion against a +1000 peak is below the positive side's
    # own lowest level, so it draws no negative contours at all.
    assert neg_thresvec is None
    assert thresvec[-1] == pytest.approx(1000.0, rel=1e-6)


def test_negative_levels_mirror_the_positive_ones_when_they_are_drawn():
    # With genuinely comparable negative signal, the two sides are exact
    # mirrors -- a contour at a given level means the same intensity
    # whichever sign it has.
    spectrum = _make_2d()
    n_f1, n_f2 = spectrum.data.shape
    yy, xx = np.mgrid[0:n_f1, 0:n_f2]
    spectrum.data = (
        1000 * np.exp(-(((xx - n_f2 // 4) ** 2) / 8 + ((yy - n_f1 // 4) ** 2) / 8))
        - 900 * np.exp(-(((xx - 3 * n_f2 // 4) ** 2) / 8 + ((yy - 3 * n_f1 // 4) ** 2) / 8))
    )
    _, _, _, thresvec, _, neg_thresvec = twod._prepare_contour(
        spectrum, contour_fraction_x=1.0, contour_fraction_y=1.0,
        ncontourpts_x=10**9, ncontourpts_y=10**9,
    )
    assert neg_thresvec is not None
    np.testing.assert_allclose(neg_thresvec, [-level for level in thresvec])


# --- Extracted-slice panel -------------------------------------------------

def _slice_test_spectrum(n_f1=32, n_f2=64):
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    y = np.linspace(150.0, 0.0, n_f1)
    x = np.linspace(12.0, -2.0, n_f2)
    data = np.zeros((n_f1, n_f2))
    for row, col in ((5, 10), (16, 30), (26, 50)):
        data[row, col] = 100.0 * (row + 1)
    return NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="frequency", ppm=y, hz=y * 100.0),
              NMRAxis(kind="frequency", ppm=x, hz=x * 400.0)],
        metadata=NMRMetadata(vendor="test", nucleus="1H", nucleus_indirect="13C",
                             sfo1=400.0, sfo1_indirect=100.0),
    )


def test_plot_2d_contour_slice_panel_pairs_lines_and_traces_by_colour():
    # User request: "plot slices from 2D contours with a dashed line
    # across the contour slice and then projecting out". The shared
    # colour is what pairs a line with its trace -- the connector lines
    # an earlier version drew are gone ("drop the lines to the extracted
    # slices, it seems too hard to maintain across different views").
    spectrum = _slice_test_spectrum()
    slices = [{"axis": "f1", "index": 6, "label": "A"},
              {"axis": "f1", "index": 17, "label": "B"},
              {"axis": "f1", "index": 27, "label": "C"}]
    fig, ax, panels = twod.plot_2d_contour(spectrum, slices=slices)

    panel = panels["slices"]
    assert panel is not None
    assert len(panel.lines) == 3                       # one trace per slice
    assert len(ax.lines) == 3                          # one dashed line per slice
    for contour_line, trace in zip(ax.lines, panel.lines):
        assert contour_line.get_color() == trace.get_color()
    assert {t.get_text() for t in panel.texts} == {"A", "B", "C"}

    # No connectors, and no panel title.
    assert not [a for a in fig.artists if isinstance(a, ConnectionPatch)]
    assert panel.get_title() == ""
    plt.close(fig)


def test_plot_2d_contour_slice_panel_spreads_traces_evenly_over_the_panel():
    # "The slices ... should spread evenly to fill the space" -- one
    # slot each across the full panel height, rather than bunched at the
    # bottom with dead space above.
    spectrum = _slice_test_spectrum()
    slices = [{"axis": "f1", "index": i} for i in (6, 17, 27)]
    fig, ax, panels = twod.plot_2d_contour(spectrum, slices=slices)

    panel = panels["slices"]
    assert panel.get_ylim() == pytest.approx((0.0, 1.0))
    baselines = sorted(float(np.min(line.get_ydata())) for line in panel.lines)
    np.testing.assert_allclose(baselines, [0.0, 1 / 3, 2 / 3], atol=1e-9)

    # Each trace normalised to its own height ("the extracted
    # projections should be normalised"), so a weak row can be compared
    # with a strong one on lineshape -- the synthetic rows differ in
    # height by construction, and must come out equal here.
    heights = [float(np.ptp(line.get_ydata())) for line in panel.lines]
    np.testing.assert_allclose(heights, heights[0])
    assert heights[0] == pytest.approx(1 / 3 * 0.85)   # slot x offset_fraction
    plt.close(fig)


def test_plot_2d_contour_slice_panel_labels_and_tracks_the_contour_axis():
    # "The axis needs a matching axis label", and the traces must stay
    # aligned under the contour through later zooming -- setting the
    # limits once at draw time was not enough, because the GUI's
    # preserve-the-view redraw restores the contour's limits afterwards.
    spectrum = _slice_test_spectrum()
    fig, ax, panels = twod.plot_2d_contour(spectrum, slices=[{"axis": "f1", "index": 6}])
    panel = panels["slices"]

    assert panel.get_xlabel() == ax.get_xlabel() != ""
    ax.set_xlim(8.0, 2.0)
    assert panel.get_xlim() == pytest.approx((8.0, 2.0))
    plt.close(fig)


def test_plot_2d_contour_slice_panel_stacks_in_contour_order():
    # The trace whose dashed line sits highest on the contour must be
    # the highest trace, or the connectors cross each other -- which
    # reads as an error even though every line joins the right trace.
    spectrum = _slice_test_spectrum()
    # Deliberately out of order: index 27 is the *highest* on the
    # contour (its F1 ppm is lowest, and the F1 axis descends downward).
    slices = [{"axis": "f1", "index": 17}, {"axis": "f1", "index": 6}, {"axis": "f1", "index": 27}]
    fig, ax, panels = twod.plot_2d_contour(spectrum, slices=slices)

    ppm = spectrum.axes[0].ppm
    lo, hi = ax.get_ylim()
    baselines = [float(np.min(line.get_ydata())) for line in panels["slices"].lines]
    screen = [(float(ppm[s["index"] - 1]) - lo) / (hi - lo) for s in slices]
    # Rank by height on the contour and by height in the panel: same order.
    assert np.argsort(screen).tolist() == np.argsort(baselines).tolist()
    plt.close(fig)


def test_plot_2d_contour_slice_panel_skips_out_of_range_and_mixed_axes():
    # A stale index (dataset reprocessed to a different size) must not
    # take the whole figure down, and mixing rows with columns would put
    # two unrelated frequency axes on one panel.
    spectrum = _slice_test_spectrum()
    slices = [{"axis": "f1", "index": 6},
              {"axis": "f1", "index": 9999},     # out of range
              {"axis": "f2", "index": 10}]       # different axis
    fig, ax, panels = twod.plot_2d_contour(spectrum, slices=slices)
    assert len(panels["slices"].lines) == 1
    plt.close(fig)


def test_plot_2d_contour_without_slices_has_no_panel():
    spectrum = _slice_test_spectrum()
    fig, ax, panels = twod.plot_2d_contour(spectrum)
    assert "slices" not in panels
    assert len(ax.lines) == 0
    plt.close(fig)
