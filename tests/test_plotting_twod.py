import matplotlib.pyplot as plt
import numpy as np
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
    assert "red" in calls
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

    # Exactly at the ceiling is still "within" it, not over.
    at_cap = (16384, 1024)
    assert at_cap[0] * at_cap[1] == twod._MAX_CONTOUR_POINTS
    assert twod._cap_contour_points(*at_cap) == at_cap


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
