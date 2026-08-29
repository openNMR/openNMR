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
    _, _, _, thresvec, _ = twod._prepare_contour(spec, factor=1e-3, clevels=20, level_mode="power", power=1.25)
    assert thresvec[-1] == pytest.approx(np.amax(spec.data))
    assert all(level <= np.amax(spec.data) + 1e-6 for level in thresvec)


def test_prepare_contour_power_mode_bunches_toward_noise_floor():
    # power > 1 should push more levels toward thresh_min (the noise
    # floor), not toward thresh_max -- matching this function's own
    # docstring ("steeper/more levels bunched near the noise floor as
    # power increases past 1").
    spec = _make_2d()
    _, _, _, linear_like, _ = twod._prepare_contour(spec, factor=1e-3, clevels=20, level_mode="power", power=1.0)
    _, _, _, bunched, _ = twod._prepare_contour(spec, factor=1e-3, clevels=20, level_mode="power", power=1.25)
    # Every level below the top one should sit lower (closer to the
    # noise floor) with the steeper power than with power=1.
    assert all(b <= lin for b, lin in zip(bunched[:-1], linear_like[:-1]))


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
