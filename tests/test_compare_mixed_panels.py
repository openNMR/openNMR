"""1D, 2D and pulse-sequence panels in one Compare figure (notebook function
``plot_compare_panels``), with panel labels and per-panel x-axis labels."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.io.pulse_sequence import parse_pulse_sequence
from opennmr.notebook import compare_panels as cp
from test_pulse_sequence import BRUKER


def spec1d(centre=5.0):
    ppm = np.linspace(10, 0, 600)
    data = np.exp(-((ppm - centre) ** 2) * 8) + 0.6 * np.exp(-((ppm - 2) ** 2) * 20)
    return NMRSpectrum(data=data, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400)],
                       metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))


def spec2d(shift=0.0):
    x = np.linspace(10, 0, 128)
    y = np.linspace(150, 50, 64)
    xx, yy = np.meshgrid(x, y)
    data = 1e6 * np.exp(-((xx - 5 - shift) ** 2 / 0.02 + (yy - 100) ** 2 / 4))
    return NMRSpectrum(data, [NMRAxis("frequency", ppm=y, hz=y * 100), NMRAxis("frequency", ppm=x, hz=x * 400)],
                       NMRMetadata(vendor="bruker", nucleus="1H", nucleus_indirect="13C",
                                   sfo1=400.0, sfo1_indirect=100.0))


@pytest.fixture
def seq():
    return parse_pulse_sequence(BRUKER, "bruker")[0]


def test_one_d_and_two_d_get_their_own_panels():
    fig, axes, link = cp.plot_compare_panels([spec1d(), spec2d()], layout="row", figsize=(8, 4))
    assert [a._opennmr_panel_kind for a in axes] == ["1d", "2d"]
    assert all(a.get_xlabel() for a in axes)               # every panel keeps its own x label


def test_several_two_d_in_one_panel_are_overlaid_and_1d_stay_superimposed():
    fig, axes, _ = cp.plot_compare_panels([[spec1d(4), spec1d(6)], [spec2d(0), spec2d(0.5)]],
                                          layout="column", figsize=(5, 7))
    assert len(axes) == 2 and len(axes[0].lines) >= 2


def test_a_pulse_sequence_fills_a_panel(seq):
    fig, axes, _ = cp.plot_compare_panels([spec1d(), {"pulse_sequence": seq}], layout="row", figsize=(9, 4))
    assert axes[1]._opennmr_panel_kind == "pulse"


def test_pulse_style_options_reach_the_panel(seq):
    fig, axes, _ = cp.plot_compare_panels(
        [{"pulse_sequence": seq}], layout="row", figsize=(6, 3),
        pulse_options={"linewidth": 3.0, "title": True, "title_text": "My sequence"})
    assert "My sequence" in {axes[0].get_title(loc=loc) for loc in ("left", "center", "right")}


def test_a_missing_pulse_sequence_says_why_instead_of_failing():
    fig, axes, _ = cp.plot_compare_panels(
        [spec1d(), {"pulse_sequence": None, "message": "No sequence file"}], layout="row", figsize=(7, 3))
    assert any("No sequence file" in t.get_text() for t in axes[1].texts)


def test_one_d_and_two_d_cannot_share_a_panel():
    with pytest.raises(ValueError):
        cp.plot_compare_panels([[spec1d(), spec2d()]], layout="row")


def test_labels_follow_order_start_and_style_on_a_mixed_grid(seq):
    fig, axes, _ = cp.plot_compare_panels(
        [spec1d(), spec2d(), spec1d(3), {"pulse_sequence": seq}], layout="grid", ncols=2,
        panel_labels={"style": "(A)", "start": "b", "order": "column"}, figsize=(9, 7))
    labels = fig._opennmr_panel_labels
    assert [t.get_text() for t in labels.texts] == ["(B)", "(D)", "(C)", "(E)"]


def test_labels_do_not_touch_the_tick_labels_or_each_other(seq):
    fig, axes, _ = cp.plot_compare_panels(
        [spec1d(), spec2d(), spec1d(3), {"pulse_sequence": seq}], layout="grid", ncols=2,
        panel_labels={}, figsize=(9, 7))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    labels = fig._opennmr_panel_labels
    boxes = [t.get_window_extent(r) for t in labels.texts]
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
    for box, ax in zip(boxes, axes):
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            if tick.get_visible() and tick.get_text():
                assert not box.overlaps(tick.get_window_extent(r))
    assert abs(boxes[0].y0 - boxes[1].y0) < 0.75            # top row on one baseline
    assert abs(boxes[0].x0 - boxes[2].x0) < 0.75            # first column on one left edge


def test_zoom_links_across_one_d_and_two_d_shift_axes():
    fig, axes, link = cp.plot_compare_panels([spec1d(), spec2d()], layout="row", figsize=(8, 4))
    axes[0].set_xlim(6, 4)
    assert axes[1].get_xlim() == pytest.approx((6, 4))
