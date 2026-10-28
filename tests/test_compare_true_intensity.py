"""Compare shows true relative intensity: one intensity scale for the whole stack
(not each spectrum stretched to its own row) and an "Unscaled" mode that shows the
intensity as recorded, as TopSpin does. Synthetic data."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.notebook import intensity_scaling as isc


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _spec(height, ns=1, vendor="bruker", centre=5.0, **meta):
    ppm = np.linspace(10, 0, 500)
    centre = ppm[int(round((10 - centre) / 10 * 499))]           # a peak that sits on a data point
    data = height * np.exp(-((ppm - centre) / 0.2) ** 2)
    return NMRSpectrum(data=data, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400)],
                       metadata=NMRMetadata(vendor=vendor, nucleus="1H", sfo1=400.0, ns=ns, **meta))


def _heights(canvas):
    canvas.figure.canvas.draw()
    out = []
    for ax in canvas._compare_axes:
        y = ax.lines[0].get_ydata()
        lo = ax.transData.transform((0, float(np.min(y))))[1]
        hi = ax.transData.transform((0, float(np.max(y))))[1]
        out.append(hi - lo)
    return out


# --- the scaling mode --------------------------------------------------------

def test_unscaled_is_a_mode_and_has_a_label():
    assert "unscaled" in isc.MODES
    assert "as recorded" in isc.intensity_label("unscaled")


def test_unscaled_multiplies_a_per_scan_reader_back_by_ns():
    s = _spec(2.0, ns=8)
    out = isc.scale_for_comparison([s], mode="unscaled")
    assert out["spectra"][0].data.max() == pytest.approx(16.0)
    assert out["factors"] == [8.0] and not out["skipped"]
    assert s.data.max() == pytest.approx(2.0)                  # the original is untouched


def test_unscaled_leaves_data_a_reader_never_divided_alone():
    s = _spec(2.0, ns=8, vendor="jeol")
    s.metadata.raw = {"params": [], "header": {}}
    assert isc.scale_for_comparison([s], mode="unscaled")["factors"] == [1.0]


def test_unscaled_ignores_mass_and_gain_but_per_scan_does_not():
    s = _spec(2.0, ns=4, mass=0.5)
    un = isc.scale_for_comparison([s], mode="unscaled", divide_rg=True)
    assert un["factors"] == [4.0] and not un["skipped"]
    assert isc.scale_for_comparison([s], mode="per_scan")["factors"] == [1.0]


def test_two_spectra_keep_their_true_ratio_when_unscaled():
    a, b = _spec(1.0, ns=2), _spec(1.0, ns=16)
    out = isc.scale_for_comparison([a, b], mode="unscaled")
    assert out["spectra"][1].data.max() / out["spectra"][0].data.max() == pytest.approx(8.0)


def test_unscaled_without_a_scan_count_says_so():
    s = _spec(2.0, ns=None)
    out = isc.scale_for_comparison([s], mode="unscaled", labels=["x"])
    assert out["factors"] == [1.0] and any("number of scans" in n for n in out["notes"])


# --- the stacked plot ----------------------------------------------------------

def test_a_shared_scale_keeps_relative_heights(qapp):
    canvas = MplCanvas()
    canvas.show_compare([_spec(1.0, centre=3), _spec(4.0, centre=5), _spec(2.0, centre=7)],
                        mode="stack", shared_scale=True)
    h = _heights(canvas)
    assert h[1] / h[0] == pytest.approx(4.0, rel=1e-3) and h[2] / h[0] == pytest.approx(2.0, rel=1e-3)


def test_without_a_shared_scale_each_fills_its_row(qapp):
    canvas = MplCanvas()
    canvas.show_compare([_spec(1.0), _spec(4.0)], mode="stack", shared_scale=False)
    h = _heights(canvas)
    assert h[0] == pytest.approx(h[1], rel=1e-6)


def test_the_shared_range_spans_every_spectrum(qapp):
    canvas = MplCanvas()
    canvas.show_compare([_spec(1.0), _spec(4.0)], mode="stack", shared_scale=True)
    lows = [float(ax.lines[0].get_ydata().min()) for ax in canvas._compare_axes]
    highs = [float(ax.lines[0].get_ydata().max()) for ax in canvas._compare_axes]
    assert canvas._compare_shared_range == pytest.approx((min(lows), max(highs)))


def test_even_pitch_and_zoom_still_hold_on_a_shared_scale(qapp):
    canvas = MplCanvas()
    canvas.show_compare([_spec(1.0), _spec(4.0), _spec(2.0)], mode="stack", shared_scale=True)
    canvas.set_stack_gap(0.3)
    canvas.figure.canvas.draw()
    feet = []
    for ax in canvas._compare_axes:
        feet.append(ax.transData.transform((0, float(ax.lines[0].get_ydata().min())))[1])
    assert np.allclose(np.diff(feet), np.diff(feet)[0])


# --- the window ----------------------------------------------------------------

@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    return MainWindow()


def _compare(window, spectra):
    for k, s in enumerate(spectra):
        window._add_derived_spectrum(s, f"S{k + 1}")
    window.spectrum_list.selectAll()
    window._show_side_tab("Compare")
    window.layout_mode_combo.setCurrentIndex(0)
    window._refresh_compare_view(switch_tab=True, force_render=True)


def test_compare_defaults_to_one_scale_and_offers_unscaled(window):
    assert window.compare_common_scale_check.isChecked()
    texts = [window.compare_normalize_combo.itemText(i) for i in range(window.compare_normalize_combo.count())]
    assert texts[:3] == ["Per scan (÷ NS)", "Max intensity", "Per scan and sample mass (÷ NS × mass)"]
    assert texts[3].startswith("Unscaled")


def test_the_window_shows_true_relative_heights_and_the_checkbox_undoes_it(window):
    _compare(window, [_spec(1.0), _spec(5.0, centre=6)])
    h = _heights(window.compare_canvas)
    assert max(h) / min(h) == pytest.approx(5.0, rel=1e-3)
    window.compare_common_scale_check.setChecked(False)
    h = _heights(window.compare_canvas)
    assert max(h) / min(h) == pytest.approx(1.0, rel=1e-6)


def test_max_intensity_still_normalises_each_spectrum(window):
    _compare(window, [_spec(1.0), _spec(5.0, centre=6)])
    window.compare_normalize_combo.setCurrentIndex(1)
    h = _heights(window.compare_canvas)
    assert max(h) / min(h) == pytest.approx(1.0, rel=1e-6)


def test_unscaled_in_the_window_restores_the_recorded_intensity(window):
    _compare(window, [_spec(1.0, ns=2), _spec(1.0, ns=8, centre=6)])
    window.compare_normalize_combo.setCurrentIndex(0)              # per scan: equal peaks
    h = _heights(window.compare_canvas)
    assert max(h) / min(h) == pytest.approx(1.0, rel=1e-3)
    window.compare_normalize_combo.setCurrentIndex(3)              # unscaled: 8 / 2 = 4 times
    h = _heights(window.compare_canvas)
    assert max(h) / min(h) == pytest.approx(4.0, rel=1e-3)
