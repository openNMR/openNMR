"""Six requests of 2026-10-15:

1. A spectrum (an extracted slice) whose baseline sits below zero was drawn
   partly under the axis: y-limits now come from the full-resolution data,
   with Plot Style gaps below spectra and between stacked spectra.
2. Process ▸ Truncate to reprocessed on every keystroke; it now applies on
   Enter / leaving the box, and says what it counts beside TopSpin's TD.
3. A kept slice lost "Apply Phase to Source 2D" when its 2D was re-read
   (a new object): the link now survives through the session's bookkeeping.
4. Right-click ▸ Calibrate Shift Here (ppm)… on a 1D; SR boxes to ±500 MHz.
5. A row of a hypercomplex 2D was phased with F1's imaginary part, so a
   90° drag barely changed it: F2's is rebuilt by the Hilbert transform.
6. Edit ▸ Undo Last Processing Change.
"""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import SR_LIMIT_HZ, MainWindow
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.plotting.style import DOWNSAMPLE_1D_MAX_POINTS
from opennmr.processing import ft as processing_ft
from opennmr.processing import slicing

from test_gui import (  # noqa: F401 -- qapp is a fixture
    _isolate_sidecars, _make_raw_1d_dataset, _make_raw_2d_dataset, qapp,
)


def _spectrum_1d(y, ppm=None, imag=None, nucleus="1H"):
    y = np.asarray(y, dtype=float)
    if ppm is None:
        ppm = np.linspace(10.0, -2.0, y.size)
    metadata = NMRMetadata(vendor="derived", nucleus=nucleus, sfo1=400.0, source_path="synthetic")
    return NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=ppm)], metadata=metadata,
                       data_imag=imag)


def _negative_baseline_with_hidden_dips(n=None):
    """A baseline at -0.3 and a tall peak, plus narrow dips that share a
    downsampling bin with a taller positive spike -- the display
    downsample (largest magnitude per bin) keeps the spike and drops the
    dip, so an autoscale of the plotted line missed the true minimum."""
    n = n or DOWNSAMPLE_1D_MAX_POINTS * 8
    y = np.full(n, -0.3)
    y[n // 2] = 10.0
    bin_size = n // DOWNSAMPLE_1D_MAX_POINTS
    for b in (100, 900, 1500):
        y[b * bin_size] = 3.0        # the spike the bin keeps
        y[b * bin_size + 1] = -2.0   # the dip it drops
    return y


# ---- 1. gaps below spectra / between stacked spectra --------------------------

def test_a_negative_baseline_is_never_under_the_axis(qapp):
    canvas = MplCanvas()
    y = _negative_baseline_with_hidden_dips()
    canvas.show_spectrum(_spectrum_1d(y))
    normalised = y / np.max(np.abs(y))
    low, high = canvas.ax.get_ylim()
    span = normalised.max() - normalised.min()
    assert low == pytest.approx(normalised.min() - MplCanvas._BASELINE_GAP * span)
    assert high >= normalised.max()
    # Full Resolution swaps the data in place; the limits already covered it.
    canvas.set_full_resolution(True)
    assert canvas.ax.get_ylim()[0] <= normalised.min()


def test_gap_below_spectra_is_a_plot_style_setting(qapp):
    canvas = MplCanvas()
    y = np.concatenate([np.full(500, -0.5), [4.0], np.full(500, -0.5)])
    canvas.show_spectrum(_spectrum_1d(y))
    canvas.set_style(baseline_gap=0.25)
    norm = y / 4.0
    span = norm.max() - norm.min()
    assert canvas.ax.get_ylim()[0] == pytest.approx(norm.min() - 0.25 * span)
    assert canvas.style_settings()["baseline_gap"] == pytest.approx(0.25)
    # ... and a redraw keeps it.
    canvas.show_spectrum(_spectrum_1d(y))
    assert canvas.ax.get_ylim()[0] == pytest.approx(norm.min() - 0.25 * span)


def test_stacked_compare_uses_the_stack_gap_between_rows(qapp):
    """2026-10-28: the gap is the *pitch* of the stack -- every spectrum the
    same distance above the one below (1 + gap + the top margin spectrum
    heights), the bottom one ``baseline_gap`` above the axis."""
    canvas = MplCanvas()
    a = _spectrum_1d(np.concatenate([np.full(300, -1.0), [5.0], np.full(300, -1.0)]))
    b = _spectrum_1d(np.concatenate([np.full(300, 0.2), [3.0], np.full(300, 0.2)]))
    canvas.set_style(baseline_gap=0.1, stack_gap=0.4)
    canvas.show_compare([a, b], mode="stack")
    top, bottom = canvas._compare_axes

    def geometry():
        canvas.figure.canvas.draw()
        feet, heights = [], []
        for ax in (top, bottom):
            y = ax.lines[0].get_ydata()
            lo = ax.transData.transform((0, float(np.min(y))))[1]
            hi = ax.transData.transform((0, float(np.max(y))))[1]
            feet.append(lo)
            heights.append(hi - lo)
        return feet, heights

    feet, heights = geometry()
    assert (feet[0] - feet[1]) / heights[0] == pytest.approx(1 + 0.4 + canvas._TOP_GAP, rel=1e-6)
    axis_line = bottom.get_window_extent().y0
    assert (feet[1] - axis_line) / heights[1] == pytest.approx(0.1, rel=1e-6)
    canvas.set_style(stack_gap=0.0)
    feet, heights = geometry()
    assert (feet[0] - feet[1]) / heights[0] == pytest.approx(1 + canvas._TOP_GAP, rel=1e-6)


def test_overlay_compare_keeps_every_trace_above_the_axis(qapp):
    canvas = MplCanvas()
    a = _spectrum_1d(np.concatenate([np.full(300, -2.0), [5.0], np.full(300, -2.0)]))
    b = _spectrum_1d(np.concatenate([np.full(300, 0.5), [3.0], np.full(300, 0.5)]))
    canvas.show_compare([a, b], mode="overlay")
    lowest = min(float(np.min(line.get_ydata())) for line in canvas.ax.lines)
    assert canvas.ax.get_ylim()[0] < lowest


def test_plot_style_panel_has_the_two_gaps(qapp):
    window = MainWindow()
    window.output_baseline_gap_spin.setValue(20.0)
    window.output_stack_gap_spin.setValue(30.0)
    kwargs = window._current_output_style_kwargs()
    assert kwargs["baseline_gap"] == pytest.approx(0.2)
    assert kwargs["stack_gap"] == pytest.approx(0.3)
    assert "baseline_gap" in MplCanvas._STYLE_KEYS and "stack_gap" in MplCanvas._STYLE_KEYS


# ---- 2. Truncate to: applied on Enter, units explained --------------------------

def test_truncate_does_not_reprocess_per_keystroke(qapp, tmp_path, monkeypatch):
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=64), vendor="bruker")
    box = window.tdeff_spin
    assert box.keyboardTracking() is False
    for name in ("tdeff_f1_spin", "zero_fill_f1_spin", "zero_fill_custom_spin"):
        assert getattr(window, name).keyboardTracking() is False, name
    calls = []
    monkeypatch.setattr(window, "_on_fid_processing_control_changed", lambda *a: calls.append(a))
    box.valueChanged.disconnect()
    box.valueChanged.connect(lambda _v: window._on_fid_processing_control_changed())
    box.lineEdit().selectAll()
    QTest.keyClicks(box.lineEdit(), "40")
    assert calls == []                       # nothing while typing
    QTest.keyClick(box.lineEdit(), Qt.Key.Key_Return)
    assert box.value() == 40 and len(calls) == 1
    # Past the FID's length: the whole FID, padded with zeros to that many
    # points (2026-10-17), and the label says so.
    box.lineEdit().selectAll()
    QTest.keyClicks(box.lineEdit(), "1024")
    QTest.keyClick(box.lineEdit(), Qt.Key.Key_Return)
    assert box.value() == 1024
    assert window._zero_fill_value() >= 1024
    assert not window.tdeff_units_label.isHidden()
    assert "of 64 complex points" in window.tdeff_units_label.text()


def test_truncate_units_label_compares_with_topspin_td(qapp, tmp_path, monkeypatch):
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=64), vendor="bruker")
    spectrum = window.session.spectra[0]
    spectrum.metadata.raw.setdefault("acqus", {})["TD"] = 1024
    window._show_truncate_units(spectrum, 434, True)
    text = window.tdeff_units_label.text()
    assert "434 complex points" in text and "TD 1024 = 512 complex" in text
    assert "less 78 for the digital filter" in text


# ---- 3. the slice keeps its link when the 2D is re-read --------------------------

def _keep_slice(window, axis="f1", index=1):
    window.canvas.extract_slice(axis, index)
    window.canvas._keep_slice_button.click()
    return window._selected_index(), window.session.spectra[-1]


def test_kept_slice_survives_its_2d_being_re_read(qapp, tmp_path, monkeypatch):
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    slice_index, kept = _keep_slice(window)
    window.session.reload(0)                         # a new object in the slot
    fresh = window.session.spectra[0]
    assert kept.metadata.raw["slice_parent"] is not fresh
    assert window._slice_source_of(kept) == (fresh, "f2")
    window._select_single_row(0)
    window._select_single_row(slice_index)
    assert window.apply_slice_phase_button.isHidden() is False


def test_slice_kept_from_a_stale_view_still_finds_its_2d(qapp, tmp_path, monkeypatch):
    # The canvas still holds the object it drew; the 2D was replaced
    # under it (re-read) before Keep was pressed.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    window.canvas.extract_slice("f1", 1)
    window.session.reload(0)
    window.canvas._keep_slice_button.click()
    kept = window.session.spectra[-1]
    assert window._slice_source_of(kept)[0] is window.session.spectra[0]
    assert window.apply_slice_phase_button.isHidden() is False


def test_slice_link_is_dropped_when_the_2d_is_removed(qapp, tmp_path, monkeypatch):
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    _i, kept = _keep_slice(window)
    window.session.reload(0)
    window.session.remove(0)
    assert window._slice_source_of(kept) == (None, None)


# ---- 4. calibrate a 1D shift; SR range ---------------------------------------

def test_sr_boxes_reach_500_mhz(qapp):
    window = MainWindow()
    for name in ("sr_shift_spin", "sr_shift_indirect_spin", "sr_cal_spin"):
        box = getattr(window, name)
        assert box.maximum() == SR_LIMIT_HZ == 5.0e8 and box.minimum() == -5.0e8, name
        box.setValue(-2.5e8)
        assert box.value() == pytest.approx(-2.5e8)


def test_calibrate_1d_shift_moves_the_peak_to_the_typed_value(qapp, tmp_path, monkeypatch):
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=256), vendor="bruker")
    spectrum = window.session.spectra[0]
    x = spectrum.axis_values(0, unit="ppm")
    peak = float(x[int(np.argmax(np.abs(spectrum.data)))])
    lo, hi = sorted(window.canvas.ax.get_xlim())
    found = window._peak_top_near(spectrum, peak + 0.004 * (hi - lo), 0.01 * (hi - lo))
    assert found == pytest.approx(peak)

    sr = window.calibrate_1d_shift(peak, peak + 1.5)
    assert sr == pytest.approx(1.5 * spectrum.metadata.sfo1, rel=1e-9)
    spectrum = window.session.spectra[0]
    x_new = spectrum.axis_values(0, unit="ppm")
    assert float(x_new[int(np.argmax(np.abs(spectrum.data)))]) == pytest.approx(peak + 1.5, abs=1e-6)
    # Again from the new position: SR accumulates, the peak lands exactly.
    window.calibrate_1d_shift(peak + 1.5, peak - 0.25)
    x_new = window.session.spectra[0].axis_values(0, unit="ppm")
    assert float(x_new[int(np.argmax(np.abs(spectrum.data)))]) == pytest.approx(peak - 0.25, abs=1e-6)


def test_the_1d_right_click_menu_offers_calibrate(qapp, tmp_path, monkeypatch):
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=256), vendor="bruker")
    canvas = window.canvas
    canvas._last_right_click_xy = (1.0, 0.5)
    titles = []
    emitted = []
    monkeypatch.setattr(window, "_calibrate_1d_shift_from_plot",
                        lambda spectrum, ppm: emitted.append(ppm))

    def _pick(menu, pos):
        actions = {a.text(): a for a in menu.actions()}
        titles.extend(actions)
        return actions.get("Calibrate Shift Here (ppm)…")

    monkeypatch.setattr(canvas, "_exec_context_menu", _pick)
    canvas._on_context_menu(canvas.rect().center())
    assert "Calibrate Shift Here (ppm)…" in titles
    assert emitted == [1.0]          # the dialog would open at the clicked ppm


# ---- 5. a row of a hypercomplex 2D is phased with F2's imaginary -------------

def _lorentzian_pair(n=1024, centre=0.3, width=0.004):
    f = np.linspace(-0.5, 0.5, n, endpoint=False)
    d = (f - centre) / width
    return 1.0 / (1.0 + d ** 2), -d / (1.0 + d ** 2)


def test_hilbert_imaginary_rebuilds_the_dispersion():
    t = np.arange(512) * 1e-4
    fid = np.exp(2j * np.pi * 1200 * t - t / 0.004) + 0.5 * np.exp(-2j * np.pi * 3000 * t - t / 0.002)
    _f, spectrum = processing_ft.fft_spectrum(processing_ft.zero_fill(fid, 2048), 1e-4)
    rebuilt = processing_ft.hilbert_imaginary(spectrum.real)
    assert np.max(np.abs(rebuilt - spectrum.imag)) < 1e-9 * np.max(np.abs(spectrum.imag))


def _hypercomplex_2d(fn_mode):
    n1, n2 = 8, 1024
    absorp = np.fft.fftshift(np.real(np.fft.fft(np.exp(-np.arange(n2) / 60.0) * (np.arange(n2) < n2 // 2))))
    data = np.tile(absorp, (n1, 1))
    f1_imag = np.full((n1, n2), 1e-3)       # what a hypercomplex 2D stores: F1's imaginary
    axes = [NMRAxis(kind="frequency", ppm=np.linspace(100, 0, n1)),
            NMRAxis(kind="frequency", ppm=np.linspace(10, 0, n2))]
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, nucleus_indirect="13C",
                           sfo1_indirect=100.0, raw={"fn_mode": fn_mode})
    return NMRSpectrum(data=data, axes=axes, metadata=metadata, data_imag=f1_imag)


@pytest.mark.parametrize("fn_mode", [4, 5, 6])
def test_a_row_of_a_hypercomplex_2d_gets_f2s_imaginary(fn_mode):
    spectrum = _hypercomplex_2d(fn_mode)
    row = slicing.get_trace(spectrum, axis="f1", index=3)
    assert row.metadata.raw["slice"]["imaginary"] == "hilbert"
    assert np.allclose(row.data_imag, processing_ft.hilbert_imaginary(row.data))
    # A 90° phase now turns the line into its dispersion, as it should:
    # the displayed trace changes by about its own size, not by 1e-3.
    rotated = np.real((row.data + 1j * row.data_imag) * np.exp(1j * np.pi / 2))
    assert np.linalg.norm(rotated - row.data) / np.linalg.norm(row.data) > 1.0
    # A column runs along F1, whose imaginary *is* the stored one.
    column = slicing.get_trace(spectrum, axis="f2", index=5)
    assert np.allclose(column.data_imag, spectrum.data_imag[:, 4])
    assert "imaginary" not in column.metadata.raw["slice"]


def test_a_row_of_a_non_hypercomplex_2d_keeps_its_imaginary():
    spectrum = _hypercomplex_2d(1)          # QF: the stored imaginary is F2's
    row = slicing.get_trace(spectrum, axis="f1", index=3)
    assert np.allclose(row.data_imag, spectrum.data_imag[2])
    assert "imaginary" not in row.metadata.raw["slice"]


# ---- 6. undo ---------------------------------------------------------------

def _commit(window, index):
    window._flush_pending_state(index)


def test_undo_steps_back_through_phase_changes(qapp, tmp_path, monkeypatch):
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=128), vendor="bruker")
    index = window._selected_index()
    original = np.array(window.session.spectra[index].data)
    window.phase0_spin.setValue(30.0)
    _commit(window, index)
    after_30 = np.array(window.session.spectra[index].data)
    window.phase0_spin.setValue(25.0)            # +25 on top: 55 in all
    _commit(window, index)
    assert window.session.spectra[index].metadata.raw["phase0"] == pytest.approx(55.0)
    assert window.can_undo(index)

    assert window._undo_last_change(index)
    spectrum = window.session.spectra[index]
    assert spectrum.metadata.raw["phase0"] == pytest.approx(30.0)
    assert np.allclose(spectrum.data, after_30)
    assert window._undo_last_change(index)
    spectrum = window.session.spectra[index]
    assert float(spectrum.metadata.raw.get("phase0", 0.0) or 0.0) == pytest.approx(0.0)
    assert np.allclose(spectrum.data, original)
    assert not window._undo_last_change(index)       # nothing further back
    # The undo itself is not something to undo.
    assert not window.can_undo(index)


def test_undo_restores_a_kept_slices_data(qapp, tmp_path, monkeypatch):
    # A slice has no raw FID: its phase is baked into its data, so the
    # undo entry has to carry the data.
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    index, _kept = _keep_slice(window)
    _commit(window, index)
    before = np.array(window.session.spectra[index].data)
    window.phase0_spin.setValue(70.0)
    _commit(window, index)
    assert not np.allclose(window.session.spectra[index].data, before)
    assert window._undo_last_change(index)
    assert np.allclose(window.session.spectra[index].data, before)
    assert float(window.session.spectra[index].metadata.raw.get("phase0", 0.0) or 0.0) == pytest.approx(0.0)


def test_undo_is_in_the_edit_menu(qapp):
    window = MainWindow()
    assert window.undo_action.text().replace("&", "") == "Undo Last Processing Change"
    assert window.undo_action.shortcut().toString() == "Ctrl+Alt+Z"
