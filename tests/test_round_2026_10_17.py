"""2026-10-17: Compare box and Plot button, Ctrl/Cmd+0/B/E, the Processing
view on entering Process, the pivot line, double-click in zoom mode,
Truncate past the acquired points, a slice proc number's phase and
nucleus, the merged baseline methods (steppable coefficients, a spline
that no longer freezes, points chosen on the plot), the pseudo-2D guide
behind a button, CP Build-up's tick/untick and drag-on-any-spectrum, BPP
Import CSV and temperature regions, and the Offset Tracker."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFocusEvent, QKeySequence
from PySide6.QtWidgets import QApplication, QPushButton

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.io import persistence as io_persistence
from opennmr.notebook import offset_tracker as ot
from opennmr.notebook import vt_relaxation as vt
from opennmr.processing import ft as processing_ft
from opennmr.processing import fitting as processing_fitting

from test_gui import (  # noqa: F401
    qapp, _load_spectrum_directly, _make_raw_1d_dataset, _isolate_sidecars,
)


def _spectrum_1d(y, ppm=None, nucleus="13C", vendor="bruker", raw=None, imag=True):
    ppm = np.linspace(100.0, -100.0, len(y)) if ppm is None else ppm
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 100.0)
    meta = NMRMetadata(vendor=vendor, nucleus=nucleus, sfo1=100.0, bf1=100.0, ns=1.0,
                       raw=dict(raw or {}))
    return NMRSpectrum(data=np.asarray(y, float), axes=[axis], metadata=meta,
                       data_imag=np.zeros(len(y)) if imag else None)


def _peak(n=2048, centre=0.0, width=2.0, slope=0.0, offset=0.0):
    ppm = np.linspace(100.0, -100.0, n)
    return ppm, np.exp(-((ppm - centre) / width) ** 2) + slope * ppm + offset


# ---- Compare ---------------------------------------------------------------------

def test_compare_plot_button_shows_the_compare_view(qapp):
    window = MainWindow()
    for c in (0.0, 5.0):
        ppm, y = _peak(centre=c)
        _load_spectrum_directly(window, _spectrum_1d(y, ppm))
    window._show_side_tab("Compare")
    window.spectrum_list.selectAll()
    window.canvas_tabs.setCurrentWidget(window.canvas)
    window.compare_plot_button.click()      # clicked(bool) no longer lands in switch_tab
    assert window.canvas_tabs.currentWidget() is window.compare_canvas


def test_compare_list_box_is_about_ten_rows(qapp):
    window = MainWindow()
    row = window._compare_list_row_height()
    assert window.compare_list.maximumHeight() == 10 * row
    assert window.compare_list.minimumHeight() == 4 * row


# ---- shortcuts and Open experiments ------------------------------------------------

def test_ctrl_0_is_the_periodic_table_and_reset_view_moved(qapp):
    window = MainWindow()
    assert window._window_panel_actions["Periodic Table"].shortcut() == QKeySequence("Ctrl+0")
    zero = [a for a in window.findChildren(type(window._fact_action))
            if a.shortcut() == QKeySequence("Ctrl+0")]
    assert len(zero) == 1
    assert window._fact_action.shortcut() == QKeySequence("Ctrl+B")
    assert window.open_experiments_action.shortcut() == QKeySequence("Ctrl+E")


def test_open_experiments_minimises_to_a_strip_and_is_remembered(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_UI_SETTINGS_PATH", str(tmp_path / "ui.json"))
    window = MainWindow()
    window._show_side_tab("Open")
    window.show()
    QApplication.processEvents()
    assert window.open_experiments_box.isVisible()
    window.open_experiments_action.trigger()            # Ctrl/Cmd+E
    assert not window.open_experiments_box.isVisible()
    assert window.open_experiments_panel.isVisible()    # the strip stays
    assert "Open experiments (0)" in window.open_experiments_toggle.text()
    assert json.loads((tmp_path / "ui.json").read_text())["open_experiments_open"] is False
    window.open_experiments_toggle.click()              # a click on the strip
    assert window.open_experiments_box.isVisible()
    assert window.open_experiments_action.isChecked()
    window.close()


# ---- Process ---------------------------------------------------------------------

def test_entering_process_always_shows_the_processing_view(qapp):
    window = MainWindow()
    ppm, y = _peak()
    _load_spectrum_directly(window, _spectrum_1d(y, ppm))   # processed data: no raw FID
    window._show_side_tab("Open")
    window.canvas_tabs.setCurrentWidget(window.canvas)
    window._show_side_tab("Process")
    assert window.canvas_tabs.currentWidget() is window.processing_preview_canvas
    index = window.canvas_tabs.indexOf(window.processing_preview_canvas)
    assert window.canvas_tabs.isTabVisible(index)


def test_pivot_line_goes_when_another_process_control_is_used(qapp):
    window = MainWindow()
    ppm, y = _peak()
    _load_spectrum_directly(window, _spectrum_1d(y, ppm))
    window._show_side_tab("Process")
    window.phase0_spin.setValue(10.0)                 # phasing: the line is there
    assert window.canvas._pivot_line is not None or window.processing_preview_canvas._pivot_line \
        is not None
    QApplication.sendEvent(window.baseline_family_combo,
                           QFocusEvent(QEvent.Type.FocusIn, Qt.FocusReason.MouseFocusReason))
    assert window.canvas._pivot_line is None
    assert window.processing_preview_canvas._pivot_line is None
    window.phase0_spin.setValue(20.0)                 # back to phasing: back again
    assert window._pivot_marker_ppm() is not None


def test_double_click_resets_the_view_in_toolbar_zoom_mode(qapp):
    canvas = MplCanvas()
    ppm, y = _peak()
    canvas.show_spectrum(_spectrum_1d(y, ppm))
    full = canvas.ax.get_xlim()
    canvas.note_natural_view()
    canvas.toolbar.zoom()                              # the toolbar's Zoom button down
    assert canvas.toolbar.mode != ""
    canvas.ax.set_xlim(10, -10)
    event = SimpleNamespace(button=1, dblclick=True, inaxes=canvas.ax, xdata=0.0, ydata=0.0,
                            x=100, y=100, key=None)
    canvas._on_press(event)
    assert canvas.ax.get_xlim() == pytest.approx(full)


def test_double_click_does_nothing_while_measuring(qapp):
    canvas = MplCanvas()
    ppm, y = _peak()
    canvas.show_spectrum(_spectrum_1d(y, ppm))
    canvas.note_natural_view()
    canvas._measurement_mode = True
    canvas.ax.set_xlim(10, -10)
    canvas._double_click_resets = True
    assert canvas._click_mode_active()


def test_truncate_can_go_past_the_acquired_points_and_pads(qapp, tmp_path, monkeypatch):
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=64), vendor="bruker")
    acquired = window._truncate_acquired_points or window.tdeff_spin.maximum()
    assert window.tdeff_spin.maximum() > acquired
    zf = window._zero_fill_choice()
    window.tdeff_spin.setValue(4 * max(zf, acquired))
    assert window._zero_fill_value() == 4 * max(zf, acquired)
    window.tdeff_spin.setValue(acquired)
    assert window._zero_fill_value() == zf


# ---- a 1D proc number of a 2D ---------------------------------------------------

def test_slice_procno_is_phased_as_processed_data(qapp, tmp_path):
    folder = tmp_path / "expt"
    folder.mkdir()
    (folder / "ser").write_bytes(b"\0")                # the 2D's raw data
    ppm = np.linspace(100.0, -100.0, 2048)
    # A complex Lorentzian, so a phase change changes the real part's shape
    # (not just its height, which the normalised plot would hide).
    z = np.exp(1j * 0.3) / (1.0 + 1j * ppm / 2.0)
    real, imag = z.real, z.imag
    spectrum = _spectrum_1d(real, ppm, raw={"procno_dimension": 1})
    spectrum.data_imag = imag
    spectrum.metadata.source_path = str(folder)
    assert not MainWindow._fid_reprocessable(spectrum)
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._show_side_tab("Process")
    assert window.canvas_tabs.currentWidget() is window.processing_preview_canvas
    def trace():
        lines = window.processing_preview_canvas.ax.lines
        return max((np.array(ln.get_ydata(), float) for ln in lines),
                   key=lambda d: float(np.max(np.abs(d))) if d.size else 0.0)
    before = trace()
    assert np.max(np.abs(before)) > 0
    window.phase0_spin.setValue(window.phase0_spin.value() + 40.0)
    after = trace()
    assert not np.allclose(before, after)              # the preview follows the phase


def test_slice_procno_keeps_its_own_nucleus_over_the_2d_sidecar(qapp, tmp_path, monkeypatch):
    folder = tmp_path / "hetcor"
    folder.mkdir()
    (folder / "ser").write_bytes(b"\0")
    # The 2D's sidecar (proc number 1) says F2 is 1H.
    (folder / "opennmr_metadata.json").write_text(json.dumps(
        {"opennmr_metadata_version": 1, "vendor": "bruker", "nucleus": "1H",
         "nucleus_indirect": "13C", "procno": 1}))
    ppm, y = _peak()
    spectrum = _spectrum_1d(y, ppm, nucleus="13C", raw={"procno_dimension": 1})
    spectrum.metadata.source_path = str(folder)
    spectrum.metadata.procno = 99                      # an F1 column TopSpin wrote
    window = MainWindow()
    window._load(str(folder), "bruker", preloaded=spectrum)
    assert window.session.spectra[-1].metadata.nucleus == "13C"


# ---- baseline -------------------------------------------------------------------

def test_smoothing_spline_baseline_is_fast_on_a_large_spectrum():
    x = np.linspace(200, -200, 32768)
    rng = np.random.default_rng(0)
    y = 1e6 * np.exp(-x ** 2 / 2) + 2e4 * np.sin(x / 40) + 3e3 * rng.standard_normal(x.size)
    t0 = time.perf_counter()
    baseline = processing_ft.compute_baseline(x, y, method="spline")
    assert time.perf_counter() - t0 < 3.0             # was ~26 s, and froze the GUI
    assert np.all(np.isfinite(baseline))
    assert np.max(np.abs(baseline)) < 1e5             # no wild swing between the edges


def test_spline_through_chosen_points_follows_the_baseline():
    x = np.linspace(100, -100, 4001)
    true = 500.0 + 3.0 * x + 0.02 * x ** 2
    y = true + 1e4 * np.exp(-(x / 3) ** 2)
    nodes = [-90, -60, -30, -15, 15, 30, 60, 90]
    baseline = processing_ft.compute_baseline(x, y, method="spline_points", node_x=nodes)
    assert np.max(np.abs(baseline - true)) < 25.0
    with pytest.raises(ValueError):
        processing_fitting.spline_through_points(x, y, [10.0])


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_automatic_polynomial_written_as_coefficients_is_exact(degree):
    from opennmr.notebook.manual_baseline import evaluate_manual_baseline
    x = np.linspace(50, 10, 1000)                      # off-centre, so x_ref matters
    y = 100 + 2 * x - 0.03 * x ** 2 + np.exp(-((x - 30) / 0.5) ** 2) * 1e3
    auto = processing_ft.automatic_baseline_as_manual(x, y, "polynomial", degree=degree)
    assert auto["exact"]
    np.testing.assert_allclose(evaluate_manual_baseline(x, "polynomial", auto["params"]),
                               auto["baseline"], rtol=1e-7, atol=1e-7)


def test_automatic_sine_written_as_coefficients_is_exact():
    from opennmr.notebook.manual_baseline import evaluate_manual_baseline
    x = np.linspace(100, -100, 2000)
    y = 50 * np.sin(x / 30 + 0.4) + 10
    auto = processing_ft.automatic_baseline_as_manual(x, y, "sinusoidal", period=2 * np.pi * 30)
    np.testing.assert_allclose(evaluate_manual_baseline(x, "sine", auto["params"]),
                               auto["baseline"], atol=1e-8)


def _baseline_window():
    window = MainWindow()
    ppm, y = _peak(slope=0.002, offset=0.05)
    _load_spectrum_directly(window, _spectrum_1d(y, ppm))
    window._show_side_tab("Process")
    return window


def test_baseline_menu_is_one_entry_per_kind(qapp):
    window = _baseline_window()
    family = window.baseline_family_combo
    assert [family.itemData(i) for i in range(family.count())] == [
        "linear", "polynomial", "sinusoidal", "spline", "whittaker"]
    assert window.baseline_method_combo.isHidden()
    family.setCurrentIndex(family.findData("spline"))
    assert window.baseline_method_combo.currentData() == "spline_auto"
    assert not window.baseline_spline_mode_combo.isHidden()
    mode = window.baseline_spline_mode_combo
    mode.setCurrentIndex(mode.findData("spline_points"))
    assert window.baseline_method_combo.currentData() == "spline_points"
    assert not window.baseline_spline_points_box.isHidden()
    # a saved hand-set polynomial shows under Polynomial
    window.baseline_method_combo.setCurrentIndex(
        window.baseline_method_combo.findData("manual_polynomial"))
    assert family.currentData() == "polynomial"


def test_stepping_a_fitted_coefficient_makes_it_hand_set(qapp):
    window = _baseline_window()
    family = window.baseline_family_combo
    family.setCurrentIndex(family.findData("polynomial"))
    window._refresh_baseline_preview()
    box = window.baseline_manual_box
    assert not box.isHidden() and box.automatic
    before = box.params()
    step = box.step_size("E")
    assert step > 0
    up, _down = box.step_buttons["E"]
    up.click()
    raw = window.session.spectra[0].metadata.raw
    assert raw["baseline_method"] == "manual_polynomial"
    assert family.currentData() == "polynomial"
    assert raw["baseline_manual_polynomial_params"][4] == pytest.approx(before[4] + step)
    assert not box.automatic
    box.auto_button.click()                            # "Fit automatically"
    assert raw["baseline_method"] == "polynomial"


def test_step_sizes_follow_the_choice(qapp):
    window = _baseline_window()
    box = window.baseline_manual_box
    box.set_scales({"E": 2.0})
    box.step_combo.setCurrentIndex(0)
    fine = box.step_size("E")
    box.step_combo.setCurrentIndex(3)
    assert box.step_size("E") == pytest.approx(1000 * fine) == pytest.approx(2.0)


def test_spline_points_are_picked_and_removed_on_the_plot(qapp):
    window = _baseline_window()
    family = window.baseline_family_combo
    family.setCurrentIndex(family.findData("spline"))
    mode = window.baseline_spline_mode_combo
    mode.setCurrentIndex(mode.findData("spline_points"))
    window.baseline_pick_points_button.setChecked(True)
    assert window.processing_preview_canvas._pick_mode
    for x in (-80.0, -40.0, 40.0, 80.0):
        window.processing_preview_canvas.peakPicked.emit(x)
    raw = window.session.spectra[0].metadata.raw
    assert raw["baseline_spline_x"] == [-80.0, -40.0, 40.0, 80.0]
    window.processing_preview_canvas.peakPicked.emit(40.2)   # on a point: removes it
    assert raw["baseline_spline_x"] == [-80.0, -40.0, 80.0]
    window.baseline_undo_point_button.click()
    assert raw["baseline_spline_x"] == [-80.0, -40.0, 40.0, 80.0]
    kwargs = window._baseline_kwargs(window.session.spectra[0])
    assert kwargs["node_x"] == [-80.0, -40.0, 40.0, 80.0]
    window.baseline_correct_button.setChecked(True)
    assert np.all(np.isfinite(window.session.spectra[0].data))


# ---- pseudo-2D guide, sections shown --------------------------------------------

def test_pseudo_2d_guide_is_behind_a_button(qapp):
    window = MainWindow()
    for host, guide in ((window._fit_guide_host, window.fit_workflow_guide),
                        (window._curve_guide_host, window.relaxation_workflow_guide)):
        assert guide.isHidden()
        assert host.toggle_button.text() == "How to fit pseudo-2D data"
        host.toggle_button.click()
        assert not guide.isHidden()
        assert guide.guide_button is not None           # Next/Back still inside


@pytest.mark.parametrize("key", ["pseudo2d_guide", "offset_tracker", "dft", "batch_pseudo2d",
                                 "bpp", "exchange"])
def test_new_sections_can_be_hidden(qapp, key):
    window = MainWindow()
    assert key in MainWindow.OPTIONAL_SECTIONS
    window._apply_section_shown(key, False)
    for name in MainWindow.OPTIONAL_SECTIONS[key][1]:
        assert getattr(window, name).isHidden(), name
    window._apply_section_shown(key, True)
    for name in MainWindow.OPTIONAL_SECTIONS[key][1]:
        assert not getattr(window, name).isHidden(), name


# ---- CP Build-up ----------------------------------------------------------------

def test_cp_tick_all_and_drag_on_any_spectrum(qapp):
    window = MainWindow()
    for c in (0.0, 5.0, 10.0):
        ppm, y = _peak(centre=c)
        _load_spectrum_directly(window, _spectrum_1d(y, ppm))
    panel = window.cp_buildup_panel
    panel.refresh()
    panel.tick_all_button.click()
    assert panel._checked_rows() == [0, 1, 2]
    panel.untick_all_button.click()
    assert panel._checked_rows() == []
    shown = []
    panel._show_spectrum = shown.append
    panel.drag_on_combo.setCurrentIndex(3)             # the third spectrum
    panel.drag_button.setChecked(True)
    assert shown and shown[-1] is window.session.spectra[2]
    assert panel.armed


# ---- BPP ------------------------------------------------------------------------

def test_bpp_import_csv_button_opens_a_file_and_paste_is_gone(qapp, tmp_path, monkeypatch):
    from opennmr.gui import dynamics_panels
    path = tmp_path / "t1.csv"
    path.write_text("T,value,larmor_MHz\n250,0.5,100\n300,0.2,100\n350,0.4,100\n")
    monkeypatch.setattr(dynamics_panels.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    window = MainWindow()
    panel = window.vt_relaxation_panel
    texts = [b.text() for b in panel.findChildren(QPushButton)]
    assert "Paste" not in texts
    panel.import_button.click()                        # used to pass False as the path
    assert panel.table.rowCount() == 3


def test_vt_regions_recover_two_barriers():
    T = np.linspace(150, 500, 36)
    tau_a = vt.correlation_time(T, 1e-12, 0.20)
    tau_b = vt.correlation_time(T, 1e-14, 0.60)
    assert np.all(tau_a > 0) and np.all(tau_b > 0)
    r_low = vt.relaxation_rate(T[T <= 260], 100.0, 1e-12, 0.20, 2e9)
    r_high = vt.relaxation_rate(T[T >= 330], 100.0, 1e-14, 0.60, 5e9)
    Ts = np.concatenate([T[T <= 260], T[T >= 330]])
    d = vt.vt_dataset(Ts, 1.0 / np.concatenate([r_low, r_high]), larmor_MHz=100.0)
    result = vt.fit_vt_regions(d, [{"T_range": (150, 260), "label": "low"},
                                   {"T_range": (330, 500), "label": "high"}])
    fits = {f["region_label"]: f for f in result["fits"]}
    assert fits["low"]["params"]["Ea_eV"] == pytest.approx(0.20, rel=0.02)
    assert fits["high"]["params"]["Ea_eV"] == pytest.approx(0.60, rel=0.02)
    import matplotlib
    matplotlib.use("Agg")
    fig, ax = vt.plot_arrhenius_regions(result, d)
    assert len(ax.lines) >= 3


def test_a_region_too_small_for_bpp_gets_an_arrhenius_line():
    # User report (2026-10-17): two regions, one fitted line -- the second
    # region held two points, too few for BPP's three free parameters, and
    # its failure only showed in the results table.
    import matplotlib
    matplotlib.use("Agg")
    T = np.array([283.3, 291.0, 299.0, 302.0, 304.5, 306.0, 308.0, 309.5, 311.0])
    tau = vt.correlation_time(T, 1e-13, 0.40)
    rate = vt.relaxation_rate(T, 61.4, 1e-13, 0.40, 1e8)
    d = vt.vt_dataset(T, 1.0 / rate, larmor_MHz=61.4)
    assert np.all(tau > 0)
    result = vt.fit_vt_regions(d, [{"T_range": (297, 311)}, {"T_range": (283, 297)}],
                               per_dataset_amplitude=True)
    first, second = result["fits"]
    assert "error" not in first and "error" not in second
    assert second["kind"] == "arrhenius" and "Arrhenius" in second["model_choice"]
    assert second["params"]["Ea_eV"] > 0
    fig, ax = vt.plot_arrhenius_regions(result, d)
    drawn = [ln for ln in ax.lines if ln.get_linestyle() in ("-", "--")]
    assert len(drawn) >= 2                                   # one line per region
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Region 1" in t for t in texts) and any("Region 2" in t for t in texts)
    # a region that cannot be fitted at all is named in the legend
    result = vt.fit_vt_regions(d, [{"T_range": (297, 311)}, {"T_range": (250, 260)}])
    fig, ax = vt.plot_arrhenius_regions(result, d)
    assert any("not fitted" in t.get_text() for t in ax.get_legend().get_texts())
    # and a model can be forced
    forced = vt.fit_vt_regions(d, [{"T_range": (297, 311), "model": "arrhenius"}])
    assert forced["fits"][0]["kind"] == "arrhenius"


def test_bpp_regions_in_the_gui(qapp):
    window = MainWindow()
    panel = window.vt_relaxation_panel
    T = np.linspace(200, 420, 12)
    rate = vt.relaxation_rate(T, 100.0, 1e-13, 0.3, 3e9)
    panel.set_rows([["data", f"{t:g}", f"{1 / r:.6g}", "", "100"] for t, r in zip(T, rate)])
    panel.add_region(200, 300, name="cold")
    panel.add_region(300, 420, name="hot")
    result = panel.run_region_fits()
    assert result is not None and len(result["fits"]) == 2
    assert all("error" not in f for f in result["fits"])
    names = [panel.results_table.item(r, 0).text() for r in range(panel.results_table.rowCount())]
    assert any(n.startswith("cold: Ea_eV") for n in names)
    assert any(n.startswith("hot: Ea_eV") for n in names)


# ---- Offset Tracker -------------------------------------------------------------

def _offset_spectra():
    x = np.linspace(50, -50, 2001)
    out = []
    for o2 in np.arange(-4000.0, 4001.0, 500.0):
        y = np.exp(-(o2 / 2000.0) ** 2) * np.exp(-x ** 2 / 2)
        meta = NMRMetadata(vendor="bruker", nucleus="13C", sfo1=100.0, bf1=100.0, ns=1.0,
                           raw={"acqus": {"O2": o2, "BF2": 400.0, "NUC2": "<1H>", "NS": 1}})
        out.append(NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x, hz=x * 100.0)],
                               metadata=meta, data_imag=np.zeros_like(y)))
    return out


def test_offset_series_and_skyline():
    series = ot.offset_series(_offset_spectra(), (5, -5), channel="O2")
    assert series["offset_ppm"][0] == pytest.approx(-10.0)
    assert series["nucleus"] == "1H" and series["bf_MHz"] == pytest.approx(400.0)
    peak = int(np.argmax(series["y"]))
    assert series["offset_ppm"][peak] == pytest.approx(0.0)
    sky = ot.skyline_spectrum(series, n_points=1001)
    assert sky.metadata.nucleus == "1H" and sky.data.size == 1001
    centre = int(np.argmin(np.abs(sky.axes[0].ppm)))
    assert sky.data[centre] == pytest.approx(series["y"][peak])
    with pytest.raises(ValueError):
        ot.channel_offset(_offset_spectra()[0], "O3")


def test_offset_tracker_panel_reads_integrates_and_keeps(qapp):
    window = MainWindow()
    for s in _offset_spectra():
        _load_spectrum_directly(window, s)
    panel = window.offset_tracker_panel
    panel.refresh()
    panel.tick_all_button.click()
    assert panel.read_offsets() == 17
    panel.set_region(5, -5)
    panel.style_combo.setCurrentIndex(panel.style_combo.findData("triangles"))
    series = panel.integrate()
    assert series is not None and len(series["y"]) == 17
    n = len(window.session)
    kept = panel.keep_skyline()
    assert kept is not None and len(window.session) == n + 1
    assert window.session.spectra[-1].metadata.nucleus == "1H"
