"""2026-10-16 round: the Integrate tile (integration regions, DNP, S/N and
CP Build-up), selecting a region integrates it, plot formatting on the
plot's right-click menu (fit components, curve fit, Relaxation vs
Temperature, exchange stack), the curve-fit peak browser by the plot, and
Dynamics ▸ Regularisation (the MATLAB_RELAX_ALGORITHM scripts)."""

import numpy as np
import pytest
from PySide6.QtWidgets import QLineEdit, QMessageBox, QPushButton

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.canvas import MplCanvas

from test_gui import qapp, _load_spectrum_directly, _set_delay_unit  # noqa: F401

DELAYS = np.geomspace(1e-3, 2.0, 16)


@pytest.fixture
def said(monkeypatch):
    out = []
    for name in ("critical", "information", "warning"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: out.append((_n, a[2] if len(a) > 2 else "")))
    return out


def _one_d(height=1.0):
    ppm = np.linspace(10.0, 0.0, 2000)
    y = height * np.exp(-0.5 * ((ppm - 5.0) / 0.05) ** 2) \
        + 0.5 * np.exp(-0.5 * ((ppm - 2.0) / 0.05) ** 2)
    return NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
                       metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))


def _series(two_peaks=True):
    ppm = np.linspace(7.0, 3.0, 800)
    rng = np.random.default_rng(3)
    rows = []
    for t in DELAYS:
        row = np.exp(-t / 0.05) * np.exp(-0.5 * ((ppm - 5) / 0.03) ** 2)
        if two_peaks:
            row += 0.6 * np.exp(-t / 0.4) * np.exp(-0.5 * ((ppm - 6) / 0.03) ** 2)
        rows.append(row + rng.normal(0, 0.002, ppm.size))
    return NMRSpectrum(data=np.array(rows),
                       axes=[NMRAxis(kind="index", index=np.arange(1, len(DELAYS) + 1)),
                             NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
                       metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0,
                                            pseudo_2d=True))


def _menu_texts(monkeypatch, canvas):
    menus = []
    monkeypatch.setattr(MplCanvas, "_exec_context_menu",
                        lambda self, menu, pos: menus.append(menu) or None)
    canvas._on_context_menu(canvas.rect().center())
    return {a.text(): a for a in menus[-1].actions()}


def _fitted_series(window):
    _load_spectrum_directly(window, _series())
    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, "s")
    window.manual_delays_edit.setText(", ".join(f"{d:.8g}" for d in DELAYS))
    window._add_region_row(4.8, 5.2)
    window._add_region_row(5.8, 6.2)
    window._on_integrate_regions_clicked()
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("exponential_decay"))
    window._on_fit_curve_button_clicked()
    assert window._relaxation_fit_results
    return window


# ----------------------------------------------------------------------
# The Integrate tile
# ----------------------------------------------------------------------

def test_integrate_tile_sits_after_fit_and_holds_the_area_tools(qapp):  # noqa: F811
    window = MainWindow()
    tiles = [window.side_tabs.tabText(i) for i in range(window.side_tabs.count())]
    assert tiles.index("Integrate") == tiles.index("Fit") + 1
    page = window.side_tabs.widget(window._integrate_tab_index)
    for section in (window._integration_section, window._dnp_section, window._snr_section,
                    window._cp_buildup_section):
        assert page.isAncestorOf(section)
    apply_page = window.side_tabs.widget(tiles.index("Apply"))
    assert apply_page.isAncestorOf(window._enmr_section)
    assert not apply_page.isAncestorOf(window._dnp_section)
    fit_page = window.side_tabs.widget(window._fitting_tab_index)
    assert not fit_page.isAncestorOf(window.region_table)
    assert not window._integration_section.isExpanded()     # sections start collapsed
    # Window ▸ Integration opens it; the numbered shortcuts stop at nine.
    window._window_panel_actions["Integrate"].trigger()
    assert window.side_tabs.currentIndex() == window._integrate_tab_index
    assert window._window_panel_actions["Periodic Table"].shortcut().isEmpty()


def test_one_select_and_integrate_button_without_the_command_symbol(qapp):  # noqa: F811
    window = MainWindow()
    page = window.side_tabs.widget(window._integrate_tab_index)
    texts = [b.text() for b in page.findChildren(QPushButton)]
    assert "Select and Integrate Regions from Plot" in texts
    assert "Integrate Regions" not in texts
    assert not any("⌘" in t for t in texts)
    assert "⌘" not in window.select_region_button.toolTip()
    menu_texts = [a.text() for a in window.menuBar().findChildren(type(window.select_region_menu_action))]
    assert "&Integrate Regions" not in menu_texts
    window.select_region_menu_action.trigger()
    assert window.select_region_button.isChecked()


def test_a_dragged_region_is_integrated_at_once(qapp, said):  # noqa: F811
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    spectrum = window.session.spectra[0]
    window.canvas.regionSelected.emit(4.7, 5.3)
    results = spectrum.metadata.raw.get("slice_fit_results")
    assert results and len(results[0]["integrals"]) == 1
    expected = 0.05 * np.sqrt(2 * np.pi)
    assert results[0]["integrals"][0] == pytest.approx(expected, rel=0.02)
    assert "Region 1 (4.7 to 5.3)" in window.integration_results_label.text()
    # a second region: both integrated, with their shares
    window.canvas.regionSelected.emit(1.7, 2.3)
    results = spectrum.metadata.raw["slice_fit_results"]
    assert len(results[0]["integrals"]) == 2
    assert "%" in window.integration_results_label.text()
    assert not said


def test_typed_removed_and_found_regions_reintegrate_but_reselection_does_not(qapp, said):  # noqa: F811
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    _load_spectrum_directly(window, _one_d(height=2.0))
    first, second = window.session.spectra
    window._select_single_row(0)
    window._add_region_row(4.7, 5.3)
    assert not first.metadata.raw.get("slice_fit_results")      # adding a blank row: no
    window.region_table.item(0, 1).setText("5.3")
    window.region_table.itemDelegate().commitData.emit(QLineEdit())   # a user's edit
    assert first.metadata.raw.get("slice_fit_results")
    # Reselecting refills the table from the spectrum's own state: nothing integrated.
    window._select_single_row(1)
    window._select_single_row(0)
    assert not second.metadata.raw.get("slice_fit_results")
    window.region_table.setCurrentCell(0, 0)
    window._on_remove_region_clicked()
    assert "No regions" in window.integration_results_label.text()
    # Find Regions Automatically integrates what it found.
    window._on_find_regions_button_clicked()
    assert len(first.metadata.raw["slice_fit_results"][0]["integrals"]) == 2
    assert not [s for s in said if s[0] == "critical"]


def test_auto_integration_on_a_2d_says_so_quietly(qapp, said):  # noqa: F811
    window = MainWindow()
    spectrum = _series()
    spectrum.metadata.pseudo_2d = False
    spectrum.axes[0] = NMRAxis(kind="frequency", ppm=np.linspace(1, 0, len(DELAYS)),
                               hz=np.linspace(1, 0, len(DELAYS)) * 400.0)
    _load_spectrum_directly(window, spectrum)
    if window.session.spectra[0].is_pseudo_2d:
        pytest.skip("synthetic 2D read as a pseudo-2D")
    window.canvas.regionSelected.emit(4.8, 5.2)
    assert not said
    assert "conventional 2D" in window.integration_results_label.text()


# ----------------------------------------------------------------------
# Plot options on the right-click menu
# ----------------------------------------------------------------------

def test_fit_plot_components_and_curve_fit_plot_are_right_click_windows(qapp, monkeypatch):  # noqa: F811
    window = _fitted_series(MainWindow())
    fit_page = window.side_tabs.widget(window._fitting_tab_index)
    assert not fit_page.isAncestorOf(window.fit_display_panel)
    dynamics = window.side_tabs.widget(
        [window.side_tabs.tabText(i) for i in range(window.side_tabs.count())].index("Dynamics"))
    assert not dynamics.isAncestorOf(window.curve_fit_plot_box)
    assert not dynamics.isAncestorOf(window.relaxation_uncertainty_checkbox)
    assert dynamics.isAncestorOf(window.curve_fit_plot_box.keep_button)
    actions = _menu_texts(monkeypatch, window.fit_canvas)
    assert "Curve Fit Plot…" in actions
    assert "Fit Plot: Components and Colours…" not in actions
    actions["Curve Fit Plot…"].trigger()
    assert window._curve_fit_plot_dialog.isVisible()
    # the options still drive the plot
    window.relaxation_uncertainty_checkbox.setChecked(True)
    assert window.fit_canvas.ax.collections                       # the shaded band
    x = np.linspace(0, 10, 200)
    comps = [np.exp(-(x - 4) ** 2), np.exp(-(x - 6) ** 2)]
    window.fit_canvas.show_slice_fit(x, sum(comps), sum(comps), components=comps,
                                     peak_labels=["A", "B"])
    actions = _menu_texts(monkeypatch, window.fit_canvas)
    assert "Fit Plot: Components and Colours…" in actions
    assert "Curve Fit Plot…" not in actions


def test_dynamics_plot_options_are_right_click_windows(qapp, monkeypatch):  # noqa: F811
    window = MainWindow()
    vt = window.vt_relaxation_panel
    assert not vt.isAncestorOf(vt.y_axis_combo)
    vt.table.setRowCount(0)
    for T, value in ((250, 0.5), (300, 0.2), (350, 0.08), (400, 0.05)):
        vt.add_row(["A", T, value, "", 400])
    vt.plot_data()
    assert window.fit_canvas.plot_owner == "vt"
    actions = _menu_texts(monkeypatch, window.fit_canvas)
    actions["Relaxation vs Temperature Plot…"].trigger()
    assert vt.plot_options_dialog.isVisible()
    label_before = window.fit_canvas.figure.axes[0].get_xlabel()
    vt.x_axis_combo.setCurrentIndex(vt.x_axis_combo.findData("T"))
    assert window.fit_canvas.figure.axes[0].get_xlabel() != label_before
    # a restyle keeps the entry
    window.fit_canvas._redraw_external()
    assert "Relaxation vs Temperature Plot…" in _menu_texts(monkeypatch, window.fit_canvas)

    ex = window.exchange_panel
    assert not ex.isAncestorOf(ex.fast_limit_check)
    ex.temperatures_edit.setText("250, 350")
    ex.simulate()
    assert window.fit_canvas.plot_owner == "exchange_stack"
    actions = _menu_texts(monkeypatch, window.fit_canvas)
    assert "Exchange Lineshape Plot…" in actions
    assert "Relaxation vs Temperature Plot…" not in actions
    lines = len(window.fit_canvas.figure.axes[0].lines)
    ex.fast_limit_check.setChecked(True)
    assert len(window.fit_canvas.figure.axes[0].lines) > lines


# ----------------------------------------------------------------------
# The curve-fit peak browser by the plot
# ----------------------------------------------------------------------

def test_peak_browser_lives_by_the_plot_view(qapp):  # noqa: F811
    window = _fitted_series(MainWindow())
    bar = window.curve_peak_bar
    assert window.canvas_tabs.cornerWidget() is bar
    dynamics = window.side_tabs.widget(
        [window.side_tabs.tabText(i) for i in range(window.side_tabs.count())].index("Dynamics"))
    assert not dynamics.isAncestorOf(window.view_relaxation_peak_spin)
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    assert not bar.isHidden()
    window.relaxation_peak_next_button.click()
    assert window.view_relaxation_peak_spin.value() == 2
    assert window.fit_canvas.shown_fit()["kind"] == "curve"
    window.canvas_tabs.setCurrentWidget(window.canvas)
    assert bar.isHidden()
    window.canvas_tabs.setCurrentWidget(window.fit_canvas)
    assert not bar.isHidden()
    # another plot on the Fit view hides it once drawn
    window.vt_relaxation_panel.add_row(["A", 300, 0.2, "", 400])
    window.vt_relaxation_panel.add_row(["A", 350, 0.1, "", 400])
    window.vt_relaxation_panel.plot_data()
    qapp.processEvents()
    assert bar.isHidden()


# ----------------------------------------------------------------------
# Dynamics ▸ Regularisation
# ----------------------------------------------------------------------

def test_regularisation_from_fit_curve_and_from_files(qapp, monkeypatch, tmp_path, said):  # noqa: F811
    window = _fitted_series(MainWindow())
    section = window._regularisation_section
    dynamics = window.side_tabs.widget(
        [window.side_tabs.tabText(i) for i in range(window.side_tabs.count())].index("Dynamics"))
    assert dynamics.isAncestorOf(section)
    section.setExpanded(True)
    panel = window.regularisation_panel
    assert panel.series_combo.count() == 2
    assert panel.kernel_combo.currentData() == "t2"
    lo, hi = panel.grid_row.grid_range()
    assert lo == pytest.approx(DELAYS.min() / 3, rel=1e-3) and hi == pytest.approx(6.0, rel=1e-3)
    panel.solver_combo.setCurrentIndex(panel.solver_combo.findData("nonnegative"))
    result = panel.run()
    assert result is not None and not said
    assert result["peaks"][0]["position"] == pytest.approx(0.05, rel=0.2)
    assert window.fit_canvas.plot_owner == "regularisation"
    assert "Regularisation Plot…" in _menu_texts(monkeypatch, window.fit_canvas)
    assert "Log mean T₂" in panel.results_label.text()
    n_axes = len(window.fit_canvas.figure.axes)
    panel.show_residuals_check.setChecked(False)
    assert len(window.fit_canvas.figure.axes) == n_axes - 1
    code = panel.notebook_code()
    namespace = {}
    import matplotlib
    matplotlib.use("Agg")
    exec(code.replace("print(", "str("), namespace)          # the copied code runs
    assert namespace["result"]["lambda"] == pytest.approx(result["lambda"])

    # a two-column file in ms, as the MATLAB scripts read
    t = np.linspace(1e-3, 0.6, 150)
    path = tmp_path / "decay.txt"
    np.savetxt(path, np.c_[t * 1e3, np.exp(-t / 0.1)], header="t/ms  y")
    panel.set_file_1d(str(path))
    panel.x_unit_combo.setCurrentIndex(panel.x_unit_combo.findText("ms"))
    panel._range_from_data()
    result = panel.run()
    assert result["peaks"][0]["position"] == pytest.approx(0.1, rel=0.15)
    out = tmp_path / "out.csv"
    monkeypatch.setattr("opennmr.gui.regularisation_panel.QFileDialog.getSaveFileName",
                        lambda *a, **k: (str(out), ""))
    panel.export_csv()
    assert out.exists()

    # 2D: a matrix and two lists
    te = np.linspace(2e-4, 0.3, 60)
    vd = np.logspace(-3, 1, 10)
    data = 0.5 * np.outer(1 - 2 * np.exp(-vd / 0.05), np.exp(-te / 0.01)) \
        + 0.5 * np.outer(1 - 2 * np.exp(-vd / 1.0), np.exp(-te / 0.1))
    np.savetxt(tmp_path / "m.txt", data)
    np.savetxt(tmp_path / "te.txt", te * 1e3)
    np.savetxt(tmp_path / "vd.txt", vd)
    panel.dimension_combo.setCurrentIndex(1)
    panel.set_file_2d("data", str(tmp_path / "m.txt"))
    panel.set_file_2d("x1", str(tmp_path / "te.txt"))
    panel.set_file_2d("x2", str(tmp_path / "vd.txt"))
    panel.x1_unit_combo.setCurrentIndex(panel.x1_unit_combo.findText("ms"))
    panel.x2_unit_combo.setCurrentIndex(panel.x2_unit_combo.findText("s"))
    panel.solver_combo.setCurrentIndex(panel.solver_combo.findData("matlab"))
    panel.grid1_row.points.setValue(20)
    panel.grid2_row.points.setValue(20)
    panel.grid1_row.set_range((1e-4, 1.0))
    panel.grid2_row.set_range((1e-3, 10.0))
    result = panel.run()
    assert result["kind"] == "2d" and result["map"].shape == (20, 20)
    assert result["log_mean1"] == pytest.approx(np.sqrt(0.01 * 0.1), rel=0.25)
    assert not said


def test_regularisation_says_what_is_missing(qapp, said):  # noqa: F811
    window = MainWindow()
    panel = window.regularisation_panel
    panel.refresh()
    assert panel.run() is None
    assert said and "Fit Curve" in said[-1][1]
    panel.dimension_combo.setCurrentIndex(1)
    assert panel.run() is None
    assert "Choose the data matrix" in said[-1][1]


def test_regularisation_is_an_optional_section(qapp):  # noqa: F811
    assert "regularisation" in MainWindow.OPTIONAL_SECTIONS
    window = MainWindow()
    window._apply_section_shown("regularisation", False)
    assert window._regularisation_section.isHidden()
