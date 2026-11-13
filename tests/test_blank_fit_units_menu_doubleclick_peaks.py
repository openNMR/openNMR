"""2026-11-13 requests: a new dataset gets a blank fit; units in Compare;
Settings… in the File menu; double-clicking a collapsible section; the
Fit peak table keeps full precision."""

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.gui.panel_state import table_text
from opennmr.gui.shortcut_sheet import collect_shortcuts, find_conflicts
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.gui.widgets.collapsible_section import CollapsibleSection

from test_gui import qapp, _menu_by_title  # noqa: F401 -- qapp is a fixture


@pytest.fixture(autouse=True)
def _no_dialogs(monkeypatch):
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)


def _spectrum(centre=5.0, sfo1=400.0, nucleus="1H", n=400):
    ppm = np.linspace(10, 0, n)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * sfo1)
    return NMRSpectrum(data=np.exp(-((ppm - centre) / 0.2) ** 2), axes=[axis],
                       metadata=NMRMetadata(vendor="simulated", nucleus=nucleus, sfo1=sfo1))


# ------------------------------------------------------------------
# 1. A new dataset starts from a blank fit
# ------------------------------------------------------------------

def test_a_removed_spectrums_peaks_do_not_land_on_the_next_one_loaded(qapp):
    # The analysis panel's state was saved by row number: remove the
    # spectrum it belonged to, load another into that row, and the old
    # peaks (and fit) were saved onto -- and shown for -- the new one.
    window = MainWindow()
    window._add_derived_spectrum(_spectrum(), "first")
    window._add_peak_row(shift=5.0, fwhm=0.4)
    assert window.peak_table.rowCount() == 1
    window._remove_rows([0])
    index = window._add_derived_spectrum(_spectrum(3.0), "second")
    assert window.peak_table.rowCount() == 0
    state = window.session.spectra[index].metadata.raw.get("analysis_state") or {}
    assert not state.get("peaks")


def test_analysis_state_follows_its_spectrum_when_rows_shift(qapp):
    window = MainWindow()
    a = window._add_derived_spectrum(_spectrum(), "a")
    b = window._add_derived_spectrum(_spectrum(3.0), "b")
    window._add_peak_row(shift=3.0, fwhm=0.4)           # on b
    window._select_single_row(a)
    assert window.peak_table.rowCount() == 0
    window._select_single_row(b)
    assert window.peak_table.rowCount() == 1
    window._remove_rows([a])                              # b moves to row 0
    window._add_derived_spectrum(_spectrum(7.0), "c")
    assert window.peak_table.rowCount() == 0              # c: blank
    window._select_single_row(0)                          # b again
    assert window.peak_table.rowCount() == 1


def test_fit_panel_inputs_are_blank_for_a_new_spectrum_and_kept_per_spectrum(qapp):
    window = MainWindow()
    panel = window.quad_fit_panel
    blank = table_text(panel.sites_table)
    first = window._add_derived_spectrum(_spectrum(nucleus="27Al", sfo1=130.3), "first")
    panel.add_site()
    panel.sites_table.item(0, 2).setText("9.9")
    edited = table_text(panel.sites_table)
    assert edited != blank
    second = window._add_derived_spectrum(_spectrum(nucleus="27Al", sfo1=130.3), "second")
    assert table_text(panel.sites_table) == blank          # a new dataset: a blank fit
    window._select_single_row(first)                       # its unfitted inputs come back
    assert table_text(panel.sites_table) == edited
    window._select_single_row(second)
    assert table_text(panel.sites_table) == blank


def test_inputs_set_up_before_the_first_spectrum_are_kept_for_it(qapp):
    window = MainWindow()
    panel = window.quad_fit_panel
    panel.add_site()
    rows = panel.sites_table.rowCount()
    window._add_derived_spectrum(_spectrum(nucleus="27Al", sfo1=130.3), "first")
    assert panel.sites_table.rowCount() == rows


# ------------------------------------------------------------------
# 2. Units in Compare
# ------------------------------------------------------------------

def test_an_overlay_compare_changes_unit_with_each_spectrums_own_frequency(qapp):
    canvas = MplCanvas()
    low, high = _spectrum(sfo1=400.0), _spectrum(sfo1=600.0)
    canvas.show_compare([low, high], mode="overlay")
    canvas.set_axis_unit("hz")
    lines = canvas.ax.lines[:2]
    np.testing.assert_allclose(np.sort(lines[0].get_xdata()), np.sort(low.axes[0].values("hz")),
                               rtol=1e-9)
    np.testing.assert_allclose(np.sort(lines[1].get_xdata()), np.sort(high.axes[0].values("hz")),
                               rtol=1e-9)
    assert canvas.axis_unit == "hz" and "Hz" in canvas.ax.get_xlabel()
    assert "different spectrometer frequencies" in canvas.compare_unit_note()
    canvas.set_axis_unit("ppm")
    np.testing.assert_allclose(np.sort(lines[0].get_xdata()), np.sort(low.axes[0].values("ppm")),
                               rtol=1e-9, atol=1e-12)
    assert canvas.compare_unit_note() == ""


def test_an_overlay_at_one_field_needs_no_warning_and_keeps_its_zoom(qapp):
    canvas = MplCanvas()
    canvas.show_compare([_spectrum(), _spectrum(3.0)], mode="overlay")
    canvas.ax.set_xlim(6.0, 4.0)
    canvas.set_axis_unit("khz")
    assert canvas.ax.get_xlim() == pytest.approx((2.4, 1.6))
    assert canvas.compare_unit_note() == ""


def test_a_stack_converts_every_row_once(qapp):
    canvas = MplCanvas()
    spectra = [_spectrum(), _spectrum(3.0), _spectrum(7.0)]
    canvas.show_compare(spectra, mode="stack")
    before = np.array(canvas._compare_axes[0].get_xlim())
    canvas.set_axis_unit("hz")
    for ax, spectrum in zip(canvas._compare_axes, spectra):
        np.testing.assert_allclose(np.sort(ax.lines[0].get_xdata()),
                                   np.sort(spectrum.axes[0].values("hz")), rtol=1e-9)
    # The rows share their x-axis: converted once, not once per row.
    for ax in canvas._compare_axes:
        assert np.array(ax.get_xlim()) == pytest.approx(before * 400.0)


def test_panels_convert_and_a_redraw_keeps_the_unit(qapp):
    canvas = MplCanvas()
    spectra = [_spectrum(), _spectrum(3.0)]
    canvas.show_compare_panels(spectra, [[0], [1]], layout="row")
    canvas.set_axis_unit("hz")
    for ax, spectrum in zip(canvas._spectrum_panels, spectra):
        assert np.max(ax.lines[0].get_xdata()) == pytest.approx(4000.0)
    canvas.show_compare_panels(spectra, [[0], [1]], layout="row")
    assert canvas.axis_unit == "hz"
    assert np.max(canvas._spectrum_panels[0].lines[0].get_xdata()) == pytest.approx(4000.0)
    canvas.show_compare(spectra, mode="overlay")
    assert np.max(canvas.ax.lines[0].get_xdata()) == pytest.approx(4000.0)


def test_compare_units_box_drives_and_follows_the_plot(qapp):
    window = MainWindow()
    combo = window.compare_unit_combo
    assert [combo.itemData(i) for i in range(combo.count())] == ["ppm", "hz", "khz"]
    canvas = window.compare_canvas
    canvas.show_compare([_spectrum(), _spectrum(3.0)], mode="stack")
    combo.setCurrentIndex(combo.findData("khz"))
    assert canvas.axis_unit == "khz"
    canvas.set_axis_unit("hz")
    QApplication.processEvents()                 # the change is announced once it has happened
    assert combo.currentData() == "hz"


def test_with_nothing_drawn_the_next_compare_uses_the_chosen_unit(qapp):
    canvas = MplCanvas()
    canvas.set_compare_unit("hz")
    canvas.show_compare([_spectrum()], mode="overlay")
    assert np.max(canvas.ax.lines[0].get_xdata()) == pytest.approx(4000.0)


# ------------------------------------------------------------------
# 3. Settings… in the File menu
# ------------------------------------------------------------------

def test_settings_is_in_file_and_window_as_one_action(qapp):
    window = MainWindow()
    file_actions = _menu_by_title(window, "File").actions()
    window_actions = _menu_by_title(window, "Window").actions()
    assert window.settings_menu_action in file_actions
    assert window.settings_menu_action in window_actions
    window.settings_menu_action.trigger()
    assert window.side_tabs.currentIndex() == window._settings_tab_index
    # One action in two menus is one shortcut, not a conflict.
    rows = collect_shortcuts(window)
    assert sum(1 for r in rows if r.portable == "Ctrl+,") == 1
    assert "Ctrl+," not in find_conflicts(rows)


# ------------------------------------------------------------------
# 5. Double-click a collapsible section
# ------------------------------------------------------------------

def _double_click(widget):
    pos = QPointF(widget.width() / 2, widget.height() / 2)
    glob = widget.mapToGlobal(pos.toPoint())
    for kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease,
                 QEvent.Type.MouseButtonDblClick, QEvent.Type.MouseButtonRelease):
        buttons = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease \
            else Qt.MouseButton.LeftButton
        event = QMouseEvent(kind, pos, QPointF(glob), Qt.MouseButton.LeftButton, buttons,
                            Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(widget, event)


def test_double_clicking_a_section_header_opens_and_closes_it(qapp):
    section = CollapsibleSection("Phase", content=QLabel("body"))
    section.resize(300, 200)
    section.show()
    QApplication.processEvents()
    header = section._header
    _double_click(header)
    assert section.isExpanded() and not section._body.isHidden()
    _double_click(header)
    assert not section.isExpanded() and section._body.isHidden()
    section.close()


def test_double_clicking_the_other_disclosure_toggles_flips_them_once(qapp):
    window = MainWindow()
    container, toggle = window._collapsible_section("Tuning", QLabel("x"))
    container.resize(300, 200)
    container.show()
    QApplication.processEvents()
    _double_click(toggle)
    assert toggle.isChecked()
    _double_click(toggle)
    assert not toggle.isChecked()
    container.close()


# ------------------------------------------------------------------
# 7. The Fit peak table keeps full precision
# ------------------------------------------------------------------

def test_peak_positions_are_used_exactly_not_as_shown(qapp):
    window = MainWindow()
    window._add_derived_spectrum(_spectrum(), "x")
    window._add_peak_row(shift=123.456789, fwhm=0.00123456, eta=0.4321)
    peak = window._read_peak_defs()[0]
    assert peak["shift"] == 123.456789                     # exactly, not 123.5
    assert peak["fwhm"] == 0.00123456
    assert window.peak_table.item(0, 0).text() == "123.4568"
    # A typed value wins.
    window.peak_table.item(0, 0).setText("120.25")
    assert window._read_peak_defs()[0]["shift"] == 120.25


def test_peak_positions_survive_a_save_and_restore_exactly(qapp):
    window = MainWindow()
    a = window._add_derived_spectrum(_spectrum(), "a")
    window._add_peak_row(shift=7.2645183, fwhm=0.0008123)
    state = window._capture_analysis_state()
    assert state["peak_exact"][0][:2] == [7.2645183, 0.0008123]
    window._add_derived_spectrum(_spectrum(3.0), "b")
    window._select_single_row(a)
    peak = window._read_peak_defs()[0]
    assert (peak["shift"], peak["fwhm"]) == (7.2645183, 0.0008123)


def test_a_fit_writes_its_result_back_exactly(qapp):
    window = MainWindow()
    window._add_derived_spectrum(_spectrum(), "x")
    window._add_peak_row(shift=4.9, fwhm=0.3)
    window.peak_table.setItem(0, 0, window._make_peak_value_item(5.000012345678))
    assert window._read_peak_defs()[0]["shift"] == 5.000012345678
    assert window._picked_line_shifts() == [5.000012345678]
