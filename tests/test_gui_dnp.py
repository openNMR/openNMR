"""Compare > DNP (gui/dnp_panel.py): the panel calls opennmr.notebook.dnp
as a notebook would, picks and drags arrive from the Compare view, and
the results are drawn, tabulated and kept with the spectra they used.

Spectra are the synthetic ones of tests/test_dnp.py, added to the
session as derived spectra, so no instrument data is needed.
"""
import numpy as np
import pytest

from opennmr.notebook import dnp
from test_dnp import DELTA, ERRORS, FIELDS, GAMMA_NA, _pair, _rolled, _sweep


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1400, 900)
    return w


def _load(window, spectra, labels):
    for s, label in zip(spectra, labels):
        window._add_derived_spectrum(s, label)
    window.spectrum_list.selectAll()
    window._show_side_tab("Compare")
    window._dnp_section.setExpanded(True)
    return window.dnp_panel


def test_dnp_box_lives_in_compare_and_starts_collapsed(window):
    assert not window._dnp_section.isExpanded()
    assert window._dnp_section.parent() is not None
    # Inside the Compare panel's own page.
    compare_page = window.compare_list.parentWidget()
    assert window._dnp_section.parentWidget() is compare_page


def test_panel_lists_the_checked_compare_spectra(window):
    panel = _load(window, _sweep(FIELDS, ERRORS), [f"s{k}" for k in range(5)])
    labels = [e["label"] for e in window._dnp_entries()]
    assert len(labels) == 5
    assert [panel.enh_combo_1.itemText(i) for i in range(panel.enh_combo_1.count())] == labels
    assert "5 spectra checked, 23Na" in panel.sweep_status_label.text()
    # Unchecking one drops it from the panel.
    window.compare_list.item(0).setCheckState(window.compare_list.item(0).checkState().Unchecked)
    assert panel.enh_combo_1.count() == 4


def test_enhancement_from_a_click_and_a_drag(window):
    on, off, pk = _pair(h_on=12500.0, h_off=1000.0)
    panel = _load(window, [on, off], ["on", "off"])
    panel.enh_combo_1.setCurrentIndex(0)
    panel.enh_combo_2.setCurrentIndex(1)

    panel.enh_pick_button.setChecked(True)
    assert window.compare_canvas._pick_mode                 # armed on the Compare view
    assert window.canvas_tabs.currentWidget() is window.compare_canvas
    window.compare_canvas.peakPicked.emit(float(pk + 1.0))
    assert not window.compare_canvas._pick_mode             # one gesture, then released
    assert not panel.enh_pick_button.isChecked()
    assert panel.enh_result["intensity"]["ratio"] == pytest.approx(12.5, rel=0.01)
    # Encouraged, not forced: the other measurement is asked for.
    assert "integrate a region too" in panel.enh_result_label.text()

    panel.enh_region_button.setChecked(True)
    assert window.compare_canvas._region_select_mode
    window.compare_canvas.regionSelected.emit(float(pk - 60), float(pk + 60))
    r = panel.enh_result
    assert r["area"]["ratio"] == pytest.approx(12.5, rel=0.01)
    assert "agree" in panel.enh_result_label.text()


def test_enhancement_baseline_choice_reaches_the_calculation(window):
    on, off, pk = _pair(h_on=3000.0, h_off=1000.0)
    _rolled(on, 4000.0)
    panel = _load(window, [on, off], ["on", "off"])
    panel.enh_combo_1.setCurrentIndex(0)
    panel.enh_combo_2.setCurrentIndex(1)
    panel.enh_peak = float(pk)
    panel.compute_enhancement()
    assert panel.enh_result["intensity"]["ratio"] == pytest.approx(3.0, rel=0.02)
    panel.enh_baseline_combo.setCurrentIndex(panel.enh_baseline_combo.findData("global"))
    assert panel.enh_result["settings"]["baseline"] == "global"
    assert abs(panel.enh_result["intensity"]["ratio"] - 3.0) > 0.3


def test_enhancement_is_kept_with_both_spectra(window):
    on, off, pk = _pair()
    panel = _load(window, [on, off], ["on", "off"])
    panel.enh_combo_1.setCurrentIndex(0)
    panel.enh_combo_2.setCurrentIndex(1)
    panel.enh_peak = float(pk)
    panel.enh_region = (float(pk - 60), float(pk + 60))
    panel.compute_enhancement()
    for s in window.session.spectra:
        record = s.metadata.raw["fits"]["dnp_enhancement"]
        labels = [e["label"] for e in window._dnp_entries()]
        assert record["target"] == f"{labels[0]} / {labels[1]}"
        assert "By peak height" in record["report"] and "By area" in record["report"]


def test_field_sweep_from_a_pick_on_the_carrier_offset_view(window):
    spectra = _sweep(FIELDS, ERRORS)
    panel = _load(window, spectra, [f"s{k}" for k in range(5)])
    assert not panel.sweep_run_button.isEnabled()           # nothing picked yet

    panel.sweep_pick_button.setChecked(True)
    ax = window.compare_canvas.figure.axes[0]
    assert "carrier" in ax.get_xlabel()                     # picking on the right axis
    window.compare_canvas.peakPicked.emit(0.0)              # kHz on that view
    assert panel.sweep_peak == 0.0
    panel.sweep_region_button.setChecked(True)
    window.compare_canvas.regionSelected.emit(-3.0, 3.0)
    assert panel.sweep_region == (-3000.0, 3000.0)
    assert panel.sweep_run_button.isEnabled()

    panel.sweep_shift_auto.setChecked(False)
    panel.sweep_shift_spin.setValue(DELTA)
    result = panel.run_sweep()
    for row, b in zip(result["rows"], FIELDS):
        assert row["field_T"] == pytest.approx(b, abs=2e-6)
    # Profile on the Fit view, table filled, report shown.
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    assert "B$_0$" in window.fit_canvas.figure.axes[-1].get_xlabel()
    assert panel.sweep_table.rowCount() == 5
    assert "SFO1 did not follow the line" in panel.sweep_report_label.text()
    # Kept with every spectrum of the sweep.
    assert all("dnp_sweep" in s.metadata.raw["fits"] for s in window.session.spectra)


def test_sweep_plot_choices_redraw_the_profile(window):
    spectra = _sweep(FIELDS, ERRORS)
    panel = _load(window, spectra, [f"s{k}" for k in range(5)])
    panel.sweep_peak, panel.sweep_region = 0.0, (-3000.0, 3000.0)
    panel.run_sweep()
    panel.sweep_quantity_combo.setCurrentIndex(panel.sweep_quantity_combo.findData("both"))
    assert len(window.fit_canvas.figure.axes) == 2
    panel.sweep_axis_combo.setCurrentIndex(2)               # ΔB0 from reference
    assert "ΔB" in window.fit_canvas.figure.axes[-1].get_xlabel()
    s1 = next(i for i in range(panel.sweep_normalise_combo.count())
              if panel.sweep_normalise_combo.itemText(i).startswith("s1"))
    panel.sweep_normalise_combo.setCurrentIndex(s1)
    assert "/ s1" in window.fit_canvas.figure.axes[0].get_ylabel()


def test_sweep_field_from_sfo1_option(window):
    spectra = _sweep(FIELDS, ERRORS)
    panel = _load(window, spectra, [f"s{k}" for k in range(5)])
    panel.sweep_peak = 0.0
    panel.sweep_field_combo.setCurrentIndex(panel.sweep_field_combo.findData("sfo1"))
    result = panel.run_sweep()
    assert {row["field_source"] for row in result["rows"]} == {"SFO1"}


def test_tracked_lines_view_and_csv_export(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    spectra = _sweep(FIELDS, ERRORS)
    panel = _load(window, spectra, [f"s{k}" for k in range(5)])
    panel.sweep_peak = 0.0
    panel.run_sweep()
    panel.sweep_aligned_check.setChecked(True)
    panel.sweep_tracked_button.click()
    assert "aligned" in window.compare_canvas.figure.axes[0].get_xlabel()
    path = tmp_path / "sweep.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    panel.sweep_csv_button.click()
    assert path.read_text(encoding="utf-8").startswith("# DNP field sweep")


def test_another_canvas_mode_releases_the_dnp_buttons(window):
    spectra = _sweep(FIELDS, ERRORS)
    panel = _load(window, spectra, [f"s{k}" for k in range(5)])
    panel.sweep_pick_button.setChecked(True)
    window.compare_canvas.set_measurement_mode(True)        # takes over the canvas
    assert not panel.sweep_pick_button.isChecked()
    window.compare_canvas.peakPicked.emit(1.0)              # a stray pick is ignored
    assert panel.sweep_peak is None


def test_messages_instead_of_errors_with_nothing_checked(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    window._dnp_section.setExpanded(True)
    panel = window.dnp_panel
    assert panel.run_sweep() is None
    assert not panel.show_enhancement_spectra()
    assert shown
