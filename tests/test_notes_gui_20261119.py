"""PFG notes §8 (2026-11-19): the Value source only offers what the stored results can supply,
and the integral-label options appear only once the labels are switched on."""
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.gui.main_window import MainWindow

from test_gui import qapp  # noqa: F401 -- fixture


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)


PEAKS = [{"popt": [1.0], "r_squared": 0.99}]
REGIONS = [{"regions": [{"integral": 1.0}], "index": 1}]


def test_slice_results_kind():
    kind = MainWindow._slice_results_kind
    assert kind(PEAKS) == "peaks"
    assert kind([{"r_squared": 0.9}]) == "peaks"          # a lineshape series fit
    assert kind(REGIONS) == "regions"
    assert kind(None) is None and kind([]) is None and kind([{"other": 1}]) is None


def _enabled(win):
    combo = win.value_source_combo
    return {combo.itemData(i): combo.model().item(i).isEnabled() for i in range(combo.count())}


def test_region_is_greyed_out_while_only_peak_fits_exist(qapp):
    win = MainWindow()
    win._slice_fit_results = PEAKS
    win.value_source_combo.setCurrentIndex(win.value_source_combo.findData("region"))
    win._sync_value_source_availability()
    enabled = _enabled(win)
    assert enabled["region"] is False
    assert enabled["area"] and enabled["amplitude"] and enabled["amplitude_raw"]
    assert win.value_source_combo.currentData() == "area"          # moved off the dead option
    assert "Fit Spectrum" not in win.value_source_combo.model().item(
        win.value_source_combo.findData("area")).toolTip()
    assert "integrat" in win.value_source_combo.model().item(
        win.value_source_combo.findData("region")).toolTip().lower()


def test_peak_options_are_greyed_out_while_only_integrals_exist(qapp):
    win = MainWindow()
    win._slice_fit_results = REGIONS
    win._sync_value_source_availability()
    enabled = _enabled(win)
    assert enabled["region"] is True
    assert not enabled["area"] and not enabled["amplitude"]
    assert win.value_source_combo.currentData() == "region"


def test_everything_is_offered_when_nothing_is_stored(qapp):
    win = MainWindow()
    win._slice_fit_results = None
    win._sync_value_source_availability()
    assert all(_enabled(win).values())


def test_coerce_can_be_switched_off(qapp):
    win = MainWindow()
    win._slice_fit_results = PEAKS
    win.value_source_combo.setCurrentIndex(win.value_source_combo.findData("region"))
    win._sync_value_source_availability(coerce=False)
    assert win.value_source_combo.currentData() == "region"


def test_integral_label_options_follow_the_checkbox(qapp):
    win = MainWindow()
    assert win.integral_labels_options.isHidden()
    win.integral_labels_check.setChecked(True)
    assert not win.integral_labels_options.isHidden()
    win.integral_labels_check.setChecked(False)
    assert win.integral_labels_options.isHidden()
    # the widgets inside are still the same objects the rest of the code reads
    assert win.integral_labels_decimals_spin.value() == 2
    assert win.integral_labels_symbol_check.isChecked()
