"""Integration regions -> pseudo-2D -> Dynamics ▸ Fit Curve (user report,
2026-10-06): after "Export Integration to Pseudo-2D…", Fit Curve on the
exported series said "Add at least one peak to fit." The export arrived
selected but carried neither the regions nor the integrals, so Fit Curve
fell through to a peak fit. It now arrives integrated, with the source's
regions and x values; and Fit Curve on any spectrum with regions but no
peaks integrates them instead of asking for a peak."""

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.gui.main_window import MainWindow

from test_gui import (qapp, _load_spectrum_directly, _make_pseudo2d_relaxation_spectrum,  # noqa: F401
                      _set_delay_unit)

TRUE_T2 = 0.05
DELAYS = np.geomspace(0.001, 0.3, 8)


@pytest.fixture
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[1:3])))
    return shown


def _window_with_series():
    window = MainWindow()
    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum(DELAYS, T2=TRUE_T2))
    window._select_single_row(0)
    window.relaxation_model_combo.setCurrentIndex(2)  # "Exponential decay (T2)"
    return window


def _type_delays(window):
    window.delay_source_combo.setCurrentIndex(window._MANUAL_LIST_SOURCE)
    _set_delay_unit(window, "s")
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in DELAYS))


def test_fit_curve_on_an_exported_integration_series(qapp, _dialogs):
    window = _window_with_series()
    _type_delays(window)
    window._add_region_row(4.5, 5.5)
    window._on_integrate_regions_clicked()
    window._on_export_integration_pseudo2d_clicked()
    exported_index = len(window.session.spectra) - 1
    assert window._selected_index() == exported_index
    exported = window.session.spectra[exported_index]
    # It arrives measured: the regions, and the integrals of its own slices.
    assert window._read_region_defs() == [(4.5, 5.5)]
    results = exported.metadata.raw.get("slice_fit_results")
    assert results and len(results) == len(DELAYS) and "regions" in results[0]
    source = window.session.spectra[0].metadata.raw["slice_fit_results"]
    np.testing.assert_allclose([r["integrals"][0] for r in results],
                               [r["integrals"][0] for r in source], rtol=1e-9)
    # ...and with the source's x values, written out (it has no vdlist of its own).
    assert window.delay_source_combo.currentIndex() == window._MANUAL_LIST_SOURCE
    window._on_fit_curve_button_clicked()
    assert not [d for d in _dialogs if d[0] in ("critical", "warning")], _dialogs
    assert window._relaxation_fit_results[0]["params"]["T2"] == pytest.approx(TRUE_T2, rel=0.05)


def test_export_carries_values_from_start_and_step(qapp, _dialogs):
    window = _window_with_series()
    window.delay_source_combo.setCurrentIndex(2)       # Start + step
    _set_delay_unit(window, "s")
    window.delay_start_spin.setValue(0.01)
    window.delay_step_spin.setValue(0.02)
    window._add_region_row(4.5, 5.5)
    window._on_integrate_regions_clicked()
    window._on_export_integration_pseudo2d_clicked()
    values = [float(v) for v in window.manual_delays_edit.text().split(",")]
    assert values == pytest.approx(0.01 + 0.02 * np.arange(len(DELAYS)))


def test_fit_curve_integrates_regions_when_there_are_no_peaks(qapp, _dialogs):
    window = _window_with_series()
    _type_delays(window)
    window._add_region_row(4.5, 5.5)                    # drawn, never integrated
    assert window.session.spectra[0].metadata.raw.get("slice_fit_results") is None
    window._on_fit_curve_button_clicked()
    assert not [d for d in _dialogs if d[0] in ("critical", "warning")], _dialogs
    assert window._relaxation_fit_results[0]["params"]["T2"] == pytest.approx(TRUE_T2, rel=0.05)
    assert "region integration" in window.curve_fit_source_label.text()


def test_fit_curve_with_nothing_measured_names_both_ways(qapp, _dialogs):
    window = _window_with_series()
    _type_delays(window)
    window._on_fit_curve_button_clicked()
    kind, (title, text) = _dialogs[-1]
    assert kind == "information" and title == "Nothing to fit yet"
    assert "Integrate" in text and "regions" in text and "Fit Spectrum" in text
    assert "Add at least one peak" not in text


def test_peaks_still_take_precedence(qapp, _dialogs):
    window = _window_with_series()
    _type_delays(window)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window._add_region_row(4.5, 5.5)
    window._on_fit_curve_button_clicked()
    results = window.session.spectra[0].metadata.raw["slice_fit_results"]
    assert "peak_defs" in results[0]                    # a peak fit, as before
    assert window._relaxation_fit_results[0]["params"]["T2"] == pytest.approx(TRUE_T2, rel=0.05)
