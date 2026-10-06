"""Relaxometry extras in the GUI (user request, 2026-10-06): the
three-exponential models in Fit Curve, Find Regions Automatically, the
integration rule, and Show Parameter Correlations."""

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow, _LEGACY_RELAXATION_CUSTOM_INDEX
from opennmr.io import history

from test_gui import qapp, _load_spectrum_directly, _set_delay_unit  # noqa: F401

DELAYS = np.geomspace(2e-4, 10, 30)
TRUE_T = (0.002, 0.02, 0.2)


@pytest.fixture
def said(monkeypatch):
    out = []
    for name in ("critical", "information", "warning"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: out.append((_n, a[2] if len(a) > 2 else "")))
    return out


def _series(seed=0, noise=0.002):
    """Two peaks; the one at 5 ppm decays with three components, the one
    at 6 ppm with one."""
    ppm = np.linspace(7.0, 3.0, 1200)
    rng = np.random.default_rng(seed)
    rows = []
    for t in DELAYS:
        a = 0.5 * np.exp(-t / 0.002) + 0.3 * np.exp(-t / 0.02) + 0.2 * np.exp(-t / 0.2)
        b = np.exp(-t / 0.05)
        rows.append(a * np.exp(-0.5 * ((ppm - 5) / 0.02) ** 2)
                    + 0.5 * b * np.exp(-0.5 * ((ppm - 6) / 0.02) ** 2)
                    + rng.normal(0, noise, ppm.size))
    return NMRSpectrum(data=np.array(rows),
                       axes=[NMRAxis(kind="index", index=np.arange(1, len(DELAYS) + 1)),
                             NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
                       metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0,
                                            pseudo_2d=True))


def _window():
    window = MainWindow()
    _load_spectrum_directly(window, _series())
    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, "s")
    window.manual_delays_edit.setText(", ".join(f"{d:.8g}" for d in DELAYS))
    return window


def test_find_regions_automatically_fills_the_table_and_is_logged(qapp, said):
    window = _window()
    window._add_region_row(0.0, 1.0)                       # replaced
    found = window._on_auto_regions_clicked()
    regions = window._read_region_defs()
    assert len(regions) == 2
    assert np.allclose(regions, sorted(found["regions"]), atol=1e-4)
    assert any(lo < 5 < hi for lo, hi in regions) and any(lo < 6 < hi for lo, hi in regions)
    assert "Regions found automatically" in window.integration_results_label.text()
    window._on_integrate_regions_clicked()
    text = window.integration_results_label.text()
    assert "trapezoid rule" in text and "found automatically" in text
    # A hand edit means the note no longer describes the regions.
    window.region_table.item(0, 0).setText("4.9")
    assert window._auto_regions_note == ""
    assert said == []


def test_threshold_modes_and_nothing_found(qapp, said):
    window = _window()
    window.auto_region_mode_combo.setCurrentIndex(
        window.auto_region_mode_combo.findData("fraction"))
    assert window.auto_region_threshold_spin.suffix() == " %"
    window.auto_region_threshold_spin.setValue(60.0)   # only the larger (5 ppm) peak
    window._on_auto_regions_clicked()
    assert len(window._read_region_defs()) == 1
    window.auto_region_mode_combo.setCurrentIndex(
        window.auto_region_mode_combo.findData("noise"))
    window.auto_region_threshold_spin.setValue(1000.0)
    window._on_auto_regions_clicked()
    assert said and "Lower the threshold" in said[-1][1]


def test_simpson_is_passed_to_the_integration(qapp, said, monkeypatch):
    window = _window()
    window._on_auto_regions_clicked()
    from opennmr.processing import relaxation as pr
    seen = {}
    real = pr.NMR2D_integrate_slices
    monkeypatch.setattr(pr, "NMR2D_integrate_slices",
                        lambda *a, **k: (seen.update(k), real(*a, **k))[1])
    window.integration_method_combo.setCurrentIndex(
        window.integration_method_combo.findData("simpson"))
    window._on_integrate_regions_clicked()
    assert seen["method"] == "simpson"
    assert "simpson rule" in window.integration_results_label.text()


def test_rule_and_threshold_are_saved_with_the_spectrum(qapp, said):
    window = _window()
    window.integration_method_combo.setCurrentIndex(
        window.integration_method_combo.findData("simpson"))
    window.auto_region_threshold_spin.setValue(8.0)
    window._on_auto_regions_clicked()
    state = window._capture_analysis_state()
    assert state["integration_method"] == "simpson" and state["auto_region_threshold"] == 8.0
    assert state["auto_regions_note"]
    # Another spectrum starts from the defaults; coming back restores.
    _load_spectrum_directly(window, _series(seed=1))
    assert window.integration_method_combo.currentData() == "trapezoid"
    assert window.auto_region_threshold_spin.value() == 5.0
    window._select_single_row(0)
    assert window.integration_method_combo.currentData() == "simpson"
    assert window.auto_region_threshold_spin.value() == 8.0
    assert window._auto_regions_note


def test_triexponential_fit_curve_advice_and_correlations(qapp, said):
    window = _window()
    window._on_auto_regions_clicked()
    window._on_integrate_regions_clicked()
    combo = window.relaxation_model_combo
    combo.setCurrentIndex(combo.findData("triexponential_decay"))
    window._on_fit_curve_button_clicked()
    assert said == [], said
    results = window._relaxation_fit_results
    assert len(results) == 2
    first = results[0]                       # regions run low to high: 5 ppm first
    for name, truth in zip(("T2a", "T2b", "T2c"), TRUE_T):
        assert first["params"][name] == pytest.approx(truth, rel=0.15)
    advice = " ".join(i["text"] for i in window.curve_fit_advice_box.items) \
        if hasattr(window.curve_fit_advice_box, "items") else ""
    # The single-component peak fitted with three: the advice says so.
    assert "not supported" in advice or "within a factor" in advice
    button = window.curve_fit_correlation_button
    assert button.isEnabled()
    button.setChecked(True)
    fig = window.fit_canvas.figure
    names, R = fig.correlation
    assert names == results[window.view_relaxation_peak_spin.value() - 1]["free_names"]
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    # The peak spin browses matrices while it is on; off goes back to the curve.
    window.view_relaxation_peak_spin.setValue(2)
    assert window.fit_canvas.figure.correlation[0] == results[1]["free_names"]
    button.setChecked(False)
    assert window.fit_canvas._external_plot is False        # the curve is back
    # Minor ticks have no meaning on a matrix: the style pass leaves them off.
    button.setChecked(True)
    ax = window.fit_canvas.figure.correlation_axes["matrix"]
    assert not ax.xaxis.get_minor_ticks() or ax.xaxis.get_minor_locator().__class__.__name__ \
        == "NullLocator"


def test_model_is_saved_by_key_and_old_states_still_restore(qapp, said):
    window = _window()
    combo = window.relaxation_model_combo
    combo.setCurrentIndex(combo.findData("triexponential_recovery"))
    state = window._capture_analysis_state()
    assert state["model_key"] == "triexponential_recovery"
    assert window._saved_model_index(state) == combo.findData("triexponential_recovery")
    # A state saved before the models were appended: index 7 was "Custom equation…".
    old = {"model": _LEGACY_RELAXATION_CUSTOM_INDEX}
    assert combo.itemData(window._saved_model_index(old)) == "custom"
    assert combo.itemData(window._saved_model_index({"model": 2})) == "exponential_decay"


def test_history_names_the_new_settings():
    old = {"analysis_state": {"regions": [["4.9", "5.1"]], "integration_method": "trapezoid",
                              "auto_regions_note": "1 region"}}
    new = {"analysis_state": {"regions": [["4.9", "5.1"]], "integration_method": "simpson",
                              "auto_regions_note": "1 region", "auto_region_threshold": 8.0,
                              "model_key": "triexponential_decay"}}
    lines = " ".join(history.change_lines(history.diff_states(old, new), old, new))
    assert "Integration rule" in lines and "Simpson" in lines
    assert "model_key" not in lines
