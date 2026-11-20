"""PFG notes §1, §2 and §7 in the GUI and the fit: the constant-offset control, the AICc component
suggestion, components renumbered by D after a fit, and the visible per-slice baseline option."""
import numpy as np
import pytest
from PySide6.QtWidgets import QDialog, QMessageBox

from opennmr.gui.main_window import MainWindow
from opennmr.processing import diffusion as dif
from opennmr.processing import relaxation as rx

from test_gui import _overlapping_pfg_window, qapp  # noqa: F401 -- fixture


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)


# ---- components are renumbered by D ---------------------------------------------------------

def test_components_come_out_fastest_first_even_from_swapped_guesses():
    b = np.linspace(0, 3e9, 24)
    y = 0.7 * np.exp(-2e-9 * b) + 0.3 * np.exp(-2e-10 * b)
    spec = dif.build_diffusion_model(2, expected_D=[3e-10, 1.5e-9])     # slow guess listed first
    fit = rx.fit_relaxation(b, y, custom_model=spec)
    assert fit["params"]["D1"] > fit["params"]["D2"]
    assert fit["params"]["I1"] == pytest.approx(0.7, rel=0.05)
    assert fit["params"]["D1"] == pytest.approx(2e-9, rel=0.05)


def test_sorting_carries_the_errors_and_leaves_fixed_components_alone():
    b = np.linspace(0, 3e9, 24)
    y = 0.7 * np.exp(-2e-9 * b) + 0.3 * np.exp(-2e-10 * b)
    spec = dif.build_diffusion_model(2, expected_D=[3e-10, 1.5e-9])
    fit = rx.fit_relaxation(b, y, y_err=np.full_like(y, 1e-3), custom_model=spec)
    # the error of D1 belongs to the (now) faster component
    assert fit["errors"]["D1"] / fit["params"]["D1"] < 0.1
    fixed = rx.fit_relaxation(b, y, custom_model=spec, fixed={"D2": 2e-10})
    assert not fixed.get("components_sorted")


def test_stretched_components_also_sort():
    spec = dif.build_diffusion_model(2, stretched=True)
    assert spec["components"] == [("I1", "D1", "beta1"), ("I2", "D2", "beta2")]
    assert dif.build_diffusion_model(1)["components"] is None


# ---- offset control -------------------------------------------------------------------------

def test_offset_control_follows_the_fit_type_and_model(qapp):
    window = MainWindow()
    window.fit_type_combo.setCurrentIndex(1)                         # diffusion
    combo = window.relaxation_model_combo
    combo.setCurrentIndex(combo.findData("stejskal_tanner"))
    assert not window.diffusion_offset_check.isHidden()
    combo.setCurrentIndex(combo.findData("diffusion_n"))
    assert not window.diffusion_offset_check.isHidden()
    window.fit_type_combo.setCurrentIndex(0)                         # relaxation
    assert window.diffusion_offset_check.isHidden()
    assert window.diffusion_select_button.isHidden()


def test_offset_adds_C_to_the_built_model(qapp):
    window = MainWindow()
    window.fit_type_combo.setCurrentIndex(1)
    combo = window.relaxation_model_combo
    combo.setCurrentIndex(combo.findData("stejskal_tanner"))
    assert not window._builds_model_spec("stejskal_tanner")
    window.diffusion_offset_check.setChecked(True)
    assert window._builds_model_spec("stejskal_tanner")
    assert window._current_model_spec()["params"] == ["I0", "D", "C"]
    assert "C" not in window._current_model_spec(include_offset=False)["params"]
    combo.setCurrentIndex(combo.findData("diffusion_n"))
    window.diffusion_components_spin.setValue(2)
    assert window._current_model_spec()["params"][-1] == "C"
    assert "C" in window._current_model_spec()["params"]


def test_score_ignores_the_offset(qapp):
    window = MainWindow()
    window.fit_type_combo.setCurrentIndex(1)
    combo = window.relaxation_model_combo
    combo.setCurrentIndex(combo.findData("stejskal_tanner"))
    window.diffusion_offset_check.setChecked(True)
    assert window._score_model() == "stejskal_tanner"


def test_curve_fit_with_an_offset_fits_C(qapp):
    window, b, _ppm = _overlapping_pfg_window(qapp)
    combo = window.relaxation_model_combo
    combo.setCurrentIndex(combo.findData("stejskal_tanner"))
    window.diffusion_offset_check.setChecked(True)
    spec = window._current_model_spec()
    y = 1.0 * np.exp(-1.8e-9 * b) + 0.01
    fit = rx.fit_relaxation(b, y, custom_model=spec)
    assert fit["params"]["C"] == pytest.approx(0.01, rel=0.2)
    assert fit["params"]["D"] == pytest.approx(1.8e-9, rel=0.05)


# ---- suggestion -----------------------------------------------------------------------------

def test_suggesting_components_reads_the_fitted_curves(qapp):
    window, b, _ppm = _overlapping_pfg_window(qapp)
    rng = np.random.default_rng(1)
    y = np.exp(-1.8e-9 * b) + rng.normal(0, 2e-3, b.size)
    window._relaxation_fit_results = [{"t": b, "y": y, "y_err": np.full_like(b, 2e-3),
                                       "peak_label": "water"}]
    window._on_select_diffusion_components_clicked()
    assert "water: 1 component" in window.diffusion_select_label.text()
    assert "SELECTED" in window.diffusion_select_label.text()
    assert window.diffusion_apply_selection_button.isEnabled()
    window._on_apply_diffusion_selection()
    assert window.relaxation_model_combo.currentData() == "diffusion_n"
    assert window.diffusion_components_spin.value() == 1
    assert not window.diffusion_offset_check.isChecked()


def test_suggesting_without_a_fit_asks_for_one(qapp):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    window._relaxation_fit_results = None
    window._on_select_diffusion_components_clicked()
    assert window.diffusion_select_label.text() == ""
    assert not window.diffusion_apply_selection_button.isEnabled()


# ---- baseline -------------------------------------------------------------------------------

def test_baseline_subtraction_is_visible_and_on_by_default(qapp):
    window = MainWindow()
    assert window.integration_baseline_check.isChecked()
    assert "baseline" in window.integration_baseline_check.toolTip().lower()
