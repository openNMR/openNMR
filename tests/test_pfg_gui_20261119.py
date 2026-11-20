"""PFG notes §4 and §5 in the GUI: SCORE checked against the integral fit (flags, optional row
alignment, automatic run beside a diffusion fit) and the shared-shape fit of the peak table."""
import numpy as np
import pytest
from PySide6.QtWidgets import QDialog, QMessageBox

from opennmr.gui.main_window import MainWindow, _ScoreSettingsDialog
from opennmr.notebook import pfg_lineshapes as L
from opennmr.notebook import pfg_score_check as S

from test_gui import _overlapping_pfg_window, qapp  # noqa: F401 -- fixture


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)


def _fit(D):
    return {"param_names": ["I0", "D"], "params": {"I0": 1.0, "D": D}}


def _select(window):
    return window.session.spectra[window._selected_index()]


def _stage_fit(window, D):
    spectrum = _select(window)
    window._relaxation_fit_results = [_fit(D)]
    window._relaxation_fit_spectrum = spectrum
    return spectrum


# ---- notebook helpers -----------------------------------------------------------------------

def test_integral_fit_for_check_picks_the_fastest_D():
    pick = S.integral_fit_for_check([_fit(1e-10), _fit(2e-9), {"junk": 1}])
    assert pick["params"]["D"] == 2e-9
    assert S.integral_fit_for_check(None) is None and S.integral_fit_for_check([]) is None


def test_format_check_reads_well():
    ok = {"check": {"ok": True, "flags": [], "water_difference": 0.012, "D_integral": [1.0]}}
    assert "✓" in S.format_check(ok) and "1.2 %" in S.format_check(ok)
    none = {"check": {"ok": True, "flags": [], "water_difference": float("nan"), "D_integral": []}}
    assert "Fit Curve" in S.format_check(none)
    bad = {"check": {"ok": False, "flags": ["water-like D differs"], "water_difference": 0.3,
                     "D_integral": [1.0]}, "alignment_shifts_ppm": [0.0, 0.02]}
    text = S.format_check(bad)
    assert "⚠ water-like D differs" in text and "aligned" in text
    assert S.format_check({}) == ""


def test_shared_shape_results_have_the_layout_fit_curve_reads():
    ppm = np.linspace(7, 3, 400)
    b = np.linspace(0, 4e9, 8)
    rows = np.array([np.exp(-2e-9 * x) * L.pseudo_voigt(ppm, 4.7, 0.2, 0.5) for x in b])
    fit = L.fit_series_shared_shapes(ppm, rows, [{"position": 4.7, "fwhm": 0.2}])
    out = L.shared_shapes_slice_results(fit, range(1, 9))
    assert len(out) == 8 and out[0]["model_fit"] and out[0]["site_labels"] == ["Peak 1"]
    assert out[0]["integrals"][0] > out[-1]["integrals"][0]
    assert out[3]["index"] == 4
    with pytest.raises(ValueError):
        L.shared_shapes_slice_results(fit, [1, 2])


# ---- SCORE beside the integral fit ---------------------------------------------------------

def test_score_agreeing_with_the_integral_fit_is_confirmed(qapp):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    _stage_fit(window, 1.8e-9)
    window._on_score_clicked()
    assert window._score_result["check"]["ok"]
    assert "✓" in window.score_check_label.text()


def test_score_disagreeing_with_the_integral_fit_is_flagged(qapp):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    _stage_fit(window, 1.2e-9)
    window._on_score_clicked()
    assert not window._score_result["check"]["ok"]
    assert "water-like D" in window.score_check_label.text()


def test_without_a_curve_fit_the_label_says_to_run_one(qapp):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    window._on_score_clicked()
    assert "Fit Curve" in window.score_check_label.text()


def test_row_alignment_is_optional_and_logged(qapp):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    assert not window.score_align_check.isChecked()
    window._on_score_clicked()
    assert "alignment_shifts_ppm" not in window._score_result
    window.score_align_check.setChecked(True)
    window._on_score_clicked()
    assert len(window._score_result["alignment_shifts_ppm"]) == 20
    assert "aligned" in window.score_check_label.text()


def test_score_runs_beside_a_diffusion_fit_for_two_and_three_components(qapp):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    assert window.score_auto_check.isChecked()
    spectrum = _stage_fit(window, 1.8e-9)
    window._auto_score_beside_fit(spectrum, window._relaxation_fit_results)
    result = window._score_result
    assert result is not None
    assert [c["k"] for c in result["beside_fit"]] == [2, 3]
    assert result["n_components"] == 2
    assert window.score_check_label.text() != ""


def test_the_automatic_run_can_be_switched_off(qapp):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    spectrum = _stage_fit(window, 1.8e-9)
    window.score_auto_check.setChecked(False)
    window._auto_score_beside_fit(spectrum, window._relaxation_fit_results)
    assert window._score_result is None


# ---- shared-shape fit of the peak table ----------------------------------------------------

def test_shared_shape_fit_of_the_peak_table(qapp):
    window, b, _ppm = _overlapping_pfg_window(qapp)
    window._add_peak_row(2.0, 0.1, 0.5)          # fast species only
    window._add_peak_row(6.5, 0.1, 0.5)          # slow species only
    window._on_shared_shape_fit_clicked()
    results = window._slice_fit_results
    assert len(results) == 20 and results[0]["model"] == "shared_shapes"
    fast = [r["integrals"][0] for r in results]
    slow = [r["integrals"][1] for r in results]
    assert fast[-1] < 0.2 * fast[0]                       # decays quickly
    assert slow[-1] == pytest.approx(slow[0] * np.exp(-2.5e-10 * b[-1]), rel=0.15)   # slowly
    assert min(fast + slow) >= 0
    assert "Shared-shape fit" in window.shared_shape_status_label.text()
    assert window.value_source_combo.currentData() == "area"


def test_shared_shape_fit_needs_peaks(qapp):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    window._on_shared_shape_fit_clicked()
    assert window._slice_fit_results is None
    assert window.shared_shape_status_label.text() == ""


def test_the_controls_exist(qapp):
    window = MainWindow()
    for name in ("score_auto_check", "score_align_check", "score_check_label",
                 "shared_shape_button", "shared_shape_anchor_check", "shared_shape_status_label"):
        assert hasattr(window, name)
