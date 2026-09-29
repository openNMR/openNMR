"""Process ▸ Baseline ▸ Manual (A x^4 + ... + E, A sin(Bx + C) + D):
fit to regions, hand-edit, apply, every slice, restore, log -- explicit
user request, 2026-09-29."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.gui.manual_baseline_box import format_regions, parse_regions
from opennmr.io import history
from opennmr.notebook.manual_baseline import evaluate_manual_baseline
from opennmr.processing import report

from test_gui import _load_spectrum_directly, _settle_live_reprocess, qapp  # noqa: F401

PPM = np.linspace(-20.0, 20.0, 2001)
PEAK = 50.0 * np.exp(-((PPM - 2.0) / 0.4) ** 2)
REGIONS = [(-20.0, -5.0), (8.0, 20.0)]
POLY = [1e-4, -2e-3, 0.02, 0.3, 4.0]
SINE = [3.0, 0.25, 0.6, 1.0]


def _spectrum_1d(baseline):
    data = PEAK + baseline
    axes = [NMRAxis(kind="frequency", ppm=PPM, hz=PPM * 400.0)]
    meta = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    return NMRSpectrum(data=data.copy(), axes=axes, metadata=meta, data_imag=np.zeros_like(data))


def _window_with(baseline, method="manual_polynomial"):
    window = MainWindow()
    _load_spectrum_directly(window, _spectrum_1d(baseline))
    window.baseline_correct_button.setChecked(False)
    window.baseline_method_combo.setCurrentIndex(window.baseline_method_combo.findData(method))
    return window


def _set_regions(window, regions):
    box = window.baseline_manual_box
    box.regions_edit.setText(format_regions(regions))
    box.regions_edit.editingFinished.emit()


# ---- the regions line ---------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("-50 – -20; 180 – 220", [(-50.0, -20.0), (180.0, 220.0)]),
    ("10-20", [(10.0, 20.0)]),
    ("220 to 180\n−5, −1", [(180.0, 220.0), (-5.0, -1.0)]),
    ("1e1 - 2e1", [(10.0, 20.0)]),
    ("", []),
])
def test_regions_are_parsed(text, expected):
    assert parse_regions(text) == expected


def test_a_bad_region_is_reported_not_raised(qapp):
    window = _window_with(evaluate_manual_baseline(PPM, "polynomial", POLY))
    box = window.baseline_manual_box
    box.regions_edit.setText("5 – 10 – 15")
    box.regions_edit.editingFinished.emit()
    assert "could not read" in box.status_label.text()
    assert "baseline_manual_regions" not in window.session.spectra[0].metadata.raw


# ---- controls -----------------------------------------------------------------

def test_manual_controls_show_only_for_manual_methods(qapp):
    window = _window_with(0.0 * PPM, method="linear")
    assert window.baseline_manual_box.isHidden()
    window.baseline_method_combo.setCurrentIndex(
        window.baseline_method_combo.findData("manual_sine"))
    box = window.baseline_manual_box
    assert not box.isHidden()
    assert "sin" in box.equation_label.text()
    assert box.param_edits["E"].isHidden() and not box.param_edits["D"].isHidden()
    window.baseline_method_combo.setCurrentIndex(
        window.baseline_method_combo.findData("manual_polynomial"))
    assert not box.param_edits["E"].isHidden()
    assert "x⁴" in box.equation_label.text()


# ---- fit, apply, hand-edit ------------------------------------------------------

def test_fit_to_regions_then_apply_polynomial(qapp):
    truth = evaluate_manual_baseline(PPM, "polynomial", POLY)
    window = _window_with(truth)
    _set_regions(window, REGIONS)
    window.baseline_manual_box.fit_button.click()
    raw = window.session.spectra[0].metadata.raw
    np.testing.assert_allclose(raw["baseline_manual_polynomial_params"], POLY, rtol=1e-6, atol=1e-12)
    assert "Fitted to" in window.baseline_manual_box.status_label.text()
    assert float(window.baseline_manual_box.param_edits["E"].text()) == pytest.approx(4.0, rel=1e-5)

    window.baseline_correct_button.setChecked(True)
    np.testing.assert_allclose(window.session.spectra[0].data, PEAK, atol=1e-5)
    window.baseline_correct_button.setChecked(False)
    np.testing.assert_allclose(window.session.spectra[0].data, PEAK + truth)


def test_fit_sine_while_the_correction_is_on(qapp):
    # The fit must use the trace *before* correction, not the flattened one.
    truth = evaluate_manual_baseline(PPM, "sine", SINE)
    window = _window_with(truth, method="manual_sine")
    _set_regions(window, REGIONS)
    window.baseline_correct_button.setChecked(True)      # first a fit with no params yet
    window.baseline_manual_box.fit_button.click()
    params = window.session.spectra[0].metadata.raw["baseline_manual_sine_params"]
    np.testing.assert_allclose(params, SINE, rtol=1e-5)
    np.testing.assert_allclose(window.session.spectra[0].data, PEAK, atol=1e-5)


def test_hand_edited_coefficient_is_what_is_subtracted(qapp):
    truth = evaluate_manual_baseline(PPM, "polynomial", POLY)
    window = _window_with(truth)
    _set_regions(window, REGIONS)
    box = window.baseline_manual_box
    box.refit_check.setChecked(True)
    box.fit_button.click()
    window.baseline_correct_button.setChecked(True)
    # Raise the offset E by 2 by hand: the corrected spectrum drops by 2,
    # and the automatic refit is switched off so it stays that way.
    edit = box.param_edits["E"]
    edit.setText(f"{float(edit.text()) + 2.0:.10g}")
    edit.textEdited.emit(edit.text())
    edit.editingFinished.emit()
    raw = window.session.spectra[0].metadata.raw
    assert raw["baseline_manual_refit"] is False and not box.refit_check.isChecked()
    np.testing.assert_allclose(window.session.spectra[0].data, PEAK - 2.0, atol=1e-4)
    assert box.status_label.text() == "Coefficients set by hand."


def test_focus_change_alone_is_not_an_edit(qapp):
    window = _window_with(evaluate_manual_baseline(PPM, "polynomial", POLY))
    box = window.baseline_manual_box
    box.refit_check.setChecked(True)
    box.param_edits["A"].editingFinished.emit()           # no textEdited first
    assert box.refit_check.isChecked()


def test_fixed_coefficients_are_held_in_the_gui_fit(qapp):
    quadratic = [0.0, 0.0, 0.02, 0.3, 4.0]
    window = _window_with(evaluate_manual_baseline(PPM, "polynomial", quadratic))
    _set_regions(window, REGIONS)
    box = window.baseline_manual_box
    box.fix_checks["A"].setChecked(True)
    box.fix_checks["B"].setChecked(True)
    box.fit_button.click()
    raw = window.session.spectra[0].metadata.raw
    assert raw["baseline_manual_polynomial_fixed"] == [True, True, False, False, False]
    params = raw["baseline_manual_polynomial_params"]
    assert params[0] == 0.0 and params[1] == 0.0
    np.testing.assert_allclose(params[2:], quadratic[2:], rtol=1e-8)
    assert box.error_labels["A"].text() == ""             # fixed: no error bar


# ---- picking on the plot, shading, overlay --------------------------------------

def test_pick_on_plot_adds_regions_and_shades_them(qapp):
    window = _window_with(evaluate_manual_baseline(PPM, "polynomial", POLY))
    box = window.baseline_manual_box
    box.pick_button.setChecked(True)
    preview = window.processing_preview_canvas
    assert preview._region_select_mode
    preview.regionSelected.emit(-18.0, -6.0)
    preview.regionSelected.emit(9.0, 19.0)
    assert window.session.spectra[0].metadata.raw["baseline_manual_regions"] == [
        [-18.0, -6.0], [9.0, 19.0]]
    assert parse_regions(box.regions_edit.text()) == [(-18.0, -6.0), (9.0, 19.0)]
    window.canvas_tabs.setCurrentWidget(preview)
    window._refresh_baseline_preview()
    shaded = [p for ax in preview.figure.get_axes() for p in ax.patches
              if p.get_gid() == preview._BASELINE_REGION_GID]
    assert len(shaded) == 2
    # Another mode taking over the canvas releases the button.
    preview.set_measurement_mode(True)
    assert not box.pick_button.isChecked()
    box.clear_button.click()
    assert window.session.spectra[0].metadata.raw["baseline_manual_regions"] == []


def test_drags_are_ignored_unless_picking(qapp):
    window = _window_with(evaluate_manual_baseline(PPM, "polynomial", POLY))
    window.processing_preview_canvas.regionSelected.emit(1.0, 2.0)
    assert not window.session.spectra[0].metadata.raw.get("baseline_manual_regions")


def test_overlay_is_the_manual_baseline(qapp):
    truth = evaluate_manual_baseline(PPM, "polynomial", POLY)
    window = _window_with(truth)
    _set_regions(window, REGIONS)
    window.baseline_manual_box.fit_button.click()
    window.canvas_tabs.setCurrentWidget(window.processing_preview_canvas)
    window._refresh_baseline_preview()
    preview = window.processing_preview_canvas
    lines = [ln for ln in preview._pivot_marker_ax.lines
             if ln.get_gid() == preview._BASELINE_PREVIEW_GID]
    assert len(lines) == 1
    x, y = preview.baseline_preview_source()
    scale = float(np.max(np.abs(y)))
    np.testing.assert_allclose(lines[0].get_ydata(),
                               evaluate_manual_baseline(x, "polynomial", POLY) / scale, atol=1e-6)


# ---- pseudo-2D --------------------------------------------------------------------

def _pseudo2d(rows=4):
    ppm = PPM
    baselines = [evaluate_manual_baseline(ppm, "sine", [1.0 + r, 0.25, 0.6, 0.5 * r])
                 for r in range(rows)]
    data = np.array([PEAK * (1.0 - 0.2 * r) + b for r, b in enumerate(baselines)])
    axes = [NMRAxis(kind="index", index=np.arange(1, rows + 1)),
            NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)]
    meta = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True)
    return NMRSpectrum(data=data, axes=axes, metadata=meta, data_imag=np.zeros_like(data)), baselines


def test_every_slice_refits_its_own_sine(qapp):
    spectrum, baselines = _pseudo2d()
    peaks = spectrum.data - np.array(baselines)
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.baseline_method_combo.setCurrentIndex(window.baseline_method_combo.findData("manual_sine"))
    _set_regions(window, REGIONS)
    window.baseline_manual_box.refit_check.setChecked(True)
    assert window.apply_all_slices_checkbox.isChecked()
    window.baseline_correct_button.setChecked(True)
    _settle_live_reprocess(window)
    np.testing.assert_allclose(window.session.spectra[0].data, peaks, atol=1e-4)


def test_every_slice_with_fixed_coefficients_subtracts_the_same_curve(qapp):
    spectrum, _baselines = _pseudo2d()
    original = spectrum.data.copy()
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.baseline_method_combo.setCurrentIndex(window.baseline_method_combo.findData("manual_sine"))
    box = window.baseline_manual_box
    box.set_params([1.0, 0.25, 0.6, 0.0])
    edit = box.param_edits["A"]
    edit.textEdited.emit(edit.text())
    edit.editingFinished.emit()
    window.baseline_correct_button.setChecked(True)
    _settle_live_reprocess(window)
    curve = evaluate_manual_baseline(PPM, "sine", [1.0, 0.25, 0.6, 0.0])
    np.testing.assert_allclose(window.session.spectra[0].data, original - curve, atol=1e-9)


# ---- per-spectrum state, warnings, log --------------------------------------------

def test_state_is_restored_per_spectrum(qapp):
    window = _window_with(evaluate_manual_baseline(PPM, "sine", SINE), method="manual_sine")
    _set_regions(window, REGIONS)
    window.baseline_manual_box.fit_button.click()
    _load_spectrum_directly(window, _spectrum_1d(0.0 * PPM))     # second spectrum, selected
    assert window.baseline_method_combo.currentData() == "linear"
    assert window.baseline_manual_box.isHidden()
    window._select_single_row(0)
    assert window.baseline_method_combo.currentData() == "manual_sine"
    box = window.baseline_manual_box
    assert parse_regions(box.regions_edit.text()) == REGIONS
    np.testing.assert_allclose(box.params(), SINE, rtol=1e-4)


def test_switching_model_keeps_each_models_numbers(qapp):
    window = _window_with(evaluate_manual_baseline(PPM, "polynomial", POLY))
    _set_regions(window, REGIONS)
    window.baseline_manual_box.fit_button.click()
    window.baseline_method_combo.setCurrentIndex(window.baseline_method_combo.findData("manual_sine"))
    assert window.baseline_manual_box.params() == [0.0, 0.0, 0.0, 0.0]
    window.baseline_method_combo.setCurrentIndex(
        window.baseline_method_combo.findData("manual_polynomial"))
    np.testing.assert_allclose(window.baseline_manual_box.params(), POLY, rtol=1e-5)


def test_warns_when_a_region_holds_signal(qapp):
    window = _window_with(evaluate_manual_baseline(PPM, "polynomial", POLY))
    _set_regions(window, [(-20.0, -5.0), (0.0, 4.0)])        # the second holds the peak
    assert "contains signal" in window.processing_status_label.text()
    _set_regions(window, [])
    assert "no baseline regions" in window.processing_status_label.text()


def test_methods_report_and_log_state_the_manual_model(qapp):
    window = _window_with(evaluate_manual_baseline(PPM, "sine", SINE), method="manual_sine")
    _set_regions(window, REGIONS)
    window.baseline_manual_box.fit_button.click()
    window.baseline_correct_button.setChecked(True)
    spectrum = window.session.spectra[0]
    phrase = report._baseline_phrase(spectrum.metadata.raw)
    assert "A·sin(B·x + C) + D (x in ppm)" in phrase and "B = 0.25" in phrase
    assert "-20 to -5 ppm" in phrase
    lines = history.state_summary(dict(spectrum.metadata.raw))
    joined = "\n".join(lines)
    assert "manual A·sin(Bx+C)+D" in joined
    assert "Baseline regions" in joined and "Baseline A, B, C, D" in joined
    assert "Baseline A, B, C, D, E" not in joined       # the other model's numbers stay out


# ---- Herzfeld-Berger skew: stated where it is entered ------------------------------

def test_skew_definition_is_stated_in_the_csa_controls(qapp):
    window = MainWindow()
    for widget in (window.sim_csa_skeweta_spin, window.sim_csa_convention_combo,
                   window.csa_convention_combo):
        assert "κ = 3(δ22 − δiso)/Ω" in widget.toolTip()
