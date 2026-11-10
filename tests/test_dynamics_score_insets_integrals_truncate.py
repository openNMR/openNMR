"""2026-11-10 round: Dynamics slices section and right-click Remove Point /
Remove Slice, the ln(I/I0) curve plot, SCORE with a multi-component model,
its residuals and report, SCORE state per experiment, a global inset list on
Compare, integral calibration on the plot, and Truncate to in TopSpin's TDeff."""
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QDialog, QInputDialog, QLabel, QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui import report_format
from opennmr.gui.inset_dialog import InsetDialog
from opennmr.gui.main_window import MainWindow, _ScoreSettingsDialog
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.gui.widgets.collapsible_section import CollapsibleSection
from opennmr.notebook import peak_integral_labels as pil
from opennmr.plotting import inset as inset_module
from opennmr.processing import ft as processing_ft
from opennmr.processing import relaxation as rx
from opennmr.processing import score as sc

from test_gui import (  # noqa: F401 -- qapp is a fixture
    _fit_a_t2_curve, _load_spectrum_directly, _make_pseudo2d_relaxation_spectrum,
    _overlapping_pfg_window, qapp,
)
from test_integrate_tile_plot_options_regularisation import _one_d


# ---- 1. Slices used in fits: collapsible; right-click Remove Point / Slice ------------

def _t2_window():
    window = MainWindow()
    _fit_a_t2_curve(window, np.geomspace(0.001, 0.3, 10), true_t2=0.05)
    return window


def test_slices_used_in_fits_is_a_collapsible_section_naming_the_excluded(qapp):
    window = _t2_window()
    group = window.excluded_slice_group
    assert isinstance(group, CollapsibleSection)
    assert not group.isExpanded()                     # folded, like its neighbours
    spectrum = window.session.spectra[window._selected_index()]
    window.set_excluded_slices(spectrum, {1, 4})
    assert group.title() == "Slices used in fits (2 excluded)"
    window.set_excluded_slices(spectrum, set())
    assert group.title() == "Slices used in fits"


def test_right_click_offers_remove_point_and_remove_slice(qapp, monkeypatch):
    window = _t2_window()
    canvas = window.fit_canvas
    assert canvas.series_slices_removable          # a pseudo-2D's own fit
    texts = []

    def capture(menu, _pos):
        texts.extend(a.text() for a in menu.actions())
        return None

    monkeypatch.setattr(canvas, "_exec_context_menu", capture)
    canvas._last_right_click_xy = (0.01, 0.5)
    canvas._on_context_menu(QPoint(5, 5))
    assert "Remove Point" in texts and "Remove Slice from Every Fit" in texts
    assert "Restore Removed Points" in texts

    # All peaks together: the points can be removed there too.
    window.curve_fit_plot_box.view_combo.setCurrentIndex(
        window.curve_fit_plot_box.view_combo.findData("all"))
    assert canvas._curve_fits_shown and not canvas._fit_mode
    texts.clear()
    canvas._last_right_click_xy = (0.01, 0.5)
    canvas._on_context_menu(QPoint(5, 5))
    assert "Remove Point" in texts


def test_remove_point_works_with_all_peaks_together(qapp):
    window = _t2_window()
    window.curve_fit_plot_box.view_combo.setCurrentIndex(
        window.curve_fit_plot_box.view_combo.findData("all"))
    original = window._relaxation_fit_original[0]
    n = len(original["t"])
    window._on_fit_point_exclude_requested(float(original["t"][3]), float(original["y"][3]))
    assert window._excluded_relaxation_indices == {0: {3}}
    assert len(window._relaxation_fit_results[0]["t"]) == n - 1
    # The removed point stays on the plot as a cross.
    assert any(line.get_marker() == "x" for line in window.fit_canvas.ax.lines)
    window._on_fit_points_reset_requested()
    assert not window._excluded_relaxation_indices.get(0)
    assert len(window._relaxation_fit_results[0]["t"]) == n


def test_remove_slice_unticks_it_and_refits_every_peak(qapp):
    window = _t2_window()
    spectrum = window.session.spectra[window._selected_index()]
    original = window._relaxation_fit_original[0]
    n = len(original["t"])
    window._on_fit_slice_remove_requested(float(original["t"][5]), float(original["y"][5]))
    assert MainWindow.excluded_slices(spectrum) == {5}
    item = window.excluded_slice_list.item(5)
    assert item.checkState() == Qt.CheckState.Unchecked
    assert len(window._relaxation_fit_results[0]["t"]) == n - 1


# ---- 2. ln(I/I0) / I/I0 curve plots -----------------------------------------------

def test_attenuation_is_the_stejskal_tanner_line():
    b = np.linspace(0.0, 3.0e9, 12)
    fit = rx.fit_relaxation(b, 5.0 * np.exp(-1.0e-9 * b), model="stejskal_tanner")
    drawn = rx.attenuation(fit, mode="ln")
    assert drawn["i0"] == pytest.approx(5.0, rel=1e-6)
    assert drawn["i0_source"] == "fitted value at x = 0"
    slope = np.polyfit(drawn["x"], drawn["y"], 1)[0]
    assert slope == pytest.approx(-1.0e-9, rel=1e-6)      # slope -D
    assert drawn["curve_y"][0] == pytest.approx(0.0, abs=1e-9)
    relative = rx.attenuation(fit, mode="relative")
    assert relative["y"][0] == pytest.approx(1.0, rel=1e-6)


def test_ln_of_a_non_positive_point_is_a_gap_not_a_number():
    values = rx.attenuation_values([2.0, 1.0, 0.0, -0.5], 2.0, "ln")
    assert values[0] == pytest.approx(0.0) and values[1] == pytest.approx(np.log(0.5))
    assert np.isnan(values[2]) and np.isnan(values[3])


def test_inversion_recovery_has_no_i0_to_normalise_by():
    t = np.geomspace(0.01, 5.0, 10)
    fit = rx.fit_relaxation(t, 1.0 - 2.0 * np.exp(-t / 0.5), model="inversion_recovery")
    with pytest.raises(ValueError, match="No positive I0"):
        rx.attenuation(fit, mode="ln")


def test_curve_fit_plot_draws_ln_and_back(qapp):
    window = _t2_window()
    box = window.curve_fit_plot_box
    box.y_mode_combo.setCurrentIndex(box.y_mode_combo.findData("ln"))
    assert window.fit_canvas.ax.get_ylabel() == r"$\ln(I/I_0)$"
    assert box.settings()["y_mode"] == "ln"
    points = next(line for line in window.fit_canvas.ax.lines if line.get_marker() == "o")
    assert np.nanmax(points.get_ydata()) < 0.1          # ln(I/I0) <= ~0
    # The stored fit is untouched: only its drawing changed.
    stored = window._relaxation_fit_results[0]
    i0 = window._curve_y_transform(stored)[1]
    np.testing.assert_allclose(points.get_ydata(), np.log(stored["y"] / i0))
    assert np.all(stored["y"] > 0)                     # still intensities, not logs
    box.y_mode_combo.setCurrentIndex(box.y_mode_combo.findData("intensity"))
    assert window.fit_canvas.ax.get_ylabel() == "Intensity"


def test_remove_point_finds_the_point_on_the_ln_axis(qapp):
    window = _t2_window()
    box = window.curve_fit_plot_box
    box.y_mode_combo.setCurrentIndex(box.y_mode_combo.findData("ln"))
    result = window._relaxation_fit_results[0]
    i0 = window._curve_y_transform(result)[1]
    original = window._relaxation_fit_original[0]
    window._on_fit_point_exclude_requested(float(original["t"][2]),
                                           float(np.log(original["y"][2] / i0)))
    assert window._excluded_relaxation_indices == {0: {2}}


# ---- 3. SCORE ----------------------------------------------------------------------

@pytest.mark.parametrize("requested, kind, expected", [
    ("stejskal_tanner_biexponential", "diffusion", "stejskal_tanner"),
    ("biexponential_decay", "relaxation", "exponential_decay"),
    ("triexponential_decay", "relaxation", "exponential_decay"),
    ("biexponential_recovery", "relaxation", "inversion_recovery"),
    ("stejskal_tanner", "diffusion", "stejskal_tanner"),
])
def test_score_kernel_model_falls_back_to_the_single_component_form(requested, kind, expected):
    chosen = sc.kernel_model(requested, kind=kind)
    assert chosen["model"] == expected
    assert chosen["substituted"] is (requested != expected)
    assert bool(chosen["note"]) is (requested != expected)


def test_score_kernel_model_keeps_a_stretched_shape():
    from opennmr.processing.diffusion import build_diffusion_model
    assert sc.kernel_model(build_diffusion_model(2), "diffusion")["model"] == "stejskal_tanner"
    assert sc.kernel_model(build_diffusion_model(2, stretched=True),
                           "diffusion")["model"] == "stejskal_tanner_stretched"


def test_score_kernel_model_still_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        sc.kernel_model("not_a_model", "diffusion")


def test_score_result_carries_residual_metrics():
    b = np.linspace(0.0, 3.0e9, 12)
    ppm = np.linspace(0.0, 10.0, 400)
    fast = np.exp(-((ppm - 3.0) / 0.05) ** 2)
    slow = np.exp(-((ppm - 3.1) / 0.05) ** 2)
    data = np.outer(np.exp(-b * 1.8e-9), fast) + np.outer(np.exp(-b * 2.5e-10), slow)
    spoiled = data.copy()
    spoiled[7] *= 1.3                                  # one bad gradient level
    result = sc.score(b, spoiled, n_components=2)
    assert result["residual_first"].shape == (ppm.size,)
    assert result["residual_by_level"].shape == (b.size,)
    assert result["rmse"] > 0 and 0 < result["relative_rmse"] < 1
    assert result["worst_level"] == 7                  # the spoiled level stands out
    report = sc.format_score_report({**result, "rows": list(range(b.size))})
    assert "RMSE" in report and "worst slice   = 8" in report


def test_residual_metrics_numbers():
    data = np.array([[2.0, 2.0], [1.0, 1.0]])
    fit = np.array([[2.0, 2.0], [1.0, 0.0]])
    q = sc.residual_metrics(data, fit)
    assert q["worst_level"] == 1
    assert q["residual_by_level"][0] == pytest.approx(0.0)
    assert q["residual_by_level"][1] == pytest.approx(np.sqrt(0.5) / 2.0)


def test_score_with_a_multi_component_model_runs_and_says_so(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("stejskal_tanner_biexponential"))
    monkeypatch.setattr(QMessageBox, "critical",
                        lambda *a, **k: pytest.fail(f"no error expected: {a[2]}"))
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()
    result = window._score_result
    assert result["model"] == "stejskal_tanner"
    assert "single-component form" in result["model_note"]
    assert "single-component form" in window.score_results_label.text()


def test_score_report_is_laid_out_like_the_curve_fit(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()
    html = window.score_results_label.html()
    assert "Components" in html and "Fit quality" in html and "RMSE" in html
    assert "<i>D</i>" in html                      # italic symbol, project rule
    # The plain report is still what is logged and saved.
    assert window.score_results_label.text().startswith("=")
    assert "peak table" in window.score_results_label.text()
    standalone = report_format.score_report_html(window._score_result)
    assert "worst slice" in standalone


def test_score_residual_can_be_ticked_onto_the_plot(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()
    listing = window.score_component_list
    residual = listing.item(listing.count() - 1)
    assert "Residual" in residual.text()
    assert residual.checkState() == Qt.CheckState.Unchecked
    residual.setCheckState(Qt.CheckState.Checked)
    assert len(window.fit_canvas._compare_axes) == 3   # two components + residual
    drawn = window._residual_spectrum_for_score()
    np.testing.assert_allclose(drawn.data, window._score_result["residual_first"])


def test_score_state_follows_the_selected_experiment(qapp, monkeypatch):
    """Explicit user report: the SCORE result's tick boxes stayed on screen
    for the next experiment selected."""
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    monkeypatch.setattr(_ScoreSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_score_clicked()
    assert window._score_result is not None
    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1]))
    window.spectrum_list.setCurrentRow(window.spectrum_list.count() - 1)
    assert window._score_result is None
    assert window.score_component_list.isHidden()
    assert window.score_annotate_check.isHidden()
    assert window.score_results_label.text() == ""
    window.spectrum_list.setCurrentRow(0)
    assert window._score_result is not None
    assert window.score_component_list.count() == 4


def test_score_dialog_names_the_substituted_model(qapp, monkeypatch):
    window, _b, _ppm = _overlapping_pfg_window(qapp)
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("stejskal_tanner_biexponential"))
    seen = {}

    def capture(self):
        seen["labels"] = [w.text() for w in self.findChildren(QLabel)]
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(_ScoreSettingsDialog, "exec", capture)
    window._on_score_clicked()
    assert any("Stejskal–Tanner" in t and "single-component form" in t for t in seen["labels"])


# ---- 4. Compare: one inset list for every spectrum ----------------------------------

def _trace(x0, name):
    ppm = np.linspace(10.0, 0.0, 2000)
    y = np.exp(-((ppm - x0) / 0.05) ** 2) + 0.2 * np.exp(-((ppm - 2.0) / 0.05) ** 2)
    return NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=ppm)],
                       metadata=NMRMetadata(vendor="derived", nucleus="1H", sfo1=400.0,
                                            source_path=name))


def _stack_with_two_insets():
    canvas = MplCanvas()
    canvas.show_compare([_trace(5.0, "A"), _trace(6.0, "B")], mode="stack",
                        legend_labels=["A", "B"])
    rows = canvas._plain_axes()
    canvas.add_inset_region((2.3, 1.7), (0.0, 0.3), ax=rows[0])
    canvas.add_inset_region((2.3, 1.7), (0.0, 0.3), ax=rows[1])
    return canvas


def test_every_rows_insets_are_listed(qapp):
    canvas = _stack_with_two_insets()
    assert canvas.inset_rows() == ["A", "B"]
    assert [spec["row"] for spec in canvas.all_inset_specs()] == [0, 1]
    dialog = InsetDialog(canvas)
    assert dialog.inset_list.count() == 2
    assert dialog.inset_list.item(0).text().startswith("A – ")
    assert dialog.inset_list.item(1).text().startswith("B – ")


def test_the_first_rows_inset_can_still_be_removed(qapp):
    """Explicit user report: after an inset on a second spectrum, the first
    one could no longer be removed."""
    canvas = _stack_with_two_insets()
    dialog = InsetDialog(canvas)
    dialog.inset_list.setCurrentRow(0)
    dialog.remove_selected()
    assert [spec["row"] for spec in canvas.all_inset_specs()] == [1]
    canvas.canvas.draw()
    rows = canvas._plain_axes()
    assert not rows[0].child_axes and len(rows[1].child_axes) == 1
    dialog.clear_insets()
    assert canvas.all_inset_specs() == []


def test_an_inset_can_be_placed_in_another_rows_area(qapp):
    canvas = _stack_with_two_insets()
    specs = canvas.all_inset_specs()
    specs[0]["host"] = 1
    canvas.set_all_inset_specs(specs)
    canvas.canvas.draw()
    rows = canvas._plain_axes()
    assert not rows[0].child_axes               # row A's inset now sits in row B
    assert len(rows[1].child_axes) == 2
    # Still row A's: removing row A's insets removes it.
    inset_module.remove_insets(rows[0])
    assert len(rows[1].child_axes) == 1
    dialog = InsetDialog(canvas)
    canvas.set_all_inset_specs(specs)
    dialog.refresh()
    assert "placed in B" in dialog.inset_list.item(0).text()


def test_inset_host_in_a_notebook():
    import matplotlib.pyplot as plt
    fig, (top, bottom) = plt.subplots(2, 1)
    x = np.linspace(0, 10, 500)
    top.plot(x, np.exp(-((x - 3) / 0.1) ** 2))
    bottom.plot(x, np.exp(-((x - 7) / 0.1) ** 2))
    inset = inset_module.add_inset(top, xlim=(2.5, 3.5), host_ax=bottom)
    assert inset in bottom.child_axes and inset not in top.child_axes
    assert inset_module.insets_of(top) == [inset]
    plt.close(fig)


# ---- 5. Integral calibration on the plot ---------------------------------------------

def test_normalised_integrals_with_a_reference():
    assert pil.normalised_integrals([2.0, 4.0, 6.0], reference=(2, 3.0)) == \
        pytest.approx([1.0, 2.0, 3.0])
    assert pil.normalised_integrals([2.0, 4.0]) == pytest.approx([1.0, 2.0])
    with pytest.raises(ValueError):
        pil.normalised_integrals([0.0, 4.0], reference=(0, 1.0))
    with pytest.raises(ValueError):
        pil.normalised_integrals([2.0, 4.0], reference=(5, 1.0))


def test_reference_region_by_click_and_by_stored_limits():
    regions = [(1.0, 2.0), (3.0, 4.0)]
    assert pil.reference_region(regions, 3.5) == 1
    assert pil.reference_region(regions, 2.4) == 0            # nearest centre
    assert pil.reference_region(regions, None,
                                calibration={"region": [2.9, 4.1], "value": 3}) == 1
    assert pil.reference_region(regions, None,
                                calibration={"region": [8.0, 9.0], "value": 3}) is None


def test_calibrate_integral_from_the_plot(qapp, monkeypatch):
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    window.spectrum_list.setCurrentRow(0)
    window._add_region_row(5.3, 4.7)
    window._add_region_row(2.3, 1.7)
    canvas = window.canvas
    assert not window.integral_labels_check.isChecked()

    monkeypatch.setattr(QInputDialog, "getDouble", staticmethod(lambda *a, **k: (3.0, True)))
    window._on_integral_calibration_requested(5.0)
    spectrum = window.session.spectra[0]
    assert spectrum.metadata.raw["integral_calibration"]["value"] == 3.0
    assert window.integral_labels_check.isChecked()           # labels turned on
    labels = dict(canvas._integral_label_items)
    assert labels[5.0].endswith("3.00")
    # The other region follows: half the height, the same width -> 1.50.
    assert labels[2.0].endswith("1.50")
    assert canvas.integral_calibrated

    texts = []
    monkeypatch.setattr(canvas, "_exec_context_menu",
                        lambda menu, _pos: texts.extend(a.text() for a in menu.actions()))
    canvas._last_right_click_xy = (5.0, 0.5)
    canvas._on_context_menu(QPoint(5, 5))
    assert "Calibrate Integral Here…" in texts and "Reset Integral Calibration" in texts

    window._on_integral_calibration_reset()
    assert "integral_calibration" not in spectrum.metadata.raw
    labels = dict(canvas._integral_label_items)
    assert labels[2.0].endswith("1.00") and labels[5.0].endswith("2.00")
    assert not canvas.integral_calibrated


# ---- 6. Truncate to in TopSpin's TDeff --------------------------------------------------

def test_tdeff_conversions():
    assert processing_ft.digital_filter_points(1024, 434) == 78
    assert processing_ft.digital_filter_points(0, 434) == 0
    assert processing_ft.tdeff_to_points(1024, 78) == 434
    assert processing_ft.tdeff_to_points(1025, 78) == 434       # odd TDeff rounds down
    assert processing_ft.tdeff_to_points(100, 78) == 1          # never below one point
    assert processing_ft.points_to_tdeff(434, 78) == 1024


def test_truncate_box_in_topspin_tdeff(qapp, tmp_path, monkeypatch):
    from test_gui import _isolate_sidecars, _make_raw_2d_dataset
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=64), vendor="bruker")
    assert window._truncate_units() == "tdeff"
    assert window.tdeff_spin.isHidden() and not window.tdeff_topspin_spin.isHidden()
    f = window._truncate_filter_points
    assert window.tdeff_topspin_spin.value() == \
        processing_ft.points_to_tdeff(window.tdeff_spin.value(), f)
    window.tdeff_topspin_spin.setValue(processing_ft.points_to_tdeff(40, f))
    assert window.tdeff_spin.value() == 40                     # what processing reads
    window.tdeff_unit_combo.setCurrentIndex(window.tdeff_unit_combo.findData("points"))
    assert not window.tdeff_spin.isHidden() and window.tdeff_topspin_spin.isHidden()
    window.tdeff_spin.setValue(30)
    assert window.tdeff_topspin_spin.value() == processing_ft.points_to_tdeff(30, f)
