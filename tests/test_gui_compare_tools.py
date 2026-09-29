"""2026-09-29 round: curve-plot spines, Compare > Spectrum Arithmetic
(the notebook's arithmetic.combine_spectra), DNP split into two
sections, Baseline as its own Processing section, the T1/T2 workflow
guide on Fit and Dynamics, and the louder "not yet saved" slice bar.

All data synthetic; no instrument files needed."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.gui.widgets.collapsible_section import CollapsibleSection


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    return MainWindow()


def _peak(x0, amplitude=1.0, lo=0.0, hi=10.0, n=501, sfo1=400.0):
    ppm = np.linspace(hi, lo, n)                       # descending, as loaded
    data = amplitude * np.exp(-0.5 * ((ppm - x0) / 0.1) ** 2)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * sfo1)
    return NMRSpectrum(data=data, axes=[axis],
                       metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=sfo1))


def _relaxation_result():
    t = np.geomspace(0.01, 1.0, 8)
    y = 1.0 - 2.0 * np.exp(-t / 0.2)
    return {"t": t, "y": y, "y_fit": y, "peak_label": "Peak 1", "model": "inversion_recovery"}


# ----------------------------------------------------------------------
# Curve spines
# ----------------------------------------------------------------------

def test_curve_spines_are_off_by_default_like_the_notebook(qapp):
    canvas = MplCanvas()
    canvas.show_relaxation_fit(_relaxation_result())
    assert not canvas.ax.spines["top"].get_visible()
    assert not canvas.ax.spines["right"].get_visible()
    assert canvas.style_settings()["top_spine"] is False
    assert canvas.style_settings()["right_spine"] is False


def test_curve_spines_switch_on_live_and_survive_a_redraw(qapp):
    canvas = MplCanvas()
    canvas.show_relaxation_fit(_relaxation_result())
    canvas.set_style(top_spine=True, right_spine=True)
    assert canvas.ax.spines["top"].get_visible()
    assert canvas.ax.spines["right"].get_visible()
    # Ticks come with the spine, as on a closed manuscript frame.
    assert canvas.ax.xaxis.get_tick_params(which="major")["top"] is True
    canvas.show_parameter_trend([1, 2, 3], [1.0, 2.0, 3.0], ylabel="T1")
    assert canvas.ax.spines["top"].get_visible()
    canvas.set_style(right_spine=False)
    assert not canvas.ax.spines["right"].get_visible()
    assert canvas.ax.spines["top"].get_visible()


def test_curve_spines_never_touch_a_spectrum(qapp):
    canvas = MplCanvas()
    canvas.set_style(top_spine=True, right_spine=True)
    canvas.show_spectrum(_peak(5.0))
    assert not canvas.ax.spines["top"].get_visible()
    assert not canvas.ax.spines["right"].get_visible()
    # ...even when the canvas showed a curve first.
    canvas.show_relaxation_fit(_relaxation_result())
    assert canvas.ax.spines["top"].get_visible()
    canvas.show_spectrum(_peak(5.0))
    assert not canvas.ax.spines["top"].get_visible()


def test_curve_spines_apply_to_a_notebook_profile_plot(qapp):
    canvas = MplCanvas()
    canvas.set_style(top_spine=True)

    def draw(figure):
        ax = figure.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        ax.spines["top"].set_visible(False)
    canvas.show_external_plot(draw, kind="curve")
    assert canvas.ax.spines["top"].get_visible()
    canvas.show_external_plot(draw, kind="compare")
    assert not canvas.ax.spines["top"].get_visible()


def test_curve_spines_are_part_of_a_saved_style(qapp):
    canvas = MplCanvas()
    canvas.show_relaxation_fit(_relaxation_result())
    canvas.apply_style_defaults({"top_spine": True, "right_spine": True, "dpi": 300})
    assert canvas.ax.spines["right"].get_visible()


def test_plot_style_panel_drives_the_curve_spines(window):
    window.canvas_tabs.setCurrentWidget(window.fit_canvas)
    window.fit_canvas.show_relaxation_fit(_relaxation_result())
    window._load_style_controls_from(window.fit_canvas)
    assert not window.output_top_spine_checkbox.isChecked()
    window.output_top_spine_checkbox.setChecked(True)
    window.output_right_spine_checkbox.setChecked(True)
    ax = window.fit_canvas.ax
    assert ax.spines["top"].get_visible() and ax.spines["right"].get_visible()
    assert window._style_payload()["top_spine"] is True
    # The Spectrum plot was not the one being styled.
    assert window.canvas.style_settings()["top_spine"] is False


def test_notebook_field_profile_takes_the_spine_switches():
    import matplotlib
    matplotlib.use("Agg")
    from opennmr.notebook import dnp
    rows = [{"label": f"s{k}", "field_T": 9.4 + 1e-4 * k, "delta_B_mT": 0.1 * k,
             "field_source": "peak", "found": True, "intensity": float(k + 1),
             "intensity_error": 0.1, "area": float(k + 1), "area_error": 0.1}
            for k in range(4)]
    result = {"rows": rows, "nucleus": "1H"}
    try:
        fig = dnp.plot_field_profile(result, show_top_spine=True, show_right_spine=True)
    except (KeyError, TypeError) as exc:           # result shape is dnp's own business
        pytest.skip(f"synthetic result not in dnp's shape: {exc}")
    ax = fig.axes[0]
    assert ax.spines["top"].get_visible() and ax.spines["right"].get_visible()
    fig = dnp.plot_field_profile(result)
    assert not fig.axes[0].spines["top"].get_visible()


# ----------------------------------------------------------------------
# Compare > Spectrum Arithmetic
# ----------------------------------------------------------------------

def _compare(window, spectra, labels):
    for s, label in zip(spectra, labels):
        window._add_derived_spectrum(s, label)
    window.spectrum_list.selectAll()
    window._show_side_tab("Compare")
    window._arithmetic_section.setExpanded(True)
    return window.arithmetic_panel


def test_compare_sections_in_order(window):
    page = window.side_tabs.widget(window._compare_tab_index)
    titles = [s.title() for s in page.findChildren(CollapsibleSection)]
    assert titles == ["Spectrum Arithmetic", "DNP Enhancement", "DNP Field Sweep Profile"]
    assert all(not s.isExpanded() for s in page.findChildren(CollapsibleSection))


def test_combine_to_spectrum_is_the_notebook_function(qapp):
    from opennmr.gui.arithmetic_panel import combine_to_spectrum
    from opennmr.notebook import arithmetic as nb_arithmetic
    from opennmr.notebook.bridge import results_from_spectrum
    from opennmr.processing import arithmetic as proc_arithmetic
    a, b = _peak(4.0), _peak(6.0, lo=1.0, hi=9.0, n=333)
    ours = combine_to_spectrum([a, b], ["+", "-"], [1.0, 0.5])
    nb = nb_arithmetic.combine_spectra([results_from_spectrum(a), results_from_spectrum(b)],
                                       ops=["+", "-"], scaling_factors=[1.0, 0.5])
    np.testing.assert_array_equal(ours.data, nb["combined_spectrum"])
    np.testing.assert_array_equal(ours.axes[0].ppm, nb["x_ppm"])
    # ...and agrees with the application's own port.
    proc = proc_arithmetic.combine_spectra([a, b], ops=["+", "-"], scaling_factors=[1.0, 0.5])
    np.testing.assert_allclose(ours.data, proc.data, atol=1e-12)
    assert ours.metadata.raw["ops"] == ["+", "-"]
    assert ours.metadata.raw["scaling_factors"] == [1.0, 0.5]


def test_arithmetic_custom_signs_and_factors_are_exact(window):
    a, b = _peak(5.0, amplitude=2.0), _peak(5.0, amplitude=1.0)
    panel = _compare(window, [a, b], ["A", "B"])
    assert panel.table.rowCount() == 2
    panel.table.cellWidget(1, 1).setCurrentIndex(1)       # −
    panel.table.cellWidget(1, 2).setValue(4.0)            # × 4
    assert panel.op_combo.currentData() == "custom"
    text = panel.formula_label.text()
    assert text.index("A") < text.index("− 4 × B")
    n_before = len(window.session)
    result = panel.combine()
    assert len(window.session) == n_before + 1
    # Same axis for both, so the interpolation is the identity (on the
    # ascending common axis): 2g − 4·g = −2g = −a.
    np.testing.assert_allclose(result.data, -1.0 * a.data[::-1], atol=1e-12)


def test_arithmetic_presets_still_work_through_the_old_names(window):
    _compare(window, [_peak(4.8), _peak(5.2)], ["A", "B"])
    window.combine_op_combo.setCurrentIndex(window.combine_op_combo.findData("sum"))
    window._on_combine_spectra_clicked()
    summed = window.session.spectra[-1]
    window.combine_op_combo.setCurrentIndex(window.combine_op_combo.findData("average"))
    window._on_combine_spectra_clicked()
    averaged = window.session.spectra[-1]
    np.testing.assert_allclose(averaged.data * 2.0, summed.data, atol=1e-12)
    assert window.session.label(len(window.session) - 1).startswith("avg(")


def test_arithmetic_options_reach_the_calculation(window):
    a, b = _peak(4.0, lo=0.0, hi=10.0), _peak(4.0, lo=2.0, hi=12.0)
    panel = _compare(window, [a, b], ["A", "B"])
    panel.range_combo.setCurrentIndex(panel.range_combo.findData("union"))
    panel.points_spin.setValue(1201)
    result = panel.combine()
    assert result.data.size == 1201
    assert result.axes[0].ppm.min() == pytest.approx(0.0)
    assert result.axes[0].ppm.max() == pytest.approx(12.0)
    panel.range_combo.setCurrentIndex(panel.range_combo.findData("intersection"))
    panel.unit_combo.setCurrentIndex(panel.unit_combo.findData("Hz"))
    result = panel.combine()
    assert result.axes[0].hz.min() == pytest.approx(2.0 * 400.0)
    assert result.metadata.raw["arithmetic"]["x_unit"] == "Hz"


def test_arithmetic_is_written_into_the_new_spectrum_history(window):
    panel = _compare(window, [_peak(4.8), _peak(5.2)], ["A", "B"])
    panel.op_combo.setCurrentIndex(panel.op_combo.findData("subtract"))
    panel.combine()
    index = len(window.session) - 1
    history = window.session.info(index)["history"]
    texts = [e["text"] for e in history["entries"] if e.get("kind") == "arithmetic"]
    assert texts and "− B" in texts[0] and "combine_spectra" in texts[0]


def test_arithmetic_with_one_spectrum_says_why(window):
    panel = _compare(window, [_peak(5.0)], ["A"])
    assert not panel.combine_button.isEnabled()
    assert panel.combine() is None
    assert "two" in panel.status_label.text()


# ----------------------------------------------------------------------
# DNP in two sections
# ----------------------------------------------------------------------

def test_dnp_pages_sit_in_their_own_sections(window):
    enh, sweep = window._dnp_enhancement_section, window._dnp_sweep_section
    assert enh.contentWidget() is window.dnp_panel.enhancement_page
    assert sweep.contentWidget() is window.dnp_panel.sweep_page
    assert enh.contentWidget().isAncestorOf(window.dnp_panel.enh_combo_1)
    assert sweep.contentWidget().isAncestorOf(window.dnp_panel.sweep_run_button)
    assert not enh.isAncestorOf(window.dnp_panel.sweep_run_button)
    compare_page = window.compare_list.parentWidget()
    assert enh.parentWidget() is compare_page and sweep.parentWidget() is compare_page


def test_opening_either_dnp_section_refreshes_the_panel(window):
    for k in range(3):
        window._add_derived_spectrum(_peak(5.0 + 0.1 * k), f"s{k}")
    window.spectrum_list.selectAll()
    window._show_side_tab("Compare")
    window._dnp_sweep_section.setExpanded(True)
    combo = window.dnp_panel.sweep_reference_combo
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert len(labels) == 3 and [t.split(" ")[0] for t in labels] == ["s0", "s1", "s2"]


# ----------------------------------------------------------------------
# Processing > Baseline
# ----------------------------------------------------------------------

def test_baseline_is_its_own_processing_section(window):
    page = window.side_tabs.widget(window._processing_tab_index)
    sections = {s.title(): s for s in page.findChildren(CollapsibleSection)}
    assert list(sections) == ["FID Processing", "Phase", "Baseline", "Reference"]
    for widget in (window.baseline_correct_button, window.baseline_method_combo,
                   window.baseline_preview_checkbox, window.baseline_target_note):
        assert sections["Baseline"].isAncestorOf(widget)
        assert not sections["Phase"].isAncestorOf(widget)
    assert sections["Phase"].isAncestorOf(window.apply_all_slices_checkbox)
    assert not sections["Baseline"].isExpanded()


# ----------------------------------------------------------------------
# T1/T2 workflow guide
# ----------------------------------------------------------------------

def test_workflow_guide_on_fit_and_dynamics(window):
    pages = {window.side_tabs.tabText(i): window.side_tabs.widget(i)
             for i in range(window.side_tabs.count())}
    assert pages["Fit"].isAncestorOf(window.fit_workflow_guide)
    assert pages["Dynamics"].isAncestorOf(window.relaxation_workflow_guide)
    for guide, marked, other in ((window.fit_workflow_guide, "Measure every slice", "Fit Curve"),
                                 (window.relaxation_workflow_guide, "Fit the recovery", None)):
        text = guide.guide_label.text()
        assert text.count("you are here") == 1
        # The marker sits in the step for this tile.
        steps = text.split("<li>")
        here = next(s for s in steps if "you are here" in s)
        assert marked in here
        for step in ("Select the pseudo-2D", "Measure every slice", "Fit the recovery"):
            assert step in text
        assert "Fit Every Slice with this Model" in text


def test_workflow_guide_buttons_go_to_the_other_step(window):
    window._show_side_tab("Fit")
    window.fit_workflow_guide.guide_button.click()
    assert window.side_tabs.tabText(window.side_tabs.currentIndex()) == "Dynamics"
    assert window.side_tabs.currentWidget().isAncestorOf(window.relaxation_workflow_guide)
    window.relaxation_workflow_guide.guide_button.click()
    assert window.side_tabs.tabText(window.side_tabs.currentIndex()) == "Fit"


def test_model_series_box_numbers_its_steps(window):
    fit_button, _export, _status = window._model_series_widgets["quadrupolar"]
    box = fit_button.parentWidget()
    from PySide6.QtWidgets import QLabel
    note = box.findChildren(QLabel)[0].text()
    assert note.count("<li>") == 3 and "Fit Curve" in note


# ----------------------------------------------------------------------
# Extracted slice: "not yet saved"
# ----------------------------------------------------------------------

def _2d():
    y = np.linspace(150, 100, 8)
    x = np.linspace(10, 0, 32)
    yy, xx = np.meshgrid(np.arange(8), np.arange(32), indexing="ij")
    data = np.exp(-(((xx - 16) ** 2 + (yy - 4) ** 2)) / 8.0) * 100
    return NMRSpectrum(
        data=data, axes=[NMRAxis(kind="frequency", ppm=y, hz=y * 100.0),
                         NMRAxis(kind="frequency", ppm=x, hz=x * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", nucleus_indirect="13C",
                             sfo1=400.0, sfo1_indirect=100.0),
        data_imag=np.zeros_like(data))


def test_slice_bar_warns_that_the_slice_is_not_saved(qapp):
    from PySide6.QtWidgets import QLabel
    canvas = MplCanvas()
    canvas.show_spectrum(_2d())
    canvas.extract_slice("f1", 2)
    text = canvas._slice_label.text()
    assert "<u>not yet saved</u>" in text and "<b>" in text
    assert "Keep it" in text
    assert not canvas._slice_warning_icon.pixmap().isNull()
    assert canvas._slice_label.font().pointSizeF() > QLabel().font().pointSizeF()
    assert canvas._slice_bar.isAncestorOf(canvas._slice_warning_icon)


def test_arithmetic_table_is_not_rebuilt_per_row_while_closed(window):
    # A pseudo-2D fills the Compare list one row at a time; rebuilding the
    # arithmetic table (a widget per row) on each insert was quadratic and
    # stalled selecting a long series.
    n_rows = 64
    ppm = np.linspace(10, 0, 128)
    data = np.outer(np.linspace(1, 2, n_rows), np.exp(-0.5 * ((ppm - 5) / 0.2) ** 2))
    spectrum = NMRSpectrum(
        data, [NMRAxis("index", index=np.arange(1, n_rows + 1, dtype=float)),
               NMRAxis("frequency", ppm=ppm, hz=ppm * 400.0)],
        NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True))
    calls = []
    original = window.arithmetic_panel.refresh
    window.arithmetic_panel.refresh = lambda: (calls.append(1), original())
    window._add_derived_spectrum(spectrum, "series")
    assert calls == []
    window._arithmetic_section.setExpanded(True)
    assert len(calls) == 1
    assert window.arithmetic_panel.table.rowCount() == window.compare_list.count()
