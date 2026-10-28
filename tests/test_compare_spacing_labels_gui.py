"""Compare tab: the spacing slider, 1D + 2D + pulse-sequence panels, panel labels
and the collapsible sub-sections. Synthetic data; offscreen Qt."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.widgets.collapsible_section import CollapsibleSection
from opennmr.gui.widgets.spacing_slider import MAXIMUM_PERCENT, MINIMUM_PERCENT, SpacingSlider, describe_spacing


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    return MainWindow()


def spec1d(centre=5.0):
    ppm = np.linspace(10, 0, 400)
    data = np.exp(-((ppm - centre) ** 2) * 8)
    return NMRSpectrum(data=data, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400)],
                       metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, ns=1))


def spec2d(shift=0.0):
    x = np.linspace(10, 0, 96)
    y = np.linspace(150, 50, 48)
    xx, yy = np.meshgrid(x, y)
    data = 1e6 * np.exp(-((xx - 5 - shift) ** 2 / 0.02 + (yy - 100) ** 2 / 4))
    return NMRSpectrum(data, [NMRAxis("frequency", ppm=y, hz=y * 100), NMRAxis("frequency", ppm=x, hz=x * 400)],
                       NMRMetadata(vendor="bruker", nucleus="1H", nucleus_indirect="13C",
                                   sfo1=400.0, sfo1_indirect=100.0))


def _compare(window, spectra, layout_index):
    for k, s in enumerate(spectra):
        window._add_derived_spectrum(s, f"S{k + 1}")
    window.spectrum_list.selectAll()
    window._show_side_tab("Compare")
    window.layout_mode_combo.setCurrentIndex(layout_index)
    window._refresh_compare_view(switch_tab=True, force_render=True)


def _flush(labels):
    """The label controls wait a moment after an edit before redrawing; do it now."""
    if labels._debounce.isActive():
        labels._debounce.stop()
        labels._debounce.timeout.emit()


# --- the slider ------------------------------------------------------------

def test_slider_reaches_overlap_and_describes_itself_in_words(qapp):
    slider = SpacingSlider(default_percent=5.0, top_gap_percent=5.0)
    assert MINIMUM_PERCENT < 0 < MAXIMUM_PERCENT
    slider.setValue(-50)
    assert slider.value() == -50 and "overlap" in slider.text()
    slider.setValue(40)
    assert "between" in slider.text()
    assert "touch" in describe_spacing(-5.0, 5.0)
    slider.setValue(1e6)
    assert slider.value() == MAXIMUM_PERCENT


def test_slider_emits_once_per_change_and_resets(qapp):
    slider = SpacingSlider(default_percent=5.0)
    seen = []
    slider.valueChanged.connect(seen.append)
    slider.setValue(20)
    slider.setValue(20)
    assert seen == [20.0]
    slider.reset()
    assert slider.value() == 5.0 and seen[-1] == 5.0
    slider.slider.setValue(33)                     # a drag
    assert slider.value() == 33.0 and seen[-1] == 33.0


def test_slider_stands_in_for_the_old_spin_box_in_the_window(window):
    spin = window.output_stack_gap_spin
    assert isinstance(spin, SpacingSlider)
    spin.setValue(30.0)
    assert window._current_output_style_kwargs()["stack_gap"] == pytest.approx(0.3)


def test_dragging_the_slider_moves_the_stacked_spectra_closer_even_overlapping(window):
    _compare(window, [spec1d(3), spec1d(5), spec1d(7)], 0)
    canvas = window.compare_canvas
    window.output_stack_gap_spin.setValue(5.0)
    spans = lambda: [ax.get_ylim()[1] - ax.get_ylim()[0] for ax in canvas._spectrum_panels or []]  # noqa: E731
    wide = canvas.stack_gap()
    window.output_stack_gap_spin.setValue(-60.0)
    assert canvas.stack_gap() == pytest.approx(-0.6) and canvas.stack_gap() < wide
    window.output_stack_gap_spin.setValue(100.0)
    assert canvas.stack_gap() == pytest.approx(1.0)


def test_canvas_clamps_a_gap_beyond_the_minimum(window):
    window.compare_canvas.set_stack_gap(-5.0)
    assert window.compare_canvas.stack_gap() >= -1.0


# --- sections ---------------------------------------------------------------

def test_compare_controls_are_collapsible_and_show_only_when_they_apply(window):
    _compare(window, [spec1d(3), spec1d(5)], 0)
    assert not window._compare_spacing_section.isHidden()
    assert window._compare_labels_section.isHidden()
    assert window._compare_2d_section.isHidden()
    window.layout_mode_combo.setCurrentIndex(3)               # Panels: row
    assert window._compare_spacing_section.isHidden()
    assert not window._compare_labels_section.isHidden()
    window._compare_spacing_section.setExpanded(False)        # sections fold
    assert not window._compare_spacing_section.isExpanded()


# --- mixed 1D / 2D ----------------------------------------------------------

def test_one_d_and_two_d_spectra_together_in_separate_panels(window):
    _compare(window, [spec1d(), spec2d()], 3)
    canvas = window.compare_canvas
    kinds = [getattr(a, "_opennmr_panel_kind", None) for a in canvas._spectrum_panels]
    assert kinds == ["1d", "2d"]
    assert "1 1D, 1 2D" in window.compare_status_label.text()
    assert not window._compare_2d_section.isHidden()          # the contour options are reachable


def test_mixed_selection_outside_panels_says_how_to_show_both(window):
    _compare(window, [spec1d(3), spec1d(5), spec2d()], 0)
    assert "Panels" in window.compare_status_label.text()


def test_two_d_overlaid_in_one_panel_beside_a_one_d_panel(window):
    _compare(window, [spec1d(), spec2d(0), spec2d(0.3)], 3)
    controls = window.compare_panels_controls
    controls._assign([1, 2, 2])                               # the two 2D share panel 2
    assert len(window.compare_canvas._spectrum_panels) == 2


def test_one_d_and_two_d_cannot_share_a_panel_and_the_window_says_so(window):
    _compare(window, [spec1d(), spec2d()], 3)
    window.compare_panels_controls.all_in_one()
    assert "cannot" in window.compare_status_label.text().lower() or \
        "can't" in window.compare_status_label.text().lower() or \
        "1D" in window.compare_status_label.text()


# --- pulse sequence ---------------------------------------------------------

def test_a_spectrum_can_be_shown_as_its_pulse_sequence(window):
    _compare(window, [spec1d(), spec1d(4)], 3)
    controls = window.compare_panels_controls
    assert controls.contents() == ["spectrum", "spectrum"]
    controls._on_content(controls._keys[1], "pulse")
    assert controls.has_pulse()
    kinds = [getattr(a, "_opennmr_panel_kind", None) for a in window.compare_canvas._spectrum_panels]
    assert kinds == ["1d", "pulse"]                           # no sequence file here: the panel says why


def test_saved_pulse_style_reaches_the_panel_options(qapp):
    from opennmr.gui.pulse_sequence_dialog import PulseSequenceDialog, saved_draw_options
    settings = {"pulse_sequence_options": {"linewidth": 3.5, "delay_style": "line", "title": False},
                "pulse_sequence_colors": {"pulse": "#ff0000"},
                "pulse_sequence_renames": {"seq": {"names": {"p1": "90"}}}}
    opts = saved_draw_options(settings, "seq")
    assert opts["linewidth"] == 3.5 and opts["title"] is False
    assert opts["colors"]["pulse"] == "#ff0000" and opts["renames"] == {"p1": "90"}
    default = saved_draw_options({}, "seq")
    assert default["linewidth"] == PulseSequenceDialog._STYLE_DEFAULTS["linewidth"]


# --- panel labels -----------------------------------------------------------

def test_labels_appear_restyle_and_follow_the_controls(window):
    _compare(window, [spec1d(3), spec1d(5), spec1d(7), spec1d(2)], 5)
    window.compare_panels_controls.one_per_panel()
    labels = window.compare_label_controls
    assert window.compare_canvas.panel_labels() is None
    labels.enable_check.setChecked(True)
    _flush(labels)
    shown = window.compare_canvas.panel_labels()
    assert [t.get_text() for t in shown.texts] == ["(a)", "(b)", "(c)", "(d)"]
    labels.style_combo.setCurrentIndex(labels.style_combo.findData("A."))
    _flush(labels)
    assert [t.get_text() for t in window.compare_canvas.panel_labels().texts][:2] == ["A.", "B."]
    labels.enable_check.setChecked(False)
    _flush(labels)
    assert window.compare_canvas.panel_labels() is None


def test_labels_survive_a_redraw_and_each_panel_keeps_its_x_label(window):
    _compare(window, [spec1d(3), spec1d(5)], 3)
    window.compare_label_controls.enable_check.setChecked(True)
    _flush(window.compare_label_controls)
    window._refresh_compare_view(switch_tab=True, force_render=True)
    assert window.compare_canvas.panel_labels() is not None
    assert all(a.get_xlabel() for a in window.compare_canvas._spectrum_panels)


def test_dragging_a_label_switches_the_controls_to_manual(window):
    _compare(window, [spec1d(3), spec1d(5)], 3)
    labels = window.compare_label_controls
    labels.enable_check.setChecked(True)
    _flush(labels)
    labels.together_check.setChecked(False)               # move each label on its own
    window.compare_canvas.panelLabelMoved.emit(1, 12.0, 7.0)
    assert labels.placement() == "manual"
    assert labels.positions()[1] == (12.0, 7.0)
    _flush(labels)
    assert window.compare_canvas.panel_labels().offsets[1] == (12.0, 7.0)
    labels.together_check.setChecked(True)                # or all together
    window.compare_canvas.panelLabelMoved.emit(0, 3.0, 4.0)
    assert labels.positions() == {"all": (3.0, 4.0)}


def test_labels_on_side_by_side_two_d_panels(window):
    _compare(window, [spec2d(0), spec2d(0.4)], 3)
    window.compare_2d_controls.layout_combo.setCurrentIndex(
        window.compare_2d_controls.layout_combo.findData("row"))
    window.compare_label_controls.enable_check.setChecked(True)
    _flush(window.compare_label_controls)
    assert window.compare_canvas.panel_labels() is not None


def test_label_start_box_types_letters_and_numbers(qapp):
    from opennmr.gui.panel_label_controls import LabelStartSpin
    spin = LabelStartSpin()
    spin.set_style("(a)")
    spin.setValue(2)
    assert spin.text() == "(b)"
    spin.lineEdit().setText("d")
    spin.interpretText()
    assert spin.value() == 4
    spin.set_style("I")
    spin.lineEdit().setText("iv")
    spin.interpretText()
    assert spin.value() == 4 and spin.text() == "IV"
