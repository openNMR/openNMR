"""Apply ▸ DNP (gui/dnp_panel.py): the step-by-step field-sweep and
enhancement workflows -- tick spectra, pick the line on the reference,
align, check/drag/confirm, measure (height and integral), then the profile
against B0 or the enhancement table (explicit user request, 2026-09-30).

Spectra are tests/test_dnp.py's synthetic 23Na ones, added to the session
as derived spectra, so no instrument data is needed.
"""
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.notebook import dnp_workflow as dw
from test_dnp import DELTA, ERRORS, FIELDS, FWHM, GAMMA_NA, _sweep


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1400, 900)
    return w


def _load(window, heights=None, panel="sweep"):
    spectra = _sweep(FIELDS, ERRORS, heights=heights)
    for k, s in enumerate(spectra):
        window._add_derived_spectrum(s, f"s{k}")
    window._show_side_tab("Integrate")
    window._dnp_section.setExpanded(True)
    section = window._dnp_sweep_section if panel == "sweep" else window._dnp_enhancement_section
    section.setExpanded(True)
    p = window.dnp_sweep_panel if panel == "sweep" else window.dnp_enhancement_panel
    p.refresh()
    return p, spectra


def _named(p, *prefixes):
    """The panel's labels for the spectra added as ``prefixes``."""
    return [e["label"] for e in p._rows if e["label"].split(" ")[0] in prefixes]


def _ref_line_ppm(spectrum):
    return float(spectrum.axes[0].ppm[np.argmax(np.abs(spectrum.data))])


def _through_alignment(p, spectra):
    p._tick_all(True)
    p.set_line(_ref_line_ppm(spectra[0]))
    p.align()
    p.confirm()


def test_step_one_lists_spectra_with_their_sfo1_field(window):
    p, spectra = _load(window)
    t = p.spectra_table
    assert t.rowCount() == 5
    assert [t.horizontalHeaderItem(c).text() for c in range(3)] == [
        "Spectrum", "SFO1 (MHz)", "B₀ from SFO1 (T)"]
    for r, (b, err) in enumerate(zip(FIELDS, ERRORS)):
        sfo1 = (GAMMA_NA * b * (1 + DELTA * 1e-6) - err) / 1e6
        assert float(t.item(r, 1).text()) == pytest.approx(sfo1, abs=1e-6)
        assert float(t.item(r, 2).text()) == pytest.approx(sfo1 * 1e6 / GAMMA_NA, abs=1e-6)
    assert not p.align_button.isEnabled()
    # Use the Open experiments selection.
    window.spectrum_list.clearSelection()
    for row in (1, 3):
        window.spectrum_list.item(row).setSelected(True)
    p.use_selected()
    assert p._labels() == [window.session.label(1), window.session.label(3)]


def test_pick_on_the_reference_then_align(window):
    p, spectra = _load(window)
    p._tick_all(True)
    p.pick_line_button.setChecked(True)
    assert window.compare_canvas._pick_mode
    assert window.canvas_tabs.currentWidget() is window.compare_canvas
    window._on_dnp_pick(_ref_line_ppm(spectra[0]) + 0.5)       # a click near the line
    assert not window.compare_canvas._pick_mode and p.line_ppm is not None
    assert p.align_button.isEnabled()
    al = p.align()
    for e, err in zip(al["entries"], ERRORS):
        assert e["found"] and e["shift_hz"] == pytest.approx(err - ERRORS[0], abs=3.0)
    assert p.align_table.rowCount() == 5
    # Not measurable until confirmed.
    assert not p.pick_shift_button.isEnabled()
    p.confirm()
    assert p.pick_shift_button.isEnabled() and "✓" in p.confirm_button.text()
    for s in p.aligned:
        assert _ref_line_ppm(s) == pytest.approx(al["peak_ppm"], abs=0.3)


def test_a_hand_shift_in_the_table_unconfirms(window):
    p, spectra = _load(window)
    _through_alignment(p, spectra)
    item = p.align_table.item(2, 2)
    item.setText("150")
    assert p.alignment["entries"][2]["manual_shift_hz"] == 150.0
    assert not p.alignment["confirmed"] and p.aligned is None
    assert not p.pick_shift_button.isEnabled()


def test_dragging_a_trace_in_the_alignment_window(window):
    p, spectra = _load(window)
    _through_alignment(p, spectra)
    p.open_align_dialog()
    dialog = p._align_dialog
    assert len(dialog.lines) == 5
    line = dialog.lines[3]
    xs, ys = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    k = int(np.argmax(ys))
    dialog.press(xs[k], ys[k])                     # on trace 3's peak
    dialog.motion(xs[k] + 4.0)
    dialog.release(xs[k] + 4.0)                    # dragged by +4 ppm
    e = p.alignment["entries"][3]
    assert e["manual_shift_hz"] == pytest.approx(
        -4.0e-6 * (e["line_frequency_hz"] - e["manual_shift_hz"]), rel=1e-3)
    moved = dw.aligned_spectrum(spectra[3], p.alignment, 3)
    assert _ref_line_ppm(moved) == pytest.approx(p.alignment["peak_ppm"] + 4.0, abs=0.3)
    assert not p.alignment["confirmed"]
    dialog.view_combo.setCurrentIndex(dialog.view_combo.findData("overlay"))
    p.reset_manual()
    assert p.alignment["entries"][3]["manual_shift_hz"] == 0.0


def test_measure_and_profile_against_b0(window):
    heights = [1000.0, 2000.0, -500.0, 4000.0, 3000.0]
    p, spectra = _load(window, heights=heights)
    _through_alignment(p, spectra)
    pk = p.alignment["peak_ppm"]
    width = FWHM / spectra[0].metadata.sfo1
    p.pick_shift_button.setChecked(True)
    window._on_dnp_pick(pk)
    assert p.measurement["height"] is not None and p.measurement["area"] is None
    assert "Do both" in p.measure_label.text()
    p.region_button.setChecked(True)
    window._on_dnp_region(pk - 8 * width, pk + 8 * width)
    m = p.measurement
    h = m["height"]["value"]
    np.testing.assert_allclose(h / h[0], np.array(heights) / heights[0], rtol=0.01, atol=0.01)
    # The profile is drawn as a curve plot, sorted by field.
    prof = p.profile
    assert prof is not None and np.all(np.diff(prof["field_T"]) > 0)
    assert window.fit_canvas.plot_kind() == "curve"
    assert len(window.fit_canvas.figure.axes) == 2          # height and integral
    # Field at the line vs from SFO1.
    at_line = prof["field_T"].copy()
    p.field_source_combo.setCurrentIndex(p.field_source_combo.findData("sfo1"))
    assert not np.allclose(p.profile["field_T"], at_line)
    assert "mT apart" in p.result_label.text() or "SFO1" in p.result_label.text()
    # Enhancement against an OFF spectrum.
    p.profile_mode_combo.setCurrentIndex(p.profile_mode_combo.findData("ratio"))
    p.off_combo.setCurrentIndex(0)
    assert p.profile["mode"] == "ratio" and len(p.profile["labels"]) == 4   # OFF left out
    p.exclude_off_check.setChecked(False)
    assert len(p.profile["labels"]) == 5
    p.profile_mode_combo.setCurrentIndex(p.profile_mode_combo.findData("one_minus"))
    assert "1 − ON / OFF" in p.result_label.text()
    # Kept with the spectra it used.
    record = window.session.spectra[0].metadata.raw["fits"]["dnp_sweep"]
    assert "Field profile" in record["report"]


def test_enhancement_workflow(window):
    heights = [1000.0, 12500.0, 8000.0, 1000.0, 1000.0]
    p, spectra = _load(window, heights=heights, panel="enhancement")
    assert p.spectra_table.columnCount() == 2
    p.set_checked(_named(p, "s0", "s1", "s2"))
    p.set_line(_ref_line_ppm(spectra[0]))
    p.align()
    p.confirm()
    pk = p.alignment["peak_ppm"]
    width = FWHM / spectra[0].metadata.sfo1
    p.set_peak(pk)
    p.set_region(pk - 8 * width, pk + 8 * width)
    p.off_combo.setCurrentIndex(0)                        # s0 is OFF
    rows = p.enhancement_rows
    assert [r["label"].split(" ")[0] for r in rows] == ["s1", "s2"]
    assert rows[0]["height"]["ratio"] == pytest.approx(12.5, rel=0.02)
    assert rows[1]["area"]["ratio"] == pytest.approx(8.0, rel=0.03)
    assert p.enh_table.rowCount() == 2 and p.enh_table.item(0, 3).text() == "yes"
    p.definition_combo.setCurrentIndex(p.definition_combo.findData("one_minus"))
    assert p.enh_table.item(0, 1).text().startswith("-11.")
    assert "1 − ON/OFF" in p.result_label.text()
    assert "dnp_enhancement" in window.session.spectra[1].metadata.raw["fits"]


def test_changing_the_spectra_resets_later_steps(window):
    p, spectra = _load(window)
    _through_alignment(p, spectra)
    p.set_checked(_named(p, "s0", "s1"))
    assert p.alignment is None and p.line_ppm is None and not p.align_button.isEnabled()


def test_messages_not_errors_when_steps_are_skipped(window, _dialogs):
    p, _spectra = _load(window)
    assert p.align() is None
    assert _dialogs and "Tick at least two" in _dialogs[-1][1]
    assert p.plot_profile(explicit=True) is None
    assert "Measure first" in _dialogs[-1][1]


def test_another_canvas_mode_releases_the_buttons(window):
    p, spectra = _load(window)
    p._tick_all(True)
    p.pick_line_button.setChecked(True)
    window.compare_canvas.set_region_select_mode(True)     # something else takes the canvas
    window.compare_canvas.set_region_select_mode(False)
    window.compare_canvas.set_pick_mode(False)
    assert not p.pick_line_button.isChecked() and p.armed is None
