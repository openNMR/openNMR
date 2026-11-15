"""Five user reports, 2026-11-15: a fit opens on one click (tested in
test_experiment_tree.py), the solution-NMR spin-system panel has no gap
inside it, typed axis limits belong to one spectrum, a Plot Style edit pushes
only what it changed, and Compare keeps plotting after other work."""
import pytest
from PySide6.QtWidgets import QMessageBox, QSizePolicy

from opennmr.gui.main_window import MainWindow

from test_gui import qapp  # noqa: F401 -- fixture
from test_gui_fit_sim_state import _quad_spectrum


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


# --- solution-state simulation: no gap ------------------------------------------------

def test_the_model_stack_cannot_grow_taller_than_its_page(qapp):  # noqa: F811
    window = MainWindow()
    sim = window.jcoupling_sim_panel
    sim.model_combo.setCurrentIndex(sim.model_combo.findData("spin_system"))
    policy = sim.model_stack.sizePolicy()
    assert policy.verticalPolicy() == QSizePolicy.Policy.Maximum
    window.close()


def test_the_page_sits_right_under_the_model_switch_in_every_model(qapp):  # noqa: F811
    """The stack used to take the height-for-width of its tallest page (the mixed
    model's wrapped note) and was centred in that: a ~270 px gap above the tables."""
    window = MainWindow()
    window.side_tabs.setCurrentIndex(
        [window.side_tabs.tabText(i) for i in range(window.side_tabs.count())].index("Simulate"))
    window._solution_sim_section.setExpanded(True)
    window.resize(1400, 1000)
    window.show()
    sim = window.jcoupling_sim_panel
    for key in ("first_order", "mixed", "spin_system"):
        sim.model_combo.setCurrentIndex(sim.model_combo.findData(key))
        qapp.processEvents()
        qapp.processEvents()
        gap = sim.model_stack.geometry().y() - sim.model_combo.geometry().bottom()
        assert 0 <= gap < 30, (key, gap)
        assert sim.model_stack.currentIndex() == sim.model_combo.currentIndex()
    window.close()


def test_the_stack_shows_one_page_and_keeps_the_stack_calls(qapp):  # noqa: F811
    from opennmr.gui.jcoupling_panels import PageSizedStack
    from PySide6.QtWidgets import QLabel
    stack = PageSizedStack()
    pages = [QLabel("a"), QLabel("b"), QLabel("c")]
    for page in pages:
        stack.addWidget(page)
    seen = []
    stack.currentChanged.connect(seen.append)
    stack.show()
    assert stack.count() == 3 and stack.currentIndex() == 0 and stack.currentWidget() is pages[0]
    assert [p.isVisibleTo(stack) for p in pages] == [True, False, False]
    stack.setCurrentIndex(2)
    assert [p.isVisibleTo(stack) for p in pages] == [False, False, True]
    assert seen == [2] and stack.indexOf(pages[2]) == 2 and stack.widget(1) is pages[1]
    stack.setCurrentIndex(9)                       # out of range: ignored
    assert stack.currentIndex() == 2


def test_no_gap_between_the_spin_tables_info_and_the_linewidth_row(qapp):  # noqa: F811
    window = MainWindow()
    sim = window.jcoupling_sim_panel
    sim.model_combo.setCurrentIndex(sim.model_combo.findData("spin_system"))
    sim.resize(420, 1600)                       # far more height than the content needs
    sim.show()
    qapp.processEvents()
    editor = sim.spin_editor
    info_bottom = editor.info_label.mapTo(sim, editor.info_label.rect().bottomLeft()).y()
    linewidth_top = sim.spin_fwhm_spin.mapTo(sim, sim.spin_fwhm_spin.rect().topLeft()).y()
    assert linewidth_top - info_bottom < 60, (info_bottom, linewidth_top)
    sim.hide()
    window.close()


# --- axis limits belong to the spectrum they were set on -------------------------------

def test_typed_axis_limits_reset_when_another_spectrum_is_selected(qapp):  # noqa: F811
    window = MainWindow()
    a = _quad_spectrum(window, "a")
    b = _quad_spectrum(window, "b")
    window._select_single_row(a)
    window.canvas.set_view_limits(x=(-20, 40))
    assert window.canvas.view_limits()["x"] == (-20.0, 40.0)
    window._select_single_row(b)
    assert window.canvas.view_limits() == {"x": None, "y": None}
    lo, hi = sorted(window.canvas.ax.get_xlim())
    assert lo < -40 and hi > 40                 # the automatic range, not -20..40
    # Limits set on b stay for b: re-selecting b itself (a redraw) keeps them,
    window.canvas.set_view_limits(x=(0, 30))
    window._on_selection_changed(b)
    assert window.canvas.view_limits()["x"] == (0.0, 30.0)
    # and going back to a opens a on its own range again.
    window._select_single_row(a)
    assert window.canvas.view_limits()["x"] is None
    window.close()


def test_forget_view_limits_is_silent_and_reports_whether_there_were_any(qapp):  # noqa: F811
    window = MainWindow()
    a = _quad_spectrum(window, "a")
    window._select_single_row(a)
    assert window.canvas.forget_view_limits() is False
    window.canvas.set_view_limits(x=(-5, 5))
    assert window.canvas.forget_view_limits() is True
    assert window.canvas.view_limits() == {"x": None, "y": None}
    window.close()


# --- a style set stays until changed -------------------------------------------------

def test_a_linewidth_edit_does_not_push_stale_panel_values_onto_the_plot(qapp):  # noqa: F811
    window = MainWindow()
    a = _quad_spectrum(window, "a")
    window._select_single_row(a)
    canvas = window.canvas
    window.canvas_tabs.setCurrentWidget(canvas)
    window._load_style_controls_from(canvas)
    # The colour scale and contour colour are changed on the plot itself
    # (not through the panel's boxes, which therefore still show the old ones).
    canvas.set_style(map_colormap="plasma", contour_colour="crimson")
    assert window.output_map_colormap_combo.currentData() != "plasma"
    window.output_linewidth_spin.setValue(window.output_linewidth_spin.value() + 1.5)
    settings = canvas.style_settings()
    assert settings["linewidth"] == pytest.approx(window.output_linewidth_spin.value())
    assert settings["map_colormap"] == "plasma"
    assert settings["contour_colour"] == "crimson"
    # An edit of the colour box itself still takes effect.
    index = window.output_map_colormap_combo.findData("magma")
    window.output_map_colormap_combo.setCurrentIndex(index)
    assert canvas.style_settings()["map_colormap"] == "magma"
    window.close()


def test_every_edit_still_reaches_the_plot(qapp):  # noqa: F811
    window = MainWindow()
    a = _quad_spectrum(window, "a")
    window._select_single_row(a)
    window._load_style_controls_from(window.canvas)
    window.output_fontsize_spin.setValue(window.output_fontsize_spin.value() + 2)
    window.output_tick_length_spin.setValue(window.output_tick_length_spin.value() + 1)
    s = window.canvas.style_settings()
    assert s["labelsize"] == pytest.approx(window.output_fontsize_spin.value())
    assert s["tick_length"] == pytest.approx(window.output_tick_length_spin.value())
    window.close()


# --- Compare keeps working -----------------------------------------------------------

def _two_sets(window):
    first = [_quad_spectrum(window, f"first {i}") for i in range(2)]
    second = [_quad_spectrum(window, f"second {i}") for i in range(2)]
    return first, second


def test_compare_plots_a_second_set_after_a_first(qapp):  # noqa: F811
    window = MainWindow()
    first, second = _two_sets(window)
    for rows in (first, second):
        window.spectrum_list.clearSelection()
        for r in rows:
            window.spectrum_list.item(r).setSelected(True)
        window._on_list_selection_changed()
        assert window.compare_list.count() == 2
        assert "Comparing 2 spectra" in window.compare_status_label.text()
        assert len(window.compare_canvas.figure.axes) >= 2
    window.close()


def test_a_compare_list_left_blocked_is_unblocked_by_the_next_selection(qapp):  # noqa: F811
    window = MainWindow()
    first, second = _two_sets(window)
    window.compare_list.blockSignals(True)      # what an interrupted rebuild left behind
    window.spectrum_list.clearSelection()
    for r in second:
        window.spectrum_list.item(r).setSelected(True)
    window._on_list_selection_changed()
    assert not window.compare_list.signalsBlocked()
    assert "Comparing 2 spectra" in window.compare_status_label.text()
    window.close()


def test_a_rebuild_that_fails_midway_still_unblocks_the_list(qapp, monkeypatch):  # noqa: F811
    window = MainWindow()
    first, _second = _two_sets(window)

    def boom(entry):
        raise RuntimeError("label failed")
    monkeypatch.setattr(window, "_compare_entry_label", boom)
    with pytest.raises(RuntimeError):
        window._rebuild_compare_list([(r, None) for r in first])
    assert not window.compare_list.signalsBlocked()
    window.close()


def test_entries_of_spectra_that_are_gone_do_not_carry_over(qapp):  # noqa: F811
    window = MainWindow()
    first, second = _two_sets(window)
    n = len(window.session)
    window._compare_pinned_entries = [(n + 3, None), (first[0], None)]
    window._compare_removed_entries = {(n + 5, None), (second[1], None)}
    window.spectrum_list.clearSelection()
    for r in second:
        window.spectrum_list.item(r).setSelected(True)
    window._on_list_selection_changed()
    assert (n + 3, None) not in window._compare_pinned_entries
    assert (n + 5, None) not in window._compare_removed_entries
    window.close()
