"""The NMR Periodic Table tile: Larmor calculator on the left, the table
in place of the plot (explicit user request, 2026-09-28)."""
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from opennmr.gui.main_window import MainWindow
from opennmr.gui.periodic_table_panel import LarmorCalculatorPanel, PeriodicTableView
from opennmr.notebook import periodic_table as pt


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    w._show_side_tab("Periodic Table")
    qapp.processEvents()
    return w


def test_tile_sits_just_before_settings_with_its_icon(qapp):
    w = MainWindow()
    tiles = [w.side_tabs.tabText(i) for i in range(w.side_tabs.count())]
    assert tiles[-2:] == ["Periodic Table", "Settings"]
    assert w._periodic_table_tab_index == len(tiles) - 2
    assert not w.side_tabs._buttons[w._periodic_table_tab_index].icon().isNull()
    # Built lazily: nothing until the tile is opened.
    assert w.periodic_table_view is None


def test_table_replaces_the_plot_only_on_its_tile(window, qapp):
    assert isinstance(window.periodic_table_view, PeriodicTableView)
    assert window.plot_stack.currentWidget() is window.periodic_table_view
    assert not window.open_experiments_box.isVisible()
    window._show_side_tab("Open")
    qapp.processEvents()
    assert window.plot_stack.currentWidget() is window.canvas_tabs
    assert window.open_experiments_box.isVisible()
    window._show_side_tab("Periodic Table")
    assert window.plot_stack.currentWidget() is window.periodic_table_view


def test_ctrl_9_opens_the_periodic_table(window):
    window._show_side_tab("Open")
    action = next(a for a in window.findChildren(type(window.settings_menu_action))
                  if a.text() == "NMR Periodic Table")
    assert action.shortcut().toString() == "Ctrl+9"
    action.trigger()
    assert window.side_tabs.tabText(window.side_tabs.currentIndex()) == "Periodic Table"


def test_every_element_has_a_tile(window):
    assert set(window.periodic_table_view.tiles) == {e.symbol for e in pt.elements()}


def test_search_highlights_matches_and_dims_the_rest(window):
    view = window.periodic_table_view
    view.search_edit.setText("plat")
    assert view.tiles["Pt"].highlighted and not view.tiles["Pt"].dimmed
    assert view.tiles["Au"].dimmed and not view.tiles["Au"].highlighted
    view.search_edit.clear()
    assert not any(t.dimmed or t.highlighted for t in view.tiles.values())


def test_enter_on_a_nuclide_opens_the_card_and_sets_the_calculator(window):
    view = window.periodic_table_view
    view.search_edit.setText("17O")
    QTest.keyClick(view.search_edit, Qt.Key.Key_Return)
    assert view.card.isVisible()
    assert view.card.name_label.text() == "Oxygen"
    assert window.larmor_panel.nuclide() == "17O"
    assert view.tiles["O"].selected


def test_clicking_a_tile_opens_its_card_and_picks_the_principal_isotope(window, qapp):
    view = window.periodic_table_view
    tile = view.tiles["Cu"]
    QTest.mouseClick(tile, Qt.MouseButton.LeftButton, pos=tile.rect().center())
    assert view.card.isVisible()
    assert view.card.table.rowCount() == 2
    assert window.larmor_panel.nuclide() == "63Cu"
    # A row in the card sends that isotope to the calculator.
    view.card.table.cellClicked.emit(1, 0)
    assert window.larmor_panel.nuclide() == "65Cu"
    # Clicking the element again keeps an isotope already of that element.
    view.open_element("Cu")
    assert window.larmor_panel.nuclide() == "65Cu"
    QTest.keyClick(view.card, Qt.Key.Key_Escape)
    assert not view.card.isVisible()


def test_card_for_an_element_without_nmr_isotopes(window):
    view = window.periodic_table_view
    view.open_element("Ar")
    assert not view.card.table.isVisibleTo(view.card)
    assert "no NMR-active isotope" in view.card.none_label.text()
    before = window.larmor_panel.nuclide()
    assert window.larmor_panel.nuclide() == before       # nothing to pick
    view.card.close_card()


def test_card_frequencies_follow_the_calculator_field(window):
    view = window.periodic_table_view
    window.larmor_panel.set_field(14.1)
    view.open_element("Al")
    assert view.card.table.item(0, 6).text() == f"{pt.larmor_frequency('27Al', 14.1):.4f}"
    window.larmor_panel.set_field(18.8)
    assert view.card.table.item(0, 6).text() == f"{pt.larmor_frequency('27Al', 18.8):.4f}"


def test_spin_legend_filters_by_any_isotope_of_that_spin(window):
    view = window.periodic_table_view
    view.legend_buttons[0.5].click()
    assert not view.tiles["N"].dimmed          # 15N is spin-1/2 though 14N dominates
    assert view.tiles["Al"].dimmed
    assert view.tiles["Ar"].dimmed
    view.legend_buttons[0.5].click()
    assert not any(t.dimmed for t in view.tiles.values())
    view.no_nmr_button.click()
    assert not view.tiles["Ar"].dimmed and view.tiles["H"].dimmed
    view.no_nmr_button.click()
    assert not any(t.dimmed for t in view.tiles.values())


def test_tile_modes(window):
    view = window.periodic_table_view
    window.larmor_panel.set_field(9.4)
    view.set_tile_mode("larmor")
    assert view.tiles["H"].bottom_text == f"{pt.larmor_frequency('1H', 9.4):.2f}"
    assert view.tiles["Ar"].bottom_text == "—"
    window.larmor_panel.set_field(11.7)          # follows the field live
    assert view.tiles["C"].bottom_text == f"{pt.larmor_frequency('13C', 11.7):.2f}"
    view.set_tile_mode("isotope")
    assert view.tiles["Al"].bottom_text == "²⁷Al 5/2"
    view.set_tile_mode("mass")
    assert view.tiles["Al"].bottom_text == "26.982"


# -- the calculator on its own -----------------------------------------------------

@pytest.fixture
def calc(qapp):
    return LarmorCalculatorPanel()


def test_calculator_defaults_to_1h_at_400(calc):
    assert calc.nuclide() == "1H"
    assert calc.proton_spin.value() == pytest.approx(400.0)
    assert calc.preset_combo.currentText() == "400 MHz"
    assert calc.larmor_MHz() == pytest.approx(400.0, abs=1e-4)


def test_field_proton_and_preset_stay_in_step(calc):
    calc.preset_combo.setCurrentIndex(calc.preset_combo.findData(600.0))
    assert calc.proton_spin.value() == pytest.approx(600.0)
    assert calc.field() == pytest.approx(pt.proton_frequency_to_field(600.0), abs=1e-6)
    calc.field_spin.setValue(14.1)
    assert calc.proton_spin.value() == pytest.approx(pt.field_to_proton_frequency(14.1), abs=1e-3)
    assert calc.preset_combo.currentText() == "Custom"
    calc.proton_spin.setValue(800.0)
    assert calc.preset_combo.currentText() == "800 MHz"


def test_result_and_working_are_the_notebook_numbers(calc):
    assert calc.set_nucleus("Al-27")
    calc.set_field(9.4)
    assert calc.result_label.text() == f"ν₀(27Al) = {pt.larmor_frequency('27Al', 9.4):.4f} MHz"
    assert "11.103079 MHz/T" in calc.working_label.text()
    assert pt.larmor_working("27Al", 9.4) in calc.result_text()


def test_unknown_nucleus_is_refused_and_the_last_one_kept(calc):
    calc.set_nucleus("13C")
    assert not calc.set_nucleus("12C")
    calc.nucleus_combo.setEditText("banana")
    calc._on_nucleus_edited()
    assert calc.nuclide() == "13C"
    assert "not an NMR-active nuclide" in calc.properties_label.text()


def test_typed_nucleus_is_accepted(calc):
    calc.nucleus_combo.setEditText("195pt")
    calc._on_nucleus_edited()
    assert calc.nuclide() == "195Pt"
    calc.nucleus_combo.setEditText("aluminium")      # an element: its principal isotope
    calc._on_nucleus_edited()
    assert calc.nuclide() == "27Al"
    assert not calc.set_nucleus("argon")             # no NMR isotope: refused


def test_all_nuclei_table_is_sorted_and_selects_the_nucleus(calc):
    calc.set_nucleus("31P")
    freqs = [float(calc.all_table.item(r, 3).text()) for r in range(calc.all_table.rowCount())]
    assert freqs == sorted(freqs, reverse=True)
    assert calc.all_table.rowCount() == len(pt.isotopes())
    selected = calc.all_table.selectedItems()
    assert selected and selected[0].data(Qt.ItemDataRole.UserRole) == "31P"
    calc.all_table.cellClicked.emit(0, 0)
    assert calc.nuclide() == calc.all_table.item(0, 0).data(Qt.ItemDataRole.UserRole)


def test_field_from_spectrum(qapp):
    calc = LarmorCalculatorPanel(spectrum_field=lambda: None)
    assert not calc.from_spectrum_button.isEnabled()
    assert not calc.use_spectrum_field()
    calc = LarmorCalculatorPanel(spectrum_field=lambda: (11.7434, "13C at 125.76 MHz"))
    assert calc.from_spectrum_button.isEnabled()
    assert calc.use_spectrum_field()
    assert calc.field() == pytest.approx(11.7434)


def test_main_window_field_from_selected_spectrum(window, monkeypatch):
    class Meta:
        bf1 = 100.6 ; sfo1 = 100.61 ; nucleus = "13C"

    class Spec:
        metadata = Meta()

    monkeypatch.setattr(window, "_selected_index", lambda: 0)
    monkeypatch.setattr(window.session, "spectra", [Spec()])
    B0, text = window._selected_spectrum_field()
    assert B0 == pytest.approx(100.6 / pt.isotope("13C").gamma_MHz_per_T)
    assert "13C" in text
    Meta.nucleus = "Xx"
    assert window._selected_spectrum_field() is None
