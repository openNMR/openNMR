"""2026-10-20 round 2: the Welcome wizard (first run, Help menu, Skip, applying
the settings) and Integrate ▸ Integration Regions ▸ Remove Region."""

import json

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices

from opennmr.gui import main_window as mw
from opennmr.gui.main_window import MainWindow
from opennmr.gui.welcome_wizard import WEBSITE_URL, WelcomeWizard

from test_gui import qapp, _load_spectrum_directly  # noqa: F401
from test_integrate_tile_plot_options_regularisation import _one_d  # noqa: F401


def _saved():
    with open(MainWindow._UI_SETTINGS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


# ---- when it appears ---------------------------------------------------------------

def test_constructing_a_window_never_opens_the_wizard(qapp):  # noqa: F811
    window = MainWindow()
    assert window.welcome_pending()
    assert getattr(window, "_welcome_dialog", None) is None


def test_first_run_opens_once_then_never_again(qapp):  # noqa: F811
    window = MainWindow()
    assert window.maybe_show_welcome() is True
    assert isinstance(window._welcome_dialog, WelcomeWizard)
    assert _saved()["welcome_seen_version"] == mw._version.__version__
    window._welcome_dialog.close()
    assert window.maybe_show_welcome() is False             # same session
    again = MainWindow()                                    # "next launch"
    assert not again.welcome_pending()
    assert again.maybe_show_welcome() is False


def test_a_new_version_shows_it_again(qapp, monkeypatch):  # noqa: F811
    window = MainWindow()
    window.maybe_show_welcome()
    window._welcome_dialog.close()
    monkeypatch.setattr(mw._version, "__version__", "99.0.0")
    assert MainWindow().welcome_pending()


def test_help_menu_reopens_it_and_links_to_the_website(qapp, monkeypatch):  # noqa: F811
    window = MainWindow()
    window.maybe_show_welcome()
    window._welcome_dialog.close()
    titles = [a.text() for a in window.menuBar().actions()]
    assert any("Help" in t for t in titles)
    first = window._welcome_dialog
    window.welcome_action.trigger()                          # seen already: still opens
    assert window._welcome_dialog is not first
    assert window._welcome_dialog.isVisible()
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    window.website_action.trigger()
    assert opened == [WEBSITE_URL] and WEBSITE_URL == "https://opennmr.github.io/"


# ---- what it does ------------------------------------------------------------------------

def test_pages_and_buttons(qapp):  # noqa: F811
    window = MainWindow()
    wizard = WelcomeWizard(window)
    assert wizard.page_count == 7 and wizard.current_page() == 0
    assert not wizard.back_button.isEnabled()
    assert wizard.next_button.text() == "Next"
    for _ in range(wizard.page_count - 1):
        wizard.go_next()
    assert wizard.current_page() == wizard.page_count - 1
    assert wizard.next_button.text() == "Apply and start"
    wizard.go_back()
    assert wizard.current_page() == wizard.page_count - 2


def test_skip_changes_nothing(qapp):  # noqa: F811
    window = MainWindow()
    before = (window.ui_font_size_spin.value(), window.ui_theme_combo.currentData(),
              window.drag_mode_combo.currentData(),
              {k: b.isChecked() for k, b in window.section_shown_checkboxes.items()})
    wizard = WelcomeWizard(window)
    wizard.font_spin.setValue(wizard.font_spin.value() + 2)     # chosen, then skipped
    wizard.section_checks["dnp"].setChecked(False)
    wizard.skip()
    assert not wizard.applied
    after = (window.ui_font_size_spin.value(), window.ui_theme_combo.currentData(),
             window.drag_mode_combo.currentData(),
             {k: b.isChecked() for k, b in window.section_shown_checkboxes.items()})
    assert after == before


def test_finishing_applies_and_remembers_the_choices(qapp):  # noqa: F811
    window = MainWindow()
    wizard = WelcomeWizard(window)
    wizard.font_spin.setValue(wizard.font_spin.value() + 1)
    wizard.theme_combo.setCurrentIndex(max(0, wizard.theme_combo.findData("dark")))
    wizard.double_click_check.setChecked(not window.double_click_reset_checkbox.isChecked())
    wizard.section_checks["dnp"].setChecked(False)
    wizard.iterations_spin.setValue(3000)
    wizard.export_unit_combo.setCurrentIndex(wizard.export_unit_combo.findData("cm"))
    expected_font = wizard.font_spin.value()
    expected_double = wizard.double_click_check.isChecked()
    while wizard.current_page() < wizard.page_count - 1:
        wizard.go_next()
    wizard.go_next()                                           # Apply and start
    assert wizard.applied
    assert window.ui_font_size_spin.value() == expected_font
    assert window.double_click_reset_checkbox.isChecked() == expected_double
    assert not window.section_shown_checkboxes["dnp"].isChecked()
    assert window.max_iterations_spin.value() == 3000
    assert window.export_unit_combo.currentData() == "cm"
    saved = _saved()
    assert "dnp" in saved["hidden_sections"]
    assert saved["export_unit"] == "cm"


def test_wizard_starts_from_the_current_settings(qapp):  # noqa: F811
    window = MainWindow()
    window.section_shown_checkboxes["enmr"].setChecked(False)
    window.ui_font_size_spin.setValue(window.ui_font_size_spin.value() + 1)
    wizard = WelcomeWizard(window)
    assert not wizard.section_checks["enmr"].isChecked()
    assert wizard.font_spin.value() == window.ui_font_size_spin.value()


# ---- Integration Regions ▸ Remove Region ------------------------------------------------------

def _regions_window(n=3):
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    for k in range(n):
        window._add_region_row(float(k), float(k) + 0.5)
    window.region_table.clearSelection()
    window.region_table.setCurrentCell(-1, -1)
    return window


def test_remove_region_removes_the_clicked_row(qapp):  # noqa: F811
    window = _regions_window()
    window.region_table.setCurrentCell(1, 0)
    window.remove_region_button.click()
    assert window.region_table.rowCount() == 2
    assert [window.region_table.item(r, 0).text() for r in range(2)] == ["0", "2"]


def test_remove_region_with_no_current_row_uses_ticks_or_the_only_row(qapp):  # noqa: F811
    window = _regions_window(3)
    window.remove_region_button.click()                        # nothing chosen: nothing removed
    assert window.region_table.rowCount() == 3
    assert "Click the region" in window.integration_results_label.text()
    window.region_table.item(2, 0).setCheckState(Qt.CheckState.Checked)
    window.remove_region_button.click()                        # the ticked one goes
    assert window.region_table.rowCount() == 2
    single = _regions_window(1)
    single.remove_region_button.click()                        # a table of one needs no selecting
    assert single.region_table.rowCount() == 0


def test_remove_region_removes_every_selected_row(qapp):  # noqa: F811
    window = _regions_window(4)
    window.region_table.selectRow(1)
    window.region_table.selectRow(3) if False else None
    sel = window.region_table.selectionModel()
    from PySide6.QtCore import QItemSelectionModel
    sel.select(window.region_table.model().index(3, 0),
               QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
    window.remove_region_button.click()
    assert window.region_table.rowCount() == 2


# ---- type of work (2026-10-24) -----------------------------------------------------------

def test_work_type_page_is_second(qapp):  # noqa: F811
    window = MainWindow()
    wizard = WelcomeWizard(window)
    wizard.go_next()
    assert wizard.current_page() == 1
    assert set(wizard.work_type_checks) == {"solution", "solid", "dnp", "pnmr"}


def test_intersection_of_chosen_work_types():
    from opennmr.gui.welcome_wizard import sections_to_remove
    assert sections_to_remove([]) == set()
    assert sections_to_remove(["dnp"]) == {"pnmr", "enmr", "imaging"}
    assert sections_to_remove(["solution", "solid"]) == {"pnmr", "dnp"}
    assert sections_to_remove(["dnp", "pnmr"]) == {"enmr", "imaging"}
    assert sections_to_remove(["solution", "dnp"]) == {"pnmr"}


def test_choosing_work_types_sets_the_layout_page_and_can_be_overridden(qapp):  # noqa: F811
    window = MainWindow()
    wizard = WelcomeWizard(window)
    wizard.work_type_checks["solution"].setChecked(True)
    off = {k for k, b in wizard.section_checks.items() if not b.isChecked()}
    assert off == {"solid", "pnmr", "dnp", "czjzek", "cp_buildup", "offset_tracker",
                   "batch_pseudo2d", "bpp", "exchange", "motion", "advanced"}
    wizard.work_type_checks["solid"].setChecked(True)
    off = {k for k, b in wizard.section_checks.items() if not b.isChecked()}
    assert off == {"pnmr", "dnp"}
    wizard.section_checks["enmr"].setChecked(False)         # still the user's call
    while wizard.current_page() < wizard.page_count - 1:
        wizard.go_next()
    wizard.go_next()
    assert not window.section_shown_checkboxes["pnmr"].isChecked()
    assert not window.section_shown_checkboxes["enmr"].isChecked()
    assert window.section_shown_checkboxes["czjzek"].isChecked()
    # unticking every type returns the layout page to the starting state
    wizard2 = WelcomeWizard(window)
    before = {k: b.isChecked() for k, b in wizard2.section_checks.items()}
    wizard2.work_type_checks["dnp"].setChecked(True)
    wizard2.work_type_checks["dnp"].setChecked(False)
    assert {k: b.isChecked() for k, b in wizard2.section_checks.items()} == before


def test_skipping_after_choosing_a_work_type_changes_nothing(qapp):  # noqa: F811
    window = MainWindow()
    before = {k: b.isChecked() for k, b in window.section_shown_checkboxes.items()}
    wizard = WelcomeWizard(window)
    wizard.work_type_checks["pnmr"].setChecked(True)
    wizard.skip()
    assert {k: b.isChecked() for k, b in window.section_shown_checkboxes.items()} == before


# ---- more colour themes (2026-10-24) ---------------------------------------------------

def test_extra_themes_are_listed_and_apply(qapp):  # noqa: F811
    from opennmr.gui import style
    from PySide6.QtGui import QPalette
    extras = ("solarized_light", "solarized_dark", "nord", "midnight", "sepia", "high_contrast")
    assert set(extras) <= set(style.THEME_MODES)
    assert all(m in style.THEME_LABELS for m in style.THEME_MODES)
    assert style.resolve_mode("nord", qapp) == "dark" and style.resolve_mode("sepia", qapp) == "light"
    seen = set()
    for mode in extras:
        style.apply_theme(mode, qapp)
        pal = qapp.palette()
        window = pal.color(QPalette.ColorRole.Window).name()
        button = pal.color(QPalette.ColorRole.Button).name()
        assert window != button                              # buttons stay offset from the panel
        seen.add(window)
    assert len(seen) == len(extras)
    style.apply_theme("light", qapp)


def test_wizard_offers_the_extra_themes(qapp):  # noqa: F811
    window = MainWindow()
    wizard = WelcomeWizard(window)
    assert wizard.theme_combo.findData("nord") >= 0 and wizard.theme_combo.findData("sepia") >= 0


def test_solution_and_solid_sections_can_be_removed_from_fit_and_simulate(qapp):  # noqa: F811
    window = MainWindow()
    for key, names in (("solution", ("_solution_fit_section", "_solution_sim_section")),
                       ("solid", ("_combined_fit_section", "_combined_sim_section"))):
        assert window.OPTIONAL_SECTIONS[key][1] == names
        assert window.section_shown_checkboxes[key].isChecked()
        window.section_shown_checkboxes[key].setChecked(False)
        assert all(getattr(window, n).isHidden() for n in names)
        window.section_shown_checkboxes[key].setChecked(True)
        assert not any(getattr(window, n).isHidden() for n in names)


def test_wizard_solution_only_removes_solid_and_solid_only_removes_solution(qapp):  # noqa: F811
    from opennmr.gui.welcome_wizard import sections_to_remove
    assert "solid" in sections_to_remove(["solution"]) and "solution" not in sections_to_remove(["solution"])
    assert "solution" in sections_to_remove(["solid"]) and "solid" not in sections_to_remove(["solid"])
    # Both kinds of work keep both.
    assert not {"solution", "solid"} & sections_to_remove(["solution", "solid"])
    window = MainWindow()
    wizard = WelcomeWizard(window)
    wizard.work_type_checks["solution"].setChecked(True)
    while wizard.current_page() < wizard.page_count - 1:
        wizard.go_next()
    wizard.go_next()
    assert window._combined_fit_section.isHidden() and window._combined_sim_section.isHidden()
    assert not window._solution_fit_section.isHidden() and not window._solution_sim_section.isHidden()
    window2 = MainWindow()
    wizard2 = WelcomeWizard(window2)
    wizard2.work_type_checks["solid"].setChecked(True)
    while wizard2.current_page() < wizard2.page_count - 1:
        wizard2.go_next()
    wizard2.go_next()
    assert window2._solution_fit_section.isHidden() and window2._solution_sim_section.isHidden()
    assert not window2._combined_fit_section.isHidden()
