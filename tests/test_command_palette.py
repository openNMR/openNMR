"""Command palette / "Where is…?" (Ctrl+H, opennmr.gui.command_palette).

Three layers:

* the matching (``normalise``, ``score``, ``rank``) -- no windows at all;
* the index, ``reveal`` and the pop-up on a small stand-in window whose
  contents are known exactly;
* the real ``MainWindow``: the menu entries and, the important one,
  **reachability** -- every section and every menu action in the real
  window must be found by the palette, so a feature the palette cannot
  see fails the suite instead of going quietly undiscoverable.
"""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QGroupBox, QLabel, QMainWindow, QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from opennmr.gui.command_palette import (
    PALETTE_KEY, CommandPalette, Entry, build_index, normalise, rank, reveal, score,
)
from opennmr.gui.shortcut_sheet import iter_menu_actions
from opennmr.gui.widgets.collapsible_section import CollapsibleSection

from test_gui import qapp, _make_1d_dataset  # noqa: F401 -- fixture


# ------------------------------------------------------------------ matching

def test_normalise_treats_british_and_american_spellings_alike():
    assert normalise("Apodisation") == normalise("apodization")
    assert normalise("Colour map") == normalise("color map")
    assert normalise("Normalise") == normalise("NORMALIZE")
    assert normalise("Greyscale") == normalise("grayscale")


def test_normalise_drops_mnemonics_punctuation_accents_and_subscripts():
    assert normalise("&Open…") == "open"
    assert normalise("T₁ (s)") == "t1 s"
    assert normalise("Phase – PH0") == "phase ph0"
    assert normalise("Zéro-fill") == "zero fill"


def _e(kind, title, path="", keywords=""):
    return Entry(kind, title, path, keywords).prepare()


def test_a_title_word_beats_a_prefix_beats_a_substring_beats_a_tooltip():
    entries = [_e("Control", "Baseline correction"), _e("Control", "Baselines list"),
               _e("Control", "Rebaseline"), _e("Control", "Other", keywords="applies a baseline")]
    assert [e.title for e in rank("baseline", entries)] == [
        "Baseline correction", "Baselines list", "Rebaseline", "Other"]


def test_every_word_must_match():
    entries = [_e("Control", "Line broadening", "Process ▸ Window"), _e("Control", "Phase", "Process")]
    assert [e.title for e in rank("line process", entries)] == ["Line broadening"]
    assert rank("line zebra", entries) == []


def test_the_path_is_searched_too():
    entries = [_e("Control", "Degree", "Process ▸ Baseline")]
    assert rank("baseline degree", entries)


def test_letters_in_order_find_a_title_as_a_last_resort():
    entries = [_e("Control", "Autophase")]
    assert rank("atph", entries) and rank("zzz", entries) == []


def test_aliases_map_what_people_say_to_what_the_program_says():
    entries = [_e("Section", "Relaxation"), _e("Control", "Line broadening"),
               _e("Control", "Export figure"), _e("Control", "Unrelated")]
    assert [e.title for e in rank("t1", entries)] == ["Relaxation"]
    assert rank("lb", entries)[0].title == "Line broadening"
    assert "Export figure" in [e.title for e in rank("png", entries)]
    assert "Line broadening" in [e.title for e in rank("apodisation", entries)]


def test_the_noise_alias_survives_the_respelling():
    entries = [_e("Control", "Signal-to-noise"), _e("Control", "Other")]
    assert [e.title for e in rank("snr", entries)] == ["Signal-to-noise"]


def test_actions_and_panels_outrank_a_notebook_function_of_the_same_name():
    entries = [_e("Notebook", "Export"), _e("Action", "Export")]
    assert [e.kind for e in rank("export", entries)] == ["Action", "Notebook"]


def test_a_blank_query_matches_nothing_and_the_limit_is_respected():
    entries = [_e("Control", f"Phase {i}") for i in range(100)]
    assert rank("", entries) == [] and score("", entries[0]) == 0
    assert len(rank("phase", entries, limit=7)) == 7


# ------------------------------------------------------- the stand-in window

class _Window(QMainWindow):
    """A window with the parts the palette reads: menus, ``side_tabs``,
    and a status bar."""

    def __init__(self):
        super().__init__()
        self.ran = []
        self._ui_settings = {}
        self.saved = 0
        file_menu = self.menuBar().addMenu("&File")
        self.open_action = file_menu.addAction("&Open…")
        self.open_action.setShortcut(QKeySequence("Ctrl+O"))
        self.open_action.setToolTip("Load an experiment from disk")
        self.open_action.triggered.connect(lambda: self.ran.append("open"))
        self.gone_action = file_menu.addAction("Close all")
        self.gone_action.setEnabled(False)
        sub = file_menu.addMenu("Recent")
        self.recent_action = sub.addAction("Example.fid")
        self.side_tabs = QTabWidget()
        self.setCentralWidget(self.side_tabs)
        # Tile 0: Open
        self.side_tabs.addTab(QWidget(), "Open")
        # Tile 1: Process = scroll area > Baseline > Manual regions > button
        self.inner_button = QPushButton("Auto baseline")
        self.inner_button.setToolTip("Fit a baseline from the spectrum edges")
        inner = CollapsibleSection("Manual regions", self.inner_button)
        self.outer = CollapsibleSection("Baseline", inner)
        self.inner = inner
        # a labelled control with a buddy, and a group box with a sub-tab widget
        self.degree_label = QLabel("Polynomial degree:")
        spin_holder = QPushButton("degree stand-in")
        self.degree_label.setBuddy(spin_holder)
        self.sub_tabs = QTabWidget()
        self.tab_button = QPushButton("Second tab button")
        self.sub_tabs.addTab(QWidget(), "First")
        self.sub_tabs.addTab(self.tab_button, "Second")
        box = QGroupBox("Fit options")
        box_layout = QVBoxLayout(box)
        box_layout.addWidget(self.sub_tabs)
        body = QWidget()
        layout = QVBoxLayout(body)
        for widget in (self.outer, self.degree_label, spin_holder, box):
            layout.addWidget(widget)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        self.side_tabs.addTab(scroll, "Process")
        self.resize(700, 500)


@pytest.fixture
def window(qapp):
    win = _Window()
    win._save_ui_settings = lambda: setattr(win, "saved", win.saved + 1)
    yield win
    win.close()


def _titles(entries, kind=None):
    return [e.title for e in entries if kind is None or e.kind == kind]


def test_the_index_has_menu_actions_panels_sections_tabs_and_controls(window):
    index = build_index(window)
    assert "Open…" in _titles(index, "Action")
    assert "Example.fid" in _titles(index, "Action")
    assert {"Open panel", "Process panel"} <= set(_titles(index, "Panel"))
    assert {"Baseline", "Manual regions"} <= set(_titles(index, "Section"))
    assert {"First", "Second"} <= set(_titles(index, "Tab"))
    assert "Fit options" in _titles(index, "Section")
    controls = _titles(index, "Control")
    assert "Auto baseline" in controls and "Polynomial degree" in controls
    assert "Second tab button" in controls


def test_menu_entries_carry_their_menu_path_shortcut_and_tooltip(window):
    entry = next(e for e in build_index(window) if e.title == "Open…")
    assert entry.path == "File" and entry.shortcut == QKeySequence("Ctrl+O").toString(
        QKeySequence.SequenceFormat.NativeText)
    assert "Load an experiment" in entry.keywords
    assert next(e for e in build_index(window) if e.title == "Example.fid").path == "File ▸ Recent"


def test_controls_know_which_sections_they_are_inside(window):
    entry = next(e for e in build_index(window) if e.title == "Auto baseline")
    assert entry.path == "Process ▸ Baseline ▸ Manual regions"
    tab_button = next(e for e in build_index(window) if e.title == "Second tab button")
    assert tab_button.path == "Process ▸ Fit options ▸ Second"


def test_the_buddy_of_a_label_is_what_it_takes_you_to(window):
    entry = next(e for e in build_index(window) if e.title == "Polynomial degree")
    assert entry.action is not None
    assert window.degree_label.buddy() is not None


def test_a_disabled_menu_action_is_listed_but_marked_unavailable(window):
    entry = next(e for e in build_index(window) if e.title == "Close all")
    assert entry.enabled is False


def test_an_excluded_action_is_not_offered(window):
    window.open_action.setProperty("palette_exclude", True)
    assert "Open…" not in _titles(build_index(window))


def test_notebook_functions_are_searchable_and_copy_their_name(window):
    entry = next(e for e in build_index(window) if e.title == "export_experiment_notebook")
    assert entry.kind == "Notebook" and "notebook" in entry.keywords.lower()
    entry.action()
    assert QGuiApplication.clipboard().text() == "opennmr.notebook.export_experiment_notebook"
    assert "Copied" in window.statusBar().currentMessage()


# ------------------------------------------------------------------- reveal

def test_reveal_opens_every_section_on_the_way_and_switches_panel(window):
    assert window.side_tabs.currentIndex() == 0
    assert not window.outer.isExpanded() and not window.inner.isExpanded()
    assert reveal(window, window.inner_button, flash=False, defer=False) is True
    assert window.side_tabs.currentIndex() == 1
    assert window.outer.isExpanded() and window.inner.isExpanded()
    assert window.inner_button.isVisibleTo(window)


def test_reveal_switches_the_inner_tab_to_the_page_with_the_control(window):
    assert window.sub_tabs.currentIndex() == 0
    reveal(window, window.tab_button, flash=False, defer=False)
    assert window.sub_tabs.currentIndex() == 1


def test_reveal_of_a_section_opens_that_section_itself(window):
    reveal(window, window.outer, flash=False, defer=False)
    assert window.outer.isExpanded() and not window.inner.isExpanded()


def test_reveal_says_so_when_the_control_is_hidden_for_this_selection(window):
    window.inner_button.hide()
    assert reveal(window, window.inner_button, flash=False, defer=False) is False
    assert "hidden" in window.statusBar().currentMessage()


def test_reveal_draws_and_then_removes_the_outline(window):
    window.show()
    before = len(window.findChildren(QWidget))
    reveal(window, window.inner_button, flash=True, defer=False)
    assert len(window.findChildren(QWidget)) == before + 1       # the outline frame


# --------------------------------------------------------------- the pop-up

def test_typing_ranks_the_results_and_enter_goes_there(window):
    palette = CommandPalette(window)
    palette.open_palette()
    palette.edit.setText("auto baseline")
    assert palette.shown()[0].title == "Auto baseline"
    assert palette.results.count() == len(palette.shown())
    QTest.keyClick(palette.edit, Qt.Key.Key_Return)
    assert not palette.isVisible()
    QTest.qWait(120)                                    # the deferred scroll/outline
    assert window.side_tabs.currentIndex() == 1 and window.inner.isExpanded()


def test_up_and_down_move_through_the_results_and_wrap(window):
    palette = CommandPalette(window)
    palette.edit.setText("baseline")
    count = palette.results.count()
    assert count >= 2 and palette.results.currentRow() == 0
    QTest.keyClick(palette.edit, Qt.Key.Key_Down)
    assert palette.results.currentRow() == 1
    QTest.keyClick(palette.edit, Qt.Key.Key_Up)
    QTest.keyClick(palette.edit, Qt.Key.Key_Up)
    assert palette.results.currentRow() == count - 1


def test_choosing_a_menu_action_runs_it(window):
    palette = CommandPalette(window)
    palette.edit.setText("open")
    entry = palette.activate(0)
    assert entry.title == "Open…" and window.ran == ["open"]


def test_a_disabled_action_does_not_run_and_the_status_bar_says_why(window):
    palette = CommandPalette(window)
    palette.edit.setText("close all")
    entry = palette.activate(0)
    assert entry.title == "Close all" and "not available" in window.statusBar().currentMessage()


def test_recent_choices_are_remembered_and_shown_for_an_empty_query(window):
    first = CommandPalette(window)
    assert first.shown() == []                                   # nothing yet
    first.edit.setText("open")
    first.activate(0)
    assert window.saved == 1 and window._ui_settings["palette_recent"]
    again = CommandPalette(window)
    assert [e.title for e in again.shown()] == ["Open…"]


def test_recent_choices_are_capped_and_deduplicated(window):
    palette = CommandPalette(window)
    for title in ("Open…", "Example.fid", "Open…"):
        palette.edit.setText(title.rstrip("…"))
        palette.activate(0)
    keys = window._ui_settings["palette_recent"]
    assert len(keys) == len(set(keys)) <= 8


def test_an_action_that_raises_does_not_take_the_window_down(window):
    def boom():
        raise RuntimeError("boom")
    palette = CommandPalette(window, entries=[Entry("Action", "Explode", action=boom).prepare()])
    palette.edit.setText("explode")
    palette.activate(0)
    assert "boom" in window.statusBar().currentMessage()


def test_nothing_matching_says_so(window):
    palette = CommandPalette(window)
    palette.edit.setText("qqqqzzzz")
    assert palette.shown() == [] and "Nothing matches" in palette.hint.text()


# --------------------------------------------------- the real main window

@pytest.fixture
def main(qapp):
    """A real window per test: conftest deletes every top-level widget a
    test leaves behind, so one cannot be shared across tests."""
    from opennmr.gui.main_window import MainWindow
    return MainWindow()


def test_help_menu_offers_the_palette_and_the_shortcut_sheet(main):
    assert main.palette_action.shortcut() == QKeySequence(PALETTE_KEY)
    assert main.shortcuts_action.shortcut() == QKeySequence("F1")
    help_titles = [a.text().replace("&", "") for a in main.help_menu.actions()]
    assert "Search Commands…" in help_titles and "Keyboard Shortcuts…" in help_titles


def test_ctrl_h_is_not_taken_by_another_action(main):
    holders = [a for _, a in iter_menu_actions(main.menuBar().actions())
               if any(s == QKeySequence(PALETTE_KEY) for s in a.shortcuts())]
    assert holders == [main.palette_action]


def test_show_command_palette_opens_and_a_second_call_closes_it(main):
    palette = main.show_command_palette()
    assert palette is not None and palette.isVisible()
    assert main.show_command_palette() is None
    assert not palette.isVisible()


def test_every_section_in_every_panel_is_reachable_from_the_palette(main):
    index = build_index(main)
    found = {(e.kind, e.title) for e in index}
    missing = []
    for i in range(main.side_tabs.count()):
        for section in main.side_tabs.widget(i).findChildren(CollapsibleSection):
            title = " ".join(section.title().split())
            if title and ("Section", title) not in found:
                missing.append((main.side_tabs.tabText(i), title))
    assert missing == []


def test_every_menu_action_is_reachable_from_the_palette(main):
    titles = {e.title for e in build_index(main) if e.kind == "Action"}
    missing = [" ▸ ".join(path + (a.text(),)) for path, a in iter_menu_actions(main.menuBar().actions())
               if not a.property("palette_exclude")
               and a.text().replace("&", "").strip() not in titles]
    assert missing == []


def test_every_panel_tile_is_reachable_from_the_palette(main):
    panels = {e.title for e in build_index(main) if e.kind == "Panel"}
    for i in range(main.side_tabs.count()):
        if main.side_tabs.isTabVisible(i):
            assert f"{main.side_tabs.tabText(i)} panel" in panels


def test_searching_for_baseline_finds_it_and_choosing_it_opens_the_section(main):
    results = rank("baseline", build_index(main))
    assert results, "the palette cannot find 'baseline' anywhere"
    section = next((e for e in results if e.kind == "Section"), None)
    assert section is not None
    section.action()
    QTest.qWait(120)
    owner = [c for i in range(main.side_tabs.count())
             for c in main.side_tabs.widget(i).findChildren(CollapsibleSection)
             if " ".join(c.title().split()) == section.title]
    assert any(c.isExpanded() for c in owner)


def test_the_index_builds_quickly_enough_to_open_on_every_keypress_of_ctrl_h(main):
    import time
    start = time.perf_counter()
    build_index(main)
    assert time.perf_counter() - start < 3.0
