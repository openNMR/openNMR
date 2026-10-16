"""Cmd/Ctrl+E: a random fact from the user's own list (~/.opennmr/facts.txt),
reachable only by the shortcut (explicit user request, 2026-10-16)."""
import random

from PySide6.QtGui import QKeySequence

from opennmr.gui import fact_box
from opennmr.gui.main_window import MainWindow

from test_gui import qapp  # noqa: F401 -- fixture


def test_reading_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "facts.txt"
    path.write_text("# header\n\nOne fact\n  Two fact  \n#not this\n", encoding="utf-8")
    assert fact_box.read_facts(str(path)) == ["One fact", "Two fact"]
    assert fact_box.read_facts(str(tmp_path / "missing.txt")) == []


def test_never_the_same_fact_twice_running():
    rng = random.Random(0)
    facts = ["a", "b"]
    seen = [fact_box.pick_fact(facts, "a", rng) for _ in range(20)]
    assert set(seen) == {"b"}
    assert fact_box.pick_fact(["only"], "only", rng) == "only"
    assert fact_box.pick_fact([], None, rng) is None


def test_ctrl_e_shows_a_fact_and_is_in_no_menu(qapp, tmp_path, monkeypatch):
    path = tmp_path / "facts.txt"
    monkeypatch.setenv("OPENNMR_FACTS_FILE", str(path))
    window = MainWindow()
    action = window._fact_action
    assert action.shortcut() == QKeySequence("Ctrl+E")
    # only the shortcut: no menu, toolbar or button carries it
    for menu_action in window.menuBar().actions():
        menu = menu_action.menu()
        assert menu is None or action not in menu.actions()
    assert action.text() == "" and action in window.actions()
    # no file yet: one is created, and the box says where
    action.trigger()
    assert path.exists() and str(path) in window._fact_dialog.fact_label.text()
    path.write_text("Fact one\nFact two\n", encoding="utf-8")
    window._fact_dialog.another_button.click()
    first = window._fact_dialog.current
    assert first in ("Fact one", "Fact two")
    window._fact_dialog.another_button.click()
    assert window._fact_dialog.current != first
    assert window._fact_dialog.isVisible()
