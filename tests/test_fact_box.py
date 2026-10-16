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
    # nothing built in and no file of the user's: the box says where to add some
    monkeypatch.setattr(fact_box, "BUILTIN_PATH", str(tmp_path / "missing.dat"))
    action.trigger()
    assert str(path) in window._fact_dialog.fact_label.text()
    path.write_text("Fact one\nFact two\n", encoding="utf-8")
    window._fact_dialog.another_button.click()
    first = window._fact_dialog.current
    assert first in ("Fact one", "Fact two")
    window._fact_dialog.another_button.click()
    assert window._fact_dialog.current != first
    assert window._fact_dialog.isVisible()


def test_built_in_list_is_compressed_and_merged_with_the_users(tmp_path, monkeypatch):
    import sys
    import zlib
    dat = tmp_path / "facts.dat"
    dat.write_bytes(zlib.compress("Built-in one\nBuilt-in two\n".encode("utf-8")))
    assert b"Built-in" not in dat.read_bytes()             # not plain text in an install
    monkeypatch.setattr(fact_box, "BUILTIN_PATH", str(dat))
    user = tmp_path / "mine.txt"
    user.write_text("Mine\nBuilt-in one\n", encoding="utf-8")
    monkeypatch.setenv("OPENNMR_FACTS_FILE", str(user))
    assert fact_box.all_facts() == ["Built-in one", "Built-in two", "Mine"]


def _edit_facts():
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "packaging" / "edit_facts.py"
    spec = importlib.util.spec_from_file_location("edit_facts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




def test_edit_facts_output_is_what_the_app_reads(tmp_path, monkeypatch):
    ff = _edit_facts()
    out = tmp_path / "facts.dat"
    ff.write(["One fact for the list.", "Two facts for the list."], str(out))
    assert fact_box.builtin_facts(str(out)) == ["One fact for the list.", "Two facts for the list."]


def test_facts_dat_round_trips_through_an_editable_text_file(tmp_path):
    ff = _edit_facts()
    dat, txt = tmp_path / "facts.dat", tmp_path / "facts.txt"
    ff.write(["First fact, long enough.", "Second fact, long enough."], str(dat))
    assert ff.export_text(str(txt), str(dat)) == 2
    assert txt.read_text(encoding="utf-8").splitlines() == ["First fact, long enough.",
                                                           "Second fact, long enough."]
    txt.write_text("# a comment\nFirst fact, long enough.\n\nA new fact I typed in.\n",
                   encoding="utf-8")
    assert ff.import_text(str(txt), str(dat)) == 2
    assert fact_box.builtin_facts(str(dat)) == ["First fact, long enough.", "A new fact I typed in."]
