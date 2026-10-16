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


def _fetch_facts():
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "packaging" / "fetch_facts.py"
    spec = importlib.util.spec_from_file_location("fetch_facts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetch_facts_reads_the_fact_column_of_every_sheet():
    import io
    import zipfile
    ff = _fetch_facts()
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    shared = (f'<sst {ns}><si><t>Episode</t></si><si><t>Fact</t></si>'
              '<si><r><t>Octopuses have </t></r><r><t>three hearts.</t></r></si></sst>')
    def sheet(rows):
        body = "".join(f'<row r="{i + 1}">{r}</row>' for i, r in enumerate(rows))
        return f'<worksheet {ns}><sheetData>{body}</sheetData></worksheet>'
    s1 = sheet(['<c r="B1" t="s"><v>0</v></c><c r="D1" t="s"><v>1</v></c>',
                '<c r="B2"><v>599</v></c><c r="D2" t="s"><v>2</v></c>'])
    s2 = sheet(['<c r="D1" t="inlineStr"><is><t>Fact</t></is></c>',
                '<c r="D2" t="inlineStr"><is><t>A snail can sleep for three years.</t></is></c>'])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/sharedStrings.xml", shared)
        z.writestr("xl/worksheets/sheet1.xml", s1)
        z.writestr("xl/worksheets/sheet2.xml", s2)
    assert ff.facts_from_xlsx(buf.getvalue()) == ["Octopuses have three hearts.",
                                                   "A snail can sleep for three years."]


def test_fetch_facts_reads_the_claim_after_each_qi_label_not_the_citation():
    ff = _fetch_facts()
    page = ("<h3>Page 2, Position 2</h3><p>The &quot;Eiffel Tower&quot; grows in summer.</p>"
            "<p><a href='https://sources.qi.com/x'>https://example.org/eiffel</a></p>"
            "<div>Page 2, Position 3: Bananas are berries but strawberries are not.</div>"
            "<p>www.example.org/bananas</p>")
    assert ff.facts_from_qi(page) == ['The "Eiffel Tower" grows in summer.',
                                      "Bananas are berries but strawberries are not."]
    assert ff.tidy(["a", "Same fact here, twice.", "same fact here, twice."]) == \
        ["Same fact here, twice."]


def test_fetch_facts_output_is_what_the_app_reads(tmp_path, monkeypatch):
    ff = _fetch_facts()
    out = tmp_path / "facts.dat"
    ff.write(["One fact for the list.", "Two facts for the list."], str(out))
    assert fact_box.builtin_facts(str(out)) == ["One fact for the list.", "Two facts for the list."]


def test_facts_dat_round_trips_through_an_editable_text_file(tmp_path):
    ff = _fetch_facts()
    dat, txt = tmp_path / "facts.dat", tmp_path / "facts.txt"
    ff.write(["First fact, long enough.", "Second fact, long enough."], str(dat))
    assert ff.export_text(str(txt), str(dat)) == 2
    assert txt.read_text(encoding="utf-8").splitlines() == ["First fact, long enough.",
                                                           "Second fact, long enough."]
    txt.write_text("# a comment\nFirst fact, long enough.\n\nA new fact I typed in.\n",
                   encoding="utf-8")
    assert ff.import_text(str(txt), str(dat)) == 2
    assert fact_box.builtin_facts(str(dat)) == ["First fact, long enough.", "A new fact I typed in."]
