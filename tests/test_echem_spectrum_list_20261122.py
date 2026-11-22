"""2026-11-22: Apply ▸ Electrochemistry -- the panel's own spectrum list, filled from Open
experiments with Add / Remove, so every open experiment is available whatever is selected in the
Open experiments list; and "Type of data to plot" (was "Beside the cell")."""
from test_gui import qapp  # noqa: F401 -- qapp is a fixture
from test_gui_echem_20261120 import window  # noqa: F401 -- the fixture: 6 open 1D experiments


def _labels(widget):
    return [widget.item(i).text() for i in range(widget.count())]


def test_mode_choice_is_named_type_of_data_to_plot(window):
    form = window.echem_panel._source_form
    assert form.labelForField(window.echem_panel.mode_combo).text() == "Type of data to plot:"


def test_every_open_experiment_is_chosen_whatever_is_selected(window):
    p = window.echem_panel
    window.spectrum_list.clearSelection()
    window.spectrum_list.item(5).setSelected(True)      # only the newest is selected
    p.refresh()
    assert p.select_combo.currentData() == "list"
    chosen, _ = p._chosen()
    assert len(chosen) == 6
    assert _labels(p.plot_list) == [e["label"] for e in chosen]
    assert p.avail_list.count() == 0


def test_experiments_opened_later_join_an_untouched_list(window):
    p = window.echem_panel
    before = len(p._chosen()[0])
    from test_gui_echem_20261120 import _spec
    window._add_derived_spectrum(_spec(7, "/data/sample/99"), "exp 99")
    p.refresh()
    assert len(p._chosen()[0]) == before + 1


def test_remove_and_add_back(window):
    p = window.echem_panel
    keys = [p._key(e) for e in p._entries]
    p.remove_spectra(keys[:2])
    assert len(p._chosen()[0]) == 4 and p.avail_list.count() == 2
    p.add_spectra([keys[0]])
    chosen = [e["label"] for e in p._chosen()[0]]
    assert len(chosen) == 5 and chosen[-1] == p._entries[0]["label"]      # added at the end
    p.add_spectra(None)
    assert len(p._chosen()[0]) == 6 and p.avail_list.count() == 0


def test_clear_empties_the_list_but_keeps_experiments_open(window):
    p = window.echem_panel
    p.remove_spectra(None)
    assert p._chosen()[0] == [] and p.avail_list.count() == 6
    assert len(window.session) >= 6


def test_move_and_sort(window):
    p = window.echem_panel
    keys = [p._key(e) for e in p._entries]
    p.remove_spectra(None)
    p.add_spectra(list(reversed(keys)))
    first = _labels(p.plot_list)
    p.sort_spectra()
    assert _labels(p.plot_list) == list(reversed(first))
    p.plot_list.item(0).setSelected(True)
    p.move_spectra(1)
    assert _labels(p.plot_list)[1] == list(reversed(first))[0]


def test_a_closed_experiment_is_dropped_from_the_list_with_a_note(window):
    p = window.echem_panel
    keys = [p._key(e) for e in p._entries]
    p.remove_spectra(keys[:1])
    p._entries = [e for e in p._entries if p._key(e) != keys[1]]      # one is closed meanwhile
    chosen, notes = p._chosen()
    assert len(chosen) == 4 and any("no longer open" in n for n in notes)


def test_table_follows_the_list(window):
    p = window.echem_panel
    keys = [p._key(e) for e in p._entries]
    p.remove_spectra(keys[:3])
    assert p.ex_table.rowCount() == 3


def test_selected_rows_option_still_works(window):
    p = window.echem_panel
    window.spectrum_list.clearSelection()
    window.spectrum_list.item(1).setSelected(True)
    window.spectrum_list.item(3).setSelected(True)
    p.select_combo.setCurrentIndex(p.select_combo.findData("selected"))
    p.refresh()
    assert len(p._chosen()[0]) == 2
