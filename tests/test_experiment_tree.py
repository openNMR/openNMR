"""Open experiments as a tree of typed entries (explicit user request,
2026-11-13; Phase 3 step 2): Spectrum, Fit, Profile fit, Simulation, and a
multi-spectrum fit as a group; a fit is replotted by opening it."""

import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.gui.main_window import MainWindow
from opennmr.notebook import experiment_tree as et

from test_gui import qapp  # noqa: F401 -- fixture
from test_gui_fit_sim_state import _fit_quad, _quad_spectrum
from test_gui_round4_fits import _window_with_kept_fits


# ---------------------------------------------------------------------------
# The tree itself (no GUI)
# ---------------------------------------------------------------------------

def _entry(row, label, raw=None):
    return {"row": row, "label": label, "entry_id": f"s-e{row + 1}", "raw": raw or {}}


def _shape(nodes):
    return [(n["label"], n["kind"], n["row"], n["virtual"], d) for n, d in et.iter_nodes(nodes)]


def test_plain_spectra_and_simulations_are_top_level_in_order():
    nodes = et.build_experiment_tree([
        _entry(0, "a"), _entry(1, "sim", {"simulation": {"kind": "csa"}}), _entry(2, "b")])
    assert _shape(nodes) == [("a", "spectrum", 0, False, 0), ("sim", "simulation", 1, False, 0),
                             ("b", "spectrum", 2, False, 0)]
    assert [n["type_label"] for n in nodes] == ["Spectrum", "Simulation", "Spectrum"]


def test_a_spectrums_fits_are_listed_beneath_it_by_type():
    raw = {"fits": {"quadrupolar": {"label": "Quadrupolar", "target": "a", "time": "t"},
                    "csa": {"label": "CSA"}},
           "slice_fit_results": [{"popt": [1.0]}, {"popt": [1.0]}],
           "relaxation_fit_results": [{"model": "exponential_decay"}],
           "fit_models": [{"id": 1, "name": "two sites", "kind": "quadrupolar",
                           "kind_label": "Quadrupolar"},
                          {"id": 2, "name": "T2", "kind": "curve", "kind_label": "Curve fit"}]}
    nodes = et.build_experiment_tree([_entry(0, "a", raw)])
    children = nodes[0]["children"]
    assert [(c["label"], c["type_label"], c["action"]) for c in children] == [
        ("Quadrupolar fit", "Fit", ("stored_fit", "quadrupolar")),
        ("CSA fit", "Fit", ("stored_fit", "csa")),
        ("Spectral fit (2 slices)", "Fit", ("slice_fit",)),
        ("Curve fit (exponential decay)", "Profile fit", ("curve_fit",)),
        ("two sites (Quadrupolar)", "Fit", ("fit_model", 1)),
        ("T2 (Curve fit)", "Profile fit", ("fit_model", 2))]
    # Each selects its spectrum's row, and is not a row itself.
    assert all(c["row"] == 0 and c["virtual"] for c in children)
    # Region integrals are not a fit.
    integrals = et.build_experiment_tree([_entry(0, "a", {"slice_fit_results": [{"integrals": [1]}]})])
    assert integrals[0]["children"] == []


def test_a_kept_fit_listed_as_an_experiment_sits_under_its_spectrum_once():
    model = {"id": 7, "name": "one site", "kind": "csa", "kind_label": "CSA/Dipolar"}
    entries = [
        _entry(0, "a", {"fit_models": [model]}),
        _entry(1, "other"),
        _entry(2, "a — fit", {"fit_model": model,
                              "fit_model_source": {"spectrum": "a", "entry_id": "s-e1", "id": 7}})]
    nodes = et.build_experiment_tree(entries)
    assert _shape(nodes) == [("a", "spectrum", 0, False, 0), ("a — fit", "fit", 2, False, 1),
                             ("other", "spectrum", 1, False, 0)]
    # By name when the ID is another session's (a saved state reopened).
    entries[2]["raw"]["fit_model_source"]["entry_id"] = "elsewhere-e9"
    assert _shape(et.build_experiment_tree(entries))[1] == ("a — fit", "fit", 2, False, 1)
    # Its spectrum gone: it stands on its own.
    alone = et.build_experiment_tree(entries[1:])
    assert [n["label"] for n in alone] == ["other", "a — fit"]


def test_a_multi_spectrum_fit_is_a_group():
    inputs = {"experiments": [{"key": "p1"}, {"key": "p2"}]}
    stored = {"fits": {"multi": {"label": "Multi-spectrum", "inputs": inputs}}}
    kept = {"id": 1, "name": "joint", "kind": "multi"}
    nodes = et.build_experiment_tree([
        _entry(0, "x", stored), _entry(1, "y", stored),
        _entry(2, "x — joint", {"fit_model": kept, "fit_model_source": {"spectrum": "x"}}),
        _entry(3, "y — joint", {"fit_model": kept, "fit_model_source": {"spectrum": "y"}})])
    assert [(n["label"], n["kind"]) for n in nodes] == [
        ("x", "spectrum"), ("y", "spectrum"),
        ("Multi-spectrum fit (2 spectra)", "multi_fit"), ("joint (2 spectra)", "multi_fit")]
    stored_group, kept_group = nodes[2], nodes[3]
    assert [(c["label"], c["row"], c["action"]) for c in stored_group["children"]] == [
        ("x", 0, ("stored_fit", "multi")), ("y", 1, ("stored_fit", "multi"))]
    assert [(c["label"], c["row"], c["virtual"]) for c in kept_group["children"]] == [
        ("x — joint", 2, False), ("y — joint", 3, False)]
    text = et.experiment_tree_text(nodes)
    assert "Multi-spectrum fit (2 spectra)  [Multi-fit]" in text
    assert "    x — joint  [Fit]" in text


def test_every_row_appears_exactly_once_whatever_the_records_say():
    # Unreadable records must never hide an open experiment.
    entries = [_entry(0, "a", {"fits": "nonsense", "fit_models": [None, 3],
                               "slice_fit_results": "x", "relaxation_fit_results": [None]}),
               _entry(1, "b", {"fit_model": "nonsense", "fit_model_source": 4}),
               _entry(2, "c", {"fit_model": {"kind": "csa"},
                               "fit_model_source": {"spectrum": "c"}})]      # its own parent
    rows = [n["row"] for n, _d in et.iter_nodes(et.build_experiment_tree(entries))
            if not n["virtual"]]
    assert sorted(rows) == [0, 1, 2]


# ---------------------------------------------------------------------------
# In the window
# ---------------------------------------------------------------------------

@pytest.fixture
def quiet(monkeypatch):
    said = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: said.append(a[2]))
    return said


def _labels(window):
    tree = window.experiment_tree
    return [(tree.node(i)["label"], tree.node(i)["type_label"]) for i in tree.items()]


def test_the_tree_is_what_is_shown_and_follows_the_list(qapp):  # noqa: F811
    window = MainWindow()
    tree = window.experiment_tree
    assert not window.spectrum_list.isVisibleTo(window.open_experiments_box)
    assert tree.isVisibleTo(window.open_experiments_box)
    first = _quad_spectrum(window, "first")
    second = _quad_spectrum(window, "second")
    window._refresh_experiment_tree()
    assert _labels(window) == [(window.session.label(first), "Spectrum"),
                               (window.session.label(second), "Spectrum")]
    assert tree.selected_rows() == [second]                 # the list's selection, mirrored
    window._select_single_row(first)
    assert tree.selected_rows() == [first]
    window._remove_rows([first])
    window._refresh_experiment_tree()
    assert [label for label, _t in _labels(window)] == [window.session.label(0)]
    window._on_clear_clicked()
    window._refresh_experiment_tree()
    assert tree.items() == []
    window.close()


def test_selecting_in_the_tree_selects_the_experiments(qapp):  # noqa: F811
    window = MainWindow()
    tree = window.experiment_tree
    a = _quad_spectrum(window, "a")
    b = _quad_spectrum(window, "b")
    window._refresh_experiment_tree()
    tree.clearSelection()
    tree.setCurrentItem(tree.item_for_row(a))               # a click on "a"
    assert window._selected_rows() == [a] and window._selected_index() == a
    tree.item_for_row(b).setSelected(True)                  # Ctrl-click "b" too
    assert window._selected_rows() == [a, b]
    assert window.compare_list.count() == 2                 # as a multi-selection always did
    window.close()


def test_a_fit_appears_under_its_spectrum_and_opening_it_replots_it(qapp, quiet):  # noqa: F811
    window = MainWindow()
    tree = window.experiment_tree
    index = _quad_spectrum(window, "data")
    panel = _fit_quad(window)
    assert panel.result is not None
    window._refresh_experiment_tree()
    assert _labels(window) == [(window.session.label(index), "Spectrum"),
                               ("Quadrupolar fit", "Fit")]
    other = _quad_spectrum(window, "other")                 # work moves elsewhere
    window._refresh_experiment_tree()
    drawn = []
    panel._draw_fit = lambda result, **k: drawn.append(result)
    fit_item = tree.find(f"{window.session.entry_id(index)}:fit:quadrupolar")
    tree.nodeActivated.emit(tree.node(fit_item))            # double-click
    assert window._selected_rows() == [index] and other != index
    assert drawn and drawn[-1] is panel.result              # the stored fit, replotted
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    assert "shown again" in window.statusBar().currentMessage()
    # The fit stays the highlighted entry, and survives the tree's own rebuild.
    assert [tree.node(i)["label"] for i in tree.selectedItems()] == ["Quadrupolar fit"]
    window._refresh_experiment_tree()
    assert [tree.node(i)["label"] for i in tree.selectedItems()] == ["Quadrupolar fit"]
    assert not quiet, quiet
    window.close()


def test_kept_fits_are_listed_and_shown(qapp, quiet, monkeypatch):  # noqa: F811
    window = MainWindow()
    tree = window.experiment_tree
    index, ids = _window_with_kept_fits(window)
    window._refresh_experiment_tree()
    kinds = [t for _l, t in _labels(window)]
    assert kinds == ["Spectrum", "Fit", "Fit"]
    shown = []
    monkeypatch.setattr(window, "_show_fit_model", lambda i, model_id: shown.append((i, model_id)))
    item = tree.find(f"{window.session.entry_id(index)}:model:{ids[1]}")
    tree.nodeActivated.emit(tree.node(item))
    assert shown == [(index, ids[1])]
    # Listed as an experiment of its own: one entry, under its spectrum.
    experiment = window._fit_model_experiment(index, ids[0])
    window._refresh_experiment_tree()
    child = tree.item_for_row(experiment)
    assert tree.node(child.parent())["row"] == index
    assert tree.node(child)["type_label"] == "Fit"
    assert len(tree.items()) == 3                             # not listed twice
    window.close()


def test_right_click_and_drag_work_as_on_the_list(qapp, monkeypatch):  # noqa: F811
    window = MainWindow()
    tree = window.experiment_tree
    index = _quad_spectrum(window, "data")
    window._refresh_experiment_tree()
    # QMenu.exec() cannot be patched on the real class (it hangs): the
    # name main_window uses is replaced, as test_gui does.
    import opennmr.gui.main_window as main_window_module
    from test_gui import _FakeMenu
    menus = []
    monkeypatch.setattr(main_window_module, "QMenu",
                        lambda *a, **k: menus.append(_FakeMenu()) or menus[-1])
    tree.menuRequested.emit(tree.node(tree.item_for_row(index)), tree.mapToGlobal(tree.pos()))
    labels = menus[-1].added_labels
    assert "Rename…" in labels and "Remove" in labels        # the list's own menu
    dragged = []
    monkeypatch.setattr(tree, "start_drag", lambda rows: dragged.append(
        tree.mime_for_rows(rows).formats()))
    tree.start_drag([index])
    assert dragged and "application/x-qabstractitemmodeldatalist" in dragged[0]
    window.close()


def test_collapsing_is_remembered_across_rebuilds(qapp, quiet):  # noqa: F811
    window = MainWindow()
    tree = window.experiment_tree
    index, _ids = _window_with_kept_fits(window)
    window._refresh_experiment_tree()
    top = tree.item_for_row(index)
    assert top.isExpanded()
    top.setExpanded(False)
    window._refresh_experiment_tree()
    assert not tree.item_for_row(index).isExpanded()
    window.close()


def test_a_refresh_asked_for_while_the_window_is_destroyed_is_ignored(qapp):  # noqa: F811
    # User report, 2026-11-14: "Internal C++ object (QTimer) already
    # deleted" on closing -- the list's rows go after the window's timer.
    import shiboken6
    window = MainWindow()
    _quad_spectrum(window, "data")
    shiboken6.delete(window._tree_refresh_timer)
    window._schedule_tree_refresh()                 # no error
    assert window._tree_refresh_timer is None
    window._schedule_tree_refresh()
    window._refresh_experiment_tree()               # nothing to do, no error
    window.spectrum_list.clear()                    # the model's signals still fire
    window.close()
