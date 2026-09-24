"""Simulate > pNMR (CIF): the step-by-step panel (gui/pnmr_panel.py).

Uses the hand-checkable structures from test_pnmr_simulator.py; every
number asserted here is the engine's, reached through the panel's own
controls, so these check the wiring, the staleness rules and that the
GUI calls generate_spectrum_dict with what the controls say."""
import json
import math

import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QMessageBox

pytest.importorskip("pymatgen")

from opennmr.gui.main_window import MainWindow  # noqa: E402
from opennmr.notebook import pnmr_simulator as pnmr  # noqa: E402

from test_gui import qapp  # noqa: F401,E402 -- fixture
from test_pnmr_simulator import (  # noqa: E402
    ROCKSALT_A, ROCKSALT_CIF, TWO_SITES_CIF,
)


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Fail loudly instead of blocking on a message box."""
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


@pytest.fixture
def cifs(tmp_path):
    rock = tmp_path / "rocksalt.cif"
    rock.write_text(ROCKSALT_CIF)
    two = tmp_path / "two_sites.cif"
    two.write_text(TWO_SITES_CIF)
    return {"rocksalt": str(rock), "two": str(two), "dir": tmp_path}


def _check(list_widget, keys):
    for i in range(list_widget.count()):
        item = list_widget.item(i)
        item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in keys
                           else Qt.CheckState.Unchecked)


def _set_cell(panel, row, header, text):
    headers = [panel.shell_table.horizontalHeaderItem(c).text()
               for c in range(panel.shell_table.columnCount())]
    panel.shell_table.item(row, headers.index(header)).setText(str(text))


def _panel(qapp):  # noqa: F811
    window = MainWindow()
    return window, window.pnmr_panel


def test_panel_lives_on_the_simulate_tile(qapp):  # noqa: F811
    window, panel = _panel(qapp)
    assert panel.window() is window
    # Nothing to do before a CIF is loaded.
    assert not panel.initialise_button.isEnabled()
    assert not panel.enumerate_button.isEnabled()
    assert not panel.plot_button.isEnabled()
    assert panel.field_spin.value() == pytest.approx(11.7)
    assert panel.broadening_spin.value() == pytest.approx(1.0)


def test_load_fills_species_sites_and_nuclide(qapp, cifs):  # noqa: F811
    _, panel = _panel(qapp)
    assert panel.load_cif(cifs["two"])
    species = [panel.species_combo.itemText(i) for i in range(panel.species_combo.count())]
    assert species == ["Na", "Ni", "Mn", "Fe"]
    assert panel.atom_of_interest() == "Na"
    labels = [panel.sites_list.item(i).data(Qt.ItemDataRole.UserRole)
              for i in range(panel.sites_list.count())]
    assert labels == ["Na1", "Na2"]
    assert panel.sites_of_interest() is None            # all ticked = all sites
    assert panel.nuclide_combo.currentText() == "23Na"
    panel.species_combo.setCurrentText("Fe")
    assert panel.nuclide_combo.currentText() == "57Fe"
    assert [panel.sites_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(panel.sites_list.count())] == ["Fe1"]


def test_dropping_a_cif_loads_it(qapp, cifs):  # noqa: F811
    _, panel = _panel(qapp)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(cifs["rocksalt"])])
    event = QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime,
                       Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    panel.cif_drop.dropEvent(event)
    assert panel.cif_path == cifs["rocksalt"]
    assert panel.atom_of_interest() == "Na+"


def test_not_a_cif_is_not_taken(qapp, cifs):  # noqa: F811
    _, panel = _panel(qapp)
    other = cifs["dir"] / "notes.txt"
    other.write_text("x")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(other))])
    event = QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime,
                       Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    panel.cif_drop.dropEvent(event)
    assert panel.cif_path is None


def test_initialise_lists_only_what_is_left_as_contributors(qapp, cifs):  # noqa: F811
    _, panel = _panel(qapp)
    panel.load_cif(cifs["two"])
    _check(panel.ignore_sites_list, {"Fe1"})
    assert panel.initialise()
    headers = [panel.shell_table.horizontalHeaderItem(c).text()
               for c in range(panel.shell_table.columnCount())]
    assert headers[3:] == ["Na", "Ni", "Mn"]            # Fe1 ignored -> no Fe column
    assert panel.shell_table.rowCount() == 1            # a first shell is offered
    assert panel.enumerate_button.isEnabled()


def test_ignoring_a_shared_site_species_says_what_else_goes(qapp, cifs):  # noqa: F811
    _, panel = _panel(qapp)
    panel.load_cif(cifs["rocksalt"])
    _check(panel.ignore_species_list, {"Br-"})
    panel.initialise()
    assert "Cl-" in panel.init_label.text() and "Also left out" in panel.init_label.text()


def test_rocksalt_workflow_matches_the_engine(qapp, cifs):  # noqa: F811
    window, panel = _panel(qapp)
    panel.load_cif(cifs["rocksalt"])
    panel.initialise()
    # The suggested first shell brackets the 6 anion neighbours only.
    bins = panel.shell_bins()
    assert bins[0][0] == "NN" and bins[0][1] == 0.0
    assert ROCKSALT_A / 2 < bins[0][2] < ROCKSALT_A / math.sqrt(2)
    _set_cell(panel, 0, "Cl-", 10)
    _set_cell(panel, 0, "Br-", -10)
    # Na+ has no neighbour in NN: its cell is greyed and not editable.
    na_col = 3 + ["Na+", "Cl-", "Br-"].index("Na+")
    assert not (panel.shell_table.item(0, na_col).flags() & Qt.ItemFlag.ItemIsEditable)
    assert "6 neighbours/site" in panel.shell_summary_label.text()

    out = cifs["dir"] / "out"
    panel.output_dir_edit.setText(str(out))
    assert panel.enumerate_sites()
    _, table = panel._result
    by_shift = dict(table.groupby("Total_shift")["p"].sum())
    assert by_shift == pytest.approx({10.0 * (2 * k - 6): math.comb(6, k) / 64 for k in range(7)})
    assert (out / "23Na_shift_table.csv").exists()
    params = json.loads((out / "23Na_simulation_parameters.json").read_text())
    assert params["broadening_sigma_ppm"] == 1.0 and params["B0_tesla"] == 11.7

    # Same answer as calling the engine directly with the same arguments.
    kwargs = panel.generate_kwargs()
    kwargs["save_shift_table"] = False
    spec, direct = pnmr.generate_spectrum_dict(**kwargs)
    assert np.allclose(direct["p"], table["p"])
    assert np.allclose(spec["real_spectrum"], panel._spectrum.data)

    assert panel.plot_button.isEnabled()
    n_before = window.spectrum_list.count()
    panel.add_to_open_check.setChecked(True)
    assert panel.plot()
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    assert window.spectrum_list.count() == n_before + 1
    assert "pNMR 23Na" in window.spectrum_list.item(window.spectrum_list.count() - 1).text()
    panel.plot()                                         # plotting again does not add a duplicate
    assert window.spectrum_list.count() == n_before + 1
    added = window.session.spectra[-1]
    assert added.metadata.nucleus == "23Na"
    assert added.metadata.sfo1 == pytest.approx(11.26952278792250 * 11.7)


def test_changing_an_input_invalidates_the_result(qapp, cifs):  # noqa: F811
    _, panel = _panel(qapp)
    panel.load_cif(cifs["two"])
    panel.initialise()
    panel.save_check.setChecked(False)
    assert panel.enumerate_sites()
    assert panel.plot_button.isEnabled()
    panel.broadening_spin.setValue(2.0)
    assert not panel.plot_button.isEnabled()
    assert not panel.plot()
    panel.broadening_spin.setValue(1.0)                  # back to what was enumerated
    assert panel.plot_button.isEnabled()
    _set_cell(panel, 0, "Ni", 55)
    assert not panel.plot_button.isEnabled()


def test_changing_the_sites_needs_initialise_again(qapp, cifs):  # noqa: F811
    _, panel = _panel(qapp)
    panel.load_cif(cifs["two"])
    panel.initialise()
    assert panel.shell_table.isEnabled()
    _check(panel.sites_list, {"Na2"})
    assert not panel.shell_table.isEnabled()
    assert not panel.enumerate_button.isEnabled()
    assert "Initialise" in panel.init_label.text()
    panel.initialise()
    assert panel.sites_of_interest() == ["Na2"]
    assert panel.enumerate_button.isEnabled()


def test_chosen_sites_and_occupancy_weighting(qapp, cifs):  # noqa: F811
    _, panel = _panel(qapp)
    panel.load_cif(cifs["two"])
    panel.initialise()
    panel.shell_table.item(0, 2).setText("3.0")
    for sp, v in (("Ni", 100), ("Mn", 200), ("Fe", 300)):
        _set_cell(panel, 0, sp, v)
    panel.save_check.setChecked(False)
    panel.enumerate_sites()
    by_shift = dict(panel._result[1].groupby("Total_shift")["p"].sum())
    assert by_shift == pytest.approx({100.0: 1 / 3, 200.0: 1 / 3, 300.0: 1 / 3})
    panel.weight_check.setChecked(False)
    panel.enumerate_sites()
    by_shift = dict(panel._result[1].groupby("Total_shift")["p"].sum())
    assert by_shift == pytest.approx({100.0: 0.25, 200.0: 0.25, 300.0: 0.5})
    _check(panel.sites_list, {"Na2"})
    panel.initialise()
    panel.enumerate_sites()
    assert dict(panel._result[1].groupby("Total_shift")["p"].sum()) == pytest.approx({300.0: 1.0})
    assert panel.spectrum_label().endswith("(Na2)")


def test_overlapping_shells_are_refused_before_enumerating(qapp, cifs):  # noqa: F811
    _, panel = _panel(qapp)
    panel.load_cif(cifs["two"])
    panel.initialise()
    panel.add_shell()
    panel.shell_table.item(1, 1).setText("0.5")          # starts inside NN
    assert "overlap" in panel.shell_error_label.text()
    assert not panel.enumerate_button.isEnabled()


def test_bad_shift_value_is_reported(qapp, cifs):  # noqa: F811
    _, panel = _panel(qapp)
    panel.load_cif(cifs["two"])
    panel.initialise()
    _set_cell(panel, 0, "Ni", "abc")
    assert "not a number" in panel.shell_error_label.text()
    assert not panel.enumerate_button.isEnabled()


def test_manual_axis_limits_are_used(qapp, cifs):  # noqa: F811
    _, panel = _panel(qapp)
    panel.load_cif(cifs["two"])
    panel.initialise()
    panel.save_check.setChecked(False)
    panel.auto_axis_check.setChecked(False)
    panel.ppm_min_spin.setValue(-50.0)
    panel.ppm_max_spin.setValue(450.0)
    panel.points_spin.setValue(1001)
    panel.enumerate_sites()
    x = panel._spectrum.axes[0].ppm
    assert x.min() == pytest.approx(-50.0) and x.max() == pytest.approx(450.0) and len(x) == 1001


def test_save_table_as_writes_csv_and_parameters(qapp, cifs):  # noqa: F811
    _, panel = _panel(qapp)
    panel.load_cif(cifs["two"])
    panel.initialise()
    panel.save_check.setChecked(False)
    panel.enumerate_sites()
    target = cifs["dir"] / "elsewhere" / "mine.csv"
    target.parent.mkdir()
    panel.save_table(str(target))
    assert target.exists()
    assert json.loads((cifs["dir"] / "elsewhere" / "mine_parameters.json").read_text())["atom_of_interest"] == "Na"


def test_notebook_code_reruns_the_same_simulation(qapp, cifs):  # noqa: F811
    _, panel = _panel(qapp)
    panel.load_cif(cifs["two"])
    panel.initialise()
    _set_cell(panel, 0, "Ni", 12.5)
    panel.save_check.setChecked(False)
    code = panel.notebook_code()
    namespace = {}
    exec(code, namespace)  # noqa: S102 -- our own generated code
    panel.enumerate_sites()
    assert np.allclose(namespace["spectrum_dict"]["real_spectrum"], panel._spectrum.data)


def test_histogram_is_drawn_on_the_simulation_plot(qapp, cifs):  # noqa: F811
    window, panel = _panel(qapp)
    panel.load_cif(cifs["rocksalt"])
    panel.initialise()
    assert window.canvas_tabs.currentWidget() is window.simulation_canvas
    ax = window.simulation_canvas.figure.axes[0]
    assert "Distance" in ax.get_xlabel()
    heights = sum(p.get_height() for p in ax.patches if hasattr(p, "get_height"))
    # 6 anions + 12 Na per site within the 6 A default radius, plus the
    # further shells; the NN bars alone hold 6 per site.
    assert heights >= 18


def test_added_shells_follow_on_without_rounding(qapp, cifs):  # noqa: F811
    # Regression: limits were once shown to 3 significant figures, so a
    # suggested 3.404 A became 3.4 and the next shell collapsed to 3.4-3.4.
    _, panel = _panel(qapp)
    panel.load_cif(cifs["rocksalt"])
    panel.initialise()
    panel.add_shell()
    panel.add_shell()
    bins = panel.shell_bins()                       # raises if any are invalid
    assert [b[0] for b in bins] == ["NN", "NNN", "N3N"]
    assert bins[0][2] == pytest.approx(round((ROCKSALT_A / 2 + ROCKSALT_A / math.sqrt(2)) / 2, 3))
    assert bins[1][1] == bins[0][2] and bins[2][1] == bins[1][2]
    assert bins[1][1] < ROCKSALT_A / math.sqrt(2) < bins[1][2]


def test_axis_warning_is_shown_parser_notes_are_only_logged(qapp, cifs):  # noqa: F811
    _, panel = _panel(qapp)
    panel.load_cif(cifs["rocksalt"])
    panel.initialise()
    _set_cell(panel, 0, "Cl-", 10)
    panel.save_check.setChecked(False)
    panel.auto_axis_check.setChecked(False)
    panel.ppm_min_spin.setValue(-20.0)
    panel.ppm_max_spin.setValue(20.0)
    panel.enumerate_sites()
    assert "outside the specified xlims" in panel.results_label.text()
    assert "_symmetry" not in panel.results_label.text()


# --- loading and saving settings ----------------------------------------------

def _configured_two_sites(panel, cif):
    """A non-default set-up touching every control load_settings restores."""
    panel.load_cif(cif)
    _check(panel.sites_list, {"Na1"})
    _check(panel.ignore_sites_list, {"Fe1"})
    panel.initialise()
    panel.shell_table.item(0, 2).setText("3.0")
    _set_cell(panel, 0, "Ni", 100)
    _set_cell(panel, 0, "Mn", -25.5)
    panel.field_spin.setValue(9.4)
    panel.broadening_spin.setValue(2.5)
    panel.points_spin.setValue(4096)
    panel.auto_axis_check.setChecked(False)
    panel.ppm_min_spin.setValue(-200.0)
    panel.ppm_max_spin.setValue(300.0)
    panel.weight_check.setChecked(False)


def test_saved_settings_restore_every_step(qapp, cifs, _no_modal_dialogs):  # noqa: F811
    _, first = _panel(qapp)
    _configured_two_sites(first, cifs["two"])
    out = cifs["dir"] / "run1"
    first.output_dir_edit.setText(str(out))
    assert first.enumerate_sites()
    saved = out / "23Na_simulation_parameters.json"

    _, second = _panel(qapp)
    assert second.load_settings(str(saved))
    assert _no_modal_dialogs == []                          # nothing to report
    want = first.generate_kwargs()
    got = second.generate_kwargs()
    assert got.pop("output_dir") is None                    # not restored, see the log
    want.pop("output_dir")
    assert got == want
    assert "Output folder left as it was" in second.log_edit.toPlainText()
    second.save_check.setChecked(False)
    first_spectrum = first._spectrum.data.copy()
    assert second.enumerate_sites()
    assert np.array_equal(second._spectrum.data, first_spectrum)


def test_dropping_settings_json_loads_them(qapp, cifs):  # noqa: F811
    _, first = _panel(qapp)
    _configured_two_sites(first, cifs["two"])
    path = cifs["dir"] / "setup.json"
    first.save_settings(str(path))
    _, second = _panel(qapp)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    second.cif_drop.dropEvent(QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime,
                                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))
    assert second.cif_path == cifs["two"]
    assert second.shell_bins() == first.shell_bins()
    assert second.enumerate_button.isEnabled()              # initialised, ready to run


def test_settings_whose_cif_moved_ask_for_it(qapp, cifs, monkeypatch):  # noqa: F811
    path = cifs["dir"] / "elsewhere.json"
    pnmr.save_simulation_parameters(str(path), cif_filename="/gone/two_sites_old.cif",
                                    atom_of_interest="Na", shell_bins=[("NN", 0.0, 3.0)],
                                    shift_dict={"NN": {"Ni": 5.0}}, nuclide="23Na", B0_tesla=11.7)
    from PySide6.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (cifs["two"], ""))
    _, panel = _panel(qapp)
    assert panel.load_settings(str(path))
    assert panel.cif_path == cifs["two"]
    assert panel.shift_dict()["NN"]["Ni"] == 5.0


def test_what_cannot_be_restored_is_listed(qapp, cifs, _no_modal_dialogs):  # noqa: F811
    path = cifs["dir"] / "odd.json"
    pnmr.save_simulation_parameters(
        str(path), cif_filename=cifs["two"], atom_of_interest="Na",
        shell_bins=[("NN", 0.0, 3.0)],
        shift_dict={"NN": {"Ni": 5.0, "Fe": 7.0}, "N9": {"Ni": 1.0}},
        ignore_sites=["Fe1", "Q7"], nuclide="23Na", B0_tesla=11.7, offset_ppm=12.0)
    _, panel = _panel(qapp)
    assert panel.load_settings(str(path))
    report = " ".join(text for _, text in _no_modal_dialogs)
    for expected in ("Q7", "Fe dropped", "'N9'", "carrier offset"):
        assert expected in report
    assert panel.shift_dict() == {"NN": {"Na": 0.0, "Ni": 5.0, "Mn": 0.0}}


def test_settings_for_a_species_the_cif_lacks_are_refused(qapp, cifs, _no_modal_dialogs):  # noqa: F811
    path = cifs["dir"] / "li.json"
    pnmr.save_simulation_parameters(str(path), cif_filename=cifs["two"], atom_of_interest="Li",
                                    shell_bins=[("NN", 0.0, 3.0)], shift_dict={"NN": {}},
                                    nuclide="7Li", B0_tesla=11.7)
    _, panel = _panel(qapp)
    assert not panel.load_settings(str(path))
    assert "no 'Li'" in _no_modal_dialogs[-1][1]


# --- large runs ---------------------------------------------------------------

@pytest.mark.parametrize("answer, ran", [("No", False), ("Yes", True)])
def test_a_run_over_the_limit_asks_first(qapp, cifs, monkeypatch, answer, ran):  # noqa: F811
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: getattr(QMessageBox.StandardButton, answer))
    _, panel = _panel(qapp)
    monkeypatch.setattr(panel, "MAX_CONFIGURATIONS", 2)     # this run makes 3 rows
    panel.load_cif(cifs["two"])
    panel.initialise()
    panel.save_check.setChecked(False)
    panel.enumerate_sites()
    assert (panel._result is not None) == ran
    assert "Large run: 3 configurations" in panel.log_edit.toPlainText()
    assert not panel.is_running()


def test_enumerator_cancels_at_the_next_step():
    from opennmr.gui.pnmr_panel import PnmrEnumerator
    worker = PnmrEnumerator()
    outcome = []
    worker.cancelled.connect(lambda: outcome.append("cancelled"))
    worker.finished.connect(lambda _p: outcome.append("finished"))
    worker.set_job({"cif_filename": "unused.cif"}, None)
    worker.cancel()
    import opennmr.notebook.pnmr_simulator as eng

    def fake(**kwargs):
        kwargs["progress"]("Enumerating environments", 1, 10)
        raise AssertionError("progress should have raised")
    original = eng.generate_spectrum_dict
    eng.generate_spectrum_dict = fake
    try:
        worker.run()
    finally:
        eng.generate_spectrum_dict = original
    assert outcome == ["cancelled"]


def _wait(qapp, condition, timeout=30.0):  # noqa: F811
    import time
    end = time.monotonic() + timeout
    while not condition():
        qapp.processEvents()
        if time.monotonic() > end:
            raise AssertionError("timed out")
        time.sleep(0.01)


@pytest.fixture
def threaded(monkeypatch):
    monkeypatch.setenv("OPENNMR_BACKGROUND_LOAD", "1")


def test_background_run_keeps_the_panel_live_and_locks_its_inputs(qapp, cifs, threaded):  # noqa: F811
    window, panel = _panel(qapp)
    panel.load_cif(cifs["rocksalt"])
    panel.initialise()
    _set_cell(panel, 0, "Cl-", 10)
    panel.save_check.setChecked(False)
    assert panel.enumerate_sites()
    assert panel._thread is not None and panel._thread.isRunning()
    if panel.is_running():                                   # it may already be done
        assert panel.enumerate_button.text() == "Cancel"
        assert not panel._shells_box.isEnabled()
        assert not panel.load_settings_button.isEnabled()
    _wait(qapp, lambda: not panel.is_running())
    assert panel._result is not None and panel.plot_button.isEnabled()
    assert panel.enumerate_button.text() == "Enumerate sites"
    assert panel._shells_box.isEnabled()
    window.close()
    assert not panel._thread.isRunning()


def _slow_engine(monkeypatch, started):
    """Replace the engine call with one that reports progress until it is
    cancelled -- a stand-in for a long enumeration."""
    import time
    import opennmr.notebook.pnmr_simulator as eng

    def slow(**kwargs):
        started.append(True)
        for i in range(10_000):
            kwargs["progress"]("Enumerating environments", i, 10_000)
            time.sleep(0.005)
        raise AssertionError("never cancelled")
    monkeypatch.setattr(eng, "generate_spectrum_dict", slow)


def test_cancel_stops_a_background_run(qapp, cifs, threaded, monkeypatch):  # noqa: F811
    _, panel = _panel(qapp)
    panel.load_cif(cifs["two"])
    panel.initialise()
    panel.save_check.setChecked(False)
    started = []
    _slow_engine(monkeypatch, started)
    panel.enumerate_sites()
    _wait(qapp, lambda: started and "Enumerating" in panel.progress_label.text())
    panel._on_enumerate_clicked()                            # the button reads "Cancel" now
    _wait(qapp, lambda: not panel.is_running())
    assert panel._result is None
    assert "cancelled" in panel.log_edit.toPlainText()
    panel.shutdown()


def test_closing_the_window_mid_run_stops_the_thread(qapp, cifs, threaded, monkeypatch):  # noqa: F811
    window, panel = _panel(qapp)
    panel.load_cif(cifs["two"])
    panel.initialise()
    started = []
    _slow_engine(monkeypatch, started)
    panel.enumerate_sites()
    _wait(qapp, lambda: bool(started))
    window.close()                                           # must not hang or abort
    assert not panel._thread.isRunning()


def test_a_long_label_list_is_not_dumped_into_the_log(qapp, cifs, tmp_path):  # noqa: F811
    structure, _ = pnmr.read_cif_with_occupancies(cifs["two"])
    structure.make_supercell([3, 3, 2])            # pymatgen labels every copy: Na1_1, ...
    path = tmp_path / "big.cif"
    structure.to(str(path))
    _, panel = _panel(qapp)
    panel.load_cif(str(path))
    line = panel.log_edit.toPlainText().splitlines()[-1]
    assert "labels in all" in line and len(line) < 400
