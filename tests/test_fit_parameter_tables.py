"""Each fit entry's table of parameters and errors: shown, copied,
exported to CSV, and the fit restored from it (explicit user request,
2026-11-13; Phase 3 step 3 of the Open experiments tree)."""

import csv
import json

import numpy as np
import pytest
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.notebook import fit_values as fv

from test_gui import qapp, _load_spectrum_directly  # noqa: F401 -- fixture
from test_gui_fit_sim_state import _fit_quad, _quad_spectrum
from test_gui_round4_fits import _window_with_kept_fits


# ---------------------------------------------------------------------------
# The table (no GUI)
# ---------------------------------------------------------------------------

def _record():
    return fv.values_record(
        {"Al1": {"δiso (ppm)": {"value": 12.0123456789, "error": 0.03},
                 "C_Q (MHz)": {"value": 4.2, "error": 0.5, "fixed": True}},
         "Al2": {"δiso (ppm)": 60.5}},
        fit="Quadrupolar", spectrum="sample · 27Al", quality={"r2": 0.9991, "note": "x"},
        source="/data/sample")


def test_a_record_from_numbers_is_a_fitted_values_record(tmp_path):
    record = _record()
    assert record[fv.FORMAT_KEY] == fv.FORMAT_VERSION
    assert record["columns"] == ["δiso (ppm)", "C_Q (MHz)"]
    assert record["values"]["Al1"]["C_Q (MHz)"] == {"value": 4.2, "error": None, "fixed": True}
    assert record["quality"] == {"r2": 0.9991}                  # numbers only
    # It is what Save the Fitted Values writes and reads.
    path = fv.save_fitted_values(record, tmp_path / "values")
    assert fv.load_fitted_values(path)["values"]["Al2"]["δiso (ppm)"]["value"] == 60.5


def test_one_row_per_parameter_in_full_precision():
    rows = fv.parameter_rows(_record())
    assert [(r["site"], r["parameter"]) for r in rows] == [
        ("Al1", "δiso (ppm)"), ("Al1", "C_Q (MHz)"), ("Al2", "δiso (ppm)")]
    assert rows[0]["value"] == 12.0123456789 and rows[0]["error"] == 0.03
    assert rows[1]["fixed"] and rows[1]["error"] is None
    cells = fv.parameter_cells(_record())
    assert cells[0] == ["Al1", "δiso (ppm)", "12.01234568", "0.03", ""]   # not rounded to the error
    assert cells[1] == ["Al1", "C_Q (MHz)", "4.2", "", "yes"]
    assert fv.quality_text(_record()) == "R² 0.9991"
    assert fv.parameter_rows({}) == [] and fv.parameter_rows({"values": "x"}) == []


def test_a_cell_that_is_not_a_number_keeps_its_text():
    record = fv.fitted_values_record(fit="J", columns=["Pattern", "J (Hz)"],
                                     rows=[("CH3", ["qd", "7.1 ± 0.2"])])
    cells = fv.parameter_cells(record)
    assert cells == [["CH3", "Pattern", "qd", "", ""], ["CH3", "J (Hz)", "7.1", "0.2", ""]]


def test_tsv_and_csv(tmp_path):
    record = _record()
    lines = fv.values_to_tsv(record).split("\n")
    assert lines[0] == "Site\tParameter\tValue\tError (1σ)\tFixed"
    assert lines[1] == "Al1\tδiso (ppm)\t12.01234568\t0.03\t"
    path = fv.values_to_csv(record, tmp_path / "out" / "values")
    assert path.endswith("values.csv")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    assert ["# Fit", "Quadrupolar"] in rows and ["# Quality", "R² 0.9991"] in rows
    start = rows.index(list(fv.PARAMETER_ROW_COLUMNS))
    assert rows[start + 1] == ["Al1", "δiso (ppm)", "12.01234568", "0.03", ""]
    assert len(rows) == start + 4


# ---------------------------------------------------------------------------
# In the window
# ---------------------------------------------------------------------------

@pytest.fixture
def quiet(monkeypatch):
    said = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: said.append(a[2]))
    return said


def _node(window, key):
    window._refresh_experiment_tree()
    tree = window.experiment_tree
    item = tree.find(key)
    assert item is not None, [tree.node(i)["key"] for i in tree.items()]
    return tree.node(item)


def test_a_panel_fits_parameters_as_a_table(qapp, quiet, tmp_path):  # noqa: F811
    window = MainWindow()
    index = _quad_spectrum(window, "data")
    panel = _fit_quad(window)
    node = _node(window, f"{window.session.entry_id(index)}:fit:quadrupolar")
    dialog = window._show_fit_parameters(node, show=False)
    assert dialog is not None and not quiet, quiet
    assert "Quadrupolar" in dialog.windowTitle()
    table = dialog.table
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    assert headers == list(fv.PARAMETER_ROW_COLUMNS)
    cells = {(table.item(r, 0).text(), table.item(r, 1).text()): r
             for r in range(table.rowCount())}
    cq = next(r for (site, name), r in cells.items() if name.startswith("C_Q"))
    # The fit's own value and its uncertainty, as numbers.
    fitted_cq = abs(float(panel.result["popt"][1]))
    assert float(table.item(cq, 2).text()) == pytest.approx(fitted_cq, rel=2e-2)
    assert float(table.item(cq, 3).text()) > 0
    # Copy, CSV, JSON.
    dialog.copy_table()
    copied = QGuiApplication.clipboard().text()
    assert copied.split("\n")[0].split("\t") == list(fv.PARAMETER_ROW_COLUMNS)
    assert len(copied.split("\n")) == table.rowCount() + 1
    written = dialog.export_csv(str(tmp_path / "quad.csv"))
    with open(written, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[rows.index(list(fv.PARAMETER_ROW_COLUMNS)) + 1][0] == table.item(0, 0).text()
    saved = dialog.save_json(str(tmp_path / "quad.json"))
    assert json.load(open(saved, encoding="utf-8"))[fv.FORMAT_KEY] == fv.FORMAT_VERSION
    assert window._copy_fit_parameters(node)
    assert "copied" in window.statusBar().currentMessage()
    window.close()


def test_restore_brings_the_fit_back_after_work_moved_on(qapp, quiet):  # noqa: F811
    window = MainWindow()
    index = _quad_spectrum(window, "data")
    panel = _fit_quad(window)
    fitted = np.array(panel.result["popt"], dtype=float)
    node = _node(window, f"{window.session.entry_id(index)}:fit:quadrupolar")
    _quad_spectrum(window, "other")                         # selected: a blank fit
    assert window._selected_index() != index
    assert window._restore_tree_fit(node)
    assert window._selected_rows() == [index]
    assert panel.result is not None
    assert np.allclose(np.array(panel.result["popt"], dtype=float), fitted)
    assert "restored" in window.statusBar().currentMessage()
    assert not quiet, quiet
    window.close()


def test_a_spectral_fits_peaks_and_errors(qapp, quiet):  # noqa: F811
    from opennmr.processing.fitting import gaussian
    ppm = np.linspace(0, 10, 900)
    rng = np.random.default_rng(1)
    spectrum = NMRSpectrum(gaussian(ppm, 5.0, 0.05, 1.0) + rng.normal(0, 1e-3, ppm.size),
                           [NMRAxis("frequency", ppm=ppm, hz=ppm * 400)],
                           NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.03, fwhm=0.08, eta=0.0)
    window._on_fit_spectrum_clicked()
    entry = window.session.entry_id(0)
    node = _node(window, f"{entry}:fit:slices")
    record = window._tree_fit_record(node)
    shift = record["values"]["Peak 1"]["Shift (ppm)"]
    assert shift["value"] == pytest.approx(5.0, abs=1e-3) and shift["error"] > 0
    assert "Area" in record["values"]["Peak 1"] or "Amplitude" in record["values"]["Peak 1"]
    # Restore: the fitted values become the peak table's.
    window.peak_table.item(0, 0).setText("8.0")
    assert window._restore_tree_fit(node)
    assert float(window.peak_table.item(0, 0).text()) == pytest.approx(5.0, abs=1e-2)
    window.close()


def test_a_kept_fit_and_its_experiment_give_the_same_table(qapp, quiet):  # noqa: F811
    window = MainWindow()
    index, ids = _window_with_kept_fits(window)
    entry = window.session.entry_id(index)
    kept = window._tree_fit_record(_node(window, f"{entry}:model:{ids[1]}"))
    assert kept is not None and "2 sites" in kept["fit"]
    assert kept["quality"].get("r2") is not None            # from the kept model itself
    experiment = window._fit_model_experiment(index, ids[1])
    window._refresh_experiment_tree()
    row_node = window.experiment_tree.node(window.experiment_tree.item_for_row(experiment))
    listed = window._tree_fit_record(row_node)
    assert listed["fit"] == kept["fit"] and listed["values"] == kept["values"]
    window.close()


def test_an_entry_without_values_says_so(qapp, quiet):  # noqa: F811
    window = MainWindow()
    index = _quad_spectrum(window, "data")
    node = {"row": index, "virtual": True, "action": ("stored_fit", "csa"), "label": "CSA fit",
            "key": "x"}
    assert window._show_fit_parameters(node, show=False) is None
    assert quiet and "no fitted values" in quiet[-1]
    assert not window._copy_fit_parameters(node)
    window.close()
