"""File ▸ Export Experiment to Notebook… in the real window.

The notebook must reproduce the spectrum the GUI shows, so the test runs
the exported cells and compares the data with the GUI's own.
"""
import json

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog

from opennmr.gui.main_window import MainWindow

from test_gui import qapp, _make_1d_dataset  # noqa: F401 -- fixture


def _export(window, monkeypatch, out):
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    window._on_export_notebook_clicked()


def test_the_file_menu_has_the_export_action(qapp):
    window = MainWindow()
    file_titles = [a.text().replace("&", "") for a in window.menuBar().actions()[0].menu().actions()]
    assert "Export Experiment to Notebook…" in file_titles
    assert window.export_notebook_action.toolTip()


def test_with_no_experiment_selected_it_says_so_and_writes_nothing(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    out = tmp_path / "x.ipynb"
    _export(window, monkeypatch, out)
    assert "Select an experiment" in window.statusBar().currentMessage()
    assert not out.exists()


def test_cancelling_the_dialog_writes_nothing(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path), vendor="bruker")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    window._on_export_notebook_clicked()
    assert not list(tmp_path.glob("*.ipynb"))


def test_the_exported_notebook_reproduces_what_the_window_shows(qapp, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    expt = _make_1d_dataset(tmp_path)
    window = MainWindow()
    window._load(expt, vendor="bruker")
    index = window._selected_index()
    shown = window.session.spectra[index]
    out = tmp_path / "out.ipynb"
    _export(window, monkeypatch, out)
    assert out.exists() and "Wrote" in window.statusBar().currentMessage()
    with open(out, encoding="utf-8") as handle:
        nb = json.load(handle)
    assert nb["metadata"]["opennmr"]["source_path"] == expt
    namespace = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec(cell["source"], namespace)
    plt.close("all")
    replayed = namespace["spectrum"]
    np.testing.assert_allclose(replayed.data, shown.data, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(replayed.axis_values(0, "ppm"), shown.axis_values(0, "ppm"))


def test_a_derived_spectrum_is_refused_with_a_message_not_a_crash(qapp, tmp_path, monkeypatch):
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    window = MainWindow()
    sim = NMRSpectrum(data=np.zeros(8), axes=[NMRAxis(kind="frequency", ppm=np.linspace(5, 0, 8))],
                      metadata=NMRMetadata(vendor="simulated", raw={}))
    window._add_derived_spectrum(sim, "sim")
    warned = []
    monkeypatch.setattr("opennmr.gui.main_window.QMessageBox.warning",
                        staticmethod(lambda *a, **k: warned.append(a[-1])))
    _export(window, monkeypatch, tmp_path / "sim.ipynb")
    assert warned and "save_spectrum_folder" in warned[0]
    assert not (tmp_path / "sim.ipynb").exists()
