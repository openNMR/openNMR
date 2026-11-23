"""User report, 2026-11-23: a pseudo-2D series fitted slice by slice and then with
Dynamics ▸ Fit Curve lost both fits on saving an openNMR state and reopening
(tree entries: "could not be shown again ... no longer stored"). The results hold
numpy arrays (and each slice an NMRSpectrum), which the JSON processing state
silently skipped. They are now saved in a file of their own."""
import os

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.gui.main_window import MainWindow
from opennmr.io import fit_results, persistence

from test_gui import qapp, _load_spectrum_directly, _make_pseudo2d_relaxation_spectrum  # noqa: F401
from test_gui_integration_to_dynamics import _window_with_series, _type_delays, _dialogs, DELAYS  # noqa: F401


def _fitted_window():
    window = _window_with_series()
    _type_delays(window)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    window._on_fit_curve_button_clicked()          # fits every slice, then the curve
    raw = window.session.spectra[0].metadata.raw
    assert raw["slice_fit_results"] and raw["relaxation_fit_results"]
    return window


def _eq(a, b, path=""):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=path)
    elif isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _eq(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _eq(x, y, f"{path}[{i}]")
    elif hasattr(a, "axes") and hasattr(a, "metadata"):          # a slice trace
        np.testing.assert_allclose(a.axes[0].ppm, b.axes[0].ppm)
        assert a.metadata.nucleus == b.metadata.nucleus and a.metadata.sfo1 == b.metadata.sfo1
        _eq(a.metadata.raw["slice"], b.metadata.raw["slice"], path + ".slice")
    elif callable(a):
        assert a is b or a.__name__ == b.__name__ or callable(b), path     # a rebuilt model function
    elif isinstance(a, float) and a != a:
        assert b != b, path
    else:
        assert a == b, path


def _same_results(a, b):
    _eq(list(a), list(b))


def test_encoding_round_trip_is_exact_and_shares_axes(qapp):
    window = _fitted_window()
    raw = window.session.spectra[0].metadata.raw
    payload = fit_results.encode_spectrum_results(window.session.spectra[0])
    import json
    json.dumps(payload)                                     # JSON-safe
    assert len(payload["slice_fit_results"]["bases"]) == 1  # the slices' axes are written once
    back = fit_results.decode_spectrum_results(payload)
    _same_results(raw["slice_fit_results"], back["slice_fit_results"])
    _same_results(raw["relaxation_fit_results"], back["relaxation_fit_results"])
    assert back["slice_fit_results"][0]["slice"].metadata.raw["slice_data_not_stored"]


def test_results_file_round_trip_and_removal(qapp, tmp_path):
    window = _fitted_window()
    spectrum = window.session.spectra[0]
    path = str(tmp_path / "r.json.gz")
    assert fit_results.write_results_file(path, spectrum)
    back = fit_results.read_results_file(path)
    _same_results(spectrum.metadata.raw["relaxation_fit_results"], back["relaxation_fit_results"])
    spectrum.metadata.raw.pop("slice_fit_results"); spectrum.metadata.raw.pop("relaxation_fit_results")
    assert not fit_results.write_results_file(path, spectrum) and not os.path.exists(path)
    (tmp_path / "bad.json.gz").write_bytes(b"not gzip")
    assert fit_results.read_results_file(str(tmp_path / "bad.json.gz")) == {}


def test_state_round_trip_brings_back_slice_and_curve_fits(qapp, _dialogs, tmp_path, monkeypatch):
    window = _fitted_window()
    source = window.session.spectra[0].metadata.raw
    folder = str(tmp_path / "work.opennmr")
    window._save_state(folder, "work", [0], copy_2d=True)
    assert os.path.isdir(os.path.join(folder, "results"))
    fresh = MainWindow()
    opened, _messages = fresh._open_state(folder)
    raw = fresh.session.spectra[opened[0]].metadata.raw
    _same_results(source["slice_fit_results"], raw["slice_fit_results"])
    _same_results(source["relaxation_fit_results"], raw["relaxation_fit_results"])
    # ...and the curve fit is live again in Dynamics, with its T2.
    assert fresh._relaxation_fit_results[0]["params"]["T2"] == pytest.approx(
        source["relaxation_fit_results"][0]["params"]["T2"])
    # Every tree entry that is a fit shows again when clicked (the report:
    # "could not be shown again ... no longer stored with this spectrum").
    fresh._refresh_experiment_tree()
    nodes = [fresh.experiment_tree.node(item) for item in fresh.experiment_tree.items()]
    actions = [n["action"][0] for n in nodes if n.get("action")]
    assert "slice_fit" in actions and "curve_fit" in actions, actions
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[1:3]))
    for node in nodes:
        if node.get("action"):
            fresh._on_tree_node_activated(node)
    assert not shown, shown


def test_sidecar_round_trip_for_a_normal_reopen(qapp, tmp_path):
    window = _fitted_window()
    spectrum = window.session.spectra[0]
    source = tmp_path / "exp"
    source.mkdir()
    spectrum.metadata.source_path = str(source)
    persistence.save_processing_sidecar(spectrum)
    assert os.path.isfile(persistence.results_sidecar_path(str(source)))
    back = persistence.load_results_sidecar(str(source))
    _same_results(spectrum.metadata.raw["slice_fit_results"], back["slice_fit_results"])


# ----------------------------------------------------------------------
# The reported case: a CSA fit applied to every slice, then Fit Curve
# ----------------------------------------------------------------------

from test_copy_fits_series_entry_browser_colours import (  # noqa: E402,F401
    window, _staged_series, _fit_curve_on)


def test_the_every_slice_model_and_its_curve_fit_reopen_from_a_state(window, tmp_path):
    index, series = _staged_series(window)
    curve = _fit_curve_on(window, index)
    assert curve
    folder = str(tmp_path / "t1.opennmr")
    window._save_state(folder, "t1", [index])

    fresh = MainWindow()
    fresh._shown = []
    opened, _messages = fresh._open_state(folder)
    new = opened[0]
    raw = fresh.session.spectra[new].metadata.raw
    assert len(raw["slice_fit_results"]) == len(series) and raw["slice_fit_results"][0]["model_fit"]
    _same_results(window.session.spectra[index].metadata.raw["relaxation_fit_results"],
                  raw["relaxation_fit_results"])

    shown = []
    from PySide6.QtWidgets import QMessageBox as _Box
    original = _Box.information
    _Box.information = staticmethod(lambda *a, **k: shown.append(a[1:3]))
    try:
        fresh._refresh_experiment_tree()
        nodes = {fresh.experiment_tree.node(i)["action"][0]: fresh.experiment_tree.node(i)
                 for i in fresh.experiment_tree.items() if fresh.experiment_tree.node(i).get("action")}
        assert {"series_fit", "curve_fit"} <= set(nodes), set(nodes)
        fresh._on_tree_node_activated(nodes["series_fit"])
        assert fresh.canvas_tabs.currentWidget() is fresh.spectral_fit_canvas
        fresh._on_tree_node_activated(nodes["curve_fit"])
    finally:
        _Box.information = original
    assert not shown, shown
    assert fresh._relaxation_fit_results[0]["params"]["T1"] == pytest.approx(
        curve[0]["params"]["T1"])
