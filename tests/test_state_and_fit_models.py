"""2026-09-29 round 3: Czjzek field input; kept fit models (several fits
of one spectrum, saved with it, compared by AIC/BIC); openNMR states
(File ▸ Save / Open openNMR State). Synthetic data only."""
import json
import math
import os

import numpy as np
import pytest

from opennmr import io
from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.io import history as io_history
from opennmr.io import state as io_state
from opennmr.notebook import fit_models as fm
from opennmr.notebook import state as nb_state

from test_gui import qapp, _make_1d_dataset, _make_2d_dataset, _settle_sidecar_save  # noqa: F401


def _two_peaks(n=400, seed=0):
    x = np.linspace(10, 0, n)
    true = np.exp(-(x - 4) ** 2 / 0.3) + 0.6 * np.exp(-(x - 4.8) ** 2 / 0.3)
    y = true + np.random.default_rng(seed).normal(0, 0.02, n)
    one = 1.2 * np.exp(-(x - 4.3) ** 2 / 0.6)
    return x, y, true, one


# ----------------------------------------------------------------------
# Fit models (notebook)
# ----------------------------------------------------------------------

def test_statistics_match_a_hand_calculation():
    x, y, true, _one = _two_peaks()
    m = fm.make_fit_model("2", "isotropic", x, y, true, n_free=6)
    st = fm.fit_statistics(m)
    n, rss = x.size, float(np.sum((y - true) ** 2))
    assert st["n"] == n and st["rss"] == pytest.approx(rss)
    assert st["aic"] == pytest.approx(n * math.log(rss / n) + 12)
    assert st["bic"] == pytest.approx(n * math.log(rss / n) + 6 * math.log(n))
    assert st["reduced_chi2"] == pytest.approx(rss / (n - 6))
    assert fm.fit_statistics(fm.make_fit_model("?", "curve", x, y, true))["aic"] is None


def test_comparison_prefers_the_right_model_and_says_how_strongly():
    x, y, true, one = _two_peaks()
    c = fm.compare_fit_models([fm.make_fit_model("1 site", "isotropic", x, y, one, n_free=3),
                               fm.make_fit_model("2 sites", "isotropic", x, y, true, n_free=6)])
    rows = {r["name"]: r for r in c["rows"]}
    assert c["comparable"] and rows["2 sites"]["delta_bic"] == 0
    assert rows["1 site"]["delta_bic"] > 10 and "very strong" in rows["1 site"]["evidence"]
    assert "1 site" in fm.format_comparison(c)


def test_models_on_different_points_are_not_ranked():
    x, y, true, one = _two_peaks()
    mask = x > 3
    c = fm.compare_fit_models([fm.make_fit_model("a", "isotropic", x, y, one, n_free=3),
                               fm.make_fit_model("b", "isotropic", x, y, true, n_free=6,
                                                 mask=mask)])
    assert not c["comparable"] and "different points" in c["note"]
    assert all(r["delta_bic"] is None for r in c["rows"])


def test_encoding_is_exact_and_json_safe():
    x, y, true, _one = _two_peaks()
    m = fm.make_fit_model("2", "quadrupolar", x, y, true, components=[true, 0 * x], labels=["A"],
                          n_free=6, mask=x > 1)
    encoded = json.loads(json.dumps(fm.encode_fit_model(m)))
    back = fm.decode_fit_model(encoded)
    for key in ("x", "y", "y_fit", "mask"):
        np.testing.assert_array_equal(back[key], m[key])
    assert back["labels"] == ["A"] and back["n_free"] == 6
    with pytest.raises(ValueError):
        fm.make_fit_model("bad", "x", x, y[:-1], true)


def test_comparison_plot_has_residuals_and_a_ppm_axis():
    import matplotlib
    matplotlib.use("Agg")
    x, y, true, one = _two_peaks()
    fig = fm.plot_fit_comparison([fm.make_fit_model("1", "isotropic", x, y, one),
                                  fm.make_fit_model("2", "isotropic", x, y, true)])
    ax = fig.fit_comparison_axes["fit"]
    assert fig.fit_comparison_axes["residuals"] is not None
    assert ax.get_xlim()[0] > ax.get_xlim()[1]             # high ppm on the left
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Data", "1", "2"]


# ----------------------------------------------------------------------
# States (io / notebook)
# ----------------------------------------------------------------------

def _derived(label="peak"):
    x = np.linspace(10, 0, 128)
    s = NMRSpectrum(np.exp(-(x - 5) ** 2), [NMRAxis("frequency", ppm=x, hz=x * 400)],
                    NMRMetadata(vendor="derived", nucleus="1H", sfo1=400.0))
    m = fm.make_fit_model("1 peak", "isotropic", x, s.data, s.data, n_free=3)
    s.metadata.raw["fit_models"] = [fm.encode_fit_model({**m, "id": 1})]
    s.metadata.raw["phase0"] = 12.5
    return s


def test_state_round_trip_log_and_standalone_copies(tmp_path):
    folder = str(tmp_path / "a.opennmr")
    s = _derived()
    io_state.save_state(folder, "My test", [{"label": "peak", "kind": "derived", "spectrum": s}],
                        notes="about it")
    log = io_state.describe_state(folder)
    assert "openNMR state: My test" in log and "about it" in log
    assert "Phase PH0: 12.5°" in log and "“1 peak” – Solution NMR" in log
    info = io_state.load_state(folder)
    entry = info["entries"][0]
    np.testing.assert_array_equal(entry["spectrum"].data, s.data)
    assert entry["spectrum"].metadata.raw["phase0"] == 12.5
    # Each copy is an ordinary openNMR dataset folder.
    assert io.load(os.path.join(folder, entry["folder"])).metadata.vendor == "opennmr"


def test_state_refuses_to_overwrite_other_folders(tmp_path):
    other = tmp_path / "busy"
    other.mkdir()
    (other / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        io_state.save_state(str(other), "x", [{"label": "p", "kind": "derived", "spectrum": _derived()}])
    folder = str(tmp_path / "s.opennmr")
    io_state.save_state(folder, "x", [{"label": "p", "kind": "derived", "spectrum": _derived()}])
    with pytest.raises(FileExistsError):
        io_state.save_state(folder, "x", [{"label": "p", "kind": "derived", "spectrum": _derived()}])
    io_state.save_state(folder, "y", [{"label": "p", "kind": "derived", "spectrum": _derived()}],
                        overwrite=True)
    assert io_state.read_state_info(folder)["name"] == "y"
    with pytest.raises(ValueError):
        io_state.save_state(str(tmp_path / "empty.opennmr"), "x", [])


def test_notebook_state_functions(tmp_path):
    folder = tmp_path / "nb.opennmr"
    nb_state.save_state(folder, [("A peak", _derived()), _derived()], name="nb")
    items = nb_state.load_state(folder)
    assert [i["label"] for i in items][0] == "A peak"
    assert items[0]["fit_models"][0]["name"] == "1 peak"
    assert items[0]["kind"] == "derived"
    assert "nb" in nb_state.describe_state(folder)


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------

@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    from opennmr.gui.main_window import MainWindow
    monkeypatch.setenv("OPENNMR_STATES_DIR", str(tmp_path / "states"))
    return MainWindow()


def test_czjzek_panels_take_the_field(window):
    sim = window.czjzek_sim_panel
    sim.nucleus_combo.setCurrentText("27Al")
    sim.field_spin.setValue(18.8)
    assert sim.larmor_spin.value() == pytest.approx(208.74, abs=0.01)
    assert sim.larmor_spin.isReadOnly()
    fit = window.czjzek_fit_panel
    x = np.linspace(100, -100, 256)
    spectrum = NMRSpectrum(np.exp(-x ** 2 / 100), [NMRAxis("frequency", ppm=x, hz=x * 156.4)],
                           NMRMetadata(vendor="bruker", nucleus="27Al", sfo1=156.4))
    window._add_derived_spectrum(spectrum, "al")
    assert fit.larmor_spin.value() == pytest.approx(156.4)
    assert fit.field_spin.value() == pytest.approx(14.09, abs=0.01)     # a 600 MHz magnet
    assert window.quad_fit_panel.field_spin is not None


def _czjzek_glass(window):
    from opennmr.notebook import czjzek as cz
    x = np.linspace(200, -200, 1024)
    a = cz.czjzek_spectrum(2.5, 3.0, 130.3e6, x, delta_iso=60, fwhm=4)["real_spec"]
    b = cz.czjzek_spectrum(2.5, 1.5, 130.3e6, x, delta_iso=10, fwhm=6)["real_spec"]
    y = 1000 * a / a.max() + 400 * b / b.max() + np.random.default_rng(3).normal(0, 3, x.size)
    spectrum = NMRSpectrum(y, [NMRAxis("frequency", ppm=x, hz=x * 130.3)],
                           NMRMetadata(vendor="bruker", nucleus="27Al", sfo1=130.3))
    return window._add_derived_spectrum(spectrum, "glass")


def test_keep_compare_restore_rename_delete(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    index = _czjzek_glass(window)
    panel = window.czjzek_fit_panel
    panel.sites_table.item(0, 1).setText("40")
    panel.sites_table.item(0, 2).setText("2")
    panel.grid_cq_spin.setValue(32)
    panel.fit()
    m1 = window._keep_shown_fit(name="1 site")
    panel.add_site(("AlO6", "15", "1", "5", "3", "0.5"))
    panel.fit()
    m2 = window._keep_shown_fit(name="2 sites")
    assert (m1["n_free"], m2["n_free"]) == (4, 8)
    assert m1["kind"] == "czjzek" and m1["panel_result"] is not None
    # Saved with the spectrum, JSON-safe, and logged by name.
    raw = window.session.spectra[index].metadata.raw
    assert len(raw["fit_models"]) == 2
    json.dumps(io_history.tracked_state(window.session.spectra[index]))
    texts = [e.get("text", "") for e in window.session.info(index)["history"]["entries"]]
    assert any("Kept fit model “2 sites”" in t for t in texts)
    # Listed and compared.
    fp = window.fit_models_panel
    assert fp.table.rowCount() == 2
    fp.set_ticked([m1["id"], m2["id"]])
    assert "Lowest BIC: 2 sites" in fp.note_label.text()
    assert fp.table.item(0, 7).text() not in ("—", "0.0")
    fp.layout_combo.setCurrentIndex(fp.layout_combo.findData("summary"))
    fp.compare()
    assert window.fit_canvas.figure.fit_comparison_axes["residuals"] is not None
    # Put back: the panel holds the 1-site fit again, and it is current.
    window._restore_fit_model(index, m1["id"])
    assert panel.results_table.rowCount() == 1
    assert raw["fits"]["czjzek"]["result"] is not None
    # Rename and delete.
    window._rename_fit_model(index, m1["id"], "one site")
    assert window._fit_models(index)[0]["name"] == "one site"
    window._delete_fit_model(index, m2["id"])
    assert [m["name"] for m in window._fit_models(index)] == ["one site"]
    # Other spectrum selected: its own (empty) list.
    window._add_derived_spectrum(_derived(), "other")
    assert [m["name"] for m in window._fit_models(window._selected_index())] == ["1 peak"]


def test_keep_a_curve_fit_and_nothing_to_keep(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    window._add_derived_spectrum(_derived(), "d")
    assert window._keep_shown_fit(name="nothing") is None and shown
    t = np.geomspace(0.01, 1, 8)
    y = 1 - 2 * np.exp(-t / 0.2)
    window.fit_canvas.show_relaxation_fit({"t": t, "y": y, "y_fit": y, "popt": [1, 2, 0.2],
                                           "peak_label": "Peak 1"}, xlabel="Delay (s)",
                                          ylabel="Intensity")
    window.canvas_tabs.setCurrentWidget(window.fit_canvas)
    model = window._keep_shown_fit(name="T1")
    assert model["kind"] == "curve" and model["n_free"] == 3
    assert model["x_label"] == "Delay (s)"


def test_right_click_keep_this_fit(window, monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    from opennmr.gui.widgets.canvas import MplCanvas
    menus = []
    monkeypatch.setattr(MplCanvas, "_exec_context_menu", lambda self, m, p: menus.append(m) or None)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("from menu", True))
    window._add_derived_spectrum(_derived(), "d")
    x = np.linspace(0, 10, 100)
    window.fit_canvas.show_slice_fit(x, np.exp(-x), np.exp(-x), components=[np.exp(-x), 0 * x],
                                     peak_labels=["A"])
    window.fit_canvas._on_context_menu(window.fit_canvas.rect().center())
    keep = [a for a in menus[-1].actions() if a.text() == "Keep This Fit…"][0]
    keep.trigger()
    assert window._fit_models(window._selected_index())[-1]["name"] == "from menu"


def test_parse_rows():
    from opennmr.gui.state_dialog import parse_rows
    assert parse_rows("1, 4-6; 3", 8) == [1, 3, 4, 5, 6]
    assert parse_rows("", 8) == []
    with pytest.raises(ValueError):
        parse_rows("9", 8)
    with pytest.raises(ValueError):
        parse_rows("a-b", 8)


def test_save_and_open_state_with_experiments_slices_and_kept_fits(window, tmp_path, monkeypatch):
    from opennmr.gui.main_window import MainWindow
    # Processed-data fixtures (the raw-FID ones patch the readers
    # globally, so a raw 1D and a raw 2D cannot share a test).
    p1 = _make_1d_dataset(tmp_path, n=64, name="one")
    p2 = _make_2d_dataset(tmp_path, n_f1=4, n_f2=16, name="two")
    window._load(p1, vendor="bruker")
    window.phase0_handle.setValue(30.0)
    window._capture_phase_reference()
    _settle_sidecar_save(window)
    window._load(p2, vendor="bruker")
    d = window._add_derived_spectrum(_derived(), "kept peak")
    parent = window.session.spectra[1]
    items = {i["index"]: i for i in window._state_items()}
    assert items[0]["kind"] == "Experiment" and items[2]["kind"] == "Derived spectrum"
    assert items[1]["rows"] == parent.data.shape[0] and "1 kept fit" in items[2]["detail"]
    folder = str(tmp_path / "work.opennmr")
    window._save_state(folder, "work", [0, 1, d], slices={1: [1, 3]}, notes="for the paper")
    info = io_state.read_state_info(folder)
    kinds = [e["kind"] for e in info["entries"]]
    assert kinds == ["experiment", "experiment", "derived", "slice", "slice"]
    assert info["entries"][0]["folder"]                          # a 1D always has a copy
    # This 2D fixture is processed data only (no ser): kept as a copy and
    # opened from it, since re-reading cannot re-apply its processing.
    assert info["entries"][1]["folder"] and info["entries"][1]["prefer_copy"]
    assert any("Saved in openNMR state “work”" in e.get("text", "")
               for e in window.session.info(0)["history"]["entries"])

    fresh = MainWindow()
    opened, messages = fresh._open_state(folder)
    assert len(opened) == 5
    assert fresh.session.spectra[opened[0]].metadata.raw.get("phase0") == pytest.approx(30.0)
    assert "reprocessed as saved" in messages[0]
    assert fresh.session.spectra[opened[1]].ndim == 2
    assert "opened from the saved copy" in messages[1]
    np.testing.assert_allclose(fresh.session.spectra[opened[1]].data, parent.data)
    # The 1D (processed data only) reopens phased as it was saved.
    np.testing.assert_allclose(fresh.session.spectra[opened[0]].data, window.session.spectra[0].data)
    assert fresh._fit_models(opened[2])[0]["name"] == "1 peak"
    np.testing.assert_allclose(fresh.session.spectra[opened[3]].data, parent.data[0])
    np.testing.assert_allclose(fresh.session.spectra[opened[4]].data, parent.data[2])

    # The 1D experiment moved away: its saved copy opens instead, and says so.
    os.rename(p1, p1 + "_moved")
    later = MainWindow()
    _opened, messages = later._open_state(folder)
    assert "no longer at" in messages[0] and "saved processed copy" in messages[0]
    np.testing.assert_allclose(later.session.spectra[0].data, window.session.spectra[0].data)


def test_file_open_on_a_state_folder_opens_the_state(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    folder = str(tmp_path / "st.opennmr")
    io_state.save_state(folder, "st", [{"label": "p", "kind": "derived", "spectrum": _derived()}])
    window._open_directory(folder)
    assert len(window.session) == 1 and "Opened 1 spectra" in shown[0]
    assert window.session.spectra[0].metadata.raw["phase0"] == 12.5


def test_a_phased_processed_only_1d_reopens_phased(qapp, tmp_path):
    # Found while building states: phasing a 1D with no FID (TopSpin's
    # processed data only) and reopening failed outright ("unexpected
    # keyword argument 'phase0'"), and once that was fixed, reopened
    # unphased under a 30° phase setting.
    from opennmr.gui.main_window import MainWindow
    path = _make_1d_dataset(tmp_path, n=64, name="processed")
    first = MainWindow()
    first._load(path, vendor="bruker")
    first.phase0_handle.setValue(30.0)
    first._capture_phase_reference()
    _settle_sidecar_save(first)
    again = MainWindow()
    again._load(path, vendor="bruker")
    assert len(again.session) == 1
    assert again.session.spectra[0].metadata.raw["phase0"] == pytest.approx(30.0)
    np.testing.assert_allclose(again.session.spectra[0].data, first.session.spectra[0].data)
