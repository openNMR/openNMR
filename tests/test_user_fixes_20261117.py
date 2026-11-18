"""User reports, 2026-11-17: a stored CP build-up fit opens from Open
experiments (it used to say it was "no longer stored"), the search box is a
little wider with a larger button, typed axis limits belong to one experiment
whichever way it is drawn, and an openNMR state remembers the colours given
to the datasets in Compare."""
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.gui.main_window import MainWindow
from opennmr.io import serialize as io_serialize
from opennmr.io import state as io_state
from opennmr.notebook import cp_buildup as cpb

from test_gui import qapp  # noqa: F401 -- fixture
from test_gui_fit_sim_state import _quad_spectrum


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


# --- a kept CP build-up fit can be shown again ----------------------------------------

def _stored_cp_result():
    """What CpBuildupPanel._store_fits keeps: one record per region."""
    t = np.array([50., 100., 200., 400., 800., 1600., 3200.])
    y = cpb.cp_is(t, 1.0, 300.0, 5000.0)
    fit = cpb.fit_cp_buildup(t, y, "is")
    record = {"model": fit["model"], "region_ppm": [10.0, -10.0], "params": fit["params"],
              "errors": fit["errors"], "fixed": fit["fixed"], "t_us": t.tolist(),
              "y": y.tolist(), "y_err": np.full(t.size, 0.01).tolist(),
              "r_squared": fit["r_squared"], "t_max_us": fit["t_max_us"]}
    return {"Region 1": record}, fit


def test_series_and_fits_are_rebuilt_from_what_was_kept():
    results, fit = _stored_cp_result()
    series_list, fits = cpb.series_fits_from_stored(results)
    assert [s["name"] for s in series_list] == ["Region 1"]
    assert np.allclose(series_list[0]["t_us"], results["Region 1"]["t_us"])
    rebuilt = fits["Region 1"]
    assert rebuilt["equation"] == fit["equation"]
    # the curve is the fitted function again, not a copy that could go stale
    assert np.allclose(np.interp(fit["curve_t"], rebuilt["curve_t"], rebuilt["curve_y"]),
                       fit["curve_y"], rtol=1e-3, atol=1e-3)


def test_a_record_without_usable_data_gives_nothing():
    assert cpb.series_fits_from_stored({}) == ([], {})
    assert cpb.series_fits_from_stored({"Region 1": {"model": "nope"}}) == ([], {})


def test_clicking_a_cp_build_up_fit_in_open_experiments_shows_it(qapp):  # noqa: F811
    window = MainWindow()
    row = _quad_spectrum(window, "cp")
    results, _fit = _stored_cp_result()
    spectrum = window.session.spectra[row]
    spectrum.metadata.raw["fits"] = {"cp_buildup": {
        "label": "CP build-up", "target": "cp", "time": "", "inputs": {}, "inputs_text": "",
        "report": "report", "result": io_serialize.encode(results)}}
    window._select_single_row(row)
    window._show_tree_fit(row, ("stored_fit", "cp_buildup"))          # no ValueError
    assert window.fit_canvas.ax.lines or window.fit_canvas.ax.containers
    window.close()


def test_a_kept_result_with_nothing_to_redraw_says_what_it_holds(qapp):  # noqa: F811
    window = MainWindow()
    row = _quad_spectrum(window, "dnp")
    window.session.spectra[row].metadata.raw["fits"] = {
        "dnp_sweep": {"label": "DNP", "report": "enhancement 12.3", "result": None}}
    with pytest.raises(ValueError, match="enhancement 12.3"):
        window._show_tree_fit(row, ("stored_fit", "dnp_sweep"))
    window.close()


# --- search box ------------------------------------------------------------------------

def test_the_search_box_is_wider_and_its_button_larger(qapp):  # noqa: F811
    window = MainWindow()
    box = window.search_box
    assert box.edit.minimumWidth() >= 230 and box.edit.maximumWidth() >= 340
    assert box.button.minimumWidth() >= 40 and box.button.minimumHeight() >= 34
    window.close()


# --- typed limits belong to one experiment, however it is drawn --------------------------

def test_typed_limits_go_when_another_experiment_is_drawn_by_any_route(qapp):  # noqa: F811
    window = MainWindow()
    a = _quad_spectrum(window, "a")
    b = _quad_spectrum(window, "b")
    window._select_single_row(a)
    window.canvas.set_view_limits(x=(-20, 40))
    # a draw that is not a selection change (the route that used to keep them)
    window._draw_selected_spectrum(window.session.spectra[b], row=b)
    assert window.canvas.view_limits() == {"x": None, "y": None}
    # the same experiment drawn again keeps what was typed for it
    window.canvas.set_view_limits(x=(0, 30))
    window._draw_selected_spectrum(window.session.spectra[b], row=b)
    assert window.canvas.view_limits()["x"] == (0.0, 30.0)
    window._draw_selected_spectrum(window.session.spectra[a], row=a)
    assert window.canvas.view_limits()["x"] is None
    window.close()


def test_the_processing_preview_canvas_lets_go_too(qapp):  # noqa: F811
    window = MainWindow()
    a = _quad_spectrum(window, "a")
    b = _quad_spectrum(window, "b")
    window._select_single_row(a)
    canvas = window.processing_preview_canvas
    canvas._view_x = (-5.0, 5.0)
    window._draw_selected_spectrum(window.session.spectra[b], row=b)
    assert canvas.view_limits()["x"] is None
    window.close()


# --- an openNMR state remembers Compare colours ------------------------------------------

def test_the_state_file_keeps_compare_colours(tmp_path):
    from test_state_and_fit_models import _derived
    folder = str(tmp_path / "c.opennmr")
    io_state.save_state(folder, "c", [{"label": "x", "kind": "derived", "spectrum": _derived(),
                                       "compare_colours": {"spectrum": "#aa3300",
                                                           "slice 3": "#0044cc"}}])
    entry = io_state.load_state(folder)["entries"][0]
    assert entry["compare_colours"] == {"spectrum": "#aa3300", "slice 3": "#0044cc"}


def test_an_old_state_without_colours_still_opens(tmp_path):
    from test_state_and_fit_models import _derived
    folder = str(tmp_path / "old.opennmr")
    io_state.save_state(folder, "o", [{"label": "x", "kind": "derived", "spectrum": _derived()}])
    assert io_state.load_state(folder)["entries"][0].get("compare_colours") == {}


def test_colours_set_in_compare_are_saved_and_come_back(qapp, tmp_path):  # noqa: F811
    window = MainWindow()
    a = _quad_spectrum(window, "a")
    b = _quad_spectrum(window, "b")
    key_a, key_b = repr((a, None)), repr((b, None))
    window._prepare_compare_series([key_a, key_b], [(a, None), (b, None)])
    window.compare_canvas._assign_series("compare", [(key_a, "a", False), (key_b, "b", False)])
    # colours given on the plot are what the state then holds
    for style in window.compare_canvas._series["compare"]:
        if style.key == key_a:
            style.colour, style.colour_overridden = "#aa3300", True
    folder = str(tmp_path / "s.opennmr")
    window._save_state(folder, "s", [a, b])
    saved = [e["compare_colours"] for e in io_state.load_state(folder)["entries"]]
    assert saved == [{"spectrum": "#aa3300"}, {}]       # in the order saved: a, b
    window.close()

    later = MainWindow()
    opened, _messages = later._open_state(folder)
    colours = later._history_info(opened[0]).get("compare_colours")
    assert colours == {"spectrum": "#aa3300"}
    # and the next Compare draw hands it to the canvas as a colour set by hand
    key = repr((opened[0], None))
    later._prepare_compare_series([key], [(opened[0], None)])
    held = {s.key: s for s in later.compare_canvas._series["compare"]}
    assert held[key].colour == "#aa3300" and held[key].colour_overridden
    later.close()


def test_reset_colours_in_compare_also_forgets_the_remembered_ones(qapp):  # noqa: F811
    window = MainWindow()
    a = _quad_spectrum(window, "a")
    window._history_info(a)["compare_colours"] = {"spectrum": "#aa3300"}
    window._prepare_compare_series([repr((a, None))], [(a, None)])
    window._forget_compare_colours_shown()
    assert window._history_info(a)["compare_colours"] == {}
    window.close()


# --- Dynamics ▸ Value source: raw amplitude ----------------------------------------------

def test_value_source_offers_the_raw_amplitude_last_so_saved_indices_hold(qapp):  # noqa: F811
    window = MainWindow()
    combo = window.value_source_combo
    keys = [combo.itemData(i) for i in range(combo.count())]
    assert keys == ["area", "amplitude", "region", "amplitude_raw"]
    window.close()
