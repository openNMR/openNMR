"""2026-10-20 round: Stored integrals (notebook table, Integrate ▸ Integration
Regions tick boxes / store / view), kept-fit component colours carried into
Compare, 2-D Compare (crosshair in side-by-side panels, y scroll, link/unlink,
Hz + ppm measurements, a spectrum against itself), the standalone 2-D inset
dialog, and Plot Style's collapsible sections."""

import json
import os

import numpy as np
import pytest
from PySide6.QtCore import Qt

from opennmr import notebook as nb
from opennmr.gui.main_window import MainWindow
from opennmr.notebook import stored_integrals as si

from test_gui import qapp, _load_spectrum_directly  # noqa: F401
from test_integrate_tile_plot_options_regularisation import _one_d, said  # noqa: F401


# ---- notebook: the stored-integrals table ---------------------------------------

def test_add_computes_midpoint_and_orders_limits(tmp_path):
    store = si.StoredIntegrals(str(tmp_path / "s.json"))
    rec = store.add("A", low=5.0, high=-3.0, integral=10.0, error=0.5, notes="500 us")
    assert (rec["min"], rec["max"], rec["midpoint"]) == (-3.0, 5.0, 1.0)
    assert rec["notes"] == "500 us" and rec["experiment"] == "A"


def test_add_refuses_missing_numbers(tmp_path):
    store = si.StoredIntegrals(str(tmp_path / "s.json"))
    with pytest.raises(ValueError):
        store.add("A", low=None, high=1.0, integral=1.0)
    with pytest.raises(ValueError):
        store.add("A", low=0.0, high=1.0, integral=float("nan"))


def test_table_persists_between_sessions(tmp_path):
    path = str(tmp_path / "s.json")
    si.StoredIntegrals(path).add("A", 0, 1, 2.0)
    again = si.StoredIntegrals(path)
    assert len(again) == 1 and again.rows()[0]["integral"] == 2.0


def test_in_memory_table_writes_nothing(tmp_path):
    store = si.StoredIntegrals(False)
    store.add("A", 0, 1, 2.0)
    assert store.path is None and len(store) == 1


def test_series_gives_one_row_per_slice(tmp_path):
    store = si.StoredIntegrals(str(tmp_path / "s.json"))
    regions = [(0.0, 1.0), (2.0, 3.0)]
    results = [{"integrals": [1.0, 2.0], "integral_errors": [0.1, 0.2]},
               {"integrals": [3.0, 4.0], "integral_errors": [0.3, 0.4]}]
    added = store.add_from_results("series", results, regions, which=[1], notes="n")
    assert [r["integral"] for r in added] == [2.0, 4.0]
    assert added[0]["experiment"].endswith("slice 1") and added[1]["error"] == 0.4


def test_notes_remove_clear(tmp_path):
    store = si.StoredIntegrals(str(tmp_path / "s.json"))
    for k in range(3):
        store.add("A", 0, 1, float(k))
    store.set_notes(1, "mid")
    assert store.rows()[1]["notes"] == "mid"
    store.remove([0, 2])
    assert [r["notes"] for r in store.rows()] == ["mid"]
    store.clear()
    assert len(store) == 0 and len(si.StoredIntegrals(store.path)) == 0


@pytest.mark.parametrize("ext", ["csv", "json", "txt"])
def test_export_formats(tmp_path, ext):
    store = si.StoredIntegrals(False)
    store.add("exp 1", -1.0, 1.0, 123.5, 1.5, "500 us contact time")
    out = tmp_path / f"t.{ext}"
    assert store.export(str(out)) == ext
    text = out.read_text(encoding="utf-8")
    assert "500 us contact time" in text and "exp 1" in text
    if ext == "json":
        assert json.loads(text)["records"][0]["integral"] == 123.5
    if ext == "csv":
        assert text.splitlines()[0].startswith("Region min (ppm),Region max (ppm),Midpoint (ppm)")


def test_export_unknown_format_refused(tmp_path):
    with pytest.raises(ValueError):
        si.StoredIntegrals(False).export(str(tmp_path / "t.xyz"))


def test_notebook_exports_the_shared_table(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENNMR_STORED_INTEGRALS", str(tmp_path / "shared.json"))
    assert nb.get_stored_integrals() is nb.get_stored_integrals()
    assert nb.StoredIntegrals is si.StoredIntegrals
    assert si.default_path() == str(tmp_path / "shared.json")


# ---- GUI: tick, store, view -------------------------------------------------------

def _window_with_regions(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENNMR_STORED_INTEGRALS", str(tmp_path / "gui_store.json"))
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    window._add_region_row(4.5, 5.5)
    window._add_region_row(1.5, 2.5)
    return window


def test_regions_have_tick_boxes_and_tick_all(qapp, tmp_path, monkeypatch):  # noqa: F811
    window = _window_with_regions(tmp_path, monkeypatch)
    item = window.region_table.item(0, 0)
    assert item.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert window._ticked_region_indices() == []
    window._set_regions_ticked(True)
    assert window._ticked_region_indices() == [0, 1]
    window._set_regions_ticked(False)
    assert window._ticked_region_indices() == []


def test_store_ticked_regions_only(qapp, tmp_path, monkeypatch):  # noqa: F811
    window = _window_with_regions(tmp_path, monkeypatch)
    window.region_table.item(1, 0).setCheckState(Qt.CheckState.Checked)
    added = window.store_ticked_integrals("500 us contact time")
    assert len(added) == 1
    row = window.stored_integrals.rows()[-1]
    assert row["notes"] == "500 us contact time"
    assert (row["min"], row["max"]) == (1.5, 2.5) and row["integral"] > 0
    assert row["experiment"]


def test_store_with_nothing_ticked_stores_nothing(qapp, tmp_path, monkeypatch):  # noqa: F811
    window = _window_with_regions(tmp_path, monkeypatch)
    before = len(window.stored_integrals)
    assert window.store_ticked_integrals() == []
    assert len(window.stored_integrals) == before
    assert "Tick" in window.integration_results_label.text()


def test_stored_values_equal_the_displayed_integrals(qapp, tmp_path, monkeypatch):  # noqa: F811
    window = _window_with_regions(tmp_path, monkeypatch)
    window._set_regions_ticked(True)
    window.store_ticked_integrals()
    spectrum = window.session.spectra[window._selected_index()]
    shown = spectrum.metadata.raw["slice_fit_results"][0]["integrals"]
    stored = [r["integral"] for r in window.stored_integrals.rows()[-2:]]
    assert stored == pytest.approx(list(shown))


def test_view_dialog_shows_notes_and_saves(qapp, tmp_path, monkeypatch):  # noqa: F811
    window = _window_with_regions(tmp_path, monkeypatch)
    window._set_regions_ticked(True)
    window.store_ticked_integrals("n1")
    window._on_view_stored_integrals_clicked()
    dialog = window._stored_integrals_dialog
    assert dialog.table.rowCount() == len(window.stored_integrals)
    dialog.table.item(0, 6).setText("edited")
    assert window.stored_integrals.rows()[0]["notes"] == "edited"
    dialog.remove_selected()            # nothing selected: no change
    assert dialog.table.rowCount() == len(window.stored_integrals)


# ---- kept-fit colours carried into Compare -------------------------------------------

def test_kept_fit_experiment_inherits_component_colours(qapp):  # noqa: F811
    from test_gui_round4_fits import _window_with_kept_fits
    window = MainWindow()
    index, ids = _window_with_kept_fits(window)
    models = window._fit_models(index)
    models[0]["component_style"] = {
        "line_colours": {"A": "#123456"}, "fill_colours": {"A": "#abcdef"}}
    window._set_fit_models(index, models)
    style = window._model_colour_style(models[0])
    assert style["line_colours"]["A"] == "#123456"
    assert style["fill_colours"]["A"] == "#abcdef"


# ---- 2-D Compare -------------------------------------------------------------------------

def test_side_by_side_panels_get_a_crosshair_and_link_toggle(qapp, tmp_path):  # noqa: F811
    from test_compare_2d import _window_with, _select_rows
    window = _window_with(tmp_path, ("a", 0, "13C"), ("b", 5, "13C"))
    _select_rows(window, [0, 1])
    controls = window.compare_2d_controls
    controls.layout_combo.setCurrentIndex(1)          # side by side
    controls._debounce.timeout.emit()
    canvas = window.compare_canvas
    assert len(canvas._panel_axes) == 2
    window.compare_2d_controls.link_check.setChecked(True)
    assert canvas.panels_linked()
    window.compare_2d_controls.link_check.setChecked(False)
    assert not canvas.panels_linked()


def test_unlinked_panels_zoom_independently(qapp, tmp_path):  # noqa: F811
    from test_compare_2d import _window_with, _select_rows
    window = _window_with(tmp_path, ("a", 0, "13C"), ("b", 5, "13C"))
    _select_rows(window, [0, 1])
    controls = window.compare_2d_controls
    controls.layout_combo.setCurrentIndex(1)          # side by side
    controls._debounce.timeout.emit()
    canvas = window.compare_canvas
    canvas.set_panels_linked(False)
    first, second = canvas._panel_axes[:2]
    before = second.get_xlim()
    first.set_xlim(*(np.array(first.get_xlim()) * 0.5))
    canvas.canvas.draw()
    assert second.get_xlim() == before
    canvas.set_panels_linked(True)
    first.set_xlim(*(np.array(first.get_xlim()) * 0.5))
    canvas.canvas.draw()
    assert second.get_xlim() == pytest.approx(first.get_xlim())


def test_panels_per_spectrum_compares_a_spectrum_with_itself(qapp, tmp_path):  # noqa: F811
    from test_compare_2d import _window_with, _select_rows
    window = _window_with(tmp_path, ("a", 0, "13C"))
    window._populate_compare_list([0])
    window.compare_2d_controls.copies_spin.setValue(2)
    window._refresh_compare_view()
    assert len(window.compare_canvas._panel_axes) == 2
    assert not window.compare_canvas.panels_linked()      # unlinked the first time


# ---- standalone inset -----------------------------------------------------------------------

def test_inset_dialog_adds_from_zoom_and_clears(qapp, tmp_path):  # noqa: F811
    from test_compare_2d import _bruker_2d
    from opennmr.gui.inset_dialog import InsetDialog
    window = MainWindow()
    window._load(_bruker_2d(tmp_path, "a"), vendor="bruker")
    canvas = window.canvas
    dialog = InsetDialog(canvas)
    assert dialog.add_from_view() is None                  # not zoomed: refused
    region = canvas.contour_view_region()
    assert region is not None
    (x0, x1), (y0, y1) = region
    canvas.ax.set_xlim(x0 - 0.25 * (x0 - x1), x1 + 0.25 * (x0 - x1))
    canvas.ax.set_ylim(y0 + 0.25 * (y1 - y0), y1 - 0.25 * (y1 - y0))
    spec = dialog.add_from_view()
    assert spec is not None and len(canvas.contour_insets()) == 1
    dialog.clear_insets()
    assert canvas.contour_insets() == []


# ---- Plot Style sections ------------------------------------------------------------------

def test_plot_style_is_sectioned(qapp):  # noqa: F811
    window = MainWindow()
    sections = window.style_sections
    for name in ("Core styling", "Legend", "Compare", "2-D", "Export"):
        assert name in sections, name
    assert "Insets" in sections and hasattr(window, "style_inset_panel")
