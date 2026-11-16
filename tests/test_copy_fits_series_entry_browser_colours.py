"""2026-11-16: (1) the "model, every slice" entry in Open experiments opens,
has a table of values and a name that says what it is; (2) a fit is copied
to another experiment -- a curve fit (T1/T2) to another profile, a lineshape
fit to another spectrum; (3) the result buttons of the fit panels are
grouped; (4) Browse a folder colours 1D and 2D experiments, set by
right-click."""

import json

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QGroupBox, QMenu, QMessageBox

from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets import data_browser as browser_module
from opennmr.gui.widgets.data_browser import DataBrowser
from opennmr.notebook import experiment_tree as et
from opennmr.notebook import fit_transfer as ft
from opennmr.notebook import series_fitting as SF
from opennmr.processing.relaxation import NMR2D_fit_relaxation

from test_gui_fit_sim_state import _fit_quad, _quad_spectrum
from test_series_fitting import (DELAYS, T1_A, T1_B, _csa_reference, _ir_csa_series, qapp)  # noqa: F401


# ---------------------------------------------------------------------------
# The series entry's name and its place in the tree (no GUI)
# ---------------------------------------------------------------------------

def test_the_every_slice_entry_is_named_for_what_it_is():
    assert et.series_fit_label("Solid-state NMR model, every slice", 12) == \
        "Solid-state NMR model applied to every slice, 12 slices"
    assert et.series_fit_label("CSA model, every slice") == "CSA model applied to every slice"
    assert et.series_fit_label("", 1) == "Model applied to every slice"


def test_the_tree_opens_the_every_slice_entry_as_per_slice_fits_not_a_panel():
    raw = {"fits": {"series_combined": {"label": "Solid-state NMR model, every slice",
                                        "target": "t", "time": "now",
                                        "inputs": {"slices": 9}, "result": None}}}
    nodes = et.build_experiment_tree([{"row": 0, "label": "a", "entry_id": "s-e1", "raw": raw}])
    child = nodes[0]["children"][0]
    assert child["action"] == ("series_fit", "combined")
    assert child["label"] == "Solid-state NMR model applied to every slice, 9 slices"
    assert "held fixed" in child["tooltip"]
    assert child["type_label"] == "Fit"


# ---------------------------------------------------------------------------
# Copying a curve fit (no GUI)
# ---------------------------------------------------------------------------

def _series_and_fits():
    x, spectrum = _ir_csa_series()
    fit = _csa_reference(spectrum, row=len(DELAYS) - 1)
    labels, xc, comps = SF.model_components("csa", fit)
    series = SF.fit_series_with_lineshapes(spectrum, xc, comps, labels, kind="csa")
    fits = NMR2D_fit_relaxation(series, model="inversion_recovery", vdlist=DELAYS,
                                value_source="area", show_report=False)
    return series, fits


def test_a_curve_fits_start_is_its_fitted_values_bounds_and_fixes():
    _series, fits = _series_and_fits()
    starts = ft.curve_fit_start(fits)
    assert [s["peak_label"] for s in starts] == ["A", "B"]
    assert starts[0]["model"] == "inversion_recovery"
    assert starts[0]["p0"]["T1"] == pytest.approx(fits[0]["params"]["T1"])
    assert starts[0]["fixed"] == {}
    assert set(starts[0]["bounds"]) == set(starts[0]["param_names"])
    # What was held fixed is known by the name being there -- also at 0.
    held = [dict(fits[0], fixed={"M0": 0.0})]
    assert ft.curve_fit_start(held)[0]["fixed"] == {"M0": True}
    assert ft.curve_fit_start(None) == [] and ft.curve_fit_start([5]) == []


def test_bounds_are_widened_never_narrowed_to_hold_the_copied_value():
    assert ft.widen_bounds(0.5, 0.0, 1.0) == (0.0, 1.0)
    assert ft.widen_bounds(5.0, 0.0, 1.0) == (0.0, 50.0)
    assert ft.widen_bounds(-5.0, -1.0, 1.0) == (-50.0, 1.0)
    low, high = ft.widen_bounds(0.001, 0.01, 1.0)
    assert low == pytest.approx(0.0001) and high == 1.0


def test_the_guess_table_rows_are_in_the_display_unit_with_fixes_marked():
    start = {"p0": {"T1": 0.15, "M0": 2.0}, "bounds": {"T1": (0.001, 100.0)},
             "fixed": {"M0": True}, "param_names": ["M0", "T1", "Other"]}
    rows = ft.guess_rows(start, ["M0", "T1", "Other"],
                         scale=lambda name: 1e-3 if name == "T1" else 1.0,
                         auto={"M0": (-10.0, 10.0)})
    # "Other" was never fitted: the table keeps its automatic row.
    assert [r[0] for r in rows] == ["M0", "T1"]
    assert rows[0] == ["M0", "2", "-10", "10", "1"]
    assert rows[1] == ["T1", "150", "1", "100000", "0"]


def test_a_curve_fit_is_refitted_on_another_profile_from_the_copied_values():
    series, fits = _series_and_fits()
    # The same measured values against delays twice as long: T1 doubles.
    copy = ft.refit_curve_like(fits, series, DELAYS * 2, value_source="area")
    assert [c["peak_label"] for c in copy] == ["A", "B"]
    assert copy[0]["params"]["T1"] == pytest.approx(2 * T1_A, rel=0.04)
    assert copy[1]["params"]["T1"] == pytest.approx(2 * T1_B, rel=0.04)
    # It started from the source's fitted values, not the model's automatic ones.
    assert copy[0]["p0"][copy[0]["param_names"].index("T1")] == \
        pytest.approx(fits[0]["params"]["T1"])
    rows = ft.compare_curve_fits(fits, copy)
    t1 = next(r for r in rows if r["peak"] == "A" and r["parameter"] == "T1")
    assert t1["copy"] == pytest.approx(2 * t1["source"], rel=0.04) and t1["copy_error"] > 0
    with pytest.raises(ValueError):
        ft.refit_curve_like([], series, DELAYS)


def test_a_held_fixed_parameter_stays_fixed_in_the_copy():
    series, fits = _series_and_fits()
    m0 = fits[0]["params"]["M0"]
    held = [dict(f, fixed={"M0": m0}) for f in fits]
    copy = ft.refit_curve_like(held, series, DELAYS, value_source="area")
    assert copy[0]["params"]["M0"] == m0 and "M0" not in copy[0]["free_names"]
    free = ft.refit_curve_like(held, series, DELAYS, value_source="area", keep_fixed=False)
    assert "M0" in free[0]["free_names"]


def test_a_panels_start_is_a_copy_with_the_fitted_sites_in_place():
    inputs = {"widgets": {"x": 1}, "sites": [{"label": "old"}]}
    start = ft.slice_fit_start(inputs, [{"label": "fitted"}])
    assert start["sites"] == [{"label": "fitted"}] and inputs["sites"] == [{"label": "old"}]
    assert ft.slice_fit_start(inputs)["sites"] == [{"label": "old"}]
    assert "copied from a to b" in ft.describe_transfer("CSA fit", "a", "b")


# ---------------------------------------------------------------------------
# In the window
# ---------------------------------------------------------------------------

@pytest.fixture
def window(qapp, monkeypatch, tmp_path):  # noqa: F811
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    monkeypatch.setattr(MainWindow, "_UI_SETTINGS_PATH", str(tmp_path / "ui_settings.json"))
    w = MainWindow()
    w._shown = shown
    return w


def _staged_series(window, name="13C T1 series"):
    """An inversion-recovery pseudo-2D with a lineshape model applied to every
    slice and staged, as the GUI leaves it."""
    _x, spectrum = _ir_csa_series()
    index = window._add_derived_spectrum(spectrum, name)
    spectrum = window.session.spectra[index]
    fit = _csa_reference(spectrum, row=len(DELAYS) - 1)
    labels, xc, comps = SF.model_components("csa", fit)
    series = SF.fit_series_with_lineshapes(spectrum, xc, comps, labels, kind="csa",
                                           reference_row=len(DELAYS))
    window._store_slice_results(spectrum, series)
    spectrum.metadata.raw["fits"] = {"series_csa": {
        "label": "CSA model, every slice", "target": name, "time": "now",
        "inputs": {"slices": len(series)}, "result": None, "report": SF.series_fit_report(series)}}
    return index, series


def _node(window, key):
    window._refresh_experiment_tree()
    item = window.experiment_tree.find(key)
    assert item is not None, [window.experiment_tree.node(i)["key"]
                              for i in window.experiment_tree.items()]
    return window.experiment_tree.node(item)


def test_the_every_slice_entry_opens_and_shows_each_slice(window):
    index, series = _staged_series(window)
    node = _node(window, f"{window.session.entry_id(index)}:fit:series_csa")
    assert node["label"] == f"CSA model applied to every slice, {len(series)} slices"
    window._on_tree_node_activated(node)
    assert not window._shown, window._shown              # the dialog of old is gone
    assert window.canvas_tabs.currentWidget() is window.spectral_fit_canvas
    text = window.spectral_fit_params_label.text()
    assert "A: area" in text and "B: area" in text and "R²" in text
    assert window.view_slice_fit_spin.maximum() == len(series)
    # Browsing the slices draws each one's own model fit.
    window.view_slice_fit_spin.setValue(3)
    assert "A: area" in window.spectral_fit_params_label.text()
    assert window._slice_fit_results[2]["index"] == 3
    assert not window._shown, window._shown


def test_the_every_slice_entry_has_a_table_and_can_be_restored(window):
    index, series = _staged_series(window)
    node = _node(window, f"{window.session.entry_id(index)}:fit:series_csa")
    record = window._tree_fit_record(node)
    heads = list(record["values"])
    assert heads[0] == "Slice 1, A" and len(heads) == 2 * len(series)
    assert set(record["values"]["Slice 1, A"]) == {"Area", "Relative amplitude", "R²"}
    assert record["values"]["Slice 1, A"]["Area"]["error"] > 0
    assert window._restore_tree_fit(node) is True
    assert not window._shown, window._shown
    # Replaced by a later spectral fit: said plainly, not a crash.
    window.session.spectra[index].metadata.raw["slice_fit_results"] = [
        {"index": 1, "integrals": [1.0]}]                   # an integration took their place
    window._on_tree_node_activated(node)
    assert window._shown and "no longer stored" in window._shown[-1][1]


def _fit_curve_on(window, index):
    window._select_single_row(index)
    window.delay_source_combo.setCurrentIndex(1)                  # manual list
    window.delay_unit_combo.setCurrentIndex(window.delay_unit_combo.findText("s"))
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in DELAYS))
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("inversion_recovery"))
    window.value_source_combo.setCurrentIndex(window.value_source_combo.findData("area"))
    window._on_fit_curve_button_clicked()
    return window._relaxation_fit_results


def test_a_curve_fit_is_copied_to_another_profile_as_its_starting_guess(window):
    source, _ = _staged_series(window, "T1 at 300 K")
    results = _fit_curve_on(window, source)
    t1_source = results[0]["params"]["T1"]
    target, _ = _staged_series(window, "T1 at 320 K")
    node = {"row": source, "action": ("curve_fit",), "label": "Curve fit"}
    assert window._copy_fit_kind(node) == ("profile", None)
    targets = [row for _label, row in window._copy_fit_targets("profile", source)]
    assert targets == [target]                         # not itself, and only measured series

    # The target's own delays and unit are set first: the guess table is in the
    # unit shown, so the unit has to be chosen before the values are copied in.
    window._select_single_row(target)
    window.delay_source_combo.setCurrentIndex(1)
    window.delay_unit_combo.setCurrentIndex(window.delay_unit_combo.findText("s"))
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in DELAYS))
    window._copy_curve_fit_to(source, target)
    assert window._selected_index() == target
    assert window.relaxation_model_combo.currentData() == "inversion_recovery"
    assert window.guess_group.isChecked()
    names = [window.guess_table.item(r, 0).text() for r in range(window.guess_table.rowCount())]
    row = names.index("T1")
    shown = float(window.guess_table.item(row, 1).text())
    assert shown * window._param_display_scale("T1") == pytest.approx(t1_source, rel=1e-4)
    # Fitting now starts from it.
    window._on_fit_curve_button_clicked()
    new = window._relaxation_fit_results
    assert new[0]["p0"][new[0]["param_names"].index("T1")] == pytest.approx(t1_source, rel=1e-4)
    assert new[0]["params"]["T1"] == pytest.approx(t1_source, rel=0.03)
    history = window._ensure_history(target)["entries"]
    assert any(e["kind"] == "fit_copy" and "Curve fit copied from" in e["text"] for e in history)
    assert not window._shown, window._shown


def test_a_curve_fit_can_be_copied_and_fitted_in_one_go(window):
    source, _ = _staged_series(window, "T1 a")
    _fit_curve_on(window, source)
    target, _ = _staged_series(window, "T1 b")
    window._select_single_row(target)
    window.delay_source_combo.setCurrentIndex(1)
    window.delay_unit_combo.setCurrentIndex(window.delay_unit_combo.findText("s"))
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in DELAYS))
    window._copy_curve_fit_to(source, target, run=True)
    assert window._relaxation_fit_results is not None
    assert window.session.spectra[target].metadata.raw["relaxation_fit_results"]
    assert not window._shown, window._shown


def test_copying_to_an_unmeasured_series_says_why(window):
    source, _ = _staged_series(window)
    _fit_curve_on(window, source)
    spectrum = _ir_csa_series()[1]
    bare = window._add_derived_spectrum(spectrum, "not measured")
    assert window._copy_fit_targets("profile", source) == []
    with pytest.raises(ValueError, match="not been measured"):
        window._copy_curve_fit_to(source, bare)


def test_a_lineshape_fit_is_copied_to_another_spectrum_as_starting_values(window):
    source = _quad_spectrum(window, "27Al first")
    panel = _fit_quad(window)
    fitted = panel.fitted_site_values()
    target = _quad_spectrum(window, "27Al second")
    node = {"row": source, "action": ("stored_fit", "quadrupolar"), "label": "Quadrupolar fit"}
    assert window._copy_fit_kind(node) == ("slice", "quadrupolar")
    assert target in [row for _label, row in window._copy_fit_targets("slice", source)]

    window._copy_slice_fit_to(source, "quadrupolar", target)
    assert window._selected_index() == target
    column = {name: col for col, name in panel.COLUMN_PARAMS.items()}["C_Q"]
    assert float(panel.sites_table.item(0, column).text()) == \
        pytest.approx(abs(fitted[0]["C_Q"]), rel=1e-5)
    history = window._ensure_history(target)["entries"]
    assert any(e["kind"] == "fit_copy" for e in history)
    # The source keeps its own fit.
    assert window.session.spectra[source].metadata.raw["fits"]["quadrupolar"]


def test_an_entry_with_nothing_to_copy_is_refused(window):
    assert window._copy_fit_kind({"action": None}) == (None, None)
    assert window._copy_fit_kind({"action": ("stored_fit", "multi")}) == (None, None)
    assert window._copy_fit_kind({"action": ("series_fit", "csa")}) == (None, None)
    assert window._on_copy_fit_to({"row": 0, "action": ("fit_model", 1)}) is False
    assert window._shown and "cannot be copied" in window._shown[-1][1]


# ---------------------------------------------------------------------------
# The result buttons are grouped
# ---------------------------------------------------------------------------

def test_each_fit_sections_result_buttons_share_one_labelled_group(window):
    assert set(window.copy_fit_buttons) == {"jcoupling", "csa", "quadrupolar", "czjzek",
                                            "combined", "isotropic"}
    for key in ("csa", "quadrupolar", "combined", "jcoupling"):
        keep = window.keep_fit_buttons[key]
        box = keep.parentWidget()
        assert isinstance(box, QGroupBox) and box.title() == "This fit"
        assert window.save_fit_values_buttons[key].parentWidget() is box
        assert window.copy_fit_buttons[key].parentWidget() is box
    csa = window.keep_fit_buttons["csa"].parentWidget()
    assert window.csa_use_fitted_button.parentWidget() is csa
    box = window.keep_fit_buttons["isotropic"].parentWidget()
    for button in (window.spectral_fit_correlation_button, window.export_fit_report_button,
                   window.export_isotropic_pseudo2d_button, window.copy_fit_buttons["isotropic"]):
        assert button.parentWidget() is box


def test_the_dynamics_result_buttons_share_one_labelled_group(window):
    box = window.curve_fit_plot_box.keep_button.parentWidget()
    assert isinstance(box, QGroupBox) and box.title() == "This curve fit"
    for button in (window.copy_curve_fit_button, window.export_curve_fit_button,
                   window.curve_fit_correlation_button):
        assert button.parentWidget() is box
    # Nothing fitted yet: nothing to copy.
    assert not window.copy_curve_fit_button.isEnabled()
    window._on_copy_current_curve_fit()
    assert window._shown and "Run Fit Curve first" in window._shown[-1][1]


# ---------------------------------------------------------------------------
# Browse a folder: 1D and 2D typefaces
# ---------------------------------------------------------------------------

_INFO = {"one": {"ndim": 1, "nucleus": "1H", "shape": (4096,)},
         "two": {"ndim": 2, "pseudo_2d": False, "nucleus": "1H", "shape": (256, 1024)},
         "series": {"ndim": 2, "pseudo_2d": True, "nucleus": "23Na", "shape": (16, 2048)}}


@pytest.fixture
def browser(qapp, tmp_path, monkeypatch):  # noqa: F811
    def peek(path, vendor=None):
        return dict(_INFO[path.replace("\\", "/").rsplit("/", 1)[-1]])

    monkeypatch.setattr(browser_module.io, "peek", peek)
    matches = []
    for name in _INFO:
        (tmp_path / name).mkdir()
        matches.append((str(tmp_path / name), "bruker"))
    b = DataBrowser()
    b.resize(400, 400)
    b.show()
    b.set_matches(str(tmp_path), matches)
    b._by_name = {name: b._leaves[str(tmp_path / name)] for name in _INFO}
    return b


def _colour(item):
    brush = item.foreground(0)
    return None if brush.style() == Qt.BrushStyle.NoBrush else brush.color().name()


def test_2d_experiments_are_coloured_and_1d_ones_are_not_by_default(browser):
    leaf = browser._by_name
    assert _colour(leaf["one"]) is None
    assert _colour(leaf["two"]) == "#3d8bfd"
    assert _colour(leaf["series"]) == "#3d8bfd"          # pseudo-2D follows 2D until set
    assert "1H, 1D" in leaf["one"].text(0) and "pseudo-2D" in leaf["series"].text(0)


def test_each_kind_takes_its_own_colour_and_says_so(browser):
    seen = []
    browser.coloursChanged.connect(seen.append)
    browser.set_colours({"1d": "#00aa00", "2d": "#3d8bfd", "pseudo2d": "#ff0000"}, emit=True)
    leaf = browser._by_name
    assert (_colour(leaf["one"]), _colour(leaf["two"]), _colour(leaf["series"])) == \
        ("#00aa00", "#3d8bfd", "#ff0000")
    assert seen == [{"1d": "#00aa00", "2d": "#3d8bfd", "pseudo2d": "#ff0000"}]
    browser.set_colours({"1d": "#00aa00"})                # restoring a saved choice: silent
    assert len(seen) == 1 and browser.colours()["2d"] == "#3d8bfd"
    browser._reset_colours()
    assert _colour(leaf["one"]) is None and seen[-1] == browser_module.DEFAULT_COLOURS


def test_unreadable_saved_colours_fall_back_to_the_default():
    assert browser_module.normalise_colours(None) == browser_module.DEFAULT_COLOURS
    assert browser_module.normalise_colours({"2d": "not a colour", "1d": 5}) == \
        browser_module.DEFAULT_COLOURS
    assert browser_module.normalise_colours({"2d": "#ABCDEF"})["2d"] == "#abcdef"


def test_a_cloud_only_or_unreadable_experiment_is_left_in_the_normal_colour(browser, tmp_path,
                                                                             monkeypatch):
    def broken(path, vendor=None):
        raise OSError("unreadable")

    monkeypatch.setattr(browser_module.io, "peek", broken)
    (tmp_path / "late").mkdir()
    browser.update_matches([(str(tmp_path / "late"), "bruker")] + browser.matches(),
                           remove_missing=False)
    browser._describe_children(browser._tree.invisibleRootItem())
    assert _colour(browser._leaves[str(tmp_path / "late")]) is None


def test_the_right_click_menu_offers_the_colours(browser):
    menu, chosen = QMenu(), {}
    browser._add_colour_menu(menu, chosen)
    submenu = menu.actions()[0].menu()
    texts = [a.text() for a in submenu.actions() if a.text()]
    assert menu.actions()[0].text() == "Typeface colours"
    assert texts == ["1D experiments…", "2D experiments…", "Pseudo-2D experiments…",
                     "Reset Colours"]
    assert all(not a.icon().isNull() for a in submenu.actions()[:3])
    assert len(chosen) == 4


def test_the_colours_are_remembered_between_sessions(window, qapp, monkeypatch, tmp_path):  # noqa: F811
    window.data_browser.set_colours({"1d": "#112233", "2d": "#445566"}, emit=True)
    saved = json.loads((tmp_path / "ui_settings.json").read_text(encoding="utf-8"))
    assert saved["browser_colours"] == {"1d": "#112233", "2d": "#445566", "pseudo2d": ""}
    again = MainWindow()
    assert again.data_browser.colours() == {"1d": "#112233", "2d": "#445566", "pseudo2d": ""}
