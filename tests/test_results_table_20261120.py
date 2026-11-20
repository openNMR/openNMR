"""Notes 8.4: one tidy results table for every kind of fit."""
import numpy as np
import pytest

from opennmr.notebook import results_table as rt
from opennmr.notebook import fit_values


def _curve_result(label="peak 1", model="biexponential", shared=()):
    return {"peak_label": label, "model": model,
            "params": {"A1": 0.6, "D1": 3e-9, "A2": 0.4, "D2": 2e-10},
            "errors": {"A1": 0.01, "D1": 1e-11, "A2": 0.01, "D2": 4e-12},
            "fixed": {}, "shared": list(shared),
            "fit_quality": {"rmse": 0.01, "r_squared": 0.999, "reduced_chi2": 1.1, "n_points": 16},
            "uncertainty": {"kind": "measured_inflated", "short": "x", "description": "y"}}


def _components_result():
    return {"kind": "diffusion", "method": "score", "n_components": 2,
            "constants": [3e-9, 2e-10], "fractions": [0.7, 0.3], "r_squared": 0.998}


def test_every_row_has_exactly_the_shared_columns_whatever_its_source():
    rows = rt.curve_fit_rows([_curve_result()], "s1") + rt.components_rows(_components_result(), "s1")
    assert rows
    for row in rows:
        assert tuple(row) == rt.COLUMNS
    assert rt.concat_results(rows) == rows


def test_curve_fit_rows_carry_units_error_kind_and_quality():
    rows = rt.curve_fit_rows([_curve_result()], "s1")
    d1 = next(r for r in rows if r["parameter"] == "D1")
    assert d1["unit"] == "m^2/s" and d1["error"] == 1e-11
    assert d1["error_kind"] == "measured_inflated"
    assert d1["fit_type"] == "curve" and d1["fit"] == "biexponential" and d1["group"] == "peak 1"
    assert d1["r_squared"] == 0.999 and d1["n_points"] == 16
    assert next(r for r in rows if r["parameter"] == "A1")["unit"] == ""


def test_components_use_the_same_parameter_names_and_units_as_a_curve_fit():
    comp = rt.components_rows(_components_result(), "s1")
    names = {(r["parameter"], r["unit"]) for r in comp}
    assert ("D1", "m^2/s") in names and ("D2", "m^2/s") in names and ("share", "%") in names
    curve = {(r["parameter"], r["unit"]) for r in rt.curve_fit_rows([_curve_result()])}
    assert ("D1", "m^2/s") in curve                      # one name, one unit, both routes
    assert next(r for r in comp if r["parameter"] == "share" and r["group"] == "component 1")["value"] == pytest.approx(70.0)


def test_components_say_that_no_error_was_estimated():
    for row in rt.components_rows(_components_result()):
        assert row["error"] is None and row["error_kind"] == "none"


def test_a_relaxation_components_result_is_in_seconds():
    res = dict(_components_result(), kind="relaxation", constants=[0.5, 2.0])
    rows = rt.components_rows(res)
    assert {r["parameter"] for r in rows} == {"T1", "T2", "share"}
    assert next(r for r in rows if r["parameter"] == "T1")["unit"] == "s"


def test_shared_parameters_are_marked():
    rows = rt.curve_fit_rows([_curve_result(shared=["D1"])])
    assert next(r for r in rows if r["parameter"] == "D1")["shared"] == 1
    assert next(r for r in rows if r["parameter"] == "D2")["shared"] == 0


def test_a_fixed_parameter_has_no_error():
    res = _curve_result()
    res["fixed"] = {"A1": 0.6}
    row = next(r for r in rt.curve_fit_rows([res]) if r["parameter"] == "A1")
    assert row["fixed"] == 1 and row["error"] is None and row["error_kind"] == "none"


def test_split_heading_gives_ascii_names_and_units():
    assert rt.split_heading("δiso (ppm)") == ("delta_iso", "ppm")
    assert rt.split_heading("C_Q (MHz)") == ("C_Q", "MHz")
    assert rt.split_heading("Share (%)") == ("share", "%")
    assert rt.split_heading("J12") == ("J12", "Hz")
    assert rt.split_heading("L fraction") == ("eta", "")
    assert rt.split_heading("R²") == ("r_squared", "")


def test_record_rows_cover_a_stored_fit_and_split_slice_rows():
    record = fit_values.values_record(
        {"Slice 3, peak 2": {"Shift (ppm)": {"value": 1.25, "error": 0.01},
                             "FWHM (ppm)": {"value": 0.1, "fixed": True}},
         "Al1": {"δiso (ppm)": {"value": 12.0, "error": 0.03}}},
        fit="Solid-state NMR", spectrum="27Al", quality={"r2": 0.99})
    rows = rt.record_rows(record, fit_type="CSA")
    shift = next(r for r in rows if r["parameter"] == "Shift")
    assert shift["slice"] == 3 and shift["group"] == "peak 2" and shift["unit"] == "ppm"
    assert next(r for r in rows if r["parameter"] == "FWHM")["fixed"] == 1
    iso = next(r for r in rows if r["parameter"] == "delta_iso")
    assert iso["group"] == "Al1" and iso["slice"] is None and iso["r_squared"] == 0.99
    assert all(r["fit_type"] == "CSA" and r["spectrum"] == "27Al" for r in rows)


def test_csv_round_trips_and_states_each_error_kind_once(tmp_path):
    rows = rt.curve_fit_rows([_curve_result()], "s1") + rt.components_rows(_components_result(), "s1")
    text = rt.results_to_csv_text(rows)
    assert text.count("# error_kind measured_inflated") == 1
    assert "# error_kind none" not in text
    path = rt.write_results_csv(rows, tmp_path / "out")
    assert path.endswith(".csv")
    back = rt.read_results_csv(path)
    assert len(back) == len(rows)
    for a, b in zip(rows, back):
        assert a["parameter"] == b["parameter"] and a["group"] == b["group"]
        assert a["value"] == pytest.approx(b["value"], rel=1e-10)
        assert (a["error"] is None) == (b["error"] is None)


def test_concatenating_two_routes_gives_one_table():
    a = rt.curve_fit_rows([_curve_result()], "s1")
    b = rt.components_rows(_components_result(), "s1")
    both = rt.concat_results(a, b)
    assert len(both) == len(a) + len(b)
    assert {r["fit_type"] for r in both} == {"curve", "components"}


def test_a_malformed_row_or_file_is_refused(tmp_path):
    with pytest.raises(KeyError):
        rt.make_row(nonsense=1)
    with pytest.raises(ValueError):
        rt.concat_results([{"a": 1}])
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        rt.read_results_csv(bad)


def test_non_finite_numbers_become_empty_not_nan():
    row = rt.make_row(parameter="x", value=float("nan"), error=float("inf"))
    assert row["value"] is None and row["error"] is None


def test_the_unit_rule_matches_the_curve_fit_csv():
    from opennmr.processing import relaxation
    for name in ("T1", "T2", "T1rho", "T2b", "tau_c", "D", "D1", "D2", "Da", "A1", "beta", "M0"):
        assert rt.parameter_unit(name) == relaxation._csv_param_unit(name), name


def test_a_real_curve_fit_exports(tmp_path):
    from opennmr.processing import relaxation
    t = np.geomspace(0.01, 5, 14)
    y = 1 - 2 * np.exp(-t / 0.8)
    fit = relaxation.fit_relaxation(t, y, model="inversion_recovery")
    fit["peak_label"] = "7Li"
    rows = rt.curve_fit_rows([fit], "li")
    t1 = next(r for r in rows if r["parameter"] == "T1")
    assert t1["value"] == pytest.approx(0.8, rel=1e-2) and t1["unit"] == "s"
    assert t1["error_kind"] in rt.ERROR_KINDS
    rt.write_results_csv(rows, tmp_path / "li.csv")


# ---- GUI / notebook wiring ------------------------------------------------

def test_the_notebook_exposes_the_module():
    import opennmr.notebook as nb
    assert nb.results_table is rt


def test_table_for_spectrum_collects_curve_and_components_from_the_spectrum():
    class _Meta:
        raw = {"relaxation_fit_results": [_curve_result()], "score_result": _components_result()}
        title = "s1"

    class _Spectrum:
        metadata = _Meta()

    rows = rt.table_for_spectrum(_Spectrum(), "s1")
    assert {r["fit_type"] for r in rows} == {"curve", "components"}
    assert all(r["spectrum"] == "s1" for r in rows)


def test_the_gui_exports_one_table(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from opennmr.gui.main_window import MainWindow
    from test_gui import _load_spectrum_directly, _make_pseudo2d_relaxation_spectrum
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum(np.geomspace(0.001, 0.3, 4))
    _load_spectrum_directly(window, spectrum)
    assert window.export_results_table_button.text().startswith("Export Results Table")

    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    window._on_export_results_table_clicked()                      # nothing fitted yet
    assert shown and "no fit results" in shown[0]

    spectrum = window.session.spectra[0]
    spectrum.metadata.raw["relaxation_fit_results"] = [_curve_result()]
    spectrum.metadata.raw["score_result"] = _components_result()
    out = tmp_path / "all.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    window._on_export_results_table_clicked()
    rows = rt.read_results_csv(out)
    assert {r["fit_type"] for r in rows} == {"curve", "components"}
    assert {r["parameter"] for r in rows} >= {"D1", "D2", "A1", "share"}
