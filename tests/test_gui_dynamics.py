"""Dynamics tile: Relaxation vs Temperature and Exchange Lineshape
sections (gui/dynamics_panels.py) and their MainWindow wiring. The
numbers are the notebook modules' (tested in test_vt_relaxation.py and
test_exchange.py); these check the panels drive them, draw on the Fit
view and log what they did."""
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.gui.dynamics_panels import (
    ExchangeLineshapePanel, VTRelaxationPanel, parse_fixed, parse_temperatures,
)
from opennmr.gui.main_window import MainWindow
from opennmr.notebook import exchange as ex
from opennmr.notebook import vt_relaxation as vt

from test_gui import qapp  # noqa: F401 -- fixture

T = np.linspace(200, 450, 14)


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def _t1_rows(larmor=116.6, series="7Li", unit_ms=False):
    R = vt.relaxation_rate(T, larmor, 2e-13, 0.32, 3e9)
    scale = 1e3 if unit_ms else 1.0
    return [[series, f"{t:g}", f"{scale / r:.8g}", f"{0.01 * scale / r:.3g}", f"{larmor:g}"]
            for t, r in zip(T, R)]


def _secondary_axes(ax):
    return [c for c in ax.child_axes if type(c).__name__ == "SecondaryAxis"]


# ------------------------------------------------------------- parsers

def test_parse_temperatures():
    assert parse_temperatures("200, 250 300") == [200.0, 250.0, 300.0]
    assert parse_temperatures("200:300:50") == [200.0, 250.0, 300.0]
    for bad in ("", "abc", "300:200", "-5", "1:2:0"):
        with pytest.raises(ValueError):
            parse_temperatures(bad)


def test_parse_fixed():
    assert parse_fixed("Ea_eV=0.3, beta = 1.8") == {"Ea_eV": 0.3, "beta": 1.8}
    assert parse_fixed("") == {}
    with pytest.raises(ValueError):
        parse_fixed("Ea_eV")


# ------------------------------------------------------------- wiring

def test_dynamics_tile_has_both_sections(qapp):
    window = MainWindow()
    pages = {window.side_tabs.tabText(i): window.side_tabs.widget(i)
             for i in range(window.side_tabs.count())}
    page = pages["Dynamics"]
    assert page.isAncestorOf(window.relaxation_workflow_guide)      # Fit Curve still first
    assert page.isAncestorOf(window.vt_relaxation_panel)
    assert page.isAncestorOf(window.exchange_panel)
    assert not window._vt_relaxation_section.isExpanded()          # sections start collapsed
    assert window.vt_relaxation_panel.canvas is window.fit_canvas
    assert window.exchange_panel.canvas is window.fit_canvas


def test_vt_rows_from_curve_fit(qapp):
    window = MainWindow()
    assert window._vt_rows_from_curve_fit() == []
    window._relaxation_fit_results = [
        {"rate_param": "T1", "params": {"T1": 0.25, "M0": 1.0}, "errors": {"T1": 0.01},
         "peak_label": "peak A"},
        {"rate_param": None, "params": {"A1": 1.0}, "errors": {}},          # biexponential: skipped
    ]
    rows = window._vt_rows_from_curve_fit()
    assert rows == [{"series": "peak A", "T_K": None, "value": 0.25, "error": 0.01,
                     "larmor_MHz": None, "quantity": "T1"}]


# ----------------------------------------------------------- VT panel

def test_vt_panel_fit_draws_ln_t1_against_1000_over_t_with_temperature_on_top(qapp):
    window = MainWindow()
    panel = window.vt_relaxation_panel
    panel.set_rows(_t1_rows() + _t1_rows(194.4))
    data = panel.datasets()
    assert [d["larmor_MHz"] for d in data] == [116.6, 194.4]
    fit = panel.run_fit()
    assert fit is not None
    assert fit["params"]["Ea_eV"] == pytest.approx(0.32, abs=0.002)
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    ax = window.fit_canvas.figure.axes[0]
    assert "1000" in ax.get_xlabel() and "ln" in ax.get_ylabel()
    tops = _secondary_axes(ax)
    assert len(tops) == 1 and "T" in tops[0].get_xlabel()
    # Plot Style's label size reaches the top axis too
    size = window.fit_canvas.style_settings()["labelsize"]
    assert tops[0].xaxis.label.get_fontsize() == pytest.approx(size)
    names = [panel.results_table.item(r, 0).text() for r in range(panel.results_table.rowCount())]
    assert "Ea_eV" in names and "tau0_s" in names
    assert "BPP" in panel.status_log.toPlainText()


def test_vt_panel_units_ms_and_celsius(qapp):
    panel = VTRelaxationPanel()
    panel.value_unit_combo.setCurrentIndex(panel.value_unit_combo.findData("ms"))
    panel.temperature_unit_combo.setCurrentIndex(panel.temperature_unit_combo.findData("C"))
    panel.set_rows([["s", "26.85", "250", "", "100"], ["s", "46.85", "125", "", "100"]])
    d = panel.datasets()[0]
    assert np.allclose(d["T_K"], [300.0, 320.0])
    assert np.allclose(d["value"], [0.25, 0.125])
    assert "°C" in panel.table.horizontalHeaderItem(1).text()
    assert "ms" in panel.table.horizontalHeaderItem(2).text()


def test_vt_panel_rejects_bad_rows(qapp, _no_modal_dialogs):
    panel = VTRelaxationPanel()
    panel.set_rows([["s", "300", "", "", "100"]])
    with pytest.raises(ValueError, match="Row 1"):
        panel.datasets()
    panel.set_rows([["s", "300", "0.1", "", ""]])
    with pytest.raises(ValueError, match="Larmor"):
        panel.datasets()
    assert panel.run_fit() is None
    assert _no_modal_dialogs and _no_modal_dialogs[-1][0] == "warning"


def test_vt_panel_paste_and_from_curve_fit(qapp):
    rows = [{"series": "pk", "T_K": 300.0, "value": 0.2, "error": 0.01, "larmor_MHz": 100.0,
             "quantity": "T2"}]
    panel = VTRelaxationPanel(curve_fit_rows=lambda: rows)
    assert panel.paste_from_clipboard("T\tT1\n300\t0.1\t0.01\t100\n320,0.2,,100,x\n") == 2
    assert panel.table.rowCount() == 2
    panel.clear_rows()
    assert panel.add_from_curve_fit() == 1
    assert panel.quantity == "T2"                      # follows the Fit Curve result
    assert panel.table.item(0, 1).text() == "300"
    assert "Fit Curve" in panel.status_log.toPlainText()


def test_vt_panel_import_csv_and_export(qapp, tmp_path):
    path = tmp_path / "t1.csv"
    lines = ["T,T1,error,larmor_MHz"] + [f"{t},{1 / r},{0.01 / r},116.6" for t, r in
                                          zip(T, vt.relaxation_rate(T, 116.6, 2e-13, 0.32, 3e9))]
    path.write_text("\n".join(lines))
    panel = VTRelaxationPanel()
    assert panel.import_csv(str(path)) == T.size
    assert panel.run_fit() is not None
    out = tmp_path / "out.csv"
    panel.export_results(str(out))
    text = out.read_text()
    assert "Ea_eV" in text and "# log" in text


def test_vt_panel_model_options_reach_the_fit(qapp):
    panel = VTRelaxationPanel()
    panel.mechanism_combo.setCurrentIndex(panel.mechanism_combo.findData("quadrupolar"))
    panel.spin_spin.setValue(2.5)
    panel.asymmetric_check.setChecked(True)
    panel.background_check.setChecked(True)
    panel.processes_spin.setValue(2)
    panel.fixed_edit.setText("beta_1=2.0")
    panel.range_check.setChecked(True)
    kw = panel.fit_kwargs()
    assert kw["mechanism"] == "quadrupolar" and kw["spin"] == 2.5
    assert kw["asymmetric"] and kw["background"] and kw["n_processes"] == 2
    assert kw["fixed"] == {"beta_1": 2.0}
    assert kw["T_range"] == (200.0, 400.0)


def test_vt_panel_compare_models_and_arrhenius_line(qapp):
    panel = VTRelaxationPanel()
    panel.set_rows(_t1_rows())
    panel.starts_spin.setValue(6)
    fits = panel.run_comparison()
    assert fits and fits[0]["delta_aicc"] == 0
    assert panel.results_table.rowCount() == 4
    assert "Model comparison" in panel.status_log.toPlainText()
    panel.range_check.setChecked(True)
    panel.t_lo_spin.setValue(200)
    panel.t_hi_spin.setValue(260)
    res = panel.run_arrhenius()
    assert res["flank"] == "low-temperature"
    assert res["Ea_eV"] == pytest.approx(0.32, rel=0.05)


def test_vt_panel_hendrickson_bray(qapp):
    window = MainWindow()
    panel = window.vt_relaxation_panel
    panel.quantity_combo.setCurrentIndex(panel.quantity_combo.findData("linewidth"))
    assert not panel.model_box.isVisibleTo(panel)
    Tw = np.linspace(150, 450, 20)
    w = vt.hendrickson_bray(Tw, 8000, 1.5e-3, 0.4, 300)
    panel.set_rows([["7Li", f"{t:g}", f"{v:.6g}", "", ""] for t, v in zip(Tw, w)])
    hb = panel.run_fit()
    assert hb["params"]["Ea_eV"] == pytest.approx(0.4, abs=0.005)
    ax = window.fit_canvas.figure.axes[0]
    assert "Delta" in ax.get_ylabel() or "Δ" in ax.get_ylabel()
    assert "Hendrickson" in panel.status_log.toPlainText()


def test_vt_panel_notebook_code_matches(qapp):
    panel = VTRelaxationPanel()
    panel.set_rows(_t1_rows())
    panel.starts_spin.setValue(6)
    fit = panel.run_fit()
    ns = {}
    exec(compile(panel.notebook_code(), "<nb>", "exec"), ns)
    assert ns["fit"]["params"]["Ea_eV"] == pytest.approx(fit["params"]["Ea_eV"], rel=1e-6)


def test_vt_panel_replots_on_axis_change(qapp):
    window = MainWindow()
    panel = window.vt_relaxation_panel
    panel.set_rows(_t1_rows())
    panel.starts_spin.setValue(4)
    panel.run_fit()
    panel.x_axis_combo.setCurrentIndex(panel.x_axis_combo.findData("1/kBT"))
    panel.top_axis_combo.setCurrentIndex(panel.top_axis_combo.findData("C"))
    ax = window.fit_canvas.figure.axes[0]
    assert "k" in ax.get_xlabel()
    assert "°C" in _secondary_axes(ax)[0].get_xlabel()


# ----------------------------------------------------- exchange panel

def test_exchange_panel_example_simulates_xrespecs_case_correctly(qapp):
    window = MainWindow()
    panel = window.exchange_panel
    assert panel.sites()[0]["shift_ppm"] == 20 and panel.hops()[0]["Ea_rev_eV"] == 0.37
    results = panel.simulate()
    assert [r["T_K"] for r in results] == [200, 225, 250, 275, 300, 325, 350]
    hot = results[-1]
    assert hot["fast_limit_ppm"] == pytest.approx(6.8, abs=0.05)
    ax = window.fit_canvas.figure.axes[0]
    assert "Li" in ax.get_xlabel() and ax.get_xlim()[0] > ax.get_xlim()[1]
    assert "Exchange lineshape" in panel.status_log.toPlainText()
    assert "narrowed" in panel.results_label.text()


def test_exchange_panel_validates_tables(qapp, _no_modal_dialogs):
    panel = ExchangeLineshapePanel()
    panel.add_hop([1, 5, 0.3, "", "1e12", 0])
    assert panel.simulate() is None
    assert "different sites" in _no_modal_dialogs[-1][1]
    panel.nucleus_edit.setText("Xx")
    assert "unknown" in panel.larmor_label.text()


def test_exchange_panel_coalescence_note_for_equal_energies(qapp):
    panel = ExchangeLineshapePanel()
    panel.hops_table.setRowCount(0)
    panel.add_hop([1, 2, 0.35, "", "1e11", 0])
    panel.simulate()
    assert "coalesces near" in panel.results_label.text()


def test_exchange_panel_fits_selected_spectra(qapp):
    larmor = 310.9
    sites = [{"label": "A", "shift_ppm": 20, "fwhm_ppm": 0.5}, {"label": "B", "shift_ppm": 0,
                                                              "fwhm_ppm": 0.5}]
    hops = [{"from": 0, "to": 1, "Ea_eV": 0.30, "nu0_Hz": 1e10}]
    axis = np.linspace(40, -20, 801)
    spectra = []
    for T_K in (220, 250, 280):
        y = ex.exchange_spectrum(sites, hops, T_K, larmor_MHz=larmor, ppm_axis=axis)["spectrum"]
        spectra.append({"label": f"{T_K} K", "ppm": axis, "y": y, "T_K": T_K,
                        "larmor_MHz": larmor, "nucleus": "7Li"})
    window = MainWindow()
    panel = window.exchange_panel
    panel.get_spectra = lambda: spectra
    panel.sites_table.setRowCount(0)
    panel.hops_table.setRowCount(0)
    panel.add_site(["A", 20, 0.5, 0])
    panel.add_site(["B", 0, 0.5, 0])
    panel.add_hop([1, 2, 0.26, "", "1e9", 0])
    panel.gauss_spin.setValue(0.0)
    panel.set_vary(["hop1.Ea_eV", "hop1.log10_nu0"])
    fit = panel.fit_selected()
    assert fit["params"]["hop1.Ea_eV"] == pytest.approx(0.30, abs=0.005)
    # the fitted values are written back into the table
    assert float(panel.hops_table.item(0, 2).text()) == pytest.approx(0.30, abs=0.005)
    assert len(window.fit_canvas.figure.axes) >= 1
    assert "Exchange lineshape fit" in panel.status_log.toPlainText()


def test_exchange_panel_needs_temperatures_for_unlabelled_spectra(qapp, _no_modal_dialogs):
    panel = ExchangeLineshapePanel()
    axis = np.linspace(40, -20, 201)
    panel.get_spectra = lambda: [{"label": "x", "ppm": axis, "y": np.exp(-axis ** 2),
                                  "T_K": None, "larmor_MHz": 300.0}] * 9
    panel.set_vary(["hop1.log10_k"])
    assert panel.fit_selected() is None                     # 9 spectra, 7 typed temperatures
    assert _no_modal_dialogs[-1][0] == "warning"


def test_exchange_panel_notebook_code_runs(qapp):
    panel = ExchangeLineshapePanel()
    panel.temperatures_edit.setText("250, 300")
    ns = {}
    exec(compile(panel.notebook_code(), "<nb>", "exec"), ns)
    assert len(ns["results"]) == 2
