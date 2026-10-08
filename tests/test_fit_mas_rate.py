"""The MAS rate as a fitting parameter (explicit user request, 2026-10-06:
"allow the MAS rate to vary, by no more than ~500 Hz of a specified value
... by default it should be fixed") -- notebook.combined_fitting,
notebook.multi_fitting and the Fit panels."""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.combined_panels import SPECTRUM_ROW, _P_FREE, _P_MAX, _P_MIN, _P_SITE
from opennmr.gui.main_window import MainWindow
from opennmr.notebook import combined_fitting as cf
from opennmr.notebook import multi_fitting as mf
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined

from test_gui import qapp  # noqa: F401 -- fixture

NU0 = 100.6e6
TRUE_MAS = 12150.0
X = np.linspace(-150, 350, 8192)


def _site(aniso=-150, eta=0.4, delta=120.0):
    return Site("13C", delta, label="C1", fwhm=0.6).add_csa(aniso, eta)


def _results(mas=TRUE_MAS, seed=0):
    y = simulate_combined([_site()], X, NU0, mas_Hz=mas, n_zcw=2584)["real_spec"]
    y = y / y.max() * 1000 + np.random.default_rng(seed).normal(0, 1, X.size)
    return {"x_ppm": X[::-1], "real_spectrum": y[::-1], "SFO1": NU0 / 1e6}


@pytest.fixture(scope="module")
def data():
    return _results()


def _p(fit, name, label=""):
    return next(p for p in fit["parameters"] if p["name"] == name and p["label"] == label)


# ---- notebook ----------------------------------------------------------------------

def test_fixed_by_default(data):
    fit = cf.fit_combined(data, [_site(-130, 0.5)], mas_Hz=12000, n_zcw=987)
    assert not any(p["name"] == "mas_Hz" for p in fit["parameters"])
    assert fit["settings"]["mas_Hz"] == 12000 and not fit["settings"]["fit_mas"]


@pytest.mark.parametrize("start", [12000.0, 11700.0, 12600.0])
def test_a_freed_rate_is_found_and_the_tensor_with_it(data, start):
    fit = cf.fit_combined(data, [_site(-130, 0.5)], mas_Hz=start, fit_mas=True, n_zcw=987)
    mas = _p(fit, "mas_Hz")
    assert mas["value"] == pytest.approx(TRUE_MAS, abs=2.0)
    assert 0 < mas["error"] < 5 and mas["free"]
    assert (mas["lower"], mas["upper"]) == (start - 500, start + 500)
    assert _p(fit, "aniso", "C1")["value"] == pytest.approx(-150, abs=2)
    assert fit["quality"]["r2"] > 0.998          # the noise limit here is 0.9989
    assert fit["settings"]["mas_Hz"] == pytest.approx(mas["value"])
    assert fit["settings"]["mas_Hz_start"] == start
    assert "fitted; started at" in fit["log"].splitlines()[0]
    # The scan that found it is kept, for anyone checking.
    assert fit["mas_scan"]["grid_Hz"][0] == start - 500
    # A held rate this far off cannot fit: it shrinks the tensor instead.
    if start != 12000.0:
        held = cf.fit_combined(data, [_site(-130, 0.5)], mas_Hz=start, n_zcw=987)
        assert held["quality"]["r2"] < fit["quality"]["r2"] - 0.01


def test_explicit_bounds_and_the_errors(data):
    fit = cf.fit_combined(data, [_site(-150, 0.4)], mas_Hz=12000, fit_mas=True, n_zcw=987,
                          mas_bounds_Hz=(11900, 12100))
    mas = _p(fit, "mas_Hz")
    assert (mas["lower"], mas["upper"]) == (11900, 12100)
    assert mas["value"] == pytest.approx(12100, abs=1)       # at its edge: the truth is beyond
    with pytest.raises(ValueError, match="must contain the starting rate"):
        cf.fit_combined(data, [_site()], mas_Hz=12000, fit_mas=True, mas_bounds_Hz=(12100, 12200))
    with pytest.raises(ValueError, match="static fit has no MAS rate"):
        cf.fit_combined(data, [_site()], mas_Hz=0, fit_mas=True)
    from opennmr.notebook import fit_advice
    text = fit_advice.advice_text(fit_advice.advise_combined_fit(fit))
    assert "The MAS rate stopped at the edge of the range it was allowed (12100 Hz)" in text


def test_multi_spectrum_refines_each_rate_and_reports_it():
    exps = [mf.make_experiment(_results(mas=m, seed=k), label=f"E{k}", nucleus="13C",
                               larmor_Hz=NU0, mas_Hz=12000, fit_mas=True, x_value=k)
            for k, m in enumerate((12150.0, 11900.0))]
    for e in exps:
        e["n_zcw"] = 987
    result = mf.fit_multi(exps, {"13C": [_site(-140, 0.45)]})
    got = [f["mas_fitted"] for f in result["experiments"]]
    assert got[0][0] == pytest.approx(12150, abs=3) and got[1][0] == pytest.approx(11900, abs=3)
    assert got[0][2] == 12000
    assert "fitted in its own fit, from 12000 Hz" in result["log"]
    _x, rates, errs, _l = mf.quantity_trend(result, mf.WINDOW, "mas_Hz")
    assert rates == pytest.approx([12150, 11900], abs=3) and np.all(errs > 0)


# ---- GUI ---------------------------------------------------------------------------

@pytest.fixture
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def _mas_row(panel):
    t = panel.parameter_table
    for r in range(t.rowCount()):
        if t.item(r, _P_SITE).text() == SPECTRUM_ROW:
            return r
    return None


def _window_with_spectrum(qapp):
    window = MainWindow()
    r = _results()
    spectrum = NMRSpectrum(data=np.asarray(r["real_spectrum"]),
                           axes=[NMRAxis(kind="frequency", ppm=np.asarray(r["x_ppm"]))],
                           metadata=NMRMetadata(vendor="simulated", nucleus="13C", sfo1=NU0 / 1e6,
                                                raw={"acqus": {"MASR": 12000}}))
    window._add_derived_spectrum(spectrum, "C")
    panel = window.combined_fit_panel
    panel.seed_from_spectrum(spectrum)
    panel.diso_spin.setValue(120.0)
    panel.fwhm_spin.setValue(0.6)
    panel.add_interaction("csa", aniso=-130.0, asym=0.5)
    panel.orientations_spin.setValue(987)
    return window, panel


def test_the_parameter_table_has_a_fixed_mas_row_under_mas(qapp, _dialogs):
    window, panel = _window_with_spectrum(qapp)
    assert panel.regime_combo.currentData() == "mas"
    r = _mas_row(panel)
    assert r is not None
    t = panel.parameter_table
    assert t.item(r, _P_FREE).checkState() == Qt.CheckState.Unchecked
    assert "11500 to 12500" in t.item(r, _P_MIN).toolTip()
    assert "fit_mas" not in panel.fit_arguments()["settings"]
    # Static: no MAS row.
    panel.regime_combo.setCurrentIndex(panel.regime_combo.findData("static"))
    assert _mas_row(panel) is None
    panel.regime_combo.setCurrentIndex(panel.regime_combo.findData("mas"))
    # Freed: ±500 Hz, or the Min/Max typed.
    t = panel.parameter_table
    r = _mas_row(panel)
    t.item(r, _P_FREE).setCheckState(Qt.CheckState.Checked)
    settings = panel.fit_arguments()["settings"]
    assert settings["fit_mas"] and settings["mas_bounds_Hz"] == (11500.0, 12500.0)
    t.item(_mas_row(panel), _P_MAX).setText("12300")
    assert panel.fit_arguments()["settings"]["mas_bounds_Hz"] == (11500.0, 12300.0)
    t.item(_mas_row(panel), _P_MIN).setText("12100")
    with pytest.raises(ValueError, match="must enclose the start"):
        panel.fit_arguments()
    assert "MAS rate" in panel.get_state()["extra"]


def test_a_gui_fit_with_the_rate_free_finds_it_and_carries_it(qapp, _dialogs):
    window, panel = _window_with_spectrum(qapp)
    panel.mas_constraint()["free"] = True
    assert panel.fit(), _dialogs
    assert _p(panel.result, "mas_Hz")["value"] == pytest.approx(TRUE_MAS, abs=3)
    t = panel.results_table
    rows = [t.verticalHeaderItem(i).text() for i in range(t.rowCount())]
    columns = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    assert rows[-1] == "(spectrum)" and "MAS rate (Hz)" in columns
    panel.use_fitted_as_start()
    assert panel.mas_spin.value() == pytest.approx(TRUE_MAS, abs=3)


def test_the_advice_button_frees_the_rate(qapp, _dialogs):
    window, panel = _window_with_spectrum(qapp)
    panel.mas_spin.setValue(12100)                     # 50 Hz off, held
    panel.add_interaction  # noqa: B018
    assert panel.fit(), _dialogs
    actions = [it.get("action") for it in panel.result["advice"]]
    assert {"type": "free_mas"} in actions
    assert panel.advice_box.isVisibleTo(panel)
    button = next(b for b in panel.advice_box.action_buttons if b.text() == "Free the MAS rate")
    button.click()
    assert panel.mas_constraint()["free"]
    assert "Done: the MAS rate is free" in panel.advice_box.done_label.text()
    assert panel.parameter_table.item(_mas_row(panel), _P_FREE).checkState() == \
        Qt.CheckState.Checked


def test_multi_panel_passes_the_choice(qapp, _dialogs, monkeypatch):
    window = MainWindow()
    panel = window.multi_fit_panel
    assert not panel.fit_mas_check.isChecked() and panel.mas_window_spin.value() == 500
    seen = {}
    real = mf.make_experiment
    monkeypatch.setattr(mf, "make_experiment", lambda *a, **k: (seen.update(k), real(*a, **k))[1])
    r = _results()
    spectrum = NMRSpectrum(data=np.asarray(r["real_spectrum"]),
                           axes=[NMRAxis(kind="frequency", ppm=np.asarray(r["x_ppm"]))],
                           metadata=NMRMetadata(vendor="simulated", nucleus="13C", sfo1=NU0 / 1e6,
                                                raw={"acqus": {"MASR": 12000}}))
    index = window._add_derived_spectrum(spectrum, "C")
    window.spectrum_list.clearSelection()
    window.spectrum_list.item(index).setSelected(True)
    panel.add_selected()
    panel.set_sites("13C", [_site(-140, 0.45)])
    panel.orientations_spin.setValue(987)
    panel.fit_mas_check.setChecked(True)
    panel.mas_window_spin.setValue(300)
    assert panel.fit(show_progress=False), _dialogs
    assert seen["fit_mas"] is True and seen["mas_window_Hz"] == 300
    assert panel.result["experiments"][0]["mas_fitted"][0] == pytest.approx(TRUE_MAS, abs=3)
    t = panel.results_table
    assert "MAS rate (Hz)" in [t.verticalHeaderItem(i).text() for i in range(t.rowCount())]
    assert panel.advice_box.isVisibleTo(panel)
