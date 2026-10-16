"""Fit > Multi-spectrum (gui/multi_fit_panel.py): experiments in any
order (from Open experiments or by number), sites per nucleus, scopes,
the fit through notebook.multi_fitting, the results/trend views, and the
fit kept with every spectrum in it."""

import json
import os

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.gui.multi_fit_panel import (_E_MAS, _E_WINDOW, _E_X, _Q_FREE, _Q_NAME, _Q_SCOPE,
                                         _Q_SITE, MultiFitPanel, parse_expnos)
from opennmr.io import history as io_history
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined

from test_gui import qapp, _make_1d_dataset, _settle_sidecar_save  # noqa: F401 -- qapp is a fixture
from opennmr.gui.widgets.collapsible_section import CollapsibleSection


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def _al(delta=12.0, cq=4.2, eta=0.35, fwhm=0.6, label="Al1"):
    return Site("27Al", delta, label=label, fwhm=fwhm).add_quadrupolar(cq, eta)


def _add(window, sites, name, nu0=130.3e6, mas=20e3, seed=0, lo=-150, hi=100, nucleus="27Al",
         transitions="ct"):
    x = np.linspace(hi, lo, 4096)
    y = simulate_combined(sites, x[::-1], nu0, mas_Hz=mas, transitions=transitions,
                          n_zcw=6765)["real_spec"][::-1]
    y = y / y.max() * 1000 + np.random.default_rng(seed).standard_normal(x.size) * 2
    spectrum = NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x)],
                           metadata=NMRMetadata(vendor="simulated", nucleus=nucleus,
                                                sfo1=nu0 / 1e6, raw={"acqus": {"MASR": mas}}))
    return window._add_derived_spectrum(spectrum, name)


def _add_selected(window, panel, indices):
    window.spectrum_list.clearSelection()
    for i in indices:
        window.spectrum_list.item(i).setSelected(True)
    panel.add_selected()


def _set_ct(panel):
    for e in panel.experiments:
        e["transitions"] = "ct"
    panel._fill_experiment_table()
    panel.orientations_spin.setValue(987)


def _row(panel, label, name):
    t = panel.parameter_table
    for r in range(t.rowCount()):
        if t.item(r, _Q_SITE).text() == label and t.item(r, _Q_NAME).data(Qt.ItemDataRole.UserRole) == name:
            return r
    raise KeyError((label, name))


def _results(panel):
    t = panel.results_table
    return {t.verticalHeaderItem(r).text(): [t.item(r, c).text() for c in range(t.columnCount())]
            for r in range(t.rowCount())}


def test_the_section_is_last_in_fit():
    assert parse_expnos("5, 302, 12-14, 2; 49 11-9") == ["5", "302", "12", "13", "14", "2", "49",
                                                          "11", "10", "9"]
    with pytest.raises(ValueError, match="'abc'"):
        parse_expnos("5, abc")


def test_section_placement(qapp):
    window = MainWindow()
    tab = window.side_tabs.widget(window._fitting_tab_index)
    titles = [s.title() for s in tab.findChildren(CollapsibleSection)]
    # The last fitting method; only kept fits and the plot display follow it.
    assert titles[-3:] == ["Multi-spectrum", "Kept fit models", "Fit plot: components and colours"]
    assert isinstance(window.multi_fit_panel, MultiFitPanel)


def test_experiments_by_number_in_the_order_typed_with_x_values(qapp, tmp_path):
    ds = tmp_path / "dataset"
    ds.mkdir()
    for k, (expno, te, date) in enumerate((("5", 300, 1000), ("302", 320, 1600), ("12", 340, 1300))):
        path = _make_1d_dataset(ds, n=16, name=expno)
        with open(os.path.join(path, "acqus"), "a") as f:
            f.write(f"##$TE= {te}\n##$DATE= {date}\n")
    window = MainWindow()
    panel = window.multi_fit_panel
    problems = panel.add_expnos(str(ds), ["5", "302", "12", "99"])
    assert problems == [f"99: not in {ds}"]
    assert [os.path.basename(e["key"]) for e in panel.experiments] == ["5", "302", "12"]
    assert [panel.experiment_table.verticalHeaderItem(r).text() for r in range(3)] == ["1", "2", "3"]
    panel.x_source_combo.setCurrentIndex(panel.x_source_combo.findData("te"))
    panel.fill_x_values()
    assert [e["x"] for e in panel.experiments] == ["300", "320", "340"]
    assert panel.x_label_edit.text() == "T (K)"
    panel.x_source_combo.setCurrentIndex(panel.x_source_combo.findData("time"))
    panel.fill_x_values()
    assert [e["x"] for e in panel.experiments] == ["0", "10", "5"]
    # Reorder, then an already-open experiment is not added twice.
    panel.experiment_table.selectRow(2)
    panel.move(-1)
    assert [os.path.basename(e["key"]) for e in panel.experiments] == ["5", "12", "302"]
    panel.add_expnos(str(ds), ["12"])
    assert len(panel.experiments) == 3 and len(window.session) == 3


def test_a_series_is_fitted_and_its_trend_shown(qapp, _dialogs):
    window = MainWindow()
    panel = window.multi_fit_panel
    cqs = [4.0, 4.1, 4.3]
    indices = [_add(window, [_al(cq=cq)], f"T{k}", seed=k) for k, cq in enumerate(cqs)]
    # Any order: fit the last first ("Add selected" adds in list order,
    # so one at a time here; Up/Down reorder).
    for i in (indices[2], indices[0], indices[1]):
        _add_selected(window, panel, [i])
    assert [e["label"] for e in panel.experiments] == [window.session.label(i) for i in
                                                       (indices[2], indices[0], indices[1])]
    for e, x in zip(panel.experiments, ("340", "300", "320")):
        e["x"] = x
    _set_ct(panel)
    panel.set_sites("27Al", [_al(delta=10, cq=3.8, eta=0.5, fwhm=1.0)])
    assert panel.preset_combo.currentData() == "series"
    assert panel.fit(show_progress=False), _dialogs
    fits = panel.result["experiments"]
    assert panel.result["mode"] == "sequential"
    got = [next(p["value"] for p in f["parameters"] if p["name"] == "C_Q") for f in fits]
    assert got == pytest.approx([4.3, 4.0, 4.1], abs=0.02)
    table = _results(panel)
    assert list(table)[0].startswith("Al1: δiso") and "Al1: C_Q (MHz)" in table
    assert panel.results_table.columnCount() == 3
    # The trend is drawn against x, sorted.
    panel.show_combo.setCurrentIndex(panel.show_combo.findText("Trend: Al1: C_Q (MHz)"))
    panel.show_selected()
    line = window.fit_canvas.ax.lines[0]
    assert list(line.get_xdata()) == [300, 320, 340]
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    panel.show_combo.setCurrentIndex(0)
    panel.show_selected()
    assert "CSA, Quad & J-coupling fit" in window.fit_canvas.ax.get_title()
    assert "experiment,x_value" in panel.csv_text().splitlines()[0]
    assert "MULTI-SPECTRUM FIT REPORT" in panel.text_report()


def test_two_fields_share_the_site_through_the_gui(qapp, _dialogs):
    window = MainWindow()
    panel = window.multi_fit_panel
    a = _add(window, [_al(fwhm=0.8)], "9.4 T", nu0=104.3e6, mas=14e3)
    b = _add(window, [_al(fwhm=0.6)], "14.1 T", nu0=156.4e6, mas=22e3, seed=1)
    _add_selected(window, panel, [a, b])
    assert [e["mas_Hz"] for e in panel.experiments] == [14e3, 22e3]
    _set_ct(panel)
    panel.set_sites("27Al", [_al(delta=10.5, cq=4.0, eta=0.5, fwhm=1.0)])
    panel.preset_combo.setCurrentIndex(panel.preset_combo.findData("fields"))
    panel.apply_preset()
    combo = panel.parameter_table.cellWidget(_row(panel, "Al1", "C_Q"), _Q_SCOPE)
    assert combo.currentData() == "global"
    assert panel.parameter_table.cellWidget(_row(panel, "Al1", "fwhm"), _Q_SCOPE).currentText() == \
        "per nucleus (Hz + ppm)"
    assert panel.fit(show_progress=False), _dialogs
    assert panel.result["mode"] == "joint"
    table = _results(panel)
    cq = table["Al1: C_Q (MHz) [global]"]
    assert cq[0] == cq[1] and float(cq[0].split(" ")[0]) == pytest.approx(4.2, abs=0.02)
    assert "width_ppm" in panel.results_label.text()


def test_isotopes_have_their_own_sites_and_scopes_can_be_changed(qapp, _dialogs):
    window = MainWindow()
    panel = window.multi_fit_panel
    h = _add(window, [Site("1H", 5.0, label="OH", fwhm=0.2)], "1H", nu0=400e6, mas=0, lo=0, hi=10,
             nucleus="1H", transitions="all")
    d = _add(window, [Site("2H", 5.0, label="OH", fwhm=0.2)], "2H", nu0=61.4e6, mas=0, lo=0,
             hi=10, nucleus="2H", seed=1, transitions="all")
    _add_selected(window, panel, [h, d])
    assert panel.nuclei() == ["1H", "2H"]
    panel.nucleus_combo.setCurrentText("1H")
    panel.site_editor.add_site({"label": "OH", "delta_iso": 5.1, "amount": 1.0, "fwhm": 0.3,
                                "lineshape": "gaussian", "eta": 0.5, "interactions": []})
    assert [d["label"] for d in panel.sites_by_nucleus["1H"]] == ["OH"]
    panel.nucleus_combo.setCurrentText("2H")
    assert panel.sites_by_nucleus["2H"] == []
    panel.copy_sites_from("1H")
    assert [s["label"] for s in panel.site_editor.site_dicts] == ["OH"]
    panel.preset_combo.setCurrentIndex(panel.preset_combo.findData("isotopes"))
    panel.apply_preset()
    # The δ offset row exists (OH in both nuclei) and starts fixed.
    r = _row(panel, "OH", "delta_offset")
    assert panel.parameter_table.item(r, _Q_FREE).checkState() == Qt.CheckState.Unchecked
    # Changing a scope by hand makes the situation Custom.
    panel.parameter_table.cellWidget(_row(panel, "OH", "fwhm"), _Q_SCOPE).setCurrentIndex(1)
    assert panel.preset_combo.currentData() == "custom"
    for e in panel.experiments:
        e["transitions"] = "all"
    panel.orientations_spin.setValue(10)
    assert panel.fit(show_progress=False), _dialogs
    table = _results(panel)
    delta = table["OH: δiso (ppm) [global]"]
    assert delta[0] == delta[1] and float(delta[0].split(" ")[0]) == pytest.approx(5.0, abs=0.01)


def test_the_fit_is_kept_with_every_spectrum_in_it_and_comes_back(qapp, _dialogs):
    window = MainWindow()
    panel = window.multi_fit_panel
    indices = [_add(window, [_al(cq=cq)], f"E{k}", seed=k) for k, cq in enumerate((4.1, 4.3))]
    _add_selected(window, panel, indices)
    _set_ct(panel)
    panel.set_sites("27Al", [_al(delta=10, cq=3.9, eta=0.5, fwhm=1.0)])
    assert panel.fit(show_progress=False), _dialogs
    shown = _results(panel)
    for position, i in enumerate(indices):
        saved = window.session.spectra[i].metadata.raw["fits"]["multi"]
        json.dumps(saved)
        assert f"(spectrum {position + 1} of 2" in saved["target"]
        text = io_history.render_log(window.session.info(i)["history"])
        assert "Multi-spectrum fit of" in text and "Al1: C_Q (MHz)" in text
    # A fresh panel state, then selecting one of the spectra brings it back.
    panel.set_state({})
    panel.clear_result()
    window._select_single_row(indices[1])
    assert [e["label"] for e in panel.experiments] == [window.session.label(i) for i in indices]
    assert _results(panel) == shown
    assert panel.show_combo.currentText() == f"Fit: {window.session.label(indices[1])}"
    panel.show_selected()  # its own curves were saved with it
    # ...but it never overwrites a different set being built.
    other = _add(window, [_al()], "other")
    panel.set_state({})
    _add_selected(window, panel, [other])
    window._select_single_row(indices[0])
    assert [e["label"] for e in panel.experiments] == [window.session.label(other)]


def test_bad_inputs_are_named(qapp, _dialogs):
    window = MainWindow()
    panel = window.multi_fit_panel
    assert not panel.fit_button.isEnabled()
    i = _add(window, [_al()], "one")
    _add_selected(window, panel, [i])
    panel.set_sites("27Al", [])
    assert not panel.fit(show_progress=False)
    assert "Add at least one 27Al site" in _dialogs[-1][1]
    panel.set_sites("27Al", [_al()])
    # "10 to 20" (and several ranges) are valid since 2026-10-16; a lone
    # limit is not.
    panel.experiment_table.item(0, _E_WINDOW).setText("10 to")
    assert not panel.fit(show_progress=False)
    assert "Window must be ppm ranges" in _dialogs[-1][1]


def test_panel_state_round_trips(qapp):
    window = MainWindow()
    panel = window.multi_fit_panel
    indices = [_add(window, [_al()], f"S{k}", seed=k) for k in range(2)]
    _add_selected(window, panel, indices)
    panel.experiment_table.item(1, _E_MAS).setText("15000")
    panel.experiment_table.item(0, _E_X).setText("280")
    panel.set_sites("27Al", [_al(), _al(label="Al2", delta=-5)])
    panel.parameter_table.cellWidget(_row(panel, "Al2", "C_Q"), _Q_SCOPE).setCurrentIndex(2)
    before = panel.get_state()
    panel.set_state(json.loads(json.dumps(before)))
    assert panel.get_state() == before
    assert panel.experiments[1]["mas_Hz"] == 15000 and panel.experiments[0]["x"] == "280"
    assert panel.parameter_table.cellWidget(_row(panel, "Al2", "C_Q"), _Q_SCOPE).currentData() == "global"
