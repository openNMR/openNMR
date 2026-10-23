"""Fit ▸ Multi-spectrum: Keep Fits in Open Experiments, and Copy sites
from (explicit user request, 2026-10-14: the first did not exist, the
second was enabled all the time and did nothing for a one-nucleus fit)."""

import json

import numpy as np
import pytest
from PySide6.QtWidgets import QInputDialog

from opennmr.notebook import fit_models as nb_fit_models
from opennmr.notebook.combined_hamiltonian import Site

from test_gui import qapp  # noqa: F401 -- qapp is a fixture
from test_gui_multi_fit import _add, _add_selected, _al, _dialogs, _set_ct  # noqa: F401
from opennmr.gui.main_window import MainWindow


def _fitted(window, cqs=(4.1, 4.3)):
    panel = window.multi_fit_panel
    indices = [_add(window, [_al(cq=cq)], f"E{k}", seed=k) for k, cq in enumerate(cqs)]
    _add_selected(window, panel, indices)
    _set_ct(panel)
    panel.set_sites("27Al", [_al(delta=10, cq=3.9, eta=0.5, fwhm=1.0)])
    return panel, indices


def test_keep_button_waits_for_a_fit(qapp, _dialogs):
    window = MainWindow()
    panel, _ = _fitted(window)
    assert panel.keep_fits_button.isVisibleTo(panel)
    assert not panel.keep_fits_button.isEnabled()
    assert panel.keep_fits("x") == []
    assert _dialogs and "Run Fit All first" in _dialogs[-1][1]


def test_fits_are_kept_listed_and_superimposed_in_compare(qapp, _dialogs):
    window = MainWindow()
    panel, indices = _fitted(window)
    assert panel.fit(show_progress=False), _dialogs
    assert panel.keep_fits_button.isEnabled()
    n_before = len(window.session)
    rows = panel.keep_fits("VT series")
    assert len(rows) == 2 and len(window.session) == n_before + 2
    for position, (i, row) in enumerate(zip(indices, rows)):
        models = window._fit_models(i)
        assert [m["name"] for m in models] == ["VT series"]
        m = models[0]
        assert m["kind"] == "multi" and m["kind_label"] == "Multi-spectrum fit"
        fit = panel.result["experiments"][position]
        assert np.allclose(m["y_fit"], fit["y_fit"]) and np.allclose(m["y"], fit["y"])
        # Its statistics are over the fit window, as the fit's own R².
        assert nb_fit_models.fit_statistics(m)["r2"] == pytest.approx(fit["quality"]["r2"],
                                                                         abs=1e-6)
        assert m["labels"] == ["Al1"]
        raw = window.session.spectra[row].metadata.raw
        assert raw["fit_model_source"]["spectrum"] == window.session.label(i)
        json.dumps(raw["fit_model"])           # saved with the session as it is
    # All of them ticked together in Compare, superimposed.
    assert window.layout_mode_combo.currentIndex() == 1
    ticked = [window.compare_list.item(k) for k in range(window.compare_list.count())]
    assert len(ticked) == 2
    # Keeping again under the same name makes second kept fits, not a mix-up.
    panel.keep_fits("again")
    assert [m["name"] for m in window._fit_models(indices[0])] == ["VT series", "again"]


def test_a_kept_multi_fit_goes_back_into_the_panel(qapp, _dialogs):
    window = MainWindow()
    panel, indices = _fitted(window)
    assert panel.fit(show_progress=False), _dialogs
    panel.keep_fits("kept")
    shown = panel.results_label.text()
    model = window._fit_models(indices[1])[0]
    assert window._fit_model_restorable(model)
    panel.set_state({})
    panel.clear_result()
    assert not panel.keep_fits_button.isEnabled()
    window._restore_fit_model(indices[1], model["id"])
    assert panel.results_label.text() == shown
    # Only that spectrum's curves come back, so only it can be kept again.
    assert panel.keepable_count() == 1


def test_keep_asks_for_a_name(qapp, _dialogs, monkeypatch):
    window = MainWindow()
    panel, indices = _fitted(window, cqs=(4.2,))
    assert panel.fit(show_progress=False), _dialogs
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("  ", True))
    panel.keep_fits()
    assert window._fit_models(indices[0])[0]["name"] == "Multi-spectrum fit"
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("", False))
    assert panel.keep_fits() == []


# ---- Copy sites from --------------------------------------------------------

def _no_other_panels(window):
    """Fit / Simulate ▸ Solid-state NMR start with a peak; empty them."""
    for panel in (window.combined_fit_panel, window.combined_sim_panel):
        panel.site_dicts = []


def _menu_texts(panel):
    panel._fill_copy_menu()
    return [(a.text(), a.isEnabled()) for a in panel._copy_menu.actions()]


def test_copy_menu_says_when_there_is_nothing(qapp, _dialogs):
    window = MainWindow()
    panel = window.multi_fit_panel
    _no_other_panels(window)
    assert not panel.copy_sites_button.isEnabled()       # no nucleus yet
    _add_selected(window, panel, [_add(window, [_al()], "E0")])
    assert panel.copy_sites_button.isEnabled()
    assert "▾" not in panel.copy_sites_button.text()     # Qt draws the arrow
    texts = _menu_texts(panel)
    assert len(texts) == 1 and not texts[0][1] and texts[0][0].startswith("Nothing to copy")


def test_copy_from_the_fit_panel_for_one_nucleus(qapp, _dialogs):
    window = MainWindow()
    panel = window.multi_fit_panel
    _add_selected(window, panel, [_add(window, [_al()], "E0")])
    _no_other_panels(window)
    fitp = window.combined_fit_panel
    fitp.nucleus_combo.setCurrentText("27Al")
    fitp.add_site({"label": "AlO6", "delta_iso": 9.0, "amount": 2.0, "fwhm": 0.7,
                   "lineshape": "gaussian", "eta": 0.5,
                   "interactions": [{"type": "quadrupolar", "C_Q": 4.0, "eta_Q": 0.3}]})
    texts = _menu_texts(panel)
    assert texts[0][0].startswith("Fit ▸ Solid-state NMR (27Al, 1 peak)") and texts[0][1]
    panel._copy_menu.actions()[0].trigger()
    assert [d["label"] for d in panel.sites_by_nucleus["27Al"]] == ["AlO6"]
    got = panel.sites_by_nucleus["27Al"][0]
    assert got["delta_iso"] == 9.0
    assert [i["C_Q"] for i in got["interactions"] if i["type"] == "quadrupolar"] == [4.0]
    assert any(panel.parameter_table.item(r, 0).text() == "AlO6"
               for r in range(panel.parameter_table.rowCount()))
    # Again: every label is already there, and it says so.
    panel._copy_menu.actions()[0].trigger()
    assert len(panel.sites_by_nucleus["27Al"]) == 1
    assert "Nothing copied" in _dialogs[-1][1]


def test_copy_between_nuclei_lists_only_those_with_sites(qapp, _dialogs):
    window = MainWindow()
    panel = window.multi_fit_panel
    h = _add(window, [Site("1H", 5.0, label="OH", fwhm=0.2)], "1H", nu0=400e6, mas=0, lo=0,
             hi=10, nucleus="1H", transitions="all")
    d = _add(window, [Site("2H", 5.0, label="OH", fwhm=0.2)], "2H", nu0=61.4e6, mas=0, lo=0,
             hi=10, nucleus="2H", seed=1, transitions="all")
    _add_selected(window, panel, [h, d])
    _no_other_panels(window)
    panel.nucleus_combo.setCurrentText("2H")
    assert [t for t, _ in _menu_texts(panel)][0].startswith("Nothing to copy")
    panel.nucleus_combo.setCurrentText("1H")
    panel.site_editor.add_site({"label": "OH", "delta_iso": 5.1, "amount": 1.0, "fwhm": 0.3,
                                "lineshape": "gaussian", "eta": 0.5, "interactions": []})
    panel.nucleus_combo.setCurrentText("2H")
    texts = _menu_texts(panel)
    assert texts[0][0] == "1H (this fit, 1 site)"
    panel._copy_menu.actions()[0].trigger()
    assert [s["label"] for s in panel.site_editor.site_dicts] == ["OH"]
    assert panel.copy_sites_from("1H") == 0
