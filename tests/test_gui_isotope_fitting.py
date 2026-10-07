"""Simulate / Fit ▸ Combined Hamiltonian with sites of other nuclei: the
per-peak Nucleus and "Same chemistry as" rows, "Add isotope partner",
and a 47Ti + 49Ti fit run from the panel."""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.combined_panels import _P_NAME, _P_SITE, CombinedHamiltonianPanel
from opennmr.gui.main_window import MainWindow
from opennmr.notebook import isotopes
from opennmr.notebook.combined_hamiltonian import Site, combined_axis_limits, simulate_combined

from test_gui import qapp  # noqa: F401 -- qapp is a fixture

NU0 = 22.55e6


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def _ti_panel(panel):
    panel.nucleus_combo.setCurrentText("47Ti")
    panel.larmor_spin.setValue(NU0 / 1e6)
    panel.transitions_combo.setCurrentIndex(panel.transitions_combo.findData("ct"))
    panel.orientations_spin.setValue(987)
    panel.set_state({"widgets": {}, "current": 0, "sites": [
        {"label": "Ti1", "delta_iso": -790.0, "amount": 1.0, "fwhm": 10.0,
         "lineshape": "gaussian", "eta": 0.5,
         "interactions": [{"type": "quadrupolar", "C_Q": 3.8, "eta_Q": 0.3}]}]})
    panel.site_list.setCurrentRow(0)


def _sim_panel():
    return CombinedHamiltonianPanel(lambda _s: None, lambda *_a: None)


def test_add_isotope_partner_links_a_49ti_peak(qapp):
    panel = _sim_panel()
    _ti_panel(panel)
    new = panel.add_isotope_partner()
    assert new["label"] == "Ti1 (49Ti)" and new["isotope"] == "49Ti"
    assert new["link"] == {"to": "Ti1", "free": []}
    assert new["interactions"][0]["C_Q"] == pytest.approx(3.8 * 24.7 / 30.2)
    # selected: shown, not editable; the note says where it sits and what it follows
    assert panel.current_site() is new
    assert not panel.diso_spin.isEnabled() and not panel.amount_spin.isEnabled()
    assert not panel.add_interaction_button.isEnabled()
    assert all(not e.isEnabled() for e in panel.interaction_editors)
    assert "+266.61 ppm" in panel.link_note.text() and "Follows Ti1" in panel.link_note.text()
    assert panel.link_combo.currentData() == "Ti1"
    assert panel.settings()["observe"] == "47Ti"
    sites = panel.build_sites()
    assert [s.nucleus for s in sites] == ["47Ti", "49Ti"]


def test_editing_the_parent_moves_the_partner(qapp):
    panel = _sim_panel()
    _ti_panel(panel)
    panel.add_isotope_partner()
    panel.site_list.setCurrentRow(0)
    panel.diso_spin.setValue(-700.0)
    panel.site_list.setCurrentRow(1)
    assert panel.diso_spin.value() == pytest.approx(-700.0)
    assert panel.simulate()
    assert set(panel.result["result"]["sites"]) == {"Ti1", "Ti1 (49Ti)"}


def test_rename_follows_and_remove_unlinks_keeping_values(qapp):
    panel = _sim_panel()
    _ti_panel(panel)
    partner = panel.add_isotope_partner()
    panel.site_list.setCurrentRow(0)
    panel.label_edit.setText("Ti(oct)")
    panel._on_label_edited("Ti(oct)")
    assert partner["link"]["to"] == "Ti(oct)"
    panel.remove_site()
    assert "link" not in partner
    assert partner["interactions"][0]["C_Q"] == pytest.approx(3.8 * 24.7 / 30.2)
    assert partner["isotope"] == "49Ti"


def test_unlinking_gives_back_its_own_parameters(qapp):
    panel = _sim_panel()
    _ti_panel(panel)
    partner = panel.add_isotope_partner()
    panel.link_combo.setCurrentIndex(0)                    # "(own parameters)"
    assert "link" not in partner
    assert panel.diso_spin.isEnabled() and panel.add_interaction_button.isEnabled()
    panel.link_combo.setCurrentIndex(panel.link_combo.findData("Ti1"))
    assert partner["link"]["to"] == "Ti1"


def test_a_background_nucleus_is_chosen_per_peak(qapp):
    panel = _sim_panel()
    panel.nucleus_combo.setCurrentText("23Na")
    panel.larmor_spin.setValue(105.84)
    panel.add_site()
    panel.site_nucleus_combo.setCurrentIndex(panel.site_nucleus_combo.findData("63Cu"))
    site = panel.current_site()
    assert site["isotope"] == "63Cu"
    assert not panel.link_combo.isEnabled()          # no other copper peak to share with
    assert "63Cu on the 23Na scale" in panel.link_note.text()
    assert panel.build_sites()[-1].nucleus == "63Cu"
    # a spin-1/2 choice drops a quadrupolar interaction it cannot have
    panel.add_interaction("quadrupolar")
    panel.site_nucleus_combo.setCurrentIndex(panel.site_nucleus_combo.findData("13C"))
    assert not [i for i in site["interactions"] if i["type"] == "quadrupolar"]


def test_the_single_nucleus_panel_is_unchanged(qapp):
    panel = _sim_panel()
    assert panel.site_nucleus_combo.currentData() == ""
    assert "observe" not in panel.settings()
    assert not panel.link_combo.isEnabled()
    assert not panel._sform.isRowVisible(panel.own_widget)
    assert "isotope" not in panel.current_site() and "link" not in panel.current_site()


def test_state_round_trip_keeps_nucleus_and_link(qapp):
    panel = _sim_panel()
    _ti_panel(panel)
    panel.add_isotope_partner()
    state = panel.get_state()
    other = _sim_panel()
    other.set_state(state)
    other.nucleus_combo.setCurrentText("47Ti")
    assert other.site_dicts[1]["link"] == {"to": "Ti1", "free": []}
    assert other.site_dicts[1]["isotope"] == "49Ti"
    assert "linked to Ti1" in state["extra"]["Site Ti1 (49Ti)"]


def test_the_multi_spectrum_editor_does_not_offer_other_nuclei(qapp):
    window = MainWindow()
    editor = window.multi_fit_panel.site_editor
    assert not editor.add_partner_button.isVisibleTo(editor)
    assert not editor._sform.isRowVisible(editor.site_nucleus_combo)
    assert not editor._sform.isRowVisible(editor.link_combo)


# ---- Fit ▸ Combined Hamiltonian ----------------------------------------------------------------

def _load_ti(window):
    truth = Site("47Ti", -800, label="t", fwhm=8.0).add_quadrupolar(4.0, 0.2)
    sites = [truth, truth.isotope_partner()]
    lo, hi = combined_axis_limits(sites, NU0, transitions="ct")
    x = np.linspace(hi, lo, 2048)
    y = simulate_combined(sites, x[::-1], NU0, transitions="ct", n_zcw=987)["real_spec"][::-1]
    y = y / y.max() * 1000 + np.random.default_rng(0).standard_normal(x.size) * 3
    spectrum = NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x)],
                           metadata=NMRMetadata(vendor="simulated", nucleus="47Ti",
                                                sfo1=NU0 / 1e6))
    return window._add_derived_spectrum(spectrum, "47Ti/49Ti static")


def _rows(panel):
    t = panel.parameter_table
    return [(t.item(r, _P_SITE).text(), t.item(r, _P_NAME).data(Qt.ItemDataRole.UserRole))
            for r in range(t.rowCount())]


def test_fit_panel_table_and_fit_with_a_linked_partner(qapp):
    window = MainWindow()
    panel = window.combined_fit_panel
    _load_ti(window)
    _ti_panel(panel)
    panel.add_isotope_partner()
    rows = _rows(panel)
    assert not [r for r in rows if r[0] == "Ti1 (49Ti)"]     # follows Ti1: no rows of its own
    panel.own_amount_check.setChecked(True)
    assert ("Ti1 (49Ti)", "amount") in _rows(panel)
    panel.own_amount_check.setChecked(False)
    assert panel.fit()
    fit = panel.result
    v = {(p["label"], p["name"]): p["value"] for p in fit["parameters"]}
    assert fit["quality"]["r2"] > 0.995
    assert v[("Ti1", "C_Q")] == pytest.approx(4.0, abs=0.02)
    assert v[("Ti1", "delta_iso")] == pytest.approx(-800, abs=1)
    labels = [panel.results_table.verticalHeaderItem(i).text()
              for i in range(panel.results_table.rowCount())]
    assert "Ti1 (49Ti): share (%)" in labels
    # fitted values back as the start keep the partner's nucleus and link
    panel.use_fitted_as_start()
    partner = panel.site_dicts[1]
    assert partner["isotope"] == "49Ti" and partner["link"]["to"] == "Ti1"
    assert panel.preview_now(quiet=False)
    assert set(panel.preview["site_spectra"]) == {"Ti1", "Ti1 (49Ti)"}


def test_partner_factor_shown_matches_the_notebook(qapp):
    panel = _sim_panel()
    _ti_panel(panel)
    panel.add_isotope_partner()
    f = isotopes.partner_factors("47Ti", "49Ti")
    assert f"× {f['amount']:.4g}" in panel.link_note.text()
