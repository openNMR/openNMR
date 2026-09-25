"""Simulate > Combined Hamiltonian (gui/combined_panels.py): sites built
by adding interactions, simulated through notebook.combined_hamiltonian
-- the GUI result must equal the notebook call it describes."""
import numpy as np
import pytest
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined

from test_gui import qapp  # noqa: F401 -- fixture


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def _panel(qapp):  # noqa: F811
    window = MainWindow()
    return window, window.combined_sim_panel


def test_build_a_site_by_adding_interactions_and_simulate(qapp, _dialogs):  # noqa: F811
    window, p = _panel(qapp)
    p.nucleus_combo.setCurrentText("27Al")
    p.larmor_spin.setValue(130.3)
    p.regime_combo.setCurrentIndex(p.regime_combo.findData("mas"))
    p.mas_spin.setValue(15000)
    p.transitions_combo.setCurrentIndex(p.transitions_combo.findData("ct"))
    p.diso_spin.setValue(12.0)
    p.fwhm_spin.setValue(0.5)
    q = p.add_interaction("quadrupolar")
    ed = p.interaction_editors[0]
    ed.cq.setValue(4.2)
    ed.eta_q.setValue(0.35)
    p.add_interaction("csa")
    csa_ed = [e for e in p.interaction_editors if e.data["type"] == "csa"][0]
    csa_ed.aniso.setValue(40.0)
    csa_ed.asym.setValue(0.2)
    csa_ed.euler[1].setValue(90.0)
    p.add_interaction("jcoupling")
    j_ed = [e for e in p.interaction_editors if e.data["type"] == "jcoupling"][0]
    j_ed.partner.setCurrentText("31P")
    j_ed.n.setValue(2)
    j_ed.j.setValue(150.0)
    site = p.site_dicts[0]
    assert q["C_Q"] == 4.2 and [i["type"] for i in site["interactions"]] == \
        ["quadrupolar", "csa", "jcoupling"]
    assert csa_ed.euler[1].parentWidget().isEnabled()          # Euler: there is an EFG
    assert p.simulate() and not _dialogs

    expected = simulate_combined(
        Site("27Al", 12.0, label="Site 1", fwhm=0.5).add_quadrupolar(4.2, 0.35)
        .add_csa(40.0, 0.2, euler=(0, 90, 0)).add_jcoupling("31P", 2, 150.0),
        p.result["axis"], larmor_Hz=130.3e6, mas_Hz=15000, transitions="ct")
    assert np.allclose(p.spectrum.data, expected["real_spec"])
    assert "C_Q 4.2 MHz" in p.results_label.text() and "MAS 15 kHz" in p.results_label.text()
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    before = len(window.session.spectra)
    p.export_button.click()
    assert len(window.session.spectra) == before + 1
    assert window.session.spectra[-1].metadata.raw["simulation"]["kind"] == "combined"


def test_one_csa_and_one_efg_per_site_and_no_efg_for_spin_half(qapp, _dialogs):  # noqa: F811
    _w, p = _panel(qapp)
    p.nucleus_combo.setCurrentText("27Al")
    assert p.add_interaction("csa") is not None
    assert p.add_interaction("csa") is None and _dialogs
    p._sync_menu()
    assert not p._actions["csa"].isEnabled() and p._actions["jcoupling"].isEnabled()
    p.add_interaction("jcoupling")
    p.add_interaction("jcoupling")                       # any number of J groups
    assert sum(i["type"] == "jcoupling" for i in p.site_dicts[0]["interactions"]) == 2
    csa_ed = [e for e in p.interaction_editors if e.data["type"] == "csa"][0]
    assert not csa_ed.euler[0].parentWidget().isEnabled()  # no EFG: Euler angles off
    p.nucleus_combo.setCurrentText("31P")
    p._sync_menu()
    assert not p._actions["quadrupolar"].isEnabled()
    assert not p.transitions_combo.isEnabled()
    csa_ed.remove_button.click()
    assert [i["type"] for i in p.site_dicts[0]["interactions"]] == ["jcoupling", "jcoupling"]


def test_several_sites_duplicate_rename_remove_and_the_per_site_plot(qapp, _dialogs):  # noqa: F811
    window, p = _panel(qapp)
    p.nucleus_combo.setCurrentText("27Al")
    p.add_interaction("quadrupolar")
    p.duplicate_site()
    assert [s["label"] for s in p.site_dicts] == ["Site 1", "Site 1 (2)"]
    p.label_edit.setText("Al(oct)")
    p._on_label_edited("Al(oct)")
    p.diso_spin.setValue(-20.0)
    assert p.site_list.item(1).text() == "Al(oct)" and p.site_dicts[1]["delta_iso"] == -20.0
    p.site_list.setCurrentRow(0)
    assert p.diso_spin.value() == 0.0                      # editor follows the selection
    p.transitions_combo.setCurrentIndex(p.transitions_combo.findData("ct"))
    assert p.simulate()
    axes = window.simulation_canvas.figure.axes
    assert [l.get_label() for l in axes[0].get_lines()] == ["Total", "Site 1", "Al(oct)"]
    p.remove_site()
    assert len(p.site_dicts) == 1 and p.site_list.count() == 1


def test_copy_as_python_reproduces_the_gui_result(qapp, _dialogs):  # noqa: F811
    _w, p = _panel(qapp)
    p.nucleus_combo.setCurrentText("13C")
    p.larmor_spin.setValue(100.6)
    p.diso_spin.setValue(20.0)
    p.add_interaction("csa", aniso=150.0, asym=0.4)
    p.add_interaction("jcoupling", nuclide="1H", n=1, J_Hz=150.0)
    assert p.simulate()
    p.copy_python()
    code = QGuiApplication.clipboard().text()
    assert "add_csa(aniso=150.0" in code
    ns = {}
    exec(code, ns)
    assert np.allclose(ns["spec"]["real_spec"], p.spectrum.data)


def test_seeded_from_the_selected_spectrum_and_refusals(qapp, _dialogs):  # noqa: F811
    window, p = _panel(qapp)
    x = np.linspace(50, -50, 512)
    spectrum = NMRSpectrum(data=np.zeros(512), axes=[NMRAxis(kind="frequency", ppm=x)],
                           metadata=NMRMetadata(vendor="simulated", nucleus="23Na", sfo1=105.8,
                                                raw={"acqus": {"MASR": 12500}}))
    window._add_derived_spectrum(spectrum, "Na")
    assert p.nucleus_combo.currentText() == "23Na" and p.larmor_spin.value() == pytest.approx(105.8)
    assert p.regime_combo.currentData() == "mas" and p.mas_spin.value() == 12500
    p.mas_spin.setValue(0)
    assert not p.simulate() and "spinning rate" in _dialogs[-1][1]
    p.remove_site()
    p.regime_combo.setCurrentIndex(0)
    assert not p.simulate() and "at least one site" in _dialogs[-1][1]
