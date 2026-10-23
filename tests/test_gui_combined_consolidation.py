"""Fit and Simulate condensed into Solid-state NMR (explicit user
request, 2026-10-06): peaks are added, interactions added to each, and
every variant of each interaction stays reachable -- quadrupolar first
order, second order and exact; J first order and strongly coupled
(exact spin system). Plus the live preview of the fit's starting model.
"""

import numpy as np
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.collapsible_section import CollapsibleSection
from opennmr.notebook import combined_fitting as cfit
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined

from test_gui import qapp  # noqa: F401 -- fixture


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def _titles(window, index):
    tab = window.side_tabs.widget(index)
    return [s.title() for s in tab.findChildren(CollapsibleSection)]


def _editor(panel, kind):
    return next(e for e in panel.interaction_editors if e.data["type"] == kind)


# ---- the menus ---------------------------------------------------------------

def test_fit_and_simulate_offer_one_combined_entry_point(qapp):
    window = MainWindow()
    assert _titles(window, window._fitting_tab_index) == [
        "Solution NMR", "Solid-state NMR",
        "Quadrupolar (Czjzek distribution)", "pNMR (CIF bond pathways)", "SIMPSON spectrum",
            "Multi-spectrum", "Kept fit models"]
    assert _titles(window, window._simulations_tab_index) == [
        "Solid-state NMR", "Czjzek distribution", "DFT tensors (CASTEP, VASP, ORCA, Gaussian)",
        "Diffusion (PFG)", "Shaped pulse (Bloch)", "pNMR (CIF)", "SIMPSON import"]
    # The retired panels still exist (kept fits and saved states restore
    # through them), out of sight.
    assert window._retired_fit_host.isHidden() and window._retired_sim_host.isHidden()
    assert window._retired_fit_host.isAncestorOf(window.quad_fit_panel)
    assert window._retired_fit_host.isAncestorOf(window.csa_sites_table)
    assert window._retired_sim_host.isAncestorOf(window.sim_quad_method_combo)
    # The add-peak wording.
    panel = window.combined_fit_panel
    assert panel.add_site_button.text() == "Add peak"
    assert "first order, second order or exact" in panel.add_interaction_button.toolTip()


def test_quadrupolar_interaction_offers_every_level_of_theory(qapp):
    window = MainWindow()
    for panel in (window.combined_fit_panel, window.combined_sim_panel):
        panel.nucleus_combo.setCurrentText("27Al")
        panel.add_interaction("quadrupolar")
        editor = _editor(panel, "quadrupolar")
        combo = editor.theory_combo
        assert [combo.itemData(i) for i in range(combo.count())] == ["second", "exact", "first"]
        # Editor -> panel, and panel -> editor.
        combo.setCurrentIndex(combo.findData("exact"))
        assert panel.order_combo.currentData() == "exact"
        assert panel.settings()["order"] == "exact"
        panel.order_combo.setCurrentIndex(panel.order_combo.findData("first"))
        assert combo.currentData() == "first"


def test_j_interaction_offers_strong_coupling_in_fit(qapp):
    window = MainWindow()
    panel = window.combined_fit_panel
    panel.nucleus_combo.setCurrentText("1H")
    panel.diso_spin.setValue(2.10)
    panel.label_edit.setText("A")
    panel._on_label_edited("A")
    panel.add_site()
    panel.diso_spin.setValue(2.25)
    panel.label_edit.setText("B")
    panel._on_label_edited("B")
    panel.add_interaction("jcoupling")
    editor = _editor(panel, "jcoupling")
    combo = editor.treatment_combo
    assert [combo.itemData(i) for i in range(combo.count())] == ["first", "strong"]
    assert window.combined_fit_model.model() == "peaks"
    editor._on_treatment(combo.findData("strong"), panel)
    # The exact spin-system model, its spins the peaks; the J stays first order.
    assert window.combined_fit_model.model() == "spin_system"
    assert window.combined_fit_model.stack.currentWidget() is window.jcoupling_fit_panel
    spins = window.jcoupling_fit_panel.spin_editor.spins()
    assert [(s["label"], s["nuclide"], s["shift_ppm"]) for s in spins] == [
        ("A", "1H", pytest.approx(2.10)), ("B", "1H", pytest.approx(2.25))]
    assert combo.currentData() == "first"
    assert window.jcoupling_fit_panel.spin_mode()
    assert window.jcoupling_fit_panel.model_combo.isHidden()
    # Spins already typed are never overwritten.
    window.combined_fit_model.show_model("peaks")
    panel.request_strong_coupling()
    assert [s["label"] for s in window.jcoupling_fit_panel.spin_editor.spins()] == ["A", "B"]


def test_strong_coupling_in_simulate_runs_the_exact_spin_system(qapp):
    window = MainWindow()
    panel = window.combined_sim_panel
    panel.nucleus_combo.setCurrentText("1H")
    panel.larmor_spin.setValue(400.0)
    panel.diso_spin.setValue(2.0)
    panel.add_site()
    panel.diso_spin.setValue(2.05)
    assert "strong" in panel._actions
    panel._sync_menu()
    assert panel._actions["strong"].isEnabled()
    panel._actions["strong"].trigger()
    j = window.jcoupling_sim_panel
    assert window.combined_sim_model.model() == "spin_system"
    assert j.nucleus_combo.currentText() == "1H"
    assert j.field_spin.value() == pytest.approx(400.0 / 42.577478, rel=1e-4)
    labels = j.spin_editor.labels()
    j.spin_editor.set_J(labels[0], labels[1], 12.0)
    assert j.simulate()
    # An AB quartet (Δν = 20 Hz, J = 12 Hz): four lines, inner taller (roofing).
    y = np.asarray(j.spectrum.data)
    x = np.asarray(j.spectrum.axes[0].ppm)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(y, height=0.05 * y.max())
    assert len(peaks) == 4
    heights = y[peaks[np.argsort(x[peaks])]]
    assert heights[1] > heights[0] and heights[2] > heights[3]


def test_compare_levels_of_theory_in_combined_simulate(qapp, monkeypatch):
    window = MainWindow()
    drawn = {}
    panel = window.combined_sim_panel
    monkeypatch.setattr(panel, "_draw_orders",
                        lambda comparison, nucleus, mhz: drawn.update(c=comparison, n=nucleus))
    panel.nucleus_combo.setCurrentText("27Al")
    panel.larmor_spin.setValue(130.3)
    panel.orientations_spin.setValue(987)
    panel.points_spin.setValue(2048)
    panel.add_interaction("quadrupolar", C_Q=4.0, eta_Q=0.3)
    panel.order_combo.setCurrentIndex(panel.order_combo.findData("exact"))
    panel.compare_orders_check.setChecked(True)
    assert panel.simulate()
    assert sorted(drawn["c"]) == ["exact", "first", "second"]
    np.testing.assert_allclose(panel.spectrum.data, drawn["c"]["exact"]["real_spec"])
    # Second order differs from first; exact is close to second (C_Q ≪ ν0).
    first, second, exact = (drawn["c"][k]["real_spec"] for k in ("first", "second", "exact"))
    assert np.abs(second - first).max() > 0.1 * second.max()
    assert np.abs(exact - second).max() < 0.1 * second.max()


def test_sends_open_the_right_model(qapp):
    window = MainWindow()
    window.combined_fit_model.show_model("spin_system")
    window._send_combined_simulation_to_fit([{"label": "S", "delta_iso": 1.0, "amount": 1.0,
                                              "fwhm": 1.0, "lineshape": "gaussian",
                                              "interactions": []}],
                                            {"regime": "static"}, "1H")
    assert window.combined_fit_model.model() == "peaks"
    window._send_spin_system_to_simulation("1H", 400e6, [{"label": "A", "nuclide": "1H",
                                                          "shift_ppm": 1.0, "n": 1}], [],
                                           1.0, "lorentzian", 0.5)
    assert window.combined_sim_model.model() == "spin_system"


# ---- the starting-model preview -----------------------------------------------

def _al_spectrum(window, delta=10.0, cq=4.0):
    x = np.linspace(80, -80, 2048)
    site = Site("27Al", delta, label="Al", fwhm=2.0).add_quadrupolar(cq, 0.3)
    y = simulate_combined([site], x[::-1], 130.3e6, transitions="ct", n_zcw=4181)["real_spec"][::-1]
    y = y / y.max() * 1000 + 5.0
    spectrum = NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x)],
                           metadata=NMRMetadata(vendor="simulated", nucleus="27Al", sfo1=130.3))
    return window._add_derived_spectrum(spectrum, "Al")


def test_starting_model_is_the_fits_own_first_step():
    x = np.linspace(-80, 60, 2048)
    truth = [Site("27Al", 10, label="A", amount=2, fwhm=2).add_quadrupolar(4, 0.3),
             Site("27Al", -20, label="B", amount=1, fwhm=2)]
    y = simulate_combined(truth, x, 130.3e6, transitions="ct", n_zcw=2584)["real_spec"] * 1e3 + 3
    results = {"x_ppm": x[::-1], "real_spectrum": y[::-1], "SFO1": 130.3}
    start = [Site("27Al", 12, label="A", fwhm=2).add_quadrupolar(4, 0.3),
             Site("27Al", -20, label="B", fwhm=2)]
    for estimate in (True, False):
        guess = cfit.starting_model(results, start, transitions="ct", n_zcw=2584,
                                    estimate_amounts=estimate)
        fixed = cfit.fit_combined(results, start, vary={"A": [], "B": []}, transitions="ct",
                                  n_zcw=2584, estimate_amounts=estimate)
        np.testing.assert_allclose(guess["start"], fixed["y_fit"], rtol=0, atol=1e-9)
        amounts = [p["value"] for p in fixed["parameters"] if p["name"] == "amount"]
        assert [guess["amounts"]["A"], guess["amounts"]["B"]] == pytest.approx(amounts)
        assert np.all(np.diff(guess["x"]) > 0)


def test_preview_draws_and_follows_each_change(qapp):
    window = MainWindow()
    _al_spectrum(window)
    panel = window.combined_fit_panel
    panel.seed_from_spectrum(window.session.spectra[0])
    panel.transitions_combo.setCurrentIndex(panel.transitions_combo.findData("ct"))
    panel.add_interaction("quadrupolar", C_Q=4.0, eta_Q=0.3)
    panel.diso_spin.setValue(10.0)
    assert panel.preview_now()
    assert window.fit_canvas.ax.get_title() == "Starting model (preview)"
    legend = window.fit_canvas.ax.get_legend()
    assert "Starting model" in [t.get_text() for t in legend.get_texts()]
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    first = panel.preview["start"].copy()
    assert "4,181 orientations" in panel.preview_label.text()
    assert "estimated from the data" in panel.preview_label.text()
    # The fit window and amounts of the fit; a change of δiso moves the model.
    panel.diso_spin.setValue(0.0)
    assert panel.preview_now()
    shift = (panel.preview["x"][np.argmax(panel.preview["start"])]
             - panel.preview["x"][np.argmax(first)])
    assert shift == pytest.approx(-10.0, abs=1.0)
    # Typed amounts, when asked for.
    panel.estimate_amounts_check.setChecked(False)
    panel.amount_spin.setValue(0.5)
    assert panel.preview_now()
    assert panel.preview["amounts"]["Site 1"] == pytest.approx(0.5 * 1005.0, rel=0.02)
    assert "as typed" in panel.preview_label.text()


def test_preview_is_live_while_shown_and_quiet_when_filled_by_code(qapp):
    window = MainWindow()
    _al_spectrum(window)
    window.show()
    window._show_side_tab("Fit")
    window._combined_fit_section.setExpanded(True)
    panel = window.combined_fit_panel
    panel.seed_from_spectrum(window.session.spectra[0])
    assert panel.isVisible()
    calls = []
    real = panel.preview_now
    panel.preview_now = lambda quiet=True: (calls.append(quiet), real(quiet))[1]
    panel._preview_timer.timeout.disconnect()
    panel._preview_timer.timeout.connect(lambda: panel.preview_now(quiet=True))
    panel.add_site()                      # adding a peak previews it
    QTest.qWait(panel.PREVIEW_DELAY_MS + 300)
    assert calls and panel.preview is not None
    n = len(calls)
    for value in (1.0, 2.0, 3.0):         # a burst of changes: one redraw
        panel.diso_spin.setValue(value)
    QTest.qWait(panel.PREVIEW_DELAY_MS + 300)
    assert len(calls) == n + 1
    panel.set_state(panel.get_state())    # restores never overdraw what is shown
    QTest.qWait(panel.PREVIEW_DELAY_MS + 300)
    assert len(calls) == n + 1
    panel.preview_check.setChecked(False)
    panel.diso_spin.setValue(4.0)
    QTest.qWait(panel.PREVIEW_DELAY_MS + 300)
    assert len(calls) == n + 1


def test_preview_without_a_spectrum_says_so_quietly(qapp, _dialogs):
    window = MainWindow()
    assert not window.combined_fit_panel.preview_now(quiet=True)
    assert "Select a spectrum" in window.combined_fit_panel.preview_label.text()
    assert not _dialogs


def test_fit_passes_the_amounts_choice(qapp, monkeypatch):
    window = MainWindow()
    _al_spectrum(window)
    panel = window.combined_fit_panel
    panel.seed_from_spectrum(window.session.spectra[0])
    seen = {}
    monkeypatch.setattr(cfit, "fit_combined",
                        lambda results, sites, **kw: (seen.update(kw), (_ for _ in ()).throw(
                            ValueError("stop")))[0])
    panel.estimate_amounts_check.setChecked(False)
    panel.fit()
    assert seen["estimate_amounts"] is False
