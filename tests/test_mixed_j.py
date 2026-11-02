"""Weak (first-order) and strong (exact) J-coupled sets on one spectrum, in the
notebook (``mixed_j_spectrum``) and in Simulate > Solution NMR (explicit user
request, 2026-11-02: switch between the weak and strong models, and have both
on the same spectrum)."""
import numpy as np
import pytest
from scipy.signal import find_peaks

import opennmr.notebook as nmr
from opennmr.notebook import mixed_j

B0 = 400.0 / 42.577478
LARMOR = 400.0e6


def _axis(lo=-0.5, hi=8.5, n=60000):
    return np.linspace(lo, hi, n)


def _weak(ppm=1.2, n=2, J=7.0, fwhm=1.0, intensity=1.0):
    return {"label": "CH3", "ppm": ppm, "fwhm": fwhm / LARMOR * 1e6,
            "relative_intensity": intensity,
            "couplings": [{"nuclide": "1H", "n": n, "J_Hz": J}]}


AB_SPINS = [{"label": "A", "nuclide": "1H", "shift_ppm": 7.30},
            {"label": "B", "nuclide": "1H", "shift_ppm": 7.26}]
AB_J = [{"a": "A", "b": "B", "J_Hz": 8.0}]


def test_mixed_is_the_sum_of_the_two_models():
    x = _axis()
    weak = [_weak()]
    mixed = nmr.mixed_j_spectrum(x, LARMOR, B0, weak, AB_SPINS, AB_J, strong_fwhm_Hz=1.0)
    only_weak = nmr.multi_site_jcoupled_spectrum(weak, LARMOR, x, fwhm=weak[0]["fwhm"])
    only_strong = nmr.spin_system_spectrum(AB_SPINS, AB_J, B0, x, fwhm_Hz=1.0)
    assert np.allclose(mixed["real_spec"], only_weak["real_spec"] + only_strong["real_spec"])
    assert len(mixed["strong"]) == 1 and mixed["skipped"] == []


def test_weak_and_strong_parts_alone_reproduce_their_models():
    x = _axis()
    only_weak = nmr.mixed_j_spectrum(x, LARMOR, B0, [_weak()])
    ref = nmr.multi_site_jcoupled_spectrum([_weak()], LARMOR, x, fwhm=_weak()["fwhm"])
    assert np.allclose(only_weak["real_spec"], ref["real_spec"]) and not only_weak["strong"]
    only_strong = nmr.mixed_j_spectrum(x, LARMOR, B0, strong_spins=AB_SPINS, strong_couplings=AB_J)
    ref = nmr.spin_system_spectrum(AB_SPINS, AB_J, B0, x, fwhm_Hz=1.0)
    assert np.allclose(only_strong["real_spec"], ref["real_spec"]) and only_strong["weak"] is None


def test_strong_part_is_roofed_and_weak_part_is_not():
    x = _axis()
    out = nmr.mixed_j_spectrum(x, LARMOR, B0, [_weak()], AB_SPINS, AB_J)
    y = out["real_spec"]
    ab = (x > 7.0) & (x < 7.6)
    peaks, _ = find_peaks(np.where(ab, y, 0), height=0.05 * y[ab].max())
    assert len(peaks) == 4                         # AB quartet, not two doublets
    h = y[peaks[np.argsort(x[peaks])]]
    assert h[1] > h[0] and h[2] > h[3]             # roofing toward each other
    tri = (x > 1.0) & (x < 1.4)
    p, _ = find_peaks(np.where(tri, y, 0), height=0.05 * y[tri].max())
    assert len(p) == 3                             # the weak CH3 is a first-order triplet
    r = y[p[np.argsort(x[p])]]
    assert r[1] == pytest.approx(2 * r[0], rel=0.02) and r[0] == pytest.approx(r[2], rel=0.02)


def test_integrals_are_the_numbers_of_nuclei():
    x = _axis()
    out = nmr.mixed_j_spectrum(x, LARMOR, B0, [_weak(n=2, intensity=3.0)], AB_SPINS, AB_J)
    dx = x[1] - x[0]
    weak_area = out["real_spec"][(x > 0.5) & (x < 2.0)].sum() * dx
    strong_area = out["real_spec"][(x > 6.5) & (x < 8.0)].sum() * dx
    assert weak_area / strong_area == pytest.approx(3.0 / 2.0, rel=0.02)


def test_independent_strong_sets_are_split_and_exact():
    spins = AB_SPINS + [{"label": "C", "nuclide": "1H", "shift_ppm": 4.10},
                        {"label": "D", "nuclide": "1H", "shift_ppm": 4.05}]
    couplings = AB_J + [{"a": "C", "b": "D", "J_Hz": 12.0}]
    sets = mixed_j.split_spin_system(spins, couplings)
    assert [[s["label"] for s in sp] for sp, _ in sets] == [["A", "B"], ["C", "D"]]
    x = _axis()
    out = nmr.mixed_j_spectrum(x, LARMOR, B0, strong_spins=spins, strong_couplings=couplings)
    # Uncoupled sets add exactly: the same as diagonalising all four spins together.
    full = nmr.spin_system_spectrum(spins, couplings, B0, x, fwhm_Hz=1.0)
    assert len(out["strong"]) == 2
    assert np.allclose(out["real_spec"], full["real_spec"], atol=1e-6 * full["real_spec"].max())


def test_a_set_without_the_observed_nucleus_is_skipped_not_drawn():
    spins = AB_SPINS + [{"label": "P", "nuclide": "31P", "shift_ppm": 0.0}]
    out = nmr.mixed_j_spectrum(_axis(), LARMOR, B0, strong_spins=spins, strong_couplings=AB_J)
    assert out["skipped"] == [["P"]] and len(out["strong"]) == 1


def test_split_rejects_a_coupling_to_an_unknown_spin():
    with pytest.raises(ValueError, match="no spin labelled Z"):
        mixed_j.split_spin_system(AB_SPINS, [{"a": "A", "b": "Z", "J_Hz": 1.0}])


# ---- the GUI -----------------------------------------------------------------
from test_gui import qapp  # noqa: E402,F401 -- fixture
from opennmr.gui.main_window import MainWindow  # noqa: E402


@pytest.fixture
def window(qapp):
    return MainWindow()


def _solution_panel(window):
    window._solution_sim_section.setExpanded(True)
    panel = window.jcoupling_sim_panel
    assert window._sim_spin_host.owner == "solution"
    panel.nucleus_combo.setCurrentText("1H")
    panel.field_spin.setValue(B0)
    return panel


def test_solution_simulate_offers_weak_strong_and_mixed(window):
    panel = _solution_panel(window)
    assert not panel.model_combo.isHidden()
    assert [panel.model_combo.itemData(i) for i in range(panel.model_combo.count())] == [
        "first_order", "spin_system", "mixed"]
    assert panel.model_combo.currentData() == "spin_system"      # as before
    # Solid-state NMR's strong-coupling model has no switch.
    window.combined_sim_model.show_model("spin_system")
    assert window._sim_spin_host.owner == "solid" and panel.model_combo.isHidden()


def test_solution_simulate_weak_model_draws_a_triplet(window):
    panel = _solution_panel(window)
    panel.model_combo.setCurrentIndex(panel.model_combo.findData("first_order"))
    assert panel.simulate()
    y, x = np.asarray(panel.spectrum.data), np.asarray(panel.spectrum.axes[0].ppm)
    peaks, _ = find_peaks(y, height=0.05 * y.max())
    assert len(peaks) == 4                      # the default CH2 site: a 1:3:3:1 quartet (3 H)


def test_solution_simulate_mixed_has_both_models_on_one_spectrum(window):
    panel = _solution_panel(window)
    panel.model_combo.setCurrentIndex(panel.model_combo.findData("mixed"))
    assert panel.simulate()
    y, x = np.asarray(panel.spectrum.data), np.asarray(panel.spectrum.axes[0].ppm)
    # Two weak sites (CH3 triplet at 1.25, CH2 quartet at 2.65) and a strong AB at ~7.28.
    def n_peaks(lo, hi):
        m = (x > lo) & (x < hi)
        return len(find_peaks(np.where(m, y, 0), height=0.03 * y.max())[0])
    assert n_peaks(1.0, 1.5) == 3 and n_peaks(2.4, 2.9) == 4 and n_peaks(7.0, 7.6) == 4
    text = panel.results_label.text()
    assert "Weak (first order)" in text and "Strong set A, B" in text
    assert panel.export_button.isEnabled()
    # The same numbers as the notebook function.
    a = panel._last_args
    ref = nmr.mixed_j_spectrum(x, a["larmor_Hz"], a["B0_tesla"],
                               [{"label": s["label"], "ppm": s["center_ppm"], "fwhm": s["fwhm"],
                                 "relative_intensity": s["relative_intensity"],
                                 "couplings": s["couplings"]} for s in a["sites"]],
                               a["spins"], a["couplings"], strong_fwhm_Hz=a["fwhm_Hz"])
    assert np.allclose(y, ref["real_spec"])


def test_mixed_with_several_strong_sets_and_only_strong(window):
    panel = _solution_panel(window)
    panel.model_combo.setCurrentIndex(panel.model_combo.findData("mixed"))
    panel.mixed_weak_editor.sites_table.setRowCount(0)          # no weak sites at all
    panel.mixed_strong_editor.set_system(
        AB_SPINS + [{"label": "C", "nuclide": "1H", "shift_ppm": 4.10},
                    {"label": "D", "nuclide": "1H", "shift_ppm": 4.05}],
        AB_J + [{"a": "C", "b": "D", "J_Hz": 12.0}])
    assert panel.simulate()
    assert panel.results_label.text().count("Strong set") == 2
    assert "Weak (first order)" not in panel.results_label.text()


def test_mixed_with_nothing_to_simulate_warns_instead_of_crashing(window, monkeypatch):
    panel = _solution_panel(window)
    panel.model_combo.setCurrentIndex(panel.model_combo.findData("mixed"))
    panel.mixed_weak_editor.sites_table.setRowCount(0)
    panel.mixed_strong_editor.set_system([], [])
    shown = []
    from opennmr.gui import jcoupling_panels
    monkeypatch.setattr(jcoupling_panels.QMessageBox, "warning",
                        lambda *args, **kw: shown.append(args[2]))
    assert panel.simulate() is False
    assert shown and "at least one" in shown[0]


def test_mixed_state_round_trips_and_notebook_code_runs(window):
    panel = _solution_panel(window)
    panel.model_combo.setCurrentIndex(panel.model_combo.findData("mixed"))
    assert panel.simulate()
    y0 = np.asarray(panel.spectrum.data).copy()
    state = panel.get_state()
    panel.mixed_weak_editor.sites_table.setRowCount(0)
    panel.model_combo.setCurrentIndex(panel.model_combo.findData("first_order"))
    panel.set_state(state)
    assert panel.mixed_mode() and panel.mixed_weak_editor.sites_table.rowCount() == 2
    assert panel.mixed_strong_editor.labels() == ["A", "B"]
    scope = {}
    exec(panel.notebook_code(), scope)                  # the copied code reproduces the figure
    assert np.allclose(scope["spec"]["real_spec"], y0)


def test_mixed_cannot_be_sent_to_fit_but_says_so(window, monkeypatch):
    panel = _solution_panel(window)
    panel.model_combo.setCurrentIndex(panel.model_combo.findData("mixed"))
    from opennmr.gui import jcoupling_panels
    told = []
    monkeypatch.setattr(jcoupling_panels.QMessageBox, "information",
                        lambda *args, **kw: told.append(args[2]))
    panel.send_to_fit()
    assert told and "one model at a time" in told[0]
