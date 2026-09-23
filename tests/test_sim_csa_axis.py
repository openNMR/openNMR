"""Simulate > CSA: the ppm axis follows the tensor, delta_iso +/- 1.5 x
|span or Delta| (explicit user request, 2026-09-23)."""
import pytest

from opennmr.gui.main_window import MainWindow
from opennmr.notebook import csa_axis_limits
from opennmr.notebook.simulations import csa_definition, make_static_csa_spectrum

from test_gui import qapp  # noqa: F401 -- fixture


@pytest.mark.parametrize("d_iso, aniso, expected", [
    (0.0, 100.0, (-150.0, 150.0)),
    (50.0, -200.0, (-250.0, 350.0)),     # sign of Delta does not matter
    (10.0, 0.0, (5.0, 15.0)),            # zero anisotropy: 5 x FWHM floor (fwhm 1)
])
def test_limits(d_iso, aniso, expected):
    assert csa_axis_limits(d_iso, aniso, fwhm=1.0) == pytest.approx(expected)


@pytest.mark.parametrize("convention, aniso, asym", [
    ("Haeberlen", 200.0, 0.0), ("Haeberlen", -200.0, 1.0),
    ("Herzfeld-Berger", 300.0, 1.0), ("Herzfeld-Berger", 300.0, -1.0)])
def test_the_whole_static_pattern_fits_inside(convention, aniso, asym):
    # The point of the rule: nothing of the pattern falls off the axis,
    # for either convention and the extreme asymmetries.
    import numpy as np
    low, high = csa_axis_limits(20.0, aniso, fwhm=2.0)
    x = np.linspace(low, high, 4096)
    y = make_static_csa_spectrum(csa_definition(20.0, aniso, asym, convention=convention), x,
                                 fwhm=2.0, n_theta=60, n_phi=120)["real_spec"]
    y = np.abs(y) / np.max(np.abs(y))
    assert max(y[0], y[-1]) < 0.01


def test_axis_follows_shift_and_anisotropy(qapp):
    window = MainWindow()
    assert window.sim_csa_auto_axis_check.isChecked()
    window.sim_csa_diso_spin.setValue(30.0)
    window.sim_csa_aniso_spin.setValue(-80.0)
    assert window.sim_csa_ppm_min_spin.value() == pytest.approx(-90.0)
    assert window.sim_csa_ppm_max_spin.value() == pytest.approx(150.0)
    assert window.sim_csa_auto_axis_check.isChecked()   # our own update is not a hand edit


def test_a_hand_set_limit_stops_the_following_and_ticking_snaps_back(qapp):
    window = MainWindow()
    window.sim_csa_ppm_max_spin.setValue(400.0)
    assert not window.sim_csa_auto_axis_check.isChecked()
    window.sim_csa_aniso_spin.setValue(10.0)
    assert window.sim_csa_ppm_max_spin.value() == pytest.approx(400.0)
    window.sim_csa_auto_axis_check.setChecked(True)
    d_iso = window.sim_csa_diso_spin.value()
    assert window.sim_csa_ppm_max_spin.value() == pytest.approx(d_iso + max(15.0, 5 * window.sim_csa_fwhm_spin.value(), 1.0))


def test_the_simulation_uses_the_followed_axis(qapp):
    window = MainWindow()
    window.sim_csa_nucleus_edit.setText("13C") if hasattr(window, "sim_csa_nucleus_edit") else None
    window.sim_csa_diso_spin.setValue(100.0)
    window.sim_csa_aniso_spin.setValue(200.0)
    window.sim_csa_orientations_spin.setValue(500)
    window._on_simulate_csa_clicked()
    x = window._simulated_csa_spectrum.axes[0].ppm
    assert min(x) == pytest.approx(-200.0) and max(x) == pytest.approx(400.0)
