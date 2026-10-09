"""Lebedev grids as a powder-averaging option (``powder_scheme='lebedev'``).

Notebook: ``lebedev_angles``, ``get_powder_grid``, every spectrum function
that takes ``powder_scheme``. GUI: the three "Powder averaging" combos
(Simulate CSA, Simulate quadrupolar, Fit CSA).
"""
import warnings

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from opennmr import notebook as nmr
from opennmr.notebook import simulations as sims


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _xyz(theta, phi):
    return np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)


def test_sizes_are_the_lebedev_laikov_family():
    sizes = list(sims.lebedev_sizes())
    assert sizes[:8] == [6, 14, 26, 38, 50, 74, 86, 110]
    assert sizes[-1] == 5810
    assert 302 in sizes and 590 in sizes


@pytest.mark.parametrize("n_min,expected", [(1, 6), (6, 6), (7, 14), (300, 302), (5810, 5810)])
def test_smallest_rule_with_at_least_n_points(n_min, expected):
    theta, phi, w = sims.lebedev_angles(n_min)
    assert len(theta) == len(phi) == len(w) == expected
    assert sims.powder_grid_size("lebedev", n_min) == expected


def test_weights_positive_normalised_and_angles_in_range():
    theta, phi, w = sims.lebedev_angles(590)
    assert np.all(w > 0) and w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all((theta >= 0) & (theta <= np.pi))
    assert np.all((phi >= 0) & (phi < 2 * np.pi))
    x, y, z = _xyz(theta, phi)
    np.testing.assert_allclose(x ** 2 + y ** 2 + z ** 2, 1.0)


def test_integrates_spherical_polynomials_exactly():
    # <z^2> = 1/3, <x^2 y^2 z^2> = 1/105, <z^4> = 1/5, <P2(cos)^2> = 1/5 over the sphere.
    theta, phi, w = sims.lebedev_angles(110)                  # degree 17
    x, y, z = _xyz(theta, phi)
    assert np.sum(w * z ** 2) == pytest.approx(1 / 3, abs=1e-13)
    assert np.sum(w * x ** 2 * y ** 2 * z ** 2) == pytest.approx(1 / 105, abs=1e-13)
    assert np.sum(w * z ** 4) == pytest.approx(1 / 5, abs=1e-13)
    assert np.sum(w * (0.5 * (3 * z ** 2 - 1)) ** 2) == pytest.approx(1 / 5, abs=1e-13)
    assert np.sum(w * x * z ** 3) == pytest.approx(0.0, abs=1e-13)


def test_asking_for_more_than_5810_uses_5810_and_warns():
    with pytest.warns(UserWarning, match="largest Lebedev grid"):
        theta, _, _ = sims.lebedev_angles(10000)
    assert len(theta) == 5810
    assert sims.powder_grid_size("lebedev", 10000) == 5810


def test_get_powder_grid_dispatches_and_rejects_unknown():
    theta, _, w = sims.get_powder_grid(10, 20, powder_scheme="lebedev", n_zcw=170)
    assert len(theta) == 170 and w.sum() == pytest.approx(1.0)
    theta, _, _ = sims.get_powder_grid(10, 20, powder_scheme="lebedev")   # n_theta*n_phi
    assert len(theta) == 230
    with pytest.raises(ValueError, match="lebedev"):
        sims.get_powder_grid(10, 20, powder_scheme="sophe")


def test_mas_sidebands_converge_with_far_fewer_points_than_zcw():
    # The reason to offer it: a few hundred Lebedev points reproduce a
    # converged sideband pattern better than ZCW at a similar count.
    csa = sims.csa_definition(0.0, -120.0, 0.4)
    ppm = np.linspace(-300, 300, 1501)
    kw = dict(larmor_Hz=100e6, mas_Hz=3000.0, n_time=128, n_sidebands=16, fwhm=1.0)

    def spec(scheme, n):
        return sims.finite_mas_csa_spectrum(csa, ppm, powder_scheme=scheme, n_zcw=n,
                                            **kw)["normalised_real_spec"]

    reference = spec("zcw", 17711)
    lebedev_error = np.abs(spec("lebedev", 302) - reference).max()
    zcw_error = np.abs(spec("zcw", 377) - reference).max()
    assert lebedev_error < 1e-4
    assert lebedev_error < zcw_error


def test_static_csa_and_quadrupolar_run_with_lebedev():
    csa = sims.csa_definition(0.0, 80.0, 0.3)
    ppm = np.linspace(-150, 150, 601)
    static = sims.make_static_csa_spectrum(csa, ppm, fwhm=5.0, powder_scheme="lebedev",
                                           n_zcw=5810)
    y = np.asarray(static["real_spec"] if isinstance(static, dict) else static)
    assert np.all(np.isfinite(y)) and y.max() > 0

    site = [{"C_Q_MHz": 2.0, "eta_Q": 0.2, "d_iso": 0.0, "fwhm": 2.0}]
    axis = np.linspace(-200, 200, 801)
    for mas in (0.0, 10000.0):
        quad = sims.quadrupolar_multisite_spectrum(site, 1.5, 105.8e6, axis, mas_Hz=mas,
                                                   powder_scheme="lebedev", n_zcw=974)
        y = np.asarray(quad["real_spec"])
        assert np.all(np.isfinite(y)) and y.max() > 0


def test_exported_from_the_single_notebook_import():
    assert nmr.lebedev_angles is sims.lebedev_angles
    assert nmr.powder_grid_size is sims.powder_grid_size


# ---- GUI ----------------------------------------------------------------

def test_every_powder_combo_offers_lebedev_with_advice(qapp):
    from opennmr.gui.main_window import MainWindow

    window = MainWindow()
    for combo in (window.sim_csa_powder_combo, window.sim_quad_powder_combo,
                  window.csa_powder_combo):
        assert combo.findData("lebedev") >= 0
        # appended, so saved states that stored an index keep their meaning
        assert combo.itemData(0) == "zcw" and combo.itemData(1) == "rectangular"
        assert "Lebedev" in combo.toolTip() and "static" in combo.toolTip()
    # Fit CSA keeps its own explanation as well as the shared one
    assert "every fit iteration costs" in window.csa_powder_combo.toolTip()


def test_grid_kwargs_and_actual_counts(qapp):
    from opennmr.gui.main_window import MainWindow

    assert MainWindow._csa_grid_kwargs("lebedev", 300) == {"n_zcw": 300}
    assert MainWindow._powder_count("lebedev", 300) == 302
    assert MainWindow._powder_count("zcw", 300) == 377
    assert MainWindow._powder_count("rectangular", 300) == 300


def test_gui_simulates_a_mas_csa_pattern_with_lebedev(qapp, monkeypatch):
    from opennmr.gui.main_window import MainWindow
    from test_gui import _fast_csa_kwargs

    window = MainWindow()
    _fast_csa_kwargs(window)
    window.sim_csa_nucleus_edit.setText("13C")
    window.sim_csa_regime_combo.setCurrentIndex(window.sim_csa_regime_combo.findData(False))
    window.sim_csa_mas_spin.setValue(5000.0)
    if window.sim_csa_larmor_spin.value() <= 0:
        window.sim_csa_larmor_spin.setValue(100.0)
    window.sim_csa_powder_combo.setCurrentIndex(window.sim_csa_powder_combo.findData("lebedev"))
    window.sim_csa_orientations_spin.setValue(300)
    seen = {}
    real = sims.csa_multisite_spectrum

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(sims, "csa_multisite_spectrum", spy)
    window._on_simulate_csa_clicked()

    assert seen.get("powder_scheme") == "lebedev" and seen.get("n_zcw") == 300
    assert window.sim_csa_export_button.isEnabled()
