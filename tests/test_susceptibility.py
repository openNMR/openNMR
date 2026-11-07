"""Susceptibility units (explicit user request, 2026-11-07): every common
unit to the SI volume susceptibility the BMS simulator uses."""

import numpy as np
import pytest

from opennmr.notebook import susceptibility as X


def test_curie_constant():
    # N_A μ_B² / 3k_B = 0.12505 emu K mol⁻¹ per μ_B² (the familiar 1/7.997)
    assert X.CURIE_CGS == pytest.approx(0.125049, rel=1e-5)
    assert 1 / X.CURIE_CGS == pytest.approx(7.9968, rel=1e-4)


def test_water_from_molar_and_mass_units():
    """Water: χ_mol = −12.97e-6 cm³ mol⁻¹ (cgs), χ_g = −0.720e-6 cm³ g⁻¹,
    χ_v = −9.03e-6 (SI) tabulated."""
    assert X.to_si_volume(-12.97e-6, "cgs_molar", molar_mass=18.015, density=0.998) == \
        pytest.approx(-9.03e-6, rel=2e-3)
    assert X.to_si_volume(-0.720e-6, "cgs_mass", density=0.998) == pytest.approx(-9.03e-6,
                                                                              rel=2e-3)


def test_unit_relations():
    assert X.to_si_volume(1e-3) == 1e-3
    assert X.to_si_volume(1000.0, "si_volume_ppm") == pytest.approx(1e-3)
    assert X.to_si_volume(1e-4, "cgs_volume") == pytest.approx(4 * np.pi * 1e-4)
    # SI mass (m³ kg⁻¹) = 4π × 10⁻³ cgs mass (cm³ g⁻¹)
    cgs = 2e-5
    assert X.to_si_volume(4 * np.pi * 1e-3 * cgs, "si_mass", density=3.0) == \
        pytest.approx(X.to_si_volume(cgs, "cgs_mass", density=3.0))
    # SI molar (m³ mol⁻¹) = 4π × 10⁻⁶ cgs molar (cm³ mol⁻¹)
    kw = {"molar_mass": 157.85, "density": 3.43}
    assert X.to_si_volume(4 * np.pi * 1e-6 * 0.0146, "si_molar", **kw) == \
        pytest.approx(X.to_si_volume(0.0146, "cgs_molar", **kw))
    v = X.to_si_volume([1e-4, 2e-4, 3e-4], "cgs_volume")
    assert np.allclose(v, 4 * np.pi * np.array([1e-4, 2e-4, 3e-4]))


def test_molar_volume_two_ways():
    vm = X.molar_volume(molar_mass=157.85, density=3.43)
    cell = vm * 4 / X.N_A * 1e24
    assert X.molar_volume(cell_volume=cell, Z=4) == pytest.approx(vm, rel=1e-12)
    with pytest.raises(ValueError, match="molar mass"):
        X.to_si_volume(0.01, "cgs_molar", density=3.0)
    with pytest.raises(ValueError, match="Z"):
        X.molar_volume(cell_volume=300.0)


def test_mu_eff_curie_weiss():
    kw = {"molar_mass": 157.85, "density": 3.43}
    chi_mol = X.CURIE_CGS * 5.92 ** 2 / 300.0
    assert X.to_si_volume(5.92, "mu_eff", T=300.0, **kw) == \
        pytest.approx(X.to_si_volume(chi_mol, "cgs_molar", **kw))
    two = X.to_si_volume(5.92, "mu_eff", T=300.0, n_ions=2, chi_dia=-1e-4, **kw)
    assert two == pytest.approx(X.to_si_volume(2 * chi_mol - 1e-4, "cgs_molar", **kw))
    cw = X.to_si_volume(5.92, "mu_eff", T=300.0, theta=-60.0, **kw)
    assert cw == pytest.approx(X.to_si_volume(X.CURIE_CGS * 5.92 ** 2 / 360.0, "cgs_molar",
                                              **kw))
    with pytest.raises(ValueError, match="above θ"):
        X.to_si_volume(5.92, "mu_eff", T=50.0, theta=60.0, **kw)


def test_bad_units_and_missing_inputs():
    with pytest.raises(ValueError, match="Unknown unit"):
        X.to_si_volume(1.0, "tesla")
    with pytest.raises(ValueError, match="density"):
        X.to_si_volume(1e-5, "cgs_mass")


def test_describe():
    text = X.describe([5.92], "mu_eff", molar_mass=157.85, density=3.43, T=300.0)
    assert "μ_eff" in text and "T = 300 K" in text and "χ_v (SI)" in text
