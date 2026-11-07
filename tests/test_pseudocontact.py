"""Pseudocontact shifts from a susceptibility tensor (point dipole), user
request 2026-11-06: the axial/rhombic formula, χ from g and Z (equal to the
hyperfine route for a metal-centred spin), lattice sums and the bulk term,
the Δχ fit, ORCA's susceptibility output, and the map."""

import math
import os

import numpy as np
import pytest

from opennmr.notebook import paramagnetic_nmr as pm
from opennmr.notebook import pseudocontact as pcs
from opennmr.notebook.crystal_nmr import H_PLANCK, MU_B, MU_N

DATA = os.path.join(os.path.dirname(__file__), "..", "test_data", "dft", "paramagnetic")
real = pytest.mark.skipif(not os.path.isdir(DATA), reason="test_data/dft/paramagnetic not present")


def test_axial_rhombic_formula():
    dax, drh = 12.0, 4.0
    chi = pcs.chi_from_axial_rhombic(dax, drh)
    for th, ph, r in ((0.3, 0.2, 5.0), (1.2, 2.0, 7.5), (math.pi / 2, 0.0, 4.0)):
        v = r * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
        expect = 1e6 * 1e-32 * (dax * (3 * math.cos(th) ** 2 - 1)
                                + 1.5 * drh * math.sin(th) ** 2 * math.cos(2 * ph)) / (
            12 * math.pi * (r * 1e-10) ** 3)
        assert pcs.pcs_iso(v, chi) == pytest.approx(expect, rel=1e-12)
        assert np.trace(pcs.pcs_tensor(v, chi)) / 3 == pytest.approx(expect, rel=1e-12)


def test_the_magic_angle_cone_has_no_axial_shift():
    chi = pcs.chi_from_axial_rhombic(10.0)
    th = math.acos(1 / math.sqrt(3))
    assert pcs.pcs_iso([math.sin(th) * 6, 0, math.cos(th) * 6], chi) == pytest.approx(0, abs=1e-12)


def test_an_isotropic_chi_gives_a_traceless_tensor_only():
    chi = 5e-32 * np.eye(3)
    t = pcs.pcs_tensor([1.0, 2.0, 3.0], chi)
    assert np.trace(t) == pytest.approx(0, abs=1e-12)
    assert np.max(np.abs(t)) > 0


def test_axial_rhombic_round_trip_and_convention():
    chi = pcs.chi_from_axial_rhombic(-7.0, -2.0, euler=(30, 60, 100), chi_iso=4.0)
    ar = pcs.axial_rhombic(chi)
    assert ar["dchi_ax"] == pytest.approx(-7.0) and ar["dchi_rh"] == pytest.approx(-2.0)
    assert ar["chi_iso"] == pytest.approx(4.0)
    assert 0 <= ar["rhombicity"] <= 2 / 3 + 1e-12
    assert np.linalg.det(ar["axes"]) == pytest.approx(1.0)


def test_point_dipole_equals_the_hyperfine_route_for_a_metal_centred_spin():
    """σ_PC from g·Z·A_dip with a point-dipole A_dip = the PCS of χ = μ0μ_B² gZgᵀ
    (isotropic g, anisotropic Z from ZFS): two independent paths."""
    S, T = 1.5, 180.0
    Z = pm.spin_susceptibility(S, T, pm.zfs_tensor(12.0, 2.0, axes=pcs.rotation_zyz(10, 70, 40)))
    g = pm.G_E * np.eye(3)
    chi = pcs.chi_from_g(g, Z)
    gN = 1.40482
    for v in ([1.2, -2.0, 2.5], [0.3, 4.0, -1.0]):
        r = np.linalg.norm(v) * 1e-10
        u = np.asarray(v) / np.linalg.norm(v)
        A = 1e-7 * pm.G_E * MU_B * gN * MU_N * (3 * np.outer(u, u) - np.eye(3)) / (
            H_PLANCK * r ** 3) / 1e6
        s = pm.paramagnetic_shielding(g, Z, A, gN)
        assert -np.trace(s) / 3 == pytest.approx(pcs.pcs_iso(v, chi), rel=1e-9)


def test_molar_cgs_to_molecular_si():
    assert pcs.chi_molecule_from_molar_cgs(1.0) == pytest.approx(4e-6 * math.pi / 6.02214076e23)


def test_lattice_sum_of_an_isotropic_chi_vanishes_and_bulk_term_signs():
    a = 6.0
    L = np.eye(3) * a
    chi = 3e-32 * np.eye(3)
    nuc = [{"label": "n", "position": np.array([a / 2, a / 2, a / 2])}]
    met = [{"label": "M", "position": np.zeros(3)}]
    res = pcs.pcs_shifts(nuc, met, chi, lattice=L, radius=25.0)
    assert res["sites"][0]["iso"] == pytest.approx(0.0, abs=1e-9)
    chi_v = 3e-32 / (a * 1e-10) ** 3
    for shape, factor in (("sphere", 0.0), ("mas_rotor", 0.0), ("cylinder_parallel", 1 / 3),
                          ("cylinder_perpendicular", -1 / 6)):
        r = pcs.pcs_shifts(nuc, met, chi, lattice=L, radius=10.0, sample_shape=shape)
        assert r["bulk"] == pytest.approx(1e6 * factor * chi_v, abs=1e-9)


def test_lattice_sum_converges_and_is_reported():
    L = np.diag([7.0, 7.0, 9.0])
    chi = pcs.chi_from_axial_rhombic(5.0, 1.0)
    nuc = [{"label": "H1", "position": np.array([1.5, 0.5, 2.0])}]
    met = [{"label": "M", "position": np.zeros(3)}]
    r40 = pcs.pcs_shifts(nuc, met, chi, lattice=L, radius=40.0)["sites"][0]
    r60 = pcs.pcs_shifts(nuc, met, chi, lattice=L, radius=60.0)["sites"][0]
    assert r40["iso"] == pytest.approx(r60["iso"], abs=0.05 * abs(r60["iso"]) + 1e-3)
    assert r40["convergence"] >= 0 and r40["n_terms"] > 100
    single = pcs.pcs_shifts(nuc, met, chi)["sites"][0]
    assert single["n_terms"] == 1


def test_fit_recovers_chi_and_needs_five_shifts():
    rng = np.random.default_rng(3)
    pos = rng.normal(size=(12, 3)) * 4.0
    chi = pcs.chi_from_axial_rhombic(-9.0, 3.0, euler=(20, 40, 60))
    d = pcs.pcs_iso(pos, chi)
    fit = pcs.fit_chi_tensor(pos, np.zeros(3), d)
    assert np.allclose(fit["chi"], chi, atol=1e-12 * 1e-32 * 10)
    # Reported in the convention 0 ≤ Δχ_rh/Δχ_ax ≤ 2/3: (−9, +3) in the input's
    # frame is (−9, −3) once its x and y are swapped -- the same tensor.
    assert fit["dchi_ax"] == pytest.approx(-9.0) and fit["dchi_rh"] == pytest.approx(-3.0)
    assert 0 <= fit["rhombicity"] <= 2 / 3
    assert fit["Q"] < 1e-10
    with pytest.raises(ValueError, match="five"):
        pcs.fit_chi_tensor(pos[:4], np.zeros(3), d[:4])


SUS_TEXT = """
                                 * O   R   C   A *
---------------------------------
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
  Ce     0.000000    0.000000    0.000000
  N      0.000000    0.000000    2.500000
  H      1.000000    0.000000    3.000000

------------------------------------------------------------------------
       TEMPERATURE DEPENDENT MOLAR MAGNETIC SUSCEPTIBILITY TENSOR
------------------------------------------------------------------------

TEMPERATURE/K:   100.00
Tensor in molecular frame (cm3*K/mol):
    0.500000     0.000000     0.000000
    0.000000     0.500000     0.000000
    0.000000     0.000000     0.800000

TEMPERATURE/K:   300.00
Tensor in molecular frame (cm3*K/mol):
    0.600000     0.000000     0.000000
    0.000000     0.600000     0.000000
    0.000000     0.000000     0.900000

------------------------------------------------------------------------
"""


def test_read_orca_susceptibility_and_interpolate(tmp_path):
    p = tmp_path / "sus.out"
    p.write_text(SUS_TEXT)
    sus = pcs.read_orca_susceptibility(p)
    assert len(sus["blocks"]) == 1 and list(sus["blocks"][0]["T"]) == [100.0, 300.0]
    assert [a["label"] for a in sus["atoms"]] == ["Ce1", "N1", "H1"]
    c = pcs.chi_at_temperature(sus, 200.0)
    assert np.allclose(c["molar_cgs"], np.diag([0.55, 0.55, 0.85]) / 200.0)
    assert c["chiT_iso"] == pytest.approx((0.55 * 2 + 0.85) / 3)
    with pytest.raises(ValueError, match="outside"):
        pcs.chi_at_temperature(sus, 400.0)
    # an axial χ along z: positive on the axis (N), as for Δχ_ax > 0
    r = pcs.pcs_shifts([sus["atoms"][1]], [sus["atoms"][0]], c["chi"])
    assert r["sites"][0]["iso"] > 0


def test_read_sus_file(tmp_path):
    p = tmp_path / "x.sus"
    p.write_text("   0.0   1.0 ---------------    0.42\n   0.0   300.0 ---------------    0.67\n")
    s = pcs.read_orca_sus_file(p)
    assert list(s["T"]) == [1.0, 300.0] and s["chiT"][-1] == pytest.approx(0.67)


@real
def test_real_ce_tensor_matches_its_isotropic_sus_file():
    sus = pcs.read_orca_susceptibility(os.path.join(DATA, "CeFeCN6_casscf_susceptibility_excerpt.out"))
    iso = pcs.read_orca_sus_file(os.path.join(DATA, "CeFeCN6_nevpt2.casscf.sus"))
    c = pcs.chi_at_temperature(sus, 298.0)
    assert c["chiT_iso"] == pytest.approx(float(np.interp(298.0, iso["T"], iso["chiT"])), rel=1e-5)


def test_map_and_python(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    chi = pcs.chi_from_axial_rhombic(8.0, 2.0)
    u, v, d, axes = pcs.pcs_map(chi, "xz", 5.0, n=21)
    assert d.shape == (21, 21) and np.allclose(axes.T @ axes, np.eye(3))
    fig = pcs.plot_pcs_map(chi, atoms=[{"label": "H1", "position": np.array([2.0, 0, 1.0])}],
                           extent=5.0)
    assert "delta" in fig.axes[-1].get_ylabel()
    p = tmp_path / "sus.out"
    p.write_text(SUS_TEXT)
    code = pcs.pcs_to_python(source="orca", path=str(p), T=200.0, structure=str(p),
                             metals=["Ce1"], element="H", radius=10.0, sample_shape=None)
    assert "read_orca_susceptibility" in code
