"""CRYSTAL paramagnetic shifts and quadrupolar tensors (user request,
2026-10-24): reading a properties output of any extension, the scaling
factor from φ, χ_M or (θ, μ_eff, S), the Fermi contact and spin dipolar
shifts, and C_Q / η_Q, checked against independent calculations (the
user's notebook formulas and CRYSTAL's own printed conversion)."""

import math

import numpy as np
import pytest

from opennmr.notebook import crystal_nmr as C
from opennmr.notebook import dft_nmr as D

# Constants of the user's script (SCRIPT_MHCF_Na_displacement_analysis), kept apart from the code
# under test so the check is independent of it.
_Q_E, _H, _ME, _MP = 1.60217653e-19, 6.6260693e-34, 0.91093826e-30, 1.67262171e-27
_HBAR = _H / (2 * np.pi)
_MUB = _Q_E * _HBAR / (2 * _ME)
_MUN = _Q_E * _HBAR / (2 * _MP)
_KB = 1.3806505e-23
_C_F = 1.99827885e+08            # the script's "CRYSTAL factor": bohr^-3 -> Hz (per g_N)


def script_phi(B0, T, th, S, mu_eff):
    return (B0 * mu_eff ** 2 * _MUB) / (3.0 * _KB * 2.0 * S * (T - th))


def script_fc_ppm(rho, B0, T, th, S, mu_eff):
    return 2.0e6 * _H * rho * _C_F * script_phi(B0, T, th, S, mu_eff) / (_MUN * B0)


HEADER = """ PROCESS            0  OF            1  WORKING

 *******************************************************************************
 *                               CRYSTAL17                                     *
 *******************************************************************************
 EEEEEEEEEE STARTING  DATE 29 01 2025 TIME 02:32:36.5

 *******************************************************************************
 Li2MnO3_test
 CRYSTAL - PROPERTIES - TYPE OF CALCULATION :  UNRESTRICTED OPEN SHELL
 *******************************************************************************
 SPIN POLARIZED SYSTEM: SUMMED SPIN DENSITY    3.00000000

 (EXCHANGE)[CORRELATION] FUNCTIONAL:(BECKE 88)[LEE-YANG-PARR]
 HYBRID EXCHANGE - PERCENTAGE OF FOCK EXCHANGE    20.0000

 DIRECT LATTICE VECTOR COMPONENTS (ANGSTROM)
         4.91717    0.00000    0.00000
         0.00000    8.52454    0.00000
        -1.62361    0.00000    9.47108
 *******************************************************************************
   ATOM N.AT.  SHELL    X(A)      Y(A)      Z(A)      EXAD       N.ELECT.
 *******************************************************************************
    1    3 LI    7     1.219     0.741    -0.030   1.700E-01       2.671
    2    3 LI    7    -1.219    -0.741     0.030   1.700E-01       2.671
    3   25 MN   16     2.053    -0.718     2.368   3.375E-01      24.653
    4    8 O    13     0.309     2.182     1.292   1.400E+00       8.357
 *******************************************************************************
 DE(K)                   0.000E+00  ENERGY LEVEL SHIFTING        0.00000
 TOTAL ENERGY -1.1135840590842E+04  CONVERGENCE ON ENER       -6.594E-08
"""

HF_LI = """
 POINT   ATOM  MASS NO  NUCLEAR G FACTOR
    {n}    {n} LI      6       0.8220575

 TENSOR IN ORIGINAL CARTESIAN AXES
 XX {xx:.6E} YY {yy:.6E} ZZ {zz:.6E} XY {xy:.6E} XZ {xz:.6E} YZ {yz:.6E}

 TENSOR IN PRINCIPAL AXES SYSTEM
 AA -2.100891E+00 BB -3.077994E+00 CC  5.178885E+00

 POINT   ATOM  MASS NO  NUCLEAR G FACTOR
    {n}    {n} LI      7       2.1709770

 TENSOR IN ORIGINAL CARTESIAN AXES
 XX {xx7:.6E} YY {yy7:.6E} ZZ {zz7:.6E} XY {xy7:.6E} XZ {xz7:.6E} YZ {yz7:.6E}

 TENSOR IN PRINCIPAL AXES SYSTEM
 AA -5.548256E+00 BB -8.128695E+00 CC  1.367695E+01
"""

#: 7Li anisotropic hyperfine tensor (MHz) of atom 1 and its (rotated) copy on atom 2.
A7 = {"xx": -4.839947, "yy": -8.100395, "zz": 12.94034, "xy": 0.08282991, "xz": -3.65703,
      "yz": 0.5754936}
G6, G7 = 0.8220575, 2.1709770
EFG_AU = {"XX": 4.553322e-03, "YY": -2.942458e-03, "ZZ": -1.610864e-03, "XY": 2.090836e-03,
          "XZ": 5.125410e-03, "YZ": -7.045908e-04}


def _hf_block(n):
    d = {k + "7": v for k, v in A7.items()}
    d.update({k: v * G6 / G7 for k, v in A7.items()})
    return HF_LI.format(n=n, **d)


def crystal_text(rho=(0.00429, 0.00429, -0.2703, 0.0139), with_spin=True, with_hf=True,
                 with_efg=True):
    t = HEADER
    if with_spin:
        t += """
 *******************************************************************************
 SPIN DENSITY AT THE NUCLEAR POSITIONS
 *******************************************************************************
 POINT  ATOM     X(AU)     Y(AU)     Z(AU)       BOHR**(-3)
"""
        for n, (sym, r) in enumerate(zip(("LI", "LI", "MN", "O"), rho), 1):
            t += f"   {n}    {n} {sym:2s}    2.3034    1.4005   -0.0565   {r:12.6f}\n"
        t += """
  CONVERSION FACTORS
     AN (MILLI T)        14.277235 * GN * P(SPIN) (BOHR**(-3))
     AN (MHZ)            14.012470 * AN (MILLI T)
     AN (CM**(-1))   0.1667820E-04 * AN (MHZ)
"""
    if with_hf:
        t += """
 *******************************************************************************
 HYPERFINE ELECTRON-NUCLEAR SPIN INTERACTION TENSOR
 ANISOTROPIC COMPONENTS - THE TENSOR IS TRACELESS
 *******************************************************************************
"""
        t += _hf_block(1) + _hf_block(2)
    if with_efg:
        t += "\n TRACELESS ELECTRIC FIELD GRADIENT TENSOR\n\n"
        for n in (1, 2):
            t += (f"\n POINT   {n} POSITION (ANGSTROM)   1.2189   0.7411  -0.0299\n\n"
                  " TENSOR IN ORIGINAL CARTESIAN AXES\n XX {XX:.6E} YY {YY:.6E} ZZ {ZZ:.6E} "
                  "XY {XY:.6E} XZ {XZ:.6E} YZ {YZ:.6E}\n\n".format(**EFG_AU))
        t += """
 CONVERSION FACTORS (CODATA RECOMMENDED VALUES 1998)
 THE PRODUCT OF THE EFG COMPONENT (AU) AND THE NUCLEAR
 QUADRUPOLE MOMENT COMPONENT (BARN) MUST BE MULTIPLIED BY 2.3496 E+2
"""
    return t


@pytest.fixture
def crystal_file(tmp_path):
    path = tmp_path / "properties.outp"          # an unusual extension: found by content
    path.write_text(crystal_text())
    return str(path)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def test_read_crystal_blocks_and_detection_by_content(crystal_file):
    calc = D.read_calculation(crystal_file)
    assert calc["code"] == "CRYSTAL" and calc["paramagnetic"]
    assert [a["label"] for a in calc["atoms"]] == ["Li1", "Li2", "Mn3", "O4"]
    assert calc["atoms"][2]["species"] == "Mn" and calc["atoms"][0]["species"] == "Li"
    assert calc["atoms"][2]["rho"] == pytest.approx(-0.2703)
    li = calc["atoms"][0]
    assert sorted(li["hf"]) == [6, 7] and li["hf"][7]["g_N"] == pytest.approx(2.170977)
    # the tensor read is the printed Cartesian one, symmetric
    t = li["hf"][7]["tensor"]
    assert t[0, 2] == t[2, 0] == pytest.approx(-3.65703) and np.trace(t) == pytest.approx(0, abs=1e-5)
    assert calc["energy_Eh"] == pytest.approx(-11135.840590842)
    assert calc["spin_total"] == pytest.approx(3.0)
    assert calc["lattice"].shape == (3, 3) and calc["lattice"][2, 0] == pytest.approx(-1.62361)
    assert "20% Fock" in calc["method"]["functional"]
    assert calc["info"]["title"] == "Li2MnO3_test"
    assert li["efg"] is not None and calc["atoms"][2]["efg"] is None
    assert calc["conversion"]["mT_per_gN_rho"] == pytest.approx(14.277235)
    assert li["sigma"] is None                       # nothing until the scaling is given


def test_other_text_is_not_crystal(tmp_path):
    path = tmp_path / "x.out"
    path.write_text("nothing to see here\n" * 50)
    with pytest.raises(ValueError, match="not a calculation openNMR reads"):
        D.read_calculation(str(path))
    with pytest.raises(ValueError, match="not CRYSTAL"):
        C.read_crystal(str(path))


def test_spin_density_table_file_alone_and_blocks_from_other_runs(tmp_path):
    no_spin = tmp_path / "scf.out"
    no_spin.write_text(crystal_text(with_spin=False))
    table = tmp_path / "spin_dens_table"
    table.write_text(
        "SPIN DENSITY AT THE NUCLEAR POSITIONS\n"
        " *******\n"
        " POINT  ATOM     X(AU)     Y(AU)     Z(AU)       BOHR**(-3)\n"
        "   1    1 LI    2.3034    1.4005   -0.0565         0.004290\n"
        "   3    3 MN   -0.7670    6.6971    4.4744        -0.270264\n")
    assert C.read_spin_density_table(str(table))[3]["rho"] == pytest.approx(-0.270264)
    plain = C.read_crystal(str(no_spin))
    assert plain["atoms"][0]["rho"] is None and any("No spin density" in n for n in plain["notes"])
    merged = C.read_crystal(str(no_spin), spin_density_file=str(table))
    assert merged["atoms"][0]["rho"] == pytest.approx(0.00429)
    assert merged["atoms"][2]["rho"] == pytest.approx(-0.270264)
    # the same blocks split over two outputs
    spin_only = tmp_path / "spin.out"
    spin_only.write_text(crystal_text(with_hf=False, with_efg=False))
    both = C.read_crystal(str(no_spin), str(spin_only))
    assert both["atoms"][2]["rho"] == pytest.approx(-0.2703) and both["atoms"][0]["hf"]
    with pytest.raises(ValueError, match="no spin-density rows"):
        bad = tmp_path / "bad"
        bad.write_text("nothing\n")
        C.read_spin_density_table(str(bad))


# ---------------------------------------------------------------------------
# The scaling factor
# ---------------------------------------------------------------------------

def test_curie_weiss_phi_matches_the_script():
    for args in ((11.7, 318.0, -300.0, 1.5, 3.87), (9.4, 298.0, 25.0, 2.5, 5.9),
                 (21.1, 100.0, -10.0, 0.5, 1.73)):
        B0, T, th, S, mu = args
        assert C.curie_weiss_phi(B0, T, th, S, mu) == pytest.approx(script_phi(*args), rel=1e-5)


def test_susceptibility_route_agrees_with_curie_law():
    """χ_M(Curie) = μ0 N_A μ_eff² μ_B² / (3 k_B T) gives the same φ as θ = 0."""
    B0, T, S, mu = 11.7, 300.0, 2.5, 5.92
    chi_si = C.MU_0 * C.N_A * mu ** 2 * C.MU_B ** 2 / (3 * C.K_B * T)
    phi_chi = C.phi_from_susceptibility(chi_si, B0, S, units="m3/mol")
    assert phi_chi == pytest.approx(C.curie_weiss_phi(B0, T, 0.0, S, mu), rel=1e-9)
    # cgs: the same susceptibility in cm3/mol (4π × 1e-6 m3/mol per cm3/mol); S = 5/2 high spin Fe3+
    # has chi ~ 4.4 cm3/mol K / T: 0.0148 cm3/mol at 300 K
    cgs = chi_si / (4 * math.pi * 1e-6)
    assert cgs == pytest.approx(4.38 / 300.0, rel=0.02)
    assert C.phi_from_susceptibility(cgs, B0, S, units="cm3/mol") == pytest.approx(phi_chi, rel=1e-9)


def test_magnetic_scaling_routes_and_errors():
    s = C.magnetic_scaling(phi=0.01, B0=9.4)
    assert s["route"] == "phi" and s["phi"] == 0.01 and "given" in s["text"]
    s = C.magnetic_scaling(chi=0.01, spin=1.5, B0=9.4)
    assert s["route"] == "susceptibility" and s["phi"] > 0
    s = C.magnetic_scaling(theta=-300, mu_eff=3.87, spin=1.5, T=318, B0=11.7)
    assert s["route"] == "curie_weiss" and s["phi"] == pytest.approx(script_phi(11.7, 318, -300, 1.5, 3.87), rel=1e-5)
    # the route taken is the most direct one given
    assert C.magnetic_scaling(phi=0.5, chi=0.1, spin=1.5)["route"] == "phi"
    with pytest.raises(ValueError, match="Give φ"):
        C.magnetic_scaling()
    with pytest.raises(ValueError, match="needs all of"):
        C.magnetic_scaling(theta=-30, spin=1.5)
    with pytest.raises(ValueError, match="spin is required"):
        C.magnetic_scaling(chi=0.01)
    with pytest.raises(ValueError, match="not positive"):          # T below the Weiss temperature
        C.magnetic_scaling(theta=400, mu_eff=3.87, spin=1.5, T=300)
    with pytest.raises(ValueError, match="positive"):
        C.magnetic_scaling(chi=-0.001, spin=1.5)
    with pytest.raises(ValueError, match="units"):
        C.phi_from_susceptibility(0.01, 11.7, 1.5, units="furlongs")


# ---------------------------------------------------------------------------
# Shifts
# ---------------------------------------------------------------------------

THETA, MU, SPIN, TEMP, FIELD = -300.0, 3.87, 1.5, 318.0, 11.7
PM = dict(theta=THETA, mu_eff=MU, spin=SPIN, T=TEMP, B0=FIELD)


def test_fermi_contact_shift_reproduces_the_script_within_its_constants(crystal_file):
    """The script's 'CRYSTAL factor' c_f = 1.99827885e8 is 0.12 % below the CODATA
    value, so the two agree to a couple of parts in a thousand."""
    calc = C.apply_scaling(C.read_crystal(crystal_file), C.magnetic_scaling(**PM),
                           components="contact")
    for atom in calc["atoms"]:
        expected = script_fc_ppm(atom["rho"], FIELD, TEMP, THETA, SPIN, MU)
        assert atom["delta_fc"] == pytest.approx(expected, rel=2.5e-3)
        # contact only: an isotropic tensor, σ = −δ
        assert np.allclose(atom["delta_hf"], atom["delta_fc"] * np.eye(3))
        assert np.allclose(atom["sigma"], -atom["delta_hf"])
    assert calc["atoms"][2]["delta_fc"] < 0 < calc["atoms"][0]["delta_fc"]   # sign follows ρ


def test_shift_is_independent_of_field_and_isotope(crystal_file):
    calc = C.read_crystal(crystal_file)
    a = C.apply_scaling(calc, C.magnetic_scaling(**{**PM, "B0": 4.7}))
    b = C.apply_scaling(calc, C.magnetic_scaling(**{**PM, "B0": 21.1}))
    for x, y in zip(a["atoms"], b["atoms"]):
        assert x["delta_fc"] == pytest.approx(y["delta_fc"], rel=1e-9)
        if x["delta_dip"] is not None:
            assert np.allclose(x["delta_dip"], y["delta_dip"], rtol=1e-9)
    # 6Li and 7Li hyperfine tensors differ by exactly g_N: one dipolar shift
    li = calc["atoms"][0]
    one = C.apply_scaling(calc, C.magnetic_scaling(**PM))["atoms"][0]["delta_dip"]
    flipped = {7: li["hf"][7]}
    calc7 = {**calc, "atoms": [{**li, "hf": flipped}] + calc["atoms"][1:]}
    seven = C.apply_scaling(calc7, C.magnetic_scaling(**PM))["atoms"][0]["delta_dip"]
    assert np.allclose(one, seven, rtol=1e-5)


def test_dipolar_shift_tensor_is_phi_over_two_times_A_over_nu0(crystal_file):
    calc = C.read_crystal(crystal_file)
    sc = C.magnetic_scaling(**PM)
    out = C.apply_scaling(calc, sc, components="dipolar")
    li = out["atoms"][0]
    nu0 = 2.170977 * 5.0507837461e-27 * FIELD / 6.62607015e-34          # 7Li at 11.7 T, Hz
    A = li["hf"][7]["tensor"]
    expected = 1e6 * 0.5 * sc["phi"] * ((A - np.trace(A) / 3 * np.eye(3)) * 1e6) / nu0
    assert li["delta_dip"] == pytest.approx(expected, rel=1e-5)
    assert np.trace(li["delta_dip"]) == pytest.approx(0, abs=1e-6)       # traceless (the rounding of the printed trace removed)
    # the user's script (HF in MHz, 23Na at 11.7 T) is the same expression
    # (ppm per MHz of hyperfine coupling, for 23Na at 11.7 T: the isotope drops out of A / nu0)
    script = 1.0e6 * 2.0 * 2.38526956e+07 * script_phi(11.7, 318, THETA, SPIN, MU) \
        / (3.4066697 * 28.02495 * 131.85e+06)
    assert script == pytest.approx(1e6 * 0.5 * script_phi(11.7, 318, THETA, SPIN, MU) * 1e6
                                   / 131.85e6, rel=1e-3)
    assert li["sigma"] == pytest.approx(-li["delta_dip"])
    both = C.apply_scaling(calc, sc)["atoms"][0]
    assert np.allclose(both["delta_hf"], both["delta_fc"] * np.eye(3) + both["delta_dip"])


def test_components_and_missing_data(tmp_path, crystal_file):
    calc = C.read_crystal(crystal_file)
    with pytest.raises(ValueError, match="components"):
        C.apply_scaling(calc, phi=0.01, components="spin")
    with pytest.raises(ValueError, match="not both"):
        C.apply_scaling(calc, C.magnetic_scaling(phi=0.01), phi=0.02)
    path = tmp_path / "nospin.out"
    path.write_text(crystal_text(with_spin=False))
    nospin = C.apply_scaling(C.read_crystal(str(path)), phi=0.01)
    assert nospin["atoms"][0]["delta_fc"] is None and nospin["atoms"][0]["delta_dip"] is not None
    with pytest.raises(ValueError, match="is for CRYSTAL"):
        C.apply_scaling({"code": "VASP", "atoms": []}, phi=0.01)


# ---------------------------------------------------------------------------
# Through dft_sites: C_Q, eta_Q and the shift tensor
# ---------------------------------------------------------------------------

def test_dft_sites_gives_shift_cq_and_eta_from_crystal(crystal_file):
    model = D.dft_sites(crystal_file, element="Li", nucleus="7Li", paramagnetic=dict(PM))
    assert model["nucleus"] == "7Li" and len(model["sites"]) == 1       # two equivalent Li merge
    rec = model["atoms"][0]
    sc = C.magnetic_scaling(**PM)
    calc = C.apply_scaling(C.read_crystal(crystal_file), sc)
    assert rec["iso"] == pytest.approx(np.trace(calc["atoms"][0]["delta_hf"]) / 3, rel=1e-9)
    # C_Q = V_zz(au) × Q(barn) × 234.96 MHz, CRYSTAL's own printed factor
    V = np.array([[4.553322e-03, 2.090836e-03, 5.125410e-03],
                  [2.090836e-03, -2.942458e-03, -7.045908e-04],
                  [5.125410e-03, -7.045908e-04, -1.610864e-03]])
    w = np.linalg.eigvalsh(V)
    vzz = w[np.argmax(np.abs(w))]
    Q = D.quadrupole_moment_barn("7Li")
    assert rec["C_Q"] == pytest.approx(vzz * Q * 234.96, rel=5e-4)
    vals = sorted(w, key=abs)
    assert rec["eta_Q"] == pytest.approx(abs((vals[0] - vals[1]) / vals[2]), rel=1e-6)
    assert rec["uses_quadrupolar"] and model["sites"][0].quadrupolar is not None \
        if hasattr(model["sites"][0], "quadrupolar") else True
    assert any("Paramagnetic shifts" in n for n in model["notes"])
    # the orbital shift is added to every site
    shifted = D.dft_sites(crystal_file, element="Li", nucleus="7Li", paramagnetic=dict(PM),
                          orbital_shift=12.5)
    assert shifted["atoms"][0]["iso"] == pytest.approx(rec["iso"] + 12.5)
    # a spin-1/2 nucleus (6Li is spin 1) vs 1/2: Mn has I = 5/2 and gets a C_Q only with an EFG
    mn = D.dft_sites(crystal_file, element="Mn", paramagnetic=dict(PM))
    assert mn["atoms"][0]["C_Q"] is None and mn["atoms"][0]["iso"] < 0


def test_dft_sites_needs_the_scaling_for_crystal_only(crystal_file, tmp_path):
    with pytest.raises(ValueError, match="paramagnetic="):
        D.dft_sites(crystal_file, element="Li")
    with pytest.raises(ValueError, match="CRYSTAL calculations only"):
        D.dft_sites({"code": "CASTEP", "path": "x", "atoms": [], "notes": []}, element="Li",
                    paramagnetic=dict(PM), sigma_ref=0.0)
    solution = D.dft_sites(crystal_file, element="Li", paramagnetic=dict(PM), state="solution")
    assert solution["atoms"][0]["C_Q"] is None and solution["atoms"][0]["aniso"] is None
    contact = D.dft_sites(crystal_file, element="Li", paramagnetic={**PM, "components": "contact"})
    assert contact["atoms"][0]["span"] == pytest.approx(0.0, abs=1e-9)    # contact alone: isotropic
    full = D.dft_sites(crystal_file, element="Li", paramagnetic=dict(PM))
    assert full["atoms"][0]["span"] > 100.0                               # dipolar adds a pattern
