"""Paramagnetic shifts from g, zero-field splitting and hyperfine tensors
(user request, 2026-11-06): the spin susceptibility Z, σ = −(μ_B/g_Nμ_N)
g·Z·A, the ORCA readers, the magnetic models, dft_sites with an ORCA EPR
output, and the temperature dependence and its fits.

Checked against ORCA's own orca_pnmr (the numbers below are copied from the
user's outputs: a Mn(II) Na2MnFe(CN)6 cluster, S = 5/2, no ZFS; and a
Pr(III) cluster, pseudo-spin S = 1 with a large D), and against independent
calculations (a numerical derivative for Z)."""

import math
import os

import numpy as np
import pytest

from opennmr.notebook import dft_nmr as D
from opennmr.notebook import paramagnetic_nmr as pm
from opennmr.notebook.crystal_nmr import K_B, MU_B

DATA = os.path.join(os.path.dirname(__file__), "..", "test_data", "dft", "paramagnetic")
real = pytest.mark.skipif(not os.path.isdir(DATA), reason="test_data/dft/paramagnetic not present")

# ---- the user's Mn cluster (input.pnmr.inp / shifts.txt, ORCA 5.0.3, 298 K) -----------
MN_G = np.array([[1.99961, 0.0001537, 0.0002208], [-1.98e-05, 1.9987445, 0.0003745],
                 [0.0003359, 5.41e-05, 1.9990185]])
MN_HFC = {  # ORCA index: (element, mass, I, P, A (MHz), σ_iso, σ tensor) -- orca_pnmr's
    0: ("H", 1, 0.5, 533.551395,
        [[0.350229, 0.84504, 0.430499], [0.84504, 0.090726, 0.38351],
         [0.430499, 0.38351, -0.46339]], 2.272,
        [[-108.171, -260.93, -132.914], [-260.825, -28.02, -118.333],
         [-132.907, -118.422, 143.008]]),
    13: ("C", 13, 0.5, 134.190303,
         [[5.868005, 0.110208, -0.296193], [0.110208, 6.177998, -0.36589],
          [-0.296193, -0.36589, 6.522662]], -7596.685,
         [[-7203.85, -135.831, 362.773], [-135.099, -7581.075, 447.489],
          [362.301, 448.825, -8005.13]]),
    35: ("Na", 23, 1.5, 141.213434,
         [[2.16259, 0.474786, -0.217468], [0.474786, 2.643565, -0.241189],
          [-0.217468, -0.241189, 1.872207]], -2596.295,
         [[-2522.887, -554.09, 253.478], [-553.572, -3082.584, 280.837],
          [253.184, 281.111, -2183.416]]),
}

# ---- the user's Pr cluster (PrFeCN6_pnmr.txt): pseudo-spin S = 1, with D ---------------
PR_G = np.array([[-0.85348, -0.041109, -0.010974], [0.008764, -2.284236, -0.041941],
                 [0.012382, -0.284094, 0.692334]])
PR_D = np.array([[46.632822, -2.963547, -0.854918], [-2.963547, -43.717324, 17.777311],
                 [-0.854918, 17.777311, 64.95082]])
PR_HFC = {
    1: ("N", 38.567697, [[-1.268141, -7.1e-05, 0.000438], [-7.1e-05, -0.960627, -0.475449],
                         [0.000438, -0.475432, -0.616721]],
        [[-491.692, -26.931, -14.957], [-5.025, -1210.941, -588.368],
         [6.618, -20.741, 106.289]]),
    2: ("N", 38.567697, [[-1.587537, 6.3e-05, 6.6e-05], [6.3e-05, -1.313771, 0.437314],
                         [6.6e-05, 0.437284, -0.883925]],
        [[-615.529, -31.603, 7.046], [-6.083, -1687.377, 582.91], [8.459, -360.697, 345.705]]),
    7: ("C", 134.190303, [[-0.106301, 0.363558, -0.381552], [0.363529, -0.199078, -0.38616],
                          [-0.381536, -0.386177, -0.0025]],
        [[-9.686, 38.519, -45.37], [136.229, -69.343, -141.891], [51.732, 23.472, -18.378]]),
}


def _m(rows):
    return "\n".join("    " + "  ".join(f"{v: .7f}" for v in r) for r in rows)


def write_orca_epr(path, atoms, hfc, g=None, mult=None, Dcm=None, shielding=None):
    """A minimal ORCA EPR/NMR output in ORCA 5's layout. ``atoms``: [(sym,
    x, y, z)]; ``hfc``: {index: (element, mass, I, P, A)}; ``shielding``:
    {index: σ tensor} for an NMR block in the same file."""
    out = ["", "                                 *****************",
           "                                 * O   R   C   A *",
           "                                 *****************", "",
           "                         Program Version 5.0.3 -  RELEASE  -", "",
           "|  1> ! PBE0 def2-TZVP", ""]
    if mult is not None:
        out.append(f" Multiplicity           Mult            ....    {mult}")
    out += ["---------------------------------", "CARTESIAN COORDINATES (ANGSTROEM)",
            "---------------------------------"]
    out += [f"  {s:<2}  {x: .6f}  {y: .6f}  {z: .6f}" for s, x, y, z in atoms]
    out += ["", ""]
    if g is not None:
        out += ["-------------------", "ELECTRONIC G-MATRIX", "-------------------", "",
                " The g-matrix: ", _m(g), "", " gel   2.0023193 2.0023193 2.0023193"]
    if Dcm is not None:
        out += ["", "--------------------------", "ZERO-FIELD-SPLITTING TENSOR",
                "--------------------------", "", "raw-matrix : ", _m(Dcm), ""]
    out += ["", "-----------------------------------------",
            "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE",
            "-----------------------------------------", ""]
    for idx, (el, mass, I, P, A) in hfc.items():
        lab = f"{idx}{el}"
        out += [" -----------------------------------------------------------",
                f" Nucleus {lab:>4}: A:ISTP= {mass:4d} I= {I:4.1f} P={P:9.4f} MHz/au**3",
                " -----------------------------------------------------------",
                " Raw HFC matrix (all values in MHz): ", " -----------------------------------",
                _m(A), "", f" A(FC)  {np.trace(A) / 3:.4f}", ""]
    for idx, sig in (shielding or {}).items():
        el = atoms[idx][0]
        out += [" --------------", f" Nucleus  {idx}{el} :", " --------------", "",
                "Total shielding tensor (ppm): ", _m(sig), ""]
    out.append("                             ****ORCA TERMINATED NORMALLY****")
    path.write_text("\n".join(out) + "\n")
    return str(path)


def _mn_file(tmp_path, shielding=None):
    atoms = [("H", 0.0, 0.0, 0.0)] + [("X", 0, 0, 0)] * 12 + [("C", 1.0, 0.0, 0.0)] \
        + [("X", 0, 0, 0)] * 21 + [("Na", 2.0, 0.0, 0.0)]
    # Fill the placeholder atoms with O (no hyperfine for them).
    atoms = [("O", 5.0 + k, 0.0, 0.0) if a[0] == "X" else a for k, a in enumerate(atoms)]
    hfc = {k: (v[0], v[1], v[2], v[3], v[4]) for k, v in MN_HFC.items()}
    return write_orca_epr(tmp_path / "mn.out", atoms, hfc, g=MN_G, mult=6, shielding=shielding)


# ---- the spin -----------------------------------------------------------------------

@pytest.mark.parametrize("S", [0.5, 1.0, 2.5, 3.5])
def test_spin_matrices_satisfy_the_commutation_relations(S):
    sx, sy, sz = pm.spin_matrices(S)
    assert np.allclose(sx @ sy - sy @ sx, 1j * sz)
    assert np.allclose(sx @ sx + sy @ sy + sz @ sz, S * (S + 1) * np.eye(int(2 * S + 1)))


def test_spin_matrices_refuse_a_bad_spin():
    with pytest.raises(ValueError):
        pm.spin_matrices(0.7)


def test_without_zfs_z_is_the_curie_law():
    Z = pm.spin_susceptibility(2.5, 298.0)
    assert np.allclose(Z, 2.5 * 3.5 / (3 * K_B * 298.0) * np.eye(3))
    # A zero D tensor is "no ZFS" too.
    assert np.allclose(pm.spin_susceptibility(2.5, 298.0, np.zeros((3, 3))), Z)


def _numerical_Z(S, T, Dcm, h=1e-4):
    """Z_kl = −∂⟨S_l⟩/∂b_k under H = S·D·S + b·S (b in J), by central
    differences -- independent of the sum over states in the code."""
    from opennmr.notebook.paramagnetic_nmr import CM1_TO_J
    ops = pm.spin_matrices(S)
    Dj = np.asarray(Dcm) * CM1_TO_J
    H0 = sum(Dj[k, l] * ops[k] @ ops[l] for k in range(3) for l in range(3))
    kT = K_B * T
    step = h * kT

    def mean_S(b):
        H = H0 + sum(b[k] * ops[k] for k in range(3))
        E, U = np.linalg.eigh(H)
        p = np.exp(-(E - E.min()) / kT)
        p /= p.sum()
        return np.array([np.real(np.sum(p * np.diag(U.conj().T @ o @ U))) for o in ops])
    Z = np.zeros((3, 3))
    for k in range(3):
        b = np.zeros(3)
        b[k] = step
        Z[k] = -(mean_S(b) - mean_S(-b)) / (2 * step)
    return Z


@pytest.mark.parametrize("S,T", [(1.0, 298.0), (1.5, 50.0), (2.5, 10.0)])
def test_z_with_zfs_equals_the_numerical_derivative_of_the_mean_spin(S, T):
    Dcm = pm.zfs_tensor(8.0, 1.5, axes=_rot(20, 50, 70)) + 2.0 * np.eye(3)  # a trace does nothing
    Z = pm.spin_susceptibility(S, T, Dcm)
    Zn = _numerical_Z(S, T, Dcm)
    assert np.allclose(Z, Zn, rtol=2e-5, atol=1e-6 * np.abs(Z).max())


def test_z_tends_to_the_curie_law_when_kt_dwarfs_d():
    Z = pm.spin_susceptibility(1.0, 1e5, pm.zfs_tensor(1.0, 0.2))
    assert np.allclose(Z, 2.0 / (3 * K_B * 1e5) * np.eye(3), rtol=1e-5)


def test_d_and_e_round_trip_with_the_usual_axes():
    R = _rot(10, 35, -40)
    de = pm.D_E_from_tensor(pm.zfs_tensor(-5.0, 1.2, axes=R))
    # The convention 0 ≤ E/D ≤ 1/3 gives E the sign of D.
    assert de["D"] == pytest.approx(-5.0) and de["E"] == pytest.approx(-1.2)
    assert 0 <= de["E_over_D"] <= 1 / 3 + 1e-12


def _rot(a, b, c):
    from opennmr.notebook.pseudocontact import rotation_zyz
    return rotation_zyz(a, b, c)


# ---- the shielding: ORCA's own numbers ------------------------------------------------

def test_mn_cluster_matches_orca_pnmr_element_by_element():
    Z = pm.spin_susceptibility(2.5, 298.0)
    for idx, (el, _m_, _I, P, A, iso, tensor) in MN_HFC.items():
        sigma = pm.paramagnetic_shielding(MN_G, Z, A, pm.g_nuclear_from_pfac(P))
        # orca_pnmr prints 3 decimals; ≤ 0.02 ppm on shieldings of up to 10⁴ ppm
        assert np.max(np.abs(sigma - np.array(tensor))) < 0.02, (idx, el)
        assert np.trace(sigma) / 3 == pytest.approx(iso, abs=0.02)


def test_pr_cluster_with_zero_field_splitting_matches_orca_pnmr():
    """With D the van Vleck (sum-over-states) Z is needed: ORCA uses it."""
    Z = pm.spin_susceptibility(1.0, 298.0, PR_D)
    for idx, (el, P, A, tensor) in PR_HFC.items():
        sigma = pm.paramagnetic_shielding(PR_G, Z, A, pm.g_nuclear_from_pfac(P))
        assert np.max(np.abs(sigma - np.array(tensor))) < 0.01, (idx, el)


def test_the_high_temperature_ss_form_is_not_what_orca_uses():
    """⟨S_k S_l⟩/kT (no off-diagonal terms) misses the Pr shifts by ~1 %: a
    regression guard on the choice of formula."""
    ops = pm.spin_matrices(1.0)
    Dj = PR_D * pm.CM1_TO_J
    H = sum(Dj[k, l] * ops[k] @ ops[l] for k in range(3) for l in range(3))
    E, U = np.linalg.eigh(H)
    p = np.exp(-(E - E.min()) / (K_B * 298.0))
    p /= p.sum()
    SS = np.array([[np.real(np.trace(np.diag(p) @ U.conj().T @ ops[k] @ ops[l] @ U))
                    for l in range(3)] for k in range(3)]) / (K_B * 298.0)
    el, P, A, tensor = PR_HFC[2]
    wrong = pm.paramagnetic_shielding(PR_G, SS, A, pm.g_nuclear_from_pfac(P))
    assert np.max(np.abs(wrong - np.array(tensor))) > 5.0


def test_pfac_gives_the_nuclear_g_factor():
    assert pm.g_nuclear_from_pfac(533.551395) == pytest.approx(5.5857, abs=2e-4)
    assert pm.g_nuclear_from_pfac(134.190303) == pytest.approx(1.4048, abs=2e-4)


def test_contact_and_pseudocontact_add_up_and_contact_follows_a_iso():
    Z = pm.spin_susceptibility(1.0, 200.0, PR_D)
    _el, P, A, _t = PR_HFC[1]
    parts = pm.shielding_contributions(PR_G, Z, A, pm.g_nuclear_from_pfac(P))
    assert parts["contact"] + parts["pseudocontact"] == pytest.approx(parts["total"])
    A2 = np.array(A) + 3.0 * np.eye(3)
    parts2 = pm.shielding_contributions(PR_G, Z, A2, pm.g_nuclear_from_pfac(P))
    assert parts2["pseudocontact"] == pytest.approx(parts["pseudocontact"])
    assert parts2["contact"] != pytest.approx(parts["contact"])


def test_isotropic_g_and_no_zfs_give_no_isotropic_pseudocontact_shift():
    Z = pm.spin_susceptibility(2.5, 298.0)
    A = np.array([[1.0, 0.3, 0.0], [0.3, -0.4, 0.1], [0.0, 0.1, -0.6]]) + 2.0 * np.eye(3)
    parts = pm.shielding_contributions(pm.G_E * np.eye(3), Z, A, 1.4048)
    assert parts["pseudocontact"] == pytest.approx(0.0, abs=1e-9)


# ---- magnetic models --------------------------------------------------------------------

def test_models_that_describe_the_same_magnet_give_the_same_z():
    S, T, g = 2.5, 298.0, 2.0
    base = pm.spin_response(S, T, "curie")["Z"]
    assert np.allclose(pm.spin_response(S, T, "curie_weiss", theta=0.0)["Z"], base)
    mu = g * math.sqrt(S * (S + 1))
    assert np.allclose(pm.spin_response(S, T, "mu_eff", g=g * np.eye(3), mu_eff=mu)["Z"], base)
    from opennmr.notebook.crystal_nmr import MU_0, N_A
    chi_si = MU_0 * N_A * g ** 2 * MU_B ** 2 * S * (S + 1) / (3 * K_B * T)
    z = pm.spin_response(S, T, "susceptibility", g=g * np.eye(3), chi=chi_si / (4e-6 * math.pi))
    assert np.allclose(z["Z"], base, rtol=1e-12)
    B0 = 9.4
    phi = g * MU_B * B0 * S * (S + 1) / (3 * K_B * T) / S     # ⟨S_z⟩/S, Curie
    assert np.allclose(pm.spin_response(S, T, "phi", g=g * np.eye(3), phi=phi, B0=B0)["Z"], base)


def test_a_weiss_constant_scales_z_by_t_over_t_minus_theta():
    z = pm.spin_response(1.5, 300.0, "curie_weiss", theta=-100.0)
    assert z["curie_ratio"] == pytest.approx(300.0 / 400.0)


def test_models_refuse_impossible_inputs():
    with pytest.raises(ValueError, match="T − θ"):
        pm.spin_response(1.0, 50.0, "curie_weiss", theta=60.0)
    with pytest.raises(ValueError):
        pm.spin_response(1.0, 300.0, "susceptibility", chi=-1.0)
    with pytest.raises(ValueError):
        pm.spin_response(1.0, 300.0, "phi", phi=0.1)
    with pytest.raises(ValueError):
        pm.spin_response(1.0, 300.0, "nonsense")


def test_zfs_with_a_measured_model_is_noted_not_used():
    r = pm.spin_response(1.0, 300.0, "curie_weiss", D_cm=PR_D)
    assert not r["has_zfs"] and r["notes"]


# ---- reading ORCA and dft_sites ---------------------------------------------------------

def test_read_orca_epr_output(tmp_path):
    calc = D.read_calculation(_mn_file(tmp_path))
    assert calc["epr"]["S"] == 2.5 and np.allclose(calc["epr"]["g"], MN_G)
    na = next(a for a in calc["atoms"] if a["species"] == "Na")
    assert na["hfc"]["mass"] == 23 and na["orca_label"] == "35Na"
    assert np.allclose(na["hfc"]["A_MHz"], MN_HFC[35][4])
    assert pm.is_paramagnetic(calc)


def test_dft_sites_with_an_orca_epr_output_gives_orca_shifts(tmp_path):
    calc = D.read_calculation(_mn_file(tmp_path))
    model = D.dft_sites(calc, element="Na", paramagnetic={"model": "curie", "T": 298.0},
                        orbital_shift=5.0, merge=False)
    rec = model["atoms"][0]
    assert rec["iso"] == pytest.approx(-MN_HFC[35][5] + 5.0, abs=0.02)
    rows = pm.contribution_rows(model)
    assert rows[0]["contact"] + rows[0]["pseudocontact"] == pytest.approx(rows[0]["total"])
    assert model["sites"][0].delta_iso == pytest.approx(rec["iso"])


def test_dft_sites_needs_a_model_for_an_orca_epr_output(tmp_path):
    calc = D.read_calculation(_mn_file(tmp_path))
    with pytest.raises(ValueError, match="hyperfine"):
        D.dft_sites(calc, element="Na")


def test_orbital_shielding_in_the_same_file_goes_through_the_reference(tmp_path):
    sig = np.diag([550.0, 560.0, 570.0])
    calc = D.read_calculation(_mn_file(tmp_path, shielding={35: sig}))
    model = D.dft_sites(calc, element="Na", paramagnetic={"model": "curie", "T": 298.0},
                        sigma_ref=500.0, delta_ref=7.0, merge=False)
    # δ = δ_ref + σ_ref − σ_orb − σ_para
    assert model["atoms"][0]["iso"] == pytest.approx(7.0 + 500.0 - 560.0 - MN_HFC[35][5],
                                                     abs=0.02)


def test_components_pick_contact_or_dipolar(tmp_path):
    calc = D.read_calculation(_mn_file(tmp_path))
    full = pm.apply_paramagnetic(calc, T=298.0)
    con = pm.apply_paramagnetic(calc, T=298.0, components="contact")
    dip = pm.apply_paramagnetic(calc, T=298.0, components="dipolar")
    for a, b, c in zip(full["atoms"], con["atoms"], dip["atoms"]):
        if a.get("para"):
            assert np.allclose(a["sigma"], b["sigma"] + c["sigma"])


def test_orca_pnmr_input_and_output_readers(tmp_path):
    inp = tmp_path / "x.pnmr.inp"
    lines = ["3 #Spin multiplicity (2S+1)", "298 #temp0", "300 #temp1", "1 #temp_step",
             "1 #have_g?"] + [" ".join(map(str, r)) for r in PR_G] + ["1 #have_D?"] + \
        [" ".join(map(str, r)) for r in PR_D] + ["1 #Number of A tensors", "2N",
                                                  "38.567697 #PFAC"] + \
        [" ".join(map(str, r)) for r in PR_HFC[2][2]]
    inp.write_text("\n".join(lines) + "\n")
    r = pm.read_orca_pnmr_input(inp)
    assert r["S"] == 1.0 and r["temperatures"] == [298.0, 299.0, 300.0]
    assert np.allclose(r["D_cm"], PR_D) and 2 in r["hfc"]


@real
def test_real_mn_output_against_orca_pnmr():
    calc = D.read_calculation(os.path.join(DATA, "orca_mn_cluster_epr.out"))
    applied = pm.apply_paramagnetic(calc, model="curie", T=298.0)
    res = pm.compare_with_orca(applied, os.path.join(DATA, "orca_mn_cluster_pnmr.txt"))
    assert res["n"] == 26
    # the .out prints A to 4 decimals: that, not the method, limits this
    assert res["max_iso_ppm"] < 0.05 and res["max_element_ppm"] < 0.08


@real
@pytest.mark.parametrize("name", ["orca_PrFeCN6_pnmr.txt", "orca_CeFeCN6_pnmr.txt",
                                  "orca_YbFeCN6_pnmr.txt", "orca_mn_cluster_pnmr.txt"])
def test_real_orca_pnmr_outputs_are_reproduced(name):
    path = os.path.join(DATA, name)
    echo = pm.read_orca_pnmr_echo(path)
    ref = pm.read_orca_pnmr_output(path)
    T = next(iter(ref))
    Z = pm.spin_susceptibility(echo["S"], T, echo["D_cm"])
    for idx, h in echo["hfc"].items():
        sigma = pm.paramagnetic_shielding(echo["g"], Z, h["A_MHz"], h["g_N"])
        assert np.max(np.abs(sigma - ref[T][f"{idx}{h['element']}"]["tensor"])) < 0.02


# ---- temperature ------------------------------------------------------------------------

def test_curie_shift_times_t_is_constant(tmp_path):
    calc = D.read_calculation(_mn_file(tmp_path))
    T = np.array([200.0, 250.0, 300.0, 400.0])
    out = pm.shift_vs_temperature(calc, T, element="Na")
    dT = out["sites"]["Na1"]["total"] * T
    assert np.allclose(dT, dT[0])


def test_zfs_adds_a_second_order_temperature_term():
    # a pseudocontact shift from ZFS alone does not scale as 1/T
    Z1 = pm.spin_susceptibility(1.0, 100.0, PR_D)
    Z2 = pm.spin_susceptibility(1.0, 400.0, PR_D)
    an1 = Z1 - np.trace(Z1) / 3 * np.eye(3)
    an2 = Z2 - np.trace(Z2) / 3 * np.eye(3)
    assert not np.allclose(an1 * 100.0, an2 * 400.0, rtol=0.05)


def test_temperature_sweep_refuses_a_single_temperature_model(tmp_path):
    calc = D.read_calculation(_mn_file(tmp_path))
    with pytest.raises(ValueError, match="one temperature"):
        pm.shift_vs_temperature(calc, [200, 300], model="susceptibility", chi=0.01)


@pytest.mark.parametrize("model,truth", [("curie", {"delta_dia": 12.0, "a": 3.2e5}),
                                         ("curie_bleaney", {"delta_dia": 12.0, "a": 3.2e5,
                                                            "b": -4.0e7})])
def test_linear_temperature_fits_recover_their_parameters(model, truth):
    T = np.linspace(220, 360, 8)
    d = truth["delta_dia"] + truth["a"] / T + truth.get("b", 0.0) / T ** 2
    fit = pm.fit_temperature(T, d, model)
    for k, v in truth.items():
        assert fit["params"][k] == pytest.approx(v, rel=1e-8)
    assert fit["r2"] == pytest.approx(1.0)


def test_curie_weiss_fit_recovers_theta():
    T = np.linspace(200, 400, 9)
    d = 5.0 + 4.0e5 / (T + 60.0)
    fit = pm.fit_temperature(T, d, "curie_weiss")
    assert fit["params"]["theta"] == pytest.approx(-60.0, abs=1e-3)
    assert fit["params"]["C"] == pytest.approx(4.0e5, rel=1e-5)


def test_fixed_diamagnetic_shift():
    T = np.linspace(250, 350, 5)
    fit = pm.fit_temperature(T, 3.0 + 1e5 / T, "curie", delta_dia=3.0)
    assert fit["params"]["a"] == pytest.approx(1e5) and fit["errors"]["delta_dia"] == 0.0


def test_temperature_fits_refuse_bad_input():
    with pytest.raises(ValueError):
        pm.fit_temperature([300, 310], [1.0], "curie")
    with pytest.raises(ValueError):
        pm.fit_temperature([300, -10], [1.0, 2.0], "curie")
    with pytest.raises(ValueError):
        pm.fit_temperature([300, 310, 320], [1, 2, 3], "nope")


def test_plots_have_italic_symbols(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    calc = D.read_calculation(_mn_file(tmp_path))
    curves = pm.shift_vs_temperature(calc, [250, 300, 350], element="Na")
    T = np.array([250.0, 300.0, 350.0])
    fit = pm.fit_temperature(T, 1e5 / T, "curie")
    fig = pm.plot_shift_temperature(curves, {"x": (T, 1e5 / T)}, {"x": fit}, x_axis="1/T",
                                    components=True)
    ax = fig.axes[0]
    assert "$T$" in ax.get_xlabel() or "mathit{T}" in ax.get_xlabel()
    assert "delta" in ax.get_ylabel()
    rows = pm.contribution_rows(pm.apply_paramagnetic(calc, T=298.0))
    fig2 = pm.plot_contributions(rows)
    assert len(fig2.axes[0].patches) == 2 * len(rows)
    with pytest.raises(ValueError):
        pm.plot_contributions([])


def test_paramagnetic_python_code_runs(tmp_path):
    path = _mn_file(tmp_path)
    code = pm.paramagnetic_to_python(path, element="Na", paramagnetic={"model": "curie",
                                                                       "T": 298.0},
                                     orbital_shift=1.0, merge=False)
    ns = {}
    exec(code, ns)
    assert ns["model"]["atoms"][0]["iso"] == pytest.approx(-MN_HFC[35][5] + 1.0, abs=0.02)
