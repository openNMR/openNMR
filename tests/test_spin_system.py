"""Strong coupling (notebook/spin_system.py), isotope-resolved coupling
groups (simulations.jcoupling_group_pattern(all_isotopes=True)), and the
nuclear-spin column of the isotope tables.

The spin-system engine is checked three independent ways: the textbook
AB quartet in closed form, the weak-coupling limit against the
first-order multiplet code, and a brute-force reference written here that
builds the full Hilbert-space Hamiltonian from Kronecker products (no
blocks, no basis bookkeeping) for systems mixing spin-1/2, spin-1,
heteronuclear and magnetically equivalent spins.
"""
import itertools

import numpy as np
import pytest

from opennmr.notebook import simulations as S
from opennmr.notebook import spin_system as SS
from opennmr.notebook.utils import find_isotope_info

B0 = 9.395
L_1H = abs(find_isotope_info("1H")["gamma"]) * B0


# --- the brute-force reference ------------------------------------------------

def _brute(spins, couplings, observe="1H"):
    ex, J = SS.expand_spin_system(spins, couplings)
    N = len(ex)
    ops = [S.spin_operators(s["I"]) for s in ex]
    dims = [o[2].shape[0] for o in ops]

    def embed(i, op):
        m = np.array([[1.0 + 0j]])
        for k in range(N):
            m = np.kron(m, op if k == i else np.eye(dims[k]))
        return m
    X = [embed(i, o[0]) for i, o in enumerate(ops)]
    Y = [embed(i, o[1]) for i, o in enumerate(ops)]
    Z = [embed(i, o[2]) for i, o in enumerate(ops)]
    nu = [s["shift_ppm"] * abs(s["gamma"]) * B0 * 1e-6 for s in ex]
    H = sum(nu[i] * Z[i] for i in range(N))
    for i, j in itertools.combinations(range(N), 2):
        if ex[i]["nuclide"] == ex[j]["nuclide"]:
            H = H + J[i, j] * (X[i] @ X[j] + Y[i] @ Y[j] + Z[i] @ Z[j])
        else:
            H = H + J[i, j] * Z[i] @ Z[j]
    E, V = np.linalg.eigh(H)
    Fm = sum(X[i] - 1j * Y[i] for i in range(N) if ex[i]["nuclide"] == observe)
    T = V.conj().T @ Fm @ V
    f = (E[None, :] - E[:, None]).ravel()
    w = (abs(T) ** 2).ravel()
    I = [s["I"] for s in ex if s["nuclide"] == observe][0]
    w = w / ((2 / 3) * I * (I + 1) * np.prod(dims))
    keep = w > 1e-9 * w.max()
    larmor = abs([s for s in ex if s["nuclide"] == observe][0]["gamma"]) * B0
    return SS._merge_lines(f[keep] / larmor * 1e6, w[keep])


CASES = {
    "ABX": ([{"label": "A", "nuclide": "1H", "shift_ppm": 7.30},
             {"label": "B", "nuclide": "1H", "shift_ppm": 7.28},
             {"label": "X", "nuclide": "31P", "shift_ppm": 20}],
            [{"a": "A", "b": "B", "J_Hz": 8.0}, {"a": "A", "b": "X", "J_Hz": 12.0},
             {"a": "B", "b": "X", "J_Hz": -3.0}]),
    "A2B3": ([{"label": "A", "nuclide": "1H", "shift_ppm": 1.30, "n": 2},
              {"label": "B", "nuclide": "1H", "shift_ppm": 1.20, "n": 3}],
             [{"a": "A", "b": "B", "J_Hz": 7.1}]),
    "AA'XX'-like with 2H and 19F2": (
        [{"label": "A", "nuclide": "1H", "shift_ppm": 4.00},
         {"label": "B", "nuclide": "1H", "shift_ppm": 3.97},
         {"label": "D", "nuclide": "2H", "shift_ppm": 4.0},
         {"label": "F", "nuclide": "19F", "shift_ppm": -120, "n": 2}],
        [{"a": "A", "b": "B", "J_Hz": 11.0}, {"a": "A", "b": "D", "J_Hz": 1.5},
         {"a": "B", "b": "F", "J_Hz": 47.0}, {"a": "A", "b": "F", "J_Hz": -5}]),
    "strongly coupled 19F pair seen from 1H": (
        [{"label": "H", "nuclide": "1H", "shift_ppm": 5.0},
         {"label": "F1", "nuclide": "19F", "shift_ppm": -100.00},
         {"label": "F2", "nuclide": "19F", "shift_ppm": -100.05}],
        [{"a": "F1", "b": "F2", "J_Hz": 250.0}, {"a": "H", "b": "F1", "J_Hz": 50.0}]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_matches_full_hilbert_space_diagonalisation(name):
    spins, couplings = CASES[name]
    t = SS.spin_system_transitions(spins, couplings, B0, observe="1H")
    f, w = _brute(spins, couplings)
    assert len(t["stick_ppm"]) == len(f)
    assert np.allclose(t["stick_ppm"], f, atol=1e-10)
    assert np.allclose(t["intensity"], w, atol=1e-10)
    assert t["intensity"].sum() == pytest.approx(t["n_observed"])     # sum rule


def test_ab_quartet_in_closed_form():
    dnu, J = 12.0, 8.0
    sp = [{"label": "A", "nuclide": "1H", "shift_ppm": 5 + dnu / 2 / L_1H * 1e6},
          {"label": "B", "nuclide": "1H", "shift_ppm": 5 - dnu / 2 / L_1H * 1e6}]
    t = SS.spin_system_transitions(sp, [{"a": "A", "b": "B", "J_Hz": J}], B0)
    D = np.hypot(dnu, J)
    centre = 5 * L_1H * 1e-6
    assert np.allclose(t["stick_Hz"] - centre, [-(D + J) / 2, -(D - J) / 2, (D - J) / 2, (D + J) / 2])
    s2 = J / D                                    # roofing: inner lines grow
    assert np.allclose(t["intensity"], np.array([1 - s2, 1 + s2, 1 + s2, 1 - s2]) / 2)


def test_equivalent_spins_hide_their_coupling():
    sp = [{"label": "A", "nuclide": "1H", "shift_ppm": 2.0, "n": 2}]
    t = SS.spin_system_transitions(sp, [], B0)
    assert len(t["stick_ppm"]) == 1 and t["intensity"][0] == pytest.approx(2.0)


def test_weak_coupling_limit_is_first_order():
    # A3X2 far apart: the A part is the first-order triplet.
    sp = [{"label": "A", "nuclide": "1H", "shift_ppm": 1.0, "n": 3},
          {"label": "X", "nuclide": "1H", "shift_ppm": 8.0, "n": 2}]
    t = SS.spin_system_transitions(sp, [{"a": "A", "b": "X", "J_Hz": 7.0}], B0)
    a = t["stick_ppm"] < 4
    f1, w1 = S.spin_multiplet_pattern(2, 0.5, 7.0)
    hz = (t["stick_ppm"][a] - 1.0) * L_1H * 1e-6
    # Second order splits each line into components ~J^2/dnu apart
    # (0.02 Hz here); gather them before comparing with first order.
    groups = np.round(hz / 7.0).astype(int)
    pos = [np.average(hz[groups == g], weights=t["intensity"][a][groups == g]) for g in (-1, 0, 1)]
    wts = [t["intensity"][a][groups == g].sum() / 3 for g in (-1, 0, 1)]
    assert np.allclose(pos, f1, atol=0.02)
    assert np.allclose(wts, w1, atol=0.01)


def test_heteronuclear_couplings_stay_first_order():
    sp = [{"label": "H", "nuclide": "1H", "shift_ppm": 3.0},
          {"label": "P", "nuclide": "31P", "shift_ppm": 3.0}]         # same ppm: irrelevant
    t = SS.spin_system_transitions(sp, [{"a": "H", "b": "P", "J_Hz": 20.0}], B0)
    assert np.allclose(t["stick_Hz"] - 3.0 * L_1H * 1e-6, [-10, 10])
    assert np.allclose(t["intensity"], [0.5, 0.5])


def test_bad_systems_are_refused():
    sp = [{"label": "A", "nuclide": "1H", "shift_ppm": 1.0}]
    with pytest.raises(ValueError, match="no spin labelled"):
        SS.spin_system_transitions(sp, [{"a": "A", "b": "Q", "J_Hz": 1}], B0)
    with pytest.raises(ValueError, match="itself"):
        SS.spin_system_transitions(sp, [{"a": "A", "b": "A", "J_Hz": 1}], B0)
    with pytest.raises(ValueError, match="No 13C"):
        SS.spin_system_transitions(sp, [], B0, observe="13C")
    big = [{"label": f"H{i}", "nuclide": "1H", "shift_ppm": i} for i in range(15)]
    with pytest.raises(ValueError, match="block"):
        SS.spin_system_transitions(big, [], B0)


def test_coupling_strength_ratio():
    sp = [{"label": "A", "nuclide": "1H", "shift_ppm": 2.00},
          {"label": "B", "nuclide": "1H", "shift_ppm": 2.02},
          {"label": "P", "nuclide": "31P", "shift_ppm": 0}]
    out = SS.coupling_strengths(sp, [{"a": "A", "b": "B", "J_Hz": 8.0},
                                     {"a": "A", "b": "P", "J_Hz": 8.0}], B0)
    assert len(out) == 1
    assert out[0]["ratio"] == pytest.approx(0.02 * L_1H * 1e-6 / 8.0)


def test_spectrum_scales():
    sp, cp = CASES["A2B3"]
    x = np.linspace(0.9, 1.6, 40001)
    area = SS.spin_system_spectrum(sp, cp, B0, x, fwhm_Hz=0.5)
    assert area["real_spec"].sum() == pytest.approx(5.0, rel=5e-3)  # 5 protons
    x = np.linspace(1.0, 1.4, 4001)                  # has a point exactly at 1.2
    one = SS.spin_system_spectrum([{"label": "A", "nuclide": "1H", "shift_ppm": 1.2}], [], B0,
                                  x, fwhm_Hz=0.5, scale="peak")
    assert one["real_spec"].max() == pytest.approx(1.0, abs=1e-9)


# --- fitting ------------------------------------------------------------------

def _data(spins, couplings, fwhm=0.7, height=500.0, baseline=0.0, n=65536, lo=6.5, hi=8.0):
    x = np.linspace(hi, lo, n)
    y = height * SS.spin_system_spectrum(spins, couplings, B0, x, fwhm_Hz=fwhm,
                                         scale="peak")["real_spec"] + baseline
    return {"x_ppm": x, "real_spectrum": y, "SFO1": L_1H / 1e6, "nuclide": "1H"}


def test_abx_fit_finds_both_solutions():
    # ABX observed on A and B is ambiguous: a second (delta_A, delta_B,
    # J_AX, J_BX) gives the *identical* AB spectrum. A single fit may land
    # on either; equivalent_solutions must report both, each reproducing
    # the data, one of them the truth.
    spins, couplings = CASES["ABX"]
    res = _data(spins, couplings)
    start = [dict(s) for s in spins]
    start[0]["shift_ppm"] += 0.002
    start[1]["shift_ppm"] -= 0.002
    popt, pcov, free, info = SS.fit_spin_system_spectrum(
        res, start, [{"a": "A", "b": "B", "J_Hz": 7.5}, {"a": "A", "b": "X", "J_Hz": 11.0},
                     {"a": "B", "b": "X", "J_Hz": -2.0}], fit_regions=[(7.2, 7.4)])
    assert not free[info["names"].index("shift X (ppm)")]                 # undetermined: held
    assert any("shift X" in w for w in info["warnings"])
    solutions = SS.equivalent_solutions(res, popt, info)
    assert len(solutions) == 2
    truth = {"shift A (ppm)": 7.30, "shift B (ppm)": 7.28, "J A-B (Hz)": 8.0,
             "J A-X (Hz)": 12.0, "J B-X (Hz)": -3.0}
    matches = []
    for sol in solutions:
        got = dict(zip(info["names"], sol["popt"]))
        model = SS.evaluate_spin_system_fit(res["x_ppm"], sol["popt"], info)
        assert np.max(np.abs(model - res["real_spectrum"])) < 1e-4 * 500   # both fit
        assert got["J A-B (Hz)"] == pytest.approx(8.0, abs=1e-4)           # J_AB is common
        assert got["fwhm (Hz)"] == pytest.approx(0.7, abs=1e-4)
        matches.append(all(abs(got[k] - v) < (1e-6 if "ppm" in k else 1e-3)
                           for k, v in truth.items()))
    assert sum(matches) == 1
    assert set(solutions[1]["differences"]) >= {"J A-X (Hz)", "J B-X (Hz)"}


def test_an_unambiguous_fit_has_one_solution():
    spins = [{"label": "A", "nuclide": "1H", "shift_ppm": 7.30},
             {"label": "B", "nuclide": "1H", "shift_ppm": 7.28}]
    couplings = [{"a": "A", "b": "B", "J_Hz": 8.0}]
    res = _data(spins, couplings)
    popt, _pcov, _free, info = SS.fit_spin_system_spectrum(res, spins, couplings,
                                                            fit_regions=[(7.2, 7.4)])
    assert dict(zip(info["names"], popt))["shift A (ppm)"] == pytest.approx(7.30, abs=1e-7)
    assert len(SS.equivalent_solutions(res, popt, info)) == 1
    spins_f, couplings_f, fwhm_f, amp_f = SS.fitted_system(popt, info)
    assert fwhm_f == pytest.approx(0.7, abs=1e-4) and amp_f == pytest.approx(500, rel=1e-4)


def test_fixed_and_bounded_parameters():
    spins, couplings = CASES["A2B3"]
    res = _data(spins, couplings, lo=0.9, hi=1.6)
    popt, _pcov, free, info = SS.fit_spin_system_spectrum(
        res, spins, [{"a": "A", "b": "B", "J_Hz": 6.0}], fixed={"J A-B (Hz)": True},
        bounds={"shift A (ppm)": (1.29, 1.31)})
    k = info["names"].index("J A-B (Hz)")
    assert popt[k] == 6.0 and not free[k]
    assert 1.29 <= popt[info["names"].index("shift A (ppm)")] <= 1.31
    with pytest.raises(ValueError, match="Unknown parameter"):
        SS.fit_spin_system_spectrum(res, spins, couplings, fixed={"J A-C (Hz)": True})


# --- isotope-resolved coupling groups ----------------------------------------

def test_boron_gives_both_isotope_multiplets():
    f, w = S.jcoupling_group_pattern(1, 1.0, nuclide="11B", all_isotopes=True)
    r = find_isotope_info("10B")["gamma"] / find_isotope_info("11B")["gamma"]
    b11 = {round(k, 6) for k in (-1.5, -0.5, 0.5, 1.5)}
    b10 = {round(k * r, 6) for k in range(-3, 4)}
    got = dict(zip(np.round(f, 6), w))
    assert set(got) == b11 | b10
    assert sum(got[k] for k in b11) == pytest.approx(0.801, abs=1e-9)
    assert sum(got[k] for k in b10) == pytest.approx(0.199, abs=1e-9)
    assert np.allclose([got[k] for k in b10], 0.199 / 7)             # 1:1:1:1:1:1:1


def test_mercury_satellites_have_the_right_spins():
    # 199Hg (I=1/2, 16.87 %) gives a doublet; 201Hg (I=3/2, 13.18 %) a
    # quartet at J x gamma(201)/gamma(199); the rest (69.95 %) a centre line.
    f, w = S.jcoupling_group_pattern(1, 100.0, nuclide="199Hg", all_isotopes=True)
    got = dict(zip(np.round(f, 4), w))
    r = find_isotope_info("201Hg")["gamma"] / find_isotope_info("199Hg")["gamma"]
    assert got[0.0] == pytest.approx(0.6995, abs=1e-9)
    assert got[50.0] == pytest.approx(0.1687 / 2, abs=1e-9)
    assert got[round(abs(1.5 * 100 * r), 4)] == pytest.approx(0.1318 / 4, abs=1e-9)


def test_all_isotopes_reduces_to_the_old_pattern_when_only_one_is_magnetic():
    a = S.jcoupling_group_pattern(2, 7.0, nuclide="13C", all_isotopes=True)
    b = S.jcoupling_group_pattern(2, 7.0, nuclide="13C")
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_all_isotopes_cannot_take_overrides():
    with pytest.raises(ValueError, match="all_isotopes"):
        S.jcoupling_group_pattern(1, 1.0, nuclide="11B", abundance=100.0, all_isotopes=True)


def test_all_isotopes_fits_one_j():
    from opennmr.notebook import jcoupling_fitting as JF
    x = np.linspace(-150.5, -151.5, 16384)          # 19F of BF4-, J(11B-F) = 1.3 Hz
    L_F = abs(find_isotope_info("19F")["gamma"]) * B0
    group = [{"nuclide": "11B", "n": 1, "J_Hz": 1.3, "all_isotopes": True}]
    freqs, w = S.combine_jcouplings(group)
    fw = 0.2 / L_F * 1e6
    y = 1000 * S.multiplet_lineshape(x, -151.0 + freqs / L_F * 1e6, w, fw, normalise="peak")
    res = {"x_ppm": x, "real_spectrum": y, "SFO1": L_F / 1e6, "nuclide": "19F"}
    # A start within about half a linewidth of the answer (a start 0.1 Hz
    # off with 0.2 Hz lines falls into a broad-line local minimum, as any
    # least-squares fit of resolved multiplets can).
    p0 = {"center_ppm": -151.0, "fwhm": 0.0006,
          "couplings": [{"nuclide": "11B", "n": 1, "J_Hz": 1.25, "all_isotopes": True}]}
    popt, *_ = JF.fit_jcoupling_spectrum(res, p0, larmor_Hz=L_F)
    assert popt[1] == pytest.approx(1.3, abs=1e-4)


# --- the isotope table's spin column ------------------------------------------

#: Nuclear spins from the IAEA nuclear-moments data, as tabulated at
#: https://www.kherb.io/docs/nmr_table.csv (checked 2026-09-24). Until
#: then the tables had the spins of 161Dy ... 209Bi shifted (199Hg as
#: 3/2, 205Tl as 4, 183W as 5/2, ...).
REFERENCE_SPINS = dict(item.split("=") for item in """
1H=0.5 2H=1 3He=0.5 3H=0.5 6Li=1 7Li=1.5 9Be=1.5 10B=3 11B=1.5 13C=0.5 14N=1 15N=0.5
17O=2.5 19F=0.5 21Ne=1.5 23Na=1.5 25Mg=2.5 27Al=2.5 29Si=0.5 31P=0.5 33S=1.5 35Cl=1.5
37Cl=1.5 39K=1.5 40K=4 41K=1.5 43Ca=3.5 45Sc=3.5 47Ti=2.5 49Ti=3.5 50V=6 51V=3.5 53Cr=1.5
55Mn=2.5 57Fe=0.5 59Co=3.5 61Ni=1.5 63Cu=1.5 65Cu=1.5 67Zn=2.5 69Ga=1.5 71Ga=1.5 73Ge=4.5
75As=1.5 77Se=0.5 79Br=1.5 81Br=1.5 83Kr=4.5 85Rb=2.5 87Rb=1.5 87Sr=4.5 89Y=0.5 91Zr=2.5
93Nb=4.5 95Mo=2.5 97Mo=2.5 99Ru=2.5 99Tc=4.5 101Ru=2.5 103Rh=0.5 105Pd=2.5 107Ag=0.5
109Ag=0.5 111Cd=0.5 113In=4.5 113Cd=0.5 115In=4.5 115Sn=0.5 117Sn=0.5 119Sn=0.5 121Sb=2.5
123Sb=3.5 123Te=0.5 125Te=0.5 127I=2.5 129Xe=0.5 131Xe=1.5 133Cs=3.5 135Ba=1.5 137Ba=1.5
138La=5 139La=3.5 141Pr=2.5 143Nd=3.5 145Nd=3.5 147Sm=3.5 149Sm=3.5 151Eu=2.5 153Eu=2.5
155Gd=1.5 157Gd=1.5 159Tb=1.5 161Dy=2.5 163Dy=2.5 165Ho=3.5 167Er=3.5 169Tm=0.5 171Yb=0.5
173Yb=2.5 175Lu=3.5 176Lu=7 177Hf=3.5 179Hf=4.5 181Ta=3.5 183W=0.5 185Re=2.5 187Re=2.5
187Os=0.5 189Os=1.5 191Ir=1.5 193Ir=1.5 195Pt=0.5 197Au=1.5 199Hg=0.5 201Hg=1.5 203Tl=0.5
205Tl=0.5 207Pb=0.5 209Bi=4.5 235U=3.5""".split())


@pytest.mark.parametrize("module", ["opennmr.notebook.utils", "opennmr.core.registry"])
def test_every_tabulated_spin_is_right(module):
    import importlib
    df = importlib.import_module(module)._GAMMA_DF
    wrong = {iso: sp for iso, sp in zip(df["Nuclide"], df["Spin"])
             if float(sp) != float(REFERENCE_SPINS[iso])}
    assert wrong == {}
