"""Bond-pathway contributions from DFT (user request, 2026-11-06): spin-flip
and single-ion calculations built synthetically with known pathways, so the
answer is written down beforehand; ambiguity of periodic images; the
additivity check; the plot and notebook code."""

import numpy as np
import pytest

from opennmr.notebook import bond_pathways as bp
from opennmr.notebook import crystal_nmr as C

A = 4.0
L = np.diag([2 * A, 2 * A, 2 * A])
MN = [np.array([i, j, k]) * A for i in (0, 1) for j in (0, 1) for k in (0, 1)]
NA = [m + np.array([A / 2, A / 2, A / 2]) for m in MN]
SCALING = {"phi": 0.05, "B0": 9.4}
UNIT = C.fermi_contact_shift(1.0, 0.05, 9.4)          # ppm per bohr⁻³ of spin density
BINS = [("NN", 0.0, 3.6)]
P_NN = {"Mn": 10.0, "Fe": -4.0}


def _calc(spins, species=None, tag="c", bins=BINS, P=None):
    """A CRYSTAL-like calculation whose Na contact shifts are exactly the sum
    of P over the (spin-signed) paramagnetic neighbours in the shells."""
    species = species or ["Mn"] * len(MN)
    P = P or {"NN": P_NN}
    atoms = []
    for k, p in enumerate(MN):
        atoms.append({"species": species[k], "label": f"{species[k]}{k + 1}", "index": k,
                      "position": p, "rho": float(spins[k]), "hf": {}, "efg": None,
                      "sigma": None})
    for k, p in enumerate(NA):
        tot = 0.0
        for j, m in enumerate(MN):
            if spins[j] == 0:
                continue
            for d in bp._image_distances(m, p, L, max(b[2] for b in bins)):
                sh = bp._shell(d, bins)
                if sh:
                    tot += spins[j] * P[sh][species[j]]
        atoms.append({"species": "Na", "label": f"Na{k + 1}", "index": 8 + k, "position": p,
                      "rho": tot / UNIT, "hf": {}, "efg": None, "sigma": None})
    return {"code": "CRYSTAL", "path": tag, "lattice": L, "atoms": atoms, "notes": [],
            "info": {}, "method": {}, "spin_total": None, "energy_Eh": None}


def test_hyperfine_shifts_of_a_crystal_calculation():
    c = _calc([1] * 8)
    s = bp.hyperfine_shifts(c, SCALING)
    assert s["Na1"] == pytest.approx(8 * 10.0)        # 8 Mn at 3.46 Å
    with pytest.raises(ValueError):
        bp.hyperfine_shifts({"code": "VASP", "atoms": [], "path": "x"}, SCALING)


def test_spin_flip_recovers_the_pathways_and_passes_additivity():
    species = ["Mn", "Fe"] * 4
    ref = _calc([1] * 8, species, "fm")
    flips = [_calc([1 if i != j else -1 for i in range(8)], species, f"flip{j}") for j in range(8)]
    res = bp.pathway_contributions(flips, reference=ref, observed="Na", magnetic=["Mn", "Fe"],
                                   shell_bins=BINS, paramagnetic=SCALING)
    assert res["shift_dict"]["NN"]["Mn"] == pytest.approx(10.0)
    assert res["shift_dict"]["NN"]["Fe"] == pytest.approx(-4.0)
    assert res["additivity"]["max"] == pytest.approx(0.0, abs=1e-9)
    s = res["summary"][("NN", "Mn")]
    assert s["n"] == 4 * 8 and s["sd"] == pytest.approx(0.0, abs=1e-9)


def test_flipped_ion_found_from_spin_density_or_named():
    ref = _calc([1] * 8, tag="fm")
    flip = _calc([1, 1, -1, 1, 1, 1, 1, 1], tag="flip")
    auto = bp.pathway_contributions([flip], reference=ref, observed="Na", magnetic="Mn",
                                    shell_bins=BINS, paramagnetic=SCALING)
    assert {r["ion"] for r in auto["rows"]} == {"Mn3"}
    named = bp.pathway_contributions([flip], reference=ref, observed="Na", magnetic="Mn",
                                     shell_bins=BINS, paramagnetic=SCALING, flipped=["Mn3"])
    assert named["shift_dict"] == auto["shift_dict"]
    same = _calc([1] * 8, tag="same")
    with pytest.raises(ValueError, match="cannot tell"):
        bp.pathway_contributions([same], reference=ref, observed="Na", magnetic="Mn",
                                 shell_bins=BINS, paramagnetic=SCALING)


def test_single_ion_calculations():
    calcs = []
    for j in (0, 5):
        spins = [0] * 8
        spins[j] = 1
        sp = ["Ga"] * 8
        sp[j] = "Mn"
        calcs.append(_calc(spins, sp, f"single{j}"))
    res = bp.pathway_contributions(calcs, observed="Na", magnetic=["Mn"], shell_bins=BINS,
                                   mode="single", paramagnetic=SCALING)
    assert res["shift_dict"]["NN"]["Mn"] == pytest.approx(10.0)
    assert res["additivity"] is None


def test_images_inside_the_shells_are_flagged():
    big = [("NN", 0.0, 3.6), ("NNN", 3.6, 7.0)]
    P = {"NN": {"Mn": 10.0}, "NNN": {"Mn": -1.0}}
    ref = _calc([1] * 8, tag="fm", bins=big, P=P)
    flip = _calc([-1] + [1] * 7, tag="flip", bins=big, P=P)
    res = bp.pathway_contributions([flip], reference=ref, observed="Na", magnetic="Mn",
                                   shell_bins=big, paramagnetic=SCALING)
    assert any(r["ambiguous"] for r in res["rows"])
    assert any("supercell" in n for n in res["notes"])


def test_input_errors():
    ref = _calc([1] * 8, tag="fm")
    with pytest.raises(ValueError, match="reference"):
        bp.pathway_contributions([ref], observed="Na", magnetic="Mn", shell_bins=BINS,
                                 paramagnetic=SCALING)
    with pytest.raises(ValueError, match="mode"):
        bp.pathway_contributions([ref], observed="Na", magnetic="Mn", shell_bins=BINS,
                                 mode="x", paramagnetic=SCALING)
    with pytest.raises(ValueError, match="shell"):
        bp.pathway_contributions([ref], reference=ref, observed="Na", magnetic="Mn",
                                 shell_bins=[], paramagnetic=SCALING)


def test_plot_and_python():
    import matplotlib
    matplotlib.use("Agg")
    ref = _calc([1] * 8, tag="fm")
    flip = _calc([-1] + [1] * 7, tag="flip")
    res = bp.pathway_contributions([flip], reference=ref, observed="Na", magnetic=["Mn"],
                                   shell_bins=BINS, paramagnetic=SCALING)
    fig = bp.plot_pathways(res)
    assert "r" in fig.axes[0].get_xlabel() and "P" in fig.axes[0].get_ylabel()
    code = bp.pathways_to_python(["a.out"], reference="fm.out", observed="Na", magnetic=["Mn"],
                                 shell_bins=BINS, paramagnetic=SCALING, mode="spin_flip")
    compile(code, "<pathways>", "exec")


# ---- which ion, and the neighbour distances (explicit user request, 2026-11-07) -------------

def _chain(spins, rho_on_metals=True, tag="c"):
    """Mn at x = 0, 6, 12, 18 Å and Na halfway between, in a 24 × 10 × 10 Å cell:
    each Na's spin density comes from its two Mn neighbours (3 Å away)."""
    lat = np.diag([24.0, 10.0, 10.0])
    atoms = []
    for k in range(4):
        atoms.append({"species": "Mn", "label": f"Mn{k + 1}", "index": k,
                      "position": np.array([6.0 * k, 0, 0]),
                      "rho": float(spins[k]) if rho_on_metals else None, "hf": {}, "efg": None,
                      "sigma": None})
    for k in range(4):
        rho = 0.01 * (spins[k] + spins[(k + 1) % 4])
        atoms.append({"species": "Na", "label": f"Na{k + 1}", "index": 4 + k,
                      "position": np.array([6.0 * k + 3.0, 0, 0]), "rho": rho, "hf": {},
                      "efg": None, "sigma": None})
    return {"code": "CRYSTAL", "path": tag, "lattice": lat, "atoms": atoms, "notes": [],
            "info": {}, "method": {}, "spin_total": None, "energy_Eh": None}


def test_identify_ions_by_spin_density_sign():
    ref = _chain([1, 1, 1, 1], tag="fm")
    found = bp.identify_ions([_chain([1, 1, -1, 1], tag="f3")], magnetic="Mn", reference=ref)
    assert found[0]["ion"] == "Mn3" and found[0]["certain"]
    assert "changed sign" in found[0]["how"] and "→" in found[0]["evidence"]


def test_identify_ions_from_the_changed_neighbours_when_the_ion_has_no_rho():
    ref = _chain([1, 1, 1, 1], rho_on_metals=False, tag="fm")
    flip = _chain([1, 1, -1, 1], rho_on_metals=False, tag="f3")
    found = bp.identify_ions([flip], magnetic="Mn", reference=ref)
    assert found[0]["ion"] == "Mn3" and found[0]["certain"]
    assert "sit around" in found[0]["how"] and "3.00 Å" in found[0]["evidence"]


def test_an_uncertain_ion_is_flagged_and_noted():
    """In the 2×2×2 cell every Na sees every Mn at the same distance: the
    fallback cannot tell, says so, and the result asks to check it."""
    ref = _calc([1] * 8, tag="fm")
    flip = _calc([1, 1, -1, 1, 1, 1, 1, 1], tag="flip")
    for c in (ref, flip):
        for a in c["atoms"]:
            if a["species"] == "Mn":
                a["rho"] = None
    found = bp.identify_ions([flip], magnetic="Mn", reference=ref)
    assert found[0]["ion"] is not None and not found[0]["certain"]
    res = bp.pathway_contributions([flip], reference=ref, observed="Na", magnetic="Mn",
                                   shell_bins=BINS, paramagnetic=SCALING)
    assert any("check it, or name it" in n for n in res["notes"])


def test_identify_ions_given_and_single():
    ref = _chain([1, 1, 1, 1], tag="fm")
    found = bp.identify_ions([_chain([1, 1, -1, 1])], magnetic="Mn", reference=ref,
                             flipped=["Mn4"])
    assert found[0]["ion"] == "Mn4" and found[0]["how"] == "given"
    with pytest.raises(ValueError, match="no atom Mn9"):
        bp.identify_ions([_chain([1, 1, -1, 1])], magnetic="Mn", reference=ref, flipped=["Mn9"])
    single = _chain([0, 0, 1, 0])
    found = bp.identify_ions([single], magnetic="Mn", mode="single")
    assert found[0]["ion"] == "Mn3" and found[0]["certain"]


def test_neighbour_distances_shells_and_plot():
    import matplotlib
    matplotlib.use("Agg")
    rows = bp.neighbour_distances(_chain([1, 1, 1, 1]), "Na", "Mn", max_r=10.0)
    d = sorted({round(r["distance"], 6) for r in rows})
    assert d[:2] == [3.0, 9.0]                   # nearest Mn, then the next along the chain
    per_na = [r for r in rows if r["nucleus"] == "Na1" and r["distance"] < 3.5]
    assert len(per_na) == 2
    shells = bp.suggest_shells(rows, max_r=10.0)
    assert shells[0][0] == "NN" and shells[0][1] < 3.0 < shells[0][2] < 9.0
    fig = bp.plot_neighbour_distances(rows, shells, max_r=10.0, title="chain")
    assert "$" in fig.axes[0].get_xlabel()
    with pytest.raises(ValueError, match="No Li"):
        bp.neighbour_distances(_chain([1, 1, 1, 1]), "Li", "Mn")
