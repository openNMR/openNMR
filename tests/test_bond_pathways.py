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
