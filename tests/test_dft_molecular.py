"""ORCA and Gaussian NMR outputs (user request, 2026-10-07) --
notebook.dft_nmr.read_orca / read_gaussian.

Synthetic files in each code's layout always run. Real outputs (from
cclib's test data, placed in test_data/dft/molecular/, gitignored) are
checked against each code's own printed values when present."""

import glob
import os
import re

import numpy as np
import pytest

from opennmr.notebook import dft_nmr as D

from test_dft_nmr import _rotation, _tensor

DATA = os.path.join(os.path.dirname(__file__), "..", "test_data", "dft", "molecular")
real = pytest.mark.skipif(not os.path.isdir(DATA), reason="test_data/dft/molecular not present")


def _rows(t):
    return ["      " + "  ".join(f"{v:12.3f}" for v in row) for row in np.asarray(t)]


def write_orca(path, atoms, efg=None, version="5.0.4", keywords="NMR PBE0 pcSseg-2",
               e2qQ_override=None):
    """atoms: [(symbol, position, sigma 3x3)]; efg: {index: (matrix au, Q barn)}."""
    L = ["", "                                 *****************",
         "                                 * O   R   C   A *",
         "                                 *****************", "",
         f"                         Program Version {version} -  RELEASE  -", "",
         f"|  1> ! {keywords}", "|  2> *xyz 0 1", "",
         "---------------------------------", "CARTESIAN COORDINATES (ANGSTROEM)",
         "---------------------------------"]
    for sym, pos, _s in atoms:
        L.append(f"  {sym:<4}" + "".join(f"{v:14.6f}" for v in pos))
    L += ["", "-------------------", "CHEMICAL SHIELDINGS (ppm)", "-------------------", ""]
    for n, (sym, _p, sigma) in enumerate(atoms):
        L += [" --------------", f" Nucleus   {n}{sym} :", " --------------", "",
              "Total shielding tensor (ppm): "] + _rows(sigma) + [""]
    for n, (V, Q) in (efg or {}).items():
        sym = atoms[n][0]
        p = D.efg_parameters(np.asarray(V) * D.AU_EFG, Q)
        e2qq = p["C_Q"] if e2qQ_override is None else e2qQ_override
        L += [" -----------------------------------------------------------",
              f" Nucleus   {n}{sym} : A:ISTP=   14 I=  1.0 P=  38.5677 MHz/au**3",
              f"                Q:ISTP=   14 I=  1.0 Q=   {Q:.4f} barn",
              " -----------------------------------------------------------",
              " Raw EFG matrix (all values in a.u.**-3):", " -----------------------------------"]
        L += ["  " + "".join(f"{v:13.4f}" for v in row) for row in np.asarray(V)]
        L += ["", " V(Tot)  " + "".join(f"{v / D.AU_EFG:13.4f}" for v in p["principal"]), " Orientation:",
              "  X       1.0000000    0.0000000    0.0000000",
              f" Quadrupole tensor eigenvalues (in MHz;Q= {Q:.4f} I=  1.0)",
              f"  e**2qQ            = {e2qq:9.3f} MHz",
              f"  e**2qQ/(4I*(2I-1))= {e2qq / 4:9.3f} MHz",
              f"  eta               = {p['eta_Q']:9.3f}", ""]
    path.write_text("\n".join(L) + "\n")
    return str(path)


def write_gaussian(path, atoms, blocks=("SCF",), route="#p NMR B3LYP/6-311+G(2d,p)",
                   orientation="Standard"):
    """atoms: [(Z, symbol, position, sigma 3x3)]; one shielding block per
    method in ``blocks`` (later ones scaled by 1.01 to tell them apart)."""
    L = [" Entering Gaussian System, Link 0=g16", " Copyright (c) 1988-2019, Gaussian, Inc.",
         " -----------------------------------", f" {route}",
         " -----------------------------------", "",
         f"                         {orientation} orientation:",
         " ---------------------------------------------------------------------",
         " Center     Atomic      Atomic             Coordinates (Angstroms)",
         " Number     Number       Type             X           Y           Z",
         " ---------------------------------------------------------------------"]
    for n, (Z, _sym, pos, _s) in enumerate(atoms, 1):
        L.append(f"   {n:4d}   {Z:8d}           0   " + "".join(f"{v:12.6f}" for v in pos))
    L.append(" ---------------------------------------------------------------------")
    for k, meth in enumerate(blocks):
        L.append(f" {meth} GIAO Magnetic shielding tensor (ppm):")
        for n, (_Z, sym, _p, sigma) in enumerate(atoms, 1):
            s = np.asarray(sigma) * (1.01 ** k)
            iso = np.trace(s) / 3
            L.append(f"  {n:4d}  {sym:<2}   Isotropic = {iso:12.4f}   Anisotropy = {0.0:12.4f}")
            # Gaussian's layout: XX= YX= ZX= / XY= YY= ZY= / XZ= YZ= ZZ=
            for col in range(3):
                L.append("   " + "   ".join(f"{'XYZ'[r]}{'XYZ'[col]}={s[r, col]:12.4f}"
                                            for r in range(3)))
            w = np.linalg.eigvalsh(0.5 * (s + s.T))
            L.append("   Eigenvalues:" + "".join(f"{v:12.4f}" for v in w))
    L += ["", " Normal termination of Gaussian 16"]
    path.write_text("\n".join(L) + "\n")
    return str(path)


SIG_N = _tensor((-120.0, -20.0, 150.0), _rotation(11))
SIG_H = _tensor((24.0, 28.0, 33.0), _rotation(12))
EFG_N = _tensor((0.20, 0.40, -0.60), _rotation(13))          # au, η = (0.2−0.4)/−0.6 = 1/3


def test_orca_shieldings_coordinates_and_method(tmp_path):
    asym = SIG_N + np.array([[0, 2.0, 0], [-2.0, 0, 0], [0, 0, 0]])  # an antisymmetric part
    path = write_orca(tmp_path / "nh3.out", [("N", (0, 0, 0.1), asym), ("H", (0.9, 0, -0.3), SIG_H),
                                             ("H", (-0.45, 0.78, -0.3), SIG_H)])
    calc = D.read_calculation(path)
    assert calc["code"] == "ORCA" and calc["lattice"] is None and calc["g0"] == "none"
    assert [a["label"] for a in calc["atoms"]] == ["N1", "H1", "H2"]
    assert np.allclose(calc["atoms"][1]["position"], (0.9, 0, -0.3))
    assert np.allclose(calc["atoms"][0]["sigma"], asym, atol=1e-3)
    assert np.allclose(D.symmetric(calc["atoms"][0]["sigma"]), SIG_N, atol=1e-3)
    assert calc["method"]["functional"] == "NMR PBE0 pcSseg-2" and calc["info"]["version"] == "5.0.4"


def test_orca_efg_matches_its_own_e2qq_and_flags_a_mismatch(tmp_path):
    atoms = [("N", (0, 0, 0), SIG_N), ("H", (1, 0, 0), SIG_H)]
    path = write_orca(tmp_path / "a.out", atoms, efg={0: (EFG_N, 0.0204)})
    calc = D.read_calculation(path)
    e = D.efg_parameters(calc["atoms"][0]["efg"], 0.0204)
    assert e["eta_Q"] == pytest.approx(1 / 3, abs=1e-4)
    assert e["Vzz"] / D.AU_EFG == pytest.approx(-0.60, abs=1e-4)
    assert e["C_Q"] == pytest.approx(D.E_OVER_H * 0.0204e-28 * -0.60 * D.AU_EFG / 1e6, rel=1e-4)
    assert calc["orca_summary"]["N1"]["Q"] == 0.0204
    assert not any("Check this file's EFG" in n for n in calc["notes"])
    bad = D.read_calculation(write_orca(tmp_path / "b.out", atoms, efg={0: (EFG_N, 0.0204)},
                                        e2qQ_override=+2.88))
    assert any("Check this file's EFG" in n for n in bad["notes"])
    model = D.dft_sites(calc, None, element="N", sigma_ref=-130.0)
    assert model["nucleus"] == "14N" and model["atoms"][0]["C_Q"] is not None


def test_orca_nucleus_that_does_not_match_the_coordinates_is_an_error(tmp_path):
    path = write_orca(tmp_path / "x.out", [("N", (0, 0, 0), SIG_N)])
    text = open(path).read().replace("Nucleus   0N", "Nucleus   0C")
    with open(path, "w") as f:
        f.write(text)
    with pytest.raises(ValueError, match="does not match"):
        D.read_calculation(path)


def test_gaussian_tensor_layout_orientation_and_route(tmp_path):
    asym = SIG_N + np.array([[0, 1.5, 0], [-1.5, 0, 0], [0, 0, 0]])
    path = write_gaussian(tmp_path / "nh3.log", [(7, "N", (0, 0, 0.1), asym),
                                                 (1, "H", (0.9, 0, -0.3), SIG_H)])
    calc = D.read_calculation(path)
    assert calc["code"] == "Gaussian" and [a["label"] for a in calc["atoms"]] == ["N1", "H1"]
    assert np.allclose(calc["atoms"][0]["sigma"], asym, atol=1e-3)     # element by element
    assert calc["info"]["orientation"] == "Standard"
    assert calc["method"]["functional"] == "#p NMR B3LYP/6-311+G(2d,p)"
    nosymm = D.read_calculation(write_gaussian(tmp_path / "n.log", [(1, "H", (0, 0, 0), SIG_H)],
                                               orientation="Input"))
    assert nosymm["info"]["orientation"] == "Input"


def test_gaussian_prefers_the_correlated_block(tmp_path):
    path = write_gaussian(tmp_path / "mp2.log", [(1, "H", (0, 0, 0), SIG_H)],
                          blocks=("SCF", "MP2"))
    calc = D.read_calculation(path)
    assert np.allclose(calc["atoms"][0]["sigma"], SIG_H * 1.01, atol=1e-3)
    assert any("MP2 shieldings used" in n for n in calc["notes"])


def test_molecular_sites_use_the_cartesian_frame(tmp_path):
    calc = D.read_calculation(write_orca(tmp_path / "m.out", [("N", (0, 0, 0), SIG_N)]))
    model = D.dft_sites(calc, None, element="N", sigma_ref=-150.0)
    with pytest.raises(ValueError, match="not periodic"):
        D.display_frame(calc, model["atoms"][0], "cell")
    assert D.display_frame(calc, model["atoms"][0], "cartesian")["axes"] == []


def test_method_differences():
    a = {"code": "CASTEP", "functional": "PBE", "cutoff": "600 eV",
         "pseudopotentials": {"H": "1|0.6|13|15|17|10(qc=8)"}}
    b = dict(a, cutoff="800 eV", pseudopotentials={"H": "1|0.8|..."})
    diffs = D.method_differences(a, b, "H")
    assert any("cutoff" in d for d in diffs) and any("pseudopotential" in d for d in diffs)
    assert D.method_differences(a, dict(a), "H") == []
    assert D.method_differences(a, {"code": "VASP"}) == ["code (CASTEP vs VASP)"]


# ---- real outputs (cclib's test data) --------------------------------------------------

@real
def test_real_gaussian_logs_match_gaussian_s_printed_values():
    for path in sorted(glob.glob(os.path.join(DATA, "g*.log"))):
        calc = D.read_calculation(path)
        text = open(path).read().splitlines()
        start = [i for i, line in enumerate(text) if "Magnetic shielding tensor" in line][-1]
        iso, eig = [], []
        for line in text[start + 1:]:
            m = re.match(r"\s*\d+\s+[A-Z][a-z]?\s+Isotropic =\s*(\S+)", line)
            if m:
                iso.append(float(m.group(1)))
            elif line.strip().startswith("Eigenvalues"):
                eig.append(sorted(float(v) for v in line.split()[1:4]))
            elif not re.match(r"\s*X[XYZ]=", line):
                break
        ours = [np.trace(a["sigma"]) / 3 for a in calc["atoms"]]
        assert np.allclose(ours, iso, atol=2e-4), path
        ours_eig = [np.sort(np.linalg.eigvalsh(D.symmetric(a["sigma"]))) for a in calc["atoms"]]
        assert np.allclose(ours_eig, eig, atol=2e-4), path


@real
def test_real_orca_outputs_match_orca_s_printed_isotropic_values():
    for path in sorted(glob.glob(os.path.join(DATA, "orca*.out"))):
        calc = D.read_calculation(path)
        text = open(path).read()
        printed = [float(v) for v in re.findall(r"^\s*Total\s+\S+\s+\S+\s+\S+\s+iso=\s*(\S+)",
                                                text, re.M)]
        ours = [np.trace(a["sigma"]) / 3 for a in calc["atoms"]]
        assert len(ours) == len(printed) == 20, path
        # 3-decimal tensors; ORCA's principal values are of sᵀs, ours of the
        # symmetric part, so only the isotropic values are compared.
        assert np.allclose(ours, printed, atol=1e-3), path
