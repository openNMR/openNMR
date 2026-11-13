"""orca_pnmr files and .xyz structures in Simulate ▸ DFT tensors (explicit user
request, 2026-11-13): "implement the pNMR/EFG tensor visualization also for
ORCA (i.e., xyz) and the output file formats for the regular ORCA output and
orca_pnmr".

* an orca_pnmr output or input (.pnmr.inp) is read as a calculation with g,
  D and hyperfine tensors, on the geometry of an .xyz or ORCA output;
* .xyz files (trajectories, ORCA's g-orientation files) are read;
* a relativistic (DKH) ORCA output's hyperfine tensors are found (its "Raw
  HFC matrix" sits ~60 lines below the nucleus header);
* the structure picture draws the Fermi-contact and spin-dipolar parts.

The reference shieldings are orca_pnmr's own printout for the user's Pr
cluster (test_paramagnetic_nmr.PR_*), so the check is independent."""

import os

import numpy as np
import pytest

from opennmr.notebook import dft_nmr as D
from opennmr.notebook import dft_structure as DS
from opennmr.notebook import paramagnetic_nmr as pm
from test_paramagnetic_nmr import PR_D, PR_G, PR_HFC

DATA = os.path.join(os.path.dirname(__file__), "..", "test_data", "dft", "paramagnetic")
real = pytest.mark.skipif(not os.path.isdir(DATA), reason="test_data/dft/paramagnetic not present")

# The Pr cluster's geometry, made up but with the elements at ORCA's indices
# of PR_HFC (1N, 2N, 7C): atom 0 is Pr.
ATOMS = [("Pr", 0.0, 0.0, 0.0), ("N", 2.5, 0.0, 0.0), ("N", -2.5, 0.0, 0.0),
         ("O", 0.0, 2.4, 0.0), ("O", 0.0, -2.4, 0.0), ("O", 0.0, 0.0, 2.4),
         ("O", 0.0, 0.0, -2.4), ("C", 3.65, 0.0, 0.0), ("H", 4.7, 0.0, 0.0)]


def _rows(m, indent=4):
    return [" " * indent + "  ".join(f"{v:11.6f}" for v in r) for r in m]


def write_pnmr_output(path, S=1.0, g=PR_G, Dcm=PR_D, hfc=None, T=298.0, inp="job.pnmr.inp"):
    """An orca_pnmr output in ORCA 6's layout (as the user's PrFeCN6_pnmr.txt).
    ``hfc``: {index: (element, P, A, σ tensor)}."""
    hfc = hfc or {k: (v[0], v[1], np.array(v[2]), np.array(v[3])) for k, v in PR_HFC.items()}
    out = [f"PNMR input file                 ... {inp}", "-" * 79,
           "                  ORCA PARAMAGNETIC NMR SHIELDING CALCULATION", "-" * 79, "",
           f"Electronic spin quantum number:  {S:.1f} ", "", "Temperature range:",
           f"  Start                         ... {T:.2f} K",
           f"  Stop                          ... {T:.2f} K",
           "  In steps of                   ...   1.00 K", "Number of temperatures          ... 1 ",
           "", "g tensor:", "        x            y            z    "] + _rows(g) + [""]
    if Dcm is None:
        out += ["Multiplicity is smaller than 3  -->  D tensor will not be used.", ""]
    else:
        out += ["D tensor (cm-1):", "        x            y            z    "] + _rows(Dcm) + [""]
    out += ["                                                     A (MHz)              ",
            "Nucleus  PFAC (MHz)  Aiso (MHz)          x             y             z    ",
            "-" * 74 + " "]
    for idx, (el, P, A, _s) in hfc.items():
        a = [f"{v:13.6f}" for v in np.ravel(A)]
        out += [" " * 33 + " ".join(a[0:3]),
                f"{idx:>4}{el:<2}  {P:12.6f} {np.trace(A) / 3:11.6f}  " + " ".join(a[3:6]),
                " " * 33 + " ".join(a[6:9]), ""]
    out += ["", "", "Paramagnetic shielding tensors (ppm):", "", f"Temperature: {T:.2f} K", "",
            "Nucleus         Iso        x          y          z   ", "-" * 54 + " "]
    for idx, (el, _P, _A, sig) in hfc.items():
        s = [f"{v:10.3f}" for v in np.ravel(sig)]
        out += [" " * 22 + " ".join(s[0:3]),
                f"{idx:>4}{el:<2}    {np.trace(sig) / 3:10.3f}    " + " ".join(s[3:6]),
                " " * 22 + " ".join(s[6:9]), ""]
    out += ["-" * 54, "", "", "Isotropic paramagnetic shielding summary:", ""]
    path.write_text("\n".join(out) + "\n")
    return str(path)


def write_pnmr_input(path, mult=3, g=PR_G, Dcm=PR_D, hfc=None, T=298.0):
    hfc = hfc or {k: (v[0], v[1], np.array(v[2])) for k, v in PR_HFC.items()}
    lines = [f"{mult} #Spin multiplicity (2S+1)", f"{T:g} #temp0", f"{T:g} #temp1",
             "1 #temp_step", "1 #have_g?"] + [" ".join(map(str, r)) for r in g]
    if Dcm is None:
        lines += ["0 #have_D?"]
    else:
        lines += ["1 #have_D?"] + [" ".join(map(str, r)) for r in Dcm]
    lines.append(f"{len(hfc)} #Number of A tensors")
    for idx, (el, P, A) in hfc.items():
        lines += [f"{idx}{el}", f"{P} #PFAC"] + [" ".join(map(str, r)) + " " for r in A]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def write_xyz(path, atoms=ATOMS, comment="Coordinates from ORCA-job job", frames=1):
    out = []
    for f in range(frames):
        out += [str(len(atoms)), comment]
        out += [f"{s:<3} {x + f:12.6f} {y:12.6f} {z:12.6f}" for s, x, y, z in atoms]
    path.write_text("\n".join(out) + "\n")
    return str(path)


# ---- .xyz ------------------------------------------------------------------------------

def test_xyz_is_read_by_content(tmp_path):
    calc = D.read_calculation(write_xyz(tmp_path / "m.dat"))
    assert calc["code"] == "XYZ" and calc["lattice"] is None
    assert [a["label"] for a in calc["atoms"][:3]] == ["Pr1", "N1", "N2"]
    assert np.allclose(calc["atoms"][7]["position"], [3.65, 0, 0])
    assert all(a["sigma"] is None and a["efg"] is None for a in calc["atoms"])


def test_xyz_trajectory_uses_the_last_frame(tmp_path):
    calc = D.read_xyz(write_xyz(tmp_path / "opt_trj.xyz", frames=3))
    assert calc["atoms"][0]["position"][0] == pytest.approx(2.0)
    assert any("3 frames" in n for n in calc["notes"])


def test_g_orientation_markers_are_removed_and_kept_as_axes(tmp_path):
    marks = [("He", 0.0, 0.0, 0.0), ("Ne", 0.0, 0.0, 0.7), ("Ar", 0.5, 0.0, 0.0),
             ("Kr", 0.0, -0.2, 0.0)]
    calc = D.read_xyz(write_xyz(tmp_path / "x.gori.xyz", ATOMS + marks,
                                comment=" g-Tensor orientation as XYZ file; He=chosen origin "
                                        "Ne=g1 Ar=g2 Kr=g3"))
    assert len(calc["atoms"]) == len(ATOMS)
    assert np.allclose(calc["g_axes"], [[0, 0, 1], [1, 0, 0], [0, -1, 0]])


def test_an_xyz_without_coordinates_is_refused(tmp_path):
    zero = [(s, 0.0, 0.0, 0.0) for s, *_ in ATOMS]
    with pytest.raises(ValueError, match="no structure"):
        D.read_xyz(write_xyz(tmp_path / "x.CAS_DTensor_HEFF_rotated.xyz", zero))


def test_xyz_with_a_bad_row_count_is_refused(tmp_path):
    p = tmp_path / "bad.xyz"
    p.write_text("5\ncomment\nN 0 0 0\nC 1 0 0\n")
    with pytest.raises(ValueError, match="says 5 atoms"):
        D.read_xyz(p)


# ---- orca_pnmr output and input ---------------------------------------------------------

def _pr_sites(calc, element="N"):
    return D.dft_sites(calc, element=element, paramagnetic={"model": "curie", "T": 298.0},
                       merge=False)


def test_orca_pnmr_output_is_a_calculation_on_its_structure(tmp_path):
    out = write_pnmr_output(tmp_path / "PrFeCN6_pnmr.txt")
    calc = D.read_calculation(out, structure=write_xyz(tmp_path / "geom.xyz"))
    assert calc["code"] == "ORCA (orca_pnmr)" and pm.is_paramagnetic(calc)
    assert calc["epr"]["S"] == 1.0 and np.allclose(calc["epr"]["D_cm"], PR_D)
    assert np.allclose(calc["epr"]["g"], PR_G)
    assert len(calc["atoms"]) == len(ATOMS) and pm.has_positions(calc)
    # nuclei are numbered from 0 in ORCA's atom order: 7C is atom 7
    assert calc["atoms"][7]["orca_label"] == "7C" and calc["atoms"][7]["label"] == "C1"
    assert np.allclose(calc["atoms"][1]["hfc"]["A_MHz"], PR_HFC[1][2])


def test_orca_pnmr_shifts_equal_what_orca_printed(tmp_path):
    out = write_pnmr_output(tmp_path / "PrFeCN6_pnmr.txt")
    calc = D.read_calculation(out, structure=write_xyz(tmp_path / "geom.xyz"))
    model = _pr_sites(calc)
    for rec in model["atoms"]:
        idx = 1 if rec["label"] == "N1" else 2
        sigma = np.array(PR_HFC[idx][3])
        assert rec["iso"] == pytest.approx(-np.trace(sigma) / 3.0, abs=0.01)
    check = next(n for n in calc["notes"] if n.startswith("Check:"))
    assert "3 nuclei" in check
    assert calc["orca_pnmr"][298.0]["7C"]["tensor"].shape == (3, 3)


def test_orca_pnmr_input_gives_the_same_shifts_as_the_output(tmp_path):
    geom = write_xyz(tmp_path / "geom.xyz")
    a = _pr_sites(D.read_calculation(write_pnmr_output(tmp_path / "o.txt"), structure=geom))
    b = _pr_sites(D.read_calculation(write_pnmr_input(tmp_path / "job.pnmr.inp"),
                                     structure=geom))
    assert [r["iso"] for r in a["atoms"]] == pytest.approx([r["iso"] for r in b["atoms"]],
                                                            abs=1e-6)


def test_input_without_d_and_a_doublet(tmp_path):
    hfc = {1: ("N", 38.567697, np.array(PR_HFC[1][2]))}
    calc = D.read_calculation(write_pnmr_input(tmp_path / "d.pnmr.inp", mult=2, Dcm=None,
                                               hfc=hfc), structure=write_xyz(tmp_path / "g.xyz"))
    assert calc["epr"]["S"] == 0.5 and calc["epr"]["D_cm"] is None


def test_a_structure_next_to_the_file_is_found(tmp_path):
    write_xyz(tmp_path / "PrFeCN6_hfc.xyz")
    out = write_pnmr_output(tmp_path / "PrFeCN6_pnmr.txt", inp="PrFeCN6_hfc.pnmr.inp")
    calc = D.read_calculation(out)
    assert calc["structure_path"].endswith("PrFeCN6_hfc.xyz") and pm.has_positions(calc)
    assert any("found next to the file" in n for n in calc["notes"])


def test_a_structure_of_another_molecule_next_to_it_is_not_used(tmp_path):
    other = [("Fe", 0, 0, 0)] + ATOMS[1:]
    write_xyz(tmp_path / "PrFeCN6_hfc.xyz", other[:3])        # too few atoms
    out = write_pnmr_output(tmp_path / "PrFeCN6_pnmr.txt", inp="PrFeCN6_hfc.pnmr.inp")
    calc = D.read_calculation(out)
    assert calc["structure_path"] is None and not pm.has_positions(calc)


def test_a_given_structure_that_does_not_match_is_refused(tmp_path):
    wrong = [("C", float(i), 0.0, 0.0) for i in range(9)]   # positions, wrong elements
    with pytest.raises(ValueError, match="does not belong"):
        D.read_calculation(write_pnmr_output(tmp_path / "o.txt"),
                           structure=write_xyz(tmp_path / "wrong.xyz", wrong))


def test_an_orca_output_can_be_the_structure(tmp_path):
    from test_paramagnetic_nmr import write_orca_epr
    orca_out = write_orca_epr(tmp_path / "job.out", [tuple(a) for a in ATOMS], {})
    calc = D.read_calculation(write_pnmr_output(tmp_path / "o.txt"), structure=orca_out)
    assert np.allclose(calc["atoms"][7]["position"], [3.65, 0, 0])
    assert all(a["sigma"] is None for a in calc["atoms"])


def test_without_a_structure_only_the_nuclei_and_no_picture(tmp_path):
    calc = D.read_calculation(write_pnmr_output(tmp_path / "lonely.txt"), structure=None)
    assert [a["orca_label"] for a in calc["atoms"]] == ["1N", "2N", "7C"]
    assert not pm.has_positions(calc)
    model = _pr_sites(calc)
    with pytest.raises(ValueError, match="no atomic positions"):
        DS.plot_dft_structure(calc, model["atoms"], frame="cartesian")
    # one site's own tensor picture needs no positions
    fig, _notes = D.plot_dft_tensors(model["atoms"][0], calc, frame="cartesian")
    assert fig is not None


def test_python_code_with_a_structure_runs(tmp_path):
    out = write_pnmr_output(tmp_path / "o.txt")
    geom = write_xyz(tmp_path / "geom.xyz")
    code = pm.paramagnetic_to_python(out, element="N", paramagnetic={"model": "curie",
                                                                       "T": 298.0},
                                     structure=geom)
    assert "structure=" in code
    scope = {}
    exec(code.replace("print(", "(lambda *a, **k: None)("), scope)
    assert scope["model"]["atoms"][0]["iso"] == pytest.approx(
        -np.trace(np.array(PR_HFC[1][3])) / 3.0, abs=0.01)


# ---- a relativistic (DKH) ORCA output ---------------------------------------------------

def test_dkh_hyperfine_blocks_are_found(tmp_path):
    from test_paramagnetic_nmr import write_orca_epr
    A = np.array(PR_HFC[1][2])
    path = tmp_path / "dkh.out"
    write_orca_epr(path, [tuple(a) for a in ATOMS], {1: ("N", 14, 1.0, 38.5677, A)},
                   g=PR_G, mult=3)
    # ORCA's DKH layout: three transformed matrices before the raw one.
    filler = []
    for order in ("0th", "1st", "2nd"):
        filler += [f" {order} order transformed DKH HFC:", " ------------------------------",
                   "      9.0   9.0   9.0", "      9.0   9.0   9.0", "      9.0   9.0   9.0", "",
                   "  A   9.0  9.0  9.0", " Orientation: ", "  X  1.0 0.0 0.0",
                   "  Y  0.0 1.0 0.0", "  Z  0.0 0.0 1.0", "", "Note: Tensor is right-handed",
                   ""] * 2
    text = path.read_text().replace(" Raw HFC matrix", "\n".join(filler) + "\n Raw HFC matrix")
    path.write_text(text)
    calc = D.read_calculation(str(path))
    assert np.allclose(calc["atoms"][1]["hfc"]["A_MHz"], A)


# ---- the paramagnetic parts on the structure --------------------------------------------

def test_contact_and_dipolar_glyphs_add_up_to_the_shift_tensor(tmp_path):
    calc = D.read_calculation(write_pnmr_output(tmp_path / "o.txt"),
                              structure=write_xyz(tmp_path / "geom.xyz"))
    rec = _pr_sites(calc)["atoms"][0]
    total = DS.tensor_of(rec, "para_contact") + DS.tensor_of(rec, "para_dipolar")
    assert np.allclose(total, DS.tensor_of(rec, "shift"), atol=1e-6)
    # the contact part is anisotropic for Pr (anisotropic g and Z)
    w = np.linalg.eigvalsh(DS.tensor_of(rec, "para_contact"))
    assert w.max() - w.min() > 1.0
    # its trace is three times the pseudocontact shift
    assert np.trace(DS.tensor_of(rec, "para_dipolar")) / 3 == pytest.approx(
        rec["para"]["pseudocontact"])


@pytest.mark.parametrize("kind", ["para_contact", "para_dipolar", "shift_aniso"])
def test_paramagnetic_structure_pictures_draw(tmp_path, kind):
    import matplotlib
    matplotlib.use("Agg")
    calc = D.read_calculation(write_pnmr_output(tmp_path / "o.txt"),
                              structure=write_xyz(tmp_path / "geom.xyz"))
    model = _pr_sites(calc)
    fig, notes = DS.plot_dft_structure(calc, model["atoms"], frame="cartesian", tensor=kind)
    assert "tensors of 2 site(s)" in notes[0]
    assert any(n.startswith("Scale:") for n in notes)


def test_a_diamagnetic_record_has_no_paramagnetic_glyph():
    assert DS.tensor_of({"shift_tensor": np.eye(3)}, "para_contact") is None


# ---- the user's files --------------------------------------------------------------------

@real
@pytest.mark.parametrize("name", ["orca_PrFeCN6_pnmr.txt", "orca_CeFeCN6_pnmr.txt",
                                  "orca_YbFeCN6_pnmr.txt", "orca_mn_cluster_pnmr.txt"])
def test_real_orca_pnmr_outputs_read_as_calculations(name):
    calc = D.read_calculation(os.path.join(DATA, name), structure=None)
    check = next(n for n in calc["notes"] if n.startswith("Check:"))
    worst = float(check.split(" to ")[1].split(" ppm")[0])
    assert worst < 0.02
