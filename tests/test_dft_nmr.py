"""DFT tensors into simulations (user request, 2026-10-06): CASTEP .magres
and VASP OUTCAR readers, referencing, CSA and EFG parameters, the CSA's
orientation in the EFG frame, site selection and merging, tensor pictures
in the cell frame -- notebook.dft_nmr.

Synthetic files with known answers always run; the user's own CASTEP and
VASP files (test_data/dft/, gitignored) are checked against each code's
own printed summary when present."""

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from opennmr.notebook import combined_hamiltonian as ch  # noqa: E402
from opennmr.notebook import dft_nmr as D  # noqa: E402
from opennmr.notebook.simulations import spin_operators  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "test_data", "dft")
HAVE_DATA = os.path.isdir(DATA) and os.path.exists(os.path.join(DATA, "sample.magres"))
real = pytest.mark.skipif(not HAVE_DATA, reason="test_data/dft not present")


def _rotation(seed):
    rng = np.random.default_rng(seed)
    q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
    return q if np.linalg.det(q) > 0 else -q


def _tensor(principal, R):
    return R @ np.diag(principal) @ R.T


def _fmt(t):
    return " ".join(f"{v:.16E}" for v in np.asarray(t).ravel())


def write_magres(path, atoms, lattice=((5.0, 0, 0), (0, 6.0, 0), (-1.0, 0, 7.0))):
    """atoms: [(species, index, position, sigma (3x3 ppm), efg (3x3 au) or None)]."""
    lines = ["#$magres-abinitio-v1.0", "[calculation]", "calc_code CASTEP", "[/calculation]",
             "[atoms]", "units lattice Angstrom"]
    if lattice is not None:
        lines.append("lattice " + _fmt(lattice))
    lines.append("units atom Angstrom")
    for sp, i, pos, _s, _e in atoms:
        lines.append(f"atom {sp} {sp} {i} " + " ".join(f"{v:.6f}" for v in pos))
    lines += ["[/atoms]", "[magres]", "units ms ppm"]
    for sp, i, _p, sigma, _e in atoms:
        lines.append(f"ms {sp} {i} " + _fmt(sigma))
    lines.append("units efg au")
    for sp, i, _p, _s, efg in atoms:
        if efg is not None:
            lines.append(f"efg {sp} {i} " + _fmt(efg))
    lines.append("[/magres]")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def write_outcar(path, species, counts, positions, valence_shift, efg_v_per_a2, g0, chi_pgv,
                 chi_core, core, lattice=((5.0, 0, 0), (0, 6.0, 0), (0, 0, 7.0))):
    """The blocks of an OUTCAR that read_outcar uses, in VASP 6's layout."""
    L = [" vasp.6.4.2 20Jul23 (build test) complex", ""]
    for sp in species:
        L.append(f"   VRHFIN ={sp}: test")
    L.append("   ions per type =    " + "   ".join(str(c) for c in counts))
    L.append("      direct lattice vectors                 reciprocal lattice vectors")
    for row in lattice:
        L.append("    " + "  ".join(f"{v:.9f}" for v in row) + "     0.1 0.0 0.0")
    L.append(" position of ions in cartesian coordinates  (Angst):")
    for p in positions:
        L.append("   " + "  ".join(f"{v:.8f}" for v in p))
    L += ["", " Electric field gradients (V/A^2)", "-" * 69,
          " ion       V_xx      V_yy      V_zz      V_xy      V_xz      V_yz", "-" * 69]
    for n, V in enumerate(efg_v_per_a2, 1):
        L.append(f"   {n}  " + "  ".join(f"{v:8.3f}" for v in
                                         (V[0, 0], V[1, 1], V[2, 2], V[0, 1], V[0, 2], V[1, 2])))
    L += ["-" * 69, "", "  Absolute Chemical Shift tensors", "  UNSYMMETRIZED TENSORS "]
    for n, T in enumerate(valence_shift, 1):
        L.append(f" ion    {n}")
        L += ["  " + "  ".join(f"{v:16.6f}" for v in row) for row in T]
    L.append("  SYMMETRIZED TENSORS ")
    for n, T in enumerate(valence_shift, 1):
        L.append(f" ion    {n}")
        L += ["  " + "  ".join(f"{v:16.6f}" for v in row) for row in T]
    L += ["", "  Approximate magnetic susceptibility, pGv (10^-6 cm^3/mole) "]
    L += [f"     {k + 1}  " + "  ".join(f"{v:16.6f}" for v in row) for k, row in enumerate(chi_pgv)]
    L += ["  G=0 CONTRIBUTION TO CHEMICAL SHIFT (field along BDIR) ",
          "  using pGv susceptibility, excluding core contribution",
          " " + "-" * 61, "   BDIR                X                 Y                 Z",
          " " + "-" * 61]
    L += [f"     {k + 1}   " + "  ".join(f"{v:16.6f}" for v in row) for k, row in enumerate(g0)]
    L += ["", "  Core NMR properties        ", "", "  typ  El   Core shift (ppm)",
          " ----------------------------"]
    L += [f"    {k + 1}  {sp}   {core[sp]:.7f}" for k, sp in enumerate(species)]
    L += [" ----------------------------", "",
          f"  Core contribution to magnetic susceptibility:    {chi_core:.2f}  10^-6 cm^3/mole"]
    path.write_text("\n".join(L) + "\n")
    return str(path)


SIG_NA1 = (540.0, 520.0, 490.0)          # shielding principal values
SIG_NA2 = (560.0, 555.0, 545.0)
EFG1 = (-0.10, -0.30, 0.40)              # au, traceless: η = (Vxx−Vyy)/Vzz = 0.5
EFG2 = (0.05, 0.15, -0.20)


@pytest.fixture
def files(tmp_path):
    R1, R2, R3 = _rotation(1), _rotation(2), _rotation(3)
    sample = write_magres(tmp_path / "sample.magres", [
        ("Na", 1, (0, 0, 0), _tensor(SIG_NA1, R1), _tensor(EFG1, R2)),
        ("Na", 2, (2.5, 3.0, 3.5), _tensor(SIG_NA2, R3), _tensor(EFG2, R1)),
        ("Na", 3, (0, 3.0, 0), _tensor(SIG_NA1, R1), _tensor(EFG1, R2)),   # equivalent to 1
        ("Cl", 1, (2.5, 0, 0), np.eye(3) * 950.0, np.zeros((3, 3))),
    ])
    ref = write_magres(tmp_path / "ref.magres", [
        ("Na", 1, (0, 0, 0), np.eye(3) * 550.0, np.zeros((3, 3))),
        ("Na", 2, (2.5, 2.5, 0), np.eye(3) * 550.2, np.zeros((3, 3))),
        ("Cl", 1, (2.5, 0, 0), np.eye(3) * 980.0, np.zeros((3, 3))),
    ])
    return {"sample": sample, "reference": ref, "R": (R1, R2, R3)}


# ---- reading and the numbers ---------------------------------------------------------

def test_magres_is_read_with_units_and_labels(files):
    calc = D.read_calculation(files["sample"])
    assert calc["code"] == "CASTEP" and calc["lattice"].shape == (3, 3)
    assert [a["label"] for a in calc["atoms"]] == ["Na1", "Na2", "Na3", "Cl1"]
    efg = calc["atoms"][0]["efg"]
    assert np.allclose(np.linalg.eigvalsh(efg), sorted(np.array(EFG1) * D.AU_EFG))
    assert D.species_of(calc) == ["Na", "Cl"]


def test_shift_csa_and_efg_parameters_have_their_textbook_values(files):
    model = D.dft_sites(files["sample"], files["reference"], element="Na", delta_ref=7.2,
                        merge=False)
    rec = model["atoms"][0]
    sigma_ref = (550.0 + 550.2) / 2
    deltas = sorted(7.2 + sigma_ref - s for s in SIG_NA1)[::-1]      # δ11 ≥ δ22 ≥ δ33
    assert rec["principal_hb"] == pytest.approx(deltas)
    iso = np.mean(deltas)
    assert rec["iso"] == pytest.approx(iso)
    assert rec["span"] == pytest.approx(deltas[0] - deltas[2])
    assert rec["skew"] == pytest.approx(3 * (deltas[1] - iso) / (deltas[0] - deltas[2]))
    # Haeberlen: δzz the furthest from δiso.
    dev = np.array(deltas) - iso
    zz = deltas[int(np.argmax(np.abs(dev)))]
    assert rec["aniso"] == pytest.approx(zz - iso) and rec["anisotropy"] == pytest.approx(
        1.5 * (zz - iso))
    # EFG: η = 0.5, C_Q = eQV_zz/h with ²³Na's Q (0.104 b).
    assert rec["eta_Q"] == pytest.approx(0.5)
    cq = D.E_OVER_H * 0.104e-28 * 0.40 * D.AU_EFG / 1e6
    assert rec["C_Q"] == pytest.approx(cq, rel=1e-9)
    assert cq == pytest.approx(0.40 * 0.104 * 234.9647, rel=1e-5)        # the usual factor
    assert model["reference"]["sigma_ref"] == pytest.approx(sigma_ref)
    assert any("σ_ref" in n for n in model["notes"])


def test_equivalent_sites_merge_into_one_with_their_count(files):
    model = D.dft_sites(files["sample"], files["reference"], element="Na", delta_ref=7.2)
    labels = [s.label for s in model["sites"]]
    assert labels == ["Na1, Na3", "Na2"] or labels == ["Na1, Na3", "Na2"][::1]
    assert [s.amount for s in model["sites"]] == [2.0, 1.0]
    assert model["site_dicts"][0]["dft"]["atoms"] == ["Na1", "Na3"]
    unmerged = D.dft_sites(files["sample"], files["reference"], element="Na", merge=False)
    assert len(unmerged["sites"]) == 3


def test_selection_by_label_and_the_errors(files):
    sample = D.read_calculation(files["sample"])
    model = D.dft_sites(sample, files["reference"], labels=["Na2"])
    assert [a["label"] for a in model["atoms"]] == ["Na2"]
    assert len(D.dft_sites(sample, files["reference"], labels="Na1, Na2")["atoms"]) == 2
    with pytest.raises(ValueError, match="one element"):
        D.dft_sites(sample, files["reference"], labels=["Na1", "Cl1"])
    with pytest.raises(ValueError, match="No site Na9"):
        D.dft_sites(sample, files["reference"], labels=["Na9"])
    with pytest.raises(ValueError, match="No K"):
        D.dft_sites(sample, files["reference"], element="K")
    with pytest.raises(ValueError, match="not an isotope"):
        D.dft_sites(sample, files["reference"], element="Na", nucleus="35Cl")
    with pytest.raises(ValueError, match="reference calculation"):
        D.dft_sites(sample, None, element="Na")
    typed = D.dft_sites(sample, None, element="Na", sigma_ref=550.0, delta_ref=0.0)
    assert typed["reference"]["source"] == "typed"
    assert any("δ_ref is 0" in n for n in typed["notes"])


def test_chlorine_uses_its_own_reference_and_isotope(files):
    model = D.dft_sites(files["sample"], files["reference"], element="Cl", delta_ref=-41.1)
    assert model["nucleus"] == "35Cl"
    assert model["atoms"][0]["iso"] == pytest.approx(-41.1 + 980.0 - 950.0)
    model37 = D.dft_sites(files["sample"], files["reference"], element="Cl", nucleus="37Cl")
    assert model37["nucleus"] == "37Cl"


def test_spin_half_sites_have_no_quadrupolar_coupling(tmp_path):
    path = write_magres(tmp_path / "c.magres", [
        ("C", 1, (0, 0, 0), _tensor((40.0, 60.0, 170.0), _rotation(4)), np.diag([0.1, 0.2, -0.3]))])
    model = D.dft_sites(path, None, sigma_ref=170.0, element="C")
    site = model["sites"][0]
    assert site.get("quadrupolar") == [] and site.get("csa")
    assert model["atoms"][0]["euler"] == (0.0, 0.0, 0.0)


def test_isotope_lists_put_the_most_receptive_first():
    assert D.nmr_isotopes("Cd")[0] == "113Cd"
    assert D.default_isotope("Li") == "7Li" and D.default_isotope("Cl") == "35Cl"
    assert D.quadrupole_moment_barn("35Cl") == pytest.approx(-0.08165)
    assert D.quadrupole_moment_barn("13C") is None


# ---- the orientation: single-crystal frequencies, computed two ways ---------------------

def _direct_ct_frequency(rec, b, nu0, spin=1.5):
    """The central-transition frequency for B0 along ``b`` (calculation
    frame) from the full tensors, by exact diagonalisation -- independent
    of the Euler angles and of the simulator's site frame."""
    Ix, Iy, Iz = spin_operators(spin)
    ops = np.stack([Ix, Iy, Iz])
    V = D.symmetric(rec["efg_tensor"])
    V = V - np.trace(V) / 3 * np.eye(3)
    K = rec["C_Q"] * 1e6 / (2 * spin * (2 * spin - 1))
    HQ = K * np.einsum("ij,ikl,jlm->km", V / rec["Vzz"], ops, ops)
    E = np.linalg.eigvalsh(HQ - nu0 * np.einsum("k,kij->ij", b, ops))
    j = int(round(spin - 0.5))
    dev = rec["shift_tensor"] - rec["iso"] * np.eye(3)
    return E[j + 1] - E[j] - nu0 + nu0 * 1e-6 * (rec["iso"] + b @ dev @ b)


def _check_orientation(model, nu0=105.84e6, n=40, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for rec, site in zip(model["atoms"], model["sites"]):
        for _ in range(n):
            b = rng.normal(size=3)
            b /= np.linalg.norm(b)
            bs = rec["site_axes"].T @ b
            th, ph = np.arccos(np.clip(bs[2], -1, 1)), np.arctan2(bs[1], bs[0])
            got = ch.site_transition_frequencies(site, nu0, th, ph, order="exact",
                                                 transitions="ct")["offset_Hz"].ravel()[0]
            worst = max(worst, abs(got - _direct_ct_frequency(rec, b, nu0)))
    return worst


def test_euler_angles_reproduce_every_single_crystal_frequency(files):
    """The strongest check there is on the orientation: with the CSA
    placed in the EFG frame by the reported Euler angles, the simulator's
    exact frequency for every direction of B0 equals the one computed from
    the two full tensors in the calculation's frame."""
    model = D.dft_sites(files["sample"], files["reference"], element="Na", merge=False)
    assert _check_orientation(model) < 1e-4                 # Hz, of ~10 kHz


def test_canonical_euler_angles_are_unique_and_small():
    rng = np.random.default_rng(5)
    for _ in range(20):
        Rc, Re = _rotation(rng.integers(1e6)), _rotation(rng.integers(1e6))
        a, b, g = D.relative_euler(Rc, Re)
        assert 0 <= b <= 90.0 + 1e-6
        # The same pair with its axes' signs flipped gives the same angles.
        flips = D._FLIPS[2]
        assert D.relative_euler(Rc @ flips, Re @ D._FLIPS[1]) == pytest.approx((a, b, g))
    assert D.relative_euler(np.eye(3), np.eye(3)) == (0.0, 0.0, 0.0)


def test_euler_round_trip_through_the_simulator_rotation():
    for seed in range(10):
        R = _rotation(seed)
        a, b, g = D.euler_zyz(R)
        assert np.allclose(ch._rotation_zyz(a, b, g), R, atol=1e-10)


# ---- VASP ----------------------------------------------------------------------------

def _vasp_files(tmp_path, include=True):
    R = _rotation(7)
    valence = [-_tensor((100.0, 120.0, 160.0), R), -_tensor((90.0, 90.0, 95.0), np.eye(3))]
    efg = [_tensor((-10.0, -30.0, 40.0), R), np.diag([1.0, 2.0, -3.0])]
    g0 = np.array([[-8.0, 0.1, 0.0], [0.1, -9.0, 0.0], [0.0, 0.0, -8.5]])
    chi = g0 / 0.08                                  # G=0 = k χ_pGv with k = 0.08
    sample = write_outcar(tmp_path / "OUTCAR", ["Na", "Cl"], [1, 1],
                          [(0, 0, 0), (2.5, 2.5, 2.5)], valence, efg, g0, chi, -10.0,
                          {"Na": -400.0, "Cl": -1000.0})
    return sample, valence, efg, g0


def test_outcar_total_shielding_is_minus_valence_g0_and_core(tmp_path):
    path, valence, efg, g0 = _vasp_files(tmp_path)
    calc = D.read_calculation(path)
    assert calc["code"] == "VASP" and [a["label"] for a in calc["atoms"]] == ["Na1", "Cl1"]
    k = np.trace(g0) / np.trace(g0 / 0.08)
    expected = -(valence[0] + g0 + k * (-10.0) * np.eye(3) + (-400.0) * np.eye(3))
    assert np.allclose(calc["atoms"][0]["sigma"], expected, atol=2e-6)
    assert np.allclose(calc["atoms"][0]["efg"], efg[0] * 1e20, atol=1e17)
    without = D.read_outcar(path, include_g0=False)
    assert np.allclose(without["atoms"][0]["sigma"], -(valence[0] - 400.0 * np.eye(3)),
                       atol=2e-6)
    assert any("G=0" in n for n in calc["notes"]) and any("G=0" in n for n in without["notes"])


def test_outcar_sites_and_orientation(tmp_path):
    path, *_ = _vasp_files(tmp_path)
    model = D.dft_sites(path, None, element="Na", sigma_ref=-400.0, delta_ref=0.0, merge=False)
    assert model["atoms"][0]["eta_Q"] == pytest.approx(0.5, abs=1e-4)
    assert _check_orientation(model) < 1e-3


def test_not_a_dft_file(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello\n")
    with pytest.raises(ValueError, match="not a calculation openNMR reads"):
        D.read_calculation(str(p))


# ---- simulation, tables, pictures -------------------------------------------------------

def test_simulate_dft_and_table(files):
    model = D.dft_sites(files["sample"], files["reference"], element="Na", delta_ref=7.2)
    spec = D.simulate_dft(model, B0_T=9.4, mas_Hz=12500, transitions="ct", n_points=2048)
    y = np.asarray(spec["real_spec"])
    assert y.size == 2048 and np.max(y) > 0
    text = D.format_table(model)
    assert "Ω (ppm)" in text and "Na1" in text and "C_Q (MHz)" in text
    csv = D.table_csv(model)
    assert csv.splitlines()[0].startswith("Site,δiso (ppm)")
    with pytest.raises(ValueError, match="Larmor"):
        D.simulate_dft(model)
    code = D.dft_to_python("a.magres", "b.magres", element="Na", delta_ref=7.2)
    assert "dft_nmr.dft_sites(sample, ref, element='Na', delta_ref=7.2)" in code


def test_tensor_pictures_in_the_cell_and_cartesian_frames(files):
    calc = D.read_calculation(files["sample"])
    model = D.dft_sites(calc, files["reference"], element="Na", delta_ref=7.2)
    rec = model["atoms"][0]
    fig, notes = D.plot_dft_tensors(rec, calc, frame="cell")
    ax = fig.axes[0]
    labels = {t.get_text() for t in ax.texts}
    assert {"a", "b", "c"} <= labels
    assert any("to a" in n and "to c" in n for n in notes)
    frame = D.display_frame(calc, rec, "cell")
    Q = D.standard_cell_rotation(calc["lattice"])
    assert np.allclose(Q @ calc["lattice"][0] / np.linalg.norm(calc["lattice"][0]), [1, 0, 0])
    assert abs((Q @ calc["lattice"][1])[2]) < 1e-12                  # b in the xy plane
    assert np.allclose(frame["rotation"], Q @ rec["site_axes"])
    fig2, _n = D.plot_dft_tensors(rec, calc, frame="cartesian", style="surface",
                                  larmor_Hz=105.8e6, relative_to="iso")
    assert "a" not in {t.get_text() for t in fig2.axes[0].texts}
    with pytest.raises(ValueError, match="frame"):
        D.display_frame(calc, rec, "lab")


def test_a_molecule_without_a_cell_uses_cartesian_axes(tmp_path):
    path = write_magres(tmp_path / "mol.magres", [
        ("C", 1, (0, 0, 0), _tensor((40.0, 60.0, 170.0), _rotation(9)), None)], lattice=None)
    calc = D.read_calculation(path)
    assert calc["lattice"] is None and any("non-periodic" in n for n in calc["notes"])
    model = D.dft_sites(calc, None, sigma_ref=170.0)
    with pytest.raises(ValueError, match="not periodic"):
        D.display_frame(calc, model["atoms"][0], "cell")
    assert D.axis_angles(calc, model["atoms"][0])[0].endswith("to z.")


# ---- the user's own files ----------------------------------------------------------------

@real
def test_castep_numbers_match_castep_s_own_summary():
    calc = D.read_calculation(os.path.join(DATA, "sample.magres"))
    castep_q = {"Na": 0.1040, "Fe": 0.16, "Cd": 1.0}
    for atom in calc["atoms"]:
        printed = calc["castep_summary"][atom["label"]]
        p = D.csa_parameters(D.symmetric(atom["sigma"]))
        assert p["iso"] == pytest.approx(printed["iso"], abs=1e-4)
        assert p["anisotropy"] == pytest.approx(printed["aniso"], abs=1e-4)
        if atom["species"] in castep_q and "Cq" in printed:
            e = D.efg_parameters(atom["efg"], castep_q[atom["species"]])
            assert e["C_Q"] == pytest.approx(printed["Cq"], abs=1e-4)
            assert e["eta_Q"] == pytest.approx(printed["eta"], abs=1e-4)


@real
def test_vasp_numbers_match_vasp_s_own_summary():
    for name in ("OUTCAR_sample", "OUTCAR_reference"):
        calc = D.read_calculation(os.path.join(DATA, name))
        for n, atom in enumerate(calc["atoms"], 1):
            row = calc["vasp_summary"][("total", n)]
            p = D.csa_parameters(-D.symmetric(atom["sigma"]))
            assert p["iso"] == pytest.approx(row[3], abs=1e-3)        # INCLUDING G=0
            assert p["span"] == pytest.approx(row[4], abs=1e-3)
            if row[4] > 1:
                assert p["skew"] == pytest.approx(row[5], abs=1e-3)


@real
def test_real_files_orientation_and_merge():
    for s, r in (("sample.magres", "ref_NaCl.magres"), ("OUTCAR_sample", "OUTCAR_reference")):
        model = D.dft_sites(os.path.join(DATA, s), os.path.join(DATA, r), element="Na",
                            delta_ref=7.21, merge=False)
        assert _check_orientation(model, n=20) < 1e-4             # Hz
    merged = D.dft_sites(os.path.join(DATA, "sample.magres"),
                         os.path.join(DATA, "ref_NaCl.magres"), element="Na", delta_ref=7.21)
    assert [s.amount for s in merged["sites"]] == [2.0, 2.0]


# ---- motional averaging and solution (2026-10-24) --------------------------------------

def _rz(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])


@pytest.fixture
def methyl(tmp_path):
    a, b, c = 24.0, 28.0, 36.0                       # shieldings (ppm) in each H's own frame
    atoms = [("H", i + 1, (np.cos(np.radians(120 * i)), np.sin(np.radians(120 * i)), 0.0),
              _tensor((a, b, c), _rz(120 * i)), None) for i in range(3)]
    atoms.append(("H", 4, (5.0, 5.0, 5.0), np.diag([20.0, 20.0, 20.0]), None))
    sample = write_magres(tmp_path / "ch3.magres", atoms, lattice=None)
    return {"sample": sample, "abc": (a, b, c)}


def test_methyl_rotation_gives_an_axial_average(methyl):
    a, b, c = methyl["abc"]
    free = D.dft_sites(methyl["sample"], None, element="H", labels=["H1", "H2", "H3"],
                       sigma_ref=30.0, merge=False)
    assert all(abs(r["eta"]) > 0.05 for r in free["atoms"])             # rigid: rhombic
    model = D.dft_sites(methyl["sample"], None, element="H", labels=["H1", "H2", "H3"],
                        sigma_ref=30.0, average=[["H1", "H2", "H3"]])
    assert len(model["atoms"]) == 1 and model["atoms"][0]["label"] == "H1–H3"
    rec = model["atoms"][0]
    assert rec["eta"] == pytest.approx(0.0, abs=1e-9)                   # about the C3 axis
    assert rec["iso"] == pytest.approx(30.0 - (a + b + c) / 3)
    # the unique axis is the one along z: the unique principal value is 30 - c
    uniq = 30.0 - c - rec["iso"]
    assert rec["aniso"] == pytest.approx(uniq)
    assert model["sites"][0].amount == 3.0
    assert model["site_dicts"][0]["dft"]["atoms"] == ["H1", "H2", "H3"]
    assert model["average"] == [["H1", "H2", "H3"]]
    assert any("averaged" in n for n in model["notes"])


def test_solution_keeps_only_the_isotropic_shift(methyl):
    model = D.dft_sites(methyl["sample"], None, element="H", labels=["H1", "H2", "H3", "H4"],
                        sigma_ref=30.0, state="solution", average=[["H1", "H2", "H3"]])
    assert [r["label"] for r in model["atoms"]] == ["H1–H3", "H4"]
    assert [r["iso"] for r in model["atoms"]] == pytest.approx([30.0 - 29.3333333, 10.0], abs=1e-5)
    for rec in model["atoms"]:
        assert rec["aniso"] is None and rec["C_Q"] is None
    for site in model["sites"]:
        assert not site.get("csa") and not site.get("quadrupolar")
    assert [s.amount for s in model["sites"]] == [3.0, 1.0]
    assert model["state"] == "solution"
    assert D.table_rows(model)                                          # the table still builds


def test_solution_without_averaging_is_just_isotropic(files):
    model = D.dft_sites(files["sample"], files["reference"], element="Na", delta_ref=7.2,
                        state="solution")
    assert all(not s.get("csa") and not s.get("quadrupolar") for s in model["sites"])
    solid = D.dft_sites(files["sample"], files["reference"], element="Na", delta_ref=7.2)
    assert [r["iso"] for r in model["atoms"]] == pytest.approx([r["iso"] for r in solid["atoms"]])


def test_averaging_errors(methyl):
    with pytest.raises(ValueError, match="at least two"):
        D.dft_sites(methyl["sample"], None, element="H", sigma_ref=30.0, average=[["H1"]])
    with pytest.raises(ValueError, match="two averaging groups"):
        D.dft_sites(methyl["sample"], None, element="H", sigma_ref=30.0,
                    average=[["H1", "H2"], ["H2", "H3"]])
    with pytest.raises(ValueError, match="not one of the chosen"):
        D.dft_sites(methyl["sample"], None, element="H", labels=["H1", "H2"], sigma_ref=30.0,
                    average=[["H1", "H3"]])
    with pytest.raises(ValueError, match="state"):
        D.dft_sites(methyl["sample"], None, element="H", sigma_ref=30.0, state="gas")
