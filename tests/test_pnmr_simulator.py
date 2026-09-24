"""CIF -> pNMR shift-table/spectrum engine (notebook/pnmr_simulator.py).

Two hand-checkable structures, written out as CIFs in each test's tmp dir:

* ``rocksalt``: Na+ on the rock-salt cation site of an Fm-3m cell, the anion
  site shared 50/50 by Cl-/Br-. Every Na has 6 anion neighbours at a/2 and
  12 Na at a/sqrt(2), so with Cl- = +10 ppm and Br- = -10 ppm the NN shift
  distribution is binomial(6, 1/2) on -60, -40, ..., +60 ppm.
* ``two_sodium_sites``: a P1 cell with Na1 (fully occupied, one Ni/Mn 50/50
  neighbour at 2.5 A) and Na2 (half occupied, one Fe neighbour at 2.5 A),
  for site selection and occupancy weighting.
"""
import json
import math
import os

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from opennmr.notebook import pnmr_simulator as pnmr  # noqa: E402
from opennmr.notebook.pnmr_fit import predict_environments  # noqa: E402

ROCKSALT_A = 5.64

ROCKSALT_CIF = f"""data_NaClBr
_cell_length_a {ROCKSALT_A}
_cell_length_b {ROCKSALT_A}
_cell_length_c {ROCKSALT_A}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225
loop_
_atom_type_symbol
_atom_type_oxidation_number
Na+ 1
Cl- -1
Br- -1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na1 Na+ 0 0 0 1.0
X1 Cl- 0.5 0.5 0.5 0.5
X1b Br- 0.5 0.5 0.5 0.5
"""

TWO_SITES_CIF = """data_two_sites
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na1 Na 0 0 0 1.0
Na2 Na 0.5 0.5 0.5 0.5
M1 Ni 0.25 0 0 0.5
M1b Mn 0.25 0 0 0.5
Fe1 Fe 0.5 0.5 0.75 1.0
"""

ROCKSALT_SHIFTS = {"NN": {"Cl-": 10.0, "Br-": -10.0}, "NNN": {"Na+": 0.0}}
ROCKSALT_SHELLS = [("NN", 0.0, 3.2), ("NNN", 3.6, 4.2)]


@pytest.fixture
def rocksalt(tmp_path):
    path = tmp_path / "rocksalt.cif"
    path.write_text(ROCKSALT_CIF)
    return str(path)


@pytest.fixture
def two_sites(tmp_path):
    path = tmp_path / "two_sites.cif"
    path.write_text(TWO_SITES_CIF)
    return str(path)


# --- describing the structure ------------------------------------------------

def test_labels_species_and_multiplicities(rocksalt):
    _, site_data = pnmr.read_cif_with_occupancies(rocksalt)
    assert pnmr.structure_species(site_data) == ["Na+", "Cl-", "Br-"]
    table = pnmr.site_label_table(site_data).set_index("Label")
    assert table.loc["Na1", "Multiplicity"] == 4          # 4 Na per conventional fcc cell
    shared = [lab for lab in table.index if lab != "Na1"]
    assert len(shared) == 1                              # one position, two CIF rows
    assert set(table.loc[shared[0], "Species"]) == {"Cl-", "Br-"}


@pytest.mark.parametrize("species, element, first", [
    ("Na+", "Na", "23Na"), ("Li", "Li", "7Li"), ("Mn3+", "Mn", "55Mn"), ("O2-", "O", "17O"),
])
def test_species_to_element_and_nuclide(species, element, first):
    assert pnmr.species_element(species) == element
    assert pnmr.candidate_nuclides(species)[0] == first


def test_vacancy_has_no_element():
    assert pnmr.species_element("Va") is None
    assert pnmr.candidate_nuclides("Va") == []


def test_neighbour_distances_are_the_rocksalt_ones(rocksalt):
    structure, site_data = pnmr.read_cif_with_occupancies(rocksalt, atom_of_interest="Na+")
    table = pnmr.neighbour_distance_table(structure, site_data, "Na+", max_r=4.2)
    per_site = table.groupby("Site")["Distance"]
    assert set(per_site.size()) == {18}                   # 6 anions + 12 Na, every Na
    d = np.sort(np.unique(table["Distance"].round(4)))
    assert d == pytest.approx([ROCKSALT_A / 2, ROCKSALT_A / math.sqrt(2)], abs=1e-4)
    # shell_distance_histogram is the same pairs, as a sorted array
    hist = pnmr.shell_distance_histogram(structure, site_data, "Na+", max_r=4.2)
    assert np.all(np.diff(hist) >= 0) and len(hist) == len(table)


def test_shell_composition_counts_occupancy_weighted_neighbours(rocksalt):
    structure, site_data = pnmr.read_cif_with_occupancies(rocksalt, atom_of_interest="Na+")
    table = pnmr.neighbour_distance_table(structure, site_data, "Na+", max_r=4.2)
    comp = pnmr.shell_composition(table, ROCKSALT_SHELLS)
    nn = comp[comp.Shell == "NN"].set_index("Species")
    assert nn.loc["Cl-", "Mean_count"] == pytest.approx(3.0)
    assert nn.loc["Br-", "Mean_count"] == pytest.approx(3.0)
    assert nn["Neighbours"].iloc[0] == pytest.approx(6.0)
    nnn = comp[comp.Shell == "NNN"].set_index("Species")
    assert list(nnn.index) == ["Na+"] and nnn.loc["Na+", "Mean_count"] == pytest.approx(12.0)


def test_ignoring_a_species_drops_its_whole_site(rocksalt):
    # Documented behaviour: a neighbour site is dropped if ANY occupant is
    # ignored, so ignoring Br- also removes the Cl- sharing its site.
    _, site_data = pnmr.read_cif_with_occupancies(rocksalt)
    assert pnmr.contributing_species(site_data) == ["Na+", "Cl-", "Br-"]
    assert pnmr.contributing_species(site_data, ignore_species=["Br-"]) == ["Na+"]
    assert pnmr.contributing_species(site_data, ignore_sites=["Na1"]) == ["Cl-", "Br-"]


# --- shells -------------------------------------------------------------------

@pytest.mark.parametrize("bins, message", [
    ([], "at least one"),
    ([("NN", 0.0, 3.0), ("NNN", 2.9, 4.0)], "overlap"),
    ([("NN", 0.0, 3.0), ("NN", 3.5, 4.0)], "Two shells"),
    ([("NN", 3.0, 3.0)], "d_min < d_max"),
    ([("p", 0.0, 3.0)], "cannot be a shell name"),
    ([("", 0.0, 3.0)], "needs a name"),
])
def test_bad_shells_are_refused(bins, message):
    with pytest.raises(ValueError, match=message):
        pnmr.validate_shell_bins(bins)


def test_touching_shells_are_fine():
    pnmr.validate_shell_bins([("NN", 0.0, 3.0), ("NNN", 3.0, 4.0)])


# --- the shift table and spectrum --------------------------------------------

def test_rocksalt_shifts_are_binomial(rocksalt, tmp_path):
    spec, table = pnmr.generate_spectrum_dict(
        rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS, nuclide="23Na", B0_tesla=11.7,
        broadening=1.0, xlims=(-100, 100), n_points=20001, output_dir=str(tmp_path / "out"))
    assert table["p"].sum() == pytest.approx(1.0)
    by_shift = table.groupby("Total_shift")["p"].sum()
    expected = {10.0 * (2 * k - 6): math.comb(6, k) / 64 for k in range(7)}
    assert dict(by_shift) == pytest.approx(expected)
    # Stick heights come through unscaled: the peaks are 20 ppm apart at
    # sigma = 1 ppm, so each maximum is its probability. (Until 2026-09-24
    # this was divided by the 4 Na sites a second time.)
    x, y = spec["x_ppm"], spec["real_spectrum"]
    for shift, p in expected.items():
        assert y[np.argmin(np.abs(x - shift))] == pytest.approx(p, rel=1e-3)
    assert spec["normalised_real_spectrum"].max() == pytest.approx(1.0)
    assert spec["BF1"] == pytest.approx(11.26952278792250 * 11.7)
    assert spec["nuclide"] == "23Na"


def test_saved_table_and_parameters(rocksalt, tmp_path):
    out = tmp_path / "chosen"
    spec, table = pnmr.generate_spectrum_dict(
        rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS, nuclide="23Na", B0_tesla=9.4,
        broadening=2.0, output_dir=str(out), ignore_sites=["Na1"])
    assert spec["path"] == str(out)
    saved = (out / "23Na_shift_table.csv")
    assert saved.exists()
    params = json.loads((out / "23Na_simulation_parameters.json").read_text())
    assert params["ignore_sites"] == ["Na1"] and params["B0_tesla"] == 9.4
    assert params["shell_bins"] == [list(b) for b in ROCKSALT_SHELLS]
    assert table.attrs["parameters"]["n_configurations"] == len(table)


def test_nothing_written_when_not_saving(rocksalt, tmp_path):
    out = tmp_path / "never"
    pnmr.generate_spectrum_dict(
        rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS, nuclide="23Na", B0_tesla=9.4,
        output_dir=str(out), save_shift_table=False)
    assert not out.exists()


@pytest.mark.parametrize("kwargs, message", [
    ({"broadening": 0.0}, "broadening"),
    ({"B0_tesla": 0.0}, "B0_tesla"),
    ({"xlims": (5, 5)}, "xlims"),
])
def test_bad_inputs_are_refused(rocksalt, kwargs, message):
    args = dict(nuclide="23Na", B0_tesla=11.7, save_shift_table=False)
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        pnmr.generate_spectrum_dict(rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS, **args)


# --- choosing sites, and weighting them --------------------------------------

TWO_SHELLS = [("NN", 0.0, 3.0)]
TWO_SHIFTS = {"NN": {"Ni": 100.0, "Mn": 200.0, "Fe": 300.0}}


def _p_by_shift(table):
    return dict(table.groupby("Total_shift")["p"].sum())


def test_sites_weighted_by_occupancy(two_sites):
    _, table = pnmr.generate_spectrum_dict(
        two_sites, "Na", TWO_SHELLS, TWO_SHIFTS, nuclide="23Na", B0_tesla=11.7,
        save_shift_table=False)
    # Na1 (occ 1) : Na2 (occ 0.5) = 2/3 : 1/3; Na1 splits 50/50 Ni/Mn.
    assert _p_by_shift(table) == pytest.approx({100.0: 1 / 3, 200.0: 1 / 3, 300.0: 1 / 3})


def test_equal_weights_on_request(two_sites):
    _, table = pnmr.generate_spectrum_dict(
        two_sites, "Na", TWO_SHELLS, TWO_SHIFTS, nuclide="23Na", B0_tesla=11.7,
        save_shift_table=False, weight_by_occupancy=False)
    assert _p_by_shift(table) == pytest.approx({100.0: 0.25, 200.0: 0.25, 300.0: 0.5})


def test_only_the_chosen_sites(two_sites):
    _, table = pnmr.generate_spectrum_dict(
        two_sites, "Na", TWO_SHELLS, TWO_SHIFTS, nuclide="23Na", B0_tesla=11.7,
        save_shift_table=False, sites_of_interest=["Na2"])
    assert _p_by_shift(table) == pytest.approx({300.0: 1.0})


def test_ignored_site_contributes_nothing(two_sites):
    _, table = pnmr.generate_spectrum_dict(
        two_sites, "Na", TWO_SHELLS, TWO_SHIFTS, nuclide="23Na", B0_tesla=11.7,
        save_shift_table=False, ignore_sites=["Fe1"])
    assert _p_by_shift(table) == pytest.approx({0.0: 1 / 3, 100.0: 1 / 3, 200.0: 1 / 3})


@pytest.mark.parametrize("sites, message", [(["Na9"], "No site labelled"),
                                            (["Fe1"], "contain no Na")])
def test_bad_site_choices_are_refused(two_sites, sites, message):
    with pytest.raises(ValueError, match=message):
        pnmr.generate_spectrum_dict(two_sites, "Na", TWO_SHELLS, TWO_SHIFTS, nuclide="23Na",
                                    B0_tesla=11.7, save_shift_table=False,
                                    sites_of_interest=sites)


def test_fit_predictions_match_the_simulation(two_sites):
    _, table = pnmr.generate_spectrum_dict(
        two_sites, "Na", TWO_SHELLS, TWO_SHIFTS, nuclide="23Na", B0_tesla=11.7,
        save_shift_table=False)
    predicted = predict_environments(two_sites, "Na", TWO_SHELLS, TWO_SHIFTS)
    assert _p_by_shift(predicted) == pytest.approx(_p_by_shift(table))


def test_equal_occupancy_sites_are_unchanged_by_weighting(rocksalt):
    common = dict(nuclide="23Na", B0_tesla=11.7, save_shift_table=False)
    _, a = pnmr.generate_spectrum_dict(rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS, **common)
    _, b = pnmr.generate_spectrum_dict(rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS,
                                       weight_by_occupancy=False, **common)
    assert np.allclose(a["p"], b["p"])


def test_suggested_shells_split_at_the_gaps(rocksalt):
    structure, site_data = pnmr.read_cif_with_occupancies(rocksalt, atom_of_interest="Na+")
    d = pnmr.shell_distance_histogram(structure, site_data, "Na+", max_r=6.0)
    bins = pnmr.suggest_shell_bins(d, max_r=6.0)
    nn, nnn = ROCKSALT_A / 2, ROCKSALT_A / math.sqrt(2)
    assert bins[0] == pytest.approx((0.0, round((nn + nnn) / 2, 3)))
    assert bins[1][0] == bins[0][1]                       # contiguous
    assert bins[1][0] < nnn < bins[1][1]
    assert all(hi <= 6.0 for _, hi in bins)
    assert pnmr.suggest_shell_bins(d, n_shells=1) == bins[:1]
    assert pnmr.suggest_shell_bins([]) == []


def test_peaks_outside_a_forced_axis_warn(rocksalt):
    with pytest.warns(pnmr.PnmrWarning, match="outside the specified xlims"):
        pnmr.generate_spectrum_dict(rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS,
                                    nuclide="23Na", B0_tesla=11.7, xlims=(-20, 20),
                                    save_shift_table=False)


# --- large runs: shared environments, fast broadening, progress, limits ------

def _reference_table(structure, site_data, aoi, shells, shifts):
    """The enumeration done the long way -- every site on its own, from
    its neighbours in the order pymatgen lists them -- to check the
    shared-environment shortcut against."""
    pnmr.mark_sites_of_interest(site_data, aoi)
    info = pnmr.get_neighbors_by_shell(structure, site_data, shells)
    names = [n for n, _, _ in shells]
    rows = []
    for i, s in enumerate(site_data):
        if not s.get(f"contains {aoi}"):
            continue
        joint = pnmr.combine_labeled_shells(
            {n: pnmr.local_environment_distribution(info[i][n]) for n in names})
        for env, p in joint.items():
            total = sum(shifts.get(sh, {}).get(sp, 0.0) for sh, comp in env for sp in comp)
            rows.append((i, round(total, 9), p))
    return sorted(rows)


@pytest.fixture
def rocksalt_supercell(tmp_path, rocksalt):
    structure, _ = pnmr.read_cif_with_occupancies(rocksalt)
    structure.make_supercell([2, 1, 1])
    path = tmp_path / "rocksalt_211.cif"
    structure.to(str(path))
    return str(path)


def test_equivalent_sites_share_one_enumeration(rocksalt_supercell):
    structure, site_data = pnmr.read_cif_with_occupancies(rocksalt_supercell, atom_of_interest="Na+")
    info = pnmr.get_neighbors_by_shell(structure, site_data, ROCKSALT_SHELLS)
    envs = pnmr.enumerate_local_environments(site_data, info, "Na+", ["NN", "NNN"])
    assert len(envs) == 8
    assert len({id(v) for v in envs.values()}) == 1          # one list, shared


def test_shared_enumeration_matches_the_long_way(rocksalt_supercell, two_sites):
    for cif, aoi, shells, shifts in ((rocksalt_supercell, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS),
                                     (two_sites, "Na", TWO_SHELLS, TWO_SHIFTS)):
        structure, site_data = pnmr.read_cif_with_occupancies(cif)
        _, table = pnmr.generate_spectrum_dict(cif, aoi, shells, shifts, nuclide="23Na",
                                               B0_tesla=11.7, save_shift_table=False,
                                               weight_by_occupancy=False)
        n_sites = table["Site"].nunique()
        got = sorted((int(r.Site), round(r.Total_shift, 9), r.p * n_sites)
                     for r in table.itertuples())
        want = _reference_table(structure, site_data, aoi, shells, shifts)
        assert [g[:2] for g in got] == [w[:2] for w in want]
        assert np.allclose([g[2] for g in got], [w[2] for w in want], rtol=1e-12)


def test_fast_broadening_equals_one_gaussian_per_row(rocksalt_supercell):
    _, table = pnmr.generate_spectrum_dict(rocksalt_supercell, "Na+", ROCKSALT_SHELLS,
                                           ROCKSALT_SHIFTS, nuclide="23Na", B0_tesla=11.7,
                                           save_shift_table=False)
    x, y = pnmr._broadened_spectrum(table, 3.0, 4001, None)
    slow = np.zeros_like(x)
    for s, w in zip(table["Total_shift"], table["p"]):
        slow += w * np.exp(-0.5 * ((x - s) / 3.0) ** 2)
    assert np.allclose(y, slow, rtol=1e-12, atol=1e-15)


def test_broadening_blocks_add_up(monkeypatch, two_sites):
    # More distinct shifts than one block: the blocks must tile exactly.
    import pandas as pd
    monkeypatch.setattr(pnmr, "_BROADEN_BLOCK", 7)
    rng = np.random.default_rng(1)
    table = pd.DataFrame({"Site": 0, "Total_shift": rng.uniform(-50, 50, 50),
                          "p": rng.dirichlet(np.ones(50))})
    x, y = pnmr._broadened_spectrum(table, 2.0, 2001, None)
    slow = sum(w * np.exp(-0.5 * ((x - s) / 2.0) ** 2) for s, w in zip(table.Total_shift, table.p))
    assert np.allclose(y, slow, rtol=1e-12)


def test_columns_come_in_shell_then_species_order(rocksalt):
    _, table = pnmr.generate_spectrum_dict(rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS,
                                           nuclide="23Na", B0_tesla=11.7, save_shift_table=False)
    assert list(table.columns) == ["Site", "Config", "NN_Br-", "NN_Cl-", "NNN_Na+",
                                   "Total_shift", "p"]


def test_estimate_matches_the_run(rocksalt_supercell, two_sites):
    for cif, aoi, shells, shifts, envs in (
            (rocksalt_supercell, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS, 1),
            (two_sites, "Na", TWO_SHELLS, TWO_SHIFTS, 2)):
        est = pnmr.estimate_configurations(cif, aoi, shells)
        _, table = pnmr.generate_spectrum_dict(cif, aoi, shells, shifts, nuclide="23Na",
                                               B0_tesla=11.7, save_shift_table=False)
        assert est["n_rows"] == len(table)
        assert est["n_sites"] == table["Site"].nunique()
        assert est["n_distinct_environments"] == envs
        assert est["n_columns"] == len(table.columns)


def test_too_many_configurations_is_refused_before_enumerating(rocksalt, tmp_path):
    calls = []
    with pytest.raises(pnmr.TooManyConfigurations) as err:
        pnmr.generate_spectrum_dict(rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS,
                                    nuclide="23Na", B0_tesla=11.7, output_dir=str(tmp_path / "o"),
                                    max_configurations=10,
                                    progress=lambda *a: calls.append(a[0]))
    assert err.value.estimate["n_rows"] == 4 * 7        # 4 sites x 7 NN compositions
    assert "Enumerating environments" not in calls
    assert not (tmp_path / "o").exists()
    pnmr.generate_spectrum_dict(rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS,
                                nuclide="23Na", B0_tesla=11.7, save_shift_table=False,
                                max_configurations=28)


def test_progress_reports_every_stage(rocksalt):
    seen = []
    pnmr.generate_spectrum_dict(rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS,
                                nuclide="23Na", B0_tesla=11.7, save_shift_table=False,
                                progress=lambda stage, done, total: seen.append((stage, done, total)))
    stages = list(dict.fromkeys(s for s, _, _ in seen))
    assert stages == ["Reading the CIF", "Enumerating environments",
                      "Building the shift table", "Assembling the table", "Broadening"]
    enum = [(d, t) for s, d, t in seen if s == "Enumerating environments"]
    assert enum == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_cancelling_writes_nothing(rocksalt, tmp_path):
    def cancel(stage, done, total):
        if stage == "Building the shift table":
            raise pnmr.EnumerationCancelled()
    with pytest.raises(pnmr.EnumerationCancelled):
        pnmr.generate_spectrum_dict(rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS,
                                    nuclide="23Na", B0_tesla=11.7,
                                    output_dir=str(tmp_path / "o"), progress=cancel)
    assert not (tmp_path / "o").exists()


def test_report_collects_notes_instead_of_printing(rocksalt, capsys, recwarn):
    notes = []
    pnmr.generate_spectrum_dict(rocksalt, "Na+", ROCKSALT_SHELLS, {"NN": {"Cl-": 10.0}},
                                nuclide="23Na", B0_tesla=11.7, save_shift_table=False,
                                xlims=(-5, 5), report=notes.append)
    assert any("'Br-'" in n for n in notes) and any("xlims" in n for n in notes)
    assert "Warning" not in capsys.readouterr().out
    assert not [w for w in recwarn if issubclass(w.category, pnmr.PnmrWarning)]


def test_unknown_nuclide_fails_before_any_work(rocksalt):
    seen = []
    with pytest.raises(KeyError):
        pnmr.generate_spectrum_dict(rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS,
                                    nuclide="99Xx", B0_tesla=11.7, save_shift_table=False,
                                    progress=lambda *a: seen.append(a))
    assert seen == []


# --- saving and loading the parameters ----------------------------------------

def _rerun(kwargs):
    return pnmr.generate_spectrum_dict(**kwargs, save_shift_table=False)


def test_parameters_round_trip(two_sites, tmp_path):
    out = tmp_path / "run1"
    spec, table = pnmr.generate_spectrum_dict(
        two_sites, "Na", TWO_SHELLS, TWO_SHIFTS, nuclide="23Na", B0_tesla=9.4, broadening=3.0,
        n_points=2048, xlims=(-100, 500), sites_of_interest=["Na1"], ignore_sites=["Fe1"],
        weight_by_occupancy=False, output_dir=str(out))
    saved = json.loads((out / "23Na_simulation_parameters.json").read_text())
    assert saved["format"] == pnmr.PARAMETERS_FORMAT and saved["version"] == 1
    assert saved["output_dir"] == str(out)
    kwargs = pnmr.load_simulation_parameters(str(out / "23Na_simulation_parameters.json"))
    assert kwargs["shell_bins"] == TWO_SHELLS and kwargs["broadening"] == 3.0
    assert "output_dir" not in kwargs and "save_shift_table" not in kwargs
    spec2, table2 = _rerun(kwargs)
    assert np.array_equal(spec2["real_spectrum"], spec["real_spectrum"])
    assert np.allclose(table2["p"], table["p"])


def test_save_settings_without_running(two_sites, tmp_path):
    path = tmp_path / "setup.json"
    pnmr.save_simulation_parameters(str(path), cif_filename=two_sites, atom_of_interest="Na",
                                    shell_bins=TWO_SHELLS, shift_dict=TWO_SHIFTS,
                                    nuclide="23Na", B0_tesla=11.7, broadening=1.5)
    kwargs = pnmr.load_simulation_parameters(str(path))
    assert kwargs["broadening"] == 1.5 and kwargs["cif_filename"] == two_sites


def test_moved_project_folder_finds_its_cif(tmp_path, two_sites):
    # Saved in <cif folder>/pnmr_simulation, then the whole folder moved.
    import shutil
    pnmr.generate_spectrum_dict(two_sites, "Na", TWO_SHELLS, TWO_SHIFTS, nuclide="23Na",
                                B0_tesla=11.7)
    moved = tmp_path.parent / (tmp_path.name + "_moved")
    shutil.copytree(tmp_path, moved)
    shutil.rmtree(tmp_path)
    kwargs = pnmr.load_simulation_parameters(
        str(moved / "pnmr_simulation" / "23Na_simulation_parameters.json"))
    assert kwargs["cif_filename"] == str(moved / "two_sites.cif")
    shutil.rmtree(moved)


def test_missing_cif_is_a_clear_error_and_can_be_overridden(two_sites, tmp_path):
    path = tmp_path / "p.json"
    pnmr.save_simulation_parameters(str(path), cif_filename="/nowhere/gone.cif",
                                    atom_of_interest="Na", shell_bins=TWO_SHELLS,
                                    shift_dict=TWO_SHIFTS, nuclide="23Na", B0_tesla=11.7)
    with pytest.raises(FileNotFoundError, match="gone.cif"):
        pnmr.load_simulation_parameters(str(path))
    assert pnmr.load_simulation_parameters(str(path), cif_filename=two_sites)["cif_filename"] == two_sites


@pytest.mark.parametrize("content, message", [
    ({"format": "something.else"}, "not pNMR simulation parameters"),
    ({"format": pnmr.PARAMETERS_FORMAT, "version": 99}, "newer openNMR"),
    ({"atom_of_interest": "Na"}, "missing"),
    ([1, 2], "not a pNMR parameters file"),
])
def test_not_a_parameters_file(tmp_path, content, message):
    path = tmp_path / "x.json"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match=message):
        pnmr.load_simulation_parameters(str(path))
