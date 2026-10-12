"""pNMR isotropic peaks -> CSA, Quad & J-coupling sites
(notebook/pnmr_export.py; explicit user request, 2026-10-12).

The rock-salt Na(Cl0.5Br0.5) cell of test_pnmr_simulator.py gives seven
environments at -60..+60 ppm with binomial(6, 1/2) weights, so every site
exported can be written down. A hand-built shift table checks grouping,
selection and the left-out share without pymatgen."""
import numpy as np
import pandas as pd
import pytest

from opennmr.notebook import combined_hamiltonian as ch
from opennmr.notebook import pnmr_export

BINOMIAL = np.array([1, 6, 15, 20, 15, 6, 1]) / 64.0
SHIFTS = np.array([-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0])


def _table():
    """Two configurations per binomial shift (split 3:1, 0.2 ppm apart), one
    tiny one far away; recorded parameters as generate_spectrum_dict's."""
    rows = []
    for s, p in zip(SHIFTS, BINOMIAL):
        rows.append({"Site": 0, "Config": len(rows), "NN_Cl-": 3, "NN_Br-": 3,
                     "Total_shift": s, "p": 0.75 * p * 0.999})
        rows.append({"Site": 0, "Config": len(rows), "NN_Cl-": 4, "NN_Br-": 2,
                     "Total_shift": s + 0.2, "p": 0.25 * p * 0.999})
    rows.append({"Site": 1, "Config": len(rows), "NN_Cl-": 6, "NN_Br-": 0,
                 "Total_shift": 300.0, "p": 0.001})
    table = pd.DataFrame(rows)
    table.attrs["parameters"] = {"nuclide": "23Na", "broadening_sigma_ppm": 3.0,
                                 "shift_dict": {"NN": {"Cl-": 10.0, "Br-": -10.0}}}
    return table


def test_table_groups_to_one_site_per_shift():
    sites, notes = pnmr_export.sites_from_pnmr_table(_table(), site_labels=["Na1", "Na2"],
                                                     return_notes=True)
    assert len(sites) == 7                                        # 300 ppm (0.1 %) left out
    assert [s["delta_iso"] for s in sites] == pytest.approx(SHIFTS[::-1] + 0.05, abs=1e-9)
    assert sum(s["amount"] for s in sites) == pytest.approx(0.999 / 1.0, rel=1e-3)
    assert sites[3]["amount"] == pytest.approx(20 / 64 * 0.999, rel=1e-6)
    assert all(s["nucleus"] == "23Na" and s["interactions"] == [] for s in sites)
    assert sites[0]["fwhm"] == pytest.approx(3.0 * 2 * np.sqrt(2 * np.log(2)))
    assert sites[0]["label"].startswith("Na1 ")
    joined = " ".join(notes)
    assert "left out" in joined and "0.1 % of the intensity" in joined
    assert "quadrupolar (I = 1.5)" in joined                      # 23Na
    # They are valid CSA, Quad & J-coupling sites, interactions can be added.
    built = [ch.Site.from_dict(d) for d in sites]
    built[0].add_quadrupolar(C_Q=2.0, eta_Q=0.1).add_csa(aniso=50.0, asym=0.2)
    assert built[0].get("quadrupolar") and built[0].get("csa")


def test_table_selected_rows_and_interactions_for_all():
    table = _table()
    rows = [6, 7]                                                 # the two 0 ppm configurations
    sites = pnmr_export.sites_from_pnmr_table(
        table, which_rows=rows, min_fraction=0.0,
        interactions=[{"type": "csa", "aniso": 100.0, "asym": 0.0, "convention": "Haeberlen",
                       "euler": (0, 0, 0)}])
    assert len(sites) == 1 and sites[0]["delta_iso"] == pytest.approx(0.05)
    assert sites[0]["amount"] == pytest.approx(1.0)               # of what was exported
    assert sites[0]["interactions"][0]["aniso"] == 100.0
    with pytest.raises(ValueError):
        pnmr_export.sites_from_pnmr_table(table, which_rows=[])


def test_table_tolerance_and_cap():
    fine = pnmr_export.sites_from_pnmr_table(_table(), tolerance=0.05, min_fraction=0.0)
    assert len(fine) == 15
    capped = pnmr_export.sites_from_pnmr_table(_table(), tolerance=0.05, min_fraction=0.0,
                                               max_sites=4)
    assert len(capped) == 4 and len({s["label"] for s in capped}) == 4


def _fit_dict():
    peaks = []
    for k, (s, p) in enumerate(zip(SHIFTS, BINOMIAL)):
        peaks.append({"type": "pseudo_voigt", "fitted_shift": s + 0.3, "fwhm": 6.5,
                      "eta": 0.4, "integral_pct": 100 * p,
                      "members": [{"site": "Na1", "label": f"env{k}", "fraction_in_group": 1.0,
                                   "p": p}]})
    return {"peaks": peaks}


def test_fit_peaks_become_sites():
    sites, notes = pnmr_export.sites_from_pnmr_fit(_fit_dict(), "23Na", return_notes=True)
    assert [s["delta_iso"] for s in sites] == pytest.approx(SHIFTS + 0.3)
    assert [s["amount"] for s in sites] == pytest.approx(BINOMIAL)
    assert all(s["lineshape"] == "pseudo_voigt" and s["eta"] == 0.4 and s["fwhm"] == 6.5
               for s in sites)
    assert sites[0]["label"] == "Na1 -59.7 ppm"
    joined = " ".join(notes)
    assert "fitted peak centre" in joined and "sidebands" in joined
    picked, notes = pnmr_export.sites_from_pnmr_fit(_fit_dict(), "23Na", which=[3, 0],
                                                    return_notes=True)
    assert [s["delta_iso"] for s in picked] == pytest.approx([-59.7, 0.3])
    assert any("2 of 7 peaks" in n for n in notes)
    with pytest.raises(ValueError, match="No peak"):
        pnmr_export.sites_from_pnmr_fit(_fit_dict(), "23Na", which=[9])
    with pytest.raises(ValueError, match="nucleus"):
        pnmr_export.sites_from_pnmr_fit(_fit_dict())


def test_spin_half_has_no_quadrupolar_note():
    _sites, notes = pnmr_export.sites_from_pnmr_fit(_fit_dict(), "31P", return_notes=True)
    assert not any("quadrupolar (I" in n for n in notes)
    _sites, notes = pnmr_export.sites_from_pnmr_fit(_fit_dict(), "7Li", return_notes=True)
    assert any("quadrupolar (I = 1.5)" in n for n in notes)


def test_python_code_rebuilds_the_sites():
    sites = pnmr_export.sites_from_pnmr_fit(_fit_dict(), "23Na", which=[0])
    code = pnmr_export.sites_python(sites)
    namespace = {}
    exec(code, namespace)                                         # noqa: S102 -- our own code
    assert namespace["sites"][0].delta_iso == pytest.approx(-59.7)


def test_real_rock_salt_fit_and_simulation(tmp_path):
    pytest.importorskip("pymatgen")
    from opennmr.notebook import pnmr_fit
    from opennmr.notebook import pnmr_simulator as pnmr
    from test_pnmr_simulator import ROCKSALT_CIF, ROCKSALT_SHELLS, ROCKSALT_SHIFTS
    cif = tmp_path / "rocksalt.cif"
    cif.write_text(ROCKSALT_CIF)
    spectrum, table = pnmr.generate_spectrum_dict(
        str(cif), "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS, nuclide="23Na", B0_tesla=9.4,
        broadening=3.0, n_points=4096, xlims=[-100, 100], save_shift_table=False,
        save_parameters=False)
    sim_sites = pnmr_export.sites_from_pnmr_table(table, min_fraction=0.0)
    assert sorted(s["delta_iso"] for s in sim_sites) == pytest.approx(SHIFTS, abs=1e-6)
    assert sorted(s["amount"] for s in sim_sites) == pytest.approx(sorted(BINOMIAL), abs=1e-6)
    fit = pnmr_fit.fit_predicted_spectrum(spectrum, str(cif), "Na+", ROCKSALT_SHELLS,
                                          ROCKSALT_SHIFTS, show_plot=False, show_report=False)
    fit_sites = pnmr_export.sites_from_pnmr_fit(fit, "23Na")
    assert sorted(s["delta_iso"] for s in fit_sites) == pytest.approx(SHIFTS, abs=0.05)
    assert sorted(s["amount"] for s in fit_sites) == pytest.approx(sorted(BINOMIAL), abs=0.01)
