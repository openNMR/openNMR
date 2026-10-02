"""Fit a spectrum against CIF-predicted pNMR environments (notebook/pnmr_fit.py),
and read back which environments make up the fit at any shift.

Uses the hand-checkable structures of test_pnmr_simulator.py:

* rock-salt Na(Cl0.5Br0.5): Cl- +10, Br- -10 ppm per NN, so seven peaks at
  -60..+60 ppm with binomial(6, 1/2) areas 1:6:15:20:15:6:1 / 64, each one
  environment of the single Na site (4 symmetry copies in the cell);
* two Na sites: Na1 (occ 1, one Ni/Mn 50:50 neighbour -> 100 / 200 ppm),
  Na2 (occ 0.5, one Fe neighbour -> 300 ppm).

The overlap read-out (environments_at_shift) is checked on a hand-built fit
whose peak heights and member fractions are chosen so the answer can be
written down.
"""
import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymatgen")

from opennmr.notebook import pnmr_fit  # noqa: E402
from opennmr.notebook import pnmr_simulator as pnmr  # noqa: E402

from test_pnmr_simulator import (  # noqa: E402
    ROCKSALT_CIF, ROCKSALT_SHELLS, ROCKSALT_SHIFTS, TWO_SITES_CIF, TWO_SHELLS, TWO_SHIFTS,
)

BINOMIAL = np.array([1, 6, 15, 20, 15, 6, 1]) / 64.0


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


def _simulated(cif, atom, shells, shifts, scale=1.0, **kwargs):
    spectrum, _ = pnmr.generate_spectrum_dict(
        cif, atom, shells, shifts, nuclide="23Na", B0_tesla=9.4, broadening=3.0,
        n_points=4096, xlims=[-100, 400] if atom == "Na" else [-100, 100],
        save_shift_table=False, save_parameters=False, **kwargs)
    spectrum = dict(spectrum)
    spectrum["real_spectrum"] = np.asarray(spectrum["real_spectrum"]) * scale
    return spectrum


@pytest.fixture
def rocksalt_fit(rocksalt):
    data = _simulated(rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS)
    return pnmr_fit.fit_predicted_spectrum(
        data, rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS,
        show_plot=False, show_report=False)


# ---------------------------------------------------------------------------
# The fit itself
# ---------------------------------------------------------------------------
def test_maximal_fit_recovers_the_binomial_areas(rocksalt_fit):
    out = rocksalt_fit
    assert list(out["kept"]["group_shift"]) == pytest.approx([-60, -40, -20, 0, 20, 40, 60])
    assert np.asarray(out["integrals_norm"]) / 100 == pytest.approx(BINOMIAL, rel=1e-6)
    assert out["tolerance"] == pytest.approx(10.0)


def test_fit_returns_its_window_data_model_and_components(rocksalt_fit):
    out = rocksalt_fit
    x, y, model = out["x_fit"], out["y_data"], out["y_model"]
    assert x.shape == y.shape == model.shape
    assert len(out["components"]) == len(out["kept"]) + 1     # + baseline
    assert np.sum(out["components"], axis=0) == pytest.approx(model)
    assert np.max(np.abs(y - model)) < 1e-6 * np.max(np.abs(y))


def test_data_is_compared_raw_whatever_the_spectrum_maximum(rocksalt):
    """fit_spectrum returns popt in raw units; the data it is shown and
    reported against must be raw too. With a maximum of 1 the normalised
    and raw copies coincide, which hid this: scale the spectrum by 1000."""
    data = _simulated(rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS, scale=1000.0)
    out = pnmr_fit.fit_predicted_spectrum(
        data, rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS,
        show_plot=False, show_report=False)
    assert out["y_data"] == pytest.approx(np.asarray(data["real_spectrum"])[out["mask"]])
    assert np.max(np.abs(out["y_data"] - out["y_model"])) < 1e-6 * np.max(out["y_data"])


def test_predicted_rows_and_members_carry_their_cif_site(two_sites):
    predicted = pnmr_fit.predict_environments(two_sites, "Na", TWO_SHELLS, TWO_SHIFTS)
    assert set(predicted["Site_label"]) == {"Na1", "Na2"}
    by_site = predicted.groupby("Site_label")["Total_shift"].apply(set).to_dict()
    assert by_site == {"Na1": {100.0, 200.0}, "Na2": {300.0}}
    groups, _ = pnmr_fit.group_environments(predicted, shift_dict=TWO_SHIFTS)
    sites = {round(g["group_shift"]): {m["site"] for m in g["members"]}
             for _, g in groups.iterrows()}
    assert sites == {100: {"Na1"}, 200: {"Na1"}, 300: {"Na2"}}


def test_fit_honours_sites_of_interest_and_occupancy_weighting(two_sites):
    data = _simulated(two_sites, "Na", TWO_SHELLS, TWO_SHIFTS)
    common = dict(show_plot=False, show_report=False, snr_threshold=0.0)
    only_na1 = pnmr_fit.fit_predicted_spectrum(
        data, two_sites, "Na", TWO_SHELLS, TWO_SHIFTS, sites_of_interest=["Na1"], **common)
    assert list(only_na1["kept"]["group_shift"]) == pytest.approx([100.0, 200.0])
    weighted = pnmr_fit.fit_predicted_spectrum(
        data, two_sites, "Na", TWO_SHELLS, TWO_SHIFTS, **common)
    flat = pnmr_fit.fit_predicted_spectrum(
        data, two_sites, "Na", TWO_SHELLS, TWO_SHIFTS, weight_by_occupancy=False, **common)
    # Na1 (occ 1) splits 1:1 over 100/200; Na2 (occ 0.5) is half as many nuclei.
    assert list(weighted["kept"]["group_p"]) == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert list(flat["kept"]["group_p"]) == pytest.approx([0.25, 0.25, 0.5])


# ---------------------------------------------------------------------------
# Collapsing symmetry copies
# ---------------------------------------------------------------------------
def test_collapse_members_merges_copies_and_is_idempotent():
    members = [
        {"site": "Na1", "label": "A", "shift": 10.0, "p": 0.1, "fraction_in_group": 0.25},
        {"site": "Na1", "label": "A", "shift": 10.0, "p": 0.1, "fraction_in_group": 0.25},
        {"site": "Na2", "label": "A", "shift": 12.0, "p": 0.1, "fraction_in_group": 0.25},
        {"site": "Na1", "label": "B", "shift": 11.0, "p": 0.1, "fraction_in_group": 0.25},
    ]
    once = pnmr_fit.collapse_members(members)
    assert [(m["site"], m["label"], m["n_copies"]) for m in once] == [
        ("Na1", "A", 2), ("Na2", "A", 1), ("Na1", "B", 1)]
    assert once[0]["fraction_in_group"] == pytest.approx(0.5)
    assert once[0]["p"] == pytest.approx(0.2)
    assert sum(m["fraction_in_group"] for m in once) == pytest.approx(1.0)
    twice = pnmr_fit.collapse_members(once)
    assert [(m["site"], m["label"], m["n_copies"], round(m["fraction_in_group"], 12))
            for m in twice] == [(m["site"], m["label"], m["n_copies"],
                                 round(m["fraction_in_group"], 12)) for m in once]


def test_the_rocksalt_peaks_each_collapse_to_one_environment(rocksalt_fit):
    peaks = pnmr_fit.summarise_peaks(rocksalt_fit)
    for peak, n_br in zip(peaks, range(6, -1, -1)):
        assert peak["n_members"] == 4                 # four copies of Na1 in the cell
        assert len(peak["members"]) == 1
        member = peak["members"][0]
        assert member["n_copies"] == 4 and member["site"] == "Na1"
        assert member["composition"]["NN"] == {"Cl-": 6 - n_br, "Br-": n_br}


# ---------------------------------------------------------------------------
# environments_at_shift, on a fit whose answer is written down
# ---------------------------------------------------------------------------
def _hand_fit():
    """Two unit Gaussians (FWHM 10) at 0 and 10 ppm, a flat zero baseline.
    Peak 0: env A (Na1) 75%, env B (Na2) 25%. Peak 1: env B (Na2) 40%,
    env C (Na1) 60%. At 5 ppm the two are equally tall."""
    x = np.linspace(-50, 60, 2201)
    sigma = 10 / (2 * math.sqrt(2 * math.log(2)))
    g0 = np.exp(-0.5 * (x / sigma) ** 2)
    g1 = np.exp(-0.5 * ((x - 10) / sigma) ** 2)
    peaks = [
        {"members": [{"site": "Na1", "label": "A", "shift": -1.0, "p": 0.3,
                      "fraction_in_group": 0.75},
                     {"site": "Na2", "label": "B", "shift": 1.0, "p": 0.1,
                      "fraction_in_group": 0.25}]},
        {"members": [{"site": "Na2", "label": "B", "shift": 9.0, "p": 0.2,
                      "fraction_in_group": 0.4},
                     {"site": "Na1", "label": "C", "shift": 11.0, "p": 0.3,
                      "fraction_in_group": 0.6}]},
    ]
    return {"x": x, "components": [g0, g1, np.zeros_like(x)], "peaks": peaks}


def test_on_an_isolated_peak_the_answer_is_its_own_breakdown():
    fit = _hand_fit()
    df = pnmr_fit.environments_at_shift(fit, -12.0, top_n=None)   # peak 1 below min_weight here
    assert list(df["environment"]) == ["A", "B"]
    assert list(df["probability"]) == pytest.approx([0.75, 0.25], abs=1e-3)
    assert set(df["peak"]) == {0}


def test_where_peaks_overlap_both_contribute_by_their_share_of_the_signal():
    fit = _hand_fit()
    df = pnmr_fit.environments_at_shift(fit, 5.0, top_n=None)
    # Equal heights at 5 ppm: w = (1/2, 1/2).
    expected = {"A": 0.5 * 0.75, "B": 0.5 * 0.25 + 0.5 * 0.4, "C": 0.5 * 0.6}
    got = dict(zip(df["environment"], df["probability"]))
    assert got == pytest.approx(expected, abs=1e-3)
    assert list(df["environment"]) == ["A", "B", "C"]
    assert df["probability"].sum() == pytest.approx(1.0)
    # B is reached through both peaks; it is reported through the one that
    # gives it more (peak 1: 0.2 against 0.125).
    row_b = df.set_index("environment").loc["B"]
    assert row_b["peak"] == 1 and row_b["fraction_of_peak"] == pytest.approx(0.4)
    assert df.attrs["peak_weights"] == pytest.approx([0.5, 0.5], abs=1e-3)


def test_top_n_and_the_text_form():
    fit = _hand_fit()
    everything = pnmr_fit.environments_at_shift(fit, 5.0, top_n=None)
    top = pnmr_fit.environments_at_shift(fit, 5.0, top_n=2)
    assert list(top["environment"]) == list(everything["environment"])[:2]
    text = pnmr_fit.format_environments(top, total=len(everything))
    lines = text.splitlines()
    assert lines[0] == "δ = 5.0 ppm"
    assert "A" in lines[1] and "Na1" in lines[1] and "37.5%" in lines[1]
    assert "peak 1" in lines[1]
    assert lines[-1].startswith("… and 1 ")


def test_on_baseline_or_outside_the_window_nothing_is_named():
    fit = _hand_fit()
    for shift in (-45.0, 55.0, 500.0, -500.0):
        df = pnmr_fit.environments_at_shift(fit, shift)
        assert df.empty, shift
        assert pnmr_fit.format_environments(df) == "No fitted peak here."
        assert pnmr_fit.peak_index_at_shift(fit, shift) is None


def test_peak_index_at_shift_is_the_dominant_peak():
    fit = _hand_fit()
    assert pnmr_fit.peak_index_at_shift(fit, 2.0) == 0
    assert pnmr_fit.peak_index_at_shift(fit, 8.0) == 1


def test_member_lists_must_match_the_components():
    fit = _hand_fit()
    fit["peaks"] = fit["peaks"][:1]
    with pytest.raises(ValueError, match="same fit"):
        pnmr_fit.environments_at_shift(fit, 0.0)


def test_read_out_on_a_real_fit(rocksalt_fit):
    df = pnmr_fit.environments_at_shift(rocksalt_fit, -40.0)
    assert df.iloc[0]["environment"] == "NN:5Br-+1Cl- NNN:12Na+"
    assert df.iloc[0]["site"] == "Na1"
    assert df.iloc[0]["probability"] == pytest.approx(1.0, abs=1e-6)
    assert pnmr_fit.peak_index_at_shift(rocksalt_fit, 19.0) == 4


# ---------------------------------------------------------------------------
# summarise_peaks, refit, sticks
# ---------------------------------------------------------------------------
def test_summarise_peaks(rocksalt_fit):
    peaks = pnmr_fit.summarise_peaks(rocksalt_fit)
    assert [p["index"] for p in peaks] == list(range(7))
    assert all(p["shift_fixed"] and p["shift_err"] is None for p in peaks)
    assert [p["fitted_shift"] for p in peaks] == pytest.approx([-60, -40, -20, 0, 20, 40, 60])
    assert sum(p["integral_pct"] for p in peaks) == pytest.approx(100.0)
    fwhm = 3.0 * 2 * math.sqrt(2 * math.log(2))      # the simulation's sigma = 3 ppm
    assert [p["fwhm"] for p in peaks] == pytest.approx([fwhm] * 7, rel=1e-4)
    assert peaks[0]["eta"] is None and peaks[0]["type"] == "gaussian"


def test_refit_returns_what_the_read_out_needs(rocksalt, rocksalt_fit):
    out = rocksalt_fit
    merged, _ = pnmr_fit.merge_groups_to_picks(out["kept"], [-50.0, 0.0, 50.0],
                                               integrals=out["integrals"])
    data = _simulated(rocksalt, "Na+", ROCKSALT_SHELLS, ROCKSALT_SHIFTS)
    refit = pnmr_fit.refit_merged_groups(merged, data, shift_float_window=10.0,
                                         show_plot=False, show_report=False)
    for key in ("x_fit", "y_data", "y_model", "components"):
        assert key in refit
    peaks = pnmr_fit.summarise_peaks(refit)
    assert len(peaks) == 3 and not any(p["shift_fixed"] for p in peaks)
    # The middle pick takes -20, 0 and +20 (each nearer 0 than +-50): 15:20:15.
    middle = {m["label"]: m["fraction_in_group"] for m in peaks[1]["members"]}
    assert sorted(middle.values()) == pytest.approx([15 / 50, 15 / 50, 20 / 50])
    df = pnmr_fit.environments_at_shift(refit, refit["x_fit"][np.argmax(refit["components"][1])])
    assert df.iloc[0]["environment"] == "NN:3Br-+3Cl- NNN:12Na+"


def test_predicted_sticks(rocksalt_fit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sticks = pnmr_fit.add_predicted_sticks(ax, rocksalt_fit, height=0.1)
    segments = sticks.get_segments()
    assert len(segments) == 7
    xs = sorted(s[0][0] for s in segments)
    tops = {round(s[0][0]): s[1][1] for s in segments}
    assert xs == pytest.approx([-60, -40, -20, 0, 20, 40, 60])
    assert tops[0] == pytest.approx(0.1) and tops[60] == pytest.approx(0.1 / 20)
    plt.close(fig)


def test_new_functions_are_exported():
    import opennmr.notebook as nmr
    for name in ("collapse_members", "peak_weights_at_shift", "peak_index_at_shift",
                 "environments_at_shift", "format_environments", "summarise_peaks",
                 "add_predicted_sticks"):
        assert getattr(nmr, name) is getattr(pnmr_fit, name)
