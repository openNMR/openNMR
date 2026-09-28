"""notebook.periodic_table: the NMR periodic table and Larmor calculator.

The nuclear data are openNMR's one gamma table (notebook.utils), so the
tests pin (a) that nothing is lost or duplicated joining it to the
elements, (b) the physics against independent literature values, and
(c) agreement with compute_BF1_SFO1, the function that references every
spectrum."""
import math

import matplotlib
import pytest

matplotlib.use("Agg")

from opennmr.notebook import periodic_table as pt
from opennmr.notebook.utils import _GAMMA_DF, compute_BF1_SFO1


def test_every_tabulated_nuclide_lands_on_exactly_one_element():
    table = set(_GAMMA_DF["Nuclide"])
    placed = [iso.nuclide for iso in pt.isotopes()]
    assert len(placed) == len(set(placed))
    assert set(placed) == table


def test_118_elements_in_a_valid_18_column_grid():
    els = pt.elements()
    assert [e.Z for e in els] == list(range(1, 119))
    positions = [pt.grid_position(e) for e in els]
    assert len(set(positions)) == 118                     # no two tiles overlap
    assert all(1 <= c <= 18 and r in (1, 2, 3, 4, 5, 6, 7, 9, 10) for r, c in positions)
    assert pt.grid_position("La") == (9, 4) and pt.grid_position("Lr") == (10, 18)
    assert pt.grid_position("Hf") == (6, 4) and pt.grid_position("Og") == (7, 18)


def test_isotopes_are_most_abundant_first():
    for el in pt.elements():
        ab = [i.abundance for i in el.isotopes]
        assert ab == sorted(ab, reverse=True), el.symbol
    assert pt.element("H").principal_isotope.nuclide == "1H"
    assert pt.element("Cu").principal_isotope.nuclide == "63Cu"
    assert not pt.element("Ar").has_nmr


@pytest.mark.parametrize("query,symbol", [
    ("Al", "Al"), ("al", "Al"), ("aluminium", "Al"), ("Aluminum", "Al"), (13, "Al"),
    ("13", "Al"), ("27Al", "Al"), ("Al-27", "Al"), ("al27", "Al"), ("²⁷Al", "Al"),
    ("caesium", "Cs"), ("cesium", "Cs"), ("deuterium", "H"),
])
def test_element_lookup_accepts_every_spelling(query, symbol):
    assert pt.element(query).symbol == symbol


def test_unknown_lookups_raise():
    with pytest.raises(KeyError):
        pt.element("Xx")
    with pytest.raises(KeyError):
        pt.element(0)
    with pytest.raises(KeyError):
        pt.isotope("12C")          # real nuclide, but I = 0
    with pytest.raises(KeyError):
        pt.isotope("banana")


def test_search_orders_exact_symbol_first_and_empty_matches_nothing():
    assert [e.symbol for e in pt.search("C")][0] == "C"
    assert {"C", "Ca", "Cl", "Cu"} <= {e.symbol for e in pt.search("c")}
    assert [e.symbol for e in pt.search("plat")] == ["Pt"]
    assert [e.symbol for e in pt.search("195Pt")] == ["Pt"]
    assert [e.symbol for e in pt.search("78")] == ["Pt"]
    assert pt.search("") == [] and pt.search("   ") == []


@pytest.mark.parametrize("spin,label", [(0.5, "1/2"), (1.0, "1"), (1.5, "3/2"), (4.5, "9/2"),
                                        (6.0, "6")])
def test_spin_labels(spin, label):
    assert pt.spin_label(spin) == label


def test_field_and_proton_frequency_round_trip():
    assert pt.field_to_proton_frequency(pt.proton_frequency_to_field(400.0)) == pytest.approx(400.0)
    # A "400 MHz" magnet is 9.3946 T (1H gamma/2pi = 42.577 MHz/T).
    assert pt.proton_frequency_to_field(400.0) == pytest.approx(9.39464, abs=1e-5)


@pytest.mark.parametrize("nuclide", ["1H", "13C", "27Al", "15N", "195Pt", "7Li", "17O"])
def test_larmor_frequency_matches_the_spectrum_referencing(nuclide):
    # Same table, same answer as compute_BF1_SFO1 (up to gamma's sign).
    bf1, _sfo1, _n = compute_BF1_SFO1(nuclide, 11.7434)
    assert pt.larmor_frequency(nuclide, 11.7434) == pytest.approx(abs(bf1), rel=1e-12)


def test_larmor_frequency_values_and_sign():
    # Independent check: 13C at a 400 MHz magnet is ~100.6 MHz; 27Al ~104.3.
    assert pt.larmor_frequency("13C", proton_MHz=400) == pytest.approx(100.60, abs=0.02)
    assert pt.larmor_frequency("27Al", proton_MHz=400) == pytest.approx(104.31, abs=0.02)
    assert pt.larmor_frequency("15N", 9.4, signed=True) < 0 < pt.larmor_frequency("15N", 9.4)
    with pytest.raises(ValueError):
        pt.larmor_frequency("1H")                       # no field
    with pytest.raises(ValueError):
        pt.larmor_frequency("1H", 9.4, proton_MHz=400)  # two fields
    with pytest.raises(ValueError):
        pt.larmor_frequency("1H", -1.0)


def test_larmor_working_shows_the_calculation():
    text = pt.larmor_working("27Al", 9.3947)
    assert "27Al" in text and "MHz/T" in text and "9.394700 T" in text
    assert f"{pt.larmor_frequency('27Al', 9.3947):.4f} MHz" in text


def test_receptivity_matches_the_literature():
    # Harris, Encyclopedia of NMR: D^C(1H) = 5.87e3, D^p(13C) = 1.70e-4,
    # D^C(19F) = 4.90e3, D^C(31P) = 391, D^C(27Al) = 1.22e3, D^C(15N) = 2.25e-2.
    assert pt.isotope("1H").receptivity_13C == pytest.approx(5.87e3, rel=0.01)
    assert pt.isotope("13C").receptivity_1H == pytest.approx(1.70e-4, rel=0.01)
    assert pt.isotope("19F").receptivity_13C == pytest.approx(4.90e3, rel=0.01)
    assert pt.isotope("31P").receptivity_13C == pytest.approx(391, rel=0.01)
    assert pt.isotope("27Al").receptivity_13C == pytest.approx(1.22e3, rel=0.01)
    assert pt.isotope("15N").receptivity_13C == pytest.approx(2.25e-2, rel=0.02)


def test_quadrupole_moment_only_for_quadrupolar_nuclei():
    assert pt.isotope("13C").quadrupole_fm2 is None
    assert pt.isotope("27Al").quadrupole_fm2 == pytest.approx(14.66)
    assert pt.isotope("2H").quadrupole_fm2 == pytest.approx(0.286)


def test_larmor_table_filters_and_sorts():
    df = pt.larmor_table(proton_MHz=600, spin=0.5)
    assert set(df["spin"]) == {0.5}
    assert list(df["larmor_MHz"]) == sorted(df["larmor_MHz"], reverse=True)
    assert df.attrs["proton_MHz"] == pytest.approx(600)
    al = pt.larmor_table(14.1, element="Al")
    assert list(al["nuclide"]) == ["27Al"]
    assert al["ratio_to_1H_pct"].iloc[0] == pytest.approx(26.0774, abs=1e-3)
    rich = pt.larmor_table(14.1, min_abundance=50, sort_by="receptivity")
    assert (rich["abundance_pct"] >= 50).all() and rich["nuclide"].iloc[0] == "1H"


def test_nearest_nuclides_finds_the_neighbour():
    near = pt.nearest_nuclides(pt.larmor_frequency("13C", proton_MHz=400), proton_MHz=400, n=3)
    assert near["nuclide"].iloc[0] == "13C"
    assert "79Br" in set(near["nuclide"])            # the classic 13C/79Br near-coincidence


@pytest.mark.parametrize("show", ["mass", "larmor", "receptivity", "isotope"])
def test_plot_periodic_table_draws(show):
    fig, ax = pt.plot_periodic_table(show=show, proton_MHz=400, highlight=["Al", "195Pt"],
                                     spin=0.5)
    texts = {t.get_text() for t in ax.texts}
    assert {"H", "Al", "Og", "Lr", "57–71"} <= texts
    if show == "larmor":
        assert f"{pt.larmor_frequency('27Al', proton_MHz=400):.1f}" in texts
    matplotlib.pyplot.close(fig)


def test_plot_larmor_needs_a_field():
    with pytest.raises(ValueError):
        pt.plot_periodic_table(show="larmor")
