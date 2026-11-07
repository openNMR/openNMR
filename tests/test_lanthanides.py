"""Lanthanide series (user request, 2026-11-06): Bleaney's C_J and
Golding–Halton's ⟨S_z⟩ checked against the free-ion ground multiplet
(Hund's rules, Landé g_J, Stevens' α_J), and the Reilley separation of
contact and pseudocontact shifts."""

import pytest

from opennmr.notebook import lanthanides as L


@pytest.mark.parametrize("ln,L_,S,J,g", [("Ce", 3, 0.5, 2.5, 6 / 7), ("Pr", 5, 1.0, 4.0, 0.8),
                                         ("Gd", 0, 3.5, 3.5, 2.0), ("Tb", 3, 3.0, 6.0, 1.5),
                                         ("Dy", 5, 2.5, 7.5, 4 / 3), ("Er", 6, 1.5, 7.5, 1.2),
                                         ("Yb", 3, 0.5, 3.5, 8 / 7), ("Eu", 3, 3.0, 0.0, 0.0)])
def test_ground_multiplets_follow_hunds_rules(ln, L_, S, J, g):
    s = L.ground_state(ln)
    assert (s["L"], s["S"], s["J"]) == (L_, S, J)
    assert s["g_J"] == pytest.approx(g)


@pytest.mark.parametrize("ln", ["Ce", "Pr", "Nd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb"])
def test_tables_agree_with_the_ground_multiplet(ln):
    """Where the ground J dominates at 300 K the tabulated values are the
    ground-multiplet ones to within the excited-state admixture (≤ 12 %)."""
    assert L.BLEANEY_CJ[ln] == pytest.approx(L.bleaney_ground_state(ln), rel=0.12)
    assert L.GOLDING_HALTON_SZ[ln] == pytest.approx(L.sz_ground_state(ln), rel=0.12)


def test_reference_points_of_the_tables():
    assert L.bleaney_ground_state("Dy") == pytest.approx(-100.0)
    assert L.BLEANEY_CJ["Gd"] == 0.0 and L.bleaney_ground_state("Gd") == 0.0
    assert L.sz_ground_state("Gd") == pytest.approx(31.5)
    assert L.bleaney_ground_state("Dy", normalise=False) == pytest.approx(-181.33, abs=0.01)
    with pytest.raises(ValueError):
        L.ground_state("Fe")


def _series(F, G, members, dia=4.0):
    return {ln: dia + F * L.GOLDING_HALTON_SZ[ln] + G * L.BLEANEY_CJ[ln] for ln in members}


SERIES = ["Ce", "Pr", "Nd", "Sm", "Eu", "Tb", "Dy", "Ho", "Er", "Tm", "Yb"]


def test_reilley_recovers_f_and_g_all_three_ways():
    res = L.reilley_analysis(_series(2.5, -0.3, SERIES), delta_dia=4.0)
    f = res["fits"][0]
    assert f["direct"]["F"] == pytest.approx(2.5) and f["direct"]["G"] == pytest.approx(-0.3)
    assert f["sz_form"]["F"] == pytest.approx(2.5) and f["sz_form"]["G"] == pytest.approx(-0.3)
    assert f["cj_form"]["F"] == pytest.approx(2.5) and f["cj_form"]["G"] == pytest.approx(-0.3)
    # Sm (⟨S_z⟩ ≈ 0) is kept out of the δ/⟨S_z⟩ form only
    assert "Sm" not in f["sz_form"]["Ln"] and "Sm" in f["cj_form"]["Ln"]
    row = next(r for r in res["rows"] if r["Ln"] == "Dy")
    assert row["contact"] + row["pseudocontact"] == pytest.approx(row["delta_para"])
    assert abs(row["residual"]) < 1e-9


def test_a_break_in_the_series_is_found_by_separate_fits():
    light, heavy = SERIES[:5], SERIES[5:]
    shifts = {**_series(2.5, -0.3, light), **_series(1.0, -0.6, heavy)}
    one = L.reilley_analysis(shifts, delta_dia=4.0)
    assert one["fits"][0]["direct"]["rms"] > 1.0
    two = L.reilley_analysis(shifts, delta_dia=4.0, groups=[light, heavy])
    assert two["fits"][0]["direct"]["F"] == pytest.approx(2.5)
    assert two["fits"][1]["direct"]["G"] == pytest.approx(-0.6)


def test_exclusions_constants_and_errors():
    shifts = _series(1.0, 0.2, SERIES)
    shifts["Ho"] += 30.0
    res = L.reilley_analysis(shifts, delta_dia=4.0, exclude=("Ho",))
    assert res["fits"][0]["direct"]["F"] == pytest.approx(1.0)
    ho = next(r for r in res["rows"] if r["Ln"] == "Ho")
    assert ho["residual"] == pytest.approx(30.0)
    res2 = L.reilley_analysis({"Dy": 10.0, "Tb": 8.0, "Er": -3.0},
                              constants={"Dy": (-100.0, 28.55)})
    assert res2["constants"]["Dy"]["C_J"] == -100.0
    with pytest.raises(ValueError, match="La"):
        L.reilley_analysis({"La": 1.0, "Dy": 2.0})


def test_plot_and_python():
    import matplotlib
    matplotlib.use("Agg")
    res = L.reilley_analysis(_series(2.5, -0.3, SERIES), delta_dia=4.0, exclude=("Eu",))
    fig = L.plot_reilley(res)
    assert len(fig.axes) == 2
    assert "S_z" in fig.axes[0].get_xlabel() and "C_J" in fig.axes[1].get_ylabel()
    code = L.lanthanide_to_python(_series(2.5, -0.3, SERIES), 4.0)
    ns = {}
    exec(code.replace("fig = ln.plot_reilley(res)\n", ""), ns)
    assert ns["res"]["fits"][0]["direct"]["F"] == pytest.approx(2.5)
