"""Shielding → shift calibrations from several reference compounds (user
request, 2026-10-07) -- notebook.dft_calibration."""

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from opennmr.notebook import dft_calibration as C  # noqa: E402
from opennmr.notebook import dft_nmr as D  # noqa: E402

from test_dft_nmr import (EFG1, SIG_NA1, _rotation, _tensor, _vasp_files,  # noqa: E402
                          write_magres)


def _pt(sigma, delta, name="p", element="H", method=None, g0="none"):
    return C.typed_point(element, sigma, delta, name=name, method=method, g0=g0)


def test_fit_is_ordinary_least_squares_of_sigma_on_delta():
    rng = np.random.default_rng(3)
    x = np.array([0.8, 1.4, 2.2, 3.9, 5.1, 7.2, 8.0])
    y = 31.2 - 0.96 * x + rng.normal(0, 0.15, x.size)
    cal = C.fit_calibration([_pt(v, u, f"p{i}") for i, (u, v) in enumerate(zip(x, y))])
    (b, a), cov = np.polyfit(x, y, 1, cov=True)          # cov scaled by SSE/(n−2)
    assert cal["slope"] == pytest.approx(b) and cal["intercept"] == pytest.approx(a)
    assert cal["slope_err"] == pytest.approx(np.sqrt(cov[0, 0]))
    assert cal["intercept_err"] == pytest.approx(np.sqrt(cov[1, 1]))
    assert cal["free_slope"] and cal["n"] == 7 and cal["dof"] == 5
    r = np.corrcoef(x, y)[0, 1]
    assert cal["r2"] == pytest.approx(r ** 2)
    # residuals are δ_calc − δ_exp, δ_calc read off the line
    for p in cal["points"]:
        assert p["delta_calc"] == pytest.approx((p["sigma"] - a) / b)
        assert p["residual"] == pytest.approx(p["delta_calc"] - p["delta_exp"])
    assert C.shift_from_sigma(cal, a + b * 4.0) == pytest.approx(4.0)


def test_one_reference_fixes_the_slope_at_minus_one():
    cal = C.fit_calibration([_pt(550.0, 7.2, "NaCl", element="Na")])
    assert cal["slope"] == -1.0 and cal["intercept"] == pytest.approx(557.2)
    assert cal["intercept_err"] is None and cal["s_sigma"] is None
    assert C.shift_uncertainty(cal, 3.0) is None
    assert any("slope is fixed at −1" in n for n in cal["notes"])


def test_fixed_slope_fits_only_the_intercept():
    pts = [_pt(31.0 - 1.0 * d + e, d, f"p{d}") for d, e in ((1.0, 0.1), (4.0, -0.2), (7.0, 0.1))]
    cal = C.fit_calibration(pts, slope=-1.0)
    y_plus_x = [p["sigma"] + p["delta_exp"] for p in pts]
    assert cal["intercept"] == pytest.approx(np.mean(y_plus_x)) and not cal["free_slope"]
    s = np.std(y_plus_x, ddof=1)
    assert cal["s_sigma"] == pytest.approx(s) and cal["intercept_err"] == pytest.approx(s / np.sqrt(3))
    assert C.shift_uncertainty(cal, 50.0) == pytest.approx(s * np.sqrt(1 + 1 / 3))


def test_degenerate_references_are_explained():
    with pytest.raises(ValueError, match="same experimental shift"):
        C.fit_calibration([_pt(30.0, 2.0, "a"), _pt(31.0, 2.0, "b")])
    with pytest.raises(ValueError, match="No usable reference"):
        C.fit_calibration([_pt(30.0, None, "a")])
    with pytest.raises(ValueError, match="slope is zero"):
        C.fit_calibration([_pt(30.0, 1.0, "a"), _pt(30.0, 5.0, "b")])
    two = C.fit_calibration([_pt(30.0, 1.0, "a"), _pt(26.0, 5.0, "b")])
    assert two["slope"] == pytest.approx(-1.0) and two["slope_err"] is None
    assert any("exactly" in n for n in two["notes"])
    pos = C.fit_calibration([_pt(30.0, 1.0, "a"), _pt(34.0, 5.0, "b"), _pt(36.1, 7.0, "c")])
    assert any("slope is positive" in n for n in pos["notes"])
    with pytest.raises(ValueError, match="one element"):
        C.fit_calibration([_pt(30.0, 1.0, "a"), _pt(550.0, 7.0, "b", element="Na")])


def test_inverse_prediction_interval_has_its_stated_coverage():
    """95 % intervals from s_δ and Student's t cover the true shift of a
    new compound 95 % of the time."""
    from scipy import stats
    rng = np.random.default_rng(7)
    x = np.linspace(0.5, 9.0, 6)
    hits, trials = 0, 3000
    for _ in range(trials):
        y = 30.0 - 0.93 * x + rng.normal(0, 0.25, x.size)
        cal = C.fit_calibration([_pt(v, u, f"p{i}") for i, (u, v) in enumerate(zip(x, y))])
        d_true = rng.uniform(0, 10)
        sigma_new = 30.0 - 0.93 * d_true + rng.normal(0, 0.25)
        d = float(C.shift_from_sigma(cal, sigma_new))
        half = stats.t.ppf(0.975, cal["dof"]) * C.shift_uncertainty(cal, d)
        hits += abs(d - d_true) <= half
    assert 0.93 < hits / trials < 0.97


def test_tensors_and_the_old_single_reference_path_agree(tmp_path):
    sample = write_magres(tmp_path / "s.magres", [
        ("Na", 1, (0, 0, 0), _tensor(SIG_NA1, _rotation(1)), _tensor(EFG1, _rotation(2)))])
    ref = write_magres(tmp_path / "r.magres", [("Na", 1, (0, 0, 0), np.eye(3) * 550.0, None)])
    old = D.dft_sites(sample, ref, element="Na", delta_ref=7.2)
    cal = C.fit_calibration([C.reference_point(D.read_calculation(ref), element="Na",
                                               delta_exp=7.2)])
    new = D.dft_sites(sample, calibration=cal, element="Na")
    assert np.allclose(old["atoms"][0]["shift_tensor"], new["atoms"][0]["shift_tensor"])
    assert np.allclose(C.shift_from_sigma(cal, D.read_calculation(sample)["atoms"][0]["sigma"]),
                       new["atoms"][0]["shift_tensor"])
    assert new["reference"]["source"] == "calibration"
    assert not any("δ_ref is 0" in n for n in new["notes"])


def test_a_fitted_slope_scales_the_whole_tensor(tmp_path):
    sample = write_magres(tmp_path / "s.magres", [
        ("Na", 1, (0, 0, 0), _tensor(SIG_NA1, _rotation(1)), None)])
    pts = [_pt(560.0, -3.0, "a", "Na"), _pt(549.0, 7.2, "b", "Na"), _pt(530.0, 25.0, "c", "Na")]
    cal = C.fit_calibration(pts)
    rec = D.dft_sites(sample, calibration={"Na": cal}, element="Na")["atoms"][0]
    b = cal["slope"]
    assert rec["span"] == pytest.approx((SIG_NA1[0] - SIG_NA1[2]) / abs(b))
    assert rec["iso_err"] == pytest.approx(C.shift_uncertainty(cal, rec["iso"]))
    with pytest.raises(ValueError, match="No calibration for Na"):
        D.dft_sites(sample, calibration={"H": C.fit_calibration([_pt(30, 1, "x")])}, element="Na")
    with pytest.raises(ValueError, match="for H, not Na"):
        D.dft_sites(sample, calibration=C.fit_calibration([_pt(30, 1, "x")]), element="Na")


def test_points_groups_and_merging(tmp_path):
    R = _rotation(4)
    path = write_magres(tmp_path / "ala.magres", [
        ("H", 1, (0, 0, 0), _tensor((28.0, 30.0, 34.0), R), None),        # CH3
        ("H", 2, (1, 0, 0), _tensor((27.0, 30.5, 34.5), R), None),
        ("H", 3, (0, 1, 0), _tensor((28.5, 29.5, 33.0), R), None),
        ("H", 4, (5, 0, 0), _tensor((28.0, 30.0, 34.0), R.T), None),      # symmetry copy of H1
        ("C", 1, (0, 0, 1), np.eye(3) * 150.0, None)])
    calc = D.read_calculation(path)
    groups = C.equivalent_groups(calc, "H")
    assert sorted(map(sorted, groups)) == [["H1", "H4"], ["H2"], ["H3"]]
    pts = C.suggest_points(calc, "H")
    assert all(p["delta_exp"] is None for p in pts) and len(pts) == 3
    merged = C.merge_points(pts)
    assert sorted(merged["labels"]) == ["H1", "H2", "H3", "H4"]
    isos = [np.trace(a["sigma"]) / 3 for a in calc["atoms"][:4]]
    assert merged["sigma"] == pytest.approx(np.mean(isos))
    with pytest.raises(ValueError, match="one file"):
        C.merge_points([pts[0], dict(pts[1], file="other")])
    assert C.reference_point(calc, ["H2"])["name"] == "ala.magres H2"       # not "H2–H2"
    assert D._group_label(["H1", "H2", "H3"]) == "H1–H3"
    p = C.reference_point(calc, ["H1", "H2", "H3"], delta_exp=1.47, name="Ala CH3")
    assert p["sigma"] == pytest.approx(np.mean(isos[:3])) and p["element"] == "H"
    with pytest.raises(ValueError, match="one element"):
        C.reference_point(calc, ["H1", "C1"])


def test_method_and_g0_mismatches_are_reported(tmp_path):
    pbe = {"code": "CASTEP", "functional": "PBE", "cutoff": "800 eV", "pseudopotentials": {}}
    rscan = dict(pbe, functional="rSCAN")
    cal = C.fit_calibration([_pt(30.0, 1.0, "a", method=pbe), _pt(26.2, 5.0, "b", method=pbe),
                             _pt(23.0, 8.0, "c", method=rscan)])
    assert any("functional" in n for n in cal["notes"])
    mixed = C.fit_calibration([_pt(550.0, 7.2, "a", "Na", g0="included"),
                               _pt(560.0, -3.0, "b", "Na", g0="excluded")])
    assert any("G=0" in n for n in mixed["notes"])
    # sample checks: method, G=0 and extrapolation
    vasp, *_ = _vasp_files(tmp_path)
    sample = D.read_outcar(vasp, include_g0=False)
    cal_na = C.fit_calibration([_pt(550.0, 7.2, "a", "Na", method=pbe, g0="fixed"),
                                _pt(560.0, -3.0, "b", "Na", method=pbe, g0="fixed")])
    notes = C.sample_notes(cal_na, sample, [100.0])
    assert any("code (CASTEP vs VASP)" in n for n in notes)
    assert any("G=0 term is excluded" in n for n in notes)
    assert any("outside the references' range" in n for n in notes)


def test_library_round_trip(tmp_path):
    lib = str(tmp_path / "cal.json")
    cal = C.fit_calibration([_pt(30.0, 1.0, "a"), _pt(26.1, 5.0, "b"), _pt(23.2, 8.0, "c")])
    C.save_calibration(cal, "1H test", lib)
    C.save_calibration(cal, "1H test", lib)                     # replaces, does not duplicate
    cal_na = C.fit_calibration([_pt(550.0, 7.2, "NaCl", "Na")])
    C.save_calibration(cal_na, "23Na NaCl", lib)
    saved = C.load_library(lib)
    assert [c["name"] for c in saved] == ["23Na NaCl", "1H test"]
    back = C.find_calibration("1H test", "H", lib)
    assert back["slope"] == pytest.approx(cal["slope"]) and len(back["points"]) == 3
    assert C.shift_from_sigma(back, 27.0) == pytest.approx(C.shift_from_sigma(cal, 27.0))
    assert C.delete_calibration("1H test", "H", lib) and not C.delete_calibration("x", None, lib)
    assert [c["name"] for c in C.load_library(lib)] == ["23Na NaCl"]
    with pytest.raises(KeyError):
        C.find_calibration("1H test", path=lib)
    with pytest.raises(ValueError, match="name"):
        C.save_calibration(cal, "  ", lib)


def test_manual_scale_and_shift_table(tmp_path):
    cal = C.manual_calibration("Na", 550.0, 7.2, 1.0)
    assert C.shift_from_sigma(cal, 540.0) == pytest.approx(17.2)
    m = C.manual_calibration("Na", 550.0, 7.2, 0.9)
    assert C.shift_from_sigma(m, 540.0) == pytest.approx(7.2 + 0.9 * 10.0)
    sample = write_magres(tmp_path / "s.magres", [
        ("Na", 1, (0, 0, 0), np.eye(3) * 540.0, None), ("H", 1, (1, 0, 0), np.eye(3) * 29.0, None),
        ("Cl", 1, (2, 0, 0), np.eye(3) * 900.0, None)])
    rows = C.shift_table(sample, {"Na": cal, "H": C.fit_calibration([_pt(30.0, 1.0, "a")])})
    assert [(r["label"], round(r["delta_iso"], 6)) for r in rows] == [("Na1", 17.2), ("H1", 2.0)]


def test_plot_and_python(tmp_path):
    R = _rotation(5)
    paths = []
    for k, sig in enumerate((31.0, 27.1, 24.2)):
        paths.append(write_magres(tmp_path / f"r{k}.magres",
                                  [("H", 1, (0, 0, 0), _tensor((sig - 3, sig, sig + 3), R), None)]))
    pts = [C.reference_point(D.read_calculation(p), element="H", delta_exp=d, name=f"r{k}")
           for k, (p, d) in enumerate(zip(paths, (1.0, 5.0, 8.0)))]
    pts.append(C.typed_point("H", 20.0, None, name="unknown yet"))
    cal = C.fit_calibration(pts)
    fig, notes = C.plot_calibration(cal)
    ax = fig.axes[0]
    assert "sigma" in ax.get_ylabel() and len(fig.axes) == 2
    assert any(r"\mathit{\sigma}_\mathrm{calc} = " in t.get_text() for t in ax.texts)
    assert {t.get_text() for t in ax.texts} >= {"r0", "r1", "r2"}
    with pytest.raises(ValueError, match="typed"):
        C.plot_calibration(C.manual_calibration("H", 30, 0))
    code = C.calibration_to_python(pts, "H")
    ns = {}
    exec(code.replace("print(", "(lambda *a: None)("), ns)
    assert ns["cal"]["slope"] == pytest.approx(cal["slope"])
    os.remove(paths[0])                                      # a file gone: a typed point
    code = C.calibration_to_python(pts, "H")
    assert "dc.typed_point('H', " in code
    rows = C.points_table(cal)
    assert rows[0][0] == "r0" and rows[0][4] != "—"
