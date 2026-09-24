"""J-coupling simulation (notebook/simulations.py) and fitting
(notebook/jcoupling_fitting.py).

Expected values are written out by hand: Pascal's triangle for spin-1/2
partners, the 1:1:1 / 1:2:3:2:1 patterns for spin-1, the binomial
satellite weights for a low-abundance partner, and fits against exact,
analytically generated multiplets on realistic (2^n-point, descending)
axes.
"""
import numpy as np
import pytest

from opennmr.notebook import simulations as S
from opennmr.notebook import jcoupling_fitting as JF
from opennmr.notebook.utils import find_isotope_info

L_1H = 400.13e6


def _lor(x, c, fwhm):
    g = fwhm / 2
    return g * g / ((x - c) ** 2 + g * g)


# --- stick patterns -----------------------------------------------------------

@pytest.mark.parametrize("n, weights", [
    (1, [1, 1]), (2, [1, 2, 1]), (3, [1, 3, 3, 1]), (4, [1, 4, 6, 4, 1])])
def test_spin_half_patterns_are_pascal(n, weights):
    f, w = S.spin_multiplet_pattern(n, 0.5, 7.0)
    assert np.allclose(w, np.array(weights) / sum(weights))
    assert np.allclose(np.diff(f), 7.0) and f[0] == pytest.approx(-n * 3.5)


@pytest.mark.parametrize("n, weights", [(1, [1, 1, 1]), (2, [1, 2, 3, 2, 1])])
def test_spin_one_patterns(n, weights):
    f, w = S.spin_multiplet_pattern(n, 1.0, 2.0)
    assert np.allclose(w, np.array(weights) / sum(weights))
    assert np.allclose(np.diff(f), 2.0)


def test_low_abundance_partner_gives_satellites():
    p = find_isotope_info("13C")["abundance"] / 100
    f, w = S.jcoupling_group_pattern(1, 140.0, nuclide="13C")
    assert list(f) == [-70.0, 0.0, 70.0]
    assert w == pytest.approx([p / 2, 1 - p, p / 2])


def test_groups_convolve():
    f, w = S.combine_jcouplings([
        {"nuclide": "1H", "n": 1, "J_Hz": 10.0, "abundance": 100.0},
        {"nuclide": "1H", "n": 2, "J_Hz": 3.0, "abundance": 100.0}])
    # a doublet of triplets: 6 lines, 1:2:1 either side of +/-5 Hz
    assert np.allclose(f, [-8, -5, -2, 2, 5, 8])
    assert np.allclose(w, np.array([1, 2, 1, 1, 2, 1]) / 8)


# --- broadening ---------------------------------------------------------------

@pytest.mark.parametrize("n", [4096, 4097])
@pytest.mark.parametrize("shape", ["gaussian", "lorentzian", "pseudo_voigt"])
def test_broadening_keeps_a_line_where_it_is(n, shape):
    # Regression: with an even number of points the Lorentzian kernel
    # was centred one sample off, so every line moved one point.
    x = np.linspace(-10, 10, n)
    delta = np.zeros(n)
    delta[n // 2] = 1.0
    y = S._apply_broadening(delta, x, fwhm=0.5, lineshape=shape, eta=0.5)
    c = n // 2
    assert np.argmax(y) == c
    k = min(c, n - 1 - c)                                # symmetric about the line
    assert np.allclose(y[c - k:c], y[c + 1:c + k + 1][::-1], rtol=1e-9, atol=1e-15)


def test_multiplet_lineshape_is_exact_anywhere():
    sticks, w = np.array([1.0, 1.2]), np.array([0.25, 0.75])
    x = np.array([1.2, 0.3, 1.0, 5.0])                  # any order, any spacing
    peak = S.multiplet_lineshape(x, sticks, w, 0.1, "lorentzian", normalise="peak")
    want = 0.25 * _lor(x, 1.0, 0.1) + 0.75 * _lor(x, 1.2, 0.1)
    assert np.allclose(peak, want, rtol=1e-12)
    area = S.multiplet_lineshape(np.linspace(-500, 500, 2_000_001), sticks, w, 0.1,
                                 "gaussian", normalise="area")
    assert area.sum() * (1000 / 2_000_000) == pytest.approx(1.0, rel=1e-6)


def test_analytic_and_grid_spectra_agree_when_well_sampled():
    x = np.linspace(0, 2, 20001)
    c = [{"nuclide": "1H", "n": 2, "J_Hz": 7.0}, {"nuclide": "31P", "n": 1, "J_Hz": 14.0}]
    a = S.jcoupled_spectrum(1.0, 100e6, c, x, fwhm=0.01)
    g = S.jcoupled_spectrum(1.0, 100e6, c, x, fwhm=0.01, method="grid")
    assert np.max(np.abs(a["real_spec"] - g["real_spec"])) < 5e-3 * g["real_spec"].max()
    assert a["real_spec"].sum() == pytest.approx(1.0, abs=5e-3)
    with pytest.raises(ValueError, match="method"):
        S.jcoupled_spectrum(1.0, 100e6, c, x, method="nope")


def test_multi_site_intensities_scale_areas():
    x = np.linspace(-1, 5, 60001)
    out = S.multi_site_jcoupled_spectrum(
        [{"ppm": 1.0, "relative_intensity": 3.0, "couplings": [{"nuclide": "1H", "n": 2, "J_Hz": 7}]},
         {"ppm": 3.0, "relative_intensity": 2.0, "couplings": []}],
        L_1H, x, fwhm=0.005)
    left = out["real_spec"][x < 2].sum()
    right = out["real_spec"][x >= 2].sum()
    assert left / right == pytest.approx(1.5, rel=1e-3)


# --- fitting ------------------------------------------------------------------

def _results(x, y, larmor=L_1H, nuclide="1H"):
    return {"x_ppm": x, "real_spectrum": y, "SFO1": larmor / 1e6, "nuclide": nuclide}


@pytest.mark.parametrize("n_points", [32768, 32769])
@pytest.mark.parametrize("centre", [3.2, 3.20013, 3.20027])
def test_triplet_fit_is_exact(n_points, centre):
    x = np.linspace(12, -2, n_points)                   # descending, as real data
    J, fwhm_hz = 7.1, 0.8
    fwhm = fwhm_hz / L_1H * 1e6
    y = 1000 * sum(w * _lor(x, centre + k * J / L_1H * 1e6, fwhm)
                   for k, w in ((-1, .25), (0, .5), (1, .25))) / 0.5
    p0 = {"label": "CH2", "center_ppm": 3.2, "fwhm": 0.003,
          "couplings": [{"nuclide": "1H", "n": 2, "J_Hz": 7.0, "abundance": 100.0}]}
    popt, pcov, free, scale, info = JF.fit_jcoupling_spectrum(
        _results(x, y), p0, fit_regions=[(3.1, 3.3)])
    assert (popt[0] - centre) * L_1H / 1e6 == pytest.approx(0, abs=1e-3)   # Hz
    assert popt[1] == pytest.approx(J, abs=1e-3)
    assert popt[2] * L_1H / 1e6 == pytest.approx(fwhm_hz, abs=1e-3)
    # amplitude = the multiplet's height at its tallest line, neighbours'
    # tails included: 1000 * (0.5 + 2 * 0.25 * L(J)) / 0.5
    tail = _lor(J, 0.0, fwhm_hz)
    assert popt[3] == pytest.approx(1000 * (0.5 + 0.5 * tail) / 0.5, rel=1e-4)


def test_two_site_two_group_fit():
    x = np.linspace(10, 0, 65536)
    f_ppm = 1.0 / L_1H * 1e6
    truth = [(4.10, [(15.2, 1), (6.9, 2)], 500.0), (1.25, [(7.3, 2)], 800.0)]

    def site(c, groups, amp):
        freqs, w = S.combine_jcouplings([{"nuclide": "1H", "n": n, "J_Hz": J, "abundance": 100.0}
                                        for J, n in groups])
        sticks = c + freqs / L_1H * 1e6
        prof = S.multiplet_lineshape(x, sticks, w, f_ppm, normalise="peak")
        top = S.multiplet_lineshape(sticks, sticks, w, f_ppm, normalise="peak").max()
        return amp * prof / top
    y = sum(site(*t) for t in truth)
    p0 = [{"label": "A", "center_ppm": 4.101, "fwhm": 0.003,
           "couplings": [{"nuclide": "1H", "n": 1, "J_Hz": 14.5, "abundance": 100.0},
                         {"nuclide": "1H", "n": 2, "J_Hz": 7.2, "abundance": 100.0}]},
          {"label": "B", "center_ppm": 1.249, "fwhm": 0.003,
           "couplings": [{"nuclide": "1H", "n": 2, "J_Hz": 7.0, "abundance": 100.0}]}]
    popt, *_ , info = JF.fit_jcoupling_spectrum(_results(x, y), p0,
                                               fit_regions=[(4.0, 4.2), (1.15, 1.35)])
    assert popt[0] == pytest.approx(4.10, abs=1e-6)
    assert popt[1:3] == pytest.approx([15.2, 6.9], abs=1e-3)
    assert popt[4] == pytest.approx(500, rel=1e-3)
    assert popt[5] == pytest.approx(1.25, abs=1e-6)
    assert popt[6] == pytest.approx(7.3, abs=1e-3)
    assert popt[8] == pytest.approx(800, rel=1e-3)


def test_model_value_does_not_depend_on_the_evaluation_axis():
    # Regression: each site was scaled to its maximum on whichever axis
    # it was evaluated on, so the report (fit region only) and the fit
    # (whole axis) described different amplitudes.
    x = np.linspace(5, 0, 8192)
    info = {"site_slices": [(0, 4)], "labels": ["A"], "larmor_Hz": L_1H,
            "lineshape": "lorentzian", "eta": 0.0, "x_ref": 2.5,
            "topologies": [[{"nuclide": "1H", "n": 1, "spin": 0.5, "abundance": 100.0}]]}
    popt = np.array([2.0, 12.0, 0.004, 1.0, 0.0, 0.0])
    full = JF.evaluate_jcoupling_fit(x, popt, info)
    part = JF.evaluate_jcoupling_fit(x[3000:3400], popt, info)
    assert np.allclose(full[3000:3400], part, rtol=1e-12)


def test_fixed_j_stays_put():
    x = np.linspace(4, 2, 16384)
    fwhm = 1.0 / L_1H * 1e6
    y = sum(_lor(x, 3.0 + k * 8.0 / L_1H * 1e6, fwhm) for k in (-0.5, 0.5))
    p0 = {"center_ppm": 3.0, "fwhm": 0.003,
          "couplings": [{"nuclide": "1H", "n": 1, "J_Hz": 6.0, "fixed_J": True,
                         "abundance": 100.0}]}
    popt, _, free, *_ = JF.fit_jcoupling_spectrum(_results(x, y), p0)
    assert popt[1] == 6.0 and not free[1]


def test_chemist_notation():
    topo = [{"nuclide": "31P", "n": 1, "spin": 0.5},
            {"nuclide": "1H", "n": 2, "spin": 0.5, "bond_label": "3J(H-H)"}]
    mult, js = JF.jcoupling_chemist_notation("1H", topo, [140.0, 7.02])
    assert mult == "dt"
    assert js == ["J(H-P) = 140.0 Hz", "3J(H-H) = 7.0 Hz"]
