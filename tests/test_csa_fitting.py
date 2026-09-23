"""The notebook CSA fitter (opennmr.notebook.csa_fitting).

Synthetic finite-MAS powder patterns with known tensors, at a real
spectrum's intensity scale (~1e10) -- the scale at which fitting raw
intensities converged badly and the amplitude uncertainties came out
wrong by the normalisation factor.
"""
import numpy as np
import pytest

from opennmr.notebook import csa_fitting as C

LARMOR_HZ = 100.0e6
MAS_HZ = 5000.0            # sidebands every 50 ppm at 100 MHz
TRUE = dict(d_iso=0.0, span_or_Delta=-100.0, skew_or_eta=0.5, fwhm=2.0)
SCALE = 1.0e10


def _mas_results(scale=SCALE, eta=0.0, lineshape="gaussian", **overrides):
    params = {**TRUE, **overrides}
    x = np.linspace(-300.0, 300.0, 2048)
    y = C.csa_model_full(x, params["d_iso"], params["span_or_Delta"], params["skew_or_eta"],
                         params["fwhm"], eta=eta, convention="Haeberlen", static=False,
                         larmor_Hz=LARMOR_HZ, mas_Hz=MAS_HZ, powder_scheme="zcw", n_zcw=400,
                         lineshape=lineshape)
    return {"x_ppm": x[::-1], "real_spectrum": (scale * y)[::-1], "SFO1": LARMOR_HZ / 1e6}


def _fit(results, p0=None, **kwargs):
    p0 = p0 or [{"d_iso": 2.0, "span_or_Delta": -80.0, "skew_or_eta": 0.4, "fwhm": 3.0}]
    options = dict(static=False, mas_Hz=MAS_HZ, powder_scheme="zcw", n_zcw=200)
    options.update(kwargs)
    return C.fit_csa_spectrum(results, p0, **options)


def test_recovers_a_known_mas_tensor():
    popt, _pcov, _free, _scale, info = _fit(_mas_results())
    assert popt[0] == pytest.approx(TRUE["d_iso"], abs=0.05)
    assert popt[1] == pytest.approx(TRUE["span_or_Delta"], abs=1.0)
    assert popt[2] == pytest.approx(TRUE["skew_or_eta"], abs=0.02)
    assert popt[4] == pytest.approx(SCALE, rel=0.01)       # amplitude in raw units
    assert info["warnings"] == []


# ---- conditioning ------------------------------------------------------

def test_use_normalised_no_longer_changes_the_answer():
    # Fitting raw intensities (use_normalised=False) left the optimiser
    # badly conditioned: on a real 79Br spectrum it stopped at a worse
    # optimum. The fit is now always conditioned internally.
    results = _mas_results()
    a = _fit(results, use_normalised=True)[0]
    b = _fit(results, use_normalised=False)[0]
    assert np.allclose(a, b, rtol=1e-6, atol=1e-6)


def test_uncertainties_are_in_raw_units():
    # Amplitude uncertainty must scale with the data; shape-parameter
    # uncertainties must not. (Amplitude errors were left in normalised
    # units -- too small by the scale factor.)
    rng = np.random.default_rng(1)
    base = _mas_results(scale=1.0)
    noise = rng.normal(0.0, 0.002, base["real_spectrum"].size)
    small = {**base, "real_spectrum": base["real_spectrum"] + noise}
    big = {**base, "real_spectrum": 1.0e8 * (base["real_spectrum"] + noise)}
    p_small, c_small, free, _s, _i = _fit(small)
    p_big, c_big, _f, _s2, _i2 = _fit(big)
    names = np.array(C._CSA_PARAM_NAMES + ["b0", "b1"])[free]
    e_small, e_big = np.sqrt(np.diag(c_small)), np.sqrt(np.diag(c_big))
    amp = list(names).index("amplitude")
    delta = list(names).index("span_or_Delta")
    assert e_big[amp] / e_small[amp] == pytest.approx(1.0e8, rel=1e-3)
    assert e_big[delta] == pytest.approx(e_small[delta], rel=1e-3)


def test_a_supplied_amplitude_is_read_in_raw_units():
    # A raw-unit amplitude in p0 was applied to the scaled data, so a
    # fixed amplitude was wrong by the scale factor.
    p0 = [{"d_iso": 2.0, "span_or_Delta": -80.0, "skew_or_eta": 0.4, "fwhm": 3.0,
           "amplitude": SCALE, "fixed": {"amplitude": True}}]
    popt, _pcov, free, _scale, _info = _fit(_mas_results(), p0=p0)
    assert not free[4]
    assert popt[4] == pytest.approx(SCALE)
    assert popt[1] == pytest.approx(TRUE["span_or_Delta"], abs=1.0)


# ---- what p0 says is honoured ------------------------------------------

def test_p0_type_selects_the_lineshape():
    # "type": "pseudo-voigt" in p0 was ignored and a Gaussian was fitted.
    results = _mas_results(eta=0.6, lineshape="pseudo_voigt")
    p0 = [{"type": "pseudo-voigt", "d_iso": 2.0, "span_or_Delta": -80.0,
           "skew_or_eta": 0.4, "fwhm": 3.0, "eta": 0.5}]
    popt, _pcov, free, _scale, info = _fit(results, p0=p0)
    assert info["sim_kwargs"]["lineshape"] == "pseudo_voigt"
    assert free[5]                                       # the L fraction is fitted
    assert popt[5] == pytest.approx(0.6, abs=0.03)


@pytest.mark.parametrize("given, expected", [
    ("Pseudo-Voigt", "pseudo_voigt"), ("gauss", "gaussian"), ("Lorentzian", "lorentzian"),
    ("pv", "pseudo_voigt")])
def test_lineshape_spellings(given, expected):
    assert C.canonical_lineshape(given) == expected


def test_lineshape_argument_overrides_p0_and_mixed_types_are_refused():
    p0 = [{"type": "lorentzian", "d_iso": 0, "span_or_Delta": 1, "skew_or_eta": 0, "fwhm": 1}]
    assert C.resolve_lineshape(p0) == "lorentzian"
    assert C.resolve_lineshape(p0, "gaussian") == "gaussian"
    assert C.resolve_lineshape([{"d_iso": 0}]) == "gaussian"   # historical default
    mixed = p0 + [{**p0[0], "label": "B", "type": "gaussian"}]
    with pytest.raises(ValueError, match="different lineshapes"):
        C.resolve_lineshape(mixed)


def test_a_sites_own_fixed_and_bounds_are_honoured():
    # "fixed" inside a site dict was ignored (the notebook call in the
    # bug report writes it there).
    p0 = [{"d_iso": 2.0, "span_or_Delta": -80.0, "skew_or_eta": 0.3, "fwhm": 3.0,
           "fixed": {"skew_or_eta": True, "d_iso": False},
           "bounds": {"span_or_Delta": (-90.0, -85.0)}}]
    popt, _pcov, free, _scale, info = _fit(_mas_results(), p0=p0)
    # -80 starts outside (-90, -85): moved inside and said so, rather
    # than curve_fit's bare "Initial guess is outside of provided bounds".
    assert any("outside its bounds" in w for w in info["warnings"])
    assert not free[2] and popt[2] == pytest.approx(0.3)
    assert free[0]
    assert -90.0 <= popt[1] <= -85.0
    assert popt[1] == pytest.approx(-90.0, abs=0.5)        # pressed against its bound


def test_an_unknown_name_in_a_sites_fixed_is_an_error():
    p0 = [{"d_iso": 0, "span_or_Delta": -80, "skew_or_eta": 0.4, "fwhm": 3,
           "fixed": {"delta_iso": True}}]
    with pytest.raises(ValueError, match="delta_iso"):
        _fit(_mas_results(), p0=p0)


# ---- sidebands in the fit window ---------------------------------------

def test_a_window_without_sidebands_is_warned_about():
    # Without sidebands every MAS rate fits the centreband equally well.
    _p, _c, _f, _s, info = _fit(_mas_results(), fit_regions=[(-30.0, 30.0)])
    assert info["warnings"] and "sideband" in info["warnings"][0]
    assert info["fit_regions"] == [(-30.0, 30.0)]


def test_a_window_with_a_sideband_is_not():
    _p, _c, _f, _s, info = _fit(_mas_results(), fit_regions=[(-70.0, 70.0)])
    assert info["warnings"] == []


def test_sideband_positions_use_the_larmor_frequency():
    # 20 kHz at 200.48 MHz: first sidebands 99.76 ppm either side.
    assert C.sideband_window_warning(0.0, 20000, 200.48e6, fit_regions=[(-99.0, 99.0)])
    assert C.sideband_window_warning(0.0, 20000, 200.48e6, fit_regions=[(-101.0, 101.0)]) is None
    assert C.sideband_window_warning(0.0, None, 200.48e6, fit_regions=[(-1, 1)]) is None
