import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.processing import fitting, relaxation, slicing
from opennmr.processing.fitting import gaussian


def _make_pseudo2d_relaxation(delays, M0=1.0, T1=0.8, C=2.0, x0=5.0, fwhm=0.05,
                               b0=0.005, n_pts=1000):
    ppm = np.linspace(3.0, 7.0, n_pts)
    n_rows = len(delays)
    data = np.zeros((n_rows, n_pts))
    for i, t in enumerate(delays):
        A = relaxation.inversion_recovery(t, M0, T1, C)
        data[i] = gaussian(ppm, x0, fwhm, A) + b0
    f1_axis = NMRAxis(kind="index", index=np.arange(1, n_rows + 1))
    f2_axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True)
    return NMRSpectrum(data=data, axes=[f1_axis, f2_axis], metadata=metadata)


# ============================================================
# Relaxation model functions
# ============================================================

def test_saturation_recovery_known_points():
    assert relaxation.saturation_recovery(0.0, 5.0, 2.0) == pytest.approx(0.0)
    assert relaxation.saturation_recovery(2.0, 5.0, 2.0) == pytest.approx(5.0 * (1 - np.exp(-1)))


def test_inversion_recovery_known_points():
    assert relaxation.inversion_recovery(0.0, 3.0, 1.5, C=2.0) == pytest.approx(-3.0)
    assert relaxation.inversion_recovery(100.0, 3.0, 1.5, C=2.0) == pytest.approx(3.0, abs=1e-6)


def test_exponential_decay_known_points():
    assert relaxation.exponential_decay(1.0, 4.0, 1.0) == pytest.approx(4.0 * np.exp(-1))


def test_biexponential_decay_known_points():
    assert relaxation.biexponential_decay(0.0, 1.0, 2.0, 3.0, 4.0) == pytest.approx(4.0)


def test_biexponential_recovery_known_points():
    val = relaxation.biexponential_recovery(0.0, 1.0, 2.0, 3.0, 4.0, C=2.0)
    assert val == pytest.approx(-4.0)


def test_stretched_decay_beta_one_matches_mono_exponential():
    t = np.linspace(0, 5, 20)
    np.testing.assert_allclose(
        relaxation.stretched_decay(t, 2.0, 1.3, beta=1.0),
        relaxation.exponential_decay(t, 2.0, 1.3),
    )


def test_stretched_recovery_known_point():
    val = relaxation.stretched_recovery(0.0, 2.0, 1.0, beta=0.7, C=2.0)
    assert val == pytest.approx(-2.0)


def test_stejskal_tanner_decay_known_points():
    assert relaxation.stejskal_tanner_decay(0.0, 5.0, 2e-9) == pytest.approx(5.0)
    b = 1.0 / 2e-9
    assert relaxation.stejskal_tanner_decay(b, 5.0, 2e-9) == pytest.approx(5.0 * np.exp(-1))


def test_stejskal_tanner_decay_stretched_beta_one_matches_plain():
    b = np.linspace(0, 5e9, 20)
    np.testing.assert_allclose(
        relaxation.stejskal_tanner_decay_stretched(b, 3.0, 1.5e-9, beta=1.0),
        relaxation.stejskal_tanner_decay(b, 3.0, 1.5e-9),
    )


def test_stejskal_tanner_decay_biexponential_known_points():
    # b=0: both components at full amplitude, matching biexponential_decay's own known-point test.
    assert relaxation.stejskal_tanner_decay_biexponential(0.0, 1.0, 2e-9, 3.0, 4e-9) == pytest.approx(4.0)


def test_stejskal_tanner_decay_biexponential_reduces_to_single_component():
    # Both components identical -> same as a single Stejskal-Tanner decay of double the amplitude.
    b = np.linspace(0, 5e9, 20)
    biexp = relaxation.stejskal_tanner_decay_biexponential(b, 2.0, 1.5e-9, 2.0, 1.5e-9)
    mono = relaxation.stejskal_tanner_decay(b, 4.0, 1.5e-9)
    np.testing.assert_allclose(biexp, mono)


def test_relaxation_models_registry_contains_expected_keys():
    assert set(relaxation.RELAXATION_MODELS) == {
        "saturation_recovery", "inversion_recovery", "exponential_decay",
        "biexponential_decay", "biexponential_recovery",
        "stretched_decay", "stretched_recovery",
        "stejskal_tanner", "stejskal_tanner_stretched", "stejskal_tanner_biexponential",
    }


# ============================================================
# format_time_value
# ============================================================

def test_format_time_value_units_and_rounding():
    assert relaxation.format_time_value(0.00152, 0.00003) == "1.52 ± 0.03 ms"
    assert relaxation.format_time_value(2.0) == "2.00 s"
    assert relaxation.format_time_value(None) == "n/a"


# ============================================================
# format_diffusion_value
# ============================================================

def test_format_diffusion_value_scientific_notation():
    assert relaxation.format_diffusion_value(2.034e-9, 0.05e-9) == "2.03e-09 ± 0.05e-09 m^2/s"
    assert relaxation.format_diffusion_value(None) == "n/a"


def test_format_diffusion_value_no_error():
    assert relaxation.format_diffusion_value(1.0e-10) == "1.00e-10 m^2/s"


# ============================================================
# fit_relaxation
# ============================================================

def test_fit_relaxation_recovers_inversion_recovery_t1():
    M0_true, T1_true, C_true = 10.0, 1.5, 2.0
    t = np.linspace(0.05, 8.0, 30)
    y = relaxation.inversion_recovery(t, M0_true, T1_true, C_true)

    fit = relaxation.fit_relaxation(t, y, model="inversion_recovery")

    assert fit["rate_param"] == "T1"
    assert fit["params"]["T1"] == pytest.approx(T1_true, rel=1e-3)
    assert fit["params"]["M0"] == pytest.approx(M0_true, rel=1e-3)
    assert fit["params"]["C"] == pytest.approx(C_true, rel=1e-2)


def test_fit_relaxation_recovers_exponential_decay_t2():
    M0_true, T2_true = 5.0, 0.3
    t = np.linspace(0.01, 2.0, 25)
    y = relaxation.exponential_decay(t, M0_true, T2_true)

    fit = relaxation.fit_relaxation(t, y, model="exponential_decay")

    assert fit["rate_param"] == "T2"
    assert fit["params"]["T2"] == pytest.approx(T2_true, rel=1e-3)
    assert fit["params"]["M0"] == pytest.approx(M0_true, rel=1e-3)


def test_fit_relaxation_unknown_model_raises():
    with pytest.raises(ValueError):
        relaxation.fit_relaxation([0, 1], [0, 1], model="not_a_model")


def test_build_custom_model_detects_params_in_first_appearance_order():
    spec = relaxation.build_custom_model("A*exp(-x/T1)+C")
    assert spec["params"] == ["A", "T1", "C"]
    assert spec["rate_param"] is None


def test_build_custom_model_fits_the_same_as_the_equivalent_builtin_model():
    # A*exp(-x/T2) is algebraically exponential_decay -- a custom
    # equation should recover the same parameters just as precisely.
    M0_true, T2_true = 5.0, 0.3
    t = np.linspace(0.01, 2.0, 25)
    y = relaxation.exponential_decay(t, M0_true, T2_true)

    spec = relaxation.build_custom_model("A*exp(-x/T2)")
    fit = relaxation.fit_relaxation(t, y, custom_model=spec)

    assert fit["param_names"] == ["A", "T2"]
    assert fit["params"]["A"] == pytest.approx(M0_true, rel=1e-3)
    assert fit["params"]["T2"] == pytest.approx(T2_true, rel=1e-3)


def test_build_custom_model_p0_overrides_apply_positionally():
    spec = relaxation.build_custom_model("A*exp(-x/T1)+C", p0_overrides=[3.0, 0.7])
    assert spec["p0"](None, None) == [3.0, 0.7, 1.0]  # C left unspecified -> default 1.0


def test_build_custom_model_rejects_unparseable_equation():
    with pytest.raises(ValueError):
        relaxation.build_custom_model("A*exp(-x/T1")  # unbalanced parens


def test_build_custom_model_rejects_missing_independent_variable():
    with pytest.raises(ValueError):
        relaxation.build_custom_model("A + B")  # no 'x'


def test_build_custom_model_rejects_no_free_parameters():
    with pytest.raises(ValueError):
        relaxation.build_custom_model("x + 3")


def test_fit_curve_uncertainty_shape_and_positivity():
    M0_true, T2_true = 5.0, 0.3
    rng = np.random.default_rng(0)
    t = np.linspace(0.01, 2.0, 25)
    y = relaxation.exponential_decay(t, M0_true, T2_true) + rng.normal(0, 0.05, t.shape)
    fit = relaxation.fit_relaxation(t, y, model="exponential_decay")

    t_fine, y_fine, y_band = relaxation.fit_curve_uncertainty(fit, n_points=50, sigma=3.0)

    assert t_fine.shape == (50,)
    assert y_fine.shape == (50,)
    assert y_band.shape == (50,)
    assert t_fine[0] == pytest.approx(t.min())
    assert t_fine[-1] == pytest.approx(t.max())
    assert np.all(np.isfinite(y_band))
    assert np.all(y_band >= 0)  # a width, never negative
    # The fine curve should agree with the model evaluated directly at
    # the same points (not some unrelated quantity).
    np.testing.assert_allclose(
        y_fine, relaxation.exponential_decay(t_fine, fit["params"]["M0"], fit["params"]["T2"]),
    )


def test_fit_curve_uncertainty_scales_with_sigma():
    M0_true, T2_true = 5.0, 0.3
    rng = np.random.default_rng(1)
    t = np.linspace(0.01, 2.0, 25)
    y = relaxation.exponential_decay(t, M0_true, T2_true) + rng.normal(0, 0.05, t.shape)
    fit = relaxation.fit_relaxation(t, y, model="exponential_decay")

    _, _, band_1sigma = relaxation.fit_curve_uncertainty(fit, n_points=30, sigma=1.0)
    _, _, band_3sigma = relaxation.fit_curve_uncertainty(fit, n_points=30, sigma=3.0)

    np.testing.assert_allclose(band_3sigma, band_1sigma * 3.0)


def test_fit_curve_uncertainty_tighter_with_less_noise():
    # A cleaner fit should be more tightly constrained -- the band
    # should shrink as the data noise shrinks, not be some fixed
    # unrelated width.
    M0_true, T2_true = 5.0, 0.3
    t = np.linspace(0.01, 2.0, 25)

    rng_low = np.random.default_rng(2)
    y_low_noise = relaxation.exponential_decay(t, M0_true, T2_true) + rng_low.normal(0, 0.01, t.shape)
    fit_low = relaxation.fit_relaxation(t, y_low_noise, model="exponential_decay")
    _, _, band_low = relaxation.fit_curve_uncertainty(fit_low, n_points=30)

    rng_high = np.random.default_rng(3)
    y_high_noise = relaxation.exponential_decay(t, M0_true, T2_true) + rng_high.normal(0, 0.5, t.shape)
    fit_high = relaxation.fit_relaxation(t, y_high_noise, model="exponential_decay")
    _, _, band_high = relaxation.fit_curve_uncertainty(fit_high, n_points=30)

    assert np.mean(band_low) < np.mean(band_high)


def test_fit_relaxation_recovers_stejskal_tanner_diffusion_coefficient():
    # A real b-value series (s/m^2), like io.bruker.read_diff_xml's own
    # "B" -- confirms the fit works at the actual physical scale
    # (b ~1e7-1e10), not just a toy 0..1 range.
    I0_true, D_true = 8.0, 2.0e-9
    b = np.linspace(0.0, 3.0e9, 16)
    y = relaxation.stejskal_tanner_decay(b, I0_true, D_true)

    fit = relaxation.fit_relaxation(b, y, model="stejskal_tanner")

    assert fit["rate_param"] == "D"
    assert fit["params"]["D"] == pytest.approx(D_true, rel=1e-3)
    assert fit["params"]["I0"] == pytest.approx(I0_true, rel=1e-3)


def test_fit_relaxation_recovers_stretched_stejskal_tanner():
    I0_true, D_true, beta_true = 4.0, 5.0e-10, 0.75
    b = np.linspace(0.0, 8.0e9, 20)
    y = relaxation.stejskal_tanner_decay_stretched(b, I0_true, D_true, beta_true)

    fit = relaxation.fit_relaxation(b, y, model="stejskal_tanner_stretched")

    assert fit["params"]["D"] == pytest.approx(D_true, rel=5e-2)
    assert fit["params"]["beta"] == pytest.approx(beta_true, rel=5e-2)


def test_fit_relaxation_recovers_biexponential_diffusion_coefficients():
    # Two well-separated components (a "fast" and a "slow" diffuser,
    # e.g. bound/free water) -- widely separated on purpose, matching
    # _p0_diffusion_biexp's own reasoning, so the fit has a real chance
    # of resolving two genuinely distinct components rather than both
    # guesses collapsing onto the same value.
    I0a_true, Da_true, I0b_true, Db_true = 3.0, 0.5e-9, 5.0, 5.0e-9
    b = np.linspace(0.0, 4.0e9, 24)
    y = relaxation.stejskal_tanner_decay_biexponential(b, I0a_true, Da_true, I0b_true, Db_true)

    fit = relaxation.fit_relaxation(b, y, model="stejskal_tanner_biexponential")

    assert fit["rate_param"] is None
    # Component order isn't identifiable from the data alone (the
    # model is symmetric under swapping the two) -- accept either
    # assignment as a correct recovery.
    fitted = fit["params"]
    as_found = {(fitted["I0a"], fitted["Da"]), (fitted["I0b"], fitted["Db"])}
    true_pairs = {(I0a_true, Da_true), (I0b_true, Db_true)}
    for (i0_found, d_found), (i0_true, d_true) in zip(
        sorted(as_found, key=lambda p: p[1]), sorted(true_pairs, key=lambda p: p[1]),
    ):
        assert i0_found == pytest.approx(i0_true, rel=1e-2)
        assert d_found == pytest.approx(d_true, rel=1e-2)


def test_fit_relaxation_fixed_parameter_is_held_fixed():
    M0_true, T1_true = 4.0, 1.0
    t = np.linspace(0.05, 5.0, 20)
    y = relaxation.inversion_recovery(t, M0_true, T1_true, C=2.0)

    fit = relaxation.fit_relaxation(t, y, model="inversion_recovery", fixed={"C": 2.0})

    assert np.isnan(fit["errors"]["C"])
    assert fit["params"]["C"] == 2.0
    assert fit["params"]["T1"] == pytest.approx(T1_true, rel=1e-3)


# ============================================================
# NMR2D_integrate_slices / NMR2D_fit_relaxation via get_slices
# ============================================================

def test_get_delays_requires_vdlist():
    with pytest.raises(ValueError):
        relaxation._get_delays([{"index": 1}], None)


def test_nmr2d_integrate_and_fit_relaxation_recovers_t1():
    delays = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.0])
    M0_true, T1_true, C_true = 1.0, 0.8, 2.0
    spec = _make_pseudo2d_relaxation(delays, M0=M0_true, T1=T1_true, C=C_true)

    slices = slicing.get_slices(spec, rows=list(range(1, len(delays) + 1)))
    integrated = relaxation.NMR2D_integrate_slices(
        slices, regions=[(4.5, 5.5)], mode="rows", noise_region=(3.0, 3.2),
    )
    rows = integrated["rows"]
    assert len(rows) == len(delays)
    for expected_index, r in zip(range(1, len(delays) + 1), rows):
        assert r["index"] == expected_index
        assert r["slice"].metadata.raw["slice"]["index"] == expected_index

    vdlist = {"name": "vdlist", "raw": [str(d) for d in delays], "seconds": delays}
    fits = relaxation.NMR2D_fit_relaxation(
        rows, model="inversion_recovery", vdlist=vdlist, show_report=False,
    )

    assert len(fits) == 1
    assert fits[0]["params"]["T1"] == pytest.approx(T1_true, rel=0.02)
    # An integrated series is labelled by its own region, not "Peak 1"
    # -- it is a region, and its bounds are what identify it.
    assert fits[0]["peak_label"] == "Region 1 (4.5-5.5 ppm)"


def test_nmr2d_fit_relaxation_value_source_amplitude_recovers_t1():
    # value_source="amplitude" reads each slice's fitted peak amplitude
    # (from a real NMR2D_fit_slices peak fit) instead of its integrated
    # area -- a different y-proxy for the same underlying decay, so it
    # should recover essentially the same T1 on synthetic data with no
    # linewidth changes across slices to confuse the two.
    delays = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.0])
    M0_true, T1_true, C_true = 1.0, 0.8, 2.0
    spec = _make_pseudo2d_relaxation(delays, M0=M0_true, T1=T1_true, C=C_true)

    slices = slicing.get_slices(spec, rows=list(range(1, len(delays) + 1)))
    peak_defs = [{"type": "gaussian", "shift": 5.0, "fwhm": 0.05}]
    fitted = fitting.NMR2D_fit_slices(
        slices, peak_defs, mode="rows", normalize=False, show_report=False, show_warnings=False,
    )
    rows = fitted["rows"]

    vdlist = {"name": "vdlist", "raw": [str(d) for d in delays], "seconds": delays}
    fits_amplitude = relaxation.NMR2D_fit_relaxation(
        rows, model="inversion_recovery", vdlist=vdlist, value_source="amplitude", show_report=False,
    )
    assert len(fits_amplitude) == 1
    assert fits_amplitude[0]["params"]["T1"] == pytest.approx(T1_true, rel=0.05)

    fits_integral = relaxation.NMR2D_fit_relaxation(
        rows, model="inversion_recovery", vdlist=vdlist, value_source="integral", show_report=False,
    )
    # Different y-proxy, same underlying decay -- should agree closely.
    assert fits_amplitude[0]["params"]["T1"] == pytest.approx(fits_integral[0]["params"]["T1"], rel=0.05)


def test_nmr2d_fit_relaxation_rejects_bad_value_source():
    with pytest.raises(ValueError):
        relaxation.NMR2D_fit_relaxation(
            [{"index": 1}], vdlist={"seconds": [1.0]}, value_source="bogus", show_report=False,
        )


# --- Known values from the acquisition, used in a custom equation ---

def test_build_custom_model_substitutes_supplied_constants():
    # User request: "it would be good if I can make fitting equations
    # using values from the input data (e.g. delays and constants and L
    # or whatever)". A symbol that is known must become a number in the
    # expression, not another thing to fit.
    spec = relaxation.build_custom_model("A*exp(-x*d1/T2)+C", constants={"d1": 2.0})
    assert spec["params"] == ["A", "T2", "C"]
    assert spec["constants"] == {"d1": 2.0}

    # Without it, d1 is just another free parameter -- the old behaviour.
    assert relaxation.build_custom_model("A*exp(-x*d1/T2)+C")["params"] == ["A", "d1", "T2", "C"]

    # And the substituted value is really in the maths.
    t = np.linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(spec["func"](t, 1.0, 0.5, 0.0), np.exp(-t * 2.0 / 0.5))


def test_build_custom_model_ignores_constants_the_equation_does_not_use():
    # The caller passes every acquisition parameter it has rather than
    # guessing which ones an equation wants, so unused names must be
    # silently irrelevant -- there are ~780 of them on a real dataset.
    spec = relaxation.build_custom_model(
        "A*exp(-x/T2)", constants={"d1": 2.0, "p1": 9.5, "ns": 8.0})
    assert spec["params"] == ["A", "T2"]
    assert spec["constants"] == {}


def test_build_custom_model_never_substitutes_the_independent_variable():
    # A vendor could record a parameter called "x"; the independent
    # variable is not up for substitution regardless.
    spec = relaxation.build_custom_model("A*exp(-x/T2)", constants={"x": 5.0})
    assert spec["params"] == ["A", "T2"]
    assert spec["constants"] == {}
    t = np.linspace(0.0, 1.0, 4)
    np.testing.assert_allclose(spec["func"](t, 1.0, 1.0), np.exp(-t))


def test_build_custom_model_reports_constants_when_nothing_is_left_to_fit():
    with pytest.raises(ValueError, match="no free parameters"):
        relaxation.build_custom_model("A*x", constants={"A": 3.0})
