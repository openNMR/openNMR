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


# ============================================================
# Factor expressions over acquisition parameters
# ============================================================

def test_evaluate_expression_plain_number():
    assert relaxation.evaluate_expression("0.5")["value"] == pytest.approx(0.5)


def test_evaluate_expression_uses_acquisition_parameters():
    # The user's own case: an X-Pulse CPMG echo list times TAU.
    result = relaxation.evaluate_expression("TAU*2*1e-6", {"TAU": 1500.0, "P90": 8.78})
    assert result["value"] == pytest.approx(0.003)
    # Only the parameters actually used are reported back, so the GUI
    # can show which ones a factor picked up.
    assert result["used"] == {"TAU": 1500.0}


def test_evaluate_expression_unknown_symbol_is_an_error_not_a_parameter():
    # The one place this must differ from build_custom_model: there is
    # nothing to fit a factor against, so a mistyped name cannot
    # quietly become a free parameter.
    with pytest.raises(ValueError, match="TAO"):
        relaxation.evaluate_expression("TAO*2", {"TAU": 1500.0})


def test_evaluate_expression_names_what_is_available():
    with pytest.raises(ValueError, match="TAU"):
        relaxation.evaluate_expression("nope", {"TAU": 1500.0})


def test_evaluate_expression_says_when_nothing_is_recorded():
    with pytest.raises(ValueError, match="no recorded acquisition parameters"):
        relaxation.evaluate_expression("TAU", {})


def test_evaluate_expression_rejects_empty_and_nonsense():
    with pytest.raises(ValueError, match="Enter a factor"):
        relaxation.evaluate_expression("   ")
    with pytest.raises(ValueError, match="Could not parse"):
        relaxation.evaluate_expression("2 +* 3")


# ============================================================
# Curve fit as CSV
# ============================================================

def _two_peak_curve_fit():
    t = np.geomspace(0.01, 0.5, 6)
    fits = [
        {"index": i + 1,
         "integrals": [float(relaxation.exponential_decay(ti, 10.0, 0.1)),
                        float(relaxation.exponential_decay(ti, 5.0, 0.3))]}
        for i, ti in enumerate(t)
    ]
    return t, relaxation.NMR2D_fit_relaxation(
        fits, model="exponential_decay", vdlist=t, show_report=False,
        peak_labels=["A", "B"])


def test_curve_fit_to_csv_writes_each_peaks_measured_values_against_x():
    t, results = _two_peak_curve_fit()
    rows = [row.split(",") for row in
            relaxation.curve_fit_to_csv(results, x_label="delay (s)",
                                        include_parameters=False).splitlines()]
    header, body = rows[0], rows[1:]
    assert header == ["delay (s)", "A value", "A fit", "B value", "B fit"]
    assert len(body) == len(t)
    np.testing.assert_allclose([float(r[0]) for r in body], t, rtol=1e-9)
    np.testing.assert_allclose([float(r[1]) for r in body],
                                relaxation.exponential_decay(t, 10.0, 0.1), rtol=1e-6)


def test_curve_fit_to_csv_scales_x_to_the_requested_unit():
    t, results = _two_peak_curve_fit()
    rows = relaxation.curve_fit_to_csv(
        results, x_label="delay (ms)", x_scale=1e-3,
        include_parameters=False).splitlines()
    np.testing.assert_allclose(
        [float(row.split(",")[0]) for row in rows[1:]], t * 1e3, rtol=1e-9)


def test_curve_fit_to_csv_names_the_value_source():
    _t, results = _two_peak_curve_fit()
    header = relaxation.curve_fit_to_csv(results, value_label="amplitude",
                                         include_parameters=False).splitlines()[0]
    assert "A amplitude" in header and "B amplitude" in header


def test_curve_fit_to_csv_leaves_a_gap_where_a_peak_has_no_point():
    # Peaks normally share an x-axis, but excluding a point from one
    # peak's fit leaves it shorter. The union of x is used and the
    # missing cell is empty -- not dropped, and not padded with a
    # number that was never measured.
    _t, results = _two_peak_curve_fit()
    for key in ("t", "y", "y_fit"):
        results[1][key] = np.asarray(results[1][key])[1:]
    rows = [row.split(",") for row in
            relaxation.curve_fit_to_csv(results, include_parameters=False).splitlines()[1:]]
    assert rows[0][3] == "" and rows[0][4] == ""
    assert rows[1][3] != ""
    assert len(rows) == 6  # every x still present


def test_curve_fit_to_csv_includes_errors_when_the_fit_had_them():
    _t, results = _two_peak_curve_fit()
    for result in results:
        result["y_err"] = np.full(len(result["y"]), 0.1)
    header = relaxation.curve_fit_to_csv(results, include_parameters=False).splitlines()[0]
    assert "A value error" in header


def test_curve_fit_to_csv_with_no_results_raises():
    with pytest.raises(ValueError, match="run Fit Curve"):
        relaxation.curve_fit_to_csv([])


# ============================================================
# Relative amplitudes
# ============================================================
# "It would be good to have relative ratios instead of absolute
# intensity values so numbers can be more easily compared in
# multi-component fits."

def test_amplitude_fractions_of_a_two_component_fit():
    fractions = relaxation.amplitude_fractions(
        {"A1": 3.0, "T2a": 0.03, "A2": 1.0, "T2b": 0.4})
    assert fractions == {"A1": pytest.approx(0.75), "A2": pytest.approx(0.25)}


def test_amplitude_fractions_recognises_every_registry_naming():
    # M0/A/A1/A2 for relaxation, I0/I0a/I0b and I1..In for diffusion.
    assert set(relaxation.amplitude_fractions({"I0a": 6.0, "I0b": 2.0})) == {"I0a", "I0b"}
    assert set(relaxation.amplitude_fractions({"I1": 5.0, "I2": 3.0, "I3": 2.0})) == {"I1", "I2", "I3"}


def test_amplitude_fractions_needs_more_than_one_amplitude():
    # A single-amplitude fit would report a constant 100%, which is not
    # a result.
    assert relaxation.amplitude_fractions({"M0": 5.0, "T2": 0.1}) == {}
    assert relaxation.amplitude_fractions({"I0": 5.0, "D": 1e-9}) == {}


def test_amplitude_fractions_ignores_rates_and_shape_parameters():
    fractions = relaxation.amplitude_fractions(
        {"A1": 3.0, "T2a": 0.03, "A2": 1.0, "T2b": 0.4, "beta": 0.8, "C": 2.0, "D1": 1e-9})
    assert set(fractions) == {"A1", "A2"}


def test_amplitude_fractions_of_a_custom_equation_is_empty():
    # Its parameters are whatever the user named them; treating the
    # wrong symbol as an amplitude would be worse than showing nothing.
    assert relaxation.amplitude_fractions({"a1": 3.0, "t1": 0.03, "a2": 1.0, "t2": 0.4}) == {}


def test_amplitude_fractions_zero_total_is_empty():
    assert relaxation.amplitude_fractions({"A1": 1.0, "A2": -1.0}) == {}


def test_amplitude_fractions_agree_with_component_weights():
    # The two must never quote different numbers for the same fit.
    from opennmr.processing.diffusion import component_weights

    params = {"I1": 5.0, "D1": 3e-9, "I2": 3.0, "D2": 5e-10, "I3": 2.0, "D3": 4e-11}
    fractions = relaxation.amplitude_fractions(params)
    np.testing.assert_allclose(
        [fractions[f"I{i}"] for i in (1, 2, 3)], component_weights(params, 3))


def _biexponential_fit():
    t = np.geomspace(0.002, 1.0, 20)
    y = relaxation.biexponential_decay(t, 7.0, 0.03, 3.0, 0.4)
    return relaxation.fit_relaxation(t, y, model="biexponential_decay")


def test_report_shows_the_share_on_the_amplitude_row():
    report = relaxation.format_relaxation_report(_biexponential_fit())
    amplitude_row = next(line for line in report.splitlines() if line.strip().startswith("A1 "))
    assert "(70.0%)" in amplitude_row


def test_report_pairs_each_share_with_its_own_rate():
    report = relaxation.format_relaxation_report(_biexponential_fit())
    assert "Relative amplitudes" in report
    section = report.split("Relative amplitudes")[1]  # heading also names "population weights"
    assert "T2a = 30.0 ms" in section
    assert "T2b = 400 ms" in section


def test_report_keeps_the_absolute_amplitude_and_its_error():
    # The ratio is for comparing; the report is still the record of
    # what was actually fitted.
    report = relaxation.format_relaxation_report(_biexponential_fit())
    amplitude_row = next(line for line in report.splitlines() if line.strip().startswith("A1 "))
    assert "7" in amplitude_row.split("(")[0]


def test_report_has_no_share_section_for_a_single_amplitude():
    t = np.geomspace(0.002, 1.0, 15)
    fit = relaxation.fit_relaxation(
        t, relaxation.exponential_decay(t, 4.0, 0.1), model="exponential_decay")
    assert "Relative amplitudes" not in relaxation.format_relaxation_report(fit)


def test_report_flags_shares_outside_zero_to_one_hundred():
    # Opposite-sign components are usually an over-parameterised model,
    # and an absolute amplitude would have hidden it.
    t = np.geomspace(0.002, 1.0, 20)
    y = relaxation.biexponential_decay(t, 5.0, 0.03, -1.0, 0.4)
    fit = relaxation.fit_relaxation(t, y, model="biexponential_decay",
                                     p0=[5.0, 0.03, -1.0, 0.4])
    report = relaxation.format_relaxation_report(fit)
    assert "125.00%" in report
    # The note is word-wrapped, so compare on collapsed whitespace.
    assert "opposite signs" in " ".join(report.split())


def test_report_orders_shares_the_way_the_model_does():
    # Alphabetical would sort a 10-component fit as I1, I10, I2, ...
    params = {f"I{i}": float(11 - i) for i in range(1, 11)}
    params.update({f"D{i}": 1e-9 / i for i in range(1, 11)})
    fit = {
        "model": "diffusion_n", "param_names": [n for i in range(1, 11)
                                                 for n in (f"I{i}", f"D{i}")],
        "params": params, "errors": {}, "rate_param": None, "spec": {},
        "fit_quality": {}, "y_err": None,
    }
    section = relaxation.format_relaxation_report(fit).split("Relative amplitudes")[1]
    shown = [line.split()[0] for line in section.splitlines()
             if line.strip().startswith("I")]
    assert shown == [f"I{i}" for i in range(1, 11)]


# ============================================================
# A value list longer than the data
# ============================================================

def test_get_delays_uses_the_first_entries_of_a_longer_list():
    # Bruker's own convention: a 24-entry vdlist with F1 TD set to 10
    # uses the first 10 values, and TopSpin does not complain.
    fits = [{"index": i} for i in range(1, 4)]
    delays = relaxation._get_delays(fits, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_allclose(delays, [1.0, 2.0, 3.0])


def test_get_delays_reports_a_shorter_list_with_the_way_out():
    # Shorter genuinely cannot be fitted -- there is no value for the
    # last slices. It used to surface as a bare IndexError rephrased as
    # "check the vdlist matches this dataset", which says what is wrong
    # but not what to do.
    fits = [{"index": i} for i in range(1, 4)]
    with pytest.raises(ValueError) as excinfo:
        relaxation._get_delays(fits, np.array([1.0, 2.0]))
    message = str(excinfo.value)
    assert "2 entries" in message and "slice 3" in message
    assert "first 2 slices" in message


def test_get_delays_follows_the_slice_indices_not_their_order():
    # Entries are looked up by slice number, so a reordered or
    # partial set of fits still gets its own values.
    fits = [{"index": 3}, {"index": 1}]
    np.testing.assert_allclose(
        relaxation._get_delays(fits, np.array([10.0, 20.0, 30.0, 40.0])), [30.0, 10.0])


# ============================================================
# Telling a fitted peak area apart from an integrated region
# ============================================================

def _fitted_slices():
    """Slices from NMR2D_fit_slices -- carrying popt, and integrals
    that are the area under each *fitted lineshape*, rather than a raw
    sum over a region nobody fitted anything to."""
    delays = np.linspace(0.01, 0.3, 6)
    spec = _make_pseudo2d_relaxation(delays, M0=1.0, T1=0.1, C=0.0)
    slices = slicing.get_slices(spec, rows=list(range(1, len(delays) + 1)))
    fitted = fitting.NMR2D_fit_slices(
        slices, [{"type": "gaussian", "shift": 5.0, "fwhm": 0.05}],
        mode="rows", show_report=False, show_warnings=False)
    return fitted["rows"], delays


def test_value_source_area_needs_a_lineshape_fit():
    """Explicit user request: the two old options "don't make it clear
    if a fully fitted spectrum is being used"."""
    fits, delays = _fitted_slices()
    # Fitted slices: area and amplitude both work.
    for source in ("area", "amplitude"):
        results = relaxation.NMR2D_fit_relaxation(fits, model="exponential_decay", vdlist=delays,
                                        value_source=source, show_report=False)
        assert results and np.isfinite(results[0]["params"]["T2"])

    # ...and asking for a raw region integral is refused, not quietly
    # answered with the fitted area.
    with pytest.raises(ValueError, match="raw integral over a region"):
        relaxation.NMR2D_fit_relaxation(fits, model="exponential_decay", vdlist=delays,
                             value_source="region", show_report=False)


def test_value_source_region_needs_integrated_slices():
    integrated = [{"integrals": [1.0], "regions": [(4.5, 5.5)], "slice": {"index": i}}
                  for i in range(4)]
    delays = np.linspace(0.01, 0.2, 4)

    for source in ("area", "amplitude"):
        with pytest.raises(ValueError, match="NMR2D_fit_slices"):
            relaxation.NMR2D_fit_relaxation(integrated, model="exponential_decay", vdlist=delays,
                                 value_source=source, show_report=False)


def test_value_source_integral_still_resolves_the_old_way():
    """The historical name meant whichever of the two the slices
    carried; it keeps doing that so existing callers are unaffected."""
    fits, delays = _fitted_slices()
    old = relaxation.NMR2D_fit_relaxation(fits, model="exponential_decay", vdlist=delays,
                                value_source="integral", show_report=False)
    new = relaxation.NMR2D_fit_relaxation(fits, model="exponential_decay", vdlist=delays,
                                value_source="area", show_report=False)
    assert old[0]["params"]["T2"] == pytest.approx(new[0]["params"]["T2"])


def test_value_source_rejects_an_unknown_name():
    fits, delays = _fitted_slices()
    with pytest.raises(ValueError, match="value_source must be one of"):
        relaxation.NMR2D_fit_relaxation(fits, model="exponential_decay", vdlist=delays,
                             value_source="nonsense", show_report=False)


# ============================================================
# Curve fit export: the fitted parameters, not just the series
# ============================================================

def test_curve_fit_parameters_to_csv_reports_the_fitted_values_and_quality():
    """Explicit user request: "Export curve fit needs to include the
    fit report parameters, like D/T1/T2 and the fit quality"."""
    _t, results = _two_peak_curve_fit()
    rows = [row.split(",") for row in
            relaxation.curve_fit_parameters_to_csv(results).splitlines()]
    header, body = rows[0], rows[1:]

    assert header[:2] == ["peak", "model"]
    assert "T2 (s)" in header and "T2 error (s)" in header
    assert "R^2" in header and "reduced chi^2" in header and "points" in header
    assert [row[0] for row in body] == ["A", "B"]

    # The values are the fit's own, not a re-derivation.
    t2 = float(body[0][header.index("T2 (s)")])
    assert t2 == pytest.approx(results[0]["params"]["T2"])
    assert float(body[0][header.index("points")]) == len(results[0]["t"])


def test_curve_fit_parameters_name_the_unit_of_a_diffusion_coefficient():
    """Raw values with the unit in the header -- a file that silently
    changed unit with a display combo would be unreadable later."""
    t = np.linspace(0, 4e9, 8)
    fits = [{"index": i + 1, "integrals": [float(10.0 * np.exp(-2.0e-9 * bi))]}
            for i, bi in enumerate(t)]
    results = relaxation.NMR2D_fit_relaxation(
        fits, model="stejskal_tanner", vdlist=t, show_report=False, peak_labels=["A"])
    header = relaxation.curve_fit_parameters_to_csv(results).splitlines()[0].split(",")
    assert "D (m^2/s)" in header
    assert "D error (m^2/s)" in header


def test_curve_fit_parameters_use_the_short_uncertainty_label():
    """The full describe_uncertainty dict is a paragraph for a report
    panel; the column only has to say which kind of error the +- is."""
    _t, results = _two_peak_curve_fit()
    rows = relaxation.curve_fit_parameters_to_csv(results).splitlines()
    header, first = rows[0].split(","), rows[1]
    assert header[-1] == "uncertainty"
    assert "description" not in first        # not the whole dict
    assert "σ" in first or "sigma" in first


def test_curve_fit_parameters_leave_a_gap_where_a_peak_lacks_one():
    """Columns are the union over peaks, so a multi-component fit and a
    single-exponential one can share a file -- an absent parameter is
    an empty cell, not a zero."""
    _t, results = _two_peak_curve_fit()
    results[1]["params"] = {k: v for k, v in results[1]["params"].items() if k != "T2"}
    results[1]["errors"] = {k: v for k, v in results[1].get("errors", {}).items() if k != "T2"}
    rows = [row.split(",") for row in
            relaxation.curve_fit_parameters_to_csv(results).splitlines()]
    column = rows[0].index("T2 (s)")
    assert rows[1][column] != ""
    assert rows[2][column] == ""


def test_curve_fit_to_csv_puts_the_parameters_before_the_series():
    """Two labelled sections: the parameters are per peak and the
    series is per point, so they have different row meanings. The
    titles are what let a reader tell which is which."""
    _t, results = _two_peak_curve_fit()
    text = relaxation.curve_fit_to_csv(results, x_label="delay (s)")
    lines = text.splitlines()

    assert lines[0] == "Curve fit parameters"
    series_at = lines.index("Curve fit series")
    assert lines[series_at - 1] == ""                 # blank line between them
    assert lines[series_at + 1].startswith("delay (s),")
    # Both sections really carry their own content.
    assert any("T2 (s)" in line for line in lines[:series_at])
    assert len(lines) - series_at - 2 == len(results[0]["t"])


def test_curve_fit_to_csv_can_still_produce_the_bare_series():
    _t, results = _two_peak_curve_fit()
    text = relaxation.curve_fit_to_csv(results, include_parameters=False)
    assert not text.startswith("Curve fit parameters")
    assert text.splitlines()[0].startswith("x,")


# ============================================================
# Decay fraction: the quality number R² cannot give you
# ============================================================

def test_decay_fraction_is_last_over_first():
    assert relaxation.decay_fraction([10.0, 5.0, 2.0]) == pytest.approx(0.2)
    assert relaxation.decay_fraction(np.array([4.0, 1.0])) == pytest.approx(0.25)


def test_decay_fraction_keeps_the_sign_of_an_inversion():
    """An inversion recovery starts negative and ends positive, so a
    negative fraction says the inversion worked. Forcing it positive
    would throw that away."""
    assert relaxation.decay_fraction([-10.0, 0.0, 9.0]) == pytest.approx(-0.9)


def test_decay_fraction_is_nan_where_it_is_undefined():
    """An undefined ratio reported as such, not as a misleading
    infinity."""
    assert np.isnan(relaxation.decay_fraction([0.0, 5.0]))
    assert np.isnan(relaxation.decay_fraction([5.0]))
    assert np.isnan(relaxation.decay_fraction([]))


def test_decay_fraction_catches_a_series_that_barely_decayed():
    """**The reason it exists.** Both of these fit perfectly -- R² is
    1.000000 for each -- but one decayed to 9% and the other only to
    98%, and the second constrains its D hardly at all. R² cannot tell
    them apart; this does."""
    b = np.linspace(0, 4e8, 8)

    def fit(rate, label):
        fits = [{"index": i + 1, "integrals": [float(10.0 * np.exp(-rate * bi))]}
                for i, bi in enumerate(b)]
        return relaxation.NMR2D_fit_relaxation(
            fits, model="stejskal_tanner", vdlist=b,
            show_report=False, peak_labels=[label])[0]

    strong, weak = fit(6e-9, "strong"), fit(5e-11, "weak")
    assert strong["fit_quality"]["r_squared"] == pytest.approx(1.0, abs=1e-6)
    assert weak["fit_quality"]["r_squared"] == pytest.approx(1.0, abs=1e-6)

    assert strong["fit_quality"]["decay_fraction"] < 0.15
    assert weak["fit_quality"]["decay_fraction"] > 0.95


def test_fit_report_states_the_decay_fraction():
    _t, results = _two_peak_curve_fit()
    report = relaxation.format_relaxation_report(results[0], width=60)
    assert "final I/I" in report
    fraction = results[0]["fit_quality"]["decay_fraction"]
    assert f"{fraction:.4g}" in report


def test_curve_fit_export_carries_the_decay_fraction():
    _t, results = _two_peak_curve_fit()
    rows = [row.split(",") for row in
            relaxation.curve_fit_parameters_to_csv(results).splitlines()]
    column = rows[0].index("final I/I0")
    assert float(rows[1][column]) == pytest.approx(
        results[0]["fit_quality"]["decay_fraction"])


def test_joint_report_gives_the_decay_fraction_per_curve():
    """**Per curve, not over the joint series.** A joint fit's own
    y-vector is every dataset concatenated, so "last over first" across
    it would divide one curve's end by another curve's start."""
    b = np.linspace(0, 4e8, 8)
    result = relaxation.fit_relaxation_global(
        [{"t": b, "y": 10.0 * np.exp(-6e-9 * b), "label": "fast"},
         {"t": b, "y": 5.0 * np.exp(-5e-11 * b), "label": "slow"}],
        model="stejskal_tanner", shared=["D"])
    report = relaxation.format_global_report(result, width=64)

    line = next(ln for ln in report.splitlines() if "I/I" in ln)
    assert "fast" in line and "slow" in line
    # Each curve's own attenuation, not one number over the pair.
    assert result["per_dataset"][0]["fit_quality"]["decay_fraction"] < 0.15
    assert result["per_dataset"][1]["fit_quality"]["decay_fraction"] > 0.95
