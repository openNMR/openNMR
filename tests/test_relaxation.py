import numpy as np
import pytest

from nmrsuite.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from nmrsuite.processing import relaxation, slicing
from nmrsuite.processing.fitting import gaussian


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


def test_relaxation_models_registry_contains_expected_keys():
    assert set(relaxation.RELAXATION_MODELS) == {
        "saturation_recovery", "inversion_recovery", "exponential_decay",
        "biexponential_decay", "biexponential_recovery",
        "stretched_decay", "stretched_recovery",
        "stejskal_tanner", "stejskal_tanner_stretched",
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
    assert fits[0]["peak_label"] == "Peak 1"
