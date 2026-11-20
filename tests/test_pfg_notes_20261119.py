"""PFG improvement notes (2026-11-19): diffusion offset, automatic model choice, error
helpers, quality checks, NMRSpectrum slices, zero-filled noise, and the visible Compare
multiply row / expanding search box."""
from types import SimpleNamespace

import numpy as np
import pytest

from opennmr.notebook import pfg_selection as P
from opennmr.notebook.two_d import NMR2D_get_slices, dataset_from_spectrum
from opennmr.processing import diffusion as D
from opennmr.processing import relaxometry
from opennmr.processing.relaxation import (UNCERTAINTY_KINDS, fit_curve_uncertainty,
                                           fit_relaxation)


def _series(offset=0.0, seed=3, n=16, bmax=5.6e9, noise=0.002):
    b = np.linspace(0.0, bmax, n)
    rng = np.random.default_rng(seed)
    y = 1.0 * np.exp(-2.0e-9 * b) + 0.4 * np.exp(-1.2e-10 * b) + offset
    return b, y + rng.normal(0, noise, n), np.full(n, noise)


# --- constant offset ------------------------------------------------------------------

def test_offset_model_has_a_C_parameter_and_adds_it():
    spec = D.build_diffusion_model(2, offset=True)
    assert spec["params"][-1] == "C" and spec["offset"] is True
    b = np.array([0.0, 1e9])
    values = [1.0, 1e-9, 0.5, 1e-10, 0.1]
    plain = D.build_diffusion_model(2)["func"](b, *values[:4])
    assert np.allclose(spec["func"](b, *values), plain + 0.1)
    lo, hi = spec["bounds"](b, np.array([1.0, 0.5]))
    assert lo[-1] == 0.0 and np.isinf(hi[-1])


def test_offset_default_is_unchanged():
    assert "C" not in D.build_diffusion_model(2)["params"]


def test_component_weights_include_the_offset_in_the_total():
    w = D.component_weights({"I1": 0.6, "I2": 0.3, "C": 0.1}, 2)
    assert w == pytest.approx([0.6, 0.3])
    assert D.component_weights({"I1": 0.6, "I2": 0.3}, 2) == pytest.approx([2 / 3, 1 / 3])
    assert D.component_weights({"I0": 1.0, "C": 1.0}, 1) == pytest.approx([0.5])
    assert D.component_weights({"I0": 1.0}, 1) == [1.0]


def test_a_fit_with_an_offset_recovers_the_plateau():
    # one fast component that has fully decayed by b_max, so the plateau is identifiable (with a
    # slow second component the plateau and its tail are degenerate)
    b = np.linspace(0.0, 5.6e9, 16)
    y = np.exp(-2.0e-9 * b) + 0.01 + np.random.default_rng(4).normal(0, 0.0005, 16)
    fit = fit_relaxation(b, y, y_err=np.full(16, 0.0005),
                         custom_model=D.build_diffusion_model(1, offset=True, expected_D=2e-9))
    assert fit["params"]["C"] == pytest.approx(0.01, abs=0.002)


# --- model selection --------------------------------------------------------------------

def test_information_criteria():
    assert P.aicc(10.0, 16, 4) == pytest.approx(16 * np.log(10 / 16) + 8 + 2 * 4 * 5 / 11)
    assert P.aicc(10.0, 5, 4) == float("inf")                  # n - k - 1 <= 0
    assert P.bic(10.0, 16, 4) == pytest.approx(16 * np.log(10 / 16) + 4 * np.log(16))


def test_selection_finds_two_components_and_lists_every_fit():
    b, y, sig = _series()
    r = P.select_diffusion_model(b, y, sig, n_max=2)
    assert len(r["rows"]) == 4
    assert r["selected"]["n_components"] == 2
    assert r["fit"] is r["selected"]["fit"]
    assert r["rows"][0]["aicc"] <= r["rows"][-1]["aicc"]        # ranked
    D_found = sorted(r["fit"]["params"][d] for d in ("D1", "D2"))
    assert D_found[1] == pytest.approx(2.0e-9, rel=0.15)
    assert "SELECTED" in r["log"] and "Rule" in r["log"]


def test_an_insignificant_offset_is_rejected_with_a_reason():
    b = np.linspace(0.0, 5.6e9, 16)
    y = np.exp(-1.0e-9 * b) + np.random.default_rng(5).normal(0, 0.002, 16)
    r = P.select_diffusion_model(b, y, np.full(16, 0.002), n_max=1, offsets=(True,))
    row = r["rows"][0]
    assert not row["valid"]
    assert any("C =" in reason for reason in row["reasons"])
    assert r["selected"] is None and "No fit passed" in r["log"]


def test_prefer_simpler_needs_a_big_aicc_gain():
    b, y, sig = _series()
    r = P.select_diffusion_model(b, y, sig, n_max=2, offsets=(False,), delta_aicc=1e9)
    assert r["selected"]["n_components"] == 1
    free = P.select_diffusion_model(b, y, sig, n_max=2, offsets=(False,), prefer_simpler=False)
    assert free["selected"]["n_components"] == 2


def test_selection_validates_its_input():
    with pytest.raises(ValueError):
        P.select_diffusion_model([1, 2, 3], [1, 2])


# --- errors -----------------------------------------------------------------------------

def test_inflated_errors_scale_by_the_birge_ratio():
    b, y, _ = _series(noise=0.02)
    fit = fit_relaxation(b, y, y_err=np.full(b.size, 0.002),          # errors 10x too small
                         custom_model=D.build_diffusion_model(2, expected_D=[2e-9, 1e-10]))
    out = P.inflated_errors(fit)
    red = fit["fit_quality"]["reduced_chi2"]
    assert red > 1 and out["scale"] == pytest.approx(np.sqrt(red))
    assert out["errors"]["D1"] == pytest.approx(fit["errors"]["D1"] * out["scale"])
    assert "Birge" in out["label"]


def test_inflated_errors_leave_scatter_based_errors_alone():
    b, y, _ = _series()
    fit = fit_relaxation(b, y, custom_model=D.build_diffusion_model(2, expected_D=[2e-9, 1e-10]))
    assert P.inflated_errors(fit)["scale"] == 1.0


def test_the_new_uncertainty_kind_is_documented():
    assert "measured_inflated" in UNCERTAINTY_KINDS
    assert "χ²" in UNCERTAINTY_KINDS["measured_inflated"]["short"]


def test_component_weights_with_errors_match_numerical_propagation():
    b, y, sig = _series()
    fit = fit_relaxation(b, y, y_err=sig,
                         custom_model=D.build_diffusion_model(2, expected_D=[2e-9, 1e-10]))
    out = P.component_weights_with_errors(fit)
    assert sum(out["weights"].values()) == pytest.approx(1.0)
    cov = np.asarray(fit["pcov"])
    free = list(fit["free_names"])
    i1, i2 = fit["params"]["I1"], fit["params"]["I2"]
    t = i1 + i2
    grad = np.zeros(len(free))
    grad[free.index("I1")] = (t - i1) / t ** 2
    grad[free.index("I2")] = -i1 / t ** 2
    assert out["errors"]["I1"] == pytest.approx(float(np.sqrt(grad @ cov @ grad)), rel=1e-6)
    assert out["offset_fraction"] is None


def test_component_weights_report_the_offset_fraction():
    b, y, sig = _series(offset=0.05, noise=0.0005)
    fit = fit_relaxation(b, y, y_err=np.full(b.size, 0.0005),
                         custom_model=D.build_diffusion_model(2, offset=True,
                                                              expected_D=[2e-9, 1e-10]))
    out = P.component_weights_with_errors(fit)
    assert out["offset_fraction"] == pytest.approx(0.05 / 1.45, abs=0.01)
    assert sum(out["weights"].values()) == pytest.approx(1.0)


def test_parameter_ratio_error():
    b, y, sig = _series()
    fit = fit_relaxation(b, y, y_err=sig,
                         custom_model=D.build_diffusion_model(2, expected_D=[2e-9, 1e-10]))
    r = P.parameter_ratio(fit, "D2", "D1")
    assert r["ratio"] == pytest.approx(fit["params"]["D2"] / fit["params"]["D1"])
    assert r["error"] > 0


def test_curve_uncertainty_scaling_widens_a_misfit_band():
    b, y, _ = _series(noise=0.02)
    fit = fit_relaxation(b, y, y_err=np.full(b.size, 0.002),
                         custom_model=D.build_diffusion_model(2, expected_D=[2e-9, 1e-10]))
    _, _, band = fit_curve_uncertainty(fit)
    _, _, scaled = fit_curve_uncertainty(fit, scale_by_chi2=True)
    assert np.all(scaled >= band - 1e-15) and scaled.max() > band.max()


# --- combining measurements ------------------------------------------------------------------

def test_combine_measurements_averages_repeats_then_samples():
    out = P.combine_measurements([1.0, 1.2, 2.0, 2.2], [0.1] * 4, ["a", "a", "b", "b"])
    assert out["mean"] == pytest.approx(1.6)
    sem = np.std([1.1, 2.1], ddof=1) / np.sqrt(2)
    fit_part = np.sqrt(2 * (np.sqrt(0.02) / 2) ** 2) / 2
    assert out["error"] == pytest.approx(np.sqrt(sem ** 2 + fit_part ** 2))
    assert out["n_samples"] == 2 and out["note"] == ""


def test_combine_measurements_single_sample_uses_the_pooled_spread():
    out = P.combine_measurements([1.0, 1.1, 2.0, 2.2, 5.0], [0.0] * 5,
                                 ["a", "a", "b", "b", "c"])
    single = P.combine_measurements([5.0], [0.0], ["c"])
    assert single["note"] == "single measurement: only the fit error is available"
    assert out["n_samples"] == 3
    assert P.combine_measurements([3.0, 3.3], None, ["x", "x"])["n_samples"] == 1
    assert "pooled" in P.combine_measurements([3.0, 3.3], [0.1, 0.1], ["x", "x"])["note"]


# --- quality checks ----------------------------------------------------------------------------

def test_free_water_law_and_its_inverse():
    assert float(P.free_water_D(298.15)) == pytest.approx(2.3e-9, rel=0.02)
    for T in (280.0, 295.15, 310.0):
        assert float(P.water_temperature_from_D(P.free_water_D(T))) == pytest.approx(T)


def test_a_slow_water_implies_a_lower_temperature():
    assert float(P.water_temperature_from_D(2.0e-9)) < 298.15 - 5


def test_d2o_limit_scales_with_temperature():
    assert P.residual_d2o_limit(298.15) == pytest.approx(1.872e-9)
    assert P.residual_d2o_limit(310.0) > P.residual_d2o_limit(290.0)


def test_tail_residual_flags_a_missing_plateau():
    b = np.linspace(0.0, 5.6e9, 16)
    y = np.exp(-2.0e-9 * b) + 0.03 + np.random.default_rng(2).normal(0, 0.0005, 16)
    fit = fit_relaxation(b, y, y_err=np.full(16, 0.0005),
                         custom_model=D.build_diffusion_model(1, expected_D=2e-9))
    flag = P.tail_residual(fit)
    assert flag["value"] > 3 and flag["needs_offset"]
    with_c = fit_relaxation(b, y, y_err=np.full(16, 0.0005),
                            custom_model=D.build_diffusion_model(1, offset=True, expected_D=2e-9))
    assert not P.tail_residual(with_c)["needs_offset"]


def test_new_functions_are_importable_from_the_notebook_package():
    import opennmr.notebook as nb
    for name in ("select_diffusion_model", "inflated_errors", "component_weights_with_errors",
                 "combine_measurements", "free_water_D", "tail_residual"):
        assert callable(getattr(nb, name))


# --- NMRSpectrum slices ------------------------------------------------------------------------

def _fake_2d(n1=4, n2=64):
    x = np.linspace(10, -10, n2)
    data = np.outer(np.arange(1, n1 + 1), np.exp(-(x / 3) ** 2))
    axes = [SimpleNamespace(ppm=None, hz=None, index=np.arange(1, n1 + 1, dtype=float)),
            SimpleNamespace(ppm=x, hz=x * 400.0, index=None)]
    md = SimpleNamespace(pseudo_2d=True, homonuclear=False, nucleus="1H", nucleus_indirect=None,
                         sfo1=400.0, sfo1_indirect=None, source_path="x")
    return SimpleNamespace(data=data, axes=axes, metadata=md)


def test_dataset_from_spectrum_has_the_import_keys():
    ds = dataset_from_spectrum(_fake_2d())
    assert ds["spectrum"].shape == (4, 64) and ds["pseudo_2d"] is True
    assert ds["NUC2"] == "1H" and ds["SF2"] == 400.0
    assert list(ds["y_index"]) == [1.0, 2.0, 3.0, 4.0]
    assert dataset_from_spectrum(ds) is ds


def test_get_slices_accepts_a_spectrum_object():
    out = NMR2D_get_slices(_fake_2d(), rows=[1, 3])
    assert len(out["rows"]) == 2 and out["columns"] == []


def test_dataset_from_spectrum_needs_two_dimensions():
    bad = _fake_2d()
    bad.data = bad.data[0]
    with pytest.raises(ValueError):
        dataset_from_spectrum(bad)


# --- noise on a zero-filled spectrum --------------------------------------------------------------

def _zero_filled(factor=8, n=512, seed=0):
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1.0, n)
    x = np.linspace(0, n - 1, n)
    fine = np.linspace(0, n - 1, n * factor)
    return fine, np.interp(fine, x, white)                       # correlated neighbours


def test_noise_stride_recovers_the_noise_of_a_zero_filled_trace():
    _x, y = _zero_filled()
    small = relaxometry._robust_noise(y)
    right = relaxometry._robust_noise(y, 8)
    auto = relaxometry._robust_noise(y, "auto")
    assert small < 0.4 * right
    assert auto == pytest.approx(right, rel=0.35)


def test_white_noise_is_unchanged_by_the_default_stride():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1.0, 4000)
    assert relaxometry._robust_noise(y) == pytest.approx(1.0, rel=0.1)
    assert relaxometry._robust_noise(y, "auto") == pytest.approx(1.0, rel=0.15)


def test_find_integration_regions_accepts_noise_stride():
    x, y = _zero_filled()
    y = y.copy()
    y += 40.0 * np.exp(-((x - 256) / 6.0) ** 2)
    res = relaxometry.find_integration_regions(x, y, noise_stride=8)
    assert len(res["regions"]) >= 1 and res["noise"] > 0.5


def test_zero_fill_scales_the_integral_error():
    x = np.linspace(0, 1, 101)
    y = np.ones(101)
    value, e1 = relaxometry.integrate(x, y, noise=0.1)
    _, e4 = relaxometry.integrate(x, y, noise=0.1, zero_fill=4.0)
    _, e_low = relaxometry.integrate(x, y, noise=0.1, zero_fill=0.5)      # never reduced
    assert value == pytest.approx(1.0)
    assert e4 == pytest.approx(2.0 * e1)
    assert e_low == pytest.approx(e1)
    assert np.isnan(relaxometry.integrate(x, y, zero_fill=4.0)[1])


# --- GUI ------------------------------------------------------------------------------------------

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from opennmr.gui.main_window import MainWindow  # noqa: E402

from test_gui import qapp  # noqa: E402,F401 -- fixture


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def test_compare_has_a_visible_multiply_row(qapp, _dialogs):
    win = MainWindow()
    assert win.compare_factor_edit.placeholderText()
    assert win.compare_factor_apply_button.text() == "Apply"
    assert "×" in win.compare_factor_reset_button.text()
    win.compare_factor_edit.setText("2")
    win._on_compare_factor_apply()                      # nothing highlighted: says so, no crash
    assert _dialogs and _dialogs[-1][0] == "information"


def test_search_box_scales_with_the_window(qapp):
    from PySide6.QtWidgets import QSizePolicy
    win = MainWindow()
    edit = win.search_box.edit
    assert edit.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert edit.maximumWidth() >= 600 and edit.minimumWidth() >= 230
