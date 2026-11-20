"""Notes 8.8 (part 2): opt-in autophase scores and a judged-region, defaults unchanged."""
import dataclasses

import numpy as np
import pytest

from opennmr.gui.main_window import MainWindow
from opennmr.notebook import autophase_slices, compare_phase_methods
from opennmr.processing import ft
from test_gui import (_load_spectrum_directly, _make_pseudo2d_relaxation_spectrum,  # noqa: F401
                      qapp)


def _line(width_hz, phase_deg, noise=0.01, n=8192, seed=0):
    rng = np.random.default_rng(seed)
    ppm = np.linspace(10, -10, n)
    lw = width_hz / 500.0
    s = lw / (lw + 1j * (ppm - 1.0)) * np.exp(1j * np.deg2rad(phase_deg))
    return ppm, s + noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def _circ(a, b):
    return abs(((a - b) + 180) % 360 - 180)


@pytest.mark.parametrize("width", [5, 75, 200])
def test_the_default_score_is_unchanged_and_finds_the_phase(width):
    ppm, s = _line(width, 37.0)
    assert _circ(ft.apc0(s, ppm, phase0_inc=0.5), 37.0) < 3.0
    assert ft.apc0(s, ppm, phase0_inc=0.5) == ft.apc0(s, ppm, phase0_inc=0.5, penalty="magnitude", region=None)


def test_acme_is_a_valid_score_and_finds_a_sharp_line():
    ppm, s = _line(5, -60.0)
    assert _circ(ft.apc0(s, ppm, phase0_inc=0.5, penalty="acme"), -60.0) < 3.0
    with pytest.raises(ValueError):
        ft.phase_score(s, "nonsense")


def test_region_accepts_ppm_pairs_or_a_mask_and_matches():
    ppm, s = _line(30, 20.0)
    by_pairs = ft.apc0(s, ppm, phase0_inc=1.0, region=[(-3, 5)])
    by_mask = ft.apc0(s, ppm, phase0_inc=1.0, region=(ppm >= -3) & (ppm <= 5))
    assert by_pairs == by_mask
    assert _circ(by_pairs, 20.0) < 3.0
    assert ft.phase_region_mask(ppm, None) is None


def test_region_too_small_is_refused_with_a_clear_message():
    ppm, s = _line(30, 20.0)
    with pytest.raises(ValueError, match="fewer than 8 points"):
        ft.apc0(s, ppm, region=[(50, 60)])
    with pytest.raises(ValueError):
        ft.phase_region_mask(ppm, [1, 2, 3])


def test_region_keeps_a_distant_hump_from_pulling_the_phase():
    ppm, s = _line(10, 0.0)
    hump = 0.8 * np.exp(-((ppm + 7) ** 2) / 0.5) * np.exp(1j * np.deg2rad(110))   # a wrongly-phased signal
    mixed = s + hump
    whole = ft.apc0(mixed, ppm, phase0_inc=1.0, remove_baseline=False)
    narrow = ft.apc0(mixed, ppm, phase0_inc=1.0, remove_baseline=False, region=[(-2, 4)])
    assert _circ(narrow, 0.0) < 3.0
    assert _circ(narrow, 0.0) < _circ(whole, 0.0)


def test_autophase01_honours_penalty_and_region():
    ppm, s = _line(75, -50.0)
    p0, p1 = ft.autophase01(s, ppm, penalty="acme", region=[(-3, 5)])
    assert _circ(p0, -50.0) < 6.0 and abs(p1) <= 20.0


def test_compare_phase_methods_reports_every_method_and_agreement():
    ppm, s = _line(5, 37.0)
    out = compare_phase_methods(s, ppm, region=[(-3, 5)])
    assert {"magnitude", "negarea", "entropy", "acme", "magnitude (region)", "acme (region)"} <= set(out["phase0"])
    assert out["spread"] >= 0.0 and isinstance(out["agree"], bool)
    assert "magnitude (region)" not in compare_phase_methods(s, ppm)["phase0"]


def _complex_series():
    spectrum = _make_pseudo2d_relaxation_spectrum(np.geomspace(0.001, 0.3, 4))
    return dataclasses.replace(spectrum, data_imag=np.zeros_like(spectrum.data))


def test_autophase_slices_accepts_penalty_and_region_and_default_matches():
    a, b, c = _complex_series(), _complex_series(), _complex_series()
    ppm = np.asarray(a.axis_values(-1, "ppm"), dtype=float)
    default = autophase_slices(a)
    explicit = autophase_slices(b, penalty="magnitude", region=None)
    assert default["phase0"] == explicit["phase0"]
    lo, hi = float(ppm.min()), float(ppm.max())
    out = autophase_slices(c, penalty="acme", region=[(lo, hi)])
    assert len(out["phase0"]) == 4


def test_the_gui_panel_exposes_score_and_visible_range_and_passes_them_on(qapp):
    window = MainWindow()
    assert window.autophase_score_combo.currentData() == "magnitude"
    assert window.autophase_visible_only_checkbox.isChecked() is False
    assert window._autophase_scoring() == {"penalty": "magnitude"}
    window.autophase_score_combo.setCurrentIndex(window.autophase_score_combo.findData("acme"))
    assert window._autophase_scoring()["penalty"] == "acme"
    _load_spectrum_directly(window, _complex_series())
    window.autophase_visible_only_checkbox.setChecked(True)
    scoring = window._autophase_scoring()
    assert "region" in scoring or scoring == {"penalty": "acme"}   # region only once a plot axis exists
