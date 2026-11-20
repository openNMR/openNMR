"""PFG notes §4 and §5: shared-shape lineshape fits, cross-talk flags, SCORE consistency
checks, row alignment and non-negative series fits."""
import math

import numpy as np
import pytest

from opennmr.notebook import pfg_lineshapes as L
from opennmr.notebook import pfg_score_check as S
from opennmr.notebook.series_fitting import _nonnegative_coefficients


def _series(n_rows=12, seed=0, noise=0.002, membrane_D=1.0e-10, bulk_D=2.0e-9):
    rng = np.random.default_rng(seed)
    ppm = np.linspace(7.0, 3.0, 800)                       # descending, like an NMR axis
    b = np.linspace(0.0, 4.0e9, n_rows)
    bulk = L.pseudo_voigt(ppm, 4.7, 0.15, 0.7)
    mem = L.pseudo_voigt(ppm, 5.1, 0.3, 0.5)
    rows = []
    for bi in b:
        rows.append(1.0 * np.exp(-bulk_D * bi) * bulk + 0.4 * np.exp(-membrane_D * bi) * mem
                    + 0.01 + rng.normal(0, noise, ppm.size))
    return ppm, b, np.array(rows)


PEAKS = [{"position": 4.72, "fwhm": 0.2, "eta": 0.5, "label": "bulk"},
         {"position": 5.05, "fwhm": 0.25, "eta": 0.5, "label": "membrane"}]


def test_pseudo_voigt_has_unit_height_and_the_right_width():
    x = np.linspace(-1, 1, 2001)
    for eta in (0.0, 0.5, 1.0):
        y = L.pseudo_voigt(x, 0.0, 0.2, eta)
        assert y.max() == pytest.approx(1.0)
        half = x[y >= 0.5]
        assert (half.max() - half.min()) == pytest.approx(0.2, abs=2e-3)


def test_shared_shapes_recover_positions_and_decays():
    ppm, b, rows = _series()
    r = L.fit_series_shared_shapes(ppm, rows, PEAKS)
    assert r["success"] and r["amplitudes"].shape == (12, 2)
    assert r["positions"] == pytest.approx([4.7, 5.1], abs=0.03)
    assert np.all(r["amplitudes"] >= 0)
    assert r["areas"][0, 0] > 3 * r["areas"][-1, 0]                # bulk decays fast
    ratio = r["areas"][-1, 1] / r["areas"][0, 1]
    assert ratio == pytest.approx(np.exp(-1.0e-10 * b[-1]), rel=0.15)     # membrane slowly
    assert "Shared-shape fit" in r["log"] and "NNLS" in r["log"]


def test_area_per_height_matches_closed_forms():
    w = 0.3
    assert L._area_per_height(w, 0.0) == pytest.approx(w * math.sqrt(math.pi / (4 * math.log(2))))
    assert L._area_per_height(w, 1.0) == pytest.approx(w * math.pi / 2)
    # numerical check on a wide window for the mixed shape (tails included)
    x = np.linspace(-400, 400, 800001)
    num = float(np.trapezoid(L.pseudo_voigt(x, 0.0, w, 0.5), x))
    assert num == pytest.approx(L._area_per_height(w, 0.5), rel=2e-3)


def test_areas_are_amplitude_times_the_shape_area():
    ppm, b, rows = _series()
    r = L.fit_series_shared_shapes(ppm, rows, PEAKS)
    for k in range(2):
        expect = r["amplitudes"][:, k] * L._area_per_height(r["fwhm"][k], r["eta"][k])
        assert r["areas"][:, k] == pytest.approx(expect)
    assert np.all(r["area_errors"] > 0)


def test_the_membrane_peak_can_be_held_downfield_of_the_bulk():
    ppm, b, rows = _series()
    start = [dict(PEAKS[0]), dict(PEAKS[1], position=4.6)]         # started on the wrong side
    free = L.fit_series_shared_shapes(ppm, rows, start)
    held = L.fit_series_shared_shapes(ppm, rows, start, anchors={1: 0})
    assert held["positions"][1] >= held["positions"][0]
    assert "downfield" in held["log"]
    assert held["rms"] <= free["rms"] * 3 + 1e-9                   # and still fits


def test_anchors_are_validated():
    ppm, b, rows = _series(n_rows=4)
    with pytest.raises(ValueError):
        L.fit_series_shared_shapes(ppm, rows, PEAKS, anchors={0: 0})
    with pytest.raises(ValueError):
        L.fit_series_shared_shapes(ppm, rows, PEAKS, anchors={0: 1, 1: 0})
    with pytest.raises(ValueError):
        L.fit_series_shared_shapes(ppm, rows[:, :10], PEAKS)
    with pytest.raises(ValueError):
        L.fit_series_shared_shapes(ppm, rows, [])


def test_window_restricts_the_fit():
    ppm, b, rows = _series(n_rows=4)
    r = L.fit_series_shared_shapes(ppm, rows, PEAKS, window=(4.0, 6.0))
    assert r["x"].min() >= 4.0 and r["x"].max() <= 6.0


def test_nonnegative_coefficients_clip_a_negative_site_and_keep_the_baseline_free():
    x = np.linspace(-1, 1, 60)
    c1, c2 = np.exp(-(x / 0.2) ** 2), np.exp(-((x - 0.5) / 0.2) ** 2)
    y = 1.0 * c1 - 0.05 * c2 - 0.3 + 0.1 * x
    design = np.column_stack([c1, c2, np.ones_like(x), x])
    free = np.linalg.lstsq(design, y, rcond=None)[0]
    nn = _nonnegative_coefficients(design, y, 2)
    assert free[1] < 0 and nn[1] == 0.0
    assert nn[2] < 0                                               # baseline may go negative


def test_crosstalk_is_flagged_when_a_peak_carries_another_peaks_decay():
    b = np.linspace(0.0, 4.0e9, 16)
    bulk = np.exp(-2.0e-9 * b)
    membrane = 0.7 * np.exp(-1.0e-10 * b) + 0.3 * np.exp(-2.0e-9 * b)       # contaminated
    areas = np.column_stack([bulk, membrane])
    result = {"areas": areas, "area_errors": np.full_like(areas, 0.002),
              "labels": ["bulk", "membrane"]}
    flags = L.flag_crosstalk(result, b)
    assert any(f["peak"] == "membrane" and f["other"] == "bulk" for f in flags)
    assert flags[0]["fraction"] == pytest.approx(0.3, abs=0.1)


def test_no_crosstalk_flag_for_clean_peaks():
    b = np.linspace(0.0, 4.0e9, 16)
    areas = np.column_stack([np.exp(-2.0e-9 * b), 0.4 * np.exp(-1.0e-10 * b)])
    result = {"areas": areas, "area_errors": np.full_like(areas, 0.002), "labels": ["a", "b"]}
    assert L.flag_crosstalk(result, b) == []


# --- SCORE consistency ---------------------------------------------------------------------

def _gauss(x, c, w):
    return np.exp(-((x - c) / w) ** 2)


def test_overlap_shares():
    x = np.linspace(0, 10, 400)
    both = np.vstack([_gauss(x, 4.5, 1.0), _gauss(x, 5.5, 1.0)])
    assert S.overlap_shares(both) == pytest.approx([0.5, 0.5], abs=0.02)
    apart = np.vstack([_gauss(x, 1.0, 0.2), _gauss(x, 9.0, 0.2)])
    assert np.all(S.overlap_shares(apart) == 0.0)
    assert np.all(S.overlap_shares(np.zeros((2, 5))) == 0.0)


def _integral_fit(D1, D2):
    return {"param_names": ["I1", "D1", "I2", "D2"],
            "params": {"I1": 1.0, "D1": D1, "I2": 0.3, "D2": D2}}


def test_a_water_D_that_disagrees_with_the_integral_fit_is_flagged():
    x = np.linspace(0, 10, 300)
    spectra = np.vstack([_gauss(x, 2.0, 0.3), _gauss(x, 8.0, 0.3)])
    good = S.check_score_result({"D": np.array([2.05e-9, 1.0e-10]), "spectra": spectra},
                                _integral_fit(2.0e-9, 1.0e-10))
    assert good["ok"] and good["flags"] == []
    bad = S.check_score_result({"D": np.array([2.5e-9, 1.0e-10]), "spectra": spectra},
                               _integral_fit(2.0e-9, 1.0e-10))
    assert not bad["ok"] and "water-like D" in bad["flags"][0]
    assert bad["water_difference"] == pytest.approx(0.25)


def test_a_split_line_is_flagged():
    x = np.linspace(0, 10, 400)
    split = np.vstack([_gauss(x, 5.0, 1.0), 0.9 * _gauss(x, 5.2, 1.0)])      # one line, two parts
    out = S.check_score_result({"D": np.array([2.0e-9, 1.8e-9]), "spectra": split})
    assert any("overlapping signal" in f for f in out["flags"])
    assert np.isnan(out["water_difference"])


def test_align_rows_removes_a_frequency_drift():
    ppm = np.linspace(8.0, 2.0, 1200)
    step = abs(ppm[1] - ppm[0])
    ref = _gauss(ppm, 5.0, 0.15)
    drifted = _gauss(ppm, 5.0 + 6 * step, 0.15)
    aligned, shifts = S.align_rows(ppm, np.vstack([ref, drifted]))
    assert np.allclose(aligned[0], ref)
    assert np.max(np.abs(aligned[1] - ref)) < 0.1
    assert abs(abs(shifts[1]) - 6 * step) < 1.5 * step and shifts[0] == 0.0


def test_score_with_check_runs_each_k():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 300)
    b = np.linspace(0.0, 4.0e9, 12)
    peaks = np.vstack([_gauss(x, 2.5, 0.3), _gauss(x, 7.0, 0.3)])
    decays = np.column_stack([np.exp(-2.0e-9 * b), np.exp(-2.0e-10 * b)])
    data = decays @ peaks + rng.normal(0, 0.002, (12, x.size))
    out = S.score_with_check(b, data, integral_fit=_integral_fit(2.0e-9, 2.0e-10), ks=(2, 3))
    assert [o["k"] for o in out] == [2, 3]
    assert out[0]["result"]["spectra"].shape == (2, x.size)
    assert set(out[0]["check"]) >= {"ok", "flags", "D_score", "water_difference"}
    assert out[0]["check"]["D_score"].max() == pytest.approx(2.0e-9, rel=0.3)


def test_the_new_functions_are_in_the_notebook_namespace():
    import opennmr.notebook as nb
    for name in ("fit_series_shared_shapes", "flag_crosstalk", "align_rows",
                 "check_score_result", "score_with_check"):
        assert callable(getattr(nb, name))
