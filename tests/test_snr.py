"""Signal-to-noise (notebook/snr.py): Apply ▸ Signal-to-Noise's engine
(explicit user request, 2026-10-12). Synthetic spectra with a known line
height and noise σ, so every number has an exact expectation."""
import math

import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import snr

X = np.linspace(100.0, -100.0, 8192)          # 0.0244 ppm per point
WIDTH = 0.5                                   # Gaussian σ of the line, ppm


def _y(height=50.0, sigma=1.0, seed=0, corr=0.0, slope=0.0, centre=0.0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=X.size)
    if corr:
        k = np.exp(-np.arange(-25, 26) ** 2 / (2 * corr ** 2))
        noise = np.convolve(noise, k / np.sqrt(np.sum(k ** 2)), "same")
    return height * np.exp(-0.5 * ((X - centre) / WIDTH) ** 2) + sigma * noise + slope * X


def _spectrum(ns=None, window=None, nucleus="7Li", **kw):
    raw = {}
    if window is not None:
        raw.update({"live_window": "exponential", "live_lb": window})
    meta = NMRMetadata(vendor="test", nucleus=nucleus, ns=ns, raw=raw)
    return NMRSpectrum(data=_y(**kw), axes=[NMRAxis(kind="frequency", ppm=X.copy())],
                       metadata=meta)


def test_known_height_and_sigma():
    r = snr.signal_to_noise((X, _y(seed=1)), (5, -5), (80, 40))
    assert r["noise_sigma"] == pytest.approx(1.0, rel=0.05)
    assert r["snr"] == pytest.approx(50.0, abs=3 * r["snr_error"])
    lo, hi = r["snr_interval"]
    assert lo < r["snr"] < hi
    assert r["sino"] == pytest.approx(r["snr"] / 2)
    assert r["position_ppm"] == pytest.approx(0.0, abs=0.2)
    assert r["detected"] and r["correlation_time"] == pytest.approx(1.0, abs=0.3)
    assert r["warnings"] == []


def test_accepts_spectrum_dict_and_pair_alike():
    y = _y(seed=2)
    a = snr.signal_to_noise((X, y), (5, -5), (80, 40))
    b = snr.signal_to_noise({"x_ppm": X, "real_spectrum": y, "nuclide": "7Li"}, (5, -5), (80, 40))
    c = snr.signal_to_noise(_spectrum(seed=2), (-5, 5), (40, 80))    # limits in either order
    assert a["snr"] == pytest.approx(b["snr"]) == pytest.approx(c["snr"])
    assert c["signal_region"] == (5.0, -5.0) and c["noise_region"] == [(80.0, 40.0)]


def test_interval_covers_the_true_value():
    """The stated 95 % interval holds the true S/N (50) at least 95 % of
    the time -- for white noise and for correlated noise."""
    for corr in (0.0, 3.0):
        hits = 0
        for seed in range(120):
            y = _y(seed=seed, corr=corr)
            r = snr.signal_to_noise((X, y), (5, -5), (80, 60))
            # the true σ of correlated noise is still 1 (kernel normalised)
            hits += r["snr_interval"][0] <= 50.0 <= r["snr_interval"][1]
        assert hits / 120 >= 0.93


def test_correlated_noise_is_measured_and_counted():
    """A Gaussian kernel of σ = 3 points has τ = 2√π·3 ≈ 10.6: the
    independent points, and so σ's own uncertainty, follow it."""
    r = snr.signal_to_noise((X, _y(seed=3, corr=3.0)), (5, -5), (80, 40))
    assert r["correlation_time"] == pytest.approx(2 * math.sqrt(math.pi) * 3, rel=0.25)
    assert r["n_noise_independent"] == pytest.approx(r["n_noise"] / r["correlation_time"])
    white = snr.signal_to_noise((X, _y(seed=3)), (5, -5), (80, 40))
    assert r["noise_sigma_error"] / r["noise_sigma"] > \
        2.5 * white["noise_sigma_error"] / white["noise_sigma"]


def test_a_linear_baseline_in_the_noise_is_removed():
    flat = snr.signal_to_noise((X, _y(seed=4)), (5, -5), (80, 40))
    tilted = snr.signal_to_noise((X, _y(seed=4, slope=0.1)), (5, -5), (80, 40))
    assert tilted["noise_sigma"] == pytest.approx(flat["noise_sigma"], rel=1e-6)
    only_mean = snr.signal_to_noise((X, _y(seed=4, slope=0.1)), (5, -5), (80, 40), detrend=0)
    assert only_mean["noise_sigma"] > 1.2 * flat["noise_sigma"]


def test_no_line_is_not_detected_at_the_stated_rate():
    """Pure noise in the signal region is "detected" about 5 % of the time."""
    false = sum(snr.signal_to_noise((X, _y(height=0.0, seed=s)), (5, -5), (80, 40))["detected"]
                for s in range(200))
    assert false / 200 < 0.12
    r = snr.signal_to_noise((X, _y(height=0.0, seed=7)), (5, -5), (80, 40))
    if not r["detected"]:
        assert any("no line is detected" in w for w in r["warnings"])


def test_detection_threshold_is_the_noise_maximum_quantile():
    from scipy.stats import norm
    assert snr.detection_threshold(1) == pytest.approx(norm.ppf(0.95))
    assert snr.detection_threshold(400) > snr.detection_threshold(10) > 2.0


def test_negative_and_absolute_lines():
    y = _y(height=-40.0, seed=5)
    neg = snr.signal_to_noise((X, y), (5, -5), (80, 40), sign="negative")
    assert neg["snr"] == pytest.approx(40, abs=4) and neg["integral_snr"] > 0
    ab = snr.signal_to_noise((X, y), (5, -5), (80, 40), sign="absolute")
    assert ab["snr"] == pytest.approx(neg["snr"])
    pos = snr.signal_to_noise((X, y), (5, -5), (80, 40))
    assert pos["snr"] < 5


def test_noise_region_checks():
    with pytest.raises(ValueError, match="overlaps"):
        snr.signal_to_noise((X, _y()), (5, -5), (3, -20))
    with pytest.raises(ValueError, match="at least"):
        snr.signal_to_noise((X, _y()), (5, -5), (80, 79.99))
    few = snr.signal_to_noise((X, _y()), (5, -5), (80, 79.5))
    assert any("independent noise points" in w for w in few["warnings"])
    # A second line inside the "noise" region is noticed.
    y = _y(seed=6) + 30 * np.exp(-0.5 * ((X - 60) / 0.3) ** 2)
    r = snr.signal_to_noise((X, y), (5, -5), (80, 40))
    assert any("noise region" in w and ("Gaussian" in w or "signal or a spike" in w)
               for w in r["warnings"])


def test_a_line_cut_by_the_signal_region_is_flagged():
    r = snr.signal_to_noise((X, _y(seed=8, centre=1.0)), (0.8, -5), (80, 40))
    assert any("edge of the signal region" in w for w in r["warnings"])


def test_several_nuclei_windows_and_integral():
    r = snr.signal_to_noise((X, _y(seed=9)), (5, -5), [(80, 60), (-60, -80)])
    assert len(r["noise_region"]) == 2 and r["n_noise"] > 1500
    # A Gaussian line's area / noise on the area, by hand.
    dx = abs(X[1] - X[0])
    n_sig = np.sum((X <= 5) & (X >= -5))
    expected = 50 * WIDTH * math.sqrt(2 * math.pi) / (r["noise_sigma"] * dx * math.sqrt(n_sig))
    assert r["integral_snr"] == pytest.approx(expected, rel=0.15)


def test_many_same_regions_and_scans_scaling():
    """Four times the scans: twice the S/N, and the comparison says that
    is what √NS predicts."""
    a = _spectrum(ns=16, height=25.0, seed=10)
    b = _spectrum(ns=64, height=50.0, seed=11)
    out = snr.signal_to_noise_many([a, b], (5, -5), (80, 40), labels=["16 scans", "64 scans"])
    assert not out["regions_differ"] and out["differing"] == []
    c = out["comparisons"][0]
    assert c["ratio"] == pytest.approx(2.0, rel=0.15)
    assert c["significant"] and c["expected_from_scans"] == pytest.approx(2.0)
    assert c["consistent_with_scans"]
    assert out["rows"][1]["snr_per_sqrt_scan"] == pytest.approx(out["rows"][1]["snr"] / 8)
    text = snr.snr_report(out)
    assert "Same regions for every spectrum" in text and "√(NS ratio) predicts 2" in text


def test_many_with_own_regions_is_flagged_everywhere():
    a, b, c = (_spectrum(seed=s) for s in (12, 13, 14))
    out = snr.signal_to_noise_many([a, b, c], (5, -5), (80, 40), labels=["a", "b", "c"],
                                   regions={"c": {"noise": (-40, -80)}})
    assert out["regions_differ"] and out["differing"] == ["c"]
    assert out["rows"][2]["noise_region"] == [(-40.0, -80.0)]
    assert out["rows"][2]["signal_region"] == (5.0, -5.0)       # shared signal kept
    assert out["comparisons"][1]["regions_differ"]
    text = snr.snr_report(out)
    assert "REGIONS DIFFERED BETWEEN SPECTRA: c" in text and "c †:" in text
    assert any("not strictly comparable" in n for n in out["notes"])
    table = snr.snr_table(out)
    assert [t["own regions"] for t in table] == [False, False, True]
    # The same regions given explicitly are not "own".
    same = snr.signal_to_noise_many([a, b], (5, -5), (80, 40),
                                    regions={1: {"signal": (-5, 5), "noise": (40, 80)}})
    assert not same["regions_differ"]


def test_many_notes_different_apodisation_and_nuclei():
    a = _spectrum(seed=15, window=5.0)
    b = _spectrum(seed=16, window=50.0, nucleus="6Li")
    out = snr.signal_to_noise_many([a, b], (5, -5), (80, 40))
    joined = " ".join(out["notes"])
    assert "apodised differently" in joined and "different nuclei" in joined


def test_same_spectrum_twice_is_not_significantly_different():
    a = _spectrum(seed=17)
    out = snr.signal_to_noise_many([a, a], (5, -5), (80, 40))
    assert out["comparisons"][0]["ratio"] == pytest.approx(1.0)
    assert not out["comparisons"][0]["significant"]


def test_report_csv_plot_and_code(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    a, b = _spectrum(seed=18), _spectrum(seed=19, height=20.0)
    out = snr.signal_to_noise_many([a, b], (5, -5), (80, 40), labels=["a", "b"],
                                   regions={1: {"signal": (3, -3)}})
    path = tmp_path / "snr.csv"
    snr.snr_to_csv(out, str(path))
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("spectrum,S/N") and "3 to -3 ppm" in text
    fig = snr.plot_snr([a, b], out)
    ax = fig.axes[0]
    assert ax.get_xlim()[0] > ax.get_xlim()[1]                 # ppm runs right to left
    assert any("S/N" in t.get_text() and "†" in t.get_text() for t in ax.texts)
    one = snr.signal_to_noise(a, (5, -5), (80, 40))
    snr.plot_snr(a, one)
    code = snr.snr_python((5, -5), [(80, 40)], regions={1: {"signal": (3, -3)}},
                          labels=["a", "b"], detrend=1)
    assert "signal_to_noise_many(spectra" in code and "regions={1:" in code
    compile(code, "<snr>", "exec")


def test_exported_by_the_single_notebook_import():
    import opennmr.notebook as nb
    assert nb.snr is snr
    assert nb.signal_to_noise is snr.signal_to_noise
    assert nb.signal_to_noise_many is snr.signal_to_noise_many


def test_plot_preview_without_a_result():
    import matplotlib
    matplotlib.use("Agg")
    a, b = _spectrum(seed=20), _spectrum(seed=21)
    fig = snr.plot_snr([a, b], regions=[{"signal": (5, -5), "noise": [(80, 40)]},
                                        {"signal": None, "noise": None}])
    assert fig.axes[0].get_xlim()[0] > fig.axes[0].get_xlim()[1]
    assert not any(t.get_text().startswith("S/N") for t in fig.axes[0].texts)
