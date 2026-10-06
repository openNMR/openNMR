"""Cubic-spline baseline with automatically placed points (user request,
2026-10-06: "a cubic-spline baseline option, with the points placed
automatically")."""

import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.processing import ft
from opennmr.processing.fitting import auto_spline_baseline, find_baseline_points

from test_gui import qapp, _load_spectrum_directly, _settle_live_reprocess  # noqa: F401


def _rolling(n=4096, sigma=2.0, seed=0):
    """A rolling baseline (amplitude ~150) under five peaks, one of them
    near an edge -- which breaks the edge-based estimators."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-20.0, 220.0, n)
    baseline = 150 * np.sin(2 * np.pi * x / 300.0) + 0.4 * x + 30
    # Gaussian peaks: a Lorentzian's long tails are signal spread under the
    # whole spectrum, which no baseline method can (or should) tell apart.
    peaks = sum(a * np.exp(-0.5 * ((x - c) / w) ** 2)
                for c, w, a in ((-15, 0.8, 800), (40, 1.5, 3000), (95, 0.6, 1500),
                                (130, 3.0, 600), (180, 1.0, 2000)))
    return x, baseline, peaks + baseline + rng.normal(0, sigma, n)


def test_baseline_points_are_found_away_from_peaks():
    x, _b, y = _rolling()
    mask = find_baseline_points(x, y)
    assert 0.3 < mask.mean() < 0.98
    for centre in (-15, 40, 95, 130, 180):
        assert not mask[np.argmin(np.abs(x - centre))]


def test_auto_spline_recovers_a_rolling_baseline_to_noise_level():
    x, true, y = _rolling()
    out = auto_spline_baseline(x, y)
    rms = float(np.sqrt(np.mean((out["baseline"] - true) ** 2)))
    assert rms < 1.0                          # noise sigma is 2
    assert float(np.max(np.abs(out["baseline"] - true))) < 6.0
    # Nodes lie on the true baseline, ascending, and cover both ends.
    nx, ny = np.asarray(out["nodes_x"]), np.asarray(out["nodes_y"])
    assert np.all(np.diff(nx) > 0)
    assert np.max(np.abs(ny - np.interp(nx, x, true))) < 3.0
    assert nx[0] < x.min() + 0.05 * np.ptp(x) and nx[-1] > x.max() - 0.05 * np.ptp(x)


def test_it_beats_the_edge_spline_and_works_on_descending_axes():
    x, true, y = _rolling()
    auto = y - np.real(ft.baseline_correct(x, y, method="spline_auto"))
    edge = y - np.real(ft.baseline_correct(x, y, method="spline"))
    rms = lambda b: float(np.sqrt(np.mean((b - true) ** 2)))   # noqa: E731
    assert rms(auto) < 0.05 * rms(edge)
    # ppm axes run high to low; the answer must not depend on it.
    flipped = ft.compute_baseline(x[::-1], y[::-1], method="spline_auto")
    np.testing.assert_allclose(flipped[::-1], auto, atol=1e-6)


def test_n_points_is_passed_through_and_changes_the_node_count():
    x, _t, y = _rolling()
    few = auto_spline_baseline(x, y, n_points=6)
    many = auto_spline_baseline(x, y, n_points=40)
    assert len(few["nodes_x"]) < len(many["nodes_x"])
    np.testing.assert_allclose(ft.compute_baseline(x, y, method="spline_auto", n_points=6),
                               few["baseline"])


def test_a_flat_noisy_spectrum_gives_a_flat_baseline():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 100, 2000)
    y = 5.0 + rng.normal(0, 1.0, x.size)
    base = auto_spline_baseline(x, y)["baseline"]
    assert np.max(np.abs(base - 5.0)) < 0.6


def _spectrum():
    x, true, y = _rolling(n=2048)
    ppm = x / 10.0
    return true, NMRSpectrum(
        data=y, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0),
        data_imag=np.zeros_like(y))


def test_gui_method_points_box_and_overlay_nodes(qapp):
    true, spectrum = _spectrum()
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    combo = window.baseline_method_combo
    combo.setCurrentIndex(combo.findData("spline_auto"))
    assert not window.baseline_points_spin.isHidden()
    assert window.baseline_degree_spin.isHidden()
    window.baseline_points_spin.setValue(12)
    raw = window.session.spectra[0].metadata.raw
    assert raw["baseline_method"] == "spline_auto" and raw["baseline_points"] == 12
    assert window._baseline_kwargs(window.session.spectra[0])["n_points"] == 12
    # The edge-peak warning is for edge-based methods only.
    assert "edge region" not in window.processing_status_label.text()

    original = window.session.spectra[0].data.copy()
    window.baseline_correct_button.setChecked(True)
    _settle_live_reprocess(window)
    removed = original - window.session.spectra[0].data
    assert float(np.sqrt(np.mean((removed - true) ** 2))) < 1.5

    # Auto (0) is stored as None; other methods hide the box.
    window.baseline_points_spin.setValue(0)
    assert raw["baseline_points"] is None
    combo.setCurrentIndex(combo.findData("linear"))
    assert window.baseline_points_spin.isHidden()


def test_canvas_draws_and_clears_the_node_markers(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas
    _true, spectrum = _spectrum()
    canvas = MplCanvas()
    canvas.show_spectrum(spectrum)
    ppm = spectrum.axes[0].ppm
    canvas.set_baseline_preview(ppm, np.zeros_like(ppm), nodes=([1.0, 2.0], [0.0, 0.1]))
    gids = [ln.get_gid() for ax in canvas.figure.get_axes() for ln in ax.lines]
    assert canvas._BASELINE_NODES_GID in gids
    canvas.set_baseline_preview()
    gids = [ln.get_gid() for ax in canvas.figure.get_axes() for ln in ax.lines]
    assert canvas._BASELINE_NODES_GID not in gids


def _as_spectrum(x, y):
    return NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x, hz=x * 400.0)],
                       metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))


def test_warns_when_little_baseline_is_found():
    # A crowded spectrum: peaks almost everywhere.
    x = np.linspace(0, 10, 4000)
    rng = np.random.default_rng(0)
    y = sum(50 * np.exp(-0.5 * ((x - c) / 0.08) ** 2) for c in np.arange(0.2, 10, 0.25))
    notes = MainWindow._auto_spline_warnings(_as_spectrum(x, y + rng.normal(0, 0.5, x.size)))
    assert any("recognised as baseline" in n or "Too little" in n for n in notes)


def test_warns_when_a_wide_region_has_no_node():
    x = np.linspace(0, 10, 4000)
    rng = np.random.default_rng(1)
    y = sum(50 * np.exp(-0.5 * ((x - c) / 0.08) ** 2) for c in np.arange(2.0, 7.0, 0.25))
    notes = MainWindow._auto_spline_warnings(_as_spectrum(x, y + rng.normal(0, 0.5, x.size)))
    assert any("guess" in n for n in notes)


def test_a_broad_hump_is_taken_for_baseline():
    """The method's known limit, pinned so it is not mistaken for a bug:
    a feature much broader than the test window looks like baseline roll."""
    x = np.linspace(0, 10, 2000)
    y = 100 * np.exp(-((x - 5) / 3.5) ** 2) + np.random.default_rng(0).normal(0, 0.5, x.size)
    assert find_baseline_points(x, y).mean() > 0.8


def test_history_and_report_name_the_method():
    from opennmr.io import history
    from opennmr.processing import report
    assert "automatic" in history._BASELINE_METHODS["spline_auto"]
    assert "automatic" in report._BASELINE_NAMES["spline_auto"]
