"""Signal-to-noise conventions side by side, TopSpin's own arithmetic, and
2D spectra (explicit user request, 2026-10-14: "enable different ways of
calculating signal-to-noise ... evaluate these values and present them
to people to compare")."""

import math

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import snr

X = np.linspace(100.0, -100.0, 4001)
X2 = np.linspace(50.0, -50.0, 256)
X1 = np.linspace(30.0, -30.0, 128)


def _one_d(height=1000.0, sigma=10.0, seed=1):
    rng = np.random.default_rng(seed)
    return X, height * np.exp(-((X - 10) / 0.5) ** 2) + rng.normal(0, sigma, X.size)


def _two_d(height=500.0, sigma=5.0, seed=2, correlate=False):
    rng = np.random.default_rng(seed)
    g2, g1 = np.meshgrid(X2, X1)
    noise = rng.normal(0, 1, g2.shape)
    if correlate:                      # neighbouring points correlated along F2, as apodised
        noise = (noise + np.roll(noise, 1, axis=1) + np.roll(noise, 2, axis=1)) / math.sqrt(3)
    z = height * np.exp(-((g2 - 5) / 1.0) ** 2 - ((g1 - 3) / 1.5) ** 2) + sigma * noise
    return z


def _two_d_spectrum(**kw):
    z = _two_d(**kw)
    return NMRSpectrum(data=z, axes=[NMRAxis(kind="frequency", ppm=X1.copy()),
                                     NMRAxis(kind="frequency", ppm=X2.copy())],
                       metadata=NMRMetadata(vendor="test", nucleus="1H", ns=8))


# ---- TopSpin's sino arithmetic -------------------------------------------------

def test_topspin_noise_follows_the_published_formula():
    y = np.array([0.3, -1.2, 0.8, 2.0, -0.5, 1.1, -0.9])
    n_points = y.size
    n = (n_points - 1) // 2
    i = np.arange(1, n + 1)
    centre = n
    s = np.sum(i * (y[centre + i] - y[centre - i]))
    expected = math.sqrt((np.sum(y ** 2) - (np.sum(y) ** 2 + 3 * s ** 2 / (n_points ** 2 - 1))
                          / n_points) / (n_points - 1))
    assert snr.topspin_noise(y) == pytest.approx(expected)
    # On flat white noise it is the ordinary standard deviation.
    flat = np.random.default_rng(0).normal(0, 2.0, 2001)
    assert snr.topspin_noise(flat) == pytest.approx(np.std(flat, ddof=1), rel=0.01)


def test_one_d_conventions_are_the_stated_factors():
    r = snr.signal_to_noise(_one_d(), (15, 5), (80, 40))
    assert r["ndim"] == 1
    assert r["sino"] == pytest.approx(r["snr"] / 2)
    assert r["snr_3sigma"] == pytest.approx(r["snr"] / 3)
    assert r["integral_sino"] == pytest.approx(r["integral_snr"] / 2)
    # TopSpin's own arithmetic: maxval from zero over twice its noise.
    assert r["sino_topspin"] == pytest.approx(r["topspin_maxval"] / (2 * r["topspin_noise"]))
    assert r["sino_topspin"] == pytest.approx(r["sino"], rel=0.03)    # flat baseline: they agree
    keys = [c["key"] for c in snr.snr_conventions(r)]
    assert keys == ["snr", "sino", "sino_topspin", "snr_3sigma", "ph_eur", "integral_snr",
                    "integral_sino"]
    text = snr.snr_report(r)
    assert "H / (2σ)" in text and "H / (3σ)" in text and "area / (2·σ_area)" in text


def test_topspin_measures_maxval_from_zero():
    x, y = _one_d()
    offset = 200.0                            # a baseline offset under everything
    r = snr.signal_to_noise((x, y + offset), (15, 5), (80, 40))
    assert r["sino"] == pytest.approx(snr.signal_to_noise((x, y), (15, 5), (80, 40))["sino"],
                                      rel=1e-6)
    assert r["sino_topspin"] > 1.15 * r["sino"]           # TopSpin counts the offset


# ---- 2D --------------------------------------------------------------------------

def test_two_d_recovers_the_true_ratio():
    r = snr.signal_to_noise_2d((X2, X1, _two_d()), (10, 0), (45, 25), (10, -4), (28, -28))
    assert r["ndim"] == 2
    assert r["snr"] == pytest.approx(100.0, rel=0.08)
    lo, hi = r["snr_interval"]
    assert lo < 100.0 < hi * 1.05
    assert r["position_ppm"] == pytest.approx((5.0, 3.0), abs=0.6)
    assert r["sino"] == pytest.approx(r["snr"] / 2) and r["snr_3sigma"] == pytest.approx(r["snr"] / 3)
    assert r["volume_sino"] == pytest.approx(r["volume_snr"] / 2)
    assert r["detected"]
    assert [c["key"] for c in snr.snr_conventions(r)] == ["snr", "sino", "snr_3sigma",
                                                           "volume_snr", "volume_sino"]


def test_two_d_counts_correlated_noise_points_once():
    white = snr.signal_to_noise_2d((X2, X1, _two_d()), (10, 0), (45, 25), (10, -4), None)
    corr = snr.signal_to_noise_2d((X2, X1, _two_d(correlate=True)), (10, 0), (45, 25),
                                  (10, -4), None)
    assert white["correlation_time_f2"] == pytest.approx(1.0, abs=0.3)
    assert corr["correlation_time_f2"] > 2.0
    assert corr["n_noise_independent"] < 0.6 * corr["n_noise"]
    assert corr["noise_region_f1"] == [(30.0, -30.0)]        # None = all of F1


def test_two_d_rejects_a_noise_box_over_the_signal():
    with pytest.raises(ValueError, match="overlaps"):
        snr.signal_to_noise_2d((X2, X1, _two_d()), (10, 0), (8, 2), (10, -4), (6, 0))


def test_two_d_spectrum_object_and_mixed_lists():
    two = _two_d_spectrum()
    out = snr.signal_to_noise_many([_one_d(), two], (15, 5), (80, 40), labels=["1d", "2d"],
                                   regions={1: {"signal": (10, 0), "noise": (45, 25)}},
                                   signal_region_f1=(10, -4), noise_region_f1=(28, -28))
    assert [r["ndim"] for r in out["rows"]] == [1, 2]
    assert out["comparisons"] == []                 # a 2D is not compared with a 1D
    assert any("1D and 2D" in n for n in out["notes"])
    assert out["rows"][1]["ns"] == 8
    rows = snr.snr_table(out)
    assert rows[1]["dimensions"] == 2 and rows[1]["signal F1 (ppm)"] == "10 to -4 ppm"


def test_two_d_csv_and_plot(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    two = _two_d_spectrum()
    out = snr.signal_to_noise_many([_one_d(), two], (15, 5), (80, 40), labels=["1d", "2d"],
                                   regions={1: {"signal": (10, 0), "noise": (45, 25),
                                                "signal_f1": (10, -4)}})
    path = tmp_path / "snr.csv"
    snr.snr_to_csv(out, str(path))
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert "Volume (V / σ_V)" in header and "Integral (area / σ_area)" in header
    fig = snr.plot_snr([_one_d(), two], out)
    assert len(fig.axes) == 2
    assert len(fig.axes[1].patches) == 2            # the signal box and the noise box


# ---- GUI ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_gui_measures_a_2d_and_shows_the_conventions_side_by_side(qapp, monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: shown.append(a[2] if len(a) > 2
                                                                           else ""))
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    one = NMRSpectrum(data=_one_d()[1], axes=[NMRAxis(kind="frequency", ppm=X.copy())],
                      metadata=NMRMetadata(vendor="test", nucleus="1H", ns=4))
    window._add_derived_spectrum(one, "one")
    window._add_derived_spectrum(_two_d_spectrum(), "two")
    window._show_side_tab("Integrate")
    window._snr_section.setExpanded(True)
    p = window.snr_panel
    p.refresh()
    labels = [e["label"] for e in p._rows]
    assert any(label.endswith("(2D)") for label in labels)
    p.set_checked([lab for lab in labels if lab.endswith("(2D)")])
    p.set_signal_region(10, 0)
    p.set_noise_region([(45, 25)])
    p.signal_f1_edit.setText("10, -4")
    p._on_f1_typed("signal")
    p.noise_f1_edit.setText("28, -28")
    p._on_f1_typed("noise")
    result = p.calculate(explicit=True)
    assert result is not None, shown
    row = result["rows"][0]
    assert row["ndim"] == 2 and row["signal_region_f1"] == (10.0, -4.0)
    names = [name for name, _values in p.conventions_rows()]
    assert names[:3] == ["RMS: H / σ", "×½ (TopSpin's factor): H / (2σ)", "×⅓: H / (3σ)"]
    # Shown in the result table (one row per quantity; row 2 is H/2σ).
    table = p.result_table
    shorts = [table.verticalHeaderItem(i).text() for i in range(table.rowCount())]
    assert not table.isHidden() and "Volume/σ" in shorts and "Area/σ" not in shorts
    i = shorts.index("H/2σ  (TopSpin ×½)")
    assert float(table.item(i, 0).text()) == pytest.approx(row["sino"], rel=1e-3)
    assert "signal_region_f1=(10.0, -4.0)" in p.notebook_code()
    assert "2D boxes, F1" in "\n".join(
        str(v) for v in window.session.spectra[1].metadata.raw["fits"]["snr"]["inputs_text"])
