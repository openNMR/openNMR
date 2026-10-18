"""2026-10-18: Offset Tracker (Integrate) -- the triangles replaced by peaks
(Gaussian, Lorentzian or pseudo-Voigt of one chosen width) centred on every
offset, the pattern of a nucleus's MAS spectrum when the offsets sit on its
sidebands; what is drawn (bars, vertical lines or peaks) is what is kept in
Open experiments. Plus the notebook's one import for the new functions."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.notebook import fitting as nb_fitting
from opennmr.notebook import offset_tracker as ot

from test_gui import qapp, _load_spectrum_directly  # noqa: F401


def _series(offsets_ppm=(-10.0, -5.0, 0.0, 5.0, 10.0), heights=(1.0, 3.0, 6.0, 2.0, 0.5),
            bf=400.0, nucleus="1H"):
    """A hand-made series: integrals ``heights`` at ``offsets_ppm``."""
    x = np.asarray(offsets_ppm, float)
    y = np.asarray(heights, float)
    return {"offset_ppm": x, "offset_hz": x * bf if bf else x * np.nan, "y": y,
            "y_err": np.zeros_like(y), "labels": [f"s{i}" for i in range(x.size)],
            "nucleus": nucleus, "bf_MHz": bf, "channel": "O2", "region": (-5.0, 5.0),
            "notes": [], "unit": "integral (a.u.)", "baseline": "local", "divide_rg": True}


def _offset_spectra():
    x = np.linspace(50, -50, 2001)
    out = []
    for o2 in np.arange(-4000.0, 4001.0, 1000.0):
        y = np.exp(-(o2 / 2000.0) ** 2) * np.exp(-x ** 2 / 2)
        meta = NMRMetadata(vendor="bruker", nucleus="13C", sfo1=100.0, bf1=100.0, ns=1.0,
                           raw={"acqus": {"O2": o2, "BF2": 400.0, "NUC2": "<1H>", "NS": 1}})
        out.append(NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x, hz=x * 100.0)],
                               metadata=meta, data_imag=np.zeros_like(y)))
    return out


# ---- notebook ------------------------------------------------------------------

def test_triangles_are_gone_and_named_in_the_error():
    assert "triangles" not in ot.STYLES and "lineshape" in ot.STYLES
    with pytest.raises(ValueError, match="replaced by 'lineshape'"):
        ot.plot_offset_series(_series(), style="triangles")
    with pytest.raises(ValueError, match="replaced by 'lineshape'"):
        ot.pattern_spectrum(_series(), style="triangles")


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian", "pseudo_voigt"])
def test_peak_profile_has_its_height_and_fwhm(shape):
    x = np.linspace(-10, 10, 200001)
    y = ot.peak_profile(x, 1.5, 2.0, shape, eta=0.3, height=4.0)
    assert y.max() == pytest.approx(4.0, rel=1e-6)
    assert x[np.argmax(y)] == pytest.approx(1.5, abs=1e-4)
    above = x[y >= 2.0]
    assert above.max() - above.min() == pytest.approx(2.0, abs=2e-4)


def test_peak_profile_matches_the_fitting_definitions():
    x = np.linspace(-5, 5, 101)
    np.testing.assert_allclose(ot.peak_profile(x, 0.3, 1.2, "gaussian", height=2.0),
                               nb_fitting.gaussian(x, 0.3, 1.2, 2.0))
    np.testing.assert_allclose(ot.peak_profile(x, 0.3, 1.2, "lorentzian", height=2.0),
                               nb_fitting.lorentzian(x, 0.3, 1.2, 2.0))
    np.testing.assert_allclose(ot.peak_profile(x, 0.3, 1.2, "pseudo_voigt", 0.4, 2.0),
                               nb_fitting.pseudo_voigt(x, 0.3, 1.2, 2.0, 0.4))
    with pytest.raises(ValueError):
        ot.peak_profile(x, 0, 1, "pseudo_voigt", eta=1.5)
    with pytest.raises(ValueError):
        ot.peak_profile(x, 0, 1, "voigt")


def test_width_units_convert_with_bf_and_refuse_without():
    s = _series(bf=400.0)
    assert ot.fwhm_in_ppm(s, 2.0, "ppm") == pytest.approx(2.0)
    assert ot.fwhm_in_ppm(s, 800.0, "Hz") == pytest.approx(2.0)
    assert ot.fwhm_in_ppm(s, 0.8, "kHz") == pytest.approx(2.0)
    with pytest.raises(ValueError, match="BF"):
        ot.fwhm_in_ppm(_series(bf=None), 800.0, "Hz")
    with pytest.raises(ValueError):
        ot.fwhm_in_ppm(s, 0.0, "ppm")
    with pytest.raises(ValueError):
        ot.fwhm_in_ppm(s, 1.0, "furlong")


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian", "pseudo_voigt"])
def test_peaks_are_centred_on_the_offsets_with_the_integrals_as_heights(shape):
    s = _series()
    pat = ot.sideband_pattern(s, lineshape=shape, fwhm=0.1, eta=0.5)
    # Positions are the offsets, never chosen: each local maximum is an offset.
    y, x = pat["y"], pat["ppm"]
    peaks = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    np.testing.assert_allclose(np.sort(x[peaks]), s["offset_ppm"], atol=2 * abs(x[1] - x[0]))
    # Narrow peaks: the curve at each offset is that offset's integral.
    at = ot.sideband_pattern(s, lineshape=shape, fwhm=0.1, eta=0.5, x=s["offset_ppm"])["y"]
    np.testing.assert_allclose(at, s["y"], rtol=2e-3, atol=2e-3 * s["y"].max())
    assert not [n for n in pat["notes"] if "overlap" in n]


def test_components_sum_to_the_pattern_and_overlap_is_reported():
    s = _series()
    pat = ot.sideband_pattern(s, "lorentzian", fwhm=6.0, components=True)
    assert pat["components"].shape == (5, pat["ppm"].size)
    np.testing.assert_allclose(pat["components"].sum(axis=0), pat["y"])
    assert any("overlap" in n for n in pat["notes"])


def test_default_width_is_a_quarter_of_the_spacing():
    s = _series()
    assert ot.default_fwhm_ppm(s) == pytest.approx(1.25)
    assert ot.sideband_pattern(s)["fwhm_ppm"] == pytest.approx(1.25)


def test_grid_reaches_the_tails_and_resolves_the_width():
    s = _series()
    g = ot.sideband_pattern(s, "gaussian", fwhm=1.0)
    l = ot.sideband_pattern(s, "lorentzian", fwhm=1.0)
    assert g["ppm"][0] >= 10.0 + 3.0 and g["ppm"][-1] <= -10.0 - 3.0
    assert l["ppm"][0] >= 10.0 + 10.0 and l["ppm"][-1] <= -10.0 - 10.0
    narrow = ot.sideband_pattern(s, "gaussian", fwhm=0.001, n_points=1024)
    step = abs(narrow["ppm"][1] - narrow["ppm"][0])
    assert 0.001 / step >= ot.MIN_POINTS_PER_FWHM - 1e-9
    assert any("enlarged" in n for n in narrow["notes"])


def test_sideband_spectrum_is_a_spectrum_of_the_pulsed_nucleus():
    s = _series()
    sp = ot.sideband_spectrum(s, "pseudo_voigt", fwhm=400.0, width_unit="Hz", eta=0.25)
    assert sp.metadata.nucleus == "1H" and sp.metadata.bf1 == pytest.approx(400.0)
    assert sp.metadata.vendor == "derived"
    ppm = sp.axes[0].ppm
    assert ppm[0] > ppm[-1]                                     # descending, as NMR axes
    np.testing.assert_allclose(sp.axes[0].hz, ppm * 400.0)
    rec = sp.metadata.raw["offset_pattern"]
    assert rec["style"] == "lineshape" and rec["lineshape"] == "pseudo_voigt"
    assert rec["fwhm_ppm"] == pytest.approx(1.0) and rec["fwhm_hz"] == pytest.approx(400.0)
    assert rec["eta"] == pytest.approx(0.25)
    assert rec["offsets_ppm"] == pytest.approx(list(s["offset_ppm"]))
    assert rec["integrals"] == pytest.approx(list(s["y"]))


def test_stick_spectrum_puts_each_integral_exactly_on_its_offset():
    s = _series()
    sp = ot.stick_spectrum(s, n_points=1001)
    nz = np.nonzero(sp.data)[0]
    np.testing.assert_allclose(np.sort(sp.axes[0].ppm[nz]), s["offset_ppm"], atol=1e-12)
    np.testing.assert_allclose(sp.data[nz][::-1], s["y"])
    assert sp.metadata.raw["offset_pattern"]["max_displacement_ppm"] < 1e-9
    uneven = _series(offsets_ppm=(-10.0, -4.9, 0.0, 5.0, 10.0))
    sp = ot.stick_spectrum(uneven, n_points=1001)
    assert sp.data.sum() == pytest.approx(uneven["y"].sum())
    assert sp.metadata.raw["offset_pattern"]["notes"]


def test_pattern_spectrum_follows_the_style():
    s = _series()
    assert ot.pattern_spectrum(s, "bars").metadata.raw["offset_pattern"]["style"] == "bars"
    assert ot.pattern_spectrum(s, "stems").metadata.raw["offset_pattern"]["style"] == "stems"
    lin = ot.pattern_spectrum(s, "lineshape", lineshape="lorentzian", fwhm=0.5)
    assert lin.metadata.raw["offset_pattern"]["lineshape"] == "lorentzian"
    # The 2026-10-17 skyline keys are still written for bars.
    assert "skyline_offsets_ppm" in ot.pattern_spectrum(s, "bars").metadata.raw


def test_offset_spacing_and_the_mas_rate_line():
    s = _series()
    sp = ot.offset_spacing(s)
    assert sp["uniform"] and sp["ppm"] == pytest.approx(5.0) and sp["hz"] == pytest.approx(2000.0)
    assert "2000 Hz" in ot.offset_report(s)
    assert not ot.offset_spacing(_series(offsets_ppm=(-10, -4, 0, 5, 10)))["uniform"]


def test_plot_lineshape_draws_one_curve_and_optional_components():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = _series()
    fig, ax = ot.plot_offset_series(s, style="lineshape", lineshape="gaussian", fwhm=1.0)
    n_plain = len(ax.lines)
    assert n_plain == 2                                        # the curve and the zero line
    plt.close(fig)
    fig, ax = ot.plot_offset_series(s, style="lineshape", lineshape="gaussian", fwhm=1.0,
                                    show_components=True)
    assert len(ax.lines) == n_plain + 5
    lo, hi = sorted(ax.get_xlim())
    assert ax.get_xlim()[0] > ax.get_xlim()[1]                 # high to low ppm
    assert lo <= -13.0 and hi >= 13.0                          # the tails are in view
    plt.close(fig)


def test_one_import_has_the_new_functions():
    import opennmr.notebook as nmr
    for name in ("offset_tracker", "sideband_pattern", "sideband_spectrum", "stick_spectrum",
                 "pattern_spectrum", "plot_offset_series", "fit_vt_regions",
                 "plot_arrhenius_regions", "available_fonts", "use_font", "save_vector_safe"):
        assert name in nmr.__all__ and hasattr(nmr, name)
    assert nmr.sideband_pattern is ot.sideband_pattern


# ---- GUI -----------------------------------------------------------------------

def _panel_with_series(window):
    for s in _offset_spectra():
        _load_spectrum_directly(window, s)
    panel = window.offset_tracker_panel
    panel.refresh()
    panel.tick_all_button.click()
    panel.read_offsets()
    panel.set_region(5, -5)
    return panel


def test_panel_offers_peaks_not_triangles(qapp):
    window = MainWindow()
    panel = window.offset_tracker_panel
    keys = [panel.style_combo.itemData(i) for i in range(panel.style_combo.count())]
    assert keys == ["bars", "stems", "lineshape"]
    shapes = [panel.lineshape_combo.itemData(i) for i in range(panel.lineshape_combo.count())]
    assert shapes == ["gaussian", "lorentzian", "pseudo_voigt"]
    assert panel.keep_button.text() == "Keep in Open Experiments"


def test_lineshape_controls_show_only_for_peaks_and_eta_only_for_pseudo_voigt(qapp):
    window = MainWindow()
    panel = window.offset_tracker_panel
    panel.style_combo.setCurrentIndex(panel.style_combo.findData("bars"))
    assert panel.lineshape_box.isHidden() and not panel.shape_combo.isHidden()
    panel.style_combo.setCurrentIndex(panel.style_combo.findData("lineshape"))
    assert not panel.lineshape_box.isHidden() and panel.shape_combo.isHidden()
    panel.lineshape_combo.setCurrentIndex(panel.lineshape_combo.findData("gaussian"))
    assert panel.eta_spin.isHidden()
    panel.lineshape_combo.setCurrentIndex(panel.lineshape_combo.findData("pseudo_voigt"))
    assert not panel.eta_spin.isHidden()


def test_width_defaults_to_a_quarter_spacing_and_converts_between_units(qapp):
    window = MainWindow()
    panel = _panel_with_series(window)
    panel.style_combo.setCurrentIndex(panel.style_combo.findData("lineshape"))
    series = panel.integrate()
    assert series is not None
    spacing = ot.offset_spacing(series)["ppm"]                 # 2.5 ppm (1000 Hz at 400 MHz)
    assert panel.fwhm_spin.value() == pytest.approx(0.25 * spacing, rel=1e-3)
    panel.width_unit_combo.setCurrentIndex(panel.width_unit_combo.findData("Hz"))
    assert panel.fwhm_spin.value() == pytest.approx(0.25 * spacing * 400.0, rel=1e-3)
    assert panel.fwhm_ppm() == pytest.approx(0.25 * spacing, rel=1e-3)
    panel.set_fwhm(500.0, "Hz")
    assert panel.pattern is not None and panel.pattern["fwhm_ppm"] == pytest.approx(1.25)
    # A chosen width survives integrating again.
    panel.integrate()
    assert panel.fwhm_spin.value() == pytest.approx(500.0)


def test_keep_in_open_experiments_keeps_the_peaks_drawn(qapp):
    window = MainWindow()
    panel = _panel_with_series(window)
    panel.style_combo.setCurrentIndex(panel.style_combo.findData("lineshape"))
    panel.lineshape_combo.setCurrentIndex(panel.lineshape_combo.findData("lorentzian"))
    series = panel.integrate()
    panel.set_fwhm(0.5, "ppm")
    n = len(window.session)
    kept = panel.keep_pattern()
    assert kept is not None and len(window.session) == n + 1
    sp = window.session.spectra[-1]
    assert sp.metadata.nucleus == "1H"
    rec = sp.metadata.raw["offset_pattern"]
    assert rec["style"] == "lineshape" and rec["lineshape"] == "lorentzian"
    assert rec["fwhm_ppm"] == pytest.approx(0.5)
    expected = ot.sideband_pattern(series, "lorentzian", fwhm=0.5, x=sp.axes[0].ppm)["y"]
    np.testing.assert_allclose(sp.data, expected)
    assert "Lorentzian peaks" in window.session.label(len(window.session) - 1)


@pytest.mark.parametrize("style", ["bars", "stems"])
def test_keep_follows_the_other_styles_too(qapp, style):
    window = MainWindow()
    panel = _panel_with_series(window)
    panel.style_combo.setCurrentIndex(panel.style_combo.findData(style))
    panel.integrate()
    kept = panel.keep_pattern()
    assert kept.metadata.raw["offset_pattern"]["style"] == style
    # The old name still works and keeps what is drawn.
    assert panel.keep_skyline().metadata.raw["offset_pattern"]["style"] == style


def test_hz_width_without_bf_is_reported_not_drawn(qapp):
    window = MainWindow()
    panel = window.offset_tracker_panel
    panel.series = _series(bf=None)
    panel.style_combo.setCurrentIndex(panel.style_combo.findData("lineshape"))
    panel.set_fwhm(100.0, "Hz")
    assert panel.pattern is None and "BF" in panel.pattern_label.text()


def test_axis_label_claims_per_unit_rg_only_when_rg_was_divided():
    spectra = _offset_spectra()                      # no RG recorded
    series = ot.offset_series(spectra, (5, -5), channel="O2", divide_rg=True)
    assert "RG" not in series["unit"]
    for s in spectra:
        s.metadata.raw["acqus"]["RG"] = 32.0
    series = ot.offset_series(spectra, (5, -5), channel="O2", divide_rg=True)
    assert "per unit RG" in series["unit"]
