"""J-coupling on isotropic peaks (explicit user request, 2026-10-23): a peak
can be a first-order multiplet whose J values are fitted."""
import numpy as np
import pytest
from test_gui import qapp  # noqa: F401 -- fixture

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import fitting as nb_fitting
from opennmr.processing import fitting, multiplets


SFO1 = 400.0


def _multiplet_spectrum(centre=5.0, counts=(2,), js=(7.2,), fwhm_hz=1.5, A=1.0, noise=0.0, seed=0):
    ppm = np.linspace(4.9, 5.1, 4000)
    hz_per_ppm = SFO1
    fwhm = fwhm_hz / hz_per_ppm
    off, w = multiplets.lines(list(counts), list(js))
    y = np.zeros_like(ppm)
    for o, wi in zip(off, w):
        y += wi * fitting.lorentzian(ppm, centre + o / hz_per_ppm, fwhm, A)
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise, y.size)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=SFO1)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * SFO1)
    return NMRSpectrum(data=y, axes=[axis], metadata=metadata)


# ---- names and lines ---------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("", []), ("s", []), ("d", [1]), ("t", [2]), ("q", [3]), ("dt", [1, 2]), ("dd", [1, 1]),
    ("triplet", [2]), ("quartet", [3]), ("septet", [6]), ("n=4", [4]), ("D T", [1, 2])])
def test_multiplet_names_are_read(text, expected):
    assert multiplets.parse_multiplet(text) == expected


def test_a_bad_multiplet_name_is_explained():
    with pytest.raises(ValueError, match="doublet of triplets"):
        multiplets.parse_multiplet("zz")


def test_names_round_trip():
    for name in ("d", "t", "q", "dt", "dd", "tq"):
        assert multiplets.multiplet_name(multiplets.parse_multiplet(name)) == name


def test_lines_are_binomial_and_normalised():
    off, w = multiplets.lines([3], [10.0])
    assert off == pytest.approx([-15, -5, 5, 15])
    assert w == pytest.approx(np.array([1, 3, 3, 1]) / 8)
    off, w = multiplets.lines([1, 2], [7.0, 2.0])
    assert len(off) == 6 and w.sum() == pytest.approx(1.0)


def test_couplings_text_is_padded_and_checked():
    assert multiplets.parse_couplings("7.5", 2) == [7.5, multiplets.DEFAULT_J_HZ]
    assert multiplets.parse_couplings("", 1) == [multiplets.DEFAULT_J_HZ]
    with pytest.raises(ValueError):
        multiplets.parse_couplings("1, 2, 3", 2)
    with pytest.raises(ValueError):
        multiplets.parse_couplings("a", 1)


# ---- the fit -----------------------------------------------------------------

def test_a_triplet_fits_its_coupling_and_keeps_its_area():
    spectrum = _multiplet_spectrum(counts=(2,), js=(7.2,))
    peaks = [{"type": "lorentzian", "shift": 5.0, "fwhm": 1.5 / SFO1, "A_init": 0.9,
              "couplings": [2], "J_init": [6.0]}]
    popt, pcov, is_free, scale, x_ref = fitting.fit_spectrum(spectrum, peaks, use_normalised=False)
    per_peak = fitting.extract_params_and_errors(popt, fitting.expand_pcov(pcov, is_free, len(popt)),
                                                 peaks, is_free)
    values = dict(zip(per_peak[0]["names"], per_peak[0]["params"]))
    assert values["J1"] == pytest.approx(7.2, abs=0.05)
    assert values["shift"] == pytest.approx(5.0, abs=1e-4)
    assert values["amplitude"] == pytest.approx(1.0, rel=0.02)
    x = spectrum.axes[0].ppm
    _y, components = fitting.evaluate_fit(x, popt, peaks, x_ref=x_ref, return_components=True)
    single = fitting.lorentzian(x, 5.0, 1.5 / SFO1, 1.0)
    order = np.argsort(x)
    area_multiplet = np.trapezoid(components[0][order], x[order])
    area_single = np.trapezoid(single[order], x[order])
    assert area_multiplet == pytest.approx(area_single, rel=0.01)


def test_a_doublet_of_triplets_fits_both_couplings():
    spectrum = _multiplet_spectrum(counts=(1, 2), js=(9.0, 2.5), fwhm_hz=0.8, noise=0.002)
    peaks = [{"type": "lorentzian", "shift": 5.0, "fwhm": 1.0 / SFO1, "A_init": 0.8,
              "couplings": [1, 2], "J_init": [8.0, 3.0]}]
    popt, pcov, is_free, *_ = fitting.fit_spectrum(spectrum, peaks, use_normalised=False)
    per_peak = fitting.extract_params_and_errors(popt, fitting.expand_pcov(pcov, is_free, len(popt)),
                                                 peaks, is_free)
    values = dict(zip(per_peak[0]["names"], per_peak[0]["params"]))
    assert values["J1"] == pytest.approx(9.0, abs=0.2)
    assert values["J2"] == pytest.approx(2.5, abs=0.2)


def test_a_fixed_coupling_stays_put():
    spectrum = _multiplet_spectrum(counts=(1,), js=(8.0,))
    peaks = [{"type": "lorentzian", "shift": 5.0, "fwhm": 1.5 / SFO1, "A_init": 0.9,
              "couplings": [1], "J_init": [5.0], "fixed": {"J1": True}}]
    popt, _pcov, is_free, *_ = fitting.fit_spectrum(spectrum, peaks, use_normalised=False)
    j_index = fitting._j_start(peaks)
    assert popt[j_index] == 5.0 and not is_free[j_index]


def test_peaks_without_couplings_are_unchanged():
    spectrum = _multiplet_spectrum(counts=(), js=())
    peaks = [{"type": "lorentzian", "shift": 5.0, "fwhm": 1.5 / SFO1, "A_init": 0.9}]
    popt, *_ = fitting.fit_spectrum(spectrum, peaks, use_normalised=False)
    assert len(popt) == 3 + 2                       # shift, fwhm, amplitude + baseline
    assert popt[0] == pytest.approx(5.0, abs=1e-4)


def test_a_multiplet_next_to_a_singlet():
    ppm = np.linspace(4.0, 6.0, 6000)
    off, w = multiplets.lines([1], [12.0])
    y = sum(wi * fitting.lorentzian(ppm, 4.5 + o / SFO1, 2.0 / SFO1, 1.0) for o, wi in zip(off, w))
    y = y + fitting.lorentzian(ppm, 5.5, 2.0 / SFO1, 0.7)
    spectrum = NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * SFO1)],
                           metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=SFO1))
    peaks = [{"type": "lorentzian", "shift": 4.5, "fwhm": 2.0 / SFO1, "A_init": 0.9,
              "couplings": [1], "J_init": [10.0]},
             {"type": "lorentzian", "shift": 5.5, "fwhm": 2.0 / SFO1, "A_init": 0.6}]
    popt, pcov, is_free, scale, x_ref = fitting.fit_spectrum(spectrum, peaks, use_normalised=False)
    full = fitting.expand_pcov(pcov, is_free, len(popt))
    per_peak = fitting.extract_params_and_errors(popt, full, peaks, is_free)
    assert "J1" in per_peak[0]["names"] and "J1" not in per_peak[1]["names"]
    assert dict(zip(per_peak[0]["names"], per_peak[0]["params"]))["J1"] == pytest.approx(12.0, abs=0.1)
    assert dict(zip(per_peak[1]["names"], per_peak[1]["params"]))["amplitude"] == pytest.approx(0.7, rel=0.02)


def test_a_series_fit_can_hold_the_coupling_from_the_first_slice():
    ppm = np.linspace(4.9, 5.1, 3000)
    rows = []
    for amp in (1.0, 0.5, 0.25):
        off, w = multiplets.lines([2], [7.0])
        rows.append(sum(wi * amp * fitting.lorentzian(ppm, 5.0 + o / SFO1, 1.5 / SFO1, 1.0)
                        for o, wi in zip(off, w)))
    data = np.vstack(rows)
    f1 = NMRAxis(kind="frequency", ppm=np.arange(3, dtype=float), hz=np.arange(3, dtype=float))
    f2 = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * SFO1)
    spectrum = NMRSpectrum(data=data, axes=[f1, f2],
                           metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=SFO1))
    from opennmr.processing import slicing
    slices = slicing.get_slices(spectrum, rows=[1, 2, 3])
    peaks = [{"type": "lorentzian", "shift": 5.0, "fwhm": 1.5 / SFO1, "A_init": 0.8,
              "couplings": [2], "J_init": [6.0]}]
    fitted = fitting.NMR2D_fit_slices(slices, peaks, mode="rows", show_report=False,
                                      show_warnings=False, seed_mode="apply_first",
                                      positive_intensity=True)
    rows_fit = fitted["rows"]
    j_index = fitting._j_start(rows_fit[0]["peak_defs"])
    assert [r["popt"][j_index] for r in rows_fit] == pytest.approx([rows_fit[0]["popt"][j_index]] * 3)
    assert rows_fit[0]["popt"][j_index] == pytest.approx(7.0, abs=0.1)
    amps = [r["popt"][2] for r in rows_fit]
    assert amps[1] / amps[0] == pytest.approx(0.5, rel=0.05)


# ---- the notebook entry point ---------------------------------------------------

def test_the_notebook_fit_handles_multiplets_too():
    ppm = np.linspace(4.9, 5.1, 3000)
    off, w = multiplets.lines([3], [6.5])
    y = sum(wi * fitting.lorentzian(ppm, 5.0 + o / SFO1, 1.2 / SFO1, 1.0) for o, wi in zip(off, w))
    results = {"x_ppm": ppm, "x_Hz": ppm * SFO1, "real_spectrum": y, "SFO1": SFO1}
    peaks = [{"type": "lorentzian", "shift": 5.0, "fwhm": 1.2 / SFO1, "A_init": 0.8,
              "couplings": [3], "J_init": [5.0]}]
    popt, pcov, is_free, scale, x_ref = nb_fitting.fit_spectrum(results, peaks, use_normalised=False)
    assert popt[nb_fitting._j_start(peaks)] == pytest.approx(6.5, abs=0.1)
    # and in Hz
    peaks_hz = [{"type": "lorentzian", "shift": 5.0 * SFO1, "fwhm": 1.2, "A_init": 0.8,
                 "couplings": [3], "J_init": [5.0]}]
    popt_hz, *_ = nb_fitting.fit_spectrum(results, peaks_hz, use_normalised=False, x_unit="Hz")
    assert popt_hz[nb_fitting._j_start(peaks_hz)] == pytest.approx(6.5, abs=0.1)


# ---- the GUI -----------------------------------------------------------------

def _gui_window_with_triplet():
    from opennmr.gui.main_window import MainWindow
    from test_fit_correlation import _load_spectrum_directly
    window = MainWindow()
    _load_spectrum_directly(window, _multiplet_spectrum(counts=(2,), js=(7.2,)))
    return window


def test_peak_table_has_multiplet_columns_and_fills_in_j(qapp):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    assert [window.peak_table.horizontalHeaderItem(c).text() for c in (4, 5)] == ["Multiplet", "J (Hz)"]
    window._add_peak_row(5.0, 0.01, 0.5)
    window.peak_table.item(0, 4).setText("dt")
    assert window.peak_table.item(0, 5).text() == "7, 7"          # a starting J per splitting


def test_gui_fit_of_a_triplet_updates_the_table(qapp):  # noqa: F811
    window = _gui_window_with_triplet()
    window._add_peak_row(5.0, 1.5 / SFO1, 0.0, multiplet="t", couplings="6")
    window._on_fit_spectrum_clicked()
    j = float(window.peak_table.item(0, 5).text())
    assert j == pytest.approx(7.2, abs=0.2)
    table = window.spectral_fit_table
    assert table.columnCount() == 7 and table.item(0, 6).text().strip() not in ("", "—")
    state = window._capture_analysis_state()
    assert state["peaks"][0][4] == "t"


def test_a_bad_multiplet_name_is_reported_in_the_row(qapp):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    window._add_peak_row(5.0, 0.01, 0.5, multiplet="zz")
    with pytest.raises(ValueError, match="Peak row 1"):
        window._read_peak_defs()


def test_fixed_row_pins_the_couplings_too(qapp):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    window._add_peak_row(5.0, 0.01, 0.5, fixed=True, multiplet="dd", couplings="9, 2")
    peak = window._read_peak_defs()[0]
    assert peak["couplings"] == [1, 1] and peak["J_init"] == [9.0, 2.0]
    assert peak["fixed"]["J1"] and peak["fixed"]["J2"]


def test_sections_are_renamed(qapp):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    from opennmr.gui.widgets.collapsible_section import CollapsibleSection
    window = MainWindow()
    titles = [s.title() for s in window.findChildren(CollapsibleSection)]
    assert "Solution NMR" in titles and titles.count("Solid-state NMR") >= 2   # Fit and Simulate
    assert "Isotropic (Gauss/Lor)" not in titles and "CSA, Quad & J-coupling" not in titles
