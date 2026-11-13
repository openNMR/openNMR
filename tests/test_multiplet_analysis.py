"""Multiplets and J couplings from picked lines (explicit user request,
2026-11-12): group the picked lines, read each group as a first-order
pattern (name, centre, J values, refined against every line), report it
the way a paper quotes it, draw the coupling tree, and hand the result to
the spectral fit as multiplet rows."""

import numpy as np
import pytest

import opennmr.notebook as nmr
from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import multiplet_analysis as ma
from opennmr.processing import multiplets as pm

MHZ = 400.0


def _lines(centre, counts, js):
    off, w = pm.lines(counts, js)
    merged = {}
    for o, ww in zip(off, w):
        merged[round(o, 6)] = merged.get(round(o, 6), 0.0) + ww
    hz = np.array(sorted(merged))
    return centre + hz / MHZ, np.array([merged[k] for k in sorted(merged)])


@pytest.mark.parametrize("counts, js, name", [
    ([1], [8.0], "d"), ([2], [7.0], "t"), ([3], [7.2], "q"), ([4], [6.9], "p"),
    ([1, 1], [10.1, 4.0], "dd"), ([2, 1], [7.0, 2.0], "td"), ([1, 2], [11.0, 3.0], "dt"),
    ([1, 1, 1], [12.0, 6.5, 2.1], "ddd"),
])
def test_first_order_patterns(counts, js, name):
    shifts, heights = _lines(3.0, counts, js)
    r = ma.analyse_multiplet(shifts, MHZ, heights)
    assert r["first_order"] and r["pattern"] == name
    assert r["J_hz"] == pytest.approx(sorted(set(js), reverse=True), abs=1e-6)
    assert r["centre_ppm"] == pytest.approx(3.0, abs=1e-9)
    assert r["rms_hz"] < 1e-6 and not r["notes"]


def test_a_singlet():
    r = ma.analyse_multiplet([1.2], MHZ, [5.0])
    assert r["pattern"] == "s" and r["J_hz"] == [] and ma.format_multiplet(r) == "1.20 (s)"


def test_noise_is_averaged_over_every_line():
    rng = np.random.default_rng(1)
    shifts, heights = _lines(2.0, [1, 1], [10.1, 4.0])
    shifts = shifts + rng.normal(0, 0.1 / MHZ, shifts.size)
    heights = heights * (1 + rng.normal(0, 0.05, heights.size))
    r = ma.analyse_multiplet(shifts, MHZ, heights)
    assert r["pattern"] == "dd"
    assert r["J_hz"] == pytest.approx([10.1, 4.0], abs=0.2) and r["rms_hz"] < 0.2


def test_not_first_order_is_m_with_its_range():
    r = ma.analyse_multiplet([2.0, 2.013, 2.031], MHZ, [1, 1.3, 0.9])
    assert r["pattern"] == "m" and not r["first_order"] and r["J_hz"] == []
    assert ma.format_multiplet(r) == "2.03–2.00 (m)"
    assert any("first-order" in n for n in r["notes"])


def test_a_triplet_without_heights_is_read_from_its_positions():
    shifts, _h = _lines(1.0, [2], [7.0])
    r = ma.analyse_multiplet(shifts, MHZ)              # no heights
    assert r["pattern"] == "t"                         # never "dd, J = 14, 7"
    assert any("No heights" in n for n in r["notes"])


def test_evenly_spaced_lines_are_told_apart_by_their_heights():
    # 1:1:1:1 is a dd with J = 2 × J'; 1:3:3:1 is a quartet.
    s, h = _lines(2.0, [1, 1], [8.0, 4.0])
    assert ma.analyse_multiplet(s, MHZ, h)["pattern"] == "dd"
    s, h = _lines(2.0, [3], [4.0])
    assert ma.analyse_multiplet(s, MHZ, h)["pattern"] == "q"


@pytest.mark.parametrize("heights", [
    [1, 6, 15, 20, 15, 6, 1],
    [1.7, 5.9, 14.7, 19.4, 16.1, 7.1, 1.7],     # the user's PilotScan septet, roughly
])
def test_a_septet_with_uneven_heights(heights):
    s = 3.96 + (np.arange(7) - 3) * 6.2 / 80.0
    r = ma.analyse_multiplet(s, 80.0, heights)
    assert r["pattern"] == "h" and r["J_hz"] == pytest.approx([6.2], abs=0.01)
    assert ma.format_multiplet(r) == "3.96 (sept, J = 6.2 Hz)"


def test_intensity_units_are_a_power_of_two():
    assert list(ma.intensity_units([1, 3, 3, 1])[0]) == [1, 3, 3, 1]
    assert list(ma.intensity_units([0.9, 2.1, 1.0])[0]) == [1, 2, 1]
    assert list(ma.intensity_units([1, 1, 1, 1])[0]) == [1, 1, 1, 1]


def test_grouping_and_the_paper_line():
    s1, h1 = _lines(7.26, [1], [8.1])
    s2, h2 = _lines(3.45, [1, 1], [10.1, 4.0])
    results = nmr.analyse_multiplets(np.r_[s2, s1], 400.13, np.r_[h2, h1])
    assert [r["pattern"] for r in results] == ["d", "dd"]
    assert nmr.format_multiplet_report(results, "1H", 400.13) == \
        "¹H NMR (400 MHz) δ 7.26 (d, J = 8.1 Hz), 3.45 (dd, J = 10.1, 4.0 Hz)."
    assert ma.group_lines([1.0, 1.01, 2.0], MHZ, max_gap_hz=20.0) == [[2], [1, 0]]


def _spectrum(peaks):
    ppm = np.linspace(8.0, 0.0, 32768)
    hz = ppm * MHZ
    y = np.zeros_like(ppm)
    for centre, counts, js in peaks:
        off, w = pm.lines(counts, js)
        for o, ww in zip(off, w):
            y += ww / (1 + ((hz - (centre * MHZ + o)) / 0.4) ** 2)
    return NMRSpectrum(y, [NMRAxis("frequency", ppm=ppm, hz=hz)],
                       NMRMetadata(vendor="bruker", nucleus="1H", sfo1=MHZ))


def _isopropanol(noise=4.0, offset=80.0):
    """An 80 MHz isopropanol/water spectrum like the user's PilotScan: water
    at 4.74 (about 80× the septet's centre line, as there), the CH septet at
    3.96 and the CH3 doublet at 1.11 (J 6.2 Hz), on a raised, sloping
    baseline with noise."""
    mhz = 80.0
    ppm = np.linspace(10.0, -2.0, 65536)
    hz = ppm * mhz
    y = offset + 5.0 * ppm + np.random.default_rng(3).normal(0, noise, ppm.size)
    for centre, counts, js, area in ((4.74, [], [], 25.0), (3.96, [6], [6.2], 1.0),
                                     (1.11, [1], [6.2], 6.0)):
        off, w = pm.lines(counts, js) if counts else (np.zeros(1), np.ones(1))
        for o, ww in zip(off, w):
            y += 4000.0 * area * ww / (1 + ((hz - (centre * mhz + o)) / 0.5) ** 2)
    return NMRSpectrum(y, [NMRAxis("frequency", ppm=ppm, hz=hz)],
                       NMRMetadata(vendor="spinsolve", nucleus="1H", sfo1=mhz))


EXPECTED = "¹H NMR (80 MHz) δ 4.74 (s), 3.96 (sept, J = 6.2 Hz), 1.11 (d, J = 6.2 Hz)."


@pytest.mark.parametrize("where", [
    {"regions": [(4.55, 4.95), (3.65, 4.25), (0.95, 1.25)]},
    {"around": [4.74, 3.96, 1.11]},            # one line per multiplet, as Auto-Pick gave
    {},                                         # the whole spectrum
])
def test_the_lines_are_found_in_the_spectrum(where):
    spectrum = _isopropanol()
    results = ma.analyse_spectrum(spectrum, **where)
    assert ma.format_report(results, "1H", 80.0) == EXPECTED
    septet = next(r for r in results if r["pattern"] == "h")
    assert septet["units"] == [1, 6, 15, 20, 15, 6, 1]          # above the raised baseline


def test_a_region_with_only_noise_reports_nothing():
    assert ma.analyse_spectrum(_isopropanol(), regions=[(7.0, 8.0)]) == []


def test_the_noise_estimate():
    assert ma.noise_level(_isopropanol(noise=4.0)) == pytest.approx(4.0, rel=0.15)


def test_heights_from_the_spectrum():
    spectrum = _spectrum([(3.0, [2], [7.0])])
    shifts, _h = _lines(3.0, [2], [7.0])
    results = nmr.analyse_multiplets(shifts, spectrum=spectrum)
    assert results[0]["pattern"] == "t" and results[0]["J_hz"] == pytest.approx([7.0], abs=0.05)


def test_clicked_lines_are_moved_to_the_top_of_their_peaks():
    # A click is rarely on the very top: up to 1.2 Hz off here, which
    # alone makes the pattern unreadable (m) until each line is snapped.
    spectrum = _spectrum([(3.0, [1, 1], [10.0, 4.0]), (1.2, [2], [7.0])])
    truth = np.r_[_lines(3.0, [1, 1], [10.0, 4.0])[0], _lines(1.2, [2], [7.0])[0]]
    clicked = truth + np.random.default_rng(0).uniform(-1.2, 1.2, truth.size) / MHZ
    assert np.max(np.abs(ma.snap_to_maxima(spectrum, clicked) - truth)) * MHZ < 0.05
    assert [r["pattern"] for r in nmr.analyse_multiplets(clicked, spectrum=spectrum)] == ["dd", "t"]
    assert [r["pattern"] for r in nmr.analyse_multiplets(clicked, spectrum=spectrum, snap=False)] \
        == ["m", "m"]


def test_tree_segments_end_on_the_lines():
    shifts, heights = _lines(3.0, [1, 1], [10.0, 4.0])
    r = ma.analyse_multiplet(shifts, MHZ, heights)
    segs = ma.tree_segments(r, top=1.0, bottom=0.5)
    feet = sorted(round(b[0], 6) for _a, b in segs if b[1] == 0.5)
    assert feet == pytest.approx(sorted(shifts), abs=1e-6)
    assert ma.tree_segments(ma.analyse_multiplet([1.0], MHZ), 1.0, 0.5) == []


def test_plot_coupling_trees_draws_on_an_axes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shifts, heights = _lines(3.0, [1, 1], [10.0, 4.0])
    fig, ax = plt.subplots()
    ax.plot(shifts, heights)
    artists = nmr.plot_coupling_trees(ax, [ma.analyse_multiplet(shifts, MHZ, heights)])
    assert len(artists) == 2
    plt.close(fig)


# ---- GUI -----------------------------------------------------------------------------------

from test_gui import qapp, _load_spectrum_directly  # noqa: E402,F401


def test_gui_analyses_the_peak_table_and_sends_multiplets_to_the_fit(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _spectrum([(3.0, [1, 1], [10.0, 4.0]), (1.2, [2], [7.0])]))
    s1, _ = _lines(3.0, [1, 1], [10.0, 4.0])
    s2, _ = _lines(1.2, [2], [7.0])
    for shift in np.r_[s1, s2]:
        window._add_peak_row(shift=float(shift), fwhm=0.002, eta=0.5)
    # The table shows 4 significant figures; the analysis uses the exact value.
    assert window._picked_line_shifts() == pytest.approx(list(np.r_[s1, s2]), abs=1e-12)
    window.multiplet_source_combo.setCurrentIndex(window.multiplet_source_combo.findData("picked"))
    results = window._on_analyse_multiplets_clicked()
    assert [r["pattern"] for r in results] == ["dd", "t"]
    assert "dd, J = 10.0, 4.0 Hz" in window.multiplet_results_label.text()
    assert "<table" in window.multiplet_results_label.html()
    assert window.multiplet_to_fit_button.isEnabled()
    window._on_copy_multiplet_report()
    from PySide6.QtWidgets import QApplication
    assert QApplication.clipboard().text().startswith("¹H NMR (400 MHz) δ 3.00 (dd")
    window._on_multiplets_to_fit_clicked()
    table = window.peak_table
    assert table.rowCount() == 2
    rows = {table.item(r, 4).text(): table.item(r, 5).text() for r in range(2)}
    assert rows == {"dd": "10.00, 4.00", "t": "7.00"}
    defs = window._read_peak_defs()                     # the fit reads them as multiplets
    assert sorted(len(d["couplings"]) for d in defs) == [1, 2]
    assert not window.multiplet_to_fit_button.isEnabled()   # used: analyse again first


def test_gui_finds_the_lines_around_one_picked_line_each(qapp):
    # The user's report: Auto-Pick gave one line per multiplet, and each
    # read as a singlet. Automatic (no regions) grows each from its line.
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _isopropanol())
    for shift in (4.7387, 3.9601, 1.1151, 7.5):          # 7.5: a line elsewhere, kept
        window._add_peak_row(shift=shift, fwhm=0.01, eta=0.5)
    results = window._on_analyse_multiplets_clicked()
    assert [r["pattern"] for r in results] == ["s", "h", "d"]
    assert window._multiplet_report_text == EXPECTED
    window._on_multiplets_to_fit_clicked()
    table = window.peak_table
    cells = sorted((table.item(r, 4).text(), table.item(r, 0).text()) for r in range(table.rowCount()))
    # The singlet's row has no multiplet; the line at 7.5 is kept as it was.
    assert [c for c, _shift in cells] == ["", "", "d", "h"]
    assert "7.5" in [shift for _c, shift in cells]
    assert window.multiplet_source_combo.findData("regions") >= 0


def test_gui_uses_the_integration_regions(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _isopropanol())
    window._integration_regions = lambda: [(4.55, 4.95), (3.65, 4.25), (0.95, 1.25)]
    results = window._on_analyse_multiplets_clicked()
    assert [r["found_by"] for r in results] == ["integration region"] * 3
    assert window._multiplet_report_text == EXPECTED


def test_results_say_which_lines_they_came_from():
    s1, h1 = _lines(7.26, [1], [8.1])
    s2, h2 = _lines(3.45, [1, 1], [10.1, 4.0])
    shifts = np.r_[s2, s1]
    results = nmr.analyse_multiplets(shifts, 400.0, np.r_[h2, h1])
    assert sorted(results[0]["indices"]) == [4, 5] and sorted(results[1]["indices"]) == [0, 1, 2, 3]
    assert results[0]["picked_ppm"] == pytest.approx(sorted(s1, reverse=True))


def test_gui_with_nothing_picked(qapp):
    # Automatic, with no regions and no lines picked, searches the whole
    # spectrum; asking for the picked lines (or around them) says what to do.
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _spectrum([(3.0, [1], [8.0])]))
    results = window._on_analyse_multiplets_clicked()
    assert [r["pattern"] for r in results] == ["d"]
    assert results[0]["found_by"] == "whole spectrum"
    for source in ("picked", "around"):
        window.multiplet_source_combo.setCurrentIndex(
            window.multiplet_source_combo.findData(source))
        assert window._on_analyse_multiplets_clicked() is None
        assert "No lines picked" in window.multiplet_results_label.text()

def test_the_notebook_exports():
    # analyse_spectrum (find the lines, then read them) was missing from
    # opennmr.notebook in 0.10.0; exported under a name that says what it does.
    assert nmr.analyse_spectrum_multiplets is ma.analyse_spectrum
    for name in ("analyse_multiplets", "analyse_multiplet", "format_multiplet_report",
                 "plot_coupling_trees", "multiplet_rows", "analyse_spectrum_multiplets"):
        assert name in nmr.__all__
