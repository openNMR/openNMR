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


def test_a_triplet_without_heights_is_not_misread_as_a_coupling_of_14_hz():
    shifts, _h = _lines(1.0, [2], [7.0])
    r = ma.analyse_multiplet(shifts, MHZ)              # no heights: 1 unit each
    assert r["pattern"] == "m"                         # never "dd, J = 14, 7"
    assert any("No heights" in n for n in r["notes"])


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


def test_editing_the_table_after_an_analysis_needs_a_new_one(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _spectrum([(3.0, [1], [8.0])]))
    for shift in _lines(3.0, [1], [8.0])[0]:
        window._add_peak_row(shift=float(shift), fwhm=0.002, eta=0.5)
    window._on_analyse_multiplets_clicked()
    assert window.multiplet_to_fit_button.isEnabled()
    window._add_peak_row(shift=1.0, fwhm=0.002, eta=0.5)       # the rows have moved on
    assert not window.multiplet_to_fit_button.isEnabled()
    window._on_multiplets_to_fit_clicked()                     # and nothing is replaced
    assert window.peak_table.rowCount() == 3


def test_results_say_which_lines_they_came_from():
    s1, h1 = _lines(7.26, [1], [8.1])
    s2, h2 = _lines(3.45, [1, 1], [10.1, 4.0])
    shifts = np.r_[s2, s1]
    results = nmr.analyse_multiplets(shifts, 400.0, np.r_[h2, h1])
    assert sorted(results[0]["indices"]) == [4, 5] and sorted(results[1]["indices"]) == [0, 1, 2, 3]
    assert results[0]["picked_ppm"] == pytest.approx(sorted(s1, reverse=True))


def test_gui_says_what_to_do_with_nothing_picked(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _spectrum([(3.0, [1], [8.0])]))
    assert window._on_analyse_multiplets_clicked() is None
    assert "No lines picked" in window.multiplet_results_label.text()
