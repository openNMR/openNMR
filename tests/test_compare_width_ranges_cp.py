"""2026-10-16 round: Compare per scan / per scan × mass (and RG); click a
peak for its FWHM (1D and 2D, an estimate under overlap); the Fit view's
Measure in ppm and Hz; fits over several ppm ranges; Apply ▸ CP Build-up
(models from Kolodziejski & Klinowski, Chem. Rev. 2002, 102, 613)."""
import dataclasses

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import cp_buildup as cpb
from opennmr.notebook import fit_ranges, intensity_scaling, peak_width
from opennmr.notebook.fitting import build_region_mask


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1400, 900)
    return w


def _lor(x, c, w, a=1.0):
    return a / (1.0 + ((x - c) / (w / 2.0)) ** 2)


def _spectrum(y, ppm, vendor="bruker", ns=16.0, mass=None, raw=None, sfo1=400.0):
    axis = NMRAxis(kind="frequency", ppm=ppm.copy(), hz=ppm * sfo1)
    meta = NMRMetadata(vendor=vendor, nucleus="1H", sfo1=sfo1, ns=ns, mass=mass, raw=dict(raw or {}))
    return NMRSpectrum(data=np.asarray(y, float), axes=[axis], metadata=meta)


# ---------------------------------------------------------------------------
# Compare: per scan, per scan × mass, RG
# ---------------------------------------------------------------------------

PPM = np.linspace(20.0, 0.0, 4001)


def test_per_scan_leaves_per_scan_readers_alone_and_divides_jeol():
    bruker = _spectrum(_lor(PPM, 10, 0.5), PPM, ns=64)
    jeol = _spectrum(64 * _lor(PPM, 10, 0.5), PPM, vendor="jeol", ns=None,
                     raw={"per_scan": False, "params": [{"name": "total_scans", "value": 64}]})
    out = intensity_scaling.scale_for_comparison([bruker, jeol], mode="per_scan")
    a, b = out["spectra"]
    assert np.allclose(a.data, bruker.data)               # already per scan: not divided again
    assert np.allclose(b.data, bruker.data)               # summed JEOL: ÷ its 64 scans
    assert intensity_scaling.number_of_scans(jeol) == 64
    assert out["notes"] == []


def test_per_scan_and_mass_divides_by_mass_and_skips_spectra_without_one():
    s1 = _spectrum(_lor(PPM, 10, 0.5), PPM, mass=0.020)
    s2 = _spectrum(_lor(PPM, 10, 0.5), PPM, mass=None)
    out = intensity_scaling.scale_for_comparison([s1, s2], mode="per_scan_mass")
    assert out["kept"] == [0] and out["skipped"] == [(1, "no sample mass set")]
    assert np.allclose(out["spectra"][0].data, s1.data / 0.020)
    assert np.allclose(s1.data, _lor(PPM, 10, 0.5))        # the original is never changed


def test_receiver_gain_divided_on_request_and_warned_otherwise():
    s1 = _spectrum(_lor(PPM, 10, 0.5), PPM, raw={"acqus": {"RG": 32.0}})
    s2 = _spectrum(4 * _lor(PPM, 10, 0.5), PPM, raw={"acqus": {"RG": 128.0}})
    plain = intensity_scaling.scale_for_comparison([s1, s2])
    assert any("Receiver gains differ" in n for n in plain["notes"])
    divided = intensity_scaling.scale_for_comparison([s1, s2], divide_rg=True)
    assert np.allclose(divided["spectra"][0].data, divided["spectra"][1].data)
    assert not any("differ" in n for n in divided["notes"])


def test_compare_normalize_offers_per_scan_and_per_scan_mass(window):
    items = [window.compare_normalize_combo.itemText(i)
             for i in range(window.compare_normalize_combo.count())]
    assert items[0].startswith("Per scan") and "mass" in items[2]
    assert window.compare_rg_check.isChecked() is False


def test_compare_divides_jeol_by_its_scans_and_by_rg_when_ticked(window):
    bruker = _spectrum(_lor(PPM, 10, 0.5), PPM, ns=64, raw={"acqus": {"RG": 10.0}})
    jeol = _spectrum(64 * _lor(PPM, 10, 0.5), PPM, vendor="jeol", ns=None,
                     raw={"per_scan": False, "params": [{"name": "scans", "value": 64}]})
    for k, s in enumerate((bruker, jeol)):
        window._add_derived_spectrum(s, f"cmp{k}")
    window.spectrum_list.clearSelection()
    for row in (0, 1):
        window.spectrum_list.item(row).setSelected(True)
    window.compare_normalize_combo.setCurrentIndex(0)
    window._refresh_compare_view(switch_tab=True, force_render=True)
    heights = [float(np.max(s.data)) for s in window.compare_canvas._compare_spectra]
    assert heights == pytest.approx([1.0, 1.0], rel=1e-6)
    window.compare_rg_check.setChecked(True)
    window._refresh_compare_view(switch_tab=True, force_render=True)
    assert "receiver gain not recorded" in window.compare_status_label.text()
    heights = [float(np.max(s.data)) for s in window.compare_canvas._compare_spectra]
    assert heights[0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Peak width (FWHM)
# ---------------------------------------------------------------------------

X = np.linspace(0.0, 20.0, 4001)


def test_isolated_peak_width_is_exact_in_ppm_and_hz():
    y = _lor(X, 10.0, 0.5) + np.random.default_rng(0).normal(0, 0.002, X.size)
    r = peak_width.peak_fwhm(X, y, 10.2, hz_per_ppm=400.0)
    assert r["fwhm"] == pytest.approx(0.5, rel=0.01)
    assert r["fwhm_hz"] == pytest.approx(200.0, rel=0.01)
    assert r["position"] == pytest.approx(10.0, abs=0.01)
    assert r["estimate"] is False and r["warnings"] == []


def test_descending_axis_and_negative_peak():
    xd = X[::-1]
    y = -_lor(xd, 4.0, 0.3)
    r = peak_width.peak_fwhm(xd, y, 4.05)
    assert r["fwhm"] == pytest.approx(0.3, rel=0.01)
    assert r["height"] < 0


def test_overlapping_peaks_give_an_estimate_with_a_warning():
    y = _lor(X, 10.0, 0.5) + _lor(X, 10.6, 0.5, 0.8)
    r = peak_width.peak_fwhm(X, y, 9.9)
    assert r["estimate"] is True and r["overlap"] is True
    assert any("ESTIMATE" in w for w in r["warnings"])
    assert any("Overlap" in w for w in r["warnings"])
    assert r["fwhm"] == pytest.approx(0.5, rel=0.15)      # symmetric-side estimate, near the truth
    assert r["alternatives"]                              # local-fit widths given as well


def test_neighbour_tail_above_ten_percent_is_flagged():
    y = _lor(X, 10.0, 0.5) + _lor(X, 11.0, 0.5)
    r = peak_width.peak_fwhm(X, y, 10.0)
    assert r["estimate"] is True


def test_2d_peak_widths_along_both_axes():
    f1 = np.linspace(0, 10, 301)
    f2 = np.linspace(0, 20, 401)
    z = _lor(f1[:, None], 5.0, 0.5) * _lor(f2[None, :], 8.0, 1.0)
    r = peak_width.peak_fwhm_2d(f1, f2, z, 5.3, 8.4, hz_per_ppm_f1=100.0, hz_per_ppm_f2=400.0)
    assert r["f1"]["fwhm"] == pytest.approx(0.5, rel=0.02)
    assert r["f2"]["fwhm"] == pytest.approx(1.0, rel=0.02)
    assert r["f2"]["fwhm_hz"] == pytest.approx(400.0, rel=0.02)
    assert r["estimate"] is False


def test_width_mode_on_a_1d_plot_measures_and_draws(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas(full_resolution=True)
    ppm = np.linspace(20, 0, 8001)
    canvas.show_spectrum(_spectrum(_lor(ppm, 10, 0.5), ppm))
    got = []
    canvas.peakWidthMeasured.connect(got.append)
    canvas.set_width_mode(True)
    assert canvas._width_button.isChecked() and canvas._click_mode_active()
    canvas._measure_width_at(canvas.ax, 10.1, 0.9)
    assert got and got[0]["fwhm"] == pytest.approx(0.5, rel=0.01)
    assert got[0]["fwhm_hz"] == pytest.approx(200.0, rel=0.01)
    assert "ppm" in got[0]["summary"] and "Hz" in got[0]["summary"]
    assert canvas._measurement_artists                      # drawn on the plot
    # one click mode at a time: Measure takes over, Width lets go and clears
    canvas.set_measurement_mode(True)
    assert canvas._width_mode is False and not canvas._measurement_artists


def test_width_mode_on_overlap_reports_estimate(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas(full_resolution=True)
    ppm = np.linspace(20, 0, 8001)
    canvas.show_spectrum(_spectrum(_lor(ppm, 10, 0.5) + _lor(ppm, 10.6, 0.5, 0.8), ppm))
    got = []
    canvas.peakWidthMeasured.connect(got.append)
    canvas.set_width_mode(True)
    canvas._measure_width_at(canvas.ax, 9.95, 0.9)
    assert got[0]["estimate"] is True and "ESTIMATE" in got[0]["summary"]


def test_width_mode_on_a_2d_contour(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()
    f1 = np.linspace(100, 0, 256)
    f2 = np.linspace(20, 0, 512)
    z = _lor(f1[:, None], 50.0, 2.0) * _lor(f2[None, :], 8.0, 0.4)
    meta = NMRMetadata(vendor="bruker", nucleus="1H", nucleus_indirect="13C", sfo1=400.0,
                       sfo1_indirect=100.0)
    spec = NMRSpectrum(data=z, axes=[NMRAxis(kind="frequency", ppm=f1, hz=f1 * 100.0),
                                     NMRAxis(kind="frequency", ppm=f2, hz=f2 * 400.0)],
                       metadata=meta)
    canvas.show_spectrum(spec)
    got = []
    canvas.peakWidthMeasured.connect(got.append)
    canvas.set_width_mode(True)
    canvas._measure_width_at(canvas.ax, 8.1, 51.0)
    r = got[0]
    assert r["kind"] == "2d"
    assert r["f2"]["fwhm"] == pytest.approx(0.4, rel=0.05)
    assert r["f1"]["fwhm"] == pytest.approx(2.0, rel=0.05)
    assert r["f1"]["fwhm_hz"] == pytest.approx(200.0, rel=0.05)


def test_width_click_off_a_spectrum_says_why(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()
    got = []
    canvas.peakWidthMeasured.connect(got.append)
    canvas.set_width_mode(True)
    canvas._measure_width_at(canvas.ax, 1.0, 1.0)
    assert "error" in got[0]


def test_main_window_reports_width_in_the_status_bar(window):
    window._on_peak_width_measured({"summary": "x: FWHM = 0.5 ppm / 200 Hz", "estimate": True})
    assert "WARNING" in window.statusBar().currentMessage()
    assert window.last_peak_width["estimate"] is True


# ---------------------------------------------------------------------------
# Fit view: Measure in ppm AND Hz
# ---------------------------------------------------------------------------

def test_fit_view_measurement_is_in_ppm_and_hz(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()
    ppm = np.linspace(20, 0, 2001)
    spectrum = _spectrum(_lor(ppm, 10, 0.5), ppm, sfo1=400.0)
    y = spectrum.data
    canvas.show_slice_fit(ppm, y, y, components=[y, np.zeros_like(y)], spectrum=spectrum)
    got = []
    canvas.measurementCompleted.connect(got.append)
    canvas.set_measurement_mode(True)
    canvas._record_measurement_point(canvas.ax, 10.0, 0.5)
    canvas._record_measurement_point(canvas.ax, 11.0, 0.5)
    assert got[0]["dx"]["ppm"] == pytest.approx(1.0)
    assert got[0]["dx"]["hz"] == pytest.approx(400.0)
    assert "ppm" in got[0]["summary"] and "Hz" in got[0]["summary"]


def test_fit_view_measurement_without_a_field_still_says_ppm(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()
    x = np.linspace(20, 0, 201)
    y = _lor(x, 10, 0.5)
    canvas.show_slice_fit(x, y, y, components=[y, np.zeros_like(y)])
    got = []
    canvas.measurementCompleted.connect(got.append)
    canvas.set_measurement_mode(True)
    canvas._record_measurement_point(canvas.ax, 10.0, 0.5)
    canvas._record_measurement_point(canvas.ax, 12.0, 0.5)
    assert "2.0000 ppm" in got[0]["summary"]


# ---------------------------------------------------------------------------
# Several fit ranges
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("5 to 10, 12 to 30", [(5, 10), (12, 30)]),
    ("5-10; 12-30", [(5, 10), (12, 30)]),
    ("(-20, -5), (3, 8)", [(-20, -5), (3, 8)]),
    ("-20--5", [(-20, -5)]),
    ("30 12 10 5", [(5, 10), (12, 30)]),
    ("5 to 10, 8 to 12 ppm", [(5, 12)]),
])
def test_parse_ppm_ranges(text, expected):
    assert fit_ranges.parse_ppm_ranges(text) == [tuple(map(float, p)) for p in expected]


@pytest.mark.parametrize("text", ["", "5 to", "a to b", "5 5"])
def test_parse_ppm_ranges_rejects(text):
    with pytest.raises(ValueError):
        fit_ranges.parse_ppm_ranges(text)


def test_ranges_leave_the_gap_out_of_the_fit_mask():
    x = np.linspace(0, 40, 401)
    regions = fit_ranges.parse_ppm_ranges("5 to 10, 12 to 30")
    mask = build_region_mask(x, regions)
    assert mask[(x > 5.5) & (x < 9.5)].all() and mask[(x > 12.5) & (x < 29.5)].all()
    assert not mask[(x > 10.5) & (x < 11.5)].any()
    assert fit_ranges.excluded_gaps(regions) == [(10.0, 12.0)]


def test_fit_panels_offer_several_ranges(window):
    for panel in (window.combined_fit_panel, window.czjzek_fit_panel):
        i = panel.window_combo.findData("ranges")
        assert i >= 0
        panel.window_combo.setCurrentIndex(i)
        panel.window_ranges_edit.setText("5 to 10, 12 to 30")
        assert panel.fit_regions() == [(5.0, 10.0), (12.0, 30.0)]
        panel.window_ranges_edit.setText("5 to")
        with pytest.raises(ValueError):
            panel.fit_regions()
    j = window.jcoupling_fit_panel
    j.window_combo.setCurrentIndex(j.window_combo.findData("ranges"))
    j.window_ranges_edit.setText("1-2, 3-4")
    assert j.fit_regions([], 400e6) == [(1.0, 2.0), (3.0, 4.0)]
    assert j._window_or_none() == ("ranges", [(1.0, 2.0), (3.0, 4.0)])


def test_ranges_box_is_kept_in_the_panel_state(window):
    from opennmr.gui import panel_state
    panel = window.combined_fit_panel
    panel.window_combo.setCurrentIndex(panel.window_combo.findData("ranges"))
    panel.window_ranges_edit.setText("5 to 10, 12 to 30")
    state = panel_state.capture_state(panel)
    panel.window_ranges_edit.setText("")
    panel.window_combo.setCurrentIndex(0)
    panel_state.restore_state(panel, state)
    assert panel.window_ranges_edit.text() == "5 to 10, 12 to 30"
    assert panel.window_combo.currentData() == "ranges"


def test_gaps_between_ranges_are_shaded_on_the_fit_plot(window):
    overlay = window._fit_gap_overlay([(5, 10), (12, 30)])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    overlay(ax)
    shaded = [p for p in ax.patches if p.get_gid() == window.FIT_GAP_GID]
    assert len(shaded) == 1
    assert window._fit_gap_overlay([(5, 10)]) is None
    plt.close(fig)


# ---------------------------------------------------------------------------
# CP build-up
# ---------------------------------------------------------------------------

T = np.array([10, 25, 50, 100, 200, 400, 700, 1000, 2000, 3000, 5000, 8000.0])


def test_eq9_reduces_to_eq10_and_eq10_is_exact_when_tis_equals_t1rho():
    assert np.allclose(cpb.cp_is_t1rho_s(T, 1, 150, 4000, 1e15), cpb.cp_is(T, 1, 150, 4000))
    # T_IS = T1rho: the limit I0 (t/T) exp(-t/T)
    assert cpb.cp_is(np.array([100.0]), 1, 500, 500)[0] == pytest.approx(0.2 * np.exp(-0.2))
    # eq 10 written out as in the review
    t = T
    textbook = 1 / (1 - 150 / 4000) * (np.exp(-t / 4000) - np.exp(-t / 150))
    assert np.allclose(cpb.cp_is(t, 1, 150, 4000), textbook)
    # eq 28 at t = 0 is zero and tends to I0 exp(-t/T1rho) at long times
    assert cpb.cp_iis(np.array([0.0]), 1, 300, 30, 6000, 0.5)[0] == pytest.approx(0.0)
    assert cpb.cp_iis(np.array([5000.0]), 1, 300, 30, 6000, 0.5)[0] == \
        pytest.approx(np.exp(-5000 / 6000), rel=1e-3)


def test_fit_recovers_classical_parameters():
    y = cpb.cp_is(T, 100, 150, 4000) + np.random.default_rng(1).normal(0, 0.5, T.size)
    fit = cpb.fit_cp_buildup(T, y, "is")
    assert fit["params"]["T_IS"] == pytest.approx(150, rel=0.03)
    assert fit["params"]["T1rho_I"] == pytest.approx(4000, rel=0.03)
    assert fit["t_max_us"] == pytest.approx(150 * 4000 / 3850 * np.log(4000 / 150), rel=0.02)
    report = cpb.cp_report(fit)
    assert "T_IS" in report and "Chem. Rev. 2002" in report


def test_fit_with_a_fixed_parameter_and_two_components():
    y = cpb.cp_is(T, 60, 50, 4000) + cpb.cp_is(T, 40, 600, 4000)
    fit = cpb.fit_cp_buildup(T, y, "two_is")
    got = sorted([(fit["params"]["T_IS_a"], fit["params"]["I0_a"]),
                  (fit["params"]["T_IS_b"], fit["params"]["I0_b"])])
    assert got[0] == pytest.approx((50, 60), rel=0.02) and got[1] == pytest.approx((600, 40), rel=0.02)
    fixed = cpb.fit_cp_buildup(T, cpb.cp_is(T, 100, 150, 4000), "is", fixed={"T1rho_I": 4000})
    assert fixed["params"]["T1rho_I"] == 4000 and "T1rho_I" not in fixed["free_names"]


def test_fit_iis_model():
    y = cpb.cp_iis(T, 100, 300, 30, 6000, 0.5)
    fit = cpb.fit_cp_buildup(T, y, "iis", start={"T_df": 200, "T2": 20, "lam": 0.5})
    assert fit["params"]["T_df"] == pytest.approx(300, rel=0.02)
    assert fit["params"]["T1rho_I"] == pytest.approx(6000, rel=0.02)
    assert any("λ" in w for w in fit["warnings"])


def test_short_series_warns_that_t1rho_is_not_sampled():
    t = np.array([10, 50, 100, 200, 300.0])
    fit = cpb.fit_cp_buildup(t, cpb.cp_is(t, 100, 150, 40000), "is")
    assert any("turned over" in w or "barely sampled" in w for w in fit["warnings"])


def _cp_spectrum(t_us, rg=1.0, vendor="bruker"):
    ppm = np.linspace(200, 0, 4001)
    amp = float(cpb.cp_is(np.array([t_us]), 100, 150, 4000)[0])
    p = [0.0] * 64
    p[1], p[15] = 2.5, float(t_us)
    raw = {"acqus": {"P": p, "D": [0.0] * 64, "RG": rg, "NS": 16}}
    return _spectrum(rg * amp * _lor(ppm, 100.0, 2.0), ppm, vendor=vendor, raw=raw, sfo1=100.0)


def test_contact_time_from_parameters_and_numbers():
    s = _cp_spectrum(1500.0)
    assert cpb.contact_time_us(s, "P15") == 1500.0
    assert cpb.contact_time_us(s, "p15") == 1500.0
    assert cpb.contact_time_us(s, 250) == 250.0
    assert cpb.contact_time_us(s, "2 ms") == 2000.0
    s.metadata.raw["acqus"]["D"][20] = 0.002
    assert cpb.contact_time_us(s, "D20") == pytest.approx(2000.0)
    with pytest.raises(ValueError, match="P1 \\(2.5"):
        cpb.contact_time_us(s, "P16")


def test_series_divides_rg_and_can_be_relative_to_a_reference():
    spectra = [_cp_spectrum(t, rg=(4.0 if i % 2 else 1.0)) for i, t in enumerate(T)]
    ref = _cp_spectrum(8000.0)
    series = cpb.cp_series(spectra, (90, 110), "P15", reference=ref, relative_to="none")
    truth = cpb.cp_is(T, 100, 150, 4000)
    ratio = series["y"] / truth
    assert np.allclose(ratio, ratio[0], rtol=1e-3)          # RG removed: one smooth curve
    rel = cpb.cp_series(spectra, (90, 110), "P15", reference=ref, relative_to="reference")
    assert rel["y"][-1] == pytest.approx(1.0, rel=1e-6)
    plain = cpb.cp_series(spectra, (90, 110), "P15", divide_rg=False)
    assert any("Receiver gains differ" in n for n in plain["notes"])


def test_cp_panel_end_to_end(window):
    for k, t in enumerate(T):
        window._add_derived_spectrum(_cp_spectrum(t, rg=2.0), f"cp{k}")
    window._add_derived_spectrum(_cp_spectrum(8000.0), "ref")
    window._show_side_tab("Apply")
    window._cp_buildup_section.setExpanded(True)
    panel = window.cp_buildup_panel
    panel.refresh()
    rows = [r for r, e in enumerate(panel._entries) if str(e["label"]).startswith("cp")]
    panel.set_checked(rows)
    panel.time_parameter_edit.setText("P15")
    assert panel.read_contact_times() == len(T)
    panel.set_reference(next(r for r, e in enumerate(panel._entries) if str(e["label"]).startswith("ref")))
    panel.add_region(hi=110.0, lo=90.0)
    assert panel.integrate()
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    assert np.allclose(np.sort(panel.series["t_us"]), T)
    panel.model_combo.setCurrentIndex(panel.model_combo.findData("is"))
    assert panel.run_fit()
    assert panel.fit["params"]["T_IS"] == pytest.approx(150, rel=0.02)
    assert panel.fit["params"]["T1rho_I"] == pytest.approx(4000, rel=0.02)
    assert "T_IS" in panel.fit_results_label.text()
    kept = window.session.spectra[rows[0]].metadata.raw["fits"]["cp_buildup"]
    assert "CP build-up" in kept["label"]


def test_cp_panel_typed_times_and_fixed_parameter(window):
    for k, t in enumerate(T[:8]):
        window._add_derived_spectrum(_cp_spectrum(t), f"cpt{k}")
    window._cp_buildup_section.setExpanded(True)
    panel = window.cp_buildup_panel
    panel.refresh()
    rows = [r for r, e in enumerate(panel._entries) if str(e["label"]).startswith("cpt")]
    panel.set_checked(rows)
    for r, t in zip(rows, T[:8]):
        panel.set_contact_time(r, float(t))
    panel.add_region(hi=110.0, lo=90.0)
    assert panel.integrate()
    panel.set_parameter("T1rho_I", start=4000.0, fixed=True)
    assert panel.run_fit()
    assert panel.fit["params"]["T1rho_I"] == 4000.0
    assert panel.fit["params"]["T_IS"] == pytest.approx(150, rel=0.02)


def test_cp_panel_refuses_missing_contact_times(window, _dialogs):
    for k in range(3):
        window._add_derived_spectrum(_cp_spectrum(100.0 * (k + 1)), f"cpm{k}")
    panel = window.cp_buildup_panel
    panel.refresh()
    panel.set_checked([r for r, e in enumerate(panel._entries) if str(e["label"]).startswith("cpm")])
    assert panel.integrate() is False
    assert any("No contact time" in text for _n, text in _dialogs)


def test_cp_section_is_in_apply_and_can_be_hidden(window):
    from opennmr.gui.main_window import MainWindow
    assert "cp_buildup" in MainWindow.OPTIONAL_SECTIONS
    assert window._cp_buildup_section.title().startswith("CP Build")
