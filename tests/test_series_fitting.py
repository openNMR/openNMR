"""Lineshape models (CSA, quadrupolar, combined) applied to every slice
of a pseudo-2D series, for T1/T2 -- notebook.series_fitting and the
GUI's "Fit Every Slice" / "Export Fit to Pseudo-2D" (explicit user
request, 2026-09-28). Known-answer tests: synthetic series built from
the models themselves, with known relaxation times."""
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import bridge, series_fitting as SF
from opennmr.notebook import csa_fitting as C
from opennmr.notebook import simulations as S
from opennmr.processing.relaxation import NMR2D_fit_relaxation

LARMOR_MHZ = 100.0
DELAYS = np.array([0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
T1_A, T1_B = 0.15, 0.9


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _csa_shape(x, d_iso, delta):
    return C.csa_model_full(x, d_iso, delta, 0.5, 3.0, convention="Haeberlen", static=True,
                            powder_scheme="zcw", n_zcw=2000)


def _pseudo2d(rows, ppm_ascending, nucleus="13C", sfo1=LARMOR_MHZ):
    """Stored the way spectra usually are: ppm descending along the row."""
    ppm = ppm_ascending[::-1]
    data = np.asarray(rows)[:, ::-1]
    return NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, len(rows) + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * sfo1)],
        metadata=NMRMetadata(vendor="simulated", nucleus=nucleus, sfo1=sfo1, pseudo_2d=True))


def _ir_csa_series(noise=0.004, seed=0, baseline=(0.05, 1e-4)):
    """Two CSA sites relaxing with different T1, inversion recovery,
    on a sloping baseline, with noise."""
    x = np.linspace(-300.0, 300.0, 1024)
    a = _csa_shape(x, -40.0, -120.0)
    b = _csa_shape(x, 60.0, 80.0)
    rng = np.random.default_rng(seed)
    rows = []
    for t in DELAYS:
        ma = 1.0 * (1 - 2 * np.exp(-t / T1_A))
        mb = 0.6 * (1 - 2 * np.exp(-t / T1_B))
        rows.append(ma * a + mb * b + baseline[0] + baseline[1] * x
                    + rng.normal(0, noise, x.size))
    return x, _pseudo2d(rows, x)


def _csa_reference(spectrum, row):
    results = bridge.results_from_spectrum(spectrum, row=row)
    sites = [{"label": "A", "d_iso": -35.0, "span_or_Delta": -110.0, "skew_or_eta": 0.45,
              "fwhm": 3.5, "type": "gaussian"},
             {"label": "B", "d_iso": 55.0, "span_or_Delta": 85.0, "skew_or_eta": 0.45,
              "fwhm": 3.5, "type": "gaussian"}]
    popt, _pcov, _free, _scale, info = C.fit_csa_spectrum(
        results, sites, static=True, powder_scheme="zcw", n_zcw=2000)
    return {"popt": popt, "info": info, "x": np.sort(results["x_ppm"])}


# -- the engine -------------------------------------------------------------------

def test_csa_inversion_recovery_recovers_both_t1s():
    x, spectrum = _ir_csa_series()
    fit = _csa_reference(spectrum, row=len(DELAYS) - 1)          # the fully relaxed slice
    labels, xc, comps = SF.model_components("csa", fit)
    assert labels == ["A", "B"] and len(comps) == 2
    series = SF.fit_series_with_lineshapes(spectrum, xc, comps, labels, kind="csa",
                                           reference_row=len(DELAYS))
    assert [r["index"] for r in series] == list(range(1, len(DELAYS) + 1))
    # Signed: early slices are inverted.
    assert series[0]["amplitudes"][0] < 0 and series[-1]["amplitudes"][0] > 0
    assert series[-1]["amplitudes"] == pytest.approx([1.0, 1.0], abs=0.02)
    fits = NMR2D_fit_relaxation(series, model="inversion_recovery", vdlist=DELAYS,
                                value_source="area", show_report=False)
    assert fits[0]["params"]["T1"] == pytest.approx(T1_A, rel=0.03)
    assert fits[1]["params"]["T1"] == pytest.approx(T1_B, rel=0.03)
    assert [f["peak_label"] for f in fits] == ["A", "B"]
    # "amplitude" (relative to the reference slice) gives the same T1s.
    fits = NMR2D_fit_relaxation(series, model="inversion_recovery", vdlist=DELAYS,
                                value_source="amplitude", show_report=False)
    assert fits[0]["params"]["T1"] == pytest.approx(T1_A, rel=0.03)
    assert fits[1]["params"]["T1"] == pytest.approx(T1_B, rel=0.03)


def test_areas_are_the_site_areas_in_each_slice():
    x, spectrum = _ir_csa_series(noise=0.0, baseline=(0.0, 0.0))
    fit = _csa_reference(spectrum, row=len(DELAYS) - 1)
    labels, xc, comps = SF.model_components("csa", fit)
    series = SF.fit_series_with_lineshapes(spectrum, xc, comps, labels)
    a = _csa_shape(x, -40.0, -120.0)
    for t, r in zip(DELAYS, series):
        truth = (1 - 2 * np.exp(-t / T1_A)) * np.trapezoid(a, x)
        assert r["integrals"][0] == pytest.approx(truth, rel=0.01, abs=0.01 * abs(np.trapezoid(a, x)))


def test_reported_errors_match_the_scatter():
    """The 1-sigma errors are honest: over noise realisations, the
    spread of a fitted amplitude matches the error the fit reports."""
    x = np.linspace(-300.0, 300.0, 1024)
    a = _csa_shape(x, -40.0, -120.0)
    b = _csa_shape(x, 60.0, 80.0)
    rng = np.random.default_rng(3)
    rows = [0.3 * a + 0.5 * b + rng.normal(0, 0.02, x.size) for _ in range(300)]
    series = SF.fit_series_with_lineshapes(_pseudo2d(rows, x), x, [a, b], ["A", "B"])
    values = np.array([r["amplitudes"] for r in series])
    reported = np.array([r["amplitude_errors"] for r in series]).mean(axis=0)
    assert values.mean(axis=0) == pytest.approx([0.3, 0.5], abs=0.01)
    assert values.std(axis=0, ddof=1) == pytest.approx(reported, rel=0.15)


def test_quadrupolar_t2_decay():
    nu0 = 130.3e6
    x = np.linspace(-80.0, 60.0, 2048)
    shape = S.quadrupolar_spectrum(2.5, 4.2e6, 0.35, nu0, x, transitions="ct",
                                   delta_iso=12.0, fwhm=0.6, n_zcw=28657)["normalised_real_spec"]
    t2 = 0.012
    delays = np.linspace(0.0005, 0.05, 10)
    rng = np.random.default_rng(1)
    rows = [np.exp(-t / t2) * shape + rng.normal(0, 0.003, x.size) for t in delays]
    spectrum = _pseudo2d(rows, x, nucleus="27Al", sfo1=nu0 / 1e6)
    from opennmr.notebook import quadrupolar_fitting as Q
    results = bridge.results_from_spectrum(spectrum, row=0)
    popt, _p, _f, _s, info = Q.fit_quadrupolar_spectrum(
        results, [{"d_iso": 12.5, "C_Q": 4.1, "eta_Q": 0.3, "fwhm": 0.7}], I=2.5)
    labels, xc, comps = SF.model_components("quadrupolar", {"popt": popt, "info": info,
                                                            "x": results["x_ppm"]})
    series = SF.fit_series_with_lineshapes(spectrum, xc, comps, labels, kind="quadrupolar")
    fit = NMR2D_fit_relaxation(series, model="exponential_decay", vdlist=delays,
                               value_source="area", show_report=False)[0]
    assert fit["params"]["T2"] == pytest.approx(t2, rel=0.03)


def test_combined_components_are_read_from_site_spectra():
    x = np.linspace(-10, 10, 201)
    fit = {"x": x[::-1], "site_spectra": {"P": np.exp(-(x[::-1] - 2) ** 2),
                                          "Q": np.exp(-(x[::-1] + 3) ** 2)}}
    labels, xc, comps = SF.model_components("combined", fit)
    assert labels == ["P", "Q"]
    assert np.all(np.diff(xc) > 0)
    assert xc[np.argmax(comps[0])] == pytest.approx(2.0)
    rows = [k * comps[0] + (5 - k) * comps[1] for k in range(1, 5)]
    series = SF.fit_series_with_lineshapes(_pseudo2d(rows, xc), xc, comps, labels,
                                           baseline="none")
    assert [r["amplitudes"] for r in series] == [pytest.approx([k, 5 - k]) for k in range(1, 5)]


def test_fit_window_and_options():
    x = np.linspace(0, 100, 500)
    c = np.exp(-((x - 30) / 2) ** 2)
    junk = 5 * np.exp(-((x - 80) / 2) ** 2)          # outside the window: ignored
    rows = [2.0 * c + junk, 1.0 * c + junk]
    series = SF.fit_series_with_lineshapes(_pseudo2d(rows, x), x, [c], ["only"],
                                           fit_regions=[(0, 60)])
    assert [r["amplitudes"][0] for r in series] == pytest.approx([2.0, 1.0], abs=1e-6)
    assert series[0]["fit_mask"].sum() < x.size
    with pytest.raises(ValueError):
        SF.fit_series_with_lineshapes(_pseudo2d(rows, x), x, [c], baseline="quadratic")
    with pytest.raises(ValueError):
        SF.model_components("isotropic", {})
    report = SF.series_fit_report(series)
    assert "only area" in report and "R²" in report


# -- the GUI --------------------------------------------------------------------------

@pytest.fixture
def window(qapp, monkeypatch):
    from opennmr.gui.main_window import MainWindow
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    w = MainWindow()
    w._shown = shown
    return w


def _load_ir(window):
    _x, spectrum = _ir_csa_series()
    index = window._add_derived_spectrum(spectrum, "13C T1 series")
    window._select_single_row(index)
    return window.session.spectra[index]          # the session's own object, as the GUI uses


def test_gui_csa_series_from_one_slice_to_fit_curve_and_export(window):
    spectrum = _load_ir(window)
    fit_button, export_button, status = window._model_series_widgets["csa"]
    assert not fit_button.isEnabled() and not export_button.isEnabled()
    # Fit the last (relaxed) slice with the CSA model, as a user would.
    window.slice_spin.setValue(len(DELAYS))
    window.csa_static_combo.setCurrentIndex(window.csa_static_combo.findData(True))
    window.csa_powder_combo.setCurrentIndex(window.csa_powder_combo.findData("zcw"))
    window.csa_orientations_spin.setValue(2000)
    window.csa_render_orientations_spin.setValue(2000)
    window._add_csa_site(("A", "-35", "-110", "0.45", "3.5"))
    window._add_csa_site(("B", "55", "85", "0.45", "3.5"))
    window.csa_window_combo.setCurrentIndex(window.csa_window_combo.findData("all"))
    window._on_csa_fit_clicked()
    assert window._csa_fit_result["row"] == len(DELAYS) - 1
    assert fit_button.isEnabled() and not export_button.isEnabled()

    fit_button.click()
    staged = spectrum.metadata.raw["slice_fit_results"]
    assert len(staged) == len(DELAYS) and staged[0]["model"] == "csa"
    assert export_button.isEnabled()
    assert window.export_csa_pseudo2d_button is export_button
    assert "Fitted 9 slices" in status.text()
    assert "2 sites" in window._describe_staged_measurement()
    assert "series_csa" in spectrum.metadata.raw["fits"]

    # Dynamics > Fit Curve recovers both T1s.
    window.delay_source_combo.setCurrentIndex(1)                  # manual list
    window.delay_unit_combo.setCurrentIndex(window.delay_unit_combo.findText("s"))
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in DELAYS))
    window.relaxation_model_combo.setCurrentIndex(
        window.relaxation_model_combo.findData("inversion_recovery"))
    window.value_source_combo.setCurrentIndex(window.value_source_combo.findData("area"))
    window._on_fit_curve_button_clicked()
    t1 = [r["params"]["T1"] for r in window._relaxation_fit_results]
    assert t1 == [pytest.approx(T1_A, rel=0.03), pytest.approx(T1_B, rel=0.03)]

    # Export to pseudo-2D: one fitted trace per slice.
    n_before = len(window.session)
    export_button.click()
    exported = window.session.spectra[n_before]
    assert exported.ndim == 2 and exported.data.shape == spectrum.data.shape
    assert np.allclose(exported.data[-1], staged[-1]["y_fit"])

    # The CSV fit report lists every slice and site.
    csv_text = window._build_fit_report_csv(staged)
    assert csv_text.count("\n") >= 2 * len(DELAYS)


def test_gui_series_buttons_need_a_fit_on_this_pseudo2d(window):
    _load_ir(window)
    for fit_button, export_button, _status in window._model_series_widgets.values():
        assert not fit_button.isEnabled() and not export_button.isEnabled()
    window._on_model_series_clicked("csa")        # nothing fitted: explains, does nothing
    assert window._shown and "Fit one slice" in window._shown[-1][1]


def test_gui_quadrupolar_and_combined_sections_have_the_series_controls(window):
    assert set(window._model_series_widgets) == {"csa", "quadrupolar", "combined"}
    assert window.quad_fit_panel.parentWidget() is not None


# -- the route from the user report: extract a slice, keep it, fit it ------------------

def _extract_and_keep(window, spectrum, row):
    """What Extract Slice ▸ Keep does: the parent is the session's own
    spectrum object (the one the canvas was showing)."""
    from opennmr.processing import slicing
    trace = slicing.get_trace(spectrum, axis="f1", index=row)
    window._on_slice_keep_requested((trace, f"F1 slice {row}", spectrum, "f1", row))
    return len(window.session) - 1


def _fit_csa_on_selected(window):
    window.csa_static_combo.setCurrentIndex(window.csa_static_combo.findData(True))
    window.csa_powder_combo.setCurrentIndex(window.csa_powder_combo.findData("zcw"))
    window.csa_orientations_spin.setValue(2000)
    window.csa_render_orientations_spin.setValue(2000)
    window._add_csa_site(("A", "-35", "-110", "0.45", "3.5"))
    window._add_csa_site(("B", "55", "85", "0.45", "3.5"))
    window.csa_window_combo.setCurrentIndex(window.csa_window_combo.findData("all"))
    window._on_csa_fit_clicked()


def test_gui_fit_on_an_extracted_slice_fits_every_slice_of_its_parent(window):
    spectrum = _load_ir(window)
    parent_index = window._selected_index()
    child = _extract_and_keep(window, spectrum, len(DELAYS))
    assert window._selected_index() == child
    fit_button, export_button, status = window._model_series_widgets["csa"]
    assert not fit_button.isEnabled()                 # nothing fitted on it yet
    _fit_csa_on_selected(window)
    assert fit_button.isEnabled()                     # the user's report: this stayed off
    assert "every slice of" in fit_button.toolTip()

    fit_button.click()
    # The parent is selected, carries the series, and Fit Curve/Export apply.
    assert window._selected_index() == parent_index
    staged = spectrum.metadata.raw["slice_fit_results"]
    assert len(staged) == len(DELAYS) and staged[0]["reference_row"] == len(DELAYS)
    assert export_button.isEnabled()
    assert "shapes from slice 9" in status.text() and "⚠" not in status.text()
    fits = NMR2D_fit_relaxation(staged, model="inversion_recovery", vdlist=DELAYS,
                                value_source="area", show_report=False)
    assert fits[0]["params"]["T1"] == pytest.approx(T1_A, rel=0.03)
    assert fits[1]["params"]["T1"] == pytest.approx(T1_B, rel=0.03)


def test_gui_parent_selected_uses_the_fit_saved_on_its_extracted_slice(window):
    spectrum = _load_ir(window)
    parent_index = window._selected_index()
    _extract_and_keep(window, spectrum, len(DELAYS))
    _fit_csa_on_selected(window)
    window._select_single_row(parent_index)           # back to the 2D: its own panels are empty
    fit_button, export_button, _status = window._model_series_widgets["csa"]
    assert window._csa_fit_result is None
    assert fit_button.isEnabled() and not export_button.isEnabled()
    fit_button.click()
    assert len(spectrum.metadata.raw["slice_fit_results"]) == len(DELAYS)
    assert export_button.isEnabled()


def test_gui_warns_when_the_extracted_slice_was_changed_on_its_own(window):
    spectrum = _load_ir(window)
    child = _extract_and_keep(window, spectrum, len(DELAYS))
    kept = window.session.spectra[child]
    kept.data = kept.data * 1.5                        # e.g. phased/scaled but not applied back
    _fit_csa_on_selected(window)
    fit_button, _export, status = window._model_series_widgets["csa"]
    fit_button.click()
    assert "⚠" in status.text() and "Apply Phase to Source 2D" in status.text()


def test_gui_slice_reopened_without_its_link_finds_its_parent_by_content(window):
    spectrum = _load_ir(window)
    parent_index = window._selected_index()
    child = _extract_and_keep(window, spectrum, 4)
    for key in ("slice_parent", "slice_axis", "slice_index"):   # as after reopening a session
        window.session.spectra[child].metadata.raw.pop(key, None)
    assert window._slice_parent_of(child) == (parent_index, 4)
