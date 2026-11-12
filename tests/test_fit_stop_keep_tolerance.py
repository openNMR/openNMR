"""Stopping a fit and keeping where it got to, and the convergence
tolerance (explicit user request, 2026-11-12: "rather than erasing all
progress midway, the current fit ... can be displayed. Users should also
be able to control the tolerance where the spectrum is considered
converged (accessible in the settings menu)")."""

import numpy as np
import pytest

from opennmr.core import fit_progress as fp

from test_fit_progress import _dialogs, _press  # noqa: F401 -- fixture, helper
from test_gui import qapp, _load_spectrum_directly, _pfg_series_window  # noqa: F401

X = np.linspace(0, 10, 400)
P0 = [1.0, 4.0, 1.5, 0.0]
BOUNDS = ([0, 0, 0.1, -1], [10, 10, 5, 1])


def _model(x, a, b, c, d):
    return a * np.exp(-((x - b) / c) ** 2) + d * np.sin(x)


Y = _model(X, 2.0, 5.0, 0.7, 0.3) + np.random.default_rng(0).normal(0, 0.01, X.size)


def _cost(p):
    r = _model(X, *p) - Y
    return float(r @ r)


@pytest.fixture(autouse=True)
def _default_tolerance():
    yield
    fp.set_tolerance(None)


def _keep_after(n):
    return lambda st: fp.KEEP if st.evaluations > n else True


def test_keep_returns_the_best_so_far_without_uncertainty():
    with fp.watch(_keep_after(12), min_interval=0.0) as status:
        popt, pcov = fp.limited_curve_fit(fp.tracked(_model, Y, n_free=4), X, Y, p0=P0,
                                          bounds=BOUNDS)
    assert status.stopped_early
    assert _cost(popt) < _cost(P0)                 # where it got to, not the start
    assert np.isnan(pcov).all()                    # not converged: no uncertainty
    full, _ = fp.limited_curve_fit(_model, X, Y, p0=P0, bounds=BOUNDS)
    assert _cost(full) < _cost(popt)               # it really had not finished


def test_keep_at_the_first_report_keeps_the_starting_point():
    with fp.watch(lambda st: fp.KEEP, min_interval=0.0) as status:
        popt, _pcov = fp.limited_curve_fit(fp.tracked(_model, Y, n_free=4), X, Y, p0=P0,
                                           bounds=BOUNDS)
    assert status.stopped_early and popt == pytest.approx(P0)


def test_after_a_keep_no_further_fit_runs_but_loops_carry_on():
    with fp.watch(_keep_after(5), min_interval=0.0):
        fp.limited_curve_fit(fp.tracked(_model, Y, n_free=4), X, Y, p0=P0, bounds=BOUNDS)
        assert fp.stopped_early()
        fp.check_cancelled()                       # does not raise
        with pytest.raises(fp.FitStopped):
            fp.limited_curve_fit(fp.tracked(_model, Y, n_free=4), X, Y, p0=P0)
        with pytest.raises(RuntimeError):          # caught like a non-converged start
            fp.limited_least_squares(lambda p: _model(X, *p) - Y, np.array(P0, float))
    assert not fp.stopped_early()                  # the next watch starts afresh


def test_keep_in_least_squares():
    with fp.watch(_keep_after(12), min_interval=0.0):
        sol = fp.limited_least_squares(
            fp.tracked_residuals(lambda p: _model(X, *p) - Y, n_free=4), np.array(P0, float),
            label="The test fit")
    assert sol.stopped_early and not sol.success and sol.not_converged is None
    assert sol.jac.shape == (X.size, 4) and _cost(sol.x) < _cost(P0)
    assert "stopped before converging" in sol.message


def test_discard_still_discards():
    with fp.watch(lambda st: False, min_interval=0.0):
        with pytest.raises(fp.FitCancelled):
            fp.limited_curve_fit(fp.tracked(_model, Y, n_free=4), X, Y, p0=P0)


def test_a_fitter_that_bypasses_the_helpers_cannot_keep():
    from scipy.optimize import curve_fit
    with fp.watch(lambda st: fp.KEEP, min_interval=0.0):
        with pytest.raises(fp.FitCancelled):
            curve_fit(fp.tracked(_model, Y, n_free=4), X, Y, p0=P0)


# ---- tolerance -----------------------------------------------------------------------------

def test_tolerance_default_changes_nothing():
    assert fp.tolerance() == fp.DEFAULT_TOLERANCE
    kwargs = {}
    fp._apply_tolerance(kwargs)
    assert kwargs == {}                            # scipy's own defaults, exactly as before


def test_tolerance_is_passed_on_once_set_unless_the_fit_sets_its_own():
    fp.set_tolerance(1e-4)
    kwargs = {}
    fp._apply_tolerance(kwargs)
    assert kwargs == {"ftol": 1e-4, "xtol": 1e-4}
    own = {"ftol": 1e-12, "xtol": 1e-12}
    fp._apply_tolerance(own)
    assert own == {"ftol": 1e-12, "xtol": 1e-12}
    with fp.convergence_tolerance(1e-10):
        assert fp.tolerance() == 1e-10
    assert fp.tolerance() == 1e-4
    with pytest.raises(ValueError):
        fp.set_tolerance(0.5)


def test_a_looser_tolerance_stops_sooner():
    counts = {}
    for tol in (1e-10, 1e-3):
        fp.set_tolerance(tol)
        calls = [0]

        def counted(x, *p):
            calls[0] += 1
            return _model(x, *p)
        fp.limited_curve_fit(counted, X, Y, p0=P0, bounds=BOUNDS)
        counts[tol] = calls[0]
    assert counts[1e-3] < counts[1e-10]


# ---- the window ----------------------------------------------------------------------------

def test_settings_tolerance_is_applied_and_remembered(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    combo = window.fit_tolerance_combo
    assert combo.currentData() == fp.DEFAULT_TOLERANCE
    combo.setCurrentIndex(combo.findData(1e-5))
    assert fp.tolerance() == 1e-5 and window._ui_settings["fit_tolerance"] == 1e-5
    assert MainWindow().fit_tolerance_combo.currentData() == 1e-5


def test_the_dialog_has_keep_and_discard(qapp):
    from opennmr.gui.fit_progress_dialog import FitProgressDialog
    dialog = FitProgressDialog("Spectral fit", "text")
    assert dialog.keep_button.text() == "Stop and Keep Best"
    dialog.keep_button.click()
    assert dialog.answer == fp.KEEP
    dialog.discard_button.click()                  # the first answer stands
    assert dialog.answer == fp.KEEP
    other = FitProgressDialog("Spectral fit", "text")
    other.reject()                                 # Esc / the close button
    assert other.answer == "discard"


def test_quadrupolar_fit_keeps_its_best_so_far(qapp, _dialogs, monkeypatch):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    from test_gui_quadrupolar_fit import _load
    window = MainWindow()
    panel = window.quad_fit_panel
    _load(window)
    panel.estimate()
    _press(monkeypatch, fp.KEEP)
    panel.fit()
    assert panel.result is not None and not _dialogs, _dialogs
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    assert "stopped early" in window.statusBar().currentMessage()


def test_a_series_stopped_part_way_is_not_kept_but_seeds_the_table(qapp, _dialogs,  # noqa: F811
                                                                  monkeypatch):
    window, _b, spectrum = _pfg_series_window(qapp)
    window.peak_table.item(0, 0).setText("5.02")         # a start a little off
    _press(monkeypatch, fp.KEEP)
    window._on_fit_spectrum_clicked()
    assert any("stopped at slice" in text for _kind, text in _dialogs)
    assert "slice_fit_results" not in spectrum.metadata.raw     # not kept as the fit
    assert window.peak_table.rowCount() == 1                    # the table, from the slice reached


def test_a_1d_fit_stopped_early_is_shown(qapp, _dialogs, monkeypatch):  # noqa: F811
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    from opennmr.gui.main_window import MainWindow
    from opennmr.processing.fitting import gaussian
    ppm = np.linspace(0, 10, 900)
    spectrum = NMRSpectrum(gaussian(ppm, 5.0, 0.05, 1.0),
                           [NMRAxis("frequency", ppm=ppm, hz=ppm * 400)],
                           NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.03, fwhm=0.08, eta=0.0)
    _press(monkeypatch, fp.KEEP)
    window._on_fit_spectrum_clicked()
    assert not _dialogs, _dialogs
    assert spectrum.metadata.raw.get("slice_fit_results") or window._slice_fit_results
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    assert "stopped early" in window.statusBar().currentMessage()
