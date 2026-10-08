"""Round of 2026-10-08 (explicit user requests):

* a user-editable maximum number of iterations for every fit, 2000 to
  start with, and suggestions for what to try when a fit reaches it;
* "for fitting any anisotropic interactions, it is sensible to have at
  least 3-5 sidebands for accuracy";
* a switch for which way scrolling over a 2D contour moves its levels;
* Cmd/Ctrl+. toggles the crosshair.
"""
import json

import numpy as np
import pytest
from PySide6.QtGui import QKeySequence

from opennmr.core import fit_progress
from opennmr.notebook import fit_advice
from opennmr.gui.widgets.canvas import MplCanvas

from test_gui_canvas import _body_scroll_event, _make_2d, qapp  # noqa: F401 -- qapp is a fixture


@pytest.fixture(autouse=True)
def _default_limit():
    fit_progress.set_max_iterations(None)
    yield
    fit_progress.set_max_iterations(None)


def _gaussian(x, a, c, w):
    return a * np.exp(-0.5 * ((x - c) / w) ** 2)


def _hard_problem():
    x = np.linspace(-10, 10, 400)
    y = _gaussian(x, 1.0, 3.0, 0.4)
    return x, y, [0.2, -4.0, 3.0]          # starts far from the answer


# ============================================================
# Maximum iterations: core
# ============================================================

def test_the_suggested_limit_is_2000_and_is_editable():
    assert fit_progress.DEFAULT_MAX_ITERATIONS == 2000
    assert fit_progress.max_iterations() == 2000
    fit_progress.set_max_iterations(750)
    assert fit_progress.max_iterations() == 750
    assert fit_progress.max_iterations(10) == 10             # an explicit value wins
    with fit_progress.iteration_limit(5):
        assert fit_progress.max_iterations() == 5
    assert fit_progress.max_iterations() == 750
    assert fit_progress.set_max_iterations(None) == 2000


@pytest.mark.parametrize("bad", [0, -3])
def test_a_limit_below_one_is_refused(bad):
    with pytest.raises(ValueError):
        fit_progress.set_max_iterations(bad)


def test_a_fit_within_the_limit_is_unchanged():
    x = np.linspace(-5, 5, 200)
    y = _gaussian(x, 2.0, 0.5, 0.8)
    popt, _pcov = fit_progress.limited_curve_fit(_gaussian, x, y, p0=[1.5, 0.3, 1.0],
                                                 bounds=([0, -5, 0.01], [10, 5, 5]))
    assert popt == pytest.approx([2.0, 0.5, 0.8], rel=1e-6)


@pytest.mark.parametrize("bounded", [True, False])
def test_reaching_the_limit_raises_with_suggestions_and_the_best_point(bounded):
    x, y, p0 = _hard_problem()
    bounds = ([0, -10, 0.01], [10, 10, 10]) if bounded else None
    with fit_progress.iteration_limit(2):
        with pytest.raises(fit_progress.FitNotConverged) as info:
            fit_progress.limited_curve_fit(_gaussian, x, y, p0=p0, bounds=bounds,
                                           label="The test fit")
    exc = info.value
    assert isinstance(exc, RuntimeError)            # multi-start fits still skip such a start
    assert exc.max_iterations == 2
    text = str(exc)
    assert text.startswith("The test fit did not converge within the maximum of 2 iterations.")
    assert "What to try next" in text
    assert "Settings ▸ Fitting" in text and "set_max_iterations" in text
    assert "Free fewer parameters (3 are free now)" in text
    assert exc.best_params is not None and exc.best_params.shape == (3,)
    assert exc.best_cost <= exc.start_cost


def test_more_iterations_let_the_same_fit_converge():
    x, y, p0 = _hard_problem()
    bounds = ([0, -10, 0.01], [10, 10, 10])
    with fit_progress.iteration_limit(2):
        with pytest.raises(fit_progress.FitNotConverged):
            fit_progress.limited_curve_fit(_gaussian, x, y, p0=p0, bounds=bounds)
    popt, _ = fit_progress.limited_curve_fit(_gaussian, x, y, p0=[0.8, 2.5, 0.5], bounds=bounds)
    assert popt[1] == pytest.approx(3.0, abs=1e-6)


def test_suggestions_say_whether_more_iterations_would_help():
    still = fit_progress.not_converged_suggestions(100, start_cost=10.0, best_cost=2.0)
    assert "still improving" in still[0] and "200" in still[0]
    stuck = fit_progress.not_converged_suggestions(100, start_cost=10.0, best_cost=10.0)
    assert "unlikely to help" in stuck[0]


def test_least_squares_limit_raises_or_flags():
    x, y, p0 = _hard_problem()

    def residuals(p):
        return _gaussian(x, *p) - y

    with fit_progress.iteration_limit(2):
        with pytest.raises(fit_progress.FitNotConverged):
            fit_progress.limited_least_squares(residuals, p0, bounds=([0, -10, 0.01], [10, 10, 10]))
        sol = fit_progress.limited_least_squares(residuals, p0, bounds=([0, -10, 0.01], [10, 10, 10]),
                                                 raise_on_limit=False)
    assert isinstance(sol.not_converged, fit_progress.FitNotConverged)
    sol = fit_progress.limited_least_squares(residuals, [0.8, 2.5, 0.5],
                                             bounds=([0, -10, 0.01], [10, 10, 10]))
    assert sol.not_converged is None and sol.success


def test_progress_counts_iterations_per_fit_against_the_limit():
    statuses = []
    x, y, _ = _hard_problem()
    with fit_progress.iteration_limit(321):
        with fit_progress.watch(lambda st: statuses.append(st) or True, min_interval=0.0):
            model = fit_progress.tracked(_gaussian, y, n_free=3)
            fit_progress.limited_curve_fit(model, x, y, p0=[0.8, 2.5, 0.5],
                                           bounds=([0, -10, 0.01], [10, 10, 10]))
    st = statuses[-1]
    assert st.max_iterations == 321
    assert st.iterations == st.evaluations // 4
    st.iteration_offset = st.evaluations              # a new fit in the same watch
    assert st.iterations == 0


def test_progress_text_shows_the_limit():
    from opennmr.gui.fit_progress_dialog import status_text
    st = fit_progress.FitStatus(evaluations=40, n_free=3, max_iterations=2000)
    assert "About 10 iterations of at most 2,000" in status_text("Fit", st)


# ============================================================
# Maximum iterations: the fitters use it
# ============================================================

def test_spectral_fit_reports_non_convergence_with_suggestions():
    from opennmr.processing import fitting          # what the GUI's Fit ▸ Spectral uses
    from test_fitting import _make_1d
    spec = _make_1d(x0=5.0, fwhm=0.05, A=1.0, kind="gaussian", b0=0.0, b1=0.0)
    peak_defs = [{"type": "gaussian", "shift": 5.2, "fwhm": 0.4}]
    with fit_progress.iteration_limit(1):
        with pytest.raises(fit_progress.FitNotConverged, match="spectral fit did not converge"):
            fitting.fit_spectrum(spec, peak_defs, use_normalised=False)
    popt, *_ = fitting.fit_spectrum(spec, peak_defs, use_normalised=False)   # default limit
    assert popt[0] == pytest.approx(5.0, abs=1e-3)


def test_relaxation_fit_reports_non_convergence():
    from opennmr.processing import relaxation
    t = np.geomspace(1e-3, 10, 20)
    y = 1 - 2 * np.exp(-t / 0.7)
    with fit_progress.iteration_limit(1):
        with pytest.raises(fit_progress.FitNotConverged, match="relaxation fit"):
            relaxation.fit_relaxation(t, y, model="inversion_recovery", p0=[0.2, 9.0, 0.5])
    result = relaxation.fit_relaxation(t, y, model="inversion_recovery")
    assert result["params"][result["rate_param"]] == pytest.approx(0.7, rel=1e-3)


def test_fitters_default_to_the_app_wide_limit():
    import inspect
    from opennmr.notebook import combined_fitting, exchange, multi_fitting, spin_system
    from opennmr.notebook import relaxation as nb_relaxation
    from opennmr.processing import relaxation
    assert inspect.signature(spin_system.fit_spin_system_spectrum).parameters["max_iterations"].default is None
    assert inspect.signature(combined_fitting.fit_combined).parameters["max_nfev"].default is None
    assert inspect.signature(multi_fitting.fit_multi).parameters["max_nfev"].default is None
    assert inspect.signature(exchange.fit_exchange_spectra).parameters["max_nfev"].default is None
    assert inspect.signature(relaxation.fit_relaxation).parameters["maxfev"].default is None
    assert inspect.signature(nb_relaxation.fit_relaxation).parameters["maxfev"].default is None


def test_notebook_exposes_the_limit_and_sideband_helpers():
    import opennmr.notebook as nb
    for name in ("set_max_iterations", "max_iterations", "iteration_limit", "FitNotConverged",
                 "DEFAULT_MAX_ITERATIONS", "sideband_count", "sideband_note", "SIDEBAND_GUIDANCE"):
        assert name in nb.__all__ and hasattr(nb, name)


# ============================================================
# Maximum iterations: Settings ▸ Fitting
# ============================================================

def test_settings_edit_and_save_the_limit(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    assert window.max_iterations_spin.value() == 2000
    window.max_iterations_spin.setValue(5000)
    assert fit_progress.max_iterations() == 5000
    with open(MainWindow._UI_SETTINGS_PATH, encoding="utf-8") as handle:
        assert json.load(handle)["max_iterations"] == 5000
    fit_progress.set_max_iterations(None)               # as a fresh launch would start
    window2 = MainWindow()
    assert window2.max_iterations_spin.value() == 5000
    assert fit_progress.max_iterations() == 5000
    window2.max_iterations_reset_button.click()
    assert fit_progress.max_iterations() == 2000


# ============================================================
# Sidebands
# ============================================================

def test_csa_span_and_sideband_count():
    # Haeberlen: span = |δ|(3 + η)/2; Herzfeld-Berger: Ω itself.
    assert fit_advice.csa_span_ppm(100, 0.0) == pytest.approx(150)
    assert fit_advice.csa_span_ppm(-100, 1.0) == pytest.approx(200)
    assert fit_advice.csa_span_ppm(120, 0.3, "Herzfeld-Berger") == pytest.approx(120)
    # 13C at 100.6 MHz, δ = 100 ppm (15.1 kHz span) at 5 kHz MAS: ~3 sidebands.
    assert fit_advice.sideband_count(100, 100.6, 5000) == pytest.approx(3.018)
    assert fit_advice.sideband_count(100, 100.6, 0) == float("inf")      # static


@pytest.mark.parametrize("n, tone", [(1.2, "warn"), (2.99, "warn"), (3.0, "note"), (4.9, "note"),
                                     (5.0, None), (12, None), (float("inf"), None)])
def test_sideband_note_follows_the_3_to_5_rule(n, tone):
    said = fit_advice.sideband_note(n, "C1", 10000)
    if tone is None:
        assert said is None
    else:
        assert said[0] == tone
        assert "C1" in said[1]
        if tone == "warn":
            assert "at least 3–5" in said[1]


def test_guidance_text_names_every_anisotropic_interaction():
    text = fit_advice.SIDEBAND_GUIDANCE
    assert "3–5" in text
    for word in ("CSA", "quadrupolar", "dipolar"):
        assert word in text


def test_fit_advice_flags_a_pattern_with_too_few_sidebands():
    # A narrow CSA (δ = 20 ppm at 100 MHz: 3 kHz span) spun at 10 kHz.
    x = np.linspace(-200, 200, 4001)
    curve = np.exp(-0.5 * (x / 0.5) ** 2)
    fit = {"x": x, "y": curve, "y_fit": curve, "site_spectra": {"C1": curve},
           "settings": {"mas_Hz": 10000.0, "larmor_Hz": 100e6},
           "sites": [{"label": "C1", "delta_iso": 0.0, "fwhm": 1.0,
                      "interactions": [{"type": "csa", "aniso": 20.0, "asym": 0.0,
                                        "convention": "Haeberlen"}]}]}
    items = fit_advice.advise_combined_fit(fit)
    texts = [i["text"] for i in items if i["tone"] == "warn"]
    assert any("C1's anisotropic pattern spreads over only about 0.3 spinning" in t
               for t in texts)


def test_start_estimate_note_uses_the_sideband_rule():
    note = fit_advice.sideband_note(fit_advice.sideband_count(10, 100, 10000), "The site", 10000)
    assert note[0] == "warn"


def test_fit_sections_show_the_sideband_tip_under_mas_only(qapp):
    from opennmr.gui.combined_panels import CombinedFitPanel
    from opennmr.gui.main_window import MainWindow
    from opennmr.gui.quadrupolar_panels import QuadrupolarFitPanel
    from opennmr.gui.widgets.sideband_hint import SHORT_TEXT
    assert "3–5" in SHORT_TEXT
    window = MainWindow()
    panels = [window.findChild(CombinedFitPanel), window.findChild(QuadrupolarFitPanel)]
    for panel in [p for p in panels if p is not None]:
        hint = panel.sideband_hint
        assert hint.toolTip() == fit_advice.SIDEBAND_GUIDANCE
        panel.regime_combo.setCurrentIndex(panel.regime_combo.findData("static"))
        assert hint.isHidden()
        panel.regime_combo.setCurrentIndex(panel.regime_combo.findData("mas"))
        assert not hint.isHidden()
    assert any(p is not None for p in panels)
    window.csa_static_combo.setCurrentIndex(window.csa_static_combo.findData(True))
    window._refresh_csa_controls()
    assert window.csa_sideband_hint.isHidden()
    window.csa_static_combo.setCurrentIndex(window.csa_static_combo.findData(False))
    window._refresh_csa_controls()
    assert not window.csa_sideband_hint.isHidden()


# ============================================================
# Contour scroll direction
# ============================================================

def _contour(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    return canvas


def test_default_scroll_up_shows_fewer_contours(qapp):
    canvas = _contour(qapp)
    assert canvas.contour_scroll_up_shows_more() is False
    before = canvas._contour_factor
    canvas._on_scroll(_body_scroll_event(canvas.ax, xdata=5.0, ydata=95.0, button="up"))
    assert canvas._contour_factor > before              # higher floor: fewer levels


def test_reversed_scroll_up_shows_more_contours(qapp):
    canvas = _contour(qapp)
    canvas.set_contour_scroll_up_shows_more(True)
    before = canvas._contour_factor
    canvas._on_scroll(_body_scroll_event(canvas.ax, xdata=5.0, ydata=95.0, button="up"))
    assert canvas._contour_factor < before              # lower floor: more levels
    before = canvas._contour_factor
    canvas._on_scroll(_body_scroll_event(canvas.ax, xdata=5.0, ydata=95.0, button="down"))
    assert canvas._contour_factor > before


@pytest.mark.parametrize("up_more", [False, True])
def test_dosy_follows_the_same_direction(qapp, up_more):
    canvas = MplCanvas()
    canvas.set_contour_scroll_up_shows_more(up_more)
    up = canvas._contour_scroll_step(type("E", (), {"button": "up"})())
    assert (up < 1) is up_more


def test_settings_and_contour_menu_share_one_saved_switch(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    box = window.contour_scroll_up_more_checkbox
    assert box.isChecked() is False
    box.setChecked(True)
    assert all(c.contour_scroll_up_shows_more() for c in window._all_canvases())
    with open(MainWindow._UI_SETTINGS_PATH, encoding="utf-8") as handle:
        assert json.load(handle)["contour_scroll_up_shows_more"] is True
    # The contour's right-click item, on one canvas, drives them all and the box.
    window.canvas._on_contour_scroll_action_toggled(False)
    assert box.isChecked() is False
    assert not any(c.contour_scroll_up_shows_more() for c in window._all_canvases())
    window.canvas._on_contour_scroll_action_toggled(True)
    window2 = MainWindow()
    assert window2.contour_scroll_up_more_checkbox.isChecked()
    assert window2.canvas.contour_scroll_up_shows_more()


# ============================================================
# Crosshair shortcut
# ============================================================

def test_ctrl_full_stop_toggles_the_crosshair(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    action = window.crosshair_menu_action
    assert action.shortcut() == QKeySequence("Ctrl+.")
    others = [a for a in window.findChildren(type(action))
              if a is not action and a.shortcut() == QKeySequence("Ctrl+.")]
    assert not others                                    # no clash
    assert action.isChecked()
    action.trigger()
    assert not action.isChecked()
    assert not any(c._crosshair_enabled for c in window._measurement_canvases()
                   if hasattr(c, "_crosshair_enabled"))
    action.trigger()
    assert action.isChecked()
