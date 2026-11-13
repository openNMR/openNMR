"""The fitting algorithm (2026-11-13): one app-wide choice (Settings ▸
Fitting), read by every fitter through core.fit_progress."""

import numpy as np
import pytest

from opennmr.core import fit_progress as fp


def _decay(x, a, t):
    return a * np.exp(-x / t)


X = np.linspace(0.0, 5.0, 60)
Y = _decay(X, 3.0, 1.2) + np.random.default_rng(1).normal(0, 0.01, X.size)


@pytest.fixture(autouse=True)
def _restore_method():
    yield
    fp.set_method(None)


def test_default_is_auto_and_unknown_names_are_refused():
    assert fp.method() == fp.DEFAULT_METHOD == "auto"
    assert set(fp.METHODS) == {"auto", "trf", "dogbox", "lm", "differential_evolution",
                               "dual_annealing"}
    with pytest.raises(ValueError):
        fp.set_method("simplex")
    with fp.fitting_method("dogbox"):
        assert fp.method() == "dogbox"
    assert fp.method() == "auto"


@pytest.mark.parametrize("name", ["auto", "trf", "dogbox", "lm"])
def test_every_local_method_fits_an_unbounded_curve(name):
    fp.set_method(name)
    popt, pcov = fp.limited_curve_fit(_decay, X, Y, p0=[1.0, 1.0])
    assert popt == pytest.approx([3.0, 1.2], rel=0.02)
    assert np.all(np.isfinite(np.diag(pcov)))
    assert fp.last_method()["method"] == name
    assert fp.last_method()["note"] == ""


@pytest.mark.parametrize("name", ["trf", "dogbox"])
def test_bounded_methods_respect_the_bounds(name):
    fp.set_method(name)
    popt, _ = fp.limited_curve_fit(_decay, X, Y, p0=[1.0, 0.5], bounds=([0, 0.1], [2.5, 5.0]))
    assert popt[0] <= 2.5 + 1e-9                          # pinned at its upper bound
    assert fp.last_method()["method"] == name


def test_levenberg_marquardt_with_bounds_uses_trf_and_says_so():
    fp.set_method("lm")
    popt, _ = fp.limited_curve_fit(_decay, X, Y, p0=[1.0, 0.5], bounds=([0, 0.1], [10, 5.0]))
    assert popt == pytest.approx([3.0, 1.2], rel=0.02)
    used = fp.last_method()
    assert used["method"] == "trf" and "cannot apply bounds" in used["note"]


@pytest.mark.parametrize("name", ["differential_evolution", "dual_annealing"])
def test_global_methods_find_the_minimum_from_a_poor_start(name):
    fp.set_method(name)
    with fp.iteration_limit(300):
        popt, pcov = fp.limited_curve_fit(_decay, X, Y, p0=[0.1, 4.9],
                                          bounds=([0.0, 0.05], [10.0, 5.0]))
    assert popt == pytest.approx([3.0, 1.2], rel=0.02)
    assert np.all(np.isfinite(np.diag(pcov)))              # the TRF finish gives them
    assert fp.last_method()["method"] == name


def test_a_global_method_without_finite_bounds_falls_back_and_says_so():
    fp.set_method("differential_evolution")
    popt, _ = fp.limited_curve_fit(_decay, X, Y, p0=[1.0, 1.0])
    assert popt == pytest.approx([3.0, 1.2], rel=0.02)
    used = fp.last_method()
    assert used["method"] == "auto" and "finite lower and upper bound" in used["note"]


def test_a_global_method_is_repeatable():
    fp.set_method("differential_evolution")
    first, _ = fp.limited_curve_fit(_decay, X, Y, p0=[0.1, 4.9], bounds=([0, 0.05], [10, 5]))
    second, _ = fp.limited_curve_fit(_decay, X, Y, p0=[0.1, 4.9], bounds=([0, 0.05], [10, 5]))
    np.testing.assert_array_equal(first, second)


def test_least_squares_fitters_follow_the_choice_too():
    def residuals(p):
        return _decay(X, *p) - Y

    for name in ("dogbox", "lm", "dual_annealing"):
        fp.set_method(name)
        bounds = ([0.0, 0.05], [10.0, 5.0]) if name != "lm" else (-np.inf, np.inf)
        with fp.iteration_limit(200):
            sol = fp.limited_least_squares(residuals, [1.0, 1.0], bounds=bounds)
            assert sol.x == pytest.approx([3.0, 1.2], rel=0.02)
            assert fp.last_method()["method"] == name
            sol = fp.least_squares(residuals, [1.0, 1.0], bounds=bounds)
            assert sol.x == pytest.approx([3.0, 1.2], rel=0.02)


def test_lm_with_a_robust_loss_uses_trf():
    fp.set_method("lm")
    sol = fp.least_squares(lambda p: _decay(X, *p) - Y, [1.0, 1.0], loss="soft_l1")
    assert sol.x == pytest.approx([3.0, 1.2], rel=0.05)
    assert "robust loss" in fp.last_method()["note"]


def test_a_fitter_that_names_its_own_method_keeps_it():
    fp.set_method("dogbox")
    fp.limited_curve_fit(_decay, X, Y, p0=[1.0, 1.0], method="trf")
    assert fp.last_method()["method"] == "trf"


def test_the_watcher_reports_the_method():
    fp.set_method("dogbox")
    seen = []
    with fp.watch(lambda st: seen.append(st.method) or True, min_interval=0.0):
        # Fitters hand their model to the watcher themselves (fp.tracked);
        # limited_curve_fit only sets the status.
        model = fp.tracked(_decay, Y, n_free=2)
        fp.limited_curve_fit(model, X, Y, p0=[1.0, 1.0])
    assert seen and set(seen) == {"dogbox"}


def test_the_notebook_names():
    from opennmr import notebook as nmr
    assert nmr.FIT_METHODS is fp.METHODS
    nmr.set_fit_method("trf")
    assert nmr.fit_method() == "trf"
    with nmr.fitting_method("lm"):
        assert fp.method() == "lm"


def test_vt_relaxation_goes_through_the_choice():
    from opennmr.notebook import vt_relaxation
    assert vt_relaxation._fit_progress is fp


# ------------------------------------------------------------------
# Settings ▸ Fitting
# ------------------------------------------------------------------

def test_settings_offers_the_methods_and_remembers_the_choice(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    combo = window.fit_method_combo
    assert [combo.itemData(i) for i in range(combo.count())] == list(fp.METHODS)
    assert combo.currentData() == "auto"
    combo.setCurrentIndex(combo.findData("dogbox"))
    assert fp.method() == "dogbox"
    assert MainWindow().fit_method_combo.currentData() == "dogbox"      # saved
    combo.setCurrentIndex(combo.findData("auto"))


def test_the_progress_pop_up_names_a_chosen_algorithm():
    from opennmr.gui.fit_progress_dialog import status_text
    status = fp.FitStatus(label="Fit", method="dogbox")
    assert "Dogbox" in status_text("Fit", status)
    assert "Algorithm" not in status_text("Fit", fp.FitStatus(label="Fit", method="auto"))


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])
