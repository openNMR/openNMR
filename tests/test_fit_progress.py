"""2026-10-01: live progress for every spectrum fit (core/fit_progress.py,
gui/fit_progress_dialog.py) -- user request: the Czjzek library's
update-as-you-go pop-up "when fitting spectra, too ... not just for the
Czjzek, but also for quadrupole, CSA, J etc"."""
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from opennmr.core import fit_progress as fp
from opennmr.core.fit_progress import FitCancelled

from test_gui import qapp  # noqa: F401 -- fixture


# ---------------------------------------------------------------------------
# The fitters, each on a small synthetic spectrum (helpers shared with
# each fitter's own tests)
# ---------------------------------------------------------------------------

def _csa():
    from test_csa_fitting import _fit, _mas_results
    return _fit(_mas_results())


def _quadrupolar():
    from opennmr.notebook import quadrupolar_fitting as QF
    from test_quadrupolar_fitting import _results
    r = _results([(12.0, 4.2, 0.35, 0.8, 1000.0)])
    return QF.fit_quadrupolar_spectrum(r, QF.estimate_quadrupolar_start(r, 2.5))


def _jcoupling():
    from opennmr.notebook import jcoupling_fitting as JF
    from test_jcoupling import L_1H, _lor, _results
    x = np.linspace(12, -2, 32768)
    fwhm = 0.8 / L_1H * 1e6
    y = 1000 * sum(w * _lor(x, 3.2 + k * 7.1 / L_1H * 1e6, fwhm)
                   for k, w in ((-1, .25), (0, .5), (1, .25))) / 0.5
    p0 = {"label": "CH2", "center_ppm": 3.2, "fwhm": 0.003,
          "couplings": [{"nuclide": "1H", "n": 2, "J_Hz": 7.0, "abundance": 100.0}]}
    return JF.fit_jcoupling_spectrum(_results(x, y), p0)


def _spin_system():
    from opennmr.notebook import spin_system as SS
    from test_spin_system import _data
    spins = [{"label": "A", "nuclide": "1H", "shift_ppm": 7.30},
             {"label": "B", "nuclide": "1H", "shift_ppm": 7.28}]
    couplings = [{"a": "A", "b": "B", "J_Hz": 8.0}]
    start = [dict(s) for s in spins]
    start[0]["shift_ppm"] += 0.001
    return SS.fit_spin_system_spectrum(_data(spins, couplings), start, couplings,
                                       fit_regions=[(7.2, 7.4)])


def _czjzek():
    from opennmr.notebook import czjzek as cz
    from test_czjzek import _two_site
    return cz.fit_czjzek_spectrum(_two_site(), [{"d_iso": 50, "sigma": 2, "fwhm": 3},
                                                {"d_iso": 15, "sigma": 1, "fwhm": 3}],
                                  n_cq=12, n_eta=4)


def _combined():
    from opennmr.notebook import combined_fitting as CF
    from opennmr.notebook.combined_hamiltonian import Site
    from test_combined_fitting import _data
    res = _data(0.0, -150, 150, 4096)
    start = [Site("27Al", 12, label="Al1", fwhm=1).add_quadrupolar(4.2, 0.35),
             Site("27Al", -8, label="Al2", fwhm=1).add_quadrupolar(6.0, 0.7)]
    return CF.fit_combined_multistart(res, start, starts={"Al1": {"C_Q": [4.0, 4.4]}},
                                      verbose=False)


def _spectral_slices():
    from opennmr.processing import fitting, slicing
    from test_fitting import _make_2d_for_fitting
    spec = _make_2d_for_fitting(amps=(0.5, 1.0, 1.5))
    slices = slicing.get_slices(spec, rows=[1, 2, 3])
    return fitting.NMR2D_fit_slices(slices, [{"type": "gaussian", "shift": 5.02, "fwhm": 0.05}],
                                    mode="rows", show_report=False, show_warnings=False)


FITTERS = {"csa": _csa, "quadrupolar": _quadrupolar, "jcoupling": _jcoupling,
           "spin_system": _spin_system, "czjzek": _czjzek, "combined": _combined,
           "spectral_slices": _spectral_slices}


@pytest.mark.parametrize("name", list(FITTERS))
def test_every_fitter_reports_its_progress(name):
    seen = []
    with fp.watch(lambda st: seen.append((st.evaluations, st.best_cost, st.start_cost,
                                          st.stage)) or True,
                  label=name, min_interval=0.0):
        FITTERS[name]()
    assert seen, f"{name} never reported"
    evaluations = [s[0] for s in seen]
    assert evaluations == sorted(evaluations) and evaluations[-1] > 1
    best, start = seen[-1][1], seen[-1][2]
    assert np.isfinite(best) and best <= start          # the misfit came down (or held)
    if name == "combined":
        assert any(s[3].startswith("start 1 of 2") for s in seen)
        assert any(s[3].startswith("start 2 of 2") for s in seen)
    if name == "spectral_slices":
        assert {s[3] for s in seen} >= {"row 1 of 3", "row 2 of 3", "row 3 of 3"}
    if name == "czjzek":
        assert any("first grid" in s[3] for s in seen)


@pytest.mark.parametrize("name", list(FITTERS))
def test_every_fitter_stops_when_asked(name):
    calls = []

    def stop_on_third(status):
        calls.append(status.evaluations)
        return len(calls) < 3

    # FitCancelled -- not "no starting point converged", not a failed
    # fit: the multi-start and slice loops must not swallow it.
    with pytest.raises(FitCancelled):
        with fp.watch(stop_on_third, min_interval=0.0):
            FITTERS[name]()
    assert len(calls) == 3


def test_a_watched_fit_gives_the_same_answer():
    plain = _csa()
    with fp.watch(lambda st: True, min_interval=0.0):
        watched = _csa()
    assert np.array_equal(plain[0], watched[0])
    assert np.array_equal(plain[1], watched[1])


def test_unwatched_fits_are_untouched():
    def model(x, a):
        return a * x
    assert fp.tracked(model, [1.0]) is model
    assert fp.tracked_residuals(model) is model
    assert not fp.active()
    fp.set_stage("nothing watches")                    # a no-op, not an error
    fp.check_cancelled()


def test_status_numbers_and_notebook_printer(capsys):
    st = fp.FitStatus(label="CSA fit", evaluations=1234, elapsed=2.5, start_cost=100.0,
                      best_cost=4.0, cost=4.2, recent_fall=0.031, stage="start 2 of 3")
    assert st.relative == pytest.approx(0.04)
    text = st.summary()
    for bit in ("CSA fit", "start 2 of 3", "1,234 evaluations", "misfit 4", "4 % of the start",
                "falling 3.1 %/s"):
        assert bit in text
    fp.print_progress._last = None
    assert fp.print_progress(st) is True
    assert "1,234 evaluations" in capsys.readouterr().out
    with fp.watch(fp.print_progress, label="Quadrupolar fit"):
        _quadrupolar()
    assert "Quadrupolar fit" in capsys.readouterr().out


def test_nested_watches_report_to_the_outermost():
    outer, inner = [], []
    with fp.watch(lambda st: outer.append(1) or True, min_interval=0.0):
        with fp.watch(lambda st: inner.append(1) or True, min_interval=0.0):
            _csa()
    assert outer and not inner


# ---------------------------------------------------------------------------
# The GUI pop-up
# ---------------------------------------------------------------------------

@pytest.fixture
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def test_popup_text(qapp):  # noqa: F811
    from opennmr.gui.fit_progress_dialog import status_text
    st = fp.FitStatus(evaluations=2048, elapsed=75.0, start_cost=10.0, best_cost=0.5,
                      recent_fall=0.0, stage="slice 2 of 5")
    text = status_text("Spectral fit", st)
    assert text.splitlines()[0] == "Spectral fit — slice 2 of 5"
    assert "2,048 model evaluations, 1 min 15 s" in text
    assert "5 % of the starting value" in text
    assert "No longer improving" in text


def _press(monkeypatch, answer):
    """The pop-up appears at once and the given button is pressed in it
    (``None``: none is); returns the texts it showed."""
    from opennmr.gui import fit_progress_dialog as fpd
    monkeypatch.setattr(fpd.FitProgress, "DELAY", 0.0)
    labels = []
    original = fpd.status_text
    monkeypatch.setattr(fpd, "status_text",
                        lambda title, st: (labels.append(original(title, st)), labels[-1])[1])

    def show(self):
        self.answer = answer
    monkeypatch.setattr(fpd.FitProgressDialog, "show", show)
    return labels


def test_quadrupolar_fit_shows_the_popup_and_stops_cleanly(qapp, _dialogs, monkeypatch):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    from test_gui_quadrupolar_fit import _load
    window = MainWindow()
    panel = window.quad_fit_panel
    _load(window)
    panel.estimate()
    assert panel.result is None

    # Discard pressed: nothing changes, nothing is reported as a failure.
    labels = _press(monkeypatch, "discard")
    panel.fit()
    assert panel.result is None
    assert not _dialogs, _dialogs
    assert QApplication.overrideCursor() is None
    assert "Quadrupolar fit stopped. Nothing was changed" in window.statusBar().currentMessage()
    assert labels and "model evaluations" in labels[0]

    # Run to the end: the pop-up updated as it went and the fit is kept.
    labels = _press(monkeypatch, None)
    panel.fit()
    assert panel.result is not None and not _dialogs, _dialogs
    assert any("Misfit" in text for text in labels)
    assert QApplication.overrideCursor() is None
