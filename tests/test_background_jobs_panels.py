"""The remaining fits and simulations as background jobs (explicit user
request, 2026-11-13: "can we build the remaining panels as background
jobs?") -- spectral, CSA, DOSY/SCORE, J-coupling, spin-system, pNMR,
Czjzek, multi-spectrum, SIMPSON, regularisation and BMS -- and what they
need: stage reports for steps that evaluate no model (a lineshape
library), printed output and warnings kept per thread, and a job's end
always reported back to its panel."""

import io
import sys
import threading
import warnings

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from opennmr.core import fit_progress, thread_io
from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui import jobs
from opennmr.gui.main_window import MainWindow

from test_gui import qapp, _load_spectrum_directly  # noqa: F401 -- fixture
from test_background_jobs import _idle, _job, _wait_for


@pytest.fixture
def background(monkeypatch):
    monkeypatch.setenv(jobs.BACKGROUND_JOBS_ENV, "1")
    said = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: said.append((_n, a[2] if len(a) > 2 else "")))
    return said


def _finish(window):
    """Wait for the window's jobs to end; then stop its thread."""
    manager = window.job_manager
    try:
        assert _wait_for(lambda: _idle(manager), seconds=240)
        QApplication.processEvents()
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# Stage reports (no GUI)
# ---------------------------------------------------------------------------

def test_a_stage_is_reported_and_can_stop_the_run():
    seen = []
    with fit_progress.watch(lambda st: seen.append(st.stage) or True, min_interval=0.0):
        fit_progress.report_stage("lineshape 1 of 3")
    assert seen == ["lineshape 1 of 3"]
    with pytest.raises(fit_progress.FitCancelled):
        with fit_progress.watch(lambda st: False, min_interval=0.0):
            fit_progress.report_stage("lineshape 1 of 3")
    fit_progress.report_stage("nothing watches: a no-op")


def test_the_library_callback_says_how_far_and_returns_false_to_stop():
    stages = []
    answers = iter([True, True, False])
    with fit_progress.watch(lambda st: stages.append(st.stage) or next(answers),
                            min_interval=0.0):
        progress = fit_progress.stage_callback("lineshape")
        assert progress(1, 10) is True and progress(2, 10) is True
        assert progress(3, 10) is False                     # Stop asked for
    assert stages[0].startswith("lineshape 1 of 10") and "left" in stages[0]


def test_a_stage_without_evaluations_reads_as_time_only(qapp):  # noqa: F811
    from opennmr.gui.fit_progress_dialog import status_text
    st = fit_progress.FitStatus(elapsed=3.0, stage="lineshape 4 of 40")
    text = status_text("Czjzek fit", st)
    assert text.splitlines()[0] == "Czjzek fit — lineshape 4 of 40"
    assert "model evaluations" not in text


# ---------------------------------------------------------------------------
# Printed output and warnings, per thread
# ---------------------------------------------------------------------------

def test_quietened_output_is_this_threads_only():
    before = sys.stdout
    outer = io.StringIO()
    sys.stdout = outer
    try:
        started, release = threading.Event(), threading.Event()
        captured = {}

        def worker():
            with thread_io.quiet_stdout() as buffer:
                print("from the fit")
                started.set()
                release.wait(10)
            captured["worker"] = buffer.getvalue()

        thread = threading.Thread(target=worker)
        thread.start()
        assert started.wait(10)
        print("from the GUI")                        # not swallowed by the fit
        with thread_io.quiet_stdout() as mine:       # nested from another thread
            print("a report")
        release.set()
        thread.join(10)
        assert captured["worker"] == "from the fit\n"
        assert mine.getvalue() == "a report\n"
        assert outer.getvalue() == "from the GUI\n"
        assert sys.stdout is outer                   # put back, whatever the order
    finally:
        sys.stdout = before


def test_collected_warnings_are_this_threads_only():
    shown = []
    original = warnings.showwarning
    warnings.showwarning = lambda message, *a, **k: shown.append(str(message))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            started, release = threading.Event(), threading.Event()
            caught = {}

            def worker():
                with thread_io.collect_warnings() as got:
                    warnings.warn("from the fit")
                    started.set()
                    release.wait(10)
                caught["worker"] = [str(w.message) for w in got]

            thread = threading.Thread(target=worker)
            thread.start()
            assert started.wait(10)
            warnings.warn("from the GUI")
            release.set()
            thread.join(10)
        assert caught["worker"] == ["from the fit"]
        assert shown == ["from the GUI"]
    finally:
        warnings.showwarning = original


# ---------------------------------------------------------------------------
# The job's end, and Keep Best only where it means something
# ---------------------------------------------------------------------------

def test_a_jobs_end_is_reported_however_it_ends(qapp, background):  # noqa: F811
    window = MainWindow()
    manager = window.job_manager
    ended = []
    try:
        gate = threading.Event()
        manager.submit(_job(window, "Runs", gate=gate, on_end=lambda: ended.append("runs")))
        manager.submit(_job(window, "Removed", on_end=lambda: ended.append("removed")))

        def boom():
            raise ValueError("no")
        manager.submit(jobs.Job("Fails", boom, lambda *a: None, fail=lambda exc: None,
                                on_end=lambda: ended.append("fails")))
        manager.remove_queued(manager.queue[0].id)
        assert ended == ["removed"]
        gate.set()
        assert _wait_for(lambda: _idle(manager))
        assert ended == ["removed", "runs", "fails"]
    finally:
        manager.shutdown()
    # Run at once (the background off), too.
    jobs.run_inline(jobs.Job("Inline", lambda: 1, lambda *a: None, watched=False,
                             on_end=lambda: ended.append("inline")))
    assert ended[-1] == "inline"


def test_keep_best_is_not_offered_for_a_simulation(qapp, background):  # noqa: F811
    window = MainWindow()
    manager = window.job_manager
    try:
        gate = threading.Event()
        manager.submit(_job(window, "Simulation", gate=gate, keepable=False))
        manager.show_dialog()
        assert not manager.dialog.keep_button.isEnabled()
        assert manager.dialog.discard_button.isEnabled()
        manager.stop_current(keep=True)                     # treated as Discard
        assert manager.current.answer == "discard"
        gate.set()
        assert _wait_for(lambda: _idle(manager))
        manager.dialog.close()
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# The panels
# ---------------------------------------------------------------------------

def _gaussian_window():
    from opennmr.processing.fitting import gaussian
    ppm = np.linspace(0, 10, 900)
    spectrum = NMRSpectrum(gaussian(ppm, 5.0, 0.05, 1.0),
                           [NMRAxis("frequency", ppm=ppm, hz=ppm * 400)],
                           NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.03, fwhm=0.08, eta=0.0)
    return window, spectrum


def test_a_spectral_fit_runs_in_the_background(qapp, background):  # noqa: F811
    window, spectrum = _gaussian_window()
    window._on_fit_spectrum_clicked()
    assert window.job_manager.current is not None          # running, not yet shown
    assert "slice_fit_results" not in spectrum.metadata.raw
    _finish(window)
    from opennmr.processing import fitting as processing_fitting
    result = spectrum.metadata.raw["slice_fit_results"][0]
    peak = processing_fitting.extract_params_and_errors(
        result["popt"], result["pcov"], result["peak_defs"], result["is_free"])[0]
    assert dict(zip(peak["names"], peak["params"]))["shift"] == pytest.approx(5.0, abs=1e-3)
    assert not background, background


def test_peak_table_edits_after_starting_do_not_change_the_fit(qapp, background):  # noqa: F811
    # The inputs are read when the fit starts.
    window, spectrum = _gaussian_window()
    gate = threading.Event()
    window.job_manager.submit(_job(window, "Before", gate=gate))   # the fit waits its turn
    window._on_fit_spectrum_clicked()
    window.peak_table.item(0, 0).setText("8.0")                    # edited while it waits
    gate.set()
    _finish(window)
    results = spectrum.metadata.raw["slice_fit_results"]
    assert results[0]["peak_defs"][0]["shift"] == pytest.approx(5.03)


def test_a_csa_fit_runs_in_the_background(qapp, background):  # noqa: F811
    from test_gui_csa import _add, _mas_spectrum
    window = MainWindow()
    _add(window, _mas_spectrum())
    window.csa_static_combo.setCurrentIndex(window.csa_static_combo.findData(False))
    window.csa_orientations_spin.setValue(200)
    window.csa_render_orientations_spin.setValue(400)
    window._add_csa_site(("A", "2", "-80", "0.4", "3"))
    window._on_csa_fit_clicked()
    assert window.job_manager.current is not None          # running, not yet shown
    window.csa_render_orientations_spin.setValue(2000)     # changed while it runs
    _finish(window)
    result = window._csa_fit_result
    assert result["popt"][1] == pytest.approx(-100.0, abs=1.5)
    assert result["spectrum_index"] == 0
    # Drawn at the orientations it was started with.
    assert result["render_n"] == window._powder_count(result["scheme"], 400)
    assert "CSA fit report" in window.csa_results_label.text()


def test_a_jcoupling_fit_runs_in_the_background(qapp, background):  # noqa: F811
    from test_gui_jcoupling import TRUTH, _editor, _set_up_two_sites, _synthetic
    window, fit, ed = _editor(qapp, "fit")
    _synthetic(window, TRUTH)
    _set_up_two_sites(ed)
    assert fit.fit()
    _finish(window)
    got = fit.fitted_sites()
    assert got[0]["couplings"][0]["J_Hz"] == pytest.approx(7.1, abs=1e-3)
    assert "J-COUPLING FIT REPORT" in fit.text_report()


def test_a_czjzek_simulation_runs_in_the_background(qapp, background):  # noqa: F811
    window = MainWindow()
    sim = window.czjzek_sim_panel
    sim.grid_cq_spin.setValue(12)
    sim.grid_eta_spin.setValue(4)
    assert sim.simulate() is None                         # started, not yet shown
    _finish(window)
    assert sim.result is not None and "⟨C_Q⟩" in sim.results_label.text()
    assert window.simulation_canvas.figure.czjzek_axes["spectrum"] is not None


def test_a_multi_spectrum_fit_keeps_its_panel_busy_until_it_ends(qapp, background,  # noqa: F811
                                                              monkeypatch):
    from opennmr.notebook import multi_fitting as mf
    window = MainWindow()
    panel = window.multi_fit_panel
    gate = threading.Event()
    stages = []

    def fake_fit(experiments, sites, progress=None, **kwargs):
        assert progress(0, 2, "Experiment 1 of 2") is True
        assert gate.wait(30)
        return {"experiments": []}

    monkeypatch.setattr(panel, "fit_arguments", lambda: {
        "experiments": [], "sites": {}, "scopes": None, "vary": None, "bounds": None,
        "amounts": None, "seed": None, "order": None, "n_zcw": None, "fit_baseline": False,
        "fit_mas": False, "mas_window_Hz": 0.0})
    monkeypatch.setattr(mf, "fit_multi", fake_fit)
    monkeypatch.setattr(panel, "_report", lambda: stages.append("reported"))
    monkeypatch.setattr(panel, "show_selected", lambda: None)
    assert panel.fit()
    assert panel._busy
    assert _wait_for(lambda: window.job_manager.current is not None
                     and window.job_manager.current.status is not None)
    assert window.job_manager.current.status.stage == "Experiment 1 of 2"
    gate.set()
    _finish(window)
    assert stages == ["reported"] and not panel._busy


def test_regularisation_runs_in_the_background(qapp, background):  # noqa: F811
    from test_integrate_tile_plot_options_regularisation import _fitted_series
    window = _fitted_series(MainWindow())
    window._regularisation_section.setExpanded(True)       # fills its series list
    panel = window.regularisation_panel
    assert panel.series_combo.count() == 2
    panel.solver_combo.setCurrentIndex(panel.solver_combo.findData("nonnegative"))
    assert panel.run() is None                             # started, not yet shown
    _finish(window)
    assert panel.result is not None
    assert panel.result["peaks"][0]["position"] == pytest.approx(0.05, rel=0.2)
    assert window.fit_canvas.plot_owner == "regularisation"


def _bms(window, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENNMR_BMS_CACHE", str(tmp_path / "sidebands"))
    t = window.paramagnetic_panel
    t.pm_bms_n_spin.setValue(15)
    t.pm_bms_box_spin.setValue(200.0)
    t.pm_bms_size_edit.setText("20")
    t.pm_bms_spread_edit.setText("2, 2, 2")
    t.pm_bms_grid_combo.setCurrentIndex(t.pm_bms_grid_combo.findData(64))
    t.pm_bms_chi_edit.setText("2.87e-3, 3.13e-3, 2.27e-3")
    return t


def test_a_bms_simulation_runs_in_the_background(qapp, background, tmp_path,  # noqa: F811
                                                 monkeypatch):
    window = MainWindow()
    t = _bms(window, tmp_path, monkeypatch)
    assert t.simulate_bms()
    t.pm_bms_chi_edit.setText("1e-3, 1e-3, 1e-3")          # edited while it runs
    _finish(window)
    assert t.bms_nmr_spectrum is not None and not background, background
    sim = t.bms_nmr_spectrum.metadata.raw["simulation"]
    assert sim["n_particles"] == 15
    # Described by the χ it was run with (the edit was put back, and said so).
    assert sim["chi_input"] == pytest.approx([2.87e-3, 3.13e-3, 2.27e-3])
    assert t.pm_bms_chi_edit.text() == "2.87e-3, 3.13e-3, 2.27e-3"


def test_a_bms_slice_asked_for_before_a_run_follows_it(qapp, background, tmp_path,  # noqa: F811
                                                       monkeypatch):
    window = MainWindow()
    t = _bms(window, tmp_path, monkeypatch)
    drawn = []
    monkeypatch.setattr(t, "_show", lambda draw, kind: drawn.append(kind) or True)
    assert t.bms_field is None
    assert t.plot_bms_slice()
    _finish(window)
    assert t.bms_field is not None and drawn and drawn[-1] == "2d"
