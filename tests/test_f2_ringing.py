"""Truncation "wiggles" in F2: the diagnosis (processing.ringing), its notebook wrapper
(notebook.ringing.check_f2_ringing) and the Process tab's button (2026-10-29). Synthetic FIDs."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.processing import ringing

N = 1024
SW = 10000.0
DT = 1.0 / SW


def make_fid(t2star, *, n=N, sigma=0.002, seed=1):
    """Three lines of decay time ``t2star`` (s) plus white noise: long -> cut off at the end."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    fid = sum(a * np.exp(2j * np.pi * f * t - t / t2star) for a, f in ((1.0, 1000.0), (0.6, 1100.0), (0.8, -2500.0)))
    return fid + sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


TRUNCATED = 1.0        # s, against an acquisition of 0.1 s: still ~90 % at the last point
DECAYED = 0.008        # s: e^-12.8 by the last point


def test_a_fid_cut_off_while_it_still_has_signal_is_found_and_sized():
    d = ringing.diagnose_truncation(make_fid(TRUNCATED), SW, sf_mhz=400.0)
    assert d["truncated"] and d["tail_snr"] > 10
    assert d["ripple"] > 0.03 and d["severity"] in ("moderate", "strong")
    assert d["period_hz"] == pytest.approx(SW / N)
    assert d["period_ppm"] == pytest.approx(SW / N / 400.0)
    assert d["suggestion"] is not None and "wiggles" in d["report"]


def test_a_fid_that_has_decayed_is_left_alone():
    d = ringing.diagnose_truncation(make_fid(DECAYED), SW)
    assert not d["truncated"] and d["severity"] == "none" and d["suggestion"] is None
    assert d["ripple"] < 0.01


def test_a_sine_bell_removes_the_wiggles():
    fid = make_fid(TRUNCATED)
    plain = ringing.diagnose_truncation(fid, SW)
    weights = ringing.window_weights("qsine", N, DT, ssb=2)
    windowed = ringing.diagnose_truncation(fid, SW, window_weights=weights)
    assert windowed["ripple_windowed"] < plain["ripple"] / 3
    assert windowed["ripple"] == pytest.approx(plain["ripple"])     # the unwindowed figure is unchanged
    order = ringing.SEVERITY_NAMES
    assert order.index(windowed["severity"]) < order.index(plain["severity"])


def test_the_suggestion_is_a_bell_that_beats_no_window_at_the_least_cost():
    fid = make_fid(TRUNCATED)
    s = ringing.suggest_window(fid, SW)
    assert s["window"] in ("sine", "qsine") and s["ssb"] in (2.0, 3.0, 4.0)
    assert s["ripple"] < ringing.ripple_fraction(fid) / 3
    assert s["noise_bandwidth"] > 1.0 and s["label"].startswith(s["window"].upper())
    # nothing narrower in the list clears the target when this one does
    if s["reaches_target"]:
        for key, ssb in ringing.CANDIDATE_WINDOWS:
            w = ringing.window_weights(key, N, DT, ssb=ssb)
            if ringing.noise_bandwidth(w) < s["noise_bandwidth"] - 1e-12:
                assert ringing.ripple_fraction(fid, w) >= ringing.TARGET_RIPPLE
    # a window already good enough needs no suggestion
    assert ringing.suggest_window(fid, SW, current_weights=ringing.window_weights("qsine", N, DT, ssb=2),
                                  target=0.5) is None


def test_noise_bandwidth_orders_the_windows_sensibly():
    assert ringing.noise_bandwidth(np.ones(100)) == pytest.approx(1.0)
    assert (ringing.noise_bandwidth(ringing.window_weights("qsine", N, DT, ssb=2))
            > ringing.noise_bandwidth(ringing.window_weights("sine", N, DT, ssb=2)) > 1.0)


def test_truncate_uses_fewer_points_and_inputs_are_checked():
    d = ringing.diagnose_truncation(make_fid(TRUNCATED), SW, truncate=512)
    assert d["n_points"] == 512 and d["period_hz"] == pytest.approx(SW / 512)
    with pytest.raises(ValueError):
        ringing.diagnose_truncation(np.ones(10, complex), SW)
    with pytest.raises(ValueError):
        ringing.diagnose_truncation(make_fid(TRUNCATED), SW, window_weights=np.ones(7))
    with pytest.raises(ValueError):
        ringing.diagnose_truncation(np.zeros(N, complex), SW)


def test_a_late_amplitude_peak_is_reported_as_a_different_problem():
    fid = make_fid(DECAYED)
    fid = np.concatenate([np.zeros(60, complex), fid[:-60]])       # an echo-like late start
    d = ringing.diagnose_truncation(fid, SW)
    assert d["onset_late"] and "echo" in d["report"]


def test_every_row_of_a_2d_is_considered_and_the_strongest_named():
    rows = np.array([make_fid(TRUNCATED, seed=k) * a for k, a in enumerate((0.1, 1.0, 0.3))])
    d = ringing.diagnose_truncation(rows, SW)
    assert d["n_rows"] == 3 and d["strongest_row"] == 1 and d["truncated"]


# --- the notebook wrapper ------------------------------------------------------------

def spectrum_with(raw_extra, t2star=TRUNCATED, rows=6):
    fid = np.array([make_fid(t2star, seed=k) for k in range(rows)])
    raw = {"fid_raw": fid, "fid_dt": DT, "procs": {"WDW": 0}, **raw_extra}
    ppm = np.linspace(-12, 12, 64)
    return NMRSpectrum(data=np.ones((rows, 64)),
                       axes=[NMRAxis(kind="index", index=np.arange(rows)),
                             NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
                       metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, raw=raw))


def test_wrapper_reads_the_recorded_window_and_flags_wiggles_without_one():
    from opennmr.notebook import ringing as nb
    d = nb.check_f2_ringing(spectrum_with({}))
    assert d["severity"] in ("moderate", "strong") and "none (TopSpin recorded no window)" in d["window_in_use"]
    assert d["report"].startswith("Window in use: none")


def test_wrapper_sees_that_a_recorded_qsine_already_removes_them():
    from opennmr.notebook import ringing as nb
    plain = nb.check_f2_ringing(spectrum_with({}))
    d = nb.check_f2_ringing(spectrum_with({"procs": {"WDW": 4, "SSB": 2}}))
    assert "QSINE, SSB 2" in d["window_in_use"]
    assert d["ripple_windowed"] < plain["ripple_windowed"] / 3


def test_wrapper_names_a_recorded_window_that_could_not_be_applied():
    from opennmr.notebook import ringing as nb
    note = "Bruker window WDW=2 (gm) is not yet supported for raw-FID reprocessing."
    d = nb.check_f2_ringing(spectrum_with({"procs": {"WDW": 2}, "window_unsupported": note}))
    assert d["window_unsupported"] == note and "cannot reproduce" in d["window_in_use"]
    assert "WITHOUT a window" in d["report"] and "TopSpin" in d["report"]


def test_wrapper_follows_the_live_process_settings_and_truncation():
    from opennmr.notebook import ringing as nb
    d = nb.check_f2_ringing(spectrum_with({"live_window": "qsine", "live_ssb": 2.0, "live_truncate": 512}))
    assert "qsine" in d["window_in_use"] and d["n_points"] == 512
    d = nb.check_f2_ringing(spectrum_with({"tdeff_f2_points": 600}))
    assert d["n_points"] == 600


def test_wrapper_can_ask_what_a_window_would_leave():
    from opennmr.notebook import ringing as nb
    sp = spectrum_with({})
    now = nb.check_f2_ringing(sp)
    asked = nb.check_f2_ringing(sp, window="qsine", window_params={"ssb": 2})
    assert asked["ripple_windowed"] < now["ripple_windowed"] / 3 and "as asked" in asked["window_in_use"]
    assert nb.check_f2_ringing(sp, window="none")["ripple_windowed"] == pytest.approx(now["ripple"])


def test_wrapper_refuses_a_spectrum_with_no_raw_fid():
    from opennmr.notebook import ringing as nb
    sp = spectrum_with({})
    del sp.metadata.raw["fid_raw"]
    with pytest.raises(ValueError, match="raw FID"):
        nb.check_f2_ringing(sp)


def test_the_notebook_package_exports_it():
    import opennmr.notebook as nbk
    assert nbk.check_f2_ringing is nbk.ringing.check_f2_ringing


# --- the Process tab ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_process_tab_applies_the_suggested_window(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    window._populate_window_combo("bruker")
    assert window.check_ringing_button.text().startswith("Check for truncation")
    diag = {"suggestion": {"window": "qsine", "ssb": 3.0, "label": "QSINE, SSB 3"}}
    assert window._apply_ringing_suggestion(diag) is True
    assert window.window_combo.currentData() == "qsine" and window.ssb_spin.value() == 3.0
    assert window.ssb_spin.isEnabled()
    assert window._apply_ringing_suggestion({"suggestion": None}) is False
    assert window._run_ringing_check() is None           # nothing selected: no crash
