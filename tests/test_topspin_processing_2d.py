"""TopSpin's 2D processing kept in openNMR (explicit user report,
2026-10-14, with Good_2D, a 13C-1H hCH: "the topspin processing are not
conserved in openNMR"; "when I toggle 'process indirect dimension' it
erases the processing that I did before and even unphased the
spectrum"; "it is not obvious how to process only the direct or
indirect dimensions").

What was wrong, measured on that dataset against TopSpin's own 2rr:

* TopSpin's ``TDeff`` was ignored -- F1 used all 563 increments where
  TopSpin uses 128 (proc2s TDeff = 256 points). Correlation with the 2rr
  0.785 -> 0.974 with it.
* A saved openNMR state held ``recorded_phase0 = 0`` (written by the
  2026-10-11 bug) and replaced the file's PHC0 = 40.6 deg on every reopen;
  and that bug's own fix -- autophase kept on when saved processing is
  read back -- had been lost from the code.
* Process Indirect Dimension re-read the file with defaults, dropping the
  window, its SSB and the F1 truncation.
"""

import json
import os
import shutil

import numpy as np
import pytest

from opennmr.io import bruker
from opennmr.io import persistence

from test_bruker_raw_2d import _make_2d_experiment_dir, _mock_2d, _synthetic_2d_fid
from test_gui import _make_raw_2d_dataset, _write_params, qapp  # noqa: F401 -- qapp is a fixture


def _hypercomplex_fid(n_inc=16, n_f2=64):
    t2 = np.arange(n_f2) / 8000.0
    rows = []
    for k in range(n_inc):
        t1 = k / 2000.0
        base = np.exp(-t2 * 300) * np.exp(2j * np.pi * 1000 * t2)
        rows += [base * np.cos(2 * np.pi * 300 * t1) * np.exp(-t1 * 50),
                 base * np.sin(2 * np.pi * 300 * t1) * np.exp(-t1 * 50)]
    return np.array(rows)


ACQUS = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0, "TD": 128}


# ---- the reader -----------------------------------------------------------------

def test_read_2d_uses_topspin_tdeff_in_f1(tmp_path, monkeypatch):
    fid = _hypercomplex_fid()
    path = _make_2d_experiment_dir(tmp_path, proc2s_overrides={"TDeff": 12})
    _mock_2d(monkeypatch, ACQUS, fid, acqu2s={"NUC1": "13C", "BF1": 100.6, "O1": 0.0,
                                               "FnMODE": 5, "TD": 32})
    spec = bruker.read_2d(path, pseudo_2d=False)
    assert spec.metadata.raw["tdeff_f1_increments"] == 6        # 12 points, 2 per increment
    explicit = bruker.read_2d(path, pseudo_2d=False, truncate_f1=6, recorded_tdeff=False)
    np.testing.assert_allclose(spec.data, explicit.data)
    every = bruker.read_2d(path, pseudo_2d=False, recorded_tdeff=False)
    assert "tdeff_f1_increments" not in every.metadata.raw
    assert not np.allclose(every.data, spec.data)
    # An explicit truncation wins over TDeff.
    mine = bruker.read_2d(path, pseudo_2d=False, truncate_f1=10)
    assert "tdeff_f1_increments" not in mine.metadata.raw


def test_read_2d_and_1d_use_topspin_tdeff_in_f2(tmp_path, monkeypatch):
    fid = _synthetic_2d_fid(n_f1=8, n_f2=64)
    path = _make_2d_experiment_dir(tmp_path)
    procs = os.path.join(path, "pdata", "1", "procs")
    with open(procs, "a") as fh:
        fh.write("##$TDeff= 80\n")
    _mock_2d(monkeypatch, ACQUS, fid)
    spec = bruker.read_2d(path, homonuclear=True)
    assert spec.metadata.raw["tdeff_f2_points"] == 40            # 80 TD points = 40 complex
    explicit = bruker.read_2d(path, homonuclear=True, truncate=40, recorded_tdeff=False)
    np.testing.assert_allclose(spec.data, explicit.data)


def test_tdeff_equal_to_td_changes_nothing(tmp_path, monkeypatch):
    fid = _hypercomplex_fid()
    path = _make_2d_experiment_dir(tmp_path, proc2s_overrides={"TDeff": 32})
    _mock_2d(monkeypatch, ACQUS, fid, acqu2s={"NUC1": "13C", "FnMODE": 5, "TD": 32})
    spec = bruker.read_2d(path, pseudo_2d=False)
    assert "tdeff_f1_increments" not in spec.metadata.raw


# ---- a saved state never replaces the file's recorded phase --------------------

def _write_sidecar(path, state, version=2):
    with open(persistence.processing_sidecar_path(path), "w", encoding="utf-8") as fh:
        json.dump({"opennmr_processing_version": version, "state": state}, fh)


def test_saved_recorded_phase_is_ignored_unless_the_user_set_it(tmp_path):
    path = str(tmp_path)
    _write_sidecar(path, {"recorded_phase0": 0.0, "recorded_phase0_f1": 0.0, "phase0": 349.64,
                          "live_window": "qsine"})
    state = persistence.load_processing_sidecar(path, vendor="bruker")
    assert "recorded_phase0" not in state and state["phase0"] == 349.64
    assert state["_saved_recorded_phase_ignored"] == {"recorded_phase0": 0.0,
                                                      "recorded_phase0_f1": 0.0}
    # Deliberately zeroed (Zero Phase 0): kept.
    _write_sidecar(path, {"recorded_phase0": 0.0, persistence.RECORDED_PHASE_USER_SET_KEY: True})
    assert persistence.load_processing_sidecar(path, vendor="bruker")["recorded_phase0"] == 0.0
    # Other vendors compute their base: unchanged.
    _write_sidecar(path, {"recorded_phase0": 12.0})
    assert persistence.load_processing_sidecar(path, vendor="xpulse")["recorded_phase0"] == 12.0


def test_saved_phase_is_read_back_on_top_of_the_recorded_one():
    from opennmr.gui.main_window import MainWindow
    kwargs = MainWindow._saved_processing_read_kwargs({"phase0": 349.64, "live_window": "qsine",
                                                       "live_ssb": 3.0, "live_truncate_f1": 128})
    assert kwargs.get("autophase", True) is True
    assert kwargs["truncate_f1"] == 128 and kwargs["window"] == "qsine"


# ---- the GUI ----------------------------------------------------------------------

def _genuine_2d(tmp_path, monkeypatch, tdeff1=8):
    fid = _hypercomplex_fid()
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=fid.shape[0], n_f2=fid.shape[1],
                                fid=fid, fn_mode=5)
    pdata = tmp_path / "rawexpt2d" / "pdata" / "1"
    _write_params(pdata / "procs", {"SW_p": 8000.0, "SI": 64, "SF": 400.13, "WDW": 4, "SSB": 3,
                                    "PHC0": 40.0, "PHC1": 0.0})
    _write_params(pdata / "proc2s", {"SW_p": 2000.0, "SI": 32, "SF": 100.6, "WDW": 0,
                                     "FT_mod": 4, "TDeff": tdeff1, "PHC0": 0.0})
    _write_params(tmp_path / "rawexpt2d" / "acqu2s", {"NUC1": "13C", "BF1": 100.6, "O1": 0.0})
    # TD in acqu2s for the reader (the fixture's mocked parameter dict).
    dic = bruker.read_acqus_files(path)
    dic["acqu2s"]["TD"] = fid.shape[0]
    dic["acqus"]["TD"] = 2 * fid.shape[1]
    return path


def test_gui_opens_with_tdeff_and_says_so(qapp, tmp_path, monkeypatch):
    from opennmr.gui.main_window import MainWindow
    path = _genuine_2d(tmp_path, monkeypatch)
    window = MainWindow()
    window._load(path, vendor="bruker")
    spec = window.session.spectra[window._selected_index()]
    assert not spec.is_pseudo_2d
    assert spec.metadata.raw["tdeff_f1_increments"] == 4
    assert window.tdeff_f1_spin.value() == 4
    assert "TDeff 8 points = 4 increments" in window.tdeff_f1_recorded_label.text()
    assert window.tdeff_f1_spin.suffix() == " increments"
    assert window.fid_f1_header.isVisibleTo(window) or not window.fid_f1_header.isHidden()
    # A live rebuild keeps TopSpin's TDeff rather than undoing it.
    before = spec.data.copy()
    window._recompute_multi_row_data(spec)
    np.testing.assert_allclose(spec.data, before, atol=1e-9 * np.abs(before).max())


def test_process_indirect_round_trip_keeps_the_processing(qapp, tmp_path, monkeypatch):
    from opennmr.gui.main_window import MainWindow
    path = _genuine_2d(tmp_path, monkeypatch)
    window = MainWindow()
    window._load(path, vendor="bruker")
    i = window._selected_index()
    window.window_combo.setCurrentIndex(window.window_combo.findData("qsine"))
    window.ssb_spin.setValue(2.0)
    window.tdeff_f1_spin.setValue(3)
    window._persist_fid_processing_controls_to_metadata(window.session.spectra[i])
    window._recompute_multi_row_data(window.session.spectra[i])
    processed = window.session.spectra[i].data.copy()
    window.process_indirect_button.setChecked(False)
    assert window.session.spectra[i].is_pseudo_2d
    assert window.session.spectra[i].metadata.raw["live_window"] == "qsine"
    window.process_indirect_button.setChecked(True)
    spec = window.session.spectra[i]
    assert not spec.is_pseudo_2d
    assert spec.metadata.raw["live_window"] == "qsine" and spec.metadata.raw["live_ssb"] == 2.0
    assert spec.metadata.raw["live_truncate_f1"] == 3 and window.tdeff_f1_spin.value() == 3
    assert spec.metadata.raw["recorded_phase0"] == 40.0
    np.testing.assert_allclose(spec.data, processed, atol=1e-6 * np.abs(processed).max())


def test_setting_opens_bruker_data_unprocessed(qapp, tmp_path, monkeypatch):
    from opennmr.gui.main_window import MainWindow
    path = _genuine_2d(tmp_path, monkeypatch)
    window = MainWindow()
    assert window.topspin_processing_checkbox.isChecked()
    window.topspin_processing_checkbox.setChecked(False)
    again = MainWindow()
    assert not again.topspin_processing_checkbox.isChecked()
    again._load(path, vendor="bruker")
    spec = again.session.spectra[again._selected_index()]
    assert spec.metadata.raw["recorded_phase0"] == 0.0
    assert "tdeff_f1_increments" not in spec.metadata.raw
    assert again.window_combo.currentData() == "none"
    before = spec.data.copy()
    again._recompute_multi_row_data(spec)        # a live rebuild agrees with the read
    np.testing.assert_allclose(spec.data, before, atol=1e-9 * np.abs(before).max())


# ---- real data ------------------------------------------------------------------

GOOD_2D = os.path.join(os.path.dirname(__file__), "..", "test_data", "Good_2D", "28")


@pytest.mark.skipif(not os.path.isdir(GOOD_2D), reason="test_data/Good_2D/28 not present")
def test_good_2d_matches_topspin(tmp_path):
    expt = tmp_path / "28"
    shutil.copytree(GOOD_2D, expt, ignore=shutil.ignore_patterns("opennmr_*"))
    ours = bruker.read_2d(str(expt), pseudo_2d=False)
    theirs = bruker.read_2d_processed(str(expt))
    assert ours.metadata.raw["tdeff_f1_increments"] == 128
    r = np.corrcoef(ours.data.ravel(), theirs.data.ravel())[0, 1]
    assert r > 0.97
