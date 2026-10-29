"""FCOR (TopSpin's first-point scaling) in the Bruker readers and in the
GUI's live recompute.

Found on a real aMAT 2D (strong signals, F1 FCOR 0.5): without it the
first F1 point was doubled, which leaks into every F1 row as a flat
offset -- "more features" in a row than TopSpin shows. The reader and
the GUI's live rebuild must apply it identically, or the spectrum jumps
the moment any control is touched.
"""

import numpy as np
import pytest

from opennmr.gui.main_window import MainWindow
from opennmr.io import bruker

from test_gui import qapp, _make_raw_2d_dataset, _write_params  # noqa: F401
from test_bruker_raw_2d import _mock_2d, _make_2d_experiment_dir, _synthetic_2d_fid


def test_first_point_factor_defaults_to_one_and_tolerates_junk():
    assert bruker.first_point_factor(None) == 1.0
    assert bruker.first_point_factor({}) == 1.0
    assert bruker.first_point_factor({"FCOR": 0.5}) == 0.5
    assert bruker.first_point_factor({"FCOR": "x"}) == 1.0


def test_apply_first_point_scaling_scales_only_the_first_point_and_copies():
    fid = np.ones((3, 8), dtype=complex)
    out = bruker.apply_first_point_scaling(fid, {"FCOR": 0.5})
    assert np.allclose(out[:, 0], 0.5)
    assert np.allclose(out[:, 1:], 1.0)
    assert np.allclose(fid, 1.0)            # input untouched
    assert bruker.apply_first_point_scaling(fid, {}) is fid   # no-op returns as is


def _read_2d(tmp_path, monkeypatch, fcor_f1=None, fcor_f2=None, name="a"):
    path = _make_2d_experiment_dir(tmp_path / name)
    for fname, fcor in (("procs", fcor_f2), ("proc2s", fcor_f1)):
        if fcor is not None:
            with open(f"{path}/pdata/1/{fname}", "a") as handle:
                handle.write(f"##$FCOR= {fcor}\n")
    fid = _synthetic_2d_fid(n_f1=8, n_f2=16)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    _mock_2d(monkeypatch, acqus, fid,
             acqu2s={"NUC1": "13C", "BF1": 100.0, "O1": 0.0, "FnMODE": 5})
    return bruker.read_2d(path, pseudo_2d=False, autophase=False)


def test_read_2d_f2_fcor_scales_first_point_of_every_row(tmp_path, monkeypatch):
    plain = _read_2d(tmp_path, monkeypatch, name="plain")
    scaled = _read_2d(tmp_path, monkeypatch, fcor_f2=0.5, name="scaled")
    assert scaled.metadata.raw["fcor"] == 0.5
    f_plain, f_scaled = plain.metadata.raw["fid"], scaled.metadata.raw["fid"]
    np.testing.assert_allclose(f_scaled[:, 0], 0.5 * f_plain[:, 0])
    np.testing.assert_allclose(f_scaled[:, 1:], f_plain[:, 1:])


def test_read_2d_f1_fcor_adds_a_flat_offset_across_f1(tmp_path, monkeypatch):
    # Scaling the first t1 point by (0.5 - 1) changes the F1 FFT by a
    # constant along F1 (an impulse at t=0 is flat in frequency).
    plain = _read_2d(tmp_path, monkeypatch, name="plain")
    scaled = _read_2d(tmp_path, monkeypatch, fcor_f1=0.5, name="scaled")
    assert scaled.metadata.raw["fcor_f1"] == 0.5
    diff = scaled.complex_data - plain.complex_data
    assert np.abs(diff).max() > 0
    np.testing.assert_allclose(diff, np.broadcast_to(diff[0], diff.shape), atol=1e-9)


def test_gui_live_recompute_matches_reader_with_fcor(qapp, tmp_path, monkeypatch):  # noqa: F811
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=16, name="fcorlive2d", f1_nuc="13C")
    base = tmp_path / "fcorlive2d" / "pdata" / "1"
    _write_params(base / "procs", {"SW_p": 8000.0, "SI": 16, "SF": 400.13, "WDW": 0, "FCOR": 0.5})
    _write_params(base / "proc2s", {"SW_p": 2000.0, "SI": 4, "SF": 100.6, "WDW": 0, "FCOR": 0.5})
    window = MainWindow()
    window._load(path, vendor="bruker")
    window.process_indirect_button.setChecked(True)
    spectrum = window.session.spectra[window._selected_index()]
    spectrum.metadata.raw["baseline_correct"] = False
    original = spectrum.complex_data.copy()
    window._recompute_multi_row_data(spectrum)   # no control touched: must be a no-op
    np.testing.assert_allclose(spectrum.complex_data, original, atol=1e-6)


def test_gui_live_row_matches_reader_row_with_fcor(qapp, tmp_path, monkeypatch):  # noqa: F811
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4, n_f2=16, name="fcorrow2d", f1_nuc="13C")
    _write_params(tmp_path / "fcorrow2d" / "pdata" / "1" / "procs",
                  {"SW_p": 8000.0, "SI": 16, "SF": 400.13, "WDW": 0, "FCOR": 0.5})
    window = MainWindow()
    window._load(path, vendor="bruker")
    spectrum = window.session.spectra[window._selected_index()]
    raw = spectrum.metadata.raw
    row = raw["fid_raw"][0]
    _, scaled = window._process_raw_fid_row(spectrum, row, raw["fid_dt"])
    raw["procs"] = dict(raw["procs"], FCOR=1.0)
    _, plain = window._process_raw_fid_row(spectrum, row, raw["fid_dt"])
    # An impulse at t=0 is flat in frequency: the whole difference is a
    # constant, equal to (0.5 - 1) * first point / NS (no zero-fill here).
    diff = scaled - plain
    np.testing.assert_allclose(diff, np.full_like(diff, diff[0]), atol=1e-9)
    assert abs(diff[0]) == pytest.approx(0.5 * abs(row[0]) / (spectrum.metadata.ns or 1.0), rel=1e-6)
