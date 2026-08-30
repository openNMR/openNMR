"""
Mocked-`ng.bruker.read`/`remove_digital_filter` coverage for `read_2d`
-- see test_bruker_raw_fid.py's module docstring for why this
approach (real code, synthetic raw data) rather than real Bruker
files, which aren't available in this environment.
"""

import numpy as np
import pytest

from opennmr.io import bruker


def _write_procs(path, params):
    lines = [f"##${k}= {v}" for k, v in params.items()]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _make_2d_experiment_dir(tmp_path, procno=1, proc2s_overrides=None):
    expt = tmp_path / "expt2d"
    pdata = expt / "pdata" / str(procno)
    pdata.mkdir(parents=True)
    _write_procs(pdata / "procs", {"SW_p": 8000.0, "SI": 64, "SF": 400.13, "WDW": 0})
    proc2s = {"SW_p": 2000.0, "SI": 32, "SF": 100.6, "WDW": 0}
    proc2s.update(proc2s_overrides or {})
    _write_procs(pdata / "proc2s", proc2s)
    return str(expt)


def _synthetic_2d_fid(n_f1=16, n_f2=32, sw2=8000.0, f0_hz=1000.0):
    t2 = np.arange(n_f2) / sw2
    row = np.exp(-t2 * 300) * np.exp(2j * np.pi * f0_hz * t2)
    return np.tile(row, (n_f1, 1)).astype(np.complex128)


def _synthetic_2d_fid_varying_peak_position(n_f1=8, n_f2=64):
    """Each row's magnitude peak sits at a different index -- unlike
    real imaging/echo data (one genuine, consistently-placed echo),
    this mimics a non-echo-shaped pseudo-2D series (e.g. T2/CPMG) where
    find_echo_top has nothing real to lock onto and lands wherever the
    (here, deliberately shifted) maximum happens to be, row to row."""
    t2 = np.arange(n_f2)
    rows = []
    for row_index in range(n_f1):
        peak_at = 10 + row_index  # a different "echo top" every row
        row = np.zeros(n_f2, dtype=np.complex128)
        row[peak_at] = 1.0 + 0.5j
        row += 0.01 * np.exp(-((t2 - peak_at) ** 2) / 8.0)  # a little width so parabolic fit is well-defined
        rows.append(row)
    return np.array(rows)


def _mock_2d(monkeypatch, acqus, fid, acqu2s=None):
    dic = {"acqus": acqus}
    if acqu2s is not None:
        dic["acqu2s"] = acqu2s
    monkeypatch.setattr(bruker.ng.bruker, "read", lambda path, **kw: (dic, fid))
    monkeypatch.setattr(bruker.ng.bruker, "remove_digital_filter", lambda d, data: data)


def test_read_2d_homonuclear_shape(tmp_path, monkeypatch):
    path = _make_2d_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_2d_fid()
    _mock_2d(monkeypatch, acqus, fid)

    spectrum = bruker.read_2d(path, homonuclear=True, autophase=False)
    assert spectrum.ndim == 2
    assert spectrum.data.shape == (32, 64)  # zero-filled to (proc2s SI, procs SI)
    assert spectrum.metadata.nucleus == "1H"
    assert spectrum.metadata.nucleus_indirect == "1H"
    assert not spectrum.is_pseudo_2d


def test_read_2d_heteronuclear_uses_acqu2s(tmp_path, monkeypatch):
    path = _make_2d_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    acqu2s = {"NUC1": "13C", "BF1": 100.6, "O1": 5000.0}
    fid = _synthetic_2d_fid()
    _mock_2d(monkeypatch, acqus, fid, acqu2s=acqu2s)

    spectrum = bruker.read_2d(path, homonuclear=False, autophase=False)
    assert spectrum.metadata.nucleus == "1H"
    assert spectrum.metadata.nucleus_indirect == "13C"
    assert spectrum.metadata.sfo1_indirect == 100.6  # from proc2s SF, not acqu2s BF1
    assert spectrum.axes[0].kind == "frequency"


def test_read_2d_pseudo2d_indirect_is_index_axis(tmp_path, monkeypatch):
    path = _make_2d_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    acqu2s = {"NUC1": "off", "BF1": 400.0, "O1": 0.0}
    fid = _synthetic_2d_fid()
    _mock_2d(monkeypatch, acqus, fid, acqu2s=acqu2s)

    spectrum = bruker.read_2d(path, homonuclear=False, autophase=False)
    assert spectrum.is_pseudo_2d
    assert spectrum.axes[0].kind == "index"
    assert spectrum.metadata.sfo1_indirect is None
    # No F1 zero-fill/FFT for pseudo-2D -- row count stays as acquired.
    assert spectrum.data.shape[0] == 16


def test_read_2d_pseudo2d_override_forces_index_axis(tmp_path, monkeypatch):
    # acqu2s claims a real nucleus, but the caller knows better (e.g.
    # acqu2s wasn't conventionally populated for this pulse sequence).
    path = _make_2d_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    acqu2s = {"NUC1": "1H", "BF1": 400.0, "O1": 0.0}
    fid = _synthetic_2d_fid()
    _mock_2d(monkeypatch, acqus, fid, acqu2s=acqu2s)

    spectrum = bruker.read_2d(path, homonuclear=False, pseudo_2d=True, autophase=False)
    assert spectrum.is_pseudo_2d
    assert spectrum.axes[0].kind == "index"


def test_read_2d_pseudo2d_override_forces_frequency_axis(tmp_path, monkeypatch):
    # acqu2s says "off" (pseudo-2D), but the caller forces a real F1
    # transform anyway.
    path = _make_2d_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    acqu2s = {"NUC1": "off", "BF1": 400.0, "O1": 0.0}
    fid = _synthetic_2d_fid()
    _mock_2d(monkeypatch, acqus, fid, acqu2s=acqu2s)

    spectrum = bruker.read_2d(path, homonuclear=False, pseudo_2d=False, autophase=False)
    assert not spectrum.is_pseudo_2d
    assert spectrum.axes[0].kind == "frequency"


def test_read_2d_autophase_runs(tmp_path, monkeypatch):
    path = _make_2d_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_2d_fid()
    _mock_2d(monkeypatch, acqus, fid)

    spectrum = bruker.read_2d(path, homonuclear=True, autophase=True)
    assert spectrum.data.shape == (32, 64)


def test_read_2d_default_phase_mode_uses_recorded_not_apc0(tmp_path, monkeypatch):
    # Same "recorded" default as read_1d (see its own test of the same
    # name) -- read_2d previously had no phase_mode at all and always
    # re-guessed via apc0, discarding F2's own recorded PHC0.
    path = _make_2d_experiment_dir(tmp_path)
    _write_procs(tmp_path / "expt2d" / "pdata" / "1" / "procs",
                  {"SW_p": 8000.0, "SI": 64, "SF": 400.13, "WDW": 0, "PHC0": 42.0})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_2d_fid()
    _mock_2d(monkeypatch, acqus, fid)

    spectrum = bruker.read_2d(path, homonuclear=True, autophase=True)

    assert spectrum.metadata.raw["phase0"] == pytest.approx(42.0)


def test_read_2d_f2_window_override_applies_per_row(tmp_path, monkeypatch):
    path = _make_2d_experiment_dir(tmp_path)  # WDW=0 recorded (no-op)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_2d_fid()
    _mock_2d(monkeypatch, acqus, fid)
    unwindowed = bruker.read_2d(path, homonuclear=True, autophase=False)

    _mock_2d(monkeypatch, acqus, fid)
    windowed = bruker.read_2d(path, homonuclear=True, autophase=False, window="hann")

    assert not np.allclose(unwindowed.data, windowed.data)
    assert windowed.data.shape == unwindowed.data.shape


def test_read_2d_zero_fill_override_changes_f2_size_only(tmp_path, monkeypatch):
    path = _make_2d_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_2d_fid()
    _mock_2d(monkeypatch, acqus, fid)

    spectrum = bruker.read_2d(path, homonuclear=True, autophase=False, zero_fill=128)
    assert spectrum.data.shape == (32, 128)  # F1 (proc2s SI) untouched, F2 overridden


def test_read_2d_fid_shift_and_linear_predict_run_without_error(tmp_path, monkeypatch):
    path = _make_2d_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_2d_fid(n_f2=64)
    _mock_2d(monkeypatch, acqus, fid)

    spectrum = bruker.read_2d(
        path, homonuclear=True, autophase=False,
        fid_shift=3.0, linear_predict_points=8, linear_predict_order=8,
    )
    assert spectrum.data.shape == (32, 64)


def test_read_2d_align_echo_handles_per_row_echo_position_disagreement(tmp_path, monkeypatch):
    # Regression test: align_echo's own auto-truncation caps each row
    # at its own len(row) - ceil(echo_top) (see prepare_fid). Applied
    # per row via np.apply_along_axis, a real (non-imaging) pseudo-2D
    # series -- rows with no genuine common echo shape, so
    # find_echo_top lands wherever each row's own maximum happens to
    # be -- used to disagree row to row and crash np.apply_along_axis
    # entirely ("could not broadcast..."), found on real Bruker T2
    # data via Align Echo. read_2d must produce one consistent shape
    # regardless.
    path = _make_2d_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_2d_fid_varying_peak_position(n_f1=8, n_f2=64)
    _mock_2d(monkeypatch, acqus, fid)

    spectrum = bruker.read_2d(path, homonuclear=True, autophase=False, align_echo=True)

    assert spectrum.data.shape[0] == 32  # F1 zero-filled to proc2s's own SI, as usual
    assert spectrum.data_imag is not None
