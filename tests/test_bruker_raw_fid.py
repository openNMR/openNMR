"""
Tests for the raw-FID path (`read_1d`/`read_2d`) that don't depend on
having a real Bruker `fid`/`ser` file: `ng.bruker.read()` and
`ng.bruker.remove_digital_filter()` -- the two calls that are
genuinely nmrglue's own tested responsibility -- are mocked with
controlled synthetic data, so everything downstream (window dispatch,
zero-fill, FFT, phasing, cropping, metadata/axis construction -- i.e.
the code actually written for this package) is exercised for real.

Full end-to-end validation against a real raw Bruker dataset is still
pending real data from the user (see the project plan's Verification
section) -- this is a narrower, but honest, slice of coverage in the
meantime.
"""

import numpy as np
import pytest

from opennmr.io import bruker


def _write_procs(path, params):
    lines = []
    for key, value in params.items():
        lines.append(f"##${key}= {value}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _make_experiment_dir(tmp_path, procno=1, procs_overrides=None):
    expt = tmp_path / "expt1"
    pdata = expt / "pdata" / str(procno)
    pdata.mkdir(parents=True)
    params = {"SW_p": 8000.0, "SI": 1024, "SF": 400.13, "WDW": 0, "LB": 0.0, "SSB": 0.0,
              "PHC0": 0.0, "PHC1": 0.0}
    params.update(procs_overrides or {})
    _write_procs(pdata / "procs", params)
    return str(expt)


def _synthetic_fid(n=512, sw=8000.0, f0_hz=1000.0, decay=200.0):
    t = np.arange(n) / sw
    return (np.exp(-t * decay) * np.exp(2j * np.pi * f0_hz * t)).astype(np.complex128)


def _mock_ng_bruker_read(monkeypatch, acqus, fid):
    dic = {"acqus": acqus}
    monkeypatch.setattr(bruker.ng.bruker, "read", lambda path, **kw: (dic, fid))
    monkeypatch.setattr(bruker.ng.bruker, "remove_digital_filter", lambda dic, data: data)


def test_read_1d_shape_and_scaling(tmp_path, monkeypatch):
    path = _make_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 4.0}
    fid = _synthetic_fid()
    _mock_ng_bruker_read(monkeypatch, acqus, fid)

    spectrum = bruker.read_1d(path, mass=2.0, autophase=False, phase0=0.0, phase1=0.0)

    assert spectrum.ndim == 1
    assert len(spectrum.data) == 1024  # zero-filled to SI
    assert spectrum.metadata.vendor == "bruker"
    assert spectrum.metadata.nucleus == "1H"
    assert spectrum.metadata.ns == 4.0
    # Descending ppm axis (NMR convention).
    assert spectrum.axes[0].ppm[0] > spectrum.axes[0].ppm[-1]


def test_read_1d_em_window_runs(tmp_path, monkeypatch):
    path = _make_experiment_dir(tmp_path, procs_overrides={"WDW": 1, "LB": 20.0})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_fid()
    _mock_ng_bruker_read(monkeypatch, acqus, fid)

    spectrum = bruker.read_1d(path, autophase=False)
    assert len(spectrum.data) == 1024


def test_read_1d_unsupported_window_raises(tmp_path, monkeypatch):
    path = _make_experiment_dir(tmp_path, procs_overrides={"WDW": 2})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_fid()
    _mock_ng_bruker_read(monkeypatch, acqus, fid)

    with pytest.raises(NotImplementedError):
        bruker.read_1d(path, autophase=False)


def test_read_1d_crop_matches_processed_axis_formula(tmp_path, monkeypatch):
    path = _make_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_fid()
    _mock_ng_bruker_read(monkeypatch, acqus, fid)

    full = bruker.read_1d(path, autophase=False)
    lo, hi = full.axes[0].ppm[-1], full.axes[0].ppm[0]
    mid = (lo + hi) / 2

    _mock_ng_bruker_read(monkeypatch, acqus, fid)  # re-arm (fid consumed by reference, not state)
    cropped = bruker.read_1d(path, f1p=lo, f2p=mid, autophase=False)
    assert len(cropped.data) < len(full.data)
    assert cropped.axes[0].ppm.max() <= mid + 1e-9


def test_read_1d_autophase_apc0_runs(tmp_path, monkeypatch):
    path = _make_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_fid()
    _mock_ng_bruker_read(monkeypatch, acqus, fid)

    spectrum = bruker.read_1d(path, autophase=True, phase_mode="apc0")
    assert "phase0" in spectrum.metadata.raw
    assert len(spectrum.data) == 1024


def test_read_1d_window_override_replaces_recorded_wdw(tmp_path, monkeypatch):
    # WDW=0 (none) recorded, but a Hann window is requested explicitly
    # -- the override should win, not the (no-op) recorded window.
    path = _make_experiment_dir(tmp_path, procs_overrides={"WDW": 0})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_fid()
    _mock_ng_bruker_read(monkeypatch, acqus, fid)
    unwindowed = bruker.read_1d(path, autophase=False)

    _mock_ng_bruker_read(monkeypatch, acqus, fid)
    windowed = bruker.read_1d(path, autophase=False, window="hann")

    assert not np.allclose(unwindowed.data, windowed.data)


def test_read_1d_zero_fill_override(tmp_path, monkeypatch):
    path = _make_experiment_dir(tmp_path)  # SI=1024 recorded
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_fid()
    _mock_ng_bruker_read(monkeypatch, acqus, fid)

    spectrum = bruker.read_1d(path, autophase=False, zero_fill=2048)
    assert len(spectrum.data) == 2048


def test_read_1d_fid_shift_changes_output(tmp_path, monkeypatch):
    path = _make_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_fid()
    _mock_ng_bruker_read(monkeypatch, acqus, fid)
    unshifted = bruker.read_1d(path, autophase=False)

    _mock_ng_bruker_read(monkeypatch, acqus, fid)
    shifted = bruker.read_1d(path, autophase=False, fid_shift=5.0)

    assert not np.allclose(unshifted.data, shifted.data)


def test_read_1d_linear_predict_extends_before_truncation_to_si(tmp_path, monkeypatch):
    # SI (1024) is bigger than the FID (512) either way, so the
    # extension is only visible via a change in the resulting
    # spectrum -- the point is that this doesn't raise and the two
    # differ (LP actually ran and fed different, longer FID data into
    # the same zero-fill/FFT pipeline).
    path = _make_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_fid(n=512)
    _mock_ng_bruker_read(monkeypatch, acqus, fid)
    plain = bruker.read_1d(path, autophase=False)

    _mock_ng_bruker_read(monkeypatch, acqus, fid)
    lp = bruker.read_1d(path, autophase=False, linear_predict_points=32, linear_predict_order=16)

    assert len(lp.data) == 1024
    # A clean single-exponential FID is exactly what LP models, so the
    # two are numerically close either way -- a loose default tolerance
    # would spuriously pass even without LP actually running.
    assert not np.allclose(plain.data, lp.data, atol=1e-9, rtol=1e-9)
