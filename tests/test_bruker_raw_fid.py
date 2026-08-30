"""
Tests for the raw-FID path (`read_1d`/`read_2d`) that don't depend on
having a real Bruker `fid`/`ser` file: `ng.bruker.read()` and
`ng.bruker.remove_digital_filter()` -- the two calls that are
genuinely nmrglue's own tested responsibility -- are mocked with
controlled synthetic data, so everything downstream (window dispatch,
zero-fill, FFT, phasing, cropping, metadata/axis construction -- i.e.
the code actually written for this package) is exercised for real.

Real raw Bruker 2D data has since been verified against directly (not
committed here -- test_data/ is gitignored, real instrument data) --
see TODO.md's "Bruker 2D verification" entry for what that surfaced
(a real nmrglue shape-guessing bug for DTYPA=2 2D data, and a real
read_2d_processed bug, both fixed) and the regression coverage it led
to: this file's own test_read_1d_default_phase_mode_uses_recorded_not_apc0,
its read_2d counterpart in test_bruker_raw_2d.py, and the
read_2d_processed/_read_auto fallback tests in test_bruker.py.
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


# _apply_bruker_window itself still raises NotImplementedError for an
# unsupported WDW code (see test_bruker.py) -- read_1d/read_2d now
# catch that internally instead of letting it propagate; see
# test_read_1d_unsupported_window_stays_on_raw_fid_path below.


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


def test_read_1d_default_phase_mode_uses_recorded_not_apc0(tmp_path, monkeypatch):
    # CRITICAL per collaborator feedback (Euan Bassey, see TODO.md):
    # a fresh load must reproduce whatever phase TopSpin itself had --
    # a user's own careful manual phasing, or TopSpin's own automatic
    # processing -- not silently discard it by re-guessing from
    # scratch. phase_mode's default changed from "apc0" to "recorded".
    path = _make_experiment_dir(tmp_path, procs_overrides={"PHC0": 37.5, "PHC1": 0.0})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_fid()
    _mock_ng_bruker_read(monkeypatch, acqus, fid)

    spectrum = bruker.read_1d(path, autophase=True)  # phase_mode not specified -- uses the default

    assert spectrum.metadata.raw["phase0"] == pytest.approx(37.5)


def test_read_1d_recorded_phase_mode_ignores_data_entirely(tmp_path, monkeypatch):
    # "recorded" reproduces PHC0 verbatim -- it must not fall back to
    # apc0's own grid search even when PHC0 happens to be 0.0 (the
    # never-manually-phased case), since that 0.0 is itself real
    # information (TopSpin's own last-applied phase).
    path = _make_experiment_dir(tmp_path, procs_overrides={"PHC0": 0.0, "PHC1": 0.0})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_fid()
    _mock_ng_bruker_read(monkeypatch, acqus, fid)

    spectrum = bruker.read_1d(path, autophase=True, phase_mode="recorded")

    assert spectrum.metadata.raw["phase0"] == pytest.approx(0.0)


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


# ============================================================
# Unsupported window function: stay on the raw-FID path
# ============================================================

def test_read_1d_unsupported_window_stays_on_raw_fid_path(tmp_path, monkeypatch):
    # Regression test: a Bruker WDW code this package doesn't (yet)
    # implement (e.g. GM) used to make the whole file unopenable
    # (read_1d itself raised NotImplementedError). It should instead
    # stay on the raw-FID path -- full reprocessing still available --
    # with the FID simply left unwindowed, flagged via
    # metadata.raw["window_unsupported"] rather than silently applying
    # no window with no explanation. The alternative of falling back
    # to TopSpin's already-processed, non-reprocessable output was
    # considered and rejected: this whole package's premise is reading
    # the raw FID, so an unsupported *parameter* shouldn't force
    # abandoning the raw-FID path entirely when everything else about
    # it (zero-fill, phase, echo-align, ...) still works fine.
    path = _make_experiment_dir(tmp_path, procs_overrides={"WDW": 2})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_fid()
    _mock_ng_bruker_read(monkeypatch, acqus, fid)

    spectrum = bruker.read_1d(path, autophase=False, zero_fill=2048)

    assert len(spectrum.data) == 2048  # zero_fill still applied normally
    assert "window_unsupported" in spectrum.metadata.raw
    assert "WDW=2" in spectrum.metadata.raw["window_unsupported"]

    # The unwindowed result should match applying window="none"
    # explicitly (both leave the FID as read, sans the recorded WDW).
    _mock_ng_bruker_read(monkeypatch, acqus, fid)
    unwindowed = bruker.read_1d(path, autophase=False, zero_fill=2048, window="exponential", window_kwargs={"lb": 0.0})
    np.testing.assert_allclose(spectrum.data, unwindowed.data)


def test_read_1d_explicit_window_override_bypasses_unsupported_wdw(tmp_path, monkeypatch):
    # An explicit window= override should apply normally (and not be
    # flagged) even when the *recorded* WDW is unsupported -- the
    # override replaces TopSpin's own choice entirely, same as for any
    # other WDW value.
    path = _make_experiment_dir(tmp_path, procs_overrides={"WDW": 2})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_fid()
    _mock_ng_bruker_read(monkeypatch, acqus, fid)

    spectrum = bruker.read_1d(path, autophase=False, window="hann")

    assert "window_unsupported" not in spectrum.metadata.raw


def test_read_2d_unsupported_window_stays_on_raw_fid_path(tmp_path, monkeypatch):
    expt = tmp_path / "expt2d"
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    _write_procs(pdata / "procs", {"SW_p": 8000.0, "SI": 32, "SF": 400.13, "WDW": 2})
    _write_procs(pdata / "proc2s", {"SW_p": 2000.0, "SI": 4, "SF": 100.6, "WDW": 0})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    acqu2s = {"NUC1": "off", "BF1": 100.6, "O1": 0.0}  # pseudo-2D
    row = _synthetic_fid(n=32)
    fid = np.tile(row, (4, 1)).astype(np.complex128)
    dic = {"acqus": acqus, "acqu2s": acqu2s}
    monkeypatch.setattr(bruker.ng.bruker, "read_acqus_file", lambda path: dic)
    monkeypatch.setattr(bruker.ng.bruker, "read", lambda path, **kw: (dic, fid))
    monkeypatch.setattr(bruker.ng.bruker, "remove_digital_filter", lambda d, data: data)
    # See test_bruker_raw_2d.py's _mock_2d for why this is needed --
    # _guess_raw_2d_shape's real-file-scale TD/DTYPA/file-size logic
    # doesn't apply to this small synthetic fixture.
    monkeypatch.setattr(bruker, "_guess_raw_2d_shape", lambda dic, ser_path: fid.shape)

    spectrum = bruker.read_2d(str(expt), autophase=False)

    assert spectrum.data.shape == (4, 32)
    assert "window_unsupported" in spectrum.metadata.raw
    assert "WDW=2" in spectrum.metadata.raw["window_unsupported"]


def test_read_2d_skips_prepare_fid_when_shift_align_lp_all_off(tmp_path, monkeypatch):
    # Real, measured regression this pins down: read_2d used to skip
    # the whole shift/align/linear-predict step per row (three
    # separate `if`s) whenever none of them were requested -- the
    # common case for a fresh load. Folding them into one shared
    # processing.ft.prepare_fid call accidentally made that call
    # (and the np.apply_along_axis wrapping it -- real per-row
    # Python-level overhead, ~1200x slower than skipping entirely for
    # a synthetic 32x16384 benchmark) unconditional. Asserts the fast
    # path by mocking prepare_fid to fail if called at all, not just
    # that read_2d still works -- a slow-but-correct regression
    # wouldn't fail any other test.
    expt = tmp_path / "expt2d_fast_path"
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    _write_procs(pdata / "procs", {"SW_p": 8000.0, "SI": 32, "SF": 400.13, "WDW": 0})
    _write_procs(pdata / "proc2s", {"SW_p": 2000.0, "SI": 4, "SF": 100.6, "WDW": 0})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    acqu2s = {"NUC1": "off", "BF1": 100.6, "O1": 0.0}  # pseudo-2D
    row = _synthetic_fid(n=32)
    fid = np.tile(row, (4, 1)).astype(np.complex128)
    dic = {"acqus": acqus, "acqu2s": acqu2s}
    monkeypatch.setattr(bruker.ng.bruker, "read_acqus_file", lambda path: dic)
    monkeypatch.setattr(bruker.ng.bruker, "read", lambda path, **kw: (dic, fid))
    monkeypatch.setattr(bruker.ng.bruker, "remove_digital_filter", lambda d, data: data)
    # See test_bruker_raw_2d.py's _mock_2d for why this is needed --
    # _guess_raw_2d_shape's real-file-scale TD/DTYPA/file-size logic
    # doesn't apply to this small synthetic fixture.
    monkeypatch.setattr(bruker, "_guess_raw_2d_shape", lambda dic, ser_path: fid.shape)

    def _fail_if_called(*_a, **_k):
        raise AssertionError("prepare_fid was called even though fid_shift/align_echo/linear_predict are all off")

    monkeypatch.setattr(bruker.processing_ft, "prepare_fid", _fail_if_called)

    spectrum = bruker.read_2d(str(expt), autophase=False, fid_shift=0.0, align_echo=False, linear_predict_points=0)

    assert spectrum.data.shape == (4, 32)
