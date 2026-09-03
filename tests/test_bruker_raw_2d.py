"""
Mocked-`ng.bruker.read`/`remove_digital_filter` coverage for `read_2d`
-- see test_bruker_raw_fid.py's module docstring for why this
approach (real code, synthetic raw data) rather than real Bruker
files, which aren't available in this environment.
"""

import numpy as np
import pytest

from opennmr.io import bruker
from opennmr.processing import ft


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
    monkeypatch.setattr(bruker.ng.bruker, "read_acqus_file", lambda path: dic)
    monkeypatch.setattr(bruker.ng.bruker, "read", lambda path, **kw: (dic, fid))
    monkeypatch.setattr(bruker.ng.bruker, "remove_digital_filter", lambda d, data: data)
    # _guess_raw_2d_shape's own TD/DTYPA/file-size-based block-rounding
    # (see its own docstring) is real-file-scale logic that doesn't
    # apply to these small synthetic fixtures -- bypass it so the mock
    # `fid` above passes straight through `_read_raw_2d_data` at its
    # own actual shape, unchanged, the same as before that function
    # existed.
    monkeypatch.setattr(bruker, "_guess_raw_2d_shape", lambda dic, ser_path: fid.shape)


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


def test_read_2d_records_probe_from_probhd(tmp_path, monkeypatch):
    path = _make_2d_experiment_dir(tmp_path)
    acqus = {
        "NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0,
        "PROBHD": "1.3 mm MAS BB/1H/19F/19F H13708/0016",
    }
    _mock_2d(monkeypatch, acqus, _synthetic_2d_fid())

    spectrum = bruker.read_2d(path, homonuclear=True, autophase=False)

    assert spectrum.metadata.probe == "1.3 mm MAS BB/1H/19F/19F H13708/0016"


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


def _synthetic_hypercomplex_2d_fid(fn_mode, n1=16, n2=32, sw1=2000.0, sw2=8000.0, f0_hz=1000.0, f1_hz=None):
    """A raw ``ser``-shaped fixture (``2*n1`` rows for States/States-
    TPPI/Echo-Antiecho, one row-pair per t1 increment) built from the
    same forward model as test_ft.py's own
    ``_synthetic_states_pair_dataset``/Echo-Antiecho construction, so
    read_2d's real pipeline (not just decode_hypercomplex_f1 in
    isolation) can be checked to actually recover a known, chosen F1
    frequency end-to-end."""
    if f1_hz is None:
        f1_hz = 3 * (sw1 / n1)  # an exact FFT bin, safe to check by bin index
    t2 = np.arange(n2) / sw2
    f2signal = np.exp(-t2 * 300) * np.exp(2j * np.pi * f0_hz * t2)
    rows = []
    for k in range(n1):
        t1 = k / sw1
        phase = 2 * np.pi * f1_hz * t1
        if fn_mode == 6:  # echo-antiecho: P-type/N-type, not cos/sin
            rows.append(np.exp(1j * phase) * f2signal)
            rows.append(np.exp(-1j * phase) * f2signal)
        else:
            sign = -1.0 if (fn_mode == 5 and k % 2) else 1.0
            rows.append(sign * np.cos(phase) * f2signal)
            rows.append(sign * np.sin(phase) * f2signal)
    return np.array(rows, dtype=np.complex128), f1_hz


@pytest.mark.parametrize("fn_mode", [4, 5])
def test_read_2d_states_and_states_tppi_recover_known_f1_frequency(tmp_path, monkeypatch, fn_mode):
    n1, sw1 = 16, 2000.0
    path = _make_2d_experiment_dir(tmp_path, proc2s_overrides={"SI": n1, "SW_p": sw1})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    acqu2s = {"NUC1": "13C", "BF1": 100.6, "O1": 0.0, "FnMODE": fn_mode}
    fid, f1_hz = _synthetic_hypercomplex_2d_fid(fn_mode, n1=n1, sw1=sw1)
    _mock_2d(monkeypatch, acqus, fid, acqu2s=acqu2s)

    spectrum = bruker.read_2d(path, homonuclear=False, autophase=False)

    assert spectrum.metadata.raw["fn_mode"] == fn_mode
    assert spectrum.data.shape[0] == n1  # row-paired down to n1, not left at the raw 2*n1
    f1_ppm = spectrum.axes[0].ppm
    f1_projection = np.abs(spectrum.data + 1j * spectrum.data_imag).sum(axis=1)
    peak_hz = f1_ppm[np.argmax(f1_projection)] * spectrum.metadata.sfo1_indirect
    assert abs(abs(peak_hz) - f1_hz) < (sw1 / n1) * 1.5


def test_read_2d_truncate_f1_keeps_only_requested_row_pairs(tmp_path, monkeypatch):
    # truncate_f1 must operate on raw row *pairs* for States-TPPI (2
    # raw rows per t1 increment), not raw rows directly -- a request
    # for 4 increments should keep 8 raw rows, decoding down to 4.
    # zero_fill_f1 pinned to the same value so the final output shape
    # unambiguously reflects the truncation -- absent that, the
    # separately-recorded proc2s SI would zero-fill back up past it
    # (matching F2's own truncate+zero-fill independence).
    n1, sw1 = 16, 2000.0
    path = _make_2d_experiment_dir(tmp_path, proc2s_overrides={"SI": n1, "SW_p": sw1})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    acqu2s = {"NUC1": "13C", "BF1": 100.6, "O1": 0.0, "FnMODE": 5}
    fid, _ = _synthetic_hypercomplex_2d_fid(5, n1=n1, sw1=sw1)
    _mock_2d(monkeypatch, acqus, fid, acqu2s=acqu2s)

    spectrum = bruker.read_2d(path, homonuclear=False, autophase=False, truncate_f1=4, zero_fill_f1=4)

    assert spectrum.data.shape[0] == 4


def test_read_2d_truncate_f1_applies_directly_for_qf(tmp_path, monkeypatch):
    # FnMODE=1 (QF) -- every genuine 2D dataset actually in this
    # project's own test_data -- has one raw row per t1 increment
    # already, no row-pairing to account for.
    path = _make_2d_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_2d_fid(n_f1=16)
    _mock_2d(monkeypatch, acqus, fid)

    spectrum = bruker.read_2d(path, homonuclear=True, autophase=False, truncate_f1=6, zero_fill_f1=6)

    assert spectrum.data.shape[0] == 6


def test_read_2d_zero_fill_f1_overrides_recorded_si(tmp_path, monkeypatch):
    path = _make_2d_experiment_dir(tmp_path)  # proc2s SI defaults to 32
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_2d_fid(n_f1=16)
    _mock_2d(monkeypatch, acqus, fid)

    spectrum = bruker.read_2d(path, homonuclear=True, autophase=False, zero_fill_f1=64)

    assert spectrum.data.shape == (64, 64)  # F1 overridden to 64, F2 untouched (procs SI)


def test_read_2d_echo_antiecho_halves_row_count_and_records_fn_mode(tmp_path, monkeypatch):
    # Integration-level wiring check only -- this synthetic fixture's
    # P/N-type forward model (a shared F2 signal times a plain +-phase
    # t1 scalar) doesn't reproduce the real raw-quadrature relationship
    # between echo/antiecho rows (see read_2d's own fn_mode==6 comment,
    # and test_read_2d_echo_antiecho_undoes_the_conjugate_only_for_the_
    # echo_row below for a fixture that does model that relationship),
    # so this doesn't assert peak position -- just that read_2d routes
    # through the decode (shape halved, fn_mode recorded) without
    # raising.
    n1, sw1 = 16, 2000.0
    path = _make_2d_experiment_dir(tmp_path, proc2s_overrides={"SI": n1, "SW_p": sw1})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    acqu2s = {"NUC1": "13C", "BF1": 100.6, "O1": 0.0, "FnMODE": 6}
    fid, _ = _synthetic_hypercomplex_2d_fid(6, n1=n1, sw1=sw1)
    _mock_2d(monkeypatch, acqus, fid, acqu2s=acqu2s)

    spectrum = bruker.read_2d(path, homonuclear=False, autophase=False)

    assert spectrum.metadata.raw["fn_mode"] == 6
    assert spectrum.data.shape[0] == n1


def test_read_2d_echo_antiecho_conjugates_and_reverses_the_echo_row(tmp_path, monkeypatch):
    # Real-data-verified regression guard for the fn_mode==6 fix in
    # read_2d (see its own comment there, and decode_hypercomplex_f1's
    # docstring, for the full two-round story). A real Echo-Antiecho
    # dataset (test_data/Bruker Example Data/Cyclosporine/203) showed
    # every genuine peak alongside an equal-magnitude ghost at its
    # F1-mirror position (ratio ~0.8-1.2x) until the "echo" (even) raw
    # row was conjugated before F2 processing (round 1) -- but a
    # conjugate-only fix, verified only on F1, turned out to introduce
    # an equal-magnitude F2-mirror ghost instead (real user report:
    # "the echo-anti echo data is now mirrored in x") -- round 2 found
    # conjugating *and reversing* the row fixes both simultaneously.
    #
    # This test locks in that exact code behaviour without needing real
    # NMR physics in the fixture: every raw ``ser`` row already passes
    # through one unconditional ``data = np.conj(data)`` (see read_2d's
    # own _CONJUGATE_QUADRATURE_NOTE) before fn_mode==6's own extra
    # conjugate-and-reverse of the even rows -- the two conjugates on
    # the same row cancel, leaving just the reversal, so the net,
    # verified-correct effect is "the echo row reaches F2 processing
    # exactly as stored on disk, but time-reversed; the antiecho row
    # still gets only the one original conjugate, unreversed". Built
    # with a single t1 point (one row pair) and a zeroed antiecho row
    # specifically so the resulting F2 row can be compared directly
    # against processing the known on-disk echo FID through the same
    # window/zero-fill/FFT pipeline, instead of needing to interpret
    # the ppm/Hz axis.
    n2, sw2, f0_hz = 32, 8000.0, 1500.0
    t2 = np.arange(n2) / sw2
    known_fid = np.exp(-t2 * 300) * np.exp(2j * np.pi * f0_hz * t2)
    # Stored pre-reversed so the net (disk value, reversed by the fix)
    # lands back on known_fid -- see the comment above.
    fid = np.array([known_fid[::-1].copy(), np.zeros(n2, dtype=np.complex128)], dtype=np.complex128)
    path = _make_2d_experiment_dir(tmp_path, proc2s_overrides={"SI": 1, "SW_p": 500.0})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    acqu2s = {"NUC1": "13C", "BF1": 100.6, "O1": 0.0, "FnMODE": 6}
    _mock_2d(monkeypatch, acqus, fid, acqu2s=acqu2s)

    spectrum = bruker.read_2d(path, homonuclear=False, autophase=False)
    actual_row = spectrum.data[0] + 1j * spectrum.data_imag[0]

    # antiecho row is all-zero, so decoded = echo_f2 + i*echo_f2.
    expected_f2 = ft.fft_spectrum(ft.zero_fill(known_fid, 64), 1.0 / sw2)[1]
    expected_row = expected_f2 * (1 + 1j)
    np.testing.assert_allclose(actual_row, expected_row)

    # Confirms this actually discriminates: conjugating without also
    # reversing (round 1's own, real-user-reported-broken fix) would
    # have produced a materially different, not just numerically-close,
    # row -- the F2-mirrored ghost the real bug report was about.
    conjugate_only_f2 = ft.fft_spectrum(ft.zero_fill(np.conj(known_fid[::-1]), 64), 1.0 / sw2)[1]
    assert np.max(np.abs(conjugate_only_f2 - expected_f2)) > 0.9 * np.max(np.abs(expected_f2))


def test_read_2d_default_fn_mode_leaves_row_count_undecoded(tmp_path, monkeypatch):
    # No FnMODE recorded at all -- defaults to 1 (QF), which is what
    # every genuine 2D dataset actually in this project's test_data
    # uses: already complex per row, no row-pairing. Regression guard
    # against decode_hypercomplex_f1 accidentally triggering by default.
    path = _make_2d_experiment_dir(tmp_path)
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_2d_fid()  # 16 raw rows, no FnMODE in acqu2s
    _mock_2d(monkeypatch, acqus, fid)

    spectrum = bruker.read_2d(path, homonuclear=True, autophase=False)

    assert spectrum.metadata.raw["fn_mode"] == 1
    assert spectrum.data.shape == (32, 64)  # unchanged from test_read_2d_homonuclear_shape


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


# ============================================================
# Raw ser shape resolution (_guess_raw_2d_shape/_read_raw_2d_data) --
# real on-disk binary files, not mocked ng.bruker.read, since these
# specifically replace nmrglue's own shape-guessing logic. See
# TODO.md's write-up of the real-data investigation this came from:
# row length confirmed via autocorrelation on real raw ser data (a
# genuine periodicity in the FID envelope) on two independent real
# States-TPPI datasets; row count confirmed by directly reading a real
# compiled pulse program's own mc-macro expansion.
# ============================================================

def _write_real_ser(path, n_rows, n_complex_per_row, dtype="<f8"):
    """A real (uninitialized-content) ``ser`` file of exactly the
    given shape, real+imag interleaved (Bruker's own convention) --
    content is irrelevant to shape-resolution tests, only the byte
    count matters."""
    n_real_words = n_rows * n_complex_per_row * 2
    np.zeros(n_real_words, dtype=dtype).tofile(path)


def test_guess_raw_2d_shape_dtypa_float64_block_rounds_to_128_words(tmp_path):
    # DTYPA=2 (float64): the direct dimension is block-padded to the
    # nearest 128-word (1024-byte) boundary -- confirmed via
    # autocorrelation on real raw ser data (see this test's module
    # docstring), not just derived from the acquisition-memory
    # block-padding convention alone.
    ser = tmp_path / "ser"
    td2 = 20  # row count -- exact fit, isolates the row-*length* rounding
    _write_real_ser(ser, n_rows=td2, n_complex_per_row=128, dtype="<f8")
    dic = {"acqus": {"TD": "200", "DTYPA": "2"}, "acqu2s": {"TD": str(td2)}}

    n_rows, n_complex_per_row = bruker._guess_raw_2d_shape(dic, str(ser))

    assert n_complex_per_row == 128  # ceil(200/128)*128 = 256 words = 128 complex
    assert n_rows == td2  # TD divides the file exactly


def test_guess_raw_2d_shape_dtypa_int32_block_rounds_to_256_words(tmp_path):
    # DTYPA absent/0 (int32): the direct dimension is block-padded to
    # the nearest 256-word (1024-byte) boundary instead -- same 1024-
    # byte block target, different word count for the smaller dtype.
    ser = tmp_path / "ser"
    td2 = 5
    _write_real_ser(ser, n_rows=td2, n_complex_per_row=128, dtype="<i4")
    dic = {"acqus": {"TD": "200"}, "acqu2s": {"TD": str(td2)}}

    n_rows, n_complex_per_row = bruker._guess_raw_2d_shape(dic, str(ser))

    assert n_complex_per_row == 128  # ceil(200/256)*256 = 256 words = 128 complex
    assert n_rows == td2


def test_guess_raw_2d_shape_falls_back_to_file_size_when_td_does_not_divide_exactly(tmp_path):
    # Real-data finding (see TODO.md): TD (acqu2s) doesn't always land
    # on the file's own exact byte count, even on a clean (non-
    # aborted-and-restarted) acquisition, for reasons not fully
    # resolved -- rather than fail, fall back to what the file's own
    # size actually supports (floor division, always in-bounds) so the
    # data still loads instead of crashing outright.
    ser = tmp_path / "ser"
    true_rows = 23  # file only actually holds 23 rows...
    _write_real_ser(ser, n_rows=true_rows, n_complex_per_row=128, dtype="<f8")
    dic = {"acqus": {"TD": "200", "DTYPA": "2"}, "acqu2s": {"TD": "500"}}  # ...but TD claims 500

    n_rows, n_complex_per_row = bruker._guess_raw_2d_shape(dic, str(ser))

    assert n_complex_per_row == 128
    assert n_rows == true_rows  # not 500 -- TD didn't divide the file exactly, so file size wins


def test_read_2d_loads_successfully_from_a_real_ser_file_td_mismatch(tmp_path):
    # End-to-end regression test for the real bug this round fixed:
    # every genuine (non-pseudo) 2D Bruker dataset actually available
    # for testing failed to load via nmrglue's own shape-guessing --
    # not just a mismatch, an outright crash/fallback with no raw data
    # recovered at all. This reproduces that exact shape (TD doesn't
    # exactly divide the real file) end to end through the real
    # read_2d, not just the shape-resolution helper alone.
    n_f1, n_f2 = 8, 32
    expt = tmp_path / "realexpt2d"
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    _write_procs(pdata / "procs", {"SW_p": 8000.0, "SI": n_f2, "SF": 400.13, "WDW": 0})
    _write_procs(pdata / "proc2s", {"SW_p": 2000.0, "SI": n_f1, "SF": 100.6, "WDW": 0})
    _write_procs(expt / "acqus", {
        "NUC1": "<1H>", "BF1": 400.0, "O1": 2000.0, "NS": 1.0, "TD": n_f2 * 2, "DTYPA": 2, "BYTORDA": 0,
    })
    _write_procs(expt / "acqu2s", {"NUC1": "<13C>", "BF1": 100.6, "O1": 0.0, "TD": n_f1 + 5})  # +5: won't divide exactly

    # A real echo-shaped FID per row, written as a genuine on-disk
    # float64 real+imag-interleaved ser -- the file only actually holds
    # n_f1 rows (acqu2s claims n_f1 + 5).
    t2 = np.arange(n_f2) / 8000.0
    row = np.exp(-t2 * 300) * np.exp(2j * np.pi * 1000.0 * t2)
    fid = np.tile(row, (n_f1, 1)).astype(np.complex128)
    interleaved = np.empty((n_f1, n_f2 * 2), dtype="<f8")
    interleaved[:, 0::2] = fid.real
    interleaved[:, 1::2] = fid.imag
    interleaved.tofile(expt / "ser")

    def _no_digital_filter(_dic, data):
        return data

    import opennmr.io.bruker as bruker_module
    orig = bruker_module.ng.bruker.remove_digital_filter
    bruker_module.ng.bruker.remove_digital_filter = _no_digital_filter
    try:
        spectrum = bruker.read_2d(str(expt), homonuclear=False, autophase=False, pseudo_2d=False)
    finally:
        bruker_module.ng.bruker.remove_digital_filter = orig

    assert spectrum.data.shape[0] == n_f1  # zero-filled to proc2s's own SI
    assert spectrum.data_imag is not None
    assert spectrum.metadata.raw.get("read_2d_shape_guess_failed") is None
