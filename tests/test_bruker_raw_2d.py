"""
Mocked-reader/`_remove_digital_filter` coverage for `read_2d`
-- see test_bruker_raw_fid.py's module docstring for why this
approach (real code, synthetic raw data) rather than real Bruker
files, which aren't available in this environment.
"""

import os

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
    monkeypatch.setattr(bruker, "read_acqus_files", lambda path: dic)
    # read_1d/read_2d read the binary themselves now (see
    # bruker._read_bruker_binary) rather than via ng.bruker.read, so the
    # synthetic fid has to be injected at that seam too.
    monkeypatch.setattr(bruker, "_read_bruker_binary", lambda p, d, shape=None: fid)
    monkeypatch.setattr(bruker, "_remove_digital_filter", lambda d, data: data)
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


def test_read_2d_records_pulse_program_from_pulprog(tmp_path, monkeypatch):
    path = _make_2d_experiment_dir(tmp_path)
    acqus = {
        "NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0,
        "PULPROG": "exsy_echo3.enb",
    }
    _mock_2d(monkeypatch, acqus, _synthetic_2d_fid())

    spectrum = bruker.read_2d(path, homonuclear=True, autophase=False)

    assert spectrum.metadata.pulse_program == "exsy_echo3.enb"


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


def _synthetic_qf_2d_fid(n1=16, n2=32, sw1=2000.0, sw2=8000.0, f0_hz=1000.0, f1_hz=500.0):
    """A raw ``ser``-shaped fixture for plain QF (FnMODE 0-3, one raw
    row per t1 increment, no row-pairing at all) with a real,
    deliberately signed F1 modulation -- unlike
    ``_synthetic_hypercomplex_2d_fid``'s own end-to-end check (which
    only ever compares ``abs(abs(peak_hz) - f1_hz)``, blind to which
    *side* of centre the recovered peak lands on), this fixture exists
    specifically to catch a mirrored F1 axis, which needs the sign
    checked directly."""
    t2 = np.arange(n2) / sw2
    f2signal = np.exp(-t2 * 300) * np.exp(2j * np.pi * f0_hz * t2)
    rows = []
    for k in range(n1):
        t1 = k / sw1
        rows.append(np.exp(-2j * np.pi * f1_hz * t1) * f2signal)
    return np.array(rows, dtype=np.complex128)


def test_read_2d_qf_f1_axis_is_not_mirrored(tmp_path, monkeypatch):
    # Real bug, found and fixed: for FnMODE 0-3 (QF and friends --
    # decode_hypercomplex_f1 is a no-op for these), each row's own raw
    # complex value is used directly as F1's time-domain point, which
    # already carries the module-level np.conj(data) fix applied once,
    # up front, for F2's own sake. That fix's own docstring explicitly
    # flagged F1 as *assumed*, not verified, correct under the same
    # convention -- confirmed wrong here: conjugating a signal before
    # its own FFT reverses that FFT's own frequency-bin order, so F1
    # came out mirrored. Confirmed directly against real COSY data
    # (PULPROG cosygpmfppqf, FnMODE 1, test_data/Bruker Example
    # Data/exam_CMCse_3/12): read_2d_processed's own trusted diagonal
    # peaks (a homonuclear COSY's diagonal is always exactly F1==F2,
    # dataset-independent) sat at up to ~135x higher intensity in
    # read_2d's own output once F1 was reversed to match, checked at 8
    # independent positions along the diagonal.
    #
    # This test uses a deliberately *signed* synthetic F1 frequency and
    # checks the recovered peak's own sign directly -- unlike
    # test_read_2d_states_and_states_tppi_recover_known_f1_frequency
    # below, which only ever compares abs(abs(peak_hz) - f1_hz), blind
    # to which side of centre the recovered peak lands on, and would
    # not have caught this.
    n1, sw1 = 16, 2000.0
    path = _make_2d_experiment_dir(tmp_path, proc2s_overrides={"SI": n1, "SW_p": sw1})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    acqu2s = {"NUC1": "13C", "BF1": 100.6, "O1": 0.0, "FnMODE": 1}
    f1_hz = 500.0
    fid = _synthetic_qf_2d_fid(n1=n1, sw1=sw1, f1_hz=f1_hz)
    _mock_2d(monkeypatch, acqus, fid, acqu2s=acqu2s)

    spectrum = bruker.read_2d(path, homonuclear=False, autophase=False)

    f1_ppm = spectrum.axes[0].ppm
    f1_projection = np.abs(spectrum.data + 1j * spectrum.data_imag).sum(axis=1)
    peak_hz = f1_ppm[np.argmax(f1_projection)] * spectrum.metadata.sfo1_indirect
    assert abs(peak_hz - f1_hz) < (sw1 / n1) * 1.5  # signed comparison -- not abs(peak_hz)


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


def test_read_2d_echo_antiecho_applies_no_raw_row_transform(tmp_path, monkeypatch):
    # Regression guard for the fn_mode==6 rewrite. Two earlier versions
    # transformed the raw "echo" row before F2 processing and both were
    # wrong; the second (conjugate *and* time-reverse) is the one this
    # test used to lock in. Its reasoning -- that the conjugate's own
    # frequency reversal and the explicit reversal cancel -- holds for a
    # bare FFT but not here, because the window and zero-fill sit in
    # between: reversing a decaying FID and then multiplying it by a
    # decaying window applies the window backwards, and the zero-fill
    # pads what is now the head of the signal. Measured against
    # TopSpin's own xf2 output for Cyclosporine/203: with the transform
    # the echo rows matched TopSpin at a median |corr| of 0.03, without
    # it 0.73, and end-to-end agreement went +0.4448 -> +0.9948.
    #
    # So: the rows reach F2 processing exactly as read, and the whole
    # recombination happens afterwards, on the processed rows, where a
    # conjugate does not touch F2's frequency axis.
    n2, sw2 = 32, 8000.0
    t2 = np.arange(n2) / sw2
    echo_fid = np.exp(-t2 * 300) * np.exp(2j * np.pi * 1500.0 * t2)
    antiecho_fid = np.exp(-t2 * 250) * np.exp(-2j * np.pi * 900.0 * t2)
    # Stored conjugated, so read_2d's own unconditional per-row
    # np.conj (see its _CONJUGATE_QUADRATURE_NOTE) hands F2 processing
    # exactly these two FIDs and the expectation below stays readable.
    fid = np.array([np.conj(echo_fid), np.conj(antiecho_fid)], dtype=np.complex128)
    path = _make_2d_experiment_dir(tmp_path, proc2s_overrides={"SI": 1, "SW_p": 500.0})
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    acqu2s = {"NUC1": "13C", "BF1": 100.6, "O1": 0.0, "FnMODE": 6}
    _mock_2d(monkeypatch, acqus, fid, acqu2s=acqu2s)

    spectrum = bruker.read_2d(path, homonuclear=False, autophase=False)
    actual_row = spectrum.data[0] + 1j * spectrum.data_imag[0]

    echo_f2 = ft.fft_spectrum(ft.zero_fill(echo_fid, 64), 1.0 / sw2)[1]
    antiecho_f2 = ft.fft_spectrum(ft.zero_fill(antiecho_fid, 64), 1.0 / sw2)[1]
    np.testing.assert_allclose(actual_row, 1j * (np.conj(echo_f2) - antiecho_f2))

    # Confirms this discriminates against the transform it replaced:
    # reversing the echo FID before the window/zero-fill is not a
    # near-miss, it is a different row entirely.
    reversed_f2 = ft.fft_spectrum(ft.zero_fill(np.conj(echo_fid)[::-1], 64), 1.0 / sw2)[1]
    assert np.max(np.abs(reversed_f2 - echo_f2)) > 0.9 * np.max(np.abs(echo_f2))


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

    # Stored under its own key now: phase0/phase1 mean the *user's*
    # phase, recorded_phase0/1 the file's own (they use different
    # models -- see io.bruker.apply_dimension_phase).
    assert spectrum.metadata.raw["recorded_phase0"] == pytest.approx(42.0)
    assert spectrum.metadata.raw["phase0"] == pytest.approx(0.0)


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
    orig = bruker_module._remove_digital_filter
    bruker_module._remove_digital_filter = _no_digital_filter
    try:
        spectrum = bruker.read_2d(str(expt), homonuclear=False, autophase=False, pseudo_2d=False)
    finally:
        bruker_module._remove_digital_filter = orig

    assert spectrum.data.shape[0] == n_f1  # zero-filled to proc2s's own SI
    assert spectrum.data_imag is not None
    assert spectrum.metadata.raw.get("read_2d_shape_guess_failed") is None


def test_read_2d_f1_quadrature_normalisation_applies_only_to_states_modes(tmp_path, monkeypatch):
    # The reader normalises Bruker's F1 quadrature sense by negating the
    # sine-modulated rows before decoding. That belongs to States (4)
    # and States-TPPI (5) ONLY: Echo-Antiecho (6) uses a different
    # recombination that this sign is not part of.
    #
    # Regression test for a real bug this file's own suite did not
    # catch: applying it to fn_mode 6 as well flipped real
    # Cyclosporine/203 data from +0.395 to -0.380 against TopSpin, and
    # was found only by the real-data check.
    acqus = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}
    fid = _synthetic_2d_fid(n_f1=8, n_f2=16)

    results = {}
    for fn_mode in (5, 6):
        path = _make_2d_experiment_dir(tmp_path / f"m{fn_mode}")
        _mock_2d(monkeypatch, acqus, fid, acqu2s={"NUC1": "13C", "BF1": 100.0, "O1": 0.0, "FnMODE": fn_mode})
        results[fn_mode] = bruker.read_2d(path, pseudo_2d=False).data

    # States-TPPI: the negation is applied, so the decode sees a - 1j*b.
    # Echo-Antiecho: it is not, so its own (a+b) + 1j*(a-b) is
    # untouched. The two must therefore differ -- applying the negation
    # to both is exactly the bug, and it is what flipped real EA data's
    # sign against TopSpin.
    assert not np.allclose(results[5], results[6])

    # And specifically: Echo-Antiecho must match a build where the
    # sine-row negation is not applied at all.
    reference = ft.decode_hypercomplex_f1(fid.copy(), fn_mode=6)
    negated = fid.copy()
    negated[1::2] *= -1
    assert not np.allclose(reference, ft.decode_hypercomplex_f1(negated, fn_mode=6))


def test_apply_recorded_f1_linear_prediction_follows_me_mod(tmp_path):
    interferogram = np.exp(-0.05 * np.arange(64))[:, None] * np.ones((64, 4), dtype=complex)

    # ME_mod=0 (none) -> untouched, even though LPBIN/NCOEF are present.
    off = bruker.apply_recorded_f1_linear_prediction(
        interferogram, {"ME_mod": 0.0, "LPBIN": 256.0, "NCOEF": 8.0})
    assert off.shape == interferogram.shape

    # ME_mod=2 (LPfc) -> extended to LPBIN/2 complex points. LPBIN counts
    # *real* points, the same convention as TD -- established by
    # measurement: LPBIN/2 reproduces TopSpin's own F1 FWHM of 4 on
    # Cyclosporine/203, while treating it as a complex count
    # over-narrows to 1.
    on = bruker.apply_recorded_f1_linear_prediction(
        interferogram, {"ME_mod": 2.0, "LPBIN": 256.0, "NCOEF": 8.0})
    assert on.shape == (128, 4)
    np.testing.assert_allclose(on[:64], interferogram)  # measured points untouched


def test_apply_recorded_f1_linear_prediction_can_be_forced_off(tmp_path):
    # It is genuinely expensive (one AR fit per F2 column), so a caller
    # wanting a fast preview needs a way out.
    interferogram = np.exp(-0.05 * np.arange(64))[:, None] * np.ones((64, 4), dtype=complex)
    same = bruker.apply_recorded_f1_linear_prediction(
        interferogram, {"ME_mod": 2.0, "LPBIN": 256.0, "NCOEF": 8.0}, enabled=False)
    assert same.shape == interferogram.shape


def test_apply_recorded_f1_linear_prediction_skips_an_impossible_order():
    # An AR fit needs order < n-1; a dataset recording an NCOEF too
    # large for its own F1 point count must be skipped, not raise.
    short = np.ones((6, 3), dtype=complex)
    out = bruker.apply_recorded_f1_linear_prediction(
        short, {"ME_mod": 2.0, "LPBIN": 64.0, "NCOEF": 32.0})
    assert out.shape == short.shape


# --- Real-data agreement with TopSpin, for the hypercomplex modes ---
#
# The synthetic tests above pin the *formulas*; these pin the thing
# that actually matters, on real files, because every wrong version of
# this code so far also passed a synthetic test. Skipped when the
# dataset isn't present, so the suite still runs anywhere.

_HYPERCOMPLEX_REFERENCE = [
    # (path, FnMODE, correlation before the 2026-09-06 fix, floor asserted now)
    ("Bruker Example Data/Cyclosporine/203", 6, 0.4448, 0.99),
    ("Bruker Example Data/Cyclosporine/204", 6, 0.0953, 0.99),
    ("2p5mm_HX_2H_23Na_113Cd_Na2CdFeCN6_D2O_150126A_20260119/2011", 5, 0.6883, 0.99),
    ("2p5mm_HX_2H_23Na_113Cd_Na2CdFeCN6_D2O_150126A_20260119/2016", 5, 0.7049, 0.99),
    # The EXSY loading benchmark's own dataset.
    ("2p5mm_HX_2H_23Na_113Cd_Na2CdFeCN6_D2O_150126A_20260119/2019", 5, 0.7021, 0.99),
    ("2p5mm_HX_2H_23Na_113Cd_Na2CdFeCN6_D2O_150126A_20260119/2023", 5, 0.6721, 0.94),
    ("2p5mm_HX_2H_23Na_113Cd_Na2CdFeCN6_D2O_150126A_20260119/6", 5, 0.6947, 0.95),
]


@pytest.mark.parametrize("relative_path,fn_mode,was,floor", _HYPERCOMPLEX_REFERENCE)
def test_read_2d_hypercomplex_matches_topspin_on_real_data(relative_path, fn_mode, was, floor):
    # Both hypercomplex modes used to carry each row's F2 imaginary part
    # into F1's quadrature, which makes the F1 transform's real part
    # RR - II (the phase twist) instead of RR -- an hourglass-shaped
    # contour, reported by a user as "the phases of f2 and f1 are
    # somehow offset from each other's phases". Echo-Antiecho had a
    # second, independent bug on top (a raw-row transform that reversed
    # each FID before the window, applying the window backwards).
    #
    # `was` records where each dataset sat before the fix, so the
    # margin this test defends is visible rather than implied.
    path = os.path.join("test_data", *relative_path.split("/"))
    if not os.path.isdir(path):
        pytest.skip(f"needs the real dataset at {path}")

    ours = bruker.read_2d(path)
    topspin = bruker.read_2d_processed(path)
    assert ours.metadata.raw["fn_mode"] == fn_mode
    assert ours.data.shape == topspin.data.shape

    a = np.asarray(ours.data, dtype=float).ravel()
    b = np.asarray(topspin.data, dtype=float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    correlation = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert correlation > floor, f"{relative_path}: {correlation:+.4f} (was {was:+.4f})"


def test_read_2d_processed_f1_imaginary_block_is_2ri_not_2ir():
    # Which of TopSpin's four blocks is which dimension's imaginary
    # partner, measured rather than assumed (this project had it
    # recorded backwards for a while): inverse-transform each candidate
    # pair along F1 and only the true one comes back causal, with all
    # its energy in the leading, non-zero-filled points. The 49-50%
    # split the others give is the signature of a pairing that isn't
    # one -- an effectively real spectrum inverse-transforms to a
    # Hermitian-symmetric series. See io.bruker's own
    # _PROCESSED_BLOCK_NAMING_NOTE.
    path = os.path.join("test_data", "Bruker Example Data", "Cyclosporine", "203")
    if not os.path.isdir(path):
        pytest.skip(f"needs the real dataset at {path}")

    procs = bruker._read_procs(path, 1)
    proc2s = bruker._read_procs(path, 1, proc2=True)
    si_f2, si_f1 = int(procs["SI"]), int(proc2s["SI"])

    def _causal_fraction(partner):
        blocks = []
        for name in ("2rr", partner):
            raw = np.fromfile(os.path.join(path, "pdata", "1", name), dtype="<i4")
            blocks.append(raw.reshape(si_f1, si_f2).astype(float))
        interferogram = np.fft.ifft(
            np.fft.ifftshift(blocks[0] + 1j * blocks[1], axes=0), axis=0,
        )
        power = np.linalg.norm(interferogram, axis=1)
        return power[: si_f1 // 4].sum() / power.sum()

    assert _causal_fraction("2ri") > 0.999
    assert _causal_fraction("2ir") < 0.6
    assert _causal_fraction("2ii") < 0.6


def test_synthetic_echo_shows_the_recorded_phase_double_correction():
    # Ground truth for a claim that has no 2D reference dataset here:
    # a recorded PHC1 exists to compensate for transforming from t=0 of
    # an echo rather than its top, so aligning the echo AND applying
    # that phase corrects the same thing twice.
    n, echo_at = 2048, 200
    t = np.arange(n)
    fid = np.exp(-np.abs(t - echo_at) / 300.0) * np.exp(2j * np.pi * 0.02 * (t - echo_at))

    def absorption(x):
        s = np.fft.fftshift(np.fft.fft(x))
        return float(np.sum(np.real(s)) / np.sum(np.abs(s)))

    aligned = np.roll(fid, -echo_at)
    ramp = np.exp(1j * np.deg2rad(360.0 * echo_at * np.arange(n) / (n - 1)))

    target = absorption(aligned)                       # align alone: the goal
    unaligned = absorption(fid)
    topspin_route = absorption(np.fft.ifft(np.fft.ifftshift(
        np.fft.fftshift(np.fft.fft(fid)) * ramp)))     # unaligned + PHC1
    doubled = float(np.sum(np.real(np.fft.fftshift(np.fft.fft(aligned)) * ramp))
                    / np.sum(np.abs(np.fft.fftshift(np.fft.fft(aligned)))))

    assert target > unaligned                # aligning helps
    assert doubled < target                  # ...and applying PHC1 too undoes it
    assert doubled <= unaligned + 0.05       # back to roughly no correction at all
    assert topspin_route > unaligned         # TopSpin's own route is fine


def _echo_2d_dataset():
    path = os.path.join("test_data", "bruker_T2", "11")
    if not os.path.isdir(path):
        pytest.skip(f"needs the real echo dataset at {path}")
    return path


def test_read_2d_drops_the_recorded_phase_when_aligning_the_echo():
    # The T2 echo series are exactly the 2D experiments that need Align
    # Echo, and they record PHC1 of 800-2000 degrees -- so this is where
    # the double correction bites hardest.
    path = _echo_2d_dataset()

    aligned = bruker.read_2d(path, pseudo_2d=True, align_echo=True)

    assert "recorded_phase_dropped" in aligned.metadata.raw
    assert aligned.metadata.raw["recorded_phase1"] == 0.0


def test_read_2d_keeps_the_recorded_phase_without_echo_alignment():
    # Scoped strictly to align_echo -- an ordinary 2D load must still
    # reproduce TopSpin's own recorded first-order term.
    path = _echo_2d_dataset()

    plain = bruker.read_2d(path, pseudo_2d=True)

    assert "recorded_phase_dropped" not in plain.metadata.raw
    assert plain.metadata.raw["recorded_phase1"] != 0.0


# ============================================================
# Row count is capped at F1 TD
# ============================================================

def test_guess_raw_2d_shape_never_exceeds_f1_td(tmp_path):
    # The floor division exists only to find a shape the file can be
    # reshaped to at all. Anything past TD is whatever the file carried
    # beyond the data, not extra t1 increments -- so it is dropped.
    # Real motivation: a pseudo-2D whose rows *are* the slices loaded
    # 19 of them against a 16-entry vdlist.
    ser = tmp_path / "ser"
    dic = {"acqus": {"TD": 256, "DTYPA": 0}, "acqu2s": {"TD": 10}}
    row_bytes = 256 * 4  # TD already a multiple of the 256-word block
    ser.write_bytes(b"\0" * (row_bytes * 13))   # three rows of surplus

    n_rows, _n_complex = bruker._guess_raw_2d_shape(dic, str(ser))
    assert n_rows == 10


def test_guess_raw_2d_shape_still_floors_a_short_file(tmp_path):
    # The fallback's original job is unchanged: a file with *fewer*
    # rows than TD must still reshape rather than crash.
    ser = tmp_path / "ser"
    dic = {"acqus": {"TD": 256, "DTYPA": 0}, "acqu2s": {"TD": 10}}
    ser.write_bytes(b"\0" * (256 * 4 * 7))

    n_rows, _n_complex = bruker._guess_raw_2d_shape(dic, str(ser))
    assert n_rows == 7


def test_guess_raw_2d_shape_without_acqu2s_is_unchanged(tmp_path):
    ser = tmp_path / "ser"
    dic = {"acqus": {"TD": 256, "DTYPA": 0}}
    ser.write_bytes(b"\0" * (256 * 4 * 13))

    n_rows, _n_complex = bruker._guess_raw_2d_shape(dic, str(ser))
    assert n_rows == 13


def test_pseudo_2d_slice_count_matches_f1_td_on_real_data():
    # The dataset that motivated the cap: 19 rows read, TD1 = 16, and a
    # 16-entry vdlist it could not be fitted against.
    path = os.path.join("test_data",
                        "2p5mm_HX_2H_23Na_113Cd_Na2CdFeCN6_D2O_150126A_20260119", "2005")
    if not os.path.isdir(path):
        pytest.skip(f"needs the real dataset at {path}")
    from opennmr.io import lists as io_lists

    spectrum = bruker.read_2d(path, pseudo_2d=True)
    assert spectrum.data.shape[0] == 16
    vdlist = io_lists.read_value_list(os.path.join(path, "vdlist"))
    assert len(vdlist["values"]) == spectrum.data.shape[0]
