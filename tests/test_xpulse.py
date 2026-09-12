import os

import numpy as np
import pytest

from opennmr.io import xpulse

FIXTURE = """\
##TITLE= synthetic test fixture
##$SFO1= 400.130000
##$SF= 400.130000
##$O1= 2400.500000
##.OBSERVENUCLEUS= 1H
##VARFORM= AFFN, AFFN, AFFN
##LAST= 0.000300
$$ SPINFLOW START
$$NP=4
$$Filter=10000
$$TDEFF=3
$$SI=8
$$ SPINFLOW END
##PAGE= N=1
##DATATABLE= (X++(R..R))
0.000000 1.0
0.000100 0.9
0.000200 0.8
0.000300 0.7
##PAGE= N=2
##DATATABLE= (X++(R..R))
0.000000 0.0
0.000100 0.0
0.000200 0.0
0.000300 0.0
##END=
"""


def _write_fixture(tmp_path):
    path = tmp_path / "spectrum.jdx"
    path.write_text(FIXTURE)
    return str(path)


# Echo top (magnitude 10.0) at index 2, with index 1 and index 3 given
# equal magnitude (2.0) so find_echo_top's parabolic interpolation
# lands exactly on that integer sample (no fractional remainder) --
# keeps the regression test below an exact permutation check rather
# than a fuzzy one across align_echo's FFT-based fractional roll.
ECHO_FIXTURE = """\
##TITLE= synthetic echo test fixture
##$SFO1= 400.130000
##$SF= 400.130000
##$O1= 2400.500000
##.OBSERVENUCLEUS= 1H
##VARFORM= AFFN, AFFN, AFFN
##LAST= 0.000700
$$ SPINFLOW START
$$NP=8
$$Filter=10000
$$TDEFF=4
$$SI=8
$$ SPINFLOW END
##PAGE= N=1
##DATATABLE= (X++(R..R))
0.000000 5.0
0.000100 2.0
0.000200 10.0
0.000300 2.0
0.000400 1.0
0.000500 0.5
0.000600 0.25
0.000700 0.1
##PAGE= N=2
##DATATABLE= (X++(R..R))
0.000000 0.0
0.000100 0.0
0.000200 0.0
0.000300 0.0
0.000400 0.0
0.000500 0.0
0.000600 0.0
0.000700 0.0
##END=
"""


def _write_echo_fixture(tmp_path):
    path = tmp_path / "echo.jdx"
    path.write_text(ECHO_FIXTURE)
    return str(path)


def test_read_1d_records_probe_from_dollar_dollar_probe(tmp_path):
    path = tmp_path / "spectrum_with_probe.jdx"
    path.write_text(FIXTURE.replace("$$Filter=10000", "$$Filter=10000\n$$Probe=HFX"))
    spectrum = xpulse.read_1d(str(path), autophase=False, baseline_correct=False)

    assert spectrum.metadata.probe == "HFX"


def test_read_1d_probe_is_none_when_not_recorded(tmp_path):
    path = _write_fixture(tmp_path)
    spectrum = xpulse.read_1d(path, autophase=False, baseline_correct=False)

    assert spectrum.metadata.probe is None


def test_read_1d_records_pulse_program_from_dot_pulsesequence(tmp_path):
    path = tmp_path / "spectrum_with_pulseq.jdx"
    path.write_text(FIXTURE.replace(
        "##.OBSERVENUCLEUS= 1H", "##.OBSERVENUCLEUS= 1H\n##.PULSESEQUENCE= HSQC_HX",
    ))
    spectrum = xpulse.read_1d(str(path), autophase=False, baseline_correct=False)

    assert spectrum.metadata.pulse_program == "HSQC_HX"


def test_read_1d_pulse_program_is_none_when_not_recorded(tmp_path):
    path = _write_fixture(tmp_path)
    spectrum = xpulse.read_1d(path, autophase=False, baseline_correct=False)

    assert spectrum.metadata.pulse_program is None


def test_read_xpulse_jcamp_decodes_affn_fid(tmp_path):
    path = _write_fixture(tmp_path)
    fid_data = xpulse._read_xpulse_jcamp(path)
    np.testing.assert_allclose(fid_data["real"], [1.0, 0.9, 0.8, 0.7])
    np.testing.assert_allclose(fid_data["imag"], [0.0, 0.0, 0.0, 0.0])
    assert fid_data["params"]["$SFO1"] == "400.130000"
    assert fid_data["exp_params"]["NP"] == 4.0


def test_read_1d_produces_nmrspectrum(tmp_path):
    path = _write_fixture(tmp_path)
    spectrum = xpulse.read_1d(path, autophase=False, baseline_correct=False)

    assert spectrum.ndim == 1
    assert len(spectrum.data) == 4
    assert spectrum.metadata.vendor == "xpulse"
    assert spectrum.metadata.nucleus == "1H"
    assert spectrum.metadata.sfo1 == 400.13
    assert len(spectrum.axes[0].ppm) == 4
    assert len(spectrum.axes[0].hz) == 4
    # NMR convention: frequency axis decreasing left-to-right.
    assert spectrum.axes[0].hz[0] > spectrum.axes[0].hz[-1]


def test_truncate_true_uses_recorded_tdeff_not_literal_true(tmp_path):
    # Regression test: `truncate=True` must read TDEFF (3) from the
    # file, not fall through to isinstance(True, int) and truncate to
    # a single point (bool is an int subclass in Python).
    path = _write_fixture(tmp_path)
    spectrum = xpulse.read_1d(path, truncate=True, autophase=False, baseline_correct=False)
    assert len(spectrum.data) == 3


def test_zero_fill_true_uses_recorded_si(tmp_path):
    path = _write_fixture(tmp_path)
    spectrum = xpulse.read_1d(path, zero_fill=True, autophase=False, baseline_correct=False)
    assert len(spectrum.data) == 8


def test_fid_shift_changes_output(tmp_path):
    path = _write_fixture(tmp_path)
    baseline = xpulse.read_1d(path, autophase=False, baseline_correct=False)
    shifted = xpulse.read_1d(path, autophase=False, baseline_correct=False, fid_shift=1.0)
    assert not np.allclose(baseline.data, shifted.data)


def test_align_echo_runs(tmp_path):
    path = _write_fixture(tmp_path)
    spectrum = xpulse.read_1d(path, autophase=False, baseline_correct=False, align_echo=True)
    assert len(spectrum.data) == 4


def test_baseline_correct_defaults_off_when_align_echo_is_on(tmp_path, monkeypatch):
    # Ported from leonmr's xpulse_image_import: a straight-line
    # baseline fit is a poor model for an echo-aligned profile, so
    # align_echo=True alone (baseline_correct left unspecified) should
    # skip baseline correction -- confirmed by checking whether
    # ft.baseline_correct actually got called.
    called = []
    from opennmr.processing import ft as processing_ft
    monkeypatch.setattr(processing_ft, "baseline_correct", lambda *a, **k: called.append(True) or a[1])

    path = _write_fixture(tmp_path)
    xpulse.read_1d(path, autophase=False, align_echo=True)
    assert called == []

    xpulse.read_1d(path, autophase=False, align_echo=False)
    assert called == [True]

    called.clear()
    xpulse.read_1d(path, autophase=False, align_echo=True, baseline_correct=True)
    assert called == [True]  # explicit override still honoured


def test_align_echo_then_truncate_discards_wrapped_pre_echo_tail(tmp_path):
    # Regression test: align_echo must run BEFORE truncate, not after.
    # align_echo rolls the echo top (here, magnitude 10.0 at index 2)
    # to index 0, which wraps the pre-echo portion (indices 0-1: 5.0,
    # 2.0) around to the tail of the array. Truncating first, against
    # the original unrolled timeline, does nothing to remove that
    # wrapped portion -- get the order backwards and the "aligned" FID
    # (10, 2, [wrapped: 5, 2] instead of the true post-echo decay 1,
    # 0.5) has a spurious bump spliced onto its tail: the "looks like
    # a time-reversal" artifact this ordering fixes.
    path = _write_echo_fixture(tmp_path)
    spectrum = xpulse.read_1d(
        path, autophase=False, baseline_correct=False, align_echo=True, truncate=True,
    )
    fid = spectrum.metadata.raw["fid"]
    assert len(fid) == 4
    np.testing.assert_allclose(np.real(fid), [10.0, 2.0, 1.0, 0.5], atol=1e-9)


def test_align_echo_caps_truncation_at_wrap_point_even_with_no_truncate_requested(tmp_path):
    # Regression test for the real remaining bug: previously, with no
    # `truncate` given at all (the actual default a user hits without
    # manually setting TDeff), align_echo alone did nothing to exclude
    # the wrapped pre-echo portion -- the file's own recorded TDEFF (4,
    # in this fixture) is what excluded it in the sibling test above,
    # not align_echo itself. Same fixture: echo top (10.0) at index 2,
    # wrap point at len(fid) - ceil(top) = 8 - 2 = 6. With truncate
    # left at its default (None/unset), align_echo must still cap the
    # result there on its own, not leave the wrapped [5.0, 2.0] tail in.
    path = _write_echo_fixture(tmp_path)
    spectrum = xpulse.read_1d(path, autophase=False, baseline_correct=False, align_echo=True)
    fid = spectrum.metadata.raw["fid"]
    assert len(fid) == 6
    np.testing.assert_allclose(np.real(fid), [10.0, 2.0, 1.0, 0.5, 0.25, 0.1], atol=1e-9)


def test_align_echo_caps_an_explicit_truncate_past_the_wrap_point(tmp_path):
    # An explicit truncate longer than the wrap point (here, 8 -- the
    # full untruncated length) must still be capped down to the wrap
    # point (6) once align_echo is on, not honoured as-is.
    path = _write_echo_fixture(tmp_path)
    spectrum = xpulse.read_1d(
        path, autophase=False, baseline_correct=False, align_echo=True, truncate=8,
    )
    fid = spectrum.metadata.raw["fid"]
    assert len(fid) == 6


def test_align_echo_does_not_widen_a_shorter_explicit_truncate(tmp_path):
    # An explicit truncate shorter than the wrap point must be
    # honoured exactly -- the cap only ever shortens, never lengthens.
    path = _write_echo_fixture(tmp_path)
    spectrum = xpulse.read_1d(
        path, autophase=False, baseline_correct=False, align_echo=True, truncate=3,
    )
    fid = spectrum.metadata.raw["fid"]
    assert len(fid) == 3
    np.testing.assert_allclose(np.real(fid), [10.0, 2.0, 1.0], atol=1e-9)


def test_linear_predict_extends_fid_before_np_truncation(tmp_path):
    # NP=4 is recorded in the fixture; without re-deriving np_points
    # after linear prediction, the pipeline's own "truncate back to the
    # recorded NP" step would silently discard the 2 predicted points.
    path = _write_fixture(tmp_path)
    spectrum = xpulse.read_1d(
        path, autophase=False, baseline_correct=False,
        linear_predict_points=2, linear_predict_order=1,
    )
    assert len(spectrum.data) == 6


def test_detect_rejects_non_jdx_files(tmp_path):
    path = tmp_path / "not_jcamp.txt"
    path.write_text("hello")
    assert xpulse._detect(str(path)) is False


def test_detect_accepts_xpulse_jdx(tmp_path):
    path = _write_fixture(tmp_path)
    assert xpulse._detect(path) is True


# ============================================================
# Arrayed experiments (e.g. PFG): array#/ subfolders as one
# pseudo-2D spectrum
# ============================================================

def _make_array_experiment(tmp_path, n_steps=3, valueslist=None):
    """A parent folder holding ``array1``..``arrayN`` subfolders, each
    with the same synthetic fixture -- mirrors X-Pulse's real PFG
    layout (``arrayN/<name>_avg.jdx`` plus a sibling ``raw/`` folder
    of individual-scan files, which must be ignored)."""
    root = tmp_path / "pfg_experiment"
    for i in range(1, n_steps + 1):
        array_dir = root / f"array{i}"
        array_dir.mkdir(parents=True)
        (array_dir / f"sample_{i}_avg.jdx").write_text(FIXTURE)
        raw_dir = array_dir / "raw"
        raw_dir.mkdir()
        (raw_dir / f"sample_{i}_1_raw.jdx").write_text(FIXTURE)
    if valueslist is not None:
        params_dir = root / "parameters"
        params_dir.mkdir(parents=True)
        (params_dir / "G1_valueslist.txt").write_text("\n".join(str(v) for v in valueslist) + "\n")
    return str(root)


def test_read_array_stacks_steps_as_pseudo_2d(tmp_path):
    path = _make_array_experiment(tmp_path, n_steps=3)
    spectrum = xpulse.read_array(path, autophase=False, baseline_correct=False)
    assert spectrum.ndim == 2
    assert spectrum.is_pseudo_2d is True
    assert spectrum.data.shape[0] == 3
    assert spectrum.metadata.raw["fid"].shape[0] == 3


def test_read_array_reads_gradient_relative_valueslist(tmp_path):
    path = _make_array_experiment(tmp_path, n_steps=3, valueslist=[0.1, 0.5, 0.9])
    spectrum = xpulse.read_array(path, autophase=False, baseline_correct=False)
    np.testing.assert_allclose(spectrum.metadata.raw["gradient_relative"], [0.1, 0.5, 0.9])


def test_read_array_gradient_relative_none_without_valueslist(tmp_path):
    path = _make_array_experiment(tmp_path, n_steps=3)
    spectrum = xpulse.read_array(path, autophase=False, baseline_correct=False)
    assert spectrum.metadata.raw["gradient_relative"] is None


def test_read_array_ignores_raw_subfolder_scans(tmp_path):
    # Each array folder's raw/ subfolder holds individual-scan repeats
    # sharing the same naming pattern -- must not be picked up as
    # "the" step file (or as a second, conflicting one).
    path = _make_array_experiment(tmp_path, n_steps=2)
    spectrum = xpulse.read_array(path, autophase=False, baseline_correct=False)
    assert spectrum.data.shape[0] == 2


def test_read_array_raises_without_array_dirs(tmp_path):
    empty = tmp_path / "not_an_experiment"
    empty.mkdir()
    try:
        xpulse.read_array(str(empty), autophase=False, baseline_correct=False)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_detect_recognizes_array_folder(tmp_path):
    path = _make_array_experiment(tmp_path, n_steps=2)
    assert xpulse._detect(path) is True


def test_detect_rejects_plain_folder_without_array_subdirs(tmp_path):
    plain = tmp_path / "plain_folder"
    plain.mkdir()
    assert xpulse._detect(str(plain)) is False


def test_read_auto_dispatches_array_folder_to_read_array(tmp_path):
    path = _make_array_experiment(tmp_path, n_steps=2)
    spectrum = xpulse._read_auto(path, autophase=False, baseline_correct=False)
    assert spectrum.ndim == 2
    assert spectrum.is_pseudo_2d is True


def test_read_auto_dispatches_single_file_to_read_1d(tmp_path):
    path = _write_fixture(tmp_path)
    spectrum = xpulse._read_auto(path, autophase=False, baseline_correct=False)
    assert spectrum.ndim == 1


# ============================================================
# Genuine 2D (NTUPLES-format JCAMP-DX, e.g. HSQC)
# ============================================================

def _make_ntuples_fixture(tmp_path, n1=8, n2=8, sw1=100.0, sw2=800.0, f1_hz=None, name="hsqc.jdx"):
    """A synthetic NTUPLES 2D fixture matching the real format
    confirmed by direct inspection of a real HSQC export: one
    (T2++(R..R)), (T2++(I..I)) PAGE pair per T1 increment, States-TPPI
    row-pairing (see processing.ft.decode_hypercomplex_f1's own
    docstring for the forward model this mirrors). Plain AFFN decimal
    text, not ASDF pseudo-digit -- deliberately paired with a
    ``##VARFORM= AFFN, AFFN, AFFN, AFFN`` header claim that's
    technically *correct* here (unlike the real file, which lies about
    this) precisely so the auto-detection path is exercised on both
    "really is AFFN" and (the real file's own) "claims AFFN but is
    ASDF" without needing two separate fixtures."""
    if f1_hz is None:
        f1_hz = 2 * (sw1 / n1)  # an exact FFT bin
    t2 = np.arange(n2) / sw2
    # Each NTUPLES page holds one *real*-valued float per T2 point --
    # confirmed directly against the real file (both its R and I pages
    # decode to plain float64, never complex) -- so the synthetic F2
    # content here must be real too, not the complex quadrature signal
    # test_ft.py's own States-TPPI fixture uses for a genuinely
    # complex-per-row Bruker row.
    f2signal = np.exp(-t2 * 50.0)
    pages = []
    for k in range(n1):
        t1 = k / sw1
        phase = 2 * np.pi * f1_hz * t1
        sign = -1.0 if k % 2 else 1.0
        real_row = sign * np.cos(phase) * f2signal
        imag_row = sign * np.sin(phase) * f2signal
        pages.append(("R", real_row, k))
        pages.append(("I", imag_row, k))

    t2_last = (n2 - 1) / sw2
    lines = [
        "##TITLE= synthetic NTUPLES fixture",
        "##JCAMPDX= 6.0",
        "##DATACLASS= NTUPLES",
        "##NUMDIM= 2",
        "$$SpinFlow Sequence Parameters Start",
        f"$$NP1= {n1}",
        f"$$NP2= {n2}",
        f"$$SW1= {sw1}",
        f"$$Filter= {sw2}",
        "$$O1= 0.0",
        "$$NS= 1",
        "$$TxPPM= 5.0",
        "$$TxPPMX= 100.0",
        "$$SpinFlow Sequence Parameters End",
        "##.OBSERVENUCLEUS= ^1H",
        "##$INDIRECTNUCLEUS= ^13C",
        "##.ACQUISITION SCHEME= TPPI",
        "##$SF= 500.000000 $$ f2",
        "##$SF= 125.000000 $$ f1",
        "##$SFO1= 500.000000 $$ f2",
        "##$SFO1= 125.000000 $$ f1",
        "##NTUPLES= nD NMR FID",
        "##VARNAME= TIME1, TIME2, FID/REAL, FID/IMAG",
        "##SYMBOL= T1, T2, R, I",
        "##.NUCLEUS=  ^13C, ^1H",
        "##VARTYPE= INDEPENDENT, INDEPENDENT, DEPENDENT, DEPENDENT",
        "##VARFORM= AFFN, AFFN, AFFN, AFFN",
        f"##VARDIM= {n1}, {n2}, {n2}, {n2}",
        "##UNITS= SECONDS, SECONDS, ARBITRARY UNITS, ARBITRARY UNITS",
        f"##FIRST= 0, 0, 0, 0",
        f"##LAST= {(n1 - 1) / sw1}, {t2_last}, 0, 0",
        "##FIRST= 0,0,0,0",
    ]
    for kind, row, t1_index in pages:
        lines.append(f"##PAGE=\tT1= {t1_index}")
        lines.append("##FIRST= 0,0,0,0")
        lines.append(f"##DATATABLE= (T2++({kind}..{kind})), PROFILE")
        for i, (t2_val, value) in enumerate(zip(t2, row)):
            lines.append(f"{t2_val:.6f} {value:.6f}")
    lines.append("##END=")

    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return str(path), f1_hz


def test_read_xpulse_ntuples_2d_recovers_known_f1_frequency(tmp_path):
    n1, sw1 = 8, 100.0
    path, f1_hz = _make_ntuples_fixture(tmp_path, n1=n1, sw1=sw1)
    fid_data = xpulse._read_xpulse_ntuples_2d(path)
    raw_rows = fid_data["complex"]
    assert raw_rows.shape[0] == 2 * n1  # row-paired, not yet decoded

    from opennmr.processing import ft as processing_ft
    decoded = processing_ft.decode_hypercomplex_f1(raw_rows, fn_mode=5)
    assert decoded.shape[0] == n1
    spectrum = np.fft.fftshift(np.fft.fft(decoded, axis=0), axes=0)
    power = np.abs(spectrum).sum(axis=1)
    freqs = np.fft.fftshift(np.fft.fftfreq(n1, d=1.0 / sw1))
    assert abs(freqs[np.argmax(power)] - f1_hz) < (sw1 / n1) / 2


def test_read_xpulse_2d_end_to_end_shape_and_metadata(tmp_path):
    path, _ = _make_ntuples_fixture(tmp_path, n1=8, n2=16)
    spectrum = xpulse.read_2d(path, autophase=False)
    assert spectrum.ndim == 2
    assert spectrum.data.shape[0] == 8  # F1: row-paired 16 -> decoded 8
    assert spectrum.metadata.nucleus == "1H"
    assert spectrum.metadata.nucleus_indirect == "13C"
    assert spectrum.metadata.raw["fn_mode"] == 5
    assert spectrum.axes[0].kind == "frequency"
    assert spectrum.axes[1].kind == "frequency"


def test_read_xpulse_2d_zero_fill_and_truncate_f1(tmp_path):
    path, _ = _make_ntuples_fixture(tmp_path, n1=8, n2=8)
    spectrum = xpulse.read_2d(path, autophase=False, truncate_f1=4, zero_fill_f1=16)
    assert spectrum.metadata.raw["fid_raw"].shape[0] == 8  # 4 pairs, row-paired
    assert spectrum.data.shape[0] == 16


def test_detect_varform_ignores_unreliable_header_claim():
    # The real file's own ##VARFORM= claims AFFN for genuinely
    # ASDF-encoded data -- confirm detection goes by content.
    assert xpulse._detect_xpulse_datatable_varform(["0.5J23K11"]) == "ASDF"
    assert xpulse._detect_xpulse_datatable_varform(["0.5 1.23 -4.5e-2"]) == "AFFN"


def test_is_ntuples_2d_file_true_for_ntuples_false_for_plain_1d(tmp_path):
    ntuples_path, _ = _make_ntuples_fixture(tmp_path)
    plain_path = _write_fixture(tmp_path)
    assert xpulse._is_ntuples_2d_file(ntuples_path) is True
    assert xpulse._is_ntuples_2d_file(plain_path) is False


def test_is_ntuples_2d_file_false_for_a_1d_file_that_still_declares_ntuples(tmp_path):
    # Real bug, found on a real file: a plain 1D X-Pulse FID export
    # (test_data/xpulse_MRI/14/*_avg.jdx) still declares
    # "##DATACLASS= NTUPLES" -- every X-Pulse export uses the NTUPLES
    # data-table format for its FID regardless of dimensionality, only
    # differing in "##NUMDIM=". The old detector checked for
    # "DATACLASS"/"NTUPLES" as bare substrings anywhere in the header,
    # so it misdetected this file as 2D and routed it into read_2d,
    # which then failed on this file's non-existent F1 spectral width.
    # _write_fixture's own "false" case above never exercised this --
    # its FIXTURE constant predates the NTUPLES format entirely, so it
    # never contained those substrings at all, discriminating nothing.
    path = tmp_path / "one_d_but_ntuples.jdx"
    path.write_text(
        "##TITLE= real-shaped 1D NTUPLES fixture\n"
        "##DATACLASS= NTUPLES\n"
        "##NUMDIM= 1\n"
        "##NTUPLES= NMRFID\n"
        "##VARNAME= TIME, FID/RE\n"
    )
    assert xpulse._is_ntuples_2d_file(str(path)) is False


def test_read_auto_dispatches_ntuples_file_to_read_2d(tmp_path):
    path, _ = _make_ntuples_fixture(tmp_path)
    spectrum = xpulse._read_auto(path, autophase=False)
    assert spectrum.ndim == 2
    assert spectrum.metadata.nucleus_indirect == "13C"


# ============================================================
# Recorded (autophase) vs user phase
# ============================================================

def _phase_state(raw):
    return (
        float(raw.get("recorded_phase0", 0.0)), float(raw.get("recorded_phase1", 0.0)),
        float(raw.get("phase0", 0.0)), float(raw.get("phase1", 0.0)),
    )


def test_autophase_result_is_recorded_not_a_user_phase():
    # The bug this fixes: an autophase result stored in the *user* slot
    # is overwritten by a saved sidecar's own phase0 on load, and the
    # rebuild-from-raw then applies the user's correction with no base
    # under it -- leaving the spectrum essentially unphased.
    import os

    path = "test_data/xpulse_PFG/4/array1"
    if not os.path.isdir(path):
        pytest.skip(f"{path} not available")
    spectrum = xpulse.read_array("test_data/xpulse_PFG/4")
    recorded0, _r1, user0, user1 = _phase_state(spectrum.metadata.raw)
    assert recorded0 != 0.0          # autophase found something
    assert user0 == 0.0 and user1 == 0.0   # and it is not a user phase


def test_a_supplied_phase_is_a_user_phase_not_a_recorded_one():
    # The mirror case: a caller-supplied phase already *is* the user's,
    # and recording it as a base too would double-count it on rebuild.
    import os

    if not os.path.isdir("test_data/xpulse_PFG/4"):
        pytest.skip("test data not available")
    spectrum = xpulse.read_array("test_data/xpulse_PFG/4", autophase=False,
                                  phase0=31.0, phase1=2.0)
    recorded0, recorded1, user0, user1 = _phase_state(spectrum.metadata.raw)
    assert (recorded0, recorded1) == (0.0, 0.0)
    assert (user0, user1) == (31.0, 2.0)


def test_recorded_and_user_phase_use_the_shared_key_names():
    # Keyed through io.bruker's own accessors so the two vendors cannot
    # drift apart about which key means what.
    from opennmr.io import bruker

    # Converted into the recorded slot's own model, not copied into it:
    # that slot means TopSpin's index ramp (exp(+i*phi)) and autophase
    # returns the pivot model (exp(-i*phi)), so a verbatim copy applies
    # the correction backwards.
    applied = xpulse._phase_metadata(12.0, 3.0, autophased=True)
    assert applied[bruker.RECORDED_PHASE_KEYS["f2"][0]] == pytest.approx(-13.5)
    assert applied[bruker.USER_PHASE_KEYS["f2"][0]] == 0.0
    assert bruker.recorded_phase(applied, "f2") == pytest.approx((-13.5, 3.0))

    supplied = xpulse._phase_metadata(12.0, 3.0, autophased=False)
    assert bruker.recorded_phase(supplied, "f2") == (0.0, 0.0)
    assert supplied[bruker.USER_PHASE_KEYS["f2"][0]] == 12.0


# ============================================================
# Acquisition parameters, for fitting equations and factors
# ============================================================

def _bare_spectrum(raw):
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    return NMRSpectrum(
        data=np.zeros(4), axes=[NMRAxis(kind="frequency", ppm=np.arange(4.0))],
        metadata=NMRMetadata(vendor="xpulse", nucleus="1H", raw=raw),
    )


def test_acquisition_constants_reads_xpulse_exp_params():
    # The user's own case needs TAU by name, from an X-Pulse file --
    # acquisition_constants used to read Bruker's acqus only.
    from opennmr.core.acquisition import acquisition_constants

    constants = acquisition_constants(_bare_spectrum(
        {"exp_params": {"TAU": 1500.0, "P90": 8.78, "NS": 16.0, "Probe": "HFX"}}))
    assert constants["TAU"] == pytest.approx(1500.0)
    assert constants["P90"] == pytest.approx(8.78)
    # A lowercase alias too, since "tau" is what anyone types.
    assert constants["tau"] == pytest.approx(1500.0)
    # Non-numeric parameters are not quantities and are left out.
    assert "Probe" not in constants


def test_acquisition_constants_keeps_case_when_xpulse_needs_it():
    # A real PFG file records both "delta" (gradient duration) and
    # "DELTA" (diffusion delay) -- different numbers. Lowercasing
    # everything the Bruker branch's way would resolve one to the
    # other with nothing on screen to show it.
    from opennmr.core.acquisition import acquisition_constants

    constants = acquisition_constants(_bare_spectrum(
        {"exp_params": {"delta": 4000.0, "DELTA": 50000.0}}))
    assert constants["delta"] == pytest.approx(4000.0)
    assert constants["DELTA"] == pytest.approx(50000.0)


def test_acquisition_constants_reads_the_jcamp_header_too():
    from opennmr.core.acquisition import acquisition_constants

    constants = acquisition_constants(_bare_spectrum(
        {"params": {"$NS": 16.0, "$P1": 8.0, ".PULSESEQUENCE": "cpmg"}}))
    assert constants["NS"] == pytest.approx(16.0)
    assert constants["P1"] == pytest.approx(8.0)
    assert ".PULSESEQUENCE" not in constants


def test_acquisition_constants_on_a_real_xpulse_pfg_dataset():
    from opennmr.core.acquisition import acquisition_constants

    path = os.path.join("test_data", "xpulse_PFG", "4")
    if not os.path.isdir(path):
        pytest.skip(f"needs the real dataset at {path}")
    constants = acquisition_constants(xpulse.read_array(path))
    assert constants["delta"] == pytest.approx(4000.0)
    assert constants["DELTA"] == pytest.approx(50000.0)
    assert constants["P90"] == pytest.approx(8.78)
